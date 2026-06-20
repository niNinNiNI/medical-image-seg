#!/usr/bin/env python3
"""
Main training script for SAM-MedSeg.

Supports:
- Few-shot training (1%, 5%, 10%, 100% data)
- LoRA fine-tuning
- Mixed precision (AMP fp16)
- Cosine annealing + warmup
- Early stopping
- TensorBoard logging
- Multiple random seeds for reproducible few-shot experiments

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --few-shot-ratio 0.05
    python train.py --config configs/config.yaml --ablation med_decoder_only
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import MedicalSegDataset, create_few_shot_subset, get_train_transforms, get_val_transforms
from models import SAMMedSeg
from losses.combined_loss import get_loss
from utils.metrics import compute_all_metrics, format_metrics


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_dataloaders(
    config: dict,
    batch_size: int,
    num_workers: int,
    few_shot_ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        config: Full configuration dict.
        batch_size: Training batch size.
        num_workers: Data loading workers.
        few_shot_ratio: Few-shot sample ratio.
        seed: Random seed.

    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg = config["data"]
    image_size = tuple(data_cfg["image_size"])

    # Create datasets
    train_dataset = MedicalSegDataset(
        data_dir=data_cfg.get("data_dir", "./data/kvasir-seg"),
        split="train",
        transform=get_train_transforms(image_size),
        image_size=image_size,
        seed=seed,
    )

    val_dataset = MedicalSegDataset(
        data_dir=data_cfg.get("data_dir", "./data/kvasir-seg"),
        split="val",
        transform=get_val_transforms(image_size),
        image_size=image_size,
        seed=seed,
    )

    test_dataset = MedicalSegDataset(
        data_dir=data_cfg.get("data_dir", "./data/kvasir-seg"),
        split="test",
        transform=get_val_transforms(image_size),
        image_size=image_size,
        seed=seed,
    )

    # Apply few-shot sampling to training set only
    if few_shot_ratio < 1.0:
        print(f"  Few-shot: using {few_shot_ratio*100:.0f}% of training data "
              f"({int(len(train_dataset)*few_shot_ratio)} samples)")
        train_dataset = create_few_shot_subset(
            train_dataset, few_shot_ratio, seed=seed
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config["training"].get("pin_memory", True),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # single sample for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config["training"].get("pin_memory", True),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config["training"].get("pin_memory", True),
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: dict,
    writer: Optional[SummaryWriter] = None,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    log_interval = config["logging"].get("log_interval", 10)
    use_amp = config["training"].get("mixed_precision", True)

    for batch_idx, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type="cuda"):
                output = model(image, gt_mask=mask)
                pred = output["final_mask"]
                loss = loss_fn(pred, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient clipping
            grad_norm = config["training"].get("grad_clip_norm", 1.0)
            if grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(image, gt_mask=mask)
            pred = output["final_mask"]
            loss = loss_fn(pred, mask)

            loss.backward()

            grad_norm = config["training"].get("grad_clip_norm", 1.0)
            if grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

            optimizer.step()

        total_loss += loss.item()

        # Logging
        if writer is not None and batch_idx % log_interval == 0:
            step = epoch * num_batches + batch_idx
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

        if batch_idx % log_interval == 0:
            print(f"  Epoch {epoch:3d} [{batch_idx:4d}/{num_batches:4d}] "
                  f"loss={loss.item():.4f}")

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation and compute metrics."""
    model.eval()
    total_loss = 0.0
    all_metrics = {"dice": [], "iou": [], "precision": [], "recall": []}

    for batch in dataloader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        output = model(image)
        pred = output["final_mask"]
        loss = loss_fn(pred, mask)
        total_loss += loss.item()

        metrics = compute_all_metrics(pred, mask)
        for k, v in metrics.items():
            if k in all_metrics:
                all_metrics[k].append(v)

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    avg_metrics["loss"] = avg_loss

    return avg_metrics


def train(
    config: dict,
    few_shot_ratio: float = 1.0,
    seed: int = 42,
    ablation: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> Dict[str, float]:
    """
    Main training loop.

    Args:
        config: Configuration dict.
        few_shot_ratio: Few-shot sample ratio.
        seed: Random seed.
        ablation: Ablation mode (None / "lora_only" / "med_decoder_only").
        resume_from: Path to checkpoint to resume from.

    Returns:
        Best validation metrics.
    """
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup logging
    ablation_suffix = f"_ablation_{ablation}" if ablation else ""
    log_dir = Path(config["project"]["log_dir"]) / f"ratio_{few_shot_ratio}_seed_{seed}{ablation_suffix}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    # Save config snapshot
    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Create dataloaders
    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 4)

    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config, batch_size, num_workers, few_shot_ratio, seed
    )

    # Initialize model
    print("\nInitializing model...")
    sam_cfg = config["sam"]
    lora_cfg = config["lora"]
    prompt_cfg = config["prompt"]

    model = SAMMedSeg(
        sam_checkpoint=sam_cfg["checkpoint_path"],
        model_type=sam_cfg["model_type"],
        freeze_layers=sam_cfg["freeze_layers"],
        lora_r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        prompt_strategy=prompt_cfg["strategy"],
        grid_size=prompt_cfg["grid_size"],
        iou_threshold=prompt_cfg["iou_threshold"],
        fusion_alpha=config["fusion"].get("alpha", 0.5),
        image_size=config["data"]["image_size"][0],
        use_checkpoint=config["training"].get("gradient_checkpointing", False),
        use_lora=ablation != "med_decoder_only",
        device=device,
    )

    # Handle ablations
    if ablation == "lora_only":
        # Disable medical decoder by setting alpha to 1.0
        model.set_fusion_alpha(1.0)
        print("  [Ablation] LoRA only — medical decoder disabled.")

    elif ablation == "med_decoder_only":
        # Disable LoRA and SAM decoder
        model.set_fusion_alpha(0.0)
        print("  [Ablation] MedDecoder only — SAM decoder disabled.")

    # Loss function
    loss_cfg = config["loss"]
    loss_fn = get_loss(loss_cfg["type"], dice_weight=loss_cfg["dice_weight"])

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
    )

    # Learning rate scheduler
    total_epochs = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs
    )

    # AMP
    scaler = GradScaler("cuda", enabled=config["training"].get("mixed_precision", True))

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    patience_counter = 0

    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_dice", 0.0)
        print(f"Resumed from {resume_from} (epoch {start_epoch})")

    # Training loop
    print(f"\n{'='*60}")
    print(f" Starting training: {total_epochs} epochs, "
          f"few_shot_ratio={few_shot_ratio}, seed={seed}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()

        # Warmup phase
        if epoch < warmup_epochs:
            warmup_lr = train_cfg["lr"] * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        # Train
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler,
            device, epoch, config, writer
        )

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)

        # LR scheduling (after warmup)
        if epoch >= warmup_epochs:
            scheduler.step()

        # TensorBoard logging
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/dice", val_metrics["dice"], epoch)
        writer.add_scalar("val/iou", val_metrics["iou"], epoch)

        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"{format_metrics(val_metrics)} | time={epoch_time:.1f}s")

        # Checkpointing
        current_dice = val_metrics["dice"]

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dice": best_dice,
            "val_metrics": val_metrics,
            "config": config,
        }, log_dir / "latest_checkpoint.pth")

        # Save best
        if current_dice > best_dice:
            best_dice = current_dice
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "val_metrics": val_metrics,
                "config": config,
            }, log_dir / "best_model.pth")
            print(f"  ✓ New best model (Dice={best_dice:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        patience = train_cfg.get("patience", 10)
        if patience_counter >= patience:
            print(f"\n  Early stopping triggered after {epoch + 1} epochs.")
            break

    # Final test evaluation
    print(f"\n{'='*60}")
    print(" Test Set Evaluation")
    print(f"{'='*60}")

    # Load best model
    best_ckpt = torch.load(log_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_metrics = validate(model, test_loader, loss_fn, device)
    print(f"  Best model (epoch {best_ckpt['epoch']}): {format_metrics(test_metrics)}")

    # Save test results
    results = {
        "few_shot_ratio": few_shot_ratio,
        "seed": seed,
        "ablation": ablation,
        "best_epoch": best_ckpt["epoch"],
        "val_dice": best_dice,
        "test_metrics": test_metrics,
        "trainable_params": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "total_params": sum(p.numel() for p in model.parameters()),
    }

    with open(log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    writer.close()
    # Clean up backward-compat resources
    if hasattr(model, 'cleanup'):
        model.cleanup()

    return results


def run_experiment_matrix(config: dict) -> None:
    """
    Run the full experiment matrix:
    - 3 few-shot ratios (1%, 5%, 10%) × 3 seeds = 9 runs
    - Plus full data (100%) baseline
    """
    few_shot_cfg = config["few_shot"]
    ratios = few_shot_cfg["ratios"]
    num_repeats = few_shot_cfg["num_repeats"]

    all_results = []

    for ratio in ratios:
        for repeat in range(num_repeats):
            seed = 42 + repeat
            print(f"\n{'#'*60}")
            print(f"# Experiment: ratio={ratio}, seed={seed} "
                  f"(repeat {repeat+1}/{num_repeats})")
            print(f"{'#'*60}")

            try:
                results = train(config, few_shot_ratio=ratio, seed=seed)
                all_results.append(results)
            except Exception as e:
                print(f"  ✗ Experiment failed: {e}")
                import traceback
                traceback.print_exc()

    # Save aggregated results
    results_dir = Path(config["project"]["log_dir"])
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f" All experiments complete! {len(all_results)} runs.")
    print(f" Results saved to {results_dir / 'all_results.json'}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM-MedSeg Training Script"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--few-shot-ratio", type=float, default=1.0,
        help="Few-shot sample ratio (0.01, 0.05, 0.10, 1.0)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--ablation", type=str, default=None,
        choices=[None, "lora_only", "med_decoder_only"],
        help="Ablation mode."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint path."
    )
    parser.add_argument(
        "--run-all", action="store_true",
        help="Run full experiment matrix."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Override data_dir to absolute path
    if "data" not in config:
        config["data"] = {}
    config["data"]["data_dir"] = str(
        PROJECT_ROOT / config.get("data", {}).get("data_dir", "data/kvasir-seg")
    )
    config["sam"]["checkpoint_path"] = str(
        PROJECT_ROOT / config["sam"]["checkpoint_path"]
    )

    if args.run_all:
        run_experiment_matrix(config)
    else:
        results = train(
            config,
            few_shot_ratio=args.few_shot_ratio,
            seed=args.seed,
            ablation=args.ablation,
            resume_from=args.resume,
        )
        print(f"\nFinal results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
