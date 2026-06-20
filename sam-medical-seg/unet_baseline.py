#!/usr/bin/env python3
"""
U-Net Baseline Training for Medical Image Segmentation.

Trains a standard U-Net as a traditional method baseline for comparison
with SAM-MedSeg. Uses the same dataset, loss, and evaluation metrics.

Usage:
    python unet_baseline.py
    python unet_baseline.py --ratio 1.0 --seed 42
    python unet_baseline.py --ratio 1.0 --seed 42 --epochs 100
    python unet_baseline.py --run-all  # train on all data proportions × 3 seeds
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import MedicalSegDataset, create_few_shot_subset, get_train_transforms, get_val_transforms
from models.unet import UNet, count_parameters
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


def get_dataloaders(
    data_dir: str,
    image_size: tuple,
    batch_size: int,
    num_workers: int,
    few_shot_ratio: float = 1.0,
    seed: int = 42,
):
    """Create train/val/test dataloaders (same as SAM-MedSeg training)."""
    train_dataset = MedicalSegDataset(
        data_dir=data_dir,
        split="train",
        transform=get_train_transforms(image_size),
        image_size=image_size,
        seed=seed,
    )
    val_dataset = MedicalSegDataset(
        data_dir=data_dir,
        split="val",
        transform=get_val_transforms(image_size),
        image_size=image_size,
        seed=seed,
    )
    test_dataset = MedicalSegDataset(
        data_dir=data_dir,
        split="test",
        transform=get_val_transforms(image_size),
        image_size=image_size,
        seed=seed,
    )

    if few_shot_ratio < 1.0:
        print(f"  Few-shot: using {few_shot_ratio*100:.0f}% of training data "
              f"({int(len(train_dataset)*few_shot_ratio)} samples)")
        train_dataset = create_few_shot_subset(train_dataset, few_shot_ratio, seed=seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
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
    writer: Optional[SummaryWriter] = None,
    grad_clip_norm: float = 1.0,
    use_amp: bool = True,
    log_interval: int = 10,
) -> float:
    """Train U-Net for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type="cuda"):
                pred = model(image)
                loss = loss_fn(pred, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(image)
            loss = loss_fn(pred, mask)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()

        if writer is not None and batch_idx % log_interval == 0:
            step = epoch * num_batches + batch_idx
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

        if batch_idx % log_interval == 0:
            print(f"  Epoch {epoch:3d} [{batch_idx:4d}/{num_batches:4d}] loss={loss.item():.4f}")

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate U-Net."""
    model.eval()
    total_loss = 0.0
    all_metrics = {"dice": [], "iou": [], "precision": [], "recall": []}

    for batch in dataloader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        pred = model(image)
        loss = loss_fn(pred, mask)
        total_loss += loss.item()

        metrics = compute_all_metrics(pred, mask)
        for k, v in metrics.items():
            if k in all_metrics:
                all_metrics[k].append(v)

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    avg_metrics["loss"] = avg_loss
    return avg_metrics


def train_unet(
    data_dir: str = "./data/kvasir-seg",
    image_size: tuple = (1024, 1024),
    batch_size: int = 4,
    num_workers: int = 4,
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,
    patience: int = 15,
    grad_clip_norm: float = 1.0,
    use_amp: bool = True,
    few_shot_ratio: float = 1.0,
    seed: int = 42,
    output_dir: str = "./experiments/results/unet_baseline",
    device_str: str = "cuda",
) -> Dict[str, float]:
    """
    Train U-Net baseline.
    """
    set_seed(seed)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory
    log_dir = Path(output_dir) / f"ratio_{few_shot_ratio}_seed_{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, image_size, batch_size, num_workers, few_shot_ratio, seed
    )
    print(f"  Train: {len(train_loader.dataset)} samples, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Model
    print("\nInitializing U-Net...")
    model = UNet(n_channels=3, n_classes=1, bilinear=True, base_channels=64)
    model = model.to(device)
    params = count_parameters(model)
    print(f"  Total: {params['total']:,} params, "
          f"Trainable: {params['trainable']:,}")

    # Loss
    loss_fn = get_loss("combined", dice_weight=0.5)
    loss_fn = loss_fn.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs
    )

    # AMP
    scaler = GradScaler("cuda", enabled=use_amp)

    # Training loop
    print(f"\n{'='*60}")
    print(f" U-Net Training: {epochs} epochs, ratio={few_shot_ratio}, seed={seed}")
    print(f"{'='*60}\n")

    best_dice = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        # Warmup
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        # Train
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler,
            device, epoch, writer, grad_clip_norm, use_amp
        )

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)

        if epoch >= warmup_epochs:
            scheduler.step()

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
            }, log_dir / "best_model.pth")
            print(f"  ✓ New best model (Dice={best_dice:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch + 1}")
            break

    # Test evaluation
    print(f"\n{'='*60}")
    print(" Test Set Evaluation")
    print(f"{'='*60}")

    best_ckpt = torch.load(log_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_metrics = validate(model, test_loader, loss_fn, device)
    print(f"  Best model (epoch {best_ckpt['epoch']}): {format_metrics(test_metrics)}")

    # Save results
    results = {
        "model": "UNet",
        "base_channels": 64,
        "few_shot_ratio": few_shot_ratio,
        "seed": seed,
        "best_epoch": best_ckpt["epoch"],
        "val_dice": best_dice,
        "test_metrics": test_metrics,
        "total_params": params["total"],
        "trainable_params": params["trainable"],
    }

    with open(log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    writer.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="U-Net Baseline Training")
    parser.add_argument("--data-dir", type=str, default="./data/kvasir-seg")
    parser.add_argument("--image-size", type=int, nargs=2, default=[1024, 1024])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Few-shot ratio (1.0 = 100% data)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="./experiments/results/unet_baseline")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all data ratios × 3 seeds")
    args = parser.parse_args()

    image_size = tuple(args.image_size)

    if args.run_all:
        ratios = [0.01, 0.05, 0.10, 1.0]
        all_results = []
        for ratio in ratios:
            for rep in range(3):
                seed = 42 + rep
                print(f"\n{'#'*60}")
                print(f"# U-Net: ratio={ratio}, seed={seed}")
                print(f"{'#'*60}")
                try:
                    results = train_unet(
                        data_dir=args.data_dir,
                        image_size=image_size,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        epochs=args.epochs,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        warmup_epochs=args.warmup_epochs,
                        patience=args.patience,
                        few_shot_ratio=ratio,
                        seed=seed,
                        output_dir=args.output_dir,
                        device_str=args.device,
                    )
                    all_results.append(results)
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    import traceback
                    traceback.print_exc()

        results_dir = Path(args.output_dir)
        with open(results_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll U-Net experiments complete: {len(all_results)} runs")
    else:
        results = train_unet(
            data_dir=args.data_dir,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            few_shot_ratio=args.ratio,
            seed=args.seed,
            output_dir=args.output_dir,
            device_str=args.device,
        )
        print(f"\nFinal results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
