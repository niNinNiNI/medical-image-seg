#!/usr/bin/env python3
"""
Evaluation script for SAM-MedSeg.

Calculates all metrics on the test set for a trained model.
Supports batch evaluation and comparison of multiple checkpoints.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --checkpoint checkpoints/best_model.pth --compare checkpoints/baseline.pth
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import MedicalSegDataset, get_val_transforms
from models import SAMMedSeg
from utils.metrics import compute_all_metrics, format_metrics
from utils.visualize import plot_single_result, plot_multi_model_comparison


def load_model(checkpoint_path: str, config: Optional[dict] = None, device: str = "cuda") -> SAMMedSeg:
    """
    Load a trained SAMMedSeg model from checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint.
        config: Configuration dict (extracted from checkpoint if None).
        device: Device string.

    Returns:
        Loaded model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if config is None:
        config = checkpoint.get("config", {})

    sam_cfg = config.get("sam", {})
    lora_cfg = config.get("lora", {})
    training_cfg = config.get("training", {})

    # CRITICAL: use_checkpoint must match the training config, otherwise
    # the CheckpointBlock wrapper's .block. prefix in state_dict keys
    # won't match the model's module hierarchy.
    use_ckpt = training_cfg.get("gradient_checkpointing", True)

    model = SAMMedSeg(
        sam_checkpoint=sam_cfg.get("checkpoint_path", "checkpoints/sam_vit_b_01ec64.pth"),
        model_type=sam_cfg.get("model_type", "vit_b"),
        freeze_layers=sam_cfg.get("freeze_layers", 9),
        lora_r=lora_cfg.get("r", 4),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        use_checkpoint=use_ckpt,
        device=torch.device(device),
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # Merge LoRA for faster inference
    if lora_cfg.get("r", 4) > 0:
        model.merge_lora()

    print(f"  Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Best val Dice: {checkpoint.get('best_dice', 'N/A')}")

    return model


@torch.no_grad()
def evaluate_model(
    model: SAMMedSeg,
    dataloader: DataLoader,
    device: str = "cuda",
    save_visualizations: bool = False,
    vis_dir: Optional[str] = None,
    n_vis: int = 10,
    include_hd95: bool = False,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained SAMMedSeg model.
        dataloader: Test/val dataloader.
        device: Device string.
        save_visualizations: Whether to save prediction visualizations.
        vis_dir: Directory for visualizations.
        n_vis: Number of samples to visualize.
        include_hd95: Whether to compute HD95.

    Returns:
        Dict of averaged metrics.
    """
    model.eval()
    all_metrics = {
        "dice": [], "iou": [], "precision": [], "recall": []
    }
    if include_hd95:
        all_metrics["hd95"] = []

    vis_count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        sample_id = batch.get("sample_id", [str(batch_idx)])[0]

        output = model(image)
        pred = output["final_mask"]

        metrics = compute_all_metrics(pred, mask, include_hd95=include_hd95)
        for k, v in metrics.items():
            if k in all_metrics:
                all_metrics[k].append(v)

        # Save visualizations
        if save_visualizations and vis_dir and vis_count < n_vis:
            vis_path = Path(vis_dir) / f"{sample_id}_pred.png"
            plot_single_result(
                image=image[0].cpu(),
                gt_mask=mask[0].cpu(),
                pred_mask=pred[0].cpu(),
                save_path=str(vis_path),
                title=f"Sample: {sample_id}",
            )
            vis_count += 1

    # Average metrics
    avg_metrics = {
        k: np.nanmean(v) for k, v in all_metrics.items()
    }

    return avg_metrics


def evaluate_multiple(
    checkpoint_paths: List[str],
    dataloader: DataLoader,
    config: Optional[dict] = None,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate and compare multiple checkpoints.

    Args:
        checkpoint_paths: List of checkpoint file paths.
        dataloader: Test dataloader.
        config: Shared config dict.
        device: Device string.
        output_dir: Directory for results.

    Returns:
        Dict of checkpoint_name → metrics.
    """
    results = {}

    for ckpt_path in checkpoint_paths:
        name = Path(ckpt_path).stem
        print(f"\n{'='*60}")
        print(f" Evaluating: {name}")
        print(f"{'='*60}")

        model = load_model(ckpt_path, config, device)
        metrics = evaluate_model(model, dataloader, device)
        results[name] = metrics

        print(f"  {format_metrics(metrics)}")

    # Compare
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(" Comparison")
        print(f"{'='*60}")
        print(f"{'Model':<30} {'Dice':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8}")
        print("-" * 70)
        for name, metrics in results.items():
            print(f"{name:<30} {metrics['dice']:>8.4f} {metrics['iou']:>8.4f} "
                  f"{metrics['precision']:>10.4f} {metrics['recall']:>8.4f}")

    # Save results
    if output_dir:
        output_path = Path(output_dir) / "evaluation_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SAM-MedSeg Evaluation Script"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint."
    )
    parser.add_argument(
        "--compare", type=str, nargs="*", default=None,
        help="Additional checkpoints to compare against."
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data/kvasir-seg",
        help="Data directory."
    )
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=[1024, 1024],
        help="Image size (H W)."
    )
    parser.add_argument(
        "--save-vis", action="store_true",
        help="Save prediction visualizations."
    )
    parser.add_argument(
        "--vis-dir", type=str, default="./experiments/visualizations",
        help="Directory for visualizations."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./experiments/results",
        help="Directory for results."
    )
    parser.add_argument(
        "--include-hd95", action="store_true",
        help="Compute HD95 metric (slower)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda / cpu)."
    )
    args = parser.parse_args()

    # Load config
    config = None
    if Path(args.config).exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load test dataset
    image_size = tuple(args.image_size)
    test_dataset = MedicalSegDataset(
        data_dir=args.data_dir,
        split="test",
        transform=get_val_transforms(image_size),
        image_size=image_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    print(f"Test set: {len(test_dataset)} samples")

    # Collect checkpoints to evaluate
    checkpoints = [args.checkpoint]
    if args.compare:
        checkpoints.extend(args.compare)

    # Run evaluation
    results = evaluate_multiple(
        checkpoint_paths=checkpoints,
        dataloader=test_loader,
        config=config,
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
