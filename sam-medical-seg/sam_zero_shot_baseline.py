#!/usr/bin/env python3
"""
SAM Zero-Shot Baseline for Medical Image Segmentation.

Evaluates the raw SAM model (ViT-B) WITHOUT any fine-tuning, LoRA, or MedDecoder.
Uses automatic grid-sampled prompt points, same strategy as SAM-MedSeg.

This serves as the LOWER BOUND for model performance comparison.

Usage:
    python sam_zero_shot_baseline.py
    python sam_zero_shot_baseline.py --data-dir ./data/kvasir-seg
    python sam_zero_shot_baseline.py --grid-size 32 --image-size 1024 1024
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import MedicalSegDataset, get_val_transforms
from utils.metrics import compute_all_metrics, format_metrics

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠ segment-anything not installed. Install with: pip install segment-anything")


def load_raw_sam(checkpoint_path: str, model_type: str = "vit_b", device: str = "cuda"):
    """
    Load raw SAM model WITHOUT any fine-tuning.

    Returns the SAM model in eval mode.
    """
    print(f"Loading raw SAM {model_type} from {checkpoint_path}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()

    # Freeze all parameters
    for param in sam.parameters():
        param.requires_grad = False

    print(f"  SAM loaded. Total params: {sum(p.numel() for p in sam.parameters()):,}")
    return sam


def generate_grid_prompts(
    image_size: int = 1024,
    grid_size: int = 32,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a uniform N×N grid of prompt points covering the image.

    Returns:
        points: (N*N, 2) — (x, y) coordinates in pixel space
        labels: (N*N,) — all 1 (foreground points)
    """
    spacing = image_size / (grid_size + 1)
    coords = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = int(spacing * (j + 1))
            y = int(spacing * (i + 1))
            coords.append([x, y])

    points = torch.tensor(coords, dtype=torch.float32, device=device)
    labels = torch.ones(len(coords), dtype=torch.long, device=device)

    return points, labels


@torch.no_grad()
def sam_zero_shot_inference(
    sam: torch.nn.Module,
    image: torch.Tensor,
    grid_size: int = 32,
    image_size: int = 1024,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Run zero-shot SAM inference on a single image.

    Args:
        sam: Raw SAM model.
        image: Input image (1, 3, H, W), normalized to [0, 1].
        grid_size: N for N×N grid of prompt points.
        image_size: Image spatial dimension (square).
        device: Device string.

    Returns:
        Predicted binary mask (1, 1, H, W) as logits.
    """
    # SAM expects pixel values in [0, 255]
    pixel_values = image * 255.0

    # Encode image
    image_embedding = sam.image_encoder(pixel_values)

    # Generate grid prompts
    points, labels = generate_grid_prompts(image_size, grid_size, device)

    # Prepare prompt embeddings
    # points: (N, 2), labels: (N,)
    # SAM prompt_encoder expects (points, labels) tuple for sparse embeddings
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=(points.unsqueeze(0), labels.unsqueeze(0)),  # add batch dim
        boxes=None,
        masks=None,
    )

    # Decode masks
    # The mask_decoder processes ALL prompt points for ONE image
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,  # single best mask
    )

    # low_res_masks: (1, 1, 256, 256)
    # Upsample to original resolution
    mask = F.interpolate(
        low_res_masks,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    return mask  # logits, (1, 1, H, W)


@torch.no_grad()
def evaluate_sam_zero_shot(
    sam: torch.nn.Module,
    dataloader: DataLoader,
    grid_size: int = 32,
    image_size: int = 1024,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate zero-shot SAM on a dataset.

    Returns dict of averaged metrics.
    """
    sam.eval()
    all_metrics = {"dice": [], "iou": [], "precision": [], "recall": []}

    for batch in tqdm(dataloader, desc="SAM Zero-Shot Eval"):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        pred = sam_zero_shot_inference(sam, image, grid_size, image_size, device)

        metrics = compute_all_metrics(pred, mask)
        for k, v in metrics.items():
            if k in all_metrics:
                all_metrics[k].append(v)

    avg_metrics = {k: float(np.nanmean(v)) for k, v in all_metrics.items()}
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(
        description="SAM Zero-Shot Baseline Evaluation"
    )
    parser.add_argument(
        "--sam-checkpoint", type=str,
        default="./checkpoints/sam_vit_b_01ec64.pth",
        help="Path to SAM checkpoint."
    )
    parser.add_argument(
        "--model-type", type=str, default="vit_b",
        help="SAM model type."
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
        "--grid-size", type=int, default=32,
        help="Grid size for prompt point sampling (N×N)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./experiments/results/sam_zero_shot",
        help="Output directory for results."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda / cpu)."
    )
    args = parser.parse_args()

    if not SAM_AVAILABLE:
        print("ERROR: segment-anything package not installed.")
        print("Install with: pip install segment-anything")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    image_size = tuple(args.image_size)
    image_size_int = image_size[0]  # assume square

    # Load SAM
    sam = load_raw_sam(args.sam_checkpoint, args.model_type, device)

    # Load test dataset
    print(f"\nLoading test dataset from {args.data_dir}...")
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
    print(f"  Test set: {len(test_dataset)} samples")

    # Evaluate
    print(f"\n{'='*60}")
    print(f" SAM Zero-Shot Evaluation")
    print(f"  Model: {args.model_type}")
    print(f"  Prompt strategy: {args.grid_size}×{args.grid_size} grid points")
    print(f"  Image size: {image_size}")
    print(f"{'='*60}\n")

    start_time = time.time()
    metrics = evaluate_sam_zero_shot(
        sam, test_loader, args.grid_size, image_size_int, device
    )
    elapsed = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print(f" SAM Zero-Shot Results")
    print(f"{'='*60}")
    print(f"  Dice:      {metrics['dice']:.4f}")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/len(test_dataset):.2f}s/sample)")
    print(f"{'='*60}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "sam_zero_shot_baseline",
        "model_type": args.model_type,
        "prompt_strategy": f"grid_{args.grid_size}x{args.grid_size}",
        "image_size": list(image_size),
        "num_test_samples": len(test_dataset),
        "test_metrics": metrics,
        "inference_time_seconds": elapsed,
        "inference_time_per_sample": elapsed / len(test_dataset),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return metrics


if __name__ == "__main__":
    main()
