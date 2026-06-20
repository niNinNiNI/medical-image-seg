#!/usr/bin/env python3
"""
Single-image inference script for SAM-MedSeg.

Supports:
- Command-line single image inference
- Batch directory inference
- Interactive Jupyter notebook mode (with visualization)

Usage:
    python inference.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pth
    python inference.py --dir path/to/images/ --checkpoint checkpoints/best_model.pth
    python inference.py --image test.jpg --checkpoint best.pth --output mask.png
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import SAMMedSeg
from data import get_inference_transforms


def load_model(checkpoint_path: str, device: str = "cuda") -> SAMMedSeg:
    """
    Load a trained SAMMedSeg model for inference.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Target device.

    Returns:
        Loaded model in eval mode with LoRA merged.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    sam_cfg = config.get("sam", {})
    lora_cfg = config.get("lora", {})

    model = SAMMedSeg(
        sam_checkpoint=sam_cfg.get("checkpoint_path", "checkpoints/sam_vit_b_01ec64.pth"),
        model_type=sam_cfg.get("model_type", "vit_b"),
        freeze_layers=sam_cfg.get("freeze_layers", 9),
        lora_r=lora_cfg.get("r", 4),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        device=torch.device(device),
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Merge LoRA for faster inference
    if lora_cfg.get("r", 4) > 0:
        model.merge_lora()

    print(f"  Loaded model (epoch {checkpoint.get('epoch', '?')}, "
          f"Dice={checkpoint.get('best_dice', 'N/A')})")

    return model


def preprocess_image(
    image_path: str,
    image_size: Tuple[int, int] = (1024, 1024),
) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    Load and preprocess a single image.

    Args:
        image_path: Path to the image file.
        image_size: Target size (H, W).

    Returns:
        preprocessed tensor (1, 3, H, W), original image (H, W, 3), original size (H, W)
    """
    # Load image
    img_pil = Image.open(image_path).convert("RGB")
    original_size = img_pil.size[::-1]  # (W, H) → (H, W)
    original_image = np.array(img_pil)

    # Resize
    img_pil = img_pil.resize(image_size[::-1], Image.BILINEAR)

    # To tensor and normalize
    img_array = np.array(img_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    return img_tensor, original_image, original_size


@torch.no_grad()
def predict(
    model: SAMMedSeg,
    image_tensor: torch.Tensor,
    device: str = "cuda",
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run inference on a preprocessed image.

    Args:
        model: SAMMedSeg model.
        image_tensor: Preprocessed image (1, 3, H, W), normalized to [0, 1].
        device: Device string.
        threshold: Binarization threshold.

    Returns:
        Binary mask as numpy array (H, W), dtype uint8, values in {0, 255}.
    """
    image_tensor = image_tensor.to(device)

    output = model(image_tensor)
    logits = output["final_mask"]  # (1, 1, H, W)

    # Binarize
    mask = (torch.sigmoid(logits) > threshold).float()
    mask = mask.squeeze().cpu().numpy()  # (H, W)

    # Convert to uint8
    mask = (mask * 255).astype(np.uint8)

    return mask


def save_mask(
    mask: np.ndarray,
    output_path: str,
    original_size: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Save predicted mask to file.

    Args:
        mask: Binary mask (H, W) uint8.
        output_path: Path to save.
        original_size: If provided, resize mask to original image size.
    """
    mask_img = Image.fromarray(mask)

    if original_size is not None:
        mask_img = mask_img.resize(original_size[::-1], Image.NEAREST)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(output_path)
    print(f"  Mask saved to: {output_path}")


def predict_single(
    model: SAMMedSeg,
    image_path: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    threshold: float = 0.5,
    image_size: Tuple[int, int] = (1024, 1024),
) -> np.ndarray:
    """
    Run inference on a single image.

    Args:
        model: Loaded SAMMedSeg model.
        image_path: Path to input image.
        output_path: Path for the output mask (optional).
        device: Device string.
        threshold: Binarization threshold.
        image_size: Input image size.

    Returns:
        Predicted binary mask.
    """
    print(f"  Processing: {image_path}")

    image_tensor, original_image, original_size = preprocess_image(
        image_path, image_size
    )

    mask = predict(model, image_tensor, device, threshold)

    if output_path:
        save_mask(mask, output_path, original_size)

    return mask, original_image


def predict_directory(
    model: SAMMedSeg,
    directory: str,
    output_dir: str,
    device: str = "cuda",
    threshold: float = 0.5,
    image_size: Tuple[int, int] = (1024, 1024),
    extensions: Optional[List[str]] = None,
) -> None:
    """
    Run inference on all images in a directory.

    Args:
        model: Loaded model.
        directory: Input directory path.
        output_dir: Output directory path.
        device: Device string.
        threshold: Binarization threshold.
        image_size: Input image size.
        extensions: List of valid image extensions.
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    image_dir = Path(directory)
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No images found in {directory}")
        return

    print(f"Found {len(image_files)} images in {directory}")

    for img_path in sorted(image_files):
        output_name = f"{img_path.stem}_mask.png"
        output_path = Path(output_dir) / output_name

        mask, _ = predict_single(
            model, str(img_path), str(output_path), device, threshold, image_size
        )

    print(f"\nDone! Masks saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM-MedSeg Inference Script"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Single image path."
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Directory of images for batch inference."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Model checkpoint path."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (for single image) or output directory (for batch)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Binarization threshold."
    )
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=[1024, 1024],
        help="Input image size (H W)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda / cpu)."
    )
    args = parser.parse_args()

    if args.image is None and args.dir is None:
        parser.error("Either --image or --dir must be specified.")

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)

    image_size = tuple(args.image_size)

    if args.image:
        # Single image inference
        output_path = args.output or f"{Path(args.image).stem}_mask.png"
        predict_single(
            model, args.image, output_path, device,
            args.threshold, image_size
        )
        print(f"✓ Inference complete. Mask saved to: {output_path}")

    elif args.dir:
        # Batch inference
        output_dir = args.output or f"{args.dir}_masks"
        predict_directory(
            model, args.dir, output_dir, device,
            args.threshold, image_size
        )


if __name__ == "__main__":
    main()
