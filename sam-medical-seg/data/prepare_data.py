#!/usr/bin/env python3
"""
Data preparation script: download, preprocess, and split Kvasir-SEG dataset.

Kvasir-SEG: Polyp segmentation dataset with 1000 images.
Source: https://datasets.simula.no/kvasir-seg/

Output structure:
    data/kvasir-seg/
    ├── images/          # raw images (symlink/copy)
    ├── masks/           # raw masks (symlink/copy)
    └── processed/
        ├── train/       # .npz files (image + mask)
        ├── val/
        └── test/
"""

import argparse
import hashlib
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Expected MD5 checksums (placeholder — update after first download)
EXPECTED_MD5 = {
    "kvasir-seg.zip": "",  # will be filled after download
}

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg/kvasir-seg.zip"


def download_dataset(data_dir: Path) -> Path:
    """Download Kvasir-SEG dataset if not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "kvasir-seg.zip"

    if zip_path.exists():
        print(f"  ✓ Archive already exists: {zip_path}")
        return zip_path

    print(f"  Downloading Kvasir-SEG from {KVASIR_URL}...")
    try:
        import urllib.request
        urllib.request.urlretrieve(KVASIR_URL, zip_path)
        print(f"  ✓ Downloaded to {zip_path}")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print(f"  Please manually download from {KVASIR_URL}")
        print(f"  and place the zip file at {zip_path}")
        sys.exit(1)

    return zip_path


def extract_dataset(zip_path: Path, data_dir: Path) -> None:
    """Extract Kvasir-SEG zip archive."""
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if images_dir.exists() and masks_dir.exists() and len(list(images_dir.glob("*"))) > 0:
        print(f"  ✓ Dataset already extracted.")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in tqdm(zf.namelist(), desc="  Extracting"):
            zf.extract(name, data_dir)

    # Kvasir-SEG extracts to kvasir-seg/images/ and kvasir-seg/masks/
    extracted_images = data_dir / "kvasir-seg" / "images"
    extracted_masks = data_dir / "kvasir-seg" / "masks"

    if extracted_images.exists():
        for f in extracted_images.iterdir():
            shutil.move(str(f), str(images_dir / f.name))
    if extracted_masks.exists():
        for f in extracted_masks.iterdir():
            shutil.move(str(f), str(masks_dir / f.name))

    # Cleanup extracted directory
    extracted_root = data_dir / "kvasir-seg"
    if extracted_root.exists():
        shutil.rmtree(extracted_root)

    print(f"  ✓ Extracted {len(list(images_dir.glob('*')))} images, "
          f"{len(list(masks_dir.glob('*')))} masks.")


def split_dataset(
    data_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[list, list, list]:
    """Split image-mask pairs into train/val/test sets with fixed seed."""
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # Shuffle with fixed seed
    rng = random.Random(seed)
    rng.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    print(f"  Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")
    return train_files, val_files, test_files


def resize_array(arr: np.ndarray, size: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    """Resize image/mask array to target size."""
    from PIL import Image as PILImage

    if arr.ndim == 3:
        img = PILImage.fromarray(arr)
    else:
        img = PILImage.fromarray(arr)

    if is_mask:
        img = img.resize(size[::-1], PILImage.NEAREST)
    else:
        img = img.resize(size[::-1], PILImage.BILINEAR)

    return np.array(img)


def preprocess_and_save(
    data_dir: Path,
    file_list: list,
    split: str,
    image_size: Tuple[int, int] = (1024, 1024),
) -> None:
    """Preprocess images and save as .npz files."""
    output_dir = data_dir / "processed" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = data_dir / "masks"

    for img_path in tqdm(file_list, desc=f"  Preprocessing {split}"):
        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))

        # Find corresponding mask
        mask_name = img_path.name
        mask_path = masks_dir / mask_name
        if not mask_path.exists():
            # Try alternative extension
            for ext in [".jpg", ".png", ".jpeg"]:
                alt = masks_dir / (img_path.stem + ext)
                if alt.exists():
                    mask_path = alt
                    break

        if not mask_path.exists():
            print(f"  ⚠ No mask found for {img_path.name}, skipping.")
            continue

        mask = np.array(Image.open(mask_path).convert("L"))

        # Resize
        image = resize_array(image, image_size, is_mask=False)
        mask = resize_array(mask, image_size, is_mask=True)

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        # Transpose to CHW
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        # Binarize mask
        mask = (mask > 127).astype(np.float32)
        mask = mask[np.newaxis, ...]  # HW -> 1HW

        # Save
        out_path = output_dir / f"{img_path.stem}.npz"
        np.savez_compressed(out_path, image=image, mask=mask)


def main():
    parser = argparse.ArgumentParser(description="Prepare Kvasir-SEG dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/kvasir-seg",
        help="Root data directory",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[1024, 1024],
        help="Target image size (H W)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download (use existing files)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset split",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_size = tuple(args.image_size)
    seed = args.seed

    print("=" * 60)
    print(" SAM-MedSeg Data Preparation")
    print("=" * 60)
    print(f"  Data dir:   {data_dir}")
    print(f"  Image size: {image_size}")
    print(f"  Seed:       {seed}")
    print()

    # Step 1: Download
    if not args.skip_download:
        print("[1/4] Downloading dataset...")
        zip_path = download_dataset(data_dir)
        extract_dataset(zip_path, data_dir)
    else:
        print("[1/4] Skipping download (--skip-download).")

    # Step 2: Split
    print("\n[2/4] Splitting dataset...")
    train_files, val_files, test_files = split_dataset(
        data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=seed,
    )

    # Step 3: Preprocess
    print("\n[3/4] Preprocessing and saving...")
    preprocess_and_save(data_dir, train_files, "train", image_size)
    preprocess_and_save(data_dir, val_files, "val", image_size)
    preprocess_and_save(data_dir, test_files, "test", image_size)

    # Step 4: Summary
    print("\n[4/4] Summary:")
    for split in ["train", "val", "test"]:
        count = len(list((data_dir / "processed" / split).glob("*.npz")))
        print(f"  {split}: {count} samples")

    print("\n✓ Data preparation complete!")
    print(f"  Preprocessed data saved to: {data_dir / 'processed'}")


if __name__ == "__main__":
    main()
