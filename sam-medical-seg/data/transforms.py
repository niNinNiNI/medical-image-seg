"""
Data augmentation pipeline for medical image segmentation.
Built on MONAI transforms for medical imaging compliance.

Reference: MONAI documentation (https://docs.monai.io)
"""

from typing import Optional, Dict, Any
import torch
from monai import transforms
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    RandFlipd,
    RandRotated,
    RandAdjustContrastd,
    RandGaussianNoised,
)

IMAGE_KEY = "image"
MASK_KEY = "mask"
ALLOWED_IMAGE_SIZE = [(512, 512), (1024, 1024)]


def get_train_transforms(image_size: tuple = (1024, 1024)) -> Compose:
    """
    Training transforms with data augmentation.

    Args:
        image_size: Target image size (H, W).

    Returns:
        MONAI Compose transform pipeline.
    """
    if image_size not in ALLOWED_IMAGE_SIZE:
        raise ValueError(f"image_size must be one of {ALLOWED_IMAGE_SIZE}")

    return Compose([
        LoadImaged(keys=[IMAGE_KEY, MASK_KEY]),
        EnsureChannelFirstd(keys=[IMAGE_KEY, MASK_KEY]),
        ScaleIntensityRanged(
            keys=[IMAGE_KEY],
            a_min=0.0, a_max=255.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Resized(
            keys=[IMAGE_KEY, MASK_KEY],
            spatial_size=image_size,
            mode=("bilinear", "nearest"),
        ),
        # Data augmentation (applied randomly only during training)
        RandFlipd(
            keys=[IMAGE_KEY, MASK_KEY],
            prob=0.5,
            spatial_axis=1,  # horizontal flip
        ),
        RandRotated(
            keys=[IMAGE_KEY, MASK_KEY],
            range_x=10.0 / 180.0 * 3.14159,  # ±10 degrees in radians
            prob=0.5,
            mode=("bilinear", "nearest"),
            keep_size=True,
        ),
        RandAdjustContrastd(
            keys=[IMAGE_KEY],
            prob=0.5,
            gamma=(0.8, 1.2),
        ),
        RandGaussianNoised(
            keys=[IMAGE_KEY],
            prob=0.3,
            std=0.01,
        ),
    ])


def get_val_transforms(image_size: tuple = (1024, 1024)) -> Compose:
    """
    Validation/evaluation transforms (NO augmentation).

    Args:
        image_size: Target image size (H, W).

    Returns:
        MONAI Compose transform pipeline.
    """
    if image_size not in ALLOWED_IMAGE_SIZE:
        raise ValueError(f"image_size must be one of {ALLOWED_IMAGE_SIZE}")

    return Compose([
        LoadImaged(keys=[IMAGE_KEY, MASK_KEY]),
        EnsureChannelFirstd(keys=[IMAGE_KEY, MASK_KEY]),
        ScaleIntensityRanged(
            keys=[IMAGE_KEY],
            a_min=0.0, a_max=255.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Resized(
            keys=[IMAGE_KEY, MASK_KEY],
            spatial_size=image_size,
            mode=("bilinear", "nearest"),
        ),
    ])


def get_inference_transforms(image_size: tuple = (1024, 1024)) -> Compose:
    """
    Single-image inference transforms (no mask required).

    Args:
        image_size: Target image size (H, W).

    Returns:
        MONAI Compose transform pipeline.
    """
    return Compose([
        LoadImaged(keys=[IMAGE_KEY]),
        EnsureChannelFirstd(keys=[IMAGE_KEY]),
        ScaleIntensityRanged(
            keys=[IMAGE_KEY],
            a_min=0.0, a_max=255.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Resized(
            keys=[IMAGE_KEY],
            spatial_size=image_size,
            mode="bilinear",
        ),
    ])
