"""
Medical image segmentation Dataset class for Kvasir-SEG and similar datasets.

Supports:
- Full dataset loading from preprocessed .npy/.npz files or raw images
- Few-shot subset sampling with fixed random seed for reproducibility
- Per-epoch resampling for few-shot scenarios
"""

import random
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class MedicalSegDataset(Dataset):
    """
    Dataset for medical image segmentation.

    Input: data directory, split (train/val/test), optional transform.
    Output: {"image": (3, H, W), "mask": (1, H, W), "prompt_points": optional}

    Supports few-shot sampling via `sample_ratio` parameter.
    Validation and test sets are never sampled.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (1024, 1024),
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Root directory containing images/ and masks/ subdirectories,
                      OR processed/ subdirectory with .npy files.
            split: One of "train", "val", "test".
            transform: MONAI Compose transform (applied to image+mask dict).
            image_size: Target image size (H, W).
            seed: Random seed for split reproducibility.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.seed = seed

        # Collect image-mask pairs
        self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found in {self.data_dir} for split '{split}'. "
                f"Run prepare_data.py first."
            )

    def _collect_samples(self) -> List[Dict[str, str]]:
        """
        Collect image-mask pairs from the data directory.
        Supports both raw images (images/*.jpg + masks/*.jpg) and
        preprocessed numpy files (processed/{split}/*.npz).
        """
        processed_dir = self.data_dir / "processed" / self.split
        images_dir = self.data_dir / "images"
        masks_dir = self.data_dir / "masks"

        samples = []

        # Prefer preprocessed .npz files
        if processed_dir.exists():
            for npz_path in sorted(processed_dir.glob("*.npz")):
                samples.append({"npz_path": str(npz_path)})
            if samples:
                return samples

        # Fallback: raw images
        if images_dir.exists() and masks_dir.exists():
            image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
            for img_path in image_files:
                mask_name = img_path.stem + ".jpg"
                mask_path = masks_dir / mask_name
                if not mask_path.exists():
                    mask_name = img_path.stem + ".png"
                    mask_path = masks_dir / mask_name
                if mask_path.exists():
                    samples.append({
                        "image_path": str(img_path),
                        "mask_path": str(mask_path),
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with keys:
            "image": torch.Tensor of shape (3, H, W), float32, normalized to [0, 1]
            "mask": torch.Tensor of shape (1, H, W), float32, binary {0, 1}
            "sample_id": str
        """
        sample = self.samples[index]

        if "npz_path" in sample:
            data = np.load(sample["npz_path"])
            image = torch.from_numpy(data["image"].astype(np.float32))
            mask = torch.from_numpy(data["mask"].astype(np.float32))
            sample_id = Path(sample["npz_path"]).stem

            # Resize if loaded size differs from target
            _, h, w = image.shape
            if (h, w) != self.image_size:
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).float(),
                    size=self.image_size,
                    mode="nearest",
                ).squeeze(0)
        else:
            # Raw image path — apply transform pipeline
            data_dict = {
                "image": sample["image_path"],
                "mask": sample["mask_path"],
            }
            if self.transform is not None:
                data_dict = self.transform(data_dict)
            image = data_dict["image"]
            mask = data_dict["mask"]
            sample_id = Path(sample["image_path"]).stem

        # Ensure correct shapes
        if image.dim() == 3:
            if image.shape[0] not in [1, 3]:
                image = image.unsqueeze(0)  # HWC -> CHW if needed
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # HW -> 1HW

        # Binarize mask
        mask = (mask > 0.5).float()

        return {
            "image": image.float(),
            "mask": mask.float(),
            "sample_id": sample_id,
        }


def create_few_shot_subset(
    dataset: Dataset,
    ratio: float,
    seed: int = 42,
) -> Subset:
    """
    Create a few-shot subset by randomly sampling from the training set.

    Args:
        dataset: Full training dataset.
        ratio: Sample ratio (0.01 = 1%, 0.05 = 5%, 0.10 = 10%, 1.0 = 100%).
        seed: Random seed for reproducibility.

    Returns:
        torch.utils.data.Subset with sampled indices.

    Note:
        - Validation and test sets should never be sampled.
        - Each few-shot ratio should be run at least 3 times with different seeds.
        - For ratio >= 1.0, returns the full dataset as Subset.
    """
    if ratio >= 1.0:
        indices = list(range(len(dataset)))
    else:
        num_samples = max(1, int(len(dataset) * ratio))
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), num_samples)

    return Subset(dataset, indices)
