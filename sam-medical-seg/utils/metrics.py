"""
Evaluation metrics for medical image segmentation.

Metrics:
- Dice Score (F1): 2 * |pred ∩ target| / (|pred| + |target|)
- IoU (Jaccard): |pred ∩ target| / |pred ∪ target|
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- HD95: 95th percentile Hausdorff Distance (boundary quality)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array safely."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _binarize(pred: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert logits to binary mask."""
    return (torch.sigmoid(pred) > threshold).float()


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Dice coefficient (F1 score).

    Args:
        pred: Prediction logits (B, 1, H, W) or binary mask.
        target: Ground truth binary mask (B, 1, H, W).
        threshold: Binarization threshold for logits.
        smooth: Smoothing term to avoid division by zero.

    Returns:
        Dice score (higher is better, range [0, 1]).
    """
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = _binarize(pred, threshold)
    elif pred.dtype != target.dtype:
        pred = pred.float()

    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    cardinality = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice.mean().item()


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Compute IoU (Jaccard index).

    Args:
        pred: Prediction logits (B, 1, H, W).
        target: Ground truth binary mask (B, 1, H, W).
        threshold: Binarization threshold.
        smooth: Smoothing term.

    Returns:
        IoU score (higher is better, range [0, 1]).
    """
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = _binarize(pred, threshold)
    elif pred.dtype != target.dtype:
        pred = pred.float()

    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def precision(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Compute precision (TP / (TP + FP))."""
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = _binarize(pred, threshold)
    elif pred.dtype != target.dtype:
        pred = pred.float()

    target = target.float()
    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))

    prec = (tp + smooth) / (tp + fp + smooth)
    return prec.mean().item()


def recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Compute recall / sensitivity (TP / (TP + FN))."""
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = _binarize(pred, threshold)
    elif pred.dtype != target.dtype:
        pred = pred.float()

    target = target.float()
    tp = (pred * target).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))

    rec = (tp + smooth) / (tp + fn + smooth)
    return rec.mean().item()


def hausdorff_distance_95(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute 95th percentile Hausdorff Distance.

    Measures the 95th percentile of distances between boundary points
    of the prediction and ground truth. Lower is better.

    Note: This is the slowest metric — compute only if needed.

    Args:
        pred: Prediction logits (1, 1, H, W) or (B, 1, H, W).
        target: Ground truth mask.
        threshold: Binarization threshold.

    Returns:
        HD95 value (lower is better).
    """
    # Handle batch dimension
    if pred.dim() == 4:
        pred = pred[0:1]
        target = target[0:1]

    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = _binarize(pred, threshold)

    pred_np = _to_numpy(pred[0, 0]).astype(bool)
    target_np = _to_numpy(target[0, 0]).astype(bool)

    # Get boundary pixels (binary erosion)
    pred_boundary = pred_np ^ binary_erosion(pred_np)
    target_boundary = target_np ^ binary_erosion(target_np)

    pred_pts = np.argwhere(pred_boundary)
    target_pts = np.argwhere(target_boundary)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float("nan")

    # Compute pairwise distances efficiently
    from scipy.spatial import cKDTree
    tree_pred = cKDTree(pred_pts.astype(np.float64))
    tree_target = cKDTree(target_pts.astype(np.float64))

    dist_pred_to_target, _ = tree_target.query(pred_pts.astype(np.float64))
    dist_target_to_pred, _ = tree_pred.query(target_pts.astype(np.float64))

    hd95 = np.percentile(
        np.concatenate([dist_pred_to_target, dist_target_to_pred]), 95
    )
    return float(hd95)


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    include_hd95: bool = False,
) -> Dict[str, float]:
    """
    Compute all standard segmentation metrics.

    Args:
        pred: Prediction logits (B, 1, H, W).
        target: Ground truth mask (B, 1, H, W).
        threshold: Binarization threshold.
        include_hd95: Whether to compute HD95 (slower).

    Returns:
        Dict of metric names to values.
    """
    metrics = {
        "dice": dice_score(pred, target, threshold),
        "iou": iou_score(pred, target, threshold),
        "precision": precision(pred, target, threshold),
        "recall": recall(pred, target, threshold),
    }

    if include_hd95:
        try:
            metrics["hd95"] = hausdorff_distance_95(pred, target, threshold)
        except Exception:
            metrics["hd95"] = float("nan")

    return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics dict as a human-readable string."""
    parts = []
    for k, v in metrics.items():
        if np.isnan(v):
            parts.append(f"{k}=NaN")
        else:
            parts.append(f"{k}={v:.4f}")
    s = " ".join(parts)
    if prefix:
        s = f"[{prefix}] {s}"
    return s
