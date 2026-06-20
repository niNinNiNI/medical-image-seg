"""
Visualization utilities for medical image segmentation results.

Generates:
- Overlay of prediction mask on original image
- Side-by-side comparison: original + ground truth + prediction
- Multi-model comparison plot (for paper Figure 5)
- Ablation study bar charts
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) tensor to (H, W, 3) uint8 numpy array."""
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu().numpy()
    else:
        img = np.asarray(tensor)

    # CHW → HWC
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    # Normalize to [0, 255]
    if img.max() <= 1.0:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert RGB
    if img.ndim == 2 or img.shape[2] == 1:
        img = np.stack([img.squeeze()] * 3, axis=-1)

    return img


def _to_mask(tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Convert logits/mask tensor to binary numpy array."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    if arr.max() > 1.0 and arr.min() < 0:
        arr = 1.0 / (1.0 + np.exp(-arr))  # sigmoid

    arr = (arr > threshold).astype(np.uint8)
    return arr.squeeze()


def overlay_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Overlay a binary mask on an image with transparency.

    Args:
        image: Input image (3, H, W) or (1, 3, H, W).
        mask: Binary/logit mask (1, H, W) or (H, W).
        alpha: Transparency weight for the overlay.
        color: RGB color for the mask overlay.

    Returns:
        (H, W, 3) uint8 numpy array of the overlaid image.
    """
    img = _to_image(image)
    mask_bin = _to_mask(mask)

    overlay = img.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bin > 0,
            (1 - alpha) * img[:, :, c] + alpha * color[c],
            img[:, :, c],
        )

    return overlay.astype(np.uint8)


def plot_single_result(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Plot original image, ground truth, and prediction side by side.

    Args:
        image: Input image (3, H, W).
        gt_mask: Ground truth mask (1, H, W).
        pred_mask: Predicted mask/logits (1, H, W).
        save_path: Path to save the figure.
        title: Optional title.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure.
    """
    img = _to_image(image)
    gt = _to_mask(gt_mask)
    pred = _to_mask(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_multi_model_comparison(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Compare multiple models' predictions on the same image.
    Used for paper Figure 5.

    Args:
        image: Input image (3, H, W).
        gt_mask: Ground truth mask (1, H, W).
        predictions: Dict of model_name → predicted mask.
        save_path: Path to save.
        figsize: Figure size (auto-computed if None).

    Returns:
        matplotlib Figure.
    """
    n_models = len(predictions)
    n_cols = min(n_models + 2, 6)  # image + GT + N models
    n_rows = math.ceil((n_models + 2) / n_cols)

    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    img = _to_image(image)
    gt = _to_mask(gt_mask)

    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")

    # Ground truth
    axes[0, 1].imshow(gt, cmap="gray")
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis("off")

    # Model predictions
    for idx, (name, pred) in enumerate(predictions.items()):
        row = (idx + 2) // n_cols
        col = (idx + 2) % n_cols
        pred_mask = _to_mask(pred)
        axes[row, col].imshow(pred_mask, cmap="gray")
        axes[row, col].set_title(name)
        axes[row, col].axis("off")

    # Hide empty subplots
    for idx in range(n_models + 2, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_ablation_bar_chart(
    results: Dict[str, Dict[str, float]],
    metric: str = "dice",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Bar chart for ablation study results.
    Used for paper Figure 6.

    Args:
        results: Dict of experiment_name → {metric_name: value, ...}.
        metric: Which metric to plot.
        save_path: Path to save.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = list(results.keys())
    values = [results[name].get(metric, 0.0) for name in names]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"Ablation Study — {metric.upper()}", fontsize=14)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_few_shot_curve(
    ratios: List[float],
    metrics_by_ratio: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Line chart showing performance vs. sample ratio.
    Used for paper Figure 7.

    Args:
        ratios: List of sample ratios (e.g., [0.01, 0.05, 0.10, 1.0]).
        metrics_by_ratio: Dict of model_name → list of metric values.
        save_path: Path to save.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert ratios to percentages for x-axis labels
    x_labels = [f"{int(r * 100)}%" for r in ratios]
    x = list(range(len(ratios)))

    markers = ["o", "s", "^", "D", "v", "<"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_by_ratio)))

    for idx, (model_name, values) in enumerate(metrics_by_ratio.items()):
        ax.plot(
            x,
            values,
            marker=markers[idx % len(markers)],
            color=colors[idx],
            linewidth=2,
            markersize=8,
            label=model_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Training Data Ratio", fontsize=12)
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Few-Shot Learning Performance", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def close_all():
    """Close all matplotlib figures to free memory."""
    plt.close("all")
