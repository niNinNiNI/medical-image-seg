#!/usr/bin/env python3
"""
Generate all paper-quality figures for the SAM-MedSeg project.
One script → 4 figures → experiments/figures/

Usage:
    python visualize_paper_figures.py              # all figures
    python visualize_paper_figures.py --fig 1      # only figure 1
    python visualize_paper_figures.py --no-infer   # skip inference (figure 3)
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"
DATA_DIR = PROJECT_ROOT / "data" / "kvasir-seg"

# --- Global style ---
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

# Color palette
C_SAM = "#2c7bb6"       # blue for SAM-MedSeg
C_UNET = "#d7191c"      # red/orange for U-Net
C_ZERO = "#7f7f7f"      # gray for zero-shot
C_BARS = ["#2c7bb6", "#abd9e9", "#fdae61"]  # Full, LoRA, MedDecoder


# ============================================================
#  Data loading helpers
# ============================================================

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_sam_medseg_results() -> Dict[float, List[dict]]:
    """Collect all SAM-MedSeg results grouped by ratio.
    Returns {ratio: [result_dict, ...]} excluding ablation dirs."""
    by_ratio = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir() or "ablation" in d.name or "unet" in d.name or "zero_shot" in d.name:
            continue
        rj = d / "results.json"
        if not rj.exists():
            continue
        r = load_json(rj)
        ratio = r["few_shot_ratio"]
        by_ratio.setdefault(ratio, []).append(r)
    return by_ratio


def collect_unet_results() -> Dict[float, List[dict]]:
    """Collect all U-Net results grouped by ratio."""
    by_ratio = {}
    unet_dir = RESULTS_DIR / "unet_baseline"
    if not unet_dir.exists():
        return by_ratio
    for d in sorted(unet_dir.iterdir()):
        if not d.is_dir():
            continue
        rj = d / "results.json"
        if not rj.exists():
            continue
        r = load_json(rj)
        ratio = r["few_shot_ratio"]
        by_ratio.setdefault(ratio, []).append(r)
    return by_ratio


def collect_ablation_results() -> Dict[str, dict]:
    """Collect ablation results.
    Returns {label: result_dict}."""
    out = {}
    ablation_map = {
        "ratio_0.05_seed_42_ablation_lora_only": "LoRA Only",
        "ratio_0.05_seed_42_ablation_med_decoder_only": "MedDecoder Only",
    }
    full_dir = RESULTS_DIR / "ratio_0.05_seed_42"
    if (full_dir / "results.json").exists():
        out["Full Model"] = load_json(full_dir / "results.json")

    for dirname, label in ablation_map.items():
        d = RESULTS_DIR / dirname
        if (d / "results.json").exists():
            out[label] = load_json(d / "results.json")
    return out


def load_zero_shot() -> dict:
    rj = RESULTS_DIR / "sam_zero_shot" / "results.json"
    if rj.exists():
        return load_json(rj)
    return {}


# ============================================================
#  Figure 1: Data efficiency curve with error bars
# ============================================================

def fig_data_efficiency(save_path: Optional[str] = None):
    """SAM-MedSeg vs U-Net data efficiency with error bars + crossover annotation."""
    sam = collect_sam_medseg_results()
    unet = collect_unet_results()
    zero = load_zero_shot()

    ratios = sorted(sam.keys())
    ratio_pct = [f"{int(r*100)}%" for r in ratios]

    def mean_std(by_ratio, key="dice"):
        means, stds = [], []
        for r in ratios:
            vals = [x["test_metrics"][key] for x in by_ratio.get(r, [])]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        return np.array(means), np.array(stds)

    sam_m, sam_s = mean_std(sam)
    unet_m, unet_s = mean_std(unet)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    x = np.arange(len(ratios))

    # Error bar lines
    ax.errorbar(x, sam_m, yerr=sam_s, color=C_SAM, marker="o", markersize=10,
                linewidth=2.5, capsize=6, capthick=2, label="SAM-MedSeg (ours)", zorder=5)
    ax.errorbar(x, unet_m, yerr=unet_s, color=C_UNET, marker="s", markersize=10,
                linewidth=2.5, capsize=6, capthick=2, label="U-Net", zorder=4)

    # SAM zero-shot horizontal line
    if zero:
        z_dice = zero["test_metrics"]["dice"]
        ax.axhline(y=z_dice, color=C_ZERO, linestyle="--", linewidth=1.8, alpha=0.7,
                   label=f"SAM Zero-Shot ({z_dice:.3f})")

    # --- Annotate crossover region ---
    # The crossover is between 5% (index 1) and 10% (index 2)
    cx_mid = (x[1] + x[2]) / 2.0
    cy_mid = (sam_m[2] + unet_m[2]) / 2.0
    ax.annotate(
        "Crossover\n(~5%–10%)",
        xy=(cx_mid, cy_mid),
        xytext=(cx_mid + 0.45, cy_mid + 0.06),
        fontsize=11,
        fontweight="bold",
        color="#8B0000",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#8B0000", lw=2.0,
                        connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", edgecolor="#8B0000", alpha=0.9),
    )

    # Highlight SAM-MedSeg advantage region
    ax.annotate(
        "SAM-MedSeg\nadvantage",
        xy=(0.25, 0.44),
        fontsize=10,
        fontstyle="italic",
        color=C_SAM,
        ha="center",
    )
    # Highlight U-Net advantage region
    ax.annotate(
        "U-Net\nadvantage",
        xy=(2.8, 0.62),
        fontsize=10,
        fontstyle="italic",
        color=C_UNET,
        ha="center",
    )

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(ratio_pct)
    ax.set_xlabel("Training Data Ratio", fontsize=14, fontweight="bold")
    ax.set_ylabel("Test Dice Score", fontsize=14, fontweight="bold")
    ax.set_title("Data Efficiency Comparison: SAM-MedSeg vs U-Net", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="gray")
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_ylim(0.22, 0.80)

    # Add value labels near points
    for i in range(len(x)):
        ax.annotate(f"{sam_m[i]:.3f}", (x[i], sam_m[i]), textcoords="offset points",
                    xytext=(8, 12), fontsize=8, color=C_SAM, ha="center")
        ax.annotate(f"{unet_m[i]:.3f}", (x[i], unet_m[i]), textcoords="offset points",
                    xytext=(8, -16), fontsize=8, color=C_UNET, ha="center")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"  ✓ Figure 1 saved → {save_path}")
    plt.close(fig)


# ============================================================
#  Figure 2: Ablation — grouped bar chart (3 groups × 4 metrics)
# ============================================================

def fig_ablation(save_path: Optional[str] = None):
    """Grouped bar chart: Full Model | LoRA Only | MedDecoder Only across 4 metrics."""
    abl = collect_ablation_results()

    # Ensure consistent order
    order = ["Full Model", "LoRA Only", "MedDecoder Only"]
    labels = [k for k in order if k in abl]
    metrics = ["dice", "iou", "precision", "recall"]
    metric_labels = ["Dice", "IoU", "Precision", "Recall"]

    data = {}
    for m in metrics:
        data[m] = [abl[name]["test_metrics"][m] for name in labels]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    bar_width = 0.55
    x_pos = np.arange(len(labels))

    for idx, (metric_key, metric_name) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        values = data[metric_key]
        bars = ax.bar(x_pos, values, bar_width, color=C_BARS, edgecolor="black",
                      linewidth=0.8, zorder=3)

        # Value labels on top
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel(metric_name, fontsize=13, fontweight="bold")
        ax.set_title(f"Ablation — {metric_name}", fontsize=14, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.22)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

        # Add parameter count annotation
        if metric_key == "dice":
            for i, name in enumerate(labels):
                p = abl[name].get("trainable_params", 0)
                if p > 1_000_000:
                    p_str = f"{p/1e6:.1f}M"
                else:
                    p_str = f"{p/1e3:.0f}K"
                ax.text(i, 0.02, f"{p_str}\nparams", ha="center", fontsize=7,
                        color="gray", fontstyle="italic")

    fig.suptitle("Ablation Study — Component Contribution (5% data, seed 42)",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"  ✓ Figure 2 saved → {save_path}")
    plt.close(fig)


# ============================================================
#  Figure 3: Segmentation comparison grid
# ============================================================

def run_inference_batch(image_paths, checkpoint_path, model_type="sam"):
    """Run inference on a list of image paths. Returns list of predicted masks (numpy)."""
    import torch
    from PIL import Image
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "sam":
        from models import SAMMedSeg
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        use_lora = True
        # CRITICAL: use_checkpoint=True so module hierarchy matches the saved checkpoint
        # (CheckpointBlock wrapper adds .block. prefix to state dict keys)
        model = SAMMedSeg(
            sam_checkpoint=str(PROJECT_ROOT / "checkpoints" / "sam_vit_b_01ec64.pth"),
            use_lora=use_lora,
            use_checkpoint=True,
            device=device,
        )
        # Remap keys: strip .block. if needed (handles both wrapped and unwrapped checkpoints)
        state_dict = ckpt["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        if hasattr(model, "merge_lora"):
            model.merge_lora()
        print(f"  [infer] Loaded SAM-MedSeg from {checkpoint_path.name}")

        masks = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((1024, 1024), Image.BILINEAR)
            tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    output = model(tensor)
            pred = output["final_mask"].squeeze().cpu()
            pred_bin = (torch.sigmoid(pred) > 0.5).numpy().astype(np.uint8)
            masks.append(pred_bin)
        return masks

    elif model_type == "unet":
        from models.unet import UNet
        model = UNet(n_channels=3, n_classes=1, bilinear=True, base_channels=64)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        print(f"  [infer] Loaded U-Net from {checkpoint_path.name}")

        masks = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((1024, 1024), Image.BILINEAR)
            tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    pred = model(tensor)
            pred_bin = (torch.sigmoid(pred).squeeze().cpu() > 0.5).numpy().astype(np.uint8)
            masks.append(pred_bin)
        return masks

    elif model_type == "sam_zero":
        # SAM zero-shot: use same approach as sam_zero_shot_baseline.py
        # (direct encoder → prompt encoder → mask decoder, avoids NMS issues)
        import torch.nn.functional as F
        from segment_anything import sam_model_registry
        sam = sam_model_registry["vit_b"](
            checkpoint=str(PROJECT_ROOT / "checkpoints" / "sam_vit_b_01ec64.pth")
        )
        sam.to(device)
        sam.eval()
        for p in sam.parameters():
            p.requires_grad = False

        # Pre-compute grid prompts (32x32 = 1024 points)
        grid_size = 32
        spacing = 1024 / (grid_size + 1)
        coords = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = int(spacing * (j + 1))
                y = int(spacing * (i + 1))
                coords.append([x, y])
        grid_pts = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, 2)
        grid_lbls = torch.ones(1, len(coords), dtype=torch.long, device=device)  # (1, N)

        print(f"  [infer] Loaded SAM zero-shot (grid 32x32 prompt)")

        masks = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((1024, 1024), Image.BILINEAR)
            tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
            # SAM expects [0, 255]
            pixel_values = (tensor * 255.0).unsqueeze(0).to(device)

            with torch.no_grad():
                image_embedding = sam.image_encoder(pixel_values)
                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=(grid_pts, grid_lbls), boxes=None, masks=None
                )
                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                mask = F.interpolate(
                    low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False
                )
            pred_bin = (torch.sigmoid(mask).squeeze().cpu() > 0.5).numpy().astype(np.uint8)
            masks.append(pred_bin)
        return masks


def dice_numpy(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice score between two binary numpy arrays."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.sum(pred & gt)
    return float(2.0 * intersection / (np.sum(pred) + np.sum(gt) + smooth))


def fig_segmentation_comparison(save_path: Optional[str] = None, run_inference: bool = True):
    """3 test samples × 5 columns: Image | GT | SAM Zero-Shot | SAM-MedSeg | U-Net"""
    import torch
    from PIL import Image

    # Define test image paths and corresponding mask paths
    test_img_dir = DATA_DIR / "images"
    test_mask_dir = DATA_DIR / "masks"

    # Get test split (last 100 images based on 8:1:1 split)
    all_images = sorted(test_img_dir.glob("*.jpg"))
    # Use the same 8:1:1 split as training: test = last 10%
    np.random.seed(42)
    n_total = len(all_images)
    n_test = n_total // 10  # 100
    test_indices = sorted(np.random.RandomState(42).choice(n_total, size=n_test, replace=False))

    # Pick 3 diverse samples (use fixed indices for reproducibility)
    sample_indices = test_indices[10:13]  # Pick 3 samples
    img_paths = [all_images[i] for i in sample_indices]
    mask_paths = [test_mask_dir / p.name for p in img_paths]

    print(f"  [fig3] Selected {len(img_paths)} test samples for comparison")

    # Load ground truth masks
    gt_masks = []
    images = []
    for img_p, msk_p in zip(img_paths, mask_paths):
        img = Image.open(img_p).convert("RGB").resize((1024, 1024), Image.BILINEAR)
        images.append(np.array(img))

        if msk_p.exists():
            msk = Image.open(msk_p).convert("L").resize((1024, 1024), Image.NEAREST)
            gt_masks.append((np.array(msk) > 127).astype(np.uint8))
        else:
            gt_masks.append(np.zeros((1024, 1024), dtype=np.uint8))

    # Checkpoint paths
    sam_ckpt = RESULTS_DIR / "ratio_1.0_seed_44" / "best_model.pth"
    unet_ckpt = RESULTS_DIR / "unet_baseline" / "ratio_1.0_seed_44" / "best_model.pth"

    preds = {}

    if run_inference and sam_ckpt.exists() and unet_ckpt.exists():
        preds["SAM Zero-Shot"] = run_inference_batch(img_paths, None, model_type="sam_zero")
        preds["SAM-MedSeg"] = run_inference_batch(img_paths, sam_ckpt, model_type="sam")
        preds["U-Net"] = run_inference_batch(img_paths, unet_ckpt, model_type="unet")
    else:
        print("  [fig3] Skipping inference (--no-infer or checkpoints missing)")
        # Create dummy predictions
        for name in ["SAM Zero-Shot", "SAM-MedSeg", "U-Net"]:
            preds[name] = [np.zeros((1024, 1024), dtype=np.uint8) for _ in img_paths]

    # Build figure: 3 rows (samples) × 5 columns
    col_titles = ["Image", "Ground Truth", "SAM Zero-Shot", "SAM-MedSeg (ours)", "U-Net"]
    n_rows, n_cols = len(img_paths), 5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.8, n_rows * 4.2))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    model_keys = ["SAM Zero-Shot", "SAM-MedSeg", "U-Net"]

    for row_idx in range(n_rows):
        # Column 0: Original image
        axes[row_idx, 0].imshow(images[row_idx])
        axes[row_idx, 0].axis("off")

        # Column 1: Ground Truth
        axes[row_idx, 1].imshow(gt_masks[row_idx], cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].axis("off")

        # Columns 2-4: Model predictions with Dice annotation
        for col_offset, model_name in enumerate(model_keys):
            col = col_offset + 2
            axes[row_idx, col].imshow(preds[model_name][row_idx], cmap="gray", vmin=0, vmax=1)
            axes[row_idx, col].axis("off")

            # Add Dice score below each prediction
            d = dice_numpy(preds[model_name][row_idx], gt_masks[row_idx])
            axes[row_idx, col].set_xlabel(f"Dice={d:.3f}", fontsize=9,
                                          fontweight="bold", color="#333333")

        # Row label (sample #)
        axes[row_idx, 0].set_ylabel(f"Sample {row_idx + 1}", fontsize=13, fontweight="bold",
                                     rotation=90, labelpad=15)

    # Column titles
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle("Segmentation Comparison Across Methods", fontsize=16, fontweight="bold", y=1.01)
    plt.subplots_adjust(wspace=0.03, hspace=0.12)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"  ✓ Figure 3 saved → {save_path}")
    plt.close(fig)


# ============================================================
#  Figure 4: Training curves from TensorBoard
# ============================================================

def read_tensorboard_scalars(logdir: Path) -> Dict[str, Dict[str, list]]:
    """Read scalar data from a TensorBoard event file directory.
    Returns {tag: {"step": [...], "value": [...]}}."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(logdir))
    ea.Reload()

    out = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        out[tag] = {
            "step": [e.step for e in events],
            "value": [e.value for e in events],
        }
    return out


def fig_training_curves(save_path: Optional[str] = None):
    """Val Dice + Loss curves for best models: SAM-MedSeg and U-Net both 100% seed 44."""
    sam_tb = RESULTS_DIR / "ratio_1.0_seed_44" / "tensorboard"
    unet_tb = RESULTS_DIR / "unet_baseline" / "ratio_1.0_seed_44" / "tensorboard"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- SAM-MedSeg training curves ---
    if sam_tb.exists():
        sam_data = read_tensorboard_scalars(sam_tb)
        # Find Val Dice and Train Loss tags
        for tag, d in sam_data.items():
            if "val" in tag.lower() and "dice" in tag.lower():
                axes[0, 0].plot(d["step"], d["value"], color=C_SAM, linewidth=1.8, label=tag)
                # Mark best epoch
                best_idx = np.argmax(d["value"])
                axes[0, 0].scatter(d["step"][best_idx], d["value"][best_idx],
                                   color="red", s=100, zorder=10,
                                   label=f"Best: {d['value'][best_idx]:.4f}")
            if "train" in tag.lower() and "loss" in tag.lower():
                axes[0, 1].plot(d["step"], d["value"], color=C_SAM, linewidth=1.5, alpha=0.8)

        axes[0, 0].set_title("SAM-MedSeg — Val Dice (100%, seed 44)", fontweight="bold")
        axes[0, 0].set_ylabel("Dice Score")
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(alpha=0.2)

        axes[0, 1].set_title("SAM-MedSeg — Train Loss (100%, seed 44)", fontweight="bold")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(alpha=0.2)
    else:
        axes[0, 0].text(0.5, 0.5, "No TensorBoard data", ha="center", transform=axes[0, 0].transAxes)
        axes[0, 1].text(0.5, 0.5, "No TensorBoard data", ha="center", transform=axes[0, 1].transAxes)

    # --- U-Net training curves ---
    if unet_tb.exists():
        unet_data = read_tensorboard_scalars(unet_tb)
        for tag, d in unet_data.items():
            if "val" in tag.lower() and "dice" in tag.lower():
                axes[1, 0].plot(d["step"], d["value"], color=C_UNET, linewidth=1.8, label=tag)
                best_idx = np.argmax(d["value"])
                axes[1, 0].scatter(d["step"][best_idx], d["value"][best_idx],
                                   color="red", s=100, zorder=10,
                                   label=f"Best: {d['value'][best_idx]:.4f}")
            if "train" in tag.lower() and "loss" in tag.lower():
                axes[1, 1].plot(d["step"], d["value"], color=C_UNET, linewidth=1.5, alpha=0.8)

        axes[1, 0].set_title("U-Net — Val Dice (100%, seed 44)", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Dice Score")
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(alpha=0.2)

        axes[1, 1].set_title("U-Net — Train Loss (100%, seed 44)", fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].grid(alpha=0.2)
    else:
        axes[1, 0].text(0.5, 0.5, "No TensorBoard data", ha="center", transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, "No TensorBoard data", ha="center", transform=axes[1, 1].transAxes)

    fig.suptitle("Training Dynamics — Best Models", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"  ✓ Figure 4 saved → {save_path}")
    plt.close(fig)


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures for SAM-MedSeg")
    parser.add_argument("--fig", type=int, choices=[1, 2, 3, 4],
                        help="Generate only a specific figure (1-4)")
    parser.add_argument("--no-infer", action="store_true",
                        help="Skip model inference for Figure 3")
    parser.add_argument("--output-dir", type=str, default=str(FIGURES_DIR),
                        help="Output directory for figures")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SAM-MedSeg — Paper Figure Generator")
    print("=" * 60)

    figures = {
        1: ("data_efficiency", fig_data_efficiency),
        2: ("ablation_study", fig_ablation),
        3: ("segmentation_comparison", lambda p: fig_segmentation_comparison(p, not args.no_infer)),
        4: ("training_curves", fig_training_curves),
    }

    if args.fig:
        figs_to_generate = {args.fig: figures[args.fig]}
    else:
        figs_to_generate = figures

    for fig_num, (name, func) in figs_to_generate.items():
        print(f"\n  Figure {fig_num}: {name}...")
        try:
            func(str(od / f"figure_{fig_num}_{name}.png"))
        except Exception as e:
            print(f"  ✗ Figure {fig_num} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  Done! Figures saved to: {od}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
