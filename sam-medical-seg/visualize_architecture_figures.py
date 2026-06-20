#!/usr/bin/env python3
"""
Generate paper-quality architecture diagrams for the SAM-MedSeg project.
One script → 4 architecture figures → experiments/figures/

Usage:
    python visualize_architecture_figures.py              # all figures
    python visualize_architecture_figures.py --fig 1      # only figure 1
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Polygon
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

# ── Color palette ──
C_SAM     = "#2c7bb6"   # blue
C_LORA    = "#fdae61"   # orange
C_MED     = "#1a9641"   # green
C_PROMPT  = "#d7191c"   # red
C_FUSION  = "#5e3c99"   # purple
C_BG      = "#f7f7f7"
C_EDGE    = "#444444"
C_ARROW   = "#555555"
C_TEXT    = "#222222"


# ============================================================
#  Helpers
# ============================================================

def draw_box(ax, x, y, w, h, text, color, alpha=0.9, fontsize=10,
             text_color="white", bold=False, edge_color=None, linewidth=1.5):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor=edge_color or color,
        alpha=alpha, linewidth=linewidth, zorder=3,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.8, style="simple",
               zorder=2):
    """Draw an arrow between two points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            connectionstyle="arc3,rad=0",
        ),
        zorder=zorder,
    )


def draw_dashed_box(ax, x, y, w, h, text, color, alpha=0.15, fontsize=9):
    """Draw a dashed boundary box with a label."""
    rect = mpatches.Rectangle(
        (x - w/2, y - h/2), w, h,
        linewidth=1.5, linestyle="--", edgecolor=color,
        facecolor=color, alpha=alpha, zorder=1,
    )
    ax.add_patch(rect)
    ax.text(x, y + h/2 + 0.12, text, ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight="bold", zorder=3)


def draw_feature_block(ax, x, y, w, h, text, color, fontsize=8):
    """Draw a small feature tensor block."""
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=color, alpha=0.7, linewidth=1, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color="white", fontweight="bold", zorder=4)


# ============================================================
#  Figure 1: Overall Architecture
# ============================================================

def draw_figure_1_overall_architecture():
    """SAM-MedSeg overall framework diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    # ── Input image ──
    draw_box(ax, 2.0, 7.5, 2.2, 1.0,
             "Input Image\n3×1024×1024", C_EDGE, alpha=0.1,
             text_color=C_TEXT, fontsize=10, edge_color=C_EDGE)

    # ── SAM ViT-B Encoder (large block) ──
    draw_dashed_box(ax, 6.0, 7.5, 5.2, 3.6, "SAM ViT-B Image Encoder (93.7M)", C_SAM)
    draw_box(ax, 4.8, 8.3, 1.6, 0.7,
             "Frozen Layers\n(1-9)", "#888888", alpha=0.5, text_color="white")
    draw_box(ax, 7.2, 8.3, 1.6, 0.7,
             "Trainable\nLayers (10-12)", "#aaaaaa", alpha=0.7, text_color="white")

    # LoRA inside the encoder block
    draw_box(ax, 6.0, 7.0, 3.0, 0.8,
             "LoRA Adapter (qkv+proj, r=4)\nTrainable: 227K (0.25%)",
             C_LORA, fontsize=9, bold=True)

    # Feature outputs from encoder
    draw_feature_block(ax, 6.0, 5.9, 1.8, 0.45,
                       "Image Embed\n256×64×64", C_SAM, fontsize=7)

    # Intermediate features for MedDecoder
    for i, (xi, name) in enumerate([
        (4.2, "feat3"), (5.0, "feat6"), (5.8, "feat9"), (6.6, "feat12")
    ]):
        draw_feature_block(ax, xi, 6.5, 0.9, 0.35,
                          f"{name}\n768×64×64", "#5a9ed4", fontsize=6)

    # ── Arrows from input to encoder ──
    draw_arrow(ax, 3.15, 7.5, 3.6, 7.5)

    # ── Prompt Optimizer ──
    draw_box(ax, 9.5, 6.0, 2.4, 1.2,
             "Prompt Optimizer\n32×32 Grid Sampling\nIoU Filter + NMS",
             C_PROMPT, fontsize=9)

    # ── SAM Mask Decoder ──
    draw_box(ax, 12.5, 6.0, 2.0, 1.2,
             "SAM\nMask Decoder\n(2-layer Trans.)",
             C_SAM, fontsize=9)

    # ── Medical Decoder ──
    draw_box(ax, 9.5, 3.5, 2.4, 1.6,
             "Medical Decoder\nU-Net Skip-Connect\n↑×2 ↑×2 ↑×2 ↑×4\n→ 1024×1024",
             C_MED, fontsize=8)

    # ── Fusion ──
    draw_box(ax, 12.5, 2.5, 2.0, 0.9,
             "Learnable Fusion\nα=0.623\nα·SAM+(1-α)·Med",
             C_FUSION, fontsize=8)

    # ── Output ──
    draw_box(ax, 14.2, 2.5, 1.6, 0.9,
             "Output Mask\n1×1024×1024",
             C_EDGE, alpha=0.15, text_color=C_TEXT, fontsize=9,
             edge_color=C_EDGE)

    # ── Arrows connecting everything ──
    # Encoder → Prompt Optimizer
    draw_arrow(ax, 7.5, 6.3, 8.3, 6.15)
    # Encoder → SAM Mask Decoder
    draw_arrow(ax, 7.5, 6.6, 11.5, 6.3)
    # Prompt Optimizer → SAM Mask Decoder
    draw_arrow(ax, 10.7, 6.0, 11.5, 6.0)
    # Encoder features → MedDecoder
    draw_arrow(ax, 7.5, 5.7, 8.9, 4.3)
    # MedDecoder → Fusion
    draw_arrow(ax, 10.0, 2.8, 11.5, 2.7)
    # SAM Decoder → Fusion
    draw_arrow(ax, 12.5, 5.4, 12.5, 4.1)
    # Fusion → Output
    draw_arrow(ax, 13.5, 2.5, 13.4, 2.5)

    # ── Annotations ──
    ax.text(8.5, 9.0, "① SAM Backbone", fontsize=10, color=C_SAM, fontweight="bold")
    ax.text(4.2, 7.65, "② LoRA", fontsize=10, color=C_LORA, fontweight="bold")
    ax.text(10.3, 5.55, "③ Prompt", fontsize=10, color=C_PROMPT, fontweight="bold")
    ax.text(14.0, 6.3, "④ SAM Dec.", fontsize=10, color=C_SAM, fontweight="bold")
    ax.text(10.3, 3.15, "⑤ MedDecoder", fontsize=10, color=C_MED, fontweight="bold")
    ax.text(14.0, 2.15, "⑥ Fusion", fontsize=10, color=C_FUSION, fontweight="bold")

    # ── Title ──
    ax.set_title("Figure 1: SAM-MedSeg Overall Architecture", fontsize=15,
                 fontweight="bold", pad=15)

    # ── Legend (param info) ──
    info_text = (
        "Total: 94.5M params | Trainable: 4.84M (5.12%) | Frozen: 89.7M\n"
        "Training: AMP fp16 + Gradient Checkpointing | Peak VRAM: 6.16 GB"
    )
    ax.text(8.0, 0.5, info_text, ha="center", va="center", fontsize=9,
            style="italic", color="#666666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                      edgecolor="#cccccc"))

    path = FIGURES_DIR / "figure_arch_1_overall.png"
    fig.savefig(path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ============================================================
#  Figure 2: LoRA Module Detail
# ============================================================

def draw_figure_2_lora_detail():
    """LoRA low-rank decomposition diagram for QKV projection."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    # ── Left side: Input ──
    draw_box(ax, 1.5, 3.5, 1.4, 0.8,
             "Input\nx ∈ ℝᵈ", C_EDGE, alpha=0.1,
             text_color=C_TEXT, fontsize=10, edge_color=C_EDGE)

    # ── Center: Original weight matrix ──
    # W0 box (frozen)
    draw_box(ax, 5.0, 4.5, 2.8, 1.8,
             "Frozen Pre-trained\nWeight\nW₀ ∈ ℝᵈˣᵈ",
             "#888888", alpha=0.5, text_color="white", fontsize=10)

    # ── LoRA decomposition ──
    draw_dashed_box(ax, 9.0, 4.5, 4.0, 2.4, "LoRA Adapter (Trainable)", C_LORA)

    # A matrix
    draw_box(ax, 7.5, 4.5, 1.2, 1.4,
             "A\n∈ ℝʳˣᵈ",
             C_LORA, alpha=0.6, fontsize=10)

    # B matrix
    draw_box(ax, 9.5, 4.5, 1.2, 1.4,
             "B\n∈ ℝᵈˣʳ",
             C_LORA, alpha=0.8, fontsize=10)

    # Rank annotation
    ax.text(8.5, 5.5, "r ≪ d\n(r=4)", ha="center", va="center",
            fontsize=9, color=C_LORA, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=C_LORA, alpha=0.8))

    # ── Right: Output ──
    draw_box(ax, 12.0, 4.0, 1.6, 1.0,
             "ΔW = A·B\nΔW ∈ ℝᵈˣᵈ",
             C_LORA, alpha=0.3, text_color=C_TEXT, fontsize=9, edge_color=C_LORA)

    draw_box(ax, 12.0, 2.0, 1.6, 0.9,
             "Output\nh = W₀x + ΔWx",
             C_EDGE, alpha=0.1, text_color=C_TEXT, fontsize=9, edge_color=C_EDGE)

    # ── Arrows ──
    draw_arrow(ax, 2.2, 3.5, 3.6, 4.1)   # input → W0
    draw_arrow(ax, 2.2, 3.5, 6.9, 4.1)   # input → A
    draw_arrow(ax, 6.4, 3.5, 8.9, 3.5)   # W0 → output (via +)
    draw_arrow(ax, 8.1, 3.5, 10.1, 3.8)  # A → B
    draw_arrow(ax, 10.1, 3.8, 11.2, 3.8) # B → ΔW
    draw_arrow(ax, 12.0, 3.5, 12.0, 2.9)  # ΔW → output

    # Plus sign
    ax.text(10.8, 3.5, "+", fontsize=14, fontweight="bold", color=C_ARROW,
            ha="center", va="center")

    # ── Bottom: SAM-specific detail ──
    ax.text(7.0, 1.2,
            "SAM ViT-B uses fused QKV projection (attn.qkv) — "
            "LoRA applied to qkv and proj modules\n"
            "r=4, α=16, dropout=0.1 | "
            "Only 227K trainable params (0.25% of ViT-B encoder)",
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                      edgecolor="#dddddd"))

    # ── QKV detail boxes ──
    for i, (xi, label) in enumerate([
        (3.5, "Q"), (5.0, "K"), (6.5, "V")
    ]):
        draw_box(ax, xi, 2.2, 0.8, 0.6, label, C_SAM, alpha=0.6, fontsize=9)
        if i > 0:
            ax.plot([xi - 0.55, xi - 0.55], [2.5, 3.5], color=C_ARROW, lw=1,
                    zorder=2)
        else:
            ax.plot([xi, xi], [2.5, 3.5], color=C_ARROW, lw=1, zorder=2)

    ax.text(5.0, 2.7, "Fused QKV\nProjection", ha="center", va="center",
            fontsize=8, color=C_TEXT)

    ax.set_title("Figure 2: LoRA Low-Rank Adaptation Detail", fontsize=15,
                 fontweight="bold", pad=15)

    path = FIGURES_DIR / "figure_arch_2_lora.png"
    fig.savefig(path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ============================================================
#  Figure 3: Prompt Optimization Flow
# ============================================================

def draw_figure_3_prompt_flow():
    """Prompt optimization strategy flow chart."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    # ── Step 1: Image input ──
    draw_box(ax, 1.5, 5.5, 1.8, 0.9,
             "Input Image\n1024×1024", C_EDGE, alpha=0.1,
             text_color=C_TEXT, fontsize=10, edge_color=C_EDGE)

    # ── Step 2: Grid Sampling ──
    draw_box(ax, 4.5, 5.5, 2.0, 1.4,
             "32×32 Uniform\nGrid Sampling\n(1024 points)",
             C_PROMPT, fontsize=9)

    # ── Step 3: Per-point SAM inference ──
    draw_box(ax, 8.0, 5.5, 2.4, 1.4,
             "SAM Per-Point\nForward Pass\n→ Mask + IoU",
             C_SAM, fontsize=9)

    # ── Step 4: IoU Filtering ──
    draw_box(ax, 11.0, 5.5, 2.0, 1.0,
             "IoU > 0.5\nThreshold\nFilter",
             C_PROMPT, alpha=0.7, fontsize=9)

    # ── Step 5: NMS ──
    draw_box(ax, 12.0, 3.5, 1.8, 1.2,
             "NMS\nIoU > 0.7\nSuppress",
             C_PROMPT, alpha=0.5, text_color=C_TEXT, fontsize=9,
             edge_color=C_PROMPT)

    # ── Step 6: Final prompts ──
    draw_box(ax, 9.5, 3.5, 1.8, 1.0,
             "Filtered\nPrompt Set\n(pts + labels)",
             C_FUSION, fontsize=9)

    # ── Step 7: Mask Decoder ──
    draw_box(ax, 6.5, 3.5, 1.8, 1.0,
             "SAM Mask\nDecoder\n(multimask=False)",
             C_SAM, fontsize=9)

    # ── Step 8: Output ──
    draw_box(ax, 3.5, 3.5, 1.8, 0.9,
             "Final\nSegmentation\nMask",
             C_EDGE, alpha=0.1, text_color=C_TEXT, fontsize=9,
             edge_color=C_EDGE)

    # ── Arrows (main flow) ──
    draw_arrow(ax, 2.4, 5.5, 3.5, 5.5)
    draw_arrow(ax, 5.5, 5.5, 6.8, 5.5)
    draw_arrow(ax, 9.2, 5.5, 10.0, 5.5)
    draw_arrow(ax, 11.0, 4.9, 11.5, 4.1)
    draw_arrow(ax, 11.5, 4.1, 10.4, 3.7)
    draw_arrow(ax, 8.6, 3.5, 7.4, 3.5)
    draw_arrow(ax, 5.6, 3.5, 4.4, 3.5)

    # ── Rejected path ──
    ax.annotate(
        "IoU ≤ 0.5\nRejected", xy=(11.0, 5.0), xytext=(13.0, 6.3),
        arrowprops=dict(arrowstyle="->", color="#999999", lw=1.2,
                        connectionstyle="arc3,rad=0.3"),
        fontsize=8, color="#999999", ha="center",
    )

    # ── Grid visualization (small inset) ──
    grid_ax = ax.inset_axes([0.02, 0.05, 0.18, 0.30])
    grid_ax.set_xlim(0, 32)
    grid_ax.set_ylim(0, 32)
    grid_ax.set_aspect("equal")
    grid_ax.axis("off")
    for i in range(0, 33, 4):
        for j in range(0, 33, 4):
            grid_ax.plot(i, j, "o", markersize=2, color=C_PROMPT, alpha=0.5)
    # highlight some "selected" points
    rng = np.random.default_rng(42)
    sel_x = rng.integers(0, 33, size=15)
    sel_y = rng.integers(0, 33, size=15)
    grid_ax.plot(sel_x, sel_y, "o", markersize=4, color="#d7191c", alpha=0.8)
    grid_ax.set_title("Grid Points (32×32)\n● Selected  ● Rejected",
                      fontsize=7, pad=2)

    # ── Step numbers ──
    steps = [
        (1.5, 6.1, "① Input"),
        (4.5, 6.4, "② Grid"),
        (8.0, 6.4, "③ Inference"),
        (11.0, 6.2, "④ Filter"),
        (12.0, 4.3, "⑤ NMS"),
        (9.5, 4.2, "⑥ Prompts"),
        (6.5, 4.2, "⑦ Decode"),
        (3.5, 4.1, "⑧ Output"),
    ]
    for x, y, label in steps:
        ax.text(x, y, label, fontsize=8, color=C_TEXT, ha="center",
                fontweight="bold")

    ax.set_title("Figure 3: Prompt Optimization Strategy", fontsize=15,
                 fontweight="bold", pad=15)

    # ── Bottom note ──
    ax.text(7.0, 1.5,
            "Strategy: Grid Sampling (setting A in paper) — "
            "Per-point SAM inference → IoU filtering → NMS dedup → "
            "Batch inference with filtered prompts",
            ha="center", va="center", fontsize=9, style="italic", color="#666666")

    path = FIGURES_DIR / "figure_arch_3_prompt_flow.png"
    fig.savefig(path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ============================================================
#  Figure 4: Medical Decoder (U-Net Skip-Connection)
# ============================================================

def draw_figure_4_med_decoder():
    """U-Net style Medical Decoder architecture with skip connections."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    # ── Encoder features (left side) ──
    encoder_features = [
        (2.0, 7.2, "feat3\nLayer 3\n768×64×64", "#2c7bb6"),
        (2.0, 5.7, "feat6\nLayer 6\n768×64×64", "#4d9ac7"),
        (2.0, 4.2, "feat9\nLayer 9\n768×64×64", "#6eb3d8"),
        (2.0, 2.7, "feat12\nLayer 12\n768×64×64", "#8fcce9"),
    ]

    for x, y, text, color in encoder_features:
        draw_box(ax, x, y, 1.6, 1.0, text, color, fontsize=8)

    # Label for encoder side
    ax.text(2.0, 8.0, "SAM Encoder\nIntermediate Features",
            ha="center", fontsize=10, fontweight="bold", color=C_SAM)

    # ── 1×1 Conv projections ──
    proj_x = 4.2
    for _, y, _, color in encoder_features:
        draw_feature_block(ax, proj_x, y, 1.2, 0.55,
                          "1×1 Conv\n→ 64 ch", color, fontsize=6)

    # ── Decoder path (center-right) ──
    # Bottleneck processing
    draw_box(ax, 7.0, 2.7, 1.8, 0.8,
             "Bottleneck\nConvBlock", C_MED, alpha=0.5, fontsize=8)

    # Decoder blocks
    decoder_blocks = [
        (7.0, 4.2, "UpSample ×2\nConvBlock\n+ concat(feat9)", C_MED, 0.6),
        (7.0, 5.7, "UpSample ×2\nConvBlock\n+ concat(feat6)", C_MED, 0.7),
        (7.0, 7.2, "UpSample ×2\nConvBlock\n+ concat(feat3)", C_MED, 0.8),
    ]
    for x, y, text, color, alpha in decoder_blocks:
        draw_box(ax, x, y, 2.0, 1.0, text, color, alpha=alpha, fontsize=7)

    # Final upsample
    draw_box(ax, 9.5, 7.2, 1.4, 0.8,
             "UpSample\n×4", C_MED, alpha=0.9, fontsize=8)

    # Output
    draw_box(ax, 12.0, 7.2, 1.6, 0.9,
             "Output\n1×1024×1024",
             C_EDGE, alpha=0.1, text_color=C_TEXT, fontsize=9, edge_color=C_EDGE)

    # ── Skip connections ──
    for (_, ey, _, color) in encoder_features:
        # 1×1 conv to decoder
        draw_arrow(ax, 3.0, ey, 3.6, ey, color=color, lw=1.2)

    # Horizontal arrows (encode → decode, skip connections)
    skip_offsets = [
        (5.4, 7.2, 6.0, 7.2),   # feat3 → decoder block 1
        (5.4, 5.7, 6.0, 5.7),   # feat6 → decoder block 2
        (5.4, 4.2, 6.0, 4.2),   # feat9 → decoder block 3
    ]
    for x1, y1, x2, y2 in skip_offsets:
        draw_arrow(ax, x1, y1, x2, y2, color=C_MED, lw=1.5)

    # Vertical upsampling arrows (between decoder blocks)
    for y_bot, y_top in [(2.7, 3.7), (3.7, 5.2), (5.2, 6.2)]:
        ax.annotate(
            "", xy=(7.0, y_top), xytext=(7.0, y_bot + 0.4),
            arrowprops=dict(arrowstyle="->", color=C_MED, lw=1.8,
                          connectionstyle="arc3,rad=0"),
        )

    # ── Final arrow ──
    draw_arrow(ax, 8.0, 7.2, 10.2, 7.2, color=C_MED, lw=1.8)
    draw_arrow(ax, 10.9, 7.2, 11.2, 7.2, color=C_ARROW, lw=1.8)

    # ── ConvBlock legend ──
    legend_box = ax.inset_axes([0.70, 0.05, 0.25, 0.18])
    legend_box.set_xlim(0, 4)
    legend_box.set_ylim(0, 4)
    legend_box.axis("off")
    legend_box.text(2, 3.5, "ConvBlock Detail", fontsize=8, fontweight="bold",
                    ha="center")
    layers = ["Conv2d(3×3)", "BatchNorm", "ReLU",
              "Conv2d(3×3)", "BatchNorm", "ReLU"]
    for i, layer in enumerate(layers):
        y_pos = 3.0 - i * 0.45
        legend_box.add_patch(FancyBboxPatch(
            (1.0, y_pos - 0.15), 2.0, 0.35,
            boxstyle="round,pad=0.05",
            facecolor=C_MED, alpha=0.15 + i * 0.08, edgecolor=C_MED,
            linewidth=0.8,
        ))
        legend_box.text(2, y_pos + 0.02, layer, fontsize=6, ha="center",
                        va="center", color=C_TEXT)

    # ── Dimension annotations ──
    dim_labels = [
        (8.5, 8.0, "64×64×64"),
        (8.5, 6.5, "128×128×64"),
        (8.5, 5.0, "256×256×64"),
        (8.5, 3.5, "512×512×64"),
    ]
    for x, y, text in dim_labels:
        ax.text(x, y, text, fontsize=7, color=C_MED, ha="center",
                style="italic")

    ax.set_title("Figure 4: Medical Skip-Connection Decoder (U-Net style)",
                 fontsize=15, fontweight="bold", pad=15)

    # ── Bottom info ──
    ax.text(7.0, 1.0,
            "MedDecoder: 4-level U-Net decoder with skip connections from SAM ViT-B intermediate layers | "
            "Base channels: 64 | Trainable: ~30.4M params",
            ha="center", va="center", fontsize=9, style="italic", color="#666666")

    path = FIGURES_DIR / "figure_arch_4_med_decoder.png"
    fig.savefig(path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ============================================================
#  Main
# ============================================================

FIGURES = {
    "1": ("Overall Architecture", draw_figure_1_overall_architecture),
    "2": ("LoRA Detail", draw_figure_2_lora_detail),
    "3": ("Prompt Flow", draw_figure_3_prompt_flow),
    "4": ("MedDecoder", draw_figure_4_med_decoder),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate SAM-MedSeg architecture figures"
    )
    parser.add_argument(
        "--fig", type=str, default=None,
        help="Figure number to generate (1-4). Omit for all."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SAM-MedSeg Architecture Figure Generator")
    print(f"Output: {FIGURES_DIR}")
    print("=" * 60)

    if args.fig:
        name, func = FIGURES[args.fig]
        print(f"\nGenerating Figure {args.fig}: {name}...")
        func()
    else:
        for num, (name, func) in FIGURES.items():
            print(f"\nGenerating Figure {num}: {name}...")
            func()

    print(f"\n{'=' * 60}")
    print(f"Done! {len(FIGURES) if not args.fig else 1} figure(s) saved to {FIGURES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
