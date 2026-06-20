# SAM-MedSeg: Few-Shot Medical Image Segmentation with Improved SAM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This project improves Meta's Segment Anything Model (SAM) for **few-shot medical image segmentation** through three components:

- **LoRA (Low-Rank Adaptation)** — 227K trainable parameters for parameter-efficient domain transfer
- **Automatic Prompt Optimization** — Grid sampling + IoU filtering + NMS for fully automatic inference
- **Medical Skip-Connection Decoder** — U-Net style decoder fusing multi-scale SAM encoder features

---

## Highlights

| Metric | Value |
|--------|-------|
| **Best SAM-MedSeg Dice** | 0.652 (100% data, seed 44) |
| **Best U-Net Dice** | 0.725 (100% data, seed 44) |
| **Few-Shot Advantage (1% data)** | SAM-MedSeg +4.0 pp over U-Net |
| **Crossover Point** | 5%–10% data (~40–80 images) |
| **Total Experiments** | 27 (SAM-MedSeg 14 + SAM 1 + U-Net 12) |
| **Trainable Params** | 4.84M / 94.5M (5.12%) |
| **Peak VRAM** | 6.16 GB (RTX 4060 Laptop) |

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA 12.1 (or 11.8)
- GPU with ≥8 GB VRAM

### Setup

```bash
# 1. Create environment
conda create -n sam-medseg python=3.10 -y
conda activate sam-medseg

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download SAM checkpoint
mkdir -p checkpoints
wget -O checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# 4. Prepare dataset
python data/prepare_data.py
```

### Reproduce Results

```bash
# Set environment variable for memory efficiency
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Train SAM-MedSeg (best config: 100% data)
python train.py --few-shot-ratio 1.0 --seed 44

# Train with different data ratios
python train.py --few-shot-ratio 0.01 --seed 42   # 1% data
python train.py --few-shot-ratio 0.05 --seed 42   # 5% data
python train.py --few-shot-ratio 0.1  --seed 42   # 10% data

# Run ablation experiments
python train.py --few-shot-ratio 0.05 --seed 42 --ablation lora_only
python train.py --few-shot-ratio 0.05 --seed 42 --ablation med_decoder_only

# U-Net baseline
python unet_baseline.py --few-shot-ratio 1.0 --seed 44

# SAM zero-shot baseline
python sam_zero_shot_baseline.py

# Evaluate best model
python evaluate.py \
  --checkpoint experiments/results/ratio_1.0_seed_44/best_model.pth

# Generate all paper figures
python visualize_paper_figures.py
python visualize_architecture_figures.py

# Generate paper and PPT
python generate_paper.py
python generate_ppt.py

# Single image inference
python inference.py --image path/to/image.jpg \
  --checkpoint experiments/results/ratio_1.0_seed_44/best_model.pth
```

## Project Structure

```
sam-medical-seg/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── configs/
│   └── config.yaml                   # Centralized hyperparameters
├── data/
│   ├── prepare_data.py               # Dataset download & preprocessing
│   ├── dataset.py                    # PyTorch Dataset class
│   └── transforms.py                 # Augmentation pipeline (MONAI)
├── models/
│   ├── sam_backbone.py               # SAM backbone loader with freeze control
│   ├── lora_adapter.py               # LoRA fine-tuning module
│   ├── prompt_optimizer.py           # Automatic prompt generation
│   ├── med_decoder.py                # U-Net style skip-connection decoder
│   └── full_model.py                 # End-to-end integrated SAMMedSeg
├── losses/
│   └── combined_loss.py              # Dice + BCE combined loss
├── utils/
│   ├── metrics.py                    # Dice, IoU, Precision, Recall
│   └── visualize.py                  # Result visualization tools
├── train.py                          # Training script (supports resume, ablation, few-shot)
├── evaluate.py                       # Evaluation & metric computation
├── inference.py                      # Single-image inference CLI
├── inference_demo.ipynb              # Interactive inference demo notebook
├── unet_baseline.py                  # U-Net baseline training
├── sam_zero_shot_baseline.py         # SAM zero-shot evaluation
├── visualize_paper_figures.py        # Generate 4 data figures
├── visualize_architecture_figures.py # Generate 4 architecture figures
├── generate_paper.py                 # Generate course paper (.docx)
├── generate_ppt.py                   # Generate presentation (.pptx)
├── experiments/
│   ├── results/                      # 27 experiment outputs
│   ├── figures/                      # 8 paper figures
│   └── *.md                          # Experiment reports & logs
├── paper/
│   ├── SAM-MedSeg_paper_draft.docx    # Course paper (~4,400 words)
│   ├── SAM-MedSeg_presentation.pptx   # Presentation (13 slides)
│   └── references.bib                 # BibTeX references (17 entries)
└── checkpoints/                      # Model weights (gitignored)
```

## Experiment Matrix

| ID | Method | Description | Best Dice |
|----|--------|-------------|:---------:|
| E1 | SAM Zero-Shot | Grid prompts, no fine-tuning | 0.271 |
| E2 | U-Net (from scratch) | Classic CNN baseline ×12 configs | 0.725 |
| E3 | SAM + LoRA only | Our variant, no MedDecoder | 0.452 |
| E4 | SAM + MedDecoder only | No LoRA adaptation | 0.426 |
| E5 | **SAM + LoRA + MedDecoder** | **Full SAM-MedSeg** ×12 configs | **0.652** |

### Data Efficiency Settings

| Ratio | Train Images | Purpose |
|:-----:|:------------:|---------|
| 1% | 8 | Extreme few-shot |
| 5% | 40 | Moderate few-shot |
| 10% | 80 | Typical few-shot |
| 100% | 800 | Full data comparison |

Each setting: 3 seeds (42, 43, 44) for statistical rigor.

## Key Results

### Data Efficiency Comparison

| Data % | U-Net | SAM-MedSeg | Winner | Gap |
|:------:|:-----:|:----------:|:------:|:---:|
| 1% | 0.385 ±0.031 | **0.425** ±0.026 | SAM-MedSeg | +4.0 pp |
| 5% | 0.443 ±0.020 | **0.467** ±0.019 | SAM-MedSeg | +2.4 pp |
| 10% | **0.509** ±0.009 | 0.484 ±0.016 | U-Net | +2.5 pp |
| 100% | **0.719** ±0.007 | 0.640 ±0.012 | U-Net | +7.9 pp |

**Crossover point: 5% → 10% data.** SAM-MedSeg is optimal when ≤50 annotated images are available.

### Ablation Study (5% data, seed 42)

| Configuration | Dice | Precision | Recall | Trainable |
|--------------|:----:|:---------:|:------:|:---------:|
| Full Model | 0.448 | 0.421 | 0.697 | 4.84M |
| LoRA Only | **0.452** | **0.441** | 0.658 | **227K** |
| MedDecoder Only | 0.426 | 0.366 | **0.772** | 30.4M |

## Figures

### Data Figures (`experiments/figures/`)
- `figure_1_data_efficiency.png` — Data efficiency curves with error bars
- `figure_2_ablation_study.png` — Ablation study bar charts
- `figure_3_segmentation_comparison.png` — Visual comparison across methods
- `figure_4_training_curves.png` — Training loss and Dice curves

### Architecture Figures (`experiments/figures/`)
- `figure_arch_1_overall.png` — Overall SAM-MedSeg architecture
- `figure_arch_2_lora.png` — LoRA low-rank decomposition detail
- `figure_arch_3_prompt_flow.png` — Prompt optimization strategy
- `figure_arch_4_med_decoder.png` — Medical decoder U-Net structure

## Deliverables

- [x] Complete source code
- [x] Course paper (~4,400 words, 17 references)
- [x] Presentation slides (13 slides)
- [x] 8 paper-quality figures
- [x] 27 experiment runs with full metrics
- [x] Inference demo (CLI + Jupyter Notebook)
- [x] BibTeX reference file (17 entries)
- [x] Pretrained model weights (best checkpoint)

## References

1. Kirillov et al., "Segment Anything", ICCV 2023
2. Ma et al., "Segment Anything in Medical Images", Nature Communications 2024
3. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
4. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
5. Isensee et al., "nnU-Net: Self-configuring Method for Biomedical Image Segmentation", Nature Methods 2021
6. Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset", MMM 2020

Full BibTeX: `paper/references.bib`

## License

MIT License
