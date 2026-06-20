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
期末大作业/                              # 课程根目录
├── README.md                           # 项目总览 README
├── .gitignore                          # Git 忽略规则
├── 期末大作业选题方案1.md                # 选题方案文档
├── 项目实施计划.md                      # 项目实施计划
├── 深度学习与计算机视觉_课程评分标准及大作业说明_.md
├── 附件2-深度学习与计算机视觉课程大作业参考模板.docx
├── kvasir-seg/                         # Kvasir-SEG 原始数据集
│   └── Kvasir-SEG/
│       └── images/                     # 1000 张息肉内窥镜图像 (.jpg)
│           ├── cju0qkwl35piu0993l0dewei2.jpg
│           ├── cju0qoxqj9q6s0835b43399p4.jpg
│           └── ... (共 1000 张)
│
└── sam-medical-seg/                    # SAM-MedSeg 核心代码
    │
    ├── 📄 根目录文件
    ├── README.md                       # 项目 README（本文件）
    ├── requirements.txt                # Python 依赖列表 (torch, MONAI, etc.)
    ├── setup.sh                        # 一键环境搭建脚本
    ├── .gitignore                      # Git 忽略规则（checkpoints/ 等）
    │
    ├── ⚙️ configs/                     # 配置模块
    │   └── config.yaml                 # 集中式超参数配置
    │       · 模型: SAM ViT-B, LoRA rank=4, MedDecoder channels=[256,128,64,32]
    │       · 训练: lr=1e-4, batch_size=4, epochs=50, early_stop=10
    │       · 数据: img_size=1024×1024, Kvasir-SEG 800/200 train/val split
    │       · 提示: grid_size=16, iou_threshold=0.7, nms_threshold=0.5
    │
    ├── 📦 data/                        # 数据处理模块
    │   ├── __init__.py                 # 数据模块导出
    │   ├── prepare_data.py             # 数据集下载 & 预处理 (288 lines)
    │   │   · Kvasir-SEG 数据集下载与解压
    │   │   · 图像归一化与掩码二值化
    │   │   · 分层 train/val 划分 (800/200)
    │   ├── dataset.py                  # PyTorch Dataset 类 (187 lines)
    │   │   · KvasirSEGDataset — 主数据集类
    │   │   · FewShotDataset — 少量样本采样包装器
    │   │   · 支持按比例/按数量采样，固定 seed 复现
    │   └── transforms.py               # 数据增强流水线 MONAI (135 lines)
    │       · 训练: RandomResizedCrop, RandomFlip, RandomRotate, ColorJitter
    │       · 验证: Resize+CenterCrop (确定性)
    │       · 归一化: ImageNet mean/std 或 Kvasir 统计量
    │
    ├── 🧠 models/                      # 模型模块（核心创新）
    │   ├── __init__.py                 # 模型模块导出
    │   ├── sam_backbone.py             # SAM 骨干网络加载器 (259 lines)
    │   │   · SAM ViT-B 图像编码器加载 (91M 参数)
    │   │   · 精细 freeze/ unfreeze 控制
    │   │   · Prompt encoder & Mask decoder 封装
    │   │   · 多尺度特征提取 (4 层 ViT 特征)
    │   ├── lora_adapter.py             # LoRA 低秩适配模块 (185 lines)
    │   │   · Q/K/V 投影层注入 LoRA (rank=4)
    │   │   · 仅 227K 可训练参数 (SAM 的 0.24%)
    │   │   · 适配 ViT 所有 12 层 Attention
    │   ├── prompt_optimizer.py          # 自动提示优化器 (248 lines)
    │   │   · 网格采样 (16×16) → 256 候选提示点
    │   │   · IoU 过滤: 保留 top-K (K=5) 高置信度提示
    │   │   · NMS 去重: 空间距离合并重复提示
    │   │   · 支持 point/box/mask 三种提示模式
    │   ├── med_decoder.py              # 医学跳跃连接解码器 (225 lines)
    │   │   · U-Net 风格解码器，融合 4 层 SAM 编码器特征
    │   │   · 通道配置: [256, 128, 64, 32]
    │   │   · 每层: Upsample + ConvBlock + SkipConnection
    │   │   · 输出头: 1×1 Conv → Sigmoid
    │   ├── unet.py                     # 标准 U-Net 实现 (162 lines)
    │   │   · 4 层编码器-解码器 (基线对比用)
    │   │   · 双卷积块 + 跳跃连接 + 上采样
    │   └── full_model.py               # SAMMedSeg 端到端集成 (285 lines)
    │       · 整合 SAM Backbone + LoRA + PromptOptimizer + MedDecoder
    │       · 前向: Image → SAM Encoder → LoRA → MedDecoder → Mask
    │       · 训练/推理模式切换
    │       · 模型保存/加载 (checkpoint 管理)
    │
    ├── 📉 losses/                      # 损失函数模块
    │   ├── __init__.py                 # 损失模块导出
    │   └── combined_loss.py            # Dice + BCE 组合损失 (178 lines)
    │       · DiceLoss (smooth=1.0)
    │       · BCEWithLogitsLoss (pos_weight 可调)
    │       · 加权组合: α·Dice + β·BCE (默认 α=0.5, β=0.5)
    │       · 支持 class weights 处理不平衡
    │
    ├── 🔧 utils/                       # 工具模块
    │   ├── __init__.py                 # 工具模块导出
    │   ├── metrics.py                  # 评估指标计算 (237 lines)
    │   │   · Dice Score, IoU (Jaccard), Precision, Recall
    │   │   · Hausdorff Distance (HD95)
    │   │   · 批量统计: mean ± std, CI 95%
    │   │   · 混淆矩阵汇总
    │   └── visualize.py                # 可视化工具 (324 lines)
    │       · 分割结果叠加 (overlay)
    │       · 多方法对比图 (U-Net / SAM / SAMMedSeg)
    │       · 训练曲线 (loss, dice per epoch)
    │       · 误差图 (FP/FN 区域高亮)
    │
    ├── 🚂 训练 & 评估脚本
    ├── train.py                        # 主训练脚本 (575 lines)
    │   · 完整训练流水线 (train + val per epoch)
    │   · 支持 --few-shot-ratio (0.01/0.05/0.1/1.0)
    │   · 支持 --seed (42/43/44) 多种子复现
    │   · 支持 --ablation (lora_only / med_decoder_only)
    │   · 支持 --resume 断点续训
    │   · 自动保存 best_model / latest_checkpoint
    │   · TensorBoard 日志记录
    ├── evaluate.py                     # 评估脚本 (299 lines)
    │   · 加载 checkpoint 计算全部指标
    │   · 输出 JSON results + 可视化分割图
    │   · 支持单模型评估和批量对比
    ├── inference.py                    # 单图推理 CLI (309 lines)
    │   · python inference.py --image <path> --checkpoint <path>
    │   · 支持 GPU/CPU 推理
    │   · 输出: 预测掩码 + overlay 可视化 + 置信度
    ├── inference_demo.ipynb            # 交互式推理 Jupyter Notebook
    ├── unet_baseline.py                # U-Net 基线训练 (432 lines)
    │   · 独立 U-Net 训练流程 (非 SAM 依赖)
    │   · 相同的 few-shot 数据设置和评估标准
    │   · 12 组实验: 4 ratios × 3 seeds
    ├── sam_zero_shot_baseline.py       # SAM 零样本基线 (297 lines)
    │   · SAM 原始权重 + 网格提示 (无训练)
    │   · 零样本 Dice 基准线: 0.271
    │
    ├── 📊 可视化 & 生成脚本
    ├── visualize_paper_figures.py      # 论文数据图生成 (4 张)
    │   · figure_1: 数据效率曲线 (error bar)
    │   · figure_2: 消融实验柱状图
    │   · figure_3: 分割效果对比图
    │   · figure_4: 训练曲线
    ├── visualize_architecture_figures.py # 架构图生成 (4 张)
    │   · figure_arch_1: 整体架构
    │   · figure_arch_2: LoRA 分解细节
    │   · figure_arch_3: 提示优化流程
    │   · figure_arch_4: 医学解码器结构
    ├── generate_paper.py               # 课程论文生成 (.docx)
    ├── generate_ppt.py                 # 演示文稿生成 (.pptx, 13 slides)
    │
    ├── 🔬 experiments/                 # 实验产出
    │   ├── figures/                    # 8 张论文级图表 (.png)
    │   │   ├── figure_1_data_efficiency.png
    │   │   ├── figure_2_ablation_study.png
    │   │   ├── figure_3_segmentation_comparison.png
    │   │   ├── figure_4_training_curves.png
    │   │   ├── figure_arch_1_overall.png
    │   │   ├── figure_arch_2_lora.png
    │   │   ├── figure_arch_3_prompt_flow.png
    │   │   └── figure_arch_4_med_decoder.png
    │   ├── results/                    # 27 组实验输出
    │   │   ├── evaluation_results.json # 汇总评估结果
    │   │   ├── sam_zero_shot/          # SAM 零样本 (1 组)
    │   │   │   └── results.json
    │   │   ├── ratio_0.01_seed_{42,43,44}/     # SAM-MedSeg 1% data (3 组)
    │   │   ├── ratio_0.05_seed_{42,43,44}/     # SAM-MedSeg 5% data (3 组)
    │   │   ├── ratio_0.05_seed_42_ablation_lora_only/        # 消融: 仅 LoRA
    │   │   ├── ratio_0.05_seed_42_ablation_med_decoder_only/  # 消融: 仅 MedDecoder
    │   │   ├── ratio_0.1_seed_{42,43,44}/      # SAM-MedSeg 10% data (3 组)
    │   │   ├── ratio_1.0_seed_{42,43,44}/      # SAM-MedSeg 100% data (3 组)
    │   │   └── unet_baseline/          # U-Net 基线 (12 组)
    │   │       ├── ratio_0.01_seed_{42,43,44}/
    │   │       ├── ratio_0.05_seed_{42,43,44}/
    │   │       ├── ratio_0.1_seed_{42,43,44}/
    │   │       └── ratio_1.0_seed_{42,43,44}/
    │   │   # 每组含: best_model.pth, latest_checkpoint.pth,
    │   │   #           config.yaml, results.json, tensorboard/
    │   ├── resume_experiments.sh       # 实验续跑脚本
    │   ├── 首次训练报告_5pct_seed42.md
    │   ├── 训练状态存档_20260615.md
    │   ├── 训练状态存档_20260616.md
    │   ├── 训练状态存档_20260617.md
    │   ├── 训练状态存档_20260620.md
    │   ├── 综合分析报告_20260617.md
    │   ├── 综合分析报告_20260618.md
    │   ├── 基线补全总结_20260618.md
    │   ├── 交叉点确认报告_20260619.md
    │   ├── 端到端流水线测试总结.md
    │   └── 项目工作总结_20260620.md
    │
    ├── 📝 paper/                       # 课程论文 & 演示
    │   ├── SAM-MedSeg_paper_draft.docx  # 课程论文 (~4,400 字)
    │   ├── SAM-MedSeg_presentation.pptx # 演示文稿 (13 页)
    │   └── references.bib              # BibTeX 参考文献 (17 条)
    │
    ├── ✅ tests/                       # 单元测试
    │   ├── __init__.py
    │   └── test_modules.py             # 模块正确性测试
    │       · 数据加载测试
    │       · 模型前向传播测试
    │       · 损失函数测试
    │       · 指标计算测试
    │
    ├── 📓 notebooks/                   # Jupyter Notebooks
    │   └── inference_demo.ipynb        # 交互式推理演示
    │
    ├── 🔗 checkpoints/                 # 模型权重 (gitignored, ~358 MB)
    │   └── sam_vit_b_01ec64.pth        # SAM ViT-B 预训练权重
    │
    └── 🐚 Shell 脚本
        ├── setup.sh                    # 环境搭建 (conda + pip + 下载 SAM 权重)
        ├── run_unet_experiments.sh     # U-Net 批量实验脚本
        └── run_unet_1pct_5pct.sh       # U-Net 1%/5% 实验脚本
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
