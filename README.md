# 面向少样本医学图像分割的 SAM 改进方法研究

> 深度学习与计算机视觉 课程大作业 | 2026年夏季学期

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 项目简介

本项目改进 Meta 的 **Segment Anything Model (SAM)**，使其适用于**少样本医学图像分割**场景。

核心思路：在冻结 SAM 大部分参数的前提下，通过三个模块实现高效迁移：

| 模块 | 说明 | 技术创新点 |
|------|------|-----------|
| **LoRA 特征适配** | 在 SAM 图像编码器的 Attention Q/V 矩阵上添加低秩适配 | 仅增加 227K 参数（0.1%），实现参数高效微调 |
| **自动提示优化** | 网格采样 + IoU 置信度筛选 + NMS 融合 | 消除手工标注提示的依赖，实现全自动推理 |
| **医学跳跃连接解码器** | U-Net 风格解码器，融合 SAM 编码器多层特征 | 双路径融合（SAM 解码器 + 医学解码器） |

### 核心实验结果

| 数据量 | U-Net Dice | SAM-MedSeg Dice | 优势 |
|:------:|:----------:|:---------------:|:----:|
| 1% (8 张) | 0.385 | **0.425** | SAM-MedSeg +4.0 pp |
| 5% (40 张) | 0.443 | **0.467** | SAM-MedSeg +2.4 pp |
| 10% (80 张) | **0.509** | 0.484 | U-Net +2.5 pp |
| 100% (800 张) | **0.719** | 0.640 | U-Net +7.9 pp |

**关键结论：当标注数据 ≤ 50 张时，SAM-MedSeg 优于 U-Net，验证了少样本场景下大模型迁移学习的价值。**

---

## 目录结构

```
期末大作业/
├── README.md                                          # 项目总览
├── .gitignore
│
├── 深度学习与计算机视觉_课程评分标准及大作业说明_.md      # 课程要求与评分标准
├── 期末大作业选题方案1.md                               # 选题方案与可行性分析
├── 项目实施计划.md                                     # 实施计划与技术路线
├── 附件2-深度学习与计算机视觉课程大作业参考模板.docx      # 论文写作模板
│
├── sam-medical-seg/                                   # ★ 核心代码与实验
│   ├── README.md                                      #   子项目详细文档
│   ├── requirements.txt                               #   Python 依赖
│   ├── configs/config.yaml                            #   超参数配置
│   ├── data/                                          #   数据加载与预处理
│   ├── models/                                        #   模型实现
│   │   ├── sam_backbone.py                            #     SAM 主干加载
│   │   ├── lora_adapter.py                            #     LoRA 适配模块
│   │   ├── prompt_optimizer.py                        #     提示优化模块
│   │   ├── med_decoder.py                             #     医学解码器
│   │   └── full_model.py                              #     完整模型整合
│   ├── losses/combined_loss.py                        #   组合损失函数
│   ├── utils/                                         #   工具函数
│   ├── train.py                                       #   训练脚本
│   ├── evaluate.py                                    #   评估脚本
│   ├── inference.py                                   #   推理脚本
│   ├── unet_baseline.py                               #   U-Net 基线
│   ├── sam_zero_shot_baseline.py                      #   SAM 零样本基线
│   ├── visualize_paper_figures.py                     #   数据图表生成
│   ├── visualize_architecture_figures.py              #   架构图生成
│   ├── generate_paper.py                              #   论文生成
│   ├── generate_ppt.py                                #   演示文稿生成
│   ├── experiments/                                   #   实验结果（27组）
│   └── paper/                                         #   论文与演示文稿
│
└── kvasir-seg/                                        # Kvasir-SEG 数据集
    └── Kvasir-SEG/                                    #   原始数据
```

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.1（或 11.8）
- GPU 显存 ≥ 8 GB（RTX 3060/4060 或 Google Colab T4）

### 安装与运行

```bash
# 1. 进入项目目录
cd sam-medical-seg

# 2. 创建虚拟环境
conda create -n sam-medseg python=3.10 -y
conda activate sam-medseg

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载 SAM 预训练权重
mkdir -p checkpoints
wget -O checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# 5. 准备数据集
python data/prepare_data.py

# 6. 训练完整模型（100% 数据）
python train.py --few-shot-ratio 1.0 --seed 44

# 7. 评估
python evaluate.py --checkpoint experiments/results/ratio_1.0_seed_44/best_model.pth

# 8. 单张推理
python inference.py --image path/to/image.jpg \
  --checkpoint experiments/results/ratio_1.0_seed_44/best_model.pth
```

更多细节见 [sam-medical-seg/README.md](sam-medical-seg/README.md)。

---

## 实验概览

### 对比实验矩阵

| 编号 | 方法 | 说明 | 最佳 Dice |
|:----:|------|------|:---------:|
| E1 | SAM Zero-Shot | 网格提示，不做微调 | 0.271 |
| E2 | U-Net（从头训练） | 经典 CNN 基线 | 0.725 |
| E3 | SAM + LoRA only | 仅特征适配 | 0.452 |
| E4 | SAM + MedDecoder only | 仅医学解码器 | 0.426 |
| E5 | **SAM + LoRA + MedDecoder** | 完整方案 | **0.652** |

### 消融实验（5% 数据，seed 42）

| 配置 | Dice | Precision | Recall | 可训练参数 |
|------|:----:|:---------:|:------:|:----------:|
| 完整模型 | 0.448 | 0.421 | 0.697 | 4.84M |
| 仅 LoRA | **0.452** | **0.441** | 0.658 | **227K** |
| 仅 MedDecoder | 0.426 | 0.366 | **0.772** | 30.4M |

---

## 交付物

- [x] 课程论文（~4,400 字，17 篇参考文献）
- [x] 演示文稿（13 页 PPT）
- [x] 项目完整源码（训练 / 评估 / 推理）
- [x] 27 组实验结果及完整指标记录
- [x] 8 张论文级图表（架构图 + 数据图）
- [x] 推理 Demo（CLI + Jupyter Notebook）
- [x] 预训练模型权重

---

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 深度学习框架 | PyTorch 2.x |
| SAM 模型 | segment-anything（Meta 官方） |
| 医学图像工具 | MONAI |
| LoRA 实现 | PEFT（HuggingFace） |
| 数据增强 | MONAI Transforms |
| 实验记录 | TensorBoard |
| 可视化 | Matplotlib + Seaborn |
| 论文/PPT 生成 | python-docx + python-pptx |

---

## 参考文献

1. Kirillov et al., "Segment Anything", *ICCV 2023*
2. Ma et al., "Segment Anything in Medical Images", *Nature Communications 2024*
3. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", *ICLR 2022*
4. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", *MICCAI 2015*
5. Isensee et al., "nnU-Net: Self-configuring Method for Biomedical Image Segmentation", *Nature Methods 2021*
6. Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset", *MMM 2020*
7. Zhang et al., "SAMed: Segment Anything Model for Medical Image Segmentation", *2023*
8. Wu et al., "SAM-Adapter: Adapting SAM for Medical Image Segmentation", *2023*

完整 BibTeX 见 `sam-medical-seg/paper/references.bib`。

---

## 许可证

MIT License
