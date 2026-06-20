#!/bin/bash
# =============================================================================
# SAM-MedSeg 实验恢复脚本
# 从 2026-06-15 存档状态恢复，运行剩余 3 组 100% 实验 + 消融实验
#
# 用法: bash experiments/resume_experiments.sh
# =============================================================================

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

echo "============================================"
echo " SAM-MedSeg 实验恢复"
echo " 存档日期: 2026-06-15"
echo " 已完成: 9/12 组 (1%/5%/10%)"
echo " 剩余: 3 组 100% + 消融实验"
echo "============================================"
echo ""

# ─── 阶段一: 100% 全量数据实验（3 seeds）───
echo ">>> 阶段一: 100% 全量数据实验 (预计 6-9 小时)"

for seed in 42 43 44; do
    result_dir="experiments/results/ratio_1.0_seed_${seed}"
    if [ -f "${result_dir}/results.json" ]; then
        echo "  [skip] ratio=1.0 seed=${seed} — 已完成"
        continue
    fi
    echo ""
    echo "  >>> 运行: ratio=1.0 seed=${seed} <<<"
    python train.py --few-shot-ratio 1.0 --seed ${seed}
    echo "  ✓ ratio=1.0 seed=${seed} 完成"
done

# ─── 阶段二: 消融实验 ───
echo ""
echo ">>> 阶段二: 消融实验 (预计 1 小时)"

# LoRA only
result_dir="experiments/results/ratio_0.05_seed_42_ablation_lora_only"
if [ -f "${result_dir}/results.json" ]; then
    echo "  [skip] LoRA only — 已完成"
else
    echo "  >>> 运行: LoRA only <<<"
    python train.py --few-shot-ratio 0.05 --seed 42 --ablation lora_only
    echo "  ✓ LoRA only 完成"
fi

# MedDecoder only
result_dir="experiments/results/ratio_0.05_seed_42_ablation_med_decoder_only"
if [ -f "${result_dir}/results.json" ]; then
    echo "  [skip] MedDecoder only — 已完成"
else
    echo "  >>> 运行: MedDecoder only <<<"
    python train.py --few-shot-ratio 0.05 --seed 42 --ablation med_decoder_only
    echo "  ✓ MedDecoder only 完成"
fi

echo ""
echo "============================================"
echo " 全部实验完成!"
echo " 结果目录: experiments/results/"
echo " TensorBoard: tensorboard --logdir experiments/results --port 6007"
echo "============================================"
