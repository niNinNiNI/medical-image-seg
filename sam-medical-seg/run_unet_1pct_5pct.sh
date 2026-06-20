#!/bin/bash
# Run U-Net 5% experiments only (1% already done)
set -e

PROJECT_DIR="/home/nini/文档/深度学习大作业/期末大作业/sam-medical-seg"
OUTPUT_DIR="$PROJECT_DIR/experiments/results/unet_baseline"
cd "$PROJECT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPERIMENTS=(
    "0.05 42"
    "0.05 43"
    "0.05 44"
)

TOTAL=${#EXPERIMENTS[@]}
COMPLETED=0

echo "=========================================="
echo " U-Net 5% Experiments (3 runs)"
echo "=========================================="

for ((i=0; i<$TOTAL; i++)); do
    entry="${EXPERIMENTS[$i]}"
    ratio=$(echo "$entry" | awk '{print $1}')
    seed=$(echo "$entry" | awk '{print $2}')
    result_file="$OUTPUT_DIR/ratio_${ratio}_seed_${seed}/results.json"

    if [ -f "$result_file" ]; then
        test_dice=$(python3 -c "import json; d=json.load(open('$result_file')); print(f\"{d['test_metrics']['dice']:.4f}\")" 2>/dev/null || echo "?")
        echo "[$((i+1))/$TOTAL] SKIP ratio=$ratio seed=$seed (done, Dice=$test_dice)"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "[$((i+1))/$TOTAL] START ratio=$ratio seed=$seed"
    echo "=========================================="

    python unet_baseline.py \
        --ratio "$ratio" \
        --seed "$seed" \
        --batch-size 2 \
        --epochs 100 \
        --patience 15 \
        --data-dir ./data/kvasir-seg \
        --image-size 1024 1024 \
        --output-dir ./experiments/results/unet_baseline \
        --device cuda 2>&1

    test_dice=$(python3 -c "import json; d=json.load(open('$result_file')); print(f\"{d['test_metrics']['dice']:.4f}\")" 2>/dev/null || echo "?")
    echo "✓ [$((i+1))/$TOTAL] DONE ratio=$ratio seed=$seed | Dice=$test_dice"
    COMPLETED=$((COMPLETED + 1))
done

echo ""
echo "=========================================="
echo " All 5% experiments complete!"
echo "=========================================="
