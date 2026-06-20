#!/bin/bash
# Run all remaining U-Net baseline experiments
# Skips already-completed configs

set -e

PROJECT_DIR="/home/nini/文档/深度学习大作业/期末大作业/sam-medical-seg"
OUTPUT_DIR="$PROJECT_DIR/experiments/results/unet_baseline"
cd "$PROJECT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# All experiments to run: ratio seed
# Order: fast ones first (few samples), slow ones last (100%)
EXPERIMENTS=(
    "0.01 42"
    "0.01 43"
    "0.01 44"
    "0.05 42"
    "0.05 43"
    "0.05 44"
    "0.10 42"
    "0.10 43"
    "0.10 44"
    "1.0 43"
    "1.0 44"
)

TOTAL=${#EXPERIMENTS[@]}
COMPLETED=0
FAILED=0

echo "=========================================="
echo " U-Net Baseline — Full Experiment Matrix"
echo " Total: $TOTAL configs to run"
echo "=========================================="
echo ""

for ((i=0; i<$TOTAL; i++)); do
    entry="${EXPERIMENTS[$i]}"
    ratio=$(echo "$entry" | awk '{print $1}')
    seed=$(echo "$entry" | awk '{print $2}')

    result_dir="$OUTPUT_DIR/ratio_${ratio}_seed_${seed}"
    result_file="$result_dir/results.json"

    if [ -f "$result_file" ]; then
        test_dice=$(python3 -c "import json; d=json.load(open('$result_file')); print(d['test_metrics']['dice'])" 2>/dev/null || echo "?")
        echo "[$((i+1))/$TOTAL] SKIP ratio=$ratio seed=$seed (already done, Test Dice=$test_dice)"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    echo ""
    echo "=========================================="
    echo "[$((i+1))/$TOTAL] START ratio=$ratio seed=$seed"
    echo "=========================================="

    start_time=$(date +%s)

    if python unet_baseline.py \
        --ratio "$ratio" \
        --seed "$seed" \
        --batch-size 2 \
        --epochs 100 \
        --patience 15 \
        --data-dir ./data/kvasir-seg \
        --image-size 1024 1024 \
        --output-dir ./experiments/results/unet_baseline \
        --device cuda 2>&1; then

        end_time=$(date +%s)
        elapsed=$(( (end_time - start_time) / 60 ))

        # Extract test dice
        test_dice=$(python3 -c "import json; d=json.load(open('$result_file')); print(d['test_metrics']['dice'])" 2>/dev/null || echo "?")

        echo ""
        echo "✓ [$((i+1))/$TOTAL] DONE ratio=$ratio seed=$seed | Test Dice=$test_dice | ${elapsed}min"
        COMPLETED=$((COMPLETED + 1))
    else
        echo ""
        echo "✗ [$((i+1))/$TOTAL] FAILED ratio=$ratio seed=$seed"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo " All U-Net Experiments Complete!"
echo "   Completed: $COMPLETED / $TOTAL"
echo "   Failed:    $FAILED"
echo "=========================================="

# Print summary
echo ""
echo "Results Summary:"
echo "----------------"
for entry in "${EXPERIMENTS[@]}"; do
    ratio=$(echo "$entry" | awk '{print $1}')
    seed=$(echo "$entry" | awk '{print $2}')
    result_file="$OUTPUT_DIR/ratio_${ratio}_seed_${seed}/results.json"
    if [ -f "$result_file" ]; then
        test_dice=$(python3 -c "import json; d=json.load(open('$result_file')); print(f\"{d['test_metrics']['dice']:.4f}\")" 2>/dev/null || echo "?")
        best_epoch=$(python3 -c "import json; d=json.load(open('$result_file')); print(d['best_epoch'])" 2>/dev/null || echo "?")
        echo "  ratio=$ratio seed=$seed  Test Dice=$test_dice  Best Epoch=$best_epoch"
    else
        echo "  ratio=$ratio seed=$seed  MISSING"
    fi
done
