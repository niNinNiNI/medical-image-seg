#!/usr/bin/env bash
# =============================================================================
# SAM-MedSeg: One-click environment setup script
# =============================================================================
set -euo pipefail

echo "============================================"
echo " SAM-MedSeg Environment Setup"
echo "============================================"

# Detect Python
PYTHON=$(which python3 || which python)
echo "[1/4] Python: $PYTHON ($($PYTHON --version))"

# Create virtual environment (optional)
if [ "${1:-}" = "--venv" ]; then
    echo "[2/4] Creating virtual environment..."
    $PYTHON -m venv venv
    source venv/bin/activate
    PIP="venv/bin/pip"
else
    PIP="pip"
fi

# Install dependencies
echo "[3/4] Installing Python dependencies..."
$PIP install --upgrade pip
$PIP install -r requirements.txt

# Download SAM checkpoint
echo "[4/4] Downloading SAM ViT-B checkpoint..."
mkdir -p checkpoints
if [ ! -f checkpoints/sam_vit_b_01ec64.pth ]; then
    wget -O checkpoints/sam_vit_b_01ec64.pth \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    echo "  ✓ SAM checkpoint downloaded."
else
    echo "  ✓ SAM checkpoint already exists."
fi

echo ""
echo "============================================"
echo " Setup complete!"
echo " Next steps:"
echo "  1. python data/prepare_data.py"
echo "  2. python train.py"
echo "============================================"
