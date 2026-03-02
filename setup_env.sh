#!/bin/bash
# Environment setup for Multi-Source Covid-19 Detection Challenge
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=== Loading Python 3.10 module ==="
module load Python3/3.10.14

echo "=== Creating virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing PyTorch (CUDA 11.8) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing ML libraries ==="
pip install \
    timm \
    albumentations \
    scikit-learn \
    pandas \
    Pillow \
    opencv-python-headless \
    tensorboard \
    tqdm \
    pyyaml

echo "=== Installing archive tools ==="
pip install py7zr rarfile

echo "=== Creating project directories ==="
mkdir -p data/{train/{covid,non_covid},val/{covid,non_covid},metadata}
mkdir -p src scripts slurm configs checkpoints logs

echo "=== Environment setup complete ==="
echo "Activate with: module load Python3/3.10.14 && source venv/bin/activate"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
