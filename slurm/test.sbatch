#!/bin/bash
#SBATCH --job-name=covid-test
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

set -e
cd /fs/nexus-scratch/anant04/covid-challenge

export TORCH_HOME=/fs/nexus-scratch/anant04/.cache/torch
mkdir -p "$TORCH_HOME"

module load Python3/3.10.14
source venv/bin/activate

echo "=== Test Prediction Job ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

# Check if test data exists; if not, run extract first
if [ ! -d "data/test" ] || [ -z "$(ls -A data/test 2>/dev/null)" ]; then
    echo "Test data not found. Running download and extract first..."
    export HOME=/fs/nexus-scratch/anant04
    mkdir -p "$HOME/.cache/gdown"
    module load unrar/7.0.9
    python scripts/download_and_extract.py \
        --datasets-dir datasets \
        --data-dir data \
        --temp-dir data/_temp_extract
fi

python src/predict_test.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/v1_ovr_best.pt \
    --data-dir data \
    --metadata-dir datasets \
    --print-results \
    --threshold 0.5 \
    --tta

echo "Done at $(date)"
