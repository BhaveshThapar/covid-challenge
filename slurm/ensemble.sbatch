#!/bin/bash
#SBATCH --job-name=covid-ensemble
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa6000:3
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/ensemble_%j.out
#SBATCH --error=logs/ensemble_%j.err

set -e
cd /fs/nexus-scratch/anant04/covid-challenge

export TORCH_HOME=/fs/nexus-scratch/anant04/.cache/torch
mkdir -p "$TORCH_HOME" logs

module load Python3/3.10.14
source venv/bin/activate

echo "=== Ensemble Job (3 GPUs) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
echo "Start: $(date)"

# Ensure test data exists
if [ ! -d "data/test" ] || [ -z "$(ls -A data/test 2>/dev/null)" ]; then
    echo "Test data not found. Running extract first..."
    export HOME=/fs/nexus-scratch/anant04
    mkdir -p "$HOME/.cache/gdown"
    module load unrar/7.0.9
    python scripts/download_and_extract.py \
        --datasets-dir datasets \
        --data-dir data \
        --temp-dir data/_temp_extract
fi

python src/ensemble.py \
    --config configs/ensemble.yaml \
    --data-dir data \
    --metadata-dir datasets \
    --output predictions_ensemble.csv \
    --gpus 0 1 2

echo "Done at $(date)"
