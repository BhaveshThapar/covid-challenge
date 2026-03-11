#!/bin/bash
#SBATCH --job-name=covid-eval
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -e
cd /fs/nexus-scratch/anant04/covid-challenge

# PyTorch hub cache in scratch (home dir full)
export TORCH_HOME=/fs/nexus-scratch/anant04/.cache/torch
mkdir -p "$TORCH_HOME"

# Load environment
module load Python3/3.10.14
source venv/bin/activate

echo "=== Eval Job Info ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "================"

python src/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/v1_ovr_best.pt \
    --data-dir data \
    --metadata-dir datasets \
    --split val \
    --output-csv validation_results.csv

echo "Done at $(date)"
