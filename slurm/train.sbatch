#!/bin/bash
#SBATCH --job-name=covid-train
#SBATCH --partition=tron
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

export BASH_ENV=/usr/share/Modules/init/bash
source /usr/share/Modules/init/bash

set -e
cd /fs/nexus-scratch/aadit/covid-challenge

module load Python3/3.10.14
source venv/bin/activate

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "================"

# Run all phases from scratch (Phase 1 → Phase 2a → Phase 2b)
python -u src/train.py \
    --config configs/default.yaml \
    --data-dir data \
    --metadata-dir datasets \
    --run-name v3 \
    --phase 0

echo "Training complete at $(date)"

# Run evaluation
python -u src/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/v3_ovr_best.pt \
    --data-dir data \
    --metadata-dir datasets

echo "Done at $(date)"

unset BASH_ENV
