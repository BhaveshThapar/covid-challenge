#!/bin/bash
#SBATCH --job-name=covid-train
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e
cd /fs/nexus-scratch/bthapar/covid-challenge

# Load environment
module load Python3/3.10.14
source venv/bin/activate

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run both phases
python src/train.py \
    --config configs/default.yaml \
    --data-dir data \
    --metadata-dir data/metadata \
    --phase 0

echo "Training complete at $(date)"

# Run evaluation
python src/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pt \
    --data-dir data \
    --metadata-dir data/metadata

echo "Done at $(date)"
