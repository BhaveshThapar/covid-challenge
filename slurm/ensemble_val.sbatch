#!/bin/bash
#SBATCH --job-name=covid-ensemble-val
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/ensemble_val_%j.out
#SBATCH --error=logs/ensemble_val_%j.err

set -e
cd /fs/nexus-scratch/bthapar/covid-challenge

export TORCH_HOME=/fs/nexus-scratch/bthapar/.cache/torch
mkdir -p "$TORCH_HOME" logs results

module load Python3/3.10.14
source venv/bin/activate

echo "=== Ensemble Validation (Probability Averaging) ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

# Probability averaging with per-source threshold sweep (5 models, 1 GPU)
python src/ensemble.py \
    --config configs/ensemble.yaml \
    --data-dir data \
    --metadata-dir datasets \
    --output results/predictions_ensemble_avg_val.csv \
    --split val \
    --tune-threshold \
    --strategy avg \
    --gpus 0

echo ""
echo "=== Comparison: Legacy Majority Vote ==="

python src/ensemble.py \
    --config configs/ensemble.yaml \
    --data-dir data \
    --metadata-dir datasets \
    --output results/predictions_ensemble_majority_val.csv \
    --split val \
    --tune-threshold \
    --strategy majority \
    --gpus 0

echo "Done at $(date)"
