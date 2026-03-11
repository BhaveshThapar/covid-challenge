#!/bin/bash
#SBATCH --job-name=covid-ensemble-val
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa6000:3
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/ensemble_val_%j.out
#SBATCH --error=logs/ensemble_val_%j.err

set -e
cd /fs/nexus-scratch/anant04/covid-challenge

export TORCH_HOME=/fs/nexus-scratch/anant04/.cache/torch
mkdir -p "$TORCH_HOME" logs

module load Python3/3.10.14
source venv/bin/activate

echo "=== Ensemble Validation + Weight Tuning ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

python src/ensemble.py \
    --config configs/ensemble.yaml \
    --data-dir data \
    --metadata-dir datasets \
    --output predictions_ensemble_val.csv \
    --split val \
    --tune-weights \
    --per-source-threshold \
    --gpus 0 1 2

echo "Done at $(date)"
