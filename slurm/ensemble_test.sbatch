#!/bin/bash
#SBATCH --job-name=covid-ensemble-test
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/ensemble_test_%j.out
#SBATCH --error=logs/ensemble_test_%j.err

set -e
cd /fs/nexus-scratch/bthapar/covid-challenge

export TORCH_HOME=/fs/nexus-scratch/bthapar/.cache/torch
mkdir -p "$TORCH_HOME" logs results

module load Python3/3.10.14
source venv/bin/activate

echo "=== Ensemble Test Predictions (9 models, 4-flip TTA, score-weighted) ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

python src/ensemble.py \
    --config configs/ensemble.yaml \
    --data-dir data \
    --metadata-dir datasets \
    --output results/predictions_ensemble_test.csv \
    --split test \
    --weighting uniform \
    --strategy avg \
    --gpus 0

echo "Done at $(date)"
echo ""
echo "=== Preview of output ==="
head -20 results/predictions_ensemble_test.csv
echo "..."
echo "Total lines: $(wc -l < results/predictions_ensemble_test.csv)"
