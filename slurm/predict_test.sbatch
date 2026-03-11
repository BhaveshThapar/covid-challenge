#!/bin/bash
#SBATCH --job-name=covid-predict-test
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/predict_test_%j.out
#SBATCH --error=logs/predict_test_%j.err

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

# Run 5-model ensemble prediction on test set
# With BN adaptation + CLAHE to handle domain shift
python src/predict.py \
    --configs configs/exp_b3_s42.yaml configs/exp_b3_s123.yaml configs/exp_b3_s7.yaml configs/exp_cnxt_s42.yaml configs/exp_effv2_s42.yaml \
    --checkpoints checkpoints/exp_b3_s42/best.pt checkpoints/exp_b3_s123/best.pt checkpoints/exp_b3_s7/best.pt checkpoints/exp_cnxt_s42/best.pt checkpoints/exp_effv2_s42/best.pt \
    --test-dir data/test \
    --adapt-bn --clahe \
    --output predictions_adapted.csv

echo "Done at $(date)"
