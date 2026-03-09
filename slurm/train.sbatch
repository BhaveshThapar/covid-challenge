#!/bin/bash
#SBATCH --job-name=covid-train
#SBATCH --partition=tron
#SBATCH --qos=high
#SBATCH --account=nexus
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e
cd /fs/nexus-scratch/aadit/covid-challenge

# Load environment
export BASH_ENV=/usr/share/Modules/init/bash
source /usr/share/Modules/init/bash
module load Python3/3.10.14
source venv/bin/activate
unset BASH_ENV

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "================"

# Run both phases
python src/train.py \
    --config configs/default.yaml \
    --data-dir data \
    --metadata-dir datasets \
    --run-name v5 \
    --phase 4

echo "Training complete at $(date)"

# Run ensemble evaluation
python src/evaluate.py \
    --config configs/default.yaml \
    --ensemble \
    --run-name v5 \
    --data-dir data \
    --metadata-dir datasets

echo "Done at $(date)"
