#!/bin/bash
#SBATCH --job-name=covid-eval
#SBATCH --partition=tron
#SBATCH --qos=high
#SBATCH --account=nexus
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

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

mkdir -p results

python src/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/v4_ovr_best.pt \
    --data-dir data \
    --metadata-dir datasets \
    --output-csv results/v4_per_scan.csv

echo "Done at $(date)"
