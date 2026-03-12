#!/bin/bash
#SBATCH --job-name=covid-diag
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err

set -e
cd /fs/nexus-scratch/bthapar/covid-challenge

export TORCH_HOME=/fs/nexus-scratch/bthapar/.cache/torch
mkdir -p "$TORCH_HOME" logs

module load Python3/3.10.14
source venv/bin/activate

echo "=== Single Model Diagnostic ==="
echo "Node: $(hostname)"
python test_single_model.py
echo "Done at $(date)"
