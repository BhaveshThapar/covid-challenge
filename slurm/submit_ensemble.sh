#!/bin/bash
# Submit all 3 ensemble experiments in parallel
# Usage: bash slurm/submit_ensemble.sh

set -e
cd /fs/nexus-scratch/bthapar/covid-challenge

echo "Submitting 3 ensemble training jobs..."

JOB1=$(sbatch --export=CONFIG=configs/exp_b3_s42.yaml slurm/train.sbatch | awk '{print $4}')
echo "  [1/3] EfficientNet-B3 seed=42:  job ${JOB1}"

JOB2=$(sbatch --export=CONFIG=configs/exp_cnxt_s42.yaml slurm/train.sbatch | awk '{print $4}')
echo "  [2/3] ConvNeXt-Tiny   seed=42:  job ${JOB2}"

JOB3=$(sbatch --export=CONFIG=configs/exp_b3_s123.yaml slurm/train.sbatch | awk '{print $4}')
echo "  [3/3] EfficientNet-B3 seed=123: job ${JOB3}"

echo ""
echo "Monitor with: squeue -u $(whoami)"
echo ""
echo "After all 3 complete, run ensemble evaluation:"
echo "  python src/ensemble_evaluate.py \\"
echo "    --configs configs/exp_b3_s42.yaml configs/exp_cnxt_s42.yaml configs/exp_b3_s123.yaml \\"
echo "    --checkpoints checkpoints/exp_b3_s42/best.pt checkpoints/exp_cnxt_s42/best.pt checkpoints/exp_b3_s123/best.pt"
