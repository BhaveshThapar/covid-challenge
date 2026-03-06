#!/bin/bash
# Submit all 5 ensemble experiments in parallel
# Usage: bash slurm/submit_ensemble_v2.sh
#
# Models:
#   1. EfficientNet-B3 seed=42   (may already be trained)
#   2. EfficientNet-B3 seed=123  (may already be trained)
#   3. ConvNeXt-Tiny   seed=42   (may already be trained)
#   4. EfficientNetV2-S seed=42  (NEW)
#   5. EfficientNet-B3 seed=7    (NEW)

set -e
cd /fs/nexus-scratch/bthapar/covid-challenge

echo "Submitting ensemble training jobs..."

# Check which models need training
submit_if_needed() {
    local config=$1
    local ckpt_dir=$2
    local desc=$3

    if [ -f "${ckpt_dir}/best.pt" ]; then
        echo "  [SKIP] ${desc}: checkpoint already exists at ${ckpt_dir}/best.pt"
    else
        local job_id=$(sbatch --export=CONFIG=${config} slurm/train.sbatch | awk '{print $4}')
        echo "  [SUBMIT] ${desc}: job ${job_id}"
    fi
}

submit_if_needed configs/exp_b3_s42.yaml   checkpoints/exp_b3_s42   "EfficientNet-B3 seed=42"
submit_if_needed configs/exp_b3_s123.yaml  checkpoints/exp_b3_s123  "EfficientNet-B3 seed=123"
submit_if_needed configs/exp_cnxt_s42.yaml checkpoints/exp_cnxt_s42 "ConvNeXt-Tiny seed=42"
submit_if_needed configs/exp_effv2_s42.yaml checkpoints/exp_effv2_s42 "EfficientNetV2-S seed=42"
submit_if_needed configs/exp_b3_s7.yaml    checkpoints/exp_b3_s7    "EfficientNet-B3 seed=7"

echo ""
echo "Monitor with: squeue -u $(whoami)"
echo ""
echo "After all jobs complete, run 5-model ensemble evaluation:"
echo "  python3 src/ensemble_evaluate.py \\"
echo "    --configs configs/exp_b3_s42.yaml configs/exp_b3_s123.yaml configs/exp_cnxt_s42.yaml configs/exp_effv2_s42.yaml configs/exp_b3_s7.yaml \\"
echo "    --checkpoints checkpoints/exp_b3_s42/best.pt checkpoints/exp_b3_s123/best.pt checkpoints/exp_cnxt_s42/best.pt checkpoints/exp_effv2_s42/best.pt checkpoints/exp_b3_s7/best.pt"
echo ""
echo "Quick 3-model ensemble (no new training needed):"
echo "  python3 src/ensemble_evaluate.py \\"
echo "    --configs configs/exp_b3_s42.yaml configs/exp_b3_s123.yaml configs/exp_cnxt_s42.yaml \\"
echo "    --checkpoints checkpoints/exp_b3_s42/best.pt checkpoints/exp_b3_s123/best.pt checkpoints/exp_cnxt_s42/best.pt"
