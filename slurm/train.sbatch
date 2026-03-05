#!/bin/bash
#SBATCH --job-name=covid-train-v2
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e
cd /fs/nexus-scratch/bthapar/covid-challenge

# Load environment
module load Python3/3.10.14
source venv/bin/activate

# Accept config via environment variable or use default
CONFIG=${CONFIG:-configs/default.yaml}

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Config: ${CONFIG}"
echo "Start: $(date)"
echo "================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create checkpoint and log dirs from config (extract via python)
CKPT_DIR=$(python -c "from src.utils import load_config; c=load_config('${CONFIG}'); print(c.get('checkpoint_dir','checkpoints'))")
LOG_DIR=$(python -c "from src.utils import load_config; c=load_config('${CONFIG}'); print(c.get('log_dir','logs'))")
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

# Run both phases
python src/train.py \
    --config "${CONFIG}" \
    --data-dir data \
    --metadata-dir data/metadata \
    --phase 0

echo "Training complete at $(date)"

# Run evaluation with TTA + threshold sweep
python src/evaluate.py \
    --config "${CONFIG}" \
    --checkpoint "${CKPT_DIR}/best.pt" \
    --data-dir data \
    --metadata-dir data/metadata

echo "Done at $(date)"
