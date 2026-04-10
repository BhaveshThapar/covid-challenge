#!/bin/bash
#SBATCH --job-name=covid-gradcam-pp
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/gradcam_pp_%j.out
#SBATCH --error=logs/gradcam_pp_%j.err

# Grad-CAM++ figures (DINOv2 slice or MIL scan). Edit paths below, then:
#   sbatch slurm/gradcam_pp.sbatch

set -e

# --- clone location (change if your scratch path differs) ---
REPO_ROOT="${REPO_ROOT:-/fs/nexus-scratch/bthapar/covid-challenge}"
cd "$REPO_ROOT"

export TORCH_HOME="${TORCH_HOME:-/fs/nexus-scratch/bthapar/.cache/torch}"
mkdir -p "$TORCH_HOME" logs figures

module load Python3/3.10.14
source venv/bin/activate

# --- job config: pick ONE of dinov2 (needs IMAGE) or mil (needs SCAN_DIR) ---
MODEL="${MODEL:-dinov2}"
CONFIG="${CONFIG:-configs/dinov2.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/v1_ovr_best.pt}"
OUTPUT="${OUTPUT:-figures/gradcam_pp_${SLURM_JOB_ID}.png}"
TARGET="${TARGET:-covid}"
VIT_BLOCK="${VIT_BLOCK:-10}"
DEVICE="${DEVICE:-cuda}"

# DINOv2: set IMAGE to a single JPEG slice
IMAGE="${IMAGE:-}"
# MIL: set SCAN_DIR to a ct_scan_* folder
SCAN_DIR="${SCAN_DIR:-}"
SLICE_INDEX="${SLICE_INDEX:-max_attn}"

echo "=== Grad-CAM++ saliency ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "MODEL=$MODEL OUTPUT=$OUTPUT"

if [ "$MODEL" = "dinov2" ]; then
  if [ -z "$IMAGE" ] || [ ! -f "$IMAGE" ]; then
    echo "ERROR: For dinov2 set IMAGE to an existing slice file, e.g."
    echo "  export IMAGE=/path/to/0.jpg"
    echo "  sbatch slurm/gradcam_pp.sbatch"
    exit 1
  fi
  python scripts/run_gradcam_pp.py \
    --model dinov2 \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --image "$IMAGE" \
    --target "$TARGET" \
    --vit-block "$VIT_BLOCK" \
    --device "$DEVICE"
else
  if [ -z "$SCAN_DIR" ] || [ ! -d "$SCAN_DIR" ]; then
    echo "ERROR: For mil set SCAN_DIR to a scan folder, e.g."
    echo "  export MODEL=mil"
    echo "  export CONFIG=configs/efficientnet.yaml"
    echo "  export CHECKPOINT=checkpoints/exp_b3_s42/best.pt"
    echo "  export SCAN_DIR=/path/to/ct_scan_xxx"
    echo "  sbatch slurm/gradcam_pp.sbatch"
    exit 1
  fi
  python scripts/run_gradcam_pp.py \
    --model mil \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --scan-dir "$SCAN_DIR" \
    --slice-index "$SLICE_INDEX" \
    --target "$TARGET" \
    --device "$DEVICE"
fi

echo "Done at $(date)"
echo "Wrote $OUTPUT"
