#!/bin/bash
#SBATCH --job-name=covid-gradcam-pp
#SBATCH --partition=tron
#SBATCH --qos=premium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/gradcam_pp_%j.out
#SBATCH --error=logs/gradcam_pp_%j.err

# Aggregate Grad-CAM++ saliency maps over the full validation set.
# Outputs three PNGs to figures/:
#   mean_cam_covid.png, mean_cam_noncovid.png, mean_cam_comparison.png
#
# Usage:
#   sbatch slurm/gradcam_pp.sbatch
#
# Override any variable without editing this file:
#   MODEL=mil CHECKPOINT=checkpoints/exp_b3_s7/best.pt sbatch slurm/gradcam_pp.sbatch

set -e

REPO_ROOT="${REPO_ROOT:-/fs/nexus-scratch/aadit/covid-challenge}"
cd "$REPO_ROOT"

export TORCH_HOME="${TORCH_HOME:-/fs/nexus-scratch/aadit/.cache/torch}"
mkdir -p "$TORCH_HOME" logs figures

module load Python3/3.10.14
source venv/bin/activate

# --- config ---
MODEL="${MODEL:-mil}"
CONFIG="${CONFIG:-configs/efficientnet.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/exp_b3_s42/best.pt}"
TARGET="${TARGET:-covid}"
DATA_DIR="${DATA_DIR:-data}"
METADATA_DIR="${METADATA_DIR:-datasets}"
SPLIT="${SPLIT:-val}"
OUTPUT_DIR="${OUTPUT_DIR:-figures}"
DEVICE="${DEVICE:-cuda}"

echo "=== Aggregate Grad-CAM++ saliency ==="
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "MODEL:      $MODEL"
echo "CHECKPOINT: $CHECKPOINT"
echo "TARGET:     $TARGET"
echo "OUTPUT_DIR: $OUTPUT_DIR"

python scripts/run_gradcam_pp.py \
  --model "$MODEL" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --dataset-mode \
  --data-dir "$DATA_DIR" \
  --metadata-dir "$METADATA_DIR" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --target "$TARGET" \
  --device "$DEVICE"

echo "Done at $(date)"
echo "Figures written to $OUTPUT_DIR/"
