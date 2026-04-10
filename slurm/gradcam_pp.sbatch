#!/bin/bash
#SBATCH --job-name=covid-gradcam-pp
#SBATCH --partition=tron
#SBATCH --qos=high
#SBATCH --account=nexus
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/gradcam_pp_%j.out
#SBATCH --error=logs/gradcam_pp_%j.err
 
# Input x Gradient saliency — top-4 most confident correct predictions per class.
# Outputs (in figures/):
#   saliency_covid_ct_scan_XXX.png      (4 files)
#   saliency_noncovid_ct_scan_XXX.png   (4 files)
#   saliency_top4_grid.png              (combined 2x4 grid)
#
# Usage:
#   sbatch slurm/gradcam_pp.sbatch

set -e

REPO_ROOT="${REPO_ROOT:-/fs/nexus-scratch/aadit/covid-challenge}"
cd "$REPO_ROOT"

export TORCH_HOME="${TORCH_HOME:-/fs/nexus-scratch/aadit/.cache/torch}"
mkdir -p "$TORCH_HOME" logs figures

# Initialize module system (required on tron — not auto-sourced)
source /etc/profile.d/modules.sh
module load Python3/3.10.14
source venv/bin/activate

# --- config ---
CHECKPOINT="${CHECKPOINT:-checkpoints/exp_b3_s42/best.pt}"
TOP_N="${TOP_N:-4}"
DATA_DIR="${DATA_DIR:-data}"
METADATA_DIR="${METADATA_DIR:-datasets}"
SPLIT="${SPLIT:-val}"
OUTPUT_DIR="${OUTPUT_DIR:-figures}"
DEVICE="${DEVICE:-cuda}"

echo "=== Input x Gradient Saliency — Top-${TOP_N} per class ==="
echo "Node:            $(hostname)"
echo "Start:           $(date)"
echo "CHECKPOINT:  $CHECKPOINT"
echo "OUTPUT_DIR:  $OUTPUT_DIR"

python scripts/run_gradcam_pp.py \
  --checkpoint "$CHECKPOINT" \
  --top-n-mode \
  --top-n "$TOP_N" \
  --data-dir "$DATA_DIR" \
  --metadata-dir "$METADATA_DIR" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE"

echo "Done at $(date)"
echo "Figures written to $OUTPUT_DIR/"
