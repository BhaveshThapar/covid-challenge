#!/bin/bash
#SBATCH --job-name=covid-extract
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

set -e
cd /fs/nexus-scratch/anant04/covid-challenge

# Use scratch as HOME so gdown writes to scratch (home dir is full)
export HOME=/fs/nexus-scratch/anant04
mkdir -p "$HOME/.cache/gdown"

# Load environment
module load Python3/3.10.14
module load unrar/7.0.9
source venv/bin/activate

echo "Starting data download and extraction at $(date)"
python scripts/download_and_extract.py \
    --datasets-dir datasets \
    --data-dir data \
    --temp-dir data/_temp_extract

echo "Done at $(date)"
