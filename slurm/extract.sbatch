#!/bin/bash
#SBATCH --job-name=covid-extract
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

set -e
cd /fs/nexus-scratch/bthapar/covid-challenge

# Load environment
module load Python3/3.10.14
module load unrar/7.0.9
source venv/bin/activate

echo "Starting data extraction at $(date)"
python scripts/extract_data.py \
    --datasets-dir datasets \
    --data-dir data \
    --temp-dir data/_temp_extract \
    --skip-validation

echo "Running data analysis at $(date)"
python scripts/analyze_data.py --data-dir data --metadata-dir data/metadata

echo "Done at $(date)"
