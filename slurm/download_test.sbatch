#!/bin/bash
#SBATCH --job-name=covid-dl-test
#SBATCH --partition=tron
#SBATCH --qos=high
#SBATCH --account=nexus
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_test_%j.out
#SBATCH --error=logs/download_test_%j.err

set -e
cd /fs/nexus-scratch/aadit/covid-challenge

# Load environment
export BASH_ENV=/usr/share/Modules/init/bash
source /usr/share/Modules/init/bash
module load Python3/3.10.14
module load unrar/7.0.9
source venv/bin/activate
unset BASH_ENV

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "================"

mkdir -p logs datasets data

python scripts/download_and_extract_test.py \
    --datasets-dir datasets \
    --data-dir data \
    --temp-dir data/_temp_test

echo "Done at $(date)"
