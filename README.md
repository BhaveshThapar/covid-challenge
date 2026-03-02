# Multi-Source Covid-19 Detection Challenge

Binary Covid/Non-Covid classification of chest CT scans across 4 hospital sources.

## Architecture

**2-stage pipeline:**
1. **Phase 1** — EfficientNet-B3 backbone pretrained on individual CT slices
2. **Phase 2** — End-to-end scan-level training with gated attention MIL pooling

```
CT Slices → EfficientNet-B3 → Attention Pooling → Covid / Non-Covid
```

**Metric:** Average macro F1 across 4 data centres

## Project Structure

```
covid-challenge/
├── src/
│   ├── model.py       # EfficientNet-B3 + Attention MIL
│   ├── dataset.py     # Slice & scan-level dataloaders
│   ├── train.py       # 2-phase training loop
│   ├── evaluate.py    # Per-source F1 evaluation
│   └── utils.py       # Metrics, checkpointing, early stopping
├── scripts/
│   ├── extract_data.py   # Archive extraction & organization
│   └── analyze_data.py   # Dataset statistics
├── slurm/
│   ├── extract.sbatch    # Data extraction SLURM job
│   └── train.sbatch      # GPU training SLURM job
├── configs/
│   └── default.yaml      # Hyperparameters
└── setup_env.sh           # Environment setup
```

## Setup

```bash
# 1. Create environment (requires Python 3.10 module on cluster)
bash setup_env.sh

# Or manually:
module load Python3/3.10.14
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm albumentations scikit-learn pandas Pillow opencv-python-headless tensorboard tqdm pyyaml unrar-cffi
```

## Data

Download the competition archives into `datasets/` and then:

```bash
# Extract (via SLURM)
sbatch slurm/extract.sbatch

# Or locally
python scripts/extract_data.py --datasets-dir datasets --data-dir data
```

Expected structure after extraction:
```
data/
├── train/
│   ├── covid/          # ct_scan_*/  folders of JPEG slices
│   └── non_covid/
├── val/
│   ├── covid/
│   └── non_covid/
└── metadata/
    ├── val_covid.csv
    └── val_non_covid.csv
```

## Training

```bash
# Submit extraction then training as a pipeline
EXTRACT_JOB=$(sbatch --parsable slurm/extract.sbatch)
sbatch --dependency=afterok:$EXTRACT_JOB slurm/train.sbatch

# Or run directly (with GPU)
source venv/bin/activate
python src/train.py --config configs/default.yaml --phase 0
```

## Evaluation

```bash
python src/evaluate.py --checkpoint checkpoints/best.pt
```

Outputs per-source macro F1 and the final challenge score:
```
  source_0:  0.xxxx
  source_1:  0.xxxx
  source_2:  0.xxxx
  source_3:  0.xxxx
  average:   0.xxxx  ★
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | EfficientNet-B3 |
| Image size | 224×224 |
| Slices per scan (train) | 32 |
| Phase 1 LR | 1e-4 |
| Phase 2 LR | 3e-5 |
| Mixed precision | FP16 |
| Early stopping patience | 5 epochs |

## Requirements

- Python 3.10+
- PyTorch 2.x + CUDA 11.8
- SLURM cluster with GPU (tested on UMD Nexus, `scavenger` partition)
