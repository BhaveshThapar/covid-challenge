# Multi-Source Covid-19 Detection Challenge

Binary Covid/Non-Covid classification of chest CT scans across 4 hospital sources.

## Architecture (aadit-dev branch)

**DenseNet-121 + RadImageNet, slice-level training, scan-level evaluation:**

```
CT Slices → DenseNet-121 (RadImageNet) → Average Slice Probs → Threshold → Covid / Non-Covid
```

Training uses progressive backbone unfreezing rather than a separate MIL aggregation stage.
Scan-level predictions simply average per-slice sigmoid probabilities (no learned attention).

**Metric:** Average macro F1 across 4 data centres

## Project Structure

```
covid-challenge/
├── src/
│   ├── model.py       # DenseNetCovidClassifier (DenseNet-121 + RadImageNet)
│   ├── dataset.py     # SliceDataset, ScanDataset, CenterBatchSampler, TTA transforms
│   ├── train.py       # Phase 1 (frozen) + Phase 2 (gradual unfreeze) training
│   ├── evaluate.py    # Scan-level inference, threshold tuning, TTA, per-source F1
│   └── utils.py       # Metrics, checkpointing, early stopping
├── scripts/
│   └── download_and_extract.py  # gdown download + archive extraction + dataset analysis
├── slurm/
│   ├── extract.sbatch    # Data download/extraction SLURM job (tron partition)
│   └── train.sbatch      # GPU training SLURM job (scavenger partition)
├── configs/
│   └── default.yaml      # Hyperparameters
└── setup_env.sh           # Environment setup
```

## Setup (on Nexus cluster)

```bash
# 1. Clone the aadit-dev branch
cd /fs/nexus-scratch/aadit
git clone -b aadit-dev https://github.com/BhaveshThapar/covid-challenge.git covid-challenge
cd covid-challenge

# 2. Create environment
bash setup_env.sh

# 3. Download RadImageNet DenseNet-121 weights
source venv/bin/activate
gdown --fuzzy "https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view?usp=sharing" \
      -O RadImageNet_pytorch.zip
unzip -q RadImageNet_pytorch.zip -d radimagenet_weights
cp radimagenet_weights/DenseNet121.pt checkpoints/radimagenet_densenet121.pt
```

## Data

Data is downloaded from Google Drive via gdown — no manual file placement needed.

```bash
# Extract data (submit SLURM job — tron partition, ~1-6 hours)
sbatch slurm/extract.sbatch
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
    ├── train_covid.csv
    ├── train_non_covid.csv
    ├── val_covid.csv
    └── val_non_covid.csv
```

## Training

```bash
# Submit training job (depends on extract completing first)
EXTRACT_JOB=$(sbatch --parsable slurm/extract.sbatch)
sbatch --dependency=afterok:$EXTRACT_JOB slurm/train.sbatch

# Or if data is already extracted:
sbatch slurm/train.sbatch

# Run directly (debug / local):
python src/train.py --config configs/default.yaml --phase 0
```

Training phases:
- **Phase 1** (epochs 1–10): Frozen backbone, head-only, lr=1e-3
- **Phase 2a** (epochs 1–15): Unfreeze `denseblock4+norm5`, lr=1e-4
- **Phase 2b** (epochs 1–15): Unfreeze `denseblock3+transition3`, lr=5e-5

Checkpoints: `checkpoints/phase1_best.pt`, `checkpoints/phase2a_best.pt`, `checkpoints/phase2b_best.pt`, `checkpoints/best.pt`

## Evaluation

```bash
python src/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pt \
    --data-dir data \
    --metadata-dir data/metadata
```

Outputs per-source F1, tuned threshold, and final challenge score:
```
=======================================================
PER-SOURCE MACRO F1 SCORES [No TTA]
=======================================================
    source_0: 0.xxxx
    source_1: 0.xxxx
    source_2: 0.xxxx
    source_3: 0.xxxx
     average: 0.xxxx  ★

Tuned threshold (TTA):  0.xx  →  avg F1: 0.xxxx

Final Challenge Score (P): 0.xxxx
```

Flags:
- `--no-tta` — skip TTA (faster)
- `--no-tune-threshold` — use default threshold of 0.5

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | DenseNet-121 (RadImageNet pretrained) |
| Image size | 224×224 |
| Slices/scan (training) | 64 (uniform sample) |
| Slices/scan (fast val) | 48 |
| Phase 1 LR | 1e-3 (head only) |
| Phase 2a LR | 1e-4 (denseblock4) |
| Phase 2b LR | 5e-5 (denseblock3) |
| Loss | BCEWithLogitsLoss + label smoothing (ε=0.05) |
| Grad clipping | max_norm=1.0 |
| Batch sampler | Center-stratified (equal center representation) |
| Threshold | Tuned on val (0.30–0.70 sweep) |
| TTA | 4 augmentations (identity, hflip, rotate ±15°) |
| Early stopping patience | 10 epochs |

## Updating from Laptop → Nexus

```bash
# Laptop: make changes, commit, push
git add -p && git commit -m "..." && git push

# Nexus: pull latest
cd /fs/nexus-scratch/aadit/covid-challenge
git pull
sbatch slurm/train.sbatch
```

## Requirements

- Python 3.10+
- PyTorch 2.x + CUDA 11.8
- SLURM cluster with GPU (tested on UMD Nexus, `tron`/`scavenger` partitions)
- `unrar` system module: `module load unrar/7.0.9`
