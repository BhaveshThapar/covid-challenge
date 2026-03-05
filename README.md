# Multi-Source Covid-19 Detection Challenge

Binary Covid/Non-Covid classification of chest CT scans across 4 hospital sources.

## Architecture (DINOv2 branch)

**DINOv2 ViT-B/14 + slice-level training, scan-level evaluation:**

```
CT Slices → DINOv2 ViT-B/14 (self-supervised) → Average Slice Probs → Threshold → Covid / Non-Covid
```

Training uses progressive backbone unfreezing (Phase 1: frozen; Phase 2a: last 2 blocks; Phase 2b: last 4 blocks).
Scan-level predictions average per-slice sigmoid probabilities (no learned attention).

**Metric:** Average macro F1 across 4 data centres

## Project Structure

```
covid-challenge/
├── src/
│   ├── model.py       # DINOv2CovidClassifier (DINOv2 ViT-B/14)
│   ├── dataset.py     # SliceDataset, ScanDataset, CenterBatchSampler, TTA transforms
│   ├── train.py       # Phase 1 (frozen) + Phase 2 (gradual unfreeze) training
│   ├── evaluate.py    # Scan-level inference, threshold tuning, TTA, per-source F1
│   └── utils.py       # Metrics, checkpointing, early stopping
├── scripts/
│   └── download_and_extract.py  # gdown download + archive extraction + dataset analysis
├── slurm/
│   ├── extract.sbatch    # Data download/extraction SLURM job (tron partition)
│   └── train.sbatch      # GPU training SLURM job (tron partition, qos=medium)
├── configs/
│   └── default.yaml      # Hyperparameters
└── setup_env.sh           # Environment setup
```

## Setup (on Nexus cluster)

```bash
# 1. Clone the Anant-dev branch
cd /fs/nexus-scratch/anant04
git clone -b Anant-dev https://github.com/BhaveshThapar/covid-challenge.git covid-challenge
cd covid-challenge

# 2. Create environment
bash setup_env.sh

# DINOv2 weights are downloaded automatically via torch.hub on first run (no manual download).
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
└── val/
    ├── covid/
    └── non_covid/

datasets/               # metadata CSVs live here (alongside raw archives)
├── train_covid.csv
├── train_non_covid.csv
├── validation_covid.csv       # NOTE: named "validation_", not "val_"
└── validation_non_covid.csv   # code handles both automatically
```

> **Note:** The metadata CSVs are in `datasets/` (alongside the raw archive files), **not**
> `data/metadata/`. Always pass `--metadata-dir datasets` to train.py and evaluate.py.
> The validation CSV names use `validation_*.csv`; the code tries `val_*.csv` first and
> falls back to `validation_*.csv` automatically.

## Training

```bash
# Submit full training job (Phase 1 → Phase 2 sequentially):
sbatch slurm/train.sbatch

# Or submit Phase 2 only (if phase1_best.pt already exists):
BASH_ENV=/usr/share/Modules/init/bash sbatch \
  --job-name=covid-phase2 \
  --partition=tron --account=nexus --qos=medium \
  --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=10:00:00 \
  --output=logs/phase2_%j.out --error=logs/phase2_%j.err \
  --wrap='cd /fs/nexus-scratch/anant04/covid-challenge &&
          source /usr/share/Modules/init/bash &&
          module load Python3/3.10.14 &&
          source venv/bin/activate &&
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          python -u src/train.py --config configs/default.yaml \
            --data-dir data --metadata-dir datasets --phase 2'

# Run directly (debug / local):
python src/train.py --config configs/default.yaml --phase 0
```

> **Partition:** Use `tron --qos=medium` (not `scavenger`) for training. Tron gives newer GPUs
> (no preemption) and `--qos=medium` is required to get 8 CPUs + 64 GB RAM
> (default QoS caps at 4 CPUs / 32 GB which is insufficient for `num_workers=8`).

Training phases:
- **Phase 1** (epochs 1–10): Frozen backbone, head-only, lr=1e-3
- **Phase 2a** (epochs 1–15): Unfreeze last 2 blocks + norm, lr=1e-4
- **Phase 2b** (epochs 1–15): Unfreeze last 4 blocks + norm, lr=5e-5

Checkpoints: `checkpoints/{run_name}_phase1_best.pt`, `{run_name}_phase2a_best.pt`, `{run_name}_phase2b_best.pt`, `{run_name}_ovr_best.pt`

## Evaluation

```bash
python src/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/v1_ovr_best.pt \
    --data-dir data \
    --metadata-dir datasets
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
| Backbone | DINOv2 ViT-B/14 (self-supervised, torch.hub) |
| Image size | 224×224 |
| Slices/scan (training) | 64 (uniform sample) |
| Slices/scan (fast val) | 48 |
| Phase 1 LR | 1e-3 (head only) |
| Phase 2a LR | 1e-4 (blocks 10–11, norm) |
| Phase 2b LR | 5e-5 (blocks 8–9) |
| Loss | BCEWithLogitsLoss + label smoothing (ε=0.05) |
| Grad clipping | max_norm=1.0 |
| Batch sampler | Center-stratified (equal center representation) |
| Threshold | Tuned on val (0.30–0.70 sweep) |
| TTA | 4 augmentations (identity, hflip, rotate ±15°) |
| Early stopping patience | 10 epochs |
| AMP | bfloat16 on Ampere GPUs; falls back to float32 on Turing/Pascal |
| Eval batch size | 1 scan at a time (prevents OOM on full-slice eval) |

## Known Issues & Cluster Notes

| Issue | Fix applied |
|-------|-------------|
| `val_covid.csv` not found (all sources = -1) | Code now tries `validation_*.csv` as fallback |
| `FileNotFoundError: phase1_best.pt` (checkpoint rotation) | `save_named()` bypasses max_keep rotation |
| `UnpicklingError` loading checkpoints (PyTorch 2.6) | `weights_only=False` in `CheckpointManager.load()` |
| NaN loss in Phase 2 (float16 overflow) | AMP uses bfloat16; falls back to float32 if unsupported |
| OOM on full-slice validation (V100, 16 GB) | `full_val_every_n_epochs: 999`; `eval.batch_size: 1` |
| 1-2 missing scan directories | Logged at startup, training continues without them |

## Updating from Laptop → Nexus

```bash
# Laptop: make changes, commit, push
git add -p && git commit -m "..." && git push

# Nexus: pull latest
cd /fs/nexus-scratch/anant04/covid-challenge
git pull
sbatch slurm/train.sbatch
```

## Requirements

- Python 3.10+
- PyTorch 2.6+ + CUDA 11.8
- SLURM cluster with GPU (tested on UMD Nexus, `tron` partition, qos=medium)
- `unrar` system module: `module load unrar/7.0.9`
