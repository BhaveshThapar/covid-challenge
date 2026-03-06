# Multi-Source Covid-19 Detection Challenge

Binary Covid/Non-Covid classification of chest CT scans across 4 hospital sources.

## Architecture (aadit-dev-v3 branch)

**DenseNet-121 + RadImageNet, scan-level MIL training with attention pooling:**

```
CT Scan (K=64 slices)
  → ROI crop each slice to 224×224 (OpenCV Otsu + connected components)
  → DenseNet-121 features (MixStyle injected after denseblock1, denseblock2)
  → Per-slice 1024-d embeddings
  → ABMIL attention: Linear(1024→128) → Tanh → Linear(128→1) → softmax over K
  → Weighted sum → 1024-d scan embedding
  → Dropout + Linear(1024→1)
  → Per-center threshold → Covid / Non-Covid
```

**Metric:** Plain average macro F1 across 4 centres (challenge score). Checkpoint selection uses weighted F1 = (F1₀ + F1₁ + 0.2·F1₂ + F1₃) / 3.2.

## Project Structure

```
covid-challenge/
├── src/
│   ├── model.py       # DenseNetMILClassifier + MixStyle (v3); DenseNetCovidClassifier kept for v1/v2 compat
│   ├── dataset.py     # ScanDataset, SliceDataset, ROI crop, CenterBatchSampler, intensity TTA transforms
│   ├── train.py       # Scan-level MIL training: Phase 1 (frozen) + Phase 2 (gradual unfreeze)
│   ├── evaluate.py    # MIL inference, intensity TTA, per-center threshold tuning, per-source F1
│   └── utils.py       # Weighted F1, checkpointing, early stopping
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
# 1. Clone the aadit-dev branch
cd /fs/nexus-scratch/aadit
git clone -b aadit-dev-v3 https://github.com/BhaveshThapar/covid-challenge.git covid-challenge
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

## Data Sampling

**Training (scan-level MIL):** `ScanDataset` is used for training — each dataset item is a full scan. 64 slices are sampled uniformly per scan; after preprocessing (ROI crop → 224×224), they are stacked into a `(K=64, 3, H, W)` bag. `CenterBatchSampler` operates at the scan level, assembling batches of 8 scans with balanced hospital center representation. `scan_collate_fn` pads bags to a fixed K and produces a boolean mask so the attention layer ignores padded positions. Each scan's bag is forwarded as `(B=8, K=64, 3, H, W)` and the model's attention pooling collapses K slices into a single scan embedding internally.

**Validation (scan-level MIL):** 48 slices are sampled per scan. The same MIL forward pass is used — attention pooling produces one logit per scan directly, with no post-hoc averaging needed. Batch size = 1 scan.

## Training

```bash
# Submit full training job (Phase 1 → Phase 2 sequentially):
sbatch slurm/train.sbatch

# Run directly (debug / local):
python src/train.py --config configs/default.yaml \
    --data-dir data --metadata-dir datasets --run-name v3 --phase 0

# Resume Phase 2b only (if phase2a_best.pt already exists):
python src/train.py --config configs/default.yaml \
    --data-dir data --metadata-dir datasets --run-name v3 --phase 3
```

> **Cluster:** `tron` partition, `--qos=hi`, RTX A6000 GPU.

Training phases:
- **Phase 1** (epochs 1–10): Frozen backbone, head-only, lr=1e-3, **batch_size=8 scans**
- **Phase 2a** (epochs 1–15): Unfreeze `denseblock4+norm5`, lr=1e-4
- **Phase 2b** (epochs 1–15): Unfreeze `denseblock3+transition3`, lr=3e-5

Checkpoints: `checkpoints/v3_phase1_best.pt`, `v3_phase2a_best.pt`, `v3_phase2b_best.pt`, `v3_ovr_best.pt`

## Evaluation

```bash
python src/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/v3_ovr_best.pt \
    --data-dir data \
    --metadata-dir datasets
```

Outputs per-center thresholds, per-source F1, weighted F1, and challenge score:
```
Per-center thresholds: {0: '0.45 (F1=0.xxxx)', 1: '0.50 ...', 2: '0.38 ...', 3: '0.47 ...'}

=======================================================
PER-SOURCE MACRO F1 SCORES [No TTA]
=======================================================
    source_0: 0.xxxx
    source_1: 0.xxxx
    source_2: 0.xxxx
    source_3: 0.xxxx
     average: 0.xxxx  ★
  [sklearn legacy (classes in y_true∪y_pred)]: avg = 0.xxxx
  Weighted F1 (checkpoint metric): 0.xxxx

Final Challenge Score (P): 0.xxxx
```

Flags:
- `--no-tta` — skip TTA (faster)
- `--no-tune-threshold` — use default threshold of 0.5

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | DenseNet-121 (RadImageNet pretrained) |
| Image preprocessing | Lung ROI heuristic crop → 224×224 (OpenCV Otsu + connected components) |
| Scan aggregation | ABMIL attention (1024→128→tanh→1→softmax→weighted sum) |
| MixStyle | Beta(0.1,0.1) feature stat mixing after denseblock1 and denseblock2; no-op at eval |
| Image size | 224×224 |
| Slices/scan (training) | 64 per scan (MIL bag) |
| Slices/scan (fast val) | 48 per scan (MIL forward) |
| Phase 1 batch size | 8 scans |
| Phase 1 LR | 1e-3 (head only) |
| Phase 2a LR | 1e-4 (denseblock4) |
| Phase 2b LR | 3e-5 (denseblock3) |
| Training augmentations | RandomGamma(85–115, p=0.5), CLAHE(p=0.3), GaussNoise(σ≈0.01, p=0.2) |
| Loss | Per-sample BCE + asymmetric center weights {0:1.0, 1:1.0, 2:0.2, 3:1.0} + label smoothing (ε=0.05) |
| Checkpoint metric | Weighted avg F1 = (F1₀ + F1₁ + 0.2·F1₂ + F1₃) / 3.2 |
| Grad clipping | max_norm=1.0 |
| Batch sampler | Center-stratified at scan level |
| Threshold | Per-center sweep 0.30–0.70 (4 independent thresholds) |
| TTA | 4 intensity passes: identity, γ=0.9, γ=1.1, CLAHE |
| F1 reporting | Dual: strict challenge formula (labels=[0,1]) + sklearn legacy |
| Early stopping patience | 10 epochs (on weighted F1) |
| AMP | bfloat16 Phase 2; float32 Phase 1 |
| Eval batch size | 1 scan |

## Known Issues & Cluster Notes

| Issue | Fix applied |
|-------|-------------|
| `val_covid.csv` not found (all sources = -1) | Code now tries `validation_*.csv` as fallback |
| `FileNotFoundError: phase1_best.pt` (checkpoint rotation) | `save_named()` bypasses max_keep rotation |
| `UnpicklingError` loading checkpoints (PyTorch 2.6) | `weights_only=False` in `CheckpointManager.load()` |
| NaN loss in Phase 2 (float16 DenseNet overflow) | AMP uses bfloat16; falls back to float32 if unsupported |
| OOM on full-slice validation (V100, 16 GB) | `full_val_every_n_epochs: 999`; `eval.batch_size: 1` |
| 1-2 missing scan directories | Logged at startup, training continues without them |

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
- PyTorch 2.6+ + CUDA 11.8
- SLURM cluster with GPU (tested on UMD Nexus, `tron` partition, `--qos=hi`, RTX A6000)
- `unrar` system module: `module load unrar/7.0.9`
