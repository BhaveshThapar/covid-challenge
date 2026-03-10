# Multi-Source Covid-19 Detection Challenge

Binary Covid/Non-Covid classification of chest CT scans across 4 hospital sources.

**Final Score: 0.9279 macro F1** (5-model ensemble with per-source threshold calibration)

## Architecture

**2-stage pipeline with 5-model ensemble:**
1. **Phase 1** — Backbone pretrained on individual CT slices (20 epochs)
2. **Phase 2** — End-to-end scan-level training with gated attention MIL pooling (30 epochs)
3. **Ensemble** — 5 diverse models with score-weighted soft voting

```
CT Slices (256x256) → Backbone → Gated Attention MIL → Ensemble Voting → Covid / Non-Covid
```

**Metric:** Average macro F1 across 4 data centres

## Project Structure

```
covid-challenge/
├── src/
│   ├── model.py              # EfficientNet-B3 + Attention MIL
│   ├── dataset.py            # Slice & scan-level dataloaders
│   ├── train.py              # 2-phase training loop
│   ├── evaluate.py           # Per-source F1 evaluation
│   ├── ensemble_evaluate.py  # Multi-model ensemble inference
│   ├── losses.py             # Focal Loss implementation
│   └── utils.py              # Metrics, checkpointing, early stopping
├── scripts/
│   ├── extract_data.py       # Archive extraction & organization
│   ├── analyze_data.py       # Dataset statistics
│   └── generate_figures.py   # Paper figure generation
├── slurm/
│   ├── extract.sbatch        # Data extraction SLURM job
│   ├── train.sbatch          # GPU training SLURM job
│   └── submit_ensemble_v2.sh # Submit all 5 ensemble models
├── configs/
│   ├── default.yaml          # Default hyperparameters
│   ├── exp_b3_s42.yaml       # EfficientNet-B3 seed 42
│   ├── exp_b3_s123.yaml      # EfficientNet-B3 seed 123
│   ├── exp_b3_s7.yaml        # EfficientNet-B3 seed 7
│   ├── exp_cnxt_s42.yaml     # ConvNeXt-Tiny seed 42
│   └── exp_effv2_s42.yaml    # EfficientNetV2-S seed 42
├── paper/
│   ├── main.tex              # LaTeX research paper
│   └── references.bib        # Bibliography
├── checkpoints/              # Trained model weights
└── setup_env.sh              # Environment setup
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
│   ├── covid/          # 564 ct_scan_*/ folders of JPEG slices
│   └── non_covid/      # 659 ct_scan_*/ folders
├── val/
│   ├── covid/          # 128 scans
│   └── non_covid/      # 180 scans
└── metadata/
    ├── val_covid.csv
    └── val_non_covid.csv
```

## Training

### Single model
```bash
source venv/bin/activate
python src/train.py --config configs/default.yaml --phase 0
```

### Full ensemble (5 models)
```bash
bash slurm/submit_ensemble_v2.sh
```

## Evaluation

### Single model
```bash
python src/evaluate.py --checkpoint checkpoints/best.pt
```

### Ensemble (final submission)
```bash
python src/ensemble_evaluate.py
```

Outputs per-source macro F1 and the final challenge score:
```
  source_0:  0.9545
  source_1:  0.8379
  source_2:  1.0000
  source_3:  0.9192
  average:   0.9279  ★
```

## Ensemble Configuration

| Model | Backbone | Seed | Techniques |
|-------|----------|------|------------|
| M1 | EfficientNet-B3 | 42 | Focal Loss, Mixup |
| M2 | EfficientNet-B3 | 123 | Focal Loss, Mixup |
| M3 | EfficientNet-B3 | 7 | SWA, per-source weights |
| M4 | ConvNeXt-Tiny | 42 | Architectural diversity |
| M5 | EfficientNetV2-S | 42 | SWA, per-source weights |

## Key Hyperparameters

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Backbone | EfficientNet-B3 | EfficientNet-B3 |
| Image size | 256x256 | 256x256 |
| Slices per scan | 64 (train) | 24 (train) / 48 (eval) |
| Learning rate | 1e-4 | 1e-5 (backbone: 1e-6) |
| Weight decay | 0.01 | 0.05 |
| Batch size | 8 | 2 (effective: 32) |
| Loss | Cross-entropy | Focal (gamma=2) |
| Mixed precision | FP16 | FP16 |
| Early stopping | — | patience=8 |

## Compute

- GPU: NVIDIA RTX A6000 (48 GB) / GTX TITAN X (12 GB)
- Memory: 64 GB system RAM, 8 CPU cores
- Training time: 6-12 hours per model (both phases)
- Total ensemble: ~50 GPU-hours

## Requirements

- Python 3.10+
- PyTorch 2.x + CUDA 11.8
- SLURM cluster with GPU access
