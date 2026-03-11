# Multi-Source COVID-19 Detection — Ensemble

Binary Covid/Non-Covid classification of chest CT scans across four hospital data sources. This branch combines three independently trained models (DINOv2, DenseNet, EfficientNet) into a weighted ensemble for final predictions.

**Challenge metric:** Average macro F1 across the four data centres (per-source macro F1, excluding missing classes).

---

## Table of Contents

1. [Models](#models)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Quick Start](#quick-start)
5. [Ensemble Deep Dive](#ensemble-deep-dive)
6. [Usage Reference](#usage-reference)
7. [Config Reference](#config-reference)
8. [Known Issues](#known-issues)

---

## Models

| Model | Backbone | Architecture | Image Size | Contributor |
|-------|----------|--------------|------------|-------------|
| **DINOv2** | ViT-B/14 (self-supervised, torch.hub) | Slice-level sigmoid → scan avg | 224×224 | Anant |
| **DenseNet** | DenseNet-121 (RadImageNet) | Slice-level sigmoid → scan avg | 224×224 | Aadit |
| **EfficientNet** | EfficientNet-B3 (timm) + Attention MIL | Scan-level softmax | 256×256 | Bhavesh |

**Model provenance:** The model classes in `src/models/` were extracted from the respective contributor branches:
- **DINOv2**: `Anant-dev`
- **DenseNet**: `aadit-dev-v6` (Aadit also has `aadit-dev-v4`; v6 was used for extraction)
- **EfficientNet**: `bhavesh/improve-diversity`

They match the original implementations in those branches. Checkpoints are trained separately by each contributor.

---

## Project Structure

```
covid-challenge/
├── src/
│   ├── models/              # Modular model definitions
│   │   ├── dinov2.py        # DINOv2CovidClassifier
│   │   ├── densenet.py      # DenseNetCovidClassifier
│   │   └── efficientnet.py  # CovidDetector, SliceClassifier, AttentionPooling
│   ├── model.py             # Re-exports (backward compat)
│   ├── dataset.py           # ScanDataset, transforms, manifests
│   ├── ensemble.py          # Multi-GPU ensemble inference + validation
│   ├── predict_test.py      # Single-model test inference
│   ├── evaluate.py          # Single-model evaluation
│   └── utils.py             # Metrics, checkpointing
├── configs/
│   ├── ensemble.yaml        # Ensemble weights, checkpoint paths
│   ├── dinov2.yaml
│   ├── densenet.yaml
│   └── efficientnet.yaml
├── slurm/
│   ├── extract.sbatch       # Data download/extraction
│   ├── test.sbatch          # Single-model test (DINOv2)
│   ├── eval.sbatch          # Single-model evaluation
│   ├── ensemble.sbatch      # Ensemble prediction on test (3 GPUs)
│   └── ensemble_val.sbatch  # Ensemble on val + weight tuning
└── scripts/
    └── download_and_extract.py
```

---

## Setup

```bash
cd /fs/nexus-scratch/anant04  # or your cluster path
git clone <repo> covid-challenge
cd covid-challenge
bash setup_env.sh
```

**Checkpoints:**
- DINOv2: `checkpoints/v1_ovr_best.pt`
- DenseNet: `v4_ovr_best.pt` (project root or path in config)
- EfficientNet: `best.pt` from [BhaveshThapar/covid-checkpoints](https://huggingface.co/BhaveshThapar/covid-checkpoints)
- DenseNet RadImageNet: `checkpoints/radimagenet_densenet121.pt` (for DenseNet init)

**Data:**
```bash
sbatch slurm/extract.sbatch
```
Expected layout: `data/{train,val,test}/{covid,non_covid}/` with `ct_scan_*` subdirs of JPEG slices; metadata CSVs in `datasets/`.

---

## Quick Start

```bash
# Ensemble prediction on test set (3 GPUs)
sbatch slurm/ensemble.sbatch

# Ensemble on validation with weight tuning
python src/ensemble.py --split val --tune-weights

# Single-model evaluation
python src/evaluate.py --model dinov2 --checkpoint checkpoints/v1_ovr_best.pt
python src/predict_test.py --model densenet --checkpoint v4_ovr_best.pt --output pred_dense.csv
```

---

## Ensemble Deep Dive

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MAIN PROCESS                                         │
│  1. Load config (weights, checkpoint paths)                              │
│  2. Spawn 3 worker processes (one per model)                             │
│  3. Each worker: CUDA_VISIBLE_DEVICES = gpu_id, load model, run inference│
│  4. Collect (scan_names, prob_covid) from each via Queue                 │
│  5. Merge: prob_ens = w1·P_dino + w2·P_dense + w3·P_eff                  │
│  6. preds = (prob_ens >= threshold).astype(int)                         │
│  7. Write CSV                                                            │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   GPU 0: DINOv2        GPU 1: DenseNet      GPU 2: EfficientNet
```

Inference runs in parallel; wall time is about the slowest model.

### How Each Model Produces P(covid)

| Model | Output | Conversion |
|-------|--------|------------|
| DINOv2 | (B, 1) logits per slice | sigmoid → mean over slices (optionally TTA) |
| DenseNet | (B, 1) logits per slice | sigmoid → mean over slices (optionally TTA) |
| EfficientNet | (B, 2) logits per scan | softmax → `probs[:, 0]` (class 0 = covid) |

All outputs are P(covid) in [0, 1], so weighted averaging is consistent.

### Weights and Threshold

- **Default:** `[0.33, 0.33, 0.34]`, threshold 0.5 (in `configs/ensemble.yaml`)
- **Override:** `--weights 0.4 0.3 0.3` (normalized automatically)
- **Tuning:** `--split val --tune-weights` runs a grid search over weights and threshold to maximize validation F1

### Per-Source Threshold

The challenge metric averages macro F1 across sources. Using a different threshold per source can improve it:

```bash
python src/ensemble.py --split val --per-source-threshold
```

### Validation Mode

With `--split val` the ensemble uses the validation set (labels available) and reports:

- Per-source macro F1
- Confusion matrices per source
- Challenge score (average F1)

```bash
python src/ensemble.py --split val
python src/ensemble.py --split val --tune-weights --per-source-threshold
```

### Workflow Summary

1. Ensure all three checkpoints are present and paths in `configs/ensemble.yaml` are correct.
2. Run validation ensemble to inspect performance:
   ```bash
   python src/ensemble.py --split val --tune-weights
   ```
3. Update `configs/ensemble.yaml` with the printed weights (or use `--weights` on the command line).
4. Run test ensemble:
   ```bash
   sbatch slurm/ensemble.sbatch
   ```
5. Submit `predictions_ensemble.csv` to the challenge.

---

## Usage Reference

### Ensemble

```bash
# Test set (default)
python src/ensemble.py [--output predictions_ensemble.csv]

# Validation set (with F1 report)
python src/ensemble.py --split val

# Tune weights and threshold on validation
python src/ensemble.py --split val --tune-weights

# Per-source threshold sweep
python src/ensemble.py --split val --per-source-threshold

# Override weights / threshold
python src/ensemble.py --weights 0.4 0.3 0.3 --threshold 0.45
```

### Single-Model Predict

```bash
python src/predict_test.py --model dinov2 --checkpoint checkpoints/v1_ovr_best.pt [--tta]
python src/predict_test.py --model densenet --checkpoint v4_ovr_best.pt --output pred_dense.csv
python src/predict_test.py --model efficientnet --checkpoint best.pt
```

### Single-Model Evaluate

```bash
python src/evaluate.py --model dinov2 --checkpoint checkpoints/v1_ovr_best.pt [--no-tta]
python src/evaluate.py --model densenet --checkpoint v4_ovr_best.pt
python src/evaluate.py --model efficientnet --checkpoint best.pt
```

---

## Config Reference

### `configs/ensemble.yaml`

```yaml
ensemble:
  weights: [0.33, 0.33, 0.34]   # [dinov2, densenet, efficientnet]
  threshold: 0.5

models:
  dinov2:
    config: configs/dinov2.yaml
    checkpoint: checkpoints/v1_ovr_best.pt
  densenet:
    config: configs/densenet.yaml
    checkpoint: v4_ovr_best.pt
  efficientnet:
    config: configs/efficientnet.yaml
    checkpoint: best.pt
```

### Model Configs

Each model has its own config for image size, `slices_per_scan`, TTA, etc. DINOv2 and DenseNet use 224×224; EfficientNet uses 256×256. DINOv2 and DenseNet use TTA (4 augs) in the ensemble; EfficientNet does not.

---

## Optimization and Tuning

| Option | Description |
|--------|-------------|
| `--tune-weights` | Grid search over weights and threshold to maximize val F1 |
| `--per-source-threshold` | Sweep a separate threshold per data source |
| `--weights w1 w2 w3` | Manual weight override (normalized to sum=1) |
| `--threshold t` | Override classification threshold |

Recommended workflow: run `--split val --tune-weights` to find best weights, then update `configs/ensemble.yaml` or pass `--weights` for test inference.

---

## Known Issues

| Issue | Notes |
|-------|-------|
| Metadata path | Use `--metadata-dir datasets` (not `data/metadata`) |
| DenseNet RadImageNet | Set `pretrained_path` in `configs/densenet.yaml`; if missing, trains from scratch |
| Empty val/test | Run `sbatch slurm/extract.sbatch` first |
| 3 GPUs required | Ensemble needs 3 GPUs; adjust `--gpus` if your machine differs |

---

## Requirements

- Python 3.10+
- PyTorch 2.x + CUDA
- albumentations, timm, torchvision, sklearn
- SLURM (for cluster jobs)
