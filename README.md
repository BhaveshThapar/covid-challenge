# Multi-Source Covid-19 Detection Challenge

Binary COVID / non-COVID classification of chest CT scans across 4 hospital sources,
for the **PHAROS-AIF-MIH** competition at **CVPR 2026**.

**Metric:** Average macro F1 across the 4 data centres.

---

## Repository Branch Structure

This repository is organized into **model-specific branches** and a **final ensemble branch**.
Each `final_*` branch trains a single backbone and produces a checkpoint that feeds into the ensemble.

```
final_EfficientNet_B3   ← you are here (default branch)
final_DenseNet-121      ← RadImageNet-pretrained DenseNet-121 (slice-level sigmoid)
final_dinoV2            ← Self-supervised DINOv2 ViT-B/14 (slice-level sigmoid)
final_ConvNeXt          ← ConvNeXt-Tiny Gated Attention MIL
final_EfficientNet_V2   ← EfficientNetV2-S Gated Attention MIL
final_ensemble          ← Combines all 9 checkpoints into the submission ensemble
```

### How the branches relate

```
┌─────────────────────┐
│  final_dinoV2       │──► checkpoints/v1_ovr_best.pt
├─────────────────────┤
│  final_DenseNet-121 │──► checkpoints/v4_ovr_best.pt
├─────────────────────┤                                     ┌────────────────────┐
│  final_EfficientNet │──► checkpoints/exp_b3_s42/best.pt   │                    │
│  _B3 (this branch)  │──► checkpoints/exp_b3_s7/best.pt    │   final_ensemble   │
│   (4 seed/SWA       │──► checkpoints/exp_b3_s7/swa_best   │                    │
│    variants)        │──► checkpoints/exp_b3_s123/best.pt ──►  9-model ensemble │
├─────────────────────┤                                     │  probability avg   │
│  final_ConvNeXt     │──► checkpoints/exp_cnxt_s42/best.pt │  + per-source      │
├─────────────────────┤                                     │    threshold sweep  │
│  final_EfficientNet │──► checkpoints/exp_effv2_s42/best   │                    │
│  _V2                │──► checkpoints/exp_effv2_s42/swa    └────────────────────┘
└─────────────────────┘
```

The **9 models** in the final ensemble are:

| # | Branch | Backbone | Inference paradigm | Checkpoint |
|---|--------|----------|--------------------|------------|
| 1 | `final_dinoV2` | DINOv2 ViT-B/14 | Slice-level sigmoid averaging | `v1_ovr_best.pt` |
| 2 | `final_DenseNet-121` | DenseNet-121 (RadImageNet) | Slice-level sigmoid averaging | `v4_ovr_best.pt` |
| 3 | `final_EfficientNet_B3` | EfficientNet-B3 (seed 42) | Gated Attention MIL softmax | `exp_b3_s42/best.pt` |
| 4 | `final_EfficientNet_B3` | EfficientNet-B3 (seed 7) | Gated Attention MIL softmax | `exp_b3_s7/best.pt` |
| 5 | `final_EfficientNet_B3` | EfficientNet-B3 (seed 7, SWA) | Gated Attention MIL softmax | `exp_b3_s7/swa_best.pt` |
| 6 | `final_EfficientNet_B3` | EfficientNet-B3 (seed 123) | Gated Attention MIL softmax | `exp_b3_s123/best.pt` |
| 7 | `final_ConvNeXt` | ConvNeXt-Tiny | Gated Attention MIL softmax | `exp_cnxt_s42/best.pt` |
| 8 | `final_EfficientNet_V2` | EfficientNetV2-S | Gated Attention MIL softmax | `exp_effv2_s42/best.pt` |
| 9 | `final_EfficientNet_V2` | EfficientNetV2-S (SWA) | Gated Attention MIL softmax | `exp_effv2_s42/swa_best.pt` |

---

## This Branch: `final_EfficientNet_B3`

### Architecture

Two-phase Gated Attention MIL pipeline:

```
Phase 1 (slice-level pretraining):
  CT Slice (224×224) → EfficientNet-B3 → Dropout → Linear(1536, 2) → CrossEntropy

Phase 2 (scan-level end-to-end):
  K slices per scan
    → EfficientNet-B3 (chunked, 8 slices at a time)
    → 1536-dim embeddings per slice
    → Gated Attention Pooling (tanh + sigmoid gate → softmax weights)
    → Weighted scan embedding (1536-dim)
    → MLP classifier (1536 → 256 → 2)
    → CrossEntropy + gradient accumulation + FP16
```

Memory optimizations in Phase 2:
- Gradient checkpointing on the backbone (~60% VRAM reduction)
- Chunked forward pass (8 slices per chunk through the backbone)
- Gradient accumulation (effective batch ~32 from physical batch 4)

### Project Structure

```
covid-challenge/                          ← repo root
├── src/
│   ├── model.py          SliceClassifier (Phase 1) + AttentionPooling + CovidDetector (Phase 2)
│   ├── dataset.py         SliceDataset, ScanDataset, WeightedRandomSampler, DataLoader builders
│   ├── train.py           2-phase training loop (Phase 1: slice, Phase 2: scan with attention MIL)
│   ├── evaluate.py        Load checkpoint → scan-level inference → per-source macro F1 + confusion matrices
│   └── utils.py           Seeding, YAML config loading, logging, F1 metrics, CheckpointManager, EarlyStopping
├── scripts/
│   ├── extract_data.py    Extract RAR/ZIP competition archives into data/{train,val}/{covid,non_covid}/
│   └── analyze_data.py    Print dataset statistics (class balance, slice counts, source distribution)
├── slurm/
│   ├── extract.sbatch     SLURM job: data extraction
│   └── train.sbatch       SLURM job: Phase 1 + Phase 2 training then evaluation
├── configs/
│   └── default.yaml       All hyperparameters (model, Phase 1, Phase 2, eval)
├── setup_env.sh           Create venv, install PyTorch + dependencies
├── EXPLAINER.md           Plain-English explanation of the model for non-ML readers
└── README.md              This file
```

### File Details

| File | What it does |
|------|-------------|
| `src/model.py` | Defines three classes: **SliceClassifier** (EfficientNet-B3 backbone + linear head for Phase 1), **AttentionPooling** (gated attention MIL from Ilse et al. ICML 2018), and **CovidDetector** (full scan-level model combining backbone + attention + classifier). `CovidDetector.from_slice_classifier()` transfers Phase 1 backbone weights into Phase 2. |
| `src/dataset.py` | **SliceDataset**: expands scans into individual slice samples (capped at 64 per scan via uniform spacing). **ScanDataset**: returns K slices per scan with padding and masking. Phase 1 uses `WeightedRandomSampler` for class-balanced sampling. Phase 2 uses `shuffle=True` on scan-level batches. `scan_collate_fn` pads variable-length scans to the batch maximum. |
| `src/train.py` | Orchestrates both training phases. Phase 1 freezes BatchNorm for early epochs, uses AdamW + cosine LR. Phase 2 initializes from Phase 1 backbone, adds gradient accumulation and FP16 mixed precision. Both phases validate per-source macro F1 each epoch and save `checkpoints/best.pt`. |
| `src/evaluate.py` | Standalone evaluation: loads a CovidDetector checkpoint, runs scan-level inference with optional FP16, prints per-source F1, confusion matrices, and the final challenge score. |
| `src/utils.py` | `set_seed()` for reproducibility, `load_config()` for YAML parsing, `get_logger()`, `compute_per_source_f1()` (macro F1 per hospital then averaged), `print_confusion_matrices()`, `EarlyStopping`, and `CheckpointManager` (save/load `.pt` files). |
| `scripts/extract_data.py` | Handles RAR and ZIP extraction from competition archives, filters out macOS metadata (`__MACOSX`), organizes into `data/{train,val}/{covid,non_covid}/ct_scan_*/`. |
| `scripts/analyze_data.py` | Prints class distribution, source distribution, and per-scan slice count statistics for any split. |
| `configs/default.yaml` | Central configuration: backbone choice, image size, slices per scan, Phase 1 and Phase 2 hyperparameters (LR, batch size, epochs, gradient accumulation, etc.), evaluation settings. |
| `setup_env.sh` | Loads Python 3.10 module, creates venv, installs PyTorch (CUDA 11.8), timm, albumentations, scikit-learn, etc. |
| `EXPLAINER.md` | Non-technical explanation of the pipeline for readers without ML background. |

---

## Setup

```bash
# On the SLURM cluster (requires Python 3.10 module)
bash setup_env.sh

# Or manually
module load Python3/3.10.14
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm albumentations scikit-learn pandas Pillow opencv-python-headless tensorboard tqdm pyyaml
```

## Data Preparation

Download the competition archives into `datasets/`, then:

```bash
sbatch slurm/extract.sbatch          # via SLURM
# or
python scripts/extract_data.py --datasets-dir datasets --data-dir data
```

Expected layout after extraction:
```
data/
├── train/
│   ├── covid/          ct_scan_*/  folders of JPEG slices
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
# Full pipeline (Phase 1 + Phase 2 + evaluation) via SLURM
sbatch slurm/train.sbatch

# Or run directly
source venv/bin/activate
python src/train.py --config configs/default.yaml --phase 0       # both phases
python src/train.py --config configs/default.yaml --phase 1       # Phase 1 only
python src/train.py --config configs/default.yaml --phase 2 \
    --resume-phase1 checkpoints/best.pt                           # Phase 2 from existing Phase 1
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

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Backbone | EfficientNet-B3 (ImageNet pretrained) | Same (from Phase 1) |
| Image size | 224×224 | 224×224 |
| Slices per scan | 64 (uniform sampling) | 16 |
| Batch size | 8 | 4 (effective ~32 via grad accum) |
| Learning rate | 1e-4 | 3e-5 |
| Optimizer | AdamW (wd=0.01) | AdamW (wd=0.01) |
| Mixed precision | Off | FP16 |
| Loss | CrossEntropy | CrossEntropy |
| Early stopping | — | 5 epochs patience |
| BN freeze | First 2 epochs | — |
| Gradient accumulation | — | 8 steps |

## Requirements

- Python 3.10+
- PyTorch 2.x + CUDA 11.8
- SLURM cluster with GPU (tested on UMD Nexus, `scavenger` partition)
