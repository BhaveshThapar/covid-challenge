# Multi-Source Covid-19 Detection Challenge

Binary COVID / non-COVID classification of chest CT scans across 4 hospital sources,
for the **PHAROS-AIF-MIH** competition at **CVPR 2026**.

**Metric:** Average macro F1 across the 4 data centres.

---

## Repository Branch Structure

Each `final_*` branch trains a single backbone; checkpoints feed into `final_ensemble`.

```
final_EfficientNet_B3   ← EfficientNet-B3 Gated Attention MIL (4 seed/SWA variants)
final_DenseNet-121      ← you are here
final_dinoV2            ← Self-supervised DINOv2 ViT-B/14 (slice-level sigmoid)
final_ConvNeXt          ← ConvNeXt-Tiny Gated Attention MIL
final_EfficientNet_V2   ← EfficientNetV2-S Gated Attention MIL
final_ensemble          ← Combines all 9 checkpoints → final submission
```

The 9-model ensemble is described in `final_ensemble`; this branch produces **checkpoint `v4_ovr_best.pt`** used by model #2 in that ensemble.

---

## This Branch: `final_DenseNet-121`

### Architecture

DenseNet-121 initialized from **RadImageNet** pretrained weights. Binary classification at the slice level; scan-level prediction via mean sigmoid averaging.

```
Phase 1 (frozen backbone, head-only):
  CT Slice (224×224) → DenseNet-121 (frozen) → Dropout(0.4) → Linear(1024, 1) → BCE

Phase 2a (unfreeze denseblock4 + norm5):
  CT Slice (224×224) → DenseNet-121 (denseblock4 trainable) → Dropout(0.4) → Linear(1024, 1) → BCE

Phase 2b (additionally unfreeze denseblock3 + transition3):
  CT Slice (224×224) → DenseNet-121 (denseblock3+4 trainable) → Dropout(0.4) → Linear(1024, 1) → BCE

Scan-level inference:
  All slices → per-slice sigmoid → mean probability → threshold → COVID / non-COVID
```

### Training Details

**Data loading:**
- Slices are loaded from JPEG folders via `SliceDataset`
- Each scan is capped at **64 slices** by deterministic uniform spacing (`np.linspace`)
- Training uses `CenterBatchSampler` which balances by **both centre and class**: each batch of 32 contains 4 slices per `(centre, label)` group (4 centres × 2 classes × 4 slices = 32)
- Smaller `(centre, label)` groups are oversampled with replacement each epoch

**Phase 1 — Head-only (10 epochs):**
- Backbone frozen; only `Dropout(0.4) → Linear(1024, 1)` trains
- AdamW, lr=1e-3, weight_decay=1e-4
- Linear warmup (1 epoch) → cosine annealing
- Label smoothing ε=0.05; BCEWithLogitsLoss with pos_weight
- No mixed precision (fast enough without)

**Phase 2a — Unfreeze denseblock4 (15 epochs):**
- Discriminative learning rates via `get_parameter_groups()`: head at 1e-3, denseblock4+norm5 at 1e-4
- CosineAnnealingWarmRestarts (T₀=5)
- bfloat16 mixed precision
- Label smoothing ε=0.05

**Phase 2b — Additionally unfreeze denseblock3 (15 epochs):**
- Reloads best Phase 2a checkpoint
- head at 1e-3, denseblock4+norm5 at 1e-4, denseblock3+transition3 at 3e-5
- Same scheduler and precision as 2a
- Best of Phase 2a vs 2b saved as `{run_name}_ovr_best.pt`

**Validation:**
- Scan-level: sample 48 slices per scan, average sigmoid probabilities, threshold at 0.5
- Metric: per-source macro F1 averaged across 4 centres
- TTA at final evaluation: 4 augmentations (identity, hflip, +15° rotate, −15° rotate)

### Project Structure

```
covid-challenge/
├── src/
│   ├── model.py          DenseNetCovidClassifier (RadImageNet init, freeze/unfreeze helpers)
│   ├── dataset.py         SliceDataset, ScanDataset, CenterBatchSampler (centre+class balanced), loaders
│   ├── train.py           3-phase training: Phase 1, Phase 2a, Phase 2b (+ Phase 2b-only resume)
│   ├── evaluate.py        Per-source macro F1, threshold tuning, TTA, confusion matrices
│   └── utils.py           Seeding, config, logging, F1 metrics, CheckpointManager, EarlyStopping
├── scripts/
│   ├── download_and_extract.py      Download + extract competition archives
│   └── download_and_extract_test.py Download + extract test set
├── slurm/
│   ├── extract.sbatch     Data extraction job
│   ├── train.sbatch       Training job (all phases)
│   ├── eval.sbatch        Evaluation job
│   └── download_test.sbatch  Test data download job
├── configs/
│   └── default.yaml       All hyperparameters
└── setup_env.sh           Environment setup
```

### File Details

| File | Purpose |
|------|---------|
| `src/model.py` | `DenseNetCovidClassifier`: loads RadImageNet weights, replaces classifier with `Dropout(0.4) → Linear(1024, 1)`. Provides `freeze_backbone()`, `unfreeze_block()`, and `get_parameter_groups()` for discriminative LR. |
| `src/dataset.py` | `SliceDataset` expands scans to slices (capped at 64). `CenterBatchSampler` groups slices by `(source, label)` and oversamples minority groups so each batch is balanced by both centre and class. `ScanDataset` + `scan_collate_fn` for scan-level validation with padding. `RawSliceScanDataset` for TTA. |
| `src/train.py` | `train_phase1()` → frozen backbone. `_run_subphase()` → shared unfreezing logic. `train_phase2()` → orchestrates 2a and 2b. `train_phase2b_only()` → resume 2b from existing 2a checkpoint (for killed jobs). |
| `src/evaluate.py` | Loads checkpoint, runs scan-level inference, sweeps threshold, prints per-source F1 and confusion matrices. Supports TTA via `get_tta_transforms()`. |
| `src/utils.py` | `set_seed()`, `load_config()`, `compute_per_source_f1()` (excludes missing classes per challenge rules), `CheckpointManager`, `EarlyStopping`. |

### Key Hyperparameters

| Parameter | Phase 1 | Phase 2a | Phase 2b |
|-----------|---------|----------|----------|
| Backbone | DenseNet-121 (RadImageNet, frozen) | denseblock4 unfrozen | + denseblock3 unfrozen |
| Image size | 224×224 | 224×224 | 224×224 |
| Slices/scan (train) | 64 | 64 | 64 |
| Batch size | 32 (via CenterBatchSampler) | 32 | 32 |
| Head LR | 1e-3 | 1e-3 | 1e-3 |
| Backbone LR | — | 1e-4 | 3e-5 (block3), 1e-4 (block4) |
| Precision | FP32 | bfloat16 | bfloat16 |
| Label smoothing | 0.05 | 0.05 | 0.05 |
| Loss | BCE + pos_weight | BCE + pos_weight | BCE + pos_weight |
| Scheduler | LinearWarmup → Cosine | CosineWarmRestarts (T₀=5) | CosineWarmRestarts (T₀=5) |

### Usage

```bash
# Setup
bash setup_env.sh

# Training (all phases)
python src/train.py --config configs/default.yaml --phase 0 --run-name v4

# Phase 2b only (resume from killed job)
python src/train.py --config configs/default.yaml --phase 3 --run-name v4

# Evaluation
python src/evaluate.py --checkpoint checkpoints/v4_ovr_best.pt
```

### Requirements

- Python 3.10+
- PyTorch 2.x + CUDA 11.8
- SLURM cluster with GPU (tested on UMD Nexus, `scavenger` partition)
