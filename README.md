# Multi-Source Covid-19 Detection Challenge

Binary COVID / non-COVID classification of chest CT scans across 4 hospital sources,
for the **PHAROS-AIF-MIH** competition at **CVPR 2026**.

**Metric:** Average macro F1 across the 4 data centres.

---

## Repository Branch Structure

Each `final_*` branch trains a single backbone; checkpoints feed into `final_ensemble`.

```
final_EfficientNet_B3   ← EfficientNet-B3 Gated Attention MIL (4 seed/SWA variants)
final_DenseNet-121      ← RadImageNet-pretrained DenseNet-121 (slice-level sigmoid)
final_dinoV2            ← you are here
final_ConvNeXt          ← ConvNeXt-Tiny Gated Attention MIL
final_EfficientNet_V2   ← EfficientNetV2-S Gated Attention MIL
final_ensemble          ← Combines all 9 checkpoints → final submission
```

This branch produces **checkpoint `v1_ovr_best.pt`** used by model #1 in the 9-model ensemble.

---

## This Branch: `final_dinoV2`

### Architecture

DINOv2 ViT-B/14 loaded from `facebookresearch/dinov2` via `torch.hub`. Self-supervised backbone (86M params, 768-dim CLS token, 14×14 patches). Binary classification at the slice level; scan-level prediction via mean sigmoid averaging.

```
Phase 1 (frozen backbone, head-only):
  CT Slice (224×224) → DINOv2 ViT-B/14 (frozen) → Dropout(0.4) → Linear(768, 1) → BCE

Phase 2a (unfreeze blocks.10, blocks.11, norm):
  CT Slice (224×224) → DINOv2 (last 2 blocks trainable) → Dropout(0.4) → Linear(768, 1) → BCE

Phase 2b (additionally unfreeze blocks.8, blocks.9):
  CT Slice (224×224) → DINOv2 (last 4 blocks trainable) → Dropout(0.4) → Linear(768, 1) → BCE

Scan-level inference:
  All slices → per-slice sigmoid → mean probability → threshold → COVID / non-COVID
```

### Training Details

**Data loading:**
- Slices loaded from JPEG folders via `SliceDataset`
- Each scan capped at **64 slices** by deterministic uniform spacing
- Training uses `CenterBatchSampler` which balances by **centre only**: each batch of 32 contains 8 slices per centre (4 centres × 8 slices = 32)
- Smaller centres oversampled with replacement each epoch

**Phase 1 — Head-only (10 epochs):**
- Backbone frozen; only `Dropout(0.4) → Linear(768, 1)` trains
- AdamW, lr=1e-3, weight_decay=1e-4
- Linear warmup (1 epoch) → cosine annealing
- Label smoothing ε=0.05; BCEWithLogitsLoss with pos_weight
- No mixed precision

**Phase 2a — Unfreeze last 2 transformer blocks (15 epochs):**
- `unfreeze_blocks_by_index([10, 11], also_norm=True)`
- Discriminative LR: head at 1e-3, blocks.10+11+norm at 1e-4
- CosineAnnealingWarmRestarts (T₀=5)
- bfloat16 mixed precision
- Gradient clipping (max_norm=1.0)

**Phase 2b — Additionally unfreeze blocks.8 and blocks.9 (15 epochs):**
- Reloads best Phase 2a checkpoint
- blocks.8+9 at 5e-5, blocks.10+11+norm at 1e-4, head at 1e-3
- Best of 2a vs 2b saved as `{run_name}_ovr_best.pt`

**Validation:**
- Scan-level: sample 48 slices, average sigmoid probabilities, threshold at 0.5
- TTA at final evaluation: 4 augmentations (identity, hflip, +15° rotate, −15° rotate)

### Project Structure

```
covid-challenge/
├── src/
│   ├── model.py          DINOv2CovidClassifier (torch.hub load, freeze/unfreeze by block index)
│   ├── dataset.py         SliceDataset, ScanDataset, CenterBatchSampler (centre-balanced), loaders
│   ├── train.py           3-phase training: Phase 1, Phase 2a, Phase 2b (+ 2b-only resume)
│   ├── evaluate.py        Per-source macro F1, threshold tuning, TTA, confusion matrices
│   ├── predict_test.py    Generate test-set predictions CSV
│   └── utils.py           Seeding, config, logging, metrics, CheckpointManager, EarlyStopping
├── scripts/
│   └── download_and_extract.py   Download + extract competition archives
├── slurm/
│   ├── extract.sbatch     Data extraction
│   ├── train.sbatch       Training (all phases)
│   ├── eval.sbatch        Evaluation
│   └── test.sbatch        Test-set inference
├── configs/
│   └── default.yaml       Hyperparameters
├── results/               Saved validation and test prediction CSVs
└── setup_env.sh           Environment setup
```

### File Details

| File | Purpose |
|------|---------|
| `src/model.py` | `DINOv2CovidClassifier`: loads ViT-B/14 from torch hub, replaces head with `Dropout(0.4) → Linear(768, 1)`. `freeze_backbone()`, `unfreeze_blocks_by_index()`, `get_parameter_groups()` for discriminative LR across transformer blocks. |
| `src/dataset.py` | `SliceDataset` expands scans to slices (capped at 64). `CenterBatchSampler` balances by centre, oversampling smaller centres. `ScanDataset` + `RawSliceScanDataset` for scan-level eval and TTA. |
| `src/train.py` | `train_phase1()` → frozen backbone. `_run_subphase()` → shared unfreezing logic with discriminative LR. `train_phase2()` → 2a then 2b. `train_phase2b_only()` for resume. All phases use `build_slice_dataloaders()`. |
| `src/evaluate.py` | Scan-level inference, threshold sweep, TTA, per-source F1, confusion matrices. |
| `src/predict_test.py` | Generate `predictions.csv` for challenge test-set submission. |

### Key Hyperparameters

| Parameter | Phase 1 | Phase 2a | Phase 2b |
|-----------|---------|----------|----------|
| Backbone | DINOv2 ViT-B/14 (frozen) | blocks.10-11 + norm | + blocks.8-9 |
| Image size | 224×224 | 224×224 | 224×224 |
| Slices/scan (train) | 64 | 64 | 64 |
| Batch size | 32 (CenterBatchSampler) | 32 | 32 |
| Head LR | 1e-3 | 1e-3 | 1e-3 |
| Block LR | — | 1e-4 | 5e-5 (8-9), 1e-4 (10-11) |
| Precision | FP32 | bfloat16 | bfloat16 |
| Scheduler | LinearWarmup → Cosine | CosineWarmRestarts (T₀=5) | CosineWarmRestarts (T₀=5) |

### Usage

```bash
bash setup_env.sh
python src/train.py --config configs/default.yaml --phase 0 --run-name v1
python src/evaluate.py --checkpoint checkpoints/v1_ovr_best.pt
```

### Requirements

- Python 3.10+
- PyTorch 2.x + CUDA 11.8
- SLURM cluster with GPU (tested on UMD Nexus)
