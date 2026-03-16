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
final_dinoV2            ← Self-supervised DINOv2 ViT-B/14 (slice-level sigmoid)
final_ConvNeXt          ← ConvNeXt-Tiny Gated Attention MIL
final_EfficientNet_V2   ← you are here
final_ensemble          ← Combines all 9 checkpoints → final submission
```

This branch produces **2 checkpoints** used by models #8 and #9 in the 9-model ensemble:
- `exp_effv2_s42/best.pt` (best regular checkpoint)
- `exp_effv2_s42/swa_best.pt` (Stochastic Weight Averaging checkpoint)

---

## This Branch: `final_EfficientNet_V2`

### Architecture

Two-phase Gated Attention MIL pipeline using **EfficientNetV2-S** (`tf_efficientnetv2_s` via `timm`) as the backbone. Same MIL pipeline as EfficientNet-B3 but with 1280-dim embeddings and an optional SWA phase.

```
Phase 1 (slice-level pretraining):
  CT Slice (256×256) → EfficientNetV2-S → Dropout(0.5) → Linear(1280, 2) → CrossEntropy

Phase 2 (scan-level end-to-end):
  24 slices per scan
    → EfficientNetV2-S (chunked, 8 slices at a time)
    → 1280-dim embeddings per slice
    → Gated Attention Pooling (tanh + sigmoid gate → softmax weights)
    → Weighted scan embedding (1280-dim)
    → MLP classifier (1280 → 512 → 2)
    → CrossEntropy + gradient accumulation + FP16

SWA phase (optional, after Phase 2):
  Load best Phase 2 checkpoint → 5 additional epochs with SWALR → average weights → update BN → save
```

### Training Details

**Data loading:**
- `SliceDataset` expands each scan into individual slices (capped at 64 per scan via uniform spacing)
- Phase 1 uses `WeightedRandomSampler` for **class-balanced** sampling
- Phase 2 uses `ScanDataset` with `shuffle=True` at scan level; 24 slices per scan

**Phase 1 — Slice-level pretraining (20 epochs):**
- Full backbone trains (ImageNet pretrained)
- AdamW, lr=1e-4, weight_decay=0.01
- Step-level linear warmup (2 epochs) → cosine decay
- BN frozen for first 2 epochs
- Gradient checkpointing
- Label smoothing ε=0.1; stochastic depth (drop_path_rate=0.3)

**Phase 2 — Scan-level MIL training (30 epochs):**
- Backbone frozen for first 3 epochs, then unfrozen with 0.1× LR factor
- AdamW with differential LR: backbone at 1e-6, attention+classifier at 1e-5
- Gradient accumulation: 16 steps (effective batch = 32)
- FP16 mixed precision
- Label smoothing ε=0.1; embedding-level mixup available
- Early stopping patience: 8 epochs

**SWA phase (5 epochs after Phase 2):**
- Loads best Phase 2 checkpoint
- `AveragedModel` + `SWALR` (swa_lr from config)
- Averages weights across 5 additional epochs
- Updates BatchNorm statistics on training data
- If SWA F1 > best Phase 2 F1, overwrites `best.pt`; always saves `swa_best.pt`

**Evaluation:**
- 48 slices per scan
- TTA (horizontal flip)
- Threshold sweep for optimal per-source F1

### Project Structure

```
covid-challenge/
├── src/
│   ├── model.py            SliceClassifier + AttentionPooling + CovidDetector (Gated Attention MIL)
│   ├── dataset.py           SliceDataset, ScanDataset, WeightedRandomSampler, scan collation
│   ├── train.py             Phase 1 + Phase 2 + optional SWA (AveragedModel, SWALR, BN update)
│   ├── evaluate.py          Scan-level inference, per-source F1, confusion matrices
│   ├── losses.py            FocalLoss implementation
│   ├── ensemble_evaluate.py Legacy ensemble evaluation
│   ├── domain_adaptation.py Experimental domain-adaptation utilities
│   └── utils.py             Seeding, config, metrics, CheckpointManager, EarlyStopping
├── scripts/
│   ├── extract_data.py        Extract competition archives
│   ├── analyze_data.py        Dataset statistics
│   └── generate_figures.py    Generate analysis figures
├── slurm/
│   ├── extract.sbatch         Data extraction
│   ├── train.sbatch           Training
│   ├── submit_ensemble.sh     Multi-config launcher
│   └── submit_ensemble_v2.sh  Updated multi-config launcher
├── configs/
│   ├── default.yaml           EfficientNetV2-S hyperparameters (this model)
│   ├── exp_effv2_s42.yaml     Seed-42 experiment config
│   ├── exp_b3_s42.yaml        EfficientNet-B3 reference config
│   ├── exp_b3_s7.yaml         EfficientNet-B3 seed-7 config
│   ├── exp_b3_s123.yaml       EfficientNet-B3 seed-123 config
│   └── exp_cnxt_s42.yaml      ConvNeXt reference config
├── tests/
│   └── test_integration.py    Integration test
└── setup_env.sh               Environment setup
```

### Key Hyperparameters

| Parameter | Phase 1 | Phase 2 | SWA |
|-----------|---------|---------|-----|
| Backbone | EfficientNetV2-S (ImageNet) | Same (from Phase 1) | Same |
| Image size | 256×256 | 256×256 | 256×256 |
| Slices/scan | 64 (slice-level) | 24 | 24 |
| Batch size | 8 | 2 (effective 32) | 2 |
| LR | 1e-4 | 1e-5 (backbone: 1e-6) | swa_lr |
| Dropout | 0.5 | 0.5 | 0.5 |
| Drop path | 0.3 | 0.3 | 0.3 |
| Label smoothing | 0.1 | 0.1 | — |
| Precision | FP32 | FP16 | FP16 |
| Backbone freeze | — | First 3 epochs | — |

### Usage

```bash
bash setup_env.sh
python src/train.py --config configs/default.yaml --phase 0
python src/evaluate.py --checkpoint checkpoints/best.pt         # regular
python src/evaluate.py --checkpoint checkpoints/swa_best.pt     # SWA variant
```

### Requirements

- Python 3.10+
- PyTorch 2.x + CUDA 11.8
- SLURM cluster with GPU (tested on UMD Nexus)
