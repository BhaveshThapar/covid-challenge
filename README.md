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
final_ConvNeXt          ← you are here
final_EfficientNet_V2   ← EfficientNetV2-S Gated Attention MIL
final_ensemble          ← Combines all 9 checkpoints → final submission
```

This branch produces **checkpoint `exp_cnxt_s42/best.pt`** used by model #7 in the 9-model ensemble.

---

## This Branch: `final_ConvNeXt`

### Architecture

Two-phase Gated Attention MIL pipeline using **ConvNeXt-Tiny** (via `timm`) as the backbone. The model architecture is identical to the EfficientNet-B3 pipeline but with a different backbone and 768-dim embeddings instead of 1536.

```
Phase 1 (slice-level pretraining):
  CT Slice (256×256) → ConvNeXt-Tiny → Dropout(0.5) → Linear(768, 2) → CrossEntropy

Phase 2 (scan-level end-to-end):
  24 slices per scan
    → ConvNeXt-Tiny (chunked, 8 slices at a time)
    → 768-dim embeddings per slice
    → Gated Attention Pooling (tanh + sigmoid gate → softmax weights)
    → Weighted scan embedding (768-dim)
    → MLP classifier (768 → 512 → 2)
    → CrossEntropy + gradient accumulation + FP16
```

### Training Details

**Data loading:**
- `SliceDataset` expands each scan into individual slices (capped at 64 per scan via uniform spacing)
- Phase 1 uses `WeightedRandomSampler` for **class-balanced** sampling (inverse class frequency weights)
- Phase 2 uses `ScanDataset` with `shuffle=True` at the scan level; each scan contributes 24 slices

**Phase 1 — Slice-level pretraining (20 epochs):**
- Full backbone trains (ImageNet pretrained)
- AdamW, lr=1e-4, weight_decay=0.01
- Step-level linear warmup (2 epochs) → cosine decay
- BN frozen for first 2 epochs
- Gradient checkpointing enabled
- Label smoothing ε=0.1; CrossEntropyLoss
- Stochastic depth (drop_path_rate=0.3)

**Phase 2 — Scan-level MIL training (30 epochs):**
- Backbone frozen for first 3 epochs, then unfrozen with 0.1× LR factor
- AdamW with differential LR: backbone at 1e-6, attention+classifier at 1e-5
- Step-level warmup (3 epochs) → cosine decay
- Gradient accumulation: 16 steps (effective batch = 32)
- FP16 mixed precision
- Label smoothing ε=0.1; embedding-level mixup available (alpha=0.0 by default)
- Early stopping patience: 8 epochs

**Evaluation:**
- 48 slices per scan at eval
- TTA (horizontal flip)
- Threshold sweep for optimal per-source F1

### Project Structure

```
covid-challenge/
├── src/
│   ├── model.py            SliceClassifier + AttentionPooling + CovidDetector (Gated Attention MIL)
│   ├── dataset.py           SliceDataset, ScanDataset, WeightedRandomSampler, scan collation
│   ├── train.py             Phase 1 (slice) + Phase 2 (scan MIL, backbone freeze/unfreeze, mixup)
│   ├── evaluate.py          Scan-level inference, per-source F1, confusion matrices
│   ├── losses.py            FocalLoss implementation
│   ├── ensemble_evaluate.py Legacy ensemble evaluation
│   ├── domain_adaptation.py Experimental domain-adaptation utilities
│   └── utils.py             Seeding, config, metrics, CheckpointManager, EarlyStopping
├── scripts/
│   ├── extract_data.py      Extract competition archives
│   └── analyze_data.py      Dataset statistics
├── slurm/
│   ├── extract.sbatch       Data extraction
│   ├── train.sbatch         Training
│   └── submit_ensemble.sh   Multi-config training launcher
├── configs/
│   ├── default.yaml         ConvNeXt-Tiny hyperparameters (this model)
│   ├── exp_cnxt_s42.yaml    Seed-42 experiment config
│   ├── exp_b3_s42.yaml      EfficientNet-B3 reference config
│   └── exp_b3_s123.yaml     EfficientNet-B3 seed-123 config
├── tests/
│   └── test_integration.py  Integration test
└── setup_env.sh             Environment setup
```

### Key Hyperparameters

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Backbone | ConvNeXt-Tiny (ImageNet) | Same (from Phase 1) |
| Image size | 256×256 | 256×256 |
| Slices/scan | 64 (slice-level) | 24 |
| Batch size | 8 | 2 (effective 32 via grad accum 16) |
| LR | 1e-4 | 1e-5 (backbone: 1e-6) |
| Dropout | 0.5 | 0.5 |
| Drop path | 0.3 | 0.3 |
| Label smoothing | 0.1 | 0.1 |
| Loss | CrossEntropy | CrossEntropy |
| Precision | FP32 | FP16 |
| Backbone freeze | — | First 3 epochs |

### Usage

```bash
bash setup_env.sh
python src/train.py --config configs/default.yaml --phase 0
python src/evaluate.py --checkpoint checkpoints/best.pt
```

### Requirements

- Python 3.10+
- PyTorch 2.x + CUDA 11.8
- SLURM cluster with GPU (tested on UMD Nexus)
