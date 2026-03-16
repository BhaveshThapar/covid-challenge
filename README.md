# Multi-Source Covid-19 Detection Challenge — Ensemble

Binary COVID / non-COVID classification of chest CT scans across 4 hospital sources,
for the **PHAROS-AIF-MIH** competition at **CVPR 2026**.

**Metric:** Average macro F1 across the 4 data centres.

---

## Repository Branch Structure

Each `final_*` branch trains a single backbone; this branch combines them all.

```
final_EfficientNet_B3   ← EfficientNet-B3 Gated Attention MIL (4 seed/SWA variants)
final_DenseNet-121      ← RadImageNet-pretrained DenseNet-121 (slice-level sigmoid)
final_dinoV2            ← Self-supervised DINOv2 ViT-B/14 (slice-level sigmoid)
final_ConvNeXt          ← ConvNeXt-Tiny Gated Attention MIL
final_EfficientNet_V2   ← EfficientNetV2-S Gated Attention MIL
final_ensemble          ← you are here — 9-model ensemble → final submission
```

---

## This Branch: `final_ensemble`

This branch does **no training**. It loads 9 pre-trained checkpoints from the other `final_*` branches, runs inference on validation or test data, fuses probabilities, tunes thresholds, and produces the final challenge submission.

### The 9 Models

Defined in `configs/ensemble.yaml`:

| # | Name in config | Backbone | Inference paradigm | Checkpoint | Source branch |
|---|----------------|----------|--------------------|------------|---------------|
| 1 | `dinov2` | DINOv2 ViT-B/14 | Slice-level sigmoid avg + 4-view TTA | `checkpoints/v1_ovr_best.pt` | `final_dinoV2` |
| 2 | `densenet` | DenseNet-121 (RadImageNet) | Slice-level sigmoid avg + 4-view TTA | `checkpoints/v4_ovr_best.pt` | `final_DenseNet-121` |
| 3 | `eff_b3_s42` | EfficientNet-B3 (seed 42) | Gated Attention MIL softmax + 4-flip TTA | `checkpoints/exp_b3_s42/best.pt` | `final_EfficientNet_B3` |
| 4 | `eff_b3_s7` | EfficientNet-B3 (seed 7) | Gated Attention MIL softmax + 4-flip TTA | `checkpoints/exp_b3_s7/best.pt` | `final_EfficientNet_B3` |
| 5 | `eff_b3_s7_swa` | EfficientNet-B3 (seed 7, SWA) | Gated Attention MIL softmax + 4-flip TTA | `checkpoints/exp_b3_s7/swa_best.pt` | `final_EfficientNet_B3` |
| 6 | `eff_b3_s123` | EfficientNet-B3 (seed 123) | Gated Attention MIL softmax + 4-flip TTA | `checkpoints/exp_b3_s123/best.pt` | `final_EfficientNet_B3` |
| 7 | `cnxt_s42` | ConvNeXt-Tiny (seed 42) | Gated Attention MIL softmax + 4-flip TTA | `checkpoints/exp_cnxt_s42/best.pt` | `final_ConvNeXt` |
| 8 | `effv2_s42` | EfficientNetV2-S (seed 42) | Gated Attention MIL softmax + 4-flip TTA | `checkpoints/exp_effv2_s42/best.pt` | `final_EfficientNet_V2` |
| 9 | `effv2_s42_swa` | EfficientNetV2-S (seed 42, SWA) | Gated Attention MIL softmax + 4-flip TTA | `checkpoints/exp_effv2_s42/swa_best.pt` | `final_EfficientNet_V2` |

### Ensemble Pipeline

```
1. Launch 9 model workers (multiprocessing, round-robin GPU assignment)
2. Each model runs inference on all scans with model-specific TTA:
   - DINOv2/DenseNet: slice-level sigmoid → mean → [P(covid), P(non-covid)]
   - MIL models: scan-level softmax → [P(covid), P(non-covid)] via 4-flip TTA
3. Collect all 9 probability arrays → shape (9, N_scans, 2)
4. Fuse via probability averaging:
   - Validation: score-weighted averaging (weight = model's individual val F1)
   - Test: uniform averaging
5. Decision rule selection (validation only):
   - Compare argmax, global threshold sweep, per-source threshold sweep
   - Pick whichever gives best validation F1
6. Output predictions CSV
```

### Per-source Threshold Tuning

On the validation split (where labels are available), the ensemble sweeps a separate threshold per data source:
- Range: [0.20, 0.80] in steps of 0.005
- For each source, selects the threshold maximizing that source's macro F1
- This accounts for systematic probability distribution differences across hospital centres

### Validation Results

Reported in `results/README.md`:

| Source | Macro F1 |
|--------|----------|
| Source 0 | 0.9317 |
| Source 1 | 0.8409 |
| Source 2 | 1.0000 |
| Source 3 | 0.9192 |
| **Average (P)** | **0.9229** |

Per-source thresholds: Source 0=0.54, Source 1=0.54, Source 2=0.20, Source 3=0.48.

### Project Structure

```
covid-challenge/
├── src/
│   ├── ensemble.py        Main ensemble driver: multi-model inference, fusion, threshold tuning
│   ├── models/
│   │   ├── __init__.py    Exports all model classes
│   │   ├── efficientnet.py SliceClassifier + AttentionPooling + CovidDetector
│   │   ├── dinov2.py       DINOv2CovidClassifier
│   │   └── densenet.py     DenseNetCovidClassifier
│   ├── model.py           Legacy single-model definitions (backward compat)
│   ├── dataset.py          SliceDataset, ScanDataset, CenterBatchSampler, RawSliceScanDataset
│   ├── evaluate.py         Single-model evaluation with threshold sweep and TTA
│   ├── predict_test.py     Single-model test-set inference
│   └── utils.py            Seeding, config, metrics, CheckpointManager
├── configs/
│   ├── ensemble.yaml       Defines all 9 models, their configs, and checkpoints
│   ├── dinov2.yaml         DINOv2 model + eval settings (224px, tta_n=4)
│   ├── densenet.yaml       DenseNet model + eval settings (224px, tta_n=4)
│   ├── efficientnet.yaml   EfficientNet-B3 model + eval settings (256px, 48 slices)
│   ├── convnext.yaml       ConvNeXt-Tiny model + eval settings (256px, 48 slices)
│   ├── efficientnetv2.yaml EfficientNetV2-S model + eval settings (256px, 48 slices)
│   └── default.yaml        Alias to dinov2.yaml
├── results/
│   ├── README.md            Detailed validation results, confusion matrices, confidence analysis
│   ├── predictions_ensemble_avg_val.csv    Validation predictions (probability averaging)
│   ├── predictions_ensemble_majority_val.csv Validation predictions (majority vote)
│   ├── predictions_ensemble_test.csv       Test-set predictions
│   ├── covid.csv            Challenge submission: COVID scan names
│   ├── non_covid.csv        Challenge submission: non-COVID scan names
│   └── analyze_val.py       Script to analyze validation predictions
├── scripts/
│   ├── download_and_extract.py    Download + extract competition data
│   └── setup_checkpoints_cluster.sh Copy checkpoints from model branches
├── slurm/
│   ├── ensemble_val.sbatch   Validation: probability averaging + majority vote comparison
│   ├── ensemble_test.sbatch  Test: generate final submission predictions
│   ├── ensemble.sbatch       Multi-GPU ensemble (3 GPUs)
│   ├── eval.sbatch           Single-model evaluation
│   ├── extract.sbatch        Data extraction
│   ├── test.sbatch           Single-model test inference
│   └── diag.sbatch           Diagnostic checks
├── test_single_model.py      Quick single-model diagnostic script
└── setup_env.sh              Environment setup
```

### File Details

| File | Purpose |
|------|---------|
| `src/ensemble.py` | `_run_efficientnet_variant()` runs a single MIL model with 4-flip TTA. `_run_slice_model()` runs DINOv2 or DenseNet with configurable TTA. Workers launch via `multiprocessing.Process` and report results via `Queue`. `sweep_threshold()` and `sweep_threshold_per_source()` tune decision boundaries. Main loop fuses all 9 models' probabilities and selects the best decision rule. |
| `configs/ensemble.yaml` | Lists all 9 models with their type, per-model config path, checkpoint path, and optional seed/backbone overrides. |
| `results/README.md` | Full breakdown of validation results: per-source F1, confusion matrices, per-source thresholds, confidence analysis, false positive/negative severity. |
| `results/analyze_val.py` | Standalone script to re-analyze saved prediction CSVs: loads predictions, aligns with ground-truth manifest, computes per-source F1. |

### Usage

```bash
# Validation (probability averaging + per-source threshold tuning)
python src/ensemble.py \
    --config configs/ensemble.yaml \
    --split val \
    --tune-threshold \
    --strategy avg \
    --gpus 0

# Test set (final submission)
python src/ensemble.py \
    --config configs/ensemble.yaml \
    --split test \
    --strategy avg \
    --weighting uniform \
    --gpus 0

# Via SLURM
sbatch slurm/ensemble_val.sbatch
sbatch slurm/ensemble_test.sbatch
```

### Requirements

- Python 3.10+
- PyTorch 2.x + CUDA 11.8
- All 9 model checkpoints must be present in `checkpoints/` (see table above)
- SLURM cluster with GPU (tested on UMD Nexus, RTX A6000)
