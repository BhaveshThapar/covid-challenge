# How Our Covid-19 Detector Works — DINOv2 Approach

## What Are We Trying to Do?

- Doctors take **CT scans** of people's chests — hundreds of 2D "slice" images through the body
- Each CT scan comes from one of **4 different hospitals** (multi-source)
- We want a model to classify: **"Does this person have Covid-19 or not?"**
- The model must perform well across **all 4 hospitals**, despite different scanners and protocols

---

## The Data

- Thousands of CT scans stored as **folders of JPEG images**
- Each folder = one patient; inside are ~200–400 slice images
- Downloaded from Google Drive via `gdown`, unpacked with `unrar` / `py7zr`
- **Training data** — scans the model learns from
- **Validation data** — scans we test on to measure performance
- Metadata CSVs (`train_covid.csv`, `validation_covid.csv`, etc.) record the source hospital (0–3) for each scan

---

## How the Model Works (The "Brain")

### The Backbone — DINOv2 ViT-B/14

- **DINOv2** is a self-supervised vision model from Meta that learns visual representations **without labels**
- Unlike supervised models (ImageNet, RadImageNet), it learns from raw images via a teacher–student setup
- **ViT-B/14** = Vision Transformer, Base size, 14×14 patches
  - 86M parameters
  - Turns each 224×224 slice into a **768-dimensional vector** (CLS token)
- This backbone gives the model a strong prior for general visual structure, which often transfers well to medical images with limited labeled data

### How We Get a Scan-Level Decision

- We pass all slices from one scan through the network individually
- Each slice gets a **sigmoid probability**: "How likely is this slice from a Covid patient?"
- We **average** these probabilities across all slices in the scan
- If the average is above a tuned threshold → **Covid**; otherwise → **Non-Covid**

---

## How We Teach It (Training)

Training happens in **3 stages**, with gradually more parameters unfrozen:

### Stage 1 — Head-Only Fine-Tuning (10 epochs)

- The DINOv2 backbone is **frozen** — its weights stay fixed
- Only the classification head (Dropout + Linear) is trained
- Fast and preserves the pre-learned visual features
- Learning rate: 1e-3

### Stage 2a — Unfreeze Last 2 Transformer Blocks (15 epochs)

- We unfreeze the **last 2 transformer blocks** (blocks 10 & 11) plus the final LayerNorm
- The head keeps training at 1e-3; backbone blocks train slower (1e-4) to avoid overwriting good features
- Cosine annealing with warm restarts (T₀ = 5 epochs)

### Stage 2b — Unfreeze 2 More Blocks (15 epochs)

- We additionally unfreeze **blocks 8 & 9**
- Learning rate for these: 5e-5
- The optimizer is re-initialized between sub-phases
- Best checkpoint across all stages is saved as `{run_name}_ovr_best.pt`

After each epoch we evaluate on validation scans (scan-level F1). Best model is saved automatically; early stopping if no improvement for 10 epochs.

---

## How We Score It

- **F1 score** — balances recall (catching Covid cases) and precision (not overcalling)
- Computed **separately for each hospital**, then averaged
- **Final score P = average F1 across the 4 sources** (the challenge metric)

### Evaluation Tricks

- **Threshold tuning**: sweep thresholds 0.30–0.70 on validation and pick the best F1
- **Test-time augmentation (TTA)**: process each scan 4 ways (identity, flip, rotate ±15°) and average slice predictions — often improves F1 by 1–3%
- **Separate threshold sweep after TTA**: TTA changes the probability distribution, so we re-sweep for the best threshold on TTA outputs

---

## How We Run It (The Cluster)

- Training runs on UMD's **Nexus HPC cluster** with GPUs
- **SLURM jobs** are submitted to run training and evaluation
- Two main jobs:
  1. **Extract** — download from Google Drive, unpack archives onto the server (e.g. scratch)
  2. **Train** — run both training phases and evaluation

- **Scratch** (`/fs/nexus-scratch/`): fast temporary storage for data and checkpoints; copy important results back to home or project storage
- **Home directory**: persistent storage for code and final outputs

---

## What Each File Does

| File | What it does |
|------|--------------|
| `src/model.py` | DINOv2 ViT-B/14 backbone + classifier head; freeze/unfreeze helpers for gradual training |
| `src/dataset.py` | Loads CT slices, applies augmentations, center-balanced batches, scan manifests |
| `src/train.py` | Runs all 3 training stages with schedulers, gradient clipping, label smoothing |
| `src/evaluate.py` | Scan-level inference, threshold tuning, TTA, per-source F1 and confusion matrices |
| `src/utils.py` | Checkpoint saving/loading, per-source F1, early stopping |
| `scripts/download_and_extract.py` | Downloads data from Google Drive, organizes folders, handles rar/zip |
| `slurm/train.sbatch` | SLURM script for training |
| `slurm/extract.sbatch` | SLURM script for download and extraction |
| `configs/default.yaml` | All hyperparameters (learning rates, epochs, image size, etc.) |

---

## End-to-End Flow

```
Google Drive archives
        ↓  gdown + unpack (extract.sbatch)
Organised folders (data/train/, data/val/)
        ↓  Load slices, augment, centre-balanced batches
CT Slice Images (224×224)
        ↓  DINOv2 ViT-B/14 (self-supervised backbone)
Per-slice probability: P(non-covid)
        ↓  Average across slices in scan
Scan-level probability
        ↓  Tuned threshold (sweep 0.30–0.70)
Covid / Non-Covid prediction
        ↓  Per-source F1
Final Score (P) = average F1 across 4 hospitals
```

---

## Technical Notes

- **Preprocessing**: DINOv2 expects ImageNet-style normalization (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]). Same as many medical imaging pipelines.
- **Mixed precision**: Phase 2 uses bfloat16 to reduce memory; float32 fallback on older GPUs.
- **First run**: DINOv2 is loaded via `torch.hub`; the first run will download the weights (~330 MB) from the internet.
