# How Our Covid-19 Detector Works — Simple Explainer

## What Are We Trying to Do?

- Doctors take **CT scans** of people's chests (like an X-ray but way more detailed — it takes hundreds of pictures of slices through your body like slices of bread)
- Each CT scan comes from one of **4 different hospitals**
- We want a computer to look at these scans and answer one question: **"Does this person have Covid-19 or not?"**
- The computer needs to get this right for patients from **all 4 hospitals**, even if each hospital's scanner looks a little different

---

## The Data

- We have thousands of CT scans, each stored as a **folder of JPEG images** (like regular photos)
- Each folder = one patient's scan, and inside are ~375 photos (one per "slice" of the chest)
- The data is stored on Google Drive and downloaded automatically using `gdown`
- Split into:
  - **Training data** — the scans the computer learns from (~1,200 patients)
  - **Validation data** — scans we test on to see how well it learned (308 patients)
- Metadata CSVs (`train_covid.csv`, `validation_covid.csv`, etc.) tell us which hospital (0–3) each scan came from

---

## How the Model Works (The "Brain")

Think of it as **a doctor who studies a whole patient chart at once, paying more attention to the most suspicious pages:**

### Step 1 — Preprocessing: Find the Lungs First
- Before the model sees anything, each CT slice goes through a **lung ROI crop**: we use OpenCV to threshold the image, find the two biggest bright regions (the two lungs), draw a bounding box around them with some padding, and crop to that box
- This removes irrelevant background (bed, bones, air) and gives the model a tighter view of the actual lung tissue
- The crop is resized to 224×224 for the model

### Step 2 — The Backbone — `DenseNet-121`
- A well-known image recognition network pretrained on **RadImageNet** — a dataset of medical images (X-rays, MRIs, CT scans), not everyday photos
- This gives it a head-start at recognising medical patterns
- It turns each 224×224 slice into **1,024 numbers** (a feature vector)
- During training, **MixStyle** is applied after the first two dense blocks: it randomly mixes the feature statistics (mean and variance) between scans from different hospitals, forcing the network to learn features that generalise across hospital scanners

### Step 3 — Attention MIL: Deciding Which Slices Matter
- All 64 slices from one scan are processed together as a **bag**
- A small attention network (2 layers) assigns each slice a weight: "how informative is this slice for the diagnosis?"
- The 1,024-d feature vectors are averaged using those weights → one **1,024-d scan embedding** that emphasises the most suspicious slices
- A final classifier layer turns that into one Covid/Non-Covid logit per scan

### Step 4 — Per-Center Threshold
- After training, the threshold (the cutoff between Covid and Not Covid) is tuned **independently for each of the 4 hospitals** by sweeping 0.30–0.70 in 0.01 steps
- This accounts for differences in how each hospital's scanner affects the probability distribution

---

## How Data Is Sampled

**During training (scan-level MIL):** Each dataset item is a whole scan. 64 slices are sampled uniformly from the scan, preprocessed with the ROI crop, and stacked into a bag of shape `(64, 3, 224, 224)`. Batches of 8 scans are assembled using `CenterBatchSampler` with **center-and-class balance**: each batch contains exactly 2 scans per hospital — 1 COVID + 1 Non-COVID — giving equal center and class representation in every gradient step. If a center×class bucket (e.g. source_2 COVID) runs out, it is resampled with replacement; the epoch ends when the largest bucket is exhausted. A boolean mask is produced so the model's attention layer can ignore any padding positions.

**During validation (scan-level MIL):** 48 slices per scan, same ROI crop and MIL forward — the model directly outputs one scan-level logit via attention pooling. No post-hoc averaging is needed.

## How We Teach It (Training)

Teaching happens in **3 stages** — think of it like gradually handing a student more freedom:

### Stage 1 — Head-Only Fine-Tuning (10 epochs)
- The DenseNet backbone is completely **frozen** — its weights don't change
- Only the tiny classification head (2 layers) is trained
- This is fast and avoids breaking the useful medical features already learned from RadImageNet
- Learning rate: 1e-3

### Stage 2a — Unfreeze the Top Block (15 epochs)
- We unfreeze `denseblock4` (the last dense block) and let those layers adapt
- The head keeps training at the same speed; denseblock4 trains slower (lr=1e-4) so we don't overwrite too fast
- The scheduler restarts every 5 epochs (cosine annealing with warm restarts)

### Stage 2b — Unfreeze One Block Deeper (15 epochs)
- We additionally unfreeze `denseblock3`, at an even slower rate (lr=3e-5)
- The optimizer is reinitialised fresh from the best Stage 2a checkpoint
- The best checkpoint across all stages is saved as `checkpoints/v3_ovr_best.pt`
- If a job is killed between Stage 2a and 2b, `--phase 3` resumes from the saved Stage 2a checkpoint

After each epoch we evaluate on the validation scans (scan-level MIL forward, 48 slices per scan) and compute the **weighted F1** = (F1₀ + F1₁ + 0.2·F1₂ + F1₃) / 3.2 — this down-weights Centre 2 which is smallest and noisiest. Checkpoints are saved when weighted F1 improves; training stops early after 10 epochs without improvement.

---

## How We Score It

- We calculate **F1 score** — a measure that penalises the model if it misses Covid patients OR cries wolf too much
- We calculate it **separately for each hospital** (so it can't just be good at one hospital and bad at others)
- **Final score = average F1 across all 4 hospitals**
- A score of `1.0` = perfect, `0.0` = completely wrong

### Extra tricks at evaluation time:
- **Per-center threshold tuning**: instead of one global cutoff, we sweep 0.30–0.70 independently for each of the 4 hospitals and pick the best threshold per hospital
- **Intensity TTA**: for each scan, we run the MIL forward 4 times with different intensity transformations (original, gamma=0.9, gamma=1.1, CLAHE contrast enhancement) and average the 4 probabilities — no geometric augmentation is used, avoiding artificial reorientation of lung anatomy
- **Independent threshold re-sweep after TTA**: TTA shifts probabilities, so the optimal per-center thresholds are re-tuned on the TTA probabilities separately
- **Dual F1 reporting**: the strict challenge formula always computes F1 over both classes 0 and 1. The sklearn "legacy" score (only classes present in the data) is also shown for comparison.

---

## How We Run It (The Cluster)

- We don't run this on a laptop — it would take weeks
- We use a shared **supercomputer cluster** (UMD's Nexus) with powerful GPUs
- We submit **SLURM jobs** — basically notes that say "please run this program when a GPU is free"
- Two jobs:
  1. **Extract job** (`tron` partition — stable, no preemption) — download from Google Drive + unpack all the zip/rar files onto the server
  2. **Train job** (`tron` partition, `--qos=hi`, RTX A6000) — train the model across all 3 phases

---

## What Each File Does

| File | What it does |
|------|-------------|
| `src/model.py` | `DenseNetMILClassifier`: DenseNet-121 backbone + MixStyle + ABMIL attention head. `DenseNetCovidClassifier` kept for backward compatibility with v1/v2 checkpoints. |
| `src/dataset.py` | Lung ROI crop (`_load_image_with_roi`), `ScanDataset` for MIL bag loading, intensity-only augmentations and TTA, `CenterBatchSampler` (scan-level), `build_scan_train_dataloader`. |
| `src/train.py` | Scan-level MIL training: per-sample BCE with asymmetric center weights, weighted F1 checkpoint selection, all 3 phases. |
| `src/evaluate.py` | MIL inference, intensity TTA, `tune_thresholds_per_center` (4 independent thresholds), `print_results` with weighted F1 display. |
| `src/utils.py` | `compute_weighted_f1`, rotation-safe checkpointing, per-hospital F1, early stopping. |
| `scripts/download_and_extract.py` | Downloads data from Google Drive and organises it into the right folder structure |
| `slurm/train.sbatch` | The "note" we hand to the supercomputer to train the model |
| `slurm/extract.sbatch` | The "note" to download and unpack the data |
| `configs/default.yaml` | All the settings (learning speed, image size, threshold sweep range, etc.) in one place |

---

## End-to-End Flow

```
Google Drive archives
        ↓  gdown + unpack (extract.sbatch)
Organised folders (data/train/, data/val/)
        ↓  Build scan manifest from metadata CSVs
        ↓  ScanDataset: sample 64 slices per scan
        ↓  ROI crop each slice (Otsu + connected components → lung bounding box → 224×224)
        ↓  Intensity augmentations (train) / normalise (val)
        ↓  CenterBatchSampler: 8 scans per batch, all 4 hospitals represented
        ↓  DenseNet-121 (RadImageNet, +MixStyle) → 1024-d per-slice embedding
        ↓  ABMIL attention → weighted sum → 1024-d scan embedding
        ↓  Linear(1024→1) → scan-level logit
        ↓  Sigmoid + per-center threshold (tuned 0.30–0.70)
Covid / Non-Covid prediction
        ↓  Compare to ground truth, per hospital
Per-hospital F1 score → Plain average = Final Challenge Score (P)
                      → Weighted average = Checkpoint selection metric
```

---

## Technical Notes (Bugs Fixed During Development)

Several non-obvious issues were discovered and fixed during cluster runs:

- **Validation CSV naming**: On disk the validation metadata files are named `validation_covid.csv` (not `val_covid.csv`). The code now tries both automatically.
- **Checkpoint rotation**: PyTorch's `torch.save` for named checkpoints like `phase1_best.pt` was being deleted by the rotation logic after 3 saves. Fixed with a `save_named()` method that bypasses rotation.
- **PyTorch 2.6 checkpoint loading**: PyTorch 2.6 changed `torch.load` to default to `weights_only=True`, rejecting checkpoints with numpy scalars. Fixed with `weights_only=False`.
- **DenseNet float16 NaN**: Mixed precision training with float16 causes overflow in DenseNet's dense connections (which concatenate feature maps from many layers). Switched to bfloat16 (same dynamic range as float32) with automatic fallback to float32 on older GPUs.
- **Full-slice validation OOM**: Running full-slice evaluation (all ~200 slices per scan) with batch_size=4 during training caused out-of-memory on a 16 GB GPU. Disabled full-slice eval during training (`full_val_every_n_epochs: 999`) — it still runs at the end via `evaluate.py`. Eval batch size also reduced to 1.
