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

Think of it as **a doctor who starts by only studying one X-ray slice at a time, then gradually learns to use their full expertise on the whole scan:**

### The Backbone — `DenseNet-121`
- A well-known image recognition network, but ours starts with weights from **RadImageNet** — a version pretrained specifically on **medical imaging** (X-rays, MRIs, CT scans), not just everyday photos
- This gives it a head-start at recognising the kinds of patterns that appear in medical images
- It turns each 224×224 photo into **1,024 numbers** that describe what it sees

### How We Get a Scan-Level Decision
- We pass all the slices from one scan through the network individually
- Each slice gets a probability: "how likely is this slice from a Covid patient?"
- We **average these probabilities** across all slices to get one number per scan
- If that number is above a tuned threshold → **Covid**; otherwise → **Not Covid**

---

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
- We additionally unfreeze `denseblock3`, at an even slower rate (lr=5e-5)
- The optimizer is reinitialised fresh to give each sub-phase a clean start
- The best checkpoint across all stages is saved as `checkpoints/best.pt`

After each epoch we check how well it does on the validation patients (using scan-level F1). We save the best version automatically and stop early if there's no improvement for 10 epochs.

---

## How We Score It

- We calculate **F1 score** — a measure that penalises the model if it misses Covid patients OR cries wolf too much
- We calculate it **separately for each hospital** (so it can't just be good at one hospital and bad at others)
- **Final score = average F1 across all 4 hospitals**
- A score of `1.0` = perfect, `0.0` = completely wrong

### Extra tricks at evaluation time:
- **Threshold tuning**: instead of always saying "≥0.5 → Covid", we sweep thresholds from 0.30 to 0.70 and pick whichever gives the best F1 on the validation set
- **Test-time augmentation (TTA)**: for each scan, we process its slices 4 ways (original, flipped, rotated +15°, rotated -15°) and average the predictions — this usually adds 1–3% F1 for free
- **Independent threshold re-sweep after TTA**: TTA shifts the probability distribution toward 0.5, so the optimal threshold changes — we re-sweep after TTA separately

---

## How We Run It (The Cluster)

- We don't run this on a laptop — it would take weeks
- We use a shared **supercomputer cluster** (UMD's Nexus) with powerful GPUs
- We submit **SLURM jobs** — basically notes that say "please run this program when a GPU is free"
- Two jobs:
  1. **Extract job** (`tron` partition — stable, no preemption) — download from Google Drive + unpack all the zip/rar files onto the server
  2. **Train job** (`tron` partition, `--qos=medium`) — train the model across all 3 stages on a newer GPU

> **Why tron and not scavenger?** Scavenger jobs can be preempted (interrupted mid-run) by higher-priority users. Tron jobs run to completion. The `medium` QoS is needed to get 8 CPU workers and 64 GB RAM.

---

## What Each File Does

| File | What it does |
|------|-------------|
| `src/model.py` | Defines the "brain" — DenseNet-121 with RadImageNet weights, plus freeze/unfreeze helpers |
| `src/dataset.py` | Teaches Python how to load CT slices, apply augmentations, and ensure batches have all 4 hospitals represented equally. Also detects missing/empty scan directories at startup. |
| `src/train.py` | Runs all 3 training stages with learning rate schedules, gradient clipping, and label smoothing |
| `src/evaluate.py` | Tests the trained model: scans are evaluated by averaging their slice predictions, then threshold tuning and TTA are applied |
| `src/utils.py` | Helper tools (saving models with rotation-safe named checkpoints, computing per-hospital scores, stopping early if not improving) |
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
        ↓  Load slices, augment, centre-balanced batches
CT Slice Images (224×224 pixels)
        ↓  DenseNet-121 (RadImageNet pretrained)
Per-slice probability: P(non-covid)
        ↓  Average across all slices in a scan
Scan-level probability
        ↓  Tuned threshold (swept 0.30–0.70)
Covid / Non-Covid prediction
        ↓  Compare to ground truth, per hospital
Per-hospital F1 score → Average = Final Score (P)
```

---

## Technical Notes (Bugs Fixed During Development)

Several non-obvious issues were discovered and fixed during cluster runs:

- **Validation CSV naming**: On disk the validation metadata files are named `validation_covid.csv` (not `val_covid.csv`). The code now tries both automatically.
- **Checkpoint rotation**: PyTorch's `torch.save` for named checkpoints like `phase1_best.pt` was being deleted by the rotation logic after 3 saves. Fixed with a `save_named()` method that bypasses rotation.
- **PyTorch 2.6 checkpoint loading**: PyTorch 2.6 changed `torch.load` to default to `weights_only=True`, rejecting checkpoints with numpy scalars. Fixed with `weights_only=False`.
- **DenseNet float16 NaN**: Mixed precision training with float16 causes overflow in DenseNet's dense connections (which concatenate feature maps from many layers). Switched to bfloat16 (same dynamic range as float32) with automatic fallback to float32 on older GPUs.
- **Full-slice validation OOM**: Running full-slice evaluation (all ~200 slices per scan) with batch_size=4 during training caused out-of-memory on a 16 GB GPU. Disabled full-slice eval during training (`full_val_every_n_epochs: 999`) — it still runs at the end via `evaluate.py`. Eval batch size also reduced to 1.
