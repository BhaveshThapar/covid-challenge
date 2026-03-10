# How Our Covid-19 Detector Works — Simple Explainer

## What Are We Trying to Do?

- Doctors take **CT scans** of people's chests (like an X-ray but way more detailed — it takes hundreds of pictures of slices through your body like slices of bread)
- Each CT scan comes from one of **4 different hospitals**
- We want a computer to look at these scans and answer one question: **"Does this person have Covid-19 or not?"**
- The computer needs to get this right for patients from **all 4 hospitals**, even if each hospital's scanner looks a little different
- Our final system achieves **0.9279 F1 score** (out of 1.0) averaged across all 4 hospitals

---

## The Data

- We have thousands of CT scans, each stored as a **folder of JPEG images** (like regular photos)
- Each folder = one patient's scan, and inside are ~50–700 photos (one per "slice" of the chest)
- The data came compressed in `.rar` and `.zip` files (like a zip file you'd get from downloading something)
- Split into:
  - **Training data** — 1,222 scans the computer learns from (564 Covid + 659 non-Covid, ~26 GB)
  - **Validation data** — 308 scans we test on to see how well it learned

---

## How the Model Works (The "Brain")

Think of it like **two students working together:**

### Student 1 — The Slice Inspector (`EfficientNet-B3`)
- Looks at **one photo at a time** from a CT scan
- Has already studied millions of regular photos (cats, dogs, cars...) so it already knows how to see shapes and patterns
- We then teach it to also recognize Covid patterns in lung slices
- It turns each photo into a list of **1,536 numbers** that describe what it sees

### Student 2 — The Attention Reader (`Attention Pooling`)
- Gets the descriptions from Student 1 for **24 sampled slices** of a scan (48 at test time)
- Figures out **which slices matter most** (e.g. the middle of the lung is more important than the very top or bottom)
- Combines everything into one final answer: **Covid or Not Covid**

### The Ensemble — 5 Models Vote Together
- We don't rely on a single model — we train **5 different models** with different backbones and settings
- At test time, all 5 models vote, weighted by how well each one performed individually
- This makes the final answer much more reliable

---

## How We Teach It (Training)

Teaching happens in **two rounds:**

### Round 1 — Practice on single photos (20 passes through all data)
- Show the model one slice at a time (resized to **256x256 pixels**)
- Tell it "this is from a Covid patient" or "this is not"
- It slowly gets better at spotting Covid patterns

### Round 2 — Practice on full scans (30 passes through all data)
- Now show it 24 slices from a scan at once
- It has to make one decision for the whole patient
- The "attention" part learns to focus on the most important slices
- Uses **Focal Loss** to focus on hard cases, and **Mixup** to prevent memorization

After each pass, we check how well it does on the test patients. We save the best version automatically.

---

## How We Score It

- We calculate **F1 score** — a measure that penalizes the model if it misses Covid patients OR cries wolf too much
- We calculate it **separately for each hospital** (so it can't just be good at one hospital and bad at others)
- Each hospital gets its own **optimized threshold** for deciding Covid vs. non-Covid
- **Final score = average F1 across all 4 hospitals = 0.9279**
- A score of `1.0` = perfect, `0.0` = completely wrong

---

## How We Run It (The Cluster)

- We don't run this on a laptop — it would take weeks
- We use a shared **GPU cluster** with powerful NVIDIA GPUs (RTX A6000 with 48 GB memory)
- We submit **SLURM jobs** — basically notes that say "please run this program when a GPU is free"
- Jobs:
  1. **Extract job** — unpack all the zip/rar files onto the server (~4-6 hours, no GPU needed)
  2. **Train job** — train one model (~6-12 hours on a single GPU)
  3. **Ensemble job** — train all 5 models (~50 GPU-hours total)

---

## What Each File Does

| File | What it does |
|------|-------------|
| `src/model.py` | Defines the "brain" — EfficientNet + Attention |
| `src/dataset.py` | Teaches Python how to read and load the CT scan images |
| `src/train.py` | Runs the two training rounds |
| `src/evaluate.py` | Tests a single trained model and prints the score for each hospital |
| `src/ensemble_evaluate.py` | Combines all 5 models and produces the final ensemble score |
| `src/losses.py` | Defines Focal Loss (focuses training on hard cases) |
| `src/utils.py` | Helper tools (saving models, computing scores, stopping early if not improving) |
| `scripts/extract_data.py` | Unpacks the zip/rar files and organizes them into folders |
| `scripts/generate_figures.py` | Creates figures for the research paper from CT scan data |
| `slurm/train.sbatch` | The "note" we hand to the cluster to train a model |
| `slurm/extract.sbatch` | The "note" to unpack the data |
| `configs/default.yaml` | Default settings (learning speed, image size, etc.) |
| `configs/exp_*.yaml` | Per-model settings for each of the 5 ensemble models |

---

## End-to-End Flow

```
Compressed archives (.rar/.zip)
        |  unpack
Organized folders (data/train/, data/val/)
        |  load & augment (12 transforms)
CT Slice Images (256x256 pixels)
        |  EfficientNet-B3 (x5 models)
Feature vectors (1536 numbers per slice)
        |  Gated Attention Pooling (24 slices -> 1 summary)
Scan embeddings
        |  Score-weighted ensemble voting
Combined prediction
        |  Per-source threshold calibration
Covid / Non-Covid prediction
        |  Compare to ground truth
Per-hospital F1 score -> Average = 0.9279
```
