# How Our Covid-19 Detector Works — Simple Explainer

## 🏥 What Are We Trying to Do?

- Doctors take **CT scans** of people's chests (like an X-ray but way more detailed — it takes hundreds of pictures of slices through your body like slices of bread)
- Each CT scan comes from one of **4 different hospitals**
- We want a computer to look at these scans and answer one question: **"Does this person have Covid-19 or not?"**
- The computer needs to get this right for patients from **all 4 hospitals**, even if each hospital's scanner looks a little different

---

## 📦 The Data

- We have thousands of CT scans, each stored as a **folder of JPEG images** (like regular photos)
- Each folder = one patient's scan, and inside are ~375 photos (one per "slice" of the chest)
- The data came compressed in `.rar` and `.zip` files (like a zip file you'd get from downloading something)
- Split into:
  - **Training data** — the scans the computer learns from (~26 GB)
  - **Validation data** — scans we test on to see how well it learned (308 patients)

---

## 🧠 How the Model Works (The "Brain")

Think of it like **two students working together:**

### Student 1 — The Slice Inspector (`EfficientNet-B3`)
- Looks at **one photo at a time** from a CT scan
- Has already studied millions of regular photos (cats, dogs, cars...) so it already knows how to see shapes and patterns
- We then teach it to also recognize Covid patterns in lung slices
- It turns each photo into a list of **1,536 numbers** that describe what it sees

### Student 2 — The Attention Reader (`Attention Pooling`)
- Gets the descriptions from Student 1 for all 32 sampled slices of a scan
- Figures out **which slices matter most** (e.g. the middle of the lung is more important than the very top or bottom)
- Combines everything into one final answer: **Covid or Not Covid**

---

## 🏋️ How We Teach It (Training)

Teaching happens in **two rounds:**

### Round 1 — Practice on single photos (5 rounds through all data)
- Show the model one slice at a time
- Tell it "this is from a Covid patient" or "this is not"
- It slowly gets better at spotting Covid patterns

### Round 2 — Practice on full scans (15 rounds through all data)
- Now show it 32 slices from a scan at once
- It has to make one decision for the whole patient
- The "attention" part learns to focus on the most important slices

After each round, we check how well it does on the test patients. We save the best version automatically.

---

## 📊 How We Score It

- We calculate **F1 score** — a measure that penalizes the model if it misses Covid patients OR cries wolf too much
- We calculate it **separately for each hospital** (so it can't just be good at one hospital and bad at others)
- **Final score = average F1 across all 4 hospitals**
- A score of `1.0` = perfect, `0.0` = completely wrong

---

## 🖥️ How We Run It (The Cluster)

- We don't run this on a laptop — it would take weeks
- We use a shared **supercomputer cluster** (UMD's Nexus) with powerful GPUs
- We submit **SLURM jobs** — basically notes that say "please run this program when a GPU is free"
- Two jobs:
  1. **Extract job** — unpack all the zip/rar files onto the server (~4–6 hours, no GPU needed)
  2. **Train job** — actually train the model (~6–12 hours on a GPU)

---

## 📁 What Each File Does

| File | What it does |
|------|-------------|
| `src/model.py` | Defines the "brain" — EfficientNet + Attention |
| `src/dataset.py` | Teaches Python how to read and load the CT scan images |
| `src/train.py` | Runs the two training rounds |
| `src/evaluate.py` | Tests the trained model and prints the score for each hospital |
| `src/utils.py` | Helper tools (saving models, computing scores, stopping early if not improving) |
| `scripts/extract_data.py` | Unpacks the zip/rar files and organizes them into folders |
| `slurm/train.sbatch` | The "note" we hand to the supercomputer to train the model |
| `slurm/extract.sbatch` | The "note" to unpack the data |
| `configs/default.yaml` | All the settings (like learning speed, image size, etc.) in one place |

---

## 🔁 End-to-End Flow

```
Compressed archives (.rar/.zip)
        ↓  unpack
Organized folders (data/train/, data/val/)
        ↓  load & augment
CT Slice Images (224×224 pixels)
        ↓  EfficientNet-B3
Feature vectors (1536 numbers per slice)
        ↓  Attention Pooling (32 slices → 1 summary)
Scan embedding
        ↓  Classifier
Covid / Non-Covid prediction
        ↓  Compare to ground truth
Per-hospital F1 score → Average = Final Score
```
