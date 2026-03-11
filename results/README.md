# Validation Results — Multi-Source COVID-19 Ensemble

This folder contains the validation output and predictions from the ensemble run (DINOv2 + DenseNet + EfficientNet), including weight tuning and per-source threshold selection.

## Contents

| File | Description |
|------|-------------|
| `ensemble_val_6364813.out` | SLURM job stdout/stderr: tuning logs, per-source F1, confusion matrices |
| `predictions_ensemble_val.csv` | Final predictions (`scan_name`, `prediction`, `prob_covid`) for 308 val scans |

---

## Challenge Metric (Official Formula)

The **Multi-Source Covid-19 Detection Challenge** evaluates models using the average of **per-source macro F1** scores:

```
P = (1/4) × Σ[i=0 to 3] ( (F1^i_covid + F1^i_noncovid) / 2 )
```

Where:
- **P** = overall challenge score
- **F1^i_covid** = F1 score for the *Covid-19* class on source *i*
- **F1^i_noncovid** = F1 score for the *non-Covid* class on source *i*
- **Macro F1 for source i** = (F1^i_covid + F1^i_noncovid) / 2

If a source has only one class (e.g. Source 2: all non-Covid), that class is excluded from the macro average for that source — the source F1 is computed only over the class(es) present.

---

## Challenge Score Calculation (This Run)

### Per-Source Macro F1

| Source | Macro F1 | Notes |
|--------|----------|-------|
| Source 0 (n=88) | **0.9317** | Both classes present |
| Source 1 (n=88) | **0.8409** | Hardest source |
| Source 2 (n=45) | **1.0000** | All non-Covid; perfect predictions |
| Source 3 (n=87) | **0.9192** | Both classes present |

### Final Challenge Score

```
P = (1/4) × (0.9317 + 0.8409 + 1.0000 + 0.9192)
  = (1/4) × 3.6918
  = 0.9229
```

**Official Challenge score: 0.9229**

---

## False Positives & False Negatives: Severity

The confusion matrix layout is: **rows = actual class**, **columns = predicted class**.

| Source | TP (Covid) | FN | FP | TN | Severity |
|--------|------------|----|----|-----|----------|
| **0** | 39 | 4 | 2 | 43 | Mild: 4 missed Covid, 2 false alarms |
| **1** | 37 | 6 | 8 | 37 | **Highest**: 6 FN, 8 FP — most errors |
| **2** | 0 | 0 | 0 | 45 | Perfect (all non-Covid) |
| **3** | 37 | 5 | 2 | 43 | Moderate: 5 missed Covid, 2 false alarms |

### Interpretation

- **Source 1** is the bottleneck: 8 false positives (healthy scans predicted as Covid) and 6 false negatives (Covid scans missed). This drives its lower F1 (0.8409).
- **Source 2** has no Covid samples in the val set, so there are no FN/FP for that class — the model correctly predicts all as non-Covid.
- **Source 0 and 3** have similar, smaller error counts; Source 0 has slightly more FN (4 vs 2 in Source 3).

### Per-Source Thresholds (After Tuning)

The ensemble used per-source thresholds for the final validation run:

| Source | Threshold |
|--------|-----------|
| 0 | 0.54 |
| 1 | 0.54 |
| 2 | 0.20 |
| 3 | 0.48 |

Source 2 uses 0.20 because all val samples are non-Covid; the low threshold mainly affects calibration, not classification. The tuned weights were **DINOv2=0.60, DenseNet=0.00, EfficientNet=0.40** (no DenseNet contribution in the best grid-search result).

---

## Model Surety (Confidence Analysis)

Using `prob_covid` from the predictions CSV:

| Predicted Class | Count | Mean prob_covid | Std | Min | Max |
|-----------------|-------|-----------------|-----|-----|-----|
| Non-Covid (0) | 125 | 0.348 | 0.079 | 0.259 | **0.540** |
| Covid (1) | 183 | 0.702 | 0.081 | **0.316** | 0.792 |

### Key Observations

1. **Confident non-Covid predictions**: Mean prob = 0.35, max 0.54 — most non-Covid predictions are clearly below the decision boundary.
2. **Confident Covid predictions**: Mean prob = 0.70 — model tends to push Covid predictions above 0.5.
3. **15.6%** of scans (48/308) have `prob_covid` in **[0.4, 0.6]** — these are the uncertain cases near the threshold.
4. Some Covid predictions have low confidence (min 0.32) — borderline cases that were still predicted as Covid.

### Calibration

The ensemble shows good separation: non-Covid predictions cluster around 0.26–0.54, Covid predictions around 0.32–0.79. The 15.6% near-boundary fraction suggests room for improvement (e.g. calibration or better feature representation) to reduce uncertainty.

---

## Run Summary

- **Node**: vulcan31.umiacs.umd.edu
- **Date**: Wed Mar 11 2026 (08:02 – 08:35 EDT)
- **Val predictions**: 308 scans → 183 Covid, 125 Non-Covid
- **Best unified threshold** (before per-source tuning): 0.54 → F1 0.9056
- **Best per-source thresholds** → F1 **0.9229**
