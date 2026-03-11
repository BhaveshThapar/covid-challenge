# DINOv2 Model — Results & Analysis

Validation and test results for Anant's DINOv2 ViT-B/14 COVID-19 detector.

## Files

| File | Description |
|------|-------------|
| `validation_results_dinov2.csv` | Per-scan validation predictions (308 scans, TTA, threshold 0.46) |
| `test_results_dinov2.csv` | Per-scan test predictions (1487 scans, threshold 0.5) |

## Model & Metrics

- **Architecture**: DINOv2 ViT-B/14 + classifier head  
- **Checkpoint**: `checkpoints/v1_ovr_best.pt`  
- **Challenge Score (val, per-source macro F1 avg)**: **0.9113** (no TTA) | 0.9053 (TTA)  
- **Test**: 1487 scans → 1088 predicted COVID, 399 non-COVID  

---

## 1. Validation Error Severity Analysis

### Overview

| Error type | Count | Description |
|------------|-------|-------------|
| **False negative** | 21 | True COVID predicted as non-COVID (missed case) |
| **False positive** | 12 | True non-COVID predicted as COVID |

### False Negatives — How Severe?

FN severity is assessed by how far the model’s `prob_covid` is from the decision boundary:

| Severity | prob_covid range | Count | Interpretation |
|----------|------------------|-------|----------------|
| **Borderline** | 0.45–0.60 | 7 | Near threshold; small threshold shift could fix |
| **Moderate** | 0.60–0.80 | 8 | Clearly above threshold; harder to correct |
| **Confident** | > 0.80 | 6 | Model very confident; likely systematic or data issue |

- **Mean prob_covid (FN)**: 0.69 (min 0.50, max 0.95)  
- 6 FNs had prob > 0.80 — the model strongly favored COVID but was still predicted non-COVID, suggesting calibration or threshold-tuning issues.  
- 7 FNs were borderline (0.45–0.60) and could be reduced with threshold tuning.

### False Positives — How Severe?

| Severity | prob_covid range | Count | Interpretation |
|----------|------------------|-------|----------------|
| **Low (confident wrong)** | < 0.35 | 7 | Model leaned non-COVID but prediction was COVID |
| **Borderline** | 0.35–0.50 | 5 | Close to threshold |

- **Mean prob_covid (FP)**: 0.32 (min 0.14, max 0.44)  
- Most FPs cluster near or below the threshold; raising the threshold would reduce them but may increase FNs.  

### Summary

- FN: 14/21 are moderate or confident, i.e. not simple borderline errors.  
- FP: 7/12 are “confident wrong” (prob < 0.35), indicating calibration or domain shift.  
- Threshold tuning (e.g. raising from 0.46) may help FPs at the cost of more FNs.

---

## 2. Test Set Confidence Analysis

### Overall Confidence

| Prediction | Count | Mean prob_covid | Range |
|------------|-------|-----------------|-------|
| COVID | 1088 | 0.874 | 0.51–0.99 |
| Non-COVID | 399 | 0.21 | 0.03–0.50 |

So:

- COVID predictions: high mean confidence (0.87).  
- Non-COVID predictions: low mean prob_covid (0.21), i.e. high confidence in non-COVID.

### Confidence Buckets (distance from 0.5)

| Bucket | |prob − 0.5| | Count | % |
|--------|----------------|-------|---|
| **High** | ≥ 0.4 | 721 | 48.5% |
| **Medium** | 0.2–0.4 | 524 | 35.2% |
| **Low** | < 0.2 | 242 | 16.3% |

- About 48.5% of test predictions are highly confident; 84% are at least medium.  
- 242 scans (16.3%) are low-confidence and should be prioritized for review or escalation.  
- 50 scans (3.4%) are in 0.45–0.55; these are the most ambiguous.

### Summary

The model is generally confident on the test set (mean prob 0.70 for COVID, well-separated from non-COVID). The main concern is the ~16% low-confidence cases, where human review or further tests would be most valuable.

---

## 3. Test Set — Parsing & Missing Scans

### Missing scan: ct_scan_492

The test results include **1487 scans**; `ct_scan_492` is **absent** from the output (sequence jumps from ct_scan_491 to ct_scan_493). It was never added to the manifest when `build_test_manifest` ran — most likely because:

- `data/test/ct_scan_492` did not exist, or  
- It had no valid slice images (empty directory or only corrupt / non-numeric files).

Scans excluded at manifest build time are not logged; they are simply omitted.

### Parsing / loading behavior

- **No scans were skipped during inference** in this run — all 1487 manifest entries produced predictions.
- If loading failed at inference time, the pipeline would report it with `status: skipped (reason)` and print a "Skipped: N" summary. None appeared.
- The pipeline uses `_load_image_safe` (TTA) or `_load_image` (no TTA); corrupt images cause exceptions that are caught and reported as skipped when they occur.

**Data type:** CT scans (not MRI).

---

## Error Lists

### False negatives (validation)

**Source 0**: ct_scan_0, ct_scan_4, ct_scan_5, ct_scan_14, ct_scan_15, ct_scan_32, ct_scan_38  
**Source 1**: ct_scan_45, ct_scan_49, ct_scan_51, ct_scan_58, ct_scan_59, ct_scan_63, ct_scan_70, ct_scan_74, ct_scan_76  
**Source 3**: ct_scan_88, ct_scan_90, ct_scan_98, ct_scan_110, ct_scan_111  

### False positives (validation)

**Source 1**: ct_scan_46, ct_scan_59, ct_scan_68, ct_scan_79, ct_scan_80, ct_scan_87  
**Source 2**: ct_scan_102, ct_scan_109, ct_scan_130, ct_scan_134  
**Source 3**: ct_scan_139, ct_scan_169  

---

## CSV Column Reference

### `validation_results_dinov2.csv`

| Column | Description |
|--------|-------------|
| scan_name | CT scan folder name |
| label | Ground truth (0=covid, 1=non_covid) |
| label_name | covid / non_covid |
| prediction | Predicted class (0 or 1) |
| pred_name | covid / non_covid |
| prob_covid | P(covid) from model |
| correct | yes / no |
| source | Data center ID (0–3) |
| error_type | correct / false_negative / false_positive |

### `test_results_dinov2.csv`

| Column | Description |
|--------|-------------|
| scan_name | CT scan folder name |
| prediction | 0=non_covid, 1=covid |
| pred_name | covid / non_covid |
| prob_covid | P(covid) from model |
| source | -1 (test has no source labels) |
| status | predicted |
| confidence | high / medium / low (by \|prob − 0.5\|) |
