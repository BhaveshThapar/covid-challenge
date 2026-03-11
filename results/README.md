# Validation Results — DINOv2 Model

Per-scan validation results for Anant's DINOv2 ViT-B/14 COVID-19 detector.

## Files

| File | Description |
|------|-------------|
| `validation_results_dinov2.csv` | Per-scan predictions on validation set (TTA, threshold 0.46) |

## Metrics (from this run)

- **Challenge Score (TTA)**: 0.9053
- **No-TTA F1**: 0.9113 (threshold 0.51)
- **TTA F1**: 0.9053 (threshold 0.46)
- **Val scans**: 308
- **Errors**: 21 false negatives, 12 false positives

## Error summary

| Error type | Count | Description |
|------------|-------|-------------|
| **False negative** | 21 | True COVID predicted as non‑COVID (missed COVID case) |
| **False positive** | 12 | True non‑COVID predicted as COVID |

### False negatives (true COVID, predicted non‑COVID)

Source 0: ct_scan_0, ct_scan_4, ct_scan_5, ct_scan_14, ct_scan_15, ct_scan_32, ct_scan_38  
Source 1: ct_scan_45, ct_scan_49, ct_scan_51, ct_scan_58, ct_scan_59, ct_scan_63, ct_scan_70, ct_scan_74, ct_scan_76  
Source 3: ct_scan_88, ct_scan_90, ct_scan_98, ct_scan_110, ct_scan_111  

### False positives (true non‑COVID, predicted COVID)

Source 1: ct_scan_46, ct_scan_59, ct_scan_68, ct_scan_79, ct_scan_80, ct_scan_87  
Source 2: ct_scan_102, ct_scan_109, ct_scan_130, ct_scan_134  
Source 3: ct_scan_139, ct_scan_169

## CSV columns

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

## Model

- **Architecture**: DINOv2 ViT-B/14 + classifier head
- **Checkpoint**: `checkpoints/v1_ovr_best.pt`
- **Inference**: TTA (4 augmentations), threshold tuned on val
