#!/usr/bin/env python3
"""
Analyze ensemble validation results: challenge score, FP/FN severity, confidence.

Run from covid-challenge root:
  python results/analyze_val.py

Requires: predictions_ensemble_val.csv and ensemble_val_6364813.out in results/
Optional: data/val, datasets/ for full FP/FN + confidence breakdown per source.
"""
import os
import sys
import csv

# Run from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def load_predictions():
    path = os.path.join(RESULTS, "predictions_ensemble_val.csv")
    if not os.path.exists(path):
        print(f"Missing {path}")
        return None
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames) if r.fieldnames else []
        has_source = "source" in fieldnames
        for row in r:
            rw = {
                "scan_name": row["scan_name"],
                "prediction": int(row["prediction"]),
                "prob_covid": float(row["prob_covid"]),
            }
            if has_source and "source" in row:
                rw["source"] = int(row["source"])
            rows.append(rw)
    return rows


def main():
    preds = load_predictions()
    if not preds:
        return 1

    probs = [p["prob_covid"] for p in preds]
    pred_labels = [p["prediction"] for p in preds]

    # Basic stats
    p0 = [probs[i] for i in range(len(probs)) if pred_labels[i] == 0]
    p1 = [probs[i] for i in range(len(probs)) if pred_labels[i] == 1]
    boundary = sum(1 for p in probs if 0.4 <= p <= 0.6)

    def _stats(arr):
        n = len(arr)
        if n == 0:
            return 0, 0.0, 0.0, 0.0, 0.0
        m = sum(arr) / n
        v = sum((x - m) ** 2 for x in arr) / n
        return n, m, v ** 0.5, min(arr), max(arr)

    n0, m0, s0, lo0, hi0 = _stats(p0)
    n1, m1, s1, lo1, hi1 = _stats(p1)

    print("=" * 60)
    print("VALIDATION PREDICTION ANALYSIS")
    print("=" * 60)
    print(f"\nTotal predictions: {len(preds)}")
    print(f"  Predicted Covid (1): {n1}, Predicted Non-Covid (0): {n0}")

    print("\n--- Model Surety (prob_covid) ---")
    print(f"  Pred=0: mean={m0:.4f}, std={s0:.4f}, range=[{lo0:.4f}, {hi0:.4f}]")
    print(f"  Pred=1: mean={m1:.4f}, std={s1:.4f}, range=[{lo1:.4f}, {hi1:.4f}]")
    print(f"  Near boundary [0.4, 0.6]: {boundary} ({100 * boundary / len(probs):.1f}%)")

    # Try to load manifest for FP/FN with confidence
    data_dir = os.path.join(ROOT, "data")
    meta_dir = os.path.join(ROOT, "datasets")
    if os.path.isdir(data_dir) and os.path.isdir(meta_dir):
        try:
            from src.dataset import build_scan_manifest
            from src.utils import compute_per_source_f1, print_confusion_matrices

            entries = build_scan_manifest(data_dir, "val", meta_dir)
            labels = [e["label"] for e in entries]
            sources = [e["source"] for e in entries]

            # Align predictions with manifest (CSV row order may differ from manifest)
            has_source_col = "source" in (preds[0] if preds else {})
            if has_source_col and len(entries) == len(preds):
                # Robust alignment by (scan_name, source)
                pred_lut = {(p["scan_name"], p["source"]): (p["prediction"], p["prob_covid"]) for p in preds}
                pred_arr = []
                prob_arr = []
                for e in entries:
                    key = (e["scan_name"], e["source"])
                    if key in pred_lut:
                        pp, pr = pred_lut[key]
                        pred_arr.append(pp)
                        prob_arr.append(pr)
                    else:
                        pred_arr.append(-1)
                        prob_arr.append(0.0)
                if len(pred_arr) != len(entries):
                    pred_arr = [p["prediction"] for p in preds]
                    prob_arr = probs
            else:
                pred_arr = [p["prediction"] for p in preds]
                prob_arr = probs

            if len(entries) != len(preds):
                print(f"\nWARNING: manifest has {len(entries)} entries, predictions have {len(preds)}")
            else:
                f1_dict = compute_per_source_f1(labels, pred_arr, sources)
                print("\n--- Per-Source F1 ---")
                for k in sorted(f1_dict.keys()):
                    v = f1_dict[k]
                    m = "  ★" if k == "average" else ""
                    print(f"  {k}: {v:.4f}{m}")

                print("\n--- Confusion Matrices ---")
                print_confusion_matrices(labels, pred_arr, sources)

                # FP/FN with prob_covid (label 0=covid, 1=non-covid; pred 0=non-covid, 1=covid)
                fn_probs = [prob_arr[i] for i in range(len(labels)) if labels[i] == 0 and pred_arr[i] == 0]
                fp_probs = [prob_arr[i] for i in range(len(labels)) if labels[i] == 1 and pred_arr[i] == 1]

                if not has_source_col and (len(fn_probs) > 50 or len(fp_probs) > 50):
                    print("\n--- FP/FN Confidence ---")
                    print("  (Skipped: row-order mismatch likely. Re-run 'ensemble --split val' to get CSV with 'source' column for correct alignment.)")
                elif fn_probs or fp_probs:
                    print("\n--- FP/FN Confidence ---")
                    if fn_probs:
                        m = sum(fn_probs) / len(fn_probs)
                        print(f"  False Negatives (true Covid, pred Non): n={len(fn_probs)}, mean prob_covid={m:.4f}")
                    if fp_probs:
                        m = sum(fp_probs) / len(fp_probs)
                        print(f"  False Positives (true Non, pred Covid): n={len(fp_probs)}, mean prob_covid={m:.4f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nCould not load manifest: {e}")
    else:
        print("\n(Data/metadata dirs not found — skipping manifest-based FP/FN analysis)")
        print("See results/README.md for confusion matrices from the .out file.")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
