"""
Evaluation script: per-source macro F1, threshold tuning, TTA, and confusion matrices.

Evaluation flow:
  1. Run TTA inference with N augmentations → collect averaged sigmoid probs.
  2. Tune threshold on val set (sweep 0.30–0.70) → print best threshold + F1.
  3. Print per-source F1, confusion matrices, final challenge score.

Flags:
  --no-tune-threshold  Use config threshold (default 0.5) instead of sweeping.
"""
import os
import sys
import argparse

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DenseNetCovidClassifier
from src.dataset import (
    build_scan_manifest, RawSliceScanDataset,
    get_tta_transforms,
)
from src.utils import (
    load_config, set_seed, compute_per_source_f1,
    print_confusion_matrices, CheckpointManager,
)


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------

def collect_scan_probs_tta(
    model: DenseNetCovidClassifier,
    val_entries: list,
    config: dict,
    device,
    tta_n: int = 4,
    use_amp: bool = True,
) -> tuple:
    """
    Run TTA inference on all validation scans.
    For each scan: load raw slice arrays once, apply N transform pipelines,
    average sigmoid probs across augmentations, then average across slices.

    Returns (probs, labels, sources) as numpy arrays.
    """
    img_size = config["data"]["image_size"]
    tta_tfms = get_tta_transforms(img_size)[:tta_n]
    k = config["eval"]["slices_per_scan"]

    raw_ds = RawSliceScanDataset(val_entries, slices_per_scan=k)

    model.eval()
    all_probs, all_labels, all_sources, all_scan_names = [], [], [], []

    for idx in tqdm(range(len(raw_ds)), desc=f"TTA Inference (n={tta_n})"):
        raw_imgs, label, source, scan_name = raw_ds[idx]   # list of np.ndarray, int, int, str

        aug_probs = []  # one entry per TTA augmentation, shape (n_slices,)
        for tfm in tta_tfms:
            tensors = torch.stack([tfm(image=img)["image"] for img in raw_imgs])  # (K, 3, H, W)
            tensors = tensors.to(device)

            with torch.no_grad():
                with autocast(enabled=use_amp):
                    logits = model(tensors).squeeze(-1)   # (K,)
            aug_probs.append(torch.sigmoid(logits).cpu().numpy())

        # Average across TTA augmentations, then average across slices
        slice_probs = np.mean(aug_probs, axis=0)   # (K,)
        scan_prob = slice_probs.mean()
        all_probs.append(scan_prob)
        all_labels.append(label)
        all_sources.append(source)
        all_scan_names.append(scan_name)

    return np.array(all_probs), np.array(all_labels), np.array(all_sources), all_scan_names


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    lo: float = 0.3,
    hi: float = 0.7,
    steps: int = 41,
) -> tuple:
    """
    Sweep thresholds and return (best_threshold, best_avg_f1).
    Optimises the per-source averaged F1 (the challenge metric).
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(lo, hi, steps):
        preds = (probs >= t).astype(int)
        f1 = compute_per_source_f1(labels, preds, sources)["average"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_results(
    probs: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    threshold: float,
    label: str = "",
):
    preds = (probs >= threshold).astype(int)
    f1_dict = compute_per_source_f1(labels, preds, sources)

    tag = f" [{label}]" if label else ""
    print(f"\n{'='*55}")
    print(f"PER-SOURCE MACRO F1 SCORES{tag}")
    print(f"{'='*55}")
    for k, v in sorted(f1_dict.items()):
        marker = "  ★" if k == "average" else ""
        print(f"  {k:>12}: {v:.4f}{marker}")

    print_confusion_matrices(labels, preds, sources)

    acc = (preds == labels).mean()
    print(f"\nOverall accuracy: {acc:.4f}")
    print(f"Final Challenge Score (P): {f1_dict['average']:.4f}")
    return f1_dict


# ---------------------------------------------------------------------------
# Per-scan COVID breakdown
# ---------------------------------------------------------------------------

def print_covid_breakdown(
    probs: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    scan_names: list,
    threshold: float,
    label: str = "",
    save_csv: str = None,
):
    """
    For each medical center, print which COVID scans (label=0) were correctly
    detected (TP: prob < threshold) vs. missed (FN: prob >= threshold).

    Optionally saves a full per-scan CSV with columns:
        scan_name, center, label, prob, pred, correct
    """
    tag = f" [{label}]" if label else ""
    print(f"\n{'='*55}")
    print(f"COVID SCAN BREAKDOWN BY CENTER{tag}  (threshold={threshold:.2f})")
    print(f"{'='*55}")

    probs = np.array(probs)
    labels = np.array(labels)
    sources = np.array(sources)

    covid_mask = labels == 0
    covid_idx = np.where(covid_mask)[0]

    rows = []  # for CSV

    for center in sorted(set(sources)):
        center_covid = [i for i in covid_idx if sources[i] == center]
        if not center_covid:
            print(f"\n  Center {center}: no COVID scans found")
            continue

        tp = [(i, probs[i]) for i in center_covid if probs[i] < threshold]
        fn = [(i, probs[i]) for i in center_covid if probs[i] >= threshold]

        # Sort: TPs by confidence descending (most confident first), FNs ascending (hardest first)
        tp.sort(key=lambda x: -x[1])
        fn.sort(key=lambda x: x[1])

        print(f"\n  Center {center} — {len(center_covid)} COVID scans  |  TP={len(tp)}  FN={len(fn)}")
        if tp:
            print(f"    CORRECT (TP={len(tp)}):")
            for i, p in tp:
                print(f"      {scan_names[i]:<40}  prob={p:.3f}")
        if fn:
            print(f"    MISSED  (FN={len(fn)}):")
            for i, p in fn:
                print(f"      {scan_names[i]:<40}  prob={p:.3f}  ← missed")

        for i, p in (tp + fn):
            pred = int(p >= threshold)
            rows.append({
                "scan_name": scan_names[i],
                "center": int(center),
                "label": int(labels[i]),
                "prob": round(float(p), 4),
                "pred": pred,
                "correct": int(pred == labels[i]),
            })

    if save_csv:
        import csv, os
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["scan_name", "center", "label", "prob", "pred", "correct"])
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["center"], r["scan_name"])))
        print(f"\n  Saved per-scan CSV → {save_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DenseNet-121 Covid-19 Detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--no-tune-threshold", action="store_true",
                        help="Use config threshold (default 0.5) instead of sweeping")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="If set, save per-scan COVID breakdown to this CSV path")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.get("phase2", {}).get("use_amp", True)

    # Model
    model = DenseNetCovidClassifier(dropout=config["model"]["dropout"]).to(device)
    epoch, score = CheckpointManager.load(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from epoch {epoch}, training score={score:.4f}")

    # Data
    val_entries = build_scan_manifest(args.data_dir, args.split, args.metadata_dir)
    print(f"Val scans: {len(val_entries)}")

    eval_cfg = config["eval"]
    lo = eval_cfg.get("threshold_lo", 0.3)
    hi = eval_cfg.get("threshold_hi", 0.7)
    steps = eval_cfg.get("threshold_steps", 41)
    tta_n = eval_cfg.get("tta_n", 4)

    # ---- TTA inference ----
    print(f"\n--- Running TTA inference (n={tta_n})...")
    probs, labels, sources, scan_names = collect_scan_probs_tta(
        model, val_entries, config, device, tta_n, use_amp
    )

    if args.no_tune_threshold:
        threshold = eval_cfg.get("threshold", 0.5)
        print(f"Using config threshold: {threshold:.2f}")
    else:
        threshold, f1_tuned = tune_threshold(probs, labels, sources, lo, hi, steps)
        print(f"Tuned threshold (TTA): {threshold:.2f}  →  avg F1: {f1_tuned:.4f}")

    print_results(probs, labels, sources, threshold, label=f"TTA n={tta_n}")
    print_covid_breakdown(probs, labels, sources, scan_names, threshold,
                          label=f"TTA n={tta_n}", save_csv=args.output_csv)


if __name__ == "__main__":
    main()
