"""
Evaluation script: per-source macro F1, threshold tuning, TTA, and confusion matrices.

Evaluation flow:
  1. Run scan-level inference without TTA → collect raw sigmoid probs.
  2. Tune threshold on val set (sweep 0.30–0.70) → print best threshold + F1.
  3. If TTA enabled, re-run inference with N augmentations → independently tune threshold again.
     (TTA shifts the probability distribution, so the optimal threshold differs.)
  4. Print per-source F1, confusion matrices, final challenge score.

Flags:
  --no-tta             Disable TTA (faster evaluation).
  --no-tune-threshold  Use config threshold (default 0.5) instead of sweeping.
  --output-csv         Save per-scan CSV: scan_name, label, prediction, prob_covid, correct.
"""
import os
import sys
import argparse
import csv

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DINOv2CovidClassifier
from src.dataset import (
    build_scan_manifest, ScanDataset, RawSliceScanDataset,
    get_val_transforms, get_tta_transforms, scan_collate_fn,
)
from src.utils import (
    load_config, set_seed, compute_per_source_f1,
    print_confusion_matrices, CheckpointManager,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------

def collect_scan_probs(
    model: DINOv2CovidClassifier,
    val_entries: list,
    config: dict,
    device,
    use_amp: bool = True,
) -> tuple:
    """
    Run inference on all validation scans (all slices, no TTA).
    Returns (probs, labels, sources) as numpy arrays.
    """
    img_size = config["data"]["image_size"]
    val_ds = ScanDataset(
        val_entries,
        get_val_transforms(img_size),
        slices_per_scan=config["eval"]["slices_per_scan"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        collate_fn=scan_collate_fn,
    )

    model.eval()
    all_probs, all_labels, all_sources = [], [], []

    with torch.no_grad():
        for images, labels, sources, masks in tqdm(val_loader, desc="Inference"):
            B, K, C, H, W = images.shape
            x_flat = images.view(B * K, C, H, W).to(device)

            with autocast(enabled=use_amp):
                logits = model(x_flat).squeeze(-1)          # (B*K,)

            probs = torch.sigmoid(logits).view(B, K)        # (B, K)
            valid = masks.float().to(device)                # (B, K)
            scan_probs = (probs * valid).sum(1) / valid.sum(1).clamp(min=1)  # (B,)

            all_probs.extend(scan_probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_sources.extend(sources.numpy())

    return np.array(all_probs), np.array(all_labels), np.array(all_sources)


def collect_scan_probs_tta(
    model: DINOv2CovidClassifier,
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
    all_probs, all_labels, all_sources = [], [], []

    for idx in tqdm(range(len(raw_ds)), desc=f"TTA Inference (n={tta_n})"):
        raw_imgs, label, source = raw_ds[idx]   # list of np.ndarray, int, int
        n_slices = len(raw_imgs)

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

    return np.array(all_probs), np.array(all_labels), np.array(all_sources)


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
    f1_dict = compute_per_source_f1(labels, preds, sources, exclude_missing_classes=True)
    f1_legacy = compute_per_source_f1(labels, preds, sources, exclude_missing_classes=False)

    tag = f" [{label}]" if label else ""
    print(f"\n{'='*55}")
    print(f"PER-SOURCE MACRO F1 SCORES{tag}")
    print(f"{'='*55}")
    for k, v in sorted(f1_dict.items()):
        marker = "  ★" if k == "average" else ""
        print(f"  {k:>12}: {v:.4f}{marker}")
    print(f"  [legacy: both classes, missing=0]: avg = {f1_legacy['average']:.4f}")

    print_confusion_matrices(labels, preds, sources)

    acc = (preds == labels).mean()
    print(f"\nOverall accuracy: {acc:.4f}")
    print(f"Final Challenge Score (P): {f1_dict['average']:.4f}")
    return f1_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 ViT-B/14 Covid-19 Detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable TTA (uses config eval.tta_n when not set)")
    parser.add_argument("--no-tune-threshold", action="store_true",
                        help="Use config threshold (default 0.5) instead of sweeping")
    parser.add_argument("--output-csv", type=str, default="",
                        help="Save per-scan results to CSV (scan_name, label, prediction, prob, correct)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.get("phase2", {}).get("use_amp", True)

    # Model
    model = DINOv2CovidClassifier(dropout=config["model"]["dropout"]).to(device)
    epoch, score = CheckpointManager.load(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from epoch {epoch}, training score={score:.4f}")

    # Data
    val_entries = build_scan_manifest(args.data_dir, args.split, args.metadata_dir)
    print(f"Val scans: {len(val_entries)}")

    eval_cfg = config["eval"]
    lo = eval_cfg.get("threshold_lo", 0.3)
    hi = eval_cfg.get("threshold_hi", 0.7)
    steps = eval_cfg.get("threshold_steps", 41)
    tta_n = 0 if args.no_tta else eval_cfg.get("tta_n", 4)

    # ---- Step 1: Inference without TTA ----
    print("\n--- Running inference (no TTA)...")
    probs, labels, sources = collect_scan_probs(model, val_entries, config, device, use_amp)

    if args.no_tune_threshold:
        thresh_base = eval_cfg.get("threshold", 0.5)
        print(f"Using config threshold: {thresh_base:.2f}")
    else:
        thresh_base, f1_tuned = tune_threshold(probs, labels, sources, lo, hi, steps)
        print(f"Tuned threshold (no TTA): {thresh_base:.2f}  →  avg F1: {f1_tuned:.4f}")

    print_results(probs, labels, sources, thresh_base, label="No TTA")

    # Track final probs/threshold for CSV (use no-TTA unless TTA runs)
    final_probs, final_thresh = probs, thresh_base

    # ---- Step 2: TTA inference (if enabled) ----
    if tta_n > 0:
        print(f"\n--- Running TTA inference (n={tta_n})...")
        tta_probs, _, _ = collect_scan_probs_tta(model, val_entries, config, device, tta_n, use_amp)

        # Independently tune threshold on TTA probabilities (distribution differs from no-TTA)
        if args.no_tune_threshold:
            thresh_tta = eval_cfg.get("threshold", 0.5)
        else:
            thresh_tta, f1_tta_tuned = tune_threshold(tta_probs, labels, sources, lo, hi, steps)
            print(f"Tuned threshold (TTA):    {thresh_tta:.2f}  →  avg F1: {f1_tta_tuned:.4f}")

        print_results(tta_probs, labels, sources, thresh_tta, label=f"TTA n={tta_n}")
        final_probs, final_thresh = tta_probs, thresh_tta

    # ---- Save per-scan CSV if requested ----
    if args.output_csv:
        scan_names = [e["scan_name"] for e in val_entries]
        preds = (final_probs >= final_thresh).astype(int)
        correct = (preds == labels)
        label_names = {0: "covid", 1: "non_covid"}
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scan_name", "label", "label_name", "prediction", "pred_name", "prob_covid", "correct", "source"])
            for name, lab, pred, prob, ok, src in zip(scan_names, labels, preds, final_probs, correct, sources):
                w.writerow([
                    name, int(lab), label_names.get(lab, str(lab)),
                    int(pred), label_names.get(pred, str(pred)),
                    f"{prob:.6f}", "yes" if ok else "no", int(src)
                ])
        print(f"\nSaved per-scan results to {args.output_csv}")


if __name__ == "__main__":
    main()
