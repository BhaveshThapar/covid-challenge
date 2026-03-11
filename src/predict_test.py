"""
Inference on the unlabeled test set.
Outputs a CSV: scan_name, prediction (0=non_covid, 1=covid), prob_covid.

Usage:
  python src/predict_test.py --checkpoint checkpoints/v1_ovr_best.pt \
      --output predictions_test.csv
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
    build_test_manifest, ScanDataset, RawSliceScanDataset,
    get_val_transforms, get_tta_transforms, scan_collate_fn,
)
from src.utils import load_config, set_seed, CheckpointManager
from torch.utils.data import DataLoader


def collect_test_probs(model, entries, config, device, use_amp=True):
    """Run inference on test scans, return (probs, scan_names)."""
    img_size = config["data"]["image_size"]
    ds = ScanDataset(
        entries,
        get_val_transforms(img_size),
        slices_per_scan=config["eval"]["slices_per_scan"],
    )
    loader = DataLoader(
        ds,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        collate_fn=scan_collate_fn,
    )

    model.eval()
    all_probs = []
    scan_names = [e["scan_name"] for e in entries]

    with torch.no_grad():
        for i, (images, _, _, masks) in enumerate(tqdm(loader, desc="Inference")):
            B, K, C, H, W = images.shape
            x_flat = images.view(B * K, C, H, W).to(device)

            with autocast(enabled=use_amp):
                logits = model(x_flat).squeeze(-1)

            probs = torch.sigmoid(logits).view(B, K)
            valid = masks.float().to(device)
            scan_probs = (probs * valid).sum(1) / valid.sum(1).clamp(min=1)
            all_probs.extend(scan_probs.cpu().numpy())

    return np.array(all_probs), scan_names


def collect_test_probs_tta(model, entries, config, device, tta_n=4, use_amp=True):
    """Run TTA inference on test scans."""
    img_size = config["data"]["image_size"]
    tta_tfms = get_tta_transforms(img_size)[:tta_n]
    k = config["eval"]["slices_per_scan"]
    raw_ds = RawSliceScanDataset(entries, slices_per_scan=k)

    model.eval()
    all_probs = []

    for idx in tqdm(range(len(raw_ds)), desc=f"TTA Inference (n={tta_n})"):
        raw_imgs, _, _ = raw_ds[idx]
        aug_probs = []
        for tfm in tta_tfms:
            tensors = torch.stack([tfm(image=img)["image"] for img in raw_imgs]).to(device)
            with torch.no_grad():
                with autocast(enabled=use_amp):
                    logits = model(tensors).squeeze(-1)
            aug_probs.append(torch.sigmoid(logits).cpu().numpy())
        slice_probs = np.mean(aug_probs, axis=0)
        all_probs.append(slice_probs.mean())

    return np.array(all_probs), [e["scan_name"] for e in entries]


def main():
    parser = argparse.ArgumentParser(description="Predict on test set (unlabeled)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--output", type=str, default="predictions_test.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta", action="store_true", help="Use TTA (4 augmentations)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.get("phase2", {}).get("use_amp", True)

    model = DINOv2CovidClassifier(dropout=config["model"]["dropout"]).to(device)
    epoch, score = CheckpointManager.load(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from epoch {epoch}, training score={score:.4f}")

    entries = build_test_manifest(args.data_dir, args.metadata_dir)
    if not entries:
        print("ERROR: No test scans found in data/test/")
        print("Run: sbatch slurm/extract.sbatch (or download/extract test set first)")
        sys.exit(1)
    print(f"Test scans: {len(entries)}")

    tta_n = config["eval"].get("tta_n", 4) if args.tta else 0
    if tta_n > 0:
        probs, scan_names = collect_test_probs_tta(
            model, entries, config, device, tta_n, use_amp
        )
    else:
        probs, scan_names = collect_test_probs(model, entries, config, device, use_amp)

    preds = (probs >= args.threshold).astype(int)
    sources = [e["source"] for e in entries]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pred_names = {0: "non_covid", 1: "covid"}
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scan_name", "prediction", "pred_name", "prob_covid", "source"])
        for name, p, prob, src in zip(scan_names, preds, probs, sources):
            w.writerow([name, int(p), pred_names.get(p, str(p)), f"{prob:.6f}", int(src)])

    print(f"Saved {len(scan_names)} predictions to {args.output}")
    print(f"  Covid (1): {(preds == 1).sum()}")
    print(f"  Non-Covid (0): {(preds == 0).sum()}")
    print(f"  Threshold: {args.threshold}")


if __name__ == "__main__":
    main()
