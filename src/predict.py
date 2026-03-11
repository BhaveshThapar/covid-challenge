"""
Inference script for unlabeled test sets.

Runs the 5-model ensemble on unlabeled CT scans and outputs predictions as CSV.
Handles flat directory structure: test/ct_scan_*/

Features:
  - Test-Time BatchNorm Adaptation (--adapt-bn): re-estimates BN stats on test data
  - CLAHE preprocessing (--clahe): normalizes intensity distributions
  - Multi-flip TTA
  - Score-weighted ensemble

Usage:
  python src/predict.py \
    --configs configs/exp_b3_s42.yaml configs/exp_b3_s123.yaml configs/exp_b3_s7.yaml configs/exp_cnxt_s42.yaml configs/exp_effv2_s42.yaml \
    --checkpoints checkpoints/exp_b3_s42/best.pt checkpoints/exp_b3_s123/best.pt checkpoints/exp_b3_s7/best.pt checkpoints/exp_cnxt_s42/best.pt checkpoints/exp_effv2_s42/best.pt \
    --test-dir data/test \
    --adapt-bn --clahe \
    --output predictions_adapted.csv
"""
import os
import sys
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CovidDetector
from src.dataset import _get_sorted_slices, _load_image, get_val_transforms, scan_collate_fn
from src.utils import load_config, set_seed, CheckpointManager
from src.ensemble_evaluate import load_model, evaluate_single, ensemble_predictions


def get_test_transforms(image_size, use_clahe=True):
    """Test transforms with optional CLAHE to normalize intensity distributions."""
    transforms = [A.Resize(image_size, image_size)]
    if use_clahe:
        transforms.append(A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0))
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return A.Compose(transforms)


def adapt_bn(model, loader, device, num_batches=None):
    """
    Adapt BatchNorm running statistics to the test distribution.

    Resets BN running stats and re-estimates them from test data,
    while keeping all other layers in eval mode.
    """
    if num_batches is None:
        num_batches = len(loader)  # use all data by default

    # Count BN layers
    bn_count = sum(1 for m in model.modules()
                   if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)))
    if bn_count == 0:
        print("  No BatchNorm layers found, skipping BN adaptation")
        return

    # Set BN layers to train mode to update running stats
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.train()
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()

    actual_batches = min(num_batches, len(loader))
    with torch.no_grad():
        for i, (images, _, _, masks) in enumerate(tqdm(loader, desc="Adapting BN", total=actual_batches)):
            if i >= num_batches:
                break
            images = images.to(device)
            masks = masks.to(device)
            with autocast(enabled=True):
                model(images, masks)

    model.eval()
    print(f"  BN adapted: {bn_count} layers updated over {actual_batches} batches")


class UnlabeledScanDataset(Dataset):
    """Dataset for unlabeled test scans in a flat directory structure."""

    def __init__(self, test_dir, transform, slices_per_scan=48):
        self.transform = transform
        self.slices_per_scan = slices_per_scan
        self.entries = []
        self.scan_slices = []

        for scan_name in sorted(os.listdir(test_dir)):
            scan_dir = os.path.join(test_dir, scan_name)
            if not os.path.isdir(scan_dir):
                continue
            # Skip Mac metadata
            if scan_name.startswith('.') or scan_name.startswith('_'):
                continue

            slices = _get_sorted_slices(scan_dir)

            # Handle nested subdirs (like ct_scan_492 with hex subdirs)
            if len(slices) == 0:
                # Check for subdirectories containing slices
                all_sub_slices = []
                for sub in sorted(os.listdir(scan_dir)):
                    sub_dir = os.path.join(scan_dir, sub)
                    if os.path.isdir(sub_dir) and not sub.startswith('.'):
                        all_sub_slices.extend(_get_sorted_slices(sub_dir))
                if all_sub_slices:
                    # Sort all slices by filename number
                    all_sub_slices.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
                    slices = all_sub_slices

            if len(slices) == 0:
                print(f"  [WARNING] Skipping empty scan: {scan_name}")
                continue

            self.entries.append({"scan_name": scan_name, "scan_dir": scan_dir})
            self.scan_slices.append(slices)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        all_slices = self.scan_slices[idx]
        k = self.slices_per_scan

        # Uniform sampling
        if k > 0 and k < len(all_slices):
            indices = np.linspace(0, len(all_slices) - 1, k, dtype=int)
            selected = [all_slices[i] for i in indices]
        else:
            selected = all_slices

        images = []
        for path in selected:
            try:
                img = _load_image(path)
                if self.transform:
                    img = self.transform(image=img)["image"]
                images.append(img)
            except Exception:
                continue

        if len(images) == 0:
            images = [torch.zeros(3, 256, 256)]

        images = torch.stack(images)
        # Use dummy label=0 and source=0 for compatibility with collate_fn
        return images, 0, 0


def predict_single(model, loader, device, tta_mode="multi"):
    """Run inference on unlabeled data, return softmax probabilities."""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images, _, _, masks in tqdm(loader, desc="Predicting"):
            images = images.to(device)
            masks = masks.to(device)

            with autocast(enabled=True):
                logits, _ = model(images, masks)

                if tta_mode == "hflip":
                    logits_hflip, _ = model(torch.flip(images, dims=[-1]), masks)
                    logits = (logits + logits_hflip) / 2.0
                elif tta_mode == "multi":
                    tta_logits = [logits]
                    logits_hflip, _ = model(torch.flip(images, dims=[-1]), masks)
                    tta_logits.append(logits_hflip)
                    logits_vflip, _ = model(torch.flip(images, dims=[-2]), masks)
                    tta_logits.append(logits_vflip)
                    logits_hvflip, _ = model(torch.flip(images, dims=[-2, -1]), masks)
                    tta_logits.append(logits_hvflip)
                    logits = torch.stack(tta_logits).mean(dim=0)

            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description="Test Set Prediction")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="P(covid) threshold; predict covid if P(covid) > threshold")
    parser.add_argument("--adapt-bn", action="store_true",
                        help="Adapt BatchNorm stats to test distribution before inference")
    parser.add_argument("--clahe", action="store_true",
                        help="Apply CLAHE preprocessing to normalize intensity distributions")
    args = parser.parse_args()

    assert len(args.configs) == len(args.checkpoints)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta_mode = "none" if args.no_tta else "multi"

    # Load models
    print("=" * 60)
    print("LOADING ENSEMBLE MODELS")
    print("=" * 60)

    models = []
    scores = []
    configs = []
    for cfg_path, ckpt_path in zip(args.configs, args.checkpoints):
        config = load_config(cfg_path)
        model, score = load_model(config, ckpt_path, device)
        models.append(model)
        scores.append(score)
        configs.append(config)

    # Build test dataset
    config0 = configs[0]
    set_seed(config0["seed"])
    img_size = config0["data"]["image_size"]
    k = config0["eval"]["slices_per_scan"]

    # Use CLAHE transforms if requested
    if args.clahe:
        print("\nUsing CLAHE preprocessing")
        transform = get_test_transforms(img_size, use_clahe=True)
    else:
        transform = get_val_transforms(img_size)

    test_ds = UnlabeledScanDataset(args.test_dir, transform, k)
    test_loader = DataLoader(
        test_ds, batch_size=config0["eval"]["batch_size"],
        shuffle=False, num_workers=config0["data"]["num_workers"],
        pin_memory=False,
        collate_fn=scan_collate_fn,
    )

    print(f"Test set: {len(test_ds)} scans")

    # Run each model
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE")
    if args.adapt_bn:
        print("(with BatchNorm adaptation)")
    print("=" * 60)

    all_model_probs = []
    for i, (model, config) in enumerate(zip(models, configs)):
        print(f"\nModel {i+1}: {config['model']['backbone']} (seed={config['seed']})")

        # Adapt BN statistics to test distribution
        if args.adapt_bn:
            adapt_bn(model, test_loader, device)

        probs = predict_single(model, test_loader, device, tta_mode=tta_mode)
        all_model_probs.append(probs)

    # Score-weighted ensemble
    ens_probs = ensemble_predictions(all_model_probs, weights=scores)

    # Apply threshold
    p_covid = ens_probs[:, 0]  # class 0 = covid
    predictions = (p_covid > args.threshold).astype(int)  # 1 = covid, 0 = non_covid

    # Write CSV
    scan_names = [e["scan_name"] for e in test_ds.entries]
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ct_scan_name", "prediction", "p_covid", "label"])
        for name, pred, p in zip(scan_names, predictions, p_covid):
            label = "covid" if pred == 1 else "non-covid"
            writer.writerow([name, pred, f"{p:.6f}", label])

    # Summary
    n_covid = predictions.sum()
    n_noncovid = len(predictions) - n_covid
    print(f"\n{'=' * 60}")
    print(f"PREDICTIONS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total scans:  {len(predictions)}")
    print(f"  COVID:        {n_covid}")
    print(f"  Non-COVID:    {n_noncovid}")
    print(f"  Threshold:    {args.threshold}")
    print(f"  CLAHE:        {args.clahe}")
    print(f"  BN Adapt:     {args.adapt_bn}")
    print(f"  Output:       {args.output}")

    # Confidence distribution
    print(f"\n  P(covid) stats: mean={p_covid.mean():.4f} std={p_covid.std():.4f} "
          f"min={p_covid.min():.4f} max={p_covid.max():.4f}")
    confident = ((p_covid > 0.7) | (p_covid < 0.3)).sum()
    uncertain = ((p_covid > 0.4) & (p_covid < 0.6)).sum()
    print(f"  Confident (p>0.7 or p<0.3): {confident} ({confident/len(p_covid)*100:.1f}%)")
    print(f"  Uncertain (0.4<p<0.6):      {uncertain} ({uncertain/len(p_covid)*100:.1f}%)")


if __name__ == "__main__":
    main()
