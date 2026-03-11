"""
Inference on the unlabeled test set.
Outputs a CSV: scan_name, prediction (0=non_covid, 1=covid), prob_covid.

Usage:
  python src/predict_test.py --model dinov2 --checkpoint checkpoints/v1_ovr_best.pt
  python src/predict_test.py --model densenet --checkpoint checkpoints/v4_ovr_best.pt --output pred_dense.csv
  python src/predict_test.py --model efficientnet --checkpoint checkpoints/best.pt
"""
import os
import sys
import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import DINOv2CovidClassifier, DenseNetCovidClassifier, CovidDetector
from src.dataset import (
    build_test_manifest, ScanDataset, RawSliceScanDataset,
    get_val_transforms, get_tta_transforms, scan_collate_fn,
)
from src.utils import load_config, set_seed, CheckpointManager
from torch.utils.data import DataLoader


MODEL_CONFIGS = {
    "dinov2": "configs/dinov2.yaml",
    "densenet": "configs/densenet.yaml",
    "efficientnet": "configs/efficientnet.yaml",
}


def _collect_slice_avg(model, entries, config, device, use_amp, tta_n=0):
    """DINOv2/DenseNet: slice-level sigmoid, scan = mean(slice probs). Returns (probs, names, skipped)."""
    img_size = config["data"]["image_size"]
    k = config["eval"]["slices_per_scan"]
    skipped = []

    if tta_n > 0:
        tta_tfms = get_tta_transforms(img_size)[:tta_n]
        raw_ds = RawSliceScanDataset(entries, slices_per_scan=k if k > 0 else -1)
        model.eval()
        all_probs, all_names = [], []
        with torch.no_grad():
            for idx in tqdm(range(len(raw_ds)), desc="TTA Inference"):
                try:
                    raw_imgs, _, _ = raw_ds[idx]
                except Exception as e:
                    skipped.append((entries[idx]["scan_name"], str(e)[:100]))
                    continue
                aug_probs = []
                for tfm in tta_tfms:
                    tensors = torch.stack([tfm(image=img)["image"] for img in raw_imgs]).to(device)
                    with autocast(enabled=use_amp):
                        logits = model(tensors).squeeze(-1)
                    aug_probs.append(torch.sigmoid(logits).cpu().numpy())
                slice_probs = np.mean(aug_probs, axis=0)
                all_probs.append(slice_probs.mean())
                all_names.append(entries[idx]["scan_name"])
        return np.array(all_probs) if all_probs else np.array([]), all_names, skipped

    ds = ScanDataset(entries, get_val_transforms(img_size), slices_per_scan=k)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=scan_collate_fn,
    )
    model.eval()
    all_probs, all_names = [], []
    loader_iter = iter(loader)
    for idx in tqdm(range(len(entries)), desc="Inference"):
        try:
            images, _, _, masks = next(loader_iter)
        except StopIteration:
            break
        except Exception as e:
            skipped.append((entries[idx]["scan_name"], str(e)[:100]))
            continue
        with torch.no_grad():
            B, K, C, H, W = images.shape
            x_flat = images.view(B * K, C, H, W).to(device)
            with autocast(enabled=use_amp):
                logits = model(x_flat).squeeze(-1)
            probs = torch.sigmoid(logits).view(B, K)
            valid = masks.float().to(device)
            scan_probs = (probs * valid).sum(1) / valid.sum(1).clamp(min=1)
            all_probs.append(scan_probs.cpu().item())
            all_names.append(entries[idx]["scan_name"])
    return np.array(all_probs) if all_probs else np.array([]), all_names, skipped


def _collect_efficientnet(model, entries, config, device, use_amp):
    """EfficientNet: scan-level (B,K,H,W), softmax, P(covid)=probs[:,0]. Returns (probs, names, skipped)."""
    img_size = config["data"]["image_size"]
    k = config["eval"]["slices_per_scan"]
    ds = ScanDataset(entries, get_val_transforms(img_size), slices_per_scan=k)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=scan_collate_fn,
    )
    model.eval()
    all_probs, all_names, skipped = [], [], []
    loader_iter = iter(loader)
    for idx in tqdm(range(len(entries)), desc="Inference"):
        try:
            images, _, _, masks = next(loader_iter)
        except StopIteration:
            break
        except Exception as e:
            skipped.append((entries[idx]["scan_name"], str(e)[:100]))
            continue
        with torch.no_grad():
            images = images.to(device)
            masks = masks.to(device)
            with autocast(enabled=use_amp):
                logits, _ = model(images, masks)
            probs_b = F.softmax(logits, dim=1)[:, 0].cpu().item()
            all_probs.append(probs_b)
            all_names.append(entries[idx]["scan_name"])
    return np.array(all_probs) if all_probs else np.array([]), all_names, skipped


def main():
    parser = argparse.ArgumentParser(description="Predict on test set (unlabeled)")
    parser.add_argument("--model", type=str, choices=["dinov2", "densenet", "efficientnet"],
                        default="dinov2", help="Model architecture")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (default: configs/<model>.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="datasets")
    parser.add_argument("--output", type=str, default="predictions_test.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta", action="store_true", help="Use TTA (dinov2/densenet only)")
    args = parser.parse_args()

    config_path = args.config or MODEL_CONFIGS[args.model]
    config = load_config(config_path)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.get("phase2", {}).get("use_amp", True)

    # Build model
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.model == "dinov2":
        model = DINOv2CovidClassifier(dropout=config["model"]["dropout"]).to(device)
    elif args.model == "densenet":
        pretrained = config["model"].get("pretrained_path", "")
        pretrained = os.path.join(base, pretrained) if pretrained and not os.path.isabs(pretrained) else pretrained
        model = DenseNetCovidClassifier(
            pretrained_path=pretrained if pretrained and os.path.exists(pretrained) else None,
            dropout=config["model"]["dropout"],
        ).to(device)
    else:  # efficientnet
        cfg = config["model"]
        model = CovidDetector(
            backbone_name=cfg["backbone"],
            pretrained=False,
            embedding_dim=cfg["embedding_dim"],
            attention_hidden_dim=cfg["attention_hidden_dim"],
            classifier_hidden_dim=cfg["classifier_hidden_dim"],
            num_classes=cfg["num_classes"],
            dropout=cfg["dropout"],
            drop_path_rate=cfg.get("drop_path_rate", 0.0),
        ).to(device)

    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(base, args.checkpoint)
    epoch, score = CheckpointManager.load(ckpt, model, device=device)
    print(f"Loaded {args.model} checkpoint (epoch {epoch}, score={score:.4f})")

    entries = build_test_manifest(args.data_dir, args.metadata_dir)
    if not entries:
        print("ERROR: No test scans found in data/test/")
        print("Run: sbatch slurm/extract.sbatch (or download/extract test set first)")
        sys.exit(1)
    print(f"Test scans: {len(entries)}")

    if args.model == "efficientnet":
        probs, scan_names, skipped = _collect_efficientnet(model, entries, config, device, use_amp)
    else:
        tta_n = config["eval"].get("tta_n", 4) if args.tta else 0
        probs, scan_names, skipped = _collect_slice_avg(model, entries, config, device, use_amp, tta_n)

    preds = (probs >= args.threshold).astype(int) if len(probs) > 0 else np.array([])
    pred_names = {0: "non_covid", 1: "covid"}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scan_name", "prediction", "pred_name", "prob_covid", "status"])
        for name, p, prob in zip(scan_names, preds, probs):
            w.writerow([name, int(p), pred_names.get(p, str(p)), f"{prob:.6f}", "predicted"])
        for name, reason in skipped:
            w.writerow([name, "", "", "", f"skipped ({reason})"])

    print(f"Saved to {args.output}")
    print(f"  Predicted: {len(scan_names)} (Covid: {(preds == 1).sum()}, Non-Covid: {(preds == 0).sum()})")
    if skipped:
        print(f"  Skipped:   {len(skipped)}")
        for name, r in skipped[:5]:
            print(f"    - {name}: {r[:60]}{'...' if len(r) > 60 else ''}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")
    print(f"  Threshold: {args.threshold}")


if __name__ == "__main__":
    main()
