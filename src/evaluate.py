"""
Evaluation script: per-source macro F1, threshold tuning, TTA, and confusion matrices.

Usage:
  python src/evaluate.py --model dinov2 --checkpoint checkpoints/v1_ovr_best.pt
  python src/evaluate.py --model densenet --checkpoint checkpoints/v4_ovr_best.pt
  python src/evaluate.py --model efficientnet --checkpoint checkpoints/best.pt
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import DINOv2CovidClassifier, DenseNetCovidClassifier, CovidDetector
from src.dataset import (
    build_scan_manifest, ScanDataset, RawSliceScanDataset,
    get_val_transforms, get_tta_transforms, scan_collate_fn,
)
from src.utils import (
    load_config, set_seed, compute_per_source_f1,
    print_confusion_matrices, CheckpointManager,
)
from torch.utils.data import DataLoader


MODEL_CONFIGS = {
    "dinov2": "configs/dinov2.yaml",
    "densenet": "configs/densenet.yaml",
    "efficientnet": "configs/efficientnet.yaml",
}


def collect_scan_probs_slice_avg(model, val_entries, config, device, use_amp, tta_n=0):
    """DINOv2/DenseNet: slice-level sigmoid, scan = mean(slice probs)."""
    img_size = config["data"]["image_size"]
    k = config["eval"]["slices_per_scan"]

    if tta_n > 0:
        tta_tfms = get_tta_transforms(img_size)[:tta_n]
        raw_ds = RawSliceScanDataset(val_entries, slices_per_scan=k if k > 0 else -1)
        model.eval()
        all_probs, all_labels, all_sources = [], [], []
        for idx in tqdm(range(len(raw_ds)), desc=f"TTA n={tta_n}"):
            raw_imgs, label, source = raw_ds[idx]
            aug_probs = []
            for tfm in tta_tfms:
                tensors = torch.stack([tfm(image=img)["image"] for img in raw_imgs]).to(device)
                with torch.no_grad():
                    with autocast(enabled=use_amp):
                        logits = model(tensors).squeeze(-1)
                aug_probs.append(torch.sigmoid(logits).cpu().numpy())
            slice_probs = np.mean(aug_probs, axis=0)
            all_probs.append(slice_probs.mean())
            all_labels.append(label)
            all_sources.append(source)
        return np.array(all_probs), np.array(all_labels), np.array(all_sources)

    val_ds = ScanDataset(val_entries, get_val_transforms(img_size), slices_per_scan=k)
    val_loader = DataLoader(
        val_ds, batch_size=config["eval"]["batch_size"], shuffle=False,
        num_workers=config["data"]["num_workers"], pin_memory=config["data"]["pin_memory"],
        collate_fn=scan_collate_fn,
    )
    model.eval()
    all_probs, all_labels, all_sources = [], [], []
    with torch.no_grad():
        for images, labels, sources, masks in tqdm(val_loader, desc="Inference"):
            B, K, C, H, W = images.shape
            x_flat = images.view(B * K, C, H, W).to(device)
            with autocast(enabled=use_amp):
                logits = model(x_flat).squeeze(-1)
            probs = torch.sigmoid(logits).view(B, K)
            valid = masks.float().to(device)
            scan_probs = (probs * valid).sum(1) / valid.sum(1).clamp(min=1)
            all_probs.extend(scan_probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_sources.extend(sources.numpy())
    return np.array(all_probs), np.array(all_labels), np.array(all_sources)


def collect_scan_probs_efficientnet(model, val_entries, config, device, use_amp):
    """EfficientNet: scan-level, softmax, P(covid)=probs[:,0]."""
    img_size = config["data"]["image_size"]
    k = config["eval"]["slices_per_scan"]
    val_ds = ScanDataset(val_entries, get_val_transforms(img_size), slices_per_scan=k)
    val_loader = DataLoader(
        val_ds, batch_size=config["eval"]["batch_size"], shuffle=False,
        num_workers=config["data"]["num_workers"], pin_memory=config["data"]["pin_memory"],
        collate_fn=scan_collate_fn,
    )
    model.eval()
    all_probs, all_labels, all_sources = [], [], []
    with torch.no_grad():
        for images, labels, sources, masks in tqdm(val_loader, desc="Inference"):
            images = images.to(device)
            masks = masks.to(device)
            with autocast(enabled=use_amp):
                logits, _ = model(images, masks)
            probs_b = F.softmax(logits, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs_b)
            all_labels.extend(labels.numpy())
            all_sources.extend(sources.numpy())
    return np.array(all_probs), np.array(all_labels), np.array(all_sources)


def tune_threshold(probs, labels, sources, lo=0.3, hi=0.7, steps=41):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(lo, hi, steps):
        preds = (probs >= t).astype(int)
        f1 = compute_per_source_f1(labels, preds, sources)["average"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def print_results(probs, labels, sources, threshold, label=""):
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate COVID-19 Detector")
    parser.add_argument("--model", type=str, choices=["dinov2", "densenet", "efficientnet"],
                        default="dinov2")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="datasets")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    parser.add_argument("--no-tune-threshold", action="store_true")
    parser.add_argument("--print-results", action="store_true",
                        help="Print per-scan results to log (for copy-paste to notes)")
    args = parser.parse_args()

    config_path = args.config or MODEL_CONFIGS[args.model]
    config = load_config(config_path)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.get("phase2", {}).get("use_amp", True)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(base, args.checkpoint)

    # Build model
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

    epoch, score = CheckpointManager.load(ckpt, model, device=device)
    print(f"Loaded {args.model} checkpoint (epoch {epoch}, score={score:.4f})")

    val_entries = build_scan_manifest(args.data_dir, args.split, args.metadata_dir)
    print(f"Val scans: {len(val_entries)}")

    eval_cfg = config["eval"]
    lo = eval_cfg.get("threshold_lo", 0.3)
    hi = eval_cfg.get("threshold_hi", 0.7)
    steps = eval_cfg.get("threshold_steps", 41)
    tta_n = 0 if args.no_tta else eval_cfg.get("tta_n", eval_cfg.get("tta", False) and 4 or 0)
    if isinstance(tta_n, bool):
        tta_n = 4 if tta_n else 0

    # Inference
    if args.model == "efficientnet":
        probs, labels, sources = collect_scan_probs_efficientnet(model, val_entries, config, device, use_amp)
        tta_n = 0  # EfficientNet TTA handled differently; skip for now
    else:
        probs, labels, sources = collect_scan_probs_slice_avg(
            model, val_entries, config, device, use_amp, tta_n=0
        )

    if args.no_tune_threshold:
        thresh = eval_cfg.get("threshold", 0.5)
        print(f"Using threshold: {thresh:.2f}")
    else:
        thresh, f1 = tune_threshold(probs, labels, sources, lo, hi, steps)
        print(f"Tuned threshold: {thresh:.2f}  →  avg F1: {f1:.4f}")

    print_results(probs, labels, sources, thresh, label="No TTA")
    final_probs, final_thresh = probs, thresh

    if tta_n > 0 and args.model != "efficientnet":
        print(f"\n--- TTA inference (n={tta_n})...")
        tta_probs, _, _ = collect_scan_probs_slice_avg(
            model, val_entries, config, device, use_amp, tta_n=tta_n
        )
        if args.no_tune_threshold:
            thresh_tta = eval_cfg.get("threshold", 0.5)
        else:
            thresh_tta, f1_tta = tune_threshold(tta_probs, labels, sources, lo, hi, steps)
            print(f"TTA tuned threshold: {thresh_tta:.2f}  →  avg F1: {f1_tta:.4f}")
        print_results(tta_probs, labels, sources, thresh_tta, label=f"TTA n={tta_n}")
        final_probs, final_thresh = tta_probs, thresh_tta

    if args.print_results:
        scan_names = [e["scan_name"] for e in val_entries]
        preds = (final_probs >= final_thresh).astype(int)
        correct = (preds == labels)
        label_names = {0: "covid", 1: "non_covid"}
        print("\n" + "=" * 60)
        print("PER-SCAN VALIDATION RESULTS (copy below into notes/Excel)")
        print("=" * 60)
        print("scan_name,label,label_name,prediction,pred_name,prob_covid,correct,source")
        for name, lab, pred, prob, ok, src in zip(scan_names, labels, preds, final_probs, correct, sources):
            print(f"{name},{lab},{label_names.get(lab, str(lab))},{pred},{label_names.get(pred, str(pred))},{prob:.6f},{'yes' if ok else 'no'},{src}")


if __name__ == "__main__":
    main()
