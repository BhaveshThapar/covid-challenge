"""
Evaluation script: per-source macro F1, threshold sweep, TTA, and confusion matrices.

Improvements:
  - Test-Time Augmentation (horizontal flip, vertical flip, combined)
  - Global classification threshold sweep for optimal F1
  - Per-source threshold sweep for independent per-centre optimization
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

from src.model import CovidDetector
from src.dataset import build_scan_dataloaders, build_scan_manifest, ScanDataset, get_val_transforms, scan_collate_fn
from src.utils import load_config, set_seed, compute_per_source_f1, print_confusion_matrices, CheckpointManager
from torch.utils.data import DataLoader


def evaluate(model, val_loader, device, use_amp=True, tta_mode="none"):
    """
    Run evaluation and collect predictions + probabilities.
    
    Args:
        tta_mode: 'none', 'hflip', 'multi' (hflip+vflip+both), or legacy True/False
    """
    # Backward compat: True → 'hflip', False → 'none'
    if tta_mode is True:
        tta_mode = "hflip"
    elif tta_mode is False:
        tta_mode = "none"

    model.eval()
    all_probs, all_labels, all_sources = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images, labels, sources, masks = batch
            images = images.to(device)
            masks = masks.to(device)

            with autocast(enabled=use_amp):
                logits, attn = model(images, masks)

                if tta_mode == "hflip":
                    images_hflip = torch.flip(images, dims=[-1])
                    logits_hflip, _ = model(images_hflip, masks)
                    logits = (logits + logits_hflip) / 2.0

                elif tta_mode == "multi":
                    # Multi-flip TTA: original + hflip + vflip + hflip+vflip
                    tta_logits = [logits]
                    # Horizontal flip
                    logits_hflip, _ = model(torch.flip(images, dims=[-1]), masks)
                    tta_logits.append(logits_hflip)
                    # Vertical flip
                    logits_vflip, _ = model(torch.flip(images, dims=[-2]), masks)
                    tta_logits.append(logits_vflip)
                    # Both flips
                    logits_hvflip, _ = model(torch.flip(images, dims=[-2, -1]), masks)
                    tta_logits.append(logits_hvflip)
                    logits = torch.stack(tta_logits).mean(dim=0)

            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_sources.extend(sources.numpy())

    return np.array(all_probs), np.array(all_labels), np.array(all_sources)


def sweep_threshold(probs, labels, sources, class_idx=0):
    """
    Sweep classification threshold on P(covid) to find optimal F1.
    class_idx=0 means class 0 is covid.
    
    Returns:
        best_f1, best_threshold, preds_at_best
    """
    p_covid = probs[:, class_idx]
    best_f1, best_thresh = 0.0, 0.5
    best_preds = None

    for t in np.arange(0.25, 0.76, 0.01):
        preds = np.zeros(len(p_covid), dtype=int)
        preds[p_covid <= t] = 1  # non-covid if P(covid) <= threshold
        # class 0 = covid, class 1 = non-covid
        f1_dict = compute_per_source_f1(labels, preds, sources)
        if f1_dict["average"] > best_f1:
            best_f1 = f1_dict["average"]
            best_thresh = t
            best_preds = preds.copy()

    return best_f1, best_thresh, best_preds


def sweep_threshold_per_source(probs, labels, sources, class_idx=0):
    """
    Sweep a separate classification threshold for each data source.
    
    Since the competition metric averages per-source F1, optimizing
    thresholds independently per source can yield a higher combined score.
    
    Returns:
        best_f1, per_source_thresholds (dict), preds_at_best
    """
    from sklearn.metrics import f1_score as sklearn_f1
    p_covid = probs[:, class_idx]
    sources_arr = np.array(sources)
    labels_arr = np.array(labels)
    unique_sources = sorted(np.unique(sources_arr))

    per_source_thresh = {}
    preds = np.zeros(len(p_covid), dtype=int)

    for src in unique_sources:
        mask = sources_arr == src
        p_src = p_covid[mask]
        l_src = labels_arr[mask]

        best_t, best_src_f1 = 0.5, 0.0
        for t in np.arange(0.20, 0.80, 0.005):
            pred_src = np.zeros(mask.sum(), dtype=int)
            pred_src[p_src <= t] = 1
            # Only compute F1 for classes present in ground truth (per organizer rules)
            present_labels = np.unique(l_src)
            f1 = sklearn_f1(l_src, pred_src, average="macro", labels=present_labels, zero_division=0)
            if f1 > best_src_f1:
                best_src_f1 = f1
                best_t = t

        per_source_thresh[f"source_{src}"] = best_t
        preds[mask] = 0
        preds[mask & (p_covid <= best_t)] = 1

    f1_dict = compute_per_source_f1(labels_arr, preds, sources_arr)
    return f1_dict["average"], per_source_thresh, preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate Covid-19 Detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    parser.add_argument("--no-threshold-sweep", action="store_true",
                        help="Disable threshold sweep")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = CovidDetector(
        backbone_name=config["model"]["backbone"],
        pretrained=False,
        embedding_dim=config["model"]["embedding_dim"],
        attention_hidden_dim=config["model"]["attention_hidden_dim"],
        classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        drop_path_rate=config["model"].get("drop_path_rate", 0.0),
    ).to(device)

    epoch, score = CheckpointManager.load(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from epoch {epoch}, score={score:.4f}")

    # Data
    entries = build_scan_manifest(args.data_dir, args.split, args.metadata_dir)
    img_size = config["data"]["image_size"]
    k = config["eval"]["slices_per_scan"]
    val_ds = ScanDataset(entries, get_val_transforms(img_size), k,
                         sampling_strategy="uniform")
    val_loader = DataLoader(
        val_ds, batch_size=config["eval"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        collate_fn=scan_collate_fn,
    )

    # Evaluate
    tta_cfg = config["eval"].get("tta", True)
    if args.no_tta:
        tta_mode = "none"
    elif isinstance(tta_cfg, str):
        tta_mode = tta_cfg  # 'hflip', 'multi', 'none'
    else:
        tta_mode = "hflip" if tta_cfg else "none"
    print(f"TTA mode: {tta_mode}")

    probs, labels, sources = evaluate(model, val_loader, device, tta_mode=tta_mode)

    # Standard argmax evaluation
    preds_argmax = probs.argmax(axis=1)
    f1_dict_argmax = compute_per_source_f1(labels, preds_argmax, sources)

    print("\n" + "=" * 50)
    print("ARGMAX RESULTS")
    print("=" * 50)
    for k_name, v in sorted(f1_dict_argmax.items()):
        marker = "  ★" if k_name == "average" else ""
        print(f"  {k_name:>12}: {v:.4f}{marker}")

    best_overall_f1 = f1_dict_argmax["average"]
    preds = preds_argmax
    f1_dict = f1_dict_argmax

    # Threshold sweep
    do_sweep = config["eval"].get("threshold_sweep", True) and not args.no_threshold_sweep
    if do_sweep:
        # Global threshold sweep
        best_f1, best_thresh, best_preds = sweep_threshold(probs, labels, sources)
        f1_dict_sweep = compute_per_source_f1(labels, best_preds, sources)

        print("\n" + "=" * 50)
        print(f"GLOBAL THRESHOLD SWEEP (best t={best_thresh:.2f})")
        print("=" * 50)
        for k_name, v in sorted(f1_dict_sweep.items()):
            marker = "  ★" if k_name == "average" else ""
            print(f"  {k_name:>12}: {v:.4f}{marker}")

        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            preds = best_preds
            f1_dict = f1_dict_sweep

        # Per-source threshold sweep
        ps_f1, ps_thresholds, ps_preds = sweep_threshold_per_source(probs, labels, sources)
        f1_dict_ps = compute_per_source_f1(labels, ps_preds, sources)

        print("\n" + "=" * 50)
        print(f"PER-SOURCE THRESHOLD SWEEP")
        print("=" * 50)
        for src, t in sorted(ps_thresholds.items()):
            print(f"  {src}: t={t:.3f}")
        for k_name, v in sorted(f1_dict_ps.items()):
            marker = "  ★" if k_name == "average" else ""
            print(f"  {k_name:>12}: {v:.4f}{marker}")

        if ps_f1 > best_overall_f1:
            best_overall_f1 = ps_f1
            preds = ps_preds
            f1_dict = f1_dict_ps
            print(f"\n  → Per-source threshold is best (+{ps_f1 - f1_dict_argmax['average']:.4f})")
        elif best_f1 > f1_dict_argmax["average"]:
            print(f"\n  → Global threshold is best (+{best_f1 - f1_dict_argmax['average']:.4f})")
        else:
            print(f"\n  → Argmax is best")

    # Confusion matrices
    print_confusion_matrices(labels, preds, sources)

    # Overall accuracy
    acc = (preds == labels).mean()
    print(f"\nOverall accuracy: {acc:.4f}")
    print(f"Final Challenge Score (P): {f1_dict['average']:.4f}")


if __name__ == "__main__":
    main()
