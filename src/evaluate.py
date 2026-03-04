"""
Evaluation script: per-source macro F1, threshold sweep, TTA, and confusion matrices.

Improvements:
  - Test-Time Augmentation (horizontal flip)
  - Classification threshold sweep for optimal F1
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


def evaluate(model, val_loader, device, use_amp=True, use_tta=False):
    """Run evaluation and collect predictions + probabilities."""
    model.eval()
    all_probs, all_labels, all_sources = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images, labels, sources, masks = batch
            images = images.to(device)
            masks = masks.to(device)

            with autocast(enabled=use_amp):
                logits, attn = model(images, masks)

                if use_tta:
                    # TTA: horizontal flip
                    images_flip = torch.flip(images, dims=[-1])
                    logits_flip, _ = model(images_flip, masks)
                    logits = (logits + logits_flip) / 2.0

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
    use_tta = config["eval"].get("tta", True) and not args.no_tta
    print(f"TTA: {'enabled' if use_tta else 'disabled'}")

    probs, labels, sources = evaluate(model, val_loader, device, use_tta=use_tta)

    # Standard argmax evaluation
    preds_argmax = probs.argmax(axis=1)
    f1_dict_argmax = compute_per_source_f1(labels, preds_argmax, sources)

    print("\n" + "=" * 50)
    print("ARGMAX RESULTS")
    print("=" * 50)
    for k_name, v in sorted(f1_dict_argmax.items()):
        marker = "  ★" if k_name == "average" else ""
        print(f"  {k_name:>12}: {v:.4f}{marker}")

    # Threshold sweep
    do_sweep = config["eval"].get("threshold_sweep", True) and not args.no_threshold_sweep
    if do_sweep:
        best_f1, best_thresh, best_preds = sweep_threshold(probs, labels, sources)
        f1_dict_sweep = compute_per_source_f1(labels, best_preds, sources)

        print("\n" + "=" * 50)
        print(f"THRESHOLD SWEEP RESULTS (best threshold={best_thresh:.2f})")
        print("=" * 50)
        for k_name, v in sorted(f1_dict_sweep.items()):
            marker = "  ★" if k_name == "average" else ""
            print(f"  {k_name:>12}: {v:.4f}{marker}")

        # Use the better result for final reporting
        if best_f1 > f1_dict_argmax["average"]:
            preds = best_preds
            f1_dict = f1_dict_sweep
            print(f"\n  Threshold sweep improved F1 by "
                  f"+{best_f1 - f1_dict_argmax['average']:.4f}")
        else:
            preds = preds_argmax
            f1_dict = f1_dict_argmax
            print(f"\n  Argmax was better, using argmax results")
    else:
        preds = preds_argmax
        f1_dict = f1_dict_argmax

    # Confusion matrices
    print_confusion_matrices(labels, preds, sources)

    # Overall accuracy
    acc = (preds == labels).mean()
    print(f"\nOverall accuracy: {acc:.4f}")
    print(f"Final Challenge Score (P): {f1_dict['average']:.4f}")


if __name__ == "__main__":
    main()
