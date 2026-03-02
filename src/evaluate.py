"""
Evaluation script: per-source macro F1, confusion matrices, and attention visualization.
"""
import os
import sys
import argparse

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CovidDetector
from src.dataset import build_scan_dataloaders, build_scan_manifest, ScanDataset, get_val_transforms, scan_collate_fn
from src.utils import load_config, set_seed, compute_per_source_f1, print_confusion_matrices, CheckpointManager
from torch.utils.data import DataLoader


def evaluate(model, val_loader, device, use_amp=True):
    """Run evaluation and collect predictions."""
    model.eval()
    all_preds, all_labels, all_sources = [], [], []
    all_attn_weights = []
    all_scan_names = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images, labels, sources, masks = batch
            images = images.to(device)
            masks = masks.to(device)

            with autocast(enabled=use_amp):
                logits, attn = model(images, masks)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_sources.extend(sources.numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_sources)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Covid-19 Detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--split", type=str, default="val")
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
    ).to(device)

    epoch, score = CheckpointManager.load(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from epoch {epoch}, score={score:.4f}")

    # Data
    entries = build_scan_manifest(args.data_dir, args.split, args.metadata_dir)
    img_size = config["data"]["image_size"]
    k = config["eval"]["slices_per_scan"]
    val_ds = ScanDataset(entries, get_val_transforms(img_size), k)
    val_loader = DataLoader(
        val_ds, batch_size=config["eval"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        collate_fn=scan_collate_fn,
    )

    # Evaluate
    preds, labels, sources = evaluate(model, val_loader, device)

    # Per-source F1
    f1_dict = compute_per_source_f1(labels, preds, sources)
    print("\n" + "=" * 50)
    print("PER-SOURCE MACRO F1 SCORES")
    print("=" * 50)
    for k, v in sorted(f1_dict.items()):
        marker = "  ★" if k == "average" else ""
        print(f"  {k:>12}: {v:.4f}{marker}")

    # Confusion matrices
    print_confusion_matrices(labels, preds, sources)

    # Overall accuracy
    acc = (preds == labels).mean()
    print(f"\nOverall accuracy: {acc:.4f}")
    print(f"Final Challenge Score (P): {f1_dict['average']:.4f}")


if __name__ == "__main__":
    main()
