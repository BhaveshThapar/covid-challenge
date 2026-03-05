"""
Ensemble evaluation: combine predictions from multiple trained models.

Supports:
  - Uniform and weighted soft-voting over softmax probabilities
  - Per-model TTA (horizontal flip)
  - Combined threshold sweep
  - Per-source confusion matrices

Usage:
  python src/ensemble_evaluate.py \
    --configs configs/exp_b3_s42.yaml configs/exp_cnxt_s42.yaml configs/exp_b3_s123.yaml \
    --checkpoints checkpoints/exp_b3_s42/best.pt checkpoints/exp_cnxt_s42/best.pt checkpoints/exp_b3_s123/best.pt \
    --data-dir data --metadata-dir data/metadata
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CovidDetector
from src.dataset import build_scan_manifest, ScanDataset, get_val_transforms, scan_collate_fn
from src.utils import load_config, set_seed, compute_per_source_f1, print_confusion_matrices, CheckpointManager
from src.evaluate import sweep_threshold


def load_model(config, checkpoint_path, device):
    """Load a trained model from config + checkpoint."""
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

    epoch, score = CheckpointManager.load(checkpoint_path, model, device=device)
    print(f"  Loaded {config['model']['backbone']} (seed={config['seed']}) "
          f"from epoch {epoch}, val_f1={score:.4f}")
    return model, score


def evaluate_single(model, val_loader, device, use_tta=True):
    """Evaluate a single model, returning softmax probabilities."""
    model.eval()
    all_probs, all_labels, all_sources = [], [], []

    with torch.no_grad():
        for images, labels, sources, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            with autocast(enabled=True):
                logits, _ = model(images, masks)

                if use_tta:
                    images_flip = torch.flip(images, dims=[-1])
                    logits_flip, _ = model(images_flip, masks)
                    logits = (logits + logits_flip) / 2.0

            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_sources.extend(sources.numpy())

    return np.array(all_probs), np.array(all_labels), np.array(all_sources)


def ensemble_predictions(all_model_probs, weights=None):
    """
    Combine predictions via weighted soft-voting.
    
    Args:
        all_model_probs: list of (N, C) probability arrays
        weights: optional list of floats (e.g., validation F1 scores)
    
    Returns:
        (N, C) averaged probability array
    """
    n_models = len(all_model_probs)
    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    ensemble_probs = np.zeros_like(all_model_probs[0])
    for probs, w in zip(all_model_probs, weights):
        ensemble_probs += w * probs

    return ensemble_probs


def main():
    parser = argparse.ArgumentParser(description="Ensemble Evaluation")
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Config files for each model")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint files for each model")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--weighting", type=str, choices=["uniform", "score"],
                        default="score",
                        help="Ensemble weighting: uniform or score-based")
    args = parser.parse_args()

    assert len(args.configs) == len(args.checkpoints), \
        "Must have same number of configs and checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tta = not args.no_tta

    # Load all models
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

    # Build val loader using the first config (data params should be identical)
    config0 = configs[0]
    set_seed(config0["seed"])
    entries = build_scan_manifest(args.data_dir, args.split, args.metadata_dir)
    img_size = config0["data"]["image_size"]
    k = config0["eval"]["slices_per_scan"]
    val_ds = ScanDataset(entries, get_val_transforms(img_size), k,
                         sampling_strategy="uniform")
    val_loader = DataLoader(
        val_ds, batch_size=config0["eval"]["batch_size"],
        shuffle=False, num_workers=config0["data"]["num_workers"],
        pin_memory=False,  # avoid warning on CPU
        collate_fn=scan_collate_fn,
    )

    # Evaluate each model individually
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL RESULTS")
    print("=" * 60)

    all_model_probs = []
    labels = None
    sources = None

    for i, (model, config) in enumerate(zip(models, configs)):
        print(f"\nModel {i+1}: {config['model']['backbone']} (seed={config['seed']})")
        probs, labels_i, sources_i = evaluate_single(
            model, val_loader, device, use_tta=use_tta)
        all_model_probs.append(probs)

        if labels is None:
            labels, sources = labels_i, sources_i

        preds = probs.argmax(axis=1)
        f1_dict = compute_per_source_f1(labels_i, preds, sources_i)
        for k_name, v in sorted(f1_dict.items()):
            marker = "  ★" if k_name == "average" else ""
            print(f"  {k_name:>12}: {v:.4f}{marker}")

    # Ensemble — uniform weighting
    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS (uniform weighting)")
    print("=" * 60)

    ens_probs_uniform = ensemble_predictions(all_model_probs, weights=None)
    preds_uniform = ens_probs_uniform.argmax(axis=1)
    f1_uniform = compute_per_source_f1(labels, preds_uniform, sources)
    for k_name, v in sorted(f1_uniform.items()):
        marker = "  ★" if k_name == "average" else ""
        print(f"  {k_name:>12}: {v:.4f}{marker}")

    # Ensemble — score-weighted
    if args.weighting == "score":
        print("\n" + "=" * 60)
        print(f"ENSEMBLE RESULTS (score-weighted: {[f'{s:.4f}' for s in scores]})")
        print("=" * 60)

        ens_probs_weighted = ensemble_predictions(all_model_probs, weights=scores)
        preds_weighted = ens_probs_weighted.argmax(axis=1)
        f1_weighted = compute_per_source_f1(labels, preds_weighted, sources)
        for k_name, v in sorted(f1_weighted.items()):
            marker = "  ★" if k_name == "average" else ""
            print(f"  {k_name:>12}: {v:.4f}{marker}")

        # Use the better ensemble
        if f1_weighted["average"] >= f1_uniform["average"]:
            ens_probs = ens_probs_weighted
            print("\n  → Using score-weighted ensemble")
        else:
            ens_probs = ens_probs_uniform
            print("\n  → Using uniform ensemble")
    else:
        ens_probs = ens_probs_uniform

    # Threshold sweep on best ensemble
    print("\n" + "=" * 60)
    print("ENSEMBLE + THRESHOLD SWEEP")
    print("=" * 60)

    best_f1, best_thresh, best_preds = sweep_threshold(ens_probs, labels, sources)
    f1_sweep = compute_per_source_f1(labels, best_preds, sources)
    print(f"  Best threshold: {best_thresh:.2f}")
    for k_name, v in sorted(f1_sweep.items()):
        marker = "  ★" if k_name == "average" else ""
        print(f"  {k_name:>12}: {v:.4f}{marker}")

    # Confusion matrices
    print_confusion_matrices(labels, best_preds, sources)

    # Summary
    acc = (best_preds == labels).mean()
    print(f"\nOverall accuracy: {acc:.4f}")
    print(f"Final Challenge Score (P): {f1_sweep['average']:.4f}")


if __name__ == "__main__":
    main()
