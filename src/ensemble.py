"""
Ensemble inference: run DINOv2, DenseNet, and EfficientNet in parallel on separate GPUs,
merge predictions via weighted average of P(covid), output final CSV.

Usage:
  python src/ensemble.py --config configs/ensemble.yaml --output predictions_ensemble.csv
  python src/ensemble.py --weights 0.4 0.3 0.3  # override weights
"""
import os
import sys
import argparse
import csv
from multiprocessing import Process, Queue

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_dinov2(gpu_id: int, config_path: str, checkpoint: str, data_dir: str,
                metadata_dir: str, split: str, out_queue: Queue):
    """Worker: DINOv2 inference on given GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from torch.cuda.amp import autocast
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.models import DINOv2CovidClassifier
    from src.dataset import (
        build_test_manifest, build_scan_manifest, ScanDataset,
        get_val_transforms, get_tta_transforms, scan_collate_fn,
    )
    from src.utils import load_config, set_seed, CheckpointManager

    config = load_config(config_path)
    set_seed(config["seed"])
    device = torch.device("cuda")
    use_amp = config.get("phase2", {}).get("use_amp", True)

    model = DINOv2CovidClassifier(dropout=config["model"]["dropout"]).to(device)
    CheckpointManager.load(checkpoint, model, device=device)

    entries = build_scan_manifest(data_dir, split, metadata_dir) if split in ("val", "train") else build_test_manifest(data_dir, metadata_dir)
    img_size = config["data"]["image_size"]
    tta_n = config["eval"].get("tta_n", 4)

    if tta_n > 0:
        from src.dataset import RawSliceScanDataset
        tta_tfms = get_tta_transforms(img_size)[:tta_n]
        k = config["eval"]["slices_per_scan"] if config["eval"]["slices_per_scan"] > 0 else -1
        raw_ds = RawSliceScanDataset(entries, slices_per_scan=k)
        model.eval()
        all_probs = []
        with torch.no_grad():
            for idx in tqdm(range(len(raw_ds)), desc="DINOv2 TTA", leave=False):
                raw_imgs, _, _ = raw_ds[idx]
                aug_probs = []
                for tfm in tta_tfms:
                    tensors = torch.stack([tfm(image=img)["image"] for img in raw_imgs]).to(device)
                    with autocast(enabled=use_amp):
                        logits = model(tensors).squeeze(-1)
                    aug_probs.append(torch.sigmoid(logits).cpu().numpy())
                slice_probs = np.mean(aug_probs, axis=0)
                all_probs.append(slice_probs.mean())
        probs = np.array(all_probs)
        scan_names = [e["scan_name"] for e in entries]
    else:
        ds = ScanDataset(entries, get_val_transforms(img_size),
                         slices_per_scan=config["eval"]["slices_per_scan"])
        loader = DataLoader(ds, batch_size=config["eval"]["batch_size"], shuffle=False,
                            num_workers=config["data"]["num_workers"], pin_memory=config["data"]["pin_memory"],
                            collate_fn=scan_collate_fn)
        model.eval()
        all_probs = []
        with torch.no_grad():
            for images, _, _, masks in tqdm(loader, desc="DINOv2", leave=False):
                B, K, C, H, W = images.shape
                x_flat = images.view(B * K, C, H, W).to(device)
                with autocast(enabled=use_amp):
                    logits = model(x_flat).squeeze(-1)
                probs_b = torch.sigmoid(logits).view(B, K)
                valid = masks.float().to(device)
                scan_probs = (probs_b * valid).sum(1) / valid.sum(1).clamp(min=1)
                all_probs.extend(scan_probs.cpu().numpy())
        probs = np.array(all_probs)
        scan_names = [e["scan_name"] for e in entries]

    out_queue.put(("dinov2", scan_names, probs))


def _run_densenet(gpu_id: int, config_path: str, checkpoint: str, data_dir: str,
                  metadata_dir: str, pretrained_path: str, split: str, out_queue: Queue):
    """Worker: DenseNet inference on given GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from torch.cuda.amp import autocast
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.models import DenseNetCovidClassifier
    from src.dataset import (
        build_test_manifest, build_scan_manifest, ScanDataset,
        get_val_transforms, get_tta_transforms, scan_collate_fn, RawSliceScanDataset,
    )
    from src.utils import load_config, set_seed, CheckpointManager

    config = load_config(config_path)
    set_seed(config["seed"])
    device = torch.device("cuda")
    use_amp = config.get("phase2", {}).get("use_amp", True)

    model = DenseNetCovidClassifier(
        pretrained_path=pretrained_path if os.path.exists(pretrained_path) else None,
        dropout=config["model"]["dropout"],
    ).to(device)
    CheckpointManager.load(checkpoint, model, device=device)

    entries = build_scan_manifest(data_dir, split, metadata_dir) if split in ("val", "train") else build_test_manifest(data_dir, metadata_dir)
    img_size = config["data"]["image_size"]
    tta_n = config["eval"].get("tta_n", 4)

    if tta_n > 0:
        tta_tfms = get_tta_transforms(img_size)[:tta_n]
        k = config["eval"]["slices_per_scan"] if config["eval"]["slices_per_scan"] > 0 else -1
        raw_ds = RawSliceScanDataset(entries, slices_per_scan=k)
        model.eval()
        all_probs = []
        with torch.no_grad():
            for idx in tqdm(range(len(raw_ds)), desc="DenseNet TTA", leave=False):
                raw_imgs, _, _ = raw_ds[idx]
                aug_probs = []
                for tfm in tta_tfms:
                    tensors = torch.stack([tfm(image=img)["image"] for img in raw_imgs]).to(device)
                    with autocast(enabled=use_amp):
                        logits = model(tensors).squeeze(-1)
                    aug_probs.append(torch.sigmoid(logits).cpu().numpy())
                slice_probs = np.mean(aug_probs, axis=0)
                all_probs.append(slice_probs.mean())
        probs = np.array(all_probs)
        scan_names = [e["scan_name"] for e in entries]
    else:
        ds = ScanDataset(entries, get_val_transforms(img_size),
                         slices_per_scan=config["eval"]["slices_per_scan"])
        loader = DataLoader(ds, batch_size=config["eval"]["batch_size"], shuffle=False,
                            num_workers=config["data"]["num_workers"], pin_memory=config["data"]["pin_memory"],
                            collate_fn=scan_collate_fn)
        model.eval()
        all_probs = []
        with torch.no_grad():
            for images, _, _, masks in tqdm(loader, desc="DenseNet", leave=False):
                B, K, C, H, W = images.shape
                x_flat = images.view(B * K, C, H, W).to(device)
                with autocast(enabled=use_amp):
                    logits = model(x_flat).squeeze(-1)
                probs_b = torch.sigmoid(logits).view(B, K)
                valid = masks.float().to(device)
                scan_probs = (probs_b * valid).sum(1) / valid.sum(1).clamp(min=1)
                all_probs.extend(scan_probs.cpu().numpy())
        probs = np.array(all_probs)
        scan_names = [e["scan_name"] for e in entries]

    out_queue.put(("densenet", scan_names, probs))


def _run_efficientnet(gpu_id: int, config_path: str, checkpoint: str, data_dir: str,
                      metadata_dir: str, split: str, out_queue: Queue):
    """Worker: EfficientNet (CovidDetector) inference on given GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    import torch.nn.functional as F
    from torch.cuda.amp import autocast
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.models import CovidDetector
    from src.dataset import build_test_manifest, build_scan_manifest, ScanDataset, get_val_transforms, scan_collate_fn
    from src.utils import load_config, set_seed, CheckpointManager

    config = load_config(config_path)
    set_seed(config["seed"])
    device = torch.device("cuda")
    use_amp = config.get("phase2", {}).get("use_amp", True)

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
    CheckpointManager.load(checkpoint, model, device=device)

    entries = build_scan_manifest(data_dir, split, metadata_dir) if split in ("val", "train") else build_test_manifest(data_dir, metadata_dir)
    img_size = config["data"]["image_size"]
    k = config["eval"]["slices_per_scan"]
    ds = ScanDataset(entries, get_val_transforms(img_size), slices_per_scan=k)
    loader = DataLoader(ds, batch_size=config["eval"]["batch_size"], shuffle=False,
                        num_workers=config["data"]["num_workers"], pin_memory=config["data"]["pin_memory"],
                        collate_fn=scan_collate_fn)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for images, _, _, masks in tqdm(loader, desc="EfficientNet", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            with autocast(enabled=use_amp):
                logits, _ = model(images, masks)
            probs_b = F.softmax(logits, dim=1)[:, 0].cpu().numpy()  # P(covid) = class 0
            all_probs.extend(probs_b)
    scan_names = [e["scan_name"] for e in entries]

    out_queue.put(("efficientnet", scan_names, np.array(all_probs)))


def _tune_weights_threshold(prob_dino, prob_dense, prob_eff, labels, sources,
                            weight_step=0.01, t_steps=21):
    """Grid search over weights and threshold. Returns (best_weights, best_thresh, best_f1)."""
    from src.utils import compute_per_source_f1
    best_f1, best_w, best_t = 0.0, None, 0.5
    eps = 1e-9
    for w1 in np.arange(0.0, 1.0 + eps, weight_step):
        for w2 in np.arange(0.0, (1.0 - w1) + eps, weight_step):
            w3 = 1.0 - w1 - w2
            if w3 < -eps:
                continue
            if w3 < 0.0:
                w3 = 0.0
            weights = np.array([w1, w2, w3])
            prob_ens = weights[0] * prob_dino + weights[1] * prob_dense + weights[2] * prob_eff
            for t in np.linspace(0.3, 0.7, t_steps):
                preds = (prob_ens >= t).astype(int)
                f1 = compute_per_source_f1(labels, preds, sources)["average"]
                if f1 > best_f1:
                    best_f1, best_w, best_t = f1, weights.copy(), t
    return best_w, best_t, best_f1


def _tune_per_source_threshold(prob_ensemble, labels, sources):
    """Sweep a separate threshold per source. Returns (best_preds, per_source_thresh, best_f1)."""
    from src.utils import compute_per_source_f1
    from sklearn.metrics import f1_score
    p = np.array(prob_ensemble)
    labels_arr = np.array(labels)
    sources_arr = np.array(sources)
    unique_sources = sorted(np.unique(sources_arr))
    per_source_thresh = {}
    preds = np.zeros(len(p), dtype=int)
    for src in unique_sources:
        mask = sources_arr == src
        if mask.sum() == 0:
            continue
        p_src = p[mask]
        l_src = labels_arr[mask]
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.20, 0.80, 0.01):
            pred_src = (p_src >= t).astype(int)
            present = np.unique(l_src)
            f1 = f1_score(l_src, pred_src, average="macro", labels=present, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        per_source_thresh[f"source_{src}"] = best_t
        preds[mask] = (p_src >= best_t).astype(int)
    f1_dict = compute_per_source_f1(labels_arr, preds, sources_arr)
    return preds, per_source_thresh, f1_dict["average"]


def main():
    parser = argparse.ArgumentParser(description="Ensemble prediction (3 models, multi-GPU)")
    parser.add_argument("--config", type=str, default="configs/ensemble.yaml")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="datasets")
    parser.add_argument("--output", type=str, default="predictions_ensemble.csv")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"],
                        help="test=unlabeled, val=validation with labels")
    parser.add_argument("--tune-weights", action="store_true",
                        help="Grid search weights+threshold on val (requires --split val)")
    parser.add_argument("--per-source-threshold", action="store_true",
                        help="Use per-source thresholds (requires --split val)")
    parser.add_argument("--threshold", type=float, default=None, help="Override config threshold")
    parser.add_argument("--weights", type=float, nargs=3, default=None,
                        help="Override weights: w_dino w_dense w_eff")
    parser.add_argument("--gpus", type=int, nargs=3, default=[0, 1, 2],
                        help="GPU IDs for dinov2, densenet, efficientnet")
    args = parser.parse_args()
    if (args.tune_weights or args.per_source_threshold) and args.split != "val":
        print("ERROR: --tune-weights and --per-source-threshold require --split val")
        sys.exit(1)

    from src.utils import load_config, compute_per_source_f1, print_confusion_matrices
    from src.dataset import build_test_manifest, build_scan_manifest

    config = load_config(args.config)
    ens = config["ensemble"]
    models_cfg = config["models"]

    weights = args.weights if args.weights else ens["weights"]
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    threshold = args.threshold if args.threshold is not None else ens["threshold"]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    meta_dir = os.path.join(base_dir, args.metadata_dir) if not os.path.isabs(args.metadata_dir) else args.metadata_dir

    split = args.split
    labels, sources = None, None
    if split == "val":
        entries_val = build_scan_manifest(data_dir, "val", meta_dir)
        if not entries_val:
            print("ERROR: No val scans found. Use --split test for unlabeled test.")
            sys.exit(1)
        labels = np.array([e["label"] for e in entries_val])
        sources = np.array([e["source"] for e in entries_val])
    elif split == "test":
        entries_test = build_test_manifest(data_dir, meta_dir)
        if not entries_test:
            print("ERROR: No test scans found.")
            sys.exit(1)

    def _resolve(path):
        p = path if os.path.isabs(path) else os.path.join(base_dir, path)
        return p

    out_queue = Queue()
    procs = []
    procs.append(Process(
        target=_run_dinov2,
        args=(args.gpus[0], _resolve(models_cfg["dinov2"]["config"]),
              _resolve(models_cfg["dinov2"]["checkpoint"]),
              data_dir, meta_dir, split, out_queue),
    ))
    densenet_cfg = load_config(_resolve(models_cfg["densenet"]["config"]))
    pretrained = densenet_cfg["model"].get("pretrained_path", "")
    pretrained = _resolve(pretrained) if pretrained else ""
    procs.append(Process(
        target=_run_densenet,
        args=(args.gpus[1], _resolve(models_cfg["densenet"]["config"]),
              _resolve(models_cfg["densenet"]["checkpoint"]),
              data_dir, meta_dir, pretrained, split, out_queue),
    ))
    procs.append(Process(
        target=_run_efficientnet,
        args=(args.gpus[2], _resolve(models_cfg["efficientnet"]["config"]),
              _resolve(models_cfg["efficientnet"]["checkpoint"]),
              data_dir, meta_dir, split, out_queue),
    ))

    for p in procs:
        p.start()
    results = {}
    for _ in range(3):
        name, scan_names, probs = out_queue.get()
        results[name] = (scan_names, probs)
    for p in procs:
        p.join()

    # Merge by scan_name (assume same order)
    scan_names = results["dinov2"][0]
    prob_dino = results["dinov2"][1]
    prob_dense = results["densenet"][1]
    prob_eff = results["efficientnet"][1]

    prob_ensemble = weights[0] * prob_dino + weights[1] * prob_dense + weights[2] * prob_eff

    # Tune weights + threshold on val (if requested)
    if split == "val" and args.tune_weights:
        print("Tuning weights and threshold...")
        weights, threshold, best_f1 = _tune_weights_threshold(
            prob_dino, prob_dense, prob_eff, labels, sources
        )
        print(f"Best weights: DINOv2={weights[0]:.2f}, DenseNet={weights[1]:.2f}, EfficientNet={weights[2]:.2f}")
        print(f"Best threshold: {threshold:.2f}  →  Val F1: {best_f1:.4f}")
        prob_ensemble = weights[0] * prob_dino + weights[1] * prob_dense + weights[2] * prob_eff

    preds = (prob_ensemble >= threshold).astype(int)

    # Per-source threshold (val only)
    if split == "val" and args.per_source_threshold:
        preds, ps_thresh, ps_f1 = _tune_per_source_threshold(prob_ensemble, labels, sources)
        print("\nPer-source thresholds:")
        for k, t in sorted(ps_thresh.items()):
            print(f"  {k}: {t:.2f}")
        print(f"Per-source threshold F1: {ps_f1:.4f}")

    if split == "val":
        f1_dict = compute_per_source_f1(labels, preds, sources)
        print(f"\n{'='*55}")
        print("PER-SOURCE MACRO F1 SCORES")
        print(f"{'='*55}")
        for k, v in sorted(f1_dict.items()):
            marker = "  ★" if k == "average" else ""
            print(f"  {k:>12}: {v:.4f}{marker}")
        print_confusion_matrices(labels, preds, sources)
        print(f"\nChallenge score: {f1_dict['average']:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        if split == "val" and sources is not None:
            w.writerow(["scan_name", "source", "prediction", "prob_covid"])
            for name, src, p, prob in zip(scan_names, sources, preds, prob_ensemble):
                w.writerow([name, int(src), int(p), f"{prob:.6f}"])
        else:
            w.writerow(["scan_name", "prediction", "prob_covid"])
            for name, p, prob in zip(scan_names, preds, prob_ensemble):
                w.writerow([name, int(p), f"{prob:.6f}"])

    print(f"Ensemble weights: DINOv2={weights[0]:.2f}, DenseNet={weights[1]:.2f}, EfficientNet={weights[2]:.2f}")
    print(f"Saved {len(scan_names)} predictions to {args.output}")
    print(f"  Covid (1): {(preds == 1).sum()}, Non-Covid (0): {(preds == 0).sum()}")


if __name__ == "__main__":
    main()
