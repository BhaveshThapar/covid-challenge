"""
Ensemble inference: run multiple models, merge via probability averaging or majority vote.

Supports:
  - EfficientNet-family (CovidDetector): scan-level softmax, keeps 2D probs
  - DINOv2 (DINOv2CovidClassifier): slice-level sigmoid, scan = mean(slice probs)
  - DenseNet (DenseNetCovidClassifier): slice-level sigmoid, scan = mean(slice probs)

Strategies:
  - avg (default): average 2D softmax probabilities, use argmax + per-source threshold sweep
  - majority: binarize each model's prob at threshold, take majority vote

Labels: 0 = covid, 1 = non-covid
P(covid) = softmax[:, 0].  When P(covid) > threshold → predict 0 (covid).

Usage:
  python src/ensemble.py --config configs/ensemble.yaml --split val --tune-threshold
  python src/ensemble.py --config configs/ensemble.yaml --split test --output predictions.csv
"""
import os
import sys
import argparse
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Worker functions — one per model type
# ---------------------------------------------------------------------------

def _run_efficientnet_variant(gpu_id: int, config_path: str, checkpoint: str,
                              data_dir: str, metadata_dir: str, split: str,
                              backbone_override: str = None,
                              seed_override: int = None,
                              model_name: str = "efficientnet"):
    """Run a single EfficientNet-family model. Returns (name, scan_names, probs_2d)."""
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
    if seed_override is not None:
        config["seed"] = seed_override
    set_seed(config["seed"])
    device = torch.device("cuda")
    use_amp = config.get("phase2", {}).get("use_amp", True)

    cfg = config["model"]
    backbone = backbone_override or cfg["backbone"]
    model = CovidDetector(
        backbone_name=backbone,
        pretrained=False,
        embedding_dim=cfg["embedding_dim"],
        attention_hidden_dim=cfg["attention_hidden_dim"],
        classifier_hidden_dim=cfg["classifier_hidden_dim"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
        drop_path_rate=cfg.get("drop_path_rate", 0.0),
    ).to(device)
    CheckpointManager.load(checkpoint, model, device=device)

    entries = (build_scan_manifest(data_dir, split, metadata_dir)
               if split in ("val", "train")
               else build_test_manifest(data_dir, metadata_dir))
    img_size = config["data"]["image_size"]
    k = config["eval"]["slices_per_scan"]
    ds = ScanDataset(entries, get_val_transforms(img_size), slices_per_scan=k)
    loader = DataLoader(ds, batch_size=config["eval"]["batch_size"], shuffle=False,
                        num_workers=config["data"]["num_workers"],
                        pin_memory=config["data"]["pin_memory"],
                        collate_fn=scan_collate_fn)

    model.eval()
    all_probs = []  # 2D: (N, num_classes)
    with torch.no_grad():
        for images, _, _, masks in tqdm(loader, desc=model_name, leave=False):
            images = images.to(device)
            masks = masks.to(device)
            with autocast(enabled=use_amp):
                # Multi-flip TTA: 4-view averaging
                tta_logits = []
                logits_orig, _ = model(images, masks)
                tta_logits.append(logits_orig)
                logits_hflip, _ = model(torch.flip(images, dims=[-1]), masks)
                tta_logits.append(logits_hflip)
                logits_vflip, _ = model(torch.flip(images, dims=[-2]), masks)
                tta_logits.append(logits_vflip)
                logits_hvflip, _ = model(torch.flip(images, dims=[-2, -1]), masks)
                tta_logits.append(logits_hvflip)
                logits = torch.stack(tta_logits).mean(dim=0)
            probs = F.softmax(logits, dim=1).cpu().numpy()  # (B, num_classes)
            all_probs.extend(probs)
    scan_names = [e["scan_name"] for e in entries]
    return model_name, scan_names, np.array(all_probs)  # (N, 2)


def _run_slice_model(gpu_id: int, config_path: str, checkpoint: str,
                     data_dir: str, metadata_dir: str, split: str,
                     model_type: str = "dinov2", pretrained_path: str = ""):
    """Run DINOv2 or DenseNet (slice-level sigmoid → mean).
    Returns (name, scan_names, probs_2d) where probs_2d[:, 0] = P(covid)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from torch.cuda.amp import autocast
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.models import DINOv2CovidClassifier, DenseNetCovidClassifier
    from src.dataset import (
        build_test_manifest, build_scan_manifest, ScanDataset,
        get_val_transforms, get_tta_transforms, scan_collate_fn, RawSliceScanDataset,
    )
    from src.utils import load_config, set_seed, CheckpointManager

    config = load_config(config_path)
    set_seed(config["seed"])
    device = torch.device("cuda")
    use_amp = config.get("phase2", {}).get("use_amp", True)

    if model_type == "dinov2":
        model = DINOv2CovidClassifier(dropout=config["model"]["dropout"]).to(device)
    else:  # densenet
        model = DenseNetCovidClassifier(
            pretrained_path=pretrained_path if pretrained_path and os.path.exists(pretrained_path) else None,
            dropout=config["model"]["dropout"],
        ).to(device)
    CheckpointManager.load(checkpoint, model, device=device)

    entries = (build_scan_manifest(data_dir, split, metadata_dir)
               if split in ("val", "train")
               else build_test_manifest(data_dir, metadata_dir))
    img_size = config["data"]["image_size"]
    tta_n = config["eval"].get("tta_n", 0)

    if tta_n > 0:
        tta_tfms = get_tta_transforms(img_size)[:tta_n]
        k = config["eval"]["slices_per_scan"] if config["eval"]["slices_per_scan"] > 0 else -1
        raw_ds = RawSliceScanDataset(entries, slices_per_scan=k)
        model.eval()
        all_p_covid = []
        with torch.no_grad():
            for idx in tqdm(range(len(raw_ds)), desc=f"{model_type} TTA", leave=False):
                raw_imgs, _, _ = raw_ds[idx]
                aug_probs = []
                for tfm in tta_tfms:
                    tensors = torch.stack([tfm(image=img)["image"] for img in raw_imgs]).to(device)
                    with autocast(enabled=use_amp):
                        logits = model(tensors).squeeze(-1)
                    aug_probs.append(torch.sigmoid(logits).cpu().numpy())
                slice_probs = np.mean(aug_probs, axis=0)
                all_p_covid.append(slice_probs.mean())
        p_covid = np.array(all_p_covid)
    else:
        k = config["eval"]["slices_per_scan"]
        ds = ScanDataset(entries, get_val_transforms(img_size), slices_per_scan=k)
        loader = DataLoader(ds, batch_size=config["eval"]["batch_size"], shuffle=False,
                            num_workers=config["data"]["num_workers"],
                            pin_memory=config["data"]["pin_memory"],
                            collate_fn=scan_collate_fn)
        model.eval()
        all_p_covid = []
        with torch.no_grad():
            for images, _, _, masks in tqdm(loader, desc=model_type, leave=False):
                B, K, C, H, W = images.shape
                x_flat = images.view(B * K, C, H, W).to(device)
                with autocast(enabled=use_amp):
                    logits = model(x_flat).squeeze(-1)
                probs_b = torch.sigmoid(logits).view(B, K)
                valid = masks.float().to(device)
                scan_probs = (probs_b * valid).sum(1) / valid.sum(1).clamp(min=1)
                all_p_covid.extend(scan_probs.cpu().numpy())
        p_covid = np.array(all_p_covid)

    # DINOv2/DenseNet sigmoid: high = non-covid (label 1), low = covid (label 0)
    # Convert to 2D [P(covid), P(non-covid)] for uniform interface
    probs_2d = np.stack([1.0 - p_covid, p_covid], axis=1)

    scan_names = [e["scan_name"] for e in entries]
    return model_type, scan_names, probs_2d


def _worker_wrapper(func, args, out_queue):
    """Wrapper for multiprocessing: calls func(*args) and puts result in queue."""
    try:
        result = func(*args)
        out_queue.put(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        out_queue.put(("ERROR", str(e), None))


# ---------------------------------------------------------------------------
# Threshold tuning (matching ensemble_evaluate.py conventions)
# Labels: 0=covid, 1=non-covid.  P(covid) = probs_2d[:, 0].
# When P(covid) > threshold → predict 0 (covid).
# When P(covid) <= threshold → predict 1 (non-covid).
# ---------------------------------------------------------------------------

def sweep_threshold(probs_2d, labels, sources, lo=0.25, hi=0.75, step=0.01):
    """Sweep classification threshold on P(covid). Returns (best_f1, best_thresh, preds)."""
    from src.utils import compute_per_source_f1
    p_covid = probs_2d[:, 0]
    labels = np.array(labels)
    sources = np.array(sources)

    best_f1, best_thresh = 0.0, 0.5
    best_preds = None
    for t in np.arange(lo, hi + step, step):
        preds = np.zeros(len(p_covid), dtype=int)
        preds[p_covid <= t] = 1  # non-covid if P(covid) <= threshold
        f1 = compute_per_source_f1(labels, preds, sources)["average"]
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)
            best_preds = preds.copy()

    return best_f1, best_thresh, best_preds


def sweep_threshold_per_source(probs_2d, labels, sources, lo=0.20, hi=0.80, step=0.005):
    """Sweep a separate threshold per data source. Returns (f1, thresholds_dict, preds)."""
    from sklearn.metrics import f1_score as sklearn_f1
    p_covid = probs_2d[:, 0]
    labels = np.array(labels)
    sources = np.array(sources)
    unique_sources = sorted(np.unique(sources))

    per_source_thresh = {}
    preds = np.zeros(len(p_covid), dtype=int)

    for src in unique_sources:
        mask = sources == src
        p_src = p_covid[mask]
        l_src = labels[mask]

        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(lo, hi + step, step):
            pred_src = np.zeros(mask.sum(), dtype=int)
            pred_src[p_src <= t] = 1  # non-covid if P(covid) <= threshold
            present_labels = np.unique(l_src)
            f1 = sklearn_f1(l_src, pred_src, average="macro",
                           labels=present_labels, zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1

        per_source_thresh[f"source_{src}"] = best_t
        # Apply best threshold for this source
        final_pred_src = np.zeros(mask.sum(), dtype=int)
        final_pred_src[p_src <= best_t] = 1
        preds[mask] = final_pred_src

    from src.utils import compute_per_source_f1
    overall_f1 = compute_per_source_f1(labels, preds, sources)["average"]
    return overall_f1, per_source_thresh, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble prediction: probability averaging or majority vote")
    parser.add_argument("--config", type=str, default="configs/ensemble.yaml")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="datasets")
    parser.add_argument("--output", type=str, default="predictions_ensemble.csv")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--tune-threshold", action="store_true",
                        help="Sweep threshold(s) on val (requires --split val)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override global threshold")
    parser.add_argument("--strategy", type=str, default="avg",
                        choices=["avg", "majority"],
                        help="'avg' (probability averaging) or 'majority' (hard vote)")
    parser.add_argument("--weighting", type=str, default="score",
                        choices=["uniform", "score"],
                        help="'uniform' (equal weights) or 'score' (weight by individual F1)")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs (models round-robin across GPUs)")
    args = parser.parse_args()

    if args.tune_threshold and args.split != "val":
        print("ERROR: --tune-threshold requires --split val")
        sys.exit(1)

    from multiprocessing import Process, Queue
    from src.utils import load_config, compute_per_source_f1, print_confusion_matrices
    from src.dataset import build_test_manifest, build_scan_manifest

    config = load_config(args.config)
    ensemble_cfg = config.get("ensemble", {})
    threshold = args.threshold if args.threshold is not None else ensemble_cfg.get("threshold", 0.5)
    strategy = args.strategy or ensemble_cfg.get("strategy", "avg")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    meta_dir = os.path.join(base_dir, args.metadata_dir) if not os.path.isabs(args.metadata_dir) else args.metadata_dir

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    split = args.split

    # Load labels if val
    labels, sources = None, None
    if split == "val":
        entries_val = build_scan_manifest(data_dir, "val", meta_dir)
        if not entries_val:
            print("ERROR: No val scans found.")
            sys.exit(1)
        labels = np.array([e["label"] for e in entries_val])
        sources = np.array([e["source"] for e in entries_val])

    def _resolve(path):
        return path if os.path.isabs(path) else os.path.join(base_dir, path)

    # Build list of worker tasks from config
    models_cfg = config["models"]
    worker_tasks = []
    for mname, mcfg in models_cfg.items():
        mtype = mcfg.get("type", "efficientnet")
        cfg_path = _resolve(mcfg["config"])
        ckpt_path = _resolve(mcfg["checkpoint"])
        gpu = gpu_ids[len(worker_tasks) % len(gpu_ids)]

        if mtype == "efficientnet":
            worker_tasks.append((
                _run_efficientnet_variant,
                (gpu, cfg_path, ckpt_path, data_dir, meta_dir, split,
                 mcfg.get("backbone_override"), mcfg.get("seed_override"),
                 mname),
            ))
        elif mtype in ("dinov2", "densenet"):
            pretrained = ""
            if mtype == "densenet":
                dense_config = load_config(cfg_path)
                pretrained = dense_config["model"].get("pretrained_path", "")
                pretrained = _resolve(pretrained) if pretrained else ""
            worker_tasks.append((
                _run_slice_model,
                (gpu, cfg_path, ckpt_path, data_dir, meta_dir, split,
                 mtype, pretrained),
            ))

    n_models = len(worker_tasks)
    print(f"Ensemble: {n_models} models, strategy={strategy}, GPUs={gpu_ids}")
    for i, (func, task_args) in enumerate(worker_tasks):
        print(f"  [{i}] {task_args[-1] if func == _run_efficientnet_variant else task_args[-2]}")

    # Launch workers
    out_queue = Queue()
    procs = []
    for func, task_args in worker_tasks:
        p = Process(target=_worker_wrapper, args=(func, task_args, out_queue))
        procs.append(p)
        p.start()

    results = {}
    for _ in range(n_models):
        name, scan_names_or_err, probs = out_queue.get()
        if name == "ERROR":
            print(f"ERROR in worker: {scan_names_or_err}")
            continue
        results[name] = (scan_names_or_err, probs)
    for p in procs:
        p.join()

    if not results:
        print("ERROR: No models returned results.")
        sys.exit(1)

    # Use first model's scan_names as reference
    first_key = list(results.keys())[0]
    scan_names = results[first_key][0]
    n_scans = len(scan_names)

    # Collect all 2D probability arrays: (n_models, n_scans, 2)
    all_probs = []
    model_names = []
    for mname in results:
        all_probs.append(results[mname][1])
        model_names.append(mname)
    all_probs = np.array(all_probs)  # (n_models, n_scans, 2)

    print(f"\nCollected probabilities from {len(model_names)} models for {n_scans} scans")
    for i, mname in enumerate(model_names):
        p = all_probs[i, :, 0]  # P(covid)
        print(f"  {mname}: P(covid) mean={p.mean():.4f}, std={p.std():.4f}, "
              f"min={p.min():.4f}, max={p.max():.4f}")

    # -----------------------------------------------------------------------
    # Combine predictions
    # -----------------------------------------------------------------------
    if strategy == "avg":
        # Compute weights
        weighting = args.weighting
        if weighting == "score" and split == "val":
            # Compute individual F1 scores to use as weights
            indiv_f1s = []
            for i in range(len(model_names)):
                indiv_preds = all_probs[i].argmax(axis=1)
                f1_i = compute_per_source_f1(labels, indiv_preds, sources)["average"]
                indiv_f1s.append(f1_i)
            weights = np.array(indiv_f1s)
            weights = weights / weights.sum()  # normalize
            print(f"\nScore-weighted averaging:")
            for mname, w, f1 in zip(model_names, weights, indiv_f1s):
                print(f"  {mname:>20}: F1={f1:.4f}, weight={w:.4f}")
            ens_probs = np.average(all_probs, axis=0, weights=weights)  # (n_scans, 2)
        elif weighting == "score":
            # Use config weights if available (for test set)
            config_weights = []
            for mname in model_names:
                w = models_cfg.get(mname, {}).get("weight", 1.0)
                config_weights.append(w)
            weights = np.array(config_weights)
            weights = weights / weights.sum()
            ens_probs = np.average(all_probs, axis=0, weights=weights)
        else:
            ens_probs = all_probs.mean(axis=0)  # (n_scans, 2)

        # Also compute uniform average for comparison
        ens_probs_uniform = all_probs.mean(axis=0)

        # Argmax prediction (no threshold)
        preds_argmax = ens_probs.argmax(axis=1)
        p_covid_ens = ens_probs[:, 0]

        if split == "val":
            f1_argmax = compute_per_source_f1(labels, preds_argmax, sources)
            print(f"\nArgmax prediction: F1={f1_argmax['average']:.4f}")

            if args.tune_threshold:
                # Global threshold sweep
                global_f1, global_thresh, global_preds = sweep_threshold(
                    ens_probs, labels, sources)
                print(f"Global threshold sweep: t={global_thresh:.2f} → F1={global_f1:.4f}")

                # Per-source threshold sweep
                ps_f1, ps_thresh, ps_preds = sweep_threshold_per_source(
                    ens_probs, labels, sources)
                print(f"\nPer-source thresholds:")
                for src, t in sorted(ps_thresh.items()):
                    print(f"  {src}: threshold={t:.3f}")
                print(f"  → Per-source F1: {ps_f1:.4f}")

                # Pick the best
                best_f1 = max(f1_argmax["average"], global_f1, ps_f1)
                if ps_f1 >= global_f1 and ps_f1 >= f1_argmax["average"]:
                    preds = ps_preds
                    print(f"\n  → Using per-source thresholds (best)")
                elif global_f1 >= f1_argmax["average"]:
                    preds = global_preds
                    print(f"\n  → Using global threshold (best)")
                else:
                    preds = preds_argmax
                    print(f"\n  → Using argmax (best)")
            else:
                preds = preds_argmax
        else:
            # Test set: use argmax or given threshold
            if args.threshold is not None:
                preds = np.zeros(n_scans, dtype=int)
                preds[p_covid_ens <= threshold] = 1
            else:
                preds = preds_argmax

    elif strategy == "majority":
        # Legacy majority vote with corrected threshold direction
        p_covid_all = all_probs[:, :, 0]  # (n_models, n_scans) — P(covid) per model
        votes_covid = np.zeros(n_scans, dtype=int)
        for i in range(len(model_names)):
            votes_covid += (p_covid_all[i] > threshold).astype(int)  # vote covid if P(covid) > threshold
        majority = np.ceil(n_models / 2.0)
        preds = np.zeros(n_scans, dtype=int)  # default: covid (label 0)
        preds[votes_covid < majority] = 1  # non-covid if fewer than majority vote covid
        p_covid_ens = all_probs.mean(axis=0)[:, 0]

        if split == "val" and args.tune_threshold:
            best_f1, best_t = 0.0, threshold
            for t in np.arange(0.25, 0.76, 0.01):
                v = np.zeros(n_scans, dtype=int)
                for i in range(len(model_names)):
                    v += (p_covid_all[i] > t).astype(int)
                p = np.zeros(n_scans, dtype=int)
                p[v < majority] = 1
                f1 = compute_per_source_f1(labels, p, sources)["average"]
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            threshold = best_t
            votes_covid = np.zeros(n_scans, dtype=int)
            for i in range(len(model_names)):
                votes_covid += (p_covid_all[i] > threshold).astype(int)
            preds = np.zeros(n_scans, dtype=int)
            preds[votes_covid < majority] = 1
            print(f"Tuned threshold: {threshold:.2f} → F1: {best_f1:.4f}")

    n_covid = (preds == 0).sum()
    n_noncovid = (preds == 1).sum()
    print(f"\n{strategy.upper()} ensemble: {n_covid} Covid (label=0), "
          f"{n_noncovid} Non-Covid (label=1)")

    # -----------------------------------------------------------------------
    # Evaluation (val only)
    # -----------------------------------------------------------------------
    if split == "val":
        f1_dict = compute_per_source_f1(labels, preds, sources)
        print(f"\n{'='*55}")
        print(f"PER-SOURCE MACRO F1 SCORES [{strategy.upper()}]")
        print(f"{'='*55}")
        for k, v in sorted(f1_dict.items()):
            marker = "  ★" if k == "average" else ""
            print(f"  {k:>12}: {v:.4f}{marker}")
        print_confusion_matrices(labels, preds, sources)
        acc = (preds == labels).mean()
        print(f"\nOverall accuracy: {acc:.4f}")
        print(f"Challenge score: {f1_dict['average']:.4f}")

        # Individual model F1s for comparison
        print(f"\n{'='*55}")
        print("INDIVIDUAL MODEL F1 SCORES (argmax)")
        print(f"{'='*55}")
        for i, mname in enumerate(model_names):
            indiv_preds = all_probs[i].argmax(axis=1)
            indiv_f1 = compute_per_source_f1(labels, indiv_preds, sources)["average"]
            print(f"  {mname:>20}: {indiv_f1:.4f}")

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    p_covid_ens = all_probs.mean(axis=0)[:, 0] if len(all_probs.shape) == 3 else all_probs.mean(axis=0)
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Always write detailed CSV for debugging
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        if split == "val" and sources is not None:
            w.writerow(["scan_name", "source", "prediction", "prob_covid"])
            for name, src, p, prob in zip(scan_names, sources, preds, p_covid_ens):
                w.writerow([name, int(src), int(p), f"{prob:.6f}"])
        else:
            w.writerow(["scan_name", "prediction", "prob_covid"])
            for name, p, prob in zip(scan_names, preds, p_covid_ens):
                w.writerow([name, int(p), f"{prob:.6f}"])

    print(f"\nSaved {len(scan_names)} predictions to {args.output}")
    n_covid = (preds == 0).sum()      # training label 0 = covid
    n_noncovid = (preds == 1).sum()   # training label 1 = non-covid
    print(f"  Covid: {n_covid}, Non-Covid: {n_noncovid}")

    # For test split: also write challenge submission files
    # Format: covid.csv and non_covid.csv — one scan name per line, no header
    if split == "test":
        covid_path = os.path.join(out_dir, "covid.csv")
        noncovid_path = os.path.join(out_dir, "non_covid.csv")

        covid_names = [name for name, p in zip(scan_names, preds) if p == 0]
        noncovid_names = [name for name, p in zip(scan_names, preds) if p == 1]

        with open(covid_path, "w") as f:
            f.write("\n".join(sorted(covid_names)) + "\n")
        with open(noncovid_path, "w") as f:
            f.write("\n".join(sorted(noncovid_names)) + "\n")

        print(f"\nChallenge submission files:")
        print(f"  {covid_path}: {len(covid_names)} scans")
        print(f"  {noncovid_path}: {len(noncovid_names)} scans")


if __name__ == "__main__":
    main()
