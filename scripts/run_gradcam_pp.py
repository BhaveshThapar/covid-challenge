#!/usr/bin/env python3
"""
Saliency maps for EfficientNet-B3 MIL — top-N most confident correct predictions.

Top-N mode (default):
  python scripts/run_gradcam_pp.py \\
      --checkpoint checkpoints/exp_b3_s42/best.pt \\
      --top-n-mode --top-n 4 \\
      --predictions-csv results/predictions_ensemble_val.csv \\
      --output-dir figures

  Outputs (in --output-dir):
    saliency_covid_ct_scan_XXX.png      — one per top-4 COVID scan
    saliency_noncovid_ct_scan_XXX.png   — one per top-4 non-COVID scan
    saliency_top4_grid.png              — combined 2x4 grid of all 8

Single-scan mode (debugging):
  python scripts/run_gradcam_pp.py \\
      --checkpoint checkpoints/exp_b3_s42/best.pt \\
      --scan-dir data/val/covid/ct_scan_001 --output figures/debug.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)

from src.dataset import (  # noqa: E402
    _get_sorted_slices,
    _load_image_safe,
    build_scan_manifest,
    get_val_transforms,
)
from src.gradcam_pp import (  # noqa: E402
    mil_input_gradient,
    overlay_heatmap_on_rgb,
    save_individual_saliency,
    save_side_by_side,
)
from src.models import CovidDetector  # noqa: E402
from src.utils import CheckpointManager, load_config, set_seed  # noqa: E402

CONFIG = "configs/efficientnet.yaml"


def _load_mil_scan(
    scan_dir: str, image_size: int, slices_per_scan: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    paths = _get_sorted_slices(scan_dir)
    if not paths:
        raise FileNotFoundError(f"No slice images in {scan_dir}")
    if slices_per_scan > 0 and len(paths) > slices_per_scan:
        idx = np.linspace(0, len(paths) - 1, slices_per_scan, dtype=int)
        paths = [paths[i] for i in idx]
    tfm = get_val_transforms(image_size)
    tensors = []
    for p in paths:
        arr = _load_image_safe(p)
        if arr is None:
            continue
        tensors.append(tfm(image=arr)["image"])
    if not tensors:
        raise RuntimeError(f"No readable slice images in {scan_dir}")
    vol = torch.stack(tensors, dim=0)
    x = vol.unsqueeze(0).to(device)
    mask = torch.ones(1, vol.shape[0], device=device)
    return x, mask


def _build_model(cfg: dict, checkpoint: str, device: torch.device) -> CovidDetector:
    cfg_m = cfg["model"]
    model = CovidDetector(
        backbone_name=cfg_m["backbone"],
        pretrained=False,
        embedding_dim=cfg_m["embedding_dim"],
        attention_hidden_dim=cfg_m["attention_hidden_dim"],
        classifier_hidden_dim=cfg_m["classifier_hidden_dim"],
        num_classes=cfg_m["num_classes"],
        dropout=cfg_m["dropout"],
        drop_path_rate=cfg_m.get("drop_path_rate", 0.0),
    )
    CheckpointManager.load(checkpoint, model, device=device)
    return model.to(device)


def _load_predictions_csv(csv_path: str) -> dict:
    """Returns {scan_name: {"prob_covid": float, "prediction": int}}."""
    result = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            result[row["scan_name"]] = {
                "prob_covid": float(row["prob_covid"]),
                "prediction": int(row["prediction"]),
            }
    return result


def _save_grid(entries: list[dict], output_dir: str) -> None:
    """2×4 grid: top row = COVID, bottom row = non-COVID, each cell = saliency overlay."""
    import matplotlib.pyplot as plt

    covid_entries    = [e for e in entries if e["true_class"] == "covid"]
    noncovid_entries = [e for e in entries if e["true_class"] == "noncovid"]
    n = max(len(covid_entries), len(noncovid_entries))

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9))

    for col, row_entries in enumerate([covid_entries, noncovid_entries]):
        class_name = "COVID" if col == 0 else "non-COVID"
        for i in range(n):
            ax = axes[col, i]
            if i < len(row_entries):
                e = row_entries[i]
                ax.imshow(e["overlay"])
                ax.set_title(
                    f"{e['scan_name']}\nP(covid)={e['prob_covid']:.3f}",
                    fontsize=8,
                )
            else:
                ax.set_visible(False)
            ax.axis("off")
        axes[col, 0].set_ylabel(class_name, fontsize=11, labelpad=8)

    plt.suptitle("Input × Gradient Saliency  |  EfficientNet-B3 MIL  |  Top-4 by confidence",
                 fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "saliency_top4_grid.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def run_top_n_mode(args, cfg, model, device) -> None:
    image_size = int(cfg["data"]["image_size"])
    k = int(cfg["eval"]["slices_per_scan"])

    # Load ground truth from manifest
    entries = build_scan_manifest(args.data_dir, args.split, args.metadata_dir)
    gt_map  = {e["scan_name"]: e["label"]    for e in entries}
    dir_map = {e["scan_name"]: e["scan_dir"] for e in entries}

    # Load ensemble confidence scores
    preds = _load_predictions_csv(args.predictions_csv)

    # Build combined rows: {scan_name, label, prob_covid, prediction, scan_dir}
    rows = []
    for scan_name, p in preds.items():
        if scan_name not in gt_map:
            continue
        rows.append({
            "scan_name":  scan_name,
            "label":      gt_map[scan_name],       # 0=covid, 1=noncovid
            "scan_dir":   dir_map[scan_name],
            "prob_covid": p["prob_covid"],
            "prediction": p["prediction"],
            "correct":    p["prediction"] == gt_map[scan_name],
        })

    # Top-N correctly classified per class, sorted by confidence
    covid_rows = sorted(
        [r for r in rows if r["label"] == 0 and r["correct"]],
        key=lambda r: r["prob_covid"], reverse=True,
    )[:args.top_n]

    noncovid_rows = sorted(
        [r for r in rows if r["label"] == 1 and r["correct"]],
        key=lambda r: r["prob_covid"],   # lowest prob_covid = most confident non-COVID
    )[:args.top_n]

    print(f"Selected {len(covid_rows)} COVID  and {len(noncovid_rows)} non-COVID scans.")

    os.makedirs(args.output_dir, exist_ok=True)
    model.eval()
    grid_entries = []

    for class_label, class_name, selected in [
        (0, "covid",    covid_rows),
        (1, "noncovid", noncovid_rows),
    ]:
        target_class = class_label
        for row in selected:
            scan_name = row["scan_name"]
            print(f"  Processing {scan_name}  (P(covid)={row['prob_covid']:.3f})")

            try:
                x, mask = _load_mil_scan(row["scan_dir"], image_size, k, device)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"    Skip: {e}")
                continue

            with torch.no_grad():
                _, attn0 = model(x, mask)
            slice_idx = int(attn0[0].argmax().item())

            try:
                rgb, overlay, logits, attn, sal_np = mil_input_gradient(
                    model, x, mask,
                    slice_index=slice_idx,
                    target_class=target_class,
                    device=device,
                )
            except Exception as e:
                print(f"    Saliency failed: {e}")
                continue

            out_path = os.path.join(
                args.output_dir, f"saliency_{class_name}_{scan_name}.png"
            )
            save_individual_saliency(
                rgb, overlay, sal_np, out_path,
                scan_name=scan_name,
                prob_covid=row["prob_covid"],
                true_class=class_name,
                attn_weight=float(attn[0, slice_idx].item()),
            )
            print(f"    Wrote {out_path}")

            grid_entries.append({
                "scan_name":  scan_name,
                "true_class": class_name,
                "prob_covid": row["prob_covid"],
                "overlay":    overlay,
            })

    if grid_entries:
        _save_grid(grid_entries, args.output_dir)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Input × Gradient saliency for EfficientNet-B3 MIL."
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Checkpoint path (use checkpoints/exp_b3_s42/best.pt)")
    p.add_argument("--target", choices=("covid", "noncovid"), default="covid",
                   help="Class to explain (default: covid)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir",      type=str, default="data")
    p.add_argument("--metadata-dir",  type=str, default="datasets")
    p.add_argument("--split",         type=str, default="val")
    p.add_argument("--output-dir",    type=str, default="figures")

    # Top-N mode
    p.add_argument("--top-n-mode", action="store_true",
                   help="Saliency for top-N most confident correct predictions per class")
    p.add_argument("--top-n", type=int, default=4)
    p.add_argument("--predictions-csv", type=str,
                   default="results/predictions_ensemble_val.csv")

    # Single-scan debug mode
    p.add_argument("--scan-dir", type=str, default="")
    p.add_argument("--output",   type=str, default="")
    p.add_argument("--slice-index", type=str, default="max_attn")

    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    cfg    = load_config(CONFIG)
    model  = _build_model(cfg, args.checkpoint, device)

    if args.top_n_mode:
        run_top_n_mode(args, cfg, model, device)
        return

    # Single-scan debug mode
    if not args.scan_dir or not args.output:
        p.error("Provide --top-n-mode, or both --scan-dir and --output.")

    image_size   = int(cfg["data"]["image_size"])
    k            = int(cfg["eval"]["slices_per_scan"])
    target_class = 0 if args.target == "covid" else 1

    x, mask = _load_mil_scan(args.scan_dir, image_size, k, device)
    model.eval()
    if args.slice_index == "max_attn":
        with torch.no_grad():
            _, attn0 = model(x, mask)
        slice_idx = int(attn0[0].argmax().item())
    else:
        slice_idx = int(args.slice_index)

    rgb, overlay, logits, attn, _ = mil_input_gradient(
        model, x, mask,
        slice_index=slice_idx,
        target_class=target_class,
        device=device,
    )
    probs = torch.softmax(logits, dim=1)[0]
    right_title = (
        f"Input × Gradient  |  slice {slice_idx}  |  "
        f"P(covid)={probs[0]:.3f}  attn={attn[0, slice_idx]:.3f}"
    )
    save_side_by_side(rgb, overlay, args.output,
                      title_left="CT slice", title_right=right_title)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
