#!/usr/bin/env python3
"""
Aggregate Grad-CAM++ saliency maps for EfficientNet-B3 MIL over the full validation set.

Supported checkpoint: checkpoints/exp_b3_s42/best.pt (efficientnet_b3 backbone only).
convnext_tiny and densenet are NOT supported — they lack the bn2 spatial hook target.

Dataset mode (default — produces aggregate figures):
  python scripts/run_gradcam_pp.py \\
      --checkpoint checkpoints/exp_b3_s42/best.pt \\
      --dataset-mode \\
      --output-dir figures

  Outputs:
    figures/mean_cam_covid.png       — mean CT + mean Grad-CAM++ for COVID scans
    figures/mean_cam_noncovid.png    — same for non-COVID scans
    figures/mean_cam_comparison.png  — 4-panel comparison of both classes

Single-scan mode (for quick debugging):
  python scripts/run_gradcam_pp.py \\
      --checkpoint checkpoints/exp_b3_s42/best.pt \\
      --scan-dir data/val/covid/ct_scan_001 \\
      --output figures/debug_cam.png
"""
from __future__ import annotations

import argparse
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
    mil_gradcam_pp,
    overlay_heatmap_on_rgb,
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


def _save_comparison(results: dict, output_dir: str, target: str) -> None:
    import matplotlib.pyplot as plt

    label_names = {0: "covid", 1: "noncovid"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col, (label, name) in enumerate(label_names.items()):
        mean_rgb, overlay, n = results[label]
        axes[0, col].imshow(mean_rgb)
        axes[0, col].set_title(f"Mean CT — {name}  (n={n})", fontsize=12)
        axes[0, col].axis("off")
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"Mean Grad-CAM++ — {name}", fontsize=12)
        axes[1, col].axis("off")
    plt.suptitle(
        f"Aggregate Grad-CAM++  |  EfficientNet-B3 MIL  |  target: {target}",
        fontsize=13,
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "mean_cam_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def run_dataset_mode(args, cfg, model, device) -> None:
    from tqdm import tqdm

    image_size = int(cfg["data"]["image_size"])
    k = int(cfg["eval"]["slices_per_scan"])
    target_class = 0 if args.target == "covid" else 1

    entries = build_scan_manifest(args.data_dir, args.split, args.metadata_dir)
    print(f"Loaded {len(entries)} scans from '{args.split}' split.")

    cam_sum: dict[int, np.ndarray | None] = {0: None, 1: None}
    img_sum: dict[int, np.ndarray | None] = {0: None, 1: None}
    counts: dict[int, int] = {0: 0, 1: 0}
    skipped = 0

    model.eval()

    for entry in tqdm(entries, desc="Aggregate GradCAM"):
        scan_dir = entry["scan_dir"]
        label = entry["label"]   # 0=covid, 1=noncovid
        scan_name = entry["scan_name"]

        try:
            x, mask = _load_mil_scan(scan_dir, image_size, k, device)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  Skip {scan_name}: {e}")
            skipped += 1
            continue

        with torch.no_grad():
            _, attn0 = model(x, mask)
        slice_idx = int(attn0[0].argmax().item())

        try:
            rgb, _, _, _, cam_np = mil_gradcam_pp(
                model, x, mask,
                slice_index=slice_idx,
                target_class=target_class,
                device=device,
            )
        except Exception as e:
            print(f"  GradCAM failed {scan_name}: {e}")
            skipped += 1
            continue

        if cam_sum[label] is None:
            cam_sum[label] = cam_np.astype(np.float64)
            img_sum[label] = rgb.astype(np.float64)
        else:
            cam_sum[label] += cam_np.astype(np.float64)
            img_sum[label] += rgb.astype(np.float64)
        counts[label] += 1

    print(f"\nDone. covid={counts[0]}, noncovid={counts[1]}, skipped={skipped}")

    os.makedirs(args.output_dir, exist_ok=True)
    label_names = {0: "covid", 1: "noncovid"}
    comparison_inputs: dict[int, tuple] = {}

    for label, name in label_names.items():
        if counts[label] == 0:
            print(f"No scans processed for '{name}', skipping.")
            continue

        mean_cam = (cam_sum[label] / counts[label]).astype(np.float32)
        mean_cam = (mean_cam - mean_cam.min()) / (mean_cam.max() - mean_cam.min() + 1e-8)
        mean_rgb = np.clip(img_sum[label] / counts[label], 0, 255).astype(np.uint8)
        overlay = overlay_heatmap_on_rgb(mean_rgb, mean_cam)

        out_path = os.path.join(args.output_dir, f"mean_cam_{name}.png")
        save_side_by_side(
            mean_rgb, overlay, out_path,
            title_left=f"Mean CT — {name}  (n={counts[label]})",
            title_right=f"Mean Grad-CAM++ — {name}  (toward {args.target})",
        )
        print(f"Wrote {out_path}")
        comparison_inputs[label] = (mean_rgb, overlay, counts[label])

    if len(comparison_inputs) == 2:
        _save_comparison(comparison_inputs, args.output_dir, args.target)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate Grad-CAM++ for EfficientNet-B3 MIL over the validation set."
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint (use checkpoints/exp_b3_s42/best.pt)")
    p.add_argument("--target", choices=("covid", "noncovid"), default="covid",
                   help="Class score to explain (default: covid)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    # Dataset mode
    p.add_argument("--dataset-mode", action="store_true",
                   help="Aggregate GradCAM over the full validation set")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--metadata-dir", type=str, default="datasets")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--output-dir", type=str, default="figures")

    # Single-scan mode (debugging only)
    p.add_argument("--scan-dir", type=str, default="",
                   help="Single scan folder (debugging; --dataset-mode preferred)")
    p.add_argument("--output", type=str, default="",
                   help="Output PNG path for single-scan mode")
    p.add_argument("--slice-index", type=str, default="max_attn",
                   help='Slice index or "max_attn" for single-scan mode')

    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    cfg = load_config(CONFIG)
    model = _build_model(cfg, args.checkpoint, device)

    if args.dataset_mode:
        run_dataset_mode(args, cfg, model, device)
        return

    # Single-scan debug mode
    if not args.scan_dir or not args.output:
        p.error("Provide --dataset-mode, or both --scan-dir and --output for single-scan mode.")

    image_size = int(cfg["data"]["image_size"])
    k = int(cfg["eval"]["slices_per_scan"])
    target_class = 0 if args.target == "covid" else 1

    x, mask = _load_mil_scan(args.scan_dir, image_size, k, device)
    model.eval()
    if args.slice_index == "max_attn":
        with torch.no_grad():
            _, attn0 = model(x, mask)
        slice_idx = int(attn0[0].argmax().item())
        log_str = f"max_attn→slice {slice_idx}"
    else:
        slice_idx = int(args.slice_index)
        log_str = f"slice {slice_idx}"

    rgb, overlay, logits, attn, _ = mil_gradcam_pp(
        model, x, mask,
        slice_index=slice_idx,
        target_class=target_class,
        device=device,
    )
    probs = torch.softmax(logits, dim=1)[0]
    right_title = (
        f"Grad-CAM++ (MIL, {log_str}, {args.target}) | "
        f"P(covid)={probs[0]:.3f} attn={attn[0, slice_idx]:.3f}"
    )
    save_side_by_side(rgb, overlay, args.output, title_left="Input slice", title_right=right_title)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
