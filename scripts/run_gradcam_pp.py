#!/usr/bin/env python3
"""
Generate Grad-CAM++ overlays for DINOv2 (slice-level) or EfficientNet MIL (scan-level).

Examples:
  # DINOv2 — single JPEG slice
  python scripts/run_gradcam_pp.py --model dinov2 --config configs/dinov2.yaml \\
      --checkpoint checkpoints/v1_ovr_best.pt --image /path/to/slice.jpg \\
      --output figures/dinov2_cam.png

  # MIL — scan folder (uniformly samples K slices like eval)
  python scripts/run_gradcam_pp.py --model mil --config configs/efficientnet.yaml \\
      --checkpoint checkpoints/exp_b3_s42/best.pt --scan-dir /data/val/covid/ct_scan_xxx \\
      --output figures/mil_cam.png --slice-index max_attn
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
    _load_image,
    _load_image_safe,
    get_val_transforms,
)
from src.gradcam_pp import (  # noqa: E402
    dinov2_gradcam_pp,
    mil_gradcam_pp,
    save_side_by_side,
)
from src.models import CovidDetector, DINOv2CovidClassifier  # noqa: E402
from src.utils import CheckpointManager, load_config, set_seed  # noqa: E402


def _load_dinov2_slice(image_path: str, image_size: int, device: torch.device) -> torch.Tensor:
    img = _load_image(image_path)
    tfm = get_val_transforms(image_size)
    t = tfm(image=img)["image"]
    return t.unsqueeze(0).to(device)


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
    k = vol.shape[0]
    x = vol.unsqueeze(0).to(device)
    mask = torch.ones(1, k, device=device)
    return x, mask


def main() -> None:
    p = argparse.ArgumentParser(description="Grad-CAM++ for DINOv2 or MIL (EfficientNet)")
    p.add_argument("--model", choices=("dinov2", "mil"), required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True, help="PNG path for side-by-side figure")
    p.add_argument("--image", type=str, default="", help="Single slice (DINOv2)")
    p.add_argument("--scan-dir", type=str, default="", help="Scan folder of slices (MIL)")
    p.add_argument(
        "--target",
        choices=("covid", "noncovid"),
        default="covid",
        help="Class score to explain (DINOv2: -logit vs logit; MIL: logits[:, class])",
    )
    p.add_argument(
        "--vit-block",
        type=int,
        default=10,
        help="DINOv2 only: backbone.blocks index to hook (10 recommended for ViT-B/14; 11 has zero patch grad)",
    )
    p.add_argument(
        "--slice-index",
        type=str,
        default="max_attn",
        help='MIL only: integer slice index or "max_attn" (highest attention weight)',
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    cfg = load_config(args.config)
    image_size = int(cfg["data"]["image_size"])
    target_covid = args.target == "covid"
    target_class = 0 if target_covid else 1

    if args.model == "dinov2":
        if not args.image:
            p.error("--image is required for DINOv2")
        model = DINOv2CovidClassifier(dropout=float(cfg["model"]["dropout"]))
        CheckpointManager.load(args.checkpoint, model, device=device)
        model.to(device)
        x = _load_dinov2_slice(args.image, image_size, device)
        rgb, overlay, logit = dinov2_gradcam_pp(
            model,
            x,
            target_covid=target_covid,
            vit_block_index=args.vit_block,
            device=device,
        )
        left_title = "Input slice"
        right_title = f"Grad-CAM++ (block {args.vit_block}, toward {args.target}) | logit={logit:.3f}"
    else:
        if not args.scan_dir:
            p.error("--scan-dir is required for MIL")
        k = int(cfg["eval"]["slices_per_scan"])
        model = CovidDetector(
            backbone_name=cfg["model"]["backbone"],
            pretrained=False,
            embedding_dim=cfg["model"]["embedding_dim"],
            attention_hidden_dim=cfg["model"]["attention_hidden_dim"],
            classifier_hidden_dim=cfg["model"]["classifier_hidden_dim"],
            num_classes=cfg["model"]["num_classes"],
            dropout=cfg["model"]["dropout"],
            drop_path_rate=cfg["model"].get("drop_path_rate", 0.0),
        )
        CheckpointManager.load(args.checkpoint, model, device=device)
        model.to(device)
        x, mask = _load_mil_scan(args.scan_dir, image_size, k, device)

        slice_idx: int
        if args.slice_index == "max_attn":
            with torch.no_grad():
                _, attn0 = model(x, mask)
            slice_idx = int(attn0[0].argmax().item())
            log_str = f"max_attn→slice {slice_idx}"
        else:
            slice_idx = int(args.slice_index)
            log_str = f"slice {slice_idx}"

        rgb, overlay, logits, attn = mil_gradcam_pp(
            model,
            x,
            mask,
            slice_index=slice_idx,
            target_class=target_class,
            device=device,
        )
        probs = torch.softmax(logits, dim=1)[0]
        left_title = "Input slice"
        right_title = (
            f"Grad-CAM++ (MIL, {log_str}, {args.target}) | "
            f"P(covid)={probs[0]:.3f} attn={attn[0, slice_idx]:.3f}"
        )

    save_side_by_side(rgb, overlay, args.output, title_left=left_title, title_right=right_title)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
