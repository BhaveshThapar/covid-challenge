"""
Grad-CAM++ saliency for paper figures (DINOv2 ViT + EfficientNet attention MIL).

ViT note: With a CLS-only head, gradients to patch tokens at the *last* block are zero.
We therefore target an earlier transformer block (default: index 10 of 12 for ViT-B).

MIL note: Hooks timm EfficientNet spatial features at ``bn2`` (BatchNormAct2d), after
chunked slice forwards, so CAM is for one slice in the context of the full scan backward pass.

References:
  Selvaraju et al., Grad-CAM
  Chattopadhay et al., Grad-CAM++ (weights implemented as in common PyTorch ports)
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import RADIMAGENET_MEAN, RADIMAGENET_STD


def gradcam_plusplus_weights(
    activations: torch.Tensor, grads: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Per-channel weights for Grad-CAM++.

    Args:
        activations: (1, C, H, W)
        grads: (1, C, H, W) d(score)/d(activations)

    Returns:
        weights: (C,)
    """
    if activations.shape != grads.shape:
        raise ValueError("activations and grads must have the same shape")
    # Grad-CAM++ (Chattopadhay et al.; pytorch-grad-cam style)
    grad_2 = grads.pow(2)
    grad_3 = grad_2 * grads
    sum_activations = activations.sum(dim=(2, 3), keepdim=True)
    denom = 2.0 * grad_3 + sum_activations * grad_3 + eps
    alpha = grad_2 / denom
    weights = (alpha * F.relu(grads)).sum(dim=(2, 3)).squeeze(0)
    return weights


def cam_from_maps(activations: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted sum + ReLU -> (H, W) CAM."""
    # activations: (1, C, H, W), weights: (C,)
    cam = (weights.view(1, -1, 1, 1) * activations).sum(dim=1).squeeze(0)
    cam = F.relu(cam)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def _upsample_cam(cam_hw: torch.Tensor, out_h: int, out_w: int) -> np.ndarray:
    """cam_hw: (H, W) -> numpy (out_h, out_w) float [0,1]."""
    t = cam_hw.unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return up.squeeze().detach().cpu().numpy().astype(np.float32)


def tensor_to_rgb_uint8(tchw: torch.Tensor) -> np.ndarray:
    """Denormalize (3,H,W) ImageNet-normalized tensor to uint8 RGB."""
    mean = torch.tensor(RADIMAGENET_MEAN, device=tchw.device, dtype=tchw.dtype).view(3, 1, 1)
    std = torch.tensor(RADIMAGENET_STD, device=tchw.device, dtype=tchw.dtype).view(3, 1, 1)
    x = tchw.detach() * std + mean
    x = x.cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    return (x * 255.0).astype(np.uint8)


def overlay_heatmap_on_rgb(
    rgb: np.ndarray, cam_01: np.ndarray, alpha: float = 0.45, colormap: str = "jet",
) -> np.ndarray:
    """rgb, cam same HxW; cam in [0,1]. Returns uint8 HxWx3."""
    import matplotlib

    if hasattr(matplotlib, "colormaps"):
        cmap = matplotlib.colormaps[colormap]
    else:
        import matplotlib.cm as cm

        cmap = cm.get_cmap(colormap)
    heat = cmap(cam_01)[:, :, :3]
    heat = (heat * 255.0).astype(np.float32)
    base = rgb.astype(np.float32)
    out = (1.0 - alpha) * base + alpha * heat
    return np.clip(out, 0, 255).astype(np.uint8)


def save_side_by_side(
    rgb: np.ndarray,
    overlay: np.ndarray,
    out_path: str,
    title_left: str = "Input",
    title_right: str = "Grad-CAM++",
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(rgb)
    ax[0].set_title(title_left)
    ax[0].axis("off")
    ax[1].imshow(overlay)
    ax[1].set_title(title_right)
    ax[1].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def dinov2_gradcam_pp(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    target_covid: bool = True,
    vit_block_index: int = 10,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Grad-CAM++ for DINOv2CovidClassifier.

    Args:
        model: DINOv2CovidClassifier
        x: (1, 3, H, W) normalized tensor
        target_covid: if True, explain score toward COVID (repo label 0). DINOv2 uses
            one logit with logit > 0 => non-COVID, so we backprop ``-logit``.
        vit_block_index: which ``model.backbone.blocks[i]`` to hook (not 11 for ViT-B/14
            with CLS-only head — patch grads vanish at the last block).

    Returns:
        rgb_uint8: (H, W, 3) input visualization
        overlay_uint8: (H, W, 3) heatmap overlay
        score_scalar: raw logit (single value)
    """
    from src.models import DINOv2CovidClassifier

    if not isinstance(model, DINOv2CovidClassifier):
        raise TypeError("model must be DINOv2CovidClassifier")

    backbone = model.backbone
    n_blocks = len(backbone.blocks)
    if vit_block_index < 0 or vit_block_index >= n_blocks:
        raise ValueError(f"vit_block_index must be in [0, {n_blocks - 1}], got {vit_block_index}")

    model.eval()
    x = x.to(device)

    captured: List[torch.Tensor] = []

    def hook(_m, _inp, out: torch.Tensor) -> None:
        out.retain_grad()
        captured.append(out)

    handle = backbone.blocks[vit_block_index].register_forward_hook(hook)
    logits = model(x)
    handle.remove()

    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError(f"Expected logits (1,1), got {tuple(logits.shape)}")

    model.zero_grad(set_to_none=True)
    logit = logits.squeeze()
    if target_covid:
        score = -logit
    else:
        score = logit
    score.backward(retain_graph=False)

    if not captured or captured[0].grad is None:
        raise RuntimeError("No gradients reached the ViT hook; try a lower vit_block_index.")

    h_act = captured[0]
    h_grad = h_act.grad
    if h_grad.shape[1] > 1 and h_grad[:, 1:, :].abs().sum().item() == 0.0:
        raise RuntimeError(
            "ViT patch-token gradients are zero at this block (common at the last block with a "
            "CLS-only head). Use a smaller --vit-block, e.g. 8–10 for ViT-B/14."
        )
    # Drop CLS token; reshape patches to spatial grid
    tok = h_act[:, 1:, :]
    gtok = h_grad[:, 1:, :]
    n_patch = tok.shape[1]
    gsize = int(round(n_patch**0.5))
    if gsize * gsize != n_patch:
        raise RuntimeError(f"Non-square patch count: {n_patch}")

    # (1, N, C) -> (1, C, H, W)
    act = tok.reshape(1, gsize, gsize, -1).permute(0, 3, 1, 2).contiguous()
    grad = gtok.reshape(1, gsize, gsize, -1).permute(0, 3, 1, 2).contiguous()

    w = gradcam_plusplus_weights(act, grad)
    cam = cam_from_maps(act, w)
    _, _, oh, ow = x.shape
    cam_np = _upsample_cam(cam, oh, ow)

    rgb = tensor_to_rgb_uint8(x[0])
    overlay = overlay_heatmap_on_rgb(rgb, cam_np)
    return rgb, overlay, float(logit.detach().cpu().item())


def _mil_spatial_layer(backbone: torch.nn.Module) -> torch.nn.Module:
    """Last spatial module before global pool for timm EfficientNet."""
    if hasattr(backbone, "bn2"):
        return backbone.bn2
    if hasattr(backbone, "act2"):
        return backbone.act2
    raise AttributeError("Could not find backbone.bn2 / backbone.act2 for MIL Grad-CAM++.")


def mil_gradcam_pp(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    slice_index: int,
    target_class: int = 0,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Grad-CAM++ on one slice of a scan with full MIL backward.

    Args:
        model: CovidDetector
        x: (1, K, 3, H, W)
        mask: (1, K) 1 = valid
        slice_index: which slice (0 .. K-1) to visualize
        target_class: 0 = COVID, 1 = non-COVID (repo convention)

    Returns:
        rgb_uint8, overlay_uint8, logits (1,2), attention (1, K)
    """
    from src.models.efficientnet import CovidDetector

    if not isinstance(model, CovidDetector):
        raise TypeError("model must be CovidDetector")

    if hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(enable=False)

    layer = _mil_spatial_layer(model.backbone)
    chunks: List[torch.Tensor] = []

    def hook(_m, _inp, out: torch.Tensor) -> None:
        out.retain_grad()
        chunks.append(out)

    handle = layer.register_forward_hook(hook)
    model.eval()
    x = x.to(device)
    mask = mask.to(device)

    logits, attn = model(x, mask)
    model.zero_grad(set_to_none=True)
    score = logits[0, target_class]
    score.backward(retain_graph=False)
    handle.remove()

    if not chunks:
        raise RuntimeError("MIL hook did not capture spatial features.")

    A = torch.cat(chunks, dim=0)
    G = torch.cat([c.grad for c in chunks], dim=0)
    if slice_index < 0 or slice_index >= A.shape[0]:
        raise IndexError(f"slice_index {slice_index} out of range for {A.shape[0]} slice forwards.")

    act = A[slice_index : slice_index + 1]
    grad = G[slice_index : slice_index + 1]
    w = gradcam_plusplus_weights(act, grad)
    cam = cam_from_maps(act, w)

    _, _, _, oh, ow = x.shape
    cam_np = _upsample_cam(cam, oh, ow)
    rgb = tensor_to_rgb_uint8(x[0, slice_index])
    overlay = overlay_heatmap_on_rgb(rgb, cam_np)
    return rgb, overlay, logits.detach(), attn.detach()
