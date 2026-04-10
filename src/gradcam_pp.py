"""
Grad-CAM++ saliency for EfficientNet-B3 attention MIL (CovidDetector).

Hooks timm EfficientNet spatial features at ``bn2`` (BatchNormAct2d), after
chunked slice forwards, so the CAM is for one slice in the context of the
full scan backward pass.

Only efficientnet_b3 and tf_efficientnetv2_s backbones are supported
(both have bn2/act2). convnext_tiny uses LayerNorm and will raise an error.

References:
  Selvaraju et al., Grad-CAM
  Chattopadhay et al., Grad-CAM++ (weights implemented as in common PyTorch ports)
"""
from __future__ import annotations

import os
from typing import List, Tuple

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
    grad_2 = grads.pow(2)
    grad_3 = grad_2 * grads
    sum_activations = activations.sum(dim=(2, 3), keepdim=True)
    denom = 2.0 * grad_2 + sum_activations * grad_3 + eps
    alpha = grad_2 / denom
    weights = (alpha * F.relu(grads)).sum(dim=(2, 3)).squeeze(0)
    return weights


def cam_from_maps(activations: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted sum + ReLU -> (H, W) CAM."""
    cam = (weights.view(1, -1, 1, 1) * activations).sum(dim=1).squeeze(0)
    cam = F.relu(cam)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def _upsample_cam(cam_hw: torch.Tensor, out_h: int, out_w: int) -> np.ndarray:
    """cam_hw: (H, W) -> numpy (out_h, out_w) float32 [0, 1]."""
    t = cam_hw.unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return up.squeeze().detach().cpu().numpy().astype(np.float32)


def tensor_to_rgb_uint8(tchw: torch.Tensor) -> np.ndarray:
    """Denormalize (3, H, W) RadImageNet-normalized tensor to uint8 RGB."""
    mean = torch.tensor(RADIMAGENET_MEAN, device=tchw.device, dtype=tchw.dtype).view(3, 1, 1)
    std = torch.tensor(RADIMAGENET_STD, device=tchw.device, dtype=tchw.dtype).view(3, 1, 1)
    x = tchw.detach() * std + mean
    x = x.cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    return (x * 255.0).astype(np.uint8)


def overlay_heatmap_on_rgb(
    rgb: np.ndarray, cam_01: np.ndarray, alpha: float = 0.45, colormap: str = "jet",
) -> np.ndarray:
    """Overlay a [0,1] CAM on an HxWx3 uint8 image. Returns uint8 HxWx3."""
    import matplotlib

    if hasattr(matplotlib, "colormaps"):
        cmap = matplotlib.colormaps[colormap]
    else:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
    heat = (cmap(cam_01)[:, :, :3] * 255.0).astype(np.float32)
    out = (1.0 - alpha) * rgb.astype(np.float32) + alpha * heat
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


def _mil_spatial_layer(backbone: torch.nn.Module, block_index: int = -5) -> torch.nn.Module:
    """
    Return the EfficientNet-B3 block to hook for Grad-CAM++.

    blocks[-5] = blocks[2] at 300x300 input → ~38x38 spatial resolution.
    This is 3x sharper than bn2 (10x10) with still-meaningful semantics.

    EfficientNet-B3 spatial sizes at 300x300 input:
        blocks[0]: 150x150  blocks[1]: 75x75   blocks[2]: 38x38
        blocks[3]: 19x19    blocks[4]: 19x19   blocks[5]: 19x19
        blocks[6]: 10x10    bn2:       10x10
    """
    if not hasattr(backbone, "blocks"):
        raise AttributeError(
            "backbone has no 'blocks' attribute. "
            "Only efficientnet_b3 is supported (convnext_tiny is not compatible)."
        )
    return backbone.blocks[block_index]


def mil_gradcam_pp(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    slice_index: int,
    target_class: int = 0,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Grad-CAM++ on one slice of a scan with full MIL backward pass.

    Args:
        model: CovidDetector (efficientnet_b3 or tf_efficientnetv2_s backbone only)
        x: (1, K, 3, H, W)
        mask: (1, K) — 1 = valid slice
        slice_index: which slice (0 .. K-1) to visualize
        target_class: 0 = COVID, 1 = non-COVID (repo convention)
        device: torch device

    Returns:
        rgb_uint8:    (H, W, 3) denormalized input slice
        overlay_uint8:(H, W, 3) heatmap overlaid on rgb
        logits:       (1, 2) detached
        attention:    (1, K) detached
        cam_np:       (H, W) float32 [0, 1] raw CAM (for aggregation)
    """
    from src.models.efficientnet import CovidDetector

    if not isinstance(model, CovidDetector):
        raise TypeError("model must be CovidDetector (EfficientNet MIL)")

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
        raise RuntimeError("MIL hook did not capture any spatial features.")

    A = torch.cat(chunks, dim=0)   # (K, C, H, W)
    G = torch.cat([c.grad for c in chunks], dim=0)  # (K, C, H, W)

    if slice_index < 0 or slice_index >= A.shape[0]:
        raise IndexError(
            f"slice_index {slice_index} out of range for {A.shape[0]} slice forwards."
        )

    act = A[slice_index: slice_index + 1]   # (1, C, H, W)
    grad = G[slice_index: slice_index + 1]  # (1, C, H, W)
    w = gradcam_plusplus_weights(act, grad)
    cam = cam_from_maps(act, w)

    _, _, _, oh, ow = x.shape
    cam_np = _upsample_cam(cam, oh, ow)
    rgb = tensor_to_rgb_uint8(x[0, slice_index])
    overlay = overlay_heatmap_on_rgb(rgb, cam_np)
    return rgb, overlay, logits.detach(), attn.detach(), cam_np


def save_individual_saliency(
    rgb: np.ndarray,
    overlay: np.ndarray,
    sal_np: np.ndarray,
    out_path: str,
    scan_name: str,
    prob_covid: float,
    true_class: str,
    attn_weight: float,
) -> None:
    """
    Two-panel figure per scan: CT slice | saliency overlay with colorbar + contour.

    Contour drawn at the 75th percentile of the saliency map to clearly
    mark the regions contributing most to the model's prediction.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: original CT
    axes[0].imshow(rgb, cmap="gray")
    axes[0].set_title("CT slice (max attention)", fontsize=11)
    axes[0].axis("off")

    # Right: saliency overlay
    axes[1].imshow(overlay)

    # Contour at 75th percentile to mark top contributing regions
    threshold = np.percentile(sal_np, 75)
    axes[1].contour(sal_np, levels=[threshold], colors="white", linewidths=1.2, alpha=0.85)

    axes[1].set_title("Input × Gradient saliency", fontsize=11)
    axes[1].axis("off")

    # Colorbar
    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap="jet", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Saliency", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])

    title = (
        f"{scan_name}  |  true: {true_class}  |  "
        f"P(covid)={prob_covid:.3f}  |  attn={attn_weight:.3f}"
    )
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def mil_input_gradient(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    slice_index: int,
    target_class: int = 0,
    device: torch.device,
    n_smooth: int = 20,
    noise_level: float = 0.15,
    blur_sigma: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    SmoothGrad saliency at full input resolution (H, W) — no upsampling.

    Adds Gaussian noise to the input n_smooth times, averages the squared
    gradients across all runs (SmoothGrad), then applies a Gaussian blur.
    This removes the salt-and-pepper noise that vanilla input×gradient produces
    and yields coherent region-level saliency maps.

    Args:
        model:       CovidDetector (EfficientNet-B3 MIL)
        x:           (1, K, 3, H, W)
        mask:        (1, K) — 1 = valid slice
        slice_index: which slice to visualize
        target_class:0 = COVID, 1 = non-COVID
        device:      torch device
        n_smooth:    number of noisy samples to average (default 20)
        noise_level: noise std as fraction of input range (default 0.15)
        blur_sigma:  Gaussian blur sigma in pixels applied after averaging (default 6)

    Returns:
        rgb_uint8:    (H, W, 3) denormalized input slice
        overlay_uint8:(H, W, 3) saliency overlaid on rgb
        logits:       (1, 2) detached — from the clean (noise-free) forward pass
        attention:    (1, K) detached — from the clean forward pass
        sal_np:       (H, W) float32 [0, 1] smoothed saliency
    """
    from scipy.ndimage import gaussian_filter
    from src.models.efficientnet import CovidDetector

    if not isinstance(model, CovidDetector):
        raise TypeError("model must be CovidDetector (EfficientNet MIL)")

    if hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(enable=False)

    model.eval()
    x = x.to(device)
    mask = mask.to(device)

    # Clean forward pass — used for logits, attn, and noise scale
    with torch.no_grad():
        logits, attn = model(x, mask)

    x_range = x.max() - x.min()
    stdev = float(noise_level * x_range.item())

    # SmoothGrad: accumulate squared gradients over noisy samples
    sal_sum = np.zeros(x.shape[3:], dtype=np.float64)  # (H, W)

    for _ in range(n_smooth):
        x_noisy = (x + torch.randn_like(x) * stdev).detach().requires_grad_(True)
        lg, _ = model(x_noisy, mask)
        model.zero_grad(set_to_none=True)
        lg[0, target_class].backward()

        # Squared gradient, max over channels → (H, W)
        grad = x_noisy.grad[0, slice_index]          # (3, H, W)
        sal  = grad.pow(2).max(dim=0)[0]
        sal_sum += sal.detach().cpu().numpy().astype(np.float64)

    sal_np = (sal_sum / n_smooth).astype(np.float32)

    # Gaussian blur to remove remaining high-frequency noise
    sal_np = gaussian_filter(sal_np, sigma=blur_sigma).astype(np.float32)

    # Normalize to [0, 1]
    s_min, s_max = sal_np.min(), sal_np.max()
    if s_max > s_min:
        sal_np = (sal_np - s_min) / (s_max - s_min + 1e-8)

    inp = x[0, slice_index].detach()
    rgb = tensor_to_rgb_uint8(inp)
    overlay = overlay_heatmap_on_rgb(rgb, sal_np)
    return rgb, overlay, logits.detach(), attn.detach(), sal_np
