"""
Training script for the Multi-Source Covid-19 Detection Challenge.

Phase 1: Frozen DenseNet-121 backbone, head-only training (slice-level)
Phase 2: Gradual backbone unfreezing — sub-phase 2a (denseblock4) then 2b (denseblock3)

Both phases train at the slice level; validation is at the scan level via evaluate_scans().
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, CosineAnnealingWarmRestarts
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DenseNetCovidClassifier
from src.dataset import (
    build_slice_dataloaders, build_scan_manifest,
    ScanDataset, get_val_transforms, scan_collate_fn,
)
from src.utils import (
    set_seed, load_config, get_logger, compute_per_source_f1,
    EarlyStopping, CheckpointManager,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               gamma: float = 2.0, pos_weight: torch.Tensor = None) -> torch.Tensor:
    """Focal loss for binary classification."""
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    pt = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


def build_criterion(config: dict, phase_key: str, pos_weight: torch.Tensor, device):
    """Return the appropriate loss function based on config."""
    pw = pos_weight.to(device)
    if config[phase_key].get("use_focal_loss", False):
        return lambda logits, targets: focal_loss(logits, targets, gamma=2.0, pos_weight=pw)
    return lambda logits, targets: F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pw
    )


# ---------------------------------------------------------------------------
# Scan-level validation helper
# ---------------------------------------------------------------------------

def evaluate_scans(
    model: DenseNetCovidClassifier,
    val_entries: list,
    config: dict,
    device,
    threshold: float = 0.5,
    slices_per_scan: int = None,
) -> dict:
    """
    Scan-level evaluation: aggregate per-slice sigmoid probabilities by averaging,
    then apply threshold to obtain a binary prediction per scan.

    Args:
        slices_per_scan: Override slices to sample per scan.
                         None → uses config['eval']['val_slices_per_scan'].

    Returns:
        dict from compute_per_source_f1: {'source_0': f1, ..., 'average': f1}
    """
    k = slices_per_scan if slices_per_scan is not None else config["eval"]["val_slices_per_scan"]
    img_size = config["data"]["image_size"]

    val_ds = ScanDataset(val_entries, get_val_transforms(img_size), slices_per_scan=k)
    val_loader = DataLoader(
        val_ds,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        collate_fn=scan_collate_fn,
    )

    model.eval()
    all_preds, all_labels, all_sources = [], [], []

    with torch.no_grad():
        for images, labels, sources, masks in val_loader:
            B, K, C, H, W = images.shape
            x_flat = images.view(B * K, C, H, W).to(device)

            logits = model(x_flat).squeeze(-1)          # (B*K,)
            probs = torch.sigmoid(logits).view(B, K)    # (B, K)

            valid = masks.float().to(device)            # (B, K)
            scan_probs = (probs * valid).sum(1) / valid.sum(1).clamp(min=1)  # (B,)
            preds = (scan_probs >= threshold).long().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_sources.extend(sources.numpy())

    return compute_per_source_f1(all_labels, all_preds, all_sources, strict_labels=False)


# ---------------------------------------------------------------------------
# Shared training loop (used by both phases)
# ---------------------------------------------------------------------------

def _run_epoch(
    model: DenseNetCovidClassifier,
    train_loader,
    criterion,
    optimizer,
    scaler: GradScaler,
    use_amp: bool,
    label_smooth: float,
    device,
    desc: str,
) -> float:
    """One training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=desc)
    for images, labels, sources in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Label smoothing: 0 → ε/2, 1 → 1 − ε/2
        smooth_labels = labels.float() * (1 - label_smooth) + 0.5 * label_smooth

        with autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images).squeeze(-1)          # (B,)
            loss = criterion(logits, smooth_labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


def _log_f1(f1_dict: dict, writer: SummaryWriter, epoch: int, logger, prefix: str = ""):
    """Log per-center and aggregate F1 to TensorBoard and text logger."""
    for k, v in sorted(f1_dict.items()):
        tag = f"f1/{prefix}{k}" if prefix else f"f1/{k}"
        writer.add_scalar(tag, v, epoch)
        logger.info(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# Phase 1: Frozen backbone, head-only
# ---------------------------------------------------------------------------

def train_phase1(config: dict, data_dir: str, metadata_dir: str, device, logger) -> DenseNetCovidClassifier:
    logger.info("=" * 60)
    logger.info("PHASE 1: Frozen Backbone — Head-Only Fine-Tuning")
    logger.info("=" * 60)

    train_loader, val_entries = build_slice_dataloaders(data_dir, metadata_dir, config)
    logger.info(f"Train slices: {len(train_loader.dataset)}, Val scans: {len(val_entries)}")

    # Model — load RadImageNet weights, freeze backbone
    pretrained_path = config["model"].get("pretrained_path", "")
    model = DenseNetCovidClassifier(
        pretrained_path=pretrained_path,
        dropout=config["model"]["dropout"],
    ).to(device)
    model.freeze_backbone()
    logger.info(f"Trainable parameters: {model.trainable_param_count():,}")

    # Compute pos_weight from training labels
    labels_list = [s[1] for s in train_loader.dataset.samples]
    n_covid = labels_list.count(0)
    n_noncovid = labels_list.count(1)
    pos_weight = torch.tensor([n_noncovid / max(n_covid, 1)])
    logger.info(f"Class counts — covid: {n_covid}, non-covid: {n_noncovid}, pos_weight: {pos_weight.item():.3f}")

    # Optimizer + real linear warmup → cosine decay
    p1 = config["phase1"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=p1["lr"], weight_decay=p1["weight_decay"],
    )
    epochs = p1["epochs"]
    warmup_e = p1["warmup_epochs"]
    linear_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_e)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_e, 1))
    scheduler = SequentialLR(optimizer, schedulers=[linear_sched, cosine_sched],
                              milestones=[warmup_e])

    criterion = build_criterion(config, "phase1", pos_weight, device)
    label_smooth = p1.get("label_smoothing", 0.05)
    use_amp = False  # head-only: fast enough without AMP
    scaler = GradScaler("cuda", enabled=use_amp)

    ckpt_mgr = CheckpointManager(config["checkpoint_dir"])
    writer = SummaryWriter(log_dir=os.path.join(config["log_dir"], "phase1"))
    early_stop = EarlyStopping(patience=p1["early_stop_patience"], mode="max")
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        avg_loss = _run_epoch(
            model, train_loader, criterion, optimizer, scaler, use_amp,
            label_smooth, device, desc=f"P1 Epoch {epoch}/{epochs}",
        )
        scheduler.step()

        f1_dict = evaluate_scans(model, val_entries, config, device)
        logger.info(f"Epoch {epoch}: train_loss={avg_loss:.4f}, per-center F1:")
        _log_f1(f1_dict, writer, epoch, logger)
        writer.add_scalar("loss/train_phase1", avg_loss, epoch)

        if f1_dict["average"] > best_f1:
            best_f1 = f1_dict["average"]
            p1_ckpt = f"{config['run_name']}_phase1_best.pt"
            ckpt_mgr.save_named(model, optimizer, epoch, best_f1, p1_ckpt)
            logger.info(f"  → New best F1: {best_f1:.4f} (saved {p1_ckpt})")

        if early_stop(f1_dict["average"]):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    writer.close()
    logger.info(f"Phase 1 complete. Best F1: {best_f1:.4f}")

    # Reload best weights before returning
    CheckpointManager.load(
        os.path.join(config["checkpoint_dir"], f"{config['run_name']}_phase1_best.pt"),
        model, device=device
    )
    return model


# ---------------------------------------------------------------------------
# Phase 2 sub-phase runner
# ---------------------------------------------------------------------------

def _run_subphase(
    model: DenseNetCovidClassifier,
    train_loader,
    val_entries: list,
    config: dict,
    device,
    logger,
    writer: SummaryWriter,
    ckpt_mgr: CheckpointManager,
    phase_key: str,           # 'phase2'
    head_lr: float,
    block_lrs: dict,
    n_epochs: int,
    save_name: str,
    epoch_offset: int = 0,
) -> tuple:
    """Run one unfreezing sub-phase. Returns (best_f1, final_epoch)."""
    p2 = config[phase_key]
    use_amp = p2.get("use_amp", True)

    param_groups = model.get_parameter_groups(head_lr, block_lrs)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=p2["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=p2["warmup_restarts_T0"])
    scaler = GradScaler("cuda", enabled=use_amp)

    pos_weight = torch.tensor([
        sum(1 for s in train_loader.dataset.samples if s[1] == 1) /
        max(sum(1 for s in train_loader.dataset.samples if s[1] == 0), 1)
    ])
    criterion = build_criterion(config, phase_key, pos_weight, device)
    label_smooth = p2.get("label_smoothing", 0.05)
    full_val_every = p2.get("full_val_every_n_epochs", 5)
    early_stop = EarlyStopping(patience=p2["early_stop_patience"], mode="max")
    best_f1 = 0.0

    logger.info(f"Trainable parameters: {model.trainable_param_count():,}")

    for epoch in range(1, n_epochs + 1):
        global_epoch = epoch_offset + epoch

        avg_loss = _run_epoch(
            model, train_loader, criterion, optimizer, scaler, use_amp,
            label_smooth, device, desc=f"{save_name} Epoch {epoch}/{n_epochs}",
        )
        scheduler.step()
        writer.add_scalar(f"loss/train_{save_name}", avg_loss, global_epoch)

        # Fast validation every epoch
        f1_dict = evaluate_scans(model, val_entries, config, device)
        logger.info(f"Epoch {epoch}: train_loss={avg_loss:.4f}, fast-val F1:")
        _log_f1(f1_dict, writer, global_epoch, logger)

        # Full-slice validation every N epochs
        if epoch % full_val_every == 0:
            full_f1 = evaluate_scans(model, val_entries, config, device, slices_per_scan=-1)
            logger.info(f"  Full-slice val F1: {full_f1['average']:.4f}")
            _log_f1(full_f1, writer, global_epoch, logger, prefix="fullslice_")

        if f1_dict["average"] > best_f1:
            best_f1 = f1_dict["average"]
            ckpt_mgr.save_named(model, optimizer, epoch, best_f1, f"{save_name}_best.pt")
            logger.info(f"  → New best F1: {best_f1:.4f} (saved {save_name}_best.pt)")

        if early_stop(f1_dict["average"]):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    return best_f1, epoch_offset + n_epochs


# ---------------------------------------------------------------------------
# Phase 2: Gradual unfreezing
# ---------------------------------------------------------------------------

def train_phase2(
    config: dict, data_dir: str, metadata_dir: str, device, logger,
    phase1_model: DenseNetCovidClassifier = None,
) -> DenseNetCovidClassifier:
    logger.info("=" * 60)
    logger.info("PHASE 2: Gradual Backbone Unfreezing")
    logger.info("=" * 60)

    train_loader, val_entries = build_slice_dataloaders(data_dir, metadata_dir, config)

    # Load Phase 1 checkpoint if model not passed in
    if phase1_model is None:
        phase1_path = os.path.join(config["checkpoint_dir"], f"{config['run_name']}_phase1_best.pt")
        model = DenseNetCovidClassifier(dropout=config["model"]["dropout"]).to(device)
        CheckpointManager.load(phase1_path, model, device=device)
        logger.info(f"Loaded Phase 1 checkpoint from {phase1_path}")
    else:
        model = phase1_model

    p2 = config["phase2"]
    ckpt_mgr = CheckpointManager(config["checkpoint_dir"])
    writer = SummaryWriter(log_dir=os.path.join(config["log_dir"], "phase2"))
    overall_best_f1 = 0.0

    # ---- Sub-phase 2a: unfreeze denseblock4 + norm5 ----
    logger.info("Sub-phase 2a: Unfreezing denseblock4 + norm5")
    model.unfreeze_block("denseblock4", "norm5")

    rn = config["run_name"]

    f1_2a, ep_offset = _run_subphase(
        model=model,
        train_loader=train_loader,
        val_entries=val_entries,
        config=config,
        device=device,
        logger=logger,
        writer=writer,
        ckpt_mgr=ckpt_mgr,
        phase_key="phase2",
        head_lr=p2["head_lr"],
        block_lrs={"denseblock4": p2["block4_lr"], "norm5": p2["block4_lr"]},
        n_epochs=p2["block4_epochs"],
        save_name=f"{rn}_phase2a",
        epoch_offset=0,
    )
    overall_best_f1 = max(overall_best_f1, f1_2a)

    # ---- Sub-phase 2b: additionally unfreeze denseblock3 + transition3 ----
    logger.info("Sub-phase 2b: Unfreezing denseblock3 + transition3")

    # Reload best from 2a to start 2b from a clean state
    CheckpointManager.load(
        os.path.join(config["checkpoint_dir"], f"{rn}_phase2a_best.pt"), model, device=device
    )
    model.unfreeze_block("denseblock3", "transition3")

    f1_2b, _ = _run_subphase(
        model=model,
        train_loader=train_loader,
        val_entries=val_entries,
        config=config,
        device=device,
        logger=logger,
        writer=writer,
        ckpt_mgr=ckpt_mgr,
        phase_key="phase2",
        head_lr=p2["head_lr"],
        block_lrs={
            "denseblock4": p2["block4_lr"], "norm5": p2["block4_lr"],
            "denseblock3": p2["block3_lr"], "transition3": p2["block3_lr"],
        },
        n_epochs=p2["block3_epochs"],
        save_name=f"{rn}_phase2b",
        epoch_offset=ep_offset,
    )
    overall_best_f1 = max(overall_best_f1, f1_2b)

    # Copy the globally best checkpoint to {run_name}_ovr_best.pt
    best_sub = f"{rn}_phase2b" if f1_2b >= f1_2a else f"{rn}_phase2a"
    best_src = os.path.join(config["checkpoint_dir"], f"{best_sub}_best.pt")
    best_dst = os.path.join(config["checkpoint_dir"], f"{rn}_ovr_best.pt")
    import shutil
    shutil.copy2(best_src, best_dst)
    logger.info(f"Best overall F1: {overall_best_f1:.4f} (from {best_sub}) → saved as {rn}_ovr_best.pt")

    writer.close()
    CheckpointManager.load(best_dst, model, device=device)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DenseNet-121 Covid-19 Detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--phase", type=int, choices=[1, 2, 0], default=0,
                        help="Phase to run: 1=head-only, 2=gradual unfreeze, 0=both")
    parser.add_argument("--run-name", type=str, default="run",
                        help="Prefix for checkpoint filenames, e.g. 'v1' → v1_phase1_best.pt, v1_ovr_best.pt")
    parser.add_argument("--radimagenet-weights", type=str, default=None,
                        help="Path to RadImageNet DenseNet-121 checkpoint (overrides config)")
    parser.add_argument("--overfit-batches", type=int, default=0,
                        help="If > 0, overfit on this many batches (debug mode)")
    args = parser.parse_args()

    config = load_config(args.config)
    config["checkpoint_dir"] = config.get("checkpoint_dir", "checkpoints")
    config["log_dir"] = config.get("log_dir", "logs")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    config["run_name"] = args.run_name

    # CLI override for RadImageNet weights path
    if args.radimagenet_weights:
        config["model"]["pretrained_path"] = args.radimagenet_weights

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("train", os.path.join(config["log_dir"], "train.log"))
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")

    phase1_model = None

    if args.phase in (0, 1):
        phase1_model = train_phase1(config, args.data_dir, args.metadata_dir, device, logger)

    if args.phase in (0, 2):
        train_phase2(config, args.data_dir, args.metadata_dir, device, logger, phase1_model)


if __name__ == "__main__":
    main()
