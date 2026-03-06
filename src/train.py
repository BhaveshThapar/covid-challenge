"""
Training script for the Multi-Source Covid-19 Detection Challenge.

Phase 1: Slice-level pretraining of the backbone
Phase 2: End-to-end scan-level training with attention pooling

Improvements over baseline:
  - Label smoothing + optional Focal Loss
  - Backbone freezing in early Phase 2 epochs + differential LR
  - Step-level linear warmup + cosine decay scheduler
  - Embedding-level mixup in Phase 2
  - drop_path_rate for stochastic depth
  - Stochastic Weight Averaging (SWA) for flatter minima
  - Per-source loss weighting for balanced cross-centre performance
"""
import os
import sys
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import SliceClassifier, CovidDetector
from src.dataset import (
    build_slice_dataloaders, build_scan_dataloaders,
)
from src.losses import FocalLoss
from src.utils import (
    set_seed, load_config, get_logger, compute_per_source_f1,
    EarlyStopping, CheckpointManager,
)


# ---------- Scheduler ---------- #

def get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup for `warmup_steps`, then cosine decay to 0."""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---------- Mixup ---------- #

def embedding_mixup(embeddings, labels, alpha=0.2):
    """
    Mixup at the embedding level (Zhang et al., ICLR 2018).
    Returns mixed embeddings and both label sets with mixing coefficient.
    """
    if alpha <= 0:
        return embeddings, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = embeddings.size(0)
    index = torch.randperm(batch_size, device=embeddings.device)

    mixed_embed = lam * embeddings + (1 - lam) * embeddings[index]
    labels_a, labels_b = labels, labels[index]
    return mixed_embed, labels_a, labels_b, lam


# ---------- Loss builder ---------- #

def build_criterion(config, phase="phase1"):
    """Build loss function based on config."""
    loss_type = config.get(phase, {}).get("loss_type", "cross_entropy")
    label_smoothing = config.get(phase, {}).get("label_smoothing", 0.0)

    if loss_type == "focal":
        gamma = config.get(phase, {}).get("focal_gamma", 2.0)
        alpha = config.get(phase, {}).get("focal_alpha", None)
        return FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
    else:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


# ---------- Phase 1 ---------- #

def train_phase1(config, data_dir, metadata_dir, device, logger):
    """Phase 1: Slice-level pretraining."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Slice-Level Pretraining")
    logger.info("=" * 60)

    # Data
    train_loader, val_loader = build_slice_dataloaders(data_dir, metadata_dir, config)
    logger.info(f"Train slices: {len(train_loader.dataset)}, Val slices: {len(val_loader.dataset)}")

    # Model
    model = SliceClassifier(
        backbone_name=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        drop_path_rate=config["model"].get("drop_path_rate", 0.0),
    ).to(device)

    # Enable gradient checkpointing to save GPU memory
    if hasattr(model.backbone, 'set_grad_checkpointing'):
        model.backbone.set_grad_checkpointing(enable=True)
        logger.info("Gradient checkpointing enabled for Phase 1")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["phase1"]["lr"],
        weight_decay=config["phase1"]["weight_decay"],
    )
    epochs = config["phase1"]["epochs"]
    warmup_epochs = config["phase1"]["warmup_epochs"]

    # Step-level warmup + cosine scheduler
    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # Loss
    criterion = build_criterion(config, "phase1")
    logger.info(f"Phase 1 loss: {criterion.__class__.__name__}")

    # Checkpoint
    ckpt_mgr = CheckpointManager(config["checkpoint_dir"])
    writer = SummaryWriter(log_dir=os.path.join(config["log_dir"], "phase1"))
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        # Freeze BN for early epochs
        if epoch <= config["phase1"]["freeze_bn_epochs"]:
            for m in model.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()

        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Phase1 Epoch {epoch}/{epochs}")
        for images, labels, sources in pbar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validate
        model.eval()
        all_preds, all_labels, all_sources = [], [], []
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for images, labels, sources in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                n_val += 1

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_sources.extend(sources.numpy())

        avg_val_loss = val_loss / max(n_val, 1)
        f1_dict = compute_per_source_f1(all_labels, all_preds, all_sources)

        logger.info(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
            f"avg_F1={f1_dict['average']:.4f}, per_source={f1_dict}"
        )
        writer.add_scalar("loss/train", avg_train_loss, epoch)
        writer.add_scalar("loss/val", avg_val_loss, epoch)
        writer.add_scalar("f1/average", f1_dict["average"], epoch)
        for k, v in f1_dict.items():
            if k != "average":
                writer.add_scalar(f"f1/{k}", v, epoch)

        # Save best
        if f1_dict["average"] > best_f1:
            best_f1 = f1_dict["average"]
            ckpt_mgr.save_best(model, optimizer, epoch, best_f1)
            logger.info(f"  → New best F1: {best_f1:.4f}")

    writer.close()
    logger.info(f"Phase 1 complete. Best F1: {best_f1:.4f}")
    return model


# ---------- Phase 2 ---------- #

def train_phase2(config, data_dir, metadata_dir, device, logger, slice_model=None):
    """Phase 2: Scan-level end-to-end training."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Scan-Level End-to-End Training")
    logger.info("=" * 60)

    # Data
    train_loader, val_loader = build_scan_dataloaders(data_dir, metadata_dir, config)
    logger.info(f"Train scans: {len(train_loader.dataset)}, Val scans: {len(val_loader.dataset)}")

    # Model — initialize from Phase 1 if available
    if slice_model is not None:
        model = CovidDetector.from_slice_classifier(slice_model, config).to(device)
        logger.info("Initialized backbone from Phase 1 slice classifier")
    else:
        model = CovidDetector(
            backbone_name=config["model"]["backbone"],
            pretrained=config["model"]["pretrained"],
            embedding_dim=config["model"]["embedding_dim"],
            attention_hidden_dim=config["model"]["attention_hidden_dim"],
            classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
            drop_path_rate=config["model"].get("drop_path_rate", 0.0),
        ).to(device)

    # Phase 2 config
    epochs = config["phase2"]["epochs"]
    warmup_epochs = config["phase2"]["warmup_epochs"]
    grad_accum = config["phase2"]["gradient_accumulation_steps"]
    use_amp = config["phase2"]["use_amp"]
    freeze_backbone_epochs = config["phase2"].get("freeze_backbone_epochs", 0)
    mixup_alpha = config["phase2"].get("mixup_alpha", 0.0)
    backbone_lr_factor = config["phase2"].get("backbone_lr_factor", 0.1)
    swa_start_epoch = config["phase2"].get("swa_start_epoch", 0)
    swa_lr = config["phase2"].get("swa_lr", 1e-5)
    source_loss_weights = config["phase2"].get("source_loss_weights", None)

    # Initially freeze backbone if configured
    backbone_frozen = False
    if freeze_backbone_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        backbone_frozen = True
        logger.info(f"Backbone frozen for first {freeze_backbone_epochs} epochs")

    # Optimizer — differential LR: backbone at lower LR
    def build_optimizer(model, config, differential=True):
        base_lr = config["phase2"]["lr"]
        wd = config["phase2"]["weight_decay"]
        if differential and not backbone_frozen:
            param_groups = [
                {"params": model.backbone.parameters(), "lr": base_lr * backbone_lr_factor},
                {"params": model.attention.parameters(), "lr": base_lr},
                {"params": model.classifier.parameters(), "lr": base_lr},
            ]
            return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)
        else:
            trainable = [p for p in model.parameters() if p.requires_grad]
            return torch.optim.AdamW(trainable, lr=base_lr, weight_decay=wd)

    optimizer = build_optimizer(model, config, differential=False)
    scaler = GradScaler(enabled=use_amp)

    # Scheduler
    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # Loss
    criterion = build_criterion(config, "phase2")
    logger.info(f"Phase 2 loss: {criterion.__class__.__name__}, mixup_alpha={mixup_alpha}")

    ckpt_mgr = CheckpointManager(config["checkpoint_dir"])
    writer = SummaryWriter(log_dir=os.path.join(config["log_dir"], "phase2"))
    patience = config["phase2"].get("early_stopping_patience", 5)
    early_stop = EarlyStopping(patience=patience, mode="max")
    best_f1 = 0.0

    global_step = 0
    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after freeze_backbone_epochs
        if backbone_frozen and epoch == freeze_backbone_epochs + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            backbone_frozen = False
            # Rebuild optimizer with differential LR
            optimizer = build_optimizer(model, config, differential=True)
            scaler = GradScaler(enabled=use_amp)
            # Reset scheduler for remaining epochs
            remaining_steps = (epochs - epoch + 1) * steps_per_epoch
            scheduler = get_cosine_warmup_scheduler(
                optimizer, warmup_steps=steps_per_epoch, total_steps=remaining_steps)
            logger.info(f"Backbone unfrozen at epoch {epoch} with {backbone_lr_factor}x LR")

        model.train()
        train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Phase2 Epoch {epoch}/{epochs}")
        for step, (images, labels, sources, masks) in enumerate(pbar):
            images = images.to(device)     # (B, K, 3, H, W)
            labels = labels.to(device)
            masks = masks.to(device)

            with autocast(enabled=use_amp):
                # Efficient mixup: single backbone pass via forward_features
                if mixup_alpha > 0 and model.training:
                    scan_embed, attn = model.forward_features(images, masks)
                    mixed_embed, labels_a, labels_b, lam = embedding_mixup(
                        scan_embed, labels, mixup_alpha)
                    logits = model.classifier(mixed_embed)
                    loss = lam * criterion(logits, labels_a) + \
                           (1 - lam) * criterion(logits, labels_b)
                else:
                    logits, attn = model(images, masks)
                    loss = criterion(logits, labels)

                # Apply per-source loss weighting if configured
                if source_loss_weights is not None:
                    src_weights = torch.tensor(source_loss_weights, device=device, dtype=torch.float32)
                    # Per-sample weight based on source
                    sample_weights = src_weights[sources.to(device)].mean()
                    loss = loss * sample_weights

                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * grad_accum
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validate
        model.eval()
        all_preds, all_labels, all_sources = [], [], []
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for images, labels, sources, masks in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

                with autocast(enabled=use_amp):
                    logits, attn = model(images, masks)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                n_val += 1

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_sources.extend(sources.numpy())

        avg_val_loss = val_loss / max(n_val, 1)
        f1_dict = compute_per_source_f1(all_labels, all_preds, all_sources)

        logger.info(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
            f"avg_F1={f1_dict['average']:.4f}, per_source={f1_dict}"
        )
        writer.add_scalar("loss/train", avg_train_loss, epoch)
        writer.add_scalar("loss/val", avg_val_loss, epoch)
        writer.add_scalar("f1/average", f1_dict["average"], epoch)
        writer.add_scalar("loss_ratio", avg_val_loss / max(avg_train_loss, 1e-8), epoch)
        for k, v in f1_dict.items():
            if k != "average":
                writer.add_scalar(f"f1/{k}", v, epoch)

        # Save best
        if f1_dict["average"] > best_f1:
            best_f1 = f1_dict["average"]
            ckpt_mgr.save_best(model, optimizer, epoch, best_f1)
            logger.info(f"  → New best F1: {best_f1:.4f}")

        # Early stopping
        if early_stop(f1_dict["average"]):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    writer.close()

    # ---- SWA phase (optional) ----
    if swa_start_epoch > 0:
        logger.info("=" * 40)
        logger.info("SWA: Stochastic Weight Averaging")
        logger.info("=" * 40)

        # Load best checkpoint as starting point for SWA
        best_ckpt_path = os.path.join(config["checkpoint_dir"], "best.pt")
        if os.path.exists(best_ckpt_path):
            CheckpointManager.load(best_ckpt_path, model, device=device)
            logger.info(f"Loaded best checkpoint for SWA base")

        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        swa_epochs = 5  # Run SWA for 5 additional epochs

        for swa_epoch in range(1, swa_epochs + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"SWA Epoch {swa_epoch}/{swa_epochs}")
            for step, (images, labels, sources, masks) in enumerate(pbar):
                images = images.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

                with autocast(enabled=use_amp):
                    logits, attn = model(images, masks)
                    loss = criterion(logits, labels)
                    loss = loss / grad_accum

                scaler.scale(loss).backward()

                if (step + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

            swa_model.update_parameters(model)
            swa_scheduler.step()
            logger.info(f"SWA epoch {swa_epoch} complete")

        # Update BN statistics for SWA model
        logger.info("Updating BatchNorm statistics for SWA model...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

        # Save SWA model
        swa_path = os.path.join(config["checkpoint_dir"], "swa_best.pt")
        torch.save({
            "epoch": -1,
            "model_state_dict": swa_model.module.state_dict(),
            "score": best_f1,
        }, swa_path)
        logger.info(f"Saved SWA model to {swa_path}")

        # Evaluate SWA model
        swa_model.eval()
        all_preds, all_labels_eval, all_sources_eval = [], [], []
        with torch.no_grad():
            for images, labels_batch, sources_batch, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                with autocast(enabled=use_amp):
                    logits, _ = swa_model(images, masks)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_eval.extend(labels_batch.numpy())
                all_sources_eval.extend(sources_batch.numpy())

        f1_dict = compute_per_source_f1(all_labels_eval, all_preds, all_sources_eval)
        logger.info(f"SWA: avg_F1={f1_dict['average']:.4f}, per_source={f1_dict}")

        if f1_dict["average"] > best_f1:
            # Overwrite best.pt with SWA model
            torch.save({
                "epoch": -1,
                "model_state_dict": swa_model.module.state_dict(),
                "score": f1_dict["average"],
            }, best_ckpt_path)
            logger.info(f"SWA model is better! Saved as best.pt (F1={f1_dict['average']:.4f})")
            best_f1 = f1_dict["average"]
        else:
            logger.info(f"SWA model not better than best ({best_f1:.4f}), keeping original")

    logger.info(f"Phase 2 complete. Best F1: {best_f1:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Covid-19 Detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--phase", type=int, choices=[1, 2, 0], default=0,
                        help="Phase to run: 1=slice, 2=scan, 0=both")
    parser.add_argument("--resume-phase1", type=str, default=None,
                        help="Path to Phase 1 checkpoint to resume from")
    parser.add_argument("--overfit-batches", type=int, default=0,
                        help="If > 0, overfit on this many batches (debug)")
    args = parser.parse_args()

    config = load_config(args.config)
    config["checkpoint_dir"] = config.get("checkpoint_dir", "checkpoints")
    config["log_dir"] = config.get("log_dir", "logs")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("train", os.path.join(config["log_dir"], "train.log"))
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {gpu_mem:.1f} GB")
    logger.info(f"Config: {config}")

    slice_model = None

    if args.phase in (0, 1):
        slice_model = train_phase1(config, args.data_dir, args.metadata_dir, device, logger)

    if args.phase in (0, 2):
        # Load Phase 1 model if resuming
        if args.resume_phase1 and slice_model is None:
            slice_model = SliceClassifier(
                backbone_name=config["model"]["backbone"],
                pretrained=False,
                num_classes=config["model"]["num_classes"],
                dropout=config["model"]["dropout"],
                drop_path_rate=config["model"].get("drop_path_rate", 0.0),
            )
            CheckpointManager.load(args.resume_phase1, slice_model, device=device)
            logger.info(f"Loaded Phase 1 checkpoint: {args.resume_phase1}")

        # Free Phase 1 model memory before Phase 2
        if slice_model is not None and device.type == 'cuda':
            logger.info("Clearing GPU memory before Phase 2...")
            torch.cuda.empty_cache()

        train_phase2(config, args.data_dir, args.metadata_dir, device, logger, slice_model)


if __name__ == "__main__":
    main()
