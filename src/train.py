"""
Training script for the Multi-Source Covid-19 Detection Challenge.

Phase 1: Slice-level pretraining of the backbone
Phase 2: End-to-end scan-level training with attention pooling
"""
import os
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import SliceClassifier, CovidDetector
from src.dataset import (
    build_slice_dataloaders, build_scan_dataloaders,
)
from src.utils import (
    set_seed, load_config, get_logger, compute_per_source_f1,
    EarlyStopping, CheckpointManager,
)


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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

    # Loss
    criterion = nn.CrossEntropyLoss()

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

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if epoch > warmup_epochs:
            scheduler.step()

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
        ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["phase2"]["lr"],
        weight_decay=config["phase2"]["weight_decay"],
    )
    epochs = config["phase2"]["epochs"]
    warmup_epochs = config["phase2"]["warmup_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    grad_accum = config["phase2"]["gradient_accumulation_steps"]
    use_amp = config["phase2"]["use_amp"]
    scaler = GradScaler(enabled=use_amp)

    criterion = nn.CrossEntropyLoss()
    ckpt_mgr = CheckpointManager(config["checkpoint_dir"])
    writer = SummaryWriter(log_dir=os.path.join(config["log_dir"], "phase2"))
    early_stop = EarlyStopping(patience=5, mode="max")
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
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
                logits, attn = model(images, masks)
                loss = criterion(logits, labels)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        if epoch > warmup_epochs:
            scheduler.step()

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
