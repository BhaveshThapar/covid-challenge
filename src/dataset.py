"""
Dataset and DataLoader classes for the Multi-Source Covid-19 Detection Challenge.

Two modes:
  1. SliceDataset  — returns individual slices (for Phase 1 pretraining)
  2. ScanDataset   — returns K slices per scan (for Phase 2 scan-level training)
"""
import os
import random
from pathlib import Path
from typing import Optional

import cv2

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------- Transforms ---------- #

def get_train_transforms(image_size: int = 224):
    return A.Compose([
        A.Resize(image_size, image_size),

        # Geometric — simulate patient positioning variance across sites
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                 scale=(0.85, 1.15), rotate=(-30, 30), p=0.7,
                 border_mode=cv2.BORDER_CONSTANT),
        A.ElasticTransform(alpha=50, sigma=5, p=0.2),

        # Intensity — simulate scanner/protocol variance (critical for cross-source)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # Occlusion — act as strong regularizer
        A.CoarseDropout(num_holes_range=(2, 8),
                        hole_height_range=(image_size // 16, image_size // 8),
                        hole_width_range=(image_size // 16, image_size // 8),
                        fill=0, p=0.4),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ---------- Helpers ---------- #

def _load_image(path: str) -> np.ndarray:
    """Load a JPEG slice and convert to RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _get_sorted_slices(scan_dir: str) -> list:
    """Get sorted list of JPEG slice paths in a scan directory."""
    exts = {".jpg", ".jpeg", ".png"}
    slices = []
    for f in os.listdir(scan_dir):
        stem, ext = os.path.splitext(f)
        # Only include files with numeric names and valid image extensions
        # This skips Mac metadata files (._0, ._1, etc.) and other junk
        if ext.lower() not in exts:
            continue
        if not stem.isdigit():
            continue
        slices.append(os.path.join(scan_dir, f))
    # Sort numerically by filename stem
    slices.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return slices


def build_scan_manifest(data_dir: str, split: str, metadata_dir: str = None):
    """
    Build a list of scan entries.
    
    Returns:
        list of dicts: [{scan_dir, label, source, scan_name}, ...]
        label: 0 = covid, 1 = non-covid
    """
    split_dir = os.path.join(data_dir, split)
    entries = []

    # Load source metadata if available
    source_map = {}
    if metadata_dir:
        for label_name, label_id in [("covid", 0), ("non_covid", 1)]:
            csv_name = f"{split}_{label_name}.csv"
            csv_path = os.path.join(metadata_dir, csv_name)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    source_map[(label_name, row["ct_scan_name"])] = int(row["data_centre"])

    skipped = 0
    for label_name, label_id in [("covid", 0), ("non_covid", 1)]:
        class_dir = os.path.join(split_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        for scan_name in sorted(os.listdir(class_dir)):
            scan_dir = os.path.join(class_dir, scan_name)
            if not os.path.isdir(scan_dir):
                continue
            # Skip scans with no valid slices (e.g. only Mac metadata files)
            if len(_get_sorted_slices(scan_dir)) == 0:
                skipped += 1
                continue
            source = source_map.get((label_name, scan_name), -1)
            entries.append({
                "scan_dir": scan_dir,
                "label": label_id,
                "source": source,
                "scan_name": scan_name,
            })

    if skipped > 0:
        print(f"  [WARNING] Skipped {skipped} empty scan directories in {split}")
    return entries


# ---------- Slice-Level Dataset (Phase 1) ---------- #

class SliceDataset(Dataset):
    """
    Returns individual slices with their scan-level label.
    Used for Phase 1 slice-level pretraining.
    """

    def __init__(self, scan_entries: list, transform=None, max_slices_per_scan: int = -1):
        """
        Args:
            scan_entries: list from build_scan_manifest
            transform: albumentations transform
            max_slices_per_scan: limit slices per scan (-1 = all)
        """
        self.transform = transform
        self.samples = []  # (slice_path, label, source)

        for entry in scan_entries:
            slices = _get_sorted_slices(entry["scan_dir"])
            if max_slices_per_scan > 0:
                # Sample uniformly
                indices = np.linspace(0, len(slices) - 1, min(max_slices_per_scan, len(slices)), dtype=int)
                slices = [slices[i] for i in indices]
            for s in slices:
                self.samples.append((s, entry["label"], entry["source"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, source = self.samples[idx]
        img = _load_image(path)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label, source


# ---------- Scan-Level Dataset (Phase 2) ---------- #

class ScanDataset(Dataset):
    """
    Returns K slices per scan for scan-level classification.
    Used for Phase 2 end-to-end training with attention pooling.
    """

    def __init__(self, scan_entries: list, transform=None, slices_per_scan: int = 32,
                 sampling_strategy: str = "uniform"):
        """
        Args:
            scan_entries: list from build_scan_manifest
            transform: albumentations transform
            slices_per_scan: number of slices to sample per scan (-1 = all)
            sampling_strategy: 'random' for training (different slices each epoch),
                             'uniform' for eval (deterministic linspace)
        """
        self.entries = scan_entries
        self.transform = transform
        self.slices_per_scan = slices_per_scan
        self.sampling_strategy = sampling_strategy

        # Pre-compute slice lists
        self.scan_slices = []
        for entry in self.entries:
            self.scan_slices.append(_get_sorted_slices(entry["scan_dir"]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        all_slices = self.scan_slices[idx]
        k = self.slices_per_scan

        # Sample k slices
        if k > 0 and k < len(all_slices):
            if self.sampling_strategy == "random":
                # Random sampling without replacement — different view each epoch
                indices = sorted(random.sample(range(len(all_slices)), k))
            else:
                # Uniform deterministic (for eval)
                indices = np.linspace(0, len(all_slices) - 1, k, dtype=int)
            selected = [all_slices[i] for i in indices]
        else:
            selected = all_slices

        # Load and transform
        images = []
        for path in selected:
            try:
                img = _load_image(path)
                if self.transform:
                    img = self.transform(image=img)["image"]
                images.append(img)
            except Exception:
                continue  # skip corrupt/unreadable slices

        # Guard: if no images loaded, return a dummy black tensor
        if len(images) == 0:
            img_size = 224
            images = [torch.zeros(3, img_size, img_size)]

        images = torch.stack(images)  # (K, 3, H, W)
        label = entry["label"]
        source = entry["source"]
        return images, label, source


def scan_collate_fn(batch):
    """
    Custom collate for ScanDataset since scans can have different numbers of slices.
    Pads to the max number of slices in the batch.
    """
    images_list, labels, sources = zip(*batch)
    max_slices = max(img.shape[0] for img in images_list)

    padded = []
    masks = []
    for img in images_list:
        n = img.shape[0]
        if n < max_slices:
            pad = torch.zeros(max_slices - n, *img.shape[1:])
            img = torch.cat([img, pad], dim=0)
            mask = torch.cat([torch.ones(n), torch.zeros(max_slices - n)])
        else:
            mask = torch.ones(n)
        padded.append(img)
        masks.append(mask)

    images = torch.stack(padded)   # (B, K, 3, H, W)
    masks = torch.stack(masks)     # (B, K)
    labels = torch.tensor(labels, dtype=torch.long)
    sources = torch.tensor(sources, dtype=torch.long)
    return images, labels, sources, masks


# ---------- DataLoader Builders ---------- #

def build_slice_dataloaders(data_dir, metadata_dir, config):
    """Build train & val slice-level dataloaders for Phase 1."""
    train_entries = build_scan_manifest(data_dir, "train", metadata_dir)
    val_entries = build_scan_manifest(data_dir, "val", metadata_dir)

    img_size = config["data"]["image_size"]
    max_val_slices = config["phase1"].get("max_val_slices_per_scan", 64)

    train_ds = SliceDataset(train_entries, get_train_transforms(img_size), max_slices_per_scan=64)
    val_ds = SliceDataset(val_entries, get_val_transforms(img_size), max_slices_per_scan=max_val_slices)

    # Class-balanced sampling for training
    labels = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    num_workers = config["data"].get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=config["phase1"]["batch_size"],
        sampler=sampler, num_workers=num_workers,
        pin_memory=config["data"]["pin_memory"], drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["phase1"]["batch_size"],
        shuffle=False, num_workers=num_workers,
        pin_memory=config["data"]["pin_memory"],
    )
    return train_loader, val_loader


def build_scan_dataloaders(data_dir, metadata_dir, config):
    """Build train & val scan-level dataloaders for Phase 2."""
    train_entries = build_scan_manifest(data_dir, "train", metadata_dir)
    val_entries = build_scan_manifest(data_dir, "val", metadata_dir)

    img_size = config["data"]["image_size"]
    k_train = config["data"]["slices_per_scan"]
    k_val = config["eval"]["slices_per_scan"]

    train_ds = ScanDataset(train_entries, get_train_transforms(img_size), k_train,
                           sampling_strategy="random")
    val_ds = ScanDataset(val_entries, get_val_transforms(img_size), k_val,
                         sampling_strategy="uniform")

    train_loader = DataLoader(
        train_ds, batch_size=config["phase2"]["batch_size"],
        shuffle=True, num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"], drop_last=True,
        collate_fn=scan_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["eval"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        collate_fn=scan_collate_fn,
    )
    return train_loader, val_loader
