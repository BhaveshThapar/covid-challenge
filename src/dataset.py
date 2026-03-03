"""
Dataset and DataLoader classes for the Multi-Source Covid-19 Detection Challenge.

Two modes:
  1. SliceDataset  — returns individual slices (for Phase 1 / Phase 2 slice-level training)
  2. ScanDataset   — returns K slices per scan (for scan-level evaluation / validation)
"""
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------- Normalization constants ---------- #

# RadImageNet uses the same preprocessing pipeline as ImageNet → identical stats.
RADIMAGENET_MEAN = [0.485, 0.456, 0.406]
RADIMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------- Transforms ---------- #

def get_train_transforms(image_size: int = 224):
    """
    Training transforms: resize to 256, random crop to image_size, augmentations, normalize.
    Augmentations applied before crop to avoid black border artifacts from rotation.
    """
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomCrop(image_size, image_size),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        A.Normalize(mean=RADIMAGENET_MEAN, std=RADIMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224):
    """Validation transforms: resize to 256, center crop to image_size, normalize."""
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=RADIMAGENET_MEAN, std=RADIMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size: int = 224) -> list:
    """
    Returns 4 augmentation pipelines for test-time augmentation (TTA).
    Augmentation is applied BEFORE CenterCrop to avoid black border artifacts.

    Pipelines: [identity, horizontal flip, rotate +15°, rotate -15°]
    """
    base = [A.Resize(256, 256)]
    crop_norm = [
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=RADIMAGENET_MEAN, std=RADIMAGENET_STD),
        ToTensorV2(),
    ]
    augmentations = [
        [],                                      # identity
        [A.HorizontalFlip(p=1.0)],               # horizontal flip
        [A.Rotate(limit=(15, 15), p=1.0)],       # rotate +15°
        [A.Rotate(limit=(-15, -15), p=1.0)],     # rotate -15°
    ]
    return [A.Compose(base + aug + crop_norm) for aug in augmentations]


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
        # Skip Mac metadata files (._0, ._1, etc.) and any non-numeric names
        if ext.lower() not in exts:
            continue
        if not stem.lstrip("0123456789").strip() == "" or not stem.isdigit():
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
        source: medical center ID (0-3), -1 if not found in metadata
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

    for label_name, label_id in [("covid", 0), ("non_covid", 1)]:
        class_dir = os.path.join(split_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        for scan_name in sorted(os.listdir(class_dir)):
            scan_dir = os.path.join(class_dir, scan_name)
            if not os.path.isdir(scan_dir):
                continue
            source = source_map.get((label_name, scan_name), -1)
            entries.append({
                "scan_dir": scan_dir,
                "label": label_id,
                "source": source,
                "scan_name": scan_name,
            })

    return entries


# ---------- Slice-Level Dataset ---------- #

class SliceDataset(Dataset):
    """
    Returns individual slices with their scan-level label.
    Used for slice-level training (Phase 1 and Phase 2).
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
            if max_slices_per_scan > 0 and len(slices) > max_slices_per_scan:
                # Sample uniformly across the scan
                indices = np.linspace(0, len(slices) - 1, max_slices_per_scan, dtype=int)
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


# ---------- Scan-Level Dataset ---------- #

class ScanDataset(Dataset):
    """
    Returns K slices per scan for scan-level evaluation/validation.
    """

    def __init__(self, scan_entries: list, transform=None, slices_per_scan: int = 32):
        """
        Args:
            scan_entries: list from build_scan_manifest
            transform: albumentations transform
            slices_per_scan: number of slices to sample per scan (-1 = all)
        """
        self.entries = scan_entries
        self.transform = transform
        self.slices_per_scan = slices_per_scan

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
            # Uniform sampling across the scan
            indices = np.linspace(0, len(all_slices) - 1, k, dtype=int)
            selected = [all_slices[i] for i in indices]
        else:
            selected = all_slices

        # Load and transform
        images = []
        for path in selected:
            img = _load_image(path)
            if self.transform:
                img = self.transform(image=img)["image"]
            images.append(img)

        images = torch.stack(images)  # (K, 3, H, W)
        label = entry["label"]
        source = entry["source"]
        return images, label, source


class RawSliceScanDataset(Dataset):
    """
    Returns raw numpy arrays for each scan's slices (for TTA in evaluate.py).
    Avoids re-reading from disk for each TTA augmentation pass.
    """

    def __init__(self, scan_entries: list, slices_per_scan: int = -1):
        self.entries = scan_entries
        self.slices_per_scan = slices_per_scan
        self.scan_slices = [_get_sorted_slices(e["scan_dir"]) for e in scan_entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        all_slices = self.scan_slices[idx]
        k = self.slices_per_scan

        if k > 0 and k < len(all_slices):
            indices = np.linspace(0, len(all_slices) - 1, k, dtype=int)
            selected = [all_slices[i] for i in indices]
        else:
            selected = all_slices

        raw_imgs = [_load_image(p) for p in selected]
        return raw_imgs, entry["label"], entry["source"]


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


# ---------- Center-Stratified Batch Sampler ---------- #

class CenterBatchSampler(Sampler):
    """
    Yields batches where each medical center (0-3) contributes roughly equally.
    Smaller centers are resampled with replacement to match the largest center's
    pool size each epoch — no data from any center is permanently discarded.

    Usage:
        sources = [s[2] for s in train_ds.samples]
        DataLoader(train_ds, batch_sampler=CenterBatchSampler(sources, batch_size=32))
    """

    def __init__(self, sources: list, batch_size: int):
        self.pools_original: dict = defaultdict(list)
        for i, s in enumerate(sources):
            self.pools_original[s].append(i)
        self.batch_size = batch_size
        self.n_centers = len(self.pools_original)
        self.max_size = max(len(v) for v in self.pools_original.values())

    def __iter__(self):
        # Oversample smaller centers (with replacement) to match largest
        pools = {}
        for c, idxs in self.pools_original.items():
            shuffled = list(idxs)
            random.shuffle(shuffled)
            while len(shuffled) < self.max_size:
                extra = list(idxs)
                random.shuffle(extra)
                shuffled.extend(extra)
            pools[c] = shuffled[:self.max_size]

        per_center = max(1, self.batch_size // self.n_centers)
        pos = {c: 0 for c in pools}
        while all(pos[c] + per_center <= self.max_size for c in pools):
            batch = []
            for c in sorted(pools):
                batch.extend(pools[c][pos[c]: pos[c] + per_center])
                pos[c] += per_center
            random.shuffle(batch)
            yield batch

    def __len__(self):
        per_center = max(1, self.batch_size // self.n_centers)
        return self.max_size // per_center


# ---------- DataLoader Builders ---------- #

def build_slice_dataloaders(data_dir: str, metadata_dir: str, config: dict):
    """
    Build train & val dataloaders for slice-level training.

    Training: CenterBatchSampler ensures center-balanced batches.
    Validation: ScanDataset with scan-level collation (used in evaluate_scans helper).
    Returns: (train_loader, val_entries) — val_entries used directly by evaluate_scans()
    """
    train_entries = build_scan_manifest(data_dir, "train", metadata_dir)
    val_entries = build_scan_manifest(data_dir, "val", metadata_dir)

    img_size = config["data"]["image_size"]
    max_slices = config["data"]["slices_per_scan"]  # per scan cap during training

    train_ds = SliceDataset(train_entries, get_train_transforms(img_size), max_slices)

    sources = [s[2] for s in train_ds.samples]
    batch_sampler = CenterBatchSampler(sources, config["phase1"]["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )
    return train_loader, val_entries


