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

import cv2
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
    Training transforms: intensity-only augmentations (no geometric ops).
    Input is already image_size×image_size from _load_image_with_roi — no Resize/Crop needed.
    """
    return A.Compose([
        A.RandomGamma(gamma_limit=(85, 115), p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(var_limit=(6.5, 6.5), mean=0, p=0.2),  # σ≈0.01 in [0,1] → var≈6.5 uint8
        A.Normalize(mean=RADIMAGENET_MEAN, std=RADIMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224):
    """Validation transforms: normalize only. ROI crop already outputs image_size×image_size."""
    return A.Compose([
        A.Normalize(mean=RADIMAGENET_MEAN, std=RADIMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size: int = 224) -> list:
    """
    Returns 4 deterministic intensity TTA pipelines (no geometric augmentation).
    Input is already image_size×image_size from _load_image_with_roi.

    Pipelines: [identity, γ=0.9, γ=1.1, CLAHE]
    """
    norm = [A.Normalize(mean=RADIMAGENET_MEAN, std=RADIMAGENET_STD), ToTensorV2()]
    return [
        A.Compose(norm),                                                             # identity
        A.Compose([A.RandomGamma(gamma_limit=(90, 90),   p=1.0)] + norm),           # γ=0.9
        A.Compose([A.RandomGamma(gamma_limit=(110, 110), p=1.0)] + norm),           # γ=1.1
        A.Compose([A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)] + norm),  # CLAHE
    ]


# ---------- Helpers ---------- #

def _load_image(path: str) -> np.ndarray:
    """Load a JPEG slice and convert to RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _center_crop_resize(img_rgb: np.ndarray, target_size: int) -> np.ndarray:
    """Square center-crop then resize to target_size×target_size (ROI fallback)."""
    H, W = img_rgb.shape[:2]
    side = min(H, W)
    y0, x0 = (H - side) // 2, (W - side) // 2
    crop = img_rgb[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def _load_image_with_roi(path: str, target_size: int = 224) -> np.ndarray:
    """
    Load a CT slice and apply lung ROI heuristic crop:
      1. Grayscale + Otsu threshold → binary mask
      2. Morphological closing (15×15 ellipse) to fill holes
      3. Find connected components; keep 2 largest with area > 1% of image
      4. Union bounding box of those 2 components, padded 15% on all sides
      5. Crop original RGB to padded bounding box, resize to target_size×target_size

    Fallback: center-crop if fewer than 2 valid components found.
    Returns RGB uint8 numpy array of shape (target_size, target_size, 3).
    """
    img_rgb = np.array(Image.open(path).convert("RGB"))
    H, W = img_rgb.shape[:2]

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_area = 0.01 * H * W
    valid = sorted(
        [(stats[i, cv2.CC_STAT_AREA], i)
         for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area],
        reverse=True,
    )

    if len(valid) < 2:
        return _center_crop_resize(img_rgb, target_size)   # fallback

    top2 = [i for _, i in valid[:2]]
    x1 = min(stats[i, cv2.CC_STAT_LEFT]                            for i in top2)
    y1 = min(stats[i, cv2.CC_STAT_TOP]                             for i in top2)
    x2 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]  for i in top2)
    y2 = max(stats[i, cv2.CC_STAT_TOP]  + stats[i, cv2.CC_STAT_HEIGHT] for i in top2)

    px, py = int(0.15 * (x2 - x1)), int(0.15 * (y2 - y1))
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)

    crop = img_rgb[y1:y2, x1:x2]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


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
            # Try primary name first, then common alias (val → validation)
            candidates = [f"{split}_{label_name}.csv"]
            if split == "val":
                candidates.append(f"validation_{label_name}.csv")
            csv_path = next(
                (os.path.join(metadata_dir, c) for c in candidates
                 if os.path.exists(os.path.join(metadata_dir, c))),
                None,
            )
            if csv_path is None:
                tried = ", ".join(os.path.join(metadata_dir, c) for c in candidates)
                print(f"WARNING: metadata CSV not found (tried: {tried}) — source will be -1")
                continue
            df = pd.read_csv(csv_path)
            # Detect column names robustly
            name_col = next((c for c in df.columns if "scan" in c.lower()), None)
            ctr_col  = next((c for c in df.columns if any(
                k in c.lower() for k in ("centre", "center", "source")
            )), None)
            if name_col is None or ctr_col is None:
                print(f"WARNING: {csv_path} columns {list(df.columns)!r} — "
                      f"expected a 'scan name' column and a 'centre' column. "
                      f"source will be -1 for {label_name}.")
                continue
            if name_col != "ct_scan_name" or ctr_col != "data_centre":
                print(f"INFO: {csv_path} — using columns '{name_col}' and '{ctr_col}'")
            for _, row in df.iterrows():
                source_map[(label_name, str(row[name_col]))] = int(row[ctr_col])

    # Track which scan names were seen in each class (for missing-scan reporting)
    csv_scan_names: dict = {}   # (label_name) -> set of names from CSV
    for (lname, sname) in source_map:
        csv_scan_names.setdefault(lname, set()).add(sname)

    missing_dir: list = []      # in CSV but no directory on disk
    empty_dir:   list = []      # directory exists but has no valid slice images

    for label_name, label_id in [("covid", 0), ("non_covid", 1)]:
        class_dir = os.path.join(split_dir, label_name)
        if not os.path.isdir(class_dir):
            continue

        scans_on_disk = set()
        for scan_name in sorted(os.listdir(class_dir)):
            scan_dir = os.path.join(class_dir, scan_name)
            if not os.path.isdir(scan_dir):
                continue
            scans_on_disk.add(scan_name)

            # Skip scans with no valid slice images
            slices = _get_sorted_slices(scan_dir)
            if not slices:
                empty_dir.append(f"{label_name}/{scan_name}")
                continue

            source = source_map.get((label_name, scan_name), -1)
            entries.append({
                "scan_dir": scan_dir,
                "label": label_id,
                "source": source,
                "scan_name": scan_name,
            })

        # CSV entries with no matching directory
        for sname in sorted(csv_scan_names.get(label_name, set()) - scans_on_disk):
            missing_dir.append(f"{label_name}/{sname}")

    # Report at end (not per-scan, so logs stay clean)
    if missing_dir:
        print(f"MISSING ({split}, in CSV but no directory on disk — {len(missing_dir)} scan(s)):")
        for s in missing_dir:
            print(f"  {s}")
    if empty_dir:
        print(f"EMPTY ({split}, directory has no valid slice images — {len(empty_dir)} scan(s)):")
        for s in empty_dir:
            print(f"  {s}")

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

        # Load with ROI crop and transform
        images = []
        for path in selected:
            img = _load_image_with_roi(path)
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

        raw_imgs = [_load_image_with_roi(p) for p in selected]
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
    Center-and-class-balanced batch sampler for scan-level MIL training.

    Each batch of B scans is structured as:
        (B // n_centers) scans per center, split evenly between COVID and Non-COVID.

    For B=8, n_centers=4: each center contributes exactly 2 scans per batch —
    1 COVID and 1 Non-COVID — so every batch has equal center AND class balance.

    Buckets: one per (center, class) pair.  If a bucket runs out before the epoch
    ends, it is resampled with replacement from itself.  An epoch ends when the
    largest bucket has been fully iterated once.

    Interaction with asymmetric loss weights: the sampler gives every center equal
    batch representation; the loss weights (e.g. center_2=0.2) then control how
    much each center's gradients count.  Both are needed.

    Args:
        sources: list of center IDs aligned with dataset indices.
        labels:  list of class labels (0=covid, 1=non-covid), same alignment.
        batch_size: total scans per batch (must be divisible by n_centers).
    """

    def __init__(self, sources: list, labels: list, batch_size: int):
        # Build (center, class) → [dataset indices]
        self.buckets: dict = defaultdict(list)
        for i, (s, lbl) in enumerate(zip(sources, labels)):
            self.buckets[(int(s), int(lbl))].append(i)

        self.batch_size = batch_size
        self.centers = sorted({int(s) for s in sources})
        self.n_centers = len(self.centers)
        self.classes = sorted({int(lbl) for lbl in labels})
        self.n_classes = len(self.classes)

        # Epoch length = largest single bucket, rounded down to full batches
        # (per-center slots per batch = batch_size // n_centers;
        #  per-bucket slots = that // n_classes)
        self.per_center = max(1, batch_size // self.n_centers)
        self.per_bucket = max(1, self.per_center // self.n_classes)
        self.max_bucket_size = max(len(v) for v in self.buckets.values())

        missing = []
        for c in self.centers:
            for lbl in self.classes:
                if (c, lbl) not in self.buckets:
                    missing.append(f"center={c} class={lbl}")
        if missing:
            print(f"WARNING CenterBatchSampler: empty buckets {missing} — "
                  "these slots will be filled by resampling sibling buckets.")

    def _make_pool(self, key) -> list:
        """Return a shuffled copy of a bucket, extended to max_bucket_size with replacement."""
        idxs = self.buckets.get(key, [])
        if not idxs:
            # Fallback: pull from same center, any class
            center = key[0]
            idxs = []
            for lbl in self.classes:
                idxs.extend(self.buckets.get((center, lbl), []))
            if not idxs:
                idxs = list(range(len(self.centers)))  # last resort
        pool = list(idxs)
        random.shuffle(pool)
        while len(pool) < self.max_bucket_size:
            extra = list(idxs)
            random.shuffle(extra)
            pool.extend(extra)
        return pool[:self.max_bucket_size]

    def __iter__(self):
        pools = {key: self._make_pool(key) for key in [
            (c, lbl) for c in self.centers for lbl in self.classes
        ]}
        pos = {key: 0 for key in pools}

        while all(pos[key] + self.per_bucket <= self.max_bucket_size for key in pools):
            batch = []
            for c in self.centers:
                for lbl in self.classes:
                    key = (c, lbl)
                    batch.extend(pools[key][pos[key]: pos[key] + self.per_bucket])
                    pos[key] += self.per_bucket
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.max_bucket_size // self.per_bucket


# ---------- DataLoader Builders ---------- #

def build_slice_dataloaders(data_dir: str, metadata_dir: str, config: dict):
    """
    Build train & val dataloaders for slice-level training (v1/v2 — kept for backward compat).

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
    labels  = [s[1] for s in train_ds.samples]
    batch_sampler = CenterBatchSampler(sources, labels, config["phase1"]["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )
    return train_loader, val_entries


def build_scan_train_dataloader(data_dir: str, metadata_dir: str, config: dict):
    """
    Build scan-level DataLoader for MIL training (v3).

    Each training sample is one scan (bag of K=slices_per_scan slices).
    CenterBatchSampler operates at scan level (not slice level).
    scan_collate_fn yields 4-tuples: (images, labels, sources, masks).

    Returns: (train_loader, val_entries)
    """
    train_entries = build_scan_manifest(data_dir, "train", metadata_dir)
    val_entries   = build_scan_manifest(data_dir,   "val", metadata_dir)

    img_size = config["data"]["image_size"]
    train_ds = ScanDataset(
        train_entries,
        get_train_transforms(img_size),
        slices_per_scan=config["data"]["slices_per_scan"],   # K=64
    )

    # Center-and-class-balanced sampler at scan level
    scan_sources = [entry["source"] for entry in train_entries]
    scan_labels  = [entry["label"]  for entry in train_entries]
    batch_sampler = CenterBatchSampler(scan_sources, scan_labels, config["phase1"]["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        collate_fn=scan_collate_fn,    # returns (images:(B,K,3,H,W), labels, sources, masks)
    )
    return train_loader, val_entries


