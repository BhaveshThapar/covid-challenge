#!/usr/bin/env python3
"""Quick diagnostic: run ONE EfficientNet checkpoint in-process (no multiprocessing) and print prob stats."""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import CovidDetector
from src.dataset import build_scan_manifest, ScanDataset, get_val_transforms, scan_collate_fn
from src.utils import load_config, set_seed, CheckpointManager, compute_per_source_f1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test with exp_b3_s42
config = load_config("configs/efficientnet.yaml")
set_seed(config["seed"])

cfg = config["model"]
model = CovidDetector(
    backbone_name=cfg["backbone"],
    pretrained=False,
    embedding_dim=cfg["embedding_dim"],
    attention_hidden_dim=cfg["attention_hidden_dim"],
    classifier_hidden_dim=cfg["classifier_hidden_dim"],
    num_classes=cfg["num_classes"],
    dropout=cfg["dropout"],
    drop_path_rate=cfg.get("drop_path_rate", 0.0),
).to(device)

ckpt_path = "checkpoints/exp_b3_s42/best.pt"
epoch, score = CheckpointManager.load(ckpt_path, model, device=device)
print(f"Loaded {ckpt_path} (epoch {epoch}, score={score:.4f})")

entries = build_scan_manifest("data", "val", "datasets")
print(f"Val scans: {len(entries)}")

ds = ScanDataset(entries, get_val_transforms(config["data"]["image_size"]),
                 slices_per_scan=config["eval"]["slices_per_scan"])
loader = DataLoader(ds, batch_size=1, shuffle=False,
                    num_workers=config["data"]["num_workers"],
                    pin_memory=config["data"]["pin_memory"],
                    collate_fn=scan_collate_fn)

model.eval()
all_probs, all_labels, all_sources = [], [], []
with torch.no_grad():
    for images, labels, sources, masks in tqdm(loader, desc="Inference"):
        images = images.to(device)
        masks = masks.to(device)
        with autocast(enabled=True):
            logits, _ = model(images, masks)
        probs_b = F.softmax(logits, dim=1)[:, 0].cpu().numpy()
        all_probs.extend(probs_b)
        all_labels.extend(labels.numpy())
        all_sources.extend(sources.numpy())

probs = np.array(all_probs)
labels = np.array(all_labels)
sources = np.array(all_sources)

print(f"\nProb stats: mean={probs.mean():.4f}, std={probs.std():.4f}, "
      f"min={probs.min():.4f}, max={probs.max():.4f}")
print(f"First 10 probs: {probs[:10]}")
print(f"First 10 labels: {labels[:10]}")

# Sweep threshold
best_t, best_f1 = 0.5, 0.0
for t in np.linspace(0.3, 0.7, 41):
    preds = (probs >= t).astype(int)
    f1 = compute_per_source_f1(labels, preds, sources)["average"]
    if f1 > best_f1:
        best_t, best_f1 = float(t), f1
print(f"\nBest threshold: {best_t:.2f} → F1: {best_f1:.4f}")

# Also compare with evaluate.py's approach
from src.evaluate import collect_scan_probs_efficientnet, tune_threshold
print("\n--- Running evaluate.py's collect_scan_probs_efficientnet ---")
probs2, labels2, sources2 = collect_scan_probs_efficientnet(model, entries, config, device, True)
print(f"Prob stats (evaluate.py): mean={probs2.mean():.4f}, std={probs2.std():.4f}")
t2, f2 = tune_threshold(probs2, labels2, sources2)
print(f"Best threshold: {t2:.2f} → F1: {f2:.4f}")
