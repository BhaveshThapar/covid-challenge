"""
Utility functions: seeding, logging, checkpointing, metrics.
"""
import os
import random
import logging
import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, confusion_matrix


def set_seed(seed: int = 42):
    """Reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    """Load YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Create a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def compute_macro_f1(y_true, y_pred):
    """Compute macro F1 (average of per-class F1)."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def compute_weighted_f1(f1_dict: dict, center_weights: dict = None) -> float:
    """
    Weighted average of per-center F1 for checkpoint selection.

    Args:
        f1_dict:        Output of compute_per_source_f1 — keys like 'source_0', 'average'.
        center_weights: {0: 1.0, 1: 1.0, 2: 0.2, 3: 1.0}
                        Falls back to plain average if None.

    Returns:
        float — weighted F1 score. Denominator = sum of all weights (e.g. 3.2).
    """
    if center_weights is None:
        return f1_dict.get("average", 0.0)
    total_w = sum(center_weights.values())
    weighted = sum(
        center_weights.get(src_id, 0.0) * f1_dict.get(f"source_{src_id}", 0.0)
        for src_id in center_weights
    )
    return weighted / max(total_w, 1e-9)


def compute_per_source_f1(y_true, y_pred, sources, strict_labels=True):
    """
    Compute macro F1 per data source, then average.

    Args:
        y_true: array of true labels (0=covid, 1=non-covid or vice versa)
        y_pred: array of predicted labels
        sources: array of source IDs (0-3)
        strict_labels: if True, always compute F1 for both class 0 and class 1,
                       matching the challenge formula (F1_covid + F1_noncovid) / 2.
                       If False, uses sklearn default (only classes present in
                       y_true ∪ y_pred) — used during training for early stopping.

    Returns:
        dict with per-source F1 and the averaged final score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sources = np.array(sources)

    unique_sources = sorted(np.unique(sources))
    source_f1 = {}
    for src in unique_sources:
        mask = sources == src
        if mask.sum() == 0:
            continue
        if strict_labels:
            f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0, labels=[0, 1])
        else:
            f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
        source_f1[f"source_{src}"] = f1

    avg_f1 = np.mean(list(source_f1.values())) if source_f1 else 0.0
    source_f1["average"] = avg_f1
    return source_f1


def print_confusion_matrices(y_true, y_pred, sources, class_names=("Covid", "Non-Covid")):
    """Print confusion matrix per source."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sources = np.array(sources)

    for src in sorted(np.unique(sources)):
        mask = sources == src
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=list(range(len(class_names))))
        print(f"\n--- Source {src} (n={mask.sum()}) ---")
        print(f"{'':>12}", end="")
        for name in class_names:
            print(f"{name:>12}", end="")
        print()
        for i, name in enumerate(class_names):
            print(f"{name:>12}", end="")
            for j in range(len(class_names)):
                print(f"{cm[i, j]:>12}", end="")
            print()


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class CheckpointManager:
    """Save and load model checkpoints."""

    def __init__(self, checkpoint_dir: str, max_keep: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.max_keep = max_keep
        self.saved = []
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, epoch, score, filename=None):
        if filename is None:
            filename = f"checkpoint_epoch{epoch}_f1{score:.4f}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "score": score,
        }, path)
        self.saved.append((score, path))
        # Keep only top-k
        self.saved.sort(key=lambda x: x[0], reverse=True)
        while len(self.saved) > self.max_keep:
            _, old_path = self.saved.pop()
            if os.path.exists(old_path):
                os.remove(old_path)
        return path

    def save_named(self, model, optimizer, epoch, score, filename: str) -> str:
        """
        Save to a fixed filename WITHOUT participating in the max_keep rotation.
        Use this for named snapshots like 'phase1_best.pt', 'best.pt' that must
        not be deleted by the rotation logic (which would happen if the same
        filename is saved more than max_keep times via save()).
        """
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "score": score,
        }, path)
        return path

    def save_best(self, model, optimizer, epoch, score):
        return self.save_named(model, optimizer, epoch, score, "best.pt")

    @staticmethod
    def load(path, model, optimizer=None, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt.get("epoch", 0), ckpt.get("score", 0.0)
