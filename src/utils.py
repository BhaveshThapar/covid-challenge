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
    """Compute macro F1 (average of per-class F1), excluding classes not in y_true."""
    present_labels = np.unique(y_true)
    return f1_score(y_true, y_pred, average="macro", labels=present_labels, zero_division=0)


def compute_per_source_f1(y_true, y_pred, sources):
    """
    Compute macro F1 per data source, then average.
    
    Per competition organizer clarification: when a source contains no samples
    for a class, that class's F1 is EXCLUDED from the macro-average (not set to 0).
    E.g., if Centre 2 has no COVID samples, its score = F1_noncovid only.
    
    Args:
        y_true: array of true labels (0=covid, 1=non-covid)
        y_pred: array of predicted labels
        sources: array of source IDs (0-3)
    
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
        # Only compute F1 for classes present in ground truth for this source
        present_labels = np.unique(y_true[mask])
        f1 = f1_score(y_true[mask], y_pred[mask], average="macro",
                      labels=present_labels, zero_division=0)
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
        return path

    def save_best(self, model, optimizer, epoch, score):
        """Save as best.pt — always overwrites, never pruned."""
        path = self.save(model, optimizer, epoch, score, filename="best.pt")
        print(f"  Saved best checkpoint: {path} (F1={score:.4f})")
        return path

    @staticmethod
    def load(path, model, optimizer=None, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt.get("epoch", 0), ckpt.get("score", 0.0)
