"""
Model: DenseNet-121 backbone with RadImageNet pretrained weights for COVID-19 CT slice classification.
Source: aadit-dev-v4 branch.

Single model class used for both training phases:
  Phase 1: Frozen backbone, head-only fine-tuning
  Phase 2: Gradual backbone unfreezing (denseblock4 → denseblock3)

Scan-level predictions aggregate per-slice sigmoid probabilities by averaging.
"""
import os

import torch
import torch.nn as nn
from torchvision import models


class DenseNetCovidClassifier(nn.Module):
    """
    DenseNet-121 for COVID-19 CT slice classification.
    Binary output: logit > 0 → non-covid (1), logit <= 0 → covid (0).

    DenseNet-121 layer names for unfreezing reference:
        features.conv0, features.norm0, features.relu0, features.pool0
        features.denseblock1, features.transition1
        features.denseblock2, features.transition2
        features.denseblock3, features.transition3
        features.denseblock4, features.norm5
        classifier
    """

    EMBED_DIM = 1024  # DenseNet-121 feature dimension before classifier

    def __init__(self, pretrained_path: str = None, dropout: float = 0.4):
        super().__init__()

        self.backbone = models.densenet121(weights=None)

        if pretrained_path and os.path.exists(pretrained_path):
            self._load_radimagenet(pretrained_path)
        elif pretrained_path:
            print(f"WARNING: RadImageNet weights not found at {pretrained_path!r}. "
                  "Training from random init.")

        # Replace classifier: Dropout(0.4) + Linear(1024, 1) — binary classification
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.EMBED_DIM, 1),
        )
        nn.init.kaiming_normal_(self.backbone.classifier[1].weight, mode="fan_out")
        nn.init.zeros_(self.backbone.classifier[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — individual slices
        Returns:
            logits: (B, 1) — raw logits (apply sigmoid for probabilities)
        """
        return self.backbone(x)

    def _load_radimagenet(self, path: str) -> None:
        """
        Load RadImageNet pretrained weights.  Handles two checkpoint formats:
          1. Direct state dict — keys start with 'features.' or 'classifier.'
          2. Full serialised nn.Module — extracts .state_dict() automatically.
        Classifier keys are always dropped so our new head is used.
        """
        obj = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(obj, dict) and any(
            k.startswith("features.") or k.startswith("classifier.")
            for k in obj.keys()
        ):
            # Direct state dict
            sd = {k: v for k, v in obj.items() if not k.startswith("classifier")}
        else:
            # Full serialised module
            src_sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
            sd = {k: v for k, v in src_sd.items() if not k.startswith("classifier")}

        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        print(f"RadImageNet weights loaded from {path!r}. "
              f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # ------------------------------------------------------------------
    # Layer freezing / unfreezing helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters; keep classifier trainable."""
        for name, p in self.backbone.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    def unfreeze_block(self, *block_names: str) -> None:
        """
        Unfreeze parameters belonging to any of the specified block names.

        Example:
            model.unfreeze_block('denseblock4', 'norm5')
            model.unfreeze_block('denseblock3', 'transition3')
        """
        for name, p in self.backbone.named_parameters():
            if any(b in name for b in block_names):
                p.requires_grad = True

    def get_parameter_groups(self, head_lr: float, block_lrs: dict) -> list:
        """
        Build AdamW parameter groups with discriminative learning rates.

        Args:
            head_lr:   Learning rate for the classifier head.
            block_lrs: Dict mapping block name fragment → lr.
                       e.g. {'denseblock4': 1e-4, 'norm5': 1e-4, 'denseblock3': 5e-5}

        Returns:
            List of {'params': [...], 'lr': lr} dicts suitable for AdamW.
        """
        assigned: set = set()
        groups = []

        # Classifier head
        head_params = [
            p for n, p in self.backbone.named_parameters()
            if "classifier" in n and p.requires_grad
        ]
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
            assigned.update(id(p) for p in head_params)

        # Named backbone blocks (order matters: more specific first)
        for block_name, lr in block_lrs.items():
            block_params = [
                p for n, p in self.backbone.named_parameters()
                if block_name in n and p.requires_grad and id(p) not in assigned
            ]
            if block_params:
                groups.append({"params": block_params, "lr": lr})
                assigned.update(id(p) for p in block_params)

        return groups

    def trainable_param_count(self) -> int:
        """Return the number of currently trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
