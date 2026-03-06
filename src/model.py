"""
Model: DenseNet-121 backbone with RadImageNet pretrained weights for COVID-19 CT classification.

Two model classes:
  DenseNetCovidClassifier — v1/v2: slice-level, average-pool scan aggregation (kept for compat)
  DenseNetMILClassifier   — v3: scan-level MIL with attention pooling + MixStyle
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# MixStyle (Zhou et al. 2021) — used by DenseNetMILClassifier
# ---------------------------------------------------------------------------

class MixStyle(nn.Module):
    """
    Interpolates per-channel feature statistics across random batch pairs during training.
    No-op at eval time. Applied after denseblock1 and denseblock2 in the MIL model.

    Reference: Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
    """

    def __init__(self, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.beta = torch.distributions.Beta(alpha, alpha)  # instantiate once, reuse
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x                                        # no-op at eval
        B = x.size(0)
        mu  = x.mean(dim=[2, 3], keepdim=True)             # (B, C, 1, 1)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        x_normed = (x - mu) / sig
        perm    = torch.randperm(B, device=x.device)
        lam     = self.beta.sample((B, 1, 1, 1)).to(x.device)  # per-sample lambda
        mu_mix  = lam * mu  + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return x_normed * sig_mix + mu_mix


# ---------------------------------------------------------------------------
# DenseNetMILClassifier — v3 model (scan-level MIL with attention pooling)
# ---------------------------------------------------------------------------

class DenseNetMILClassifier(nn.Module):
    """
    DenseNet-121 backbone + Attention MIL head for scan-level COVID-19 classification.

    Input:  (B, K, 3, H, W) — B scans, K slices each
    Output: (B, 1)           — scan-level logits (apply sigmoid for probabilities)

    Attention (ABMIL tanh variant, Ilse et al. 2018):
      per-slice 1024-d embedding → Linear(1024, hidden) → Tanh → Linear(hidden, 1)
      → softmax over K slices → weighted sum → 1024-d scan embedding
      → Dropout + Linear(1024, 1)

    MixStyle is injected after denseblock1 and denseblock2 (training only, no-op at eval).

    DenseNet-121 features sub-module names (torchvision):
        conv0, norm0, relu0, pool0,
        denseblock1, transition1,
        denseblock2, transition2,
        denseblock3, transition3,
        denseblock4, norm5
    """

    EMBED_DIM = 1024

    def __init__(
        self,
        pretrained_path: str = None,
        dropout: float = 0.4,
        mil_hidden_dim: int = 128,
        mixstyle_alpha: float = 0.1,
    ):
        super().__init__()

        self.backbone = models.densenet121(weights=None)

        if pretrained_path and os.path.exists(pretrained_path):
            self._load_radimagenet(pretrained_path)
        elif pretrained_path:
            print(f"WARNING: RadImageNet weights not found at {pretrained_path!r}. "
                  "Training from random init.")

        # Replace backbone classifier with Identity — aggregation is done via attention
        self.backbone.classifier = nn.Identity()

        self.mixstyle = MixStyle(alpha=mixstyle_alpha)

        # Attention MLP: 1024 → hidden → tanh → 1
        self.attention = nn.Sequential(
            nn.Linear(self.EMBED_DIM, mil_hidden_dim),
            nn.Tanh(),
            nn.Linear(mil_hidden_dim, 1),
        )

        # Classification head (same structure as DenseNetCovidClassifier)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.EMBED_DIM, 1),
        )

        # Initialisation
        nn.init.kaiming_normal_(self.classifier[1].weight, mode="fan_out")
        nn.init.zeros_(self.classifier[1].bias)
        nn.init.xavier_uniform_(self.attention[0].weight)
        nn.init.zeros_(self.attention[0].bias)
        nn.init.xavier_uniform_(self.attention[2].weight)
        nn.init.zeros_(self.attention[2].bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Manual DenseNet feature extraction with MixStyle after blocks 1 and 2.

        Args:
            x: (N, 3, H, W) — N individual slices
        Returns:
            embeddings: (N, 1024)
        """
        f = self.backbone.features
        x = f.conv0(x); x = f.norm0(x); x = f.relu0(x); x = f.pool0(x)
        x = f.denseblock1(x);  x = self.mixstyle(x)    # MixStyle (no-op at eval)
        x = f.transition1(x)
        x = f.denseblock2(x);  x = self.mixstyle(x)    # MixStyle (no-op at eval)
        x = f.transition2(x)
        x = f.denseblock3(x);  x = f.transition3(x)
        x = f.denseblock4(x);  x = f.norm5(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)                      # (N, 1024)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        MIL forward pass.

        Args:
            x:    (B, K, 3, H, W) — batch of scan bags
            mask: (B, K) float tensor, 1.0 for valid slices, 0.0 for padding.
                  Pass None when all slices are valid (e.g. fixed K during training).
        Returns:
            logits: (B, 1) — raw logits
        """
        B, K, C, H, W = x.shape
        embeds   = self.forward_features(x.view(B * K, C, H, W)).view(B, K, self.EMBED_DIM)
        attn_raw = self.attention(embeds)               # (B, K, 1)
        if mask is not None:
            # Large negative on padding positions → effectively zero attention after softmax
            attn_raw = attn_raw + (1.0 - mask.unsqueeze(-1)) * (-1e9)
        attn_w     = torch.softmax(attn_raw, dim=1)    # (B, K, 1)
        scan_embed = (attn_w * embeds).sum(dim=1)      # (B, 1024)
        return self.classifier(scan_embed)              # (B, 1)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters; keep classifier + attention trainable."""
        for name, p in self.backbone.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    def freeze_attention(self) -> None:
        """Freeze attention MLP. Call in Phase 1 so attention re-trains on adapted features in P2."""
        for p in self.attention.parameters():
            p.requires_grad = False

    def unfreeze_attention(self) -> None:
        """Re-enable attention training. Call at Phase 2 start before building optimizer."""
        for p in self.attention.parameters():
            p.requires_grad = True

    def unfreeze_block(self, *block_names: str) -> None:
        """Unfreeze parameters whose name contains any of the given block name fragments."""
        for name, p in self.backbone.named_parameters():
            if any(b in name for b in block_names):
                p.requires_grad = True

    def get_parameter_groups(self, head_lr: float, block_lrs: dict) -> list:
        """
        Build AdamW parameter groups with discriminative learning rates.

        Head group = self.classifier + self.attention (on self, not self.backbone).
        self.backbone.classifier is nn.Identity() and has no parameters.

        Args:
            head_lr:   LR for classifier head and attention MLP.
            block_lrs: Dict mapping block name fragment → lr.
        Returns:
            List of {'params': [...], 'lr': lr} dicts for AdamW.
        """
        assigned: set = set()
        groups = []

        # Head params: self.classifier and self.attention are on self (not self.backbone)
        head_params = [
            p for n, p in self.named_parameters()
            if ("classifier" in n or "attention" in n)
            and "backbone" not in n          # backbone.classifier is Identity, no params
            and p.requires_grad
        ]
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
            assigned.update(id(p) for p in head_params)

        # Backbone block groups
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

    def _load_radimagenet(self, path: str) -> None:
        """Load RadImageNet pretrained weights (identical logic to DenseNetCovidClassifier)."""
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and any(
            k.startswith("features.") or k.startswith("classifier.") for k in obj.keys()
        ):
            sd = {k: v for k, v in obj.items() if not k.startswith("classifier")}
        else:
            src_sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
            sd = {k: v for k, v in src_sd.items() if not k.startswith("classifier")}
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        print(f"RadImageNet weights loaded from {path!r}. "
              f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")


# ---------------------------------------------------------------------------
# DenseNetCovidClassifier — v1/v2 model (kept for backward compatibility)
# ---------------------------------------------------------------------------

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
