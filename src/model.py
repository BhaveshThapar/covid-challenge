"""
Model: DINOv2 ViT-B/14 backbone for COVID-19 CT slice classification.

DINOv2 is a self-supervised vision model (Meta) that learns robust representations
without labels. ViT-B/14 has 86M params, 768-dim embeddings, 14×14 patch size.

Single model class used for both training phases:
  Phase 1: Frozen backbone, head-only fine-tuning
  Phase 2: Gradual backbone unfreezing (last 2 transformer blocks → last 4 blocks)

Scan-level predictions aggregate per-slice sigmoid probabilities by averaging.
"""
import torch
import torch.nn as nn


class DINOv2CovidClassifier(nn.Module):
    """
    DINOv2 ViT-B/14 for COVID-19 CT slice classification.
    Binary output: logit > 0 → non-covid (1), logit <= 0 → covid (0).

    ViT-B/14 structure for unfreezing:
        patch_embed, cls_token, pos_embed
        blocks.0 ... blocks.11  (12 transformer blocks)
        norm
        head (our classifier: Dropout + Linear)
    """

    EMBED_DIM = 768  # ViT-B/14 feature dimension

    def __init__(self, dropout: float = 0.4, hub_repo: str = "facebookresearch/dinov2",
                 hub_model: str = "dinov2_vitb14"):
        super().__init__()

        # Load DINOv2 backbone from torch hub (self-supervised, no labels)
        self.backbone = torch.hub.load(hub_repo, hub_model, pretrained=True, verbose=False)

        # Replace head: Dropout + Linear(768, 1) for binary classification
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.EMBED_DIM, 1),
        )
        nn.init.kaiming_normal_(self.backbone.head[1].weight, mode="fan_out")
        nn.init.zeros_(self.backbone.head[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — individual slices (224×224 expected)
        Returns:
            logits: (B, 1) — raw logits (apply sigmoid for probabilities)
        """
        # DINOv2 forward: returns head(cls_token) — our head gives (B, 1)
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Layer freezing / unfreezing helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters except the classification head."""
        for name, p in self.backbone.named_parameters():
            if "head" not in name:
                p.requires_grad = False

    def unfreeze_block(self, *block_names: str) -> None:
        """
        Unfreeze parameters belonging to any of the specified block names.

        ViT block names: "blocks.11", "blocks.10", "norm", etc.
        For chunked layouts (block_chunks>1), use "blocks.3.2", "blocks.3.1" for last blocks.

        Example:
            model.unfreeze_block("blocks.11", "blocks.10", "norm")
            model.unfreeze_block("blocks.9", "blocks.8")
        """
        for name, p in self.backbone.named_parameters():
            if any(b in name for b in block_names):
                p.requires_grad = True

    def unfreeze_blocks_by_index(self, indices: list, also_norm: bool = False) -> None:
        """
        Unfreeze transformer blocks by 0-based index (0-11 for ViT-B).
        Handles both flat (blocks.0..blocks.11) and chunked layouts.
        """
        for name, p in self.backbone.named_parameters():
            if "head" in name:
                continue
            if also_norm and "norm" in name and "norm1" not in name and "norm2" not in name:
                p.requires_grad = True
                continue
            if "blocks." not in name:
                continue
            parts = name.split(".")
            try:
                if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                    # Chunked: blocks.3.2 → block_idx = 3*3+2 = 11
                    chunk, inner = int(parts[1]), int(parts[2])
                    block_idx = chunk * 3 + inner
                elif len(parts) >= 2 and parts[1].isdigit():
                    block_idx = int(parts[1])
                else:
                    continue
                if block_idx in indices:
                    p.requires_grad = True
            except (ValueError, IndexError):
                pass

    def get_parameter_groups(self, head_lr: float, block_lrs: dict) -> list:
        """
        Build AdamW parameter groups with discriminative learning rates.

        Args:
            head_lr:   Learning rate for the classifier head.
            block_lrs: Dict mapping block name fragment → lr.
                       e.g. {"blocks.11": 1e-4, "blocks.10": 1e-4, "norm": 1e-4}

        Returns:
            List of {"params": [...], "lr": lr} dicts suitable for AdamW.
        """
        assigned = set()
        groups = []

        # Classifier head
        head_params = [
            p for n, p in self.backbone.named_parameters()
            if "head" in n and p.requires_grad
        ]
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
            assigned.update(id(p) for p in head_params)

        # Named backbone blocks (order matters: more specific first)
        for block_name, lr in block_lrs.items():
            def _match_block(n: str) -> bool:
                if block_name == "norm":
                    return "norm" in n and "blocks." not in n  # top-level norm only
                return block_name in n

            block_params = [
                p for n, p in self.backbone.named_parameters()
                if _match_block(n) and p.requires_grad and id(p) not in assigned
            ]
            if block_params:
                groups.append({"params": block_params, "lr": lr})
                assigned.update(id(p) for p in block_params)

        return groups

    def trainable_param_count(self) -> int:
        """Return the number of currently trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
