"""
Model: EfficientNet-B3 backbone + Attention-based MIL pooling for scan-level classification.

Two modes:
  - SliceClassifier: for Phase 1 slice-level pretraining
  - CovidDetector:   for Phase 2 scan-level end-to-end training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SliceClassifier(nn.Module):
    """
    Phase 1: Slice-level classifier.
    EfficientNet backbone + simple linear head.
    """

    def __init__(self, backbone_name="efficientnet_b3", pretrained=True,
                 num_classes=2, dropout=0.3, drop_path_rate=0.0):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0, drop_path_rate=drop_path_rate)
        self.embed_dim = self.backbone.num_features  # 1536 for efficientnet_b3
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) — single slices
        Returns:
            logits: (B, num_classes)
        """
        features = self.backbone(x)       # (B, embed_dim)
        logits = self.head(features)      # (B, num_classes)
        return logits

    def extract_features(self, x):
        """Extract features without classification head."""
        return self.backbone(x)


class AttentionPooling(nn.Module):
    """
    Gated attention mechanism for MIL (Multiple Instance Learning).
    Learns to weight the importance of each slice in a scan.
    
    Reference: Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, h, mask=None):
        """
        Args:
            h: (B, K, embed_dim) — slice embeddings
            mask: (B, K) — 1 for valid slices, 0 for padding
        Returns:
            z: (B, embed_dim) — scan-level embedding
            attention_weights: (B, K) — attention weights per slice
        """
        # Gated attention
        v = self.attention_V(h)          # (B, K, hidden_dim)
        u = self.attention_U(h)          # (B, K, hidden_dim)
        scores = self.attention_w(v * u).squeeze(-1)  # (B, K)

        # Mask padding
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=1)  # (B, K)

        # Weighted sum
        z = torch.bmm(attention_weights.unsqueeze(1), h).squeeze(1)  # (B, embed_dim)
        return z, attention_weights


class CovidDetector(nn.Module):
    """
    Phase 2: Full scan-level model.
    EfficientNet backbone → Attention Pooling → Classification head.
    Memory-safe: uses gradient checkpointing and chunked slice processing.
    """

    def __init__(self, backbone_name="efficientnet_b3", pretrained=True,
                 embedding_dim=1536, attention_hidden_dim=256,
                 classifier_hidden_dim=256, num_classes=2, dropout=0.3,
                 drop_path_rate=0.0):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0, drop_path_rate=drop_path_rate)
        self.embed_dim = self.backbone.num_features

        # Enable gradient checkpointing to save ~60% GPU memory
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable=True)

        self.attention = AttentionPooling(self.embed_dim, attention_hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward_features(self, x, mask=None):
        """
        Extract scan-level embedding without classification.
        
        Args:
            x: (B, K, 3, H, W) — K slices per scan
            mask: (B, K) — valid slice mask (optional)
        Returns:
            scan_embed: (B, embed_dim) — scan-level embedding
            attention_weights: (B, K) — attention weights per slice
        """
        B, K, C, H, W = x.shape

        # Process slices in chunks to prevent OOM
        x_flat = x.view(B * K, C, H, W)
        chunk_size = 8  # max slices through backbone at once
        features_list = []
        for i in range(0, B * K, chunk_size):
            chunk = x_flat[i:i + chunk_size]
            feat = self.backbone(chunk)
            features_list.append(feat)
        features = torch.cat(features_list, dim=0)  # (B*K, embed_dim)
        features = features.view(B, K, -1)           # (B, K, embed_dim)

        # Attention pooling
        scan_embed, attn_weights = self.attention(features, mask)  # (B, embed_dim), (B, K)
        return scan_embed, attn_weights

    def forward(self, x, mask=None):
        """
        Full forward pass: backbone → attention → classifier.
        
        Args:
            x: (B, K, 3, H, W) — K slices per scan
            mask: (B, K) — valid slice mask (optional)
        Returns:
            logits: (B, num_classes)
            attention_weights: (B, K)
        """
        scan_embed, attn_weights = self.forward_features(x, mask)
        logits = self.classifier(scan_embed)     # (B, num_classes)
        return logits, attn_weights

    @classmethod
    def from_slice_classifier(cls, slice_model: SliceClassifier, config: dict):
        """
        Initialize CovidDetector from a pretrained SliceClassifier,
        transferring the backbone weights.
        """
        model = cls(
            backbone_name=config["model"]["backbone"],
            pretrained=False,
            embedding_dim=config["model"]["embedding_dim"],
            attention_hidden_dim=config["model"]["attention_hidden_dim"],
            classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
            drop_path_rate=config["model"].get("drop_path_rate", 0.0),
        )
        # Copy backbone weights
        model.backbone.load_state_dict(slice_model.backbone.state_dict())
        return model
