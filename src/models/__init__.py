"""
Modular model definitions for COVID-19 CT classification.

- DINOv2CovidClassifier: DINOv2 ViT-B/14 (Anant) — Anant-dev
- DenseNetCovidClassifier: DenseNet-121 + RadImageNet (Aadit) — aadit-dev-v4
- CovidDetector: EfficientNet-B3 + Attention MIL (Bhavesh) — bhavesh/improve-diversity
"""
from .dinov2 import DINOv2CovidClassifier
from .densenet import DenseNetCovidClassifier
from .efficientnet import CovidDetector, SliceClassifier, AttentionPooling

__all__ = [
    "DINOv2CovidClassifier",
    "DenseNetCovidClassifier",
    "CovidDetector",
    "SliceClassifier",
    "AttentionPooling",
]
