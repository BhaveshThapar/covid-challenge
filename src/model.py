"""
Unified model re-exports for backward compatibility.

All models live in src.models. Import from here or from src.models directly:

  from src.model import DINOv2CovidClassifier
  from src.models import DINOv2CovidClassifier, DenseNetCovidClassifier, CovidDetector
"""
from src.models import (
    DINOv2CovidClassifier,
    DenseNetCovidClassifier,
    CovidDetector,
    SliceClassifier,
    AttentionPooling,
)

__all__ = [
    "DINOv2CovidClassifier",
    "DenseNetCovidClassifier",
    "CovidDetector",
    "SliceClassifier",
    "AttentionPooling",
]
