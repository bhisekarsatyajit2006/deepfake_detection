"""Deepfake detection — shared library for the local web app and diagnostics.

Single source of truth for the model architecture, image transforms, and the
video → face → classification inference pipeline. The Flask app (``webapp``)
and the diagnostic tool (``scripts/diagnose.py``) both import from here so the
model is defined exactly once.

Training runs on Kaggle (see ``training/train_kaggle.py``) and is intentionally
kept standalone — it cannot import this package inside a Kaggle kernel.
"""

from deepfake.config import Settings, settings
from deepfake.model import DeepfakeDetector, load_model
from deepfake.transforms import build_val_transform

__all__ = [
    "Settings",
    "settings",
    "DeepfakeDetector",
    "load_model",
    "build_val_transform",
]
