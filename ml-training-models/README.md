# ML Model Training & Dataset Pipeline (PyTorch / TensorFlow)

This directory contains the machine learning training scripts, PyTorch model definitions, dataset preprocessing pipelines, and Jupyter notebooks used for training deepfake detection models (EfficientNet, Xception, ResNet).

## Directory Overview

- `models/`: PyTorch & TensorFlow model definition files and pretrained weight loaders.
- `training/`: Training loop scripts, cross-entropy loss configurations, and evaluation metrics.
- `notebooks/`: Jupyter notebooks (`.ipynb`) for dataset exploration, feature distribution analysis, and ROC curve plotting.
- `scripts/`: Data augmentation, facial frame extraction, and landmark alignment utilities.
- `requirements-train.txt`: Python dependencies for GPU-accelerated model training (PyTorch, torchvision, OpenCV, scikit-learn).
