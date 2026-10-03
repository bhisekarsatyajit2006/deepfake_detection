"""EfficientNet-B4 deepfake classifier + robust checkpoint loading.

Architecture matches DeepShield (Yashikaysn29/deepshield on HuggingFace):
    EfficientNet-B4 backbone (1792-d features)
    → Linear(1792, 256) → ReLU → Dropout(0.4) → Linear(256, 1) → Sigmoid
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from pathlib import Path

from deepfake.config import Settings, settings


class DeepfakeDetector(nn.Module):
    """EfficientNet-B4 with DeepShield classification head."""

    def __init__(self, dropout: float = 0.4, pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        in_features = self.backbone.num_features  # 1792 for B4
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x)).squeeze(1)


def load_model(cfg: Settings = settings) -> tuple[DeepfakeDetector, float]:
    """Load checkpoint robustly. Falls back to best_model.pth if primary not found."""
    model = DeepfakeDetector(dropout=0.4, pretrained=False)
    threshold = cfg.threshold

    # Try primary path, then best_model.pth fallback
    model_path = cfg.model_path
    if not model_path.exists():
        fallback = model_path.parent / "best_model.pth"
        if fallback.exists():
            print(f"[Model] Primary not found; loading fallback: {fallback}")
            model_path = fallback
        else:
            print(f"[WARN] No checkpoint found — random weights (results meaningless)")
            return model.to(cfg.device).eval(), threshold

    try:
        ckpt = torch.load(model_path, map_location=cfg.device, weights_only=True)
    except Exception:
        print("[WARN] weights_only=True failed, retrying with weights_only=False")
        ckpt = torch.load(model_path, map_location=cfg.device, weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print(f"[Model] Checkpoint keys: epoch={ckpt.get('epoch','?')} val_auc={ckpt.get('val_auc','?')}")
    else:
        state_dict = ckpt
        print("[Model] Loaded raw state_dict")

    missing = set(model.state_dict().keys()) - set(state_dict.keys())
    unexpected = set(state_dict.keys()) - set(model.state_dict().keys())
    if missing:
        print(f"[WARN] {len(missing)} missing keys: {list(missing)[:3]}")
    if unexpected:
        print(f"[WARN] {len(unexpected)} unexpected keys: {list(unexpected)[:3]}")

    model.load_state_dict(state_dict, strict=False)
    model.to(cfg.device).eval()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)
        prob = model(dummy).item()
    print(f"[Model] Loaded OK — dummy prob = {prob:.4f}  device = {cfg.device}")
    return model, threshold
