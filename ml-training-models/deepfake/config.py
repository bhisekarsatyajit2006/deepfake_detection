"""Central configuration — every constant and env-var override lives here.

Previously these values were copy-pasted across app.py, deepfake_pipeline.py,
deepfake_train.py and diagnose.py. Now there is one place to change them.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Project root = the directory containing this package.
ROOT = Path(__file__).resolve().parent.parent

# ImageNet normalisation (backbone was pretrained on ImageNet).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _env_path(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val) if val else default


@dataclass(frozen=True)
class Settings:
    """Runtime settings, overridable via environment variables."""

    # ── Paths ──────────────────────────────────────────────────────────────
    videos_dir: Path = field(
        default_factory=lambda: _env_path("VIDEOS_DIR", ROOT / "videos"))
    model_path: Path = field(
        default_factory=lambda: _env_path(
            "MODEL_PATH", ROOT / "models" / "efficientnet_b4_deepfake_resumed_final.pth"))
    csv_path: Path = field(
        default_factory=lambda: _env_path("CSV_PATH", ROOT / "metadata.csv"))

    # ── Model / preprocessing (must match the training pipeline) ────────────
    img_size: int = 224
    dropout: float = 0.5
    threshold: float = field(
        default_factory=lambda: float(os.environ.get("CLF_THRESHOLD", "0.463")))

    # ── Frame sampling ──────────────────────────────────────────────────────
    frame_step: int = 10       # keep every Nth frame
    max_frames: int = 30       # cap frames analysed per video

    # ── MTCNN face detection ────────────────────────────────────────────────
    min_confidence: float = 0.95
    min_face_px: int = 60
    margin: float = 0.20       # padding around the detected box

    # ── Web server ──────────────────────────────────────────────────────────
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "5000")))

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Import this shared instance everywhere.
settings = Settings()
