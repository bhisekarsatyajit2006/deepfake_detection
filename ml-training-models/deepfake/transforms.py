"""Image transforms — the validation/inference transform used at serve time.

Must exactly match the ``val_transform`` used during training: resize to
img_size, ImageNet-normalise, to CHW float tensor. Input is an RGB uint8 HWC
numpy array.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from deepfake.config import IMAGENET_MEAN, IMAGENET_STD, Settings, settings


def build_val_transform(cfg: Settings = settings) -> A.Compose:
    return A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
