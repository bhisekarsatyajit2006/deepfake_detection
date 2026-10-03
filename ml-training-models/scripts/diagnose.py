"""Checkpoint / prediction debugger for the "always REAL" class of bugs.

Inspects a .pth checkpoint, rebuilds the model via the SHARED architecture
(deepfake.model.DeepfakeDetector — no more copy-pasted definition), probes it
with constant inputs, and optionally traces a real video end-to-end.

Usage:
    python -m scripts.diagnose --model models/efficientnet_b4_deepfake_resumed_final.pth
    python -m scripts.diagnose --model <path.pth> --video sample.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepfake.config import IMAGENET_MEAN, IMAGENET_STD, settings  # noqa: E402
from deepfake.model import DeepfakeDetector                        # noqa: E402
from deepfake.transforms import build_val_transform                # noqa: E402

IMG = settings.img_size


def inspect_checkpoint(path: Path):
    print("\n" + "=" * 60 + "\n1. CHECKPOINT INSPECTION\n" + "=" * 60)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  Type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"  Top-level keys: {list(ckpt.keys())}")
        sd = ckpt.get("state_dict", ckpt)
        for k, v in ckpt.items():
            if k not in ("state_dict", "cfg"):
                print(f"  {k}: {v}")
    else:
        sd = ckpt
    keys = list(sd.keys())
    print(f"  state_dict: {len(keys)} tensors; first 3: {keys[:3]}")
    for hk in keys:
        if "head" in hk:
            t = sd[hk]
            print(f"    HEAD '{hk}': shape={tuple(t.shape)} "
                  f"mean={t.float().mean():.4f} std={t.float().std():.4f}")
    return sd


def load_and_probe(sd) -> DeepfakeDetector:
    print("\n" + "=" * 60 + "\n2. MODEL LOAD + WEIGHT PROBE\n" + "=" * 60)
    model = DeepfakeDetector()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Missing keys   ({len(missing)}): {list(missing)[:5]}")
    print(f"  Unexpected keys({len(unexpected)}): {list(unexpected)[:5]}")
    for name, p in model.named_parameters():
        if "head" in name and "bias" in name:
            print(f"  Head bias: {p.item():+.4f} -> "
                  f"sigmoid={torch.sigmoid(p.detach()).item():.4f}")
    model.eval()
    with torch.no_grad():
        for val, name in [(0.0, "zeros"), (1.0, "ones"), (-1.0, "neg_ones")]:
            x = torch.full((1, 3, IMG, IMG), val)
            logit = model(x).item()
            print(f"  Input={name:8s} -> logit={logit:+.4f}  "
                  f"prob={torch.sigmoid(torch.tensor(logit)).item():.4f}")
    return model


def test_preprocessing():
    print("\n" + "=" * 60 + "\n3. PREPROCESSING CHECK\n" + "=" * 60)
    frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = build_val_transform()(image=rgb)["image"]
    expected = (128 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    got = tensor[0].mean().item()
    print(f"  shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    print(f"  channel[0] expected ~{expected:.4f}, got {got:.4f} -> "
          f"{'OK' if abs(got - expected) < 0.01 else 'WRONG'}")


def test_video(video_path: Path, model, threshold):
    print("\n" + "=" * 60 + "\n4. VIDEO INFERENCE TEST\n" + "=" * 60)
    from deepfake.inference import DeepfakeAnalyzer
    analyzer = DeepfakeAnalyzer.__new__(DeepfakeAnalyzer)  # reuse crop/infer without reloading
    analyzer.cfg = settings
    analyzer.model = model
    analyzer.threshold = threshold
    analyzer.transform = build_val_transform()
    from mtcnn import MTCNN
    analyzer.detector = MTCNN()

    result = analyzer.analyze_video(video_path)
    if "error" in result:
        print(f"  {result['error']}")
        return
    print(f"  faces={result['faces_detected']}  avg_prob={result['avg_prob']}  "
          f"verdict={result['prediction']}  (threshold={threshold})")
    if result["max_prob"] - min(result["all_probs"]) < 0.05:
        print("  WARNING: probs nearly identical — weights may not have loaded.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pth file")
    ap.add_argument("--video", default=None, help="Optional video for end-to-end test")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)

    sd = inspect_checkpoint(model_path)
    model = load_and_probe(sd)
    test_preprocessing()
    if args.video:
        test_video(Path(args.video), model, settings.threshold)
    else:
        print("\n  (Pass --video path/to/video.mp4 for an end-to-end test)")
    print("\n" + "=" * 60 + "\nDONE\n" + "=" * 60)


if __name__ == "__main__":
    main()
