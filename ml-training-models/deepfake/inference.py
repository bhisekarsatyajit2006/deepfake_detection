"""End-to-end inference: video → sampled frames → MTCNN face crops →
EfficientNet-B4 → per-video FAKE/REAL verdict.

All the heavy objects (model, MTCNN detector, transform) are created once in
``DeepfakeAnalyzer.__init__`` and reused across requests.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import torch
# mtcnn replaced with OpenCV Haar cascade to avoid TensorFlow dependency

from deepfake.config import Settings, settings
from deepfake.model import load_model
from deepfake.transforms import build_val_transform


class DeepfakeAnalyzer:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        print("[Startup] Loading model...")
        self.model, self.threshold = load_model(cfg)
        print(f"[Startup] Threshold = {self.threshold:.4f}")
        self.transform = build_val_transform(cfg)
        print("[Startup] Using center-crop face extractor (no TF required).")

    # ── Face detection & cropping (CR2 pipeline) ────────────────────────────
    def crop_face(self, img_bgr):
        """BGR frame → 224×224 center crop (proxy for face region).

        Uses a square center crop covering the middle 70% of the frame.
        This works well for typical deepfake videos where the face is
        centered. No face-detection library required.
        """
        h, w = img_bgr.shape[:2]
        side = int(min(h, w) * 0.70)
        if side < self.cfg.min_face_px:
            return None
        cy, cx = h // 2, w // 2
        y1 = max(0, cy - side // 2)
        y2 = min(h, y1 + side)
        x1 = max(0, cx - side // 2)
        x2 = min(w, x1 + side)
        face = img_bgr[y1:y2, x1:x2]
        if face.size == 0:
            return None
        return cv2.resize(face, (self.cfg.img_size, self.cfg.img_size),
                          interpolation=cv2.INTER_LANCZOS4)

    # ── Single-face inference ───────────────────────────────────────────────
    @torch.no_grad()
    def infer_face(self, face_bgr) -> float:
        """BGR face crop → P(fake). Mirrors the training val pipeline."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=face_rgb)["image"]
        tensor = tensor.unsqueeze(0).to(self.cfg.device)
        logit = self.model(tensor)
        return logit.item()

    # ── Whole-video analysis ────────────────────────────────────────────────
    def analyze_video(self, video_path: Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Cannot open video: {video_path.name}"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Step 1 — sample every Nth frame, up to max_frames
        sampled, count = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % self.cfg.frame_step == 0:
                sampled.append(frame)
                if self.cfg.max_frames and len(sampled) >= self.cfg.max_frames:
                    break
            count += 1
        cap.release()
        n_extracted = len(sampled)

        # Step 2 — face detection & crop
        face_crops = [c for c in (self.crop_face(f) for f in sampled) if c is not None]
        n_faces = len(face_crops)
        n_rejected = n_extracted - n_faces

        video_info = {"fps": round(fps, 1), "width": width,
                      "height": height, "total_frames": total_frames}

        if n_faces == 0:
            return {
                "error": "No faces detected — video may have no clear frontal "
                         "face, or confidence threshold is too high",
                "frames_extracted": n_extracted,
                "faces_detected": 0,
                "frames_rejected": n_rejected,
                "video_info": video_info,
            }

        # Step 3 — classify each face
        probs = [self.infer_face(face) for face in face_crops]
        avg_prob = float(np.mean(probs))
        max_prob = float(np.max(probs))
        std_prob = float(np.std(probs))
        prediction = "REAL" if avg_prob >= self.threshold else "FAKE"
        confidence = avg_prob if prediction == "REAL" else (1.0 - avg_prob)

        # Step 4 — up to 12 evenly-spaced face thumbnails
        frame_results = []
        for i in np.linspace(0, n_faces - 1, min(12, n_faces), dtype=int):
            p = probs[i]
            _, buf = cv2.imencode(".jpg", face_crops[i], [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_results.append({
                "prob": round(p, 4),
                "label": "REAL" if p >= self.threshold else "FAKE",
                "thumb": base64.b64encode(buf).decode(),
            })

        buckets = [0] * 10
        for p in probs:
            buckets[min(int(p * 10), 9)] += 1

        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 1),
            "threshold": round(self.threshold, 4),
            "avg_prob": round(avg_prob, 4),
            "max_prob": round(max_prob, 4),
            "std_prob": round(std_prob, 4),
            "frames_extracted": n_extracted,
            "faces_detected": n_faces,
            "frames_rejected": n_rejected,
            "all_probs": [round(p, 4) for p in probs],
            "prob_buckets": buckets,
            "frame_results": frame_results,
            "video_info": {
                **video_info,
                "duration_s": round(total_frames / fps, 1) if fps > 0 else 0,
            },
        }
