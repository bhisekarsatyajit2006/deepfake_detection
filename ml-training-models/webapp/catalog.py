"""Video catalogue — browse VIDEOS_DIR and join optional CSV ground-truth."""

from __future__ import annotations

import csv
from pathlib import Path

from deepfake.config import Settings, settings

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class VideoCatalog:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.meta = self._load_csv_meta()

    def _load_csv_meta(self) -> dict:
        meta: dict[str, dict] = {}
        if not self.cfg.csv_path.exists():
            return meta
        with open(self.cfg.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fp = row.get("File Path", "").strip()
                if fp:
                    meta[Path(fp).name] = row
        return meta

    def folders(self) -> list[dict]:
        root = self.cfg.videos_dir
        if not root.exists():
            return []
        out = []
        for item in sorted(root.iterdir()):
            if item.is_dir():
                vids = [v for v in item.iterdir() if v.suffix.lower() in VIDEO_EXTS]
                out.append({"name": item.name, "count": len(vids)})
        return out

    def videos_in(self, folder_name: str) -> list[dict]:
        folder = self.cfg.videos_dir / folder_name
        if not folder.exists():
            return []
        videos = []
        for v in sorted(folder.iterdir()):
            if v.suffix.lower() not in VIDEO_EXTS:
                continue
            m = self.meta.get(v.name, {})
            videos.append({
                "name": v.name,
                "path": str(v.relative_to(self.cfg.videos_dir)),
                "label": m.get("Label", "unknown").lower(),
                "frame_count": m.get("Frame Count", "—"),
                "width": m.get("Width", "—"),
                "height": m.get("Height", "—"),
                "codec": m.get("Codec", "—"),
                "size_mb": m.get("File Size(MB)", "—"),
            })
        return videos
