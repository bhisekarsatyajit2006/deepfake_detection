"""
=============================================================================
  Deepfake Detection — Preprocessing (Kaggle, self-contained)
  Stage 1 of 2: video -> sampled frames -> MTCNN face crops -> frame index
  -------------------------------------------------------------------------
  Reads a FaceForensics++ metadata CSV ("File Path", "Label") and, per video:
    1. samples 1 frame every FRAME_STEP frames (max MAX_FRAMES)
    2. keeps the highest-confidence MTCNN face (conf >= MIN_CONF,
       >= MIN_FACE_PX), crops it with MARGIN padding
    3. resizes to TARGET_SIZE (Lanczos) and writes it as JPEG

  Output layout — consumed as-is by training/train_kaggle.py:
    <WORKING>/preprocessed/<label>/<video_stem>/frame_XXXXX.jpg
    <WORKING>/dataset_index.csv          frame index: frame_path,label,video_id,split
    <WORKING>/preprocess_progress.json   resume checkpoint (every SAVE_EVERY videos)

  Re-running resumes where it left off — the progress file is the source of
  truth, so an interrupted Kaggle session costs nothing.

  NOTE: This file is INTENTIONALLY standalone (it does not import the local
  `deepfake` package) because it is pasted into / run on a Kaggle kernel where
  that package is not installed. Its sampling and MTCNN constants duplicate
  deepfake/config.py ON PURPOSE — inference must crop faces exactly the way
  training saw them, so keep the two in sync.

  HOW TO RUN ON KAGGLE
    1. New Notebook (CPU is fine — MTCNN dominates, not the GPU).
    2. Add Data: the FaceForensics++ C23 dataset + its metadata CSV.
    3. Set CSV_FILENAME below, then Run All.
    4. Save /kaggle/working as a Kaggle dataset and add it to the training
       kernel (see training/train_kaggle.py, SOURCE="scan").

  SPLITTING THE WORK ACROSS SESSIONS
    A full run is longer than one Kaggle session. Give each session a slice of
    the CSV via ROW_OFFSET / ROW_LIMIT (session A: offset 0, limit 3500;
    session B: offset 3500, ...), publish each /kaggle/working as its own
    dataset, then list them all in the trainer — it merges and deduplicates.

  Without a /kaggle directory the script runs locally: reads from data/,
  writes to output/.
=============================================================================
"""

import subprocess
import sys
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()

if ON_KAGGLE:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "mtcnn", "opencv-python-headless", "tqdm"])

import json
import random
import shutil
import warnings

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ┌───────────────────────────────────────────────────────────────────────────┐
# │  USER CONFIG                                                              │
# └───────────────────────────────────────────────────────────────────────────┘
CSV_FILENAME = "FF++_Metadata_Shuffled.csv"   # searched recursively under DATASET_INPUT

# CSV slice — for splitting the work across several Kaggle sessions.
ROW_OFFSET = 0        # skip this many data rows (e.g. 3500 for the second session)
ROW_LIMIT = None      # process at most this many rows (None = all remaining)

# Optional quality filters (both OFF by default — the model was trained without
# them; turning one on changes the input distribution, so retrain if you do).
APPLY_GAUSSIAN = False   # light Gaussian blur, reduces MPEG blocking artefacts
APPLY_CLAHE = False      # adaptive histogram equalisation, evens out illumination

# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_INPUT = Path("/kaggle/input") if ON_KAGGLE else Path("data")
WORKING = Path("/kaggle/working") if ON_KAGGLE else Path("output")
PREPROCESSED_DIR = WORKING / "preprocessed"
PROGRESS_FILE = WORKING / "preprocess_progress.json"
INDEX_FILE = WORKING / "dataset_index.csv"

# ── Frame sampling + MTCNN — mirror of deepfake/config.py (keep in sync) ─────
FRAME_STEP = 10             # keep 1 frame every N (~3 fps at 30 fps)
MAX_FRAMES = 30             # hard cap per video
TARGET_SIZE = (224, 224)    # EfficientNet-B4 input resolution
MARGIN = 0.20               # padding around the detected box (20 %)
MIN_CONF = 0.95             # MTCNN minimum confidence
MIN_FACE_PX = 60            # minimum face width / height in pixels

SAVE_EVERY = 10             # checkpoint progress every N videos
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & VALIDATE THE METADATA CSV
# ═════════════════════════════════════════════════════════════════════════════
def find_csv() -> Path:
    candidates = sorted(DATASET_INPUT.rglob(CSV_FILENAME))
    if not candidates:
        raise FileNotFoundError(
            f"{CSV_FILENAME} not found under {DATASET_INPUT}.\n"
            "On Kaggle: add the dataset to this kernel. "
            "Locally: put it under data/.")
    if len(candidates) > 1:
        print(f"[WARN] Multiple CSVs found — using {candidates[0]}")
    return candidates[0]


CSV_PATH = find_csv()
print(f"CSV : {CSV_PATH}")

skiprows = range(1, ROW_OFFSET + 1) if ROW_OFFSET else None
meta = pd.read_csv(CSV_PATH, skiprows=skiprows, nrows=ROW_LIMIT)
meta.columns = [c.strip() for c in meta.columns]   # strip accidental spaces

missing = {"File Path", "Label"} - set(meta.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(meta.columns)}")

meta["label_norm"] = meta["Label"].str.strip().str.lower()

print(f"\nCSV loaded : {len(meta)} rows (offset={ROW_OFFSET}, limit={ROW_LIMIT})")
print("Label distribution:")
print(meta["label_norm"].value_counts().to_string())


# ═════════════════════════════════════════════════════════════════════════════
# 2.  HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def resolve_path(raw: str):
    """Locate a video: absolute, relative to the CSV's dataset dir, or by name."""
    p = Path(raw)
    if p.exists():
        return p
    alt = CSV_PATH.parent.parent / p
    if alt.exists():
        return alt
    matches = list(DATASET_INPUT.rglob(p.name))
    return matches[0] if matches else None


def extract_frames(video_path: Path, output_dir: Path) -> list:
    """Sample up to MAX_FRAMES frames (1 every FRAME_STEP) into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    saved_paths, count, saved = [], 0, 0
    while saved < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if count % FRAME_STEP == 0:
            out = output_dir / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(out), frame)
            saved_paths.append(out)
            saved += 1
        count += 1

    cap.release()
    return saved_paths


def build_detector():
    try:
        from mtcnn import MTCNN
        det = MTCNN()
        print("[INFO] MTCNN detector ready.")
        return det
    except Exception as e:
        print(f"[WARN] MTCNN unavailable ({e}). Frames will be resized without "
              "a face crop — the resulting dataset will NOT match what the "
              "detector sees at inference time.")
        return None


def crop_face(img_bgr, detector):
    """Best face + MARGIN, resized to TARGET_SIZE. None = discard this frame."""
    if detector is None:
        return cv2.resize(img_bgr, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

    h, w = img_bgr.shape[:2]
    detections = detector.detect_faces(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not detections:
        return None

    best = max(detections, key=lambda d: d["confidence"])
    if best["confidence"] < MIN_CONF:
        return None

    x, y, bw, bh = best["box"]
    x, y = max(0, x), max(0, y)          # MTCNN can return negative coords
    if bw < MIN_FACE_PX or bh < MIN_FACE_PX:
        return None

    pad_x, pad_y = int(bw * MARGIN), int(bh * MARGIN)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
    face = img_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None

    return cv2.resize(face, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)


def apply_gaussian_filter(img, kernel_size=(3, 3), sigma=0.5):
    """Light Gaussian smoothing — reduces MPEG blocking artefacts."""
    return cv2.GaussianBlur(img, kernel_size, sigma)


def apply_clahe(img_bgr):
    """Adaptive histogram equalisation on the L channel (LAB space)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        print(f"[RESUME] {len(data['processed_videos'])} videos already done.")
        return data
    return {"processed_videos": {}, "records": []}


def save_progress(progress: dict) -> None:
    """Persist atomically — a killed session must not corrupt the checkpoint."""
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f)
    tmp.replace(PROGRESS_FILE)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MAIN LOOP — frame extraction + face crop
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 1 — FRAME EXTRACTION + FACE CROP")
print("=" * 60)
print(f"  FRAME_STEP   = {FRAME_STEP}  (1 frame every {FRAME_STEP})")
print(f"  MAX_FRAMES   = {MAX_FRAMES}  per video")
print(f"  TARGET_SIZE  = {TARGET_SIZE[0]}x{TARGET_SIZE[1]} px (Lanczos)")
print(f"  MTCNN        conf >= {MIN_CONF} | min face = {MIN_FACE_PX} px "
      f"| margin = {int(MARGIN * 100)}%")
print(f"  Gaussian     = {APPLY_GAUSSIAN}   CLAHE = {APPLY_CLAHE}")
print(f"  Save every   = {SAVE_EVERY} videos")
print("=" * 60 + "\n")

detector = build_detector()
progress = load_progress()
processed_videos = progress["processed_videos"]   # {stem: {"kept": n, "rejected": n}}
records = progress["records"]                     # [{frame_path, label, video_id}]

total = len(meta)
videos_this_run = 0

for idx, (raw_path, label) in enumerate(
        tqdm(zip(meta["File Path"], meta["label_norm"]), total=total, desc="Videos")):

    video_path = resolve_path(raw_path)
    if video_path is None:
        tqdm.write(f"[SKIP] Not found: {raw_path}")
        continue

    # Parent name is part of the stem: FF++ reuses bare numbers across folders.
    video_stem = f"{video_path.parent.name}_{video_path.stem}"
    video_id = f"{label}/{video_stem}"

    if video_stem in processed_videos:      # done in an earlier run
        continue

    out_dir = PREPROCESSED_DIR / label / video_stem
    raw_dir = out_dir / "_raw"

    raw_frames = extract_frames(video_path, raw_dir)
    if not raw_frames:
        tqdm.write(f"[WARN] No frames extracted from {video_path.name}")
        processed_videos[video_stem] = {"kept": 0, "rejected": 0}
        videos_this_run += 1
        continue

    kept = rejected = 0
    for raw_frame_path in raw_frames:
        img_bgr = cv2.imread(str(raw_frame_path))
        if img_bgr is None:
            rejected += 1
            continue

        cropped = crop_face(img_bgr, detector)
        if cropped is None:                 # no confident, large-enough face
            rejected += 1
            continue

        if APPLY_GAUSSIAN:
            cropped = apply_gaussian_filter(cropped)
        if APPLY_CLAHE:
            cropped = apply_clahe(cropped)

        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / raw_frame_path.name
        cv2.imwrite(str(final_path), cropped)
        records.append({"frame_path": str(final_path),
                        "label": label,
                        "video_id": video_id})
        kept += 1

    shutil.rmtree(raw_dir, ignore_errors=True)   # raw frames were scratch space

    processed_videos[video_stem] = {"kept": kept, "rejected": rejected}
    videos_this_run += 1
    tqdm.write(f"[{idx + 1:>5}/{total}] {video_path.name:<40} "
               f"kept={kept}  rejected={rejected}")

    if videos_this_run % SAVE_EVERY == 0:
        progress["processed_videos"], progress["records"] = processed_videos, records
        save_progress(progress)
        tqdm.write(f"  [OK] Progress saved ({len(processed_videos)} videos, "
                   f"{len(records)} frames)")

progress["processed_videos"], progress["records"] = processed_videos, records
save_progress(progress)

print(f"\nTotal videos processed : {len(processed_videos)}")
print(f"Total valid frames     : {len(records)}")


# ═════════════════════════════════════════════════════════════════════════════
# 4.  BALANCE & SPLIT  (video-aware — a video never crosses splits)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2 — BALANCE & SPLIT (video-aware)")
print("=" * 60)

frames_df = pd.DataFrame(records)
if frames_df.empty:
    raise RuntimeError("No frames were saved. Check that 'File Path' in the CSV "
                       "points at accessible video files.")

real_vids = sorted(frames_df.loc[frames_df.label == "real", "video_id"].unique())
fake_vids = sorted(frames_df.loc[frames_df.label == "fake", "video_id"].unique())
print(f"Unique videos — real: {len(real_vids)}  fake: {len(fake_vids)}")

# Under-sample at VIDEO level (not frame level) to equalise the classes.
min_vids = min(len(real_vids), len(fake_vids))
random.Random(SEED).shuffle(real_vids)
random.Random(SEED).shuffle(fake_vids)
real_vids, fake_vids = real_vids[:min_vids], fake_vids[:min_vids]
print(f"After balance — {min_vids} videos per class")


def split_video_ids(vids: list) -> dict:
    n = len(vids)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    return {"train": set(vids[:n_train]),
            "val": set(vids[n_train:n_train + n_val]),
            "test": set(vids[n_train + n_val:])}


split_of_video = {}
for vids in (real_vids, fake_vids):
    for split_name, vid_set in split_video_ids(vids).items():
        for vid in vid_set:
            split_of_video[vid] = split_name

balanced = frames_df[frames_df.video_id.isin(split_of_video)].copy()
balanced["split"] = balanced.video_id.map(split_of_video)
balanced = balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)

print("\nSplit statistics:")
for split in ("train", "val", "test"):
    sub = balanced[balanced.split == split]
    print(f"  {split:5s} | {len(sub):>6} frames | "
          f"real={(sub.label == 'real').sum():>5}  "
          f"fake={(sub.label == 'fake').sum():>5}  "
          f"videos={sub.video_id.nunique()}")

balanced.to_csv(INDEX_FILE, index=False)
print(f"\nDataset index saved -> {INDEX_FILE}")
print("Columns:", list(balanced.columns))
print(balanced.head())


# ═════════════════════════════════════════════════════════════════════════════
# 5.  SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
total_kept = sum(v["kept"] for v in processed_videos.values())
total_rejected = sum(v["rejected"] for v in processed_videos.values())
seen = total_kept + total_rejected
rejection_rate = (total_rejected / seen * 100) if seen else 0.0

print("\n" + "=" * 60)
print("  PREPROCESSING SUMMARY")
print("=" * 60)
print(f"  Videos processed   : {len(processed_videos)}")
print(f"  Frames kept        : {total_kept}")
print(f"  Frames rejected    : {total_rejected}  ({rejection_rate:.1f}% — MTCNN filter)")
print(f"  Frames in index    : {len(balanced)}  (after balance)")
print(f"  Gaussian filter    : {'ON' if APPLY_GAUSSIAN else 'OFF'}")
print(f"  CLAHE              : {'ON' if APPLY_CLAHE else 'OFF'}")
print(f"  Output directory   : {PREPROCESSED_DIR}")
print(f"  Dataset index      : {INDEX_FILE}")
print(f"  Progress file      : {PROGRESS_FILE}")
print("=" * 60)
print("\nDone. Publish this working directory as a Kaggle dataset, then run "
      "training/train_kaggle.py against it.")
