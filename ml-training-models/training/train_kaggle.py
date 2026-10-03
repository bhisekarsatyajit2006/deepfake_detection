"""
=============================================================================
  Deepfake Detection — Training (Kaggle, self-contained)
  EfficientNet-B4 · two-phase fine-tuning · video-aware split
  -------------------------------------------------------------------------
  Replaces the old deepfake_pipeline.py + deepfake_train.py. Those two only
  differed in how they discovered the frame list; this script unifies them
  behind SOURCE below.

  NOTE: This file is INTENTIONALLY standalone (it does not import the local
  `deepfake` package) because it is pasted into / run on a Kaggle kernel where
  that package is not installed. Its model definition duplicates
  deepfake/model.py ON PURPOSE — keep the two in sync if you change the
  architecture.

  HOW TO RUN ON KAGGLE
    1. New Notebook, Accelerator = GPU (T4 x2 / P100).
    2. Add Data: the datasets holding the preprocessed face crops
       (and, for the json/csv sources, the checkpoint/index dataset).
    3. Set SOURCE below, then Run All.
    4. Download /kaggle/working/efficientnet_b4_deepfake_final.pth from Output.

  SOURCE options
    "scan" : glob <dataset>/preprocessed/<label>/<video>/*.jpg directly.
             Simplest — needs only the image datasets. (Recommended.)
    "json" : merge two preprocess_progress.json checkpoints (their "records").
    "csv"  : merge two dataset_index.csv files.
=============================================================================
"""

import subprocess
import sys


def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *pkgs])


pip_install("timm", "albumentations", "opencv-python-headless",
            "scikit-learn", "matplotlib", "seaborn", "tqdm")

import json
import random
import inspect
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve,
)

warnings.filterwarnings("ignore")

# ┌───────────────────────────────────────────────────────────────────────────┐
# │  USER CONFIG                                                                │
# └───────────────────────────────────────────────────────────────────────────┘
SOURCE = "scan"          # "scan" | "json" | "csv"

# For SOURCE="json": two preprocess_progress.json checkpoints
CHECKPOINT_A = Path("/kaggle/input/checkpoint-trima/preprocess_progress.json")
CHECKPOINT_B = Path("/kaggle/input/checkpoint-trima-2/preprocess_progress.json")

# For SOURCE="csv": two dataset_index.csv files + their image dataset slugs
CSV_A = Path("/kaggle/input/preprocess-trima-1/dataset_index.csv")
CSV_B = Path("/kaggle/input/preprocess-trima-2/dataset_index.csv")

# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no GPU — training will be very slow. Set Accelerator = GPU.")

KAGGLE_INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
CKPT_DIR = WORKING / "checkpoints"
FIG_DIR = WORKING / "figures"
for d in (CKPT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

CFG = {
    "model_name": "efficientnet_b4",
    "img_size": 224,
    "batch_size": 64 if DEVICE.type == "cuda" else 16,
    "epochs_phase1": 10,
    "lr_phase1": 1e-4,
    "epochs_phase2": 40,
    "lr_phase2": 1e-5,
    "weight_decay": 1e-2,
    "dropout": 0.5,
    "label_smoothing": 0.05,
    "grad_clip": 1.0,
    "use_amp": DEVICE.type == "cuda",
    "early_stop_patience": 8,
    "num_workers": 2,
}
print(json.dumps({k: str(v) for k, v in CFG.items()}, indent=2))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = CFG["img_size"]


# ═════════════════════════════════════════════════════════════════════════════
# 1.  BUILD THE FRAME INDEX  (three interchangeable sources)
# ═════════════════════════════════════════════════════════════════════════════
def build_from_scan() -> pd.DataFrame:
    roots = [p for p in KAGGLE_INPUT.rglob("preprocessed") if p.is_dir()]
    if not roots:
        raise FileNotFoundError(
            "No 'preprocessed/' folder under /kaggle/input. Add the image datasets.")
    print("Scanning:", *[f"\n  {r}" for r in roots])
    rows = []
    for root in roots:
        for label in ("real", "fake"):
            ldir = root / label
            if not ldir.is_dir():
                continue
            for vdir in ldir.iterdir():
                if vdir.is_dir():
                    vid = f"{label}/{vdir.name}"
                    for frame in vdir.glob("*.jpg"):
                        rows.append((str(frame), label, vid))
    return pd.DataFrame(rows, columns=["frame_path", "label", "video_id"])


def build_from_json() -> pd.DataFrame:
    def load(path):
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        recs = json.loads(path.read_text()).get("records", [])
        print(f"  {path} -> {len(recs)} records")
        return recs
    recs = load(CHECKPOINT_A) + load(CHECKPOINT_B)
    df = pd.DataFrame(recs)[["frame_path", "label", "video_id"]]
    return df.drop_duplicates(subset=["frame_path"])


def build_from_csv() -> pd.DataFrame:
    def load(path):
        if not path.exists():
            raise FileNotFoundError(f"Index not found: {path}")
        d = pd.read_csv(path)
        print(f"  {path} -> {len(d)} rows")
        return d
    df = pd.concat([load(CSV_A), load(CSV_B)], ignore_index=True)
    return df[["frame_path", "label", "video_id"]].drop_duplicates(subset=["frame_path"])


print("\n" + "=" * 60 + f"\n  STEP 1 — BUILD INDEX (source={SOURCE})\n" + "=" * 60)
df = {"scan": build_from_scan, "json": build_from_json, "csv": build_from_csv}[SOURCE]()

# Keep only frames whose files actually exist on disk.
before = len(df)
df = df[df["frame_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
if len(df) < before:
    print(f"  Dropped {before - len(df)} rows with missing image files.")
if df.empty:
    raise RuntimeError("No usable frames found — check that the image datasets are added.")

print(f"\nTotal frames : {len(df):,}")
print(df["label"].value_counts().to_string())


# ═════════════════════════════════════════════════════════════════════════════
# 2.  BALANCE & SPLIT  (video-aware — a video never crosses splits)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60 + "\n  STEP 2 — BALANCE & SPLIT\n" + "=" * 60)
real_vids = df[df.label == "real"]["video_id"].unique().tolist()
fake_vids = df[df.label == "fake"]["video_id"].unique().tolist()
print(f"  Unique videos — real: {len(real_vids)}  fake: {len(fake_vids)}")

min_vids = min(len(real_vids), len(fake_vids))
random.Random(SEED).shuffle(real_vids)
random.Random(SEED).shuffle(fake_vids)
real_vids, fake_vids = real_vids[:min_vids], fake_vids[:min_vids]
print(f"  After balance  — {min_vids} videos per class")


def split_video_ids(vids):
    n = len(vids)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    return {"train": set(vids[:n_train]),
            "val": set(vids[n_train:n_train + n_val]),
            "test": set(vids[n_train + n_val:])}


real_splits, fake_splits = split_video_ids(real_vids), split_video_ids(fake_vids)


def assign_split(row):
    splits = real_splits if row["label"] == "real" else fake_splits
    for name, vid_set in splits.items():
        if row["video_id"] in vid_set:
            return name
    return None


balanced = df[df.video_id.isin(set(real_vids) | set(fake_vids))].copy()
balanced["split"] = balanced.apply(assign_split, axis=1)
balanced = balanced[balanced.split.notna()].reset_index(drop=True)

for split in ("train", "val", "test"):
    sub = balanced[balanced.split == split]
    print(f"  {split:5s} | {len(sub):>6} frames | "
          f"real={(sub.label == 'real').sum():>5}  fake={(sub.label == 'fake').sum():>5}  "
          f"videos={sub.video_id.nunique()}")
balanced.to_csv(WORKING / "merged_dataset_index.csv", index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  TRANSFORMS
# ═════════════════════════════════════════════════════════════════════════════
_ic = inspect.signature(A.ImageCompression.__init__).parameters
_gn = inspect.signature(A.GaussNoise.__init__).parameters
ImageCompression = (A.ImageCompression(quality_range=(40, 95), p=0.4)
                    if "quality_range" in _ic
                    else A.ImageCompression(quality_lower=40, quality_upper=95, p=0.4))
GaussNoise = (A.GaussNoise(std_range=(0.04, 0.25), p=0.3)
              if "std_range" in _gn
              else A.GaussNoise(var_limit=(10.0, 80.0), p=0.3))

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.4),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.2),
    A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=0.2),
    ImageCompression, GaussNoise,
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(max_holes=6, max_height=20, max_width=20,
                    min_holes=1, fill_value=0, p=0.2),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


# ═════════════════════════════════════════════════════════════════════════════
# 4.  DATASET / LOADERS
# ═════════════════════════════════════════════════════════════════════════════
class DeepfakeDataset(Dataset):
    LABEL_MAP = {"real": 0, "fake": 1}

    def __init__(self, frame_df, transform=None):
        self.df = frame_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["frame_path"])
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        label = torch.tensor(self.LABEL_MAP[row["label"]], dtype=torch.float32)
        return img, label, row["video_id"]


def make_loaders(b):
    kw = dict(num_workers=CFG["num_workers"], pin_memory=(DEVICE.type == "cuda"))
    return (
        DataLoader(DeepfakeDataset(b[b.split == "train"], train_transform),
                   batch_size=CFG["batch_size"], shuffle=True, **kw),
        DataLoader(DeepfakeDataset(b[b.split == "val"], val_transform),
                   batch_size=CFG["batch_size"], shuffle=False, **kw),
        DataLoader(DeepfakeDataset(b[b.split == "test"], val_transform),
                   batch_size=CFG["batch_size"], shuffle=False, **kw),
    )


train_loader, val_loader, test_loader = make_loaders(balanced)
print(f"\nBatches — train {len(train_loader)} | val {len(val_loader)} | test {len(test_loader)}")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  MODEL  (mirror of deepfake/model.py — keep in sync)
# ═════════════════════════════════════════════════════════════════════════════
class DeepfakeDetector(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=True, num_classes=0, global_pool="avg")
        self.head = nn.Sequential(nn.Dropout(p=dropout),
                                  nn.Linear(self.backbone.num_features, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_mbconv(self, n=2):
        for m in (self.backbone.bn2, self.backbone.conv_head):
            for p in m.parameters():
                p.requires_grad = True
        for stage in list(self.backbone.blocks)[-n:]:
            for p in stage.parameters():
                p.requires_grad = True


model = DeepfakeDetector(CFG["dropout"]).to(DEVICE)
print(f"Total params : {sum(p.numel() for p in model.parameters()):,}")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  LOSS / AMP / TRAIN + EVAL HELPERS
# ═════════════════════════════════════════════════════════════════════════════
class BCESmooth(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.s = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets * (1 - self.s) + 0.5 * self.s)


criterion = BCESmooth(CFG["label_smoothing"])
scaler = torch.amp.GradScaler("cuda", enabled=CFG["use_amp"])


def amp_ctx():
    return torch.amp.autocast("cuda", enabled=CFG["use_amp"])


def train_epoch(model, loader, optimizer, scheduler=None):
    model.train()
    total_loss, labels_all, probs_all = 0.0, [], []
    for imgs, labels, _ in tqdm(loader, desc="  train", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with amp_ctx():
            logits = model(imgs)
            loss = criterion(logits, labels)
        labels_all.extend(labels.detach().cpu().numpy())
        probs_all.extend(torch.sigmoid(logits).detach().float().cpu().numpy())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], CFG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()
        total_loss += loss.item() * len(imgs)
    auc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.0
    return total_loss / len(loader.dataset), auc


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, labels_all, probs_all, vids_all = 0.0, [], [], []
    for imgs, labels, vids in tqdm(loader, desc="  eval ", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with amp_ctx():
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item() * len(imgs)
        labels_all.extend(labels.cpu().numpy())
        probs_all.extend(torch.sigmoid(logits).float().cpu().numpy())
        vids_all.extend(vids)
    auc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.0
    return total_loss / len(loader.dataset), auc, probs_all, labels_all, vids_all


def best_threshold(labels, probs):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(labels, (np.array(probs) >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


# ═════════════════════════════════════════════════════════════════════════════
# 7.  TRAINING  (two-phase, early stopping on val AUC)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60 + "\n  STEP 3 — TRAINING\n" + "=" * 60)
history = defaultdict(list)
best_val_auc, clf_threshold, no_improve = 0.0, 0.5, 0


def run_phase(phase, epochs, lr):
    global best_val_auc, clf_threshold, no_improve
    print(f"\n-- Phase {phase} " + "-" * 48)
    if phase == 1:
        model.freeze_backbone(); print("  Backbone frozen — head only")
    else:
        model.unfreeze_last_mbconv(2); print("  Unfrozen last 2 MBConv blocks + head")
    print(f"  Trainable params : "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader))

    for epoch in range(1, epochs + 1):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_auc, val_probs, val_labels, _ = evaluate(model, val_loader)
        thr = best_threshold(val_labels, val_probs)
        val_f1 = f1_score(val_labels, (np.array(val_probs) >= thr).astype(int),
                          zero_division=0)
        history["phase"].append(phase); history["epoch"].append(epoch)
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc); history["val_auc"].append(val_auc)
        history["threshold"].append(thr)
        print(f"  [P{phase} E{epoch:03d}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} train_auc={train_auc:.4f} "
              f"val_auc={val_auc:.4f} val_f1={val_f1:.4f} thr={thr:.3f}")
        if val_auc > best_val_auc:
            best_val_auc, clf_threshold, no_improve = val_auc, thr, 0
            torch.save({"state_dict": model.state_dict(), "cfg": CFG,
                        "best_val_auc": float(best_val_auc),
                        "best_threshold": float(clf_threshold),
                        "epoch": epoch, "phase": phase},
                       CKPT_DIR / "best_model.pth")
            print(f"  [saved] best_model.pth (val_auc={best_val_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= CFG["early_stop_patience"]:
                print(f"  Early stop after {no_improve} epochs w/o improvement.")
                break


run_phase(1, CFG["epochs_phase1"], CFG["lr_phase1"])
no_improve = 0
run_phase(2, CFG["epochs_phase2"], CFG["lr_phase2"])


# ═════════════════════════════════════════════════════════════════════════════
# 8.  TEST EVALUATION  (frame-level + video-level mean aggregation)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60 + "\n  STEP 4 — TEST EVALUATION\n" + "=" * 60)
ckpt = torch.load(CKPT_DIR / "best_model.pth", map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"])
clf_threshold = ckpt["best_threshold"]
print(f"Loaded best: phase={ckpt['phase']} epoch={ckpt['epoch']} "
      f"val_auc={ckpt['best_val_auc']:.4f} thr={clf_threshold:.3f}")

_, test_auc, test_probs, test_labels, test_vids = evaluate(model, test_loader)
test_preds = (np.array(test_probs) >= clf_threshold).astype(int)
frame_metrics = {
    "auc": float(test_auc),
    "acc": float(accuracy_score(test_labels, test_preds)),
    "precision": float(precision_score(test_labels, test_preds, zero_division=0)),
    "recall": float(recall_score(test_labels, test_preds, zero_division=0)),
    "f1": float(f1_score(test_labels, test_preds, zero_division=0)),
}

vid_buckets = defaultdict(lambda: {"probs": [], "label": None})
for prob, label, vid in zip(test_probs, test_labels, test_vids):
    vid_buckets[vid]["probs"].append(prob)
    vid_buckets[vid]["label"] = int(label)
vid_scores = np.array([np.mean(v["probs"]) for v in vid_buckets.values()])
vid_labels = np.array([v["label"] for v in vid_buckets.values()])
vid_preds = (vid_scores >= clf_threshold).astype(int)
video_metrics = {
    "auc": float(roc_auc_score(vid_labels, vid_scores)) if len(set(vid_labels)) > 1 else 0.0,
    "acc": float(accuracy_score(vid_labels, vid_preds)),
    "precision": float(precision_score(vid_labels, vid_preds, zero_division=0)),
    "recall": float(recall_score(vid_labels, vid_preds, zero_division=0)),
    "f1": float(f1_score(vid_labels, vid_preds, zero_division=0)),
}
print("\nFrame-level:")
for k, v in frame_metrics.items():
    print(f"  {k:<10}: {v:.4f}")
print("\nVideo-level:")
for k, v in video_metrics.items():
    print(f"  {k:<10}: {v:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# 9.  FIGURES
# ═════════════════════════════════════════════════════════════════════════════
def save_fig(name):
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {FIG_DIR / name}")


hist_df = pd.DataFrame(history)
for phase in (1, 2):
    ph = hist_df[hist_df.phase == phase]
    if ph.empty:
        continue
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot(ph.epoch, ph.train_loss, label="Train")
    ax[0].plot(ph.epoch, ph.val_loss, label="Val")
    ax[0].set_title(f"Phase {phase} — Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ph.epoch, ph.train_auc, label="Train AUC")
    ax[1].plot(ph.epoch, ph.val_auc, color="tab:orange", label="Val AUC")
    ax[1].set_title(f"Phase {phase} — AUC"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); save_fig(f"curves_phase{phase}.png")

cm = confusion_matrix(vid_labels, vid_preds)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix — Video Level")
plt.tight_layout(); save_fig("confusion_matrix.png")

if len(set(vid_labels)) > 1:
    fpr, tpr, _ = roc_curve(vid_labels, vid_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={video_metrics['auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC — Video Level"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); save_fig("roc_curve.png")

# Score distribution — shows how well the two classes separate, and where the
# tuned threshold sits between them.
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(vid_scores[vid_labels == 0], bins=30, alpha=.6, color="green", label="REAL")
ax.hist(vid_scores[vid_labels == 1], bins=30, alpha=.6, color="red", label="FAKE")
ax.axvline(clf_threshold, color="black", ls="--", lw=1.5,
           label=f"Threshold={clf_threshold:.3f}")
ax.set_xlabel("P(fake)"); ax.set_ylabel("Videos")
ax.set_title("Score Distribution — Video Level")
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); save_fig("score_distribution.png")


# ═════════════════════════════════════════════════════════════════════════════
# 10.  EXPORT  (filename matches the app's default MODEL_PATH)
# ═════════════════════════════════════════════════════════════════════════════
with open(CKPT_DIR / "metrics.json", "w") as f:
    json.dump({"frame": frame_metrics, "video": video_metrics,
               "history": dict(history)}, f, indent=2)

final = WORKING / "efficientnet_b4_deepfake_final.pth"
torch.save({"state_dict": model.state_dict(), "cfg": CFG,
            "best_val_auc": float(best_val_auc),
            "best_threshold": float(clf_threshold),
            "frame_metrics": frame_metrics,
            "video_metrics": video_metrics}, final)
print(f"\nFINAL MODEL -> {final}")
print(f"  best val AUC = {best_val_auc:.4f}   threshold = {clf_threshold:.3f}")
print("  Download it from the Output panel and place it in the local models/ folder.")

# Optional ONNX export — for runtimes without PyTorch. The Flask app uses the
# .pth above; this is a convenience, so a failure here must not fail the run.
try:
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    onnx_path = WORKING / "efficientnet_b4_deepfake.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["image"], output_names=["logit"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX        -> {onnx_path}")
except Exception as e:
    print(f"ONNX export skipped: {e}")

print("Done.")
