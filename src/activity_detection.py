"""
Human activity recognition: MobileNetV2 (CNN) features  ->  LSTM / BiLSTM.

Pipeline
  1. Sample frames from a video (or read pre-extracted frames).
  2. MobileNetV2 (ImageNet weights, frozen) turns every frame into a
     1280-d feature vector  -> CNN part.
  3. Feature vectors are cut into windows of ACTIVITY_WINDOW frames.
  4. The LSTM (activity_cnn_lstm.keras) or BiLSTM (activity_bilstm.keras)
     classifies each window; window probabilities are averaged per video.

Model contract (created in notebooks 04 / 05):
  * input  : float32 (N, ACTIVITY_WINDOW, 1280)
  * output : softmax over ACTIVITY_CLASSES ["Sitting", "Standing", "Walking"]

NOTE: the supplied dataset has only 4 videos, so this module is a working
proof-of-concept pipeline, not a statistically meaningful model.
"""
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from src.config import (
    ACTIVITY_BILSTM_PATH,
    ACTIVITY_CLASSES,
    ACTIVITY_CNN_LSTM_PATH,
    ACTIVITY_DIR,
    ACTIVITY_FRAMES_DIR,
    ACTIVITY_IMG_SIZE,
    ACTIVITY_LABELS_CSV,
    ACTIVITY_NUM_FRAMES,
    ACTIVITY_WINDOW,
    ACTIVITY_WINDOW_STRIDE,
)

MODEL_PATHS = {
    "cnn_lstm": ACTIVITY_CNN_LSTM_PATH,
    "bilstm": ACTIVITY_BILSTM_PATH,
}


# ------------------------------------------------------------------ frames
def read_video_frames(video_path, num_frames: int = ACTIVITY_NUM_FRAMES) -> List[np.ndarray]:
    """Read `num_frames` evenly spaced BGR frames (resized) from a video file."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: Dict[int, np.ndarray] = {}

    if total > 0:
        wanted = np.linspace(0, total - 1, num_frames).round().astype(int)
        wanted_set = set(wanted.tolist())
        last = int(wanted.max())
        i = 0
        while i <= last:
            if not cap.grab():
                break
            if i in wanted_set:
                ok, frame = cap.retrieve()
                if ok:
                    frames[i] = cv2.resize(frame, ACTIVITY_IMG_SIZE)
            i += 1
        cap.release()
        ordered = [frames[int(k)] for k in wanted if int(k) in frames]
    else:  # frame count unknown -> read everything, then sample
        all_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            all_frames.append(cv2.resize(frame, ACTIVITY_IMG_SIZE))
        cap.release()
        if not all_frames:
            raise ValueError(f"No frames could be read from: {video_path}")
        idx = np.linspace(0, len(all_frames) - 1, num_frames).round().astype(int)
        ordered = [all_frames[k] for k in idx]

    if not ordered:
        raise ValueError(f"No frames could be read from: {video_path}")
    return ordered


def read_frames_folder(folder) -> List[np.ndarray]:
    """Read frame_00.jpg, frame_01.jpg, ... from a folder (sorted by name)."""
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    frames = []
    for p in files:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(cv2.resize(img, ACTIVITY_IMG_SIZE))
    if not frames:
        raise ValueError(f"No readable frames in: {folder}")
    return frames


def find_frames_folder(video_path):
    """Folders in datasets/human_activity/frames are named '<prefix>__<video stem>'."""
    stem = Path(video_path).stem
    if ACTIVITY_FRAMES_DIR.exists():
        for d in ACTIVITY_FRAMES_DIR.iterdir():
            if d.is_dir() and d.name.endswith("__" + stem):
                return d
    return None


# ---------------------------------------------------------------- features
@lru_cache(maxsize=1)
def get_feature_extractor():
    """Frozen MobileNetV2 -> one 1280-d vector per frame (downloads weights once)."""
    import tensorflow as tf

    return tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(*ACTIVITY_IMG_SIZE, 3),
    )


def extract_features(frames_bgr: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
    """List of BGR frames -> float32 array (n_frames, 1280)."""
    import tensorflow as tf

    if not frames_bgr:
        raise ValueError("No frames given.")
    batch = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, ACTIVITY_IMG_SIZE, interpolation=cv2.INTER_AREA)
        batch.append(rgb.astype("float32"))
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(np.stack(batch))
    feats = get_feature_extractor().predict(batch, batch_size=batch_size, verbose=0)
    return feats.astype("float32")


def make_windows(
    features: np.ndarray,
    window: int = ACTIVITY_WINDOW,
    stride: int = ACTIVITY_WINDOW_STRIDE,
) -> np.ndarray:
    """(T, D) -> (num_windows, window, D). Short sequences are padded by repeating the last frame."""
    features = np.asarray(features, dtype="float32")
    if features.ndim != 2:
        raise ValueError("features must have shape (T, D).")
    if len(features) < window:
        pad = np.repeat(features[-1:], window - len(features), axis=0)
        features = np.concatenate([features, pad], axis=0)
    starts = range(0, len(features) - window + 1, stride)
    return np.stack([features[s:s + window] for s in starts], axis=0)


def load_activity_dataset(
    window: int = ACTIVITY_WINDOW,
    stride: int = ACTIVITY_WINDOW_STRIDE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the training set from activity_labels.csv.

    Returns
      X      (N, window, 1280) float32
      y      (N,) int labels (index into ACTIVITY_CLASSES)
      groups (N,) source_group name -> use it to split by video, never by window
    """
    df = pd.read_csv(ACTIVITY_LABELS_CSV)
    X, y, groups = [], [], []
    for _, row in df.iterrows():
        video_path = ACTIVITY_DIR / row["video_path"]
        folder = find_frames_folder(video_path)
        frames = read_frames_folder(folder) if folder else read_video_frames(video_path)
        windows = make_windows(extract_features(frames), window, stride)
        X.append(windows)
        y.extend([ACTIVITY_CLASSES.index(row["label"])] * len(windows))
        groups.extend([row["source_group"]] * len(windows))
    return np.concatenate(X, axis=0), np.array(y, dtype="int64"), np.array(groups)


# --------------------------------------------------------------- inference
@lru_cache(maxsize=2)
def load_activity_model(model_type: str = "cnn_lstm"):
    import tensorflow as tf

    if model_type not in MODEL_PATHS:
        raise ValueError(f"model_type must be one of {list(MODEL_PATHS)}")
    path = MODEL_PATHS[model_type]
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Train it in the notebooks first.")
    return tf.keras.models.load_model(path, compile=False)


def predict_frames(frames_bgr: List[np.ndarray], model_type: str = "cnn_lstm") -> Dict:
    """Classify an activity from a list of BGR frames."""
    windows = make_windows(extract_features(frames_bgr))
    probs = load_activity_model(model_type).predict(windows, verbose=0)
    mean_probs = probs.mean(axis=0)
    idx = int(np.argmax(mean_probs))
    return {
        "label": ACTIVITY_CLASSES[idx],
        "confidence": float(mean_probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(ACTIVITY_CLASSES, mean_probs)},
        "num_frames": len(frames_bgr),
        "num_windows": int(len(windows)),
        "model_type": model_type,
    }


def predict_video(video_path, model_type: str = "cnn_lstm") -> Dict:
    """Classify the activity in a video file."""
    frames = read_video_frames(video_path, ACTIVITY_NUM_FRAMES)
    return predict_frames(frames, model_type)