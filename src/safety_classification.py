"""
Safety-event prioritisation with an ANN (Keras).

IMPORTANT: this model classifies an *event record* (counts and durations) into
a predefined category. It is an event-prioritisation aid trained on SYNTHETIC
data - it is NOT an assessment of any person's risk or intent.

Artifacts (created in notebook 07):
    models/safety_ann.keras     Dense network, softmax over SAFETY_CLASSES
    models/safety_scaler.pkl    sklearn StandardScaler fitted on the training set
Feature order is SAFETY_FEATURES:
    duration_seconds, people_count, mask_issue_count, motion_change_count
"""
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import (
    SAFETY_ANN_PATH,
    SAFETY_CLASSES,
    SAFETY_CSV,
    SAFETY_FEATURES,
    SAFETY_SCALER_PATH,
    SAFETY_TARGET,
)

# Hand-written review-priority rule per category (not learned; edit as you like).
PRIORITY_BY_CATEGORY = {
    "normal_demo": "Low",
    "mask_observation_demo": "Medium",
    "long_group_event_demo": "Medium",
    "movement_observation_demo": "High",
}


def load_safety_data() -> Tuple[np.ndarray, np.ndarray]:
    """Return X (N, 4) float32 and y (N,) int labels (index into SAFETY_CLASSES)."""
    df = pd.read_csv(SAFETY_CSV)
    missing = [c for c in SAFETY_FEATURES + [SAFETY_TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {SAFETY_CSV.name}: {missing}")
    df = df.dropna(subset=SAFETY_FEATURES + [SAFETY_TARGET])
    mapping = {c: i for i, c in enumerate(SAFETY_CLASSES)}
    y = df[SAFETY_TARGET].map(mapping)
    if y.isna().any():
        raise ValueError("Unknown labels found: " + str(sorted(set(df[SAFETY_TARGET]) - set(mapping))))
    X = df[SAFETY_FEATURES].to_numpy(dtype="float32")
    return X, y.to_numpy(dtype="int64")


@lru_cache(maxsize=1)
def load_safety_artifacts():
    """Return (model, scaler)."""
    import joblib
    import tensorflow as tf

    for path in (SAFETY_ANN_PATH, SAFETY_SCALER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run notebook 07 first.")
    model = tf.keras.models.load_model(SAFETY_ANN_PATH, compile=False)
    scaler = joblib.load(SAFETY_SCALER_PATH)
    return model, scaler


def predict_safety(
    duration_seconds: float,
    people_count: int,
    mask_issue_count: int,
    motion_change_count: int,
) -> Dict:
    """Classify one event record."""
    values = [duration_seconds, people_count, mask_issue_count, motion_change_count]
    if any(v is None or v < 0 for v in values):
        raise ValueError("All safety features must be non-negative numbers.")

    model, scaler = load_safety_artifacts()
    x = scaler.transform(np.array([values], dtype="float32"))
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = SAFETY_CLASSES[idx]
    return {
        "label": label,
        "confidence": float(probs[idx]),
        "priority": PRIORITY_BY_CATEGORY[label],
        "probabilities": {c: float(p) for c, p in zip(SAFETY_CLASSES, probs)},
        "inputs": dict(zip(SAFETY_FEATURES, values)),
    }