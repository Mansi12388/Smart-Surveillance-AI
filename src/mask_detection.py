"""
Face-mask classification (inference).

Model contract (created in notebooks 02 / 03):
  * input  : float32 RGB image, shape (N, 224, 224, 3), pixel values 0-255
             (the saved model contains its own Rescaling layer, so NO manual
              normalisation is done here)
  * output : softmax over MASK_CLASSES
             ["mask_weared_incorrect", "with_mask", "without_mask"]
"""
from functools import lru_cache
from typing import Dict, List

import cv2
import numpy as np

from src.config import (
    MASK_CLASSES,
    MASK_CNN_PATH,
    MASK_IMG_SIZE,
    MASK_MOBILENET_PATH,
)
from src.face_detection import crop_face, detect_faces, ensure_bgr

LABEL_COLORS = {                      # BGR
    "with_mask": (0, 200, 0),
    "without_mask": (0, 0, 255),
    "mask_weared_incorrect": (0, 165, 255),
}


def get_model_path():
    """Prefer the MobileNetV2 model; fall back to the scratch CNN."""
    if MASK_MOBILENET_PATH.exists():
        return MASK_MOBILENET_PATH
    if MASK_CNN_PATH.exists():
        return MASK_CNN_PATH
    raise FileNotFoundError(
        "No trained mask model found. Run notebook 02 or 03 first. Expected:\n"
        f"  {MASK_MOBILENET_PATH}\n  {MASK_CNN_PATH}"
    )


@lru_cache(maxsize=1)
def load_mask_model():
    import tensorflow as tf  # imported lazily: keeps imports fast

    return tf.keras.models.load_model(get_model_path(), compile=False)


def preprocess_faces(faces_bgr: List[np.ndarray]) -> np.ndarray:
    """List of BGR face crops -> float32 RGB batch (N, 224, 224, 3), range 0-255."""
    batch = []
    for face in faces_bgr:
        face = ensure_bgr(face)
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, MASK_IMG_SIZE, interpolation=cv2.INTER_AREA)
        batch.append(rgb.astype("float32"))
    return np.stack(batch, axis=0)


def classify_faces(faces_bgr: List[np.ndarray]) -> List[Dict]:
    """Classify a list of face crops in a single model call."""
    if not faces_bgr:
        return []
    probs = load_mask_model().predict(preprocess_faces(faces_bgr), verbose=0)
    results = []
    for p in probs:
        idx = int(np.argmax(p))
        results.append(
            {
                "label": MASK_CLASSES[idx],
                "confidence": float(p[idx]),
                "probabilities": {c: float(v) for c, v in zip(MASK_CLASSES, p)},
            }
        )
    return results


def classify_face(face_bgr: np.ndarray) -> Dict:
    """Classify one already-cropped face image."""
    return classify_faces([face_bgr])[0]


def detect_and_classify(image_bgr: np.ndarray, use_face_detector: bool = True) -> List[Dict]:
    """Find faces, then classify each one.

    If no face is found (or use_face_detector=False) the whole image is treated
    as one face crop. This matters because Haar cascades often fail on masked
    faces, and the training images are already tight face crops.
    """
    image_bgr = ensure_bgr(image_bgr)
    h, w = image_bgr.shape[:2]

    boxes = detect_faces(image_bgr) if use_face_detector else []
    face_detected = len(boxes) > 0
    if not face_detected:
        boxes = [(0, 0, w, h)]

    crops = [crop_face(image_bgr, b) if face_detected else image_bgr for b in boxes]
    predictions = classify_faces(crops)

    detections = []
    for box, pred in zip(boxes, predictions):
        detections.append({"box": list(box), "face_detected": face_detected, **pred})
    return detections


def summarize_detections(detections: List[Dict]) -> Dict:
    """Counts used by the dashboard and by the safety ANN."""
    counts = {c: 0 for c in MASK_CLASSES}
    for d in detections:
        counts[d["label"]] += 1
    return {
        "people_count": len(detections),
        "mask_issue_count": counts["without_mask"] + counts["mask_weared_incorrect"],
        "counts": counts,
    }


def annotate_image(image_bgr: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Return a copy of the image with boxes and 'label confidence' text."""
    out = ensure_bgr(image_bgr).copy()
    thickness = max(2, int(round(min(out.shape[:2]) / 200)))
    scale = max(0.5, min(out.shape[:2]) / 600)
    for d in detections:
        x, y, w, h = d["box"]
        color = LABEL_COLORS.get(d["label"], (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        text = f'{d["label"]} {d["confidence"] * 100:.0f}%'
        y_text = y - 8 if y - 8 > 12 else y + 18
        cv2.putText(out, text, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
    return out