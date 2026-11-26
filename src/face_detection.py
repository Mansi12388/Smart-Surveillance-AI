"""
Face detection with OpenCV's Haar cascade, plus small image helpers.

All images in this project are OpenCV-style BGR uint8 numpy arrays.
"""
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # (x, y, w, h)


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes (e.g. an uploaded file) into a BGR array."""
    if not data:
        raise ValueError("Empty image data.")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Upload a valid JPG/PNG file.")
    return image


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Return a 3-channel uint8 BGR image from grayscale / BGRA / BGR input."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


@lru_cache(maxsize=1)
def _get_cascade() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from: {path}")
    return cascade


def detect_faces(
    image_bgr: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (40, 40),
) -> List[Box]:
    """Detect frontal faces. Returns a list of (x, y, w, h) boxes, largest first.

    Note: Haar cascades can miss faces that are covered by a mask or turned
    sideways. Callers should handle an empty result (see mask_detection.py).
    """
    image_bgr = ensure_bgr(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _get_cascade().detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def crop_face(image_bgr: np.ndarray, box: Box, margin: float = 0.15) -> np.ndarray:
    """Crop a face box (with a relative margin) safely inside the image."""
    h_img, w_img = image_bgr.shape[:2]
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(w_img, x + w + mx), min(h_img, y + h + my)
    return image_bgr[y1:y2, x1:x2]


def draw_boxes(image_bgr: np.ndarray, boxes: List[Box], color=(0, 255, 0)) -> np.ndarray:
    """Return a copy of the image with rectangles drawn around the boxes."""
    out = ensure_bgr(image_bgr).copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    return out