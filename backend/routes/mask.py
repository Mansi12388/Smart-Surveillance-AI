"""POST /mask/predict - face-mask detection on an uploaded image."""
import base64

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.schemas import MaskResponse
from src.database import log_event
from src.face_detection import decode_image
from src.mask_detection import annotate_image, detect_and_classify, summarize_detections

router = APIRouter(prefix="/mask", tags=["Face mask"])

MAX_IMAGE_BYTES = 10 * 1024 * 1024
SEVERITY_ORDER = ["without_mask", "mask_weared_incorrect", "with_mask"]  # worst first


@router.post("/predict", response_model=MaskResponse)
def predict_mask(
    file: UploadFile = File(..., description="JPG or PNG image"),
    use_face_detector: bool = Query(True, description="Detect faces first; otherwise classify the whole image"),
    annotate: bool = Query(False, description="Also return the image with boxes drawn"),
):
    data = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")

    image = decode_image(data)                       # ValueError -> HTTP 400
    detections = detect_and_classify(image, use_face_detector)
    summary = summarize_detections(detections)

    overall = next((lab for lab in SEVERITY_ORDER if summary["counts"][lab] > 0), "with_mask")
    confidence = max(d["confidence"] for d in detections if d["label"] == overall)

    event_id = log_event(
        "mask", overall, confidence,
        input_text=file.filename,
        details={**summary, "use_face_detector": use_face_detector},
    )

    annotated = None
    if annotate:
        ok, buf = cv2.imencode(".jpg", annotate_image(image, detections))
        if ok:
            annotated = base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "event_id": event_id,
        "label": overall,
        "detections": detections,
        "summary": summary,
        "annotated_image": annotated,
    }