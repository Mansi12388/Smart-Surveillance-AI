"""POST /activity/predict - activity recognition on an uploaded video."""
import os
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.schemas import ActivityResponse
from src.activity_detection import predict_video
from src.database import log_event

router = APIRouter(prefix="/activity", tags=["Human activity"])

ALLOWED_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
MAX_VIDEO_BYTES = 100 * 1024 * 1024


@router.post("/predict", response_model=ActivityResponse)
def predict_activity(
    file: UploadFile = File(..., description="Short video: avi, mp4, mov, mkv or webm"),
    model_type: Literal["cnn_lstm", "bilstm"] = Query("cnn_lstm"),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported video type. Use one of {sorted(ALLOWED_EXTENSIONS)}.")

    data = file.file.read(MAX_VIDEO_BYTES + 1)
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Video is larger than 100 MB.")

    # OpenCV needs a real file path, so write the upload to a temporary file
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        result = predict_video(tmp_path, model_type)     # ValueError -> HTTP 400
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    event_id = log_event(
        "activity", result["label"], result["confidence"],
        input_text=file.filename,
        details={
            "model_type": result["model_type"],
            "num_frames": result["num_frames"],
            "num_windows": result["num_windows"],
            "probabilities": result["probabilities"],
        },
    )
    return {**result, "event_id": event_id}