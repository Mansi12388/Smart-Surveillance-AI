"""POST /safety/predict - event-prioritisation with the ANN.

Classifies an event record (counts and durations). It is NOT an assessment of
any person's risk or intent.
"""
from fastapi import APIRouter

from backend.schemas import SafetyRequest, SafetyResponse
from src.database import log_event
from src.safety_classification import predict_safety

router = APIRouter(prefix="/safety", tags=["Safety events"])


@router.post("/predict", response_model=SafetyResponse)
def predict(request: SafetyRequest):
    result = predict_safety(
        duration_seconds=request.duration_seconds,
        people_count=request.people_count,
        mask_issue_count=request.mask_issue_count,
        motion_change_count=request.motion_change_count,
    )
    event_id = log_event(
        "safety", result["label"], result["confidence"],
        details={
            "priority": result["priority"],
            "inputs": result["inputs"],
            "probabilities": result["probabilities"],
        },
    )
    return {**result, "event_id": event_id}