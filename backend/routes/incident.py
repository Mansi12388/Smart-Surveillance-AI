"""POST /incident/predict - classify a written incident description with BERT."""
from fastapi import APIRouter

from backend.schemas import IncidentRequest, IncidentResponse
from src.database import log_event
from src.incident_classification import predict_incident

router = APIRouter(prefix="/incident", tags=["Incident text"])


@router.post("/predict", response_model=IncidentResponse)
def predict(request: IncidentRequest):
    result = predict_incident(request.text)              # ValueError -> HTTP 400
    event_id = log_event(
        "incident", result["label"], result["confidence"],
        input_text=result["text"],
        details={"probabilities": result["probabilities"], "summary": result["summary"]},
    )
    return {**result, "event_id": event_id}