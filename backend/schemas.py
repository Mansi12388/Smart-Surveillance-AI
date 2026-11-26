"""Pydantic request / response models for the FastAPI backend."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ------------------------------------------------------------------ mask
class Detection(BaseModel):
    box: List[int] = Field(description="[x, y, width, height] in pixels")
    face_detected: bool = Field(description="False when the whole image was used as one face crop")
    label: str
    confidence: float
    probabilities: Dict[str, float]


class MaskSummary(BaseModel):
    people_count: int
    mask_issue_count: int
    counts: Dict[str, int]


class MaskResponse(BaseModel):
    event_id: int
    label: str = Field(description="Most severe label found in the image")
    detections: List[Detection]
    summary: MaskSummary
    annotated_image: Optional[str] = Field(default=None, description="Base64-encoded JPEG")


# -------------------------------------------------------------- activity
class ActivityResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow the field name "model_type"

    event_id: int
    label: str
    confidence: float
    probabilities: Dict[str, float]
    num_frames: int
    num_windows: int
    model_type: str


# -------------------------------------------------------------- incident
class IncidentRequest(BaseModel):
    text: str = Field(min_length=3, max_length=500, examples=["Person fell near the entrance while walking."])


class IncidentResponse(BaseModel):
    event_id: int
    text: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    summary: str


# ---------------------------------------------------------------- safety
class SafetyRequest(BaseModel):
    duration_seconds: float = Field(ge=0, le=86400, examples=[120])
    people_count: int = Field(ge=0, le=10000, examples=[4])
    mask_issue_count: int = Field(ge=0, le=10000, examples=[0])
    motion_change_count: int = Field(ge=0, le=10000, examples=[3])


class SafetyResponse(BaseModel):
    event_id: int
    label: str
    confidence: float
    priority: str
    probabilities: Dict[str, float]
    inputs: Dict[str, float]


# ------------------------------------------------------- events / health
class EventOut(BaseModel):
    id: int
    created_at: str
    module: str
    label: str
    confidence: Optional[float] = None
    input_text: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class StatsOut(BaseModel):
    total: int
    by_module: Dict[str, int]
    by_label: List[Dict[str, Any]]


class HealthOut(BaseModel):
    status: str
    models: Dict[str, bool]