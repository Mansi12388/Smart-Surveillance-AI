"""
FastAPI application.

Run from the project root (the folder that contains src/ and backend/):
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
Interactive docs: http://127.0.0.1:8000/docs

Models are loaded on the first request to each endpoint, so that first call is slow.
"""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routes import activity, incident, mask, safety
from backend.schemas import EventOut, HealthOut, StatsOut
from src import database
from src.config import (
    ACTIVITY_BILSTM_PATH,
    ACTIVITY_CNN_LSTM_PATH,
    BERT_MODEL_DIR,
    MASK_CNN_PATH,
    MASK_MOBILENET_PATH,
    SAFETY_ANN_PATH,
    SAFETY_SCALER_PATH,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    yield


app = FastAPI(
    title="Smart Surveillance AI",
    description="Face-mask detection, activity recognition, incident text classification "
                "and safety-event prioritisation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mask.router)
app.include_router(activity.router)
app.include_router(incident.router)
app.include_router(safety.router)


# ------------------------------------------------------- error handling
@app.exception_handler(FileNotFoundError)
async def model_not_found_handler(request, exc: FileNotFoundError):
    """A trained model file is missing -> tell the client which notebook to run."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def bad_input_handler(request, exc: ValueError):
    """Invalid user input (unreadable image / video, empty text, unknown module...)."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ------------------------------------------------------------ endpoints
@app.get("/health", response_model=HealthOut, tags=["System"])
def health():
    """Which trained models are present on disk."""
    return {
        "status": "ok",
        "models": {
            "mask_cnn": MASK_CNN_PATH.exists(),
            "mask_mobilenetv2": MASK_MOBILENET_PATH.exists(),
            "activity_cnn_lstm": ACTIVITY_CNN_LSTM_PATH.exists(),
            "activity_bilstm": ACTIVITY_BILSTM_PATH.exists(),
            "bert_incident": (BERT_MODEL_DIR / "config.json").exists(),
            "safety_ann": SAFETY_ANN_PATH.exists() and SAFETY_SCALER_PATH.exists(),
        },
    }


@app.get("/events", response_model=List[EventOut], tags=["Event log"])
def list_events(
    limit: int = Query(50, ge=1, le=500),
    module: Optional[str] = Query(None, description="mask | activity | incident | safety"),
):
    return database.get_events(limit=limit, module=module)


@app.get("/stats", response_model=StatsOut, tags=["Event log"])
def stats():
    return database.get_stats()


@app.delete("/events", tags=["Event log"])
def delete_events():
    return {"deleted": database.clear_events()}