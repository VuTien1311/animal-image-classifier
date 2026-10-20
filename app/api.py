"""REST API for the animal classifier, served with ONNX Runtime (no PyTorch in the serving image).

Run:
    uvicorn app.api:app --port 8000
    curl -F "file=@cat.jpg" "http://localhost:8000/predict?top_k=3"

Configuration (environment variables):
    MODEL_PATH      ONNX model written by scripts/deploy.py (default below)
    MAX_UPLOAD_MB   larger uploads are rejected with 413 (default 10)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import UnidentifiedImageError
from pydantic import BaseModel

from animal_classifier import __version__
from animal_classifier.predictor import OnnxPredictor

DEFAULT_MODEL_PATH = "models/convnext_tiny/model.int8.onnx"


class LabelScore(BaseModel):
    label: str
    score: float


class Prediction(BaseModel):
    label: str
    score: float
    top_k: list[LabelScore]
    model: str
    latency_ms: float


class Health(BaseModel):
    status: str
    model: str
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = OnnxPredictor(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    app.state.max_upload_bytes = int(float(os.environ.get("MAX_UPLOAD_MB", "10")) * 2**20)
    yield


app = FastAPI(title="Animal Classifier API", version=__version__, lifespan=lifespan)


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", model=app.state.predictor.model_name, version=__version__)


@app.get("/classes", response_model=list[str])
def classes() -> list[str]:
    return app.state.predictor.class_names


@app.post("/predict", response_model=Prediction)
def predict(
    file: Annotated[UploadFile, File(description="JPEG/PNG/WebP image")],
    top_k: Annotated[int, Query(ge=1, le=10)] = 3,
) -> Prediction:
    # A sync endpoint: FastAPI runs it in a thread pool, so CPU-bound inference does not block the event loop.
    data = file.file.read(app.state.max_upload_bytes + 1)
    if len(data) > app.state.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File too large")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    start = time.perf_counter()
    try:
        result = app.state.predictor.predict(data, top_k=top_k)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Could not decode the file as an image") from error
    latency_ms = (time.perf_counter() - start) * 1000
    return Prediction(**result, model=app.state.predictor.model_name, latency_ms=round(latency_ms, 2))
