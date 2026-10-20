"""ONNX export, INT8 quantisation, the ONNX predictor and the FastAPI service."""

import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from animal_classifier.export import export_onnx, quantize_int8, read_metadata
from animal_classifier.models import create_model
from animal_classifier.predictor import OnnxPredictor

CLASSES = ["cat", "dog", "fish"]


@pytest.fixture(scope="module")
def onnx_model(tmp_path_factory):
    torch.manual_seed(0)
    model = create_model("simple_cnn", num_classes=len(CLASSES)).eval()
    path = export_onnx(
        model,
        tmp_path_factory.mktemp("model") / "model.onnx",
        CLASSES,
        "simple_cnn",
        resize_size=72,
        crop_size=64,
    )
    return model, path


def png_bytes(color=(200, 40, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (90, 70), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_onnx_matches_pytorch(onnx_model):
    model, path = onnx_model
    batch = np.random.default_rng(0).standard_normal((3, 3, 64, 64)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(batch)).numpy()
    np.testing.assert_allclose(OnnxPredictor(path).logits(batch), expected, atol=1e-4)


def test_onnx_metadata_drives_preprocessing(onnx_model):
    _, path = onnx_model
    assert json.loads(read_metadata(path)["class_names"]) == CLASSES
    predictor = OnnxPredictor(path)
    assert (predictor.resize_size, predictor.crop_size) == (72, 64)
    result = predictor.predict(png_bytes(), top_k=2)
    assert result["label"] in CLASSES and len(result["top_k"]) == 2
    scores = [item["score"] for item in predictor.predict(png_bytes(), top_k=3)["top_k"]]
    assert sum(scores) == pytest.approx(1.0)


def test_int8_quantisation(onnx_model, tmp_path):
    _, path = onnx_model
    calibration = list(np.random.default_rng(0).standard_normal((8, 3, 64, 64)).astype(np.float32))
    int8_path = quantize_int8(path, tmp_path / "model.int8.onnx", calibration)
    assert int8_path.stat().st_size < path.stat().st_size
    assert OnnxPredictor(int8_path).predict(png_bytes())["label"] in CLASSES


@pytest.fixture
def client(onnx_model, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import app

    monkeypatch.setenv("MODEL_PATH", str(onnx_model[1]))
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.01")
    with TestClient(app) as test_client:
        yield test_client


def test_api_health_and_classes(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/classes").json() == CLASSES


def test_api_predict(client):
    response = client.post("/predict?top_k=2", files={"file": ("cat.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in CLASSES and len(body["top_k"]) == 2 and body["latency_ms"] >= 0


@pytest.mark.parametrize(
    "content,query,status",
    [
        (b"not an image", "", 400),
        (b"", "", 400),
        (png_bytes(), "?top_k=0", 422),
        (b"x" * 20_000, "", 413),  # MAX_UPLOAD_MB=0.01 -> ~10 KB limit
    ],
)
def test_api_rejects_bad_requests(client, content, query, status):
    response = client.post(f"/predict{query}", files={"file": ("x.png", content, "image/png")})
    assert response.status_code == status


def test_measure_latency_covers_torch_and_onnx(onnx_model):
    from animal_classifier.benchmark import measure_latency, time_calls

    model, path = onnx_model
    latency = measure_latency(model, image_size=64, onnx_models={"onnx_cpu_fp32": path}, warmup=1, runs=3)
    assert {"torch_cpu_fp32", "onnx_cpu_fp32"} <= set(latency)
    assert all(stats["median_ms"] > 0 for stats in latency.values())
    assert time_calls(lambda: None, warmup=0, runs=5)["p95_ms"] >= 0
