# Serving image: FastAPI + ONNX Runtime only (no PyTorch), so the image stays small and starts fast.
#
# Export the model first, then build:
#   python scripts/deploy.py --checkpoint outputs/convnext_tiny/best.pt
#   docker build -t animal-classifier .
#   docker run -p 8000:8000 animal-classifier
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so this layer is cached when only the model or app code changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

COPY app ./app
ARG MODEL=models/convnext_tiny/model.int8.onnx
COPY ${MODEL} ./model.onnx
ENV MODEL_PATH=/app/model.onnx

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
