"""Inference wrappers with an identical interface for PyTorch checkpoints and ONNX models.

    predictor = OnnxPredictor("models/model.onnx")       # serving: no PyTorch needed
    predictor = TorchPredictor("outputs/x/best.pt")      # research: also supports Grad-CAM
    predictor.predict("cat.jpg", top_k=3)
    # {"label": "cat", "score": 0.97, "top_k": [{"label": "cat", "score": 0.97}, ...]}

Both use :func:`preprocess.preprocess`, the same function used for evaluation during training.
ONNX Runtime and PyTorch are imported inside the classes, so the serving image (ONNX Runtime only)
can import this module without PyTorch installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .preprocess import ImageSource, load_image, preprocess

ONNX_METADATA_KEYS = ("class_names", "model_name", "resize_size", "crop_size")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def format_prediction(probs: np.ndarray, class_names: list[str], top_k: int = 3) -> dict:
    order = np.argsort(probs)[::-1][:top_k]
    top = [{"label": class_names[i], "score": float(probs[i])} for i in order]
    return {"label": top[0]["label"], "score": top[0]["score"], "top_k": top}


class OnnxPredictor:
    """ONNX Runtime predictor. Class names and preprocessing sizes are read from the model metadata."""

    def __init__(self, model_path: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
        metadata = self.session.get_modelmeta().custom_metadata_map
        missing = [key for key in ONNX_METADATA_KEYS if key not in metadata]
        if missing:
            raise ValueError(f"{model_path} lacks metadata {missing}; export it with scripts/deploy.py")
        self.class_names: list[str] = json.loads(metadata["class_names"])
        self.model_name = metadata["model_name"]
        self.resize_size = int(metadata["resize_size"])
        self.crop_size = int(metadata["crop_size"])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: ImageSource) -> np.ndarray:
        return preprocess(load_image(image), self.resize_size, self.crop_size)

    def logits(self, batch: np.ndarray) -> np.ndarray:
        """Raw logits for an already preprocessed Nx3xHxW float32 batch."""
        return self.session.run(None, {self.input_name: batch.astype(np.float32, copy=False)})[0]

    def predict(self, image: ImageSource, top_k: int = 3) -> dict:
        logits = self.logits(self.preprocess(image)[None])[0]
        return format_prediction(softmax(logits), self.class_names, top_k)


class TorchPredictor:
    """PyTorch predictor for a trainer checkpoint (``best.pt``), with Grad-CAM explanations."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        import torch

        from .models import load_checkpoint

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.class_names: list[str] = checkpoint["class_names"]
        self.model_name: str = checkpoint["model_name"]
        data_cfg = checkpoint["config"]["data"]
        self.resize_size, self.crop_size = data_cfg["resize_size"], data_cfg["image_size"]
        self._gradcam = None

    def _tensor(self, image: ImageSource):
        import torch

        array = preprocess(load_image(image), self.resize_size, self.crop_size)
        return torch.from_numpy(array)[None].to(self.device)

    def predict(self, image: ImageSource, top_k: int = 3) -> dict:
        import torch

        with torch.no_grad():
            logits = self.model(self._tensor(image))[0].float().cpu().numpy()
        return format_prediction(softmax(logits), self.class_names, top_k)

    def explain(self, image: ImageSource, top_k: int = 3) -> tuple[dict, np.ndarray]:
        """Prediction plus a Grad-CAM overlay (HxWx3 uint8) on the centre-cropped input."""
        from .gradcam import GradCAM, overlay_heatmap
        from .models import gradcam_layer
        from .preprocess import center_crop, resize_shorter_side

        if self._gradcam is None:
            self._gradcam = GradCAM(self.model, gradcam_layer(self.model, self.model_name))
        rgb = load_image(image)
        heatmap, probs = self._gradcam(self._tensor(rgb))
        cropped = np.asarray(center_crop(resize_shorter_side(rgb, self.resize_size), self.crop_size))
        return format_prediction(probs, self.class_names, top_k), overlay_heatmap(cropped, heatmap)
