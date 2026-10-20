"""ONNX export, INT8 quantisation and evaluation of exported models.

The exported file carries its own metadata (class names, model name, preprocessing sizes), so the
serving code needs nothing but the .onnx file; see :class:`predictor.OnnxPredictor`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import onnx
import torch

from .data import read_split
from .metrics import classification_metrics
from .predictor import OnnxPredictor
from .preprocess import load_image, preprocess

# ---------------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------------


def export_onnx(
    model: torch.nn.Module,
    path: str | Path,
    class_names: list[str],
    model_name: str,
    resize_size: int = 256,
    crop_size: int = 224,
    opset: int = 18,
) -> Path:
    """Export ``model`` (FP32, dynamic batch size) with serving metadata; returns the file path.

    The model is moved to the CPU in place and put in eval mode.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.float().eval().to("cpu")
    example = torch.randn(2, 3, crop_size, crop_size)
    program = torch.onnx.export(
        model,
        (example,),
        dynamo=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=1024)},),
        opset_version=opset,
        verbose=False,  # its progress messages contain emoji, which crash a cp1252 Windows console
    )
    program.save(str(path), external_data=False)
    add_metadata(path, serving_metadata(class_names, model_name, resize_size, crop_size))
    return path


def serving_metadata(class_names: list[str], model_name: str, resize_size: int, crop_size: int) -> dict:
    return {
        "class_names": json.dumps(class_names),
        "model_name": model_name,
        "resize_size": str(resize_size),
        "crop_size": str(crop_size),
    }


def add_metadata(path: str | Path, metadata: dict[str, str]) -> None:
    model = onnx.load(str(path))
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        prop = existing.get(key) or model.metadata_props.add()
        prop.key, prop.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def read_metadata(path: str | Path) -> dict[str, str]:
    return {prop.key: prop.value for prop in onnx.load(str(path), load_external_data=False).metadata_props}


# ---------------------------------------------------------------------------------------------------
# INT8 quantisation
# ---------------------------------------------------------------------------------------------------

CONV_MATMUL_OPS = ("Conv", "MatMul", "Gemm")


def calibration_images(
    data_root: str | Path, resize_size: int, crop_size: int, n: int = 300, seed: int = 1
) -> list[np.ndarray]:
    """A seeded random sample of preprocessed *train* images for INT8 calibration (never val/test)."""
    samples = random.Random(seed).sample(read_split(data_root, "train"), n)
    return [preprocess(load_image(Path(data_root) / s.path), resize_size, crop_size) for s in samples]


def quantize_int8(
    fp32_path: str | Path,
    int8_path: str | Path,
    calibration: list[np.ndarray],
    op_types: tuple[str, ...] | None = None,
    moving_average: bool = False,
) -> Path:
    """Static INT8 quantisation (QDQ format, per-channel weights), calibrated on preprocessed images.

    Static quantisation fixes activation ranges ahead of time, which is what makes convolutions
    faster on CPU; dynamic quantisation would only speed up the linear layers.

    Args:
        op_types: operator types to quantise; None means every type ONNX Runtime supports.
            ``CONV_MATMUL_OPS`` keeps element-wise operations (SiLU, sigmoid, squeeze-and-excitation
            products, LayerNorm) in float. Which setting works best depends on the architecture;
            scripts/deploy.py tries several and picks one on the val split.
        moving_average: estimate activation ranges with a moving average of per-batch min/max
            instead of the global min/max, so rare outliers do not stretch the range.
    """
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, arrays: list[np.ndarray]):
            self._batches = iter([{"image": array[None].astype(np.float32)} for array in arrays])

        def get_next(self):
            return next(self._batches, None)

    fp32_path, int8_path = Path(fp32_path), Path(int8_path)
    prepared = int8_path.with_name(int8_path.stem + ".prep.onnx")
    quant_pre_process(str(fp32_path), str(prepared))
    quantize_static(
        str(prepared),
        str(int8_path),
        Reader(calibration),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        op_types_to_quantize=list(op_types) if op_types else None,
        extra_options={"CalibMovingAverage": moving_average},
    )
    prepared.unlink()
    add_metadata(int8_path, read_metadata(fp32_path))
    return int8_path


# ---------------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------------


def evaluate_onnx(model_path: str | Path, data_root: str | Path, split: str, batch_size: int = 64) -> dict:
    """Accuracy and macro-F1 of an ONNX model on a split, through the serving code path.

    Images are preprocessed by OnnxPredictor exactly as in production, in batches so memory stays flat.
    """
    predictor = OnnxPredictor(model_path)
    samples = read_split(data_root, split)
    predictions = []
    for start in range(0, len(samples), batch_size):
        batch = [predictor.preprocess(Path(data_root) / s.path) for s in samples[start : start + batch_size]]
        predictions.append(predictor.logits(np.stack(batch)).argmax(axis=1))
    labels = [s.label for s in samples]
    metrics = classification_metrics(labels, np.concatenate(predictions), predictor.class_names)
    return {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
