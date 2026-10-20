"""Prepare a trained model for serving: export to ONNX, quantise to INT8 and verify both.

Steps
  1. Export to ONNX (FP32) and check that its logits match PyTorch on real val images.
  2. Post-training static INT8 quantisation. Four calibration settings are tried and the one with
     the best *val* accuracy is kept. If even the best one costs more than --max-int8-drop points on
     val, no INT8 model is shipped (EfficientNet-B0 is the example in this project).
  3. Test accuracy of the shipped ONNX models through the serving code path (test is used once).

Latency is measured separately by scripts/benchmark.py (see there for why).

Writes models/<experiment>/model.onnx (and model.int8.onnx) and results/deployment/<experiment>.json.

Example:
    python scripts/deploy.py --checkpoint outputs/convnext_tiny/best.pt
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from animal_classifier.config import Config
from animal_classifier.data import read_split
from animal_classifier.export import (
    CONV_MATMUL_OPS,
    calibration_images,
    evaluate_onnx,
    export_onnx,
    quantize_int8,
)
from animal_classifier.models import load_checkpoint
from animal_classifier.predictor import OnnxPredictor
from animal_classifier.preprocess import load_image, preprocess
from animal_classifier.utils import count_parameters, save_json

# name -> (operator types to quantise, moving-average calibration). The ONNX Runtime default comes
# first, so it wins ties.
INT8_SETTINGS = {
    "all ops, min/max": (None, False),
    "all ops, moving average": (None, True),
    "conv/matmul only, min/max": (CONV_MATMUL_OPS, False),
    "conv/matmul only, moving average": (CONV_MATMUL_OPS, True),
}


def check_parity(model: torch.nn.Module, onnx_path: Path, data_root: str, resize: int, crop: int) -> float:
    """Largest absolute logit difference between PyTorch and ONNX Runtime on 32 val images."""
    samples = read_split(data_root, "val")[:32]
    images = np.stack([preprocess(load_image(Path(data_root) / s.path), resize, crop) for s in samples])
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(images)).numpy()
    return float(np.abs(torch_logits - OnnxPredictor(onnx_path).logits(images)).max())


def select_int8(fp32_path: Path, int8_path: Path, data_root: str, calibration: list, max_drop: float) -> dict:
    """Try every INT8 setting, keep the best on val, and ship it only if it costs <= max_drop points."""
    val = {"fp32": evaluate_onnx(fp32_path, data_root, "val")}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        candidates = {}
        for i, (name, (op_types, moving_average)) in enumerate(INT8_SETTINGS.items()):
            candidates[name] = quantize_int8(
                fp32_path, Path(tmp) / f"{i}.onnx", calibration, op_types, moving_average
            )
            val[name] = evaluate_onnx(candidates[name], data_root, "val")
            print(f"  INT8 [{name}] val accuracy {val[name]['accuracy']:.4f}")

        best = max(INT8_SETTINGS, key=lambda name: val[name]["accuracy"])
        drop = (val["fp32"]["accuracy"] - val[best]["accuracy"]) * 100
        shipped = drop <= max_drop
        int8_path.unlink(missing_ok=True)
        if shipped:
            shutil.copy(candidates[best], int8_path)

    return {"val_accuracy": val, "selected": best, "val_drop_points": drop, "shipped": shipped}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, help="default: models/<experiment>")
    parser.add_argument("--calibration-images", type=int, default=300)
    parser.add_argument("--max-int8-drop", type=float, default=1.0, help="max val accuracy loss, in points")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint, "cpu")
    cfg = Config.from_dict(checkpoint["config"])
    root, resize, crop = cfg.data.root, cfg.data.resize_size, cfg.data.image_size
    out_dir = args.out_dir or Path("models") / cfg.experiment
    print(f"== {cfg.experiment}")

    # 1. FP32 export + parity with PyTorch.
    fp32_path = export_onnx(
        model, out_dir / "model.onnx", checkpoint["class_names"], cfg.model.name, resize, crop
    )
    max_diff = check_parity(model, fp32_path, root, resize, crop)
    print(f"FP32 ONNX: max |logit difference| vs PyTorch = {max_diff:.1e}")

    # 2. INT8, setting chosen on val.
    calibration = calibration_images(root, resize, crop, n=args.calibration_images)
    int8_path = out_dir / "model.int8.onnx"
    quantization = select_int8(fp32_path, int8_path, root, calibration, args.max_int8_drop)
    verdict = "shipped" if quantization["shipped"] else f"not shipped (> {args.max_int8_drop} points)"
    print(
        f"INT8: best setting '{quantization['selected']}', "
        f"val drop {quantization['val_drop_points']:.2f} points -> {verdict}"
    )

    # 3. Test accuracy of what is shipped.
    shipped = {"fp32": fp32_path} | ({"int8": int8_path} if quantization["shipped"] else {})
    test = {name: evaluate_onnx(path, root, "test") for name, path in shipped.items()}
    for name, metrics in test.items():
        print(f"{name.upper()} test accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}")

    report = {
        "experiment": cfg.experiment,
        "model_name": cfg.model.name,
        "parameters": count_parameters(model),
        "file_mb": {name: path.stat().st_size / 2**20 for name, path in shipped.items()},
        "max_abs_logit_diff_vs_torch": max_diff,
        "quantization": quantization | {"calibration_images": len(calibration)},
        "test": test,
    }
    save_json(report, args.results / "deployment" / f"{cfg.experiment}.json")


if __name__ == "__main__":
    main()
