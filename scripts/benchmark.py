"""Measure the single-image latency of every deployed model, all in one session.

Latency is measured here rather than inside deploy.py. Right after minutes of all-core work
(quantisation, evaluation), this laptop's CPU is thermally throttled and the same model measured up
to 2x slower. Measuring all models back-to-back on an idle machine keeps the conditions comparable.

For each experiment in models/: PyTorch on CPU (FP32) and GPU (bf16), and ONNX Runtime on CPU for
model.onnx and, if it was shipped, model.int8.onnx. Batch size 1; median and p95 over --runs calls.

Writes results/deployment/latency.json.

Example:
    python scripts/benchmark.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from animal_classifier.benchmark import environment, measure_latency
from animal_classifier.models import load_checkpoint
from animal_classifier.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    report = {
        "batch_size": 1,
        "warmup": args.warmup,
        "runs": args.runs,
        "environment": environment(),
        "models": {},
    }
    for model_dir in sorted(p for p in args.models_dir.iterdir() if (p / "model.onnx").exists()):
        experiment = model_dir.name
        model, checkpoint = load_checkpoint(args.outputs / experiment / "best.pt")
        onnx_models = {"onnx_cpu_fp32": model_dir / "model.onnx"}
        if (model_dir / "model.int8.onnx").exists():
            onnx_models["onnx_cpu_int8"] = model_dir / "model.int8.onnx"

        latency = measure_latency(
            model, checkpoint["config"]["data"]["image_size"], onnx_models, args.warmup, args.runs
        )
        report["models"][experiment] = latency
        print(experiment)
        for name, stats in latency.items():
            print(f"  {name:<16} median {stats['median_ms']:7.2f} ms   p95 {stats['p95_ms']:7.2f} ms")

    save_json(report, args.results / "deployment" / "latency.json")


if __name__ == "__main__":
    main()
