"""Single-image inference latency (batch size 1, the online-serving case)."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch


def cpu_name() -> str:
    if platform.system() == "Windows":
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def environment() -> dict:
    import onnxruntime as ort

    return {
        "cpu": cpu_name(),
        "cpu_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
    }


def time_calls(fn: Callable[[], object], warmup: int, runs: int, sync=lambda: None) -> dict[str, float]:
    """Median and 95th-percentile wall time (ms) of ``fn`` over ``runs`` calls after ``warmup`` calls."""
    for _ in range(warmup):
        fn()
    sync()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        sync()
        timings.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95))}


def measure_latency(
    model: torch.nn.Module,
    image_size: int,
    onnx_models: dict[str, Path],
    warmup: int = 20,
    runs: int = 100,
) -> dict[str, dict[str, float]]:
    """Latency of the PyTorch model on CPU (FP32) and GPU (bf16), and of each ONNX file on CPU."""
    import onnxruntime as ort

    image = torch.randn(1, 3, image_size, image_size)
    model = model.float().eval().cpu()
    latency = {}
    with torch.inference_mode():
        latency["torch_cpu_fp32"] = time_calls(lambda: model(image), warmup, runs)
        if torch.cuda.is_available():
            gpu_model, gpu_image = model.cuda(), image.cuda()

            def gpu_call():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    gpu_model(gpu_image)

            latency["torch_gpu_bf16"] = time_calls(gpu_call, warmup, runs, torch.cuda.synchronize)
            model.cpu()

    for name, path in onnx_models.items():
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed = {session.get_inputs()[0].name: image.numpy()}
        latency[name] = time_calls(lambda s=session, f=feed: s.run(None, f), warmup, runs)
    return latency
