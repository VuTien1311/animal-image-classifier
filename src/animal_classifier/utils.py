"""Small helpers: reproducibility, devices, logging and JSON I/O."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def single_threaded_blas_for_subprocesses() -> None:
    """Keep worker processes (DataLoader workers, process pools) lean.

    Workers only decode and transform images, but NumPy's OpenBLAS pre-allocates buffers for every CPU
    thread at import: ~1 GB of extra committed memory per worker on a 16-thread machine, which ran
    Windows out of memory with 8+ workers. Child processes inherit the environment when they are
    spawned; the already-running parent process is unaffected.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def resolve_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


_LOG_FORMAT = logging.Formatter("%(asctime)s %(levelname)s %(message)s")


def get_logger() -> logging.Logger:
    """Project logger writing to stdout. Third-party INFO logs (HTTP requests etc.) are not captured."""
    logger = logging.getLogger("animal_classifier")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_LOG_FORMAT)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


@contextmanager
def log_to_file(path: str | Path) -> Iterator[logging.Logger]:
    """Also write the project log to ``path`` inside the block; the file is closed on exit."""
    logger = get_logger()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(_LOG_FORMAT)
    logger.addHandler(handler)
    try:
        yield logger
    finally:
        logger.removeHandler(handler)
        handler.close()


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
