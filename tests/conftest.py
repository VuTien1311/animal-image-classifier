"""Shared fixtures: a tiny synthetic dataset in the processed layout, and a fast CPU config."""

from __future__ import annotations

import csv
import json
import os

# Tests run on the CPU, exactly as in CI, even on a machine with a GPU. ("-1" rather than "": on
# Windows, setting a variable to the empty string removes it.)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from animal_classifier.config import Config  # noqa: E402

CLASSES = ["cat", "dog", "fish"]
COLORS = [(200, 40, 40), (40, 200, 40), (40, 40, 200)]  # one colour per class -> learnable
SPLIT_SIZES = {"train": 8, "val": 3, "test": 3}


def make_image(rng: np.random.Generator, color, width: int = 80, height: int = 64) -> Image.Image:
    noise = rng.integers(-30, 30, size=(height, width, 3))
    return Image.fromarray(np.clip(np.array(color) + noise, 0, 255).astype(np.uint8))


@pytest.fixture
def tiny_dataset(tmp_path):
    """data/processed-style dataset: 3 classes, 8/3/3 images per class for train/val/test."""
    rng = np.random.default_rng(0)
    rows = []
    for split, per_class in SPLIT_SIZES.items():
        for label, (name, color) in enumerate(zip(CLASSES, COLORS, strict=True)):
            for i in range(per_class):
                relative = f"images/{split}/{name}/{i}.jpg"
                path = tmp_path / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                make_image(rng, color).save(path)
                rows.append({"path": relative, "label": label, "class_name": name, "split": split})
    with open(tmp_path / "splits.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "class_name", "split"])
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    return tmp_path


@pytest.fixture
def tiny_config(tiny_dataset, tmp_path) -> Config:
    """SimpleCNN on 64px images, CPU only, no worker processes: trains in seconds."""
    cfg = Config(experiment="test_run", output_dir=str(tmp_path / "outputs"))
    cfg.data.root = str(tiny_dataset)
    cfg.data.resize_size, cfg.data.image_size = 72, 64
    cfg.data.batch_size = 4
    cfg.data.num_workers = cfg.data.eval_num_workers = 0
    cfg.model.name, cfg.model.pretrained, cfg.model.drop_rate = "simple_cnn", False, 0.1
    cfg.optim.warmup_epochs = 0.5
    cfg.train.epochs, cfg.train.amp, cfg.train.channels_last = 2, False, False
    return cfg
