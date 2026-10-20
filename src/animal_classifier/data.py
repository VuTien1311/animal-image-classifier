"""Datasets, transforms and data loaders for the processed Animals-10 splits.

``scripts/prepare_data.py`` writes the processed dataset as::

    data/processed/
        classes.json          # ["butterfly", "cat", ...]  (index == label)
        splits.csv            # path,label,class_name,split   (split in {train, val, test})
        images/<split>/<class>/<name>.jpg
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from .config import DataConfig
from .preprocess import IMAGENET_MEAN, IMAGENET_STD, load_image, preprocess
from .utils import single_threaded_blas_for_subprocesses

SPLITS_FILE = "splits.csv"
CLASSES_FILE = "classes.json"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    path: str  # relative to the processed dataset root
    label: int


def read_class_names(root: str | Path) -> list[str]:
    return json.loads((Path(root) / CLASSES_FILE).read_text(encoding="utf-8"))


def read_split(root: str | Path, split: str) -> list[Sample]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
    with open(Path(root) / SPLITS_FILE, newline="", encoding="utf-8") as f:
        return [Sample(row["path"], int(row["label"])) for row in csv.DictReader(f) if row["split"] == split]


def stratified_split(labels: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split indices into train/val so every class keeps the same proportion in both parts."""
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    train_idx, val_idx = [], []
    for label in np.unique(labels_arr):
        idx = rng.permutation(np.flatnonzero(labels_arr == label))
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


class ImageDataset(Dataset):
    def __init__(self, root: str | Path, samples: list[Sample], transform):
        self.root = Path(root)
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = load_image(self.root / sample.path)
        return self.transform(image), sample.label


class EvalTransform:
    """Deterministic eval transform built on :func:`preprocess.preprocess` (the serving code path)."""

    def __init__(self, resize_size: int, crop_size: int):
        self.resize_size = resize_size
        self.crop_size = crop_size

    def __call__(self, image) -> torch.Tensor:
        return torch.from_numpy(preprocess(image, self.resize_size, self.crop_size))


def build_train_transform(cfg: DataConfig):
    """Augmentation policy.

    none     - same as evaluation (the baseline recipe)
    standard - random resized crop + flip + colour jitter
    strong   - random resized crop + flip + TrivialAugment + random erasing
    """
    if cfg.augment == "none":
        return EvalTransform(cfg.resize_size, cfg.image_size)

    ops = [T.RandomResizedCrop(cfg.image_size, scale=(0.5, 1.0)), T.RandomHorizontalFlip()]
    if cfg.augment == "standard":
        ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    elif cfg.augment == "strong":
        ops.append(T.TrivialAugmentWide())
    else:
        raise ValueError(f"Unknown augment policy '{cfg.augment}'")
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if cfg.augment == "strong":
        ops.append(T.RandomErasing(p=0.25))
    return T.Compose(ops)


def class_counts(samples: list[Sample], num_classes: int) -> np.ndarray:
    return np.bincount([s.label for s in samples], minlength=num_classes)


def class_weights(counts: np.ndarray, mode: str) -> torch.Tensor | None:
    """Loss weights against class imbalance, normalised so the average weight is 1."""
    if mode == "none":
        return None
    counts = counts.astype(np.float64)
    if mode == "inverse":
        weights = 1.0 / counts
    elif mode == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown class weighting '{mode}'")
    weights *= len(counts) / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def build_loader(cfg: DataConfig, split: str, train: bool, seed: int = 0) -> DataLoader:
    samples = read_split(cfg.root, split)
    transform = build_train_transform(cfg) if train else EvalTransform(cfg.resize_size, cfg.image_size)
    dataset = ImageDataset(cfg.root, samples, transform)
    generator = torch.Generator().manual_seed(seed)
    num_workers = cfg.num_workers if train else cfg.eval_num_workers
    if num_workers > 0:
        single_threaded_blas_for_subprocesses()

    sampler = None
    if train and cfg.balanced_sampler:
        counts = class_counts(samples, num_classes=max(s.label for s in samples) + 1)
        sample_weights = [1.0 / counts[s.label] for s in samples]
        sampler = WeightedRandomSampler(sample_weights, len(samples), replacement=True, generator=generator)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=train,
        generator=generator,
    )
