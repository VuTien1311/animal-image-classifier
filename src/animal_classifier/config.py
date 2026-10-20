"""Typed experiment configuration, loaded from YAML with dotted command-line overrides.

Example:
    cfg = load_config("configs/convnext_tiny.yaml", overrides=["train.epochs=3", "data.batch_size=32"])
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data/processed"
    image_size: int = 224
    resize_size: int = 256
    augment: str = "standard"  # none | standard | strong
    batch_size: int = 64
    num_workers: int = 4  # train loader; each worker is a separate process (~1 GB on Windows)
    eval_num_workers: int = 2
    balanced_sampler: bool = False


@dataclass
class ModelConfig:
    name: str = "simple_cnn"  # "simple_cnn" or any timm model name (optionally with a pretrained tag)
    pretrained: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0


@dataclass
class OptimConfig:
    name: str = "adamw"  # adamw | sgd
    lr: float = 1e-3
    weight_decay: float = 0.05
    momentum: float = 0.9
    scheduler: str = "cosine"  # cosine | constant
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.01


@dataclass
class TrainConfig:
    epochs: int = 30
    amp: bool = True
    channels_last: bool = True
    label_smoothing: float = 0.0
    class_weighting: str = "none"  # none | inverse | sqrt_inverse
    grad_clip_norm: float = 0.0
    early_stopping_patience: int = 0  # 0 disables early stopping
    monitor: str = "macro_f1"


@dataclass
class DistillConfig:
    teacher_checkpoint: str | None = None
    alpha: float = 0.5  # weight of the soft (teacher) term
    temperature: float = 4.0


@dataclass
class Config:
    experiment: str = "default"
    seed: int = 42
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Config:
        return _build(cls, values)


def _coerce(hint: Any, value: Any, key: str) -> Any:
    # PyYAML (YAML 1.1) reads "1e-3" as a string, so numeric fields are coerced explicitly.
    if hint is float and isinstance(value, (int, str)) and not isinstance(value, bool):
        return float(value)
    if hint is int and isinstance(value, str):
        return int(value)
    if hint in (int, float, bool, str) and not isinstance(value, hint):
        raise TypeError(f"Config key '{key}' expects {hint.__name__}, got {value!r}")
    return value


def _build(cls: type, values: dict[str, Any], prefix: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise KeyError(f"Unknown config keys in '{prefix or 'root'}': {unknown}")

    kwargs = {}
    for name, value in values.items():
        hint, key = hints[name], f"{prefix}{name}"
        if dataclasses.is_dataclass(hint):
            if not isinstance(value, dict):
                raise TypeError(f"Config section '{key}' must be a mapping")
            kwargs[name] = _build(hint, value, prefix=f"{key}.")
        elif isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
            kwargs[name] = value
        else:
            kwargs[name] = _coerce(hint, value, key)
    return cls(**kwargs)


def apply_overrides(values: dict[str, Any], overrides: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides in place; values are parsed as YAML scalars."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like 'section.key=value', got '{item}'")
        dotted, raw = item.split("=", 1)
        *parents, leaf = dotted.strip().split(".")
        node = values
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return values


def load_config(path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> Config:
    path = Path(path)
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values.setdefault("experiment", path.stem)
    apply_overrides(values, overrides)
    return Config.from_dict(values)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
