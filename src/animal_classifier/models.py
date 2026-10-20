"""Model factory: a small from-scratch CNN baseline plus any architecture from timm."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 conv-BN-LeakyReLU layers followed by 2x2 max pooling (halves the resolution)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


class SimpleCNN(nn.Module):
    """From-scratch baseline: 5 conv blocks (8 -> 128 channels) + a 3-layer MLP head, 4.0M parameters.

    The adaptive pool before the head fixes the feature map at 7x7, so the model accepts any input
    size. It is deliberately simple: the point of the project is to measure how far training recipe,
    pre-training and distillation move a model, starting from this reference.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        widths = [3, 8, 16, 32, 64, 128]
        self.features = nn.Sequential(*(_conv_block(i, o) for i, o in pairwise(widths)))
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1] * 7 * 7, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),  # raw logits: no activation before CrossEntropyLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))


def create_model(
    name: str,
    num_classes: int,
    pretrained: bool = False,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes, dropout=drop_rate)

    import timm

    kwargs: dict[str, Any] = {"pretrained": pretrained, "num_classes": num_classes, "drop_rate": drop_rate}
    if drop_path_rate > 0:
        kwargs["drop_path_rate"] = drop_path_rate
    return timm.create_model(name, **kwargs)


# Last spatial feature map of each family, used as the Grad-CAM target.
_GRADCAM_LAYERS = (
    ("simple_cnn", "features.4.5"),  # last activation before the final max-pool (14x14)
    ("resnet", "layer4"),
    ("convnext", "stages.3"),
    ("efficientnet", "bn2"),
    ("mobilenetv3", "blocks"),
)


def gradcam_layer(model: nn.Module, model_name: str) -> nn.Module:
    for prefix, layer_name in _GRADCAM_LAYERS:
        if model_name.startswith(prefix):
            return model.get_submodule(layer_name)
    raise ValueError(f"No Grad-CAM layer registered for '{model_name}'")


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a model from a checkpoint written by the trainer; returns (model in eval mode, checkpoint)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = create_model(checkpoint["model_name"], num_classes=len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
