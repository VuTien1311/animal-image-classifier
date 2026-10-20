"""Grad-CAM (Selvaraju et al., 2017): which image regions drive a prediction.

The class score's gradient w.r.t. a convolutional feature map is averaged per channel to weight that
channel; the weighted sum of channels (after ReLU) is a coarse heat map of the evidence for the class.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output: torch.Tensor) -> None:
        # The hook stays registered, so it also fires for ordinary no-grad predictions; only
        # Grad-CAM passes (which enable gradients) need the activations and their gradient.
        if output.requires_grad:
            self._activations = output
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad

    def __call__(self, image: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``image``: normalised 1x3xHxW tensor. Returns (HxW heat map in [0, 1], class probabilities)."""
        self.model.eval()
        with torch.enable_grad():
            image = image.requires_grad_(True)
            logits = self.model(image)
            probs = logits.softmax(dim=1)[0]
            if class_index is None:
                class_index = int(probs.argmax())
            self.model.zero_grad(set_to_none=True)
            logits[0, class_index].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # one weight per channel
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-8)
        return cam.detach().cpu().numpy(), probs.detach().cpu().numpy()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] heat map (jet colour map) over an HxWx3 uint8 image."""
    from matplotlib import colormaps

    colored = colormaps["jet"](heatmap)[..., :3]
    blended = (1 - alpha) * image.astype(np.float32) / 255.0 + alpha * colored
    return (blended.clip(0, 1) * 255).round().astype(np.uint8)
