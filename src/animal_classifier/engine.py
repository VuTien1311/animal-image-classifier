"""One training epoch and one evaluation pass."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class EvalResult:
    loss: float  # mean cross-entropy (no label smoothing / class weights, so runs are comparable)
    logits: np.ndarray  # (N, num_classes) float32
    labels: np.ndarray  # (N,)

    @property
    def predictions(self) -> np.ndarray:
        return self.logits.argmax(axis=1)

    @property
    def probabilities(self) -> np.ndarray:
        shifted = self.logits - self.logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)


def autocast(device: torch.device, enabled: bool):
    """bfloat16 mixed precision on CUDA. Unlike float16 it needs no GradScaler."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _to_device(images: torch.Tensor, labels: torch.Tensor, device: torch.device, channels_last: bool):
    images = images.to(device, non_blocking=True)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images, labels.to(device, non_blocking=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    *,
    amp: bool = True,
    channels_last: bool = False,
    grad_clip_norm: float = 0.0,
    teacher: nn.Module | None = None,
    desc: str = "train",
) -> dict[str, float]:
    """Train for one epoch. With a ``teacher`` the criterion is called as ``criterion(s, y, t)``."""
    model.train()
    # Accumulate on the device to avoid a GPU->CPU sync on every step.
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    seen = 0

    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for step, (images, labels) in enumerate(progress):
        images, labels = _to_device(images, labels, device, channels_last)
        with autocast(device, amp):
            logits = model(images)
            teacher_logits = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(images)

        logits = logits.float()  # compute the loss in float32
        if teacher_logits is None:
            loss = criterion(logits, labels)
        else:
            loss = criterion(logits, labels, teacher_logits.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        loss_sum += loss.detach() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum()
        seen += labels.size(0)
        if step % 25 == 0:
            progress.set_postfix(loss=f"{loss_sum.item() / seen:.4f}")

    return {"loss": loss_sum.item() / seen, "accuracy": correct.item() / seen}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool = True,
    channels_last: bool = False,
    desc: str = "eval",
) -> EvalResult:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, all_logits, all_labels = 0.0, [], []
    for images, labels in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images, labels = _to_device(images, labels, device, channels_last)
        with autocast(device, amp):
            logits = model(images)
        logits = logits.float()
        loss_sum += criterion(logits, labels).item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    labels = torch.cat(all_labels).numpy()
    return EvalResult(loss=loss_sum / len(labels), logits=torch.cat(all_logits).numpy(), labels=labels)
