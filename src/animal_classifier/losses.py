"""Loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Knowledge distillation (Hinton et al., 2015).

    loss = (1 - alpha) * hard_loss(student, labels) + alpha * T^2 * KL(teacher_T || student_T)

    where ``*_T`` are softmax distributions at temperature T. The T^2 factor keeps the gradient scale
    of the soft term independent of the temperature.
    """

    def __init__(self, hard_loss: nn.Module, alpha: float = 0.5, temperature: float = 4.0):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.hard_loss = hard_loss
        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self, student_logits: torch.Tensor, labels: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        t = self.temperature
        soft = F.kl_div(
            F.log_softmax(student_logits / t, dim=1),
            F.log_softmax(teacher_logits / t, dim=1),
            reduction="batchmean",
            log_target=True,
        ) * (t * t)
        hard = self.hard_loss(student_logits, labels)
        return (1.0 - self.alpha) * hard + self.alpha * soft
