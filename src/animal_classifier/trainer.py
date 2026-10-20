"""Training orchestration: config -> data -> model -> optimisation loop -> checkpoints and logs.

Each run writes to ``outputs/<experiment>/``:
    config.yaml      exact config used (reproducibility)
    train.log        human-readable log
    history.csv      per-epoch train/val metrics
    tensorboard/     scalars + validation confusion matrix
    best.pt          weights of the epoch with the best validation score (for evaluation/serving)
    last.pt          full training state (for --resume)
    summary.json     best epoch, best score, training time
"""

from __future__ import annotations

import csv
import logging
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from .config import Config, OptimConfig, save_config
from .data import build_loader, class_counts, class_weights, read_class_names
from .engine import evaluate, train_one_epoch
from .losses import DistillationLoss
from .metrics import classification_metrics, confusion
from .models import create_model, load_checkpoint
from .utils import count_parameters, log_to_file, resolve_device, save_json, seed_everything
from .viz import plot_confusion_matrix


@dataclass
class EarlyStopping:
    """Tracks the best validation score; ``should_stop`` after ``patience`` epochs without improvement."""

    patience: int
    best_score: float = -math.inf
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, score: float, epoch: int) -> bool:
        if score > self.best_score:
            self.best_score, self.best_epoch, self.bad_epochs = score, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_epochs >= self.patience


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """AdamW or SGD. Biases and normalisation weights (1-D tensors) are excluded from weight decay."""
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (no_decay if param.ndim <= 1 else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg.name == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    if cfg.name == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum)
    raise ValueError(f"Unknown optimizer '{cfg.name}'")


def build_scheduler(optimizer, cfg: OptimConfig, steps_per_epoch: int, epochs: int) -> LambdaLR:
    """Per-step learning-rate multiplier: linear warm-up, then cosine decay (or constant)."""
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if cfg.scheduler == "constant":
            return 1.0
        if cfg.scheduler != "cosine":
            raise ValueError(f"Unknown scheduler '{cfg.scheduler}'")
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_multiplier)


def _build_criterion(cfg: Config, train_labels_counts, device: torch.device) -> nn.Module:
    weights = class_weights(train_labels_counts, cfg.train.class_weighting)
    hard_loss = nn.CrossEntropyLoss(
        weight=None if weights is None else weights.to(device), label_smoothing=cfg.train.label_smoothing
    )
    if cfg.distill.teacher_checkpoint:
        return DistillationLoss(hard_loss, alpha=cfg.distill.alpha, temperature=cfg.distill.temperature)
    return hard_loss


def _load_teacher(cfg: Config, device: torch.device) -> nn.Module | None:
    if not cfg.distill.teacher_checkpoint:
        return None
    teacher, _ = load_checkpoint(cfg.distill.teacher_checkpoint, device)
    if cfg.train.channels_last:
        teacher = teacher.to(memory_format=torch.channels_last)
    return teacher.requires_grad_(False)


def _write_history(history: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train(cfg: Config, resume: str | Path | None = None, overwrite: bool = False) -> Path:
    """Train according to ``cfg`` and return the run directory (see the module docstring)."""
    run_dir = cfg.run_dir
    if run_dir.exists() and resume is None:
        if not overwrite:
            raise FileExistsError(f"{run_dir} already exists: use --overwrite or --resume")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with log_to_file(run_dir / "train.log") as log:
        _fit(cfg, run_dir, log, resume)
    return run_dir


def _fit(cfg: Config, run_dir: Path, log: logging.Logger, resume: str | Path | None) -> None:
    save_config(cfg, run_dir / "config.yaml")
    seed_everything(cfg.seed)
    device = resolve_device()
    torch.backends.cudnn.benchmark = True

    # Data
    class_names = read_class_names(cfg.data.root)
    train_loader = build_loader(cfg.data, "train", train=True, seed=cfg.seed)
    val_loader = build_loader(cfg.data, "val", train=False)
    counts = class_counts(train_loader.dataset.samples, len(class_names))
    log.info(
        "Train %d images, val %d images, %d classes",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(class_names),
    )

    # Model, loss, optimisation
    model = create_model(
        cfg.model.name, len(class_names), cfg.model.pretrained, cfg.model.drop_rate, cfg.model.drop_path_rate
    ).to(device)
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    teacher = _load_teacher(cfg, device)
    criterion = _build_criterion(cfg, counts, device)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim, len(train_loader), cfg.train.epochs)
    stopper = EarlyStopping(cfg.train.early_stopping_patience)
    history: list[dict] = []
    start_epoch = 0
    log.info(
        "Model %s: %.2fM parameters on %s%s",
        cfg.model.name,
        count_parameters(model) / 1e6,
        device,
        f" (teacher: {cfg.distill.teacher_checkpoint})" if teacher is not None else "",
    )

    # Resume: restore weights, optimiser/scheduler state, early-stopping counters and history.
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        stopper = EarlyStopping(**state["early_stopping"])
        history, start_epoch = state["history"], state["epoch"]
        log.info("Resumed from %s at epoch %d", resume, start_epoch)

    metadata = {"model_name": cfg.model.name, "class_names": class_names, "config": cfg.to_dict()}
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    run_start = time.perf_counter()

    # Epoch loop: train, validate, log, checkpoint, maybe stop early.
    for epoch in range(start_epoch + 1, cfg.train.epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            amp=cfg.train.amp,
            channels_last=cfg.train.channels_last,
            grad_clip_norm=cfg.train.grad_clip_norm,
            teacher=teacher,
            desc=f"epoch {epoch}/{cfg.train.epochs}",
        )
        val = evaluate(model, val_loader, device, amp=cfg.train.amp, channels_last=cfg.train.channels_last)
        val_metrics = classification_metrics(val.labels, val.predictions, class_names)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val.loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_time_s": time.perf_counter() - epoch_start,
        }
        history.append(row)
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key.replace("_", "/", 1), value, epoch)

        # best.pt: weights + metadata only (for evaluation and serving); last.pt: everything to resume.
        if stopper.update(val_metrics[cfg.train.monitor], epoch):
            torch.save(
                {**metadata, "model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics},
                run_dir / "best.pt",
            )
            cm = confusion(val.labels, val.predictions, len(class_names))
            writer.add_figure("val/confusion_matrix", plot_confusion_matrix(cm, class_names), epoch)

        torch.save(
            {
                **metadata,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "early_stopping": asdict(stopper),
                "history": history,
                "epoch": epoch,
            },
            run_dir / "last.pt",
        )
        _write_history(history, run_dir / "history.csv")
        log.info(
            "epoch %3d | train loss %.4f acc %.4f | val loss %.4f acc %.4f macro-F1 %.4f | %.0fs%s",
            epoch,
            row["train_loss"],
            row["train_accuracy"],
            row["val_loss"],
            row["val_accuracy"],
            row["val_macro_f1"],
            row["epoch_time_s"],
            "  *best*" if stopper.best_epoch == epoch else "",
        )
        if stopper.should_stop:
            log.info("Early stopping: no improvement for %d epochs", stopper.patience)
            break

    writer.close()
    summary = {
        "experiment": cfg.experiment,
        "model_name": cfg.model.name,
        "parameters": count_parameters(model),
        "epochs_trained": len(history),
        "best_epoch": stopper.best_epoch,
        f"best_val_{cfg.train.monitor}": stopper.best_score,
        "train_time_min": (time.perf_counter() - run_start) / 60,
    }
    save_json(summary, run_dir / "summary.json")
    log.info("Done: best val %s %.4f at epoch %d", cfg.train.monitor, stopper.best_score, stopper.best_epoch)
