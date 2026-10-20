"""Matplotlib figures used in TensorBoard and the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are only written to files / TensorBoard

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str = "Confusion matrix"):
    """Row-normalised confusion matrix (each row = recall distribution of one true class)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    normalised = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    ax.set(xticks=ticks, yticks=ticks, xlabel="Predicted label", ylabel="True label", title=title)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = normalised[i, j]
            if value >= 0.005:
                color = "white" if value > 0.5 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.tight_layout()
    return fig


def plot_image_grid(images: list[np.ndarray], titles: list[str], ncols: int = 5, cell_size: float = 2.6):
    nrows = max(1, int(np.ceil(len(images) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_size, nrows * (cell_size + 0.35)))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    # The last row may have more axes than images; the extra axes stay blank.
    for ax, image, title in zip(np.atleast_1d(axes).ravel(), images, titles, strict=False):
        ax.imshow(image)
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
