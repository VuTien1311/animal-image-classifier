"""Grad-CAM comparison: where do different models look?

Samples --num-images test images from different classes (seeded, so the selection is not
cherry-picked) and draws one column per image: the input on the first row, then one row per model
with its Grad-CAM heat map. Each title shows the model's prediction and confidence; wrong
predictions are red.

Example:
    python scripts/gradcam_grid.py --checkpoints outputs/simple_cnn/best.pt outputs/convnext_tiny/best.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from animal_classifier.data import read_class_names, read_split
from animal_classifier.predictor import TorchPredictor
from animal_classifier.preprocess import center_crop, load_image, resize_shorter_side
from animal_classifier.viz import save_figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/figures/gradcam_comparison.png"))
    args = parser.parse_args()

    class_names = read_class_names(args.data_root)
    rng = random.Random(args.seed)
    test = read_split(args.data_root, "test")
    labels = rng.sample(range(len(class_names)), args.num_images)
    selected = [rng.choice([s for s in test if s.label == label]) for label in labels]

    predictors = [TorchPredictor(path) for path in args.checkpoints]
    names = [path.parent.name for path in args.checkpoints]
    nrows = 1 + len(predictors)
    fig, axes = plt.subplots(nrows, len(selected), figsize=(2.4 * len(selected), 2.6 * nrows), squeeze=False)

    for col, sample in enumerate(selected):
        image = load_image(Path(args.data_root) / sample.path)
        axes[0, col].imshow(np.asarray(center_crop(resize_shorter_side(image, 256), 224)))
        axes[0, col].set_title(f"input: {class_names[sample.label]}", fontsize=9)
        for row, (predictor, name) in enumerate(zip(predictors, names, strict=True), start=1):
            prediction, overlay = predictor.explain(image)
            correct = prediction["label"] == class_names[sample.label]
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(
                f"{name}: {prediction['label']} {prediction['score']:.0%}",
                fontsize=8,
                color="black" if correct else "red",
            )
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
