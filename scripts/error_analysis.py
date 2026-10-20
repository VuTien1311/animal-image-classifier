"""Error analysis on the test set, from the predictions saved by scripts/evaluate.py.

1. The most confident mistakes, as an image grid. For a strong model these are often label errors
   in the dataset rather than model errors (see results/error_analysis/label_audit.md).
2. Suspected label issues: images where the model gives the annotated class less than
   --max-label-prob while being at least --min-confidence sure of another class. They are written to
   a CSV for manual review.

Writes results/figures/errors_<experiment>.png and, in results/error_analysis/,
label_issues_<experiment>.csv and summary_<experiment>.json.

Example:
    python scripts/error_analysis.py --predictions outputs/convnext_tiny/test_predictions.npz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from animal_classifier.predictor import softmax
from animal_classifier.preprocess import load_image
from animal_classifier.utils import save_json
from animal_classifier.viz import plot_image_grid, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--predictions", required=True, type=Path, help="outputs/<experiment>/test_predictions.npz"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--top", type=int, default=20, help="mistakes shown in the figure")
    parser.add_argument("--max-label-prob", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    experiment = args.predictions.parent.name
    data = np.load(args.predictions)
    probs, labels, paths = softmax(data["logits"]), data["labels"], data["paths"]
    class_names = data["class_names"].tolist()
    predicted, confidence = probs.argmax(axis=1), probs.max(axis=1)
    label_prob = probs[np.arange(len(labels)), labels]
    wrong = np.flatnonzero(predicted != labels)

    # 1. Most confident mistakes.
    worst = wrong[np.argsort(-confidence[wrong])][: args.top]
    images = [np.asarray(load_image(args.data_root / paths[i]).resize((224, 168))) for i in worst]
    titles = [
        f"label: {class_names[labels[i]]}\npred: {class_names[predicted[i]]} {confidence[i]:.0%}"
        for i in worst
    ]
    save_figure(
        plot_image_grid(images, titles, ncols=5), args.results / "figures" / f"errors_{experiment}.png"
    )

    # 2. Suspected label issues, most confident first.
    suspects = np.flatnonzero((label_prob < args.max_label_prob) & (confidence >= args.min_confidence))
    suspects = suspects[np.argsort(-confidence[suspects])]
    out_dir = args.results / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"label_issues_{experiment}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "given_label", "predicted_label", "confidence", "prob_of_given_label"])
        for i in suspects:
            given, guess = class_names[labels[i]], class_names[predicted[i]]
            writer.writerow([paths[i], given, guess, f"{confidence[i]:.4f}", f"{label_prob[i]:.4f}"])

    summary = {
        "predictions": args.predictions.as_posix(),
        "test_images": int(len(labels)),
        "errors": int(len(wrong)),
        "suspected_label_issues": int(len(suspects)),
        "criteria": {"max_label_prob": args.max_label_prob, "min_confidence": args.min_confidence},
    }
    save_json(summary, out_dir / f"summary_{experiment}.json")
    print(f"{experiment}: {len(wrong)}/{len(labels)} test errors, {len(suspects)} possible label errors")


if __name__ == "__main__":
    main()
