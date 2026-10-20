"""Evaluate a trained checkpoint on the held-out test split.

The test split is used only here, once per final model; model selection happens on val.

Writes:
    results/metrics/<experiment>.json            accuracy, macro-F1, per-class metrics, top confusions
    results/figures/confusion_<experiment>.png   row-normalised confusion matrix
    outputs/<experiment>/test_predictions.npz    logits, labels, image paths (for error analysis)

Example:
    python scripts/evaluate.py --checkpoint outputs/convnext_tiny/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from animal_classifier.config import Config
from animal_classifier.data import build_loader
from animal_classifier.engine import evaluate
from animal_classifier.metrics import classification_metrics, confusion, top_confusions
from animal_classifier.models import load_checkpoint
from animal_classifier.utils import count_parameters, resolve_device, save_json
from animal_classifier.viz import plot_confusion_matrix, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--data-root", help="override the dataset root stored in the checkpoint")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    device = resolve_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    cfg = Config.from_dict(checkpoint["config"])
    cfg.data.batch_size, cfg.data.eval_num_workers = args.batch_size, args.num_workers
    if args.data_root:
        cfg.data.root = args.data_root
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)

    class_names = checkpoint["class_names"]
    loader = build_loader(cfg.data, args.split, train=False)
    result = evaluate(model, loader, device, amp=cfg.train.amp, channels_last=cfg.train.channels_last)

    metrics = classification_metrics(result.labels, result.predictions, class_names)
    cm = confusion(result.labels, result.predictions, len(class_names))
    report = {
        "experiment": cfg.experiment,
        "model_name": cfg.model.name,
        "split": args.split,
        "num_images": len(result.labels),
        "parameters": count_parameters(model),
        "best_epoch": checkpoint["epoch"],
        "loss": result.loss,
        **metrics,
        "top_confusions": top_confusions(cm, class_names),
        "confusion_matrix": cm.tolist(),
    }

    suffix = "" if args.split == "test" else f"_{args.split}"
    save_json(report, args.results / "metrics" / f"{cfg.experiment}{suffix}.json")
    title = (
        f"{cfg.experiment} ({args.split}): accuracy {metrics['accuracy']:.3f}, "
        f"macro-F1 {metrics['macro_f1']:.3f}"
    )
    save_figure(
        plot_confusion_matrix(cm, class_names, title),
        args.results / "figures" / f"confusion_{cfg.experiment}{suffix}.png",
    )
    np.savez_compressed(
        args.checkpoint.parent / f"{args.split}_predictions.npz",
        logits=result.logits,
        labels=result.labels,
        paths=np.array([sample.path for sample in loader.dataset.samples]),
        class_names=np.array(class_names),
    )
    print(
        f"{cfg.experiment} on {args.split}: accuracy {metrics['accuracy']:.4f} | "
        f"macro-F1 {metrics['macro_f1']:.4f} | balanced accuracy {metrics['balanced_accuracy']:.4f}"
    )
    for pair in report["top_confusions"]:
        print(f"  {pair['true']:>9} -> {pair['predicted']:<9} {pair['count']}")


if __name__ == "__main__":
    main()
