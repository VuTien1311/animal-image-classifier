"""Aggregate the JSON results into results/RESULTS.md and two summary figures.

Every number in the README comes from RESULTS.md, which is generated from the files written by
evaluate.py, clip_eval.py and deploy.py; nothing is typed in by hand.

Writes results/RESULTS.md, results/figures/accuracy_vs_latency.png, results/figures/learning_curves.png.

Example:
    python scripts/make_report.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker

from animal_classifier.utils import load_json
from animal_classifier.viz import save_figure

# experiment -> (display name, what the weights were trained on before Animals-10)
EXPERIMENTS = {
    "simple_cnn_baseline": ("SimpleCNN, baseline recipe", "none"),
    "simple_cnn": ("SimpleCNN, improved recipe", "none"),
    "simple_cnn_class_weighted": ("SimpleCNN, improved recipe + class-weighted loss", "none"),
    "simple_cnn_distill": ("SimpleCNN, distilled from ConvNeXt-T", "none (teacher: ConvNeXt-T)"),
    "efficientnet_b0": ("EfficientNet-B0, fine-tuned", "ImageNet-1k"),
    "resnet50": ("ResNet-50, fine-tuned", "ImageNet-1k"),
    "convnext_tiny": ("ConvNeXt-T, fine-tuned", "ImageNet-22k -> 1k"),
    "clip_zero_shot": ("CLIP ViT-L/14, zero-shot", "DFN-2B image-text pairs"),
    "clip_linear_probe": ("CLIP ViT-L/14, linear probe", "DFN-2B image-text pairs"),
}


def load_all(folder: Path) -> dict[str, dict]:
    """``folder/<experiment>.json`` for every known experiment that has one, in EXPERIMENTS order."""
    return {
        name: load_json(folder / f"{name}.json") for name in EXPERIMENTS if (folder / f"{name}.json").exists()
    }


def markdown_table(header: list[str], rows: list[list[str]], numeric_from: int = 1) -> list[str]:
    """Markdown table; columns from ``numeric_from`` on are right-aligned."""
    align = ["---" if i < numeric_from else "---:" for i in range(len(header))]
    return ["| " + " | ".join(row) + " |" for row in [header, align, *rows]]


def ms(latency: dict, key: str) -> str:
    return f"{latency[key]['median_ms']:.1f} ms" if key in latency else "-"


def model_section(metrics: dict, outputs: Path) -> list[str]:
    rows = []
    for name, m in metrics.items():
        display, pretraining = EXPERIMENTS[name]
        summary_path = outputs / name / "summary.json"
        summary = load_json(summary_path) if summary_path.exists() else None
        rows.append(
            [
                display,
                pretraining,
                f"{m['parameters'] / 1e6:.1f}M",
                f"{summary['epochs_trained']} ({summary['best_epoch']})" if summary else "-",
                f"{summary['train_time_min']:.0f} min" if summary else "-",
                f"{m['accuracy']:.2%}",
                f"{m['macro_f1']:.4f}",
                f"{m['balanced_accuracy']:.2%}",
            ]
        )
    header = [
        "Model",
        "Pre-training",
        "Params",
        "Epochs (best)",
        "Train time",
        "Accuracy",
        "Macro-F1",
        "Balanced acc.",
    ]
    return ["## Model comparison (test set)", "", *markdown_table(header, rows, numeric_from=2)]


def deployment_section(deployments: dict, latency: dict) -> list[str]:
    env = latency["environment"]
    rows = []
    for name, d in deployments.items():
        q, test, size, times = d["quantization"], d["test"], d["file_mb"], latency["models"][name]
        if q["shipped"]:
            int8 = f"{size['int8']:.1f} MB"
            accuracy = f"{test['fp32']['accuracy']:.2%} -> {test['int8']['accuracy']:.2%}"
        else:
            int8 = f"not shipped (-{q['val_drop_points']:.1f} pts on val)"
            accuracy = f"{test['fp32']['accuracy']:.2%} (FP32 only)"
        rows.append(
            [
                EXPERIMENTS[name][0],
                f"{size['fp32']:.1f} MB",
                int8,
                ms(times, "torch_cpu_fp32"),
                ms(times, "onnx_cpu_fp32"),
                ms(times, "onnx_cpu_int8"),
                ms(times, "torch_gpu_bf16"),
                accuracy,
            ]
        )
    header = [
        "Model",
        "Size FP32",
        "Size INT8",
        "PyTorch CPU",
        "ONNX CPU FP32",
        "ONNX CPU INT8",
        "PyTorch GPU (bf16)",
        "Test acc. FP32 -> INT8",
    ]
    return [
        "## Deployment: size and single-image latency",
        "",
        f"Batch size 1, median of {latency['runs']} runs after {latency['warmup']} warm-up runs; all models "
        f"measured in one session by `scripts/benchmark.py`. CPU: {env['cpu']} (PyTorch uses "
        f"{env['cpu_threads']} threads); GPU: {env['gpu']}. ONNX Runtime {env['onnxruntime']}, "
        f"PyTorch {env['torch']}.",
        "",
        *markdown_table(header, rows),
    ]


def quantization_section(deployments: dict) -> list[str]:
    settings = list(next(iter(deployments.values()))["quantization"]["val_accuracy"])[1:]  # drop "fp32"
    rows = []
    for name, d in deployments.items():
        val = d["quantization"]["val_accuracy"]
        deltas = [f"{(val[s]['accuracy'] - val['fp32']['accuracy']) * 100:+.2f}" for s in settings]
        rows.append(
            [EXPERIMENTS[name][0], f"{val['fp32']['accuracy']:.2%}", *deltas, d["quantization"]["selected"]]
        )
    return [
        "## INT8 quantisation study (val set)",
        "",
        "Change in val accuracy (points) for each post-training quantisation setting. The best setting",
        "per model is kept; INT8 is shipped only if it costs at most 1 point.",
        "",
        *markdown_table(["Model", "FP32", *settings, "Selected"], rows),
    ]


def per_class_section(metrics: dict) -> list[str]:
    shown = [
        n for n in ("simple_cnn_baseline", "simple_cnn", "convnext_tiny", "clip_linear_probe") if n in metrics
    ]
    first = metrics[shown[0]]["per_class"]
    rows = [
        [cls, str(first[cls]["support"]), *(f"{metrics[n]['per_class'][cls]['f1']:.3f}" for n in shown)]
        for cls in first
    ]
    header = ["Class", "Test images", *(EXPERIMENTS[n][0] for n in shown)]
    return ["## Per-class F1 (test set)", "", *markdown_table(header, rows)]


def confusion_section(metrics: dict) -> list[str]:
    best = max(metrics, key=lambda n: metrics[n]["macro_f1"])
    pairs = [f"- {p['true']} -> {p['predicted']}: {p['count']}" for p in metrics[best]["top_confusions"]]
    return [f"## Most frequent confusions of the best model ({EXPERIMENTS[best][0]})", "", *pairs]


def save_accuracy_vs_latency(metrics: dict, deployments: dict, latency: dict, path: Path) -> None:
    """Test macro-F1 vs. ONNX Runtime CPU latency; an arrow shows the INT8 speed-up where it was shipped."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, name in enumerate(deployments):
        times, y, color = latency["models"][name], metrics[name]["macro_f1"], f"C{i}"
        fp32 = times["onnx_cpu_fp32"]["median_ms"]
        ax.scatter(fp32, y, s=60, color=color, label=EXPERIMENTS[name][0])
        if "onnx_cpu_int8" in times:
            int8 = times["onnx_cpu_int8"]["median_ms"]
            ax.annotate("", xy=(int8, y), xytext=(fp32, y), arrowprops={"arrowstyle": "->", "color": color})
            ax.scatter(int8, y, s=45, marker="D", color=color)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("CPU latency per image (ms, log scale): circle = FP32, diamond = INT8")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Accuracy vs. CPU latency (ONNX Runtime, batch size 1)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    save_figure(fig, path)


def save_learning_curves(outputs: Path, path: Path) -> None:
    """Validation macro-F1 per epoch for every trained run (read from outputs/<run>/history.csv)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (display, _) in EXPERIMENTS.items():
        history = outputs / name / "history.csv"
        if history.exists():
            with open(history, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            ax.plot(
                [int(r["epoch"]) for r in rows],
                [float(r["val_macro_f1"]) for r in rows],
                marker="o",
                markersize=3,
                label=display,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Learning curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    metrics = load_all(args.results / "metrics")
    if not metrics:
        raise SystemExit("No results/metrics/*.json found: run scripts/evaluate.py first")
    deployments = load_all(args.results / "deployment")
    latency_path = args.results / "deployment" / "latency.json"
    num_test = next(iter(metrics.values()))["num_images"]

    sections = [
        [
            "# Results",
            "",
            "Generated by `scripts/make_report.py` from the JSON files in `results/`. "
            f"Test metrics use the held-out test split ({num_test} images), which was not used for model "
            "selection, early stopping or choosing quantisation settings (the val split is used for those).",
        ],
        model_section(metrics, args.outputs),
    ]
    if deployments and latency_path.exists():
        latency = load_json(latency_path)
        sections += [deployment_section(deployments, latency), quantization_section(deployments)]
        figure = args.results / "figures" / "accuracy_vs_latency.png"
        save_accuracy_vs_latency(metrics, deployments, latency, figure)
    sections += [per_class_section(metrics), confusion_section(metrics)]

    report = "\n\n".join("\n".join(section) for section in sections) + "\n"
    (args.results / "RESULTS.md").write_text(report, encoding="utf-8")
    save_learning_curves(args.outputs, args.results / "figures" / "learning_curves.png")
    print(f"Wrote {args.results / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
