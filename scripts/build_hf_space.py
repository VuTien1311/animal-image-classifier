"""Assemble a ready-to-push Hugging Face Space (Gradio SDK) in dist/hf_space/.

The Space runs app/gradio_app.py on a CPU with the given checkpoint (Grad-CAM needs PyTorch, so this
uses the .pt checkpoint rather than ONNX).

Example:
    python scripts/build_hf_space.py --checkpoint outputs/convnext_tiny/best.pt
    # then push dist/hf_space/ to a new Space, see README "Deploy the demo"
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import gradio

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: Animal Classifier
emoji: 🐾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
---

Animals-10 classifier (ConvNeXt-T fine-tuned) with Grad-CAM explanations.
Source code, training and evaluation: see the project repository.
"""

SPACE_APP = '''"""Hugging Face Spaces entry point."""

from pathlib import Path

from animal_classifier.predictor import TorchPredictor
from gradio_app import build_demo

examples = sorted(str(p) for p in Path("examples").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
demo = build_demo(TorchPredictor("model.pt", device="cpu"), examples)
demo.launch()
'''

REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
timm
numpy
pillow
pyyaml
matplotlib
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--examples", type=Path, default=ROOT / "assets" / "examples")
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "hf_space")
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    shutil.copytree(
        ROOT / "src" / "animal_classifier",
        args.out / "animal_classifier",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy(ROOT / "app" / "gradio_app.py", args.out / "gradio_app.py")
    shutil.copy(args.checkpoint, args.out / "model.pt")
    if args.examples.exists():
        shutil.copytree(args.examples, args.out / "examples")

    (args.out / "app.py").write_text(SPACE_APP, encoding="utf-8")
    (args.out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    readme = SPACE_README.format(gradio_version=gradio.__version__)
    (args.out / "README.md").write_text(readme, encoding="utf-8")
    size_mb = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 2**20
    print(f"Space ready in {args.out} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
