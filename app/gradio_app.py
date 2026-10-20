"""Interactive demo: upload a photo, get the top-3 classes and a Grad-CAM heat map.

Run:
    python app/gradio_app.py --checkpoint outputs/convnext_tiny/best.pt

Also the Hugging Face Spaces entry point (see scripts/build_hf_space.py).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gradio as gr

from animal_classifier.predictor import TorchPredictor

DESCRIPTION = """
Classifies photos into 10 animal classes: butterfly, cat, chicken, cow, dog, elephant, horse, sheep,
spider, squirrel. The heat map (Grad-CAM) highlights the image regions that drove the prediction.
"""


def build_demo(predictor: TorchPredictor, examples: list[str] | None = None) -> gr.Blocks:
    def classify(image):
        if image is None:
            return None, None
        prediction, overlay = predictor.explain(image)
        return {item["label"]: item["score"] for item in prediction["top_k"]}, overlay

    with gr.Blocks(title="Animal Classifier") as demo:
        gr.Markdown(f"# Animal Classifier\n{DESCRIPTION}\nModel: `{predictor.model_name}`")
        with gr.Row():
            image = gr.Image(type="pil", label="Photo")
            with gr.Column():
                label = gr.Label(num_top_classes=3, label="Prediction")
                heatmap = gr.Image(label="Grad-CAM", interactive=False)
        gr.Button("Classify", variant="primary").click(classify, inputs=image, outputs=[label, heatmap])
        if examples:
            gr.Examples(examples, inputs=image)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT", "outputs/convnext_tiny/best.pt"))
    parser.add_argument("--examples", type=Path, default=Path("assets/examples"))
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    examples = sorted(
        str(p) for p in args.examples.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    demo = build_demo(TorchPredictor(args.checkpoint, device=args.device), examples)
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
