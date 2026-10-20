"""CLIP baselines on the same splits: zero-shot and a linear probe on frozen image embeddings.

zero-shot     No training images: each test image is assigned the class whose text embedding is
              closest, using prompts such as "a photo of a cat." averaged over several templates.
linear probe  Logistic regression on frozen, L2-normalised CLIP image embeddings of the train split;
              the regularisation strength C is chosen on val (macro-F1), test is used once.

Default model: open_clip ViT-L/14 with Apple's DFN-2B weights. These weights were trained with
QuickGELU activations, so the architecture must be "ViT-L-14-quickgelu"; plain "ViT-L-14" (GELU)
loads without error but silently runs a slightly different network.

Writes results/metrics/clip_{zero_shot,linear_probe}.json (same format as scripts/evaluate.py),
confusion matrices, and outputs/clip_*/test_predictions.npz for error analysis.

Example:
    python scripts/clip_eval.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from animal_classifier.data import ImageDataset, read_class_names, read_split
from animal_classifier.metrics import classification_metrics, confusion, top_confusions
from animal_classifier.utils import resolve_device, save_json, seed_everything
from animal_classifier.viz import plot_confusion_matrix, save_figure

TEMPLATES = (
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a blurry photo of a {}.",
    "a low resolution photo of a {}.",
    "a cropped photo of a {}.",
    "a photo of the {}, a type of animal.",
    "a drawing of a {}.",
    "a cartoon {}.",
)


@torch.no_grad()
def image_embeddings(model, preprocess, root: str, split: str, device, batch_size: int, workers: int):
    samples = read_split(root, split)
    loader = DataLoader(ImageDataset(root, samples, preprocess), batch_size=batch_size, num_workers=workers)
    features = []
    for images, _ in tqdm(loader, desc=f"embed {split}", leave=False):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            features.append(model.encode_image(images.to(device)).float().cpu())
    features = torch.nn.functional.normalize(torch.cat(features), dim=-1).numpy()
    return features, np.array([s.label for s in samples]), np.array([s.path for s in samples])


@torch.no_grad()
def text_classifier(model, tokenizer, class_names: list[str], device) -> np.ndarray:
    """One embedding per class: the normalised mean of its normalised prompt embeddings."""
    weights = []
    for name in class_names:
        tokens = tokenizer([template.format(name) for template in TEMPLATES]).to(device)
        embeddings = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1)
        weights.append(torch.nn.functional.normalize(embeddings.mean(dim=0), dim=-1))
    return torch.stack(weights).cpu().numpy()


def save_run(
    name: str,
    model_label: str,
    logits: np.ndarray,
    labels: np.ndarray,
    paths: np.ndarray,
    class_names: list[str],
    extra: dict,
    results: Path,
    outputs: Path,
) -> dict:
    predictions = logits.argmax(axis=1)
    metrics = classification_metrics(labels, predictions, class_names)
    cm = confusion(labels, predictions, len(class_names))
    report = {
        "experiment": name,
        "model_name": model_label,
        "split": "test",
        "num_images": len(labels),
        **extra,
        **metrics,
        "top_confusions": top_confusions(cm, class_names),
        "confusion_matrix": cm.tolist(),
    }
    save_json(report, results / "metrics" / f"{name}.json")
    title = f"{name} (test) - accuracy {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f}"
    save_figure(plot_confusion_matrix(cm, class_names, title), results / "figures" / f"confusion_{name}.png")
    (outputs / name).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outputs / name / "test_predictions.npz",
        logits=logits,
        labels=labels,
        paths=paths,
        class_names=np.array(class_names),
    )
    print(f"{name}: accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="ViT-L-14-quickgelu")
    parser.add_argument("--pretrained", default="dfn2b")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device()
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    class_names = read_class_names(args.data_root)
    model_label = f"{args.model} ({args.pretrained})"
    encoder_params = sum(p.numel() for p in model.visual.parameters())

    start = time.perf_counter()
    embedded = {
        split: image_embeddings(
            model, preprocess, args.data_root, split, device, args.batch_size, args.workers
        )
        for split in ("train", "val", "test")
    }
    print(f"Embedded all splits in {time.perf_counter() - start:.0f}s")
    test_x, test_y, test_paths = embedded["test"]

    # Zero-shot: cosine similarity to class text embeddings (scaled like CLIP's logits).
    classifier = text_classifier(model, tokenizer, class_names, device)
    zero_shot_logits = 100.0 * test_x @ classifier.T
    save_run(
        "clip_zero_shot",
        model_label,
        zero_shot_logits,
        test_y,
        test_paths,
        class_names,
        {"parameters": encoder_params, "templates": list(TEMPLATES)},
        args.results,
        args.outputs,
    )

    # Linear probe: choose C on val, refit on train, evaluate once on test.
    (train_x, train_y, _), (val_x, val_y, _) = embedded["train"], embedded["val"]
    search = {}
    for c in (0.1, 0.3, 1.0, 3.0, 10.0):
        probe = LogisticRegression(C=c, max_iter=2000).fit(train_x, train_y)
        search[c] = classification_metrics(val_y, probe.predict(val_x), class_names)["macro_f1"]
        print(f"  C={c:<5} val macro-F1 {search[c]:.4f}")
    best_c = max(search, key=search.get)
    probe = LogisticRegression(C=best_c, max_iter=2000).fit(train_x, train_y)
    save_run(
        "clip_linear_probe",
        model_label,
        probe.decision_function(test_x),
        test_y,
        test_paths,
        class_names,
        {
            "parameters": encoder_params + probe.coef_.size + probe.intercept_.size,
            "best_C": best_c,
            "val_macro_f1_by_C": {str(k): v for k, v in search.items()},
        },
        args.results,
        args.outputs,
    )


if __name__ == "__main__":
    main()
