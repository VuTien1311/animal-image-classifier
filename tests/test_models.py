import numpy as np
import pytest
import torch
import torch.nn as nn

from animal_classifier.gradcam import GradCAM, overlay_heatmap
from animal_classifier.losses import DistillationLoss
from animal_classifier.models import SimpleCNN, create_model, gradcam_layer, load_checkpoint


@pytest.mark.parametrize("size", [224, 160, 96])
def test_simple_cnn_accepts_any_input_size(size):
    model = SimpleCNN(num_classes=10).eval()
    assert model(torch.randn(2, 3, size, size)).shape == (2, 10)


def test_simple_cnn_outputs_raw_logits():
    # CrossEntropyLoss needs unbounded logits; a final ReLU (a common mistake) would forbid negatives.
    torch.manual_seed(0)
    logits = SimpleCNN(num_classes=10).eval()(torch.randn(8, 3, 96, 96))
    assert (logits < 0).any()


@pytest.mark.parametrize("name", ["simple_cnn", "resnet18"])
def test_gradcam_heatmap(name):
    torch.manual_seed(0)
    model = create_model(name, num_classes=5).eval()
    heatmap, probs = GradCAM(model, gradcam_layer(model, name))(torch.randn(1, 3, 96, 96))
    assert heatmap.shape == (96, 96)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    assert probs.shape == (5,) and probs.sum() == pytest.approx(1.0, abs=1e-5)
    assert overlay_heatmap(np.zeros((96, 96, 3), np.uint8), heatmap).dtype == np.uint8


def test_checkpoint_round_trip(tmp_path):
    model = create_model("simple_cnn", num_classes=3).eval()
    torch.save(
        {"model": model.state_dict(), "model_name": "simple_cnn", "class_names": ["a", "b", "c"]},
        tmp_path / "best.pt",
    )
    restored, checkpoint = load_checkpoint(tmp_path / "best.pt")
    x = torch.randn(1, 3, 64, 64)
    assert checkpoint["class_names"] == ["a", "b", "c"]
    assert torch.allclose(model(x), restored(x))


def test_distillation_loss():
    torch.manual_seed(0)
    student, teacher = torch.randn(4, 5, requires_grad=True), torch.randn(4, 5)
    labels = torch.tensor([0, 1, 2, 3])
    ce = nn.CrossEntropyLoss()

    # alpha = 0 is plain cross-entropy
    plain = DistillationLoss(ce, alpha=0.0)(student, labels, teacher)
    assert plain.item() == pytest.approx(ce(student, labels).item())
    # a student that matches the teacher pays no distillation penalty
    matched = DistillationLoss(ce, alpha=0.5)(teacher, labels, teacher)
    assert matched.item() == pytest.approx(0.5 * ce(teacher, labels).item(), abs=1e-6)
    # gradients flow to the student
    DistillationLoss(ce, alpha=0.7)(student, labels, teacher).backward()
    assert student.grad is not None and student.grad.abs().sum() > 0


def test_predictions_still_work_after_gradcam(tmp_path):
    from animal_classifier.predictor import TorchPredictor

    model = create_model("simple_cnn", num_classes=3).eval()
    checkpoint = {
        "model": model.state_dict(),
        "model_name": "simple_cnn",
        "class_names": ["a", "b", "c"],
        "config": {"data": {"resize_size": 72, "image_size": 64}},
    }
    torch.save(checkpoint, tmp_path / "best.pt")
    predictor = TorchPredictor(tmp_path / "best.pt", device="cpu")
    image = np.random.default_rng(0).integers(0, 256, (80, 100, 3), dtype=np.uint8)

    explained, overlay = predictor.explain(image)
    predicted = predictor.predict(image)  # ordinary no-grad inference after a Grad-CAM pass
    assert overlay.shape == (64, 64, 3)
    assert predicted["label"] == explained["label"]
    assert predicted["score"] == pytest.approx(explained["score"], abs=1e-6)
