from collections import Counter

import numpy as np
import pytest
import torch

from animal_classifier.data import build_loader, class_weights, read_class_names, read_split, stratified_split


def test_stratified_split_keeps_class_proportions_and_is_disjoint():
    labels = [0] * 100 + [1] * 50 + [2] * 10
    train_idx, val_idx = stratified_split(labels, val_fraction=0.2, seed=0)

    assert set(train_idx).isdisjoint(val_idx)
    assert sorted(train_idx + val_idx) == list(range(len(labels)))
    assert Counter(labels[i] for i in val_idx) == {0: 20, 1: 10, 2: 2}


def test_stratified_split_is_reproducible():
    labels = list(np.random.default_rng(0).integers(0, 5, 200))
    assert stratified_split(labels, 0.1, seed=3) == stratified_split(labels, 0.1, seed=3)
    assert stratified_split(labels, 0.1, seed=3) != stratified_split(labels, 0.1, seed=4)


def test_read_split(tiny_dataset):
    assert read_class_names(tiny_dataset) == ["cat", "dog", "fish"]
    assert len(read_split(tiny_dataset, "train")) == 24
    with pytest.raises(ValueError):
        read_split(tiny_dataset, "holdout")


@pytest.mark.parametrize("augment", ["none", "standard", "strong"])
def test_train_loader_batches(tiny_config, augment):
    tiny_config.data.augment = augment
    images, labels = next(iter(build_loader(tiny_config.data, "train", train=True)))
    assert images.shape == (4, 3, 64, 64) and images.dtype == torch.float32
    assert labels.shape == (4,)


def test_eval_loader_is_deterministic_and_complete(tiny_config):
    loader = build_loader(tiny_config.data, "test", train=False)
    first = torch.cat([images for images, _ in loader])
    second = torch.cat([images for images, _ in loader])
    assert len(first) == 9
    assert torch.equal(first, second)


def test_class_weights():
    counts = np.array([100, 25])
    assert class_weights(counts, "none") is None
    inverse = class_weights(counts, "inverse")
    assert inverse.mean().item() == pytest.approx(1.0)
    assert inverse[1] / inverse[0] == pytest.approx(4.0)
    sqrt_inverse = class_weights(counts, "sqrt_inverse")
    assert sqrt_inverse[1] / sqrt_inverse[0] == pytest.approx(2.0)
