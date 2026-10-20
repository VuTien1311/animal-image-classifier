from pathlib import Path

import pytest

from animal_classifier.config import Config, apply_overrides, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_shipped_config_loads(path):
    cfg = load_config(path)
    assert cfg.experiment == path.stem
    assert isinstance(cfg.optim.lr, float)


def test_overrides_are_parsed_as_yaml_scalars():
    cfg = load_config(
        CONFIG_DIR / "convnext_tiny.yaml", ["train.epochs=3", "data.augment=none", "optim.lr=1e-3"]
    )
    assert cfg.train.epochs == 3
    assert cfg.data.augment == "none"
    assert cfg.optim.lr == pytest.approx(1e-3)


def test_scientific_notation_string_is_coerced_to_float():
    # PyYAML reads "1e-3" (no dot) as a string.
    cfg = Config.from_dict({"optim": {"lr": "1e-3"}})
    assert cfg.optim.lr == pytest.approx(1e-3)


def test_unknown_key_is_rejected():
    with pytest.raises(KeyError, match="lrr"):
        Config.from_dict({"optim": {"lrr": 0.1}})


def test_wrong_type_is_rejected():
    with pytest.raises(TypeError):
        Config.from_dict({"train": {"amp": "yes please"}})


def test_round_trip_through_dict():
    cfg = load_config(CONFIG_DIR / "simple_cnn_distill.yaml")
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_apply_overrides_creates_nested_sections():
    assert apply_overrides({}, ["a.b.c=1"]) == {"a": {"b": {"c": 1}}}
