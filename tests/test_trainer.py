import csv
import json

import pytest

from animal_classifier.trainer import EarlyStopping, train


def test_training_end_to_end_and_resume(tiny_config):
    run_dir = train(tiny_config)

    for name in ("config.yaml", "train.log", "history.csv", "best.pt", "last.pt", "summary.json"):
        assert (run_dir / name).exists(), name
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["epochs_trained"] == 2

    # Resume to a longer schedule: history continues instead of restarting.
    tiny_config.train.epochs = 3
    train(tiny_config, resume=run_dir / "last.pt")
    with open(run_dir / "history.csv", newline="") as f:
        assert [int(row["epoch"]) for row in csv.DictReader(f)] == [1, 2, 3]


def test_existing_run_is_not_silently_overwritten(tiny_config):
    tiny_config.train.epochs = 1
    train(tiny_config)
    with pytest.raises(FileExistsError):
        train(tiny_config)
    train(tiny_config, overwrite=True)


def test_early_stopping():
    stopper = EarlyStopping(patience=2)
    assert stopper.update(0.5, epoch=1)
    assert not stopper.update(0.4, epoch=2)
    assert not stopper.should_stop
    assert not stopper.update(0.5, epoch=3)  # equal is not an improvement
    assert stopper.should_stop and stopper.best_epoch == 1
