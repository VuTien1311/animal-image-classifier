"""Train a model from a YAML config.

Examples:
    python scripts/train.py --config configs/convnext_tiny.yaml
    python scripts/train.py --config configs/simple_cnn.yaml --set train.epochs=5 data.batch_size=64
    python scripts/train.py --config configs/convnext_tiny.yaml --resume outputs/convnext_tiny/last.pt
"""

from __future__ import annotations

import argparse

from animal_classifier.config import load_config
from animal_classifier.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="path to a YAML config in configs/")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--resume", help="continue from a last.pt checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="delete an existing run directory first")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    train(cfg, resume=args.resume, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
