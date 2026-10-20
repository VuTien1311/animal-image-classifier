"""Calibrate the near-duplicate threshold used by scripts/prepare_data.py.

For every processed train image, find the closest test image with a 64-bit and a 256-bit dHash and
count, per distance bucket, how many pairs there are and how many have *different* labels. Genuine
duplicates almost always share a label, so the different-label rate approximates the false-match rate
(unrelated images that happen to hash alike, typically objects on plain white backgrounds). Example
pairs per bucket are saved so the threshold can also be checked by eye.

Writes results/data/dedup_calibration.json and results/figures/dedup_pairs_256bit_<lo>-<hi>.png.
Run after prepare_data.py (it reads every processed image, including the ones it later dropped).
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from animal_classifier.data_quality import dhash, hamming_distances
from animal_classifier.utils import save_json, single_threaded_blas_for_subprocesses
from animal_classifier.viz import plot_image_grid, save_figure


def hash_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path)
    return dhash(image, 8), dhash(image, 16)


def nearest(query: np.ndarray, reference: np.ndarray, chunk_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Index of and distance to the nearest reference hash for every query hash."""
    index, distance = [], []
    for start in range(0, len(query), chunk_size):
        d = hamming_distances(query[start : start + chunk_size], reference)
        index.append(d.argmin(axis=1))
        distance.append(d.min(axis=1))
    return np.concatenate(index), np.concatenate(distance)


def bucket_table(distance: np.ndarray, different_label: np.ndarray, edges: list[int]) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (distance >= lo) & (distance < hi)
        n = int(mask.sum())
        rows.append(
            {
                "distance": f"[{lo}, {hi})",
                "pairs": n,
                "different_label": int(different_label[mask].sum()),
                "different_label_rate": float(different_label[mask].mean()) if n else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    files = {split: sorted((args.data_root / "images" / split).rglob("*.jpg")) for split in ("train", "test")}
    labels = {split: np.array([p.parent.name for p in paths]) for split, paths in files.items()}
    single_threaded_blas_for_subprocesses()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        hashes = {split: list(pool.map(hash_image, paths, chunksize=128)) for split, paths in files.items()}

    report = {"train_images": len(files["train"]), "test_images": len(files["test"])}
    rng = random.Random(args.seed)
    for name, bits_index in (("64bit", 0), ("256bit", 1)):
        train = np.stack([h[bits_index] for h in hashes["train"]])
        test = np.stack([h[bits_index] for h in hashes["test"]])
        index, distance = nearest(train, test)
        different = labels["train"] != labels["test"][index]
        edges = list(range(0, 9)) if name == "64bit" else list(range(0, 52, 4))
        report[name] = bucket_table(distance, different, edges)

        if name == "256bit":  # example pairs around the chosen threshold
            for lo, hi in ((16, 24), (24, 32), (32, 40)):
                candidates = np.flatnonzero((distance >= lo) & (distance < hi)).tolist()
                images, titles = [], []
                for i in rng.sample(candidates, min(6, len(candidates))):
                    for path, label in (
                        (files["train"][i], labels["train"][i]),
                        (files["test"][index[i]], labels["test"][index[i]]),
                    ):
                        images.append(np.asarray(Image.open(path).convert("RGB").resize((200, 150))))
                        titles.append(f"{path.parts[-3]}: {label} (d={distance[i]})")
                figure = plot_image_grid(images, titles, ncols=4)
                save_figure(figure, args.results / "figures" / f"dedup_pairs_256bit_{lo}-{hi}.png")

    save_json(report, args.results / "data" / "dedup_calibration.json")
    for name in ("64bit", "256bit"):
        print(f"\n{name}: nearest test image for each train image")
        for row in report[name]:
            rate = "-" if row["different_label_rate"] is None else f"{row['different_label_rate']:.0%}"
            print(f"  d in {row['distance']:<9} pairs {row['pairs']:>6}  different label {rate}")


if __name__ == "__main__":
    main()
