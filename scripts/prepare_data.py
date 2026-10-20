"""Build the processed dataset from the raw Animals-10 folders.

Steps
  1. Scan <src>/{train,test}/<class>/*  (class index = alphabetical order of the folder names).
  2. Decode every image (corrupt files are reported and skipped), apply EXIF orientation, resize the
     shorter side to --size and save as JPEG q95. Decoding small JPEGs makes data loading much faster.
     Blank (single-colour) images cannot be classified by anyone and are removed from every split.
  3. Perceptual-hash (256-bit dHash) every image and
       - drop train images that are near-duplicates of a test image   (train/test leakage)
       - keep one image per near-duplicate group inside train         (train/val leakage)
     Apart from blank images, the official test split is never modified.
  4. Stratified train/val split of the remaining train images. Val is used for model selection and
     early stopping; test is only used once, for the final numbers.
  5. Write splits.csv, classes.json, a data report and figures.

Usage:
    python scripts/prepare_data.py --src D:/Deep/data/animals --out data/processed
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from animal_classifier.data import CLASSES_FILE, SPLITS_FILE, stratified_split
from animal_classifier.data_quality import dhash, find_cross_matches, find_duplicate_groups, is_blank
from animal_classifier.preprocess import load_image, resize_shorter_side
from animal_classifier.utils import save_json, single_threaded_blas_for_subprocesses
from animal_classifier.viz import plot_image_grid, save_figure

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, required=True, help="raw dataset with train/ and test/ folders")
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=256, help="shorter side after resizing")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-hash-distance", type=int, default=28, help="dHash Hamming distance (of 256)")
    parser.add_argument("--keep-duplicates", action="store_true", help="report duplicates but keep them")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", type=Path, default=Path("results"))
    return parser.parse_args()


def scan(src: Path) -> tuple[list[str], list[dict]]:
    class_names = sorted(p.name for p in (src / "train").iterdir() if p.is_dir())
    records, used_names = [], set()
    for split in ("train", "test"):
        for label, class_name in enumerate(class_names):
            for file in sorted((src / split / class_name).iterdir()):
                if file.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                stem, n = file.stem, 1
                while (split, class_name, stem) in used_names:  # a.jpg and a.png -> a.jpg, a_1.jpg
                    stem, n = f"{file.stem}_{n}", n + 1
                used_names.add((split, class_name, stem))
                records.append(
                    {
                        "source": str(file),
                        "path": f"images/{split}/{class_name}/{stem}.jpg",
                        "label": label,
                        "class_name": class_name,
                        "split": split,
                    }
                )
    return class_names, records


def process_image(task: tuple[str, str, int]) -> dict:
    source, destination, size = task
    try:
        image = load_image(source)
        width, height = image.size
        resized = resize_shorter_side(image, size)
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        resized.save(destination, format="JPEG", quality=95)
        return {
            "ok": True,
            "width": width,
            "height": height,
            "hash": dhash(resized),
            "blank": is_blank(resized),
        }
    except Exception as error:  # noqa: BLE001 - any decode failure means a corrupt file
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def deduplicate(records: list[dict], max_distance: int, keep: bool) -> tuple[list[dict], dict, list]:
    """Remove train images that leak into test or duplicate other train images."""
    train = [r for r in records if r["split"] == "train"]
    test = [r for r in records if r["split"] == "test"]
    train_hashes = np.stack([r["hash"] for r in train])
    test_hashes = np.stack([r["hash"] for r in test])

    leaks = find_cross_matches(train_hashes, test_hashes, max_distance)
    leaked = {i for i, _, _ in leaks}
    train_groups = find_duplicate_groups(train_hashes, max_distance)
    redundant = {i for group in train_groups for i in group[1:]}  # keep the first image of each group
    test_groups = find_duplicate_groups(test_hashes, max_distance)

    def conflicting(groups, items):
        return sum(len({items[i]["label"] for i in group}) > 1 for group in groups)

    stats = {
        "max_hamming_distance": max_distance,
        "train_images_near_duplicate_of_test": len(leaked),
        "leaks_with_different_label": sum(train[i]["label"] != test[j]["label"] for i, j, _ in leaks),
        "train_duplicate_groups": len(train_groups),
        "train_redundant_copies": len(redundant),
        "train_groups_with_conflicting_labels": conflicting(train_groups, train),
        "test_duplicate_groups": len(test_groups),
        "test_groups_with_conflicting_labels": conflicting(test_groups, test),
        "removed_from_train": 0 if keep else len(leaked | redundant),
    }
    examples = [(train[i], test[j], d) for i, j, d in leaks]
    if keep:
        return records, stats, examples
    drop = {train[i]["path"] for i in leaked | redundant}
    return [r for r in records if r["path"] not in drop], stats, examples


def save_duplicate_examples(out: Path, examples: list, figure_path: Path, n_pairs: int = 8, seed: int = 0):
    """Side-by-side train/test pairs so the duplicate threshold can be verified by eye."""
    if not examples:
        return
    pairs = random.Random(seed).sample(examples, min(n_pairs, len(examples)))
    images, titles = [], []
    for train_rec, test_rec, distance in pairs:
        for rec, split in ((train_rec, "train"), (test_rec, "test")):
            images.append(np.asarray(Image.open(out / rec["path"]).convert("RGB")))
            titles.append(f"{split}: {rec['class_name']} (d={distance})")
    save_figure(plot_image_grid(images, titles, ncols=4), figure_path)


def save_class_distribution(records: list[dict], class_names: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    counts = Counter((r["split"], r["class_name"]) for r in records)
    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(10, 4))
    for offset, split in zip((-0.27, 0, 0.27), ("train", "val", "test"), strict=True):
        ax.bar(x + offset, [counts[(split, c)] for c in class_names], width=0.27, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("images")
    ax.set_title("Class distribution per split")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, path)


def main() -> None:
    args = parse_args()
    class_names, records = scan(args.src)
    print(f"Found {len(records)} images in {len(class_names)} classes")

    tasks = [(r["source"], str(args.out / r["path"]), args.size) for r in records]
    single_threaded_blas_for_subprocesses()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(process_image, tasks, chunksize=64))

    corrupt, blank, valid = [], [], []
    for record, result in zip(records, results, strict=True):
        if not result["ok"]:
            corrupt.append({"file": record["source"], "error": result["error"]})
        elif result["blank"]:
            blank.append({"file": record["source"], "split": record["split"], "label": record["class_name"]})
        else:
            valid.append({**record, **result})
    print(f"Decoded {len(valid)} images, {len(corrupt)} corrupt, {len(blank)} blank")

    kept, duplicate_stats, examples = deduplicate(valid, args.max_hash_distance, args.keep_duplicates)
    print("Duplicates:", duplicate_stats)

    # Stratified train/val split of what is left of the official train split.
    train = [r for r in kept if r["split"] == "train"]
    _, val_idx = stratified_split([r["label"] for r in train], args.val_fraction, args.seed)
    for i in val_idx:
        train[i]["split"] = "val"

    fields = ["path", "label", "class_name", "split"]
    with open(args.out / SPLITS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    save_json(class_names, args.out / CLASSES_FILE)

    per_split = defaultdict(dict)
    for (split, class_name), n in sorted(Counter((r["split"], r["class_name"]) for r in kept).items()):
        per_split[split][class_name] = n
    widths = np.array([r["width"] for r in valid])
    heights = np.array([r["height"] for r in valid])
    report = {
        "source_folder": args.src.name,
        "classes": class_names,
        "images_found": len(records),
        "corrupt_files": corrupt,
        "blank_images": blank,
        "duplicates": duplicate_stats,
        "split_sizes": {split: sum(counts.values()) for split, counts in per_split.items()},
        "class_counts": per_split,
        "original_resolution": {
            "width": {"min": int(widths.min()), "median": int(np.median(widths)), "max": int(widths.max())},
            "height": {
                "min": int(heights.min()),
                "median": int(np.median(heights)),
                "max": int(heights.max()),
            },
        },
        "processed_shorter_side": args.size,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
    }
    save_json(report, args.results / "data" / "data_report.json")
    save_class_distribution(kept, class_names, args.results / "figures" / "class_distribution.png")
    save_duplicate_examples(
        args.out, examples, args.results / "figures" / "near_duplicates.png", seed=args.seed
    )
    print("Split sizes:", report["split_sizes"])
    print(f"Wrote {args.out / SPLITS_FILE} and {args.results / 'data' / 'data_report.json'}")


if __name__ == "__main__":
    main()
