"""Data-quality checks: blank images and near-duplicate detection with perceptual hashing.

An MD5 check only finds byte-identical files. Web-scraped datasets often contain the same photo
resized, re-encoded or watermarked, which MD5 misses but a perceptual hash catches. If such copies
sit in both train and test, the test score is inflated (data leakage).

Hash size matters: with a 64-bit dHash, unrelated objects on plain white backgrounds collide at
distance 3-6. A 256-bit dHash (16x16) separates them much better; see the README for the calibration.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def dhash(image: Image.Image, hash_size: int = 16) -> np.ndarray:
    """Difference hash: one bit per horizontally adjacent pixel pair of a small grayscale thumbnail.

    Returns the ``hash_size**2`` bits packed into a uint8 array (32 bytes for the default 16x16).
    """
    thumbnail = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(thumbnail, dtype=np.int16)
    return np.packbits(pixels[:, 1:] > pixels[:, :-1])


def is_blank(image: Image.Image, min_std: float = 2.0) -> bool:
    """True for (almost) single-colour images, e.g. failed downloads saved as a black frame."""
    return float(np.asarray(image.convert("L"), dtype=np.float32).std()) < min_std


def hamming_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between packed hashes: (n, bytes) x (m, bytes) -> (n, m)."""
    return np.bitwise_count(a[:, None, :] ^ b[None, :, :]).sum(axis=-1, dtype=np.int32)


def find_cross_matches(
    query: np.ndarray, reference: np.ndarray, max_distance: int, chunk_size: int = 128
) -> list[tuple[int, int, int]]:
    """For each query hash, the closest reference hash if within ``max_distance``.

    Returns ``(query_index, reference_index, distance)`` tuples.
    """
    matches = []
    for start in range(0, len(query), chunk_size):
        distances = hamming_distances(query[start : start + chunk_size], reference)
        nearest = distances.argmin(axis=1)
        nearest_distance = distances[np.arange(len(nearest)), nearest]
        for offset in np.flatnonzero(nearest_distance <= max_distance):
            matches.append((start + int(offset), int(nearest[offset]), int(nearest_distance[offset])))
    return matches


def find_duplicate_groups(hashes: np.ndarray, max_distance: int, chunk_size: int = 128) -> list[list[int]]:
    """Group indices whose hashes are within ``max_distance`` (connected components, via union-find).

    Only groups with two or more members are returned.
    """
    parent = list(range(len(hashes)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(hashes), chunk_size):
        distances = hamming_distances(hashes[start : start + chunk_size], hashes)
        rows, cols = np.nonzero(distances <= max_distance)
        for row, col in zip(rows, cols, strict=True):
            i, j = start + int(row), int(col)
            if i < j:
                parent[root(j)] = root(i)

    groups: dict[int, list[int]] = {}
    for i in range(len(hashes)):
        groups.setdefault(root(i), []).append(i)
    return [members for members in groups.values() if len(members) > 1]
