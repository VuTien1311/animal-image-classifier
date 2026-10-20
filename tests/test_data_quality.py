import io

import numpy as np
from PIL import Image, ImageDraw

from animal_classifier.data_quality import (
    dhash,
    find_cross_matches,
    find_duplicate_groups,
    hamming_distances,
    is_blank,
)


def scene(seed: int, size=(320, 240)) -> Image.Image:
    """A smooth synthetic 'photo': gradient background plus a few random ellipses."""
    rng = np.random.default_rng(seed)
    gradient = np.linspace(0, 255, size[0], dtype=np.float32)[None, :, None] * rng.uniform(0.3, 1, 3)
    image = Image.fromarray(np.broadcast_to(gradient, (size[1], size[0], 3)).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for _ in range(4):
        x, y = rng.integers(0, size[0] - 60), rng.integers(0, size[1] - 60)
        draw.ellipse(
            [x, y, x + rng.integers(30, 120), y + rng.integers(30, 120)],
            fill=tuple(int(c) for c in rng.integers(0, 256, 3)),
        )
    return image


def recompressed(image: Image.Image, scale: float = 0.6, quality: int = 70) -> Image.Image:
    small = image.resize((int(image.width * scale), int(image.height * scale)))
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue()))


def distance(a: Image.Image, b: Image.Image) -> int:
    return int(hamming_distances(dhash(a)[None], dhash(b)[None])[0, 0])


def test_dhash_is_robust_to_resizing_and_jpeg_but_separates_different_images():
    original = scene(0)
    assert dhash(original).shape == (32,)  # 256 bits
    assert distance(original, recompressed(original)) <= 10
    assert distance(original, scene(1)) > 40


def test_cross_matches_and_groups():
    a, b, c = scene(0), scene(1), scene(2)
    query = np.stack([dhash(recompressed(a)), dhash(c)])
    reference = np.stack([dhash(b), dhash(a)])
    assert find_cross_matches(query, reference, max_distance=10) == [(0, 1, distance(recompressed(a), a))]

    hashes = np.stack([dhash(a), dhash(b), dhash(recompressed(a)), dhash(c), dhash(recompressed(c))])
    assert sorted(find_duplicate_groups(hashes, max_distance=10)) == [[0, 2], [3, 4]]


def test_is_blank():
    assert is_blank(Image.new("RGB", (64, 64), (0, 0, 0)))
    assert not is_blank(scene(0))
