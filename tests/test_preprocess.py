import numpy as np
import pytest
from PIL import Image
from torchvision import transforms as T

from animal_classifier.preprocess import IMAGENET_MEAN, IMAGENET_STD, denormalize, load_image, preprocess


@pytest.mark.parametrize("width,height", [(640, 427), (300, 500), (256, 256), (1000, 257), (250, 180)])
def test_matches_torchvision_eval_transform(width, height):
    """The NumPy serving pipeline must produce the same tensor as torchvision (no train/serve skew)."""
    rng = np.random.default_rng(width * height)
    image = Image.fromarray(rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8))
    reference = T.Compose(
        [T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )

    expected = reference(image).numpy()
    actual = preprocess(image, resize_size=256, crop_size=224)

    assert actual.shape == (3, 224, 224) and actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_transparent_png_is_composited_on_white():
    # Black silhouette whose shape lives only in the alpha channel (as in the raw dataset).
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[8:24, 8:24, 3] = 255
    rgb = np.asarray(load_image(Image.fromarray(rgba, mode="RGBA")))
    assert rgb[0, 0].tolist() == [255, 255, 255]  # transparent background -> white
    assert rgb[16, 16].tolist() == [0, 0, 0]  # opaque silhouette stays black


def test_load_image_accepts_bytes_and_arrays(tmp_path):
    array = np.full((10, 12, 3), 128, dtype=np.uint8)
    path = tmp_path / "x.png"
    Image.fromarray(array).save(path)
    for source in (path, str(path), path.read_bytes(), array):
        assert load_image(source).size == (12, 10)


def test_denormalize_inverts_normalisation():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8))
    restored = denormalize(preprocess(image, 224, 224))
    np.testing.assert_array_equal(restored, np.asarray(image))
