"""Torch-free image preprocessing shared by evaluation, the ONNX predictor and the API.

The eval pipeline is Resize(shorter side) -> CenterCrop -> ToTensor -> Normalize, re-implemented with
PIL + NumPy so the serving container does not need PyTorch. Training-time evaluation uses these exact
functions too, so there is no train/serve skew by construction; tests/test_preprocess.py also checks
that the result matches torchvision's implementation bit for bit.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageSource = str | Path | bytes | bytearray | np.ndarray | Image.Image


def to_rgb(image: Image.Image, background=(255, 255, 255)) -> Image.Image:
    """Convert to RGB, compositing transparent images onto a white background.

    A plain ``convert("RGB")`` drops the alpha channel: a black-on-transparent PNG silhouette (the
    dataset has 49 transparent PNGs) becomes an all-black image.
    """
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    if image.mode in ("RGBA", "LA"):
        canvas = Image.new("RGBA", image.size, (*background, 255))
        return Image.alpha_composite(canvas, image.convert("RGBA")).convert("RGB")
    return image.convert("RGB")


def load_image(source: ImageSource) -> Image.Image:
    """Open an image from a path, raw bytes, a HxWx3 uint8 array or a PIL image as RGB.

    EXIF orientation is applied so phone photos are not classified sideways.
    """
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, (bytes, bytearray)):
        image = Image.open(io.BytesIO(source))
    elif isinstance(source, np.ndarray):
        image = Image.fromarray(source)
    else:
        image = Image.open(source)
    return to_rgb(ImageOps.exif_transpose(image))


def resize_shorter_side(image: Image.Image, size: int) -> Image.Image:
    """Same output as ``torchvision.transforms.Resize(size)`` on a PIL image."""
    width, height = image.size
    short, long = (width, height) if width <= height else (height, width)
    new_short, new_long = size, int(size * long / short)
    new_w, new_h = (new_short, new_long) if width <= height else (new_long, new_short)
    if (new_w, new_h) == (width, height):
        return image
    return image.resize((new_w, new_h), Image.Resampling.BILINEAR)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    """Same output as ``torchvision.transforms.CenterCrop(size)`` when the image is large enough."""
    width, height = image.size
    if size > width or size > height:
        raise ValueError(f"Crop size {size} is larger than image {width}x{height}; resize first")
    top = int(round((height - size) / 2.0))
    left = int(round((width - size) / 2.0))
    return image.crop((left, top, left + size, top + size))


def normalize(image: Image.Image, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """HxWx3 uint8 image -> 3xHxW float32 array, like ``ToTensor()`` followed by ``Normalize``."""
    array = np.asarray(image, dtype=np.uint8).astype(np.float32) / 255.0
    array = array.transpose(2, 0, 1)
    mean = np.asarray(mean, dtype=np.float32)[:, None, None]
    std = np.asarray(std, dtype=np.float32)[:, None, None]
    return (array - mean) / std


def preprocess(
    image: ImageSource,
    resize_size: int = 256,
    crop_size: int = 224,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> np.ndarray:
    """Full eval preprocessing; returns a contiguous 3 x crop_size x crop_size float32 array."""
    if not isinstance(image, Image.Image):
        image = load_image(image)
    image = center_crop(resize_shorter_side(image, resize_size), crop_size)
    return np.ascontiguousarray(normalize(image, mean, std))


def denormalize(array: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """Inverse of :func:`normalize`: 3xHxW float array -> HxWx3 uint8 image (for visualisation)."""
    mean = np.asarray(mean, dtype=np.float32)[:, None, None]
    std = np.asarray(std, dtype=np.float32)[:, None, None]
    image = (array * std + mean).clip(0, 1).transpose(1, 2, 0)
    return (image * 255).round().astype(np.uint8)
