"""RGB decoding, resizing and host augmentation used by image record steps."""

from __future__ import annotations

import dataclasses
import importlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from dew.data.sources.urls import Fetch
    from dew.data.steps.image import (
        Albumentations,
        Augmentation,
        AugmentationKey,
        Caption,
        CenterCrop,
        ColorJitter,
        DecodeImage,
        DeviceAugment,
        Flip,
        ImageProcessor,
        Interpolation,
        MaxAspect,
        MinSide,
        NotFlat,
        Processor,
        RandomResizedCrop,
        Resize,
        Torchvision,
    )


def import_opencv() -> None:
    """Import OpenCV in the thread that opens a loader, before its readers start.

    The reader threads reach their first decode together, so each would
    otherwise make the first import of cv2 at once. An import that fails in
    one of them leaves the others the half-built module, which surfaces as
    `module 'cv2' has no attribute 'INTER_AREA'` instead of the failure
    itself. Imported here, a broken install raises its own error when the
    loader opens. The import stays out of the module's top so that reading
    text never loads OpenCV.
    """
    importlib.import_module("cv2")



def as_rgb(pixels: np.ndarray) -> np.ndarray:
    """Grey, grey and alpha, RGB or RGBA pixels, 8 or 16 bits, as RGB uint8.

    Grey is replicated and a 16-bit sample kept to its high byte. An image
    with transparency is composited onto white, as img2dataset does for the
    url shards the online loader streams.
    """
    if pixels.dtype == np.uint16:
        pixels = np.right_shift(pixels, 8).astype(np.uint8)
    pixels = pixels.reshape(*pixels.shape[:2], -1)
    if pixels.shape[-1] in (2, 4):
        # c * a / 255 + 255 - a, rounded to nearest in integers; the exact
        # quotient is k / 255, which is never a tie.
        alpha = pixels[..., -1:].astype(np.uint32)
        pixels = ((pixels[..., :-1] * alpha + 255 * (255 - alpha) + 127) // 255).astype(np.uint8)
    return np.ascontiguousarray(np.broadcast_to(pixels, (*pixels.shape[:2], 3)))



def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """`image` at `size` square; area interpolation down, cubic up."""
    import cv2
    if image.shape[:2] == (size, size):
        return image
    interpolation = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interpolation)


@dataclasses.dataclass(frozen=True)
class Augment:
    """The host augmentations `augment_image` applies: a random horizontal
    flip with `flip` and a colour jitter with `jitter`. No Augment at all
    leaves the image as it is.
    """

    flip: bool
    jitter: bool


# ColorJitter(brightness=0.2, contrast=0.05, saturation=0.2, hue=0): the three
# factors' uniform ranges, applied in a random order per record.
_JITTER_RANGES = ((0.8, 1.2), (0.95, 1.05), (0.8, 1.2))


def augment_image(augment: Augment | None, image: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Flips and colour-jitters `image`, seeded by the record's own rng.

    Every draw comes from grain's per-record rng, a Philox keyed by the
    record index, so a record's augmentation is the same however many
    workers, threads or processes produced its batch. The jitter is
    torchvision's float ColorJitter (`jitter_host`); uint8 pixels go through
    float32 and are rounded once at the end.
    """
    if augment is None:
        return image
    if augment.flip and rng.random() < 0.5:
        image = image[:, ::-1]
    if not augment.jitter:
        return np.ascontiguousarray(image)

    from .image_augmentation import jitter_host

    factors = np.asarray([rng.uniform(low, high) for low, high in _JITTER_RANGES])
    pixels = jitter_host(image.astype(np.float32), factors, rng.permutation(3))
    return np.rint(pixels).astype(np.uint8)


def fetch(url: str = "url", to: str = "jpg", *, timeout: float = 10.0, max_bytes: int = 64 * 2**20) -> Fetch:
    """A record step (`filter_map`) that fetches the image at the record's
    `url` field into `to` as bytes, in one attempt of at most `timeout`; a
    failed fetch is counted and dropped, a transient one fetched again on a
    later pass. An indexed read runs `Loading.threads` fetches at once in each
    worker. The bytes are volatile: a published cache after the fetch keeps
    them for exact resume."""
    from dew.data.sources.urls import Fetch

    return Fetch(url, to, timeout, max_bytes)


def decode(field: str = "jpg", to: str = "image") -> DecodeImage:
    """A record step that decodes the image in `field` to RGB uint8 pixels
    under `to`; bytes that are no image are counted and dropped."""
    from dew.data.steps.image import DecodeImage

    return DecodeImage(field, to=to)


def min_side(size: int, field: str = "image") -> MinSide:
    """A filter keeping pixels at least `size` on their shorter side."""
    from dew.data.steps.image import MinSide

    return MinSide(size, field)


def max_aspect(ratio: float, field: str = "image") -> MaxAspect:
    """A filter keeping pixels at most `ratio` times as long as wide."""
    from dew.data.steps.image import MaxAspect

    return MaxAspect(ratio, field)


def not_flat(field: str = "image") -> NotFlat:
    """A filter dropping pixels of one flat colour."""
    from dew.data.steps.image import NotFlat

    return NotFlat(field)


def resize(size: int | tuple[int, int], method: Interpolation | None = None, *,
           field: str = "image") -> Resize:
    """A record step that resizes the pixels in `field` to `size`, a square
    side or (height, width), with OpenCV's `method` interpolation, by default
    area down and bicubic up."""
    from dew.data.steps.image import Resize

    return Resize(size, method, field)


def center_crop(size: int, *, field: str = "image") -> CenterCrop:
    """A record step that keeps the central `size` square of the pixels in
    `field`, resizing a smaller image up first."""
    from dew.data.steps.image import CenterCrop

    return CenterCrop(size, field=field)


def random_resized_crop(size: int, *, scale: tuple[float, float] = (0.08, 1.0),
                        ratio: tuple[float, float] = (3 / 4, 4 / 3),
                        field: str = "image") -> RandomResizedCrop:
    """A random step, torchvision's RandomResizedCrop: a crop of `scale` of
    the area at `ratio` of aspect, resized to a `size` square."""
    from dew.data.steps.image import RandomResizedCrop

    return RandomResizedCrop(size, scale, ratio, field=field)


def flip(p: float = 0.5, *, field: str = "image") -> Flip:
    """A random step that mirrors the pixels in `field` left to right with probability `p`."""
    from dew.data.steps.image import Flip

    return Flip(p, field)


def color_jitter(brightness: float = 0.2, contrast: float = 0.05, saturation: float = 0.2, *,
                 field: str = "image") -> ColorJitter:
    """A random step, torchvision's ColorJitter of brightness, contrast and saturation."""
    from dew.data.steps.image import ColorJitter

    return ColorJitter(brightness, contrast, saturation, field)


def caption(template: str | None = None, *, classes: tuple[str, ...] = ()) -> Caption:
    """A record step that writes "caption": `template` formatted with the
    record's fields ("{label}" is the class name `classes` gives its label),
    or with none, the record's own caption or text."""
    from dew.data.steps.image import Caption

    return Caption(template, classes)


def processor(image_processor: ImageProcessor, *, field: str = "image",
              to: str = "pixel_values") -> Processor:
    """A record step that runs a Hugging Face image processor on the image in
    `field`, its `pixel_values` under `to`."""
    from dew.data.steps.image import Processor

    return Processor(image_processor, field, to)


def albumentations(transform: Augmentation, *, field: str = "image") -> Albumentations:
    """A random step that runs an albumentations pipeline on the image in
    `field`, seeded from the record's generator, so its draws repeat exactly
    on resume and at any thread or worker count."""
    from dew.data.steps.image import Albumentations

    return Albumentations(transform, field)


def torchvision(transform: Callable[[torch.Tensor], torch.Tensor], *, field: str = "image") -> Torchvision:
    """A random step that runs a torchvision v2 transform on the image in
    `field` (HWC uint8 in and out), seeded from the record's generator."""
    from dew.data.steps.image import Torchvision

    return Torchvision(transform, field)


def augmentation_key() -> AugmentationKey:
    """A random step that draws the key `device_augment` augments its record with."""
    from dew.data.steps.image import AugmentationKey

    return AugmentationKey()


def device_augment(size: int, *, crop_scale: tuple[float, float] = (1.0, 1.0), flip: bool = True,
                   jitter: bool = False) -> DeviceAugment:
    """A batch step that crops, flips and jitters a batch's images on the
    host's accelerator, each record by the key `augmentation_key` drew."""
    from dew.data.steps.image import DeviceAugment

    return DeviceAugment(size, crop_scale, flip, jitter)
