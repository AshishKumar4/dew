"""Image records and branch-scoped host stages."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch
from dew.data.images import Augment, as_rgb, augment_image, decode_image, import_opencv, resize_image
from dew.data.pipeline import Rejected


@dataclasses.dataclass(frozen=True)
class DecodeImage(grain.MapTransform):
    field: str = "image"
    at_least: int | None = None
    rejects = True

    def map(self, element: Batch) -> Batch | Rejected:
        import_opencv()
        image = element[self.field]
        if isinstance(image, bytes):
            try:
                image = decode_image(image, at_least=self.at_least)
            except ValueError:
                return Rejected("undecodable")
        else:
            image = as_rgb(np.asarray(image))
        return {**element, "image": image}


@dataclasses.dataclass(frozen=True)
class Resize(grain.MapTransform):
    size: int

    def map(self, element: Batch) -> Batch:
        return {**element, "image": resize_image(np.asarray(element["image"]), self.size)}


@dataclasses.dataclass(frozen=True)
class Caption(grain.MapTransform):
    template: str | None = None
    classes: tuple[str, ...] = ()

    def map(self, element: Batch) -> Batch:
        if self.template is None:
            caption = str(element.get("caption", element.get("text", "")))
        else:
            label = element.get("label", "")
            if self.classes:
                label = self.classes[int(label)]
            caption = self.template.format(**{**element, "label": label})
        return {**element, "caption": caption}


@dataclasses.dataclass(frozen=True)
class CenterCrop(grain.MapTransform):
    size: int

    def map(self, element: Batch) -> Batch:
        image = np.asarray(element["image"])
        top, left = max(0, (image.shape[0] - self.size) // 2), max(0, (image.shape[1] - self.size) // 2)
        return {
            **element,
            "image": resize_image(image[top : top + self.size, left : left + self.size], self.size),
        }


@dataclasses.dataclass(frozen=True)
class RandomCrop(grain.RandomMapTransform):
    size: int
    scale: tuple[float, float] = (0.8, 1.0)
    flip: bool = False
    jitter: bool = False

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        from dew.data.image_augmentation import apply_host, draw_host

        image = np.asarray(element["image"])
        parameters = draw_host(rng, image.shape, flip=self.flip, jitter=self.jitter, crop_scale=self.scale)
        pixels = apply_host(image, parameters, self.size)
        return {**element, "image": np.clip(np.rint(pixels), 0, 255).astype(np.uint8)}


@dataclasses.dataclass(frozen=True)
class Flip(grain.RandomMapTransform):
    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        return {**element, "image": augment_image(Augment(flip=True, jitter=False),
                                                  np.asarray(element["image"]), rng)}


@dataclasses.dataclass(frozen=True)
class ColorJitter(grain.RandomMapTransform):
    flip: bool = False

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        return {
            **element,
            "image": augment_image(Augment(flip=self.flip, jitter=True), np.asarray(element["image"]), rng),
        }


@dataclasses.dataclass(frozen=True)
class AugmentationKey(grain.RandomMapTransform):
    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        return {**element, "image_augmentation_key": rng.integers(0, 2**32, size=2, dtype=np.uint32)}


@dataclasses.dataclass(frozen=True)
class EncodeCaptions:
    reader: Callable[[Sequence[str]], Batch] | None = None

    def __call__(self, batch: Batch) -> Batch:
        held = dict(batch)
        captions = [str(value) for value in held.pop("caption")]
        return held if self.reader is None else {**held, **self.reader(captions)}


@dataclasses.dataclass(frozen=True)
class DeviceAugment:
    size: int
    crop_scale: tuple[float, float] = (1.0, 1.0)
    flip: bool = True
    jitter: bool = False

    def __call__(self, batch: Batch) -> Batch:
        fields = dict(batch)
        keys = fields.pop("image_augmentation_key")
        fields["image"] = self._run(fields["image"], keys)
        return fields

    @functools.cached_property
    def _run(self):
        import jax

        from dew.data.image_augmentation import augment_batch

        return jax.jit(
            functools.partial(
                augment_batch, size=self.size, crop_scale=self.crop_scale, flip=self.flip, jitter=self.jitter
            )
        )
