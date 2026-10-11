"""Image records and branch-scoped host stages."""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

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
        if isinstance(image, Mapping):
            body = image.get("bytes")
            if isinstance(body, bytes):
                image = body
            else:
                from dew.data.storage import Store
                from dew.records import text

                image = Store(text(image.get("path"), "image.path")).read()
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
    reader: Callable[[Sequence[str]], Batch] | None = dataclasses.field(default=None, repr=False)
    binding = "captions"

    @property
    def identity(self) -> str:
        reader = self.reader
        if reader is None:
            return "dew.drop-captions@1"
        kind = reader if inspect.ismethod(reader) or inspect.isfunction(reader) else type(reader)
        return f"dew.encode-captions:{kind.__module__}.{kind.__qualname__}"

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


@dataclasses.dataclass(frozen=True)
class ImageRecord(grain.RandomMapTransform):
    size: int
    staging_size: int
    field: str = "image"
    caption: str | None = None
    templates: tuple[str, ...] = ()
    columns: tuple[str, ...] = ("caption", "text")
    classes: tuple[str, ...] = ()
    flip: bool = False
    jitter: bool = False
    crop_scale: tuple[float, float] = (1., 1.)
    backend: Literal["host", "device"] = "host"

    def __post_init__(self) -> None:
        if min(self.size, self.staging_size) < 1:
            raise ValueError("image size and staging_size must be positive")
        if self.backend not in ("host", "device"):
            raise ValueError("image augmentation backend must be host or device")
        if len(self.crop_scale) != 2 or not 0 < self.crop_scale[0] <= self.crop_scale[1] <= 1:
            raise ValueError("crop_scale must satisfy 0 < low <= high <= 1")
        import_opencv()

    def __setstate__(self, state: Mapping[str, object]) -> None:
        import_opencv()
        self.__dict__.update(state)

    def random_map(self, element: Batch | bytes, rng: np.random.Generator) -> Batch:
        import_opencv()
        if isinstance(element, bytes):
            from dew.data.sources.codec import unpack_dict_of_byte_arrays

            decoded = unpack_dict_of_byte_arrays(element)
            if "shape" in decoded:
                height, width = np.frombuffer(decoded["shape"], np.int32)
                pixels = np.frombuffer(decoded["image"], np.uint8).reshape(int(height), int(width), 3)
                label = None if "label" not in decoded else int(np.frombuffer(decoded["label"], np.int32)[0])
                element = {"image": pixels, "caption": decoded["caption"].decode(), "label": label}
            else:
                element = {"image": decoded["jpg"], "caption": decoded["txt"].decode()}
        image = element[self.field]
        label = element.get("label")
        if self.classes and label is None:
            raise ValueError("caption classes need an image label")
        described = label if not self.classes else self.classes[int(element["label"])]
        if isinstance(image, Mapping):
            body = image.get("bytes")
            if isinstance(body, bytes):
                image = body
            else:
                from dew.data.storage import Store
                from dew.records import text

                image = Store(text(image.get("path"), "image.path")).read()
        if self.templates:
            chosen = self.templates[int(rng.integers(len(self.templates)))]
            caption = chosen.format(described, label=described)
        elif self.caption is not None:
            caption = self.caption.format(**{**element, "label": described})
        elif not self.columns:
            caption = ""
        else:
            caption = next((str(element[key]) for key in self.columns if key in element), "")
        if isinstance(image, bytes):
            image = decode_image(image, at_least=self.staging_size)
        else:
            image = as_rgb(np.asarray(image))
        image = resize_image(image, self.staging_size)
        if self.backend == "host":
            if (self.flip or self.jitter) and (self.crop_scale != (1., 1.) or self.staging_size != self.size):
                image = RandomCrop(self.size, self.crop_scale, self.flip, self.jitter).random_map(
                    {"image": image}, rng)["image"]
            else:
                image = augment_image(Augment(self.flip, self.jitter), image, rng)
        held: Batch = {"image": image, "caption": caption}
        if self.backend == "device":
            held["image_augmentation_key"] = rng.integers(0, 2**32, size=2, dtype=np.uint32)
        if label is not None:
            held["label"] = np.int32(label)
        return held


@dataclasses.dataclass(frozen=True)
class ConfigureImage:
    image: ImageRecord

    def __call__(self, metadata: Mapping[str, object]) -> ImageRecord:
        from dew.records import record, strings

        labels = record(metadata.get("class_names", {}), "class names").get("label")
        return self.image if self.image.classes or labels is None else dataclasses.replace(
            self.image, classes=strings(labels, "class_names.label"))
