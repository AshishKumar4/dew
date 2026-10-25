"""Image record steps: decoding, filters, resizing, crops, captions and augmentation, on the
host or on the accelerator."""

from __future__ import annotations

import copy
import dataclasses
import importlib
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

import grain.python as grain
import numpy as np
from numpy.typing import ArrayLike

from dew.data.dataset import Batch
from dew.data.facts import resized, shaped, unshaped
from dew.data.image_augmentation import JITTER
from dew.data.image_codec import as_rgb, decode_image
from dew.data.pipeline import DeviceStep, Rejected
from dew.data.plan import callable_name
from dew.records import record

if TYPE_CHECKING:
    import jax
    import torch

    from dew.data.sources.urls import Fetch


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


def resize_image(image: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """`image` at `size`, a square side or (height, width); area interpolation
    where it shrinks along either side, cubic where it only grows."""
    height, width = (size, size) if isinstance(size, int) else size
    return resized_to(image, height, width, None)


def fetch(url: str = "url", to: str = "jpg", *, timeout: float = 10.0, max_bytes: int = 64 * 2**20,
          threads: int | None = None) -> Fetch:
    """A record step (`filter_map`) that fetches the image at the record's
    `url` field into `to` as bytes, in one attempt of at most `timeout` once
    its host is resolved (`Fetch`); a
    failed fetch is counted and dropped, a transient one fetched again on a
    later pass. `threads` fetches that many at once on the stage's own pool, in
    each loading worker, delivered in the order read; without it an indexed
    read runs `Loading.threads` fetches at once in each worker. The bytes are
    volatile: a published cache after the fetch keeps them for exact resume."""
    from dew.data.sources.urls import Fetch

    return Fetch(url, to, timeout, max_bytes, threads)


def _payload(image: bytes | Mapping[str, object] | ArrayLike) -> bytes | ArrayLike:
    """An Arrow or Hugging Face image value's encoded bytes, read through
    `Store` when it holds only a path, and any other value as it is."""
    if not isinstance(image, Mapping):
        return image
    body = image.get("bytes")
    if isinstance(body, bytes):
        return body
    from dew.data.storage import Store
    from dew.records import text

    return Store(text(image.get("path"), "image.path")).read()


@dataclasses.dataclass(frozen=True)
class DecodeImage(grain.MapTransform):
    """The image in `field` as RGB uint8 pixels under `to`, including prepared ArrayRecord
    pixels, labels and captions; bytes that are no image are counted and dropped. With
    `rejects=False`, use `map` and fail on bad bytes. `at_least` bounds a reduced JPEG decode."""

    field: str = "jpg"
    to: str = "image"
    _: dataclasses.KW_ONLY
    at_least: int | None = None
    rejects: bool = True

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """RGB pixels under `to`, each image its own height and width, in place of the bytes in `field`."""
        return shaped(unshaped(facts, self.field), self.to, [None, None, 3])

    def __post_init__(self) -> None:
        import_opencv()

    def __setstate__(self, state: Mapping[str, object]) -> None:
        import_opencv()
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def map(self, element: Batch | bytes) -> Batch | Rejected:
        field = self.field
        if isinstance(element, bytes):
            from dew.data.sources.codec import unpack_dict_of_byte_arrays

            decoded = unpack_dict_of_byte_arrays(element)
            field = field if field in decoded else "image" if "shape" in decoded else "jpg"
            element = {key: value for key, value in decoded.items()
                       if key not in ("shape", "label", "caption", "txt")}
            if "shape" in decoded:
                height, width = np.frombuffer(decoded["shape"], np.int32)
                element[field] = np.frombuffer(decoded[field], np.uint8).reshape(int(height), int(width), 3)
            if "label" in decoded:
                element["label"] = np.frombuffer(decoded["label"], np.int32)[0]
            element["caption"] = decoded.get("caption", decoded.get("txt", b"")).decode()
        image = _payload(element[field])
        if isinstance(image, bytes):
            try:
                image = decode_image(image, at_least=self.at_least)
            except ValueError:
                if not self.rejects:
                    raise
                return Rejected("undecodable")
        else:
            image = as_rgb(np.asarray(image))
        # The pixels replace the bytes they came from.
        kept = {key: value for key, value in element.items() if key != field}
        return {**kept, self.to: image}


@dataclasses.dataclass(frozen=True)
class MinSide:
    """Whether the pixels in `field`, `[..., H, W, C]`, are at least `size` on their shorter side."""

    size: int
    field: str = "image"
    reason = "too_small"

    def __call__(self, element: Batch) -> bool:
        return min(np.shape(element[self.field])[-3:-1]) >= self.size


@dataclasses.dataclass(frozen=True)
class MaxAspect:
    """Whether the pixels in `field` are at most `ratio` times as long as wide."""

    ratio: float
    field: str = "image"
    reason = "too_elongated"

    def __call__(self, element: Batch) -> bool:
        shorter, longer = sorted(np.shape(element[self.field])[-3:-1])
        return longer <= self.ratio * shorter


@dataclasses.dataclass(frozen=True)
class NotFlat:
    """Whether the pixels in `field` hold more than one value."""

    field: str = "image"
    reason = "flat"

    def __call__(self, element: Batch) -> bool:
        pixels = np.asarray(element[self.field])
        return bool(pixels.min() != pixels.max())


_INTERPOLATIONS = {"area": "INTER_AREA", "bilinear": "INTER_LINEAR", "bicubic": "INTER_CUBIC",
                   "nearest": "INTER_NEAREST", "lanczos": "INTER_LANCZOS4"}
type Interpolation = Literal["area", "bilinear", "bicubic", "nearest", "lanczos"]


def resized_to(image: np.ndarray, height: int, width: int, method: Interpolation | None) -> np.ndarray:
    """`image` at `height` x `width` by OpenCV's `method` interpolation; where None, area where it
    shrinks along either side and bicubic where it only grows. The one resize every image and video
    step reads."""
    import cv2

    if image.shape[:2] == (height, width):
        return image
    if method is None:
        method = "area" if image.shape[0] > height or image.shape[1] > width else "bicubic"
    return cv2.resize(image, (width, height), interpolation=getattr(cv2, _INTERPOLATIONS[method]))


def _cropped(image: np.ndarray, rows: int, columns: int, method: Interpolation,
             resize: Literal["cover", "grow"]) -> np.ndarray:
    """`image` cut to `rows` x `columns` at its centre, first resized by `method`, in proportion, to
    cover them: always (`cover`), or only where it is smaller (`grow`)."""
    height, width = image.shape[:2]
    if resize == "cover" or height < rows or width < columns:
        scale = max(rows / height, columns / width)
        height, width = max(rows, round(height * scale)), max(columns, round(width * scale))
        image = resized_to(image, height, width, method)
    top, left = (height - rows) // 2, (width - columns) // 2
    return image[top : top + rows, left : left + columns]


@dataclasses.dataclass(frozen=True)
class Resize(grain.MapTransform):
    """The pixels in `field` resized to `size`, a square side or (height, width),
    by OpenCV's `method` interpolation, or by area down and bicubic up where None."""

    size: int | tuple[int, int]
    method: Interpolation | None = None
    _: dataclasses.KW_ONLY
    field: str = "image"

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        height, width = (self.size, self.size) if isinstance(self.size, int) else self.size
        return resized(facts, self.field, height, width)

    def map(self, element: Batch) -> Batch:
        height, width = (self.size, self.size) if isinstance(self.size, int) else self.size
        image = np.asarray(element[self.field])
        return {**element, self.field: resized_to(image, height, width, self.method)}


@dataclasses.dataclass(frozen=True)
class Fit(grain.MapTransform):
    """The pixels in `field` fitted into a `size` square whole: the longer side resized to `size`
    (area down, bicubic up), the shorter in proportion, rounded, and the rest padded on both sides
    with `fill`, the odd pixel at the bottom or right. A web image keeps all it shows, as the
    retired online loader fitted it; `Resize` stretches it and `CenterCrop` cuts it instead."""

    size: int
    fill: int = 255
    _: dataclasses.KW_ONLY
    field: str = "image"

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        return resized(facts, self.field, self.size, self.size)

    def map(self, element: Batch) -> Batch:
        image = np.asarray(element[self.field])
        height, width = image.shape[:2]
        scale = self.size / max(height, width)
        fitted = (max(1, round(height * scale)), max(1, round(width * scale)))
        resized = resized_to(image, *fitted, None).reshape(*fitted, *image.shape[2:])
        tall, wide = self.size - fitted[0], self.size - fitted[1]
        padding = ((tall // 2, tall - tall // 2), (wide // 2, wide - wide // 2),
                   *((0, 0) for _ in image.shape[2:]))
        return {**element, self.field: np.pad(resized, padding, constant_values=self.fill)}


@dataclasses.dataclass(frozen=True)
class CenterCrop(grain.MapTransform):
    """The central `size` square of the pixels in `field`; a smaller image is
    first resized up so its shorter side is `size`."""

    size: int
    method: Interpolation = "bicubic"
    _: dataclasses.KW_ONLY
    field: str = "image"

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """The images `size` squares, as many channels as they held."""
        return resized(facts, self.field, self.size, self.size)

    def map(self, element: Batch) -> Batch:
        image = np.asarray(element[self.field])
        return {**element, self.field: _cropped(image, self.size, self.size, self.method, "grow")}


@dataclasses.dataclass(frozen=True)
class RandomCrop(grain.RandomMapTransform):
    """A crop of `scale` of the area resized to `size`, optionally flipped and colour-jittered."""

    size: int
    _: dataclasses.KW_ONLY
    scale: tuple[float, float] = (0.8, 1.0)
    flip: bool = False
    jitter: bool = False

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        return resized(facts, "image", self.size, self.size)

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        from dew.data.image_augmentation import apply_host, draw_host

        image = np.asarray(element["image"])
        parameters = draw_host(rng, image.shape, flip=self.flip, jitter=self.jitter, crop_scale=self.scale)
        pixels = apply_host(image, parameters, self.size)
        return {**element, "image": np.clip(np.rint(pixels), 0, 255).astype(np.uint8)}


@dataclasses.dataclass(frozen=True)
class Bucket(grain.MapTransform):
    """The pixels in `field` resized to cover the (height, width) of `sizes` nearest their aspect
    ratio and cut to it at the centre, with that size's index under `to`, so records of one bucket
    stack."""

    sizes: tuple[tuple[int, int], ...]
    method: Interpolation = "area"
    field: str = "image"
    to: str = "bucket"

    def __post_init__(self) -> None:
        if not self.sizes or any(height < 1 or width < 1 for height, width in self.sizes):
            raise ValueError(f"buckets are positive (height, width) sizes, not {self.sizes}")

    def map(self, element: Batch) -> Batch:
        image = np.asarray(element[self.field])
        height, width = image.shape[:2]
        aspect = np.log(width / height)
        index = int(np.argmin([abs(np.log(columns / rows) - aspect) for rows, columns in self.sizes]))
        rows, columns = self.sizes[index]
        return {**element, self.field: _cropped(image, rows, columns, self.method, "cover"),
                self.to: np.int32(index)}


def bucket(sizes: Sequence[tuple[int, int]], method: Interpolation = "area", *, field: str = "image",
           to: str = "bucket") -> Bucket:
    """A record step that resizes the pixels in `field` to cover the (height, width) of `sizes`
    nearest their aspect ratio and cuts that size from the centre, writing its index under `to`; a
    batch grouped by `to` stacks images of one shape."""
    return Bucket(tuple((int(height), int(width)) for height, width in sizes), method, field, to)


@dataclasses.dataclass(frozen=True)
class RandomResizedCrop(grain.RandomMapTransform):
    """torchvision's RandomResizedCrop: a crop of `scale` of the area and
    `ratio` of aspect, drawn from the record's generator, resized to a `size`
    square; after ten draws that do not fit, the central crop of the nearest
    aspect."""

    size: int
    _: dataclasses.KW_ONLY
    scale: tuple[float, float] = (0.08, 1.0)
    ratio: tuple[float, float] = (3 / 4, 4 / 3)
    method: Interpolation = "bilinear"
    field: str = "image"

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        return resized(facts, self.field, self.size, self.size)

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        image = np.asarray(element[self.field])
        height, width = image.shape[:2]
        area, bounds = height * width, np.log(self.ratio)
        for _ in range(10):
            target = area * rng.uniform(*self.scale)
            aspect = np.exp(rng.uniform(*bounds))
            crop_width, crop_height = round(np.sqrt(target * aspect)), round(np.sqrt(target / aspect))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                top = int(rng.integers(0, height - crop_height + 1))
                left = int(rng.integers(0, width - crop_width + 1))
                break
        else:
            aspect = width / height
            if aspect < min(self.ratio):
                crop_width, crop_height = width, round(width / min(self.ratio))
            elif aspect > max(self.ratio):
                crop_height, crop_width = height, round(height * max(self.ratio))
            else:
                crop_width, crop_height = width, height
            top, left = (height - crop_height) // 2, (width - crop_width) // 2
        crop = image[top : top + crop_height, left : left + crop_width]
        return {**element, self.field: resized_to(crop, self.size, self.size, self.method)}


@dataclasses.dataclass(frozen=True)
class Flip(grain.RandomMapTransform):
    """The pixels in `field` mirrored left to right with probability `p`."""

    p: float = 0.5
    _: dataclasses.KW_ONLY
    field: str = "image"

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """The images as their shape was: a mirror moves pixels, not the frame."""
        return facts

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        image = np.asarray(element[self.field])
        flipped = np.ascontiguousarray(image[:, ::-1]) if rng.random() < self.p else image
        return {**element, self.field: flipped}


@dataclasses.dataclass(frozen=True)
class ColorJitter(grain.RandomMapTransform):
    """torchvision's float ColorJitter of the pixels in `field`: brightness,
    contrast and saturation factors drawn from [1 - x, 1 + x], applied in a
    random order, rounded once."""

    brightness: float = JITTER.brightness
    contrast: float = JITTER.contrast
    saturation: float = JITTER.saturation
    _: dataclasses.KW_ONLY
    field: str = "image"

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """The images as their shape was, uint8 still: the jitter changes their values alone."""
        return facts

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        from dew.data.image_augmentation import Spread, jitter_host

        ranges = Spread(self.brightness, self.contrast, self.saturation).ranges()
        factors = np.asarray([rng.uniform(low, high) for low, high in ranges])
        pixels = jitter_host(np.asarray(element[self.field]).astype(np.float32), factors, rng.permutation(3))
        return {**element, self.field: np.rint(pixels).astype(np.uint8)}


class ImageProcessor(Protocol):
    """A Hugging Face image processor, as `Processor` calls it."""

    def __call__(self, *, images: np.ndarray, return_tensors: str) -> Mapping[str, np.ndarray]: ...


class Augmentation(Protocol):
    """An albumentations pipeline (`Compose`), as `Albumentations` seeds and calls it."""

    def set_random_seed(self, seed: int) -> None: ...

    def __call__(self, *, image: np.ndarray) -> Mapping[str, np.ndarray]: ...


@dataclasses.dataclass(frozen=True, eq=False)
class Processor(grain.MapTransform):
    """A Hugging Face image processor's `pixel_values` for the image in `field`, under `to`."""

    processor: ImageProcessor
    _: dataclasses.KW_ONLY
    field: str = "image"
    to: str = "pixel_values"

    def map(self, element: Batch) -> Batch:
        processed = self.processor(images=np.asarray(element[self.field]), return_tensors="np")
        return {**element, self.to: np.asarray(processed["pixel_values"][0])}


_torch_rng = threading.Lock()


class _ThreadCopy(threading.local):
    """A thread's own copy of a stage's pipeline, made when the thread first runs the stage."""

    transform: Augmentation | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class Albumentations(grain.RandomMapTransform):
    """An albumentations pipeline on the image in `field`, seeded from the
    record's generator through `Compose.set_random_seed`. Each reading thread
    seeds and calls its own copy of the pipeline, so no other thread's seed,
    from this stage or another holding the same pipeline, lands between the
    two, and a record's draw is the same at any thread count and after a
    resume."""

    transform: Augmentation
    _: dataclasses.KW_ONLY
    field: str = "image"
    # Each instance's own, set by __post_init__ and __setstate__; no field of its configuration.
    _copy: ClassVar[_ThreadCopy]

    def __post_init__(self) -> None:
        # Made before any reader thread runs, so every thread finds the same one.
        object.__setattr__(self, "_copy", _ThreadCopy())

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        if self._copy.transform is None:
            self._copy.transform = copy.deepcopy(self.transform)
        transform = self._copy.transform
        transform.set_random_seed(int(rng.integers(0, 2**32 - 1)))
        return {**element, self.field: transform(image=np.asarray(element[self.field]))["image"]}

    def __getstate__(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if key != "_copy"}

    def __setstate__(self, state: dict) -> None:
        # A worker makes its own copies.
        self.__dict__.update(state)
        object.__setattr__(self, "_copy", _ThreadCopy())


@dataclasses.dataclass(frozen=True, eq=False)
class Torchvision(grain.RandomMapTransform):
    """A torchvision v2 transform on the image in `field` (HWC uint8 in and
    out), seeded from the record's generator. torch draws from one generator a
    process, so the transform runs one record at a time in each process, the
    generator seeded inside that turn and restored after it."""

    transform: Callable[[torch.Tensor], torch.Tensor]
    _: dataclasses.KW_ONLY
    field: str = "image"

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        import torch

        seed = int(rng.integers(0, 2**63 - 1))
        pixels = torch.from_numpy(np.ascontiguousarray(element[self.field])).permute(2, 0, 1)
        with _torch_rng, torch.random.fork_rng(devices=[]):
            # The CPU generator alone, the one fork_rng restores; an accelerator's stays as it was.
            torch.default_generator.manual_seed(seed)
            transformed = self.transform(pixels)
        return {**element, self.field: np.ascontiguousarray(np.asarray(transformed).transpose(1, 2, 0))}


CAPTION_COLUMNS = ("caption", "text")
"""The fields a record's own caption is read from, the first it holds (`Caption`)."""


@dataclasses.dataclass(frozen=True)
class Caption(grain.MapTransform):
    """A record's caption and int32 numeric labels, from columns or a class template."""

    template: str | None = None
    classes: tuple[str, ...] = ()
    columns: tuple[str, ...] = CAPTION_COLUMNS

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """A caption, a string of no fixed shape, in place of any the record held, beside the fields
        as they were."""
        return unshaped(facts, "caption")

    def map(self, element: Batch) -> Batch:
        label = element.get("label")
        if self.classes:
            if label is None:
                raise ValueError("caption classes need an image label")
            label = self.classes[int(label)]
        if self.template is None:
            caption = next((str(element[key]) for key in self.columns if key in element), "")
        else:
            caption = self.template.format(label, **{**element, "label": label})
        held = {**element, "caption": caption}
        if element.get("label") is not None and np.asarray(element["label"]).dtype.kind in "biuf":
            held["label"] = np.int32(element["label"])
        return held


@dataclasses.dataclass(frozen=True)
class RandomCaption(Caption, grain.RandomMapTransform):
    """A caption template drawn from the record's generator."""

    templates: tuple[str, ...] = ()

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        chosen = self.templates[int(rng.integers(len(self.templates)))]
        return dataclasses.replace(self, template=chosen).map(element)


def caption(template: str | None = None, *, templates: tuple[str, ...] = (),
            classes: tuple[str, ...] = (),
            columns: tuple[str, ...] = CAPTION_COLUMNS) -> Caption | RandomCaption:
    """A record step that writes "caption": `template` formatted with the
    record's fields ("{label}" is the class name `classes` gives its label),
    or with none, the first of `columns`. With `templates`, a random step
    instead draws one template per record; `{}` and `{label}` both name its class."""
    if templates:
        return RandomCaption(template, classes, columns, templates)
    return Caption(template, classes, columns)


@dataclasses.dataclass(frozen=True)
class ConfigureCaption:
    caption: Caption

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """A caption, a string of no fixed shape, in place of any the record held, beside the rest."""
        return unshaped(facts, "caption")

    def __call__(self, metadata: Mapping[str, object]) -> Caption:
        from dew.records import strings

        labels = record(metadata.get("class_names", {}), "class names").get("label")
        return (
            self.caption
            if self.caption.classes or labels is None
            else dataclasses.replace(self.caption, classes=strings(labels, "class_names.label"))
        )


@dataclasses.dataclass(frozen=True)
class AugmentationKey(grain.RandomMapTransform):
    """The key `DeviceAugment` augments the record with, drawn from the record's generator."""

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        return shaped(facts, "image_augmentation_key", [2])

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        return {**element, "image_augmentation_key": rng.integers(0, 2**32, size=2, dtype=np.uint32)}


CAPTIONS = "captions"
"""The field a diffusion objective keeps each batch's captions under as host text (`EncodeCaptions`),
which a metric or a reward reads with its own tokenizer (`dew.eval.images.CLIPScore`)."""


@dataclasses.dataclass(frozen=True)
class EncodeCaptions:
    """Each batch's captions encoded by `reader` for the conditions that read them, or dropped without
    one; `keep` names a field to hold them beside as host text, which a metric reads against its own
    tokenizer (`dew.eval.images.CLIPScore`) and no compiled step takes (`split_fields`). A reader that
    encodes a field of that name too is refused (ValueError), rather than one overwriting the other."""

    reader: Callable[[Sequence[str]], Batch] | None = dataclasses.field(default=None, repr=False)
    keep: str | None = None

    @property
    def identity(self) -> str:
        kept = "" if self.keep is None else f"+keep={self.keep}"
        if self.reader is None:
            return "dew.drop-captions@1" if self.keep is None else f"dew.keep-captions@1:{self.keep}"
        return f"dew.encode-captions{kept}:{callable_name(self.reader)}"

    def __call__(self, batch: Batch) -> Batch:
        held = dict(batch)
        if "caption" not in held:
            return held
        captions = [str(value) for value in held.pop("caption")]
        encoded = held if self.reader is None else {**held, **self.reader(captions)}
        if self.keep is None:
            return encoded
        if self.keep in encoded:
            raise ValueError(f"the captions kept as {self.keep!r} would overwrite the field of that name the "
                             "conditions read; keep them under another")
        return {**encoded, self.keep: np.asarray(captions)}


@dataclasses.dataclass(frozen=True)
class EncodeFields:
    """Each batch with the fields `reader` encodes from it (`InputSpec.encode_fields`), as
    `EncodeCaptions` does its captions."""

    reader: Callable[[Batch], Batch] = dataclasses.field(repr=False)

    @property
    def identity(self) -> str:
        return f"dew.encode-fields:{callable_name(self.reader)}"

    def __call__(self, batch: Batch) -> Batch:
        return {**batch, **self.reader(batch)}


@dataclasses.dataclass(frozen=True)
class DeviceAugment(DeviceStep):
    """Crop, flip and jitter each image of the training batch inside the
    compiled step, from its record's own key (`AugmentationKey`)."""

    size: int
    _: dataclasses.KW_ONLY
    crop_scale: tuple[float, float] = (1.0, 1.0)
    flip: bool = True
    jitter: bool = False

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """Each image cropped to a `size` square, its key spent."""
        return unshaped(resized(facts, "image", self.size, self.size), "image_augmentation_key")

    def __call__(self, batch: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        from dew.data.image_augmentation import augment_batch

        fields = dict(batch)
        keys = fields.pop("image_augmentation_key")
        fields["image"] = augment_batch(fields["image"], keys, size=self.size, crop_scale=self.crop_scale,
                                        flip=self.flip, jitter=self.jitter)
        return fields
