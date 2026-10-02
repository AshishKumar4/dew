"""Image datasets: TFDS, Hugging Face hub and arrayrecord shards, one transform.

Every image dataset resizes, augments and captions its records the same way.
What differs is where the records come from and how one is read, which is
the three hooks a subclass fills in. Records leave as
`{"image": uint8 [size, size, 3], "caption": str}`, plus `"label"` when the
source carries a class index. `load(tokenize=)` is where a run's own
condition reads the captions: the dataset carries the text, the encoder
decides what tokens it becomes. cv2, tensorflow_datasets and HF datasets are
imported on use, so `import dew.data` costs none of them.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import os
import struct
from collections.abc import Mapping, Sized
from typing import Literal

import grain.python as pygrain
import jax
import numpy as np

from dew.registry import datasets

from .dataset import (
    CAPTION,
    Batch,
    Dataset,
    DatasetSpec,
    Reader,
    Records,
    Tokenize,
    checked_count,
    hold_out,
    mapped,
    tokenized,
    train_stream,
    validation_pass,
)
from .sources.hf import HFOptions, HubOptions
from .tokens import bounded

Augmentation = Literal["none", "flip_only", "flip_jitter"]


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


def unpack_dict_of_byte_arrays(packed_data: bytes) -> dict[str, bytes]:
    """The `str -> bytes` entries of one packed arrayrecord record.

    Each entry is a uint32 key length, the utf-8 key, a uint32 value length
    and the value, in that order. `pack_dict_of_byte_arrays` writes it.
    """
    unpacked_dict = {}
    offset = 0
    while offset < len(packed_data):
        key_length = struct.unpack_from('I', packed_data, offset)[0]
        offset += struct.calcsize('I')
        key = packed_data[offset:offset+key_length].decode('utf-8')
        offset += key_length
        byte_array_length = struct.unpack_from('I', packed_data, offset)[0]
        offset += struct.calcsize('I')
        byte_array = packed_data[offset:offset+byte_array_length]
        offset += byte_array_length
        unpacked_dict[key] = byte_array
    return unpacked_dict

def pack_dict_of_byte_arrays(unpacked: dict) -> bytes:
    """`unpacked`'s entries length-prefixed in dict order, the layout
    `unpack_dict_of_byte_arrays` reads."""
    packed = bytearray()
    for key, byte_array in unpacked.items():
        encoded = key.encode('utf-8')
        packed += struct.pack('I', len(encoded))
        packed += encoded
        packed += struct.pack('I', len(byte_array))
        packed += byte_array
    return bytes(packed)


def decode_image(encoded: bytes, *, at_least: int | None = None) -> np.ndarray:
    """An encoded image as RGB uint8, in the orientation its pixels are stored.

    Grey is replicated and a 16-bit sample kept to its high byte. An image
    with transparency is composited onto white, as img2dataset does for the
    url shards the online loader streams. EXIF orientation is ignored, as
    PIL's `Image.open` ignores it, on the reduced decodes too, which would
    otherwise apply it.

    With `at_least`, an opaque image is decoded at the largest 1/2, 1/4 or
    1/8 reduction that keeps both sides >= `at_least` (the DCT scale of a
    JPEG), so the resize after it still only shrinks and most of the decode
    is skipped.

    Every failure is a ValueError. PIL reads the header first, since it
    refuses a decompression bomb from the header alone where cv2 would
    decode up to 2**30 pixels, and cv2 hands back None for a half-written
    file.
    """
    import cv2
    height, width, transparent = _header(encoded)
    buffer = np.frombuffer(encoded, dtype=np.uint8)
    # IMREAD_UNCHANGED keeps the alpha the colour flags drop, and ignores EXIF.
    flags = cv2.IMREAD_UNCHANGED if transparent else cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    if at_least is not None and not transparent:
        for factor, reduced in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4),
                                (2, cv2.IMREAD_REDUCED_COLOR_2)):
            if min(height, width) // factor >= at_least:
                flags = reduced | cv2.IMREAD_IGNORE_ORIENTATION
                break
    try:
        image = cv2.imdecode(buffer, flags)
    except cv2.error as error:
        raise ValueError(f"cv2 refused {len(encoded)} bytes of image") from error
    if image is None:
        raise ValueError(f"cv2 could not decode {len(encoded)} bytes of image")
    if image.dtype == np.uint16:
        image = np.right_shift(image, 8).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 4:
        # c * a / 255 + 255 - a, rounded to nearest in integers; the exact
        # quotient is k / 255, which is never a tie.
        alpha = image[..., 3:].astype(np.uint32)
        image = ((image[..., :3] * alpha + 255 * (255 - alpha) + 127) // 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _header(encoded: bytes) -> tuple[int, int, bool]:
    """(height, width, whether it carries transparency) from the header alone;
    PIL reads no pixels for this."""
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(encoded)) as header:
            width, height = header.size
            transparent = "A" in header.getbands() or "transparency" in header.info
    except Exception as error:
        raise ValueError(f"could not read the header of {len(encoded)} bytes of image") from error
    return height, width, transparent


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """`image` at `size` square; area interpolation down, cubic up."""
    import cv2
    if image.shape[:2] == (size, size):
        return image
    interpolation = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interpolation)


@dataclasses.dataclass(frozen=True)
class Augment:
    """Says which augmentations a mode applies: flip for flip_only, both for
    flip_jitter. 'none' maps to no Augment at all."""

    flip: bool
    jitter: bool


def image_augmentations(mode: Augmentation) -> Augment | None:
    """The augmentations `mode` names: flip_only (DiT style), flip_jitter,
    or none (deterministic evaluation and debugging)."""
    if mode == 'none':
        return None
    if mode == 'flip_only':
        return Augment(flip=True, jitter=False)
    if mode == 'flip_jitter':
        return Augment(flip=True, jitter=True)
    raise ValueError(f"augmentation {mode!r} is not one of none, flip_only, flip_jitter")


# ColorJitter(brightness=0.2, contrast=0.05, saturation=0.2, hue=0): the three
# factors' uniform ranges, applied in a random order per record.
_JITTER_RANGES = ((0.8, 1.2), (0.95, 1.05), (0.8, 1.2))
_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def _gray(pixels: np.ndarray) -> np.ndarray:
    """0.299 R + 0.587 G + 0.114 B as float32; cvtColor is the fast path here."""
    import cv2
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)


def augment_image(augment: Augment | None, image: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Flips and colour-jitters `image`, seeded by the record's own rng.

    Every draw comes from grain's per-record rng, a Philox keyed by the
    record index, so a record's augmentation is the same however many
    workers, threads or processes produced its batch. uint8 pixels go through
    float32 and are rounded and clipped once at the end.
    """
    if augment is None:
        return image
    if augment.flip and rng.random() < 0.5:
        image = image[:, ::-1]
    if not augment.jitter:
        return np.ascontiguousarray(image)

    import cv2
    brightness, contrast, saturation = (rng.uniform(low, high) for low, high in _JITTER_RANGES)
    pixels = image.astype(np.float32)
    for index in rng.permutation(3):
        if index == 0:
            pixels = pixels * brightness
        elif index == 1:
            # Contrast is measured against the image as it stands at its
            # place in the order, not the incoming one.
            mean = _gray(pixels).mean()
            pixels = pixels * contrast + mean * (1 - contrast)
        else:
            # Saturation as one matrix: out_c = s * x_c + (1 - s) * gray.
            matrix = np.eye(3, dtype=np.float32) * saturation + np.outer(
                np.ones(3, np.float32), (1 - saturation) * _LUMA)
            pixels = cv2.transform(pixels, matrix)
    np.rint(pixels, out=pixels)
    return np.clip(pixels, 0, 255, out=pixels).astype(np.uint8)


@functools.cache
def class_names(path: str) -> tuple[str, ...]:
    """The class names of a labels file, one per line, read once per process."""
    with open(os.path.expanduser(path)) as handle:
        return tuple(line.strip() for line in handle)


def record_caption(element, columns: tuple[str, ...] = ("caption", "text")) -> str:
    """The caption a record already carries, for datasets that ship their text:
    the first of `columns` it holds."""
    for key in columns:
        if key in element:
            return element[key]
    raise KeyError(
        f"an image record needs one of the columns {list(columns)}, this one has "
        f"{sorted(element)}")


def _fields(element: Batch | bytes, name: str) -> Batch:
    """A record's fields, or the refusal that this source holds bytes.

    A spec whose `source` reads a table of features reads them by name here;
    one whose source is arrayrecord bytes unpacks them itself.
    """
    if isinstance(element, bytes):
        raise TypeError(
            f"{name} reads records of named fields, and this source holds "
            f"{len(element)} bytes; unpack them in record() instead")
    return element


class ImageTransform(pygrain.RandomMapTransform):
    """Resizes, augments and captions one record, seeded by the record's own rng.

    It is built where its loader opens and unpickled where a spawned worker
    starts, both before any reader thread runs, and both import OpenCV
    (`import_opencv`).
    """

    def __init__(self, spec: ImageDataset):
        import_opencv()
        self.spec = spec
        self.augments = image_augmentations(spec.augmentation)

    def __setstate__(self, state: Mapping[str, object]) -> None:
        import_opencv()
        self.__dict__.update(state)

    def random_map(self, element: Batch | bytes, rng: np.random.Generator) -> Batch:
        image, caption, label = self.spec.record(element, rng)
        size = self.spec.staging_size
        if isinstance(image, bytes):
            image = decode_image(image, at_least=size)
        image = resize_image(image, size)
        if self.spec.augmentation_backend == "host":
            if (self.augments is not None
                    and (self.spec.crop_scale != (1.0, 1.0) or size != self.spec.image_size)):
                from .image_augmentation import apply_host, draw_host

                parameters = draw_host(rng, image.shape, flip=self.augments.flip, jitter=self.augments.jitter,
                                       crop_scale=self.spec.crop_scale)
                pixels = apply_host(image, parameters, self.spec.image_size)
                image = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
            else:
                image = augment_image(self.augments, image, rng)
        record = {"image": image, CAPTION: caption}
        if self.spec.augmentation_backend == "device":
            # Grain keys its Philox by the data seed and global record position.
            # Only the small key crosses the host augmentation boundary; the
            # device draws crop, flip and colour independently for every row.
            record["image_augmentation_key"] = rng.integers(0, 2**32, size=2, dtype=np.uint32)
        if label is not None:
            # the class index, which the JEPA linear/kNN probes score against
            record["label"] = np.int32(label)
        return record


@dataclasses.dataclass(frozen=True)
class ImageDataset(DatasetSpec):
    """Reads captioned images through grain, resized to `image_size`.

    Validation comes from one of two places. `val_split` names a split of the
    dataset's own, which is opened as a second source and scored in record
    order, `val_batches` batches of it or all of it when that is None.
    Without one, `val_batches` batches of records are held out of the head of
    the training source, in canonical order, so FID and CLIP are never
    measured on records the model trained on. None or 0 holds nothing out and
    validates nothing.

    `count` takes that many records from the head of the source. A source
    that reports no length needs it set.
    """

    image_size: int = 128
    augmentation: Augmentation = "flip_jitter"
    augmentation_backend: Literal["host", "device"] = "host"
    """Host OpenCV/NumPy, or JAX augmentation of the decoded device batch."""
    crop_scale: tuple[float, float] = (1.0, 1.0)
    """Uniform retained area fraction, resized bilinearly; (1, 1) keeps the full image."""
    augmentation_size: int | None = None
    """Square decoded staging size before random crop; None uses image_size."""
    val_batches: int | None = 4
    val_split: str | None = None
    count: int | None = None

    def __post_init__(self):
        if self.augmentation_backend not in ("host", "device"):
            raise ValueError("augmentation_backend must be host or device")
        if len(self.crop_scale) != 2 or not 0 < self.crop_scale[0] <= self.crop_scale[1] <= 1:
            raise ValueError("crop_scale must satisfy 0 < low <= high <= 1")
        if self.augmentation_size is not None and (
                type(self.augmentation_size) is not int or self.augmentation_size < 1):
            raise ValueError("augmentation_size must be a positive integer or None")

    @property
    def staging_size(self) -> int:
        """The dense host shape; none retains the old evaluation resize."""
        if self.augmentation == "none":
            return self.image_size
        return self.image_size if self.augmentation_size is None else self.augmentation_size

    def processed(self, stream: Reader) -> Reader:
        """Tokenize on the host, then optionally augment the pixels on device.

        `mapped` forwards Grain's saved position and close/stop methods. No
        second RNG state or batch counter needs checkpointing, and changing
        reader threads, process shares or batch size changes no record draw.
        """
        if self.augmentation_backend == "host":
            return stream
        from .image_augmentation import augment_batch

        augment = image_augmentations(self.augmentation)
        run = jax.jit(functools.partial(augment_batch, size=self.image_size,
                      flip=augment is not None and augment.flip,
                      jitter=augment is not None and augment.jitter,
                      crop_scale=self.crop_scale if augment is not None else (1.0, 1.0)))

        def stage(batch):
            fields = dict(batch)
            keys = fields.pop("image_augmentation_key")
            fields["image"] = run(fields["image"], keys)
            return fields

        return mapped(stream, stage)

    def source(self, split: str | None = None) -> Records:
        """Opens the records by index (`__getitem__`, and `__len__` unless
        `count` says how many there are).

        `split` names a split other than the one this spec reads, which is
        how `val_split` opens a second source. A dataset whose records are
        one pile refuses it.
        """
        raise NotImplementedError

    def record(self, element: Batch | bytes,
               rng: np.random.Generator) -> tuple[np.ndarray | bytes, str, int | None]:
        """One record as `(image, caption, class index or None)`.

        The image is RGB uint8, or the encoded bytes for the transform to
        decode at the size it needs.
        """
        raise NotImplementedError

    def records(self, source: Records) -> int:
        """The records the run uses, from the head of the source.

        A source that cannot count itself (no `__len__`) is the other case, where
        the spec's own `count` is the whole record of how many there are.
        """
        name = type(self).__name__
        if self.count is None:
            if not isinstance(source, Sized):
                raise ValueError(
                    f"{name} reports no length, so it needs count= set to the "
                    "records it holds")
            return len(source)
        if isinstance(source, Sized):
            return checked_count(self.count, len(source), name)
        return self.count

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        source = self.source()
        # A named split is its own records, so nothing is held out of
        # training; without one the head of the source is the split.
        held_out = 0 if self.val_split else (self.val_batches or 0) * batch
        train, validation = hold_out(source, self.records(source), held_out,
                                     type(self).__name__)
        if self.val_split:
            validation = self.source(self.val_split)
        # Validation reads the deterministic full-image resize a reference
        # metric compares against, not the training crop, flip and jitter.
        evaluated = dataclasses.replace(
            self, augmentation="none", augmentation_backend="host", crop_scale=(1.0, 1.0))
        scored = None if validation is None else tokenized(
            validation_pass(validation, [ImageTransform(evaluated)], batch=batch,
                            seed=self.seed, loading=self.loading), tokenize)
        if self.val_split and scored is not None:
            scored = bounded(scored, self.val_batches)
        training = tokenized(train_stream(train, [ImageTransform(self)], batch=batch,
                                          seed=self.seed, loading=self.loading), tokenize)
        return Dataset(
            train=self.processed(training),
            val=None if scored is None else evaluated.processed(scored),
            records=len(train),
            batch=batch,
            held_out=0 if validation is None or self.val_split else held_out,
        )


@datasets("tfds_images")
@dataclasses.dataclass(frozen=True)
class TFDSImages(ImageDataset):
    """Reads a prepared TFDS image dataset's ArrayRecords, captioned from its
    class names: the `image` and `label` features of oxford_flowers102,
    cifar10, food101 and the like.

    Preparation runs separately. Reading uses TFDS metadata and NumPy image
    decoding through its read-only builder, without TensorFlow or dataset
    generation code in the training process.

    A record's caption is one of `caption_templates`, drawn from the
    record's rng, filled with its class name.
    """

    path: str | None = None
    """Prepared version directory containing dataset_info.json and ArrayRecords."""
    split: str = "all"
    labels: str | None = None
    """Class-name file override; unset reads label.labels.txt in path."""
    caption_templates: tuple[str, ...] = ("a photo of a {}",)

    def source(self, split: str | None = None):
        if not self.path:
            raise ValueError(
                "TFDSImages needs path= (--data.path) pointing to prepared TFDS "
                "ArrayRecords. Prepare the dataset separately with "
                "download_and_prepare(file_format='array_record'), then pass "
                "the builder.data_dir version directory to training.")
        import tensorflow_datasets as tfds

        from .sources.tfds import prepared_source

        return prepared_source(self.path, split or self.split,
                               decoders={"image": tfds.decode.SkipDecoding()})

    def record(self, element: Batch | bytes, rng):
        element = _fields(element, "TFDSImages")
        label = int(element["label"])
        # The template comes from the record's rng, like the augmentation.
        # A module-global random.choice would key a record's caption to how
        # many workers and processes produced the batch.
        template = self.caption_templates[int(rng.integers(len(self.caption_templates)))]
        labels = self.labels
        if labels is None:
            if self.path is None:
                raise ValueError("TFDSImages captions need labels= or a prepared path=.")
            labels = os.path.join(self.path, "label.labels.txt")
        return element["image"], template.format(class_names(labels)[label]), label


@datasets("hf_images")
@dataclasses.dataclass(frozen=True)
class HFImages(ImageDataset):
    """Reads a Hugging Face hub dataset of images by index.

    `name` is the repo id and `split` the split to read. `options` is
    everything else `datasets.load_dataset` takes, the same value the `hf`
    provider holds, so a dataset behind a config name, a revision, its own
    `data_files` or a token is read here too.

    The columns say where a record keeps its fields. `image_column` holds the
    image; the caption is the first of `caption_columns` a record has, and
    an empty tuple reads an uncaptioned dataset, such as a class-labelled
    one, for an unconditional or class-conditional run. A `label` column,
    where the dataset has one, is the class index a record carries. A column
    the split does not hold is refused when the spec loads.

    Its images arrive decoded by `datasets` rather than by `decode_image`,
    so a JPEG's EXIF orientation is applied, where `decode_image` keeps the
    stored one.
    """

    name: str = ""
    split: str = "train"
    options: HubOptions = dataclasses.field(default_factory=HFOptions)
    image_column: str = "image"
    caption_columns: tuple[str, ...] = ("caption", "text")

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        if not self.caption_columns and tokenize is not None:
            raise TypeError(
                "HFImages with caption_columns=() reads no captions, so tokenize= has "
                "nothing to read; name the caption column, or train without conditions")
        return super().load(batch=batch, tokenize=tokenize)

    def source(self, split: str | None = None):
        from .sources.hf import HFDatasetSource
        if not self.name:
            raise ValueError("HFImages needs name= set to a hub dataset repo id")
        source = HFDatasetSource(name=self.name, split=split or self.split,
                                 options=self.options)
        self._check_columns(source.columns, split or self.split)
        return source

    def _check_columns(self, columns: list[str], split: str) -> None:
        """Refuse a column field the split does not hold, naming what it holds."""
        held = f"{self.name!r} split {split!r} has the columns {sorted(columns)}"
        if self.image_column not in columns:
            raise ValueError(f"image_column={self.image_column!r}, and {held}; name its image column")
        if self.caption_columns and not set(self.caption_columns) & set(columns):
            raise ValueError(
                f"caption_columns={self.caption_columns!r}, and {held}; name its caption "
                f"column, or set caption_columns=() for a dataset without captions")

    def record(self, element: Batch | bytes, rng):
        element = _fields(element, "HFImages")
        caption = record_caption(element, self.caption_columns) if self.caption_columns else ""
        label = element.get("label")
        return element[self.image_column], caption, None if label is None else int(label)


@datasets("array_record_images")
@dataclasses.dataclass(frozen=True)
class ArrayRecordImages(ImageDataset):
    """Reads image and caption pairs from arrayrecord shards under
    `path/<shard>/`, each record a packed dict.

    Two layouts are read. 'jpg'/'txt' entries hold an encoded image, decoded
    on read. The `prepare_images.py` layout holds 'image'/'shape'/'caption',
    where the image is uint8 HxWx3 already at training size, the shape is two
    little-endian int32s, and 'label' is optional.

    `path` is the bucket mount or directory the shards live under. An empty
    `shards` reads every arrayrecord file in `path` itself, which is the
    layout prepare_images.py writes.
    """

    path: str | None = None
    shards: tuple[str, ...] = ()

    def source(self, split: str | None = None):
        if split is not None:
            raise ValueError(
                f"{type(self).__name__} reads the shards under path= as one pile "
                f"and holds no named split, so val_split={split!r} names nothing; "
                "leave it unset and val_batches holds records out of the head")
        if not self.path:
            raise ValueError(
                f"{type(self).__name__} needs path= set: its records live under "
                "<path>/<shard>/ for each of its shards")
        roots = self.shards or ("",)
        files = []
        for shard in roots:
            root = os.path.join(self.path, shard)
            files += [os.path.join(root, f) for f in sorted(os.listdir(root))
                      if 'array_record' in f]
        return pygrain.ArrayRecordDataSource(files)

    def record(self, element: Batch | bytes, rng):
        if not isinstance(element, bytes):
            raise TypeError(
                f"{type(self).__name__} reads arrayrecord shards, whose records "
                f"are packed bytes; this one is {type(element).__name__}")
        element = unpack_dict_of_byte_arrays(element)
        if 'image' in element:
            height, width = np.frombuffer(element['shape'], dtype=np.int32)
            image = np.frombuffer(element['image'], dtype=np.uint8).reshape(
                int(height), int(width), 3)
            label = element.get('label')
            return (image, element['caption'].decode('utf-8'),
                    None if label is None else int(np.frombuffer(label, np.int32)[0]))
        return element['jpg'], element['txt'].decode('utf-8'), None
