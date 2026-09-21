"""Image datasets: TFDS, Hugging Face hub and arrayrecord shards, one transform.

Every image dataset resizes, augments and captions its records the same way;
what differs is where the records come from and how one is read, which is
the three hooks a subclass fills in. Records leave as
`{"image": uint8 [size, size, 3], "caption": str}`, plus `"label"` when the
source carries a class index, and `load(tokenize=)` is where a run's own
condition reads the captions: the dataset carries the text, the encoder
decides what tokens it becomes. cv2, tensorflow_datasets
and HF datasets are imported on use, so `import dew.data` costs none of
them.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import struct as st
from typing import Literal

import grain.python as pygrain
import numpy as np

from dew.registry import datasets

from .dataset import (
    CAPTION,
    Batch,
    Dataset,
    DatasetSpec,
    Records,
    Tokenize,
    hold_out,
    local_batch,
    tokenized,
    train_stream,
    validation_pass,
)
from .providers import Counted
from .sources.hf import HFOptions, HubOptions
from .tokens import bounded

Augmentation = Literal["none", "flip_only", "flip_jitter"]


def unpack_dict_of_byte_arrays(packed_data):
    """Unpacks a dictionary of byte arrays from a packed binary format."""
    unpacked_dict = {}
    offset = 0
    while offset < len(packed_data):
        key_length = st.unpack_from('I', packed_data, offset)[0]
        offset += st.calcsize('I')
        key = packed_data[offset:offset+key_length].decode('utf-8')
        offset += key_length
        byte_array_length = st.unpack_from('I', packed_data, offset)[0]
        offset += st.calcsize('I')
        byte_array = packed_data[offset:offset+byte_array_length]
        offset += byte_array_length
        unpacked_dict[key] = byte_array
    return unpacked_dict

def pack_dict_of_byte_arrays(unpacked: dict) -> bytes:
    """The inverse of unpack_dict_of_byte_arrays: `str -> bytes` entries,
    length-prefixed, in dict order."""
    packed = bytearray()
    for key, byte_array in unpacked.items():
        encoded = key.encode('utf-8')
        packed += st.pack('I', len(encoded))
        packed += encoded
        packed += st.pack('I', len(byte_array))
        packed += byte_array
    return bytes(packed)


def decode_image(encoded: bytes, *, at_least: int | None = None) -> np.ndarray:
    """An encoded image as RGB uint8.

    With `at_least`, a JPEG is decoded at the largest 1/2, 1/4 or 1/8 DCT
    reduction that keeps both sides >= `at_least`, so the resize after it
    still only shrinks and most of the decode is skipped. cv2 hands back None
    for a half-written file; that becomes an error here.
    """
    import cv2
    buffer = np.frombuffer(encoded, dtype=np.uint8)
    flags = cv2.IMREAD_UNCHANGED
    if at_least is not None:
        shortest = min(_encoded_size(encoded))
        for factor, reduced in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4),
                                (2, cv2.IMREAD_REDUCED_COLOR_2)):
            if shortest // factor >= at_least:
                flags = reduced
                break
    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise ValueError(f"cv2 could not decode {len(encoded)} bytes of image")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB)


def _encoded_size(encoded: bytes) -> tuple[int, int]:
    """(height, width) from the header alone; PIL reads no pixels for this."""
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(encoded)) as header:
            width, height = header.size
    except Exception as error:
        raise ValueError(f"could not read the size of {len(encoded)} bytes of image") from error
    return height, width


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """`image` at `size` square; area interpolation down, cubic up."""
    import cv2
    if image.shape[:2] == (size, size):
        return image
    interpolation = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interpolation)


@dataclasses.dataclass(frozen=True)
class Augment:
    """The augmentations one mode applies: flip for flip_only, both for
    flip_jitter. 'none' maps to no Augment at all."""

    flip: bool
    jitter: bool


def image_augmentations(mode: Augmentation) -> Augment | None:
    """The augmentations for one mode: flip_only (DiT style), flip_jitter,
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
    """Flip and colour-jitter `image`, seeded by the record's own rng.

    Every draw comes from grain's per-record rng (Philox keyed by the record
    index), so a record's augmentation is the same however many workers,
    threads or processes produced its batch. uint8 pixels go through float32
    and are rounded and clipped once at the end.
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


PROMPT_TEMPLATES = (
    "a photo of a {}",
    "a photo of a {} flower",
    "This is a photo of a {}",
    "This is a photo of a {} flower",
    "A photo of a {} flower",
)


@functools.cache
def class_names(path: str) -> tuple[str, ...]:
    """The class names of a labels file, one per line, read once per process."""
    with open(os.path.expanduser(path)) as handle:
        return tuple(line.strip() for line in handle)


def record_caption(element) -> str:
    """The caption a record already carries, for datasets that ship their text.

    Hub image datasets keep it in a 'caption' or a 'text' column.
    """
    for key in ("caption", "text"):
        if key in element:
            return element[key]
    raise KeyError(
        "an image record needs a 'caption' or a 'text' column, this one has "
        f"{sorted(element)}")


def _fields(element: Batch | bytes, name: str) -> Batch:
    """A record's fields, or the refusal that this source holds bytes.

    The three hooks a spec fills in belong together: a spec whose `source`
    reads a table of features reads them by name here, and one whose source
    is arrayrecord bytes unpacks them itself.
    """
    if isinstance(element, bytes):
        raise TypeError(
            f"{name} reads records of named fields, and this source holds "
            f"{len(element)} bytes; unpack them in record() instead")
    return element


class ImageTransform(pygrain.RandomMapTransform):
    """Resize, augment and caption one record, seeded by the record's own rng."""

    def __init__(self, spec: ImageDataset):
        self.spec = spec
        self.augments = image_augmentations(spec.augmentation)

    def random_map(self, element: Batch | bytes, rng: np.random.Generator) -> Batch:
        image, caption, label = self.spec.record(element, rng)
        if isinstance(image, bytes):
            image = decode_image(image, at_least=self.spec.image_size)
        image = augment_image(self.augments, resize_image(image, self.spec.image_size), rng)
        record = {"image": image, CAPTION: caption}
        if label is not None:
            # the class index, which the JEPA linear/kNN probes score against
            record["label"] = np.int32(label)
        return record


@dataclasses.dataclass(frozen=True)
class ImageDataset(DatasetSpec):
    """Captioned images through grain, resized to `image_size`.

    Validation comes from one of two places. `val_split` names a split of
    the dataset's own, which is opened as a second source and scored in
    record order, `val_batches` batches of it or all of it when that is
    None. Without one, `val_batches` batches of records are held out of the
    head of the training source, in canonical order, so FID and CLIP are
    still never measured on records the model trained on; None or 0 holds
    nothing out and validates nothing.

    `count` takes that many records from the head of the source. A source
    that reports no length needs it set.
    """

    image_size: int = 128
    augmentation: Augmentation = "flip_jitter"
    val_batches: int | None = 4
    val_split: str | None = None
    count: int | None = None

    def source(self, split: str | None = None) -> Records:
        """Random access over the records (`__getitem__`, and `__len__` unless
        `count` says how many there are).

        `split` names a split other than the one this spec reads, which is
        how `val_split` opens a second source; a dataset whose records are
        one pile refuses it.
        """
        raise NotImplementedError

    def record(self, element: Batch | bytes,
               rng: np.random.Generator) -> tuple[np.ndarray | bytes, str, int | None]:
        """One record as `(image, caption, class index or None)`; the image is
        RGB uint8, or the encoded bytes for the transform to decode at the size
        it needs."""
        raise NotImplementedError

    def records(self, source: Records) -> int:
        """The records the run uses, from the head of the source.

        A source that cannot count itself is `Counted`'s other case: the
        spec's own `count` is then the whole record of how many there are.
        """
        name = type(self).__name__
        if self.count is None:
            if not isinstance(source, Counted):
                raise ValueError(
                    f"{name} reports no length, so it needs count= set to the "
                    "records it holds")
            return len(source)
        if isinstance(source, Counted) and self.count > len(source):
            raise ValueError(
                f"count {self.count} is more than the {len(source)} records of {name}")
        return self.count

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        source = self.source()
        # A named split is its own records, so nothing is held out of
        # training; without one the head of the source is the split.
        held_out = 0 if self.val_split else (self.val_batches or 0) * batch
        train, validation = hold_out(source, self.records(source), held_out,
                                     type(self).__name__)
        rows = local_batch(batch)
        if self.val_split:
            validation = self.source(self.val_split)
        scored = None if validation is None else tokenized(
            validation_pass(validation, [ImageTransform(self)], batch=rows,
                            seed=self.seed, loading=self.loading), tokenize)
        if self.val_split and scored is not None:
            scored = bounded(scored, self.val_batches)
        return Dataset(
            train=tokenized(train_stream(train, [ImageTransform(self)], batch=rows, seed=self.seed, loading=self.loading), tokenize),
            val=scored,
            records=len(train),
            batch=batch,
        )


@datasets("oxford_flowers102")
@dataclasses.dataclass(frozen=True)
class OxfordFlowers(ImageDataset):
    """Prepared Oxford Flowers ArrayRecords, captioned from their class names.

    Preparation runs separately. Reading uses TFDS metadata and NumPy image
    decoding through its read-only builder, without TensorFlow or dataset
    generation code in the training process.
    """

    path: str | None = None
    """Prepared version directory containing dataset_info.json and ArrayRecords."""
    split: str = "all"
    labels: str | None = None
    """Class-name file override; unset reads label.labels.txt in path."""

    def source(self, split: str | None = None):
        if not self.path:
            raise ValueError(
                "OxfordFlowers needs path= (--data.path) pointing to prepared "
                "TFDS ArrayRecords. Prepare oxford_flowers102 separately with "
                "download_and_prepare(file_format='array_record'), then pass "
                "the builder.data_dir version directory to training.")
        import tensorflow_datasets as tfds

        from .sources.tfds import prepared_source

        return prepared_source(self.path, split or self.split,
                               decoders={"image": tfds.decode.SkipDecoding()})

    def record(self, element: Batch | bytes, rng):
        element = _fields(element, "OxfordFlowers")
        label = int(element["label"])
        # The template comes from the record's rng, like the augmentation.
        # A module-global random.choice would key a record's caption to how
        # many workers and processes produced the batch.
        template = PROMPT_TEMPLATES[int(rng.integers(len(PROMPT_TEMPLATES)))]
        labels = self.labels
        if labels is None:
            if self.path is None:
                raise ValueError("OxfordFlowers captions need labels= or a prepared path=.")
            labels = os.path.join(self.path, "label.labels.txt")
        return element["image"], template.format(class_names(labels)[label]), label


@datasets("hf_images")
@dataclasses.dataclass(frozen=True)
class HFImages(ImageDataset):
    """A Hugging Face hub dataset of images with a 'caption' or 'text' column,
    read through grain's random access.

    `name` is the repo id and `split` the split to read; `options` is
    everything else `datasets.load_dataset` takes, the same value the `hf`
    provider holds, so a dataset behind a config name, a revision, its own
    `data_files` or a token is read here too.
    """

    name: str = ""
    split: str = "train"
    options: HubOptions = HFOptions()

    def source(self, split: str | None = None):
        from .sources.hf import HFDatasetSource
        if not self.name:
            raise ValueError("HFImages needs name= set to a hub dataset repo id")
        return HFDatasetSource(name=self.name, split=split or self.split,
                               options=self.options)

    def record(self, element: Batch | bytes, rng):
        element = _fields(element, "HFImages")
        label = element.get("label")
        return element["image"], record_caption(element), None if label is None else int(label)


@datasets("array_record_images")
@dataclasses.dataclass(frozen=True)
class ArrayRecordImages(ImageDataset):
    """Image and caption pairs in arrayrecord shards under `path/<shard>/`,
    each record a packed dict. Two layouts: 'jpg'/'txt' entries (encoded,
    decoded on read) and the `prepare_images.py` layout 'image'/'shape'/
    'caption' (uint8 HxWx3 already at training size, shape two little-endian
    int32s, 'label' optional). `path` is the bucket mount or directory the
    shards live under; an empty `shards` reads every arrayrecord file in
    `path` itself, which is the layout prepare_images.py writes."""

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


# The msml612 shards live in gs://msml612-diffusion-data, read through a gcs
# fuse mount handed over as `path`.

@datasets("laion12m_coco")
@dataclasses.dataclass(frozen=True)
class Laion12mCoco(ArrayRecordImages):
    """laion-aesthetics-12M (score >= 6) plus MS-COCO 2017: 228 shards, 236 GiB, about 15M samples."""
    shards: tuple[str, ...] = ("arrayrecord2/laion12m_coco",)


@datasets("laion2b_aesthetic")
@dataclasses.dataclass(frozen=True)
class Laion2bAesthetic(ArrayRecordImages):
    """laion-2B-en aesthetic >= 4.2 subset: 569 shards, 550 GiB, larger but noisier."""
    shards: tuple[str, ...] = ("arrayrecord2/laion2B-en-aesthetic",)


@datasets("diffusiondb")
@dataclasses.dataclass(frozen=True)
class DiffusionDB(ArrayRecordImages):
    """diffusiondb (SD synthetic images and prompts): 31 shards, 60 GiB, 1.97M samples."""
    shards: tuple[str, ...] = ("arrayrecord2/diffusiondb",)


@datasets("cc3m")
@dataclasses.dataclass(frozen=True)
class CC3M(ArrayRecordImages):
    """Conceptual Captions 3M: 50 shards, 37 GiB, about 3.3M samples (shard 00039 missing)."""
    shards: tuple[str, ...] = ("arrayrecord2/cc3m",)


@datasets("combined_msml612")
@dataclasses.dataclass(frozen=True)
class CombinedMsml612(ArrayRecordImages):
    """The four msml612 datasets together, about 883 GiB and 20M samples."""
    shards: tuple[str, ...] = (
        "arrayrecord2/laion12m_coco",
        "arrayrecord2/laion2B-en-aesthetic",
        "arrayrecord2/diffusiondb",
        "arrayrecord2/cc3m",
    )


# Older shard layouts; the paths may not exist on the current bucket.

@datasets("cc12m")
@dataclasses.dataclass(frozen=True)
class CC12M(ArrayRecordImages):
    shards: tuple[str, ...] = ("arrayrecord2/cc12m",)


@datasets("laiona_coco")
@dataclasses.dataclass(frozen=True)
class LaionaCoco(ArrayRecordImages):
    shards: tuple[str, ...] = ("arrayrecord2/laion-aesthetics-12m+mscoco-2017",)


@datasets("aesthetic_coyo")
@dataclasses.dataclass(frozen=True)
class AestheticCoyo(ArrayRecordImages):
    shards: tuple[str, ...] = ("arrayrecords/aestheticCoyo_0.25clip_6aesthetic",)


@datasets("combined_aesthetic")
@dataclasses.dataclass(frozen=True)
class CombinedAesthetic(ArrayRecordImages):
    shards: tuple[str, ...] = (
        "arrayrecord2/laion-aesthetics-12m+mscoco-2017",
        "arrayrecords/aestheticCoyo_0.25clip_6aesthetic",
        "arrayrecord2/cc12m",
        "arrayrecords/aestheticCoyo_0.25clip_6aesthetic",
    )


@datasets("laiona_coco_coyo")
@dataclasses.dataclass(frozen=True)
class LaionaCocoCoyo(ArrayRecordImages):
    shards: tuple[str, ...] = (
        "arrayrecords/aestheticCoyo_0.25clip_6aesthetic",
        "arrayrecord2/laion-aesthetics-12m+mscoco-2017",
        "arrayrecords/aestheticCoyo_0.25clip_6aesthetic",
    )


@datasets("combined_30m")
@dataclasses.dataclass(frozen=True)
class Combined30M(ArrayRecordImages):
    shards: tuple[str, ...] = (
        "arrayrecord2/laion-aesthetics-12m+mscoco-2017",
        "arrayrecord2/cc12m",
        "arrayrecord2/aestheticCoyo_0.26_clip_5.5aesthetic_256plus",
        "arrayrecord2/playground+leonardo_x4+cc3m.parquet",
    )
