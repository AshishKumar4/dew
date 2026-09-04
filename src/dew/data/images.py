"""Image datasets: TFDS, Hugging Face hub and arrayrecord shards, one transform.

Every image dataset resizes, augments and captions its records the same way;
what differs is where the records come from and how one is read, which is
the three hooks a subclass fills in. Records leave as
`{"image": uint8 [size, size, 3], "caption": str}`, plus `"label"` when the
source carries a class index, and `load(tokenize=)` is where a run's own
condition reads the captions: the dataset carries the text, the encoder
decides what tokens it becomes. cv2, albumentations, tensorflow_datasets
and HF datasets are imported on use, so `import dew.data` costs none of
them.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import os
import struct as st
import threading
import weakref
from typing import Any, Literal

import grain.python as pygrain
import numpy as np

from dew.registry import datasets

from .dataset import (CAPTION, Dataset, DatasetSpec, Loading, hold_out, local_batch, tokenized,
                      train_stream, validation_pass)

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


def decode_image(encoded: bytes) -> np.ndarray:
    """An encoded image as RGB uint8 at its native size.

    cv2 decodes to BGR(A) and hands back None for a half-written file, which
    the colour conversion turns into an error rather than a black record.
    """
    import cv2
    image = cv2.imdecode(np.asarray(bytearray(encoded), dtype="uint8"), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cv2 could not decode {len(encoded)} bytes of image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """`image` at `size` square; area interpolation down, cubic up."""
    import cv2
    interpolation = cv2.INTER_CUBIC if size > 256 else cv2.INTER_AREA
    return cv2.resize(image, (size, size), interpolation=interpolation)


def image_augmentations(mode: Augmentation):
    """The albumentations pipeline for one augmentation mode: flip_only (DiT
    style), flip_jitter, or none (deterministic evaluation and debugging)."""
    import albumentations as A

    if mode == 'none':
        return A.Compose([])
    if mode == 'flip_only':
        return A.Compose([A.HorizontalFlip(p=0.5)])
    if mode == 'flip_jitter':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.05, saturation=0.2, hue=0, p=1.0),
        ])
    raise ValueError(f"augmentation {mode!r} is not one of none, flip_only, flip_jitter")


# Each thread's copies of the pipelines it has augmented with, keyed by the
# shared pipeline they were copied from.
_thread_pipelines = threading.local()


def augment_image(augments, image, rng: np.random.Generator):
    """Apply a pipeline from image_augmentations to one image.

    The seed is drawn from grain's per-record rng (Philox keyed by the record
    index), so every record gets the same augmentation regardless of how many
    workers, threads or processes produced the batch. albumentations keeps the
    generators a call draws from on the pipeline itself, so a pipeline shared
    between grain's prefetch threads had one record's seed applied to another
    record's pixels; each thread seeds and runs a copy of its own. numpy's
    global RNG is never touched from inside data-loading workers.
    """
    copies = getattr(_thread_pipelines, "copies", None)
    if copies is None:
        copies = _thread_pipelines.copies = weakref.WeakKeyDictionary()
    pipeline = copies.get(augments)
    if pipeline is None:
        pipeline = copies[augments] = copy.deepcopy(augments)
    pipeline.set_random_seed(int(rng.integers(0, 2**32 - 1)))
    return pipeline(image=image)['image']


PROMPT_TEMPLATES = (
    "a photo of a {}",
    "a photo of a {} flower",
    "This is a photo of a {}",
    "This is a photo of a {} flower",
    "A photo of a {} flower",
)


@functools.lru_cache(maxsize=None)
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


class ImageTransform(pygrain.RandomMapTransform):
    """Resize, augment and caption one record; the record's rng seeds both."""

    def __init__(self, spec: "ImageDataset"):
        self.spec = spec
        self.augments = image_augmentations(spec.augmentation)

    def random_map(self, element: Any, rng: np.random.Generator) -> dict[str, Any]:
        image, caption, label = self.spec.record(element, rng)
        image = augment_image(self.augments, resize_image(image, self.spec.image_size), rng)
        record = {"image": image, CAPTION: caption}
        if label is not None:
            # the class index, which the JEPA linear/kNN probes score against
            record["label"] = np.int32(label)
        return record


@dataclasses.dataclass(frozen=True)
class ImageDataset(DatasetSpec):
    """Captioned images through grain, resized to `image_size`.

    `val_batches` batches of records are held out of the head of the source,
    in canonical order, as the validation split, so FID and CLIP are never
    measured on records the model trained on; None or 0 holds nothing out.
    `count` uses that many records from the head of the source, and is what
    a source that reports no length needs set.
    """

    image_size: int = 128
    augmentation: Augmentation = "flip_jitter"
    val_batches: int | None = 4
    count: int | None = None
    seed: int = 0
    loading: Loading = Loading()

    def source(self) -> Any:
        """Random access over the records: `__getitem__`, and `__len__` unless
        `count` says how many there are."""
        raise NotImplementedError

    def record(self, element, rng: np.random.Generator) -> tuple[np.ndarray, str, int | None]:
        """One record as `(rgb uint8 image, caption, class index or None)`."""
        raise NotImplementedError

    def records(self, source) -> int:
        """The records the run uses, from the head of the source."""
        name = type(self).__name__
        if self.count is None:
            if not hasattr(source, "__len__"):
                raise ValueError(
                    f"{name} reports no length, so it needs count= set to the "
                    "records it holds")
            return len(source)
        if hasattr(source, "__len__") and self.count > len(source):
            raise ValueError(
                f"count {self.count} is more than the {len(source)} records of {name}")
        return self.count

    def load(self, *, batch: int, tokenize=None) -> Dataset:
        source = self.source()
        train, val = hold_out(source, self.records(source),
                              (self.val_batches or 0) * batch, type(self).__name__)
        return Dataset(
            train=tokenized(train_stream(train, [ImageTransform(self)], batch=local_batch(batch), seed=self.seed, loading=self.loading), tokenize),
            val=None if val is None else tokenized(
                validation_pass(val, [ImageTransform(self)], batch=local_batch(batch), seed=self.seed, loading=self.loading), tokenize),
            records=len(train),
            batch=batch,
        )


@datasets("oxford_flowers102")
@dataclasses.dataclass(frozen=True)
class OxfordFlowers(ImageDataset):
    """Oxford Flowers 102 from the TFDS data dir, captioned from its class
    name through a prompt template the record's rng picks."""

    split: str = "all"
    labels: str = "~/tensorflow_datasets/oxford_flowers102/2.1.1/label.labels.txt"

    def source(self):
        import tensorflow_datasets as tfds
        return tfds.data_source("oxford_flowers102", split=self.split, try_gcs=False)

    def record(self, element, rng):
        label = int(element["label"])
        # The template comes from the record's rng, like the augmentation: a
        # module-global random.choice made a record's caption depend on how
        # many workers and processes produced the batch.
        template = PROMPT_TEMPLATES[int(rng.integers(len(PROMPT_TEMPLATES)))]
        return element["image"], template.format(class_names(self.labels)[label]), label


@datasets("hf_images")
@dataclasses.dataclass(frozen=True)
class HFImages(ImageDataset):
    """A Hugging Face hub dataset of images with a 'caption' or 'text' column,
    read through grain's random access. `name` is the repo id."""

    name: str = ""
    split: str = "train"

    def source(self):
        from .sources.hf import HFDatasetSource
        if not self.name:
            raise ValueError("HFImages needs name= set to a hub dataset repo id")
        return HFDatasetSource(name=self.name, split=self.split)

    def record(self, element, rng):
        label = element.get("label")
        return element["image"], record_caption(element), None if label is None else int(label)


@dataclasses.dataclass(frozen=True)
class ArrayRecordImages(ImageDataset):
    """Image and caption pairs in arrayrecord shards under `path/<shard>/`,
    each record a packed dict with a 'jpg' and a 'txt' entry. `path` is the
    bucket mount or directory the shards live under."""

    path: str | None = None
    shards: tuple[str, ...] = ()

    def source(self):
        if not self.path:
            raise ValueError(
                f"{type(self).__name__} needs path= set: its records live under "
                "<path>/<shard>/ for each of its shards")
        files = []
        for shard in self.shards:
            root = os.path.join(self.path, shard)
            files += [os.path.join(root, f) for f in sorted(os.listdir(root))
                      if 'array_record' in f]
        return pygrain.ArrayRecordDataSource(files)

    def record(self, element, rng):
        element = unpack_dict_of_byte_arrays(element)
        return decode_image(element['jpg']), element['txt'].decode('utf-8'), None


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
