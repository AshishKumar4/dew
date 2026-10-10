"""Typed recipe values that build record pipelines."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from dew.data.compile import compile
from dew.data.dataset import Batch, Dataset, Loading
from dew.data.pipeline import HostStep, ObjectiveData, Pipeline, Stage
from dew.data.sources.base import Read, Shard, ShardManifest, Source
from dew.data.steps.chat import PreferencePair, RenderChat, RenderPrompt
from dew.data.steps.generic import Formatting
from dew.data.steps.image import (
    AugmentationKey,
    Caption,
    CenterCrop,
    ColorJitter,
    DecodeImage,
    DeviceAugment,
    EncodeCaptions,
    Flip,
    RandomCrop,
    Resize,
)
from dew.data.steps.text import Chunks, LengthPolicy, Tokenize
from dew.data.steps.video import Audio, DecodeClip, FixedClip

type SourceValue = Source | str | Sequence[Batch] | Mapping[str, Sequence]
type Validation = HoldOut | Split | SourceValue | None


@dataclasses.dataclass(frozen=True)
class HoldOut:
    records: int | None = None
    fraction: float | None = None
    by: Literal["index", "hash"] = "index"

    def __post_init__(self) -> None:
        if (self.records is None) == (self.fraction is None):
            raise ValueError("HoldOut needs records or fraction, exactly one")
        if self.records is not None and self.records < 0:
            raise ValueError("HoldOut records must be nonnegative")
        if self.fraction is not None and not 0 <= self.fraction < 1:
            raise ValueError("HoldOut fraction must be in [0, 1)")


@dataclasses.dataclass(frozen=True)
class Split:
    name: str = "validation"


@dataclasses.dataclass(frozen=True)
class _SliceRecords:
    source: Source
    start: int
    stop: int | None = None

    def describe(self) -> str:
        return f"{self.source.describe()}[{self.start}:{self.stop}]"

    def snapshot(self) -> Mapping[str, object]:
        return {**self.source.snapshot(), "slice": [self.start, self.stop]}

    def supports(self, read: Read) -> bool:
        return read != "streamed" and self.source.supports(read)

    def records(self, read: Read):
        from dew.data.dataset import SourceSlice

        records = self.source.records(read)
        stop = len(records) if self.stop is None else self.stop
        if not 0 <= self.start <= stop <= len(records):
            raise ValueError(f"{self.describe()} exceeds the source's {len(records)} records")
        return SourceSlice(records, self.start, stop)

    def metadata(self) -> Mapping[str, object]:
        return self.source.metadata()

    def lengths(self) -> np.ndarray | None:
        lengths = self.source.lengths()
        return None if lengths is None else lengths[self.start:self.stop]

    def manifest(self) -> ShardManifest:
        raise ValueError("indexed HoldOut needs an index; use fraction= with by='hash' for a stream")


@dataclasses.dataclass(frozen=True)
class _HashRows:
    fraction: float
    held: bool
    shard: str = "index"

    def __call__(self, index: int, row: Batch) -> Batch:
        from dew.data.pipeline import Rejected

        digest = hashlib.sha256(f"{self.shard}:{index}".encode()).digest()
        accepted = int.from_bytes(digest[:8], "big") < int(self.fraction * 2**64)
        return row if accepted == self.held else Rejected("holdout")


@dataclasses.dataclass(frozen=True)
class _HashSource:
    source: Source
    fraction: float
    held: bool

    def describe(self) -> str:
        return f"{self.source.describe()} hash holdout={self.fraction},held={self.held}"

    def snapshot(self) -> Mapping[str, object]:
        return {**self.source.snapshot(), "holdout": self.fraction, "held": self.held}

    def supports(self, read: Read) -> bool:
        return self.source.supports(read)

    def records(self, read: Read):
        return self.source.records(read)

    def metadata(self) -> Mapping[str, object]:
        return self.source.metadata()

    def lengths(self) -> np.ndarray | None:
        return self.source.lengths()

    def manifest(self) -> ShardManifest:
        manifest = self.source.manifest()
        return dataclasses.replace(manifest, reader=functools.partial(self._reader, manifest))

    def _reader(self, manifest: ShardManifest, shard: Shard):
        return manifest.reader(shard).map_with_index(_HashRows(self.fraction, self.held, shard.id))


def _source(value: SourceValue, *, split: str = "train", format: str | None = None) -> Source:
    from dew.data.sources import source

    return source(value, split=split, format=format)


def _branches(value: SourceValue, validation: Validation, *, split: str, prefix: tuple[Stage, ...],
              format: str | None = None) -> tuple[Pipeline, Pipeline | None]:
    source = _source(value, split=split, format=format)
    held: Source | None = None
    train = Pipeline(source, prefix)
    if isinstance(validation, HoldOut):
        if validation.by == "hash":
            if validation.fraction is None:
                raise ValueError("hash HoldOut needs fraction=")
            train = Pipeline(_HashSource(source, validation.fraction, False), prefix)
            held = _HashSource(source, validation.fraction, True)
            train = train.map_with_index(_HashRows(validation.fraction, False), identity="dew-holdout@1")
        else:
            count = validation.records
            if count is None:
                facts = source.metadata()
                if "rows" not in facts:
                    raise ValueError(f"fraction HoldOut of {source.describe()} needs authoritative rows")
                count = int(facts["rows"] * validation.fraction)
            train = Pipeline(_SliceRecords(source, count), prefix)
            held = _SliceRecords(source, 0, count) if count else None
    elif isinstance(validation, Split):
        held = _source(value, split=validation.name, format=format)
    elif validation is not None:
        held = _source(validation, format=format)
    evaluated = None if held is None else Pipeline(held, prefix)
    if isinstance(validation, HoldOut) and validation.by == "hash" and evaluated is not None:
        evaluated = evaluated.map_with_index(_HashRows(validation.fraction, True), identity="dew-holdout@1")
    return train, evaluated


@runtime_checkable
class Recipe(Protocol):
    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]: ...


@dataclasses.dataclass(frozen=True)
class Records:
    source: SourceValue
    validation: Validation = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        prefix = Pipeline(_source(self.source)).map(Formatting()).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        return train, held, ()


@dataclasses.dataclass(frozen=True)
class Text:
    source: SourceValue
    tokenizer: str = "byte"
    seq_len: int = 256
    field: str = "text"
    eos: bool = False
    pack: Literal["first-fit", "best-fit", "concat"] = "concat"
    packing_bins: int = 8
    validation: Validation = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        prefix = Pipeline(_source(self.source)).map(Tokenize(self.tokenizer, self.field, self.eos)).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        train = train.flat_map(Chunks(self.seq_len + 1)).pack(self.seq_len + 1, bins=self.packing_bins,
                                                           policy=self.pack)
        held = None if held is None else held.flat_map(Chunks(self.seq_len + 1)).pack(
            self.seq_len + 1, bins=self.packing_bins, policy=self.pack)
        return train, held, ()


@dataclasses.dataclass(frozen=True)
class Chat:
    source: SourceValue
    tokenizer: str
    seq_len: int = 256
    column: str = "messages"
    long: Literal["truncate", "drop", "chunk"] = "truncate"
    packing_bins: int = 8
    validation: Validation = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        prefix = Pipeline(_source(self.source)).map(RenderChat(self.tokenizer, self.column)).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)

        def policy(pipeline: Pipeline) -> Pipeline:
            if self.long == "chunk":
                pipeline = pipeline.flat_map(Chunks(self.seq_len + 1, ("text", "text_roles")))
            else:
                pipeline = pipeline.map(LengthPolicy(self.seq_len + 1, long=self.long, aligned=("text_roles",)))
            return pipeline.pack(self.seq_len + 1, fields=("text", "text_roles"), bins=self.packing_bins)

        return policy(train), None if held is None else policy(held), ()


@dataclasses.dataclass(frozen=True)
class Prompts:
    source: SourceValue
    tokenizer: str
    max_prompt_len: int = 256
    pad_id: int = 0
    reward_bytes: int = 1024
    long: Literal["drop", "truncate"] = "drop"
    thinking: bool | None = None
    validation: Validation = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        step = RenderPrompt(self.tokenizer, self.max_prompt_len, self.pad_id, self.reward_bytes,
                            self.long, self.thinking)
        prefix = Pipeline(_source(self.source)).map(step).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        return train, held, ()


@dataclasses.dataclass(frozen=True)
class Preferences:
    source: SourceValue
    seq_len: int = 256
    pad_id: int = 0
    validation: Validation = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        prefix = Pipeline(_source(self.source)).map(PreferencePair(self.seq_len, self.pad_id)).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        return train, held, ()


@dataclasses.dataclass(frozen=True)
class Augment:
    crop: bool = False
    flip: bool = True
    jitter: bool = False
    crop_scale: tuple[float, float] = (1., 1.)
    backend: Literal["host", "device"] = "device"


@dataclasses.dataclass(frozen=True)
class Images:
    source: SourceValue
    image_size: int = 128
    staging_size: int | None = None
    field: str = "image"
    caption: str | None = None
    augment: Augment | None = dataclasses.field(default_factory=Augment)
    validation: Validation = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        size = self.staging_size or self.image_size
        prefix = Pipeline(_source(self.source)).map(DecodeImage(self.field, size), Resize(size),
                                                   Caption(self.caption)).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        augment = self.augment
        if augment is not None:
            if augment.backend == "device":
                train = train.random_map(AugmentationKey())
            elif augment.crop or size != self.image_size or augment.crop_scale != (1., 1.):
                train = train.random_map(RandomCrop(self.image_size, augment.crop_scale, augment.flip, augment.jitter))
            elif augment.jitter:
                train = train.random_map(ColorJitter(augment.flip))
            elif augment.flip:
                train = train.random_map(Flip())
        if held is not None:
            held = held.map(Resize(self.image_size))
        return train, held, (EncodeCaptions(),)

    def train_batch(self) -> tuple[HostStep, ...]:
        augment = self.augment
        return () if augment is None or augment.backend != "device" else (
            DeviceAugment(self.image_size, augment.crop_scale, augment.flip, augment.jitter),)


@dataclasses.dataclass(frozen=True)
class Video:
    source: SourceValue
    frames: int = 16
    frame_size: int = 256
    audio_padding: int = 3
    audio_model: str = "facebook/wav2vec2-base-960h"
    validation: Validation = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        train, held = _branches(self.source, self.validation, split=self.split, prefix=())
        audio = Audio(self.audio_model)
        decoder = DecodeClip(self.frames, self.frame_size, self.audio_padding, audio._processor().sampling_rate)
        train = train.random_map(decoder).map(audio)
        held = None if held is None else held.map(FixedClip(decoder), audio)
        return train, held, (EncodeCaptions(),)


def load(recipe: Recipe | str, *, batch: int, read: Read = "indexed", provision: bool = False,
         objective: ObjectiveData | Callable[[Mapping[str, object]], ObjectiveData] | None = None,
         seed: int = 0, loading: Loading = Loading(workers="auto"), host: tuple[HostStep, ...] = (),
         **fields) -> Dataset:
    """Resolve a recipe alias and compile its training and evaluation pipelines."""
    if isinstance(recipe, str):
        if "://" in recipe or "/" in recipe:
            raise ValueError("load needs a recipe or alias; a bare URI names a source")
        from dew.registry import datasets

        recipe = datasets.build(recipe, **fields)
    elif fields:
        declared = {field.name for field in dataclasses.fields(recipe)}
        unknown = fields.keys() - declared
        if unknown:
            raise ValueError(f"unknown recipe fields {sorted(unknown)}")
        recipe = dataclasses.replace(recipe, **fields)
    if not isinstance(recipe, Recipe):
        raise TypeError("load expects a typed recipe with pipelines()")
    train, held, shared = recipe.pipelines()
    train = train.shuffle(seed).repeat().batch(batch)
    if isinstance(recipe, Images):
        train = train.map_batch(*recipe.train_batch())
    held = None if held is None else held.batch(batch, drop_remainder=False)
    return compile(train, validation=held, host=(*shared, *host), objective=objective,
                   read=read, provision=provision, loading=loading, seed=seed)
