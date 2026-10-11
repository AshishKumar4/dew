"""Typed recipe values that build record pipelines."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np

from dew.data.compile import compile
from dew.data.dataset import Batch, Dataset, Loading
from dew.data.dataset import Records as IndexedRecords
from dew.data.pipeline import Artifact, HostStep, Mixture, ObjectiveData, Pipeline, RandomMap, Rejected, Stage
from dew.records import integer

if TYPE_CHECKING:
    from dew.data.cache import Cache, Generation
    from dew.data.sources.files import Format
from dew.data.sources.base import Read, Shard, ShardManifest, Source
from dew.data.steps.chat import PreferencePair, RenderChat, RenderPrompt
from dew.data.steps.generic import Formatting
from dew.data.steps.image import (
    AugmentationKey,
    Caption,
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
_DEFAULT_LOADING = Loading(workers="auto")


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
        return None if lengths is None else lengths[self.start : self.stop]

    def manifest(self) -> ShardManifest:
        raise ValueError("indexed HoldOut needs an index; use fraction= with by='hash' for a stream")


@dataclasses.dataclass(frozen=True)
class _HashRows:
    fraction: float
    held: bool
    shard: str = "index"

    def __call__(self, index: int, row: Batch) -> Batch | Rejected:
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
        import grain.python as grain

        rows = self.source.records(read)
        keys = tuple(index for index in range(len(rows)) if (
            int.from_bytes(hashlib.sha256(f"index:{index}".encode()).digest()[:8], "big")
            < int(self.fraction * 2**64)) == self.held)
        return grain.MapDataset.source(keys).map(_SelectedRow(rows))

    def metadata(self) -> Mapping[str, object]:
        return self.source.metadata()

    def lengths(self) -> np.ndarray | None:
        return None

    def manifest(self) -> ShardManifest:
        manifest = self.source.manifest()
        return dataclasses.replace(manifest, reader=functools.partial(self._reader, manifest))

    def _reader(self, manifest: ShardManifest, shard: Shard):
        from dew.data.mix import Accepted

        return Accepted(manifest.reader(shard).map_with_index(_HashRows(self.fraction, self.held, shard.id)),
                        self.describe())


@dataclasses.dataclass(frozen=True)
class _SelectedRow:
    rows: IndexedRecords

    def __call__(self, index: int) -> Batch | bytes | None:
        return self.rows[index]


def _source(value: SourceValue, *, split: str = "train", format: Format | None = None) -> Source:
    from dew.data.sources import source

    return source(value, split=split, format=format)


def _branches(
    value: SourceValue,
    validation: Validation,
    *,
    split: str,
    prefix: tuple[Stage, ...],
    format: Format | None = None,
) -> tuple[Pipeline, Pipeline | None]:
    source = _source(value, split=split, format=format)
    held: Source | None = None
    train = Pipeline(source, prefix)
    if isinstance(validation, HoldOut):
        if validation.by == "hash":
            if validation.fraction is None:
                raise ValueError("hash HoldOut needs fraction=")
            train = Pipeline(_HashSource(source, validation.fraction, held=False), prefix)
            held = _HashSource(source, validation.fraction, held=True)
        else:
            count = validation.records
            if count is None:
                facts = source.metadata()
                if "rows" not in facts:
                    raise ValueError(f"fraction HoldOut of {source.describe()} needs authoritative rows")
                if validation.fraction is None:
                    raise ValueError("fraction HoldOut needs fraction=")
                count = int(integer(facts["rows"], "rows") * validation.fraction)
            train = Pipeline(_SliceRecords(source, count), prefix)
            held = _SliceRecords(source, 0, count) if count else None
    elif isinstance(validation, Split):
        held = _source(value, split=validation.name, format=format)
    elif validation is not None:
        held = _source(validation, format=format)
    evaluated = None if held is None else Pipeline(held, prefix)
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
    stride: int | None = None
    field: str = "text"
    eos: bool = False
    pack: Literal["first-fit", "best-fit", "concat"] | None = None
    packing_bins: int = 8
    validation: Validation = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        from dew.data.sources.files import TokenFiles

        stored = _source(self.source, split=self.split)
        if isinstance(stored, TokenFiles):
            return self._token_pipelines(stored)
        prefix = Pipeline(stored).map(Tokenize(self.tokenizer, self.field, self.eos)).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        if self.pack is None:
            train = train.windows(self.seq_len, stride=self.stride)
            held = None if held is None else held.windows(self.seq_len)
        else:
            train = train.flat_map(Chunks(self.seq_len + 1)).pack(
                self.seq_len + 1, bins=self.packing_bins, policy=self.pack
            )
            held = (
                None
                if held is None
                else held.flat_map(Chunks(self.seq_len + 1)).pack(
                    self.seq_len + 1, bins=self.packing_bins, policy=self.pack
                )
            )
        return train, held, ()


    def _token_pipelines(self, stored) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        from dew.data.steps.windows import Documents, WindowSource

        held = self.validation
        if held is None and stored.metadata().get("val_tokens", 0):
            held = dataclasses.replace(stored, split="val")
        sources = (stored, None if held is None else _source(held))

        def windows(source, stride):
            if self.pack is None:
                return Pipeline(WindowSource(source, source.tokens, self.seq_len, stride))
            if source.tokens.eos_id is None:
                raise ValueError(f"packed text needs producer document boundaries or EOS: {source.describe()}")
            return Pipeline(source).flat_map(Documents(source.tokens.eos_id)).pack(
                self.seq_len + 1, bins=self.packing_bins, policy=self.pack)

        return (windows(sources[0], self.stride or self.seq_len),
                None if sources[1] is None else windows(sources[1], self.seq_len), ())


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
                pipeline = pipeline.map(
                    LengthPolicy(self.seq_len + 1, long=self.long, aligned=("text_roles",))
                )
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
        step = RenderPrompt(
            self.tokenizer, self.max_prompt_len, self.pad_id, self.reward_bytes, self.long, self.thinking
        )
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
    crop_scale: tuple[float, float] = (1.0, 1.0)
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
        prefix = (
            Pipeline(_source(self.source))
            .map(DecodeImage(self.field, size), Resize(size), Caption(self.caption))
            .stages
        )
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        augment = self.augment
        if augment is not None:
            if augment.backend == "device":
                train = train.random_map(AugmentationKey())
                train = train.map_batch(DeviceAugment(self.image_size, augment.crop_scale,
                                                       augment.flip, augment.jitter))
            elif augment.crop or size != self.image_size or augment.crop_scale != (1.0, 1.0):
                train = train.random_map(
                    RandomCrop(self.image_size, augment.crop_scale, augment.flip, augment.jitter)
                )
            elif augment.jitter:
                train = train.random_map(ColorJitter(augment.flip))
            elif augment.flip:
                train = train.random_map(Flip())
        if held is not None:
            held = held.map(Resize(self.image_size))
        return train, held, (EncodeCaptions(),)

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
        decoder = DecodeClip(self.frames, self.frame_size, self.audio_padding, audio.processor.sampling_rate)
        train = train.random_map(decoder).map(audio)
        held = None if held is None else held.map(FixedClip(decoder), audio)
        return train, held, (EncodeCaptions(),)


def load(
    recipe: Recipe | str,
    *,
    batch: int,
    read: Read = "indexed",
    provision: bool = False,
    cache: Cache | Generation | str | None = None,
    objective: ObjectiveData | Callable[[Mapping[str, object]], ObjectiveData] | None = None,
    seed: int = 0,
    loading: Loading = _DEFAULT_LOADING,
    host: tuple[HostStep, ...] = (),
    **fields,
) -> Dataset:
    """Resolve a recipe alias and compile its training and evaluation pipelines."""
    if isinstance(recipe, str):
        if "://" in recipe or "/" in recipe:
            raise ValueError("load needs a recipe or alias; a bare URI names a source")
        from dew.registry import datasets

        resolved = datasets.build(recipe, **fields)
        if not isinstance(resolved, Recipe):
            raise TypeError(f"dataset alias {recipe!r} does not name a Pipeline recipe")
        recipe = resolved
    elif fields:
        if not dataclasses.is_dataclass(recipe):
            raise TypeError("recipe field overrides require a frozen dataclass")
        declared = {field.name for field in dataclasses.fields(recipe)}
        unknown = fields.keys() - declared
        if unknown:
            raise ValueError(f"unknown recipe fields {sorted(unknown)}")
        recipe = dataclasses.replace(recipe, **fields)
    if not isinstance(recipe, Recipe):
        raise TypeError("load expects a typed recipe with pipelines()")
    train, held, shared = recipe.pipelines()
    if cache is not None:
        train = _cached(train, cache)
        held = None if held is None else _cached(held, cache)
    train = _batched(_ordered(train, read, seed), batch, train=True)
    held = None if held is None else _batched(held, batch, train=False)
    return compile(
        train,
        validation=held,
        host=(*shared, *host),
        objective=objective,
        read=read,
        provision=provision,
        loading=loading,
        seed=seed,
    )


def _ordered(pipeline: Pipeline, read: Read, seed: int) -> Pipeline:
    if isinstance(pipeline.source, Mixture):
        children = tuple(_ordered(child, read, seed) for child in pipeline.source.pipelines)
        return dataclasses.replace(pipeline, source=dataclasses.replace(pipeline.source, pipelines=children))
    if read == "memory":
        at = next(
            (index for index, stage in enumerate(pipeline.stages) if isinstance(stage, RandomMap)),
            len(pipeline.stages),
        )
        order = Pipeline(pipeline.source).shuffle(seed).repeat().stages
    else:
        cache = next(
            (index for index, stage in enumerate(pipeline.stages) if isinstance(stage, Artifact)), -1
        )
        at = cache + 1
        order = (
            Pipeline(pipeline.source)
            .shuffle(seed, window=64 if read == "streamed" or cache >= 0 else None)
            .repeat()
            .stages
        )
    return dataclasses.replace(pipeline, stages=(*pipeline.stages[:at], *order, *pipeline.stages[at:]))


def _batched(pipeline: Pipeline, batch: int, *, train: bool) -> Pipeline:
    from dew.data.pipeline import MapBatch

    at = next((index for index, stage in enumerate(pipeline.stages)
               if isinstance(stage, MapBatch)), len(pipeline.stages))
    prefix = dataclasses.replace(pipeline, stages=pipeline.stages[:at]).batch(batch, drop_remainder=train)
    return dataclasses.replace(prefix, stages=(*prefix.stages, *pipeline.stages[at:]))


def _cached(pipeline: Pipeline, cache: Cache | Generation | str) -> Pipeline:
    from dew.data.pipeline import MapBatch, Pack, Windows

    if isinstance(pipeline.source, Mixture):
        children = tuple(_cached(child, cache) for child in pipeline.source.pipelines)
        return dataclasses.replace(pipeline, source=dataclasses.replace(pipeline.source, pipelines=children))
    at = next((index for index, stage in enumerate(pipeline.stages)
               if isinstance(stage, (RandomMap, MapBatch, Pack, Windows))), len(pipeline.stages))
    prefix = dataclasses.replace(pipeline, stages=pipeline.stages[:at]).cache(cache)
    return dataclasses.replace(prefix, stages=(*prefix.stages, *pipeline.stages[at:]))
