"""Typed recipe values that build record pipelines."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import random
from types import UnionType
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, TypeAliasType, runtime_checkable

import numpy as np
import tyro

from dew.data.compile import compile
from dew.data.dataset import Batch, DataPhase, Dataset, Loading, record_argument
from dew.data.dataset import Records as IndexedRecords
from dew.data.pipeline import Artifact, Configured, HostStep, Mixture, ObjectiveData, Phases, Pipeline, RandomMap, Rejected, Stage
from dew.records import integer

if TYPE_CHECKING:
    from dew.data.cache import Cache, Generation
    from dew.data.sources.files import Format
from dew.data.sources.base import Read, Shard, ShardManifest, Source
from dew.data.steps.chat import ConfigurePrompt, PreferencePair, RenderChat, RenderPrompt
from dew.data.steps.generic import Formatting
from dew.data.steps.image import (
    AugmentationKey,
    Caption,
    ColorJitter,
    ConfigureImage,
    DecodeImage,
    DeviceAugment,
    EncodeCaptions,
    Flip,
    ImageRecord,
    RandomCrop,
    Resize,
)
from dew.data.steps.text import Chunks, LengthPolicy, Tokenize, ValidTargets
from dew.data.steps.video import Audio, DecodeClip, VideoPath

type SourceValue = Source | str | Sequence[Batch] | Mapping[str, Sequence]
type Validation = HoldOut | Split | Pipeline | Source | str | Sequence[Batch] | Mapping[str, Sequence] | None
_DEFAULT_LOADING = Loading(workers="auto")


def source_argument(annotation: type | UnionType | TypeAliasType):
    def read(words):
        from dew.registry import from_record

        word = words[0]
        if word == "None":
            return None
        if word.startswith(("{", "[")):
            value = json.loads(word)
            return from_record(annotation, value, dtypes=False) if isinstance(value, Mapping) and "class" in value else value
        return word

    def write(value):
        from dew.registry import to_record

        return [value if isinstance(value, str) else "None" if value is None else json.dumps(to_record(value, None))]

    return tyro.constructors.PrimitiveConstructorSpec(nargs=1, metavar="URL|JSON", instance_from_str=read,
        is_instance=lambda value: value is None or isinstance(value, (str, Source, Mapping, Sequence))
                                  or dataclasses.is_dataclass(value),
        str_from_instance=write)


@dataclasses.dataclass(frozen=True)
class HoldOut:
    records: int | None = None
    fraction: float | None = None
    by: Literal["index", "hash", "random"] = "index"
    most: int | None = None
    seed: int = 0

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
        stated = self.source.metadata()
        return {**stated, **({"held_out": self.start} if self.start else {})}

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


@dataclasses.dataclass(frozen=True)
class _RandomSource:
    source: Source
    split: HoldOut
    held: bool

    def describe(self) -> str:
        return f"{self.source.describe()} random holdout={self.split!r},held={self.held}"

    def snapshot(self) -> Mapping[str, object]:
        return {**self.source.snapshot(), "holdout": dataclasses.asdict(self.split), "held": self.held}

    def supports(self, read: Read) -> bool:
        return read != "streamed" and self.source.supports(read)

    def _indices(self, count: int) -> tuple[int, ...]:
        size = self.split.records if self.split.records is not None else int(count * (self.split.fraction or 0.))
        size = min(size, self.split.most) if self.split.most is not None else size
        order = list(range(count))
        random.Random(self.split.seed).shuffle(order)
        chosen = set(order[:size])
        return tuple(index for index in range(count) if (index in chosen) == self.held)

    def records(self, read: Read):
        import grain.python as grain

        rows = self.source.records(read)
        return grain.MapDataset.source(self._indices(len(rows))).map(_SelectedRow(rows))

    def metadata(self) -> Mapping[str, object]:
        return self.source.metadata()

    def lengths(self) -> np.ndarray | None:
        lengths = self.source.lengths()
        return None if lengths is None else lengths[list(self._indices(len(lengths)))]

    def manifest(self) -> ShardManifest:
        raise ValueError("random-index HoldOut needs indexed records; use by='hash' for streams")


def _source(value: SourceValue | Mapping[str, float], *, split: str = "train",
            format: Format | None = None) -> Source:
    from dew.data.sources import source

    return source(value, split=split, format=format)


def _branches(
    value: SourceValue | Mapping[str, float],
    validation: Validation,
    *,
    split: str,
    prefix: tuple[Stage, ...],
    format: Format | None = None,
) -> tuple[Pipeline, Pipeline | None]:
    source = _source(value, split=split, format=format)
    held: Source | None = None
    train = Pipeline(source, prefix)
    if isinstance(validation, Pipeline):
        return train, validation
    if isinstance(validation, HoldOut):
        if validation.by == "random":
            train = Pipeline(_RandomSource(source, validation, held=False), prefix)
            held = _RandomSource(source, validation, held=True)
        elif validation.by == "hash":
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


@runtime_checkable
class _HostBinding(Protocol):
    binding: str
    def __call__(self, batch: Batch) -> Batch: ...


def _host_stages(defaults: tuple[HostStep, ...], explicit: tuple[HostStep, ...]) -> tuple[HostStep, ...]:
    bindings = {stage.binding: stage for stage in explicit if isinstance(stage, _HostBinding)}
    configured = tuple(bindings.pop(stage.binding, stage) if isinstance(stage, _HostBinding) else stage
                       for stage in defaults)
    return (*configured, *(stage for stage in explicit if not isinstance(stage, _HostBinding)),
            *bindings.values())


@dataclasses.dataclass(frozen=True)
class Records:
    source: Annotated[SourceValue, source_argument(SourceValue)]
    validation: Annotated[Validation, source_argument(Validation)] = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        prefix = Pipeline(_source(self.source)).map(Formatting()).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        return train, held, ()


@dataclasses.dataclass(frozen=True)
class Text:
    source: Annotated[SourceValue | Mapping[str, float], source_argument(SourceValue)] = ""
    tokenizer: str = "byte"
    seq_len: int = 256
    stride: int | None = None
    field: str = "text"
    eos: bool = False
    pack: Literal["first-fit", "best-fit", "concat"] | None = None
    packing_bins: int = 8
    validation: Annotated[Validation, source_argument(Validation)] = None
    split: str = "train"
    phases: Annotated[tuple[DataPhase, ...], record_argument(tuple[DataPhase, ...])] = ()
    token_holdout: HoldOut | None = None

    def __post_init__(self) -> None:
        if self.seq_len < 1 or self.stride is not None and self.stride < 1:
            raise ValueError("text seq_len and stride must be positive")
        if self.packing_bins < 1:
            raise ValueError("packing_bins must be positive")

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        from dew.data.sources.files import TokenFiles

        if self.phases:
            if self.source:
                raise ValueError("Text takes source or phases, not both")
            branches = [dataclasses.replace(self, source=phase.path, phases=()).pipelines()
                        for phase in self.phases]
            return Pipeline(Phases(tuple(branch[0] for branch in branches),
                                   tuple(phase.until_step for phase in self.phases))), branches[0][1], ()
        if isinstance(self.source, Mapping) and all(isinstance(weight, (int, float))
                                                   for weight in self.source.values()):
            branches = [dataclasses.replace(self, source=path).pipelines() for path in sorted(self.source)]
            if not branches:
                raise ValueError("Text mixture has no corpora")
            weighted = {path: float(weight) for path, weight in self.source.items()
                        if isinstance(weight, (int, float))}
            weights = [weighted[path] for path in sorted(weighted)]
            trains = [branch[0] for branch in branches]
            held = [branch[1] for branch in branches]
            train = trains[0] if len(trains) == 1 else trains[0].mix(*trains[1:], weights=weights, tokens=ValidTargets())
            scored = [branch for branch in held if branch is not None]
            validation = (None if not scored else scored[0] if len(scored) == 1
                          else scored[0].mix(*scored[1:], weights=weights))
            return train, validation, ()
        stored = _source(self.source, split=self.split)
        if isinstance(stored, TokenFiles):
            return self._token_pipelines(stored)
        prefix = Pipeline(stored).map(Tokenize(self.tokenizer, self.field, self.eos)).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        if self.token_holdout is not None:
            if self.validation is not None or self.token_holdout.fraction is None:
                raise ValueError("token_holdout needs fraction= and no separate validation source")
            held = train.token_split(self.token_holdout.fraction, held=True)
            train = train.token_split(self.token_holdout.fraction, held=False)
        if self.pack is None:
            train = train.windows(self.seq_len, stride=self.stride)
            held = None if held is None else held if isinstance(self.validation, Pipeline) else held.windows(self.seq_len)
        else:
            train = train.flat_map(Chunks(self.seq_len + 1)).pack(
                self.seq_len + 1, bins=self.packing_bins, policy=self.pack
            )
            held = self.validation if isinstance(self.validation, Pipeline) else (
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
        if isinstance(held, Split):
            held = dataclasses.replace(stored, split=held.name)
        elif isinstance(held, HoldOut):
            raise ValueError("token-store validation uses its named split; slice a windows Pipeline for HoldOut")
        if held is None and stored.metadata().get("val_tokens", 0):
            held = dataclasses.replace(stored, split="val")
        sources = (stored, None if held is None or isinstance(held, Pipeline) else _source(held))

        def windows(source, stride):
            if self.pack is None:
                return Pipeline(WindowSource(source, source.tokens, self.seq_len, stride))
            if source.tokens.eos_id is None:
                raise ValueError(f"packed text needs producer document boundaries or EOS: {source.describe()}")
            return Pipeline(source).flat_map(Documents(source.tokens.eos_id)).pack(
                self.seq_len + 1, bins=self.packing_bins, policy=self.pack)

        return (windows(sources[0], self.seq_len if self.stride is None else self.stride),
                held if isinstance(held, Pipeline) else None if sources[1] is None else windows(sources[1], self.seq_len), ())


@dataclasses.dataclass(frozen=True)
class Chat:
    source: Annotated[SourceValue, source_argument(SourceValue)]
    tokenizer: str
    seq_len: int = 256
    column: str = "messages"
    long: Literal["truncate", "drop", "chunk"] = "truncate"
    packing_bins: int = 8
    validation: Annotated[Validation, source_argument(Validation)] = None
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

        return policy(train), None if held is None else held if isinstance(self.validation, Pipeline) else policy(held), ()


@dataclasses.dataclass(frozen=True)
class Prompts:
    source: Annotated[SourceValue, source_argument(SourceValue)]
    tokenizer: str
    max_prompt_len: int = 256
    pad_id: int = 0
    reward_bytes: int | None = None
    long: Literal["drop", "truncate"] = "drop"
    thinking: bool | None = None
    validation: Annotated[Validation, source_argument(Validation)] = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        source = _PromptFacts(_source(self.source, split=self.split), self.reward_bytes)
        step = Configured(ConfigurePrompt(self.tokenizer, self.max_prompt_len, self.pad_id,
                                         self.long, self.thinking), needs=("reward_bytes",))
        prefix = Pipeline(source).map(step).stages
        validation = self.validation
        if validation is not None and not isinstance(validation, (HoldOut, Split, Pipeline)):
            validation = _PromptFacts(_source(validation), self.reward_bytes)
        train, held = _branches(source, validation, split=self.split, prefix=prefix)
        return train, held, ()


@dataclasses.dataclass(frozen=True)
class _PromptFacts:
    source: Source
    width: int | None

    @functools.cached_property
    def _measured(self) -> dict[str, int]:
        return {}

    def describe(self) -> str:
        return self.source.describe()

    def snapshot(self) -> Mapping[str, object]:
        return self.source.snapshot()

    def supports(self, read: Read) -> bool:
        return self.source.supports(read)

    def records(self, read: Read):
        from dew.data.prompts import _ground_truth, _text

        rows = self.source.records(read)
        if read == "memory" and self.width is None and "reward_bytes" not in self._measured:
            self._measured["reward_bytes"] = max(1, max((len(text.encode("utf-8"))
                for index in range(len(rows)) for row in (rows[index],) if isinstance(row, Mapping)
                for text in (_text(row.get("data_source")), _ground_truth(row, "prompt"),
                             _text(row.get("extra_info")))), default=1))
        return rows

    def metadata(self) -> Mapping[str, object]:
        return {**self.source.metadata(), **self._measured,
                **({} if self.width is None else {"reward_bytes": self.width})}

    def lengths(self) -> np.ndarray | None:
        return self.source.lengths()

    def manifest(self) -> ShardManifest:
        return self.source.manifest()


@dataclasses.dataclass(frozen=True)
class Preferences:
    source: Annotated[SourceValue, source_argument(SourceValue)]
    seq_len: int = 256
    pad_id: int = 0
    validation: Annotated[Validation, source_argument(Validation)] = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        prefix = Pipeline(_source(self.source)).map(PreferencePair(self.seq_len, self.pad_id)).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        return train, held, ()


@dataclasses.dataclass(frozen=True)
class Augment:
    crop: bool = False
    flip: bool = True
    jitter: bool = True
    crop_scale: tuple[float, float] = (1.0, 1.0)
    backend: Literal["host", "device"] = "device"

    def __post_init__(self) -> None:
        if self.backend not in ("host", "device"):
            raise ValueError("image augmentation backend must be host or device")
        if len(self.crop_scale) != 2 or not 0 < self.crop_scale[0] <= self.crop_scale[1] <= 1:
            raise ValueError("crop_scale must satisfy 0 < low <= high <= 1")


@dataclasses.dataclass(frozen=True)
class Images:
    source: Annotated[SourceValue, source_argument(SourceValue)]
    image_size: int = 128
    staging_size: int | None = None
    field: str = "image"
    caption: str | None = None
    caption_templates: tuple[str, ...] = ()
    caption_columns: tuple[str, ...] = ("caption", "text")
    classes: tuple[str, ...] = ()
    labels: str | None = None
    augment: Augment | None = dataclasses.field(default_factory=Augment)
    validation: Annotated[Validation, source_argument(Validation)] = None
    split: str = "train"

    def __post_init__(self) -> None:
        if self.image_size < 1 or self.staging_size is not None and self.staging_size < self.image_size:
            raise ValueError("image_size must be positive and staging_size must be at least image_size")

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        from dew.data.sources.tfds import TFDS

        size = self.image_size if self.staging_size is None else self.staging_size
        source = _source(self.source, split=self.split)
        if isinstance(source, TFDS):
            import tensorflow_datasets as tfds

            source = dataclasses.replace(source, decoders={self.field: tfds.decode.SkipDecoding()})
        train, held = _branches(source, self.validation, split=self.split, prefix=())
        augment = self.augment
        classes = self.classes
        if self.labels is not None:
            from dew.data.storage import Store

            classes = tuple(Store(self.labels).read().decode().splitlines())
        transform = ImageRecord(self.image_size, size, self.field, self.caption, self.caption_templates,
                                self.caption_columns, classes, flip=augment is not None and augment.flip,
                                jitter=augment is not None and augment.jitter,
                                crop_scale=(1., 1.) if augment is None else augment.crop_scale,
                                backend="host" if augment is None else augment.backend)
        train = train.map(Configured(ConfigureImage(transform), needs=(), random=True))
        if augment is not None and augment.backend == "device":
            train = train.map_batch(DeviceAugment(self.image_size, augment.crop_scale, augment.flip, augment.jitter))
        if held is not None:
            evaluated = dataclasses.replace(transform, staging_size=self.image_size, backend="host",
                                            flip=False, jitter=False, crop_scale=(1., 1.))
            held = held.map(Configured(ConfigureImage(evaluated), needs=(), random=True))
        return train, held, (EncodeCaptions(),)

@dataclasses.dataclass(frozen=True)
class Video:
    source: Annotated[SourceValue, source_argument(SourceValue)]
    frames: int = 16
    frame_size: int = 256
    audio_padding: int = 3
    audio_model: str = "facebook/wav2vec2-base-960h"
    caption: str = ""
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".webm")
    validation: Annotated[Validation, source_argument(Validation)] = None
    split: str = "train"

    @property
    def audio_seconds(self) -> float:
        from dew.data.sources.av_utils import FPS

        return (self.frames + 2 * self.audio_padding) / FPS

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        from pathlib import PurePosixPath

        from dew.data.sources.files import Files

        source = _source(self.source, split=self.split, format="files")
        if isinstance(source, Files):
            extensions = tuple(extension.lower() if extension.startswith(".") else "." + extension.lower()
                               for extension in self.extensions)
            source = dataclasses.replace(source, url=tuple(path for path in source.paths
                                                          if PurePosixPath(path).suffix.lower() in extensions))
        prefix = Pipeline(source).map(VideoPath(self.caption)).stages
        train, held = _branches(source, self.validation, split=self.split, prefix=prefix)
        audio = Audio(self.audio_model)
        decoder = DecodeClip(self.frames, self.frame_size, self.audio_padding, audio.processor.sampling_rate)
        train = train.random_map(decoder).map(audio)
        held = None if held is None else held.random_map(decoder).map(audio)
        return train, held, (EncodeCaptions(),)


def load(
    recipe: Recipe | str,
    *,
    batch: int | None = None,
    tokens: int | None = None,
    lengths: tuple[int, ...] = (),
    read: Read = "indexed",
    provision: bool = False,
    cache: Cache | Generation | str | None = None,
    validation_records: int | None = None,
    objective: ObjectiveData | None = None,
    seed: int = 0,
    loading: Loading = _DEFAULT_LOADING,
    host: tuple[HostStep, ...] = (),
    **fields,
) -> Dataset:
    """Resolve a recipe alias and compile its training and evaluation pipelines."""
    if (batch is None) == (tokens is None):
        raise ValueError("load needs exactly one of batch rows or tokens")
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
    if held is not None and validation_records is not None:
        held = held.take(validation_records)
    if cache is not None:
        train = _cached(train, cache)
        held = None if held is None else _cached(held, cache)
    train = _batched(_ordered(train, read, seed), batch, train=True, tokens=tokens, lengths=lengths)
    held = None if held is None else _batched(held, batch, train=False, tokens=tokens, lengths=lengths)
    return compile(
        train,
        validation=held,
        host=_host_stages(shared, host),
        objective=objective,
        read=read,
        provision=provision,
        loading=loading,
        seed=seed,
    )


def _ordered(pipeline: Pipeline, read: Read, seed: int) -> Pipeline:
    if isinstance(pipeline.source, Phases):
        children = tuple(_ordered(child, read, seed) for child in pipeline.source.pipelines)
        return dataclasses.replace(pipeline, source=dataclasses.replace(pipeline.source, pipelines=children))
    if isinstance(pipeline.source, Mixture):
        children = tuple(_ordered(child, read, seed) for child in pipeline.source.pipelines)
        return dataclasses.replace(pipeline, source=dataclasses.replace(pipeline.source, pipelines=children))
    if read == "memory":
        at = next(
            (index for index, stage in enumerate(pipeline.stages)
             if isinstance(stage, RandomMap) or isinstance(stage, Configured) and stage.random),
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


def _batched(pipeline: Pipeline, batch: int | None, *, train: bool, tokens: int | None = None,
             lengths: tuple[int, ...] = ()) -> Pipeline:
    from dew.data.pipeline import BatchStage, MapBatch

    stated = [stage.size for stage in pipeline.stages if isinstance(stage, BatchStage)]
    if stated:
        if stated != [batch]:
            raise ValueError(f"pipeline batches {stated}, but load requested batch={batch}")
        return pipeline

    at = next((index for index, stage in enumerate(pipeline.stages)
               if isinstance(stage, MapBatch)), len(pipeline.stages))
    prefix = dataclasses.replace(pipeline, stages=pipeline.stages[:at]).batch(
        batch, tokens=tokens, lengths=lengths, drop_remainder=train)
    return dataclasses.replace(prefix, stages=(*prefix.stages, *pipeline.stages[at:]))


def _cached(pipeline: Pipeline, cache: Cache | Generation | str) -> Pipeline:
    from dew.data.pipeline import MapBatch, Pack, Windows

    if isinstance(pipeline.source, (Mixture, Phases)):
        children = tuple(_cached(child, cache) for child in pipeline.source.pipelines)
        return dataclasses.replace(pipeline, source=dataclasses.replace(pipeline.source, pipelines=children))
    at = next((index for index, stage in enumerate(pipeline.stages)
               if isinstance(stage, (RandomMap, MapBatch, Pack, Windows))
               or isinstance(stage, Configured) and stage.random), len(pipeline.stages))
    prefix = dataclasses.replace(pipeline, stages=pipeline.stages[:at]).cache(cache)
    return dataclasses.replace(prefix, stages=(*prefix.stages, *pipeline.stages[at:]))
