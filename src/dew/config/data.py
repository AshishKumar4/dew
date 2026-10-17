"""Typed recipe values that build record pipelines."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, runtime_checkable

import numpy as np

from dew.data.analysis import _cache_at, _order_at, _preferred_read, _promoted, _until
from dew.data.compile import compile
from dew.data.dataset import Batch, DataPhase, Loader, Loading, record_argument
from dew.data.holdout import (
    HoldOut,
    SourceValue,
    Split,
    Validation,
    _branches,
    _named_split,
    _source,
    source_argument,
)
from dew.data.load import RECORD_HOLDOUT, TEXT_HOLDOUT
from dew.data.pipeline import Configured, HostStep, MapBatch, Mixture, ObjectiveData, Phases, Pipeline, mix

if TYPE_CHECKING:
    from dew.data.cache import Cache
    from dew.data.generation import Generation
from dew.data import audio, images, video
from dew.data.sources.base import Read, ShardManifest, Source, _consumed
from dew.data.steps.chat import ConfigurePrompt, PreferencePair, RenderChat, ResponseCanvases
from dew.data.steps.generic import Formatting
from dew.data.steps.image import ConfigureCaption, EncodeCaptions
from dew.data.steps.text import Chunks, LengthPolicy, Tokenize

_DEFAULT_LOADING = Loading(workers="auto")


@runtime_checkable
class RunData(Protocol):
    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]: ...


class _Loadable(RunData):
    def load(self, **options) -> Loader:
        """Compile this run's data with `loader(self, **options)`."""
        return loader(self, **options)


@runtime_checkable
class _HostBinding(Protocol):
    binding: str

    def __call__(self, batch: Batch) -> Batch: ...


def _host_stages(defaults: tuple[HostStep, ...], explicit: tuple[HostStep, ...]) -> tuple[HostStep, ...]:
    bindings = {stage.binding: stage for stage in explicit if isinstance(stage, _HostBinding)}
    configured = tuple(
        bindings.pop(stage.binding, stage) if isinstance(stage, _HostBinding) else stage for stage in defaults
    )
    return (
        *configured,
        *(stage for stage in explicit if not isinstance(stage, _HostBinding)),
        *bindings.values(),
    )


@dataclasses.dataclass(frozen=True)
class RecordsData(_Loadable):
    source: Annotated[SourceValue, source_argument(SourceValue)]
    validation: Annotated[Validation, source_argument(Validation)] = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        prefix = Pipeline(_source(self.source)).map(Formatting()).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        return train, held, ()


@dataclasses.dataclass(frozen=True)
class TextData(_Loadable):
    source: Annotated[SourceValue | Mapping[str, float], source_argument(SourceValue)] = ""
    tokenizer: str = "byte"
    seq_len: int = 256
    stride: int | None = None
    field: str = "text"
    eos: bool = True
    pack: Literal["first-fit", "best-fit", "concat"] | None = None
    packing_bins: int = 8
    validation: Annotated[Validation, source_argument(Validation)] = None
    split: str = "train"
    phases: Annotated[tuple[DataPhase, ...], record_argument(tuple[DataPhase, ...])] = ()
    token_holdout: HoldOut | None = TEXT_HOLDOUT

    def __post_init__(self) -> None:
        if self.seq_len < 1 or (self.stride is not None and self.stride < 1):
            raise ValueError("text seq_len and stride must be positive")
        if self.packing_bins < 1:
            raise ValueError("packing_bins must be positive")

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        from dew.data.sources.tokens import TokenFiles

        if self.phases:
            if self.source:
                raise ValueError("TextData takes source or phases, not both")
            branches = [
                dataclasses.replace(self, source=phase.path, phases=()).pipelines() for phase in self.phases
            ]
            return (
                Pipeline(
                    Phases(
                        tuple(branch[0] for branch in branches),
                        tuple(phase.until_step for phase in self.phases),
                    )
                ),
                branches[0][1],
                (),
            )
        if isinstance(self.source, Mapping) and all(
            isinstance(weight, (int, float)) for weight in self.source.values()
        ):
            # Each corpus with its weight, so a corpus without a held split leaves the held mixture
            # with its weight.
            weights = {path: float(weight) for path, weight in self.source.items()
                       if isinstance(weight, (int, float))}
            pairs = [(dataclasses.replace(self, source=path).pipelines(), weight)
                     for path, weight in sorted(weights.items())]
            if not pairs:
                raise ValueError("TextData mixture has no corpora")
            trains = [(branches[0], weight) for branches, weight in pairs]
            scored = [(branches[1], weight) for branches, weight in pairs if branches[1] is not None]

            def mixed(weighted, **options):
                pipelines, shares = zip(*weighted, strict=True)
                return pipelines[0] if len(pipelines) == 1 else mix(pipelines, shares, **options)

            return mixed(trains, by="tokens"), mixed(scored) if scored else None, ()
        stored = _source(self.source, split=self.split)
        if isinstance(stored, TokenFiles):
            return self._token_pipelines(stored)
        prefix = Pipeline(stored).map(Tokenize(self.tokenizer, self.field, self.eos)).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        if self.token_holdout is not None and self.validation is None:
            if self.token_holdout.fraction is None:
                raise ValueError("token_holdout needs fraction=")
            held = train.token_split(self.token_holdout.fraction, held=True)
            train = train.token_split(self.token_holdout.fraction, held=False)
        if self.pack is None:
            train = train.windows(self.seq_len, stride=self.stride)
            held = (
                None
                if held is None
                else held
                if isinstance(self.validation, Pipeline)
                else held.windows(self.seq_len)
            )
        else:
            train = train.flat_map(Chunks(self.seq_len + 1)).pack(
                self.seq_len + 1, bins=self.packing_bins, policy=self.pack
            )
            held = (
                self.validation
                if isinstance(self.validation, Pipeline)
                else (
                    None
                    if held is None
                    else held.flat_map(Chunks(self.seq_len + 1)).pack(
                        self.seq_len + 1, bins=self.packing_bins, policy=self.pack
                    )
                )
            )
        return train, held, ()

    def _token_pipelines(self, stored) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        from dew.data.steps.windows import WindowSource

        held = self.validation
        if isinstance(held, Split):
            held = _named_split(stored, held.name)
        elif isinstance(held, HoldOut):
            raise ValueError(
                "token-store validation uses its named split; slice a windows Pipeline for HoldOut"
            )
        if held is None and stored.holds("val"):
            held = _named_split(stored, "val")
        sources = (stored, None if held is None or isinstance(held, Pipeline) else _source(held))

        def windows(source, stride):
            if self.pack is None:
                return Pipeline(WindowSource(source, source.tokens, self.seq_len, stride))
            if not source.documented:
                raise ValueError(
                    f"packed text needs producer document boundaries or EOS: {source.describe()}"
                )
            # A document longer than a window is cut into window-long pieces, which the
            # source's plan of document lengths cuts alike (`PackedSource.planned`).
            return Pipeline(source).flat_map(Chunks(self.seq_len + 1)).pack(
                self.seq_len + 1, bins=self.packing_bins, policy=self.pack)

        return (
            windows(sources[0], self.seq_len if self.stride is None else self.stride),
            held
            if isinstance(held, Pipeline)
            else None
            if sources[1] is None
            else windows(sources[1], self.seq_len),
            (),
        )


@dataclasses.dataclass(frozen=True)
class ChatData(_Loadable):
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
                pipeline = pipeline.filter_map(
                    LengthPolicy(self.seq_len + 1, long=self.long, aligned=("text_roles",))
                )
            return pipeline.pack(self.seq_len + 1, fields=("text", "text_roles"), bins=self.packing_bins)

        return (
            policy(train),
            None if held is None else held if isinstance(self.validation, Pipeline) else policy(held),
            (),
        )


@dataclasses.dataclass(frozen=True)
class ChatCanvasData(_Loadable):
    """Conversations rendered as `ChatData` renders them, each assistant turn
    then one `BlockDiffusionObjective` row (`ResponseCanvases`), never packed:
    the objective splits every row at `prompt_length`, so the turn alone
    fills the canvases it scores."""

    source: Annotated[SourceValue, source_argument(SourceValue)]
    tokenizer: str
    prompt_length: int
    canvas_size: int
    canvases: int = 1
    pad_id: int = 0
    column: str = "messages"
    validation: Annotated[Validation, source_argument(Validation)] = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        # The canvases' eos is the rendering's own tokenizer's, one pinned handle for both.
        renderer = RenderChat(self.tokenizer, self.column)
        eos = renderer.vocabulary.eos_id
        if eos is None:
            raise ValueError(f"tokenizer {self.tokenizer!r} declares no eos token to end a response's canvas")
        layout = ResponseCanvases(self.prompt_length, self.canvas_size, self.canvases, eos, self.pad_id)
        rendered = Pipeline(_source(self.source)).map(renderer)
        prefix = rendered.flat_map(layout).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        return train, held, ()


@dataclasses.dataclass(frozen=True)
class PromptData(_Loadable):
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
        step = Configured(
            ConfigurePrompt(self.tokenizer, self.max_prompt_len, self.pad_id, self.long, self.thinking),
            needs=("reward_bytes",),
        )
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

    def with_split(self, name: str) -> _PromptFacts:
        """These facts over the source's own split `name`, which a source without splits refuses."""
        return dataclasses.replace(self, source=_named_split(self.source, name))

    def consumed(self, read: Read) -> _PromptFacts:
        consumed = dataclasses.replace(self, source=_consumed(self.source, read))
        consumed._measured.update(self._measured)
        return consumed

    @functools.cached_property
    def _measured(self) -> dict[str, int]:
        return {}

    def describe(self) -> str:
        return self.source.describe()

    def fingerprint(self) -> Mapping[str, object]:
        return self.source.fingerprint()

    def supports(self, read: Read) -> bool:
        return self.source.supports(read)

    def records(self, read: Read):
        from dew.data.prompts import _ground_truth, _text

        rows = self.source.records(read)
        if read == "memory" and self.width is None and "reward_bytes" not in self._measured:
            self._measured["reward_bytes"] = max(
                1,
                max(
                    (
                        len(text.encode("utf-8"))
                        for index in range(len(rows))
                        for row in (rows[index],)
                        if isinstance(row, Mapping)
                        for text in (
                            _text(row.get("data_source")),
                            _ground_truth(row, "prompt"),
                            _text(row.get("extra_info")),
                        )
                    ),
                    default=1,
                ),
            )
        return rows

    def metadata(self) -> Mapping[str, object]:
        return {
            **self.source.metadata(),
            **self._measured,
            **({} if self.width is None else {"reward_bytes": self.width}),
        }

    def lengths(self) -> np.ndarray | None:
        return self.source.lengths()

    def manifest(self) -> ShardManifest:
        return self.source.manifest()


@dataclasses.dataclass(frozen=True)
class PreferenceData(_Loadable):
    source: Annotated[SourceValue, source_argument(SourceValue)]
    seq_len: int = 256
    pad_id: int = 0
    validation: Annotated[Validation, source_argument(Validation)] = None
    split: str = "train"

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        prefix = Pipeline(_source(self.source)).map(PreferencePair(self.seq_len, self.pad_id)).stages
        train, held = _branches(self.source, self.validation, split=self.split, prefix=prefix)
        return train, held, ()


TINY_SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclasses.dataclass(frozen=True)
class TinyShakespeareData(TextData):
    """Karpathy's tiny Shakespeare corpus as a byte-level `TextData` recipe, one
    line a document, the corpus the language-model examples train on."""

    source: Annotated[SourceValue | Mapping[str, float], source_argument(SourceValue)] = TINY_SHAKESPEARE

@dataclasses.dataclass(frozen=True)
class Augment:
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
class ImageData(_Loadable):
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
    validation: Annotated[Validation, source_argument(Validation)] = RECORD_HOLDOUT
    split: str = "train"

    def __post_init__(self) -> None:
        if self.image_size < 1 or (self.staging_size is not None and self.staging_size < self.image_size):
            raise ValueError("image_size must be positive and staging_size must be at least image_size")

    def pipelines(self) -> tuple[Pipeline, Pipeline | None, tuple[HostStep, ...]]:
        from dew.data.sources.tfds import TFDS

        # No augmentation stages nothing larger than the output: the record resizes to it directly.
        size = self.image_size if self.staging_size is None or self.augment is None else self.staging_size
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
        caption = images.caption(self.caption, templates=self.caption_templates,
                                 classes=classes, columns=self.caption_columns)
        configured = Configured(ConfigureCaption(caption), needs=(), random=bool(self.caption_templates))

        def prepared(chain: Pipeline, staging: int) -> Pipeline:
            return (chain.map(images.decode(self.field, at_least=staging, rejects=False))
                    .map(images.resize(staging)).map(configured))

        train = prepared(train, size)
        if augment is not None and augment.backend == "host":
            if augment.crop_scale != (1.0, 1.0) or size != self.image_size:
                train = train.random_map(images.random_crop(self.image_size, scale=augment.crop_scale,
                                                            flip=augment.flip, jitter=augment.jitter))
            else:
                if augment.flip:
                    train = train.random_map(images.flip())
                if augment.jitter:
                    train = train.random_map(images.color_jitter())
        elif augment is not None:
            train = train.random_map(images.augmentation_key()).then(MapBatch(images.device_augment(
                self.image_size, crop_scale=augment.crop_scale, flip=augment.flip, jitter=augment.jitter)))
        held = None if held is None else prepared(held, self.image_size)
        return train, held, (EncodeCaptions(),)


@dataclasses.dataclass(frozen=True)
class VideoData(_Loadable):
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

        from dew.data.processors import AutoAudioProcessor
        from dew.data.sources.av_utils import FPS
        from dew.data.sources.files import Files

        source = _source(self.source, split=self.split, format="files")
        if isinstance(source, Files):
            extensions = tuple(
                extension.lower() if extension.startswith(".") else "." + extension.lower()
                for extension in self.extensions
            )
            source = dataclasses.replace(
                source,
                url=tuple(path for path in source.paths if PurePosixPath(path).suffix.lower() in extensions),
            )
        prefix = Pipeline(source).map(_VideoRow(self.caption)).stages
        train, held = _branches(source, self.validation, split=self.split, prefix=prefix)
        # The sound at the rate the audio model reads, `audio_padding` frames either side of the frames.
        rate = AutoAudioProcessor(tensor_type="np", modelname=self.audio_model).sampling_rate
        padding = self.audio_padding / FPS

        def clips(chain: Pipeline) -> Pipeline:
            # A listed clip too short or that does not decode is an error, as it always was here, so the
            # run knows its count.
            pick = dataclasses.replace(video.pick_clip(self.frames / FPS, margin=padding), rejects=False)
            frames = dataclasses.replace(video.decode(self.frames, fps=FPS, size=self.frame_size),
                                         rejects=False)
            sound = dataclasses.replace(audio.decode(rate, context=padding), rejects=False)
            return chain.random_map(pick).map(frames, sound).remove_columns("mp4", "clip")

        return clips(train), None if held is None else clips(held), (EncodeCaptions(),)


@dataclasses.dataclass(frozen=True)
class _VideoRow:
    """A listed clip's path as the decoder reads it, and its caption."""

    caption: str = ""

    def __call__(self, element: Batch) -> Batch:
        from dew.records import text

        path = text(element.get("video_path", element.get("path")), "video_path")
        return {"mp4": path, "caption": element.get("caption", self.caption)}


def loader(
    recipe: RunData | str,
    *,
    batch: int | None = None,
    tokens: int | None = None,
    read: Read | None = "indexed",
    provision: bool = False,
    cache: Cache | Generation | str | None = None,
    validation_records: int | None = None,
    objective: ObjectiveData | None = None,
    seed: int = 0,
    loading: Loading = _DEFAULT_LOADING,
    host: tuple[HostStep, ...] = (),
    **fields,
) -> Loader:
    """Resolve a recipe alias and compile its training and evaluation pipelines.
    With no `read`, the recipe's sources choose it, as `compile` does."""
    if (batch is None) == (tokens is None):
        raise ValueError("load needs exactly one of batch rows or tokens")
    if isinstance(recipe, str):
        if "://" in recipe or "/" in recipe:
            raise ValueError("load needs a recipe or alias; a bare URI names a source")
        from dew.registry import datasets

        recipe = datasets.build(recipe, **fields)
    elif fields:
        if not dataclasses.is_dataclass(recipe):
            raise TypeError("recipe field overrides require a frozen dataclass")
        # The recipe's own constructor refuses a field it does not declare.
        recipe = dataclasses.replace(recipe, **fields)
    if not isinstance(recipe, RunData):
        raise TypeError("load expects a typed recipe with pipelines()")
    train, held, shared = recipe.pipelines()
    if held is not None and validation_records is not None:
        held = held.take(validation_records)
    if cache is not None:
        train = _cached(train, cache)
        held = None if held is None else _cached(held, cache)
    train, held = _promoted(train), None if held is None else _promoted(held)

    def compiled(at: Read) -> tuple[Pipeline, ...]:
        ordered = _batched(_ordered(train, at, seed), batch, train=True, tokens=tokens)
        if held is None:
            return (ordered,)
        return ordered, _batched(held, batch, train=False, tokens=tokens)

    read = read or _preferred_read(compiled, provision=provision)
    branches = compiled(read)
    return compile(
        branches[0],
        validation=branches[1] if len(branches) > 1 else None,
        host=_host_stages(shared, host),
        objective=objective,
        read=read,
        provision=provision,
        loading=loading,
        seed=seed,
    )


def _ordered(pipeline: Pipeline, read: Read, seed: int) -> Pipeline:
    if isinstance(pipeline.source, (Mixture, Phases)):
        children = tuple(_ordered(child, read, seed) for child in pipeline.source.pipelines)
        return dataclasses.replace(pipeline, source=dataclasses.replace(pipeline.source, pipelines=children))
    at, indexed = _order_at(pipeline.stages, read)
    order = Pipeline(pipeline.source).shuffle(seed, window=None if indexed else 64).repeat().stages
    return dataclasses.replace(pipeline, stages=(*pipeline.stages[:at], *order, *pipeline.stages[at:]))


def _batched(
    pipeline: Pipeline,
    batch: int | None,
    *,
    train: bool,
    tokens: int | None = None,
) -> Pipeline:
    from dew.data.pipeline import BatchStage, MapBatch, TokenBudget

    stated = [(stage.size, None) if isinstance(stage, BatchStage) else (None, stage.tokens)
              for stage in pipeline.stages if isinstance(stage, (BatchStage, TokenBudget))]
    if stated:
        if stated != [(batch, tokens)]:
            raise ValueError(f"pipeline batches {stated} (rows, tokens), but load requested batch={batch}, "
                             f"tokens={tokens}")
        return pipeline

    at = _until(pipeline.stages, MapBatch)
    prefix = dataclasses.replace(pipeline, stages=pipeline.stages[:at]).batch(
        batch, tokens=tokens, drop_remainder=train
    )
    return dataclasses.replace(prefix, stages=(*prefix.stages, *pipeline.stages[at:]))


def _cached(pipeline: Pipeline, cache: Cache | Generation | str) -> Pipeline:
    if isinstance(pipeline.source, (Mixture, Phases)):
        children = tuple(_cached(child, cache) for child in pipeline.source.pipelines)
        return dataclasses.replace(pipeline, source=dataclasses.replace(pipeline.source, pipelines=children))
    at = _cache_at(pipeline.stages)
    prefix = dataclasses.replace(pipeline, stages=pipeline.stages[:at]).cache(cache)
    return dataclasses.replace(prefix, stages=(*prefix.stages, *pipeline.stages[at:]))
