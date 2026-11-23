"""What every condition encoder is (`ConditionEncoder`), apart from the encoders themselves
(`dew.inputs.encoders`), so a spec of a model's inputs reads the contract without the towers. The work an
encoder does, which a model reads, is the model's own type (`dew.nn.flops.ConditionWork`).

`ConditionEncoder.bound(record)` rebuilds an encoder from a run's record, the
registry's `{"class": ..., "fields": ...}` whose fields `recorded_fields` wrote.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Generic, Self

import jax
import numpy as np
from flax import linen as nn
from typing_extensions import TypeVar

from dew import _records
from dew.batch import Variables
from dew.nn.progress import END, Progress
from dew.nn.sharding import boxed_axes
from dew.registry import component, to_record

Raw = TypeVar("Raw")
"""One item of a modality's raw data: a prompt, a waveform."""

Encoded = TypeVar("Encoded", default=object, covariant=True)
"""The conditioning value a modality's encoder produces: a `TextContext`, a
`DenoisingCondition`, an array. An encoder that leaves it unstated promises
its callers nothing about the value beyond what it hands the model. An encoder
only produces it (`encode`), so one of a narrower value serves where any
encoder is asked for (covariant)."""


class ConditionEncoder(ABC, Generic[Raw, Encoded]):
    """Turns one modality's raw data into a conditioning value.

    Subclasses implement `from_pretrained`, `tokenize`, `encode` and
    `recorded_fields`, the record a run keeps of them (`dew.registry.Recorded`).
    """

    if TYPE_CHECKING:
        # Each encoder is a dataclass, which a run rebuilds with new fields (`dataclasses.replace`).
        __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]

    params: Variables
    parameter_collections: ClassVar[tuple[str, ...] | None] = None
    """The collections that hold learned weights, frozen ones included, or None
    for a bare parameter tree. Collections not named here keep their dtype."""
    reads_captions: ClassVar[bool] = True
    """Whether a dataset's captions are this encoder's raw data. An encoder
    that does not read captions reads a batch field its dataset writes
    itself, such as audio, and `InputSpec.tokenize` leaves that field alone."""
    keyword: ClassVar[str] = "textcontext"
    """The model keyword this encoder's value is passed under. Each value
    type goes to the models written for it: a `TextContext` to the native
    architectures' `textcontext`, and a `DenoisingCondition` to the
    published families' `conditioning`."""

    def parameter_axes(self) -> dict[tuple[str, ...], tuple[str | None, ...]]:
        """The axes each parameter of `params` is drawn in, by its path in `params`, read off its towers'
        draws as `params` lays them out (`drawn`, `dew.nn.sharding.boxed_axes`): the table that places a
        run holding this encoder frozen (`dew.objectives.diffusion.DiffusionObjective.parameter_axes`) and
        a pipeline loaded on a mesh, each path the one its parameter is held at, so no tower's declaration
        is missed by a path its draw lays out otherwise (row 4407). An encoder whose towers name no axes
        holds none."""
        return boxed_axes(self.drawn())

    def drawn(self) -> Mapping:
        """Its towers' parameters as their draws make them, traced for shapes alone (`jax.eval_shape`, no
        array made), laid out as `params` holds them: the shapes a pipeline loaded without them draws its
        denoiser over (`dew.interop.pretrained.pipeline_axes`). None drawn for an encoder that draws
        none."""
        return {}

    @classmethod
    @abstractmethod
    def from_pretrained(cls, checkpoint: str, *, params: Variables | None = None) -> Self:
        """Load the tower named `checkpoint`. This is the one call that opens files.

        Anything else a checkpoint needs is a keyword field with a default;
        `recorded_fields` records those fields, and `bound` rebuilds the
        encoder from them. When you pass `params`, the load uses them as they are. It
        reads the checkpoint's metadata but none of its weights, and keeps
        the values, dtypes and placement of `params`.
        """

    @abstractmethod
    def tokenize(self, texts: Sequence[Raw]) -> Mapping[str, np.ndarray]:
        """Turn raw data into the host arrays `encode` reads, one row per item."""

    @abstractmethod
    def encode(self, params: Variables, tokens) -> Encoded:
        """Encode tokens into the conditioning value on device, under `params`."""

    def captions(self, tokens) -> tuple[str, ...]:
        """Return the text the tokens stand for, to show in a rendered artifact.

        A modality that is not text returns an empty tuple.
        """
        return ()

    @abstractmethod
    def recorded_fields(self) -> Mapping[str, object]:
        """Return the keyword fields that `from_pretrained` rebuilds this encoder from, its checkpoint
        among them: the record a run keeps (`dew.registry.Recorded`)."""

    def token_counts(self, texts: Sequence[Raw]) -> np.ndarray:
        """How many tokens each of `texts` is before `tokenize` cuts it to the encoder's width, which a
        reader that keeps a text's first tokens reports as a cut (`dew.decision.layout.with_context`): the
        tokens `tokenize` keeps, for an encoder that cuts none; one that cuts states its own."""
        return np.sum(np.asarray(self.tokenize(texts)["attention_mask"]) != 0, axis=-1)

    def dependencies(self) -> Mapping[str, object]:
        """What `tokenize` makes of raw data depends on: the encoder and the fields it is
        rebuilt from, which hold its context length and its checkpoint's revision."""
        return _records.record(to_record(self, ConditionEncoder), "encoder")

    @staticmethod
    def bound(record: Mapping[str, object], *, params: Variables | None = None) -> ConditionEncoder:
        """Rebuild the encoder a record names, `{"class": alias or import path, "fields": ...}`
        (`dew.registry.component`), around `params` where given.

        The fields are the keywords `recorded_fields` wrote, scalars and dtype names its
        `from_pretrained` reads itself, passed back to it, so each encoder keeps its own concrete
        signature. `checkpoint` is the one field every encoder takes; a record without one raises
        ValueError. With `params` the load reads the checkpoint's metadata and tokenizer, never its
        weights (`from_pretrained`).
        """
        encoder, fields = component(ConditionEncoder, record)
        checkpoint = fields.get("checkpoint")
        if not isinstance(checkpoint, str):
            raise ValueError(f"the {record['class']} record names no checkpoint to rebuild from")
        rest = {name: value for name, value in fields.items() if name != "checkpoint"}
        return encoder.from_pretrained(checkpoint, **rest, params=params)


@dataclasses.dataclass(frozen=True)
class BackboneCall:
    """The live model an encoder that reads the backbone (`BackboneEncoder`) encodes with, as the model's
    own pass holds it: its module and variables, an adapter's included, whether it trains, its draws and
    the run's progress (`dew.nn.progress`)."""

    module: nn.Module
    variables: Variables
    train: bool = False
    rngs: dict[str, jax.Array] | None = None
    progress: Progress = END
    loop_passes: jax.Array | None = None
    """The count of its loop's passes the call runs, where given (`dew.batch.LOOP_PASSES`); every pass where
    None."""


class BackboneEncoder(ConditionEncoder[Raw, Encoded]):
    """An encoder that reads the backbone: the model it conditions is its tower, so it holds no weights and
    no checkpoint of its own, and a dispatch (`dew.inputs.spec.encode_conditions`) hands it the live model
    (`encode_backbone`), as it hands no other encoder. Its tokens are those its model's data routes to its
    stream, which the data writes; a record rebuilds it from its fields alone (`ConditionEncoder.bound`)."""

    reads_captions: ClassVar[bool] = False

    @classmethod
    def from_pretrained(cls, checkpoint: str, *, params: Variables | None = None) -> Self:
        raise TypeError(f"{cls.__name__} encodes with the model it conditions, which holds its weights; it "
                        "loads no checkpoint")

    def tokenize(self, texts: Sequence[Raw]) -> Mapping[str, np.ndarray]:
        raise TypeError(f"{type(self).__name__} reads the tokens its model's data routes to its stream")

    def encode(self, params: Variables, tokens) -> Encoded:
        raise TypeError(f"{type(self).__name__} encodes with the live model (encode_backbone), which its "
                        "dispatch hands it")

    @abstractmethod
    def encode_backbone(self, backbone: BackboneCall, tokens) -> Encoded:
        """Encode `tokens` with the live model `backbone`, on device."""
