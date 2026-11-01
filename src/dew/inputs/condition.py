"""What every condition encoder is (`ConditionEncoder`) and the work it does (`ConditionWork`), apart
from the encoders themselves (`dew.inputs.encoders`), so a spec of a model's inputs, a protocol or a
model reads the contract without the towers.

`ConditionEncoder.bound(record)` rebuilds an encoder from a run's record, the
registry's `{"class": ..., "fields": ...}` whose fields `recorded_fields` wrote.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, Self

import numpy as np
from typing_extensions import TypeVar

from dew import _records
from dew.batch import Variables
from dew.nn.flops import counted_flops
from dew.nn.protocols import Counted
from dew.registry import component, to_record

Raw = TypeVar("Raw")
"""One item of a modality's raw data: a prompt, a waveform."""

Encoded = TypeVar("Encoded", default=object)
"""The conditioning value a modality's encoder produces: a `TextContext`, a
`DenoisingCondition`, an array. An encoder that leaves it unstated promises
its callers nothing about the value beyond what it hands the model."""


@dataclass(frozen=True)
class ConditionWork:
    """A condition encoder's forward FLOPs and the sequence its denoiser reads.

    The output lengths may differ from encoded input (system prompts
    cropped, CLIP and T5 concatenated, or visible zero tokens retained).
    Keeping both under the encoder makes its geometry the encode contract.
    """

    flops: float
    lengths: np.ndarray
    spans: np.ndarray
    features: int
    pooled_features: int | None = None
    time_ids: int | None = None

    @classmethod
    def of_tokens(cls, tower: Counted, tokens: Mapping[str, np.ndarray], features: int) -> Self:
        """The work of a `tower` that reads each row's unmasked `tokens` and hands its denoiser the
        whole fixed-length sequence, `features` wide: its FLOPs over the real tokens, their lengths,
        and spans of the token budget."""
        lengths = np.sum(np.asarray(tokens["attention_mask"]) != 0, axis=-1, dtype=np.int64)
        spans = np.full(lengths.shape, np.shape(tokens["input_ids"])[1], np.int64)
        return cls(counted_flops(tower, lengths), lengths, spans, features)


class ConditionEncoder(ABC, Generic[Raw, Encoded]):
    """Turns one modality's raw data into a conditioning value.

    Subclasses implement `from_pretrained`, `tokenize`, `encode` and
    `recorded_fields`, the record a run keeps of them (`dew.registry.Recorded`).
    """

    if TYPE_CHECKING:
        # Each encoder is a dataclass, whose fields hold its towers (`dew.nn.kernels.options.quantizations`).
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
