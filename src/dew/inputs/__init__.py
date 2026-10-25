"""What a generative objective is fed: the sample field and its conditions.

`InputSpec` names the batch field the model learns to generate, and the
conditions it is given, keyed by the model's own keyword arguments. A
`Condition` holds an encoder, the batch field it reads and the raw datum that
stands for "no condition", which classifier-free guidance and conditioning
dropout substitute. The spec only describes the inputs; `encode_conditions`
runs its encoders over their weights, the one encoding a training step and a
sampling task share.

Image and video batches arrive as uint8 pixels in [0, 255], as the data
workers write them. `unit_range` is the one conversion to [-1, 1], the range
every diffusion loss, sample, artifact and image metric uses, and
`dew.artifacts.uint8_pixels` is the one conversion back.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from dew import registry
from dew.nn.inputs import ModelInputs
from dew.nn.precision import at_least_fp32
from dew.nn.text_encoders import DEFAULT_MODEL
from dew.objectives.base import Batch, Variables
from dew.registry import DtypeName

from .diffusion import DiffusionConditioner
from .encoders import CharTable, CLIPText, ConditionEncoder, HFAudio, T5Text, rebuild


def unit_range(pixels: jax.typing.ArrayLike) -> jax.Array:
    """Convert pixels in [0, 255] to [-1, 1].

    uint8 and float32 pixels give float32, and a float64 run's float64 pixels
    stay float64.
    """
    pixels = jnp.asarray(pixels)
    return (pixels.astype(at_least_fp32(pixels.dtype)) - 127.5) / 127.5


@dataclass(frozen=True)
class Field:
    """A batch field and its per-example shape, such as `Field("image", (128, 128, 3))`."""

    key: str
    shape: tuple[int, ...]
    variable_axes: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(size) for size in self.shape))
        if any(axis < 0 or axis >= len(self.shape) for axis in self.variable_axes):
            raise ValueError("variable field axes must index its per-sample shape")


@dataclass(frozen=True)
class Condition:
    """Names one conditioning input: its encoder, the batch field holding its
    tokens, and the raw datum for the unconditional branch."""

    encoder: ConditionEncoder
    field: str = "text"
    unconditional: str | float | Mapping[str, object] = ""

    def to_json(self) -> dict:
        return {
            "encoder": {"class": registry.import_path(type(self.encoder)), "fields": self.encoder.to_json()},
            "field": self.field,
            "unconditional": self.unconditional,
        }

    @classmethod
    def from_json(cls, record: Mapping, *, params: Variables | None = None) -> Condition:
        encoder = record["encoder"]
        return cls(
            encoder=rebuild(encoder["class"], encoder["fields"], params=params),
            field=record["field"],
            unconditional=record["unconditional"],
        )


@dataclass(frozen=True)
class InputSpec:
    """Names the sample field and the conditions, keyed by the model keyword each is passed under.

    For example, `conditions={"textcontext": Condition(...)}`. Two conditions
    cannot share a batch field.

    A captioning dataset passes its text to `tokenize`. Every condition that
    reads captions tokenizes them under its own field, so the encoder a run
    names decides the ids and the context length, and the dataset only
    carries the words. A condition on another modality reads the field its
    dataset writes; audio conditioning, for example, reads a clip's `audio`.
    """

    sample: Field
    conditions: Mapping[str, Condition] = field(default_factory=dict)
    mask: Field | None = None
    """A binary image mask for explicit masked-image latent conditioning."""

    def __post_init__(self):
        fields = [condition.field for condition in self.conditions.values()]
        if len(set(fields)) != len(fields):
            raise ValueError(
                f"two conditions share the batch field {sorted(set(fields))}: "
                "the objective reads batch[condition.field] under each keyword, "
                "so the second tokenization would overwrite the first. "
                "Name a field per condition"
            )

    def tokenize(self, captions: Sequence[str]) -> dict[str, Mapping[str, np.ndarray]]:
        """Return the batch fields this run's caption conditions read, tokenized from `captions`.

        The result is empty for a run with no caption conditions, so the
        captions stop at the loader and no string array reaches a device.
        """
        return {
            condition.field: condition.encoder.tokenize(captions)
            for condition in self.conditions.values()
            if condition.encoder.reads_captions
        }

    @property
    def encoders(self) -> tuple[tuple[str, ConditionEncoder], ...]:
        """Each condition's encoder by the keyword it is passed under, a hashable value a compiled
        encoding takes as a static argument (`encode_conditions`)."""
        return tuple((keyword, condition.encoder) for keyword, condition in self.conditions.items())

    def dependencies(self) -> Mapping[str, object]:
        """What `tokenize` and `encode_fields` make of a batch depends on: each condition's field
        and its encoder's own (`ConditionEncoder.dependencies`)."""
        return {keyword: {"field": condition.field, "encoder": condition.encoder.dependencies()}
                for keyword, condition in self.conditions.items()}

    def encode_fields(self, batch: Batch) -> dict[str, Mapping[str, np.ndarray]]:
        """Return the batch fields this run's other conditions read, each encoded from the rows its
        field holds: a clip's `audio` waveforms become the audio model's features, as `tokenize`
        makes ids of captions, so the dataset carries the sound and the run's encoder featurizes it."""
        return {
            condition.field: condition.encoder.tokenize(list(batch[condition.field]))
            for condition in self.conditions.values()
            if not condition.encoder.reads_captions
        }

    def check(self, batch: Batch) -> None:
        """Raise `ValueError` if `batch` is missing a field this spec names or has a wrong shape.

        The sample and the mask must have their declared per-example shapes.
        A `ModelInputs` field's shape is taken from its token rows."""
        for condition in self.conditions.values():
            if condition.field not in batch:
                raise ValueError(
                    f"objective.inputs needs condition field {condition.field!r} in the training batch"
                )
        for declared in (self.sample, self.mask):
            if declared is None:
                continue
            if declared.key not in batch:
                raise ValueError(f"objective.inputs needs field {declared.key!r} in the training batch")
            value = batch[declared.key]
            actual = np.shape(value.tokens if isinstance(value, ModelInputs) else value)[1:]
            matches = len(actual) == len(declared.shape) and all(
                0 < size <= bound if axis in declared.variable_axes else size == bound
                for axis, (size, bound) in enumerate(zip(actual, declared.shape, strict=True))
            )
            if not matches:
                raise ValueError(
                    f"objective.inputs field {declared.key!r} declares shape {declared.shape}, "
                    f"but the first training batch has shape {tuple(actual)}"
                )

    def to_json(self) -> dict:
        return {
            "sample": {
                "key": self.sample.key,
                "shape": list(self.sample.shape),
                **({"variable_axes": list(self.sample.variable_axes)} if self.sample.variable_axes else {}),
            },
            "conditions": {keyword: condition.to_json() for keyword, condition in self.conditions.items()},
            **(
                {
                    "mask": {
                        "key": self.mask.key,
                        "shape": list(self.mask.shape),
                        **(
                            {"variable_axes": list(self.mask.variable_axes)}
                            if self.mask.variable_axes
                            else {}
                        ),
                    }
                }
                if self.mask is not None
                else {}
            ),
        }

    @classmethod
    def from_json(cls, record: Mapping, *, params: Mapping[str, Variables] | None = None) -> InputSpec:
        """Rebuild the spec from its JSON record, around the given condition parameters.

        `params` maps each condition's keyword to its encoder's parameters.
        When it is None, each encoder loads its own weights."""
        sample = record["sample"]
        return cls(
            sample=Field(sample["key"], tuple(sample["shape"]), tuple(sample.get("variable_axes", ()))),
            conditions={
                keyword: Condition.from_json(condition, params=None if params is None else params[keyword])
                for keyword, condition in record["conditions"].items()
            },
            mask=Field(
                record["mask"]["key"],
                tuple(record["mask"]["shape"]),
                tuple(record["mask"].get("variable_axes", ())),
            )
            if "mask" in record
            else None,
        )


@dataclasses.dataclass(frozen=True)
class TextCondition:
    """Text conditioning: which registered encoder reads the batch's tokens, and from which checkpoint."""

    encoder: str = "clip_text"
    checkpoint: str = DEFAULT_MODEL
    dtype: DtypeName | None = None
    """The encoder's compute dtype; None follows the model's compute dtype."""
    param_dtype: DtypeName = "float32"
    """The storage dtype when loading source weights; supplied params keep theirs."""
    field: str = "text"
    """The batch field holding the tokenized text."""
    unconditional: str = ""
    """The prompt the unconditional branch is encoded from."""
    max_length: int | None = None
    """The token length every prompt is padded to.

    None keeps the encoder's own default, which for CLIP is the checkpoint's
    context length.
    """
    revision: str | None = None
    """The checkpoint's git revision.

    With it, a rerun conditions on the weights the run named, even after the
    branch has moved on.
    """

    def build(self, *, params: Variables | None = None,
              dtype: DtypeName | None = None) -> Condition:
        """Build the encoder over supplied params, without loading source weights or casting storage.

        `dtype` is the run's own compute dtype, which an unset `self.dtype` follows:
        the tower runs next to the model it conditions in every step, so a checkpoint
        stored in float32 is no reason to run it in float32.
        """
        fields = {name: value for name, value in
                  (("max_length", self.max_length), ("revision", self.revision))
                  if value is not None}
        return Condition(
            rebuild(self.encoder, {"checkpoint": self.checkpoint,
                                   "dtype": dtype if self.dtype is None else self.dtype,
                                   "param_dtype": self.param_dtype, **fields}, params=params),
            field=self.field, unconditional=self.unconditional)


@dataclasses.dataclass(frozen=True)
class AudioCondition:
    """Audio conditioning on each clip, through an audio tower.

    `checkpoint` names the `hf_audio` tower, whose feature extractor encodes the
    waveform in the batch's `audio` field on the host.
    The dataset's clip length sets the waveform length used to encode every
    clip and the silent unconditional input, so the two always agree.
    """

    encoder: ClassVar[str] = "hf_audio"
    checkpoint: str = "facebook/wav2vec2-base-960h"
    dtype: DtypeName | None = None
    """The tower's compute dtype; None follows the model's compute dtype."""
    param_dtype: DtypeName = "float32"
    """The storage dtype when loading source weights; supplied params keep theirs."""

    def build(self, seconds: float, *, params: Variables | None = None,
              dtype: DtypeName | None = None) -> Condition:
        """Build the tower over supplied params for waveforms of `seconds` (the clips' `audio_seconds`,
        `Dataset.facts`), without loading source weights or casting storage."""
        return Condition(
            rebuild(self.encoder, {"checkpoint": self.checkpoint, "seconds": seconds,
                                   "dtype": dtype if self.dtype is None else self.dtype,
                                   "param_dtype": self.param_dtype}, params=params),
            field="audio", unconditional=0.0)


def encode_conditions(encoders: Sequence[tuple[str, ConditionEncoder]], weights: Mapping[str, Variables],
                      tokens: Mapping[str, object]) -> dict:
    """Each of `encoders` (`InputSpec.encoders`) over the tokens under its keyword, with the weights
    under its keyword: the conditions a model is passed, by keyword. The one encoding a training
    step (`DiffusionObjective.encode`) and a sampling task (`TextToImage`) run."""
    return {keyword: encoder.encode(weights[keyword], tokens[keyword]) for keyword, encoder in encoders}


__all__ = [
    "AudioCondition",
    "CLIPText",
    "CharTable",
    "Condition",
    "ConditionEncoder",
    "DiffusionConditioner",
    "Field",
    "HFAudio",
    "InputSpec",
    "T5Text",
    "TextCondition",
    "unit_range",
]
