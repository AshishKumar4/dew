"""What a generative objective is fed: the sample field and its conditions.

`InputSpec` names the batch field the model learns to generate, and the
conditions it is given, keyed by the model's own keyword arguments. A
`Condition` holds an encoder, the batch field it reads and the raw datum that
stands for "no condition", which classifier-free guidance and conditioning
dropout substitute. Nothing here runs a model. The spec only describes the
inputs, and the objective does the encoding.

Image and video batches arrive as uint8 pixels in [0, 255], as the data
workers write them. `unit_range` is the one conversion to [-1, 1], the range
every diffusion loss, sample, artifact and image metric uses, and
`dew.artifacts.uint8_pixels` is the one conversion back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from dew import registry
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Batch, Variables

from .diffusion import DiffusionConditioner
from .encoders import CharTable, CLIPText, ConditionEncoder, HFAudio, T5Text, rebuild


def unit_range(pixels: jax.typing.ArrayLike) -> jax.Array:
    """Convert pixels in [0, 255] to [-1, 1].

    uint8 and float32 pixels give float32, and a float64 run's float64 pixels
    stay float64.
    """
    pixels = jnp.asarray(pixels)
    return (pixels.astype(jnp.promote_types(pixels.dtype, jnp.float32)) - 127.5) / 127.5


@dataclass(frozen=True)
class Field:
    """A batch field and its per-example shape, such as `Field("image", (128, 128, 3))`."""

    key: str
    shape: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(size) for size in self.shape))


@dataclass(frozen=True)
class Condition:
    """Names one conditioning input: its encoder, the batch field holding its
    tokens, and the raw datum for the unconditional branch."""

    encoder: ConditionEncoder
    field: str = "text"
    unconditional: str | float | Mapping[str, object] = ""

    def to_json(self) -> dict:
        return {"encoder": {"class": registry.import_path(type(self.encoder)),
                            "fields": self.encoder.to_json()},
                "field": self.field,
                "unconditional": self.unconditional}

    @classmethod
    def from_json(cls, record: Mapping, *, params: Variables | None = None) -> Condition:
        encoder = record["encoder"]
        return cls(encoder=rebuild(encoder["class"], encoder["fields"], params=params),
                   field=record["field"], unconditional=record["unconditional"])


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
                "Name a field per condition")

    def tokenize(self, captions: Sequence[str]) -> dict[str, Mapping[str, np.ndarray]]:
        """Return the batch fields this run's caption conditions read, tokenized from `captions`.

        The result is empty for a run with no caption conditions, so the
        captions stop at the loader and no string array reaches a device.
        """
        return {condition.field: condition.encoder.tokenize(captions)
                for condition in self.conditions.values() if condition.encoder.reads_captions}

    def check(self, batch: Batch) -> None:
        """Raise `ValueError` if `batch` is missing a field this spec names or has a wrong shape.

        The sample and the mask must have their declared per-example shapes.
        A `ModelInputs` field's shape is taken from its token rows."""
        for condition in self.conditions.values():
            if condition.field not in batch:
                raise ValueError(f"objective.inputs needs condition field {condition.field!r} "
                                 "in the training batch")
        for declared in (self.sample, self.mask):
            if declared is None:
                continue
            if declared.key not in batch:
                raise ValueError(f"objective.inputs needs field {declared.key!r} in the training batch")
            value = batch[declared.key]
            actual = np.shape(value.tokens if isinstance(value, ModelInputs) else value)[1:]
            if tuple(actual) != declared.shape:
                raise ValueError(f"objective.inputs field {declared.key!r} declares shape {declared.shape}, "
                                 f"but the first training batch has shape {tuple(actual)}")

    def to_json(self) -> dict:
        return {
            "sample": {"key": self.sample.key, "shape": list(self.sample.shape)},
            "conditions": {keyword: condition.to_json() for keyword, condition in self.conditions.items()},
            **(
                {"mask": {"key": self.mask.key, "shape": list(self.mask.shape)}}
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
            sample=Field(sample["key"], tuple(sample["shape"])),
            conditions={
                keyword: Condition.from_json(condition, params=None if params is None else params[keyword])
                for keyword, condition in record["conditions"].items()
            },
            mask=Field(record["mask"]["key"], tuple(record["mask"]["shape"])) if "mask" in record else None,
        )


__all__ = ["CLIPText", "CharTable", "Condition", "ConditionEncoder", "DiffusionConditioner", "Field",
           "HFAudio", "InputSpec", "T5Text", "unit_range"]
