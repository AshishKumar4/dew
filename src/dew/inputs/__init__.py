"""What a generative objective is fed: the sample field and its conditions.

`InputSpec` names the batch field the model learns to generate and, keyed by
the model's own keyword arguments, the conditions it is given. A `Condition`
is an encoder, the batch field it reads and the raw datum that stands for
"no condition", which classifier-free guidance and conditioning dropout
substitute. Nothing here runs a model: the spec is a description, and the
objective encodes.

Image and video batches arrive as uint8 pixels in [0, 255], the way the data
workers write them; `unit_range` is the one conversion to the [-1, 1] range
every diffusion loss, sample, artifact and image metric lives in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dew import registry
from .encoders import CharTable, CLIPText, ConditionEncoder, T5Text, rebuild


def unit_range(pixels: jax.typing.ArrayLike) -> jax.Array:
    """uint8 pixels in [0, 255] as float32 in [-1, 1]."""
    return (jnp.asarray(pixels, jnp.float32) - 127.5) / 127.5


@dataclass(frozen=True)
class Field:
    """A batch field and its per-example shape: `Field("image", (128, 128, 3))`."""

    key: str
    shape: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(size) for size in self.shape))


@dataclass(frozen=True)
class Condition:
    """One conditioning input: the encoder, the batch field holding its
    tokens, and the raw datum for the unconditional branch."""

    encoder: ConditionEncoder
    field: str = "text"
    unconditional: str | float = ""

    def to_json(self) -> dict:
        return {"encoder": {"name": registry.encoders.name_of(type(self.encoder)),
                            "fields": self.encoder.to_json()},
                "field": self.field,
                "unconditional": self.unconditional}

    @classmethod
    def from_json(cls, data: Mapping) -> "Condition":
        encoder = data["encoder"]
        return cls(encoder=rebuild(encoder["name"], encoder["fields"]),
                   field=data["field"], unconditional=data["unconditional"])


@dataclass(frozen=True)
class InputSpec:
    """The sample field and the conditions, keyed by the model keyword each
    is passed under: `{"textcontext": Condition(...)}`.

    `tokenize` is what a captioning dataset hands its text to: every
    condition tokenizes the batch's captions under its own field, so the
    encoder a run names decides the ids and the context length and a
    dataset carries the words alone.
    """

    sample: Field
    conditions: Mapping[str, Condition] = field(default_factory=dict)

    def __post_init__(self):
        fields = [condition.field for condition in self.conditions.values()]
        if len(set(fields)) != len(fields):
            raise ValueError(
                f"two conditions share the batch field {sorted(set(fields))}: "
                "the objective reads batch[condition.field] under each keyword, "
                "so the second tokenization would overwrite the first. "
                "Name a field per condition")

    def tokenize(self, captions: Sequence[str]) -> dict[str, Mapping[str, np.ndarray]]:
        """The batch fields this run's conditions read out of `captions`.

        Empty for a run that conditions on nothing, which is how the
        captions stop at the loader instead of riding a string array into a
        device.
        """
        return {condition.field: condition.encoder.tokenize(captions)
                for condition in self.conditions.values()}

    def to_json(self) -> dict:
        return {"sample": {"key": self.sample.key, "shape": list(self.sample.shape)},
                "conditions": {keyword: condition.to_json()
                               for keyword, condition in self.conditions.items()}}

    @classmethod
    def from_json(cls, data: Mapping) -> "InputSpec":
        """Rebuild the spec, loading each encoder's weights; the one place a
        spec opens files."""
        sample = data["sample"]
        return cls(sample=Field(sample["key"], tuple(sample["shape"])),
                   conditions={keyword: Condition.from_json(condition)
                               for keyword, condition in data["conditions"].items()})


__all__ = ["Field", "Condition", "InputSpec", "ConditionEncoder", "CLIPText", "T5Text",
           "CharTable", "rebuild", "unit_range"]
