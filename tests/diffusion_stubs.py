"""What the diffusion suites share: a small registered text encoder and a
batch the data axis divides."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from dew.inputs import ConditionEncoder
from dew.nn.dit import TextContext
from dew.objectives.base import Variables
from dew.registry import encoders

RES = 8
TOKENS = 5
FEATURES = 6
VOCAB = 11


@encoders("stub_text")
@dataclass(frozen=True, eq=False)
class StubText(ConditionEncoder[str]):
    """A text encoder with a table of `VOCAB` vectors: tokenize maps a prompt to
    ids by character behind a start token, encode looks them up. Small, and
    shaped like CLIP's output, so the models' text keyword takes it; registered,
    so a run's text condition can name it."""

    checkpoint: str
    params: Variables

    @classmethod
    def from_pretrained(cls, checkpoint: str, *, params=None, **fields):
        if params is None:
            params = {"table": jnp.asarray(
                np.random.RandomState(0).normal(size=(VOCAB, FEATURES)).astype(np.float32))}
        return cls(checkpoint=checkpoint, params=params)

    def tokenize(self, data):
        ids = np.zeros((len(data), TOKENS), np.int32)
        mask = np.zeros((len(data), TOKENS), np.int32)
        for row, text in enumerate(data):
            codes = [1] + [2 + (ord(char) % (VOCAB - 2)) for char in text[:TOKENS - 1]]
            ids[row, :len(codes)] = codes
            mask[row, :len(codes)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def encode(self, params, tokens):
        return TextContext(hidden=params["table"][jnp.asarray(tokens["input_ids"])],
                           mask=jnp.asarray(tokens["attention_mask"]))

    def captions(self, tokens):
        return tuple("".join(chr(97 + int(i)) for i in row[row > 1])
                     for row in np.asarray(tokens["input_ids"]))

    def to_json(self):
        return {"checkpoint": self.checkpoint}


PROMPTS = ["a red bird", "two cats"]


def batch_for(objective, size: int) -> dict:
    """One row per device, the prompts in turn: a batch the data axis divides,
    in the channels the objective's sample field names."""
    rows, channels = jax.device_count(), objective.inputs.sample.shape[-1]
    pixels = np.tile(np.arange(size * size * channels, dtype=np.uint8).reshape(
        1, size, size, channels), (rows, 1, 1, 1))
    return {"image": pixels,
            **objective.inputs.tokenize([PROMPTS[row % len(PROMPTS)] for row in range(rows)])}
