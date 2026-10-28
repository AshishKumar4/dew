"""What the diffusion suites share: a small text encoder, a batch the data
axis divides, and a table that reads a class off a prompt."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from dew.batch import Variables
from dew.inputs import CharTable, ConditionEncoder
from dew.inputs.encoders import char_codes
from dew.nn.dit import TextContext

RES = 8
TOKENS = 5
FEATURES = 6
VOCAB = 11


@dataclass(frozen=True, eq=False)
class StubText(ConditionEncoder[str]):
    """A text encoder with a table of `vocab` vectors `features` wide: tokenize
    maps a prompt to `tokens` ids by character behind a start token, encode
    looks them up. Small, and shaped like CLIP's output, so the models' text
    keyword takes it; a run's text condition names it by its import path,
    `STUB_TEXT`."""

    checkpoint: str
    params: Variables
    tokens: int = TOKENS
    vocab: int = VOCAB

    @classmethod
    def from_pretrained(cls, checkpoint: str, *, params=None, tokens: int = TOKENS, features: int = FEATURES,
                        vocab: int = VOCAB, **fields):
        if params is None:
            params = {"table": jnp.asarray(
                np.random.RandomState(0).normal(size=(vocab, features)).astype(np.float32))}
        return cls(checkpoint=checkpoint, params=params, tokens=tokens, vocab=vocab)

    def tokenize(self, data):
        return char_codes(data, self.tokens, self.vocab)

    def encode(self, params, tokens):
        return TextContext(hidden=params["table"][jnp.asarray(tokens["input_ids"])],
                           mask=jnp.asarray(tokens["attention_mask"]))

    def captions(self, tokens):
        return tuple("".join(chr(97 + int(i)) for i in row[row > 1])
                     for row in np.asarray(tokens["input_ids"]))

    def recorded_fields(self):
        return {"checkpoint": self.checkpoint, "tokens": self.tokens,
                "features": int(self.params["table"].shape[1]), "vocab": self.vocab}


STUB_TEXT = f"{__name__}:StubText"

PROMPTS = ["a red bird", "two cats"]


def batch_for(objective, size: int) -> dict:
    """One row per device, the prompts in turn: a batch the data axis divides,
    in the channels the objective's sample field names (`diffusion_support.image_batch`)."""
    from diffusion_support import image_batch

    return image_batch(objective, PROMPTS, size)


def label_table(labels, null: float = 0.0) -> CharTable:
    """A character table of one feature: the prompt `str(i)` reads
    `labels[i]`, and the padding id 0 reads `null`, the class a dropped
    condition stands for."""
    table = CharTable.from_pretrained(tokens=2, features=1)
    entries = np.zeros((table.vocab, 1), np.float32)
    entries[0] = null
    for index, label in enumerate(labels):
        entries[table.tokenize([str(index)])["input_ids"][0, 1]] = label
    return CharTable.from_pretrained(tokens=2, features=1, params={"table": jnp.asarray(entries)})
