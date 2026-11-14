"""Encoders that turn raw conditioning data into the value a model keyword takes.

An encoder tokenizes on the host, in the data workers or before a sampling
call. It encodes on device, as a pure function of explicit parameters. The
parameters are a leaf of the objective's tree, and the trainer's layout
places them like any other leaf, so a frozen tower's weights reach the
compiled step as arguments, not as constants compiled into it.

`ConditionEncoder.bound(record)` rebuilds an encoder from a run's record, the
registry's `{"class": ..., "fields": ...}` whose fields `recorded_fields` wrote.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype

from dew.batch import Variables
from dew.inputs.condition import ConditionEncoder
from dew.nn.blocks import table_rows
from dew.nn.dit import TextContext
from dew.nn.flops import ConditionWork
from dew.registry import dtype_name, resolve_dtype


def char_codes(texts: Sequence[str], tokens: int, vocab: int) -> dict[str, np.ndarray]:
    """Each text as `tokens` ids of a `vocab`-row table, behind a start token, with its mask
    (`CharTable.tokenize`)."""
    # id 0 is padding and 1 is the start token, so a character takes the
    # rest of the table: its code point wrapped into the vocabulary above
    # those two. Two characters that wrap together share a vector, which
    # a table this small is for.
    ids = np.zeros((len(texts), tokens), np.int32)
    mask = np.zeros((len(texts), tokens), np.int32)
    for row, text in enumerate(texts):
        codes = [1] + [2 + (ord(char) % (vocab - 2)) for char in text[:tokens - 1]]
        ids[row, :len(codes)] = codes
        mask[row, :len(codes)] = 1
    return {"input_ids": ids, "attention_mask": mask}


@dataclass(frozen=True, eq=False)
class CharTable(ConditionEncoder[str, TextContext]):
    """Encodes text as a table lookup: one id per character, one fixed random
    vector per id.

    It costs nothing and downloads nothing, so tests, benchmarks and smoke
    runs use it as their text encoder. Its output has the form of a real
    encoder's, a `TextContext` with a mask, so a model that takes CLIP's
    output takes this one unchanged.
    """

    params: Variables
    tokens: int = 8
    features: int = 16
    vocab: int = 130
    seed: int = 0
    dtype: Dtype | None = None
    param_dtype: str = "float32"

    @classmethod
    def from_pretrained(cls, checkpoint: str = "char_table", *, dtype=None,
                        tokens: int = 8, features: int = 16, vocab: int = 130,
                        seed: int = 0, param_dtype: str = "float32",
                        params: Variables | None = None):
        """Return an encoder with the table `seed` draws, or with the table `params` already holds.

        The table has no checkpoint: `checkpoint` names it as `char_table`, and any other name is
        refused rather than read as a random table."""
        if checkpoint != "char_table":
            raise ValueError(f"the character table draws its table from `seed` and reads no checkpoint; "
                             f"name it 'char_table', not {checkpoint!r}")
        compute = resolve_dtype(dtype)
        storage = resolve_dtype(param_dtype)
        if params is None:
            table = np.random.RandomState(seed).normal(size=(vocab, features))
            params = {"table": jnp.asarray(table, storage)}
        return cls(params=params, tokens=tokens, features=features, vocab=vocab, seed=seed,
                   dtype=compute, param_dtype=param_dtype)

    def tokenize(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        return char_codes(texts, self.tokens, self.vocab)

    def conditioning_flops(self, tokens) -> ConditionWork:
        lengths = np.sum(np.asarray(tokens["attention_mask"]) != 0, axis=-1, dtype=np.int64)
        return ConditionWork(0.0, lengths, np.full(lengths.shape, np.shape(tokens["input_ids"])[1], np.int64),
                             self.features)

    def encode(self, params, tokens) -> TextContext:
        table = params["table"]
        dtype = table.dtype if self.dtype is None else self.dtype
        hidden = table_rows(table, jnp.asarray(tokens["input_ids"]), dtype)
        return TextContext(hidden=hidden, mask=jnp.asarray(tokens["attention_mask"]))

    def captions(self, tokens) -> tuple[str, ...]:
        # Each id inverts to the lowest code point that wraps to it, which is
        # the character itself below `vocab - 2`: all of ASCII by default.
        return tuple("".join(chr(int(i) - 2) for i in row[row > 1])
                     for row in np.asarray(tokens["input_ids"]))

    def recorded_fields(self) -> dict:
        return {"checkpoint": "char_table", "tokens": self.tokens,
                "features": self.features, "vocab": self.vocab, "seed": self.seed,
                "dtype": dtype_name(self.dtype), "param_dtype": self.param_dtype}


type AudioRow = np.ndarray | float | Mapping[str, object]
"""One waveform, mono at the extractor's rate; a constant, which fills the
clip (0.0 is silence); or a conditioning record holding either under `audio`."""


__all__ = ["AudioRow", "CharTable", "ConditionEncoder", "char_codes"]
