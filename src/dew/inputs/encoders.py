"""From raw conditioning data to the value a model keyword takes.

An encoder tokenizes on the host, in the data workers or before a sampling
call, and encodes on device as a pure function of explicit parameters. The
parameters are a leaf of the objective's tree, placed by the trainer's layout
like any other, so a frozen tower's weights arrive at the compiled step as
arguments, not as constants baked into it.

An encoder is rebuilt from a run's record by `rebuild(name, fields)`, where
`fields` is what `to_json` wrote.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, Mapping, Self, Sequence

import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype
from typing_extensions import TypeVar

from dew.nn.dit import TextContext
from dew.nn.text_encoders import (
    DEFAULT_MODEL,
    DEFAULT_T5_MODEL,
    CLIPTextModel,
    CLIPTextTransformer,
    CLIPTowerOutput,
    T5EncoderModel,
    T5EncoderTransformer,
)
from dew.objectives.base import FROZEN, Variables
from dew.registry import dtype_name, encoders, resolve_dtype

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

Raw = TypeVar("Raw")
"""One item of a modality's raw data: a prompt, a waveform."""

Encoded = TypeVar("Encoded", default=object)
"""The conditioning value a modality's encoder produces: a `TextContext`, a
`DenoisingCondition`, an array. An encoder that leaves it unstated promises
its callers nothing about the value beyond what it hands the model."""


class ConditionEncoder(ABC, Generic[Raw, Encoded]):
    """Carries one modality from raw data to a conditioning value."""

    params: Variables
    parameter_collections: ClassVar[tuple[str, ...] | None] = None
    """None declares a bare parameter tree; otherwise these collections own
    learned weights, including frozen ones. Other collections retain their dtype."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, checkpoint: str, *, params: Variables | None = None) -> Self:
        """Loads the tower named `checkpoint`, the one call that opens files.

        Whatever else a checkpoint needs is a keyword field with a default,
        which is what `to_json` records and the registry rebuilds from.
        Supplied params are authoritative: the load reads metadata and never
        source weights, and keeps their values, dtypes and placement.
        """

    @abstractmethod
    def tokenize(self, texts: Sequence[Raw]) -> Mapping[str, np.ndarray]:
        """Raw data to the host arrays `encode` reads, one row per item."""

    @abstractmethod
    def encode(self, params: Variables, tokens) -> Encoded:
        """Tokens to the conditioning value, on device, under `params`."""

    def captions(self, tokens) -> tuple[str, ...]:
        """What the tokens say, for a rendered artifact.

        A modality that is not text has nothing to say and answers nothing.
        """
        return ()

    @abstractmethod
    def to_json(self) -> dict:
        """The keyword fields `from_pretrained` rebuilds this encoder from."""


def rebuild(name: str, fields: Mapping[str, object], *,
            params: Variables | None = None) -> ConditionEncoder:
    """The named encoder rebuilt from its JSON fields.

    A run's record stores the registry name with the keyword fields `to_json`
    wrote. Those fields are unpacked here, so each encoder's `from_pretrained`
    keeps its own concrete signature. The checkpoint is the one field every
    encoder takes and is read here; the rest are the encoder's own and its
    signature checks them.
    """
    checkpoint = fields.get("checkpoint")
    if not isinstance(checkpoint, str):
        raise ValueError(f"the {name} record names no checkpoint to rebuild from")
    rest = {key: value for key, value in fields.items() if key != "checkpoint"}
    return encoders[name].from_pretrained(checkpoint, **rest, params=params)


@encoders("clip_text")
@dataclass(frozen=True, eq=False)
class CLIPText(ConditionEncoder[str, TextContext]):
    """The CLIP text tower, vendored in `dew.nn.text_encoders`, with the
    checkpoint's tokenizer.

    `tokenize` pads every prompt to the checkpoint's context length and
    returns the ids with the attention mask. `encode` returns the last hidden
    state with that mask as a `TextContext`, so a model can pool over the
    real tokens only.
    """

    checkpoint: str
    parameter_collections: ClassVar[tuple[str, ...] | None] = ("params", FROZEN)
    transformer: CLIPTextTransformer
    params: Variables
    tokenizer: PreTrainedTokenizerBase
    dtype: Dtype | None = None

    revision: str | None = None
    """The checkpoint's git revision, recorded so a rebuild reads the same
    weights and the same tokenizer."""
    param_dtype: str = "float32"

    @classmethod
    def from_pretrained(cls, checkpoint: str = DEFAULT_MODEL, *, dtype=None,
                        revision: str | None = None, param_dtype: str = "float32",
                        params: Variables | None = None) -> CLIPText:
        from transformers import AutoTokenizer

        dtype = resolve_dtype(dtype)
        model = CLIPTextModel.from_pretrained(
            checkpoint, dtype=dtype, revision=revision, param_dtype=param_dtype, variables=params)
        return cls(checkpoint=checkpoint, transformer=model.transformer, params=model.variables,
                   tokenizer=AutoTokenizer.from_pretrained(checkpoint, revision=revision),
                   dtype=dtype, param_dtype=param_dtype, revision=revision)

    def tokenize(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        tokens = self.tokenizer(list(texts), padding="max_length",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors="np")
        return {"input_ids": np.asarray(tokens["input_ids"], np.int32),
                "attention_mask": np.asarray(tokens["attention_mask"], np.int32)}

    def encode(self, params, tokens) -> TextContext:
        mask = jnp.asarray(tokens["attention_mask"])
        hidden = self.transformer.apply(params, jnp.asarray(tokens["input_ids"]), mask)
        # The tower returns its own output type; apply's mutable-collections
        # pair would mean collections were asked for, and none were.
        assert isinstance(hidden, CLIPTowerOutput)
        return TextContext(hidden=hidden.last_hidden_state, mask=mask)

    def captions(self, tokens) -> tuple[str, ...]:
        return tuple(self.tokenizer.batch_decode(
            np.asarray(tokens["input_ids"]), skip_special_tokens=True))

    def to_json(self) -> dict:
        return {"checkpoint": self.checkpoint, "dtype": dtype_name(self.dtype),
                "param_dtype": self.param_dtype,
                **({} if self.revision is None else {"revision": self.revision})}


@encoders("t5")
@dataclass(frozen=True, eq=False)
class T5Text(ConditionEncoder[str, TextContext]):
    """The T5 encoder tower, vendored in `dew.nn.text_encoders`, with the
    checkpoint's tokenizer.

    It is the text half of an SD3.5/Flux-class run, whose MMDiT conditions on
    T5-XXL's last hidden states. `tokenize` pads every prompt to `max_length`
    and returns the ids with the attention mask. `encode` returns the last
    hidden state with that mask as a `TextContext`.
    """

    checkpoint: str
    parameter_collections: ClassVar[tuple[str, ...] | None] = ("params", FROZEN)
    transformer: T5EncoderTransformer
    params: Variables
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 256
    dtype: Dtype | None = None

    revision: str | None = None
    """The checkpoint's git revision, recorded so a rebuild reads the same
    weights and the same tokenizer."""
    param_dtype: str = "float32"

    @classmethod
    def from_pretrained(cls, checkpoint: str = DEFAULT_T5_MODEL, *, dtype=None,
                        revision: str | None = None,
                        max_length: int = 256, param_dtype: str = "float32",
                        params: Variables | None = None) -> T5Text:
        from transformers import AutoTokenizer

        dtype = resolve_dtype(dtype)
        model = T5EncoderModel.from_pretrained(
            checkpoint, dtype=dtype, revision=revision, param_dtype=param_dtype, variables=params)
        return cls(checkpoint=checkpoint, transformer=model.transformer, params=model.variables,
                   tokenizer=AutoTokenizer.from_pretrained(checkpoint, revision=revision),
                   max_length=max_length, dtype=dtype, param_dtype=param_dtype, revision=revision)

    def tokenize(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        tokens = self.tokenizer(list(texts), padding="max_length", max_length=self.max_length,
                                truncation=True, return_tensors="np")
        return {"input_ids": np.asarray(tokens["input_ids"], np.int32),
                "attention_mask": np.asarray(tokens["attention_mask"], np.int32)}

    def encode(self, params, tokens) -> TextContext:
        mask = jnp.asarray(tokens["attention_mask"])
        hidden = self.transformer.apply(params, jnp.asarray(tokens["input_ids"]), mask)
        # The tower returns one array; a tuple would mean apply() returned
        # mutable collections, and none were asked for.
        assert not isinstance(hidden, tuple)
        return TextContext(hidden=hidden, mask=mask)

    def captions(self, tokens) -> tuple[str, ...]:
        return tuple(self.tokenizer.batch_decode(
            np.asarray(tokens["input_ids"]), skip_special_tokens=True))

    def to_json(self) -> dict:
        return {"checkpoint": self.checkpoint, "dtype": dtype_name(self.dtype),
                "param_dtype": self.param_dtype,
                "max_length": self.max_length,
                **({} if self.revision is None else {"revision": self.revision})}


@encoders("char_table")
@dataclass(frozen=True, eq=False)
class CharTable(ConditionEncoder[str, TextContext]):
    """Encodes text as a table lookup: one id per character, one fixed random
    vector per id.

    It costs nothing and downloads nothing, which makes it the text encoder
    of tests, benchmarks and smoke runs. It has the shape of a real one, a
    `TextContext` with a mask, so a model that takes CLIP's output takes this
    one unchanged.
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
        compute = resolve_dtype(dtype)
        storage = resolve_dtype(param_dtype)
        if params is None:
            table = np.random.RandomState(seed).normal(size=(vocab, features))
            params = {"table": jnp.asarray(table, storage)}
        return cls(params=params, tokens=tokens, features=features, vocab=vocab, seed=seed,
                   dtype=compute, param_dtype=param_dtype)

    def tokenize(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        # id 0 is padding and 1 is the start token, so a character takes the
        # rest of the table: its code point wrapped into the vocabulary above
        # those two. Two characters that wrap together share a vector, which
        # a table this small is for.
        ids = np.zeros((len(texts), self.tokens), np.int32)
        mask = np.zeros((len(texts), self.tokens), np.int32)
        for row, text in enumerate(texts):
            codes = [1] + [2 + (ord(char) % (self.vocab - 2)) for char in text[:self.tokens - 1]]
            ids[row, :len(codes)] = codes
            mask[row, :len(codes)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def encode(self, params, tokens) -> TextContext:
        hidden = params["table"][jnp.asarray(tokens["input_ids"])]
        if self.dtype is not None:
            hidden = hidden.astype(self.dtype)
        return TextContext(hidden=hidden, mask=jnp.asarray(tokens["attention_mask"]))

    def captions(self, tokens) -> tuple[str, ...]:
        return tuple("".join(chr(97 + (int(i) - 2) % 26) for i in row[row > 1])
                     for row in np.asarray(tokens["input_ids"]))

    def to_json(self) -> dict:
        return {"checkpoint": "char_table", "tokens": self.tokens,
                "features": self.features, "vocab": self.vocab, "seed": self.seed,
                "dtype": dtype_name(self.dtype), "param_dtype": self.param_dtype}


__all__ = ["CLIPText", "CharTable", "ConditionEncoder", "T5Text", "rebuild"]
