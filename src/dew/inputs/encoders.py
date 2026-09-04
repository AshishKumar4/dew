"""From raw conditioning data to the value a model keyword takes.

An encoder tokenizes on the host, in the data workers or before a sampling
call, and encodes on device as a pure function of explicit parameters. The
parameters are a leaf of the objective's tree, placed by the trainer's layout
like any other, so a frozen tower's weights arrive at the compiled step as
arguments and never as constants baked into it.

An encoder is rebuilt from a run's record by `encoders[name].from_pretrained(
**fields)`, where `fields` is what `to_json` wrote.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.dit import TextContext
from dew.nn.text_encoders import DEFAULT_MODEL, CLIPTextModel, CLIPTextTransformer
from dew.registry import dtype_name, encoders, resolve_dtype


class ConditionEncoder(ABC):
    """A modality's path from raw data to a conditioning value."""

    params: Any

    @classmethod
    @abstractmethod
    def from_pretrained(cls, checkpoint: str, **fields) -> "ConditionEncoder":
        """Load the tower named `checkpoint`; the one call that opens files."""

    @abstractmethod
    def tokenize(self, data: Sequence) -> Any:
        """Raw data to the host arrays `encode` reads, one row per item."""

    @abstractmethod
    def encode(self, params, tokens) -> Any:
        """Tokens to the conditioning value, on device, under `params`."""

    def captions(self, tokens) -> tuple[str, ...]:
        """What the tokens say, for a rendered artifact; a modality that is
        not text has nothing to say."""
        return ()

    @abstractmethod
    def to_json(self) -> dict:
        """The keyword fields `from_pretrained` rebuilds this encoder from."""


@encoders("clip_text")
@dataclass(frozen=True, eq=False)
class CLIPText(ConditionEncoder):
    """The CLIP text tower, vendored in `dew.nn.text_encoders`, with the
    checkpoint's tokenizer.

    `tokenize` pads every prompt to the checkpoint's context length and
    returns the ids with the attention mask; `encode` returns the last hidden
    state with that mask as a `TextContext`, so a model can pool over the real
    tokens only.
    """

    checkpoint: str
    transformer: CLIPTextTransformer
    params: Any
    tokenizer: Any
    dtype: Optional[Any] = None

    @classmethod
    def from_pretrained(cls, checkpoint: str = DEFAULT_MODEL, *, dtype=None,
                        revision: Optional[str] = None) -> "CLIPText":
        from transformers import AutoTokenizer

        dtype = resolve_dtype(dtype)
        model = CLIPTextModel.from_pretrained(checkpoint, dtype=dtype, revision=revision)
        return cls(checkpoint=checkpoint, transformer=model.transformer, params=model.variables,
                   tokenizer=AutoTokenizer.from_pretrained(checkpoint), dtype=dtype)

    def tokenize(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        tokens = self.tokenizer(list(texts), padding="max_length",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors="np")
        return {"input_ids": np.asarray(tokens["input_ids"], np.int32),
                "attention_mask": np.asarray(tokens["attention_mask"], np.int32)}

    def encode(self, params, tokens) -> TextContext:
        mask = jnp.asarray(tokens["attention_mask"])
        hidden = self.transformer.apply(params, jnp.asarray(tokens["input_ids"]), mask)
        return TextContext(hidden=hidden.last_hidden_state, mask=mask)

    def captions(self, tokens) -> tuple[str, ...]:
        return tuple(self.tokenizer.batch_decode(
            np.asarray(tokens["input_ids"]), skip_special_tokens=True))

    def to_json(self) -> dict:
        return {"checkpoint": self.checkpoint, "dtype": dtype_name(self.dtype)}


@encoders("audio")
@dataclass(frozen=True, eq=False)
class Audio(ConditionEncoder):
    """Audio conditioning through a Hugging Face feature extractor and a pure
    `apply(params, features) -> [B, L, D]` over whatever keys the extractor
    emits (`input_values` for wav2vec2/HuBERT, `input_features` for
    Whisper/AST), so switching audio models changes nothing here.

    No loader builds the tower: transformers 5 ships no Flax audio models,
    and a torch one cannot take the numpy arrays `tokenize` produces.
    `from_pretrained` says so. Until an audio tower is vendored the way
    `dew.nn.text_encoders` vendors CLIP, an encoder is constructed with an
    `apply` and `params` of the caller's own.
    """

    checkpoint: str
    extractor: Any
    apply: Callable[[Any, Mapping[str, Any]], jax.Array]
    params: Any
    sampling_rate: int

    @classmethod
    def from_pretrained(cls, checkpoint: str = "facebook/wav2vec2-base-960h", **fields):
        raise NotImplementedError(
            f"no loader can build {checkpoint!r}: transformers 5 ships no Flax audio "
            "models, and a torch model cannot take the numpy arrays tokenize produces. "
            "Vendor the audio tower the way dew.nn.text_encoders vendors CLIP, then "
            "construct Audio with its apply and params and "
            "AutoFeatureExtractor.from_pretrained(checkpoint) as the extractor")

    def tokenize(self, audio: Sequence) -> dict[str, np.ndarray]:
        return dict(self.extractor(audio, sampling_rate=self.sampling_rate,
                                   padding=True, return_tensors="np"))

    def encode(self, params, tokens) -> jax.Array:
        return self.apply(params, tokens)

    def to_json(self) -> dict:
        return {"checkpoint": self.checkpoint, "sampling_rate": self.sampling_rate}


@encoders("char_table")
@dataclass(frozen=True, eq=False)
class CharTable(ConditionEncoder):
    """Text as a table lookup: each character is an id and each id a fixed
    random vector. It costs nothing and downloads nothing, so it is the text
    encoder of tests, benchmarks and smoke runs, and it has the shape of a
    real one (`TextContext` with a mask), so a model that takes CLIP's output
    takes this one unchanged.
    """

    params: dict
    tokens: int = 8
    features: int = 16
    vocab: int = 130

    @classmethod
    def from_pretrained(cls, checkpoint: str = "char_table", *, dtype=None,
                        tokens: int = 8, features: int = 16, vocab: int = 130,
                        seed: int = 0):
        table = np.random.RandomState(seed).normal(size=(vocab, features))
        return cls(params={"table": jnp.asarray(table, resolve_dtype(dtype) or jnp.float32)},
                   tokens=tokens, features=features, vocab=vocab)

    def tokenize(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        # id 0 is padding, 1 is the start token, characters follow.
        ids = np.zeros((len(texts), self.tokens), np.int32)
        mask = np.zeros((len(texts), self.tokens), np.int32)
        for row, text in enumerate(texts):
            codes = [1] + [2 + (ord(char) % (self.vocab - 2)) for char in text[:self.tokens - 1]]
            ids[row, :len(codes)] = codes
            mask[row, :len(codes)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def encode(self, params, tokens) -> TextContext:
        return TextContext(hidden=params["table"][jnp.asarray(tokens["input_ids"])],
                           mask=jnp.asarray(tokens["attention_mask"]))

    def captions(self, tokens) -> tuple[str, ...]:
        return tuple("".join(chr(97 + (int(i) - 2) % 26) for i in row[row > 1])
                     for row in np.asarray(tokens["input_ids"]))

    def to_json(self) -> dict:
        return {"checkpoint": "char_table", "tokens": self.tokens,
                "features": self.features, "vocab": self.vocab}


__all__ = ["ConditionEncoder", "CLIPText", "Audio", "CharTable"]
