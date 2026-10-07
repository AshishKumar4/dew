"""Encoders that turn raw conditioning data into the value a model keyword takes.

An encoder tokenizes on the host, in the data workers or before a sampling
call. It encodes on device, as a pure function of explicit parameters. The
parameters are a leaf of the objective's tree, and the trainer's layout
places them like any other leaf, so a frozen tower's weights reach the
compiled step as arguments, not as constants compiled into it.

`rebuild(name, fields)` rebuilds an encoder from a run's record, where
`fields` is what `to_json` wrote.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Generic, Self

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
    from torchax.interop import JittableModule
    from transformers import PreTrainedTokenizerBase

    from dew.data.processors import AutoAudioProcessor

Raw = TypeVar("Raw")
"""One item of a modality's raw data: a prompt, a waveform."""

Encoded = TypeVar("Encoded", default=object)
"""The conditioning value a modality's encoder produces: a `TextContext`, a
`DenoisingCondition`, an array. An encoder that leaves it unstated promises
its callers nothing about the value beyond what it hands the model."""


class ConditionEncoder(ABC, Generic[Raw, Encoded]):
    """Turns one modality's raw data into a conditioning value.

    Subclasses implement `from_pretrained`, `tokenize`, `encode` and
    `to_json`.
    """

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
        `to_json` records those fields, and the registry rebuilds the encoder
        from them. When you pass `params`, the load uses them as they are. It
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
    def to_json(self) -> dict:
        """Return the keyword fields that `from_pretrained` rebuilds this encoder from."""


def rebuild(name: str, fields: Mapping[str, object], *,
            params: Variables | None = None) -> ConditionEncoder:
    """Rebuild the encoder an alias or import path `name`s from its JSON fields.

    A run's record stores the encoder's import path and the keyword fields `to_json`
    wrote. This function unpacks those fields into the encoder's
    `from_pretrained`, so each encoder keeps its own concrete signature.
    `checkpoint` is the one field every encoder takes, and this function
    reads it; a record without one raises ValueError. The other fields
    belong to the encoder, and its signature checks them.
    """
    checkpoint = fields.get("checkpoint")
    if not isinstance(checkpoint, str):
        raise ValueError(f"the {name} record names no checkpoint to rebuild from")
    rest = {key: value for key, value in fields.items() if key != "checkpoint"}
    return encoders[name].from_pretrained(checkpoint, **rest, params=params)


@dataclass(frozen=True, eq=False)
class _TextTower(ConditionEncoder[str, TextContext]):
    """A pretrained text tower with the checkpoint's own tokenizer.

    `tokenize` pads every prompt to `context` tokens and returns the ids with
    the attention mask. `encode` returns the tower's last hidden state with
    that mask as a `TextContext`, so a model can pool over the real tokens
    only. A subclass names its own transformer, says how long a prompt it
    pads to, and reads the state out of whatever its `apply` returns.
    """

    checkpoint: str
    parameter_collections: ClassVar[tuple[str, ...] | None] = ("params", FROZEN)
    transformer: CLIPTextTransformer | T5EncoderTransformer
    params: Variables
    tokenizer: PreTrainedTokenizerBase
    dtype: Dtype | None = None

    revision: str | None = None
    """The checkpoint's git revision, recorded so a rebuild reads the same
    weights and the same tokenizer."""
    param_dtype: str = "float32"

    @property
    def context(self) -> int:
        """How many tokens a prompt is padded to."""
        raise NotImplementedError

    def states(self, answer: object) -> jnp.ndarray:
        """The tower's last hidden state, out of what `apply` handed back.

        Flax's `apply` is typed as returning anything. The CLIP tower returns
        its output record and the T5 tower the states themselves; anything
        else means collections were mutated that no tower asks for."""
        if isinstance(answer, CLIPTowerOutput):
            return answer.last_hidden_state
        if isinstance(answer, jnp.ndarray):
            return answer
        raise TypeError(f"{type(self).__name__} returned {type(answer).__name__}, not hidden states")

    @property
    def recorded(self) -> dict:
        """The fields this tower records beyond the ones every tower has."""
        return {}

    def tokenize(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        tokens = self.tokenizer(list(texts), padding="max_length", max_length=self.context,
                                truncation=True, return_tensors="np")
        return {"input_ids": np.asarray(tokens["input_ids"], np.int32),
                "attention_mask": np.asarray(tokens["attention_mask"], np.int32)}

    def encode(self, params, tokens) -> TextContext:
        mask = jnp.asarray(tokens["attention_mask"])
        answer = self.transformer.apply(params, jnp.asarray(tokens["input_ids"]), mask)
        return TextContext(hidden=self.states(answer), mask=mask)

    def captions(self, tokens) -> tuple[str, ...]:
        return tuple(self.tokenizer.batch_decode(
            np.asarray(tokens["input_ids"]), skip_special_tokens=True))

    def to_json(self) -> dict:
        return {"checkpoint": self.checkpoint, "dtype": dtype_name(self.dtype),
                "param_dtype": self.param_dtype, **self.recorded,
                **({} if self.revision is None else {"revision": self.revision})}


@dataclass(frozen=True, eq=False)
class CLIPText(_TextTower):
    """Encodes text with a CLIP text tower and its checkpoint's tokenizer.

    The tower is the one vendored in `dew.nn.text_encoders`. Prompts are
    padded to the checkpoint's own context length, which the tokenizer
    reports. `encode` returns the tower's last hidden state and the
    attention mask as a `TextContext`, so a model can pool over the real
    tokens only.
    """

    transformer: CLIPTextTransformer

    @classmethod
    def from_pretrained(cls, checkpoint: str = DEFAULT_MODEL, *, dtype=None,
                        revision: str | None = None, param_dtype: str = "float32",
                        params: Variables | None = None) -> CLIPText:
        from dew.data.text import load_tokenizer

        dtype = resolve_dtype(dtype)
        model = CLIPTextModel.from_pretrained(
            checkpoint, dtype=dtype, revision=revision, param_dtype=param_dtype, variables=params)
        return cls(checkpoint=checkpoint, transformer=model.transformer, params=model.variables,
                   tokenizer=load_tokenizer(checkpoint, revision=revision),
                   dtype=dtype, param_dtype=param_dtype, revision=revision)

    @property
    def context(self) -> int:
        return self.tokenizer.model_max_length



@dataclass(frozen=True, eq=False)
class T5Text(_TextTower):
    """Encodes text with a T5 encoder tower and its checkpoint's tokenizer.

    The tower is the one vendored in `dew.nn.text_encoders`. It is the text
    half of an SD3.5/Flux-class run, whose MMDiT conditions on T5-XXL's last
    hidden states. Prompts are padded to `max_length` (256 by default), which
    the run's record stores. `encode` returns the tower's last hidden state
    and the attention mask as a `TextContext`.
    """

    transformer: T5EncoderTransformer
    max_length: int = 256

    @classmethod
    def from_pretrained(cls, checkpoint: str = DEFAULT_T5_MODEL, *, dtype=None,
                        revision: str | None = None,
                        max_length: int = 256, param_dtype: str = "float32",
                        params: Variables | None = None) -> T5Text:
        from dew.data.text import load_tokenizer

        dtype = resolve_dtype(dtype)
        model = T5EncoderModel.from_pretrained(
            checkpoint, dtype=dtype, revision=revision, param_dtype=param_dtype, variables=params)
        return cls(checkpoint=checkpoint, transformer=model.transformer, params=model.variables,
                   tokenizer=load_tokenizer(checkpoint, revision=revision),
                   max_length=max_length, dtype=dtype, param_dtype=param_dtype, revision=revision)

    @property
    def context(self) -> int:
        return self.max_length

    @property
    def recorded(self) -> dict:
        return {"max_length": self.max_length}



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

        There is nothing to load, so `checkpoint` is ignored. It is in the
        signature because `rebuild` passes every encoder the checkpoint name
        its `to_json` wrote, and this encoder always writes `"char_table"`.
        """
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
        # Each id inverts to the lowest code point that wraps to it, which is
        # the character itself below `vocab - 2`: all of ASCII by default.
        return tuple("".join(chr(int(i) - 2) for i in row[row > 1])
                     for row in np.asarray(tokens["input_ids"]))

    def to_json(self) -> dict:
        return {"checkpoint": "char_table", "tokens": self.tokens,
                "features": self.features, "vocab": self.vocab, "seed": self.seed,
                "dtype": dtype_name(self.dtype), "param_dtype": self.param_dtype}


type AudioRow = np.ndarray | float | Mapping[str, object]
"""One waveform, mono at the extractor's rate; a constant, which fills the
clip (0.0 is silence); or a conditioning record holding either under `audio`."""


def _last_hidden_state(model, features, *, key: str):
    """`model`'s last hidden state for its input `key`, run by torchax."""
    return model(**{key: features}).last_hidden_state


@dataclass(frozen=True, eq=False)
class HFAudio(ConditionEncoder[AudioRow, TextContext]):
    """Encodes audio as a transformers model's last hidden state, using the checkpoint's feature extractor.

    The model is called with one array, the extractor's first model input
    (`model_input_names[0]`), and torchax lowers transformers' PyTorch
    forward to JAX. A checkpoint works when `AutoModel` builds it, when its
    forward (or, for an encoder-decoder, its encoder's forward) takes that
    one input and returns `last_hidden_state`, and when torchax lowers every
    op it runs. wav2vec2 (`input_values`) and Whisper's encoder
    (`input_features`) are the tested ones. Weight-norm parametrizations are
    folded into plain weights, which compute the same result for a frozen
    tower. The states come back as a `TextContext` in which every position
    is real, so a model that cross-attends to text attends to them
    unchanged.

    Every waveform is cut or zero-padded to `seconds`, so every clip has the
    same length, including the constant the unconditional branch is encoded
    from. A dataset that writes the extractor's arrays itself
    (`VideoDataset`'s `audio` field) passes them to `encode` unchanged, which
    is why this encoder reads no captions.
    """

    checkpoint: str
    seconds: float
    module: JittableModule
    params: Variables
    audio: AutoAudioProcessor
    dtype: Dtype | None = None
    param_dtype: str = "float32"
    parameter_collections: ClassVar[tuple[str, ...] | None] = ("params",)
    reads_captions: ClassVar[bool] = False

    @classmethod
    def from_pretrained(cls, checkpoint: str = "facebook/wav2vec2-base-960h", *, seconds: float = 1.0,
                        dtype=None, param_dtype: str = "float32",
                        params: Variables | None = None) -> HFAudio:
        from dew.data.processors import AutoAudioProcessor
        from dew.interop.pickles import host_view
        from dew.interop.torchax_fallback import INSTALL

        try:
            import torch
            from torch.nn.utils import parametrize
            from torchax.interop import JittableModule, extract_all_buffers
            from transformers import AutoConfig, AutoModel
        except ImportError as error:
            raise ImportError(
                f"hf_audio runs transformers' audio model through torchax: {INSTALL}"
            ) from error
        if params is None:
            model = AutoModel.from_pretrained(checkpoint, dtype=getattr(torch, param_dtype))
        else:
            # Supplied weights replace every tensor, so the tower is built
            # without storage.
            with torch.device("meta"):
                model = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint))
        if model.config.is_encoder_decoder:
            model = model.get_encoder()
        model.eval()
        for layer in model.modules():
            if parametrize.is_parametrized(layer):
                for name in list(layer.parametrizations):
                    parametrize.remove_parametrizations(layer, name)
        module = JittableModule(model)
        if params is None:
            params = {"params": {name: host_view(leaf, name) for name, leaf in module.params.items()},
                      "buffers": {name: host_view(leaf, name)
                                  for name, leaf in extract_all_buffers(model)[1].items()}}
        # Torch's copies are no longer read: the forward runs on the arrays above.
        torch.nn.Module.to(model, "meta")
        return cls(checkpoint=checkpoint, seconds=seconds, module=module, params=params,
                   audio=AutoAudioProcessor(modelname=checkpoint), dtype=resolve_dtype(dtype),
                   param_dtype=param_dtype)

    @property
    def input_name(self) -> str:
        """The name of the extractor's array that the model reads."""
        return self.audio.processor.model_input_names[0]

    @property
    def samples(self) -> int:
        """The number of samples every waveform is cut or padded to."""
        return round(self.seconds * self.audio.sampling_rate)

    def waveform(self, row: AudioRow) -> np.ndarray:
        """Return one row as a float32 waveform `samples` long.

        A constant fills the whole clip, and a mapping holds the waveform or
        the constant under `"audio"`. Text raises ValueError.
        """
        value = row.get("audio") if isinstance(row, Mapping) else row
        if value is None:
            raise ValueError("an audio conditioning record holds its waveform under 'audio'")
        if isinstance(value, str):
            raise ValueError(f"hf_audio encodes waveforms, and was handed the text {value!r}")
        if isinstance(value, (int, float)):
            return np.full(self.samples, value, np.float32)
        wave = np.asarray(value, np.float32).reshape(-1)[:self.samples]
        return np.pad(wave, (0, self.samples - len(wave)))

    def tokenize(self, texts: Sequence[AudioRow]) -> dict[str, np.ndarray]:
        features = self.audio([self.waveform(row) for row in texts])
        return {self.input_name: np.asarray(features[self.input_name], np.float32)}

    def encode(self, params, tokens) -> TextContext:
        import torchax

        # The tower and its input run in one dtype: the compute dtype, or
        # float32 when none is set, whatever the weights are stored in.
        compute = jnp.float32 if self.dtype is None else self.dtype

        def in_compute(leaf):
            value = jnp.asarray(leaf)
            return value.astype(compute) if jnp.issubdtype(value.dtype, jnp.floating) else value

        weights = {name: in_compute(leaf) for name, leaf in params["params"].items()}
        buffers = {name: jnp.asarray(leaf) for name, leaf in params.get("buffers", {}).items()}
        env = torchax.default_env()
        arguments = env.j2t_iso((weights, buffers, in_compute(tokens[self.input_name])))
        with env:
            hidden = self.module.functional_call(
                partial(_last_hidden_state, key=self.input_name), *arguments)
        hidden = env.t2j_iso(hidden)
        return TextContext(hidden=hidden, mask=jnp.ones(hidden.shape[:2], jnp.int32))

    def to_json(self) -> dict:
        return {"checkpoint": self.checkpoint, "seconds": self.seconds,
                "dtype": dtype_name(self.dtype), "param_dtype": self.param_dtype}


__all__ = ["CLIPText", "CharTable", "ConditionEncoder", "HFAudio", "T5Text", "rebuild"]
