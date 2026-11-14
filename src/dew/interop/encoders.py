"""The condition encoders that bind a released checkpoint: CLIP's and T5's text towers (`CLIPText`,
`T5Text`) and transformers' audio models (`HFAudio`), each built by its `from_pretrained` from the
checkpoint it names. They are dew.interop's, where binding a checkpoint belongs; the checkpoint-free
encoders are dew.inputs' (`dew.inputs.encoders`), and a run names either by its registry entry
(`dew.registry.encoders`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype

from dew.batch import FROZEN, Variables
from dew.inputs.condition import ConditionEncoder
from dew.inputs.encoders import AudioRow
from dew.interop.text_encoders import (
    _t5_path,
    _text_path,
    bind_text_tower,
    t5_embedding,
    translate_config,
    translate_t5_config,
)
from dew.nn.dit import TextContext
from dew.nn.flops import ConditionWork
from dew.nn.sharding import abstract
from dew.nn.text_checkpoints import DEFAULT_MODEL, DEFAULT_T5_MODEL
from dew.nn.text_encoders import CLIPTextTransformer, CLIPTowerOutput, T5EncoderTransformer
from dew.registry import dtype_name, resolve_dtype

if TYPE_CHECKING:
    from torchax.interop import JittableModule
    from transformers import PreTrainedTokenizerBase

    from dew.data.processors import FeatureExtractor
    from dew.data.text import PinnedTokenizer



def _pinned_tokenizer(checkpoint: str, revision: str | None) -> PinnedTokenizer:
    """A text tower's tokenizer files resolved once (`pin_tokenizer`): a local directory as it is,
    a Hub repo at `revision`, so the tokenizer read and its recorded hashes are one set of files."""
    from dew.data.text import pin_tokenizer

    hub = revision is not None and not Path(checkpoint).is_dir()
    return pin_tokenizer(f"{checkpoint}@{revision}" if hub else checkpoint)


@dataclass(frozen=True, eq=False)
class _TextTower(ConditionEncoder[str, TextContext]):
    """A pretrained text tower with the checkpoint's own tokenizer.

    `from_pretrained` builds the tower at `dtype` and stores its weights at
    `param_dtype`, FP32 masters by default whatever the checkpoint stores.
    Supplied `params` are bound unchanged, and only the configuration and
    the tokenizer are read.

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
    tokenizer_files: PinnedTokenizer
    """The files `tokenizer` was read from, resolved once (`_pinned_tokenizer`)."""
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

    def dependencies(self) -> Mapping[str, object]:
        """The encoder's fields and the hashes of the tokenizer files it reads, so a tokenizer
        that moved under an unpinned checkpoint name changes the identity too."""
        return {**super().dependencies(), "tokenizer": self.tokenizer_files.snapshot}

    def tokenize(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        tokens = self.tokenizer(list(texts), padding="max_length", max_length=self.context,
                                truncation=True, return_tensors="np")
        return {"input_ids": np.asarray(tokens["input_ids"], np.int32),
                "attention_mask": np.asarray(tokens["attention_mask"], np.int32)}

    def conditioning_flops(self, tokens) -> ConditionWork:
        return ConditionWork.of_tokens(self.transformer, tokens, self.transformer.emb_features)

    def drawn(self) -> Mapping:
        """The transformer drawn over a row of token ids and its mask, as `encode` reads it
        (`ConditionEncoder.drawn`)."""
        ids = jnp.zeros((1, 2), jnp.int32)
        return abstract(self.transformer.init, jax.random.key(0), ids, jnp.ones_like(ids))

    def encode(self, params, tokens) -> TextContext:
        mask = jnp.asarray(tokens["attention_mask"])
        answer = self.transformer.apply(params, jnp.asarray(tokens["input_ids"]), mask)
        return TextContext(hidden=self.states(answer), mask=mask)

    def captions(self, tokens) -> tuple[str, ...]:
        return tuple(self.tokenizer.batch_decode(
            np.asarray(tokens["input_ids"]), skip_special_tokens=True))

    def recorded_fields(self) -> dict:
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
        dtype = resolve_dtype(dtype)
        transformer, variables = bind_text_tower("CLIP text", translate_config, _text_path, checkpoint,
                                                 revision, params, dtype, param_dtype)
        pinned = _pinned_tokenizer(checkpoint, revision)
        return cls(checkpoint=checkpoint, transformer=transformer, params=variables,
                   tokenizer=pinned.transformers(), tokenizer_files=pinned,
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
        dtype = resolve_dtype(dtype)
        transformer, variables = bind_text_tower("T5", translate_t5_config, _t5_path, checkpoint, revision,
                                                 params, dtype, param_dtype, validate=t5_embedding)
        pinned = _pinned_tokenizer(checkpoint, revision)
        return cls(checkpoint=checkpoint, transformer=transformer, params=variables,
                   tokenizer=pinned.transformers(), tokenizer_files=pinned,
                   max_length=max_length, dtype=dtype, param_dtype=param_dtype, revision=revision)

    @property
    def context(self) -> int:
        return self.max_length

    @property
    def recorded(self) -> dict:
        return {"max_length": self.max_length}


def _last_hidden_state(model, features, *, key: str):
    """`model`'s last hidden state for its input `key`, run by torchax."""
    return model(**{key: features}).last_hidden_state


def _audio_files() -> tuple[str, ...]:
    """The files an audio tower's configuration and feature extractor are read from, as transformers names
    them: `CONFIG_NAME` and the versioned configurations beside it (`config.*.json`), of which
    `PreTrainedConfig.get_config_dict` reads one; `FEATURE_EXTRACTOR_NAME`; and `PROCESSOR_NAME`, whose
    nested extractor settings `FeatureExtractionMixin.get_feature_extractor_dict` reads first."""
    from transformers.utils import CONFIG_NAME, FEATURE_EXTRACTOR_NAME, PROCESSOR_NAME

    versioned = CONFIG_NAME.replace(".json", ".*.json")
    return (CONFIG_NAME, versioned, FEATURE_EXTRACTOR_NAME, PROCESSOR_NAME)


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
    from, and the checkpoint's feature extractor (`extractor`) reads them at
    its own sampling rate and padding, as Whisper's pads to 30 seconds. A
    run's batches carry the waveform under the condition's field (a
    clip's `audio`), which the objective encodes through `tokenize` on the host
    (`InputSpec.encode_fields`), so this encoder reads no captions.
    """

    checkpoint: str
    seconds: float
    module: JittableModule
    params: Variables
    extractor: FeatureExtractor
    files: Mapping[str, object]
    """The checkpoint's configuration and feature-extractor files the tower and `extractor` were read from,
    hashed (`dew.data.text.pin_files`), which its identity holds (`dependencies`)."""
    dtype: Dtype | None = None
    param_dtype: str = "float32"
    revision: str | None = None
    """The checkpoint's git revision, recorded so a rebuild reads the same weights and extractor."""
    parameter_collections: ClassVar[tuple[str, ...] | None] = ("params",)
    reads_captions: ClassVar[bool] = False

    @classmethod
    def from_pretrained(cls, checkpoint: str = "facebook/wav2vec2-base-960h", *, seconds: float = 1.0,
                        dtype=None, param_dtype: str = "float32", revision: str | None = None,
                        params: Variables | None = None) -> HFAudio:
        from dew.data.text import pin_files
        from dew.interop.pickles import host_view
        from dew.interop.torchax_fallback import INSTALL

        try:
            import torch
            from torch.nn.utils import parametrize
            from torchax.interop import JittableModule, extract_all_buffers
            from transformers import AutoConfig, AutoFeatureExtractor, AutoModel
        except ImportError as error:
            raise ImportError(
                f"hf_audio runs transformers' audio model through torchax: {INSTALL}"
            ) from error
        # The configuration and extractor read from one commit's files, whose hashes the identity holds,
        # and the weights from that commit.
        hub = revision is not None and not Path(checkpoint).is_dir()
        files, directory = pin_files(f"{checkpoint}@{revision}" if hub else checkpoint, _audio_files())
        if params is None:
            model = AutoModel.from_pretrained(checkpoint, revision=files["revision"],
                                              dtype=getattr(torch, param_dtype))
        else:
            # Supplied weights replace every tensor, so the tower is built
            # without storage.
            with torch.device("meta"):
                model = AutoModel.from_config(AutoConfig.from_pretrained(directory))
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
                   extractor=AutoFeatureExtractor.from_pretrained(directory), files=files,
                   dtype=resolve_dtype(dtype), param_dtype=param_dtype, revision=revision)

    @property
    def input_name(self) -> str:
        """The name of the extractor's array that the model reads."""
        return self.extractor.model_input_names[0]

    @property
    def samples(self) -> int:
        """The number of samples every waveform is cut or padded to."""
        from dew.data.processors import sampling_rate
        from dew.data.sources.av_utils import sample_count

        return sample_count(self.seconds, sampling_rate(self.extractor))

    def waveform(self, row: AudioRow | str) -> np.ndarray:
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
        from dew.data.processors import sampling_rate

        waveforms = [self.waveform(row) for row in texts]
        features = self.extractor(waveforms, sampling_rate=sampling_rate(self.extractor), return_tensors="np")
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

    def recorded_fields(self) -> dict:
        return {"checkpoint": self.checkpoint, "seconds": self.seconds,
                "dtype": dtype_name(self.dtype), "param_dtype": self.param_dtype, "revision": self.revision}

    def dependencies(self) -> Mapping[str, object]:
        """The encoder's fields and the hashes of the configuration and extractor files it reads, so an
        extractor that moved under an unpinned checkpoint name changes the identity too."""
        return {**super().dependencies(), "files": self.files}


__all__ = ["CLIPText", "HFAudio", "T5Text"]
