"""Native CLIP conditioning and image preprocessing for latent diffusion."""
from __future__ import annotations

import html
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dew.diffusion.process import DenoisingCondition
from dew.inputs.encoders import ConditionEncoder
from dew.nn.blocks import torch_nearest_resize
from dew.nn.safety import CLIPSafetyHead
from dew.nn.text_encoders import CLIPTextTransformer, T5EncoderTransformer
from dew.objectives.base import Variables
from dew.registry import dtype_name

if TYPE_CHECKING:
    from transformers import CLIPTokenizer, PreTrainedTokenizerBase

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.training.distributed import Layout, MeshSpec


def _prompt(record: Mapping[str, object], key: str, default: str) -> str:
    """One text slot of a conditioning record."""
    text = record.get(key, default)
    if not isinstance(text, str):
        raise ValueError(f"A text-conditioning record's {key} must be a string")
    return text


def _row_guidance(record: Mapping[str, object], default: float | None) -> float:
    """The guidance a row is walked at.

    A record's own where it names one, and the checkpoint's pipeline default
    otherwise. A model that reads no guidance (`default` None) refuses a
    record that names one.
    """
    value = record.get("guidance")
    if value is None:
        return 0.0 if default is None else default
    if default is None:
        raise ValueError("This checkpoint's model reads no guidance value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError("A record's guidance must be a finite number")
    return float(value)


def latent_image_conditions(autoencoder, params, pixels, mask, key):
    """Turns normalized pixels and a binary pixel mask into native UNet inputs.

    Returns mask [B,h,w,1] and masked_image [B,h,w,C] under the model keyword
    names. Spatial inputs stay present in both guidance branches. The mask
    shrinks to the latent grid by torch's nearest rule, as diffusers'
    inpainting pipelines shrink it (`prepare_mask_latents`).
    """
    mask = jnp.asarray(mask, jnp.float32)
    if mask.shape != (*pixels.shape[:-1], 1):
        raise ValueError("Image and mask geometry must match, with one mask channel")
    mask = (mask >= 0.5).astype(jnp.float32)
    latent = autoencoder.encode(params, pixels * (mask < 0.5), key)
    mask = torch_nearest_resize(mask, *latent.shape[1:3])
    return {"mask": mask, "masked_image": latent}



class _TextFeatures(NamedTuple):
    """The three states a family reads off one CLIP tower."""

    last: jax.Array
    penultimate: jax.Array
    pooled: jax.Array


def _text_features(tower, ids):
    """The CLIP tower's last, penultimate and pooled states for `ids`.

    The layers run here rather than through the tower's own call because the
    families that read the penultimate hidden states need the value from
    before the last layer, which a plain forward does not keep.

    The pooled slot is the tower's own: the checkpoints whose eos id is 2
    take the largest id in the row, which is that eos, and the rest take the
    first position that equals the eos id.
    """
    hidden = tower.token_embedding(ids) + tower.position_embedding(jnp.arange(ids.shape[1]))
    penultimate = hidden  # the state before the last layer; the embedding when there is none
    for layer in tower.layers:
        penultimate = hidden
        hidden = layer(hidden)
    hidden = tower.final_layer_norm(hidden)
    index = jnp.argmax(ids if tower.eos_token_id == 2 else ids == tower.eos_token_id, axis=-1)
    return _TextFeatures(hidden, penultimate, hidden[jnp.arange(ids.shape[0]), index])


Composition = Literal["clip", "clip_pooled", "sd3", "flux"]
"""How a checkpoint family composes its text towers into one conditioning.

- `clip`: one CLIP tower's last hidden states, which is Stable Diffusion 1
  and 2.
- `clip_pooled`: two towers' penultimate states concatenated along the width,
  the second tower's projected pooled vector and the size/crop time ids,
  which is SDXL and its refiner.
- `sd3`: the same width concatenation padded out to the T5 width, with the T5
  tower's final states concatenated along the sequence, and BOTH towers'
  projected pooled vectors concatenated as the pooled one.
- `flux`: the T5 tower's final states alone, with the first tower's pooled
  vector unprojected and no CLIP tokens in the sequence.
"""


@dataclass(frozen=True, eq=False)
class T5Segment:
    """Holds the T5 tower a family reads beside its CLIP ones.

    The fields are the tower, its tokenizer, the component name its
    parameters and tokenizer live under, and the sequence budget its pipeline
    pads to. That name is an SD3 directory's third text encoder or a Flux
    directory's second.
    """

    tower: T5EncoderTransformer
    tokenizer: PreTrainedTokenizerBase
    name: str
    tokens: int


@dataclass(eq=False)
class DiffusionConditioner(ConditionEncoder[str | Mapping[str, object]]):
    """Encodes text into the conditioning a published latent diffusion checkpoint reads.

    This one encoder handles every family's `composition` (`clip`,
    `clip_pooled`, `sd3` or `flux`). The composition decides which towers
    run, which of their states the model reads, and how the pooled vector is
    built. The towers are the native CLIP and T5 towers, called the way their
    own source pipelines call them. The SD3 and Flux pipelines pass their T5
    ids with no attention mask. `T5EncoderTransformer` called without a mask
    does the same, so no generic T5 default changes for these families.
    """

    towers: tuple[CLIPTextTransformer, ...]
    tokenizers: tuple[CLIPTokenizer, ...]
    names: tuple[str, ...]
    params: Variables
    checkpoint: str
    height: int
    width: int
    context_width: int
    """The width of the token sequence the denoiser reads: the UNet's
    `cross_attention_dim`, the joint transformers' `joint_attention_dim`. The
    SD3 composition pads its CLIP states out to this width, and the zero
    segment its pipeline writes for an absent third encoder has this width
    too."""
    composition: Composition = "clip"
    aesthetics: bool = False
    t5: T5Segment | None = None
    guidance: float | None = None
    param_dtype: str = "float32"
    keyword: ClassVar[str] = "conditioning"

    @classmethod
    def from_pretrained(cls, checkpoint: str, *, dtype: str | None = "bfloat16",
                        param_dtype: str = "float32", revision: str | None = None,
                        attention_impl: str = "auto", params: Variables | None = None,
                        mesh: MeshSpec | None = None, layout: Layout | None = None):
        from dew.interop.pretrained import load_diffusion_conditioner

        return load_diffusion_conditioner(checkpoint, cls, dtype=dtype, param_dtype=param_dtype,
                                          revision=revision, attention_impl=attention_impl, params=params,
                                          mesh=mesh, layout=layout)

    @property
    def stacked(self) -> bool:
        """Whether the CLIP ids share one array with a tower axis.

        Every family except plain Stable Diffusion stacks them. The XL
        refiner stacks its single tower too, because its pipeline still
        writes a tower's row and not a bare batch.
        """
        return self.composition != "clip"

    def __post_init__(self):
        towers, t5 = len(self.towers), self.t5 is not None
        if towers != len(self.tokenizers) or towers != len(self.names):
            raise ValueError("Every text tower needs its own tokenizer and its own name")
        expected = {"clip": (1, 1), "clip_pooled": (1, 2), "sd3": (2, 2), "flux": (1, 1)}[
            self.composition]
        if not expected[0] <= towers <= expected[1]:
            raise ValueError(f"{self.composition} conditioning runs "
                             f"{'-'.join(str(bound) for bound in sorted(set(expected)))} "
                             f"CLIP towers, not {towers}")
        if self.guidance is not None and self.composition != "flux":
            raise ValueError(f"{self.composition} conditioning carries no guidance input")
        if self.composition not in ("sd3", "flux"):
            if t5:
                raise ValueError(f"{self.composition} conditioning has no T5 segment")
        elif self.composition == "flux" and not t5:
            # SD3's pipeline writes a zero segment for an absent third encoder;
            # Flux's `_get_t5_prompt_embeds` has no such path.
            raise ValueError("Flux conditioning needs its T5 encoder")

    def tokenize(self, texts: Sequence[str | Mapping[str, object]]):
        """Tokenize one row per item, sending each text slot to the tower whose source pipeline reads it.

        An item is a prompt string or a mapping of slots, and a missing
        `second` or `third` slot repeats `text`. `text` goes to the first CLIP
        tower and `second` to the second one. The T5 tower reads its own
        slot, which is `third` when the family has two CLIP towers beside it
        and `second` when it has one.
        """
        rows, second, third, zero, negative, guidance = [], [], [], [], [], []
        for prompt in texts:
            record: Mapping[str, object] = {"text": prompt} if isinstance(prompt, str) else prompt
            text = _prompt(record, "text", "")
            rows.append(text)
            second.append(_prompt(record, "second", text))
            third.append(_prompt(record, "third", text))
            zero.append(bool(record.get("zero", False)))
            negative.append(bool(record.get("negative", False)))
            guidance.append(_row_guidance(record, self.guidance))
        ids = [tokenizer(second if index == 1 else rows, padding="max_length",
                         max_length=tokenizer.model_max_length, truncation=True,
                         return_tensors="np").input_ids
               for index, tokenizer in enumerate(self.tokenizers)]
        tokens = {"input_ids": np.stack(ids, axis=1) if self.stacked else ids[0],
                  "zero_condition": np.asarray(zero, bool), "negative": np.asarray(negative, bool)}
        if self.guidance is not None:
            tokens["guidance"] = np.asarray(guidance, np.float32)
        if self.t5 is not None:
            tokens["t5_input_ids"] = self.t5.tokenizer(
                third if self.composition == "sd3" else second, padding="max_length",
                max_length=self.t5.tokens, truncation=True, add_special_tokens=True,
                return_tensors="np").input_ids
        return tokens

    def time_ids(self, count, dtype):
        """Return SDXL's micro-conditioning for `count` rows.

        Each row is the original size, a zero crop offset, then the target
        size or, for the refiner, its aesthetic score.
        """
        size = (self.height, self.width)
        values = (*size, 0, 0, *((6.0,) if self.aesthetics else size))
        return jnp.broadcast_to(jnp.asarray(values, dtype), (count, len(values)))

    def _clip(self, params, ids) -> list[_TextFeatures]:
        outputs = []
        for index, (name, tower) in enumerate(zip(self.names, self.towers, strict=True)):
            output = tower.apply({"params": params[name]["text_model"]},
                                 ids[:, index] if self.stacked else ids, method=_text_features)
            assert isinstance(output, _TextFeatures)
            outputs.append(output)
        return outputs

    def _projected(self, params, name: str, pooled):
        return pooled @ params[name]["text_projection"]["kernel"]

    def _t5_states(self, params, tokens, rows: int, dtype) -> jax.Array:
        """The T5 segment, or the zero segment SD3 writes without a third encoder.

        That zero segment is `tokenizer_max_length` long, which is the CLIP
        tokenizer's window and not the T5 sequence the call asked for.
        """
        if self.t5 is None:
            return jnp.zeros((rows, self.tokenizers[0].model_max_length, self.context_width), dtype)
        # The SD3 and Flux pipelines call their T5 encoder with ids only.
        states = self.t5.tower.apply({"params": params[self.t5.name]},
                                     jnp.asarray(tokens["t5_input_ids"]))
        assert isinstance(states, jax.Array)
        return states

    def _zeroed(self, value, zero):
        """A dropped row's conditioning is zeros, in the shape it arrives."""
        return jnp.where(zero.reshape(zero.shape + (1,) * (value.ndim - 1)),
                         jnp.zeros_like(value), value)

    def encode(self, params, tokens) -> DenoisingCondition:
        ids = tokens["input_ids"]
        rows, zero = ids.shape[0], jnp.asarray(tokens["zero_condition"])
        outputs = self._clip(params, ids)
        if self.composition == "clip":
            return DenoisingCondition(self._zeroed(outputs[0].last, zero))
        if self.composition == "clip_pooled":
            hidden = jnp.concatenate([value.penultimate for value in outputs], axis=-1)
            pooled = self._projected(params, self.names[-1], outputs[-1].pooled)
            time_ids = tokens.get("time_ids")
            if time_ids is None:
                time_ids = self.time_ids(rows, hidden.dtype)
                if self.aesthetics:
                    # The refiner pipeline's own aesthetic scores: 6.0 for a
                    # positive prompt and 2.5 for a negative one.
                    time_ids = time_ids.at[:, -1].set(jnp.where(tokens["negative"], 2.5, 6.0))
            return DenoisingCondition(self._zeroed(hidden, zero),
                                      self._zeroed(pooled, zero), time_ids)
        if self.composition == "flux":
            hidden = self._t5_states(params, tokens, rows, outputs[0].pooled.dtype)
            # Flux reads the CLIP pooled vector unprojected.
            pooled = outputs[0].pooled
            guidance = (None if self.guidance is None
                        else jnp.asarray(tokens["guidance"], hidden.dtype))
            return DenoisingCondition(self._zeroed(hidden, zero),
                                      self._zeroed(pooled, zero), guidance=guidance)
        # sd3: the CLIP states padded out to the joint width, the T5 segment
        # after them along the sequence, and both projected pooled vectors.
        clip = jnp.concatenate([value.penultimate for value in outputs], axis=-1)
        padded = jnp.pad(clip, ((0, 0), (0, 0), (0, self.context_width - clip.shape[-1])))
        hidden = jnp.concatenate(
            [padded, self._t5_states(params, tokens, rows, clip.dtype)], axis=1)
        pooled = jnp.concatenate(
            [self._projected(params, name, value.pooled)
             for name, value in zip(self.names, outputs, strict=True)], axis=-1)
        return DenoisingCondition(self._zeroed(hidden, zero), self._zeroed(pooled, zero))

    def captions(self, tokens):
        ids = tokens["input_ids"][:, 0] if self.stacked else tokens["input_ids"]
        return tuple(self.tokenizers[0].batch_decode(np.asarray(ids), skip_special_tokens=True))

    def to_json(self):
        return {"checkpoint": self.checkpoint, "dtype": dtype_name(self.towers[0].dtype),
                "param_dtype": self.param_dtype}

    def save_assets(self, destination: Path) -> None:
        """Write into `destination` the tokenizer files an exported directory keeps beside the weights."""
        for name, tokenizer in zip(self.names, self.tokenizers, strict=True):
            folder = destination / ("tokenizer" + name.removeprefix("text_encoder"))
            tokenizer.save_pretrained(folder)
            # A CLIP tokenizer's own vocabulary and merges beside its config.
            tokenizer.backend_tokenizer.model.save(str(folder))
        if self.t5 is not None:
            # The T5 tokenizer ships one file, which `save_pretrained` writes, in
            # the slot its own family keeps it.
            self.t5.tokenizer.save_pretrained(
                destination / ("tokenizer" + self.t5.name.removeprefix("text_encoder")))


def _residual_states(decoder, ids):
    """The last decoder layer's output, before the final norm.

    `QwenImage21Pipeline` reads `hidden_states[-1]` with the norm hooked out,
    since that is what the transformer was trained on. The layers read the
    embeddings the decoder's forward gives them, scaled by its own
    `scaled_embeddings`; the decoders this reads run one residual stream.
    """
    return decoder.stack(decoder.scaled_embeddings(decoder.token_embeddings(ids)), train=False,
                         decode=False, positions=None, segment_ids=None, per_layer_input=None)


@dataclass(eq=False)
class _LanguageText(ConditionEncoder[str | Mapping[str, object]]):
    """What the conditioners of a language-model or UMT5 text tower share: the
    tower and its tokenizer, loaded from the pipeline's directory `checkpoint`
    at a prompt budget of `tokens`, which the record keeps, and the tokenizer's
    files, copied from its `assets` folder as they came: they are read, never
    trained."""

    tower: CausalTransformer | T5EncoderTransformer
    tokenizer: PreTrainedTokenizerBase
    params: Variables
    checkpoint: str
    tokens: int = field(default=512, kw_only=True)
    param_dtype: str = field(default="float32", kw_only=True)
    keyword: ClassVar[str] = "conditioning"
    assets: ClassVar[str] = "tokenizer"

    @classmethod
    def from_pretrained(cls, checkpoint: str, *, dtype: str | None = "bfloat16",
                        param_dtype: str = "float32", revision: str | None = None,
                        attention_impl: str = "auto", tokens: int = 512,
                        params: Variables | None = None, mesh: MeshSpec | None = None,
                        layout: Layout | None = None):
        from dew.interop.pretrained import load_diffusion_conditioner

        return load_diffusion_conditioner(checkpoint, cls, dtype=dtype, param_dtype=param_dtype,
                                          revision=revision, attention_impl=attention_impl,
                                          tokens=tokens, params=params, mesh=mesh, layout=layout)

    def captions(self, tokens):
        return tuple(self.tokenizer.batch_decode(np.asarray(tokens["input_ids"]), skip_special_tokens=True))

    def to_json(self):
        return {"checkpoint": self.checkpoint, "dtype": dtype_name(self.tower.dtype),
                "param_dtype": self.param_dtype, "tokens": self.tokens}

    def save_assets(self, destination: Path) -> None:
        shutil.copytree(Path(self.checkpoint) / self.assets, destination / self.assets, dirs_exist_ok=True)


@dataclass(eq=False)
class QwenImageConditioner(_LanguageText):
    """The text conditioning of a Qwen-Image 2.1 checkpoint: its Qwen3-VL
    encoder's language model over the pipeline's text-to-image template.

    `QwenImage21Pipeline._get_qwen_prompt_embeds` formats each prompt into
    its template, reads the last decoder layer's output before the final
    norm, and drops the system turn's tokens. A prompt with no image puts the
    encoder's three rotary axes at one position, so its interleaved sections
    rotate as the plain one-axis table `tower` applies.

    Rows are padded on the right to the system turn plus `tokens`. The
    encoder is causal, so a real token never reads a later pad and its state
    is the one the pipeline's left padding gives; the condition's `mask`
    marks those tokens and the pads carry zeros, as the pipeline's stacking
    writes them. An empty prompt is the single space the pipeline encodes in
    its place. A prompt past the budget is refused rather than cut, since
    the template's closing turn would go with it.
    """

    height: int
    width: int
    drop: int = field(init=False)
    """The system turn's token count, which the pipeline derives the same way."""

    SYSTEM: ClassVar[str] = "Comprehend and analyze the provided prompt."
    USER: ClassVar[tuple[str, str]] = ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")
    assets = "processor"

    def __post_init__(self):
        system = [{"role": "system", "content": [{"type": "text", "text": self.SYSTEM}]}]
        self.drop = len(self.tokenizer.apply_chat_template(system, tokenize=True, return_dict=False))

    def tokenize(self, texts: Sequence[str | Mapping[str, object]]):
        system = f"<|im_start|>system\n{self.SYSTEM}<|im_end|>\n"
        rows = []
        for prompt in texts:
            record: Mapping[str, object] = {"text": prompt} if isinstance(prompt, str) else prompt
            rows.append(f"{system}{self.USER[0]}{_prompt(record, 'text', '') or ' '}{self.USER[1]}")
        length = self.drop + self.tokens
        encoded = self.tokenizer(rows, padding="max_length", padding_side="right",
                                 max_length=length)
        if any(len(ids) > length for ids in encoded.input_ids):
            raise ValueError(f"A prompt runs past the {self.tokens}-token budget; raise `tokens`")
        return {"input_ids": np.asarray(encoded.input_ids, np.int32),
                "attention_mask": np.asarray(encoded.attention_mask, np.int32)}

    def encode(self, params, tokens) -> DenoisingCondition:
        states = jnp.asarray(self.tower.apply({"params": params["text_encoder"]["params"]},
                                              jnp.asarray(tokens["input_ids"]), method=_residual_states))
        valid = jnp.asarray(tokens["attention_mask"], bool)[:, self.drop:]
        context = jnp.where(valid[..., None], states[:, self.drop:], 0)
        return DenoisingCondition(context, mask=valid)

    def captions(self, tokens):
        texts = self.tokenizer.batch_decode(np.asarray(tokens["input_ids"])[:, self.drop:],
                                            skip_special_tokens=True)
        return tuple(text.removeprefix("user\n").removesuffix("\nassistant\n") for text in texts)


@dataclass(eq=False)
class HiddenStatesConditioner(_LanguageText):
    """Text conditioning read off a language model's hidden states: FLUX.2's
    and Z-Image's.

    Each pipeline formats a prompt with the encoder's chat template, pads
    every row on the right to `tokens`, runs the encoder with the padding
    mask, and concatenates, for each token, `hidden_states[k]` for k in
    `layers`: the output of decoder layer k, the embeddings being k = 0.
    `Flux2Pipeline` (a Mistral-3 encoder, `template="mistral3"`) writes a
    system turn and a user turn and stacks layers 10, 20 and 30;
    `Flux2KleinPipeline` (Qwen3, `template="qwen3"`) a user turn and the
    assistant's opening with thinking off, layers 9, 18 and 27;
    `ZImagePipeline` the same with `thinking` on, and reads the one layer
    before the last. `mask` marks the real tokens: FLUX.2's transformer
    reads every row whole, pads and their states included, and Z-Image's
    only the real tokens. A prompt past the budget is refused rather than
    cut, since the template's closing turn would go with it.
    """

    height: int
    width: int
    template: Literal["mistral3", "qwen3"]
    layers: tuple[int, ...]
    thinking: bool = False
    guidance: float | None = None
    """The distilled guidance FLUX.2 [dev] embeds, its pipeline's default;
    None for a transformer that embeds none."""

    SYSTEM: ClassVar[str] = (
        "You are an AI that reasons about image descriptions. You give structured responses "
        "focusing on object "
        "relationships, object\nattribution and actions without speculation."
    )
    """FLUX.2's `SYSTEM_MESSAGE`, from black-forest-labs/flux2 at 5a5d316b."""

    def _conversation(self, prompt: str) -> tuple[list[dict], dict]:
        if self.template == "qwen3":
            return ([{"role": "user", "content": prompt}],
                    {"add_generation_prompt": True, "enable_thinking": self.thinking})
        return ([{"role": "system", "content": [{"type": "text", "text": self.SYSTEM}]},
                 {"role": "user", "content": [{"type": "text", "text": prompt.replace("[IMG]", "")}]}],
                {"add_generation_prompt": False})

    def tokenize(self, texts: Sequence[str | Mapping[str, object]]):
        rows, guidance = [], []
        for prompt in texts:
            record: Mapping[str, object] = {"text": prompt} if isinstance(prompt, str) else prompt
            conversation, options = self._conversation(_prompt(record, "text", ""))
            rows.append(self.tokenizer.apply_chat_template(conversation, tokenize=False, **options))
            guidance.append(_row_guidance(record, self.guidance))
        # The template writes the special tokens itself: [dev]'s tokenizing
        # `apply_chat_template` has the tokenizer add none, and the Qwen
        # tokenizers [klein] and Z-Image call add none of their own.
        encoded = self.tokenizer(rows, padding="max_length", padding_side="right", max_length=self.tokens,
                                 add_special_tokens=False)
        if any(len(ids) > self.tokens for ids in encoded.input_ids):
            raise ValueError(f"A prompt runs past the {self.tokens}-token budget; raise `tokens`")
        tokens: dict[str, np.ndarray] = {"input_ids": np.asarray(encoded.input_ids, np.int32),
                                         "attention_mask": np.asarray(encoded.attention_mask, np.int32)}
        if self.guidance is not None:
            tokens["guidance"] = np.asarray(guidance, np.float32)
        return tokens

    def encode(self, params, tokens) -> DenoisingCondition:
        from dew.nn.backbones.causal_transformer import INTERMEDIATES, layer_output, layer_outputs

        _, kept = self.tower.apply(
            {"params": params["text_encoder"]["params"]}, jnp.asarray(tokens["input_ids"]),
            attention_mask=jnp.asarray(tokens["attention_mask"], bool), method="hidden_states",
            capture_intermediates=layer_outputs, mutable=[INTERMEDIATES])
        states = [layer_output(kept[INTERMEDIATES], layer - 1) for layer in self.layers]
        guidance = None if self.guidance is None else jnp.asarray(tokens["guidance"], jnp.float32)
        return DenoisingCondition(jnp.concatenate(states, axis=-1),
                                  mask=jnp.asarray(tokens["attention_mask"], bool), guidance=guidance)


@dataclass(eq=False)
class WanConditioner(_LanguageText):
    """The text conditioning of a Wan 2.1 checkpoint: its UMT5 encoder's last
    hidden states.

    `WanPipeline._get_t5_prompt_embeds` cleans each prompt as its
    `prompt_clean` does: ftfy's repairs, HTML entities unescaped twice, then
    every run of whitespace one space. The source collapses with the `regex`
    module's `\\s`, Unicode's White_Space; the standard library's also takes
    U+001C to U+001F, which ftfy has already removed as control characters,
    so the two agree here. It tokenizes the prompt with its end-of-sequence
    token, padded and cut to `tokens`, runs the encoder under the padding
    mask, keeps each row's states up to its own token count and fills the
    rest of the row with zeros. Its transformer reads every position, the
    zeros included, so the condition carries no mask.
    """

    def tokenize(self, texts: Sequence[str | Mapping[str, object]]):
        try:
            import ftfy
        except ImportError as missing:  # the wan extra's one package
            raise ValueError("Wan's pipelines clean each prompt with ftfy; install it with "
                             "`pip install 'dewml[wan]'`") from missing
        rows = []
        for prompt in texts:
            record: Mapping[str, object] = {"text": prompt} if isinstance(prompt, str) else prompt
            text = html.unescape(html.unescape(ftfy.fix_text(_prompt(record, "text", "")))).strip()
            rows.append(re.sub(r"\s+", " ", text).strip())
        encoded = self.tokenizer(rows, padding="max_length", max_length=self.tokens, truncation=True,
                                 add_special_tokens=True, return_tensors="np")
        return {"input_ids": np.asarray(encoded.input_ids, np.int32),
                "attention_mask": np.asarray(encoded.attention_mask, np.int32)}

    def encode(self, params, tokens) -> DenoisingCondition:
        mask = jnp.asarray(tokens["attention_mask"])
        states = jnp.asarray(self.tower.apply({"params": params["text_encoder"]},
                                              jnp.asarray(tokens["input_ids"]), mask))
        lengths = jnp.sum(mask != 0, axis=1)
        kept = jnp.arange(states.shape[1])[None, :, None] < lengths[:, None, None]
        return DenoisingCondition(jnp.where(kept, states, 0))


@lru_cache(maxsize=32)
def _cubic_weights(source: int, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Compact Pillow bicubic taps, with its 22-bit integer coefficients."""
    scale = source / target
    support = 2.0 * max(scale, 1.0)
    taps = int(np.ceil(2 * support)) + 1
    indices = np.zeros((target, taps), np.int32)
    weights = np.zeros((target, taps), np.int32)
    for row in range(target):
        center = (row + 0.5) * scale
        left = max(int(center - support + 0.5), 0)
        right = min(int(center + support + 0.5), source)
        positions = (np.arange(left, right) - center + 0.5) / max(scale, 1.0)
        x = np.abs(positions)
        kernel = np.where(x < 1, (1.5 * x - 2.5) * x * x + 1,
                          np.where(x < 2, ((-0.5 * x + 2.5) * x - 4) * x + 2, 0.0))
        normalized = kernel / kernel.sum() * (1 << 22)
        indices[row, :right - left] = np.arange(left, right)
        weights[row, : right - left] = np.trunc(normalized + np.where(normalized >= 0, 0.5, -0.5)).astype(
            np.int32
        )
    return indices, weights


@dataclass(frozen=True)
class CLIPImageTransform:
    """Runs the published CLIP preprocessing as JAX arithmetic over uint8 NHWC
    pixels.

    The reference is the PIL image processor: `CLIPImageProcessor` as
    transformers 4 ran it, which the safety fixtures were generated with
    (tools/diffusers_pipeline_reference.py), and which transformers 5 ships
    as `CLIPImageProcessorPil`. Transformers 5's `CLIPImageProcessor`
    resamples through torchvision instead and lands up to one uint8 level
    away.
    """
    size: int | tuple[int, int]
    crop: tuple[int, int]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    rescale: float = 1 / 255
    resize: bool = True
    center_crop: bool = True
    normalize: bool = True

    @classmethod
    def from_config(cls, config):
        if config.get("resample", 3) != 3:
            raise ValueError("Only bicubic CLIP preprocessing is implemented")
        size = config.get("size", {"shortest_edge": 224})
        if isinstance(size, dict):
            size = size["shortest_edge"] if "shortest_edge" in size else (size["height"], size["width"])
        crop = config.get("crop_size", {"height": 224, "width": 224})
        crop = (crop, crop) if isinstance(crop, int) else (crop["height"], crop["width"])
        return cls(
            size,
            crop,
            tuple(config.get("image_mean", (0.48145466, 0.4578275, 0.40821073))),
            tuple(config.get("image_std", (0.26862954, 0.26130258, 0.27577711))),
            config.get("rescale_factor", 1 / 255) if config.get("do_rescale", True) else 1.0,
            config.get("do_resize", True),
            config.get("do_center_crop", True),
            config.get("do_normalize", True),
        )

    def __call__(self, pixels):
        pixels = jnp.asarray(pixels, jnp.float32)
        _, height, width, _ = pixels.shape
        if self.resize:
            if isinstance(self.size, int):
                # transformers' get_resize_output_image_size: the shorter side
                # takes the size exactly, the longer int(size * long / short).
                if width <= height:
                    target = (int(self.size * height / width), self.size)
                else:
                    target = (self.size, int(self.size * width / height))
            else:
                target = self.size
            row_indices, rows = (jnp.asarray(value) for value in _cubic_weights(height, target[0]))
            column_indices, columns = (jnp.asarray(value) for value in _cubic_weights(width, target[1]))
            # Each integer pass rounds and clamps before the next pass, as
            # Pillow does. Sparse taps avoid a dense HxW resampling matrix.
            pixels = pixels.astype(jnp.int32)
            horizontal = jnp.sum(
                jnp.take(pixels, column_indices, axis=2) * columns[None, None, :, :, None], axis=3
            )
            pixels = jnp.clip((horizontal + (1 << 21)) >> 22, 0, 255)
            vertical = jnp.sum(jnp.take(pixels, row_indices, axis=1) * rows[None, :, :, None, None], axis=2)
            pixels = jnp.clip((vertical + (1 << 21)) >> 22, 0, 255).astype(jnp.float32)
        if self.center_crop:
            crop_h, crop_w = self.crop
            pad_h, pad_w = max(0, crop_h - pixels.shape[1]), max(0, crop_w - pixels.shape[2])
            if pad_h or pad_w:
                # transformers' center_crop puts the odd pad row on top and the
                # odd pad column on the left, ceil((crop - side) / 2).
                pixels = jnp.pad(pixels, ((0, 0), (pad_h - pad_h // 2, pad_h // 2),
                                          (pad_w - pad_w // 2, pad_w // 2), (0, 0)))
            top, left = (pixels.shape[1] - crop_h) // 2, (pixels.shape[2] - crop_w) // 2
            pixels = pixels[:, top:top + crop_h, left:left + crop_w]
        pixels = pixels * self.rescale
        if self.normalize:
            pixels = (pixels - jnp.asarray(self.mean, jnp.float32)) / jnp.asarray(self.std, jnp.float32)
        return pixels.transpose(0, 3, 1, 2)


@dataclass(frozen=True, eq=False)
class ImageSafety:
    """The checkpoint's frozen safety head over decoded images, under jit."""
    model: CLIPSafetyHead
    transform: CLIPImageTransform

    def __call__(self, variables, images):
        pixels = jnp.clip(jnp.round((images + 1) * 127.5), 0, 255)
        flagged = self.model.apply({"params": variables["encoders"]["safety"]}, self.transform(pixels))
        assert isinstance(flagged, jax.Array)
        return jnp.where(flagged[:, None, None, None], -jnp.ones_like(images), images)
