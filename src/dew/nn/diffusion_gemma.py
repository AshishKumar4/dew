"""DiffusionGemma's shared encoder and decoder, and its self-conditioning MLP.

The MLP follows the Transformers implementation: a scaled pre-norm, a gated
feed-forward and a post-norm without scale. The previous step's logits become
soft embeddings through an fp32 softmax against the scaled embedding table.
On the first inference step, an explicit self-conditioning mask zeros those
embeddings. The official SFT objective supplies zero logits for its dropout
branch instead, and zero logits give uniform soft embeddings, which are not a
zero signal.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike, VariableDict

from dew.interop.weights import ParamTree, translate_parameters
from dew.nn.attention import RMSNorm
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import DecoderBank
from dew.nn.moe import gated_product
from dew.nn.multimodal import VisionConditioner
from dew.nn.precision import at_least_fp32
from dew.nn.protocols import OutputTable
from dew.registry import models

if TYPE_CHECKING:
    from dew.records import JSON


# The layers follow Transformers' modeling_diffusion_gemma.py:790-823.
class SelfConditioning(nn.Module):
    """Folds the previous step's soft embeddings into the canvas embeddings."""

    hidden_size: int
    intermediate_size: int
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.pre_norm = RMSNorm(
            epsilon=self.norm_eps, dtype=self.dtype, name="pre_norm")
        self.post_norm = RMSNorm(
            epsilon=self.norm_eps, with_scale=False, dtype=self.dtype,
            name="post_norm")
        dense = functools.partial(nn.Dense, use_bias=False,
                                  dtype=self.dtype, precision=self.precision)
        self.gate_proj = dense(self.intermediate_size, name="gate_proj")
        self.up_proj = dense(self.intermediate_size, name="up_proj")
        self.down_proj = dense(self.hidden_size, name="down_proj")

    def __call__(self, inputs_embeds, signal):
        normed = self.pre_norm(signal)
        gated = self.down_proj(gated_product('geglu')(self.gate_proj(normed), self.up_proj(normed)))
        return self.post_norm(inputs_embeds + gated)


def soft_embeddings(logits: jax.typing.ArrayLike, embed_weight: jax.typing.ArrayLike,
                    scale: float) -> jax.Array:
    """Previous logits as soft embeddings: fp32 softmax against the table
    (the logits' own dtype where it is wider, `at_least_fp32`).

    The table is contracted in its stored dtype with fp32 accumulation, which
    is the upcast product up to summation order and materialises no fp32 copy
    of the vocabulary-sized table.
    """
    logits = jnp.asarray(logits)
    wide = at_least_fp32(logits.dtype)
    probs = jax.nn.softmax(logits.astype(wide), axis=-1)
    return jnp.einsum('...v,vd->...d', probs, jnp.asarray(embed_weight),
                      preferred_element_type=wide) * jnp.asarray(scale, wide)


@models("diffusion_gemma")
class DiffusionGemma(nn.Module):
    """Reads one text parameter tree causally for the context and bidirectionally for canvases.

    ``encode`` appends clean tokens to the cache. ``__call__`` refines a canvas
    against that frozen cache and feeds the previous logits through
    self-conditioning. ``hidden_states`` and ``logits`` read clean tokens
    through the same causal encoder over the whole sequence, with no cache.
    Each method is a separate apply. The encoder and the decoder share one
    scope, so their parameters are identical and no second tree is stored.
    """

    text: CausalTransformer
    canvas_length: int
    conditioner: VisionConditioner | None = None

    def setup(self):
        if not self.text.causal:
            raise ValueError("DiffusionGemma.text is the causal encoder view")
        width = self.text.mlp_features
        if not isinstance(width, int):
            raise ValueError("DiffusionGemma requires one intermediate MLP width")
        self.decoder = self.text.clone(causal=False)
        nn.share_scope(self.decoder, self.text)
        self.self_conditioning = SelfConditioning(
            hidden_size=self.text.emb_features, intermediate_size=width,
            norm_eps=self.text.norm_eps, dtype=self.text.dtype, precision=self.text.precision)

    @property
    def vocab_size(self) -> int:
        return self.text.vocab_size

    @property
    def bank_sites(self) -> tuple[DecoderBank, ...]:
        # decoder shares text's scope; it is another reader, not another bank.
        return tuple(DecoderBank(("text", *site.namespace), site.view, site.scanned)
                     for site in self.text.bank_sites)

    @property
    def max_seq_len(self) -> int:
        return self.text.max_seq_len

    @property
    def emb_features(self) -> int:
        return self.text.emb_features

    @property
    def dtype(self) -> Dtype | None:
        return self.text.dtype

    @property
    def precision(self) -> PrecisionLike:
        return self.text.precision

    @property
    def final_logit_softcap(self) -> float | None:
        return self.text.final_logit_softcap

    @property
    def causal(self) -> bool:
        """Whether `hidden_states` attend causally: they are the encoder's,
        which `setup` holds causal."""
        return self.text.causal

    @property
    def mask_token_id(self) -> int | None:
        return self.text.mask_token_id

    def with_trainable_layer_scalars(self) -> DiffusionGemma:
        """Return this model with its layer scalars as parameters, the model the published SFT trains.

        A source whose scalars are constants ("frozen") is read by it through
        `trainable_variables`. A model without layer scalars is refused.
        """
        if self.text.layer_scalar not in ("frozen", "trainable"):
            raise ValueError(f"layer_scalar is {self.text.layer_scalar!r}, so this model has no layer "
                             f"scalars to train; the published SFT trains a 'frozen' or 'trainable' one")
        return self.clone(text=self.text.clone(layer_scalar="trainable"))

    def trainable_variables(self, variables: VariableDict) -> VariableDict:
        """Return this model's `variables` as `with_trainable_layer_scalars()` reads them.

        Google's release makes each layer's scalar a parameter, and
        Transformers declares the same tensor a buffer, which a "frozen"
        model keeps under `constants`. Each moves to its layer under `params`,
        constants it leaves empty are dropped, and no array is copied. A
        "trainable" model's tree is returned as it is. A frozen model's tree
        that already holds a trainable scalar is refused.
        """
        if self.text.layer_scalar == "trainable":
            return variables
        if self.text.layer_scalar != "frozen":
            raise ValueError(f"layer_scalar is {self.text.layer_scalar!r}, so this model's tree holds "
                             f"no layer scalars to move")
        values = dict(variables)
        params = dict(values["params"])
        text = dict(params["text"])
        constants = dict(values["constants"])
        text_constants = dict(constants["text"])
        for index in range(self.text.num_layers):
            layer = f"layers_{index}"
            fixed = dict(text_constants[layer])
            learned = dict(text[layer])
            if "layer_scalar" in learned:
                raise ValueError("frozen source unexpectedly contains a trainable layer_scalar")
            learned["layer_scalar"] = fixed.pop("layer_scalar")
            text[layer] = learned
            if fixed:
                text_constants[layer] = fixed
            else:
                del text_constants[layer]
        params["text"] = text
        values["params"] = params
        if text_constants:
            constants["text"] = text_constants
        else:
            del constants["text"]
        if constants:
            values["constants"] = constants
        else:
            del values["constants"]
        return values

    def init_cache(self, batch_size: int):
        self.text.init_cache(batch_size)

    # The training and cache hooks (`dew.nn.protocols`), its text model's,
    # which the decoder clones.

    @nn.nowrap
    def with_cache_capacity(self, capacity: int) -> Self:
        return self.clone(text=self.text.with_cache_capacity(capacity))

    @nn.nowrap
    def recompute_record(self) -> JSON:
        return self.text.recompute_record()

    @nn.nowrap
    def recompute_more(self) -> Self | None:
        text = self.text.recompute_more()
        return None if text is None else self.clone(text=text)

    @nn.nowrap
    def restore_recompute(self, record: JSON) -> Self:
        return self.clone(text=self.text.restore_recompute(record))

    @property
    def keeps_triton_gemm(self) -> bool:
        return self.text.keeps_triton_gemm

    def encode(self, tokens, *, positions=None, segment_ids=None, image_indices=None,
               attention_mask=None, image_groups=None, rotary_positions=None,
               attention_pairwise_mask=None, attention_key_positions=None,
               conditioning: Mapping[str, jax.Array] | None = None, train: bool = False,
               states: bool = False):
        """Append a clean prompt or committed canvas to the cache, evaluating media only when given.

        It returns the logits, or with `states` the final normalized states
        before the head. A loss that scores the vocabulary a tile at a time
        reads the states, so the vocabulary-sized logits of a whole row never
        exist at once.
        """
        read = self.text.hidden_states if states else self.text
        tokens, embeddings = self._clean_inputs(tokens, image_indices, conditioning, train)
        return read(tokens, decode=True, train=train, positions=positions,
                    segment_ids=segment_ids, input_embeddings=embeddings,
                    attention_mask=attention_mask,
                    image_groups=image_groups, rotary_positions=rotary_positions,
                    attention_pairwise_mask=attention_pairwise_mask,
                    attention_key_positions=attention_key_positions)

    def _clean_inputs(self, tokens, image_indices, conditioning: Mapping[str, jax.Array] | None,
                      train: bool) -> tuple[jax.Array, jax.Array | None]:
        """The token ids the encoder reads clean text as, and with media the
        embeddings that replace theirs, image slots filled from the conditioner."""
        if not conditioning:
            if image_indices is not None:
                raise ValueError("image_indices require conditioning payloads")
            return tokens, None
        if self.conditioner is None or image_indices is None:
            raise ValueError("image conditioning requires a vision conditioner and image_indices")
        safe = jnp.where(image_indices >= 0, 0, tokens)
        embedded = self.text.embed_tokens(safe)
        embedded = (embedded * jnp.asarray(math.sqrt(self.text.emb_features),
                     self.text.embed_tokens.embedding.dtype)).astype(embedded.dtype)
        fused = self.conditioner.fuse(safe, embedded, image_indices, conditioning, train=train)
        return fused.tokens, fused.embeddings

    def hidden_states(self, tokens, *, train: bool = False, image_indices=None,
                      conditioning: Mapping[str, jax.Array] | None = None, **fields):
        """Return the causal encoder's final normalized states over clean `tokens`, writing no cache.

        The encoder reads the text tree and fuses media as `encode` does, but
        over the whole sequence at once and without the cache `encode` fills
        for a canvas to attend to: the read a loss or a probe takes of clean
        text. `fields` go to `CausalTransformer.hidden_states`.
        """
        tokens, embeddings = self._clean_inputs(tokens, image_indices, conditioning, train)
        return self.text.hidden_states(tokens, train=train, input_embeddings=embeddings, **fields)

    def logits(self, tokens, *, train: bool = False, **fields):
        """Return the causal encoder's fp32 logits over clean `tokens`, writing no cache (`hidden_states`)."""
        return self.text.logits_from_hidden(self.hidden_states(tokens, train=train, **fields))

    def logits_from_hidden(self, hidden):
        """Return the logits of final states `hidden`, through the head the encoder and the decoder share."""
        return self.text.logits_from_hidden(hidden)

    def output_table(self) -> OutputTable | None:
        """Return the shared head as the matrix final states contract, or
        None where none does (`CausalTransformer.output_table`)."""
        return self.text.output_table()

    def __call__(self, tokens, *, self_conditioning_logits=None,
                 self_conditioning_mask=None, train: bool = False, positions=None,
                 attention_pairwise_mask=None, attention_key_positions=None,
                 states: bool = False):
        """Return the canvas logits, or with `states` the final normalized states before the head.

        `encode` explains why a loss reads the states. Self-conditioning reads
        `self_conditioning_logits`, and rows where `self_conditioning_mask` is
        false get a zero signal, as they do when no logits are given."""
        tokens = jnp.asarray(tokens, jnp.int32)
        if self.is_initializing() and self.conditioner is not None:
            self.conditioner.initialize_parameters()
        embedded = self.decoder.embed_tokens(tokens)
        table = self.decoder.embed_tokens.embedding
        scaled = (embedded * jnp.asarray(math.sqrt(self.text.emb_features), table.dtype)).astype(
            embedded.dtype
        )
        if self_conditioning_logits is None:
            signal = jnp.zeros_like(scaled)
        else:
            signal = soft_embeddings(self_conditioning_logits, table,
                                     math.sqrt(self.text.emb_features)).astype(scaled.dtype)
            if self_conditioning_mask is not None:
                signal = jnp.where(jnp.asarray(self_conditioning_mask)[:, None, None], signal, 0)
        conditioned = self.self_conditioning(scaled, signal)
        # Parameter initialization needs no prefix. Loaded inference always
        # takes the frozen-cache branch, which refuses an absent prefill.
        read = self.decoder.hidden_states if states else self.decoder
        return read(tokens, train=train, decode=not self.is_initializing(),
                    input_embeddings=conditioned,
                    positions=positions, attention_pairwise_mask=attention_pairwise_mask,
                    attention_key_positions=attention_key_positions)


def translate_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> ParamTree:
    """Self-conditioning parameters, cast per weight before the layout copy."""
    paths = {
        "pre_norm.weight": ("pre_norm", "scale"),
        "gate_proj.weight": ("gate_proj", "kernel"),
        "up_proj.weight": ("up_proj", "kernel"),
        "down_proj.weight": ("down_proj", "kernel"),
    }
    seen = set()

    def path_of(name: str) -> tuple[str, ...]:
        bare = name.removeprefix("self_conditioning.")
        if bare not in paths:
            raise ValueError(f"unknown tensor name {name!r}")
        path = paths[bare]
        if path[0] in seen:
            raise ValueError(f"duplicate self-conditioning tensor {name!r}")
        seen.add(path[0])
        return path

    params = translate_parameters(hf_tensors, path_of, param_dtype)
    missing = {path[0] for path in paths.values()} - seen
    if missing:
        raise ValueError(f"missing self-conditioning tensors for {sorted(missing)}")
    return params


__all__ = ["DiffusionGemma", "SelfConditioning"]
