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
from flax.typing import Dtype, PrecisionLike

from dew.interop.weights import ParamTree, translate_parameters
from dew.nn.attention import RMSNorm
from dew.nn.backbones.causal_transformer import CausalTransformer, DecoderBank
from dew.nn.moe import gated_product
from dew.nn.multimodal import VisionConditioner
from dew.nn.precision import at_least_fp32
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
    self-conditioning. Each method is a separate apply. The encoder and the
    decoder share one scope, so their parameters are identical and no second
    tree is stored.
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

    def init_cache(self, batch_size: int):
        self.text.init_cache(batch_size)

    # What a task and the trainer read off the model, answered by its text
    # model (`dew.nn.protocols`); the decoder is a clone of it and follows.

    @nn.nowrap
    def with_cache_capacity(self, capacity: int) -> Self:
        """This model with `capacity` cache slots per row for the clean
        prefix the canvases read (`CacheCapacity`)."""
        return self.clone(text=self.text.with_cache_capacity(capacity))

    @nn.nowrap
    def recompute_record(self) -> JSON:
        """Its text model's rung (`Recomputing`)."""
        return self.text.recompute_record()

    @nn.nowrap
    def recompute_more(self) -> Self | None:
        """This model with its text model one rung up (`Recomputing`), or None at its top."""
        text = self.text.recompute_more()
        return None if text is None else self.clone(text=text)

    @nn.nowrap
    def restore_recompute(self, record: JSON) -> Self:
        """This model with its text model at `record`'s rung where that is above its own (`Recomputing`)."""
        return self.clone(text=self.text.restore_recompute(record))

    @property
    def keeps_triton_gemm(self) -> bool:
        """Its text model's (`TritonGemm`)."""
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
        if not conditioning:
            if image_indices is not None:
                raise ValueError("image_indices require conditioning payloads")
            return read(tokens, decode=True, train=train, positions=positions,
                        segment_ids=segment_ids, attention_mask=attention_mask,
                        image_groups=image_groups, rotary_positions=rotary_positions,
                        attention_pairwise_mask=attention_pairwise_mask,
                        attention_key_positions=attention_key_positions)
        if self.conditioner is None or image_indices is None:
            raise ValueError("image conditioning requires a vision conditioner and image_indices")
        safe = jnp.where(image_indices >= 0, 0, tokens)
        embedded = self.text.embed_tokens(safe)
        embedded = (embedded * jnp.asarray(math.sqrt(self.text.emb_features),
                     self.text.embed_tokens.embedding.dtype)).astype(embedded.dtype)
        fused = self.conditioner.fuse(safe, embedded, image_indices, conditioning, train=train)
        return read(fused.tokens, decode=True, train=train, positions=positions,
                    segment_ids=segment_ids, input_embeddings=fused.embeddings,
                    attention_mask=attention_mask,
                    image_groups=image_groups, rotary_positions=rotary_positions,
                    attention_pairwise_mask=attention_pairwise_mask,
                    attention_key_positions=attention_key_positions)

    def head_weight(self, params):
        """Return the `[D, vocab]` head the encoder and the decoder score with, in its stored dtype.

        It is read from the text tree of `params` by `CausalTransformer.head_weight`."""
        return self.text.head_weight(params["text"])

    def vocabulary_bias(self, params):
        """The decoder's vocabulary bias, for the same affine head its forward scores."""
        return self.text.vocabulary_bias(params['text'])

    def head_table(self, params):
        """Return the head as the text tree stores it, and whether its rows are the vocabulary.

        It is read from the text tree of `params` by `CausalTransformer.head_table`."""
        return self.text.head_table(params["text"])

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
