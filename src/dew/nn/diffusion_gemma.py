"""DiffusionGemma's self-conditioning MLP, in the reference layout.

The decoder folds the previous step's logits back into its input embeddings
through this gated MLP with a scaled pre-norm and a scale-free post-norm
(TF/models/diffusion_gemma/modeling_diffusion_gemma.py:790-823, the norms at
:147-165). The previous logits become soft embeddings through
`soft_embeddings` (softmax in fp32 against the embedding table times its
scale, :1271-1280), zeroed wherever training disables conditioning for the
row. Weights load under the module's own tensor names.
"""

from __future__ import annotations

from collections.abc import Mapping
import functools
import math
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.multimodal import VisionConditioner
from dew.registry import models


class SelfConditioning(nn.Module):
    """The previous step's soft embeddings folded into the canvas embeddings."""

    hidden_size: int
    intermediate_size: int
    norm_eps: float = 1e-6
    dtype: Optional[Dtype] = None
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
        gated = self.down_proj(
            jax.nn.gelu(self.gate_proj(normed), approximate=True)
            * self.up_proj(normed))
        return self.post_norm(inputs_embeds + gated)


def soft_embeddings(logits: jax.typing.ArrayLike, embed_weight: jax.typing.ArrayLike,
                    scale: float) -> jax.Array:
    """Previous logits as soft embeddings: fp32 softmax against the table."""
    probs = jax.nn.softmax(jnp.asarray(logits, jnp.float32), axis=-1)
    return (probs @ jnp.asarray(embed_weight, jnp.float32)) * jnp.asarray(
        scale, jnp.float32)


@models("diffusion_gemma")
class DiffusionGemma(nn.Module):
    """One text parameter tree, read causally for context and bidirectionally for canvases.

    ``encode`` appends clean tokens to the cache. ``__call__`` refines a canvas
    against that frozen cache and feeds previous logits through self-conditioning.
    Each method is a separate apply: sharing scopes keeps the encoder and decoder
    parameters identical without storing a second tree.
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
    def max_seq_len(self) -> int:
        return self.text.max_seq_len

    def init_cache(self, batch_size: int):
        self.text.init_cache(batch_size)

    def encode(self, tokens, *, positions=None, segment_ids=None, image_indices=None,
               attention_mask=None, image_groups=None, rotary_positions=None,
               attention_pairwise_mask=None, attention_key_positions=None,
               conditioning: Mapping[str, jax.Array] | None = None, train: bool = False):
        """Append a clean prompt or committed canvas, evaluating media only when supplied."""
        if not conditioning:
            if image_indices is not None:
                raise ValueError("image_indices require conditioning payloads")
            return self.text(tokens, decode=True, train=train, positions=positions,
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
        slots = jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape)
        return self.text(fused.tokens, decode=True, train=train, positions=positions,
                         segment_ids=segment_ids, input_embeddings=fused.embeddings,
                         embedding_positions=slots, attention_mask=attention_mask,
                         image_groups=image_groups, rotary_positions=rotary_positions,
                         attention_pairwise_mask=attention_pairwise_mask,
                         attention_key_positions=attention_key_positions)

    def __call__(self, tokens, *, self_conditioning_logits=None,
                 self_conditioning_mask=None, train: bool = False, positions=None,
                 attention_pairwise_mask=None, attention_key_positions=None):
        tokens = jnp.asarray(tokens, jnp.int32)
        embedded = self.decoder.embed_tokens(tokens)
        table = self.decoder.embed_tokens.embedding
        scaled = (embedded * jnp.asarray(math.sqrt(self.text.emb_features), table.dtype)).astype(embedded.dtype)
        if self_conditioning_logits is None:
            signal = jnp.zeros_like(scaled)
        else:
            signal = soft_embeddings(self_conditioning_logits, table,
                                     math.sqrt(self.text.emb_features)).astype(scaled.dtype)
            if self_conditioning_mask is not None:
                signal = jnp.where(jnp.asarray(self_conditioning_mask)[:, None, None], signal, 0)
        conditioned = self.self_conditioning(scaled, signal)
        indices = jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape)
        # Parameter initialization needs no prefix. Loaded inference always
        # takes the frozen-cache branch, which refuses an absent prefill.
        return self.decoder(tokens, train=train, decode=not self.is_initializing(),
                            input_embeddings=conditioned, embedding_positions=indices,
                            positions=positions, attention_pairwise_mask=attention_pairwise_mask,
                            attention_key_positions=attention_key_positions)


def translate_weights(hf_tensors: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Self-conditioning tensors into the module parameter tree in fp32."""
    paths = {
        "pre_norm.weight": ("pre_norm", "scale"),
        "gate_proj.weight": ("gate_proj", "kernel"),
        "up_proj.weight": ("up_proj", "kernel"),
        "down_proj.weight": ("down_proj", "kernel"),
    }
    params: dict[str, dict[str, np.ndarray]] = {}
    for name, tensor in hf_tensors.items():
        bare = name.removeprefix("self_conditioning.")
        if bare not in paths:
            raise ValueError(f"unknown tensor name {name!r}")
        module, key = paths[bare]
        if module in params:
            raise ValueError(f"duplicate self-conditioning tensor {name!r}")
        leaf = np.asarray(tensor, np.float32)
        params[module] = {key: np.ascontiguousarray(leaf.T) if key == "kernel" else leaf}
    missing = {path[0] for path in paths.values()} - set(params)
    if missing:
        raise ValueError(f"missing self-conditioning tensors for {sorted(missing)}")
    return params
