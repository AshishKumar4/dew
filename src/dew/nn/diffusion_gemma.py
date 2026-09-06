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

from typing import Any, Dict, Mapping, Optional

import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm


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


def translate_weights(hf_tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """Self-conditioning tensors into the module's params tree, in fp32."""
    paths = {
        "self_conditioning.pre_norm.weight": ("pre_norm", "scale"),
        "self_conditioning.gate_proj.weight": ("gate_proj", "kernel"),
        "self_conditioning.up_proj.weight": ("up_proj", "kernel"),
        "self_conditioning.down_proj.weight": ("down_proj", "kernel"),
        "pre_norm.weight": ("pre_norm", "scale"),
        "gate_proj.weight": ("gate_proj", "kernel"),
        "up_proj.weight": ("up_proj", "kernel"),
        "down_proj.weight": ("down_proj", "kernel"),
    }
    params: dict = {}
    for name, tensor in hf_tensors.items():
        if name not in paths:
            raise ValueError(f"unknown tensor name {name!r}")
        leaf = np.asarray(tensor, np.float32)
        if paths[name][-1] == "kernel":
            leaf = np.ascontiguousarray(leaf.T)
        node = params
        for key in paths[name][:-1]:
            node = node.setdefault(key, {})
        node[paths[name][-1]] = leaf
    return params
