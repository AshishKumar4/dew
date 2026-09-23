"""Block attention residuals: Kimi K3's residual as a softmax over depth.

Kimi K3 (arXiv 2607.24653) replaces the running residual sum with Attention
Residuals. The layers are cut into blocks of `block_size`. Each block's
output sum is kept, and a sublayer does not read the running sum: it reads
a softmax-weighted mixture of every finished block and the partial sum of
the current one. The weights come from one learned pseudo-query per site
(`KimiDecoderLayer._forward_attn_residual` and `_apply_attn_res`,
modeling_kimi_linear.py:973-1088 of moonshotai/Kimi-K3 at f831ab6):

    v      = [block_0, ..., block_{n-1}, partial]          # [B, S, n + 1, D]
    k      = v / sqrt(mean(v^2) + eps)                     # fp32, per source
    scores = k @ (norm.weight * proj.weight)               # [B, S, n + 1]
    h      = softmax(scores) @ v                           # fp32, then v's dtype

Layer `i` opens a block when `i % block_size == 0`: after its attention
input is read, the partial sum it received becomes a finished block and the
partial restarts from the attention output. Layer 0 reads the embeddings as
they are and makes them block 0. Every other sublayer input is a mixture,
and after the last layer a model-level site mixes all blocks with the final
partial before the final norm (`_apply_output_attn_res`, :1226-1233).

The residual state between layers is one array `[B, S, blocks + 1, D]`:
the finished blocks in order, unfilled slots zero, and the partial last.
Which slots a layer reads is fixed by its index, so a layer's slice is
static and the state keeps one shape through a scan or a pipeline.

Each site holds the checkpoint's two tensors folded as the reference folds
them: `scale` `[D]` is the norm's weight (`*_res_norm.weight`) and `kernel`
`[D, 1]` the projection's (`*_res_proj.weight`, `[1, D]` in the torch
layout). A zero kernel scores every source alike, so a fresh site averages.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import linen as nn

from .sharding import logical_axes


@dataclasses.dataclass(frozen=True)
class AttentionResiduals:
    """The block size the depth softmax groups layers by, `attn_res_block_size`."""

    block_size: int

    def blocks(self, num_layers: int) -> int:
        """The finished blocks after `num_layers` layers: every layer opening one counts."""
        return -(-num_layers // self.block_size)

    def site(self, index: int) -> ResidualSite:
        """Layer `index`'s place in the blocks."""
        return ResidualSite(finished=-(-index // self.block_size), opens=index % self.block_size == 0)


@dataclasses.dataclass(frozen=True)
class ResidualSite:
    """One layer's static place in the depth mixture.

    `finished` counts the blocks done before the layer runs, which its
    attention input mixes with the partial sum; `opens` makes the layer
    close the partial it received into block `finished` and restart it.
    """

    finished: int
    opens: bool


def sources(state, finished: int, partial):
    """The `finished` blocks of `state` with `partial` after them: what one site mixes."""
    return jnp.concatenate([state[:, :, :finished], partial[:, :, None, :]], axis=2)


@logical_axes({}, heuristic=(("attention_res",), ("mlp_res",), ("output_res",)))
class DepthAttention(nn.Module):
    """Mix `[B, S, n, D]` sources into `[B, S, D]` by one learned pseudo-query.

    Both contractions run at full fp32 precision: they are `n + 1` wide, so
    the cost is the elementwise work, and the reference computes them in fp32.
    """

    emb_features: int
    norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, values):
        scale = self.param('scale', nn.initializers.ones, (self.emb_features,), jnp.float32)
        kernel = self.param('kernel', nn.initializers.zeros, (self.emb_features, 1), jnp.float32)
        work = values.astype(jnp.float32)
        keys = work * jax.lax.rsqrt(jnp.mean(jnp.square(work), axis=-1, keepdims=True) + self.norm_eps)
        query = scale.astype(jnp.float32) * kernel[:, 0].astype(jnp.float32)
        highest = jax.lax.Precision.HIGHEST
        weights = jax.nn.softmax(jnp.einsum('bsnd,d->bsn', keys, query, precision=highest), axis=-1)
        return jnp.einsum('bsn,bsnd->bsd', weights, work, precision=highest).astype(values.dtype)
