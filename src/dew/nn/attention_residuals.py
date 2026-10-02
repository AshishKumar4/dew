"""Block attention residuals: Kimi K3's residual as a softmax over depth.

Kimi K3 (arXiv 2607.24653) cuts the layers into blocks of `block_size`, keeps
each block's output sum, and has every sublayer read a softmax mixture of the
finished blocks and the current partial sum, weighted by one learned
pseudo-query per site (`KimiDecoderLayer._forward_attn_residual`,
`_apply_attn_res`, modeling_kimi_linear.py:973-1088 of moonshotai/Kimi-K3 at
f831ab6):

    v      = [block_0, ..., block_{n-1}, partial]          # [B, S, n + 1, D]
    k      = v / sqrt(mean(v^2) + eps)                     # fp32, per source
    scores = k @ (norm.weight * proj.weight)               # [B, S, n + 1]
    h      = softmax(scores) @ v                           # fp32, then v's dtype

Layer `i` opens a block when `i % block_size == 0`: after reading its
attention input, the partial it received becomes a finished block and the
partial restarts from the attention output. Layer 0 reads the embeddings as
block 0, and after the last layer a model-level site mixes all blocks with
the final partial before the final norm (`_apply_output_attn_res`,
:1226-1233). The state is one `[B, S, blocks + 1, D]` array, unfilled slots
zero and the partial last, so each layer's slice is static through a scan or
pipeline. A site's `scale` `[D]` is `*_res_norm.weight` and `kernel` `[D, 1]`
`*_res_proj.weight`; a zero kernel averages.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import linen as nn

from .precision import at_least_fp32


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
        wide = at_least_fp32(values.dtype)
        work = values.astype(wide)
        keys = work * jax.lax.rsqrt(jnp.mean(jnp.square(work), axis=-1, keepdims=True) + self.norm_eps)
        query = scale.astype(wide) * kernel[:, 0].astype(wide)
        highest = jax.lax.Precision.HIGHEST
        weights = jax.nn.softmax(jnp.einsum('bsnd,d->bsn', keys, query, precision=highest), axis=-1)
        return jnp.einsum('bsn,bsnd->bsd', weights, work, precision=highest).astype(values.dtype)
