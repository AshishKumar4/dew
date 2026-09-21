"""Manifold-constrained hyper-connections: the residual as `hc_mult` streams.

GLM-5.3-Flash and DeepSeek V4 carry the residual through the decoder as a
stack of `hc_mult` streams, `[B, S, H, D]`, instead of one vector. Each
sublayer reads one collapse of the streams and writes back into every one
of them through a mixing the streams themselves choose (mHC, Xie et al.
2026, section 2.2 eq. 8). Both references compute the same thing, module
for module (`Glm5NextTextHyperConnection`, modeling_glm5_next.py:219-295;
`DeepseekV4HyperConnection`, modeling_deepseek_v4.py:867-943):

    flat  = rmsnorm(streams.reshape(B, S, H*D))          # no weight, fp32
    mixes = flat @ fn^T                                  # [(2 + H) * H]
    pre   = sigmoid(mixes[:H] * scale[0] + base[:H]) + eps
    post  = 2 sigmoid(mixes[H:2H] * scale[1] + base[H:2H])
    comb  = softmax(mixes[2H:] * scale[2] + base[2H:], over the H x H row) + eps
    comb  = sinkhorn(comb)     # hc_sinkhorn_iters alternating normalisations
    collapsed = sum_h pre[h] streams[h]                  # the sublayer's input
    streams'  = post[:, None] * sublayer(collapsed)[None, :] + comb^T @ streams

`comb` is projected toward the doubly stochastic matrices by Sinkhorn-Knopp:
a column normalisation first, then `iters - 1` rounds of row and column
normalisation, each with `eps` in the denominator, in fp32. It is applied
transposed, `streams'[k] = sum_j comb[j, k] streams[j]`, which the reference
spells `matmul(comb.transpose(-1, -2), residual)`; Sinkhorn leaves the
matrix asymmetric, so the direction is part of the math.

The two references differ only in how the streams collapse at the end:
GLM takes their unweighted mean (`Glm5NextTextHyperHead`,
modeling_glm5_next.py:298-302) and V4 a learned collapse of the same shape
as `pre` (`DeepseekV4HyperHead`, modeling_deepseek_v4.py:946-962), which the
record's `head` names.

Parameter names are the checkpoints': `fn` `[(2 + H) H, H D]` in the
torch Linear's `[out, in]` layout (the release stores it as a raw tensor,
not a Linear), `base` `[(2 + H) H]` and `scale` `[3]` under the block's
`attn_hc` and `ffn_hc`; the weighted head's `hc_fn`, `hc_base`, `hc_scale`
under `hc_head`.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype

from .sharding import logical_axes

HEADS = ('mean', 'weighted')


@dataclasses.dataclass(frozen=True)
class HyperConnections:
    """The record a model names on `hyper_connections`, by the references'
    config fields: the stream count, the Sinkhorn floor and iteration
    count, and which head collapses the streams before the final norm."""

    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    head: str = 'mean'

    def __post_init__(self):
        if self.hc_mult < 1:
            raise ValueError(f"hc_mult counts residual streams, got {self.hc_mult}")
        if self.hc_sinkhorn_iters < 1:
            raise ValueError(
                f"hc_sinkhorn_iters counts Sinkhorn normalisations from one, got {self.hc_sinkhorn_iters}")
        if self.head not in HEADS:
            raise ValueError(f"head collapses the streams, one of {HEADS}, got {self.head!r}")


def expand_streams(x, hc_mult: int):
    """The embeddings copied into every stream: `[B, S, D]` -> `[B, S, H, D]`
    (modeling_glm5_next.py:1477, modeling_deepseek_v4.py:1310)."""
    return jnp.broadcast_to(x[:, :, None, :], (*x.shape[:2], hc_mult, x.shape[-1]))


def _unweighted_rmsnorm(flat, eps: float):
    """`x * rsqrt(mean(x^2) + eps)` with no weight, in fp32
    (`Glm5NextTextUnweightedRMSNorm`, modeling_glm5_next.py:210-216)."""
    return flat * jax.lax.rsqrt(jnp.mean(jnp.square(flat), axis=-1, keepdims=True) + eps)


def sinkhorn(comb, iters: int, eps: float):
    """`iters` alternating normalisations of `[..., H, H]`, the column one
    first, each with `eps` in its denominator (modeling_glm5_next.py:287-290)."""
    comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (jnp.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)
    return comb


def mix_streams(post, comb, output, streams):
    """The sublayer's output written into every stream over the mixed residual:
    `post[k] output + sum_j comb[j, k] streams[j]` (modeling_glm5_next.py:1316-1318)."""
    dtype = streams.dtype
    mixed = jnp.einsum('bsjk,bsjd->bskd', comb.astype(dtype), streams)
    return post.astype(dtype)[..., None] * output[..., None, :] + mixed


@logical_axes({}, heuristic=(("attn_hc",), ("ffn_hc",), ("hc_head",)))
class HyperConnection(nn.Module):
    """One site's mHC mapping: `(post, comb, collapsed)` of the streams.

    `post` `[B, S, H]` and `comb` `[B, S, H, H]` are fp32, as the reference
    computes them, for `mix_streams`; `collapsed` `[B, S, D]` is in the
    streams' dtype and is what the sublayer's norm reads.
    """

    spec: HyperConnections
    emb_features: int
    norm_eps: float = 1e-5
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, streams):
        hc = self.spec.hc_mult
        mix = (2 + hc) * hc
        fn = self.param('fn', nn.initializers.normal(0.02), (mix, hc * self.emb_features), jnp.float32)
        base = self.param('base', nn.initializers.zeros, (mix,), jnp.float32)
        scale = self.param('scale', nn.initializers.ones, (3,), jnp.float32)
        flat = _unweighted_rmsnorm(streams.reshape(*streams.shape[:2], hc * streams.shape[-1]).astype(jnp.float32),
                                   self.norm_eps)
        mixes = flat @ fn.T
        pre = nn.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + self.spec.hc_eps
        post = 2 * nn.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
        logits = mixes[..., 2 * hc:].reshape(*mixes.shape[:-1], hc, hc) * scale[2] + base[2 * hc:].reshape(hc, hc)
        comb = sinkhorn(jax.nn.softmax(logits, axis=-1) + self.spec.hc_eps,
                        self.spec.hc_sinkhorn_iters, self.spec.hc_eps)
        collapsed = jnp.sum(pre[..., None] * streams.astype(jnp.float32), axis=2).astype(streams.dtype)
        return post, comb, collapsed


class HyperHead(nn.Module):
    """DeepSeek V4's learned collapse of the streams before the final norm
    (modeling_deepseek_v4.py:958-962): `pre` of the mHC mapping alone, with
    its own `hc_fn` `[H, H D]`, `hc_base` `[H]` and `hc_scale` `[1]`."""

    spec: HyperConnections
    emb_features: int
    norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, streams):
        hc = self.spec.hc_mult
        fn = self.param('hc_fn', nn.initializers.normal(0.02), (hc, hc * self.emb_features), jnp.float32)
        base = self.param('hc_base', nn.initializers.zeros, (hc,), jnp.float32)
        scale = self.param('hc_scale', nn.initializers.ones, (1,), jnp.float32)
        flat = _unweighted_rmsnorm(streams.reshape(*streams.shape[:2], hc * streams.shape[-1]).astype(jnp.float32),
                                   self.norm_eps)
        pre = nn.sigmoid(flat @ fn.T * scale + base) + self.spec.hc_eps
        return jnp.sum(pre[..., None] * streams.astype(jnp.float32), axis=2).astype(streams.dtype)


def collapse_streams(streams, head: HyperHead | None):
    """The streams as the one vector the final norm reads: their mean without
    a head (modeling_glm5_next.py:301-302) or the weighted head's collapse."""
    if head is None:
        return jnp.mean(streams, axis=2)
    return head(streams)
