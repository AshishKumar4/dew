"""Manifold-constrained hyper-connections: the residual as `hc_mult` streams.

GLM-5.3-Flash and DeepSeek V4 carry the residual through the decoder as a
stack of `hc_mult` streams, `[B, S, H, D]`, instead of one vector. Each
sublayer reads one collapse of the streams and writes back into every one
of them through a mixing the streams themselves choose (mHC, Xie et al.
2026, section 2.2 eq. 8). GLM-5.3-Flash and DeepSeek V4 compute this
identically. The steps are:

    flat  = rmsnorm(streams.reshape(B, S, H*D))          # no weight, fp32
    mixes = flat @ fn^T                                  # [(2 + H) * H]
    pre   = sigmoid(mixes[:H] * scale[0] + base[:H]) + eps
    post  = 2 sigmoid(mixes[H:2H] * scale[1] + base[H:2H])
    comb  = softmax(mixes[2H:] * scale[2] + base[2H:], over the H x H row) + eps
    comb  = sinkhorn(comb)     # hc_sinkhorn_iters alternating normalisations
    collapsed = sum_h pre[h] streams[h]                  # the sublayer's input
    streams'  = post[:, None] * sublayer(collapsed)[None, :] + comb^T @ streams

References: `Glm5NextTextHyperConnection` (modeling_glm5_next.py:219-295)
and `DeepseekV4HyperConnection` (modeling_deepseek_v4.py:867-943).

`comb` is projected toward the doubly stochastic matrices by Sinkhorn-Knopp:
a column normalisation first, then `iters - 1` rounds of row and column
normalisation, each with `eps` in the denominator, in fp32. It is applied
transposed, `streams'[k] = sum_j comb[j, k] streams[j]`. Sinkhorn leaves the
matrix asymmetric, so the direction is part of the math.

The two references differ only in how the streams collapse at the end:
GLM takes their unweighted mean (`Glm5NextTextHyperHead`,
modeling_glm5_next.py:298-302) and V4 a learned collapse of the same shape
as `pre` (`DeepseekV4HyperHead`, modeling_deepseek_v4.py:946-962), which the
record's `head` names.

DeepSeek-V4.1 runs Single-Pass mHC (arXiv 2609.19969, section 2.4.1, eq. 6):
a sublayer collapses the streams by the `pre` the previous site computed,
not its own, so a block's attention reads the previous block's feed-forward
`pre` and its feed-forward the attention's, and the stack's first sublayer
reads the first stream alone (V4.1 inference/model.py:968-994, :1159-1163).
The record's `single_pass` names that schedule, under which the stack
carries `Carried(streams, pre)`; its final norm reads the collapse by the
last site's `pre` (:1268), the head 'carried'.

Parameter names are the checkpoints': `fn` `[(2 + H) H, H D]` in the
torch Linear's `[out, in]` layout (the release stores it as a raw tensor,
not a Linear), `base` `[(2 + H) H]` and `scale` `[3]` under the block's
`attn_hc` and `ffn_hc`; the weighted head's `hc_fn`, `hc_base`, `hc_scale`
under `hc_head`.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .attention import unweighted_rmsnorm
from .sharding import logical_axes

HEADS = ('mean', 'weighted', 'carried')


@dataclasses.dataclass(frozen=True)
class HyperConnections:
    """Holds the stream count, the Sinkhorn floor and iterations, the head
    and the schedule.

    A model names this on `hyper_connections`. The field names are the
    references' own config fields. `head` picks what collapses the streams
    before the final norm; 'carried' is the last site's `pre`, which only
    the `single_pass` schedule carries (the module doc).
    """

    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    head: str = 'mean'
    single_pass: bool = False

    def __post_init__(self):
        if self.hc_mult < 1:
            raise ValueError(f"hc_mult counts residual streams, got {self.hc_mult}")
        if self.hc_sinkhorn_iters < 1:
            raise ValueError(
                f"hc_sinkhorn_iters counts Sinkhorn normalisations from one, got {self.hc_sinkhorn_iters}")
        if self.head not in HEADS:
            raise ValueError(f"head collapses the streams, one of {HEADS}, got {self.head!r}")
        if self.head == 'carried' and not self.single_pass:
            raise ValueError("the carried head collapses by the pre the Single-Pass schedule "
                             "carries, so it needs single_pass")


class Carried(NamedTuple):
    """Single-Pass mHC's residual: the streams `[B, S, H, D]` and the fp32
    `pre` `[B, S, H]` the next sublayer collapses them by."""

    streams: jax.Array
    pre: jax.Array


def expand_streams(x, hc_mult: int):
    """Copy the embeddings into every stream: `[B, S, D]` -> `[B, S, H, D]`."""
    return jnp.broadcast_to(x[:, :, None, :], (*x.shape[:2], hc_mult, x.shape[-1]))


def sinkhorn(comb, iters: int, eps: float):
    """Normalise `[..., H, H]` alternately `iters` times, the columns first.

    Each division carries `eps` in its denominator
    (modeling_glm5_next.py:287-290).
    """
    comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (jnp.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)
    return comb


def mix_streams(post, comb, output, streams):
    """Write the sublayer's output into every stream over the mixed residual.

    Stream k becomes `post[k] output + sum_j comb[j, k] streams[j]`
    (modeling_glm5_next.py:1316-1318).
    """
    dtype = streams.dtype
    mixed = jnp.einsum('bsjk,bsjd->bskd', comb.astype(dtype), streams)
    return post.astype(dtype)[..., None] * output[..., None, :] + mixed


@logical_axes({}, heuristic=(("attn_hc",), ("ffn_hc",), ("hc_head",)))
class HyperConnection(nn.Module):
    """Map the streams to one site's `(post, comb, collapsed)`.

    `post` `[B, S, H]` and `comb` `[B, S, H, H]` are fp32, as the reference
    computes them, and go to `mix_streams`. `collapsed` `[B, S, D]` is in
    the streams' dtype and is what the sublayer's norm reads.
    """

    spec: HyperConnections
    emb_features: int
    norm_eps: float = 1e-5

    def __call__(self, streams):
        pre, post, comb = self.mapping(streams)
        return post, comb, collapse_by(pre, streams)

    @nn.compact
    def mapping(self, streams):
        """The site's `(pre, post, comb)` over the streams, all fp32."""
        hc = self.spec.hc_mult
        mix = (2 + hc) * hc
        fn = self.param('fn', nn.initializers.normal(0.02), (mix, hc * self.emb_features), jnp.float32)
        base = self.param('base', nn.initializers.zeros, (mix,), jnp.float32)
        scale = self.param('scale', nn.initializers.ones, (3,), jnp.float32)
        flat = unweighted_rmsnorm(
            streams.reshape(*streams.shape[:2], hc * streams.shape[-1]).astype(jnp.float32),
            self.norm_eps)
        mixes = flat @ fn.T
        pre = nn.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + self.spec.hc_eps
        post = 2 * nn.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
        logits = mixes[..., 2 * hc:].reshape(*mixes.shape[:-1], hc, hc) * scale[2] + base[2 * hc:].reshape(hc, hc)
        comb = sinkhorn(jax.nn.softmax(logits, axis=-1) + self.spec.hc_eps,
                        self.spec.hc_sinkhorn_iters, self.spec.hc_eps)
        return pre, post, comb


def collapse_by(pre, streams):
    """`sum_h pre[h] streams[h]` in fp32, back in the streams' dtype."""
    return jnp.sum(pre[..., None] * streams.astype(jnp.float32), axis=2).astype(streams.dtype)


def first_stream(streams):
    """The `pre` Single-Pass mHC's first sublayer collapses by: the first
    stream alone (V4.1 inference/model.py:1159-1163), fp32 `[B, S, H]`."""
    return jnp.zeros(streams.shape[:3], jnp.float32).at[..., 0].set(1.0)


class HyperHead(nn.Module):
    """Collapse the streams as DeepSeek V4 learns to, before the final norm.

    This is `pre` of the mHC mapping alone, with its own `hc_fn` `[H, H D]`,
    `hc_base` `[H]` and `hc_scale` `[1]` (modeling_deepseek_v4.py:958-962).
    """

    spec: HyperConnections
    emb_features: int
    norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, streams):
        hc = self.spec.hc_mult
        fn = self.param('hc_fn', nn.initializers.normal(0.02), (hc, hc * self.emb_features), jnp.float32)
        base = self.param('hc_base', nn.initializers.zeros, (hc,), jnp.float32)
        scale = self.param('hc_scale', nn.initializers.ones, (1,), jnp.float32)
        flat = unweighted_rmsnorm(
            streams.reshape(*streams.shape[:2], hc * streams.shape[-1]).astype(jnp.float32),
            self.norm_eps)
        pre = nn.sigmoid(flat @ fn.T * scale + base) + self.spec.hc_eps
        return collapse_by(pre, streams)


def collapse_streams(streams, head: HyperHead | None):
    """Collapse the streams to the one vector the final norm reads.

    Without a head that is their mean (modeling_glm5_next.py:301-302), and
    with one it is the head's weighted sum.
    """
    if head is None:
        return jnp.mean(streams, axis=2)
    return head(streams)
