"""Qwen4-Exp's QSA: an attention layer that reads only the key blocks its indexer picks.

`Qwen4ExpTextQSAIndexer` (modeling_qwen4_exp.py:611-718 in transformers 5.16.1) picks, for each query,
the keys its attention reads among those it sees:

1. One projection of the layer's input gives `indexer_n_heads` query heads and one key head, each
   `indexer_head_dim` wide; the query heads are RMS-normed (`(1 + w)`) and turned by the layer's own
   rotary at the query's place.
2. The keys a query sees, in order, fall into blocks of `indexer_compress_ratio`, counted from the first;
   a block's key is the mean of its raw keys, normed and turned at its first key's place.
3. A query scores each complete block `sum_h relu(q_h . k_block) / sqrt(indexer_head_dim)` and keeps
   the best `indexer_budget // indexer_compress_ratio`, every one where it sees fewer, so a query
   seeing up to `indexer_budget` keys reads them all; the keys past its last complete block, fewer than
   a block, it reads whatever their score.

The attention is Qwen3.5's gated attention under that selection, in place of the keys it sees
(`CausalSelfAttention.qsa`). The blocks of a key are the same for every query that sees it, a causal
layer's visibility running in order over each document's real keys, so each block pools once
(`qsa_selection`). The indexer's raw keys and their places wait in the `cache` collection beside the
layer's keys, at the same slots, as `index_keys` and `index_places`.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.dense import Dense

from .norms import RMSNorm
from .precision import float32_dot_general
from .sparse_selection import index_agreements, top_k_selection


@dataclasses.dataclass(frozen=True)
class QSA:
    """The indexer's geometry, by Qwen4-Exp's config names (one key head, `indexer_kv_heads` 1)."""

    indexer_n_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int

    def __post_init__(self):
        if min(self.indexer_n_heads, self.indexer_head_dim, self.indexer_budget,
               self.indexer_compress_ratio) < 1:
            raise ValueError(f"QSA's heads, width, budget and block are positive, got {self}")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError(f"QSA keeps whole blocks: indexer_budget {self.indexer_budget} is a multiple of "
                             f"indexer_compress_ratio {self.indexer_compress_ratio}")

    @property
    def blocks(self) -> int:
        """The complete blocks a query keeps."""
        return self.indexer_budget // self.indexer_compress_ratio


class QSAIndexer(nn.Module):
    """The indexer's own weights and its heads: `index_qk_proj`, `q_layernorm`, `k_layernorm`."""

    spec: QSA
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        heads, width = self.spec.indexer_n_heads, self.spec.indexer_head_dim
        self.index_qk_proj = Dense((heads + 1) * width, use_bias=False, dtype=self.dtype,
                                   precision=self.precision, dot_general=float32_dot_general)
        norm = functools.partial(RMSNorm, epsilon=self.norm_eps, scale_offset=True, dtype=self.dtype)
        self.q_layernorm = norm()
        self.k_layernorm = norm()

    def __call__(self, x):
        """The normed query heads `[B, S, heads, width]` and the raw key `[B, S, width]` of `x`."""
        heads, width = self.spec.indexer_n_heads, self.spec.indexer_head_dim
        projected = self.index_qk_proj(x)
        query = projected[..., :heads * width].reshape(*x.shape[:2], heads, width)
        return self.q_layernorm(query), projected[..., heads * width:]

    def pooled(self, keys):
        """A block's key from its raw keys' mean, normed: `[B, G, width]`."""
        return self.k_layernorm(keys)


def qsa_selection(spec: QSA, query, keys, visible, turn, indexer: QSAIndexer):
    """Each query's selected keys `[B, S, T]`, a subset of the `visible` ones `[B, S, T]` (the module doc).

    `query` `[B, S, heads, width]` is turned already; `keys` `[B, T, width]` are the raw keys;
    `turn(block_keys, first)` turns block keys `[B, G, width]` at their first keys' slots `[B, G]`. A key's
    rank among the keys a query sees is the same for every query that sees it, so it is read off the last
    query that does."""
    batch, length, total = visible.shape
    ratio = spec.indexer_compress_ratio
    rank = jnp.cumsum(visible, axis=-1, dtype=jnp.int32) - 1  # [B, S, T]
    seen = jnp.any(visible, axis=1)  # [B, T]
    last = (length - 1) - jnp.argmax(visible[:, ::-1, :], axis=1)  # the last query that sees each key
    key_rank = jnp.take_along_axis(rank, last[:, None, :], axis=1)[:, 0]  # [B, T]
    # Blocks numbered over the row in slot order: a seen key at a rank divisible by the block opens one.
    opens = seen & (key_rank % ratio == 0)
    block = jnp.where(seen, jnp.cumsum(opens, axis=-1, dtype=jnp.int32) - 1, total)  # [B, T]; total: none
    slots = jnp.arange(total)
    wide = jnp.promote_types(keys.dtype, jnp.float32)

    def pooled(keys, block, seen):
        count = jax.ops.segment_sum(seen.astype(jnp.int32), block, num_segments=total + 1)[:total]
        summed = jax.ops.segment_sum(jnp.where(seen[:, None], keys.astype(wide), 0), block,
                                     num_segments=total + 1)[:total]
        first = jax.ops.segment_min(jnp.where(seen, slots, total), block, num_segments=total + 1)[:total]
        last_slot = jax.ops.segment_max(jnp.where(seen, slots, -1), block, num_segments=total + 1)[:total]
        return summed / jnp.maximum(count, 1)[:, None], count, first, last_slot

    mean, count, first, last_slot = jax.vmap(pooled)(keys, block, seen)
    block_keys = turn(indexer.pooled(mean.astype(keys.dtype)), jnp.minimum(first, total - 1))
    agreements = index_agreements(query, block_keys)  # [B, S, heads, G]
    scores = jnp.sum(jnp.maximum(agreements, 0), axis=-2) * spec.indexer_head_dim ** -0.5
    # A block is a candidate for a query that sees its last key, whole.
    complete = (count == ratio)[:, None, :] & jnp.take_along_axis(
        visible, jnp.broadcast_to(jnp.maximum(last_slot, 0)[:, None, :], (batch, length, total)), axis=-1)
    kept = top_k_selection(scores, complete, spec.blocks)  # [B, S, blocks]
    chosen = jnp.zeros((batch, length, total + 1), bool).at[
        jnp.arange(batch)[:, None, None], jnp.arange(length)[None, :, None], jnp.where(kept >= 0, kept, total)
    ].set(True)[..., :total]
    key_block = jnp.broadcast_to(jnp.minimum(block, total - 1)[:, None, :], (batch, length, total))
    in_complete = jnp.take_along_axis(complete, key_block, axis=-1) & (block < total)[:, None, :]
    picked = jnp.take_along_axis(chosen, key_block, axis=-1)
    return visible & (picked | ~in_complete)
