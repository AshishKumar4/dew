"""DeepSeek V4's attention: shared-KV MQA over a sliding window, extended by
a compressor.

The reference is transformers 5.16.1
`models/deepseek_v4/modeling_deepseek_v4.py`, read as the specification.
Every layer is one attention (`DeepseekV4Attention`, :746-864). The query is
low-rank and normed per head without a weight. One key/value head is shared
by every query head. Rotary positions turn the trailing `rope_head_dim` of
each head in interleaved pairs. A learned sink sits beside the logits. The
values are the rotated keys, so the output is de-rotated at the query's
position after the softmax. The output projection is grouped and low-rank.

A sliding layer attends the last `sliding_window` positions,
its own included (masking_utils sliding_window_overlay). A compressed layer
concatenates onto those keys the entries its compressor emits, one per
window of `compress_rate` tokens (:353-435 the heavily compressed HCA
branch, :580-693 the compressed sparse CSA branch with its lightning
indexer, :437-578), and a per-query bias says which entries a query may
attend: HCA every entry whose window closed before the query, CSA the
indexer's top-k of those.

The compressors emit no entry for a trailing partial window: the reference
drops it outside a cache (:398, :630) and buffers it inside one. Cached
calls keep projected incomplete windows, CSA's preceding Ca window and
completed rotated entries (:209-291); no old hidden state is reprojected.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from dew.nn.attention import (
    RMSNorm,
    _cache_positions,
    causal_attention_mask,
    document_mask,
    rotary_freqs,
    unweighted_rmsnorm,
)
from dew.nn.inputs import AttentionMetadata
from dew.nn.kv_cache import KVCache, write_cache
from dew.nn.mixers import MixerBase, MixerContext, mixers
from dew.nn.mla import YarnScaling, yarn_inv_freq
from dew.nn.sharding import logical_axes
from dew.nn.sparse_selection import top_k_keys

COMPRESSORS = ('csa', 'hca')


def rope_freqs(positions, rope_dim: int, theta: float, yarn: YarnScaling | None):
    """cos/sin per pair over the rope width, `[..., rope_dim // 2]`.

    V4's rotary scales neither table: its YaRN entry forces
    `attention_factor` to 1.0 (configuration_deepseek_v4.py:294-320) and
    the plain entry never scales (modeling_deepseek_v4.py:136), so the
    ramp changes the frequencies alone.
    """
    if yarn is None:
        return rotary_freqs(positions, rope_dim, theta)
    inv_freq = yarn_inv_freq(rope_dim, theta, yarn)
    angles = jnp.asarray(positions, jnp.float32)[..., None] * inv_freq
    return jnp.cos(angles), jnp.sin(angles)


def rotate_trailing(x, cos, sin):
    """Rotate the trailing `2 * cos.shape[-1]` channels of `x` in interleaved
    pairs, the leading channels untouched (modeling_deepseek_v4.py:326-350).

    `x` is `[B, S, H, D]` or `[B, S, D]`; `cos`/`sin` are `[S, P]` or
    `[B, S, P]` with `P` pairs. The pair `(x[2i], x[2i+1])` turns by one
    angle and lands back in place, so a rotation by the negated sine undoes
    it, which the attention output relies on.
    """
    pairs = cos.shape[-1]
    lead = x.shape[:-1]
    if cos.ndim == 2:
        cos = jnp.broadcast_to(cos, (lead[0], *cos.shape))
        sin = jnp.broadcast_to(sin, (lead[0], *sin.shape))
    if x.ndim == 4:
        cos, sin = cos[:, :, None, :], sin[:, :, None, :]
    rope = x[..., -2 * pairs:].astype(jnp.float32).reshape(*lead, pairs, 2)
    first, second = rope[..., 0], rope[..., 1]
    rotated = jnp.stack([first * cos - second * sin, second * cos + first * sin], axis=-1)
    return jnp.concatenate([x[..., :-2 * pairs], rotated.reshape(*lead, 2 * pairs).astype(x.dtype)], axis=-1)


def pool_windows(kv, gate, position_bias, rate: int, overlap: bool):
    """The compressor's entries: one softmax-gated sum per window of `rate` tokens.

    `kv` and `gate` are `[B, S, W]` with `W` the entry width, or `[B, S, 2W]`
    for the two-series CSA layout; `position_bias` is `[rate, W or 2W]` and
    adds to the gate of each position inside its window
    (modeling_deepseek_v4.py:405-409 HCA, :636-666 CSA). The trailing
    partial window is dropped, as the reference's stateless path does.

    With `overlap`, entry `w` combines window `w - 1`'s first series (Ca)
    with window `w`'s second series (Cb) over `2 * rate` slots, and window
    0's Ca slots hold zero keys under a `-inf` gate, so they weigh nothing
    (:641-660).
    """
    batch, length, features = kv.shape
    windows = length // rate
    kv = kv[:, :windows * rate].reshape(batch, windows, rate, features)
    gate = gate[:, :windows * rate].reshape(batch, windows, rate, features) + position_bias
    if overlap:
        width = features // 2
        previous_kv = jnp.concatenate(
            [jnp.zeros_like(kv[:, :1, :, :width]), kv[:, :-1, :, :width]], axis=1)
        previous_gate = jnp.concatenate(
            [jnp.full_like(gate[:, :1, :, :width], -jnp.inf), gate[:, :-1, :, :width]], axis=1)
        kv = jnp.concatenate([previous_kv, kv[..., width:]], axis=2)
        gate = jnp.concatenate([previous_gate, gate[..., width:]], axis=2)
    weights = jax.nn.softmax(gate.astype(jnp.float32), axis=2).astype(kv.dtype)
    return jnp.sum(kv * weights, axis=2)


def entries_visible(positions, entries: int, rate: int):
    """Which compressed entries a query may attend: `[B, S, entries]`, bool.

    Entry `w` covers source positions `w * rate` through `w * rate + rate - 1`
    and is visible once the query sits at or past its last token,
    `w < (position + 1) // rate` (modeling_deepseek_v4.py:426-433).
    """
    threshold = (jnp.asarray(positions) + 1) // rate
    if threshold.ndim == 1:
        threshold = threshold[None]
    return jnp.arange(entries)[None, None, :] < threshold[:, :, None]


def append_windows(kv, gate, slots, buffers, previous, rate: int, width: int):
    """Pool newly closed windows without reprojecting old tokens.

    The incomplete window survives a call boundary. CSA additionally keeps
    the preceding window's Ca series (modeling_deepseek_v4.py:209-243,
    277-291); a first window has no previous Ca contribution. Slots are
    compact per-row cache positions, with -1 marking an invalid token.
    """
    batch, length, _ = kv.shape
    maximum = (length + rate - 1) // rate
    pooled = jnp.zeros((batch, maximum, width), kv.dtype)
    emitted = jnp.full((batch, maximum), -1, jnp.int32)
    counts = jnp.zeros((batch,), jnp.int32)

    def step(carry, inputs):
        buffered, prior, values, indices, count = carry
        key, logits, position = inputs
        within = jnp.where(position >= 0, position % rate, -1)
        buffered = (write_cache(buffered[0], key[:, None], within[:, None]),
                    write_cache(buffered[1], logits[:, None], within[:, None]))
        closed = (position >= 0) & ((position + 1) % rate == 0)

        def emit(state):
            old, values, indices, count = state
            window_key, window_gate = buffered
            if old is not None:
                window_key = jnp.concatenate([old[0], window_key[..., width:]], axis=1)
                window_gate = jnp.concatenate([old[1], window_gate[..., width:]], axis=1)
            weights = jax.nn.softmax(window_gate.astype(jnp.float32), axis=1).astype(window_key.dtype)
            entry = jnp.sum(window_key * weights, axis=1)
            at = jnp.where(closed, count, -1)[:, None]
            values = write_cache(values, entry[:, None], at)
            indices = write_cache(indices[..., None], (position // rate)[:, None, None], at)[..., 0]
            if old is not None:
                old = tuple(jnp.where(closed[:, None, None], current[..., :width], before)
                            for current, before in zip(buffered, old, strict=True))
            return old, values, indices, count + closed.astype(jnp.int32)

        prior, values, indices, count = jax.lax.cond(
            jnp.any(closed), emit, lambda state: state, (prior, values, indices, count))
        return (buffered, prior, values, indices, count), None

    (buffers, previous, pooled, emitted, _), _ = jax.lax.scan(
        step, (buffers, previous, pooled, emitted, counts),
        (jnp.swapaxes(kv, 0, 1), jnp.swapaxes(gate, 0, 1), jnp.swapaxes(slots, 0, 1)))
    return pooled, emitted, buffers, previous


class GroupedLinear(nn.Module):
    """Block-diagonal grouped projection: `[..., g, in]` to `[..., g, features]`
    through one `[g, in, features]` kernel (`DeepseekV4GroupedLinear`,
    modeling_deepseek_v4.py:294-323, whose weight `[g * features, in]` is
    this kernel with each group's block transposed)."""

    groups: int
    features: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel', nn.initializers.lecun_normal(),
                            (self.groups, x.shape[-1], self.features), jnp.float32)
        dtype = x.dtype if self.dtype is None else self.dtype
        return jnp.einsum('...gi,gio->...go', x.astype(dtype), kernel.astype(dtype),
                          precision=self.precision)


class CompressedEntries(nn.Module):
    """The projections behind a compressor's entries: `kv_proj` and
    `gate_proj` into the entry width (twice it for the two-series CSA
    layout), the position bias, and the entry norm; `entries` pools the
    windows and rotates each entry at its window's first position, `w *
    rate` (modeling_deepseek_v4.py:375-413, :603-671).

    Both the layer's compressor and the indexer's own compressor are one of
    these under their checkpoint names; the indexer's sits under
    `compressor/indexer` with its scoring leaves beside them, the nesting
    the checkpoint keeps (conversion_mapping.py:487).
    """

    width: int
    rate: int
    overlap: bool
    rope_dim: int
    rope_theta: float
    yarn: YarnScaling | None
    norm_eps: float
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        series = 2 if self.overlap else 1
        dense = functools.partial(nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
        self.kv_proj = dense(series * self.width, name='kv_proj')
        self.gate_proj = dense(series * self.width, name='gate_proj')
        self.position_bias = self.param(
            'position_bias', nn.initializers.zeros, (self.rate, series * self.width), jnp.float32)
        self.kv_norm = RMSNorm(epsilon=self.norm_eps, scale_after_cast=True,
                               dtype=self.dtype, name='kv_norm')

    def entries(self, x):
        """The rotated entries `[B, T, width]`, `T = S // rate`."""
        pooled = self.kv_norm(pool_windows(
            self.kv_proj(x), self.gate_proj(x), self.position_bias.astype(x.dtype),
            self.rate, self.overlap))
        cos, sin = rope_freqs(jnp.arange(pooled.shape[1]) * self.rate, self.rope_dim,
                              self.rope_theta, self.yarn)
        return rotate_trailing(pooled, cos, sin)

    @nn.compact
    def cached_entries(self, x, slots, capacity: int, write: bool):
        """Append closed windows to the rotated-entry cache; allocation writes none."""
        kv, gate = self.kv_proj(x), self.gate_proj(x)
        gate = gate + self.position_bias[jnp.maximum(slots, 0) % self.rate].astype(gate.dtype)
        shape = (x.shape[0], self.rate, kv.shape[-1])
        buffer_kv = self.variable('cache', 'buffer_kv', jnp.zeros, shape, kv.dtype)
        buffer_gate = self.variable('cache', 'buffer_gate', jnp.zeros, shape, gate.dtype)
        entries = self.variable('cache', 'compressed', jnp.zeros,
                                (x.shape[0], capacity // self.rate, self.width), kv.dtype)
        prior = None
        overlap_kv = overlap_gate = None
        if self.overlap:
            overlap_kv = self.variable('cache', 'overlap_kv', jnp.zeros,
                                       (x.shape[0], self.rate, self.width), kv.dtype)
            overlap_gate = self.variable('cache', 'overlap_gate', jnp.full,
                                         (x.shape[0], self.rate, self.width), -jnp.inf, gate.dtype)
            prior = (overlap_kv.value, overlap_gate.value)
        pooled, emitted, buffered, prior = append_windows(
            kv, gate, slots, (buffer_kv.value, buffer_gate.value), prior, self.rate, self.width)
        pooled = self.kv_norm(pooled)
        cos, sin = rope_freqs(emitted * self.rate, self.rope_dim, self.rope_theta, self.yarn)
        rotated = rotate_trailing(pooled, cos, sin)
        if write:
            buffer_kv.value, buffer_gate.value = buffered
            entries.value = write_cache(entries.value, rotated, emitted)
            if overlap_kv is not None and overlap_gate is not None and prior is not None:
                overlap_kv.value, overlap_gate.value = prior
        return entries.value


class IndexScorer(nn.Module):
    """The lightning indexer's score of every query against every entry:
    `sum_h w_h relu(q_h . k) / sqrt(head_dim)` with `w = weights_proj(x) /
    sqrt(n_heads)`, in fp32 (`DeepseekV4IndexerScorer`,
    modeling_deepseek_v4.py:437-450)."""

    n_heads: int
    head_dim: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, query, keys, x):
        scores = jnp.maximum(jnp.einsum('bshd,btd->bsht', query.astype(jnp.float32),
                                        keys.astype(jnp.float32), precision=self.precision), 0) * self.head_dim ** -0.5
        weights = nn.Dense(self.n_heads, use_bias=False, dtype=self.dtype,
                           precision=self.precision, name='weights_proj')(x)
        return jnp.einsum('bsht,bsh->bst', scores, weights.astype(jnp.float32) * self.n_heads ** -0.5,
                          precision=self.precision)


class LightningIndexer(CompressedEntries):
    """CSA's lightning indexer: its own compressor at `head_dim` over the same
    windows (the inherited projections), queries from the query residual,
    and the V3.2 score (modeling_deepseek_v4.py:453-578).

    The indexer reads its inputs detached, as V3.2's does: the top-k is a
    selection the main loss cannot reach, and no indexer loss is trained
    here (the reference trains none either).
    """

    n_heads: int = 1
    top_k: int = 1

    def setup(self):
        if self.width <= self.rope_dim:
            raise ValueError(
                f"the indexer rotates a {self.rope_dim}-wide rope slice out of heads "
                f"of width {self.width}, so the heads have to be wider")
        super().setup()
        self.q_b_proj = nn.Dense(self.n_heads * self.width, use_bias=False,
                                 dtype=self.dtype, precision=self.precision, name='q_b_proj')
        self.scorer = IndexScorer(n_heads=self.n_heads, head_dim=self.width,
                                  dtype=self.dtype, precision=self.precision, name='scorer')

    def select(self, x, q_resid, positions, cos, sin, cache=None):
        """The entries each query attends: `[B, S, T]`, bool.

        The top-k among the entries whose window closed before the query
        (:568-575: later entries score `-inf`, and a pick past the causal
        threshold is dropped, so a query with fewer than k allowed entries
        attends all of them).
        """
        x = jax.lax.stop_gradient(x)
        q_resid = jax.lax.stop_gradient(q_resid)
        batch, length, _ = x.shape
        keys = self.entries(x) if cache is None else self.cached_entries(x, *cache)
        query = rotate_trailing(
            self.q_b_proj(q_resid).reshape(batch, length, self.n_heads, self.width), cos, sin)
        scores = self.scorer(query, keys, x)
        return top_k_keys(scores, entries_visible(positions, keys.shape[1], self.rate), self.top_k)


class Compressor(CompressedEntries):
    """The layer's compressor: its entries plus, on a CSA layer, the lightning
    indexer that selects among them (modeling_deepseek_v4.py:580-693); an HCA
    layer attends every entry whose window closed (:353-435)."""

    index_n_heads: int | None = None
    index_head_dim: int | None = None
    index_topk: int | None = None

    def setup(self):
        super().setup()
        if (self.index_n_heads is not None and self.index_head_dim is not None
                and self.index_topk is not None):
            self.indexer = LightningIndexer(
                width=self.index_head_dim, rate=self.rate, overlap=True, rope_dim=self.rope_dim,
                rope_theta=self.rope_theta, yarn=self.yarn, norm_eps=self.norm_eps,
                n_heads=self.index_n_heads, top_k=self.index_topk,
                dtype=self.dtype, precision=self.precision, name='indexer')

    def __call__(self, x, q_resid, positions, cos, sin, cache=None):
        """The entries `[B, T, width]` and which of them each query attends `[B, S, T]`."""
        entries = self.entries(x) if cache is None else self.cached_entries(x, *cache)
        if self.index_topk is not None:
            return entries, self.indexer.select(x, q_resid, positions, cos, sin, cache)
        visible = entries_visible(positions, entries.shape[1], self.rate)
        return entries, jnp.broadcast_to(visible, (x.shape[0], *visible.shape[1:]))


@logical_axes({
    ("q_a_proj",): ("embed", "qlora"),
    ("q_b_proj",): ("qlora", "attention"),
    ("q_a_norm",): ("qlora",),
    # One shared key/value head: nothing to shard on the output side.
    ("kv_proj",): ("embed", None),
    ("o_a_proj",): (None, "attention", None),
    ("o_b_proj",): (None, "embed"),
    # The compressors' projections are the entry width or twice it; the
    # longer suffixes keep them off the dense feed-forward's rule.
    ("compressor", "kv_proj"): ("embed", None),
    ("compressor", "gate_proj"): ("embed", None),
    ("indexer", "kv_proj"): ("embed", None),
    ("indexer", "gate_proj"): ("embed", None),
    ("indexer", "q_b_proj"): ("qlora", "index"),
    ("scorer", "weights_proj"): ("embed", "index"),
})
class DeepseekV4Attention(nn.Module):
    """One V4 attention layer, sliding or compressed (see the module doc).

    `compressor` is None for a sliding layer, 'hca' or 'csa' otherwise, with
    `compress_rate` its window; a CSA layer needs the indexer's
    `index_n_heads`, `index_head_dim` and `index_topk`. `rope_theta` and
    `yarn` are the layer's rope (the main one on a sliding layer, the
    compress one otherwise), which the compressor and indexer share.
    Compressed windows follow the physical row, not document boundaries.
    Packed documents are refused rather than allowing a pooled entry to
    carry information across segments.
    """

    emb_features: int
    num_heads: int
    head_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    rope_dim: int
    sliding_window: int
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    yarn: YarnScaling | None = None
    compressor: str | None = None
    compress_rate: int | None = None
    index_topk: int | None = None
    index_n_heads: int | None = None
    index_head_dim: int | None = None
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        if self.rope_dim % 2 or not 0 < self.rope_dim <= self.head_dim:
            raise ValueError(
                f"the rope slice rotates pairs inside the head, so rope_dim is even and "
                f"at most head_dim ({self.head_dim}), got {self.rope_dim}")
        if (self.num_heads * self.head_dim) % self.o_groups:
            raise ValueError(
                f"o_groups ({self.o_groups}) has to divide the heads' width "
                f"({self.num_heads * self.head_dim})")
        if self.compressor is not None and self.compressor not in COMPRESSORS:
            raise ValueError(f"compressor is one of {COMPRESSORS} or None, got {self.compressor!r}")
        if (self.compressor is None) != (self.compress_rate is None):
            raise ValueError("a compressor and its compress_rate come together")
        index = (self.index_topk, self.index_n_heads, self.index_head_dim)
        if self.compressor == 'csa' and any(field is None for field in index):
            raise ValueError("a csa layer selects with the indexer, which index_topk, "
                             "index_n_heads and index_head_dim describe")
        if self.compressor != 'csa' and any(field is not None for field in index):
            raise ValueError("only a csa layer carries the indexer")
        dense = functools.partial(nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
        norm = functools.partial(RMSNorm, epsilon=self.norm_eps, scale_after_cast=True, dtype=self.dtype)
        self.q_a_proj = dense(self.q_lora_rank, name='q_a_proj')
        self.q_a_norm = norm(name='q_a_norm')
        self.q_b_proj = dense(self.num_heads * self.head_dim, name='q_b_proj')
        self.kv_proj = dense(self.head_dim, name='kv_proj')
        self.kv_norm = norm(name='kv_norm')
        self.o_a_proj = GroupedLinear(groups=self.o_groups, features=self.o_lora_rank,
                                      dtype=self.dtype, precision=self.precision, name='o_a_proj')
        self.o_b_proj = dense(self.emb_features, name='o_b_proj')
        self.sinks = self.param('sinks', nn.initializers.zeros, (self.num_heads,), jnp.float32)
        if self.compressor is not None and self.compress_rate is not None:
            self.compress = Compressor(
                width=self.head_dim, rate=self.compress_rate, overlap=self.compressor == 'csa',
                rope_dim=self.rope_dim, rope_theta=self.rope_theta, yarn=self.yarn,
                norm_eps=self.norm_eps, index_n_heads=self.index_n_heads,
                index_head_dim=self.index_head_dim, index_topk=self.index_topk,
                dtype=self.dtype, precision=self.precision, name='compressor')

    @nn.compact
    def __call__(self, x, decode: bool = False, positions=None, segment_ids=None,
                 kv_store=None, attention_metadata: AttentionMetadata | None = None):
        if segment_ids is not None and self.compressor is not None:
            raise ValueError("segment_ids are not supported by deepseek_v4 compressed windows")
        if attention_metadata is not None and attention_metadata.rotary_positions is not None:
            raise ValueError("the deepseek_v4 mixer does not implement multi-axis rotary positions")
        valid = (None if attention_metadata is None or attention_metadata.valid is None
                 else jnp.asarray(attention_metadata.valid, bool))
        batch, length, _ = x.shape
        cache = None
        cached_key = None
        if decode:
            if segment_ids is not None:
                raise ValueError("decode accepts row validity, not packed segment_ids")
            if valid is None and length > self.max_seq_len:
                raise ValueError(f"{length} tokens exceed max_seq_len={self.max_seq_len}")
            slots, allocated = _cache_positions(self, batch, length, self.max_seq_len, valid)
            cache = (slots, self.max_seq_len, allocated)
            cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                       (batch, self.max_seq_len, self.head_dim), x.dtype)
            if positions is None:
                positions = slots
        if positions is None:
            positions = jnp.arange(length) if valid is None else jnp.maximum(jnp.cumsum(valid, axis=1) - 1, 0)
        positions = jnp.asarray(positions)
        rows = jnp.arange(length)
        cos, sin = rope_freqs(positions, self.rope_dim, self.rope_theta, self.yarn)

        q_resid = self.q_a_norm(self.q_a_proj(x))
        query = checkpoint_name(self.q_b_proj(q_resid), 'q_proj').reshape(
            batch, length, self.num_heads, self.head_dim)
        query = rotate_trailing(unweighted_rmsnorm(query, self.norm_eps), cos, sin)
        keys = rotate_trailing(checkpoint_name(self.kv_norm(self.kv_proj(x)), 'kv_proj'), cos, sin)

        # The sliding window over the row's own positions: key j at or before
        # query i and within the window, its own position included
        # (masking_utils sliding_window_overlay, modeling_deepseek_v4.py:205).
        query_pos, key_pos = rows[:, None], rows[None, :]
        allowed = (key_pos <= query_pos) & (key_pos > query_pos - self.sliding_window)
        allowed = jnp.broadcast_to(allowed, (batch, length, length))
        if cache is not None and cached_key is not None:
            slots, _, allocated = cache
            if allocated:
                cached_key.value = write_cache(cached_key.value, keys, slots)
            keys = cached_key.value
            allowed = causal_attention_mask(
                slots, keys.shape[1], self.sliding_window,
                key_valid=self.get_variable('cache', 'cache_valid'))[:, 0]
        if segment_ids is not None:
            allowed = allowed & document_mask(segment_ids)
        if valid is not None and not decode:
            allowed = allowed & valid[:, None, :]

        if self.compressor is not None:
            selected_positions = positions if cache is None else cache[0]
            entries, selected = self.compress(x, q_resid, selected_positions, cos, sin, cache)
            if entries.shape[1]:
                keys = jnp.concatenate([keys, entries], axis=1)
                allowed = jnp.concatenate([allowed, selected], axis=-1)

        # One shared key/value head under every query head, the per-head
        # sink beside the logits, softmax in fp32 and the sink dropped
        # (:708-736).
        logits = jnp.einsum('bshd,btd->bhst', query.astype(jnp.float32), keys.astype(jnp.float32),
                            precision=self.precision) * self.head_dim ** -0.5
        logits = jnp.where(allowed[:, None], logits, -jnp.inf)
        sinks = jnp.broadcast_to(self.sinks[None, :, None, None], (batch, self.num_heads, length, 1))
        probs = jax.nn.softmax(jnp.concatenate([logits, sinks], axis=-1), axis=-1)[..., :-1]
        context = jnp.einsum('bhst,btd->bshd', probs.astype(keys.dtype), keys, precision=self.precision)
        # The values are the rotated keys, so the rope slice of the output is
        # turned back at the query's position (:853-859).
        context = rotate_trailing(checkpoint_name(context, 'context'), cos, -sin)
        mixed = self.o_a_proj(context.reshape(batch, length, self.o_groups, -1))
        output = checkpoint_name(self.o_b_proj(mixed.reshape(batch, length, -1)), 'o_proj')
        return output if valid is None else jnp.where(valid[..., None], output, 0)


@mixers("deepseek_v4")
@dataclasses.dataclass(frozen=True)
class DeepseekV4Mixer(MixerBase):
    """The `deepseek_v4` kind, by the config's field names.

    The model names the sliding layer (`compressor` None); the two
    compressed kinds name their `compressor` ('csa' or 'hca') and its
    `compress_rate` on their `LayerKind.mixer`, with the indexer's three
    fields on the csa kind. The window is the context's `sliding_window`
    (every V4 layer has one), the rope base and YaRN ramp the context's
    kind-resolved ones, and `rope_head_dim` the rotated slice
    (`head_dim * partial_rotary_factor`). The context's `num_kv_heads`,
    `qk_norm`, `v_norm`, `k_eq_v`, `kv_shared`, `attention_scale`,
    `partial_rotary_factor`, `output_gate` and `attention_sinks` are the
    standard attention's dials and are not read: V4's single KV head,
    unweighted query norm, sinks and de-rotated values are the layer's own.
    """

    q_lora_rank: int = 1024
    o_groups: int = 8
    o_lora_rank: int = 1024
    rope_head_dim: int = 64
    compressor: str | None = None
    compress_rate: int | None = None
    index_topk: int | None = None
    index_n_heads: int | None = None
    index_head_dim: int | None = None

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        if not ctx.causal:
            raise ValueError("the deepseek_v4 mixer is causal: its window and compressors read the past alone")
        if ctx.sliding_window is None:
            raise ValueError("every deepseek_v4 layer attends a sliding window, which its kind names")
        if ctx.attention_chunk is not None:
            raise ValueError("a deepseek_v4 layer attends a sliding window, not a chunk")
        if ctx.kv_shared:
            raise ValueError("the deepseek_v4 mixer shares no keys across layers")
        if ctx.kv_cache != KVCache():
            raise ValueError("the deepseek_v4 mixer keeps its own compressed cache; it takes no kv_cache layout")
        if ctx.yarn is not None and ctx.yarn.rope_theta != ctx.rope_theta:
            raise ValueError(
                f"the yarn record's rope_theta ({ctx.yarn.rope_theta}) and the layer's "
                f"({ctx.rope_theta}) disagree; the rope base is configured once, on the kind")
        return functools.partial(
            DeepseekV4Attention,
            emb_features=ctx.emb_features,
            num_heads=ctx.num_heads,
            head_dim=ctx.head_dim,
            q_lora_rank=self.q_lora_rank,
            o_groups=self.o_groups,
            o_lora_rank=self.o_lora_rank,
            rope_dim=self.rope_head_dim,
            sliding_window=ctx.sliding_window,
            max_seq_len=ctx.max_seq_len,
            rope_theta=ctx.rope_theta,
            yarn=ctx.yarn,
            compressor=self.compressor,
            compress_rate=self.compress_rate,
            index_topk=self.index_topk,
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            norm_eps=ctx.norm_eps,
            dtype=ctx.dtype,
            precision=ctx.precision)
