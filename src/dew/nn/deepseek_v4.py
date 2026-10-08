"""DeepSeek V4's attention: shared-KV MQA over a sliding window, extended by
a compressor.

The reference is transformers 5.16.1 `models/deepseek_v4/modeling_deepseek_v4.py`
(`DeepseekV4Attention`, :746-864). The query is low-rank and normed per head
without a weight, one key/value head serves every query head, rotary turns
the trailing `rope_head_dim` of each head in interleaved pairs, a learned
sink sits beside the logits, the values are the rotated keys (so the output
is de-rotated at the query's position), and the output projection is grouped
and low-rank.

A sliding layer attends the last `sliding_window` positions, its own
included. A compressed layer adds the entries its compressor emits, one per
window of `compress_rate` tokens (HCA :353-435, CSA :580-693 with its
indexer :437-578), with a per-query bias: HCA attends every entry whose
window closed before the query, CSA the indexer's top-k of those. A trailing
partial window emits nothing outside a cache (:398, :630) and is buffered
inside one; cached calls never reproject old hidden states (:209-291).

DeepSeek-V4.1-Flash replaces CSA and HCA with CSA2 (arXiv 2609.19969, section
2.3; the release's inference/model.py at the revision
tools/deepseek_v41_reference.py pins, cited as v41:line). Its compressor
pools non-overlapping windows without position bias in fp32, and at rate 1
is the normed projection alone (v41:429-485). Its indexer projects keys from
the compressor's latent before RoPE (v41:488-580). KV, index keys and
selections are shared across layers through the kv_store (`_CSA2_*`): a Full
layer computes all three, a Reindex layer rescores the latest keys, a Reuse
layer attends the latest selection (v41:613-763), and the first Full layer
builds the candidate pool later Reindex layers search (section 2.3.2,
v41:583-610). The query is not normed per head (v41:770-772), and
quantization-aware training rounds the cache as the release stores it
(section 2.4.4), each rounding after its RoPE with a straight-through
gradient: window keys through FP8 E4M3 per 32 channels with power-of-two
scales, compressed entries through FP4 E2M1 per 16 with E4M3 scales, index
queries and keys through FP4 per 32 with power-of-two scales (v41:545-552,
:705-707, :759-760; kernel.py:40-204).
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
from jax.typing import DTypeLike

from dew.nn.attention import (
    RMSNorm,
    cache_positions,
    cached_validity,
    causal_attention_mask,
    document_mask,
    unweighted_rmsnorm,
)
from dew.nn.fake_quant import fake_quant_fp4, fake_quant_fp8
from dew.nn.inputs import AttentionMetadata
from dew.nn.kv_cache import KVCache, write_cache
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.nn.precision import at_least_fp32
from dew.nn.rope import YarnScaling, rotary_freqs, rotate, yarn_inv_freq
from dew.nn.sharding import RESIDUAL, LogicalAxes, constrain, down_projection, logical_axes
from dew.nn.sparse_selection import candidate_pool, selection_mask, top_k_keys, top_k_selection

COMPRESSORS = ('csa', 'hca', 'csa2')
CANDIDATES = ('source', 'restrict')
_CSA2_ENTRIES = 'csa2_entries'
_CSA2_INDEX_KEYS = 'csa2_index_keys'
_CSA2_SELECTED = 'csa2_selected'
_CSA2_CANDIDATES = 'csa2_candidates'
"""The kv_store names a CSA2 layer publishes under: the latest Full layer's
rotated entries and index keys, the latest selection and the candidate
pool's entries. One name each, since every layer reads the latest (section
2.3.1)."""


def rope_freqs(positions, rope_dim: int, theta: float, yarn: YarnScaling | None, *,
               dtype: DTypeLike):
    """cos/sin per pair over the rope width, `[..., rope_dim // 2]`, in `dtype`
    (`at_least_fp32` of the activations it rotates).

    V4's rotary scales neither table: its YaRN entry forces
    `attention_factor` to 1.0 (configuration_deepseek_v4.py:294-320) and
    the plain entry never scales (modeling_deepseek_v4.py:136), so the
    ramp changes the frequencies alone.
    """
    if yarn is None:
        return rotary_freqs(positions, rope_dim, theta, dtype=dtype)
    inv_freq = yarn_inv_freq(rope_dim, theta, yarn, dtype=dtype)
    angles = jnp.asarray(positions, inv_freq.dtype)[..., None] * inv_freq
    return jnp.cos(angles), jnp.sin(angles)


_rotate_rope = functools.partial(rotate, pairs='adjacent', span='trailing')
"""V4's rotary: the trailing rope channels of a head or an entry turn in
interleaved pairs that land back in place, the nope channels before them
untouched (modeling_deepseek_v4.py:326-350), so a rotation by the negated
sine undoes it, which the attention output relies on."""


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
    gate = gate[:, :windows * rate].reshape(batch, windows, rate, features)
    if position_bias is not None:
        gate = gate + position_bias
    if overlap:
        width = features // 2
        previous_kv = jnp.concatenate(
            [jnp.zeros_like(kv[:, :1, :, :width]), kv[:, :-1, :, :width]], axis=1)
        previous_gate = jnp.concatenate(
            [jnp.full_like(gate[:, :1, :, :width], -jnp.inf), gate[:, :-1, :, :width]], axis=1)
        kv = jnp.concatenate([previous_kv, kv[..., width:]], axis=2)
        gate = jnp.concatenate([previous_gate, gate[..., width:]], axis=2)
    weights = jax.nn.softmax(gate.astype(at_least_fp32(gate.dtype)), axis=2).astype(kv.dtype)
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
            weights = jax.nn.softmax(window_gate.astype(at_least_fp32(window_gate.dtype)),
                                     axis=1).astype(window_key.dtype)
            entry = jnp.sum(window_key * weights, axis=1)
            at = jnp.where(closed, count, -1)[:, None]
            values = write_cache(values, entry[:, None], at)
            indices = write_cache(
                indices[..., None], (position // rate).astype(indices.dtype)[:, None, None], at
            )[..., 0]
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
    `gate_proj` into the entry width (twice it for the two-series CSA layout),
    the position bias and the entry norm; `entries` pools the windows and
    rotates each entry at its window's first position, `w * rate`
    (modeling_deepseek_v4.py:375-413, :603-671). The indexer's compressor sits
    under `compressor/indexer`, as the checkpoint nests it
    (conversion_mapping.py:487).

    V4.1's compressor (v41:429-485) has no position bias and pools in fp32
    (`pool_dtype`); at rate 1 there is no gate and the normed projection is the
    entry. `quantized` rounds the rotated entries through FP4 (v41:759-760).
    """

    width: int
    rate: int
    overlap: bool
    rope_dim: int
    rope_theta: float
    yarn: YarnScaling | None
    norm_eps: float
    position_bias: bool = True
    pool_dtype: Dtype | None = None
    quantized: bool = False
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @property
    def gated(self) -> bool:
        return self.rate > 1

    def setup(self):
        series = 2 if self.overlap else 1
        projected = self.pool_dtype if self.gated and self.pool_dtype is not None else self.dtype
        dense = functools.partial(nn.Dense, use_bias=False, dtype=projected, precision=self.precision)
        self.kv_proj = dense(series * self.width, name='kv_proj')
        if self.gated:
            self.gate_proj = dense(series * self.width, name='gate_proj')
            if self.position_bias:
                self.bias = self.param(
                    'position_bias', nn.initializers.zeros, (self.rate, series * self.width), jnp.float32)
        self.kv_norm = RMSNorm(epsilon=self.norm_eps, scale_after_cast=True,
                               dtype=self.dtype, name='kv_norm')

    def _projections(self, x):
        """`kv` and the gate over it, in the pooling's dtype (v41:464-465)."""
        if not self.gated:
            return self.kv_proj(x), None
        if self.pool_dtype is not None:
            x = x.astype(self.pool_dtype)
        return self.kv_proj(x), self.gate_proj(x)

    def _normed(self, pooled, dtype):
        """The entry norm over pooled windows, back in the model's `dtype`
        first where the pooling ran in its own (v41:485)."""
        return self.kv_norm(pooled if self.pool_dtype is None else pooled.astype(dtype))

    def latents(self, x):
        """The normed entries before RoPE `[B, T, width]`, `T = S // rate`."""
        kv, gate = self._projections(x)
        if gate is None:
            return self.kv_norm(kv)
        bias = self.bias.astype(x.dtype) if self.position_bias else None
        return self._normed(pool_windows(kv, gate, bias, self.rate, self.overlap), x.dtype)

    def rotate(self, latents, windows):
        """Rotate each entry at its window's first position, `window * rate`."""
        cos, sin = rope_freqs(windows * self.rate, self.rope_dim, self.rope_theta, self.yarn,
                              dtype=at_least_fp32(latents.dtype))
        return _rotate_rope(latents, cos, sin)

    def entries(self, x):
        """The rotated entries `[B, T, width]`, `T = S // rate`."""
        return self.entries_and_latents(x)[0]

    def entries_and_latents(self, x):
        """The entries and the latents they rotate."""
        latents = self.latents(x)
        return self._stored(latents, jnp.arange(latents.shape[1])), latents

    def _stored(self, latents, windows):
        rotated = self.rotate(latents, windows)
        return fake_quant_fp4(rotated, 16, e4m3_scale=True) if self.quantized else rotated

    @nn.compact
    def cached_entries(self, x, slots, capacity: int, write: bool):
        """Append closed windows to the rotated-entry cache; allocation writes none.

        Returns the cache, the latents this call closed and the window each
        one is (-1 for none). The incomplete window stays in the cache; a
        window of one token closes with the token, so at rate 1 every valid
        token is an entry at once, with no buffer and no scan.
        """
        kv, gate = self._projections(x)
        if gate is None:
            latents, emitted = self.kv_norm(kv), slots
        else:
            if self.position_bias:
                gate = gate + self.bias[jnp.maximum(slots, 0) % self.rate].astype(gate.dtype)
            shape = (x.shape[0], self.rate, kv.shape[-1])
            buffer_kv = self.variable('cache', 'buffer_kv', jnp.zeros, shape, kv.dtype)
            buffer_gate = self.variable('cache', 'buffer_gate', jnp.zeros, shape, gate.dtype)
            prior = overlap = None
            if self.overlap:
                overlap = (self.variable('cache', 'overlap_kv', jnp.zeros,
                                         (x.shape[0], self.rate, self.width), kv.dtype),
                           self.variable('cache', 'overlap_gate', jnp.full,
                                         (x.shape[0], self.rate, self.width), -jnp.inf, gate.dtype))
                prior = (overlap[0].value, overlap[1].value)
            pooled, emitted, buffered, prior = append_windows(
                kv, gate, slots, (buffer_kv.value, buffer_gate.value), prior, self.rate, self.width)
            if write:
                buffer_kv.value, buffer_gate.value = buffered
                if overlap is not None and prior is not None:
                    overlap[0].value, overlap[1].value = prior
            latents = self._normed(pooled, x.dtype)
        entries = self.variable('cache', 'compressed', jnp.zeros,
                                (x.shape[0], capacity // self.rate, self.width), latents.dtype)
        if write:
            entries.value = write_cache(entries.value, self._stored(latents, emitted), emitted)
        return entries.value, latents, emitted


class IndexScorer(nn.Module):
    """The lightning indexer's score of every query against every entry:
    `sum_h w_h relu(q_h . k) / sqrt(head_dim)` with `w = weights_proj(x) /
    sqrt(n_heads)`, in fp32 (`DeepseekV4IndexerScorer`,
    modeling_deepseek_v4.py:437-450)."""

    n_heads: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, query, keys, x):
        weights = nn.Dense(self.n_heads, use_bias=False, dtype=self.dtype,
                           precision=self.precision, name='weights_proj')(x)
        return _index_scores(query, keys, weights, self.precision)


def _index_scores(query, keys, weights, precision=None):
    """`sum_h w_h relu(q_h . k) / sqrt(head_dim) / sqrt(n_heads)` in fp32
    (`at_least_fp32`), over keys `[B, T, D]` every query shares or
    `[B, S, T, D]` per query."""
    heads, width = query.shape[-2:]
    wide = at_least_fp32(query.dtype)
    shared = 'bshd,btd->bsht' if keys.ndim == 3 else 'bshd,bstd->bsht'
    scores = jnp.maximum(jnp.einsum(shared, query.astype(wide),
                                    keys.astype(wide), precision=precision), 0) * width ** -0.5
    return jnp.einsum('bsht,bsh->bst', scores, weights.astype(wide) * heads ** -0.5,
                      precision=precision)


class LightningIndexer(CompressedEntries):
    """CSA's lightning indexer: its own compressor at `head_dim` over the same
    windows, queries from the query residual, and the V3.2 score
    (modeling_deepseek_v4.py:453-578). It reads its inputs detached and trains
    no loss, as the reference's.
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
        self.scorer = IndexScorer(n_heads=self.n_heads, dtype=self.dtype, precision=self.precision,
                                  name='scorer')

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
        keys = self.entries(x) if cache is None else self.cached_entries(x, *cache)[0]
        query = _rotate_rope(
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
        entries = self.entries(x) if cache is None else self.cached_entries(x, *cache)[0]
        if self.index_topk is not None:
            return entries, self.indexer.select(x, q_resid, positions, cos, sin, cache)
        visible = entries_visible(positions, entries.shape[1], self.rate)
        return entries, jnp.broadcast_to(visible, (x.shape[0], *visible.shape[1:]))


class Csa2Indexer(nn.Module):
    """CSA2's indexer (v41:488-580): queries from the query residual, scored
    against index keys a Full layer projects from its compressor's latent (`wk`,
    `k_norm`) and the other layers read from the store, under V3.2's leaf names.
    It reads its inputs detached and trains no loss, as the reference's.
    """

    n_heads: int
    head_dim: int
    rope_dim: int
    owns_keys: bool
    quantized: bool
    norm_eps: float
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        if self.head_dim <= self.rope_dim:
            raise ValueError(
                f"the indexer rotates a {self.rope_dim}-wide rope slice out of heads "
                f"of width {self.head_dim}, so the heads have to be wider")
        dense = functools.partial(nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
        self.wq_b = dense(self.n_heads * self.head_dim, name='wq_b')
        self.weights_proj = dense(self.n_heads, name='weights_proj')
        if self.owns_keys:
            self.wk = dense(self.head_dim, name='wk')
            self.k_norm = RMSNorm(epsilon=self.norm_eps, scale_after_cast=True,
                                  dtype=self.dtype, name='k_norm')

    def _fp4(self, x):
        return fake_quant_fp4(x, 32, e4m3_scale=False) if self.quantized else x

    def keys(self, latents, cos, sin):
        """Index keys `[B, T, head_dim]` off the latents, rotated at their
        windows' positions (v41:537-547)."""
        return self._fp4(_rotate_rope(self.k_norm(self.wk(jax.lax.stop_gradient(latents))), cos, sin))

    def scores(self, x, q_resid, keys, cos, sin):
        """Every query's score of every entry, `[B, S, T]` (v41:550-557)."""
        x, q_resid = jax.lax.stop_gradient(x), jax.lax.stop_gradient(q_resid)
        batch, length, _ = x.shape
        query = _rotate_rope(
            self.wq_b(q_resid).reshape(batch, length, self.n_heads, self.head_dim), cos, sin)
        return _index_scores(self._fp4(query), jax.lax.stop_gradient(keys), self.weights_proj(x),
                            self.precision)


def _published(kv_store, name: str, layer: str):
    if kv_store is None or name not in kv_store:
        raise ValueError(
            f"a CSA2 {layer} layer reads {name} from the latest layer that publishes it; "
            "the model threads one kv_store down its stack when kv_shared_layers are set, "
            "and a Full layer has to come first")
    return kv_store[name]


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
    ("indexer", "wq_b"): ("qlora", "index"),
    ("indexer", "weights_proj"): ("embed", "index"),
    ("scorer", "weights_proj"): ("embed", "index"),
})
class DeepseekV4Attention(nn.Module):
    """One V4 attention layer, sliding or compressed (see the module doc).

    `compressor` is None, 'hca', 'csa' or V4.1's 'csa2', with `compress_rate`
    its window; CSA needs `index_n_heads`, `index_head_dim` and `index_topk`.
    `rope_theta` and `yarn` are the layer's rope, shared by its compressor and
    indexer. Compressed windows follow the physical row, so packed documents
    are refused. For 'csa2', `kv_shared` makes a Reuse layer and `reindex` a
    Reindex one, else a Full layer; `candidates` 'source' builds the candidate
    pool and 'restrict' searches it. `query_norm` is V4's per-head query norm,
    and `kv_qat` V4.1's FP8/FP4 rounding.
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
    kv_shared: bool = False
    reindex: bool = False
    candidates: str | None = None
    candidate_blocks: int | None = None
    candidate_block_size: int | None = None
    query_norm: bool = True
    kv_qat: bool = False
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
        if self.compressor in ('csa', 'csa2') and any(field is None for field in index):
            raise ValueError(f"a {self.compressor} layer selects with the indexer, which "
                             "index_topk, index_n_heads and index_head_dim describe")
        if self.compressor not in ('csa', 'csa2') and any(field is not None for field in index):
            raise ValueError("only a csa or csa2 layer carries the indexer")
        if self.compressor != 'csa2' and (self.kv_shared or self.reindex or self.candidates):
            raise ValueError("sharing entries, reindexing and candidate pools are CSA2's")
        if self.reindex and self.kv_shared:
            raise ValueError("a Reindex layer computes its own selection, which a Reuse "
                             "layer reads instead; a layer is one or the other")
        if self.candidates is not None:
            if self.candidates not in CANDIDATES:
                raise ValueError(f"candidates is one of {CANDIDATES} or None, got {self.candidates!r}")
            if self.candidate_blocks is None or self.candidate_block_size is None:
                raise ValueError("a candidate pool is candidate_blocks blocks of candidate_block_size")
            # A Reuse layer rides its Full layer's kind and computes no selection.
            if not self.kv_shared and (self.candidates == 'source') == self.reindex:
                raise ValueError("a Full layer builds the candidate pool and a Reindex layer "
                                 "searches within it")
            if (self.candidate_blocks - 1) * self.candidate_block_size + 1 < (self.index_topk or 0):
                raise ValueError(
                    f"a pool of {self.candidate_blocks} blocks of {self.candidate_block_size} can "
                    f"hold fewer than index_topk ({self.index_topk}) visible entries, where the "
                    f"reference's top-k falls back on entries outside the pool in tie order")
        if self.kv_qat and (self.head_dim % 32 or (self.index_head_dim or 32) % 32):
            raise ValueError("the cache rounds the keys per 32 channels and the index keys "
                             "per 32, so head_dim and index_head_dim are multiples of 32")
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
        if self.compressor == 'csa2':
            rate, index = self._csa2_geometry()
            if not self.kv_shared and not self.reindex:
                self.compress = CompressedEntries(
                    width=self.head_dim, rate=rate, overlap=False,
                    rope_dim=self.rope_dim, rope_theta=self.rope_theta, yarn=self.yarn,
                    norm_eps=self.norm_eps, position_bias=False, pool_dtype=at_least_fp32(self.dtype),
                    quantized=self.kv_qat,
                    dtype=self.dtype, precision=self.precision, name='compressor')
            if not self.kv_shared:
                self.indexer = Csa2Indexer(
                    n_heads=index[1], head_dim=index[2], rope_dim=self.rope_dim,
                    owns_keys=not self.reindex, quantized=self.kv_qat,
                    norm_eps=self.norm_eps, dtype=self.dtype, precision=self.precision, name='indexer')
        elif self.compressor is not None and self.compress_rate is not None:
            self.compress = Compressor(
                width=self.head_dim, rate=self.compress_rate, overlap=self.compressor == 'csa',
                rope_dim=self.rope_dim, rope_theta=self.rope_theta, yarn=self.yarn,
                norm_eps=self.norm_eps, index_n_heads=self.index_n_heads,
                index_head_dim=self.index_head_dim, index_topk=self.index_topk,
                dtype=self.dtype, precision=self.precision, name='compressor')

    def _csa2_geometry(self) -> tuple[int, tuple[int, int, int]]:
        """A CSA2 layer's rate and its indexer's (top_k, heads, head width),
        which setup has already required."""
        rate, index = self.compress_rate, (self.index_topk, self.index_n_heads, self.index_head_dim)
        assert rate is not None and index[0] is not None and index[1] is not None and index[2] is not None
        return rate, (index[0], index[1], index[2])

    def _csa2(self, x, q_resid, positions, cos, sin, cache, kv_store):
        """CSA2's entries `[B, T, head_dim]` and the ones each query attends
        `[B, S, T]`, by the layer's mode (section 2.3.1, v41:722-763).

        A Full layer publishes its index keys on every call. The release
        republishes them only when one of the layer's groups closes
        (v41:537-548), so in a call where none does, a Reindex layer after
        it reads the keys of whichever layer last published; Dew does not
        reproduce that."""
        if self.kv_shared:
            return (_published(kv_store, _CSA2_ENTRIES, 'Reuse'),
                    _published(kv_store, _CSA2_SELECTED, 'Reuse'))
        rate, (top_k, _, index_head_dim) = self._csa2_geometry()
        if self.reindex:
            entries = _published(kv_store, _CSA2_ENTRIES, 'Reindex')
            keys = _published(kv_store, _CSA2_INDEX_KEYS, 'Reindex')
        elif cache is None:
            entries, latents = self.compress.entries_and_latents(x)
            windows = jnp.arange(latents.shape[1])
            keys = self.indexer.keys(latents, *rope_freqs(windows * rate, self.rope_dim,
                                                          self.rope_theta, self.yarn,
                                                          dtype=at_least_fp32(latents.dtype)))
        else:
            slots, capacity, allocated = cache
            entries, latents, windows = self.compress.cached_entries(x, slots, capacity, allocated)
            held = self.variable('cache', 'index_keys', jnp.zeros,
                                 (x.shape[0], capacity // rate, index_head_dim), latents.dtype)
            fresh = self.indexer.keys(latents, *rope_freqs(windows * rate, self.rope_dim,
                                                           self.rope_theta, self.yarn,
                                                           dtype=at_least_fp32(latents.dtype)))
            if allocated:
                held.value = write_cache(held.value, fresh, windows)
            keys = held.value
        visible = entries_visible(positions, entries.shape[1], rate)
        visible = jnp.broadcast_to(visible, (x.shape[0], *visible.shape[1:]))
        if self.candidates == 'restrict':
            # A Reindex layer scores the pool's entries alone, their keys
            # gathered per query (section 2.3.2): O(pool), not O(entries).
            pool = _published(kv_store, _CSA2_CANDIDATES, 'Reindex')
            held = jnp.maximum(pool, 0)
            scores = self.indexer.scores(x, q_resid, keys[jnp.arange(x.shape[0])[:, None, None], held],
                                         cos, sin)
            picks = top_k_selection(scores, (pool >= 0) & jnp.take_along_axis(visible, held, -1), top_k)
            chosen = jnp.where(picks >= 0, jnp.take_along_axis(pool, jnp.maximum(picks, 0), -1), -1)
            selected = selection_mask(chosen, entries.shape[1])
        else:
            scores = self.indexer.scores(x, q_resid, keys, cos, sin)
            if (self.candidates == 'source' and self.candidate_blocks and self.candidate_block_size
                    and kv_store is not None):
                kv_store[_CSA2_CANDIDATES] = candidate_pool(
                    scores, visible, self.candidate_blocks, self.candidate_block_size)
            selected = top_k_keys(scores, visible, top_k)
        if kv_store is not None:
            if not self.reindex:
                kv_store[_CSA2_ENTRIES] = entries
                kv_store[_CSA2_INDEX_KEYS] = keys
            kv_store[_CSA2_SELECTED] = selected
        return entries, selected

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
            slots, allocated = cache_positions(self, batch, length, valid)
            cache = (slots, self.max_seq_len, allocated)
            cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                       (batch, self.max_seq_len, self.head_dim), x.dtype)
            if positions is None:
                positions = slots
        if positions is None:
            positions = jnp.arange(length) if valid is None else jnp.maximum(jnp.cumsum(valid, axis=1) - 1, 0)
        positions = jnp.asarray(positions)
        rows = jnp.arange(length)
        cos, sin = rope_freqs(positions, self.rope_dim, self.rope_theta, self.yarn,
                              dtype=at_least_fp32(x.dtype))

        # The query latent and the shared key head project the residual down,
        # where `down_projection` places them: on each tensor shard's own
        # tokens where the link pays for gathering them back.
        place = down_projection(x, self.q_lora_rank + self.head_dim)
        q_resid = constrain(self.q_a_norm(self.q_a_proj(constrain(x, place))), place)
        query = checkpoint_name(self.q_b_proj(constrain(q_resid, RESIDUAL)), 'q_proj').reshape(
            batch, length, self.num_heads, self.head_dim)
        if self.query_norm:
            query = unweighted_rmsnorm(query, self.norm_eps)
        query = _rotate_rope(query, cos, sin)
        # V4.1's window cache keeps FP8, RoPE included (v41:700-707)
        keys = self._window_keys(x, cos, sin, place)

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
                key_valid=cached_validity(self, keys.shape[1]))[:, 0]
        if segment_ids is not None:
            allowed = allowed & document_mask(segment_ids)
        if valid is not None and not decode:
            allowed = allowed & valid[:, None, :]

        if self.compressor is not None:
            selected_positions = positions if cache is None else cache[0]
            if self.compressor == 'csa2':
                entries, selected = self._csa2(x, q_resid, selected_positions, cos, sin, cache, kv_store)
            else:
                entries, selected = self.compress(x, q_resid, selected_positions, cos, sin, cache)
            if entries.shape[1]:
                keys = jnp.concatenate([keys, entries], axis=1)
                allowed = jnp.concatenate([allowed, selected], axis=-1)

        output = self._attend(query, keys, allowed, cos, sin)
        return output if valid is None else jnp.where(valid[..., None], output, 0)

    def _window_keys(self, x, cos, sin, place: LogicalAxes):
        """The shared key head, projected where `place` puts it and gathered
        for the query heads the tensor axis splits."""
        projected = constrain(self.kv_norm(self.kv_proj(constrain(x, place))), place)
        keys = constrain(_rotate_rope(checkpoint_name(projected, 'kv_proj'), cos, sin), RESIDUAL)
        return fake_quant_fp8(keys, 32) if self.kv_qat else keys

    def _attend(self, query, keys, allowed, cos, sin):
        batch, length = query.shape[:2]
        # One shared key/value head under every query head, the per-head
        # sink beside the logits, softmax in fp32 and the sink dropped
        # (:708-736).
        wide = at_least_fp32(query.dtype)
        logits = jnp.einsum('bshd,btd->bhst', query.astype(wide), keys.astype(wide),
                            precision=self.precision) * self.head_dim ** -0.5
        logits = jnp.where(allowed[:, None], logits, -jnp.inf)
        sinks = jnp.broadcast_to(self.sinks[None, :, None, None], (batch, self.num_heads, length, 1))
        probs = jax.nn.softmax(jnp.concatenate([logits, sinks], axis=-1), axis=-1)[..., :-1]
        context = jnp.einsum('bhst,btd->bshd', probs.astype(keys.dtype), keys, precision=self.precision)
        # The values are the rotated keys, so the rope slice of the output is
        # turned back at the query's position (:853-859).
        context = _rotate_rope(checkpoint_name(context, 'context'), cos, -sin)
        mixed = self.o_a_proj(context.reshape(batch, length, self.o_groups, -1))
        return checkpoint_name(self.o_b_proj(mixed.reshape(batch, length, -1)), 'o_proj')


DRAFT_CONTEXT = 'draft_context'
DRAFT_VALID = 'draft_valid'
"""The kv_store names a DSpark stage hands its attention the target model's
context under, and which of its positions are real (`DSparkAttention`)."""


class DSparkAttention(DeepseekV4Attention):
    """A DSpark drafter stage's attention (V4.1 inference/model.py:1032-1074):
    a sliding V4 layer whose keys are the target's context and the draft
    block's own.

    `kv_store[DRAFT_CONTEXT]` `[B, M, D]` is the target's context up to the
    drafting position and `kv_store[DRAFT_VALID]` `[B, M]` which of it is real;
    its keys come through this layer's `kv_proj` into a sliding window. The
    block `x` `[B, K, D]` attends that window and all of itself, both
    directions. Cached, each call appends the real context to the window cache
    (an empty block only does that, the release's prefill), and a call without
    context drafts after the cache; uncached, the context is whole.
    """

    @nn.compact
    def __call__(self, x, decode: bool = False, positions=None, segment_ids=None,
                 kv_store=None, attention_metadata: AttentionMetadata | None = None):
        store = {} if kv_store is None else kv_store
        main = store.get(DRAFT_CONTEXT)
        batch = x.shape[0]
        if decode:
            cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                       (batch, self.max_seq_len, self.head_dim), x.dtype)
            if main is not None:
                slots, allocated = cache_positions(self, batch, main.shape[1], store.get(DRAFT_VALID))
                if allocated:
                    cached_key.value = write_cache(cached_key.value, self._window_keys(
                        main, *rope_freqs(slots, self.rope_dim, self.rope_theta, self.yarn,
                                          dtype=at_least_fp32(main.dtype)),
                        down_projection(main, self.head_dim)), slots)
            main_keys = cached_key.value
            last = jnp.asarray(self.get_variable('cache', 'cache_index')) - 1
        elif main is None:
            raise ValueError(f"uncached, a DSpark stage's attention reads the whole context "
                             f"from kv_store[{DRAFT_CONTEXT!r}]")
        else:
            main_keys = self._window_keys(main, *rope_freqs(jnp.arange(main.shape[1]), self.rope_dim,
                                                            self.rope_theta, self.yarn,
                                                            dtype=at_least_fp32(main.dtype)),
                                          down_projection(main, self.head_dim))
            last = jnp.full((batch,), main.shape[1] - 1)
        length = x.shape[1]
        if length == 0:
            return x
        positions = last[:, None] + 1 + jnp.arange(length)
        cos, sin = rope_freqs(positions, self.rope_dim, self.rope_theta, self.yarn,
                              dtype=at_least_fp32(x.dtype))
        place = down_projection(x, self.q_lora_rank + self.head_dim)
        q_resid = constrain(self.q_a_norm(self.q_a_proj(constrain(x, place))), place)
        query = self.q_b_proj(constrain(q_resid, RESIDUAL)).reshape(
            batch, length, self.num_heads, self.head_dim)
        if self.query_norm:
            query = unweighted_rmsnorm(query, self.norm_eps)
        query = _rotate_rope(query, cos, sin)
        slots = jnp.arange(main_keys.shape[1])
        window = (slots[None] <= last[:, None]) & (slots[None] > last[:, None] - self.sliding_window)
        allowed = jnp.concatenate([
            jnp.broadcast_to(window[:, None], (batch, length, main_keys.shape[1])),
            jnp.ones((batch, length, length), bool)], axis=-1)
        keys = jnp.concatenate([main_keys, self._window_keys(x, cos, sin, place)], axis=1)
        return self._attend(query, keys, allowed, cos, sin)


@dataclasses.dataclass(frozen=True)
class DeepseekV4Mixer(MixerBase):
    """The `deepseek_v4` kind, by the config's field names.

    The compressed kinds name `compressor` ('csa', 'hca' or V4.1's 'csa2') and
    `compress_rate` on their `LayerKind.mixer`, the csa kind the indexer's three
    fields. The window, rope base and YaRN ramp are the context's, and
    `rope_head_dim` the rotated slice. The context's standard attention dials
    (`num_kv_heads`, the norms, `kv_shared` outside CSA2, sinks and scales) are
    not read. A V4.1 Reuse layer is one listed in `kv_shared_layers`, in the
    kind of the Full layer before it; `reindex` names the Reindex kind and
    `candidates` the pool's role.
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
    reindex: bool = False
    candidates: str | None = None
    candidate_blocks: int | None = None
    candidate_block_size: int | None = None
    query_norm: bool = True
    kv_qat: bool = False

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        return self._built(DeepseekV4Attention, ctx)

    def publishes(self, kv_shared: bool) -> bool:
        """Whether a layer of this kind leaves what later layers read in the
        kv_store: a CSA2 Full layer its entries, index keys and selection, a
        Reindex layer its selection."""
        return self.compressor == 'csa2' and (self.reindex or not kv_shared)

    def drafter(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        """This kind's attention as a DSpark stage's (`DSparkAttention`)."""
        if self.compressor is not None:
            raise ValueError("a DSpark stage attends a sliding window: its kind has no compressor")
        return self._built(DSparkAttention, ctx)

    def _built(self, attention: type[DeepseekV4Attention], ctx: MixerContext) -> Callable[..., nn.Module]:
        if not ctx.causal:
            raise ValueError(
                "the deepseek_v4 mixer is causal: its window and compressors read the past alone"
            )
        if ctx.sliding_window is None:
            raise ValueError("every deepseek_v4 layer attends a sliding window, which its kind names")
        if ctx.attention_chunk is not None:
            raise ValueError("a deepseek_v4 layer attends a sliding window, not a chunk")
        if ctx.kv_shared and self.compressor != 'csa2':
            raise ValueError("of the deepseek_v4 kinds only CSA2 shares its entries across layers")
        if ctx.kv_cache != KVCache():
            raise ValueError(
                "the deepseek_v4 mixer keeps its own compressed cache; it takes no kv_cache layout"
            )
        if ctx.yarn is not None and ctx.yarn.rope_theta != ctx.rope_theta:
            raise ValueError(
                f"the yarn record's rope_theta ({ctx.yarn.rope_theta}) and the layer's "
                f"({ctx.rope_theta}) disagree; the rope base is configured once, on the kind")
        return functools.partial(
            attention,
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
            kv_shared=ctx.kv_shared,
            reindex=self.reindex,
            candidates=self.candidates,
            candidate_blocks=self.candidate_blocks,
            candidate_block_size=self.candidate_block_size,
            query_norm=self.query_norm,
            kv_qat=self.kv_qat,
            norm_eps=ctx.norm_eps,
            dtype=ctx.dtype,
            precision=ctx.precision)
