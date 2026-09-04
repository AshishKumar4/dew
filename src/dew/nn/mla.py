"""Multi-head latent attention: DeepSeek-V3 MLA and the V3.2 sparse indexer.

The reference is transformers 5.16.1
(models/deepseek_v3/modeling_deepseek_v3.py and
models/deepseek_v32/modeling_deepseek_v32.py), read as the specification.
MLA compresses keys and values into one low-rank latent per token plus a
small decoupled rotary head: `kv_a_proj_with_mqa` maps the hidden states to
`[kv_lora_rank + qk_rope_head_dim]`, the latent is normed, and `kv_b_proj`
expands it back out to every head's nope keys and values. Queries are
low-rank the same way when `q_lora_rank` is set (a plain `q_proj` when it is
None, which no released checkpoint uses). Decode caches the compressed
latents, which is what the V3 reference holds; the V3.2 reference caches the
expanded keys and values instead, so the sparse variant does that too.

The rotary head rotates interleaved pairs (even/odd slices, one frequency
each) rather than the rotate-half convention `apply_rotary` implements, so
the pairwise rotation lives here next to its only caller. YaRN scaling,
which both released DeepSeek configs ask for, reshapes the inverse
frequencies and multiplies the attention scale; the frequency ramp is the
reference's `_compute_yarn_parameters` and the scale multiplier its
`yarn_apply_mscale`, both with `dim` at the rope width, which is where
DeepSeek points `config.head_dim`.
"""

import dataclasses
import functools
import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import causal_attention_mask, scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import (
    RMSNorm,
    apply_rotary,
    rotary_freqs,
)
from dew.nn.sharding import logical_axes


@dataclasses.dataclass(frozen=True)
class YarnScaling:
    """YaRN rope scaling, the reference's `rope_parameters` fields.

    Both released DeepSeek configs carry this spelling (rope_type yarn,
    factor 40 off 4096 base positions), so the record keeps the reference's
    names: translation renames nothing. `rope_theta` repeats the mixer's own
    base, and the two must agree, so the scaling is configured once (the
    mscale lives in the attention as a query pre-scale, not in the rope).
    """

    rope_type: str = 'yarn'
    rope_theta: float = 10000.0
    factor: float = 40.0
    original_max_position_embeddings: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: Optional[float] = None
    mscale_all_dim: Optional[float] = None
    truncate: bool = True
    # An explicit cos/sin amplitude, which the reference applies instead of
    # deriving one; None derives it from factor and the mscales above.
    attention_factor: Optional[float] = None


def yarn_inv_freq(head_dim: int, theta: float, yarn: YarnScaling) -> jax.Array:
    """YaRN inverse frequencies over the rope width: `[head_dim // 2]`.

    Mirrors `modeling_rope_utils._compute_yarn_parameters` with `dim` at the
    head dim, which is what DeepSeek's configs do by pointing `head_dim` at
    the rope slice. Low dims interpolate towards `1 / (factor * pos_freqs)`,
    high dims keep extrapolating, and the linear ramp between the correction
    bounds blends them.
    """
    dim = head_dim
    pairs = dim // 2
    pos_freqs = theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    inv_extrapolation = 1.0 / pos_freqs
    inv_interpolation = 1.0 / (yarn.factor * pos_freqs)

    def correction_dim(rotations: float) -> float:
        """The dimension seeing `rotations` turns over the original context."""
        return (dim * math.log(yarn.original_max_position_embeddings
                               / (rotations * 2 * math.pi))
                / (2 * math.log(theta)))

    low = correction_dim(yarn.beta_fast)
    high = correction_dim(yarn.beta_slow)
    if yarn.truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    span = high - low
    if span == 0:
        # The reference nudges a degenerate bound rather than dividing by zero.
        span = 0.001
    ramp = jnp.clip((jnp.arange(pairs, dtype=jnp.float32) - low) / span, 0, 1)
    return inv_interpolation * ramp + inv_extrapolation * (1 - ramp)

def yarn_attention_factor(yarn: YarnScaling) -> float:
    """The cos/sin multiplier of `_compute_yarn_parameters`.

    Both released configs set mscale and mscale_all_dim to 1.0, so this is
    1.0 for them; the general form stays, because a config that sets them
    apart rotates at a different amplitude and must not silently lose it.
    """
    if yarn.attention_factor is not None:
        return float(yarn.attention_factor)
    def mscale(scale: float, weight: float) -> float:
        return 1.0 if scale <= 1 else 0.1 * weight * math.log(scale) + 1.0

    if yarn.mscale and yarn.mscale_all_dim:
        return float(mscale(yarn.factor, yarn.mscale)
                     / mscale(yarn.factor, yarn.mscale_all_dim))
    return float(mscale(yarn.factor, 1.0))


def yarn_query_scale(yarn: YarnScaling) -> float:
    """The attention-scale multiplier of `yarn_apply_mscale`, squared.

    The reference folds this into the softmax scale, while dew's kernels
    scale by `1 / sqrt(head_dim)` themselves, so the query carries the
    ratio, the same way `attention_scale` does on the standard mixer. For
    the released configs this is `(0.1 * ln(40) + 1) ** 2`.
    """
    if not yarn.mscale_all_dim or yarn.factor <= 1:
        return 1.0
    mscale = 0.1 * yarn.mscale_all_dim * math.log(yarn.factor) + 1.0
    return mscale * mscale


def mla_rope_freqs(positions, head_dim: int, theta: float,
                   yarn: Optional[YarnScaling]):
    """cos/sin over the rope width, plain or YaRN-scaled: `[P, head_dim // 2]`.

    Plain rope is `rotary_freqs`, the one layout every mixer shares. YaRN
    replaces the inverse frequencies with the ramp and scales the resulting
    cos/sin by its attention factor, which is what the reference's rotary
    embedding returns.
    """
    if yarn is None:
        return rotary_freqs(positions, head_dim, theta)
    inv_freq = yarn_inv_freq(head_dim, theta, yarn)
    positions = jnp.asarray(positions, jnp.float32)
    if positions.ndim == 1:
        angles = positions[:, None] * inv_freq[None, :]
    else:
        angles = positions[:, :, None] * inv_freq[None, None, :]
    factor = yarn_attention_factor(yarn)
    return jnp.cos(angles) * factor, jnp.sin(angles) * factor


def apply_rotary_interleave(x, freqs_cos, freqs_sin):
    """Rotate `[B, S, H, D]` heads pairwise, DeepSeek's rope convention.

    Pairs `(x0, x1), (x2, x3), ...` each rotate by one frequency
    (`modeling_deepseek_v3.apply_rotary_pos_emb_interleave`): the even and
    odd slices turn against the first half of the cos/sin, and the halves
    stack real over imaginary rather than interleaving back. Query and key
    take the same layout, so the dot product keeps the complex structure;
    matching the layout exactly is what parity needs.
    """
    if freqs_cos.ndim == 3:
        cos = freqs_cos[:, :, None, :]
        sin = freqs_sin[:, :, None, :]
    else:
        cos = freqs_cos[None, :, None, :]
        sin = freqs_sin[None, :, None, :]
    fp32 = x.astype(jnp.float32)
    even, odd = fp32[..., 0::2], fp32[..., 1::2]
    out = jnp.concatenate([even * cos - odd * sin, odd * cos + even * sin],
                          axis=-1)
    return out.astype(x.dtype)


def open_latent_cache(module: nn.Module, latent, rot, index_keys, max_seq_len):
    """Fixed-size latent cache in the `open_kv_cache` style.

    Declares `cached_latent`/`cached_rot` (and `cached_index` when the
    indexer scores) plus the shared `cache_index`, sized from the step's own
    arrays at the full decode length, and returns the absolute positions of
    the appended tokens with the writer that appends them. Positions come
    out before the write, because the decoupled rope head is rotated before
    it enters the cache, exactly like the standard mixer's keys.
    """
    if max_seq_len is None:
        raise ValueError(
            "decoding needs max_seq_len: the latent cache is allocated once, "
            "at the full decode length, and never grows.")
    batch, length = latent.shape[0], latent.shape[1]
    if length > max_seq_len:
        raise ValueError(
            f"{length} tokens do not fit a latent cache of {max_seq_len}.")
    allocated = module.has_variable('cache', 'cached_latent')
    cached_latent = module.variable(
        'cache', 'cached_latent', jnp.zeros,
        (batch, max_seq_len, latent.shape[-1]), latent.dtype)
    cached_rot = module.variable(
        'cache', 'cached_rot', jnp.zeros,
        (batch, max_seq_len, rot.shape[-1]), rot.dtype)
    cached_index = None
    if index_keys is not None:
        cached_index = module.variable(
            'cache', 'cached_index', jnp.zeros,
            (batch, max_seq_len, index_keys.shape[-1]), index_keys.dtype)
    cache_index = module.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, jnp.int32))
    index = cache_index.value

    def append(new_latent, new_rot, new_index_keys):
        if not allocated:
            return new_latent, new_rot, new_index_keys
        zero = jnp.array(0, index.dtype)
        cached_latent.value = jax.lax.dynamic_update_slice(
            cached_latent.value, new_latent.astype(cached_latent.value.dtype),
            (zero, index, zero))
        cached_rot.value = jax.lax.dynamic_update_slice(
            cached_rot.value, new_rot.astype(cached_rot.value.dtype),
            (zero, index, zero))
        if cached_index is not None:
            if new_index_keys is None:
                raise ValueError(
                    "the indexer scores, so decode has to append its keys too")
            cached_index.value = jax.lax.dynamic_update_slice(
                cached_index.value,
                new_index_keys.astype(cached_index.value.dtype),
                (zero, index, zero))
        cache_index.value = index + length
        full_index = None if cached_index is None else cached_index.value
        return cached_latent.value, cached_rot.value, full_index

    return index + jnp.arange(length), append


def open_expanded_cache(module: nn.Module, key, value, index_keys,
                        max_seq_len):
    """Fixed-size expanded K/V cache for the sparse variant.

    The V3.2 reference caches the expanded keys and values rather than the
    latents, so decode reads them back instead of re-expanding the whole
    history every step. Same declare/append shape as `open_latent_cache`,
    with the indexer's keys alongside.
    """
    if max_seq_len is None:
        raise ValueError(
            "decoding needs max_seq_len: the sparse cache is allocated once, "
            "at the full decode length, and never grows.")
    batch, length = key.shape[0], key.shape[1]
    if length > max_seq_len:
        raise ValueError(
            f"{length} tokens do not fit a sparse cache of {max_seq_len}.")
    allocated = module.has_variable('cache', 'cached_key')
    cached_key = module.variable(
        'cache', 'cached_key', jnp.zeros,
        (batch, max_seq_len) + key.shape[2:], key.dtype)
    cached_value = module.variable(
        'cache', 'cached_value', jnp.zeros,
        (batch, max_seq_len) + value.shape[2:], value.dtype)
    cached_index = module.variable(
        'cache', 'cached_index', jnp.zeros,
        (batch, max_seq_len, index_keys.shape[-1]), index_keys.dtype)
    cache_index = module.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, jnp.int32))
    index = cache_index.value

    def append(new_key, new_value, new_index_keys):
        if not allocated:
            return new_key, new_value, new_index_keys
        zero = jnp.array(0, index.dtype)
        cached_key.value = jax.lax.dynamic_update_slice(
            cached_key.value, new_key.astype(cached_key.value.dtype),
            (zero, index) + (zero,) * (new_key.ndim - 2))
        cached_value.value = jax.lax.dynamic_update_slice(
            cached_value.value, new_value.astype(cached_value.value.dtype),
            (zero, index) + (zero,) * (new_value.ndim - 2))
        cached_index.value = jax.lax.dynamic_update_slice(
            cached_index.value,
            new_index_keys.astype(cached_index.value.dtype),
            (zero, index, zero))
        cache_index.value = index + length
        return cached_key.value, cached_value.value, cached_index.value

    return index + jnp.arange(length), append


@logical_axes({
    ("wq_b",): ("qlora", "index"),
    ("wk",): ("embed", "index"),
    # The key norm is a LayerNorm, so scale and bias share the indexer width.
    ("k_norm",): ("index",),
    # Maps the hidden states onto one weight per indexer head.
    ("weights_proj",): ("embed", "index"),
})
class SparseIndexer(nn.Module):
    """DeepSeek sparse attention's top-k token selector, per query.

    A lightweight scorer beside the main MLA projections
    (`modeling_deepseek_v32.DeepseekV32Indexer`): `wq_b` reads the query
    residual, `wk` reads the hidden states into keys the cache holds, and
    `weights_proj` weights the heads into one score per key. `select`
    returns the `int32` top-k key positions, which the mixer folds into the
    attention mask the way the reference's eager path does. There is no
    fast-kernel index path here, so the mask fold is the only path, on every
    backend.

    The indexer rotates with the plain rotate-half convention, unlike the
    main rope head's interleaved pairs; the reference calls the two
    different functions and so does this.
    """

    q_lora_rank: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    top_k: int
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        if self.head_dim <= self.rope_head_dim:
            raise ValueError(
                f"the indexer rotates a {self.rope_head_dim}-wide rope slice "
                f"out of heads of width {self.head_dim}, so the heads have "
                "to be wider")
        dense = functools.partial(
            nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
        self.wq_b = dense(self.n_heads * self.head_dim, name='wq_b')
        self.wk = dense(self.head_dim, name='wk')
        # A LayerNorm with weight and bias at a hardcoded 1e-6: the reference
        # names no config for it (modeling_deepseek_v32.py, DeepseekV32Indexer).
        self.k_norm = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name='k_norm')
        self.weights_proj = dense(self.n_heads, name='weights_proj')

    def keys(self, hidden):
        """The indexer's keys for these hidden states: `[B, S, head_dim]`."""
        return self.k_norm(self.wk(hidden))

    def rotated_keys(self, keys, freqs_cos, freqs_sin):
        """`keys` with their rope slice rotated at their own positions.

        Rotation happens once, before caching, so a cached key keeps the
        angle of its position while later queries rotate at theirs, which is
        what the reference's `update_indexer` ordering does.
        """
        k_rot, k_pass = jnp.split(keys, [self.rope_head_dim], axis=-1)
        k_rot = apply_rotary(k_rot[:, :, None, :], freqs_cos, freqs_sin)
        return jnp.concatenate([k_rot[:, :, 0, :], k_pass], axis=-1)

    def select(self, hidden, q_resid, keys, freqs_cos, freqs_sin, mask):
        """Top-k key positions per query: `[B, S, K]`, int32.

        `keys` are the rotated keys of every candidate (the cache on
        decode), `mask` the `[B, S, T]` additive float bias of what the
        query may not attend, and the query side is computed, never cached.
        Scores run in fp32: the head weighting multiplies by
        `n_heads ** -0.5` in fp32 in the reference, and the relu keeps only
        the positive agreements.
        """
        batch, length, _ = hidden.shape
        total = keys.shape[-2]
        query = self.wq_b(q_resid).reshape(
            batch, length, self.n_heads, self.head_dim)
        q_rot, q_pass = jnp.split(query, [self.rope_head_dim], axis=-1)
        query = jnp.concatenate(
            [apply_rotary(q_rot, freqs_cos, freqs_sin), q_pass], axis=-1)
        scores = jnp.matmul(
            query.astype(jnp.float32),
            jnp.expand_dims(keys.astype(jnp.float32).transpose(0, 2, 1), -3))
        scores = jnp.maximum(scores * (self.head_dim ** -0.5), 0)
        weights = self.weights_proj(hidden).astype(jnp.float32)
        weights = weights * (self.n_heads ** -0.5)
        index_scores = jnp.matmul(weights[..., None, :], scores).squeeze(-2)
        index_scores = index_scores + mask
        return jax.lax.top_k(
            index_scores, min(self.top_k, total))[1].astype(jnp.int32)


@logical_axes({
    ("q_a_proj",): ("embed", "qlora"),
    ("q_b_proj",): ("qlora", "attention"),
    # The output concatenates the latent and the decoupled rope head, which
    # ride into the cache together; the name picks the sharded side.
    ("kv_a_proj_with_mqa",): ("embed", "kvlora"),
    ("kv_b_proj",): ("kvlora", "attention"),
    ("o_proj",): ("attention", "embed"),
    ("q_a_layernorm",): ("qlora",),
    ("kv_a_layernorm",): ("kvlora",),
})
class MultiHeadLatentAttention(nn.Module):
    """DeepSeek's multi-head latent attention, dense or sparse.

    decode=True runs against the cache, like the standard mixer: the first
    call writes the whole prompt and each later call appends one token. The
    dense variant caches the compressed latent and the rotated rope head;
    the sparse (V3.2 indexer) variant caches the expanded keys and values
    with the indexer's keys, which is what each reference holds. `positions`
    and `segment_ids` behave as on the standard mixer: absolute positions
    for the cache slots, per-document positions and a block-diagonal mask
    for a packed batch.
    `yarn` replaces the plain rope base with the YaRN ramp; when it is set
    the mixer's `rope_theta` has to equal the record's, so the scaling is
    configured once, and the mscale reaches the logits as a query
    pre-scale. causal=False is full attention with no cache, the mode a
    non-causal reader would take; decode=True raises there.
    """

    emb_features: int
    num_heads: int
    max_seq_len: int
    q_lora_rank: Optional[int]
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    causal: bool = True
    rope_theta: float = 10000.0
    rope_interleave: bool = True
    yarn: Optional[YarnScaling] = None
    norm_eps: float = 1e-6
    attention_bias: bool = False
    index_topk: Optional[int] = None
    index_n_heads: Optional[int] = None
    index_head_dim: Optional[int] = None
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    attention_impl: Optional[str] = None
    force_fp32_for_softmax: bool = True

    def setup(self):
        if self.qk_rope_head_dim % 2:
            raise ValueError(
                "rotary positions rotate pairs, so the rope head dim must be "
                f"even, got {self.qk_rope_head_dim}")
        if self.qk_nope_head_dim < 1 or self.v_head_dim < 1:
            raise ValueError(
                "the nope head dim and the value head dim must be positive, "
                f"got {self.qk_nope_head_dim} and {self.v_head_dim}")
        index_fields = (self.index_topk, self.index_n_heads, self.index_head_dim)
        sparse = self.index_topk is not None
        if sparse != all(field is not None for field in index_fields):
            raise ValueError(
                "the indexer needs its top-k, head count and head dim "
                "together, all set or all unset")
        if sparse and self.q_lora_rank is None:
            raise ValueError(
                "the indexer reads the query residual, which only exists "
                "with a q_lora_rank")
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        dense = functools.partial(
            nn.Dense, use_bias=self.attention_bias, dtype=self.dtype,
            precision=self.precision)
        if self.q_lora_rank is None:
            self.q_proj = dense(self.num_heads * qk_head_dim, name='q_proj')
        else:
            self.q_a_proj = dense(self.q_lora_rank, name='q_a_proj')
            self.q_a_layernorm = RMSNorm(
                epsilon=self.norm_eps, dtype=self.dtype, name='q_a_layernorm')
            self.q_b_proj = nn.Dense(
                self.num_heads * qk_head_dim, use_bias=False, dtype=self.dtype,
                precision=self.precision, name='q_b_proj')
        self.kv_a_proj_with_mqa = dense(
            self.kv_lora_rank + self.qk_rope_head_dim, name='kv_a_proj_with_mqa')
        self.kv_a_layernorm = RMSNorm(
            epsilon=self.norm_eps, dtype=self.dtype, name='kv_a_layernorm')
        self.kv_b_proj = nn.Dense(
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            use_bias=False, dtype=self.dtype, precision=self.precision,
            name='kv_b_proj')
        self.o_proj = dense(self.emb_features, name='o_proj')
        if self.sparse:
            assert self.q_lora_rank is not None
            self.indexer = SparseIndexer(
                q_lora_rank=self.q_lora_rank, n_heads=self.index_n_heads,
                head_dim=self.index_head_dim,
                rope_head_dim=self.qk_rope_head_dim, top_k=self.index_topk,
                dtype=self.dtype, precision=self.precision, name='indexer')

    @property
    def sparse(self) -> bool:
        """Whether the V3.2 indexer selects the keys per query."""
        return self.index_topk is not None

    @property
    def query_scale(self) -> float:
        """The softmax-scale ratio the query carries into the kernel.

        The kernel scales by `1 / sqrt(qk_head_dim)` itself; YaRN's mscale
        rides on the query, so a plain rope carries exactly 1.0.
        """
        if self.yarn is None:
            return 1.0
        return yarn_query_scale(self.yarn)

    def _queries(self, x):
        """`[B, S, H, nope+rope]` queries and the residual the indexer reads."""
        batch, length, _ = x.shape
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.q_lora_rank is None:
            q_resid = None
            queries = self.q_proj(x)
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(x))
            queries = self.q_b_proj(q_resid)
        return queries.reshape(batch, length, self.num_heads, qk_head_dim), q_resid

    def _latents(self, x):
        """The normed KV latent and the raw decoupled rope head."""
        compressed = self.kv_a_proj_with_mqa(x)
        latent, rot = jnp.split(compressed, [self.kv_lora_rank], axis=-1)
        return self.kv_a_layernorm(latent), rot

    def _rotate(self, part, freqs_cos, freqs_sin):
        if self.rope_interleave:
            return apply_rotary_interleave(part, freqs_cos, freqs_sin)
        return apply_rotary(part, freqs_cos, freqs_sin)

    def _expand(self, latent, rot):
        """Latent and rope head into per-head keys and values."""
        batch, length = latent.shape[0], latent.shape[1]
        width = self.qk_nope_head_dim + self.v_head_dim
        kv = self.kv_b_proj(latent).reshape(batch, length, self.num_heads, width)
        nope, values = jnp.split(kv, [self.qk_nope_head_dim], axis=-1)
        rot = jnp.broadcast_to(
            rot[:, :, None, :], (batch, length, self.num_heads, self.qk_rope_head_dim))
        return jnp.concatenate([nope, rot], axis=-1), values

    @nn.compact
    def __call__(self, x, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None):
        causal, mask = self.causal, None
        implementation = self.attention_impl
        batch, length, _ = x.shape
        queries, q_resid = self._queries(x)
        # jnp splits at indices where torch splits into sizes: one cut point,
        # since the widths add up exactly.
        q_pass, q_rot = jnp.split(
            queries, [self.qk_nope_head_dim], axis=-1)
        latent, rot = self._latents(x)
        if decode:
            if positions is not None or segment_ids is not None:
                raise ValueError(
                    "decode positions come from the cache index, so an "
                    "explicit positions or segment_ids has no meaning there")
            # The cache hands out the slots first, because the rope heads
            # rotate at absolute positions: the queries at this step's slots
            # and the appended latents at theirs, while the cached ones keep
            # the angles of the slots they were written at.
            if self.sparse:
                assert q_resid is not None
                # The cache is shaped by the expansion; its values are
                # recomputed after rotation below, so only shapes flow here.
                shape_key, shape_value = self._expand(latent, rot)
                index_keys = self.indexer.keys(x)
                positions, append = open_expanded_cache(
                    self, shape_key, shape_value, index_keys,
                    self.max_seq_len)
                freqs_cos, freqs_sin = mla_rope_freqs(
                    positions, self.qk_rope_head_dim, self.rope_theta,
                    self.yarn)
                q_rot = self._rotate(q_rot, freqs_cos, freqs_sin)
                rot = self._rotate(
                    rot[:, :, None, :], freqs_cos, freqs_sin)[:, :, 0, :]
                index_keys = self.indexer.rotated_keys(
                    self.indexer.keys(x), freqs_cos, freqs_sin)
                key, value, index_full = append(
                    *self._expand(latent, rot), index_keys)
                mask = self._index_mask(
                    x, q_resid, positions, key.shape[1], index_full,
                    freqs_cos, freqs_sin)
            else:
                positions, append = open_latent_cache(
                    self, latent, rot, None, self.max_seq_len)
                freqs_cos, freqs_sin = mla_rope_freqs(
                    positions, self.qk_rope_head_dim, self.rope_theta,
                    self.yarn)
                q_rot = self._rotate(q_rot, freqs_cos, freqs_sin)
                rot = self._rotate(
                    rot[:, :, None, :], freqs_cos, freqs_sin)[:, :, 0, :]
                latent, rot, _ = append(latent, rot, None)
                key, value = self._expand(latent, rot)
                mask = causal_attention_mask(positions, key.shape[-3])
            causal = False
        else:
            if positions is None:
                positions = jnp.arange(length)
            else:
                positions = jnp.asarray(positions)
            freqs_cos, freqs_sin = mla_rope_freqs(
                positions, self.qk_rope_head_dim, self.rope_theta, self.yarn)
            q_rot = self._rotate(q_rot, freqs_cos, freqs_sin)
            rot = self._rotate(
                rot[:, :, None, :], freqs_cos, freqs_sin)[:, :, 0, :]
            key, value = self._expand(latent, rot)
            if segment_ids is not None:
                segment_ids = jnp.asarray(segment_ids)
                inside = ((segment_ids[:, :, None] == segment_ids[:, None, :])
                          & (segment_ids[:, :, None] != 0))[:, None]
                mask = inside
                if causal:
                    mask = jnp.logical_and(
                        inside, causal_attention_mask(
                            jnp.arange(length), length))
                causal = False
                if implementation in ('auto', 'cudnn'):
                    implementation = 'xla'
            if self.sparse:
                assert q_resid is not None
                index_keys = self.indexer.rotated_keys(
                    self.indexer.keys(x), freqs_cos, freqs_sin)
                mask = self._index_mask(
                    x, q_resid, positions, length, index_keys,
                    freqs_cos, freqs_sin, base=mask)
        scale = self.query_scale
        query = jnp.concatenate([q_pass, q_rot], axis=-1)
        if scale != 1.0:
            query = query * scale
        attention = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            implementation=implementation, causal=causal, mask=mask)
        return self.o_proj(attention.reshape(
            batch, length, self.num_heads * self.v_head_dim))

    def _index_mask(self, x, q_resid, positions, kv_len: int, index_keys,
                    freqs_cos, freqs_sin, base=None):
        """The attention bool mask with the indexer's top-k folded in.

        `True` attends: causality (or the packed base) and the selected keys
        meet, the way the reference's eager path `masked_fill`s the additive
        mask. Decode positions are cache slots, so causality reads them.
        """
        batch, length, _ = x.shape
        keep = causal_attention_mask(positions, kv_len)
        if base is not None:
            keep = jnp.logical_and(base, keep)
        queries = jnp.arange(length)
        bias = jnp.where(keep, 0.0, jnp.finfo(jnp.float32).min)
        if bias.ndim == 4:
            bias = bias[:, 0]
        chosen = self.indexer.select(
            x, q_resid, index_keys, freqs_cos, freqs_sin, bias)
        selected = jnp.zeros((batch, length, kv_len), jnp.bool_).at[
            jnp.arange(batch)[:, None, None],
            queries[None, :, None], chosen].set(True)
        return jnp.logical_and(keep, selected[:, None])
