"""Multi-head latent attention: DeepSeek-V3 MLA and the V3.2 sparse indexer.

The reference is transformers 5.16.1
(models/deepseek_v3/modeling_deepseek_v3.py and
models/deepseek_v32/modeling_deepseek_v32.py), read as the specification;
models/glm_moe_dsa/modeling_glm_moe_dsa.py is the same block with the
indexer rotating interleaved pairs and IndexShare layers attending the
previous full layer's selection.
MLA compresses keys and values into one low-rank latent per token plus a
small decoupled rotary head: `kv_a_proj_with_mqa` maps the hidden states to
`[kv_lora_rank + qk_rope_head_dim]`, the latent is normed, and `kv_b_proj`
expands it back out to every head's nope keys and values. Queries are
low-rank the same way when `q_lora_rank` is set (a plain `q_proj` when it is
None, which no released checkpoint uses). Decode caches the compressed
latents, as the V3 reference does; the V3.2 reference caches the expanded
keys and values instead, so the sparse variant does that too.

The rotary head rotates interleaved pairs (even/odd slices, one frequency
each), not the rotate-half pairs `apply_rotary` rotates, so the pairwise
rotation lives here next to its only caller. YaRN scaling, which both
released DeepSeek configs ask for, reshapes the inverse frequencies and
multiplies the attention scale; the frequency ramp is the reference's
`_compute_yarn_parameters` and the scale multiplier its `yarn_apply_mscale`,
both with `dim` at the rope width, where DeepSeek points `config.head_dim`.
"""

import dataclasses
import functools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.scipy.special import xlogy

from dew.nn.attention import (
    LayerNorm,
    RMSNorm,
    _cache_positions,
    apply_rotary,
    causal_attention_mask,
    document_mask,
    kernel_for_materialized_mask,
    max_attention_logits,
    rotary_freqs,
    scaled_dot_product_attention,
)
from dew.nn.inputs import AttentionMetadata
from dew.nn.kv_cache import KVCache, write_cache
from dew.nn.sharding import logical_axes
from dew.nn.sparse_selection import selection_mask, sparse_latent_attention, top_k_selection


@dataclasses.dataclass(frozen=True)
class YarnScaling:
    """YaRN rope scaling, the reference's `rope_parameters` fields.

    Both released DeepSeek configs carry this spelling (rope_type yarn,
    factor 40 off 4096 base positions), so the record keeps the reference's
    names and a translation renames nothing. `rope_theta` repeats the
    mixer's own base, and the two must agree, so the scaling is configured
    once (the mscale is applied in the attention as a query pre-scale).
    """

    rope_type: str = 'yarn'
    rope_theta: float = 10000.0
    factor: float = 40.0
    original_max_position_embeddings: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float | None = None
    mscale_all_dim: float | None = None
    truncate: bool = True
    # An explicit cos/sin amplitude, which the reference applies instead of
    # deriving one; None derives it from factor and the mscales above.
    attention_factor: float | None = None


def yarn_inv_freq(head_dim: int, theta: float, yarn: YarnScaling) -> jax.Array:
    """YaRN inverse frequencies over the rope width: `[head_dim // 2]`.

    Mirrors `modeling_rope_utils._compute_yarn_parameters` with `dim` at the
    head dim, as DeepSeek's configs do by pointing `head_dim` at the rope
    slice. Low dims interpolate towards `1 / (factor * pos_freqs)`,
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
        # The reference nudges a degenerate bound to keep the division finite.
        span = 0.001
    ramp = jnp.clip((jnp.arange(pairs, dtype=jnp.float32) - low) / span, 0, 1)
    return inv_interpolation * ramp + inv_extrapolation * (1 - ramp)

def yarn_attention_factor(yarn: YarnScaling) -> float:
    """The cos/sin multiplier of `_compute_yarn_parameters`.

    Both released configs set mscale and mscale_all_dim to 1.0, so this is
    1.0 for them; a config that sets them apart rotates at a different
    amplitude.
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
                   yarn: YarnScaling | None):
    """cos/sin over the rope width, plain or YaRN-scaled: `[P, head_dim // 2]`.

    Plain rope is `rotary_freqs`, the one layout every mixer shares. YaRN
    replaces the inverse frequencies with the ramp and scales the resulting
    cos/sin by its attention factor, as the reference's rotary embedding
    does.
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
    stack real over imaginary without interleaving back. Query and key
    take the same layout, so the dot product keeps the complex structure.
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


def open_latent_cache(module: nn.Module, latent, rot, index_keys, max_seq_len, *, valid=None):
    """Compact per-row latent/rotary cache with optional sparse-index keys."""
    if max_seq_len is None:
        raise ValueError("decoding needs max_seq_len for its fixed-capacity latent cache")
    batch, length = latent.shape[:2]
    if valid is None and length > max_seq_len:
        raise ValueError(f"{length} tokens do not fit a latent cache of {max_seq_len}.")
    cached_latent = module.variable("cache", "cached_latent", jnp.zeros,
                                    (batch, max_seq_len, latent.shape[-1]), latent.dtype)
    cached_rot = module.variable("cache", "cached_rot", jnp.zeros,
                                 (batch, max_seq_len, rot.shape[-1]), rot.dtype)
    cached_index = None
    if index_keys is not None:
        cached_index = module.variable("cache", "cached_index", jnp.zeros,
                                       (batch, max_seq_len, index_keys.shape[-1]), index_keys.dtype)
    positions, allocated = _cache_positions(module, batch, length, max_seq_len, valid)

    def append(new_latent, new_rot, new_index_keys):
        if allocated:
            cached_latent.value = write_cache(cached_latent.value, new_latent, positions)
            cached_rot.value = write_cache(cached_rot.value, new_rot, positions)
            if cached_index is not None:
                if new_index_keys is None:
                    raise ValueError("the indexer scores, so decode must append its keys")
                cached_index.value = write_cache(cached_index.value, new_index_keys, positions)
        full_index = None if cached_index is None else cached_index.value
        return cached_latent.value, cached_rot.value, full_index

    return positions, append


def open_expanded_cache(module: nn.Module, key, value, index_keys, max_seq_len, *, valid=None):
    """Per-row expanded K/V and sparse-index cache, as DeepSeek V3.2 stores it.

    `index_keys` is None on a layer that owns no indexer and reads another
    layer's selection; its cache holds the expanded keys and values alone.
    """
    if max_seq_len is None:
        raise ValueError("decoding needs max_seq_len for its fixed-capacity sparse cache")
    batch, length = key.shape[:2]
    if valid is None and length > max_seq_len:
        raise ValueError(f"{length} tokens do not fit a sparse cache of {max_seq_len}.")
    cached_key = module.variable("cache", "cached_key", jnp.zeros,
                                 (batch, max_seq_len, *key.shape[2:]), key.dtype)
    cached_value = module.variable("cache", "cached_value", jnp.zeros,
                                   (batch, max_seq_len, *value.shape[2:]), value.dtype)
    cached_index = None
    if index_keys is not None:
        cached_index = module.variable("cache", "cached_index", jnp.zeros,
                                       (batch, max_seq_len, index_keys.shape[-1]), index_keys.dtype)
    positions, allocated = _cache_positions(module, batch, length, max_seq_len, valid)

    def append(new_key, new_value, new_index_keys):
        if allocated:
            cached_key.value = write_cache(cached_key.value, new_key, positions)
            cached_value.value = write_cache(cached_value.value, new_value, positions)
            if cached_index is not None:
                cached_index.value = write_cache(cached_index.value, new_index_keys, positions)
        full_index = None if cached_index is None else cached_index.value
        return cached_key.value, cached_value.value, full_index

    return positions, append


INDEXER = 'indexer'
"""The indexer's name under its attention layer, which the parameter paths
of the indexer's weights carry and nothing else in the tree does."""

INDEXER_COLLECTION = 'indexer'
"""The collection an attention layer sows its per-query indexer KL under
when a caller opens it, as the LM objective does to train the indexer."""


def indexer_kl(scores, query, key, keep, scale: float):
    """Per query, KL of the indexer's softmax from the attention: `[B, S]`, fp32.

    The indexer's objective (arXiv 2512.02556, eq. 3 and 4; MaxText
    `attention_mla.py` `calculate_indexer_loss`): the target is the main
    attention's distribution summed over its heads and L1-normalised, the
    indexer's distribution the softmax of its `[B, S, T]` `scores`, both
    over the keys `keep` allows, so the dense warm-up passes the causal
    (and packed) mask and sparse training the top-k selection. `query` and
    `key` are the main heads, `[B, S, H, D]` and `[B, T, H, D]`, `scale`
    the logit scale the kernel applies on top of them. The target is a
    constant of the loss, both projections detached as the reference
    detaches them, so the gradient reaches the indexer alone. The heads are
    summed one at a time, which keeps the footprint at `[B, S, T]` rather
    than the `[B, H, S, T]` logits.
    """
    masked = jnp.finfo(jnp.float32).min
    batch, length, total = scores.shape
    keep = jnp.broadcast_to(keep, (batch, length, total))
    query = jax.lax.stop_gradient(query).astype(jnp.float32) * scale
    key = jax.lax.stop_gradient(key).astype(jnp.float32)

    def head(summed, projections):
        q, k = projections
        logits = jnp.where(keep, jnp.einsum('bsd,btd->bst', q, k), masked)
        return summed + jax.nn.softmax(logits, axis=-1), None

    summed, _ = jax.lax.scan(
        head, jnp.zeros((batch, length, total), jnp.float32),
        (jnp.moveaxis(query, 2, 0), jnp.moveaxis(key, 2, 0)))
    target = summed / jnp.sum(summed, axis=-1, keepdims=True)
    log_indexer = jax.nn.log_softmax(
        jnp.where(keep, scores.astype(jnp.float32), masked), axis=-1)
    # xlogy keeps a key the target gives no mass at zero rather than nan.
    return jnp.sum(xlogy(target, target) - target * log_indexer, axis=-1)


@logical_axes({
    ("wq_b",): ("qlora", "index"),
    ("wk",): ("embed", "index"),
    # The key norm is a LayerNorm, so scale and bias share the indexer width.
    ("k_norm",): ("index",),
    # Maps the hidden states onto one weight per indexer head.
    ("weights_proj",): ("embed", "index"),
})
class SparseIndexer(nn.Module):
    """DeepSeek sparse attention's lightning indexer: a score per query and key.

    A lightweight scorer beside the main MLA projections
    (`modeling_deepseek_v32.DeepseekV32Indexer`): `wq_b` reads the query
    residual, `wk` reads the hidden states into keys the cache holds, and
    `weights_proj` weights the heads into one score per key. `select`
    keeps the top-k keys of each query. A whole-sequence pass attends only
    those, in the latent space (`sparse_latent_attention`); the cache path
    folds them into the attention mask the way the reference's eager path
    does.

    The indexer reads its inputs detached. The reference trains it apart
    from the main model: the top-k is a selection, so the main loss reaches
    the indexer's weights nowhere, and the indexer's own loss (`indexer_kl`)
    reaches the main model nowhere.

    The indexer rotates its rope slice in its family's convention: V3.2's
    rotates half-split pairs (`modeling_deepseek_v32.py` calls
    `apply_rotary_pos_emb` in the indexer beside the main head's interleaved
    rotation) and GLM's rotates interleaved pairs like its main head
    (modeling_glm_moe_dsa.py:231-232), which `rope_interleave` selects.
    """

    q_lora_rank: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    top_k: int | None = None
    rope_interleave: bool = False
    dtype: Dtype | None = None
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
        self.k_norm = LayerNorm(epsilon=1e-6, dtype=self.dtype, name='k_norm')
        self.weights_proj = dense(self.n_heads, name='weights_proj')

    def keys(self, hidden):
        """The indexer's keys for these hidden states: `[B, S, head_dim]`."""
        return self.k_norm(self.wk(jax.lax.stop_gradient(hidden)))

    def _rotate(self, part, freqs_cos, freqs_sin):
        if self.rope_interleave:
            return apply_rotary_interleave(part, freqs_cos, freqs_sin)
        return apply_rotary(part, freqs_cos, freqs_sin)

    def rotated_keys(self, keys, freqs_cos, freqs_sin):
        """`keys` with their rope slice rotated at their own positions.

        Rotation happens once, before caching, so a cached key keeps the
        angle of its position while later queries rotate at theirs, which is
        what the reference's `update_indexer` ordering does.
        """
        k_rot, k_pass = jnp.split(keys, [self.rope_head_dim], axis=-1)
        k_rot = self._rotate(k_rot[:, :, None, :], freqs_cos, freqs_sin)
        return jnp.concatenate([k_rot[:, :, 0, :], k_pass], axis=-1)

    def scores(self, hidden, q_resid, keys, freqs_cos, freqs_sin):
        """Index scores of every query against every key: `[B, S, T]`, fp32.

        `keys` are the rotated keys of every candidate (the cache on
        decode); the query side is computed on each call. Scores run in
        fp32: the head weighting multiplies by `n_heads ** -0.5` in fp32 in
        the reference, and the relu keeps only the positive agreements.
        """
        hidden = jax.lax.stop_gradient(hidden)
        q_resid = jax.lax.stop_gradient(q_resid)
        batch, length, _ = hidden.shape
        query = self.wq_b(q_resid).reshape(
            batch, length, self.n_heads, self.head_dim)
        q_rot, q_pass = jnp.split(query, [self.rope_head_dim], axis=-1)
        query = jnp.concatenate(
            [self._rotate(q_rot, freqs_cos, freqs_sin), q_pass], axis=-1)
        scores = jnp.matmul(
            query.astype(jnp.float32),
            jnp.expand_dims(keys.astype(jnp.float32).transpose(0, 2, 1), -3))
        scores = jnp.maximum(scores * (self.head_dim ** -0.5), 0)
        weights = self.weights_proj(hidden).astype(jnp.float32)
        weights = weights * (self.n_heads ** -0.5)
        return jnp.matmul(weights[..., None, :], scores).squeeze(-2)

    def select(self, scores, keep):
        """The top-k keys of each query among those `keep` allows, as
        `top_k_selection` returns them."""
        if self.top_k is None:
            raise ValueError("selecting keys needs the indexer's top_k")
        return top_k_selection(scores, keep, self.top_k)


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
    with the indexer's keys, as each reference does. `positions`
    and `segment_ids` behave as on the standard mixer: absolute positions
    for the cache slots, per-document positions and a block-diagonal mask
    for a packed batch.
    `yarn` replaces the plain rope base with the YaRN ramp; when it is set
    the mixer's `rope_theta` has to equal the record's, so the scaling is
    configured once, and the mscale reaches the logits as a query
    pre-scale. causal=False is full attention with no cache, the mode a
    non-causal reader would take; decode=True raises there. The two latent
    norms are the model's RMSNorm under the model's `scale_offset` and
    `scale_after_cast`, since the reference builds them from the same class
    as every other norm of the layer.

    `index_n_heads` and `index_head_dim` together put the indexer beside
    the attention; `index_topk` makes it select, which is the released
    V3.2 layer. Without a top-k the attention stays dense and the indexer
    only scores, the state of V3.2's dense warm-up (arXiv 2512.02556,
    section 2.1.1), where a fresh indexer learns the dense attention of a
    frozen model. Whenever the indexer is present and the `indexer`
    collection is open, a training pass sows the per-query `indexer_kl`
    under `kl`, over the keys the attention itself used.
    `index_rope_interleave` is the indexer's own rotation convention.

    `index_shared` is GLM's IndexShare layer (modeling_glm_moe_dsa.py:313-318,
    432-446): it owns no indexer and attends the keys the last earlier
    selecting layer chose, which that layer stashes under `kv_store_key` in
    the `kv_store` the block threads down the stack, as a KV-sharing
    attention layer reads its provider's keys and values. It caches the
    expanded keys and values like any sparse layer, without indexer keys.

    `attention_impl` reaches the shared kernel path, which pads these
    values to the query's width for a fused kernel and hands back their own
    columns (`widen_value_heads`), so `auto` picks cudnn or xla off the
    query width like any other layer's: `qk_nope_head_dim +
    qk_rope_head_dim`, 192 in the released V3 configs, which is past the
    width cudnn tiles, so those run on xla. A mask this layer materializes
    outside decode (packed documents, row validity, the indexer's
    selection) takes 'auto' and 'cudnn' to xla, since cudnn reads a bool
    mask as an additive bias and refuses one at an odd length while
    training.
    """

    emb_features: int
    num_heads: int
    max_seq_len: int
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    causal: bool = True
    rope_theta: float = 10000.0
    rope_interleave: bool = True
    yarn: YarnScaling | None = None
    # DeepSeek and GLM norm the latents at RMSNorm's 1e-6 default whatever
    # the trunk's rms_norm_eps says (modeling_deepseek_v3.py:392,404,
    # modeling_glm_moe_dsa.py:349,361).
    norm_eps: float = 1e-6
    scale_offset: bool = False
    scale_after_cast: bool = False
    attention_bias: bool = False
    index_topk: int | None = None
    index_n_heads: int | None = None
    index_head_dim: int | None = None
    index_rope_interleave: bool = False
    index_shared: bool = False
    kv_store_key: str | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl

    def setup(self):
        if self.qk_rope_head_dim % 2:
            raise ValueError(
                "rotary positions rotate pairs, so the rope head dim must be "
                f"even, got {self.qk_rope_head_dim}")
        if self.qk_nope_head_dim < 1 or self.v_head_dim < 1:
            raise ValueError(
                "the nope head dim and the value head dim must be positive, "
                f"got {self.qk_nope_head_dim} and {self.v_head_dim}")
        if (self.index_n_heads is None) != (self.index_head_dim is None):
            raise ValueError(
                "the indexer needs its head count and head dim together, "
                "both set or both unset")
        if self.index_topk is not None and self.index_n_heads is None:
            raise ValueError(
                "index_topk selects with the indexer, which index_n_heads "
                "and index_head_dim have to describe")
        if self.index_shared and (self.index_n_heads is not None or self.index_topk is not None):
            raise ValueError(
                "index_shared reads another layer's selection, so the layer "
                "carries no indexer of its own: index_n_heads, index_head_dim "
                "and index_topk have to be None")
        if self.index_shared and self.kv_store_key is None:
            raise ValueError(
                "index_shared reads the selection under kv_store_key, which "
                "names the layer type that stashes it")
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        dense = functools.partial(
            nn.Dense, use_bias=self.attention_bias, dtype=self.dtype,
            precision=self.precision)
        norm = functools.partial(
            RMSNorm, epsilon=self.norm_eps,
            scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast, dtype=self.dtype)
        if self.q_lora_rank is None:
            # The reference's plain q_proj takes no bias, whatever
            # attention_bias says; only the low-rank projections do.
            self.q_proj = nn.Dense(
                self.num_heads * qk_head_dim, use_bias=False, dtype=self.dtype,
                precision=self.precision, name='q_proj')
        else:
            self.q_a_proj = dense(self.q_lora_rank, name='q_a_proj')
            self.q_a_layernorm = norm(name='q_a_layernorm')
            self.q_b_proj = nn.Dense(
                self.num_heads * qk_head_dim, use_bias=False, dtype=self.dtype,
                precision=self.precision, name='q_b_proj')
        self.kv_a_proj_with_mqa = dense(
            self.kv_lora_rank + self.qk_rope_head_dim, name='kv_a_proj_with_mqa')
        self.kv_a_layernorm = norm(name='kv_a_layernorm')
        self.kv_b_proj = nn.Dense(
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            use_bias=False, dtype=self.dtype, precision=self.precision,
            name='kv_b_proj')
        self.o_proj = dense(self.emb_features, name='o_proj')
        if self.index_n_heads is not None and self.index_head_dim is not None:
            if self.q_lora_rank is None:
                raise ValueError(
                    "the indexer reads the query residual, which only exists "
                    "with a q_lora_rank")
            self.indexer = SparseIndexer(
                q_lora_rank=self.q_lora_rank, n_heads=self.index_n_heads,
                head_dim=self.index_head_dim,
                rope_head_dim=self.qk_rope_head_dim, top_k=self.index_topk,
                rope_interleave=self.index_rope_interleave,
                dtype=self.dtype, precision=self.precision, name=INDEXER)

    @property
    def indexed(self) -> bool:
        """Whether the V3.2 indexer scores the keys beside the attention."""
        return self.index_n_heads is not None

    @property
    def sparse(self) -> bool:
        """Whether each query attends a selection of the keys: this layer's
        indexer's, or the one it shares."""
        return self.index_topk is not None or self.index_shared

    def _select(self, index_scores, keep, kv_store):
        """The keys each query attends among `keep`, as `top_k_selection`
        returns them: this layer's top-k, stashed for the layers sharing it,
        or the stashed selection read."""
        if self.index_shared:
            if kv_store is None or self.kv_store_key not in kv_store:
                raise ValueError(
                    f"layer shares the index selection under {self.kv_store_key!r} "
                    "but no earlier selecting layer stashed one; the model has to "
                    "pass one kv_store dict down its layer stack")
            indices = kv_store[self.kv_store_key]
            keep = jnp.broadcast_to(keep, (*indices.shape[:2], keep.shape[-1]))
            allowed = jnp.take_along_axis(keep, jnp.maximum(indices, 0), axis=-1)
            return jnp.where(allowed & (indices >= 0), indices, -1)
        selected = self.indexer.select(index_scores, keep)
        if kv_store is not None and self.kv_store_key is not None:
            kv_store[self.kv_store_key] = selected
        return selected

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
        # The projections carry the names a remat policy saves or offloads
        # (causal_transformer.RESIDUALS); kv_b_proj is the fused K/V one.
        if self.q_lora_rank is None:
            q_resid = None
            queries = self.q_proj(x)
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(x))
            queries = self.q_b_proj(q_resid)
        queries = checkpoint_name(queries, 'q_proj')
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

    def _rotated(self, q_rot, rot, positions):
        """Rotate the query's and the latent's rope heads at `positions`.

        Returns the rotated pair and the angles, which the indexer rotates
        its own keys with. The decoupled rope head is shared by every query
        head, so it rotates with a head axis added and taken away again.
        """
        freqs_cos, freqs_sin = mla_rope_freqs(
            positions, self.qk_rope_head_dim, self.rope_theta, self.yarn)
        return (self._rotate(q_rot, freqs_cos, freqs_sin),
                self._rotate(rot[:, :, None, :], freqs_cos, freqs_sin)[:, :, 0, :],
                freqs_cos, freqs_sin)

    def _expand(self, latent, rot):
        """Latent and rope head into per-head keys and values."""
        batch, length = latent.shape[0], latent.shape[1]
        width = self.qk_nope_head_dim + self.v_head_dim
        kv = checkpoint_name(self.kv_b_proj(latent), 'kv_proj').reshape(
            batch, length, self.num_heads, width)
        nope, values = jnp.split(kv, [self.qk_nope_head_dim], axis=-1)
        rot = jnp.broadcast_to(
            rot[:, :, None, :], (batch, length, self.num_heads, self.qk_rope_head_dim))
        return jnp.concatenate([nope, rot], axis=-1), values

    @nn.compact
    def __call__(self, x, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None,
                 attention_metadata: AttentionMetadata | None = None):
        causal, mask, objective = self.causal, None, None
        implementation = self.attention_impl
        _, length, _ = x.shape
        valid = None if attention_metadata is None else attention_metadata.valid
        if attention_metadata is not None and attention_metadata.rotary_positions is not None:
            raise ValueError("MLA does not implement multi-axis rotary positions")
        logical_positions = positions
        queries, q_resid = self._queries(x)
        # jnp splits at indices where torch splits into sizes: one cut point,
        # since the widths add up exactly.
        q_pass, q_rot = jnp.split(
            queries, [self.qk_nope_head_dim], axis=-1)
        latent, rot = self._latents(x)
        if decode:
            if segment_ids is not None:
                raise ValueError("decode accepts row validity, not packed segment_ids")
            # The cache hands out the slots first, because the rope heads
            # rotate at absolute positions: the queries at this step's slots
            # and the appended latents at theirs, while the cached ones keep
            # the angles of the slots they were written at.
            if self.sparse:
                # The cache is shaped by the expansion; its values are
                # recomputed after rotation below, so only shapes flow here.
                shape_key, shape_value = self._expand(latent, rot)
                index_keys = self.indexer.keys(x) if self.indexed else None
                positions, append = open_expanded_cache(
                    self, shape_key, shape_value, index_keys,
                    self.max_seq_len, valid=valid)
                q_rot, rot, freqs_cos, freqs_sin = self._rotated(
                    q_rot, rot,
                    positions if logical_positions is None else logical_positions)
                if index_keys is not None:
                    index_keys = self.indexer.rotated_keys(index_keys, freqs_cos, freqs_sin)
                key, value, index_full = append(
                    *self._expand(latent, rot), index_keys)
                index_scores = None
                if self.indexed:
                    assert q_resid is not None
                    index_scores = self.indexer.scores(
                        x, q_resid, index_full, freqs_cos, freqs_sin)
                keep = causal_attention_mask(
                    positions, key.shape[1],
                    key_valid=self.get_variable("cache", "cache_valid"))[:, 0]
                mask = selection_mask(self._select(index_scores, keep, kv_store),
                                      key.shape[1])[:, None]
            else:
                positions, append = open_latent_cache(
                    self, latent, rot, None, self.max_seq_len, valid=valid)
                q_rot, rot, _, _ = self._rotated(
                    q_rot, rot,
                    positions if logical_positions is None else logical_positions)
                latent, rot, _ = append(latent, rot, None)
                key, value = self._expand(latent, rot)
                mask = causal_attention_mask(
                    positions, key.shape[-3], key_valid=self.get_variable("cache", "cache_valid"))
            causal = False
        else:
            if positions is None:
                positions = (jnp.arange(length) if valid is None else
                             jnp.maximum(jnp.cumsum(valid, axis=1) - 1, 0))
            else:
                positions = jnp.asarray(positions)
            q_rot, rot, freqs_cos, freqs_sin = self._rotated(q_rot, rot, positions)
            if segment_ids is not None:
                inside = document_mask(segment_ids)[:, None]
                mask = inside
                if causal:
                    mask = jnp.logical_and(
                        inside, causal_attention_mask(
                            jnp.arange(length), length))
                causal = False
            if self.indexed or self.index_shared:
                # The keys a query may attend before selection: the rows'
                # causal order (packed positions restart per document, so
                # they order nothing here) and the packed base when there
                # is one. The indexer selects among those, and the KL
                # measures over whatever the attention then uses.
                keep = None if mask is None else mask[:, 0]
                if causal:
                    rows = causal_attention_mask(jnp.arange(length), length)[:, 0]
                    keep = rows if keep is None else jnp.logical_and(keep, rows)
                if keep is None:
                    keep = jnp.ones((1, length, length), jnp.bool_)
                if valid is not None:
                    # Selection and its objective range over real keys only.
                    keep = jnp.logical_and(keep, jnp.asarray(valid, bool)[:, None, :])
                index_scores = None
                if self.indexed:
                    assert q_resid is not None
                    index_keys = self.indexer.rotated_keys(
                        self.indexer.keys(x), freqs_cos, freqs_sin)
                    index_scores = self.indexer.scores(
                        x, q_resid, index_keys, freqs_cos, freqs_sin)
                if self.sparse:
                    selection = self._select(index_scores, keep, kv_store)
                    if self._attends_sparsely(selection, length):
                        return self._sparse_attention(
                            q_pass, q_rot, latent, rot, selection, index_scores)
                    keep = selection_mask(selection, length)
                    mask, causal = keep[:, None], False
                if (index_scores is not None and not self.is_initializing()
                        and self.is_mutable_collection(INDEXER_COLLECTION)):
                    objective = (index_scores, keep)
            key, value = self._expand(latent, rot)
        if valid is not None:
            queries_valid = jnp.asarray(valid, bool)[:, None, :, None]
            mask = queries_valid if mask is None else mask & queries_valid
            if not decode:
                mask = mask & jnp.asarray(valid, bool)[:, None, None, :]
        query = self._scaled_query(q_pass, q_rot)
        if mask is not None and not decode:
            # Packed documents, row validity and the indexer's selection all
            # materialize a mask on the training path, so they take the xla
            # kernel, the way the standard mixer's own masks do. Decoding
            # runs no backward pass, so its cache mask keeps the fused path.
            implementation = kernel_for_materialized_mask(
                implementation, query, dtype=self.dtype, precision=self.precision)
        if objective is not None:
            self._sow_indexer_kl(*objective, query, key)
        # The per-head maxima the QK-Clip reads, with the nope width the
        # clip needs to split the latent projections: computed only when a
        # caller opened the collection. Under the sparse indexer the mask
        # holds the selected keys, so the maximum is over those.
        if self._qk_open():
            self.sow("qk", "max_logits", max_attention_logits(
                query, key, causal=causal, mask=mask))
            self.sow("qk", "qk_nope", jnp.asarray(self.qk_nope_head_dim))
        attention = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            implementation=implementation, causal=causal, mask=mask)
        return self._output(attention)

    def _qk_open(self) -> bool:
        return not self.is_initializing() and self.is_mutable_collection("qk")

    def _attends_sparsely(self, selection, total: int) -> bool:
        """Whether a whole-sequence pass runs `sparse_latent_attention`.

        It does when the selection is narrower than the keys, so the dense
        kernel would read keys the selection drops. A selection as wide as
        the sequence is every allowed key, which the dense kernel under the
        allowed mask computes. An open `qk` collection reads the dense
        logits for its maxima, so it keeps the masked kernel, and so does a
        call whose `kv_b_proj` input is drawn on (`stochastic_input`): the
        absorbed path feeds kv_b_proj the identity, where a dropout draw
        would be one mask for every token rather than one per token.
        """
        return (selection.shape[-1] < total and not self._qk_open()
                and not self.stochastic_input("kv_b_proj"))

    def stochastic_input(self, name: str) -> bool:
        """Whether this call draws randomness on the input of the submodule
        `name`. The layer draws none; a method interceptor that does, as
        `dew.lora`'s dropout does on its targets, answers for itself."""
        del name
        return False

    def _scaled_query(self, q_pass, q_rot):
        """The whole query with YaRN's mscale on it, as every kernel reads it."""
        query = jnp.concatenate([q_pass, q_rot], axis=-1)
        scale = self.query_scale
        return query * scale if scale != 1.0 else query

    def _sow_indexer_kl(self, index_scores, index_keep, query, key):
        """The indexer's objective, per query, over the keys the attention
        itself attends: dense attention's causal set in the warm-up, the
        selected set under the top-k. The kernel scales the (yarn-prescaled)
        query by the head width, so the target does."""
        self.sow(INDEXER_COLLECTION, "kl", indexer_kl(
            index_scores, query, key, index_keep,
            1.0 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)))

    def _sparse_attention(self, q_pass, q_rot, latent, rot, selection, index_scores):
        """Attend the selection in the latent space (`sparse_latent_attention`).

        `kv_b_proj` applied to the identity is its matrix as the layer
        computes it, dtype policy and any deterministic interceptor's branch
        (a LoRA adapter without dropout in this call, a quantization rule)
        included, so the absorbed weights are the ones the expanded path
        multiplies by. The indexer's objective, when open, still reads the
        expanded keys over the selection mask.
        """
        heads, nope = self.num_heads, self.qk_nope_head_dim
        matrix = self.kv_b_proj(jnp.eye(self.kv_lora_rank, dtype=latent.dtype)).reshape(
            self.kv_lora_rank, heads, nope + self.v_head_dim)
        key_weight, value_weight = jnp.split(matrix, [nope], axis=-1)
        scale = self.query_scale
        if (index_scores is not None and not self.is_initializing()
                and self.is_mutable_collection(INDEXER_COLLECTION)):
            key, _ = self._expand(latent, rot)
            self._sow_indexer_kl(index_scores, selection_mask(selection, latent.shape[1]),
                                 self._scaled_query(q_pass, q_rot), key)
        attention = sparse_latent_attention(
            q_pass * scale if scale != 1.0 else q_pass,
            q_rot * scale if scale != 1.0 else q_rot,
            latent, rot, key_weight, value_weight, selection,
            scale=1.0 / math.sqrt(nope + self.qk_rope_head_dim), precision=self.precision)
        return self._output(attention)

    def _output(self, attention):
        batch, length = attention.shape[:2]
        return checkpoint_name(self.o_proj(checkpoint_name(attention, 'context').reshape(
            batch, length, self.num_heads * self.v_head_dim)), 'o_proj')


# The standard attention mixer reads the YaRN ramp above, so the registry
# hub imports this module while it imports the hub; the registry side of
# this module comes after the ramp so either import order resolves.
from dew.nn.mixers import MixerBase, MixerContext, mixers


@mixers("mla")
@dataclasses.dataclass(frozen=True)
class MLAMixer(MixerBase):
    """The `mla` kind: DeepSeek's latent attention under the reference's names.

    A config names it as `mixer={"kind": "mla", ...}` with the fields of a
    DeepSeek config.json, so translation renames nothing; `yarn` is the
    rope-scaling record (or None for plain rope). `index_n_heads` and
    `index_head_dim` put the V3.2 indexer beside the attention and
    `index_topk` makes it select, the released sparse layer; the heads
    without a top-k is dense attention with an indexer scoring beside it,
    the state the dense warm-up trains (all None is dense MLA). The rope
    base is the model's `rope_theta`, transformed by the yarn ramp rather
    than replaced, so scaling is configured once, and the mscale is applied
    in the attention as a query pre-scale.

    `index_rope_interleave` is the indexer's rotation convention: V3.2
    rotates half-split pairs, GLM interleaved ones.

    The context's grouped-query geometry (`num_kv_heads`, `head_dim`) has
    no meaning here and is not read, as the backbone documents; `qk_norm` is
    not read either, since the latent norms are the design's own and always
    present. A layer the backbone marks `kv_shared` is GLM's IndexShare
    layer (modeling_glm_moe_dsa.py:313-318): what an MLA layer shares is
    its indexer's selection, so the layer builds no indexer and attends the
    keys its provider chose, which needs the kind to select. The dials a
    standard attention honours and this cannot (a values norm, a window, an
    attention scale, a partial rotary) are refused.
    """

    q_lora_rank: int | None = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_interleave: bool = True
    yarn: YarnScaling | None = None
    index_topk: int | None = None
    index_n_heads: int | None = None
    index_head_dim: int | None = None
    index_rope_interleave: bool = False

    @property
    def indexed(self) -> bool:
        """Whether the layer carries the indexer, selecting or not."""
        return self.index_n_heads is not None

    @property
    def sparse(self) -> bool:
        """Whether the indexer selects the keys each query attends."""
        return self.index_topk is not None

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        unsupported = {
            "v_norm": ctx.v_norm,
            "k_eq_v": ctx.k_eq_v,
            "sliding_window": ctx.sliding_window,
            "attention_chunk": ctx.attention_chunk,
            "attention_scale": ctx.attention_scale,
            "partial_rotary_factor": ctx.partial_rotary_factor,
            "kv_cache": ctx.kv_cache != KVCache(),
        }
        asked = sorted(name for name, value in unsupported.items() if value)
        if asked:
            raise ValueError(
                f"the mla mixer has no {', '.join(asked)}: the latent "
                "attention scales by its head dims and the yarn mscale, "
                "rotates its rope head whole, attends the whole sequence "
                "and norms its latents, not its values")
        if self.yarn is not None and self.yarn.rope_theta != ctx.rope_theta:
            raise ValueError(
                f"the yarn record's rope_theta ({self.yarn.rope_theta}) and "
                f"the layer's ({ctx.rope_theta}) disagree; the rope base is "
                "configured once, on the model")
        if ctx.kv_shared and not self.sparse:
            raise ValueError(
                "a sharing mla layer attends the selection of an earlier "
                "layer's indexer, so the kind has to select: index_topk, "
                "index_n_heads and index_head_dim have to be set")
        return functools.partial(
            MultiHeadLatentAttention,
            emb_features=ctx.emb_features,
            num_heads=ctx.num_heads,
            max_seq_len=ctx.max_seq_len,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            causal=ctx.causal,
            rope_theta=ctx.rope_theta,
            rope_interleave=self.rope_interleave,
            yarn=self.yarn,
            scale_offset=ctx.scale_offset,
            scale_after_cast=ctx.scale_after_cast,
            attention_bias=ctx.attention_bias,
            index_topk=None if ctx.kv_shared else self.index_topk,
            index_n_heads=None if ctx.kv_shared else self.index_n_heads,
            index_head_dim=None if ctx.kv_shared else self.index_head_dim,
            index_rope_interleave=self.index_rope_interleave,
            index_shared=ctx.kv_shared,
            kv_store_key=ctx.kv_store_key,
            dtype=ctx.dtype,
            precision=ctx.precision,
            attention_impl=ctx.attention_impl)
