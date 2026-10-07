"""Attend over queries, keys and values, cache them, and block them up.

One kernel path serves every attention module here. The KV cache and the
UNet transformer blocks sit beside it; the blocks come from diffusers'
attention_flax.py.
"""

import contextlib
import contextvars
import dataclasses
import functools
import importlib
import importlib.util
import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.dtypes import canonicalize_dtype, promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as P

from dew.telemetry.devices import deterministic_ops_requested

from .attention_sinks import attention_with_sinks
from .kernels import decode_attention
from .kernels.generation import bf16_dot_runs
from .kv_cache import Append, KVCache, KVStore, filled_slots
from .precision import (
    at_default_precision,
    at_least_fp32,
    bf16_operand_precision,
    precision_names,
    rounded_to,
)
from .rope import apply_rotary
from .sharding import (
    HEADS,
    KV_HEADS,
    SEQUENCE_AXIS,
    LayoutRefused,
    constrain,
    logical_axes,
    logical_spec,
    manual_map,
    mesh_axes,
    row_axes,
    sequence_shards,
    split_positions,
)

AttentionImpl = Literal["auto", "reference", "xla", "cudnn", "triton", "tpu"]
"""Names which kernel an attention call runs.

Every layer that carries the choice spells it the same way: a `ModelConfig`
field, a module's `attention_impl`, and `scaled_dot_product_attention`'s
`implementation`.

'reference' is the portable einsum and softmax, and the only path that reads
dtype, precision and force_fp32_for_softmax. 'xla' and 'cudnn' are
`jax.nn.dot_product_attention`'s own two. 'triton' is tokamax's Pallas-Triton
flash kernel (`triton_attention`), which needs tokamax installed
(docs/installation.md). 'tpu' is the pallas splash kernel, with the older pallas
flash kernel behind it for the calls splash's mask descriptor cannot carry.
'auto', every module's default, resolves per trace (`resolve_implementation`):
triton where cudnn's kernel runs and `triton_runs`; cudnn where its kernel
runs; tpu where splash's does;
the reference path where the call asks for arithmetic only it honours; and
xla anywhere else.
"""


def repeat_kv_heads(x, num_heads: int):
    """Repeat grouped key/value heads out to the query heads: [B, S, K, D] -> [B, S, N, D].

    Query head n reads key/value head n // (N // K), the grouping
    jax.nn.dot_product_attention uses internally, so the same param tree runs
    on the kernels that group heads themselves and on the ones that need the
    keys materialized.
    """
    kv_heads = x.shape[-2]
    if kv_heads == num_heads:
        return x
    if num_heads % kv_heads:
        raise ValueError(
            f"grouped-query attention needs the query heads ({num_heads}) to be "
            f"a multiple of the key/value heads ({kv_heads}).")
    return jnp.repeat(x, num_heads // kv_heads, axis=-2)


def causal_attention_mask(query_positions, kv_len: int, sliding_window=None, *, key_valid=None):
    """Boolean [B, 1, T, S] mask from shared [T] or per-row [B, T] slots.

    A negative query slot denotes an invalid token. Cache validity excludes
    unfilled slots; a window retains the query and its preceding W-1 keys.
    """
    positions = jnp.asarray(query_positions)
    if positions.ndim == 1:
        positions = positions[None, :]
    q_pos = positions[:, :, None]
    k_pos = jnp.arange(kv_len)[None, None, :]
    mask = (q_pos >= 0) & (k_pos <= q_pos)
    if sliding_window is not None:
        mask = mask & (k_pos > q_pos - sliding_window)
    if key_valid is not None:
        mask = mask & jnp.asarray(key_valid, bool)[:, None, :]
    return mask[:, None]


def alibi_bias(key_positions, num_heads: int, *, dtype) -> jax.Array:
    """BLOOM/Falcon's per-head linear key-position bias, `[B, H, 1, K]`.

    The slopes follow transformers' `build_alibi_tensor`, including the odd
    powers appended for a non-power-of-two head count. Dropping the query's
    position subtracts one constant from every visible logit of that query,
    which leaves softmax unchanged.
    """
    power = 2 ** math.floor(math.log2(num_heads))
    base = np.float32(2 ** (-(2 ** -(math.log2(power) - 3))))
    slopes = np.power(base, np.arange(1, power + 1, dtype=np.float32))
    if power != num_heads:
        extra = np.float32(2 ** (-(2 ** -(math.log2(2 * power) - 3))))
        slopes = np.concatenate((slopes, np.power(
            extra, np.arange(1, 2 * (num_heads - power), 2, dtype=np.float32))))
    positions = jnp.atleast_2d(key_positions)
    return (jnp.asarray(slopes)[None, :, None, None]
            * positions[:, None, None, :]).astype(dtype)


def window_sides(causal: bool, sliding_window: int) -> tuple[int, int]:
    """The keys a window of `sliding_window` keeps before and after each query,
    counted as `jax.nn.dot_product_attention`'s `local_window_size` counts them.

    A causal window keeps the query and the w - 1 keys before it. A
    bidirectional one keeps the keys within w - 1 positions on either side,
    |q - k| < w: ModernBERT's |q - k| <= local_attention // 2, which is w =
    local_attention // 2 + 1 (masking_utils.py:141-158, transformers 5.16.1).
    A model layer passes one only where its kind asks for it
    (`LayerKind.bidirectional_window`).
    """
    return sliding_window - 1, 0 if causal else sliding_window - 1


def structural_mask(query_positions, kv_len: int, causal: bool,
                    sliding_window: int | None) -> jax.Array | None:
    """The `[B, 1, T, S]` keys that causality and a window leave each query,
    from shared `[T]` or per-row `[B, T]` positions, or None when neither
    applies. A negative query position sees nothing."""
    if causal:
        return causal_attention_mask(query_positions, kv_len, sliding_window)
    if sliding_window is None:
        return None
    positions = jnp.asarray(query_positions)
    if positions.ndim == 1:
        positions = positions[None, :]
    distance = positions[:, :, None] - jnp.arange(kv_len)[None, None, :]
    return ((positions[:, :, None] >= 0) & (jnp.abs(distance) < sliding_window))[:, None]


def document_mask(segment_ids) -> jax.Array:
    """Keep each packed document to itself: `[B, S, S]` boolean.

    Two positions see each other when they carry the same segment id.
    Segment 0 is padding, which sees nothing and is seen by nothing. This is
    the mask every kernel but splash and the pallas flash kernel reads for
    `segment_ids`; those two compare the ids per block.
    """
    segment_ids = jnp.asarray(segment_ids)
    return ((segment_ids[:, :, None] == segment_ids[:, None, :])
            & (segment_ids[:, :, None] != 0))


def chunk_mask(query_positions, key_positions, chunk_size: int):
    """Boolean `[.., 1, T, S]` keeping keys in the query's chunk, both absolute.

    transformers' `chunked_overlay`: `kv // chunk == q // chunk`, positions
    counted from the sequence start, so a packed document's positions place
    its chunks from its own first token.
    """
    query_chunks = jnp.asarray(query_positions) // chunk_size
    key_chunks = jnp.asarray(key_positions) // chunk_size
    same = query_chunks[..., :, None] == key_chunks[..., None, :]
    return same[..., None, :, :] if same.ndim == 3 else same[None, None]


def combined_attention_mask(query_length: int, key_length: int, causal: bool,
                            sliding_window: int | None,
                            mask: jax.Array | None) -> jax.Array | None:
    """Fold causality and a sliding window into `mask`.

    A causal flag keeps the keys at or before each query's row; a window
    narrows that to the most recent keys, or to the nearest on either side
    without the flag (`window_sides`). Both read the row index, the way the
    fused kernels take them as flags. Unset stays unset, so a caller that
    distinguishes no mask from an all-true one keeps doing so.
    """
    structural = structural_mask(jnp.arange(query_length), key_length, causal, sliding_window)
    if structural is not None:
        mask = structural if mask is None else jnp.logical_and(mask, structural)
    return mask


def with_key_lengths(mask: jax.Array | None, lengths: jax.Array, key_length: int) -> jax.Array:
    """`mask` narrowed to each row's first `lengths[b]` keys, `[B, 1, 1, K]`
    where there was no mask. This is the mask a `key_value_seq_lengths`
    call means on the paths whose kernel takes no lengths."""
    kept = (jnp.arange(key_length) < lengths[:, None])[:, None, None, :]
    return kept if mask is None else jnp.logical_and(mask, kept)


def max_attention_logits(query: jax.Array, key: jax.Array, *, causal: bool = False,
                         sliding_window: int | None = None,
                         mask: jax.Array | None = None,
                         bias: jax.Array | None = None) -> jax.Array:
    """Return the largest pre-softmax logit per query head: `[batch, heads]`, fp32.

    The logits are the scaled dot products the kernels softmax. Masked
    positions read -inf, so causality and packing never trip the maximum.
    Grouped key heads repeat out to the query heads first, the grouping the
    kernels run. A logit softcap and attention sinks stay out: the clip that
    reads this bounds the raw query-key growth, and a softcapped model bounds
    it already.
    """
    heads = query.shape[-2]
    key = repeat_kv_heads(key, heads)
    scale = 1.0 / math.sqrt(query.shape[-1])
    logits = jnp.einsum('...qhd,...khd->...hqk',
                        query.astype(jnp.float32) * jnp.asarray(scale, jnp.float32),
                        key.astype(jnp.float32))
    if bias is not None:
        logits = logits + bias.astype(jnp.float32)
    combined = combined_attention_mask(
        query.shape[-3], key.shape[-3], causal, sliding_window, mask)
    if combined is not None:
        logits = jnp.where(combined, logits, -jnp.inf)
    return jnp.max(logits, axis=(-2, -1))


def normalized_in_fp32(normalize, static_argnums: tuple[int, ...] = ()):
    """Wrap a norm body in `jax.checkpoint` so nothing fp32 is retained.

    The norms reduce in fp32 for bf16 stability, and differentiated as written
    the backward pass would keep two fp32 copies of the activations per norm
    (2.35 GiB on one SimpleDiT step). Saving nothing keeps the arguments (the
    bf16 input, the weight and the bias) and leaves the residuals to the policy
    that recomputes the block (`decoder_block.RESIDUALS`). The wrapped function
    takes arrays first and static arguments last, so no flax lifting sits
    between the checkpoint and the module; forward values are unchanged.
    """
    return jax.checkpoint(normalize, policy=jax.checkpoint_policies.nothing_saveable,
                          static_argnums=static_argnums)


@functools.partial(normalized_in_fp32, static_argnums=(2, 3, 4, 5, 6))
def rms_normalized(x, scale, epsilon: float, dtype, scale_offset: bool, scale_after_cast: bool,
                   fp32_statistics: bool):
    """Normalize `x` by its root mean square, then apply `scale`.

    The families differ on which side of the cast the weight goes, which
    `scale_after_cast` picks. `scale` of None is the weightless norm.
    `fp32_statistics` of False computes the square, the mean and the
    normalization in the input's dtype.
    """
    if fp32_statistics:
        y = x.astype(at_least_fp32(x.dtype))
        y = y * jax.lax.rsqrt(jnp.mean(jnp.square(y), axis=-1, keepdims=True) + epsilon)
    else:
        # The reference rounds every step to the input dtype. XLA fuses the
        # chain and carries fp32 between the ops, so each rounding is explicit.
        y = rounded_to(x * rounded_to(jax.lax.rsqrt(rounded_to(
            rounded_to(jnp.mean(rounded_to(jnp.square(x), x.dtype), axis=-1, keepdims=True), x.dtype)
            + epsilon, x.dtype)), x.dtype), x.dtype)
    if scale is None:
        # A pure normalization with no learned weight, as Gemma 4 norms
        # its values (modeling_gemma4.py, Gemma4RMSNorm with_scale=False).
        return y.astype(dtype)
    weight = (1.0 + scale) if scale_offset else scale
    if scale_after_cast:
        # The reference casts the activations, then multiplies by its weight:
        # a bf16 product for a bf16 weight, an fp32 one for an fp32 master
        # that the next layer rounds. Rounding in place keeps both roundings
        # under jit, where XLA drops a narrowing cast that a widening one
        # follows, and keeps the weight's product and gradient in fp32.
        return rounded_to(rounded_to(y, dtype) * weight, dtype).astype(dtype)
    return (y * weight).astype(dtype)


@functools.partial(normalized_in_fp32, static_argnums=(3, 4))
def layer_normalized(x, scale, bias, epsilon: float, dtype):
    """Normalize `x` over its last axis, op for op as flax computes it.

    E[x] and E[x^2] are reduced in fp32 and the variance comes from the
    pair, clipped at zero. The weight folds into the inverse deviation
    before it meets the centered activations. `scale` and `bias` of None
    are the affine-free norm.
    """
    y = x.astype(at_least_fp32(x.dtype))
    row_mean = jnp.mean(y, axis=-1)
    variance = jnp.maximum(0.0, jnp.mean(jax.lax.square(y), axis=-1) - jax.lax.square(row_mean))
    scaling = jax.lax.rsqrt(jnp.expand_dims(variance, -1) + epsilon)
    width = (1,) * (x.ndim - 1) + (-1,)
    if scale is not None:
        scaling = scaling * jnp.reshape(scale, width)
    y = (y - jnp.expand_dims(row_mean, -1)) * scaling
    if bias is not None:
        y = y + jnp.reshape(bias, width)
    return y.astype(dtype)


def unweighted_rmsnorm(x, eps: float):
    """Normalize the last axis by its root mean square, with no learned weight.

    The reduction runs in fp32 and the inverse deviation is cast back
    before it multiplies, which is what `DeepseekV4UnweightedRMSNorm`
    (modeling_deepseek_v4.py:66-72) and its GLM twin do.
    """
    fp32 = x.astype(at_least_fp32(x.dtype))
    inverse = jax.lax.rsqrt(jnp.mean(jnp.square(fp32), axis=-1, keepdims=True) + eps)
    return x * inverse.astype(x.dtype)


class RMSNorm(nn.Module):
    """Normalize the last axis by its root mean square, reducing in fp32 by default.

    `scale_offset` stores the weight as Gemma does, (1 + w), initialized to
    zeros, so a Gemma checkpoint's weights land unchanged. Gemma multiplies in
    fp32 and casts the product (modeling_gemma3.py:147-150); Llama and Qwen3
    cast first and multiply by the weight (modeling_qwen3.py:61-64), which
    `scale_after_cast` reproduces. The two agree at fp32 and differ under bf16.
    `fp32_statistics=False` with `scale_after_cast` is timm's `RmsNorm2d`
    order, x * rsqrt(mean(x^2) + eps) * w in the input dtype. The backward pass
    keeps the input in its own dtype (`normalized_in_fp32`).
    """
    epsilon: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    with_scale: bool = True
    fp32_statistics: bool = True
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, x):
        dtype = self.dtype if self.dtype is not None else x.dtype
        scale = self.param(
            'scale',
            nn.initializers.zeros if self.scale_offset else nn.initializers.ones,
            (x.shape[-1],), jnp.float32) if self.with_scale else None
        return rms_normalized(x, scale, self.epsilon, dtype, self.scale_offset,
                              self.scale_after_cast, self.fp32_statistics)


class LayerNorm(nn.Module):
    """Normalize the last axis as flax's `nn.LayerNorm` does, op for op.

    The parameter tree and the fp32 arithmetic are flax's, so checkpoints move
    both ways and forward values match bit for bit. The backward pass keeps the
    input as it arrived (`normalized_in_fp32`) where flax keeps fp32 copies.
    Other axes, masks, pmapped axes and the exact variance are flax's to serve.
    """
    epsilon: float = 1e-6
    use_scale: bool = True
    use_bias: bool = True
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        width = (x.shape[-1],)
        scale = self.param('scale', nn.initializers.ones, width,
                           self.param_dtype) if self.use_scale else None
        bias = self.param('bias', nn.initializers.zeros, width,
                          self.param_dtype) if self.use_bias else None
        return layer_normalized(x, scale, bias, self.epsilon,
                                canonicalize_dtype(x, scale, bias, dtype=self.dtype))


def _cache_positions(module: nn.Module, batch: int, length: int, valid):
    """Allocate compact cache slots for real tokens, independently per row.

    A row's tokens fill its slots in order, so the cursor alone says which
    hold one (`cached_validity`)."""
    valid = jnp.ones((batch, length), bool) if valid is None else jnp.asarray(valid, bool)
    if valid.shape != (batch, length):
        raise ValueError(f"cache validity must be {(batch, length)}, got {valid.shape}")
    allocated = module.has_variable("cache", "cache_index")
    index = module.variable("cache", "cache_index", jnp.zeros, (batch,), jnp.int32)
    positions = index.value[:, None] + jnp.cumsum(valid, axis=1, dtype=jnp.int32) - 1
    positions = jnp.where(valid, positions, -1)
    if allocated:
        index.value = index.value + jnp.sum(valid, axis=1, dtype=jnp.int32)
    return positions, allocated


def cached_validity(module: nn.Module, length: int) -> jax.Array:
    """Which of `module`'s `length` cache slots hold a token, `[rows, length]`:
    those before the row's cursor, which `_cache_positions` fills in order.

    Derived where it is read rather than stored beside the cursor, so a
    decode step writes no `[rows, capacity]` copy of it a layer: 28 kernels
    a step on Qwen3-0.6B, 0.6% of a 32-slot serving step on an RTX 4080
    (docs/performance.md)."""
    return filled_slots(module.get_variable("cache", "cache_index"), length)


_DEFAULT_CACHE = KVCache()


def open_kv_cache(module: nn.Module, key, max_seq_len, *, valid=None, layout: KVCache = _DEFAULT_CACHE):
    """Fixed-size K/V with a cursor and cached validity for each batch row.

    Returns [B, S] compact slot positions and a writer. Invalid tokens have
    position -1 and do not advance the cursor. The allocation-only call leaves
    every row empty. The writer returns full cache arrays, including unused
    slots which the caller excludes with `cached_validity`. `layout` chooses the
    storage behind the slots, dense or paged, full or quantized
    (`dew.nn.kv_cache`); the slots and the cursor are the same for all.
    """
    shards = sequence_shards()
    if shards > 1:
        raise LayoutRefused(
            f"decoding is not supported under a sequence axis of {shards}: the KV "
            "cache holds whole sequences. Generate on a mesh with sequence=1.")
    if max_seq_len is None:
        raise ValueError("decoding needs max_seq_len for its fixed-capacity cache")
    batch, length, heads, head_dim = key.shape
    if valid is None and length > max_seq_len:
        raise ValueError(f"{length} tokens do not fit a KV cache of {max_seq_len}.")
    store = KVStore.open(module, layout, batch, max_seq_len, heads, head_dim, key.dtype)
    positions, allocated = _cache_positions(module, batch, length, valid)
    return positions, Append(store, positions, allocated)


def _pad_rows(x, rows: int):
    """`x` with `rows` zero rows appended on its sequence axis, [B, S, H, D]."""
    return x if rows == 0 else jnp.pad(x, ((0, 0), (0, rows), (0, 0), (0, 0)))


def widen_value_heads(query, value):
    """Zero-pad `value`'s head axis out to the query's width.

    `jax.nn.dot_product_attention` refuses a value narrower than the key, which
    is DeepSeek's latent attention (`v_head_dim` 128 against a 192-wide query in
    every released V2/V3 config). Each value column is its own sum, so a zero
    column gives a zero output column the caller crops; transformers 5.16.1 pads
    FlashAttention the same way (integrations/flash_attention.py:63). A wider
    value raises: padding the query and key would move the kernel's 1/sqrt(d)
    off the query's width.
    """
    width, v_width = query.shape[-1], value.shape[-1]
    if v_width > width:
        raise ValueError(
            "fused attention runs one head width for the keys and the values, "
            f"and a value head of {v_width} is wider than the query head of "
            f"{width}: the kernel would scale the logits by the padded width. "
            "Use the reference implementation (attention_impl 'reference').")
    return jnp.pad(value, ((0, 0),) * (value.ndim - 1) + ((0, width - v_width),))


def cudnn_attention(query, key, value, bias, mask, causal, sliding_window,
                    key_value_seq_lengths=None):
    """Run jax's cudnn flash attention over any sequence length.

    cudnn has no backward pass for an odd query or key length (77 CLIP text
    tokens, most text-plus-image sequences), so one zero row pads it even. The
    padded query row is sliced off and the padded key hidden by the kernel's
    key_value_seq_lengths, which never reach past the real keys.
    tests/test_kernels.py pins the equality with the xla path.
    """
    q_len, kv_len = query.shape[-3], key.shape[-3]
    # jax's cudnn call takes a mask or bias only at the full [.., Q, K]
    # (check_layout in jax/_src/cudnn/fused_attention_stablehlo.py), so a
    # key-padding mask broadcast over the queries, [B, 1, 1, K], is widened.
    def spanning(x):
        return None if x is None else jnp.broadcast_to(x, (*x.shape[:-2], q_len, kv_len))
    mask, bias = spanning(mask), spanning(bias)
    q_pad, kv_pad = q_len % 2, kv_len % 2
    if q_pad or kv_pad:
        query = _pad_rows(query, q_pad)
        key, value = _pad_rows(key, kv_pad), _pad_rows(value, kv_pad)
        # Masks and biases broadcast over [B, H, Q, K]; only the last two
        # dimensions grow, and a padded query row or key column sees nothing.
        def pad_tail(x, fill):
            return jnp.pad(x, ((0, 0),) * (x.ndim - 2) + ((0, q_pad), (0, kv_pad)),
                           constant_values=fill)
        mask = None if mask is None else pad_tail(mask, fill=False)
        bias = None if bias is None else pad_tail(bias, 0)
    kv_lengths = key_value_seq_lengths
    if kv_lengths is None and kv_pad:
        kv_lengths = jnp.full(key.shape[:1], kv_len, jnp.int32)
    # A window of (l, r) keeps the l keys before the query and the r after it
    # on both the xla and the cudnn path, the sides `window_sides` counts.
    out = jax.nn.dot_product_attention(
        query, key, value, bias=bias, mask=mask, is_causal=causal,
        key_value_seq_lengths=kv_lengths,
        local_window_size=None if sliding_window is None else window_sides(causal, sliding_window),
        implementation='cudnn')
    return out[:, :q_len] if q_pad else out


def stripe(x, shards: int, axis: int = 1):
    """Reorder `axis` so every shard holds equal causal work.

    The axis is cut into 2 * shards chunks and shard i takes chunks i and
    2 * shards - 1 - i, MaxText's `reorder_sequence` order: for two shards
    [0..7] becomes [0, 1, 6, 7, 2, 3, 4, 5]. Reshape, slice, flip and stack
    lower to collective-permutes of the chunks that move; a gather along a split
    axis lowers to an all-gather of the whole array
    (tests/test_sequence_parallel.py measures both).
    """
    axis %= x.ndim
    length = x.shape[axis]
    if length % (2 * shards):
        raise LayoutRefused(
            f"sequence parallelism over {shards} shards pairs chunks of the "
            f"sequence to balance the causal work, which needs the sequence "
            f"length to be a multiple of {2 * shards}, got {length}")
    chunk = length // (2 * shards)
    chunks = x.reshape((*x.shape[:axis], 2 * shards, chunk, *x.shape[axis + 1:]))
    first = jax.lax.slice_in_dim(chunks, 0, shards, axis=axis)
    second = jnp.flip(jax.lax.slice_in_dim(chunks, shards, 2 * shards, axis=axis), axis=axis)
    return jnp.stack([first, second], axis=axis + 1).reshape(x.shape)


def unstripe(x, shards: int, axis: int = 1):
    """The inverse of `stripe`: `axis` back in sequence order."""
    axis %= x.ndim
    chunk = x.shape[axis] // (2 * shards)
    pairs = x.reshape((*x.shape[:axis], shards, 2, chunk, *x.shape[axis + 1:]))
    first = jax.lax.index_in_dim(pairs, 0, axis=axis + 1, keepdims=False)
    second = jnp.flip(jax.lax.index_in_dim(pairs, 1, axis=axis + 1, keepdims=False), axis=axis)
    return jnp.concatenate([first, second], axis=axis).reshape(x.shape)


def sequence_parallel_attention(kernel, query, key, value, shards: int, *, causal,
                                sliding_window, mask, bias, sinks, key_value_seq_lengths=None):
    """Run `kernel` over a sequence the mesh's sequence axis splits.

    `exchanged_heads_attention` (Ulysses) takes a call whose query heads divide
    by tensor times sequence and whose lengths divide by the shard count;
    `gathered_keys_attention` takes any head count and length. Where both run,
    a causal, windowed or masked call takes the all-to-all: its kernel sees
    whole sequences and skips masked blocks, while the gather's striped
    explicit mask is a dense bias no kernel skips (2.0 to 4.0 times the
    all-to-all's time for causal GQA at 4k to 64k tokens on two RTX 3090s,
    docs/guides/multi-node.md), and it never sends more bytes. A call with no
    mask takes the exchange that sends fewer (`all_to_all_moves_less`).
    """
    tensor = _tensor_shards(query)
    heads, kv_heads = query.shape[-2], key.shape[-2]
    masked = causal or sliding_window is not None or mask is not None
    exchangeable = (heads % (tensor * shards) == 0
                    and query.shape[-3] % shards == 0 and key.shape[-3] % shards == 0)
    run = (exchanged_heads_attention
           if exchangeable and (masked or all_to_all_moves_less(heads, kv_heads, tensor, shards))
           else gathered_keys_attention)
    return run(kernel, query, key, value, shards, causal=causal,
               sliding_window=sliding_window, mask=mask, bias=bias, sinks=sinks,
               key_value_seq_lengths=key_value_seq_lengths)


def all_to_all_moves_less(heads: int, kv_heads: int, tensor: int, shards: int) -> bool:
    """Whether Ulysses sends fewer bytes per device than gathering the keys,
    for a call with no mask.

    Counted per device, per batch row, in units of S * D * (n - 1) / n for
    a sequence of S rows, head width D and n = `shards`, over one tensor
    shard's heads: H = heads / tensor, K = the key heads it holds.

    The all-to-all sends (n - 1) / n of four local blocks of S / n rows:
    the query and the output at H heads, the key and the value at K', the
    key heads repeated out to lcm(kv_heads, tensor * n) over tensor. That
    is 2 (H + K') / n.

    The gather receives the (n - 1) / n of the whole keys and values it
    does not hold, 2 K, the key heads repeated out to lcm(kv_heads, tensor)
    over tensor. A causal, windowed or masked call also stripes its S / n
    query rows and unstripes its output rows, which adds 2 H / n, so for
    such a call the all-to-all never sends more: K' / n is at most K.

    At H = 32, K = 8, n = 2 with one tensor shard, in those units the
    all-to-all sends 40, the gather 16: grouped-query attention with no mask
    gathers. The gather wins ties, as the path that takes every shape.
    Mask and bias blocks are left out; both paths carry them.
    """
    local_heads = heads // tensor
    repeated = math.lcm(kv_heads, tensor * shards) // tensor
    gathered = math.lcm(kv_heads, tensor) // tensor
    return 2 * (local_heads + repeated) / shards < 2 * gathered


def _entry(spec: P, dimension: int):
    """What `spec` names for `dimension`: a spec leaves off trailing whole ones."""
    return spec[dimension] if dimension < len(spec) else None


def _tensor_shards(query) -> int:
    """How many ways the mesh's tensor axes split the query heads."""
    mesh = jax.sharding.get_abstract_mesh()
    return math.prod(mesh.shape[axis]
                     for axis in mesh_axes(_entry(logical_spec(HEADS, query.shape), 2)))


def _four_dimensional(x):
    """A mask or a bias `[.., Q, K]` broadcastable to `[B, H, Q, K]`, as four
    dimensions; a broadcast dimension of 1 divides by no mesh axis, so the
    rule table leaves it whole."""
    return x.reshape((1,) * (4 - x.ndim) + x.shape)


def exchanged_heads_attention(kernel, query, key, value, shards: int, *, causal,
                              sliding_window, mask, bias, sinks, key_value_seq_lengths=None):
    """DeepSpeed Ulysses: trade a slice of the sequence for a slice of the heads.

    One all-to-all per operand over the sequence axis turns each shard's S/n
    rows of every head into all S rows of H/n heads; the kernel attends whole
    sequences and one more all-to-all puts the output back in rows (Jacobs et
    al. 2023, arXiv:2309.14509). Causality, windows, document masks and splash's
    block skipping work as on one device, causal work is equal without a
    reorder, autodiff transposes each all-to-all, and a shard holds S*K*D/n of
    each key and value against the gather's S*K*D.

    Heads split over tensor first and sequence within it, so H divides by
    tensor times sequence; key and value heads repeat only to the least common
    multiple of their count and that split. A mask or bias head dimension,
    learned sinks and key lengths split with what they belong to. Batch rows
    split over every batch axis that still divides them, in mesh order, and are
    attended alike where it does not.
    """
    mesh = jax.sharding.get_abstract_mesh()
    heads = query.shape[2]
    queries = logical_spec(HEADS, query.shape)
    tensor = mesh_axes(_entry(queries, 2))
    split = shards * math.prod(mesh.shape[axis] for axis in tensor)
    if heads % split or query.shape[1] % shards or key.shape[1] % shards:
        raise LayoutRefused(
            f"the all-to-all sequence exchange splits the {heads} query heads "
            f"{split} ways (tensor times sequence) and the query and key lengths "
            f"({query.shape[1]}, {key.shape[1]}) {shards} ways, and one of them does "
            "not divide. `gathered_keys_attention` takes any head count and key "
            "length; `sequence_parallel_attention` picks it for such a call.")
    kv_heads = math.lcm(key.shape[-2], split)
    key, value = repeat_kv_heads(key, kv_heads), repeat_kv_heads(value, kv_heads)
    keys = logical_spec(KV_HEADS, key.shape)

    # After the all-to-all a shard holds its tensor shard's heads split again
    # over the sequence axis; a mask, a bias or sinks with a head dimension
    # arrive split the same way, their query and key dimensions whole.
    exchanged_heads = (*tensor, SEQUENCE_AXIS)
    extras = {}
    for name, x in (('mask', mask), ('bias', bias)):
        if x is not None:
            x = _four_dimensional(x)
            extras[name] = (x, P(row_axes(x.shape[0]),
                                 exchanged_heads if x.shape[1] == heads else None))
    if sinks is not None:
        extras['sinks'] = (sinks, P(exchanged_heads))
    if key_value_seq_lengths is not None:
        extras['key_value_seq_lengths'] = (key_value_seq_lengths, logical_spec(
            ("activation_batch",), key_value_seq_lengths.shape))

    def local(query, key, value, *arrays):
        query, key, value = (jax.lax.all_to_all(x, SEQUENCE_AXIS, 2, 1, tiled=True)
                             for x in (query, key, value))
        out = kernel(query, key, value, causal=causal, sliding_window=sliding_window,
                     **dict(zip(extras, arrays, strict=True)))
        return jax.lax.all_to_all(out, SEQUENCE_AXIS, 1, 2, tiled=True)

    exchanged = manual_map(
        local, (queries, keys, keys, *(spec for _, spec in extras.values())), queries)
    return exchanged(query, key, value, *(x for x, _ in extras.values()))


def gathered_keys_attention(kernel, query, key, value, shards: int, *, causal,
                            sliding_window, mask, bias, sinks, key_value_seq_lengths=None):
    """Run `kernel` on each shard's slice of the queries against the whole
    keys and values, which every shard gathers over the sequence axis.

    Any head count and length, at the cost of every shard holding every key and
    value. Batch rows and heads split as `exchanged_heads_attention` splits
    them where the query heads divide by the tensor axis, and heads stay whole
    otherwise. A query length the shards do not divide is padded and the
    padding dropped; such a key length arrives whole. A causal, windowed or
    masked call stripes its queries for equal work and carries their positions
    in the mask, which the kernel reads in place of its causal flag, so the mask
    and the caller's rotary angles stay exact.
    """
    q_len, kv_len = query.shape[1], key.shape[1]
    # Each tensor shard's query heads beside the key heads they read.
    kv_heads = math.lcm(key.shape[-2], _tensor_shards(query))
    key, value = repeat_kv_heads(key, kv_heads), repeat_kv_heads(value, kv_heads)

    reordered = causal or sliding_window is not None or mask is not None
    if reordered:
        def rows_in_order(x: jax.Array | None) -> jax.Array | None:
            # A broadcast query row has the same value in either order.
            if x is not None and x.ndim >= 2 and x.shape[-2] == q_len:
                return stripe(x, shards, axis=-2)
            return x

        query = stripe(query, shards)
        mask, bias = rows_in_order(mask), rows_in_order(bias)
        structural = structural_mask(
            stripe(jnp.arange(q_len), shards, axis=0), kv_len, causal, sliding_window)
        if structural is not None:
            mask = structural if mask is None else jnp.logical_and(mask, structural)
    else:
        padding = -q_len % shards
        query = _pad_rows(query, padding)
        if bias is not None and bias.ndim >= 2 and bias.shape[-2] == q_len:
            bias = jnp.pad(bias, ((0, 0),) * (bias.ndim - 2) + ((0, padding), (0, 0)))

    # A key length the shards do not divide takes no sequence axis, and
    # arrives whole instead of gathered.
    queries, keys = logical_spec(HEADS, query.shape), logical_spec(KV_HEADS, key.shape)
    gathered = SEQUENCE_AXIS in mesh_axes(_entry(keys, 1))
    extras = {}
    for name, x in (('mask', mask), ('bias', bias)):
        if x is not None:
            x = _four_dimensional(x)
            extras[name] = (x, logical_spec(
                ("activation_batch", "activation_heads", "activation_length", None), x.shape))
    if sinks is not None:
        extras['sinks'] = (sinks, logical_spec(("activation_heads",), sinks.shape))
    if key_value_seq_lengths is not None:
        extras['key_value_seq_lengths'] = (key_value_seq_lengths, logical_spec(
            ("activation_batch",), key_value_seq_lengths.shape))

    def local(query, key, value, *arrays):
        if gathered:
            key, value = (jax.lax.all_gather(x, SEQUENCE_AXIS, axis=1, tiled=True)
                          for x in (key, value))
        return kernel(query, key, value, causal=False, sliding_window=None,
                      **dict(zip(extras, arrays, strict=True)))

    attended = manual_map(
        local, (queries, keys, keys, *(spec for _, spec in extras.values())), queries)
    out = attended(query, key, value, *(x for x, _ in extras.values()))
    return unstripe(out, shards) if reordered else out[:, :q_len]


CUDNN_DTYPES = (jnp.bfloat16, jnp.float16)
CUDNN_MAX_HEAD_DIM = 128


def cudnn_runs(query, softcap=None) -> bool:
    """Report whether cudnn's fused kernel takes this query.

    It needs a gpu of sm80 or later (cuDNN refuses bf16 and fp16 below it:
    "SDPA FP16/BF16 requires SM80"), one of cudnn's two dtypes, a head
    dimension it tiles, and no logit softcap, which no fused kernel applies. Only 'auto'
    asks this; an explicit 'cudnn' refuses by name instead.

    A run under `--xla_gpu_deterministic_ops` is excluded as well. Under
    that flag XLA's cudnn attention backward path crashes at execution time
    when one executable holds two structurally identical backward calls,
    which every multi-layer model has (openxla/xla#46500).

    Every fused kernel runs at the query's head width, values included:
    `widen_value_heads` pads a narrower value up to it. So this reads the
    query alone and holds for the whole call.
    """
    head_dim = query.shape[-1]
    return (jax.default_backend() == 'gpu' and bf16_dot_runs() and query.dtype in CUDNN_DTYPES
            and head_dim % 8 == 0 and head_dim <= CUDNN_MAX_HEAD_DIM
            and softcap is None and not deterministic_ops_requested())


def triton_runs(query, sliding_window=None, mask=None, bias=None) -> bool:
    """Report whether 'auto' sends a call cudnn would take to tokamax's
    Pallas-Triton kernel instead: tokamax is installed, the heads are at most
    64 wide, and the call has no window, mask (key lengths included) or bias.

    Those are the calls it measured faster on. On an RTX 4080, bf16, in the
    training step: SimpleDiT-B's attention (32 x 256 tokens, 12 heads of 64)
    5.62 to 4.33 ms and the step 73.1 to 71.5 ms; the 3-layer decoder's (16 x
    512 causal) 1.82 to 1.24 and the step 50.8 to 50.2. At 128-wide heads it
    is not: Qwen3-0.6B's (1 x 1024, 16 query and 8 key heads) 9.13 to 9.24 ms,
    and a 2048-token call with 8 heads over one key head 0.63 against 0.67
    ms. A window gains nothing (0.58 against 0.59 ms at 256 of 1024), so it
    stays on cuDNN with the masks.
    """
    return (query.shape[-1] <= 64 and sliding_window is None and mask is None and bias is None
            and importlib.util.find_spec('tokamax') is not None)


def triton_attention(query, key, value, causal: bool):
    """tokamax's Pallas-Triton flash kernel over `[B, S, H, D]` arrays, at
    tokamax's heuristic config, with jax.nn's default scale of 1/sqrt(D).

    It takes grouped key heads and any length itself. A Pallas call has no
    partitioning rule, so on a mesh it runs inside `manual_map` on each
    shard's rows and heads, with the key heads repeated until the tensor
    axes split them as they split the query's. tokamax is not a dependency
    of Dew (docs/installation.md says how to install it), so it is imported here.
    """
    try:
        tokamax = importlib.import_module('tokamax')
    except ImportError as e:
        raise ValueError("attention implementation 'triton' needs tokamax: "
                         "uv pip install 'tokamax>=0.0.15' (docs/installation.md)") from e

    def local(query, key, value):
        return tokamax.dot_product_attention(query, key, value, is_causal=causal,
                                             implementation='triton')

    if jax.sharding.get_abstract_mesh().empty:
        return local(query, key, value)
    kv_heads = math.lcm(key.shape[-2], _tensor_shards(query))
    key, value = repeat_kv_heads(key, kv_heads), repeat_kv_heads(value, kv_heads)
    queries, keys = logical_spec(HEADS, query.shape), logical_spec(KV_HEADS, key.shape)
    return manual_map(local, (queries, keys, keys), queries)(query, key, value)


VALUE_BLOCK = 256
"""Keys per block of `weighted_values`' fp32 sum on XLA:CPU."""


def weighted_values(equation, weights, value, *, precision=None) -> jax.Array:
    """The probability-value product, with the probabilities in the value's dtype.

    An fp32 softmax leaves fp32 probabilities, and `jnp.einsum` promotes to
    the wider operand, so multiplying them by a bf16 value would run both
    sides at the fp32 rate - forward and backward, since the product's
    cotangent is then fp32 too. Rounding the probabilities first is what
    `jax.nn.dot_product_attention` does on its own xla path
    (`probs.astype(key.dtype)`), so the two paths multiply alike; the softmax
    that produced them still reduced in fp32, which is where the precision
    was needed.

    `equation` is flax's value product, '...hqk,...khd->...qhd'. On XLA:CPU
    an fp32 product sums each block of VALUE_BLOCK keys apart and the
    blocks' sums after. YNNPACK, XLA:CPU's dot, sums up to about 1024 keys
    in one chain: fp32 attention over 512, 2048 and 8192 keys rounded 1.34,
    1.75 and 1.63 times as far from float64 as torch's SDPA does, and 0.98,
    0.96 and 0.87 times in blocks. Writing each block's product costs a
    forward 3% to 9% at 512 to 1024 keys on two threads (docs/performance.md).
    """
    if equation != '...hqk,...khd->...qhd':
        raise ValueError(f"weighted_values takes flax's value product, got {equation!r}")
    weights = weights.astype(value.dtype)
    if value.dtype != jnp.float32 or value.shape[-3] <= VALUE_BLOCK or jax.default_backend() != 'cpu':
        return jnp.einsum(equation, weights, value, precision=precision)
    return _blocked_values(weights, value, precision)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _blocked_values(weights, value, precision) -> jax.Array:
    """`weighted_values`' product summed in blocks of VALUE_BLOCK keys. Its
    gradients are the plain product's: neither sums over the keys, and the
    blocked form's own cost the backward an eighth more on two threads."""
    keys = value.shape[-3]
    blocks = -(-keys // VALUE_BLOCK)
    pad = blocks * VALUE_BLOCK - keys
    weights = jnp.pad(weights, [(0, 0)] * (weights.ndim - 1) + [(0, pad)])
    value = jnp.pad(value, [(0, 0)] * (value.ndim - 3) + [(0, pad), (0, 0), (0, 0)])
    weights = weights.reshape(*weights.shape[:-1], blocks, VALUE_BLOCK)
    value = value.reshape(*value.shape[:-3], blocks, VALUE_BLOCK, *value.shape[-2:])
    return jnp.einsum('...hqck,...ckhd->...qhdc', weights, value, precision=precision).sum(-1)


def _blocked_values_forward(weights, value, precision):
    return _blocked_values(weights, value, precision), (weights, value)


def _blocked_values_backward(precision, saved, cotangent):
    weights, value = saved
    return (jnp.einsum('...qhd,...khd->...hqk', cotangent, value, precision=precision),
            jnp.einsum('...hqk,...qhd->...khd', weights, cotangent, precision=precision))


_blocked_values.defvjp(_blocked_values_forward, _blocked_values_backward)


def softcapped_attention(query, key, value, softcap: float, dtype=None, precision=None,
                         force_fp32_for_softmax=True, mask=None, bias=None):
    """Attend with Gemma 2's tanh softcap on the logits, in plain XLA ops.

    The reference scales the logits, squashes them into (-softcap, softcap)
    as `softcap * tanh(logits / softcap)`, adds the mask and takes the
    softmax in fp32 (modeling_gemma2.py:192-208). No fused kernel has that
    tanh, so this is flax's reference attention with the cap between the
    scaling and the mask. Heads arrive repeated and the mask structural, as
    the reference path prepares them.
    """
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype
    logits = jnp.einsum('...qhd,...khd->...hqk',
                        query / jnp.sqrt(query.shape[-1]).astype(dtype), key,
                        precision=precision)
    logits = jnp.tanh(logits / softcap) * softcap
    if bias is not None:
        logits = logits + bias
    if mask is not None:
        logits = jnp.where(mask, logits, jnp.finfo(dtype).min)
    if force_fp32_for_softmax and dtype != jnp.float32:
        weights = jax.nn.softmax(logits.astype(at_least_fp32(dtype)))
    else:
        weights = jax.nn.softmax(logits).astype(dtype)
    return weighted_values('...hqk,...khd->...qhd', weights, value, precision=precision)


def scaled_dot_product_attention(query, key, value, dtype=None, precision=None,
                                 force_fp32_for_softmax=True, implementation='auto',
                                 causal=False, sliding_window=None, mask=None, bias=None,
                                 sinks=None, softcap=None, segment_ids=None,
                                 key_value_seq_lengths=None, dropout_rate=0.0,
                                 dropout_rng=None, deterministic=True):
    """Attend over [B, S, H, D] queries, keys and values: [B, S, H, Dv] out.

    `implementation` names the kernel (`AttentionImpl`, dispatched by
    `attention_kernel`); the parameter tree never depends on it. Keys and values
    may carry fewer heads (grouped-query attention) and the value a narrower
    head width (DeepSeek's latent attention, `widen_value_heads`), so kernel
    selection reads the query alone.

    `causal` restricts query i to keys 0..i, top-left aligned like jax's
    is_causal, and `sliding_window=w` to the w most recent of those, or
    without `causal` to the keys within w - 1 of i (`window_sides`); both read
    the row index, so a decode step against a cache passes `mask` instead.
    `bias` is additive and broadcasts to [B, H, Q, K] (T5's position table).
    `sinks` is one learned value-free logit per query head in the denominator.
    `softcap` is Gemma 2's tanh on the scaled logits. `segment_ids` `[B, S]`
    keeps each packed document to itself, 0 as padding (`document_mask`).
    `key_value_seq_lengths` `[B]` ends each row's keys, which cuDNN skips; a
    mask that only ends rows early costs cuDNN a dense bias per head, so pass
    lengths for right padding.

    A mesh sequence axis above one runs `sequence_parallel_attention`. Active
    dropout runs Flax's reference attention, independently per row, head and
    query, and refuses explicit fused kernels and sequence parallelism. The
    result carries the checkpoint name 'attention_output', so a remat policy
    can save it instead of replaying the kernel (`remat_block` in dit.py).
    """
    shards = sequence_shards()
    if not 0 <= dropout_rate < 1:
        raise ValueError(f"dropout_rate must be within [0, 1), got {dropout_rate}")
    if dropout_rate and not deterministic:
        out = _dropout_attention(
            query, key, value, dtype, precision, force_fp32_for_softmax, implementation,
            causal, sliding_window, mask, bias, sinks, softcap, segment_ids,
            key_value_seq_lengths, dropout_rate, dropout_rng, shards)
        return checkpoint_name(out, 'attention_output')
    if shards > 1 and segment_ids is not None:
        # The exchanges split an explicit mask along the sequence and have no
        # split for the ids, so the documents travel as the mask they stand
        # for, with causality and the window folded in beside them.
        mask = combined_attention_mask(query.shape[-3], key.shape[-3], causal, sliding_window,
                                       with_documents(mask, segment_ids))
        causal, sliding_window, segment_ids = False, None, None
        implementation = kernel_for_materialized_mask(
            implementation, query, dtype=dtype, precision=precision,
            force_fp32_for_softmax=force_fp32_for_softmax)
    kernel = functools.partial(
        attention_kernel, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, implementation=implementation,
        softcap=softcap)
    if shards > 1:
        out = sequence_parallel_attention(
            kernel, query, key, value, shards, causal=causal,
            sliding_window=sliding_window, mask=mask, bias=bias, sinks=sinks,
            key_value_seq_lengths=key_value_seq_lengths)
    else:
        out = kernel(query, key, value, causal=causal, sliding_window=sliding_window,
                     mask=mask, bias=bias, sinks=sinks, segment_ids=segment_ids,
                     key_value_seq_lengths=key_value_seq_lengths)
    return checkpoint_name(out, 'attention_output')


def _dropout_attention(query, key, value, dtype, precision, force_fp32_for_softmax, implementation,
                       causal, sliding_window, mask, bias, sinks, softcap, segment_ids,
                       key_value_seq_lengths, dropout_rate, dropout_rng, shards):
    """Apply one reference probability draw, retaining visibility and projection precision."""
    if implementation not in ('auto', 'reference', 'xla'):
        raise ValueError(
            f"{implementation} attention cannot apply probability dropout; use reference or auto"
        )
    if shards > 1:
        raise ValueError("attention probability dropout requires sequence_shards=1")
    if sinks is not None or softcap is not None:
        raise ValueError("attention probability dropout does not support sinks or softcap")
    if dropout_rng is None:
        raise ValueError("training attention dropout requires dropout_rng")
    mask = combined_attention_mask(
        query.shape[-3], key.shape[-3], causal, sliding_window,
        mask if segment_ids is None else with_documents(mask, segment_ids))
    if key_value_seq_lengths is not None:
        mask = with_key_lengths(mask, key_value_seq_lengths, key.shape[-3])
    return nn.dot_product_attention(
        query, repeat_kv_heads(key, query.shape[-2]), repeat_kv_heads(value, query.shape[-2]),
        bias=bias, mask=mask, dtype=dtype, dropout_rate=dropout_rate,
        dropout_rng=dropout_rng, broadcast_dropout=False, deterministic=False,
        force_fp32_for_softmax=(force_fp32_for_softmax
                                and at_least_fp32(dtype or query.dtype) == jnp.float32),
        qk_attn_weights_einsum=functools.partial(jnp.einsum, precision=precision),
        attn_weights_value_einsum=functools.partial(weighted_values, precision=precision))


def with_documents(mask, segment_ids):
    """`mask` narrowed to each packed document's own keys, `[B, 1, S, S]`."""
    inside = document_mask(segment_ids)[:, None]
    return inside if mask is None else jnp.logical_and(mask, inside)


def refuse_reference_only_arguments(implementation, query, dtype, precision,
                                    force_fp32_for_softmax):
    """Raise for the arguments only the reference path reads.

    A fused kernel accumulates the logits and runs the softmax in fp32 in
    the inputs' own dtype, whatever the caller asked for. Each message names
    the implementation and the reference path, so a caller can pick one.
    """
    if precision_names(precision) & {'HIGH', 'HIGHEST'}:
        raise ValueError(
            f"attention implementation '{implementation}' cannot honor "
            f"precision={precision}: fused attention accumulates the logits and "
            "runs the softmax in fp32 regardless. Leave precision at DEFAULT, or "
            "use the reference implementation (attention_impl 'reference').")
    if not force_fp32_for_softmax:
        raise ValueError(
            f"attention implementation '{implementation}' cannot honor "
            "force_fp32_for_softmax=False: fused attention runs the softmax in "
            "fp32 regardless. Leave it True, or use the reference implementation "
            "(attention_impl 'reference').")
    if dtype is not None and jnp.dtype(dtype) != query.dtype:
        raise ValueError(
            f"attention implementation '{implementation}' cannot honor "
            f"dtype={dtype}: fused attention computes in the inputs' dtype "
            f"({query.dtype}). Pass dtype=None or leave the inputs in that dtype, or use "
            "the reference implementation (attention_impl 'reference').")


FOLDED_PAIRS = 16
"""Most query positions times key heads `folded_attention` takes."""


def folds(query, key, bias, mask, causal, sliding_window) -> bool:
    """Whether an xla call goes to `folded_attention`: on a GPU, a call of
    few query positions over keys of the query's heads, masked by key
    lengths at most, such as a decode step's grouped query
    (`CausalSelfAttention._decode_attention`) where cudnn does not run."""
    return (jax.default_backend() == 'gpu' and query.shape[-2] == key.shape[-2]
            and query.shape[-3] * key.shape[-2] <= FOLDED_PAIRS
            and bias is None and mask is None and not causal and sliding_window is None)


def folded_attention(query, key, value, key_value_seq_lengths):
    """jax.nn's xla attention for query `[B, T, N, D]` over keys and values
    `[B, S, N, D]` of the same heads, with every query head against every
    (key, head) pair.

    XLA's GPU products take each head's keys apart, so jax.nn's attention
    transposed the whole cache, keys and values, at every decode step where
    cudnn does not run: a tenth of Qwen3.5-0.8B's decode program at 128
    rows, its 256-wide heads. Here both products read the keys and values
    as `[B, S * N, D]`, their own layout, and the cross-head products are
    thrown away: N times the multiplications, of a step bound by reading
    the cache. The arithmetic is jax.nn's otherwise (fp32 logits of the
    operands, the scale after, an fp32 softmax of logits masked to -0.7
    times the largest fp32, probabilities in the value's dtype). Six layers
    of Qwen3.5's decode took 1.21 against 2.48 ms at 128 rows and 0.33
    against 0.47 at 32 on an RTX 4080 (docs/performance.md).
    """
    batch, positions, heads, width = query.shape
    keys = key.shape[-3]
    wide = jnp.promote_types(query.dtype, jnp.float32)
    precision = bf16_operand_precision(query.dtype)
    logits = jnp.einsum('bxd,byd->bxy', query.reshape(batch, positions * heads, width),
                        key.reshape(batch, keys * heads, width), precision=precision,
                        preferred_element_type=wide)
    logits = jnp.diagonal(logits.reshape(batch, positions, heads, keys, heads), axis1=2, axis2=4)
    logits = logits * jnp.asarray(1 / math.sqrt(width), wide)  # [B, T, S, N]
    if key_value_seq_lengths is not None:
        valid = jnp.arange(keys)[None, None, :, None] < key_value_seq_lengths[:, None, None, None]
        logits = jnp.where(valid, logits, jnp.asarray(-0.7 * jnp.finfo(jnp.float32).max, wide))
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=2).astype(value.dtype)
    # Each query head's probabilities over its own head's keys, zeros elsewhere.
    spread = jnp.moveaxis(probs, 3, 2)[..., None] * jnp.eye(heads, dtype=probs.dtype)[:, None, :]
    out = jnp.einsum('bxy,byd->bxd', spread.reshape(batch, positions * heads, keys * heads),
                     value.reshape(batch, keys * heads, value.shape[-1]))
    return out.reshape(batch, positions, heads, value.shape[-1])


def fused_attention(query, key, value, bias, mask, causal, sliding_window, implementation, *,
                    softcap, sinks, segment_ids, key_value_seq_lengths):
    """Run the fused kernel `implementation` names, at the query's head width.

    A narrower value rides in padded and its columns come back out, at no
    runtime branch. 'cudnn' refuses here what its kernel cannot take: sinks, a
    softcap, deterministic ops, and dtypes and head widths outside its tiles.
    `key_value_seq_lengths` reaches cudnn and xla as lengths and 'tpu' as the
    mask it means.
    """
    v_head_dim = value.shape[-1]
    if v_head_dim != query.shape[-1]:
        value = widen_value_heads(query, value)

    if implementation == 'cudnn':
        if sinks is not None:
            raise ValueError("attention implementation 'cudnn' cannot honor sinks")
        if softcap is not None:
            raise ValueError(
                f"attention implementation 'cudnn' cannot apply an attention logit "
                f"softcap of {softcap}: the fused kernel has no tanh between its "
                "scaling and its softmax. Use attention_impl 'xla' or the reference "
                "implementation (attention_impl 'reference').")
        if deterministic_ops_requested():
            raise ValueError(
                "attention implementation 'cudnn' cannot run under "
                "--xla_gpu_deterministic_ops: on this JAX and XLA its backward pass "
                "is unusable, because an executable holding two identical fused "
                "attention backward calls, which every multi-layer model has, fails "
                "at execution time (openxla/xla#46500). Use attention_impl 'xla', "
                "which is deterministic, or drop the flag.")
        if query.dtype not in CUDNN_DTYPES:
            raise ValueError(
                "cudnn attention needs bf16 or fp16 inputs, the query is "
                f"{query.dtype}. Set dtype bfloat16, or attention_impl 'xla' to "
                "keep this precision.")
        head_dim = query.shape[-1]
        if head_dim % 8 or head_dim > CUDNN_MAX_HEAD_DIM:
            raise ValueError(
                f"cudnn attention needs a head dimension that is a multiple of 8 "
                f"and at most {CUDNN_MAX_HEAD_DIM}, got {head_dim}; use attention_impl "
                "'xla' for this shape.")
        if sliding_window is not None and not causal:
            raise ValueError(
                "cudnn attention windows only the keys before a query, and this "
                "bidirectional call windows both sides; use attention_impl 'xla'.")
        out = cudnn_attention(query, key, value, bias, mask, causal, sliding_window,
                              key_value_seq_lengths)
    elif implementation == 'triton':
        if any(x is not None for x in (sinks, softcap, bias, mask, sliding_window, key_value_seq_lengths)):
            raise ValueError(
                "attention implementation 'triton' takes no sinks, softcap, bias, mask, "
                "window or key lengths; use attention_impl 'cudnn' or 'xla' for this call.")
        out = triton_attention(query, key, value, causal)
    elif (implementation == 'xla' and folds(query, key, bias, mask, causal, sliding_window)
          and key_value_seq_lengths is not None and decode_attention.fits(query, key)):
        out = decode_attention.attend(query, key, value, key_value_seq_lengths)
    elif implementation == 'xla' and folds(query, key, bias, mask, causal, sliding_window):
        out = folded_attention(query, key, value, key_value_seq_lengths)
    elif implementation == 'xla':
        # A window of (l, r) keeps the l keys before the query and the r after
        # it on both the xla and the cudnn path, the sides `window_sides` counts.
        out = jax.nn.dot_product_attention(
            query, key, value, bias=bias, mask=mask, is_causal=causal,
            key_value_seq_lengths=key_value_seq_lengths,
            local_window_size=None if sliding_window is None else window_sides(causal, sliding_window),
            implementation='xla')
    else:  # 'tpu'
        if key_value_seq_lengths is not None:
            mask = with_key_lengths(mask, key_value_seq_lengths, key.shape[-3])
        out = tpu_attention(query, key, value, bias, mask, causal, sliding_window,
                            softcap=softcap, sinks=sinks, segment_ids=segment_ids,
                            interpret=jax.default_backend() != 'tpu')
    return out if v_head_dim == out.shape[-1] else out[..., :v_head_dim]


_FORWARD_MODE = contextvars.ContextVar("forward_mode_attention", default=False)


@contextlib.contextmanager
def forward_mode_attention():
    """Trace the attention calls inside for `jax.jvp`.

    cuDNN's and the TPU's fused kernels are reverse-mode only (`jax.custom_vjp`).
    Inside this context each such call keeps the fused kernel for its value and
    takes its tangent from the reference path's JVP, which materializes the
    `[B, H, Q, K]` probabilities once. Consistency models (sCM, rCM) run their
    JVP inside it and train outside it, on the fused backward.
    """
    token = _FORWARD_MODE.set(True)
    try:
        yield
    finally:
        _FORWARD_MODE.reset(token)


def attention_kernel(query, key, value, dtype=None, precision=None,
                     force_fp32_for_softmax=True, implementation='auto',
                     causal=False, sliding_window=None, mask=None, bias=None, sinks=None,
                     softcap=None, segment_ids=None, key_value_seq_lengths=None):
    """Dispatch one whole-sequence attention call to the kernel it names,
    differentiable in forward mode under `forward_mode_attention`.

    'auto' resolves here, once, against this call's shapes and backend, and
    `_attention_kernel` runs the call on the kernel it resolved to.
    """
    if sliding_window is not None and sliding_window < 1:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")
    if sinks is not None and softcap is not None:
        raise ValueError(
            "attention sinks and a logit softcap have no reference that "
            "combines them, so no path takes both")
    lengths = (None if key_value_seq_lengths is None
               else jnp.asarray(key_value_seq_lengths, jnp.int32))
    masked = mask if lengths is None else with_key_lengths(mask, lengths, key.shape[-3])
    resolved = resolve_implementation(
        implementation, query, key, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap, sinks=sinks,
        causal=causal, sliding_window=sliding_window, mask=masked, bias=bias)
    call = functools.partial(
        _attention_kernel, dtype=dtype, precision=precision, force_fp32_for_softmax=force_fp32_for_softmax,
        causal=causal, sliding_window=sliding_window, mask=mask, masked=masked, softcap=softcap,
        segment_ids=segment_ids, lengths=lengths)
    if not _FORWARD_MODE.get() or resolved not in ('cudnn', 'triton', 'tpu'):
        return call(query, key, value, bias=bias, sinks=sinks, implementation=resolved)

    def fused(query, key, value, bias, sinks):
        return call(query, key, value, bias=bias, sinks=sinks, implementation=resolved)

    def reference(query, key, value, bias, sinks):
        return call(query, key, value, bias=bias, sinks=sinks, implementation='reference')

    return forward_differentiable(fused, reference, query, key, value, bias, sinks)


def forward_differentiable(value, tangent, *arrays):
    """`value(*arrays)`, whose JVP is `tangent`'s: a kernel that defines only
    its reverse-mode derivative takes forward mode through a reference
    computing the same function."""
    @jax.custom_jvp
    def attend(*arrays):
        return value(*arrays)

    @attend.defjvp
    def rule(primals, tangents):
        return attend(*primals), jax.jvp(tangent, primals, tangents)[1]

    return attend(*arrays)


def _attention_kernel(query, key, value, *, bias, sinks, implementation, dtype, precision,
                      force_fp32_for_softmax, causal, sliding_window, mask, masked, softcap,
                      segment_ids, lengths):
    """Run one whole-sequence attention call on the kernel `attention_kernel` resolved.

    `masked` is `mask` with the key `lengths` in it. Splash takes sinks, a
    softcap and packed segment ids itself; elsewhere segment ids become the
    document mask, and sinks and a softcap run their own XLA paths, since
    `jax.nn.dot_product_attention` has neither. The rest goes to
    `fused_attention` once the arguments it cannot honour have raised. Key
    lengths reach cudnn and xla as they are and every other path as the mask
    they mean, which splash cannot describe, so 'auto' never picks 'tpu' for
    them.
    """
    if segment_ids is not None and implementation != 'tpu':
        mask = with_documents(mask, segment_ids)
        masked = with_documents(masked, segment_ids)
        # cuDNN would take the document mask as an additive bias
        # (`kernel_for_materialized_mask`), and the triton route takes no
        # mask, so a packed call runs on xla.
        if implementation in ('cudnn', 'triton'):
            implementation = 'xla'
    if sinks is not None and implementation in ('reference', 'xla'):
        mask = combined_attention_mask(
            query.shape[-3], key.shape[-3], causal, sliding_window, masked)
        return attention_with_sinks(
            query, key, value, sinks, mask=mask, bias=bias, dtype=dtype,
            precision=precision, force_fp32_for_softmax=force_fp32_for_softmax)
    if implementation == 'reference' or (softcap is not None and implementation == 'xla'):
        heads = query.shape[-2]
        key = repeat_kv_heads(key, heads)
        value = repeat_kv_heads(value, heads)
        mask = combined_attention_mask(
            query.shape[-3], key.shape[-3], causal, sliding_window, masked)
        if softcap is not None:
            return softcapped_attention(
                query, key, value, softcap, dtype=dtype, precision=precision,
                force_fp32_for_softmax=force_fp32_for_softmax, mask=mask, bias=bias)
        # flax pins a forced softmax to float32 itself; a wider compute
        # dtype already runs it at least that wide.
        return nn.dot_product_attention(
            query, key, value, bias=bias, mask=mask, dtype=dtype, broadcast_dropout=False,
            dropout_rng=None, precision=None,
            force_fp32_for_softmax=(force_fp32_for_softmax
                                    and at_least_fp32(dtype or query.dtype) == jnp.float32),
            deterministic=True,
            # flax takes the precision through these two or through
            # `precision`, never both, and its own value product would
            # multiply the fp32 probabilities by the bf16 value.
            qk_attn_weights_einsum=functools.partial(jnp.einsum, precision=precision),
            attn_weights_value_einsum=functools.partial(weighted_values, precision=precision))

    refuse_reference_only_arguments(
        implementation, query, dtype, precision, force_fp32_for_softmax)
    return fused_attention(query, key, value, bias, mask, causal, sliding_window,
                           implementation, softcap=softcap, sinks=sinks,
                           segment_ids=segment_ids, key_value_seq_lengths=lengths)


def reference_only(query, dtype, precision, force_fp32_for_softmax) -> bool:
    """Whether a call asks for arithmetic only the reference path performs.

    The three are what `refuse_reference_only_arguments` raises for: a
    matmul precision above DEFAULT, a softmax outside fp32, and a compute
    dtype other than the inputs'. A fused kernel runs none of them.
    """
    return (bool(precision_names(precision) & {'HIGH', 'HIGHEST'})
            or not force_fp32_for_softmax
            or (dtype is not None and jnp.dtype(dtype) != query.dtype))


def _xla_kernel_narrows(query) -> bool:
    """Whether jax.nn's xla attention would compute this call below its
    query's precision, so 'auto' and 'xla' take the reference path, which
    computes in the query's dtype (`dew.nn.precision.at_least_fp32`).

    It rounds a float64 query's softmax to float32
    (`_dot_product_attention_core`: "Softmax and it is always carried out in
    fp32"), and names the BF16_BF16_F32 algorithm for a bf16 one, which a GPU
    older than sm80 rejects at run time, past jax's own fallback."""
    return query.dtype == jnp.float64 or (
        query.dtype == jnp.bfloat16 and jax.default_backend() == 'gpu' and not bf16_dot_runs())


def _xla_kernel_chains(query, key) -> bool:
    """Whether jax.nn's xla attention would sum this call's fp32 value
    product over more than VALUE_BLOCK keys in one chain, as YNNPACK,
    XLA:CPU's dot, does, so 'auto' and 'xla' take the reference path, whose
    `weighted_values` sums it in blocks."""
    return query.dtype == jnp.float32 and key.shape[-3] > VALUE_BLOCK and jax.default_backend() == 'cpu'


def resolve_implementation(implementation, query, key, *, dtype=None, precision=None,
                           force_fp32_for_softmax=True, softcap=None, sinks=None, causal=False,
                           sliding_window=None, mask=None, bias=None) -> str:
    """The concrete kernel an `AttentionImpl` names for this call.

    Only 'auto' chooses, against the call's shapes and this machine's
    backend (both 'auto' and 'xla' take the reference path where jax.nn's
    xla kernel would narrow the call, `_xla_kernel_narrows`): the reference
    path when the call asks for arithmetic no fused kernel performs
    (`reference_only`), else cudnn where `cudnn_runs` and the call has no
    sinks and no bidirectional window (triton in its place where
    `triton_runs`), the tpu kernel where
    `tpu_runs`, and xla anywhere else. Any other name is returned as it is,
    so an explicit kernel still refuses what it cannot honour by name.
    """
    if implementation not in ('auto', 'reference', 'xla', 'cudnn', 'triton', 'tpu'):
        raise ValueError(f"Unknown attention implementation: {implementation}")
    if implementation in ('auto', 'xla') and (_xla_kernel_narrows(query) or _xla_kernel_chains(query, key)):
        return 'reference'
    if implementation != 'auto':
        return implementation
    if reference_only(query, dtype, precision, force_fp32_for_softmax):
        return 'reference'
    # cuDNN keeps a window behind the query only (jax.nn.dot_product_attention
    # refuses a right window without the causal mask), so a bidirectional
    # window goes past it.
    if sinks is None and cudnn_runs(query, softcap) and (causal or sliding_window is None):
        return 'triton' if triton_runs(query, sliding_window, mask, bias) else 'cudnn'
    if tpu_runs(query, key, causal=causal, sliding_window=sliding_window, mask=mask, bias=bias):
        return 'tpu'
    return 'xla'


def kernel_for_materialized_mask(implementation: str, query, *, dtype=None, precision=None,
                                 force_fp32_for_softmax=True) -> str:
    """Send a call that built an explicit mask to the xla kernel.

    cuDNN has no mask argument: jax hands it a bool mask as an additive bias of
    -2**41 in the compute dtype (`combine_bias_and_mask` in
    jax/_src/cudnn/fused_attention_stablehlo.py), and `check_is_flash_attention`
    then refuses an odd length while training. The xla kernel masks by
    exclusion with the same fp32 softmax, at 83.6 ms and 5.80 GiB a step against
    cuDNN's fixed window at 75.8 ms and 4.99 GiB
    (docs/concepts/language_models.md). 'auto' first takes the reference path
    for arithmetic only it performs (`reference_only`).
    """
    if implementation == 'auto' and reference_only(query, dtype, precision,
                                                   force_fp32_for_softmax):
        return 'reference'
    if implementation in ('auto', 'cudnn', 'xla') and _xla_kernel_narrows(query):
        return 'reference'
    return 'xla' if implementation in ('auto', 'cudnn') else implementation


def _blocks(x, block: int, blocks: int):
    """`[B, S, ...]` padded at the end to `blocks * block` rows, as `[B, blocks, block, ...]`."""
    x = jnp.pad(x, [(0, 0), (0, blocks * block - x.shape[1])] + [(0, 0)] * (x.ndim - 2))
    return x.reshape(x.shape[0], blocks, block, *x.shape[2:])


def _banded(x, before=None):
    """Each block of `[B, n, W, ...]` behind the block before it: `[B, n, 2W, ...]`.

    `before` `[B, 1, W, ...]` is the block before the first: under a sequence
    axis, the previous shard's last `W` rows. None is zeros, whose rows are
    negative and so outside every query's causal reach.
    """
    if before is None:
        before = jnp.zeros_like(x[:, :1])
    return jnp.concatenate([jnp.concatenate([before, x[:, :-1]], axis=1), x], axis=2)


def local_attention(query, key, value, *, window: int | None = None, chunk: int | None = None,
                    positions=None, segment_ids=None, valid=None, dtype=None, precision=None,
                    force_fp32_for_softmax=True, implementation='auto', sinks=None,
                    softcap=None):
    """Causal self-attention in which each query reads only nearby keys, in
    memory linear in the sequence.

    `window=w` is sliding attention: a query reads itself and the w - 1 keys
    before it (Mistral, Gemma, MaxText's `sliding_window_size`). `chunk=c` is
    chunked local attention: a query reads the keys at or before it in its
    chunk, `position // c` (Llama 4's `attention_chunk_size`). Exactly one is
    set. `positions` `[S]` or `[B, S]` places the chunks (None is the row
    index), `segment_ids` keeps packed documents apart (0 is padding) and
    `valid` `[B, S]` drops the keys it marks False.

    No `[S, S]` array is built above two spans. A window cudnn (no sinks or
    packing) or splash takes as a flag runs there, skipping blocks outside it.
    Chunks that start at row multiples fold into the batch. Everything else
    runs banded: query blocks of the span against their own block and the one
    before under a `[W, 2W]` mask, so the logits are `[S, 2W]` per head. Banding
    pads to whole spans, so at most two spans take the dense call.
    """
    if (window is None) == (chunk is None):
        raise ValueError("local attention takes exactly one of window and chunk")
    span = window if window is not None else chunk
    assert span is not None
    if span < 1:
        raise ValueError(f"a local span is a positive number of keys, got {span}")
    length = query.shape[1]
    if key.shape[1] != length:
        raise ValueError(
            f"local attention is self-attention: {length} queries against "
            f"{key.shape[1]} keys")
    whole = functools.partial(
        scaled_dot_product_attention, query, key, value, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, sinks=sinks, softcap=softcap)
    resolved = resolve_implementation(
        implementation, query, key, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap, sinks=sinks,
        causal=True, sliding_window=window)
    # A mask built here is at most `[S, 2W]` per head, so cudnn takes it as
    # its additive bias where cudnn runs (docs/performance.md: 16384 packed
    # tokens at window 4096 in 0.60 GiB of temporaries on an L4, where xla
    # asked for 10.0 GiB at 8192).
    masked = 'cudnn' if resolved == 'cudnn' else kernel_for_materialized_mask(
        implementation, query, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax)
    if window is not None and valid is None:
        # An explicit 'tpu' at a length splash cannot tile would reach the
        # flash kernel, whose default 128-row blocks must divide the length
        # (flash_attention.py:113-128, 1707-1715), so it would raise there.
        on_splash = resolved == 'tpu' and length % SPLASH_LANES == 0
        if on_splash or (segment_ids is None and (
                (resolved in ('cudnn', 'tpu') and sinks is None) or length <= 2 * span)):
            return whole(implementation=implementation, causal=True, sliding_window=window,
                         segment_ids=segment_ids)
    shards = sequence_shards()
    # Under a sequence axis each shard's first block reads the previous
    # shard's last `span` rows, which only one neighbour holds when a shard's
    # slice is at least that long. A wider span takes the dense call, whose
    # mask the sequence exchange carries.
    halo = length % shards == 0 and length // shards >= span
    if length <= 2 * span or not halo:
        places = jnp.arange(length) if positions is None else positions
        mask = (None if chunk is None or (positions is None and length <= chunk)
                else chunk_mask(places, places, chunk))
        if segment_ids is not None:
            mask = with_documents(mask, segment_ids)
        if valid is not None:
            live = jnp.asarray(valid, bool)[:, None, None, :]
            mask = live if mask is None else mask & live
        if mask is None:
            return whole(implementation=implementation, causal=True, sliding_window=window)
        return whole(implementation=masked, mask=combined_attention_mask(
            length, length, causal=True, sliding_window=window, mask=mask))
    kernel = functools.partial(
        attention_kernel, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap)
    if shards > 1:
        out = _local_over_sequence(
            kernel, query, key, value, shards, window=window, chunk=chunk, positions=positions,
            segment_ids=segment_ids, valid=valid, sinks=sinks, implementation=implementation,
            masked=masked)
    else:
        out = _local_blocks(functools.partial(kernel, sinks=sinks), query, key, value,
                            window=window, chunk=chunk, positions=positions,
                            segment_ids=segment_ids, valid=valid,
                            implementation=implementation, masked=masked)
    return checkpoint_name(out, 'attention_output')


def _local_over_sequence(kernel, query, key, value, shards: int, *, window, chunk, positions,
                         segment_ids, valid, sinks, implementation, masked):
    """`_local_blocks` over a sequence the mesh's sequence axis splits, in a
    `shard_map`: each shard computes its own rows, and one `ppermute` hands its
    first block the previous shard's last `span` keys, values, positions,
    segment ids and validity (zeros on the first shard). That halo is all that
    crosses the interconnect; chunks starting at the shard boundaries need none.
    Heads split over the tensor axis as the exchanges split them."""
    batch, length = query.shape[:2]
    span = window if window is not None else chunk
    assert span is not None
    queries = logical_spec(HEADS, query.shape)
    kv_heads = math.lcm(key.shape[2], _tensor_shards(query))
    key, value = repeat_kv_heads(key, kv_heads), repeat_kv_heads(value, kv_heads)
    keys = logical_spec(KV_HEADS, key.shape)
    fields = {}
    if positions is not None:
        fields['positions'] = jnp.broadcast_to(jnp.asarray(positions), (batch, length))
    if segment_ids is not None:
        fields['segment_ids'] = segment_ids
    if valid is not None:
        fields['valid'] = jnp.asarray(valid, bool)
    rows = logical_spec(("activation_batch", "activation_length"), (batch, length))
    specs = [rows] * len(fields)
    if sinks is not None:
        fields['sinks'] = sinks
        specs.append(logical_spec(("activation_heads",), sinks.shape))
    names = tuple(fields)
    local_length = length // shards
    # Chunks that start at every shard's first row stay inside their shard.
    aligned = chunk is not None and positions is None and local_length % chunk == 0

    def local(query, key, value, *arrays):
        given = dict(zip(names, arrays, strict=True))
        forward = [(shard, shard + 1) for shard in range(shards - 1)]

        def last_rows(x):
            if x is None:
                return None
            return jax.lax.ppermute(x[:, local_length - span:], SEQUENCE_AXIS, forward)

        before = None if aligned else tuple(
            last_rows(x) for x in (key, value, given.get('positions'),
                                   given.get('segment_ids'), given.get('valid')))
        return _local_blocks(
            functools.partial(kernel, sinks=given.get('sinks')), query, key, value,
            window=window, chunk=chunk, positions=given.get('positions'),
            segment_ids=given.get('segment_ids'), valid=given.get('valid'),
            implementation=implementation, masked=masked,
            offset=jax.lax.axis_index(SEQUENCE_AXIS) * local_length, before=before)

    attended = manual_map(local, (queries, keys, keys, *specs), queries)
    return attended(query, key, value, *fields.values())


def _local_blocks(kernel, query, key, value, *, window, chunk, positions, segment_ids, valid,
                  implementation, masked, offset=0, before=None):
    """`local_attention`'s blocked computation over the `[B, S, ...]` rows it
    is handed, which start at row `offset` of the sequence.

    `before` holds what the first rows read before them: the previous `span`
    rows' keys, values, positions, segment ids and validity, each None where
    the call has none. None is the start of the sequence, where nothing comes
    before, and chunks start at row multiples from there."""
    batch, length = query.shape[0], query.shape[1]
    span = window if window is not None else chunk
    assert span is not None
    blocks = -(-length // span)
    flags_only = segment_ids is None and valid is None

    def folded(x, width: int):
        return x.reshape(batch * blocks, width, *x.shape[3:])

    def row_blocks(x):
        return _blocks(jnp.broadcast_to(jnp.asarray(x), (batch, length)), span, blocks)

    def preceding(index):
        # The halo field at `index`, as the one block before the first.
        return None if before is None else before[index][:, None]

    if chunk is not None and positions is None and before is None:
        mask = None
        if not flags_only:
            keep = jnp.ones((batch, blocks, span, span), bool)
            if segment_ids is not None:
                segments = row_blocks(segment_ids)
                keep = keep & ((segments[..., :, None] == segments[..., None, :])
                               & (segments[..., :, None] != 0))
            if valid is not None:
                keep = keep & row_blocks(jnp.asarray(valid, bool))[..., None, :]
            mask = keep.reshape(batch * blocks, 1, span, span)
            implementation = masked
        out = kernel(*(folded(_blocks(x, span, blocks), span) for x in (query, key, value)),
                     implementation=implementation, causal=True, mask=mask)
    else:
        rows = offset + jnp.arange(blocks * span).reshape(1, blocks, span)
        key_rows = jnp.concatenate([rows - span, rows], axis=-1)
        keep = ((key_rows[..., None, :] >= 0)
                & (key_rows[..., None, :] <= rows[..., :, None]))
        if window is not None:
            keep = keep & (rows[..., :, None] - key_rows[..., None, :] < window)
        keep = jnp.broadcast_to(keep, (batch, blocks, span, 2 * span))
        if chunk is not None and positions is None:
            keep = keep & (rows[..., :, None] // chunk == key_rows[..., None, :] // chunk)
        elif chunk is not None:
            places = row_blocks(positions) // chunk
            earlier = preceding(2)
            keep = keep & (places[..., :, None] == _banded(
                places, None if earlier is None else earlier // chunk)[..., None, :])
        if segment_ids is not None:
            segments = row_blocks(segment_ids)
            keep = keep & ((segments[..., :, None] == _banded(segments, preceding(3))[..., None, :])
                           & (segments[..., :, None] != 0))
        if valid is not None:
            keep = keep & _banded(row_blocks(jnp.asarray(valid, bool)),
                                  preceding(4))[..., None, :]
        out = kernel(folded(_blocks(query, span, blocks), span),
                     folded(_banded(_blocks(key, span, blocks), preceding(0)), 2 * span),
                     folded(_banded(_blocks(value, span, blocks), preceding(1)), 2 * span),
                     implementation=masked,
                     mask=keep.reshape(batch * blocks, 1, span, 2 * span))
    return out.reshape(batch, blocks * span, *out.shape[2:])[:, :length]


# Splash's tile size. At 1024 the forward kernel holds 4 MiB of fp32 logits
# plus its operands. On a TPU v6e, Qwen3-0.6B's attention (8 x 1024 tokens,
# 16 query heads over 8, 128 wide, causal) took 1.28 ms forward and backward
# at 1024 with the fused backward, against 1.81 at 512 with separate dq and
# dkv kernels and 1.42 at 512 fused, with the same errors against fp32
# (docs/performance.md). `splash_block_sizes` narrows this to a divisor of
# each sequence, as the mask blocking needs.
SPLASH_BLOCK = 1024
# The kernel tiles the key axis by lanes: the compute block must be a whole
# number of them (`{bkv_compute=} must be a multiple of {NUM_LANES=}`,
# splash_attention_kernel.py:970-971) and the mask blocking must divide both
# sequences (splash_attention_mask_info.py:565-571), so a length that is not
# a multiple of this has no legal block size and never reaches splash.
SPLASH_LANES = 128
# The two input dtypes a TPU matmul takes; fp16 has no MXU path. The kernel
# accumulates in fp32 whatever comes in: the logits carry
# preferred_element_type=float32, and the running max, sum and output stay
# fp32 to the last block, which is the reference softmax.
SPLASH_DTYPES = (jnp.bfloat16, jnp.float32)
# How much of an explicit boolean mask splash will carry. The array is read
# on the host while the executable is built and its unresolved blocks are
# stored dense inside it, so this bounds host and executable bytes, not
# device memory that grows with the batch. 4 Mi cells is a 16-head 512x512.
SPLASH_DENSE_MASK_CELLS = 1 << 22

# The shortest sequence 'auto' hands splash. tools/qualify_splash.py on one
# v6e (jax 0.11.1, bf16, 16 query heads over 8 key heads of width 128, 16384
# causal tokens a step), a training step: splash 3.24 ms against XLA's 2.62
# at 256 keys, 2.57 against 4.64 at 512, 28.9 against 144.2 at 16384; width
# 64 crosses at the same length. Softcap, sinks and packed ids cost splash
# at most 16% more from 512 to 16384.
SPLASH_MIN_LENGTH = 512


def tpu_runs(query, key, *, causal=False, sliding_window=None, mask=None, bias=None) -> bool:
    """Report whether splash takes this call.

    It needs a tpu backend, one of the two TPU matmul dtypes, sequence lengths
    its mask blocking tiles and at least `SPLASH_MIN_LENGTH`, a mask it can
    describe, and no bias; softcap, sinks and segment ids are kernel arguments.
    Only 'auto' asks this, after `cudnn_runs`; an explicit 'tpu' sends what this
    turns down to the pallas flash kernel. The head width is not read: the
    kernel pads the value width to lanes (splash_attention_kernel.py:732) and
    the query width is a contraction Mosaic pads.

    A call under an automatic sequence axis above one is turned down: that is
    `gathered_keys_attention`, and a pallas call with no partitioning rule
    would gather the queries back. `exchanged_heads_attention` calls the kernel
    inside a `shard_map` where `sequence_shards` is 1, so splash takes those.
    """
    if jax.default_backend() != 'tpu' or query.dtype not in SPLASH_DTYPES:
        return False
    if sequence_shards() > 1 or bias is not None:
        return False
    q_len, kv_len = query.shape[-3], key.shape[-3]
    if q_len % SPLASH_LANES or kv_len % SPLASH_LANES or min(q_len, kv_len) < SPLASH_MIN_LENGTH:
        return False
    return splash_mask_descriptor(
        q_len, kv_len, query.shape[-2], causal, sliding_window, mask) is not None


def tpu_attention(query, key, value, bias, mask, causal, sliding_window, *,
                  softcap, sinks, segment_ids, interpret: bool):
    """Attend on pallas: splash where its descriptor covers the call, flash
    everywhere else.

    Splash is block-sparse: its mask is a descriptor built with the executable,
    so a causal or windowed sequence costs its live blocks, and packed segment
    ids are a kernel argument. It has no bias argument and needs an explicit
    mask readable on the host. Flash takes both as one additive [B, H, Q, K]
    array over the whole rectangle, so it serves exactly:

    - an additive `bias` (T5's relative position table);
    - a traced `mask` (a cache decode mask, sequence parallelism's striped
      mask), one past `SPLASH_DENSE_MASK_CELLS`, or one that varies by row;
    - a length that is not a multiple of `SPLASH_LANES`.

    Flash has no softcap and no sinks, so such a call raises. `interpret` runs
    splash under pallas's interpreter, which is how the parity tests run its
    arithmetic off a TPU; it is far slower than XLA, so only an explicit 'tpu'
    uses it. The flash kernel has no interpreter and runs only on a TPU.
    """
    q_len, kv_len = query.shape[-3], key.shape[-3]
    descriptor = None
    if bias is None and not (q_len % SPLASH_LANES or kv_len % SPLASH_LANES):
        descriptor = splash_mask_descriptor(
            q_len, kv_len, query.shape[-2], causal, sliding_window, mask)
    if descriptor is None and (softcap is not None or sinks is not None):
        raise ValueError(
            "attention implementation 'tpu' takes a softcap and sinks only on "
            "the splash kernel, and this call has a bias, a mask splash cannot "
            "describe or a length that is not a multiple of "
            f"{SPLASH_LANES}. Use attention_impl 'xla'.")

    def attend(query, key, value, bias, sinks):
        if descriptor is None:
            return pallas_flash_attention(query, key, value, bias, mask, causal,
                                          sliding_window, segment_ids)
        return splash_attention(query, key, value, descriptor, softcap=softcap, sinks=sinks,
                                segment_ids=segment_ids, interpret=interpret)

    if jnp.finfo(query.dtype).bits == 16 and not interpret:
        # Mosaic multiplies 16-bit operands at the default precision only.
        # Their products are exact and accumulate in fp32; splash's fp32 P·V
        # takes one bf16 pass, as at any default-precision run
        # (`at_default_precision`).
        return at_default_precision(attend)(query, key, value, bias, sinks)
    return attend(query, key, value, bias, sinks)


def splash_attention(query, key, value, descriptor, *, softcap, sinks, segment_ids,
                     interpret: bool):
    """Run the splash kernel over [B, S, H, D] arrays under `descriptor`.

    The kernel takes one [H, S, D] example at a time, so the batch rides a
    vmap. It applies no scale, so 1/sqrt(d) goes onto the query in the query's
    dtype, where flax's reference puts it, which makes the two agree exactly in
    fp32. Grouped key heads stay grouped (splash reads q_heads % kv_heads).
    `softcap` is applied to the scaled logits before the mask, as in
    `softcapped_attention`; `sinks` `[H]` seed each head's running maximum and
    denominator, as in `attention_with_sinks`, with their gradient. The kernel
    lets equal segment ids attend, so padding (id 0) reads other padding where
    `document_mask` reads nothing; no document's query reads padding either way.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel

    kernel = splash_attention_kernel.make_splash_mha(
        descriptor, block_sizes=splash_block_sizes(query.shape[-3], key.shape[-3]),
        head_shards=1, q_seq_shards=1, attn_logits_soft_cap=softcap, interpret=interpret)
    scale = jnp.asarray(1.0 / math.sqrt(query.shape[-1]), query.dtype)
    segments = (None if segment_ids is None
                else splash_attention_kernel.SegmentIds(segment_ids, segment_ids))

    def example(q, k, v, ids):
        return kernel(q, k, v, segment_ids=ids, sinks=sinks)

    attended = jax.vmap(example)(jnp.moveaxis(query, -2, -3) * scale,
                                 jnp.moveaxis(key, -2, -3), jnp.moveaxis(value, -2, -3),
                                 segments)
    if isinstance(attended, tuple):
        # Only a save_residuals=True kernel returns a pair, and this one is
        # built without it; the check narrows the kernel's union return type.
        raise ValueError("splash returned residuals this path has no consumer for")
    return jnp.moveaxis(attended, -3, -2)


def splash_block_sizes(q_len: int, kv_len: int):
    """Narrow `SPLASH_BLOCK` to a divisor of each sequence, forward and backward.

    The greatest common divisor divides its sequence, as the mask blocking
    requires, and stays a multiple of `SPLASH_LANES`, as the key compute block
    requires. The backward runs as one kernel for dq, dk and dv, which reads
    each logits block once where separate dq and dkv kernels read it twice;
    its blocks are set because a kernel built without them raises "Need to
    specify backward blocks." at its first gradient.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel

    block_q, block_kv = math.gcd(SPLASH_BLOCK, q_len), math.gcd(SPLASH_BLOCK, kv_len)
    return splash_attention_kernel.BlockSizes(
        block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv,
        block_q_dkv=block_q, block_kv_dkv=block_kv, block_kv_dkv_compute=block_kv,
        use_fused_bwd_kernel=True)


def splash_mask_descriptor(q_len: int, kv_len: int, heads: int, causal: bool,
                           sliding_window: int | None, mask):
    """Build splash's mask for one call, or None when splash cannot describe it.

    CausalMask and LocalMask are comparisons the kernel evaluates per block, so
    emptied blocks leave the grid and nothing Q*K-sized is stored. Dew's window
    holds the causal flag's side already (`window_sides`), so it replaces the
    flag. An explicit boolean mask
    is ANDed in as dense blocks, so only a concrete, small one is taken.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

    shape = (q_len, kv_len)
    if sliding_window is not None:
        structural = splash_attention_mask.LocalMask(shape, window_sides(causal, sliding_window), 0)
    elif causal:
        structural = splash_attention_mask.CausalMask(shape)
    else:
        structural = splash_attention_mask.FullMask(shape)
    if mask is None:
        return splash_attention_mask.MultiHeadMask([structural] * heads)
    per_head = splash_dense_mask(mask, q_len, kv_len, heads)
    if per_head is None:
        return None
    dense = [splash_attention_mask.NumpyMask(rows) for rows in per_head]
    if isinstance(structural, splash_attention_mask.FullMask):
        return splash_attention_mask.MultiHeadMask(dense)
    return splash_attention_mask.MultiHeadMask(
        [splash_attention_mask.LogicalAnd(structural, head) for head in dense])


def splash_dense_mask(mask, q_len: int, kv_len: int, heads: int):
    """Return an explicit mask as one host [Q, K] boolean array per query head.

    None when splash cannot carry it: a traced mask (the descriptor is built
    with the executable; a decode mask over slots is one), a batch axis wider
    than one (splash has none), or one past `SPLASH_DENSE_MASK_CELLS`, whose
    dense blocks would cost more than the additive-bias path.
    """
    import numpy as np
    from jax.core import Tracer

    if isinstance(mask, Tracer):
        return None
    dense = np.asarray(mask, bool)
    while dense.ndim > 3 and dense.shape[0] == 1:
        dense = dense[0]
    if dense.ndim == 2:
        dense = dense[None]
    if dense.ndim != 3 or dense.shape[-2:] != (q_len, kv_len):
        return None
    if dense.shape[0] not in (1, heads) or dense.size > SPLASH_DENSE_MASK_CELLS:
        return None
    return [dense[head % dense.shape[0]] for head in range(heads)]


def pallas_flash_attention(query, key, value, bias, mask, causal, sliding_window, segment_ids):
    """Run the pallas TPU flash kernel, with every mask but the documents as
    an additive bias.

    A window and an explicit mask become one [B, H, Q, K] array of zeros and the
    dtype's minimum; causality is a flag and packed documents its `segment_ids`,
    with splash's padding rule. That rectangle is what splash avoids, so only the
    calls `tpu_attention` names come here. 1/sqrt(d) goes onto the query and
    `sm_scale` stays 1, because the kernel adds `ab` before it multiplies by
    `sm_scale` (flash_attention.py:402-409), which would scale T5's bias too.
    """
    from jax.experimental.pallas.ops.tpu.flash_attention import SegmentIds, flash_attention

    heads = query.shape[-2]
    key = repeat_kv_heads(key, heads)
    value = repeat_kv_heads(value, heads)
    # pallas wants [B, H, S, D]
    q = jnp.moveaxis(query, -2, -3) * jnp.asarray(1.0 / math.sqrt(query.shape[-1]), query.dtype)
    k = jnp.moveaxis(key, -2, -3)
    v = jnp.moveaxis(value, -2, -3)
    combined = None
    if bias is not None:
        combined = jnp.broadcast_to(bias.astype(q.dtype),
                                    (q.shape[0], q.shape[1], q.shape[2], k.shape[2]))
    # The kernel takes causality as its flag, so only a window becomes a band.
    band = None if sliding_window is None else structural_mask(
        jnp.arange(query.shape[-3]), key.shape[-3], causal, sliding_window)
    if band is not None:
        mask = band if mask is None else jnp.logical_and(mask, band)
    if mask is not None:
        seated = jnp.broadcast_to(
            jnp.where(mask, 0, jnp.finfo(q.dtype).min).astype(q.dtype),
            (q.shape[0], q.shape[1], q.shape[2], k.shape[2]))
        combined = seated if combined is None else combined + seated
    segments = None if segment_ids is None else SegmentIds(segment_ids, segment_ids)
    return jnp.moveaxis(
        flash_attention(q, k, v, ab=combined, segment_ids=segments, causal=causal), -3, -2)


@logical_axes({
    ("to_q",): ("embed", "heads", "head_dim"),
    ("to_k",): ("embed", "heads", "head_dim"),
    ("to_v",): ("embed", "heads", "head_dim"),
    ("to_out_0",): ("heads", "head_dim", "embed"),
})
class NormalAttention(nn.Module):
    """Attend over a `[B, S, C]` or `[B, H, W, C]` input with multiple heads.

    `freqs_cis` rotates the queries and keys of a self-attention call;
    `rotary_freqs` gives the pair, and None leaves them unrotated.
    `use_bias` is the projections' bias; `qkv_bias`, when given, the
    query, key and value projections' alone, as timm's ViT attention
    (U-ViT's) has them without one and its output projection with one.
    `mask`, boolean and broadcasting to `[B, H, S, S_context]`, keeps the keys
    each query may read.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_bias: bool = True
    qkv_bias: bool | None = None
    force_fp32_for_softmax: bool = True
    qk_norm: bool = False  # RMSNorm on q/k per head (SD3-style bf16 logit safety)
    attention_impl: str = "auto"  # an AttentionImpl

    def setup(self):
        dense = functools.partial(
            nn.DenseGeneral,
            features=[self.heads, self.dim_head],
            axis=-1,
            precision=self.precision,
            use_bias=self.use_bias if self.qkv_bias is None else self.qkv_bias,
            dtype=self.dtype
        )
        self.query = dense(name="to_q")
        self.key = dense(name="to_k")
        self.value = dense(name="to_v")

        if self.qk_norm:
            self.q_norm = RMSNorm(epsilon=1e-6, dtype=self.dtype, name="q_norm")
            self.k_norm = RMSNorm(epsilon=1e-6, dtype=self.dtype, name="k_norm")

        self.proj_attn = nn.DenseGeneral(
            self.query_dim,
            axis=(-2, -1),
            precision=self.precision,
            use_bias=self.use_bias,
            dtype=self.dtype,
            name="to_out_0",
        )

    @nn.compact
    def __call__(self, x, context=None, freqs_cis=None, mask=None):
        orig_x_shape = x.shape
        if len(x.shape) == 4:
            x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        context = x if context is None else context
        if len(context.shape) == 4:
            context = context.reshape(
                (context.shape[0], context.shape[1] * context.shape[2], context.shape[3]))
        # [B, S, heads, head_dim], column-parallel under a tensor axis; a
        # context's positions split over a sequence axis where its link pays
        # for it (`split_positions`).
        query = constrain(self.query(x), HEADS)
        key, value = split_positions(
            context, (self.heads, self.dim_head),
            lambda tokens: (constrain(self.key(tokens), HEADS), constrain(self.value(tokens), HEADS)))
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        if freqs_cis is not None:
            freqs_cos, freqs_sin = freqs_cis
            query = apply_rotary(query, freqs_cos, freqs_sin)
            key = apply_rotary(key, freqs_cos, freqs_sin)

        hidden_states = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            implementation=self.attention_impl, mask=mask,
        )
        proj = self.proj_attn(constrain(hidden_states, HEADS))
        return proj.reshape(orig_x_shape)


class FlaxGEGLU(nn.Module):
    """Project to twice the hidden width, then gate one half by the other.

    The gate is GELU, the gated linear unit of Shazeer 2020
    (https://arxiv.org/abs/2002.05202). The hidden width is four times
    `dim`.
    """

    dim: int
    dtype: Dtype | None = jnp.float32
    precision: PrecisionLike = jax.lax.Precision.DEFAULT
    approximate: bool = True

    def setup(self):
        inner_dim = self.dim * 4
        self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype, precision=self.precision)

    def __call__(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=-1)
        return hidden_linear * nn.gelu(hidden_gelu, approximate=self.approximate)


@logical_axes({("net_0", "proj"): ("embed", "mlp"), ("net_2",): ("mlp", "embed")})
class FlaxFeedForward(nn.Module):
    """Run GEGLU then a linear layer back to `dim`, as diffusers does.

    The checkpoint keys `net_0` and `net_2` are the indices the reference's
    Sequential gives the two layers.
    """

    dim: int
    dtype: Dtype | None = jnp.float32
    precision: PrecisionLike = jax.lax.Precision.DEFAULT
    dropout: float = 0.0
    approximate_gelu: bool = True

    def setup(self):
        self.net_0 = FlaxGEGLU(
            self.dim, dtype=self.dtype, precision=self.precision, approximate=self.approximate_gelu
        )
        self.net_2 = nn.Dense(self.dim, dtype=self.dtype, precision=self.precision)
        self.dropout_layer = nn.Dropout(self.dropout)

    def __call__(self, hidden_states, *, train: bool = False):
        hidden_states = self.net_0(hidden_states)
        if self.dropout:
            hidden_states = self.dropout_layer(hidden_states, deterministic=not train)
        return self.net_2(hidden_states)


class BasicTransformerBlock(nn.Module):
    """Run self-attention, cross-attention over `context`, then feed-forward.

    Each is pre-normed with a residual. `use_cross_only` drops the
    self-attention. `only_pure_attention` runs the cross-attention alone,
    with no norm and no residual, which the UNets' stages do by default.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_bias: bool = True
    use_cross_only:bool = False
    only_pure_attention:bool = False
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-4
    attention_impl: str = "auto"  # an AttentionImpl

    def setup(self):
        attention = functools.partial(
            NormalAttention,
            query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            precision=self.precision,
            use_bias=self.use_bias,
            dtype=self.dtype,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            attention_impl=self.attention_impl,
        )
        self.attention1 = attention(name='Attention1')
        self.attention2 = attention(name='Attention2')

        self.ff = FlaxFeedForward(dim=self.query_dim, dtype=self.dtype, precision=self.precision)
        self.norm1 = RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)
        self.norm2 = RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)
        self.norm3 = RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)

    def __call__(self, hidden_states, context=None):
        if self.only_pure_attention:
            return self.attention2(hidden_states, context)

        if not self.use_cross_only:
            hidden_states = hidden_states + self.attention1(self.norm1(hidden_states))
        hidden_states = hidden_states + self.attention2(self.norm2(hidden_states), context)
        return hidden_states + self.ff(self.norm3(hidden_states))


@dataclasses.dataclass(frozen=True)
class Stage:
    """One resolution stage's attention in a UNet; a stage without is `None`.

    Every field is a `TransformerBlock` setting. The head width is the stage's
    channel count over `heads`, which the unet knows and a config does not.
    `dew.registry.from_record` builds one from a record (`{"heads": 8}` from a
    command line), so a misspelled field raises at the build boundary.
    `dtype` and `precision` of None mean the model's. The softmax runs in
    fp32 by default, as every fused kernel computes it.
    """

    heads: int
    use_projection: bool = False
    use_self_and_cross: bool = True
    only_pure_attention: bool = True
    force_fp32_for_softmax: bool = True
    norm_inputs: bool = True
    explicitly_add_residual: bool = True
    norm_epsilon: float = 1e-4
    dtype: Dtype | None = None
    precision: PrecisionLike = None


def stage_attention(stage: Stage, channels: int, attention_impl: str, dtype: Dtype | None,
                    precision: PrecisionLike, name: str) -> "TransformerBlock":
    """Build the block a `Stage` describes, at the stage's channel count.

    A stage that names no dtype or precision takes the model's, and a model
    that names no dtype attends in fp32.
    """
    return TransformerBlock(
        heads=stage.heads, dim_head=channels // stage.heads,
        dtype=stage.dtype or dtype or jnp.float32,
        attention_impl=attention_impl, use_projection=stage.use_projection,
        use_self_and_cross=stage.use_self_and_cross,
        precision=stage.precision or precision,
        only_pure_attention=stage.only_pure_attention,
        force_fp32_for_softmax=stage.force_fp32_for_softmax,
        norm_inputs=stage.norm_inputs, explicitly_add_residual=stage.explicitly_add_residual,
        norm_epsilon=stage.norm_epsilon,
        name=name)


class TransformerBlock(nn.Module):
    """Run a `BasicTransformerBlock`, optionally at its own head width.

    `use_projection` puts a dense projection into and out of
    `heads * dim_head` around the block; without it the block runs at the
    input width.
    """
    heads: int = 4
    dim_head: int = 32
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_projection: bool = False
    use_self_and_cross:bool = True
    only_pure_attention:bool = False
    force_fp32_for_softmax: bool = True
    attention_impl: str = "auto"  # an AttentionImpl
    norm_inputs: bool = True
    explicitly_add_residual: bool = True
    norm_epsilon: float = 1e-4

    @nn.compact
    def __call__(self, x, context=None):
        channels = x.shape[-1]
        if self.norm_inputs:
            x = RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)(x)

        def project(features: int, name: str):
            return nn.Dense(features=features, use_bias=False, precision=self.precision,
                            dtype=self.dtype, name=name)

        if self.use_projection:
            inner_dim = self.heads * self.dim_head
            hidden = project(inner_dim, 'project_in')(x)
        else:
            inner_dim, hidden = channels, x
        hidden = BasicTransformerBlock(
            query_dim=inner_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            name='Attention',
            precision=self.precision,
            use_bias=False,
            dtype=self.dtype,
            use_cross_only=(not self.use_self_and_cross),
            only_pure_attention=self.only_pure_attention,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            attention_impl=self.attention_impl,
            norm_epsilon=self.norm_epsilon
        )(hidden, hidden if context is None else context)
        if self.use_projection:
            hidden = project(channels, 'project_out')(hidden)

        if self.only_pure_attention or self.explicitly_add_residual:
            hidden = x + hidden
        return hidden
