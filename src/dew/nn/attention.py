"""Attend over queries, keys and values, cache them, and block them up.

One kernel path serves every attention module here. The KV cache and the
UNet transformer blocks sit beside it; the blocks come from diffusers'
attention_flax.py.
"""

import dataclasses
import functools
import math
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import canonicalize_dtype, promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as P

from dew.telemetry.devices import deterministic_ops_requested

from .attention_sinks import attention_with_sinks
from .kv_cache import Append, KVCache, KVStore, filled_slots
from .precision import precision_names
from .sharding import SEQUENCE_AXIS, STAGE_AXIS, TENSOR_AXIS, logical_axes, row_axes, sequence_shards

AttentionImpl = Literal["auto", "reference", "xla", "cudnn", "tpu"]
"""Names which kernel an attention call runs.

Every layer that carries the choice spells it the same way: a `ModelConfig`
field, a module's `attention_impl`, and `scaled_dot_product_attention`'s
`implementation`.

'reference' is the portable einsum and softmax, and the only path that reads
dtype, precision and force_fp32_for_softmax. 'xla' and 'cudnn' are
`jax.nn.dot_product_attention`'s own two. 'tpu' is the pallas splash kernel,
with the older pallas flash kernel behind it for the calls splash's mask
descriptor cannot carry. 'auto', every module's default, resolves per trace
(`resolve_implementation`): cudnn where its kernel runs, tpu where splash's
does, the reference path where the call asks for arithmetic only it
honours, and xla anywhere else.
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


def document_mask(segment_ids) -> jax.Array:
    """Keep each packed document to itself: `[B, S, S]` boolean.

    Two positions see each other when they carry the same segment id.
    Segment 0 is padding, which sees nothing and is seen by nothing. A
    packed batch carries its structure here, so the caller ANDs causality
    in rather than handing the kernels their causal flag.
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
    narrows that to the most recent keys. Both read the row index, the way
    the fused kernels take them as flags. Unset stays unset, so a caller
    that distinguishes no mask from an all-true one keeps doing so.
    """
    if causal or sliding_window is not None:
        structural = causal_attention_mask(
            jnp.arange(query_length), key_length, sliding_window)
        mask = structural if mask is None else jnp.logical_and(mask, structural)
    return mask


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

    The norms reduce in fp32 for bf16 stability. Differentiated as written,
    the backward pass would keep two fp32 copies of the activations per
    norm. Recomputing them keeps only the bf16 input. On one SimpleDiT step
    this saved 2.35 GiB.

    The policy saves nothing, so the backward pass holds the arguments: the
    input, the weight and the bias. Recomputing the reductions rather than
    naming them leaves the residuals to the policy that recomputes this
    block (`causal_transformer.RESIDUALS`).

    The wrapped function takes arrays first and static arguments last. A
    caller passes its parameters as plain arrays, so no flax lifting sits
    between the checkpoint and the module. The forward values are unchanged;
    only the buffer assignment moves.
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
        y = x.astype(jnp.float32)
        y = y * jax.lax.rsqrt(jnp.mean(jnp.square(y), axis=-1, keepdims=True) + epsilon)
    else:
        # The reference rounds every step to the input dtype. XLA fuses the
        # chain and carries fp32 between the ops, so each rounding is explicit.
        y = _rounded(x * _rounded(jax.lax.rsqrt(_rounded(
            _rounded(jnp.mean(_rounded(jnp.square(x), x.dtype), axis=-1, keepdims=True), x.dtype)
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
        return _rounded(_rounded(y, dtype) * weight, dtype).astype(dtype)
    return (y * weight).astype(dtype)


def _rounded(x, dtype):
    """Round `x` to `dtype`'s precision, keeping its own dtype."""
    bits = jnp.finfo(dtype)
    return jax.lax.reduce_precision(x, exponent_bits=bits.nexp, mantissa_bits=bits.nmant)


@functools.partial(normalized_in_fp32, static_argnums=(3, 4))
def layer_normalized(x, scale, bias, epsilon: float, dtype):
    """Normalize `x` over its last axis, op for op as flax computes it.

    E[x] and E[x^2] are reduced in fp32 and the variance comes from the
    pair, clipped at zero. The weight folds into the inverse deviation
    before it meets the centered activations. `scale` and `bias` of None
    are the affine-free norm.
    """
    y = x.astype(jnp.float32)
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
    fp32 = x.astype(jnp.float32)
    inverse = jax.lax.rsqrt(jnp.mean(jnp.square(fp32), axis=-1, keepdims=True) + eps)
    return x * inverse.astype(x.dtype)


class RMSNorm(nn.Module):
    """Normalize the last axis by its root mean square, reducing in fp32 by default.

    `scale_offset` stores the weight as Gemma does, (1 + w), and flips the
    initializer to zeros. The identity is the starting point either way, so
    a Gemma checkpoint's stored weights land unchanged.

    The families differ in where the scale meets the activation dtype. Gemma
    multiplies in fp32 and casts the product (modeling_gemma3.py:147-150).
    Llama and Qwen3 cast first and multiply by the weight
    (modeling_qwen3.py:61-64), which `scale_after_cast` reproduces. The two
    agree at fp32 and differ under bf16.

    timm's `RmsNorm2d` (layers/fast_norm.py `rms_norm2d`) computes
    x * rsqrt(mean(x^2) + eps) * w in the input dtype.
    `fp32_statistics=False` with `scale_after_cast` is that order.

    The reduction runs under `normalized_in_fp32`, so the backward pass
    keeps this call's input in its own dtype, not an fp32 copy.
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

    The parameter tree, the names and the fp32 arithmetic are flax's, so a
    checkpoint written by either loads into the other and the forward values
    match bit for bit. What differs is the backward pass: flax keeps an fp32
    copy of the input and the fp32 centered activations, and this keeps the
    input as it arrived, through `normalized_in_fp32`.

    The fields are the subset the tree sets. A norm over other axes, under a
    mask, across a pmapped axis, or on the slower exact variance is flax's
    to serve.
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


@dataclasses.dataclass(frozen=True)
class RopeScaling:
    """Scales rotary frequencies by Llama 3.1's ramp, under the reference's names.

    `_compute_llama3_parameters` (transformers modeling_rope_utils.py:580)
    divides a frequency by `factor` when its wavelength exceeds
    original_max_position_embeddings / low_freq_factor, and leaves it alone
    below original_max_position_embeddings / high_freq_factor. In between it
    interpolates linearly on
    (original_max_position_embeddings / wavelength - low_freq_factor)
    / (high_freq_factor - low_freq_factor).

    `rope_type` is the record's discriminator and only 'llama3' is this
    ramp; YaRN is a mixer kind's own value.
    """

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int
    rope_type: str = 'llama3'

    def __post_init__(self):
        if self.rope_type != 'llama3':
            raise ValueError(
                f"rope_scaling applies the llama3 ramp, got rope_type {self.rope_type!r}")
        if self.factor < 1.0:
            raise ValueError(f"rope_scaling factor is at least 1, got {self.factor}")
        if self.high_freq_factor <= self.low_freq_factor:
            raise ValueError(
                f"rope_scaling needs high_freq_factor above low_freq_factor, got "
                f"{self.high_freq_factor} and {self.low_freq_factor}")
        if self.original_max_position_embeddings < 1:
            raise ValueError(
                "rope_scaling original_max_position_embeddings is the pretraining "
                f"context, got {self.original_max_position_embeddings}")

    def apply(self, inv_freq):
        """Return the scaled inverse frequencies, the reference's arithmetic in fp32."""
        old_context_len = float(self.original_max_position_embeddings)
        wavelen = 2 * math.pi / inv_freq
        divided = jnp.where(wavelen > old_context_len / self.low_freq_factor,
                            inv_freq / self.factor, inv_freq)
        smooth = ((old_context_len / wavelen - self.low_freq_factor)
                  / (self.high_freq_factor - self.low_freq_factor))
        smoothed = (1 - smooth) * divided / self.factor + smooth * divided
        medium = jnp.logical_and(wavelen >= old_context_len / self.high_freq_factor,
                                 wavelen <= old_context_len / self.low_freq_factor)
        return jnp.where(medium, smoothed, divided)


def rotary_freqs(positions, head_dim: int, theta: float, rot_dim: int | None = None,
                 partial_rotary_type: str = 'proportional',
                 rope_scaling: RopeScaling | None = None):
    """Return cos and sin of the rotary angles at absolute `positions`: [P, pairs].

    `positions` may be [P] for one sequence, or [B, P] for a packed batch
    whose documents each restart at 0; the angle axes line up with the
    trailing [B, S] either way. The angles are computed in fp32, so a token
    rotates the same in a prefill and in a single decode step.

    `rot_dim` narrows the rotation to the first rot_dim dimensions.
    `partial_rotary_type` names which published convention that is, because
    the two rotate different angles:

    - 'proportional' (Gemma 4, modeling_rope_utils.py
      `_compute_proportional_rope_parameters`): the exponents run over the
      full head_dim, `theta ** (2i / head_dim)` for the rot_dim // 2 rotated
      pairs. The rest keep frequency zero, where the rotation is the
      identity. The output is head_dim // 2 wide.
    - 'default' (Qwen3.5, modeling_qwen3_5.py:117-124
      `Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters`): the rope
      is rot_dim-dimensional, `theta ** (2i / rot_dim)`, and the output is
      rot_dim // 2 wide. `apply_rotary` passes the trailing dimensions
      through, the reference's `q_rot, q_pass` split.

    With rot_dim None both are the full rotation and the type is moot.
    `rope_scaling` is Llama 3.1's ramp over the base frequencies. It applies
    before a proportional rope pads its zero-frequency tail, as the
    reference's `dim = head_dim * partial_rotary_factor` does.
    """
    if partial_rotary_type not in ('proportional', 'default'):
        raise ValueError(
            "partial_rotary_type names the convention of a partial rotary, "
            f"'proportional' or 'default', got {partial_rotary_type!r}")
    pairs = head_dim // 2 if rot_dim is None else rot_dim // 2
    divisor = head_dim if rot_dim is None or partial_rotary_type == 'proportional' else rot_dim
    inv_freq = 1.0 / (theta ** (jnp.arange(0, 2 * pairs, 2, dtype=jnp.float32) / divisor))
    if rope_scaling is not None:
        inv_freq = rope_scaling.apply(inv_freq)
    if rot_dim is not None and partial_rotary_type == 'proportional':
        padding = head_dim // 2 - pairs
        inv_freq = jnp.concatenate([inv_freq, jnp.zeros((padding,), jnp.float32)])
    positions = jnp.asarray(positions, jnp.float32)
    if positions.ndim == 1:
        angles = positions[:, None] * inv_freq[None, :]
    else:
        angles = positions[:, :, None] * inv_freq[None, None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rotary(x, freqs_cos, freqs_sin, scale: float | None = None):
    """Rotate [B, S, H, D] heads, rotate-half convention as in the HF decoders.

    The freqs are [S, pairs] for one sequence, or [B, S, pairs] when a packed
    batch restarts positions per document. Freqs narrower than D // 2 rotate
    the first 2 * pairs dimensions and pass the rest through, which is the
    sliced partial rotary of `rotary_freqs(partial_rotary_type='default')`.
    `scale` multiplies the whole head inside the fp32 arithmetic, so a
    query's attention scale narrows once, with the product.
    """
    cos = jnp.concatenate([freqs_cos, freqs_cos], axis=-1)
    sin = jnp.concatenate([freqs_sin, freqs_sin], axis=-1)
    if cos.ndim == 3:
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
    else:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    fp32 = x.astype(jnp.float32)
    rotated_dims = cos.shape[-1]
    fp32, passed = fp32[..., :rotated_dims], fp32[..., rotated_dims:]
    x1, x2 = jnp.split(fp32, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    out = fp32 * cos + rotated * sin
    if passed.shape[-1]:
        out = jnp.concatenate([out, passed], axis=-1)
    return (out if scale is None else out * scale).astype(x.dtype)


def _cache_positions(module: nn.Module, batch: int, length: int, capacity: int, valid):
    """Allocate compact cache slots for real tokens, independently per row."""
    valid = jnp.ones((batch, length), bool) if valid is None else jnp.asarray(valid, bool)
    if valid.shape != (batch, length):
        raise ValueError(f"cache validity must be {(batch, length)}, got {valid.shape}")
    allocated = module.has_variable("cache", "cache_index")
    index = module.variable("cache", "cache_index", jnp.zeros, (batch,), jnp.int32)
    cached_valid = module.variable("cache", "cache_valid", jnp.zeros, (batch, capacity), bool)
    positions = index.value[:, None] + jnp.cumsum(valid, axis=1, dtype=jnp.int32) - 1
    positions = jnp.where(valid, positions, -1)
    if allocated:
        index.value = index.value + jnp.sum(valid, axis=1, dtype=jnp.int32)
        cached_valid.value = filled_slots(index.value, capacity)
    return positions, allocated


def open_kv_cache(module: nn.Module, key, max_seq_len, *, valid=None, layout: KVCache = KVCache()):
    """Fixed-size K/V with a cursor and cached validity for each batch row.

    Returns [B, S] compact slot positions and a writer. Invalid tokens have
    position -1 and do not advance the cursor. The allocation-only call leaves
    every row empty. The writer returns full cache arrays, including unused
    slots which the caller excludes with cache_valid. `layout` chooses the
    storage behind the slots, dense or paged, full or quantized
    (`dew.nn.kv_cache`); the slots and the cursor are the same for all.
    """
    shards = sequence_shards()
    if shards > 1:
        raise ValueError(
            f"decoding is not supported under a sequence axis of {shards}: the KV "
            "cache holds whole sequences. Generate on a mesh with sequence=1.")
    if max_seq_len is None:
        raise ValueError("decoding needs max_seq_len for its fixed-capacity cache")
    batch, length, heads, head_dim = key.shape
    if valid is None and length > max_seq_len:
        raise ValueError(f"{length} tokens do not fit a KV cache of {max_seq_len}.")
    store = KVStore.open(module, layout, batch, max_seq_len, heads, head_dim, key.dtype)
    positions, allocated = _cache_positions(module, batch, length, max_seq_len, valid)
    return positions, Append(store, positions, allocated)


def _pad_rows(x, rows: int):
    """`x` with `rows` zero rows appended on its sequence axis, [B, S, H, D]."""
    return x if rows == 0 else jnp.pad(x, ((0, 0), (0, rows), (0, 0), (0, 0)))


def widen_value_heads(query, value):
    """Zero-pad `value`'s head axis out to the query's width.

    `jax.nn.dot_product_attention` checks the value against the key's whole
    shape, so a value narrower than the query is refused before any kernel
    sees it. DeepSeek's latent attention is that shape: `v_head_dim` is 128
    against a 192-wide query in every released V2/V3 config. The second
    product is a separate sum per value column, so a zero column gives a
    zero output column and the caller crops it off. transformers 5.16.1
    pads FlashAttention the same way (integrations/flash_attention.py:63).

    A value wider than the query has no such rewrite. Padding the query and
    the key instead would move the kernel's 1/sqrt(d) scale off the query's
    width, so this raises and names the reference path.
    """
    width, v_width = query.shape[-1], value.shape[-1]
    if v_width > width:
        raise ValueError(
            "fused attention runs one head width for the keys and the values, "
            f"and a value head of {v_width} is wider than the query head of "
            f"{width}: the kernel would scale the logits by the padded width. "
            "Use the reference implementation (attention_impl 'reference').")
    return jnp.pad(value, ((0, 0),) * (value.ndim - 1) + ((0, width - v_width),))


def cudnn_attention(query, key, value, bias, mask, causal, sliding_window):
    """Run jax's cudnn flash attention over any sequence length.

    cudnn's kernel has no backward pass for an odd query or key length; jax
    raises NotImplementedError from inside the gradient. 77 CLIP text tokens
    are odd, as is every concatenated text-plus-image sequence. One zero row
    of padding makes the length even. A padded query row's output is sliced
    off and a padded key is hidden by the kernel's own padding mask
    (key_value_seq_lengths), so every real query attends to the keys it had.
    tests/test_kernels.py pins the equality with the xla path.
    """
    q_len, kv_len = query.shape[-3], key.shape[-3]
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
    kv_lengths = None if kv_pad == 0 else jnp.full(key.shape[:1], kv_len, jnp.int32)
    # A left window of l means the l+1 most recent keys on both the xla and
    # the cudnn path, which is the window this function counts.
    out = jax.nn.dot_product_attention(
        query, key, value, bias=bias, mask=mask, is_causal=causal,
        key_value_seq_lengths=kv_lengths,
        local_window_size=None if sliding_window is None else (sliding_window - 1, 0),
        implementation='cudnn')
    return out[:, :q_len] if q_pad else out


def stripe(x, shards: int, axis: int = 1):
    """Reorder `axis` so every shard holds equal causal work.

    The axis is cut into 2 * shards chunks; shard i takes chunks i and
    2 * shards - 1 - i. For two shards, [0..7] becomes [0, 1, 6, 7, 2, 3,
    4, 5], which is MaxText's `reorder_sequence` order.

    A reshape, two slices, a flip and a stack, not a gather with the
    permutation: GSPMD lowers these to collective-permutes of the chunks
    that change shard, half the rows. It lowers a gather along a split axis
    to an all-gather of the whole array on every device
    (tests/test_sequence_parallel.py measures both).
    """
    axis %= x.ndim
    length = x.shape[axis]
    if length % (2 * shards):
        raise ValueError(
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
                                sliding_window, mask, bias, sinks):
    """Run `kernel` over a sequence the mesh's sequence axis splits.

    Two exchanges are exact for every call they take, and they differ in
    what crosses the interconnect. `exchanged_heads_attention` (Ulysses)
    takes a call whose query heads divide by tensor times sequence and whose
    query and key lengths both divide by the shard count, and runs where
    `all_to_all_moves_less` says it moves fewer bytes. Every other call runs
    `gathered_keys_attention`, which takes any head count and any key
    length: joint and cross attention over an odd length, a head count the
    split does not divide, and grouped-query calls with no mask.
    """
    mesh = jax.sharding.get_abstract_mesh()
    tensor = (mesh.shape[TENSOR_AXIS]
              if TENSOR_AXIS in mesh.axis_names and TENSOR_AXIS not in mesh.manual_axes else 1)
    heads, kv_heads = query.shape[-2], key.shape[-2]
    reordered = causal or sliding_window is not None or mask is not None
    exchangeable = (heads % (tensor * shards) == 0
                    and query.shape[-3] % shards == 0 and key.shape[-3] % shards == 0)
    run = (exchanged_heads_attention
           if exchangeable and all_to_all_moves_less(heads, kv_heads, tensor, shards,
                                                     reordered=reordered)
           else gathered_keys_attention)
    return run(kernel, query, key, value, shards, causal=causal,
               sliding_window=sliding_window, mask=mask, bias=bias, sinks=sinks)


def all_to_all_moves_less(heads: int, kv_heads: int, tensor: int, shards: int, *,
                          reordered: bool) -> bool:
    """Whether Ulysses sends fewer bytes per device than gathering the keys.

    Counted per device, per batch row, in units of S * D * (n - 1) / n for
    a sequence of S rows, head width D and n = `shards`, over one tensor
    shard's heads: H = heads / tensor, K = the key heads it holds.

    The all-to-all sends (n - 1) / n of four local blocks of S / n rows:
    the query and the output at H heads, the key and the value at K', the
    key heads repeated out to lcm(kv_heads, tensor * n) over tensor. That
    is 2 (H + K') / n.

    The gather receives the (n - 1) / n of the whole keys and values it
    does not hold, 2 K. A causal, windowed or masked call also stripes its
    S / n query rows and unstripes its output rows, and that reorder moves
    (n - 1) / n of both, 2 H / n. So the gather sends 2 K, plus 2 H / n
    when reordered.

    At H = 32, K = 8, n = 2 with one tensor shard, in units of S * D the
    all-to-all sends 20 either way, the gather 8 unmasked and 24 causal:
    grouped-query attention with no mask gathers, causal attention
    exchanges. The gather wins ties, as the path that takes every shape.
    Mask and bias blocks are left out; both paths carry them.
    """
    local_heads = heads // tensor
    repeated = math.lcm(kv_heads, tensor * shards) // tensor
    # Key heads the tensor axis cannot split stay whole on every shard.
    gathered = kv_heads // tensor if kv_heads % tensor == 0 else kv_heads
    exchanged = 2 * (local_heads + repeated) / shards
    return exchanged < 2 * gathered + (2 * local_heads / shards if reordered else 0)


def exchanged_heads_attention(kernel, query, key, value, shards: int, *, causal,
                              sliding_window, mask, bias, sinks):
    """DeepSpeed Ulysses: trade a slice of the sequence for a slice of the heads.

    Each shard holds S/n rows of every head. One all-to-all per operand over
    the sequence axis hands it all S rows of H/n heads instead, the kernel
    attends those heads over the whole sequence, and one more all-to-all
    puts the output back in rows (Jacobs et al. 2023, arXiv:2309.14509).
    The kernel sees whole sequences, so causality, a window, a packed
    document mask and splash's block skipping work as on one device, every
    shard does the same causal work without a reorder, and autodiff
    transposes each all-to-all into the reverse one for the backward pass.
    No shard ever holds the whole of any key or value, which is what the
    all-gather exchange costs: a shard holds S*K*D of each against
    S*K*D/n here.

    Heads split over the tensor axis first and the sequence axis within
    that, so H has to divide by tensor times sequence. Grouped key and value
    heads are repeated only as far as that split needs, to the least common
    multiple of their count and the split, which keeps each shard's query
    heads beside the key heads they read. A mask or bias with a head
    dimension is split with the heads, and its query and key dimensions
    arrive whole, as the kernel reads them. Learned sinks split with the
    heads they belong to.

    Batch rows split over every batch axis that still divides them, in mesh
    order; a batch too small for the rest is attended alike on those axes'
    shards, the way GSPMD replicates a dimension it cannot split.
    """
    mesh = jax.sharding.get_abstract_mesh()
    usable = [axis for axis in mesh.axis_names
              if axis not in mesh.manual_axes and mesh.shape[axis] > 1]
    batch, _, heads, _ = query.shape
    rows = row_axes(batch)
    tensor = (TENSOR_AXIS,) if TENSOR_AXIS in usable else ()
    split = shards * math.prod(mesh.shape[axis] for axis in tensor)
    if heads % split or query.shape[1] % shards or key.shape[1] % shards:
        raise ValueError(
            f"the all-to-all sequence exchange splits the {heads} query heads "
            f"{split} ways (tensor times sequence) and the query and key lengths "
            f"({query.shape[1]}, {key.shape[1]}) {shards} ways, and one of them does "
            "not divide. `gathered_keys_attention` takes any head count and key "
            "length; `sequence_parallel_attention` picks it for such a call.")
    kv_heads = math.lcm(key.shape[-2], split)
    key, value = repeat_kv_heads(key, kv_heads), repeat_kv_heads(value, kv_heads)

    row_entry = rows or None
    head_entry = (*tensor, SEQUENCE_AXIS)
    in_rows = P(row_entry, SEQUENCE_AXIS, tensor or None, None)

    def whole_rows(x):
        # [.., Q, K] broadcastable to [B, H, Q, K]: split its rows and heads
        # where it has them, keep its query and key dimensions whole.
        x = x.reshape((1,) * (4 - x.ndim) + x.shape)
        return x, P(row_entry if x.shape[0] == batch else None,
                    head_entry if x.shape[1] == heads else None, None, None)

    extras = {name: whole_rows(x) for name, x in (('mask', mask), ('bias', bias)) if x is not None}
    if sinks is not None:
        extras['sinks'] = (sinks, P(head_entry))
    names = tuple(extras)

    def local(query, key, value, *arrays):
        query, key, value = (jax.lax.all_to_all(x, SEQUENCE_AXIS, 2, 1, tiled=True)
                             for x in (query, key, value))
        given = dict(zip(names, arrays, strict=True))
        out = kernel(query, key, value, causal=causal, sliding_window=sliding_window,
                     mask=given.get('mask'), bias=given.get('bias'), sinks=given.get('sinks'))
        return jax.lax.all_to_all(out, SEQUENCE_AXIS, 1, 2, tiled=True)

    # Every axis the context leaves automatic goes manual here, those the
    # specs do not name included, over which the operands are replicated. A
    # Mosaic kernel (splash) refuses to lower where any axis is still left
    # to the partitioner, whatever its size. The pipeline also needs it: it
    # vmaps its stages with spmd_axis_name=stage, and a vmapped shard_map
    # can only split the new dimension over an axis it holds manual.
    manual = {axis for axis in mesh.axis_names if axis not in mesh.manual_axes}
    # Pallas kernels state no varying-manual-axes type for their outputs, so
    # splash inside the map needs the check off, as MaxText wraps it. Every
    # operand here is split on the axes the specs name and nothing is
    # reduced, so the check has nothing to catch.
    exchanged = jax.shard_map(
        local, in_specs=(in_rows,) * 3 + tuple(spec for _, spec in extras.values()),
        out_specs=in_rows, axis_names=manual, check_vma=False)
    return exchanged(query, key, value, *(x for x, _ in extras.values()))


def gathered_keys_attention(kernel, query, key, value, shards: int, *, causal,
                            sliding_window, mask, bias, sinks):
    """Run `kernel` with the queries split along the mesh's sequence axis.

    The keys and values are gathered whole once. Batch rows stay split over
    every other mesh axis but tensor and stage, which hold a width and a
    pipeline stage and never a row. The heads are left to GSPMD, so a width
    the rules put on the tensor axis stays there. This takes any head count,
    and costs every shard the whole of every key and value.

    A causal, windowed or masked call reorders the queries with `stripe` so
    each shard holds equal causal work, and puts the output back with
    `unstripe`. The queries' positions travel in the mask, which keeps the
    mask and the caller's rotary angles exact under the reorder. Under the
    reorder the kernels see that mask and not their causal flag, because a
    striped row's position is no longer its index. A call with no mask has
    equal work on every row and keeps its order.
    """
    mesh = jax.sharding.get_abstract_mesh()
    rows = tuple(axis for axis in mesh.axis_names
                 if axis not in (TENSOR_AXIS, SEQUENCE_AXIS, STAGE_AXIS))
    split = P(rows or None, SEQUENCE_AXIS, P.UNCONSTRAINED, P.UNCONSTRAINED)
    whole = P(rows or None, None, P.UNCONSTRAINED, P.UNCONSTRAINED)
    constrain = jax.lax.with_sharding_constraint
    query = constrain(query, split)
    key, value = constrain(key, whole), constrain(value, whole)
    if not (causal or sliding_window is not None or mask is not None):
        return constrain(kernel(query, key, value, causal=False, sliding_window=None,
                                mask=None, bias=bias, sinks=sinks), split)

    q_len, kv_len = query.shape[-3], key.shape[-3]
    query = constrain(stripe(query, shards), split)

    def rows_in_order(x: jax.Array | None) -> jax.Array | None:
        # A broadcast query row has the same value in either order.
        if x is not None and x.ndim >= 2 and x.shape[-2] == q_len:
            return stripe(x, shards, axis=-2)
        return x

    mask, bias = rows_in_order(mask), rows_in_order(bias)
    if causal or sliding_window is not None:
        structural = causal_attention_mask(
            stripe(jnp.arange(q_len), shards, axis=0), kv_len, sliding_window)
        mask = structural if mask is None else jnp.logical_and(mask, structural)
    out = kernel(query, key, value, causal=False, sliding_window=None, mask=mask, bias=bias,
                 sinks=sinks)
    return constrain(unstripe(constrain(out, split), shards), split)


CUDNN_DTYPES = (jnp.bfloat16, jnp.float16)
CUDNN_MAX_HEAD_DIM = 128


def cudnn_runs(query, softcap=None) -> bool:
    """Report whether cudnn's fused kernel takes this query.

    It needs a gpu backend, one of cudnn's two dtypes, a head dimension it
    tiles, and no logit softcap, which no fused kernel applies. Only 'auto'
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
    return (jax.default_backend() == 'gpu' and query.dtype in CUDNN_DTYPES
            and head_dim % 8 == 0 and head_dim <= CUDNN_MAX_HEAD_DIM
            and softcap is None and not deterministic_ops_requested())


def weighted_values(equation, weights, value, *, precision=None):
    """The probability-value product, with the probabilities in the value's dtype.

    An fp32 softmax leaves fp32 probabilities, and `jnp.einsum` promotes to
    the wider operand, so multiplying them by a bf16 value would run both
    sides at the fp32 rate - forward and backward, since the product's
    cotangent is then fp32 too. Rounding the probabilities first is what
    `jax.nn.dot_product_attention` does on its own xla path
    (`probs.astype(key.dtype)`), so the two paths multiply alike; the softmax
    that produced them still reduced in fp32, which is where the precision
    was needed.
    """
    return jnp.einsum(equation, weights.astype(value.dtype), value, precision=precision)


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
        weights = jax.nn.softmax(logits.astype(jnp.float32))
    else:
        weights = jax.nn.softmax(logits).astype(dtype)
    return weighted_values('...hqk,...khd->...qhd', weights, value, precision=precision)


def scaled_dot_product_attention(query, key, value, dtype=None, precision=None,
                                 force_fp32_for_softmax=True, implementation='auto',
                                 causal=False, sliding_window=None, mask=None, bias=None,
                                 sinks=None, softcap=None):
    """Attend over [B, S, H, D] queries, keys and values.

    Picks the kernel `implementation` names and returns [B, S, H, Dv]. The
    parameter tree never changes with the kernel, so checkpoints move
    between backends. `AttentionImpl` lists the paths; `attention_kernel`
    dispatches to them and raises for the arguments each one cannot honour.

    Keys and values may carry fewer heads than the query, which is
    grouped-query attention; the paths that cannot group heads themselves
    get them repeated out. The value's head width may be narrower than the
    query's, which is DeepSeek's latent attention. The reference path takes
    that as it is and a fused path pads and crops it (`widen_value_heads`),
    so the kernel selection reads the query's width either way.

    `causal` restricts query i to keys 0..i, top-left aligned like jax's
    is_causal, and `sliding_window=w` narrows that to the w most recent
    keys. Both read the row index, so decoding against a KV cache passes
    `mask` instead: a step's single query sits at the cache index, not at
    row 0. `bias` is an additive float array broadcastable to [B, H, Q, K];
    T5's relative position table travels in it. `sinks` holds one learned,
    value-free logit per query head, which the reference and xla paths put
    in the denominator. `softcap` is Gemma 2's tanh on the scaled logits,
    which only `softcapped_attention` applies.

    Under a mesh whose sequence axis is above one, the call runs through
    `sequence_parallel_attention`.

    The result leaves here under the checkpoint name 'attention_output', so
    a remat policy can save it instead of replaying the kernel in the
    backward pass (`remat_block` in dit.py). The name is inert outside
    jax.checkpoint.
    """
    kernel = functools.partial(
        attention_kernel, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, implementation=implementation,
        softcap=softcap)
    shards = sequence_shards()
    if shards > 1:
        out = sequence_parallel_attention(
            kernel, query, key, value, shards, causal=causal,
            sliding_window=sliding_window, mask=mask, bias=bias, sinks=sinks)
    else:
        out = kernel(query, key, value, causal=causal, sliding_window=sliding_window,
                     mask=mask, bias=bias, sinks=sinks)
    return checkpoint_name(out, 'attention_output')


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


def fused_attention(query, key, value, bias, mask, causal, sliding_window, implementation):
    """Run the fused kernel `implementation` names, at the query's head width.

    Every fused kernel runs one head width for the keys and the values, so a
    narrower value rides in padded and its own columns come back out. The
    widths a caller passes are static, so this costs no runtime branch.
    """
    v_head_dim = value.shape[-1]
    if v_head_dim != query.shape[-1]:
        value = widen_value_heads(query, value)

    if implementation == 'cudnn':
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
        out = cudnn_attention(query, key, value, bias, mask, causal, sliding_window)
    elif implementation == 'xla':
        # A left window of l means the l+1 most recent keys on both the xla and
        # the cudnn path, which is the window this function counts.
        out = jax.nn.dot_product_attention(
            query, key, value, bias=bias, mask=mask, is_causal=causal,
            local_window_size=None if sliding_window is None else (sliding_window - 1, 0),
            implementation='xla')
    elif implementation == 'tpu':
        out = tpu_attention(query, key, value, bias, mask, causal, sliding_window,
                            interpret=jax.default_backend() != 'tpu')
    else:
        raise ValueError(f"Unknown attention implementation: {implementation}")
    return out if v_head_dim == out.shape[-1] else out[..., :v_head_dim]


def attention_kernel(query, key, value, dtype=None, precision=None,
                     force_fp32_for_softmax=True, implementation='auto',
                     causal=False, sliding_window=None, mask=None, bias=None, sinks=None,
                     softcap=None):
    """Dispatch one whole-sequence attention call to the kernel it names.

    Sinks and a softcap have no fused kernel, so they run their own path.
    'auto' resolves here, against this call's shapes and this machine's
    backend. What is left goes to `fused_attention`, after the arguments it
    cannot honour raise.
    """
    if sliding_window is not None and sliding_window < 1:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")
    if sinks is not None:
        if implementation not in ('reference', 'auto', 'xla'):
            raise ValueError(f"attention implementation '{implementation}' cannot honor sinks")
        if softcap is not None:
            raise ValueError(
                "attention sinks and a logit softcap have no reference that "
                "combines them, so the sink path takes no softcap")
        mask = combined_attention_mask(
            query.shape[-3], key.shape[-3], causal, sliding_window, mask)
        return attention_with_sinks(
            query, key, value, sinks, mask=mask, bias=bias, dtype=dtype,
            precision=precision, force_fp32_for_softmax=force_fp32_for_softmax)

    implementation = resolve_implementation(
        implementation, query, key, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap, causal=causal,
        sliding_window=sliding_window, mask=mask, bias=bias)
    if softcap is not None and implementation in ('cudnn', 'tpu'):
        raise ValueError(
            f"attention implementation '{implementation}' cannot apply an "
            f"attention logit softcap of {softcap}: the fused kernel has no tanh "
            "between its scaling and its softmax. Use attention_impl 'xla' or "
            "the reference implementation (attention_impl 'reference').")
    if implementation == 'cudnn' and deterministic_ops_requested():
        raise ValueError(
            "attention implementation 'cudnn' cannot run under "
            "--xla_gpu_deterministic_ops: on this JAX and XLA its backward pass "
            "is unusable, because an executable holding two identical fused "
            "attention backward calls, which every multi-layer model has, fails "
            "at execution time (openxla/xla#46500). Use attention_impl 'xla', "
            "which is deterministic, or drop the flag.")

    if implementation == 'reference' or softcap is not None:
        heads = query.shape[-2]
        key = repeat_kv_heads(key, heads)
        value = repeat_kv_heads(value, heads)
        mask = combined_attention_mask(
            query.shape[-3], key.shape[-3], causal, sliding_window, mask)
        if softcap is not None:
            return softcapped_attention(
                query, key, value, softcap, dtype=dtype, precision=precision,
                force_fp32_for_softmax=force_fp32_for_softmax, mask=mask, bias=bias)
        return nn.dot_product_attention(
            query, key, value, bias=bias, mask=mask, dtype=dtype, broadcast_dropout=False,
            dropout_rng=None, precision=None,
            force_fp32_for_softmax=force_fp32_for_softmax, deterministic=True,
            # flax takes the precision through these two or through
            # `precision`, never both, and its own value product would
            # multiply the fp32 probabilities by the bf16 value.
            qk_attn_weights_einsum=functools.partial(jnp.einsum, precision=precision),
            attn_weights_value_einsum=functools.partial(weighted_values, precision=precision))

    refuse_reference_only_arguments(
        implementation, query, dtype, precision, force_fp32_for_softmax)
    return fused_attention(query, key, value, bias, mask, causal, sliding_window,
                           implementation)


def reference_only(query, dtype, precision, force_fp32_for_softmax) -> bool:
    """Whether a call asks for arithmetic only the reference path performs.

    The three are what `refuse_reference_only_arguments` raises for: a
    matmul precision above DEFAULT, a softmax outside fp32, and a compute
    dtype other than the inputs'. A fused kernel runs none of them.
    """
    return (bool(precision_names(precision) & {'HIGH', 'HIGHEST'})
            or not force_fp32_for_softmax
            or (dtype is not None and jnp.dtype(dtype) != query.dtype))


def resolve_implementation(implementation, query, key, *, dtype=None, precision=None,
                           force_fp32_for_softmax=True, softcap=None, causal=False,
                           sliding_window=None, mask=None, bias=None) -> str:
    """The concrete kernel an `AttentionImpl` names for this call.

    Only 'auto' chooses, against the call's shapes and this machine's
    backend: the reference path when the call asks for arithmetic no fused
    kernel performs (`reference_only`), else cudnn where `cudnn_runs`, the
    tpu kernel where `tpu_runs`, and xla anywhere else. Any other name is
    returned as it is, so an explicit kernel still refuses what it cannot
    honour by name.
    """
    if implementation not in ('auto', 'reference', 'xla', 'cudnn', 'tpu'):
        raise ValueError(f"Unknown attention implementation: {implementation}")
    if implementation != 'auto':
        return implementation
    if reference_only(query, dtype, precision, force_fp32_for_softmax):
        return 'reference'
    if cudnn_runs(query, softcap):
        return 'cudnn'
    if tpu_runs(query, key, softcap, causal=causal, sliding_window=sliding_window,
                mask=mask, bias=bias):
        return 'tpu'
    return 'xla'


def kernel_for_materialized_mask(implementation: str, query, *, dtype=None, precision=None,
                                 force_fp32_for_softmax=True) -> str:
    """Send a call that built an explicit mask to the xla kernel.

    cuDNN has no mask argument: causality and the window are flags, and jax
    hands the kernel a bool mask as an additive bias of -2**41 in the
    compute dtype instead (`combine_bias_and_mask` in
    jax/_src/cudnn/fused_attention_stablehlo.py). That also makes
    `check_is_flash_attention` refuse an odd length while training. The xla
    kernel masks by exclusion, on every backend and with the same fp32
    softmax. It costs 83.6 ms and 5.80 GiB a step where the fixed window on
    cuDNN costs 75.8 ms and 4.99 GiB (docs/concepts/language_models.md).

    'auto' first takes the reference path for arithmetic only it performs
    (`reference_only`), as `resolve_implementation` does.
    """
    if implementation == 'auto' and reference_only(query, dtype, precision,
                                                   force_fp32_for_softmax):
        return 'reference'
    return 'xla' if implementation in ('auto', 'cudnn') else implementation


def _blocks(x, block: int, blocks: int):
    """`[B, S, ...]` padded at the end to `blocks * block` rows, as `[B, blocks, block, ...]`."""
    x = jnp.pad(x, [(0, 0), (0, blocks * block - x.shape[1])] + [(0, 0)] * (x.ndim - 2))
    return x.reshape(x.shape[0], blocks, block, *x.shape[2:])


def _banded(x):
    """Each block of `[B, n, W, ...]` behind the block before it: `[B, n, 2W, ...]`.

    The first block's predecessor is zeros, whose rows are negative and so
    outside every query's causal reach.
    """
    previous = jnp.pad(x[:, :-1], [(0, 0), (1, 0)] + [(0, 0)] * (x.ndim - 2))
    return jnp.concatenate([previous, x], axis=2)


def local_attention(query, key, value, *, window: int | None = None, chunk: int | None = None,
                    positions=None, segment_ids=None, valid=None, dtype=None, precision=None,
                    force_fp32_for_softmax=True, implementation='auto', sinks=None,
                    softcap=None):
    """Causal self-attention in which each query reads only nearby keys, in
    memory linear in the sequence.

    `window=w` is sliding attention: a query reads itself and the w - 1 keys
    before it (Mistral, Gemma, MaxText's `sliding_window_size`). `chunk=c`
    is chunked local attention: a query reads the keys at or before it whose
    position shares its chunk, `position // c` (Llama 4's
    `attention_chunk_size`, MaxText's `chunk_attn_window_size`). Exactly one
    of the two is set.

    `positions` places the chunks: `[S]` or `[B, S]`, advancing by one per
    row inside a document, as a packed batch's do; None is the row index.
    `segment_ids` keeps each packed document to itself (0 is padding) and
    `valid` `[B, S]` drops the keys it marks False.

    No `[S, S]` array is built above two spans. A sliding window without
    sinks that cudnn or splash takes as a flag runs there, whose kernels
    skip the blocks outside it. Chunks that start at row multiples fold into the
    batch, one causal call per chunk, which every kernel takes. Everything
    else runs banded: the queries in blocks of the span, each against its
    own block and the one before, which holds every key a query of the
    block may read, under a `[W, 2W]` mask per block. The xla kernel's
    logits are then `[S, 2W]` per head where a dense mask costs `[S, S]`.
    Banding pads the sequence to whole spans, so a sequence of at most two
    spans, where `[S, S]` is no larger than `[S, 2W]`, takes the dense
    call instead.
    """
    if (window is None) == (chunk is None):
        raise ValueError("local attention takes exactly one of window and chunk")
    span = window if window is not None else chunk
    assert span is not None
    if span < 1:
        raise ValueError(f"a local span is a positive number of keys, got {span}")
    batch, length = query.shape[0], query.shape[1]
    if key.shape[1] != length:
        raise ValueError(
            f"local attention is self-attention: {length} queries against "
            f"{key.shape[1]} keys")
    kernel = functools.partial(
        attention_kernel, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, sinks=sinks, softcap=softcap)
    masked = kernel_for_materialized_mask(
        implementation, query, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax)
    flags_only = segment_ids is None and valid is None
    if window is not None and flags_only:
        resolved = resolve_implementation(
            implementation, query, key, dtype=dtype, precision=precision,
            force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap, causal=True,
            sliding_window=window)
        # No fused kernel honours sinks, so a sink call bands wherever the
        # window would otherwise go to one.
        if (resolved in ('cudnn', 'tpu') and sinks is None) or length <= 2 * span:
            return scaled_dot_product_attention(
                query, key, value, dtype=dtype, precision=precision,
                force_fp32_for_softmax=force_fp32_for_softmax, implementation=implementation,
                causal=True, sliding_window=window, sinks=sinks, softcap=softcap)
    if length <= 2 * span:
        places = jnp.arange(length) if positions is None else positions
        mask = (None if chunk is None or (positions is None and length <= chunk)
                else chunk_mask(places, places, chunk))
        if segment_ids is not None:
            inside = document_mask(segment_ids)[:, None]
            mask = inside if mask is None else mask & inside
        if valid is not None:
            live = jnp.asarray(valid, bool)[:, None, None, :]
            mask = live if mask is None else mask & live
        if mask is not None:
            mask = combined_attention_mask(length, length, causal=True,
                                           sliding_window=window, mask=mask)
            implementation = masked
        return scaled_dot_product_attention(
            query, key, value, dtype=dtype, precision=precision,
            force_fp32_for_softmax=force_fp32_for_softmax, implementation=implementation,
            causal=mask is None, mask=mask, sinks=sinks, softcap=softcap)

    blocks = -(-length // span)

    def folded(x, width: int):
        return x.reshape(batch * blocks, width, *x.shape[3:])

    def row_blocks(x):
        return _blocks(jnp.broadcast_to(jnp.asarray(x), (batch, length)), span, blocks)

    if chunk is not None and positions is None:
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
        rows = jnp.arange(blocks * span).reshape(1, blocks, span)
        key_rows = jnp.concatenate([rows - span, rows], axis=-1)
        keep = ((key_rows[..., None, :] >= 0)
                & (key_rows[..., None, :] <= rows[..., :, None]))
        if window is not None:
            keep = keep & (rows[..., :, None] - key_rows[..., None, :] < window)
        keep = jnp.broadcast_to(keep, (batch, blocks, span, 2 * span))
        if chunk is not None:
            places = row_blocks(positions) // chunk
            keep = keep & (places[..., :, None] == _banded(places)[..., None, :])
        if segment_ids is not None:
            segments = row_blocks(segment_ids)
            keep = keep & ((segments[..., :, None] == _banded(segments)[..., None, :])
                           & (segments[..., :, None] != 0))
        if valid is not None:
            keep = keep & _banded(row_blocks(jnp.asarray(valid, bool)))[..., None, :]
        out = kernel(folded(_blocks(query, span, blocks), span),
                     folded(_banded(_blocks(key, span, blocks)), 2 * span),
                     folded(_banded(_blocks(value, span, blocks)), 2 * span),
                     implementation=masked,
                     mask=keep.reshape(batch * blocks, 1, span, 2 * span))
    out = out.reshape(batch, blocks * span, *out.shape[2:])[:, :length]
    return checkpoint_name(out, 'attention_output')


# Splash's tile size. At 512 the forward kernel holds 1 MiB of fp32 logits
# plus its operands, which fits VMEM at decoder head widths. jax's default
# of 128 gives the MXU a sixteenth of that per pass. `splash_block_sizes`
# narrows this to a divisor of each sequence, as the mask blocking needs.
SPLASH_BLOCK = 512
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


def tpu_runs(query, key, softcap=None, *, causal=False, sliding_window=None,
             mask=None, bias=None) -> bool:
    """Report whether splash takes this call.

    It needs a tpu backend, one of the two dtypes a TPU matmul reads,
    sequence lengths its mask blocking can tile, a mask it can describe, and
    neither a bias nor a softcap. Only 'auto' asks this, after `cudnn_runs`;
    an explicit 'tpu' sends the calls this turns down to the older pallas
    flash kernel instead.

    The head width is not read, because splash does not constrain it. The
    kernel pads the value width to a whole number of lanes and slices the
    result back (`pl.cdiv(head_dim_v, NUM_LANES)`,
    splash_attention_kernel.py:732). The query and key width is only a
    dot_general's contraction dimension inside the kernel, which Mosaic pads
    like any other minor axis. The sequence axes are the ones the kernel and
    its mask blocking state a divisibility for.

    A call under an automatic sequence axis above one is turned down as
    well: that is `gathered_keys_attention`, whose split is GSPMD constraints
    around a whole-sequence kernel. A pallas call with no partitioning rule
    would have the queries gathered back to serve it, which is the split
    that exchange exists to keep. `exchanged_heads_attention` calls the
    kernel inside a `shard_map` that holds the sequence axis manual, where
    `sequence_shards` is 1 and each instance holds whole sequences, so
    splash does take the calls that exchange runs.
    """
    if jax.default_backend() != 'tpu' or query.dtype not in SPLASH_DTYPES:
        return False
    if sequence_shards() > 1:
        return False
    if softcap is not None or bias is not None:
        return False
    q_len, kv_len = query.shape[-3], key.shape[-3]
    if q_len % SPLASH_LANES or kv_len % SPLASH_LANES:
        return False
    return splash_mask_descriptor(
        q_len, kv_len, query.shape[-2], causal, sliding_window, mask) is not None


def tpu_attention(query, key, value, bias, mask, causal, sliding_window, *,
                  interpret: bool):
    """Attend on pallas: splash where its descriptor covers the call, flash
    everywhere else.

    Splash is the block-sparse kernel. Its mask is a descriptor built while
    the executable is, the blocks that descriptor empties are never visited,
    and a causal or windowed long sequence costs its live blocks rather than
    its rectangle. What it has no form for is a value of the trace: there is
    no bias argument at all, and the mask has to be readable on the host.
    Flash takes both as one additive [B, H, Q, K] array and pays the whole
    rectangle for them, so it stays for exactly these calls:

    - an additive `bias`, which is T5's relative position table;
    - a `mask` that is a tracer (a KV-cache decode mask over slots, a packed
      batch's segment mask, the striped mask sequence parallelism builds), or
      one past `SPLASH_DENSE_MASK_CELLS`, or one that differs by batch row;
    - a query or key length that is not a multiple of `SPLASH_LANES`, which
      leaves the mask blocking no block size that divides its sequence.

    `interpret` runs splash under pallas's interpreter instead of Mosaic,
    which is what lets the same arithmetic, forward and backward, run off a
    TPU and what the parity tests use. It is far slower than XLA's attention,
    so it exists for an explicit 'tpu' and never for 'auto'. The flash kernel
    takes no such argument, so the calls that fall through to it run on a TPU
    and nowhere else.
    """
    q_len, kv_len = query.shape[-3], key.shape[-3]
    descriptor = None
    if bias is None and not (q_len % SPLASH_LANES or kv_len % SPLASH_LANES):
        descriptor = splash_mask_descriptor(
            q_len, kv_len, query.shape[-2], causal, sliding_window, mask)
    if descriptor is not None:
        return splash_attention(query, key, value, descriptor, interpret=interpret)
    return pallas_flash_attention(query, key, value, bias, mask, causal, sliding_window)


def splash_attention(query, key, value, descriptor, *, interpret: bool):
    """Run the splash kernel over [B, S, H, D] arrays under `descriptor`.

    The kernel takes one example at a time, as [H, S, D] with the head width
    minor, so the batch rides in on a vmap and the seam is two transposes.
    It applies no scale of its own, unlike the flash kernel's `sm_scale`, so
    the 1/sqrt(d) goes onto the query in the query's own dtype. That is
    where flax's reference path puts it, which makes the two agree exactly
    in fp32 rather than to a rounding of the scale. Grouped key heads stay
    grouped, because splash reads q_heads % kv_heads itself, so this path
    never materializes the repeated keys the flash path needs.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel

    kernel = splash_attention_kernel.make_splash_mha(
        descriptor, block_sizes=splash_block_sizes(query.shape[-3], key.shape[-3]),
        head_shards=1, q_seq_shards=1, interpret=interpret)
    scale = jnp.asarray(1.0 / math.sqrt(query.shape[-1]), query.dtype)
    attended = jax.vmap(kernel)(jnp.moveaxis(query, -2, -3) * scale,
                                jnp.moveaxis(key, -2, -3), jnp.moveaxis(value, -2, -3))
    if isinstance(attended, tuple):
        # make_splash_mha(save_residuals=True) returns the logsumexp beside
        # the output, and this builds a kernel without it, so the pair is a
        # kernel someone else built. Narrowed rather than asserted, because a
        # cast is evidence nobody produced.
        raise ValueError("splash returned residuals this path has no consumer for")
    return jnp.moveaxis(attended, -3, -2)


def splash_block_sizes(q_len: int, kv_len: int):
    """Narrow `SPLASH_BLOCK` to a divisor of each sequence, forward and backward.

    The greatest common divisor satisfies both of the kernel's rules at
    once. It divides its sequence, which the mask blocking requires, and it
    stays a multiple of `SPLASH_LANES`, which the key compute block
    requires, because the constant and the admitted lengths are both
    multiples of it. The backward blocks are filled in because a kernel
    built without them raises "Need to specify backward blocks." from inside
    its own vjp, which a training run would meet at its first gradient.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel

    block_q, block_kv = math.gcd(SPLASH_BLOCK, q_len), math.gcd(SPLASH_BLOCK, kv_len)
    return splash_attention_kernel.BlockSizes(
        block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv,
        block_q_dkv=block_q, block_kv_dkv=block_kv, block_kv_dkv_compute=block_kv,
        block_q_dq=block_q, block_kv_dq=block_kv)


def splash_mask_descriptor(q_len: int, kv_len: int, heads: int, causal: bool,
                           sliding_window: int | None, mask):
    """Build splash's mask for one call, or None when splash cannot describe it.

    The structural part is a function of the two indices, not an array.
    CausalMask and LocalMask carry the comparison the kernel evaluates per
    block, so the blocks they empty leave the grid and nothing proportional
    to Q*K is stored. Dew's window is causal already, so a window replaces
    the causal flag here instead of sitting beside it. An explicit boolean
    mask has no such form: it is ANDed in as dense blocks, so only a
    concrete, small one is taken.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

    shape = (q_len, kv_len)
    if sliding_window is not None:
        structural = splash_attention_mask.LocalMask(shape, (sliding_window - 1, 0), 0)
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

    Returns None when splash cannot carry it, which three things cause. The
    mask is a value of the trace: the descriptor is built while the
    executable is, so a tracer cannot be read, and a decode mask over cache
    slots and a packed batch's segment mask are both tracers. Or it has a
    batch axis wider than one: splash indexes by head and position and has
    no batch axis. Or it is large: unresolved blocks are stored dense inside
    the executable, so past `SPLASH_DENSE_MASK_CELLS` the additive-bias path
    is cheaper.

    numpy and the tracer type are imported here because this is the one
    place the mask leaves the trace, and splash's NumpyMask is a host array.
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


def pallas_flash_attention(query, key, value, bias, mask, causal, sliding_window):
    """Run the pallas TPU flash kernel, with every mask as an additive bias.

    The kernel has no mask argument, so a window and an explicit mask become
    one [B, H, Q, K] float array of zeros and the dtype's minimum, added to
    the logits. Only causality is a flag it takes. That array is the whole
    rectangle, which is what splash exists to avoid, so this path runs only
    for the calls `tpu_attention` names. The 1/sqrt(d) scale is the kernel's
    own `sm_scale` here.
    """
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

    heads = query.shape[-2]
    key = repeat_kv_heads(key, heads)
    value = repeat_kv_heads(value, heads)
    # pallas wants [B, H, S, D]
    q = jnp.moveaxis(query, -2, -3)
    k = jnp.moveaxis(key, -2, -3)
    v = jnp.moveaxis(value, -2, -3)
    combined = None
    if bias is not None:
        combined = jnp.broadcast_to(bias.astype(q.dtype),
                                    (q.shape[0], q.shape[1], q.shape[2], k.shape[2]))
    if sliding_window is not None:
        band = causal_attention_mask(
            jnp.arange(query.shape[-3]), key.shape[-3], sliding_window)
        mask = band if mask is None else jnp.logical_and(mask, band)
    if mask is not None:
        seated = jnp.broadcast_to(
            jnp.where(mask, 0, jnp.finfo(q.dtype).min).astype(q.dtype),
            (q.shape[0], q.shape[1], q.shape[2], k.shape[2]))
        combined = seated if combined is None else combined + seated
    return jnp.moveaxis(
        flash_attention(q, k, v, ab=combined, causal=causal,
                        sm_scale=1.0 / math.sqrt(query.shape[-1])), -3, -2)


@logical_axes({
    ("to_q",): ("embed", "heads", "head_dim"),
    ("to_k",): ("embed", "heads", "head_dim"),
    ("to_v",): ("embed", "heads", "head_dim"),
    ("to_out_0",): ("heads", "head_dim", "embed"),
})
class NormalAttention(nn.Module):
    """Attend over a `[B, S, C]` or `[B, H, W, C]` input with multiple heads.

    `causal` makes it a decoder attention, where query i sees keys 0..i.
    `decode=True` on a call runs it against a fixed-size KV cache allocated
    at max_seq_len instead: the first call writes the whole prompt, later
    calls append one token each. Neither flag touches the param tree, so a
    model trained without either reloads into a decoding one unchanged.
    `freqs_cis` rotates the queries and keys of a self-attention call;
    `rotary_freqs` gives the pair, and None leaves them unrotated.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_bias: bool = True
    force_fp32_for_softmax: bool = True
    qk_norm: bool = False  # RMSNorm on q/k per head (SD3-style bf16 logit safety)
    attention_impl: str = "auto"  # an AttentionImpl
    causal: bool = False
    max_seq_len: int | None = None  # KV cache length, required to decode

    def setup(self):
        dense = functools.partial(
            nn.DenseGeneral,
            features=[self.heads, self.dim_head],
            axis=-1,
            precision=self.precision,
            use_bias=self.use_bias,
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
    def __call__(self, x, context=None, decode: bool = False, freqs_cis=None):
        orig_x_shape = x.shape
        if len(x.shape) == 4:
            x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        context = x if context is None else context
        if len(context.shape) == 4:
            context = context.reshape(
                (context.shape[0], context.shape[1] * context.shape[2], context.shape[3]))
        query = self.query(x)
        key = self.key(context)
        value = self.value(context)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        if freqs_cis is not None:
            freqs_cos, freqs_sin = freqs_cis
            query = apply_rotary(query, freqs_cos, freqs_sin)
            key = apply_rotary(key, freqs_cos, freqs_sin)

        causal, mask = self.causal, None
        if decode:
            # Position lives in the cache slot now, not in the row index, so
            # causality travels as a mask over the slots.
            positions, append = open_kv_cache(self, key, self.max_seq_len)
            key, value = append(key, value)
            mask = causal_attention_mask(positions, key.shape[-3])
            causal = False

        hidden_states = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            implementation=self.attention_impl, causal=causal, mask=mask,
        )
        proj = self.proj_attn(hidden_states)
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
        self.net_0 = FlaxGEGLU(self.dim, dtype=self.dtype, precision=self.precision, approximate=self.approximate_gelu)
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

    @nn.compact
    def __call__(self, hidden_states, context=None):
        if self.only_pure_attention:
            return self.attention2(hidden_states, context)

        if not self.use_cross_only:
            hidden_states = hidden_states + self.attention1(self.norm1(hidden_states))
        hidden_states = hidden_states + self.attention2(self.norm2(hidden_states), context)
        return hidden_states + self.ff(self.norm3(hidden_states))


@dataclasses.dataclass(frozen=True)
class Stage:
    """Describes one resolution stage's attention in a UNet.

    A stage with no attention is `None` instead. Every field is a
    `TransformerBlock` setting. The head width is the stage's channel count
    divided by `heads`, which the unet knows and a config does not, so there
    is no `dim_head` field. `use_linear_attention` selects the projection
    kind, a dense layer or a 1x1 convolution; it is not linear attention.
    `dew.registry.from_record` builds one from a record at the build
    boundary, so a stage arrives as `{"heads": 8}` from a command line and a
    misspelled field raises there.

    `precision` of None means the model's. `dew.registry.with_precision`
    writes the model's dtype into every stage.
    """

    heads: int
    use_linear_attention: bool = True
    use_projection: bool = False
    use_self_and_cross: bool = True
    only_pure_attention: bool = True
    force_fp32_for_softmax: bool = False
    norm_inputs: bool = True
    explicitly_add_residual: bool = True
    norm_epsilon: float = 1e-4
    dtype: Dtype | None = jnp.float32
    precision: PrecisionLike = None


def stage_attention(stage: Stage, channels: int, attention_impl: str,
                    precision: PrecisionLike, name: str) -> "TransformerBlock":
    """Build the block a `Stage` describes, at the stage's channel count.

    A stage that names no precision takes the model's.
    """
    return TransformerBlock(
        heads=stage.heads, dim_head=channels // stage.heads, dtype=stage.dtype,
        attention_impl=attention_impl, use_projection=stage.use_projection,
        use_self_and_cross=stage.use_self_and_cross,
        precision=stage.precision or precision,
        only_pure_attention=stage.only_pure_attention,
        force_fp32_for_softmax=stage.force_fp32_for_softmax,
        norm_inputs=stage.norm_inputs, explicitly_add_residual=stage.explicitly_add_residual,
        use_linear_attention=stage.use_linear_attention, norm_epsilon=stage.norm_epsilon,
        name=name)


class TransformerBlock(nn.Module):
    """Run a `BasicTransformerBlock`, optionally at its own head width.

    `use_projection` puts a projection into and out of `heads * dim_head`
    around the block; without it the block runs at the input width.
    `use_linear_attention` picks what that projection is, a dense layer or
    a 1x1 convolution. It does not select linear attention.
    """
    heads: int = 4
    dim_head: int = 32
    use_linear_attention: bool = True
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
        inner_dim = self.heads * self.dim_head
        C = x.shape[-1]
        if self.norm_inputs:
            x = RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)(x)
        if self.use_projection:
            if self.use_linear_attention:
                projected_x = nn.Dense(features=inner_dim,
                                       use_bias=False, precision=self.precision,
                                       dtype=self.dtype, name='project_in')(x)
            else:
                projected_x = nn.Conv(
                    features=inner_dim, kernel_size=(1, 1),
                    strides=(1, 1), padding='VALID', use_bias=False, dtype=self.dtype,
                    precision=self.precision, name='project_in_conv',
                )(x)
        else:
            projected_x = x
            inner_dim = C

        context = projected_x if context is None else context

        projected_x = BasicTransformerBlock(
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
        )(projected_x, context)

        if self.use_projection:
            if self.use_linear_attention:
                projected_x = nn.Dense(features=C, precision=self.precision,
                                       dtype=self.dtype, use_bias=False,
                                       name='project_out')(projected_x)
            else:
                projected_x = nn.Conv(
                    features=C, kernel_size=(1, 1),
                    strides=(1, 1), padding='VALID', use_bias=False, dtype=self.dtype,
                    precision=self.precision, name='project_out_conv',
                )(projected_x)

        if self.only_pure_attention or self.explicitly_add_residual:
            projected_x = x + projected_x
        return projected_x
