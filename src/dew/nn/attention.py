"""Attention: the one kernel path, the KV cache, and the blocks the UNets
use, the latter ported from diffusers' attention_flax.py."""

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
from .sharding import SEQUENCE_AXIS, STAGE_AXIS, TENSOR_AXIS, logical_axes, sequence_shards

AttentionImpl = Literal["auto", "reference", "xla", "cudnn", "tpu"]
"""Which kernel an attention call runs, named once for every layer that
carries the choice: a `ModelConfig`, the modules' `attention_impl` field and
`scaled_dot_product_attention`'s `implementation`.

'reference' is the portable einsum and softmax, the only path that reads
dtype, precision and force_fp32_for_softmax; 'xla' and 'cudnn' are
`jax.nn.dot_product_attention`'s own two; 'tpu' is the pallas splash kernel,
with the older pallas flash kernel behind it for the calls splash's mask
descriptor cannot carry; 'auto' is cudnn where its kernel runs and xla anywhere
else, resolved per trace. A module field
spells 'reference' as None as well, which is what a module built in code
without the field set runs.
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


def combined_attention_mask(query_length: int, key_length: int, causal: bool,
                            sliding_window: int | None,
                            mask: jax.Array | None) -> jax.Array | None:
    """`mask` with the structural positions folded in: a causal flag keeps
    keys at or before each query's row, a window narrows that to the most
    recent keys, both read off the row index the way the fused kernels take
    them as flags. Unset stays unset, so a caller that distinguishes no mask
    from an all-true one keeps doing so."""
    if causal or sliding_window is not None:
        structural = causal_attention_mask(
            jnp.arange(query_length), key_length, sliding_window)
        mask = structural if mask is None else jnp.logical_and(mask, structural)
    return mask


def max_attention_logits(query: jax.Array, key: jax.Array, *, causal: bool = False,
                         sliding_window: int | None = None,
                         mask: jax.Array | None = None,
                         bias: jax.Array | None = None) -> jax.Array:
    """Per query head, the largest pre-softmax logit: `[batch, heads]`, fp32.

    The logits are the scaled dot products the kernels softmax, masked
    positions reading -inf so causality and packing never trip the maximum.
    Grouped key heads repeat out to the query heads first, the same grouping
    the kernels run. A logit softcap and attention sinks stay out: the clip
    that reads this bounds the raw query-key growth, and a softcapped model
    bounds it already."""
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
    """`normalize` rematerialized, so the residual stream crosses into the
    backward pass in the dtype the caller holds it in.

    Every norm here reduces over the width in fp32, which is what keeps a
    bf16 run stable. Differentiated as written, that upcast is also what the
    backward pass keeps: the fp32 copy of the norm's input and the fp32
    normalized activations are both residuals, so a bf16 stream is saved at
    fp32 twice per norm. On a SimpleDiT step (patch 4, emb 256, 8 layers, 4
    heads, bf16, 128px, batch 64) that was 36 fp32 [64, 1024, 256] tensors,
    2.35 GiB, against 48 bf16 ones for the stream itself; under this it is
    one, the fp32 output head's own promoted input.

    The checkpoint saves nothing, so what the backward pass holds is the
    arguments: the norm's input, its weight and its bias. It recomputes the
    reductions rather than naming them, which keeps the recomputed block's
    residuals a matter for the policy that recomputes it
    (`causal_transformer.RESIDUALS`) and not for the norms underneath.

    The wrapped function takes arrays first and its static arguments last, so
    a caller reads its own parameters out of the variable tree and hands them
    over as plain arrays: the checkpoint stays a jax transform over a pure
    function, with no flax lifting between it and the module. The forward
    values are the same ones in the same order, so a checkpoint and a
    converged run are untouched; only the buffer assignment moves.
    """
    return jax.checkpoint(normalize, policy=jax.checkpoint_policies.nothing_saveable,
                          static_argnums=static_argnums)


@functools.partial(normalized_in_fp32, static_argnums=(2, 3, 4, 5))
def rms_normalized(x, scale, epsilon: float, dtype, scale_offset: bool, scale_after_cast: bool):
    """`RMSNorm`'s body: the root-mean-square normalization in fp32 and the
    learned weight applied on whichever side of the cast the family puts it.
    `scale` of None is the weightless norm."""
    y = x.astype(jnp.float32)
    y = y * jax.lax.rsqrt(jnp.mean(jnp.square(y), axis=-1, keepdims=True) + epsilon)
    if scale is None:
        # A pure normalization with no learned weight, as Gemma 4 norms
        # its values (modeling_gemma4.py, Gemma4RMSNorm with_scale=False).
        return y.astype(dtype)
    weight = (1.0 + scale) if scale_offset else scale
    if scale_after_cast:
        return y.astype(dtype) * weight.astype(dtype)
    return (y * weight).astype(dtype)


@functools.partial(normalized_in_fp32, static_argnums=(3, 4))
def layer_normalized(x, scale, bias, epsilon: float, dtype):
    """`LayerNorm`'s body, op for op as flax computes it: E[x] and E[x^2] over
    the width in fp32, the variance from the pair and clipped at zero, the
    learned weight folded into the inverse deviation before it meets the
    centered activations. `scale` and `bias` of None are the affine-free
    norm."""
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


class RMSNorm(nn.Module):
    """RMSNorm normalized in fp32, with Gemma's (1 + w) scale behind a flag.

    scale_offset also flips the initializer to zeros, so the identity is the
    starting point either way and a Gemma checkpoint's stored weights land
    unchanged.

    The families differ in where the scale meets the activation dtype. Gemma
    multiplies in fp32 and casts the product (modeling_gemma3.py:147-150);
    Llama and Qwen3 cast the normalized activations first and multiply by
    the scale in that dtype (modeling_qwen3.py:61-64), which
    scale_after_cast reproduces. The two agree at fp32 and differ under bf16.

    The fp32 reduction runs under `normalized_in_fp32`, so what the backward
    pass keeps is this call's input in its own dtype, not an fp32 copy.
    """
    epsilon: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    with_scale: bool = True
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, x):
        dtype = self.dtype if self.dtype is not None else x.dtype
        scale = self.param(
            'scale',
            nn.initializers.zeros if self.scale_offset else nn.initializers.ones,
            (x.shape[-1],), jnp.float32) if self.with_scale else None
        return rms_normalized(x, scale, self.epsilon, dtype,
                              self.scale_offset, self.scale_after_cast)


class LayerNorm(nn.Module):
    """flax's `nn.LayerNorm` over the last axis, keeping the residual stream
    in the dtype the caller holds it in.

    Same parameter tree, same names, same fp32 arithmetic, so a checkpoint
    written by either loads into the other and the forward values match bit
    for bit. What differs is what the backward pass keeps: flax's version
    leaves it the fp32 copy of the input and the fp32 centered activations,
    and this one, through `normalized_in_fp32`, leaves it the input as it
    arrived.

    The fields are the subset the tree sets. A norm over other axes, under a
    mask, across a pmapped axis or on the slower exact variance is flax's to
    serve, and a caller that needs one takes `nn.LayerNorm` and its fp32
    residual with it.
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
    """Llama 3.1's frequency ramp, under the reference's own names.

    `_compute_llama3_parameters` (transformers modeling_rope_utils.py:580)
    divides the inverse frequencies whose wavelength exceeds
    original_max_position_embeddings / low_freq_factor by `factor`, leaves
    those below original_max_position_embeddings / high_freq_factor alone,
    and interpolates linearly in between on
    (original_max_position_embeddings / wavelength - low_freq_factor)
    / (high_freq_factor - low_freq_factor). `rope_type` is the record's
    discriminator, and only 'llama3' is this ramp; YaRN is a mixer kind's
    own value.
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
        """The scaled inverse frequencies, the reference's arithmetic in fp32."""
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
    """cos/sin of the rotary angles at absolute `positions`: [P, pairs].

    `positions` may be [P] (one sequence) or [B, P] (a packed batch whose
    documents each restart at 0); the angle axes line up with the trailing
    [B, S] either way. Computed in fp32 so a token gets the same rotation
    whether it arrives in a prefill or comes back as a single decode step.

    rot_dim narrows the rotation to the first rot_dim dimensions, and
    `partial_rotary_type` names which of the two published conventions
    that is, because they rotate different angles:

    - 'proportional' (Gemma 4, modeling_rope_utils.py
      `_compute_proportional_rope_parameters`): the exponents run over the
      full head_dim, `theta ** (2i / head_dim)` for the rot_dim // 2 rotated
      pairs, and the rest keep frequency zero (cosine one, sine zero, so the
      rotation is the identity there). The output is head_dim // 2 wide.
    - 'default' (Qwen3.5, modeling_qwen3_5.py:117-124
      `Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters`): the
      rope is a rot_dim-dimensional one, `theta ** (2i / rot_dim)`, and the
      output is rot_dim // 2 wide; `apply_rotary` passes the trailing
      dimensions through untouched, the reference's `q_rot, q_pass` split
      (modeling_qwen3_5.py:581-591).

    With rot_dim None both are the full rotation and the type is moot.
    `rope_scaling` is Llama 3.1's ramp over the base frequencies, applied
    before a proportional rope pads its zero-frequency tail, as the reference
    does (`dim = head_dim * partial_rotary_factor`).
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
        cached_valid.value = jnp.arange(capacity)[None, :] < index.value[:, None]
    return positions, allocated


def _write_cache(buffer: jax.Array, values: jax.Array, positions: jax.Array) -> jax.Array:
    """Append at per-row slots; invalid queries do not change cache storage."""
    slots = jnp.where(positions >= 0, positions, buffer.shape[1])
    return buffer.at[jnp.arange(buffer.shape[0])[:, None], slots].set(
        values.astype(buffer.dtype), mode="drop")


def open_kv_cache(module: nn.Module, key, max_seq_len, *, valid=None):
    """Fixed-size K/V with a cursor and cached validity for each batch row.

    Returns [B, S] compact slot positions and a writer. Invalid tokens have
    position -1 and do not advance the cursor. The allocation-only call leaves
    every row empty. The writer returns full cache arrays, including unused
    slots which the caller excludes with cache_valid.
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
    cached_key = module.variable("cache", "cached_key", jnp.zeros,
                                 (batch, max_seq_len, heads, head_dim), key.dtype)
    cached_value = module.variable("cache", "cached_value", jnp.zeros,
                                   (batch, max_seq_len, heads, head_dim), key.dtype)
    positions, allocated = _cache_positions(module, batch, length, max_seq_len, valid)

    def append(key, value):
        if allocated:
            cached_key.value = _write_cache(cached_key.value, key, positions)
            cached_value.value = _write_cache(cached_value.value, value, positions)
        return cached_key.value, cached_value.value

    return positions, append


def _pad_rows(x, rows: int):
    """`x` with `rows` zero rows appended on its sequence axis, [B, S, H, D]."""
    return x if rows == 0 else jnp.pad(x, ((0, 0), (0, rows), (0, 0), (0, 0)))


def widen_value_heads(query, value):
    """`value` with its head axis zero-padded to the query's width, which is
    what the fused kernels take.

    `jax.nn.dot_product_attention` checks the value against the key's whole
    shape (`_check_shape_and_dtype` in jax/_src/nn/functions.py), so a value
    narrower than the query is refused before any kernel sees it, and
    DeepSeek's latent attention is exactly that shape: `v_head_dim` is 128
    where the queries carry `qk_nope_head_dim + qk_rope_head_dim`, 192, in
    every released V2/V3 config. The attention's second product is a separate
    sum per value column, so a zero column produces a zero output column and
    leaves the real ones untouched; the caller crops them off, and the
    arithmetic on the kept columns is the fused kernel's own. transformers
    5.16.1 hands FlashAttention the same padding
    (integrations/flash_attention.py:63, `pad(value, [0, head_dim -
    v_head_dim])`, cropped again after the call).

    A value *wider* than the query has no such rewrite: padding the query and
    the key instead would move the kernel's own 1/sqrt(d) scale off the
    query's width, so it raises and names the reference path.
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
    """jax's cudnn flash attention over any sequence length.

    cudnn's kernel has no backward pass for an odd query or key length (jax
    raises NotImplementedError from inside the gradient, so a run found out at
    its first training step), and 77 CLIP text tokens are odd, as is every
    concatenated text-plus-image sequence. One zero row of padding makes the
    length even: a padded query row's output is sliced off, so nothing reads
    it, and a padded key is hidden by the kernel's own padding mask
    (key_value_seq_lengths), so every real query attends to exactly the keys
    it had. The arithmetic on the real rows is the fused kernel's, in fp32
    like the xla path's; tests/test_kernels.py pins the equality.
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
    """`x` with `axis` in the striped order of load-balanced context
    parallelism: the axis is cut into 2 * shards chunks and shard i holds
    chunks i and 2 * shards - 1 - i, so under a causal mask every shard's
    queries see the same number of (query, key) pairs. For two shards,
    [0, 1, 2, 3, 4, 5, 6, 7] becomes [0, 1, 6, 7, 2, 3, 4, 5], the order
    MaxText's reorder_sequence writes (src/maxtext/utils/maxtext_utils.py).

    Written as a reshape, two slices, a flip and a stack, not as a gather
    with the permutation: GSPMD lowers these to
    collective-permutes of the chunks that change shard, half of the rows,
    and a gather along a split axis to an all-gather of the whole array on
    every device (measured in tests/test_sequence_parallel.py).
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
                                sliding_window, mask, bias):
    """`kernel` over queries split along the sequence axis of the mesh in
    context, with the keys and values gathered whole once.

    The batch rows of every activation stay split over the mesh's other
    axes but tensor and stage, which hold a width and a pipeline stage and
    never a row; the heads are left to GSPMD, so a width the rules put on
    the tensor axis stays there. A
    causal call, a windowed one and a masked one reorder the queries with
    `stripe` so each shard holds equal causal work, carry the queries'
    positions into the mask (`causal_attention_mask` reads positions, so the
    mask and the rotary angles the caller already applied stay exact under
    the reorder) and put the output back in sequence order with
    `unstripe`. A call with no mask at all has equal work on every row and
    keeps its order. Under the reorder the kernels see an explicit mask, not
    their causal flag: a striped row's position is in the mask, not its index.
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
                                mask=None, bias=bias), split)

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
    out = kernel(query, key, value, causal=False, sliding_window=None, mask=mask, bias=bias)
    return constrain(unstripe(constrain(out, split), shards), split)
CUDNN_DTYPES = (jnp.bfloat16, jnp.float16)
CUDNN_MAX_HEAD_DIM = 128


def cudnn_runs(query, softcap=None) -> bool:
    """Whether cudnn's fused kernel takes this query: a gpu backend, one of its
    two dtypes, a head dimension it tiles, and no logit softcap, which no
    fused kernel applies. 'auto' asks this; an explicit 'cudnn' refuses by
    name instead.

    A run under `--xla_gpu_deterministic_ops` is excluded as well, because
    XLA's cudnn attention backward path crashes at execution time when one
    executable holds two structurally identical backward calls under that
    flag, which every multi-layer model has (openxla/xla#46500).

    The query's head width is the one every fused kernel runs at, values
    included: a narrower value (DeepSeek's latent attention) is padded to it
    by `widen_value_heads`, so this predicate reads the query alone and holds
    for the whole call."""
    head_dim = query.shape[-1]
    return (jax.default_backend() == 'gpu' and query.dtype in CUDNN_DTYPES
            and head_dim % 8 == 0 and head_dim <= CUDNN_MAX_HEAD_DIM
            and softcap is None and not deterministic_ops_requested())


def softcapped_attention(query, key, value, softcap: float, dtype=None, precision=None,
                         force_fp32_for_softmax=True, mask=None, bias=None):
    """Attention with Gemma 2's tanh softcap on the logits, in plain XLA ops.

    The reference scales the logits, squashes them into (-softcap, softcap)
    as `softcap * tanh(logits / softcap)`, adds the mask and takes the softmax
    in fp32 (modeling_gemma2.py:192-208). No fused kernel has that tanh, so
    this is flax's reference attention with the cap between the scaling and
    the mask; heads arrive already repeated and the mask already structural,
    as the reference path prepares them.
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
    return jnp.einsum('...hqk,...khd->...qhd', weights, value, precision=precision)


def scaled_dot_product_attention(query, key, value, dtype=None, precision=None,
                                 force_fp32_for_softmax=True, implementation=None,
                                 causal=False, sliding_window=None, mask=None, bias=None,
                                 sinks=None, softcap=None):
    """The one attention kernel path for every attention module.

    Inputs are [B, S, H, D]. Keys and values may carry fewer heads than the
    query (grouped-query attention); the paths that cannot group heads
    themselves get them repeated out. The value's head width may be narrower
    than the query's, which is DeepSeek's latent attention (`v_head_dim`
    against `qk_nope_head_dim + qk_rope_head_dim`): the reference path takes
    it as it is, and a fused path pads the value to the query's width and
    crops its own columns back out (`widen_value_heads`), so the selection
    below reads the query's width for either shape. The param trees of the
    callers never change with the implementation, so checkpoints are
    interchangeable across hardware:

    - 'reference', which a module field also spells None: flax reference
      attention (einsum + softmax), the portable default and the only path
      that reads dtype, precision and force_fp32_for_softmax.
    - 'auto': 'cudnn' where its kernel runs (a gpu backend, bf16 or fp16
      inputs, a query head width that is a multiple of 8 and at most 128, no
      softcap, and no `--xla_gpu_deterministic_ops` on the run), 'xla'
      anywhere else. Resolved per trace, so a config logged as 'auto' still
      runs on the next machine. A tpu backend gets xla here until the rule
      reads splash's own constraints.
    - 'xla' / 'cudnn': jax.nn.dot_product_attention, which dispatches to the
      fused cudnn flash kernel on supported GPUs. It takes no dtype, precision
      or softmax argument: the logits accumulate and the softmax runs in fp32
      whatever the inputs are. A dtype other than the inputs' own raises a
      ValueError, and so do a HIGH or HIGHEST precision and
      force_fp32_for_softmax=False. cudnn takes any sequence length
      (`cudnn_attention` pads an odd one), and only bf16 or fp16 inputs. A
      value wider than the query has no fused rewrite and raises, and so does
      an explicit 'cudnn' under `--xla_gpu_deterministic_ops`, whose backward
      pass XLA cannot execute (openxla/xla#46500).
    - 'tpu': the pallas splash kernel (`tpu_attention`), whose mask is a
      block-sparse descriptor built at trace time, so a causal or windowed
      long sequence costs its live blocks rather than its rectangle. The
      1/sqrt(d) scale goes onto the query, in the query's dtype, because the
      kernel has no scale argument of its own (the deleted EfficientAttention
      passed none to a kernel that wanted one, which inflated the logits by
      sqrt(d) and made its checkpoints poisonous). The older pallas flash
      kernel stays behind it for the calls splash has no form for: an
      additive bias, a mask that is a value of the trace, and a length that
      is not a multiple of 128. Off a tpu backend the kernel runs under
      pallas's interpreter, which computes the same numbers far more slowly;
      'auto' never selects it there.

    causal restricts query i to keys 0..i, top-left aligned like jax's
    is_causal; sliding_window=w narrows that to the w most recent keys. Both
    are structural, over the row index, so decoding against a KV cache passes
    `mask` instead (built by causal_attention_mask over the cache slots): a
    step's single query sits at the cache index, not at row 0. The fused
    kernels take causality and the window as flags, which saves the memory of
    a materialized mask; splash takes them as a descriptor, which additionally
    saves visiting the blocks they empty, and the pallas flash kernel behind
    it has no mask argument at all, so an explicit mask rides in there as an
    additive bias.
    `bias` is an additive float array broadcastable to [B, H, Q, K], added to
    the logits on every path; T5's relative position table travels in it, and
    it is the one argument splash has no form for.
    `sinks` holds one learned, value-free logit per query head. The reference
    and xla paths include it in the denominator; auto chooses xla, and the
    fused cudnn and tpu kernels refuse it.

    Under a mesh in context whose sequence axis is above one, the call runs
    through `sequence_parallel_attention`: the queries split over that axis,
    the keys and values are gathered whole, and a causal or masked call
    balances its work across the shards.

    `softcap` is Gemma 2's tanh on the scaled logits before the mask and the
    softmax. No fused kernel applies it, so a softcapped call runs
    `softcapped_attention` under both the reference and the xla
    implementation, honouring dtype, precision and force_fp32_for_softmax the
    way the reference path does; 'auto' resolves it to xla, and cudnn or tpu
    raise a ValueError that names the implementation.

    Whichever path ran, the result leaves here as the checkpoint name
    'attention_output', which is what lets a remat policy save it instead of
    replaying the kernel in the backward pass (`remat_block` in dit.py). The
    name is inert outside jax.checkpoint.
    """
    kernel = functools.partial(
        attention_kernel, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, implementation=implementation,
        sinks=sinks, softcap=softcap)
    shards = sequence_shards()
    if shards > 1:
        out = sequence_parallel_attention(
            kernel, query, key, value, shards, causal=causal,
            sliding_window=sliding_window, mask=mask, bias=bias)
    else:
        out = kernel(query, key, value, causal=causal, sliding_window=sliding_window,
                     mask=mask, bias=bias)
    return checkpoint_name(out, 'attention_output')


def precision_names(precision: PrecisionLike) -> frozenset[str]:
    """The names a `PrecisionLike` spells, upper case and unordered.

    flax's alias is four shapes at once: None, a string, a
    `jax.lax.Precision`, or a pair of either for the two operands. A fused
    path has to refuse HIGH and HIGHEST whichever shape the caller wrote, so
    this reads the union rather than asking each member what it is: the enum
    carries the name, a string is the name, and the pair is both operands'.
    """
    if precision is None:
        return frozenset()
    written = (precision,) if isinstance(precision, str | jax.lax.Precision) else precision
    return frozenset((one.name if isinstance(one, jax.lax.Precision) else one).upper()
                     for one in written if one is not None)


def attention_kernel(query, key, value, dtype=None, precision=None,
                     force_fp32_for_softmax=True, implementation=None,
                     causal=False, sliding_window=None, mask=None, bias=None, sinks=None,
                     softcap=None):
    """`scaled_dot_product_attention`'s kernel dispatch over whole sequences."""
    if sliding_window is not None and sliding_window < 1:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")
    if sinks is not None:
        if implementation not in (None, 'reference', 'auto', 'xla'):
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

    if implementation == 'auto':
        implementation = 'cudnn' if cudnn_runs(query, softcap) else 'xla'
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

    if implementation in (None, 'reference') or softcap is not None:
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
            dropout_rng=None, precision=precision,
            force_fp32_for_softmax=force_fp32_for_softmax, deterministic=True)

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

    # Every fused kernel runs one head width for the keys and the values, so a
    # narrower value rides in padded and its own columns come back out; the
    # widths a caller passes are static, so this costs no runtime branch.
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


# Splash's tile sizes, one constant for every block the kernel names. The
# forward kernel holds a [block_q, block_kv] fp32 logit tile, an fp32
# [block_q, head_dim] output accumulator and the q, k and v blocks in VMEM at
# once: at 512 that is 1 MiB of logits and a few hundred KiB of operands,
# double-buffered by the pipeline and still far inside a core's VMEM at the
# head widths a decoder has, while jax's own BlockSizes.get_default() of 128
# hands the MXU a sixteenth of that tile per pass. `splash_block_sizes`
# narrows it to a divisor of each sequence, which is what the mask blocking
# needs; a longer sequence therefore gets 512 and a short one gets itself.
SPLASH_BLOCK = 512
# The kernel tiles the key axis by lanes: the compute block must be a whole
# number of them (`{bkv_compute=} must be a multiple of {NUM_LANES=}`,
# splash_attention_kernel.py:970-971) and the mask blocking must divide both
# sequences (splash_attention_mask_info.py:565-571), so a length that is not
# a multiple of this has no legal block size and never reaches splash.
SPLASH_LANES = 128
# What the kernel accumulates in is fp32 whatever comes in: the logits carry
# preferred_element_type=float32 and the running max, sum and output stay
# fp32 to the last block, which is the reference softmax. These are the two
# input dtypes a TPU matmul takes; fp16 has no MXU path.
SPLASH_DTYPES = (jnp.bfloat16, jnp.float32)
# How much of an explicit boolean mask splash will carry. The array is read
# on the host while the executable is built and its unresolved blocks are
# stored dense inside it, so this bounds host and executable bytes, not
# device memory that grows with the batch. 4 Mi cells is a 16-head 512x512.
SPLASH_DENSE_MASK_CELLS = 1 << 22


def tpu_attention(query, key, value, bias, mask, causal, sliding_window, *,
                  interpret: bool):
    """The pallas TPU path: splash where its descriptor covers the call, the
    older pallas flash kernel everywhere else.

    Splash is the block-sparse kernel. Its mask is a descriptor built while
    the executable is, the blocks that descriptor empties are never visited,
    and a causal or windowed long sequence costs what its live blocks cost
    rather than what its rectangle does. What it has no form for is a value
    of the trace: there is no bias argument at all, and the mask has to be
    readable on the host. Flash takes both, as one additive [B, H, Q, K]
    array, and pays the whole rectangle for them, so it stays for exactly
    these calls:

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
    """The splash kernel over [B, S, H, D] arrays under `descriptor`.

    The kernel takes one example at a time, as [H, S, D] with the head width
    minor, so the batch rides in on a vmap and the seam is two transposes.
    It applies no scale of its own, unlike the flash kernel's `sm_scale`, so
    the 1/sqrt(d) goes onto the query in the query's own dtype: that is where
    flax's reference path puts it (`query / jnp.sqrt(depth)` in
    dot_product_attention_weights), which is what makes the two agree exactly
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
    """`SPLASH_BLOCK` narrowed to a divisor of each sequence, forward and
    backward.

    The greatest common divisor satisfies both of the kernel's stated rules
    at once: it divides its sequence, which the mask blocking requires, and
    it stays a multiple of `SPLASH_LANES`, which the key compute block
    requires, because the constant and the admitted lengths are both
    multiples of it. The backward blocks are filled in because a kernel built
    without them raises "Need to specify backward blocks." from inside its
    own vjp, which a training run would meet at its first gradient rather
    than at trace time.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel

    block_q, block_kv = math.gcd(SPLASH_BLOCK, q_len), math.gcd(SPLASH_BLOCK, kv_len)
    return splash_attention_kernel.BlockSizes(
        block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv,
        block_q_dkv=block_q, block_kv_dkv=block_kv, block_kv_dkv_compute=block_kv,
        block_q_dq=block_q, block_kv_dq=block_kv)


def splash_mask_descriptor(q_len: int, kv_len: int, heads: int, causal: bool,
                           sliding_window: int | None, mask):
    """Splash's mask for one call, or None when the call's mask is not one
    splash describes.

    The structural part is a function of the two indices, not an array:
    CausalMask and LocalMask carry the comparison the kernel evaluates per
    block, so the blocks they empty are dropped from the grid and nothing
    proportional to Q*K is stored. Dew's window is causal already
    (`causal_attention_mask` keeps k <= q before it narrows to the w most
    recent keys, which is what the xla path spells `local_window_size=(w-1,
    0)`), so a window replaces the causal flag here instead of sitting beside
    it. An explicit boolean mask has no such form: it is ANDed in as dense
    blocks, which is why only a concrete, small one is taken.
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
    """An explicit mask as one host [Q, K] boolean array per query head, or
    None when splash cannot carry it.

    Three things put a mask out of reach. It is a value of the trace: the
    descriptor is built while the executable is, so a mask that exists only
    as a tracer cannot be read, and a decode mask over cache slots and a
    packed batch's segment mask are both that. It has a batch axis wider than
    one: splash indexes its mask by head and by position and has no batch
    axis at all. Or it is large: the blocks the descriptor cannot resolve to
    all-on or all-off are stored dense inside the executable, so past
    `SPLASH_DENSE_MASK_CELLS` the additive-bias path is the cheaper one.

    numpy and the tracer type are imported here rather than at the module:
    this is the one place the mask leaves the trace, and splash's NumpyMask
    is a host array.
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
    """The pallas TPU flash kernel, with every mask as an additive bias.

    The kernel has no mask argument, so a window and an explicit mask become
    one [B, H, Q, K] float array of zeros and the dtype's minimum, added to
    the logits; only causality is a flag it takes. That array is the whole
    rectangle, which is what splash exists to avoid, so this path runs for
    the calls `tpu_attention` names and not for the others. The 1/sqrt(d)
    scale is the kernel's own `sm_scale` here.
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
    """Multi-head attention over a `[B, S, C]` or `[B, H, W, C]` input.

    causal makes it a decoder attention (query i sees keys 0..i). decode=True
    on a call runs it against a fixed-size KV cache instead, allocated at
    max_seq_len: the first call writes the whole prompt, later calls append one
    token each. Neither flag touches the param tree, so a model trained without
    either reloads into a decoding one unchanged. `freqs_cis` rotates the
    queries and keys of a self-attention call (`rotary_freqs` gives the pair);
    None leaves them unrotated.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_bias: bool = True
    force_fp32_for_softmax: bool = True
    qk_norm: bool = False  # RMSNorm on q/k per head (SD3-style bf16 logit safety)
    attention_impl: str | None = None  # an AttentionImpl, or None for 'reference'
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
            self.q_norm = nn.RMSNorm(dtype=self.dtype, name="q_norm")
            self.k_norm = nn.RMSNorm(dtype=self.dtype, name="k_norm")

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
    """A linear layer into the gated linear unit of Shazeer 2020
    (https://arxiv.org/abs/2002.05202): half the projection gates the other
    half through GELU. The hidden width is four times `dim`."""

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
    """GEGLU then a linear layer back to `dim`, diffusers' `FlaxFeedForward`.
    The checkpoint keys `net_0` and `net_2` are the indices the reference's
    Sequential gives the two layers."""

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
    """Self-attention, cross-attention over `context`, feed-forward, each
    pre-normed with a residual. `use_cross_only` drops the self-attention;
    `only_pure_attention` runs the cross-attention alone with no norm and no
    residual; the UNets' stages use it by default."""
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
    attention_impl: str | None = None

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
        self.norm1 = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)
        self.norm2 = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)
        self.norm3 = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)

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
    """One resolution stage's attention in a UNet, or `None` for a stage that
    has none.

    Every field is a `TransformerBlock` dial. The block's head width is the
    stage's channel count divided by `heads`, which the unet knows and a
    config does not, so there is no `dim_head` field. `dew.registry.from_record` builds one from a record at
    the build boundary, so a stage still arrives as `{"heads": 8}` from a
    command line or a run record, and a misspelled field raises there.

    `dtype` defaults to float32; `with_precision` writes the model's dtype
    into every stage. `precision` is the one field whose None means "the
    model's".
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


def stage_attention(stage: Stage, channels: int, attention_impl: str | None,
                    precision: PrecisionLike, name: str) -> "TransformerBlock":
    """The block a UNet stage's `Stage` describes, at the stage's channel
    count; a stage that names no precision takes the model's."""
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
    """A `BasicTransformerBlock` behind an optional projection into and out of
    `heads * dim_head`, dense (`use_linear_attention`) or a 1x1 convolution.
    Without the projection the block runs at the input width."""
    heads: int = 4
    dim_head: int = 32
    use_linear_attention: bool = True
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_projection: bool = False
    use_self_and_cross:bool = True
    only_pure_attention:bool = False
    force_fp32_for_softmax: bool = True
    attention_impl: str | None = None
    norm_inputs: bool = True
    explicitly_add_residual: bool = True
    norm_epsilon: float = 1e-4

    @nn.compact
    def __call__(self, x, context=None):
        inner_dim = self.heads * self.dim_head
        C = x.shape[-1]
        if self.norm_inputs:
            x = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)(x)
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
