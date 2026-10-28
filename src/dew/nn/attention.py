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
import math
from collections.abc import Callable
from typing import Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
import tokamax
from flax import linen as nn
from flax.linen.dtypes import canonicalize_dtype, promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as P

from dew.mesh import SEQUENCE_AXIS, LayoutRefused, logical_spec, mesh_axes
from dew.nn.dense import Dense, DenseGeneral
from dew.nn.dropout import Dropout

from .attention_kernels import ATTENTION, DECODE, SPLASH_LANES, TPU_BACKEND, AttentionCase, decodes
from .attention_sinks import sink_softmax
from .decode_geometry import folded_attention, folds
from .documents import (
    CrossDocuments,
    cross_document_mask,
    document_ids,
    document_offsets,
    query_order,
    with_documents,
)
from .flops import cross_attention_flops, projection_flops, self_attention_flops
from .inputs import cu_seqlens
from .kernels import decode_attention, generation, norm_swiglu
from .kernels.generation import bf16_dot_runs, gpu_placed
from .kernels.selection import forced, select, selected
from .kernels.templates import attention as template, attention_route
from .precision import (
    asks_default_precision,
    at_default_precision,
    at_least_fp32,
    capped_logits,
    configured_precision,
    float32_dot_general,
    float32_einsum,
    float32_products,
    operand_precision,
    rms_statistics,
    rounded_to,
    wide_cpu_arithmetic,
)
from .rope import rotate
from .sharding import (
    HEADS,
    KV_HEADS,
    SCORES,
    constrain,
    logical_axes,
    manual_map,
    sequence_shards,
    split_positions,
)
from .visibility import same_chunk, window_sides, windowed

AttentionImpl = Literal["auto", "reference", "xla", "cudnn", "triton", "flash", "tpu", "template"]
"""Names which kernel an attention call runs.

Every layer that carries the choice spells it the same way: a `ModelConfig`
field, a module's `attention_impl`, and `scaled_dot_product_attention`'s
`implementation`.

'reference' is the portable einsum and softmax, and the only path that reads
dtype, precision and force_fp32_for_softmax. 'xla' and 'cudnn' are
`jax.nn.dot_product_attention`'s own two. 'triton' is tokamax's Pallas-Triton
flash kernel (`triton_attention`). 'flash' is FlashAttention (`flash_attention`),
FlashAttention-3 on Hopper and FlashAttention-2 elsewhere, from the
dew_flash_attn wheel (docs/installation.md). 'tpu' is the
pallas splash kernel, with the older pallas flash kernel behind it for the
calls splash's mask descriptor cannot carry.
'template' is the attention template's route (`attention_route.attend`): the call's visibility as
spans, a softcap as a score hook and sinks as the initial state, on the template's Triton or Mosaic
mainloop on a GPU and its plain-JAX one elsewhere. 'auto' takes it on every GPU, where the call's
declared old route does not take it (`attention_kernels._attention_preferred`, `DECLARED`).
'auto', every module's default, resolves per trace (`resolve_implementation`),
so a configuration logged as 'auto' runs on the next machine too: the attention selector (`ATTENTION`,
`dew.nn.kernels.selection.select`) takes the call's declared old route where a receipt measured it faster,
else the template, where its requirements hold (each kernel's `SUPPORTS`), else the first of flash,
triton, cuDNN, splash, xla and the reference path that takes it; a call of few
positions over key lengths is the decode selector's (`DECODE`). Validity reaches
the kernels as coordinates (causality, a window, documents, key lengths); an
arbitrary mask only the generic paths take. A call off the kernel the target's
row measured fastest is logged at INFO once a process with the reason, and
every choice is reported as the target's route (`dew.nn.kernels.generation.route`).
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
    positions = jnp.atleast_2d(jnp.asarray(query_positions, jnp.int32))
    seen = visibility_mask(positions.shape[1], kv_len, causal=True, sliding_window=sliding_window,
                           query_slots=positions)
    # The filled slots ANDed after the spans, which then search no keys' ids: a batch-sharded cache's
    # validity takes no gather under explicit sharding (test_parallelism_matrix's data4_explicit).
    return seen if key_valid is None else seen & jnp.asarray(key_valid, bool)[:, None, None, :]


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


def structural_mask(query_positions, kv_len: int, causal: bool,
                    sliding_window: int | None) -> jax.Array | None:
    """The `[B, 1, T, S]` keys that causality and a window leave each query,
    from shared `[T]` or per-row `[B, T]` positions, or None when neither
    applies. A negative query position sees nothing."""
    if not causal and sliding_window is None:
        return None
    positions = jnp.atleast_2d(jnp.asarray(query_positions, jnp.int32))
    return visibility_mask(positions.shape[1], kv_len, causal, sliding_window, query_slots=positions)


def chunk_mask(query_positions, key_positions, chunk_size: int):
    """Boolean `[.., 1, T, S]` keeping keys in the query's chunk, both absolute.

    transformers' `chunked_overlay`: `kv // chunk == q // chunk`, positions
    counted from the sequence start, so a packed document's positions place
    its chunks from its own first token.
    """
    same = same_chunk(jnp.asarray(query_positions)[..., :, None], jnp.asarray(key_positions)[..., None, :],
                      chunk_size)
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
    if not causal and sliding_window is None:
        return mask
    return visibility_mask(query_length, key_length, causal, sliding_window, mask=mask)


def key_spans(query_length: int, key_length: int, causal: bool, sliding_window: int | None,
              segment_ids=None, key_value_seq_lengths=None, key_value_seq_starts=None, *, block_groups=None,
              image_groups=None, key_valid=None, key_runs=None, places=None, chunk: int | None = None,
              query_slots=None, rows: int | None = None) -> template.KeySpans:
    """The keys each query of a call sees as the template's spans (`template.spans`), from the
    coordinates `scaled_dot_product_attention` takes: causality, a window, a chunk of `chunk` places
    (`chunk_mask`), documents (`CrossDocuments` aligned to their last keys), key lengths and starts, block
    and image groups, and the keys' validity `key_valid` `[B, K]`, and each query's key runs `key_runs`
    `[B, Q, R, 2]`, whose union it may see (`template.spans`'s `runs`). `places` `[B, S]` are a call's own
    tokens' places a window or a chunk counts by (`counted_places`), by default their indices, or a pair,
    its queries' `[B, Q]` and its keys' `[B, K]`. `query_slots` `[B, Q]` are the queries' own keys in a
    cache call, -1 for padding (`template.spans`' `slots`). `rows` is the call's batch, by default the
    coordinates' own, or one where none carries it, which every row shares."""
    pair = places if isinstance(places, tuple) else (places, places)
    if rows is None:
        # The batch the coordinates broadcast to: a row's lengths and starts `[B]`, every other one's
        # leading axis where it has a row axis.
        rowed = [None if segment_ids is None else document_ids(segment_ids)[0], block_groups, image_groups,
                 key_valid, key_runs, pair[0], query_slots]
        batches = [(jnp.shape(x)[0],) for x in (key_value_seq_lengths, key_value_seq_starts) if x is not None]
        batches += [(jnp.shape(x)[0],) for x in rowed if x is not None and jnp.ndim(x) >= 2]
        rows = int(np.broadcast_shapes(*batches, (1,))[0])
    lengths, starts = (None if x is None else jnp.asarray(x, jnp.int32)
                       for x in (key_value_seq_lengths, key_value_seq_starts))
    positions = None
    if pair[0] is not None and pair[1] is not None:
        positions = (jnp.broadcast_to(jnp.atleast_2d(pair[0]), (rows, query_length)),
                     jnp.broadcast_to(jnp.atleast_2d(pair[1]), (rows, key_length)))
    valid = None if key_valid is None else jnp.broadcast_to(key_valid, (rows, key_length))
    slots = None if query_slots is None else jnp.broadcast_to(query_slots, (rows, query_length))
    return template.spans(query_length, key_length, rows, causal=causal, sliding_window=sliding_window,
                          chunk=chunk, positions=positions, segment_ids=segment_ids, lengths=lengths,
                          starts=starts, block_groups=block_groups, image_groups=image_groups, valid=valid,
                          runs=key_runs, slots=slots)


def visibility_mask(query_length: int, key_length: int, causal: bool, sliding_window: int | None, mask=None,
                    segment_ids=None, key_value_seq_lengths=None, key_value_seq_starts=None, *,
                    block_groups=None, image_groups=None, key_valid=None, key_runs=None,
                    places=None, chunk: int | None = None, query_slots=None) -> jax.Array:
    """The keys each query sees, `[B or 1, 1, Q, K]`: the call's spans (`key_spans`) as a mask
    (`template.dense`), narrowed by `mask`. It is the mask a call's coordinates mean on the paths
    whose kernel does not read them all."""
    seen = template.dense(key_spans(query_length, key_length, causal, sliding_window, segment_ids,
                                    key_value_seq_lengths, key_value_seq_starts, block_groups=block_groups,
                                    image_groups=image_groups, key_valid=key_valid, key_runs=key_runs,
                                    places=places, chunk=chunk, query_slots=query_slots))[:, None]
    return seen if mask is None else jnp.logical_and(seen, mask)


def visible_rows(query_length: int, key_length: int, causal: bool, sliding_window: int | None,
                 mask=None, segment_ids=None, key_value_seq_lengths=None,
                 key_value_seq_starts=None, *, block_groups=None, image_groups=None, key_valid=None,
                 key_runs=None, places=None) -> jax.Array | None:
    """Whether each query row sees a key, `[B or 1, Q, H or 1, 1]` over the
    output, or None where every row does.

    A row that sees no key reads nothing, so attention gives it exactly zero
    on every implementation (`scaled_dot_product_attention`): a softmax over
    no keys has no value, and a kernel left to it returns the mean of the
    values or, in Splash's backward, gradients from a uniform weight over
    masked keys. The call's spans decide a row (`template.reads`); an
    explicit mask is reduced over its keys, folded with them
    (`visibility_mask`). Without a mask, key bounds, key validity, counted
    places or another sequence's documents, a row sees its own key where the keys reach it,
    which the program knows as it traces, unless it is a packed row's padding.
    """
    if (mask is None and key_value_seq_lengths is None and key_value_seq_starts is None and key_valid is None
            and key_runs is None and places is None and not isinstance(segment_ids, CrossDocuments)):
        # The keys row i may see run from 0 to the last one; causality keeps 0 in reach, and a
        # window ends w - 1 keys before the row (`windowed`). A call's own documents keep each row's
        # own key, and block and image groups only add keys beside it.
        rows = np.arange(query_length)[None]
        last = np.full((1, 1), key_length - 1)
        reach = (last >= 0) if sliding_window is None else windowed(rows, last, sliding_window, causal=True)
        if segment_ids is None:
            return None if reach.all() else jnp.asarray(reach)[:, :, None, None]
        return (jnp.asarray(reach) & (jnp.asarray(segment_ids) != 0))[:, :, None, None]
    if mask is None:
        spans = key_spans(query_length, key_length, causal, sliding_window, segment_ids,
                          key_value_seq_lengths, key_value_seq_starts, block_groups=block_groups,
                          image_groups=image_groups, key_valid=key_valid, key_runs=key_runs, places=places)
        return template.reads(spans)[:, :, None, None]
    seen = jnp.any(visibility_mask(query_length, key_length, causal, sliding_window, mask, segment_ids,
                                   key_value_seq_lengths, key_value_seq_starts, block_groups=block_groups,
                                   image_groups=image_groups, key_valid=key_valid, key_runs=key_runs,
                                   places=places), -1)
    return jnp.swapaxes(seen, -1, -2)[..., None]


def with_key_lengths(mask: jax.Array | None, lengths: jax.Array, key_length: int) -> jax.Array:
    """`mask` narrowed to each row's first `lengths[b]` keys, `[B, 1, 1, K]`
    where there was no mask. This is the mask a `key_value_seq_lengths`
    call means on the paths whose kernel takes no lengths."""
    return visibility_mask(1, key_length, causal=False, sliding_window=None, mask=mask,
                           key_value_seq_lengths=lengths)


def with_key_starts(mask: jax.Array | None, starts: jax.Array | None, key_length: int) -> jax.Array | None:
    """`mask` narrowed to each row's keys from `starts[b]`, `[B, 1, 1, K]` where there was no mask:
    the mask a `key_value_seq_starts` call means on the paths whose kernel takes no starts. None
    starts leave `mask` as it is."""
    if starts is None:
        return mask
    return visibility_mask(1, key_length, causal=False, sliding_window=None, mask=mask,
                           key_value_seq_starts=starts)


def max_attention_logits(query: jax.Array, key: jax.Array, *, causal: bool = False,
                         sliding_window: int | None = None,
                         mask: jax.Array | None = None,
                         bias: jax.Array | None = None) -> jax.Array:
    """Return the largest pre-softmax logit per query head: `[batch, heads]`, at least fp32
    (`at_least_fp32` of the operands'), so a float64 model's maxima are float64.

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
    wide = at_least_fp32(jnp.result_type(query.dtype, key.dtype))
    logits = float32_einsum('...qhd,...khd->...hqk',
                            query.astype(wide) * jnp.asarray(scale, wide),
                            key.astype(wide), precision=operand_precision(wide))
    if bias is not None:
        logits = logits + bias.astype(wide)
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
    that recomputes the block (`mixer_base.RESIDUALS`). The wrapped function
    takes arrays first and static arguments last, so no flax lifting sits
    between the checkpoint and the module; forward values are unchanged.
    """
    return jax.checkpoint(normalize, policy=jax.checkpoint_policies.nothing_saveable,
                          static_argnums=static_argnums)


@functools.partial(normalized_in_fp32, static_argnums=(2, 3, 4, 5, 6, 7))
def _xla_rms_normalized(x, scale, epsilon: float, dtype, scale_offset: bool, scale_after_cast: bool,
                        fp32_statistics: bool, activation=None):
    """Normalize `x` by its root mean square, then apply `scale`.

    The families differ on which side of the cast the weight goes, which
    `scale_after_cast` picks. `scale` of None is the weightless norm.
    `fp32_statistics` of False computes the square, the mean and the
    normalization in the input's dtype. `activation` (None for none) runs
    on the result before its last cast to `dtype`, as `GroupNorm`'s does.
    """
    activated = (lambda y: y) if activation is None else activation
    if fp32_statistics:
        y = x.astype(at_least_fp32(x.dtype))
        y = y * rms_statistics(y, epsilon)[1]
    else:
        # The reference rounds every step to the input dtype. XLA fuses the
        # chain and carries fp32 between the ops, so each rounding is explicit.
        y = rounded_to(x * rounded_to(jax.lax.rsqrt(rounded_to(
            rounded_to(jnp.mean(rounded_to(jnp.square(x), x.dtype), axis=-1, keepdims=True), x.dtype)
            + epsilon, x.dtype)), x.dtype), x.dtype)
    if scale is None:
        # A pure normalization with no learned weight, as Gemma 4 norms
        # its values (modeling_gemma4.py, Gemma4RMSNorm with_scale=False).
        return activated(y).astype(dtype)
    weight = (1.0 + scale) if scale_offset else scale
    if scale_after_cast:
        # The reference casts the activations, then multiplies by its weight:
        # a bf16 product for a bf16 weight, an fp32 one for an fp32 master
        # that the next layer rounds. Rounding in place keeps both roundings
        # under jit, where XLA drops a narrowing cast that a widening one
        # follows, and keeps the weight's product and gradient in fp32.
        return rounded_to(activated(rounded_to(y, dtype) * weight), dtype).astype(dtype)
    return activated(y * weight).astype(dtype)


def rms_normalized(x, scale, epsilon: float, dtype, scale_offset: bool, scale_after_cast: bool,
                   fp32_statistics: bool, *, implementation: str = 'auto', axes=None):
    """Normalize the last axis and apply an optional vector weight.

    `scale_offset` adds one to the stored weight; `scale_after_cast` rounds
    the normalized activation before multiplying and rounds the product
    again. `fp32_statistics=False` retains the input-dtype arithmetic.
    `implementation` selects the generation's measured kernel by default;
    off a GPU, 'pallas' too runs XLA's arithmetic. The fused VJP retains only the
    narrow input, weight and per-row mean square, not fp32 activations.
    Unsupported variants run the checkpointed XLA reference instead
    (`norm_swiglu.rms_kernel`), including float64 and input-dtype statistics.
    `axes`, `x`'s logical axes, place the kernel's rows on a mesh; without
    them a split mesh runs XLA.
    """
    return stream_rms_normalized(x, None, scale, epsilon, dtype, scale_offset, scale_after_cast,
                                 fp32_statistics, implementation=implementation, axes=axes)[1]


def stream_rms_normalized(x, branch, scale, epsilon: float, dtype, scale_offset: bool,
                          scale_after_cast: bool, fp32_statistics: bool, *, implementation: str = 'auto',
                          axes=None, activation=None):
    """A pre-norm block's seam: the residual stream `x` plus `branch` (None
    adds nothing) and that stream's `rms_normalized`, as `(stream, normed)`.

    The caller carries the returned stream on, so the fused kernel
    (`norm_swiglu.stream_rms_norm`) owns both ends of the norm: its forward
    reads the two addends once and writes the sum beside the norm, and its
    reverse adds the stream's own cotangent to the norm's in the same pass.
    XLA's add and norm are the portable path and the oracle. `activation`
    (None for none) runs on the norm before its last cast, on XLA's path.
    """
    chosen = norm_swiglu.rms_kernel(implementation, x, scale, dtype, fp32_statistics, branch=branch,
                                    axes=axes, activated=activation is not None)
    def portable(x, branch, scale):
        stream = x if branch is None else x + branch
        return stream, _xla_rms_normalized(stream, scale, epsilon, dtype, scale_offset, scale_after_cast,
                                           fp32_statistics, activation)

    if chosen == 'pallas':
        return norm_swiglu.stream_rms_norm(x, branch, scale, epsilon, dtype, scale_offset, scale_after_cast,
                                           portable, axes=axes)
    return portable(x, branch, scale)


def gated_rms_normalized(x, gate, scale, epsilon: float, dtype, activation: str, *,
                         implementation: str = 'auto', axes=None):
    """Qwen3-Next's gated norm (`Qwen3NextRMSNormGated`,
    modeling_qwen3_next.py:57-74): `x` normalized in fp32, times `scale`,
    times `activation(gate)` ('silu' or 'sigmoid'), cast once to `dtype`.
    The fused kernel (`norm_swiglu.gated_rms_norm`) reads both inputs once a
    pass; XLA's arithmetic is the portable path and the oracle. `axes`, the
    inputs' logical axes, place the kernel's rows on a mesh.
    """
    def portable(x, gate, scale):
        wide = at_least_fp32(x.dtype)
        y = unweighted_rmsnorm(x.astype(wide), epsilon, implementation='xla')
        y = y if scale is None else y * scale.astype(wide)
        gate = gate.astype(wide)
        return (y * (jax.nn.silu(gate) if activation == 'silu' else jax.nn.sigmoid(gate))).astype(dtype)

    chosen = norm_swiglu.gated_rms_kernel(implementation, x, gate, scale, dtype, activation, axes=axes)
    if chosen == 'pallas':
        return norm_swiglu.gated_rms_norm(x, gate, scale, epsilon, dtype, activation, portable, axes=axes)
    return portable(x, gate, scale)


def _moment_normalized(x, scale, bias, axes: tuple[int, ...], epsilon: float, requested):
    """`x` normalized over `axes`, reduced in the module's `requested` dtype,
    else `x`'s, at least fp32, as flax lays the computation out
    (flax.linen.normalization._compute_stats) but for the variance: the
    centered mean square E[(x - E[x])^2], as torch's GroupNorm takes it, where
    flax takes E[x^2] - E[x]^2. That difference cancels where a group's mean
    dwarfs its spread, and its gradient carries the loss: FLUX.2's published
    VAE's input gradient sat twice torch's distance from float64, its worst
    seeds 6 to 10 times. The centered form has no such cancellation, at one
    more pass over the group. The weight is folded into the inverse deviation
    before it meets the centered activations. `scale` and `bias` (None for
    none) broadcast against `x`'s trailing axes."""
    statistics = at_least_fp32(x.dtype if requested is None else requested)
    reduced = x.astype(statistics)
    mean = jnp.mean(reduced, axis=axes, keepdims=True)
    variance = jnp.mean(jax.lax.square(reduced - mean), axis=axes, keepdims=True)
    scaling = jax.lax.rsqrt(variance + epsilon)
    if scale is not None:
        scaling = scaling * scale
    # flax centers the input as it arrived, promoted with the statistics' dtype.
    y = (x.astype(jnp.promote_types(x.dtype, statistics)) - mean) * scaling
    return y if bias is None else y + bias


@functools.partial(normalized_in_fp32, static_argnums=(3, 4, 5, 6, 7))
def group_normalized(x, scale, bias, groups: int, epsilon: float, activation, requested, dtype):
    """`x` `[batch, ..., channels]` normalized over each of `groups` channel
    groups and every axis but the batch, as flax's `GroupNorm` lays it out, its
    variance centered as torch's (`_moment_normalized`). `activation` (None for none) runs on the result
    before its one cast to `dtype`."""
    grouped = x.reshape(*x.shape[:-1], groups, x.shape[-1] // groups)
    scale, bias = (None if value is None else value.reshape(grouped.shape[-2:]) for value in (scale, bias))
    y = _moment_normalized(grouped, scale, bias, (*range(1, x.ndim - 1), x.ndim), epsilon, requested)
    y = y.reshape(x.shape)
    return (y if activation is None else activation(y)).astype(dtype)


def l2_normalized(x, eps: float = 1e-12):
    """torch's `F.normalize`: `x` over its L2 norm on the last axis, the norm
    held at `eps` at least.

    The norm is the root of the squared norm held at `eps`², the same value,
    so a zero vector's gradient is zero: the root of a sum that is zero has
    an infinite slope, which the floor's zero gradient would turn into NaN.
    The norm is then held at `eps` again, a no-op that keeps the division off
    a bare square root, which XLA rewrites to a product with rsqrt, so the
    quotient rounds as torch's does."""
    norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(x), axis=-1, keepdims=True), eps * eps))
    return x / jnp.maximum(norm, eps)


def unweighted_rmsnorm(x, eps: float, *, implementation: str = 'auto'):
    """Normalize the last axis by its root mean square, with no learned weight.

    The reduction runs in fp32 and the inverse deviation is cast back
    before it multiplies, which is what `DeepseekV4UnweightedRMSNorm`
    (modeling_deepseek_v4.py:66-72) and its GLM twin do; the fused RMSNorm
    takes it where the target's row names it, rounding the inverse as they do
    (`norm_swiglu.rms_norm`'s `inverse_rounded`).
    """
    if norm_swiglu.rms_kernel(implementation, x, None, x.dtype, fp32_statistics=True) == 'pallas':
        def portable(x, scale):
            return unweighted_rmsnorm(x, eps, implementation='xla')

        return norm_swiglu.rms_norm(x, None, eps, x.dtype, offset=False, after_cast=False, portable=portable,
                                    inverse_rounded=True)
    fp32 = x.astype(at_least_fp32(x.dtype))
    inverse = rms_statistics(fp32, eps)[1]
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
    keeps the input in its own dtype (`normalized_in_fp32`). `activation`
    (None for none) runs on the norm before its last cast, as `GroupNorm`'s.
    `groups` takes the statistics over each of that many equal, contiguous
    parts of the axis, under one weight over the whole axis, as
    `K2HorizonRMSNorm` does (modeling_k2_horizon.py:613-636).
    """
    epsilon: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    with_scale: bool = True
    fp32_statistics: bool = True
    dtype: Dtype | None = None
    implementation: str = 'auto'
    activation: Callable[[jax.Array], jax.Array] | None = None
    groups: int = 1

    def __call__(self, x, *, axes=None):
        """The norm of `x`. `axes`, `x`'s logical axes, place the fused kernel on a mesh."""
        return self.seam(x, axes=axes)[1]

    @nn.compact
    def seam(self, x, branch=None, *, axes=None):
        """A pre-norm block's residual seam: `(stream, normed)` for the stream
        `x + branch` (`x` for a None branch) and its norm, whose fused kernel
        owns the add and the stream's cotangent (`stream_rms_normalized`)."""
        dtype = self.dtype if self.dtype is not None else x.dtype
        scale = self.param(
            'scale',
            nn.initializers.zeros if self.scale_offset else nn.initializers.ones,
            (x.shape[-1],), jnp.float32) if self.with_scale else None
        if self.groups == 1:
            return stream_rms_normalized(x, branch, scale, self.epsilon, dtype, self.scale_offset,
                                         self.scale_after_cast, self.fp32_statistics,
                                         implementation=self.implementation, axes=axes,
                                         activation=self.activation)
        if x.shape[-1] % self.groups:
            raise ValueError(f"a {x.shape[-1]}-wide norm splits into no {self.groups} equal groups")

        def grouped(value):
            return None if value is None else value.reshape(*value.shape[:-1], self.groups, -1)

        # The residual summed first, as it broadcasts; then the groups' own last axis, which the one
        # norm body reduces over under the weight laid out alike, the width's logical axis naming
        # the groups and each group's own span none.
        stream = x if branch is None else x + branch
        _, normed = stream_rms_normalized(
            grouped(stream), None, grouped(scale), self.epsilon, dtype, self.scale_offset,
            self.scale_after_cast, self.fp32_statistics, implementation=self.implementation,
            axes=None if axes is None else (*axes, None), activation=self.activation)
        return stream, normed.reshape(stream.shape)


class LayerNorm(nn.remat(nn.LayerNorm, policy=jax.checkpoint_policies.nothing_saveable)):
    """flax's `nn.LayerNorm`, rematerialized saving nothing: its backward pass keeps the input as
    it arrived and its parameters, not the two fp32 copies flax's own differentiation keeps
    (2.35 GiB of them on one SimpleDiT step). The parameter tree, the arithmetic
    and the module's name are flax's, so checkpoints and scopes move both ways."""


class GroupNorm(nn.Module):
    """Flax's `nn.GroupNorm` over the last axis, with its parameter tree, its
    variance centered as torch's GroupNorm takes it (`_moment_normalized`),
    and an `activation` applied to its fp32 output before the one cast to
    `dtype`.

    Diffusers' norms keep their fp32 affine output into the activation and
    the next layer casts its result (vae_flax.py:154 and its UNet), so a
    norm that rounded first and activated after would round where they do
    not. The backward pass keeps the input as it arrived
    (`normalized_in_fp32`), so the activation's fp32 input is recomputed,
    not stored.
    """
    num_groups: int = 32
    epsilon: float = 1e-6
    activation: Callable[[jax.Array], jax.Array] | None = None
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        width = (x.shape[-1],)
        scale = self.param('scale', nn.initializers.ones, width, self.param_dtype)
        bias = self.param('bias', nn.initializers.zeros, width, self.param_dtype)
        return group_normalized(x, scale, bias, self.num_groups, self.epsilon, self.activation, self.dtype,
                                canonicalize_dtype(x, scale, bias, dtype=self.dtype))


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
    tests/test_kernels.py pins the equality with the xla path. On a mesh it
    runs per shard (`_per_shard`).
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

    def local(query, key, value, bias=None, mask=None, key_value_seq_lengths=None):
        # A window of (l, r) keeps the l keys before the query and the r after it
        # on both the xla and the cudnn path, the sides `window_sides` counts.
        return jax.nn.dot_product_attention(
            query, key, value, bias=bias, mask=mask, is_causal=causal,
            key_value_seq_lengths=key_value_seq_lengths,
            local_window_size=None if sliding_window is None else window_sides(causal, sliding_window),
            implementation='cudnn')

    out = _per_shard(local, query, key, value,
                    bias=(None if bias is None else _four_dimensional(bias), SCORES),
                    mask=(None if mask is None else _four_dimensional(mask), SCORES),
                    key_value_seq_lengths=(kv_lengths, ("activation_batch",)))
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
                                sliding_window, mask, bias, sinks, key_value_seq_lengths=None,
                                dropout_rate=0.0, dropout_seed=None):
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
    A dropping call drops what one device would in either (`template.DropPlaces`).
    """
    tensor = tensor_shards(query)
    heads, kv_heads = query.shape[-2], key.shape[-2]
    masked = causal or sliding_window is not None or mask is not None
    exchangeable = (heads % (tensor * shards) == 0
                    and query.shape[-3] % shards == 0 and key.shape[-3] % shards == 0)
    run = (exchanged_heads_attention
           if exchangeable and (masked or all_to_all_moves_less(heads, kv_heads, tensor, shards))
           else gathered_keys_attention)
    return run(kernel, query, key, value, shards, causal=causal,
               sliding_window=sliding_window, mask=mask, bias=bias, sinks=sinks,
               key_value_seq_lengths=key_value_seq_lengths, dropout_rate=dropout_rate,
               dropout_seed=dropout_seed)


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


def tensor_shards(query) -> int:
    """How many ways the mesh's tensor axes split the query heads."""
    mesh = jax.sharding.get_abstract_mesh()
    return math.prod(mesh.shape[axis]
                     for axis in mesh_axes(_entry(logical_spec(HEADS, query.shape), 2)))


def shard_heads(query) -> int:
    """The query heads one shard holds (`tensor_shards`)."""
    return query.shape[-2] // tensor_shards(query)


def split_with_queries(key, value, split: int):
    """`key` and `value` with their heads repeated to the least common multiple
    of their count and `split`, the ways the mesh splits the query heads, so
    each shard holds the key heads its query heads read."""
    heads = math.lcm(key.shape[-2], split)
    return repeat_kv_heads(key, heads), repeat_kv_heads(value, heads)


def _per_shard(local, query, key, value, *, coordinates: bool = False, **extras):
    """`local(query, key, value, **extras)` on each shard's rows and heads
    (`manual_map`). The key heads a shard reads are the ones its key shard holds,
    at their offset in that shard, where the key heads' split begins with the query
    heads' split; a key-head split that does not is set aside, and the key heads are
    held whole on every shard. Where the tensor axes split the query heads more ways
    than the key heads, each shard takes its share of the repeats `split_with_queries`
    makes, so the key heads' gradient is one sum across the shards in the map's
    transpose. Whole off a mesh, or inside a map that already holds every axis manual
    (the sequence exchanges), where the tensor axes split nothing. A Pallas kernel has
    no partitioning rule, and jax's cuDNN attention partitions through
    `custom_partitioning`, whose lowering on several devices names the partitioner by
    a process address (`str(id(...))` in its backend config), which the persistent
    cache key keeps, so every fresh process would compile again; under a fully manual
    map jax lowers neither. `extras` maps each further argument to its array, or None,
    and the logical axes it splits on: a dimension named for the rows or the query heads splits as the
    query's own rows or heads do (`query_split`), whatever the rules give that name alone, so a shard's
    coordinates are its own rows' and heads' where axes contend for them.
    With `coordinates`, `local` also takes `global_rows` and `global_heads`, the shard's global batch rows
    and query heads from the query's own split, so a draw keyed on them is the same whatever the rules give
    the rows and heads alone.
    """
    given = {name: pair for name, pair in extras.items() if pair[0] is not None}
    queries, keys = logical_spec(HEADS, query.shape), logical_spec(KV_HEADS, key.shape)
    query_axes, key_axes = mesh_axes(_entry(queries, 2)), mesh_axes(_entry(keys, 2))
    row_axes = mesh_axes(_entry(queries, 0))
    if key_axes and query_axes[: len(key_axes)] != key_axes:
        keys, key_axes = P(*keys[:2]), ()
    split, kv_heads = tensor_shards(query), key.shape[-2]
    heads = math.lcm(kv_heads, split)
    share, copies = heads // split, heads // kv_heads
    names = tuple(given)

    def shard_start(axes, extent):
        # Where a shard's extent begins along the mesh axes that split it (0 where no axis does).
        return jax.lax.axis_index(axes) * extent if axes else 0

    def shard(query, key, value, *arrays):
        if copies > 1 or key.shape[-2] != share:
            # This shard's share of the repeated key heads (head n is key head n // copies), read from the key
            # heads it holds, which begin at its own key shard's offset.
            read = (jax.lax.axis_index(query_axes) * share + jnp.arange(share)) // copies
            if key_axes:
                read = read - jax.lax.axis_index(key_axes) * key.shape[-2]
            key, value = (jnp.take(array, read, axis=-2) for array in (key, value))
        arguments = dict(zip(names, arrays, strict=True))
        if coordinates:
            rows_at = shard_start(row_axes, query.shape[0])
            arguments["global_rows"] = rows_at + jnp.arange(query.shape[0], dtype=jnp.int32)
            heads_at = shard_start(query_axes, query.shape[2])
            arguments["global_heads"] = heads_at + jnp.arange(query.shape[2], dtype=jnp.int32)
        return local(query, key, value, **arguments)

    # An extra may be a tree of arrays (`CrossDocuments`), each leaf split on its axes.
    specs = (queries, keys, keys,
             *(jax.tree.map(lambda leaf, axes=axes: query_split(axes, leaf, queries, query.shape), x)
               for x, axes in given.values()))
    return manual_map(shard, specs, queries)(query, key, value, *(x for x, _ in given.values()))


def query_split(axes, leaf, queries: P, shape: tuple[int, ...]) -> P:
    """The spec of a per-shard argument `leaf` whose dimensions `axes` names, from the query's own split
    `queries` of its `shape`: a dimension of the rows, the queries or the query heads takes the query's,
    where it is as long (a broadcast dimension of 1 stays whole), any other none. Every manual map over
    attention's rows splits its row-scoped arguments so (`_per_shard`, the sequence exchanges, local
    attention over a sequence), so a shard's masks, lengths and ids are its own rows' where axes contend for
    them, whatever the rules give those names alone."""
    of = {"activation_batch": 0, "activation_length": 1, "activation_heads": 2}
    return P(*(_entry(queries, of[axis]) if axis in of and length == shape[of[axis]] else None
               for axis, length in zip(axes, jnp.shape(leaf), strict=False)))


def _four_dimensional(x):
    """A mask or a bias `[.., Q, K]` broadcastable to `[B, H, Q, K]`, as four
    dimensions; a broadcast dimension of 1 divides by no mesh axis, so the
    rule table leaves it whole."""
    return x.reshape((1,) * (4 - x.ndim) + x.shape)


def exchanged_heads_attention(kernel, query, key, value, shards: int, *, causal,
                              sliding_window, mask, bias, sinks, key_value_seq_lengths=None,
                              dropout_rate=0.0, dropout_seed=None):
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
    attended alike where it does not. A dropping call draws each shard's
    weights at their global rows and heads, its sequences whole
    (`template.DropPlaces`), so it drops what one device would.
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
    key, value = split_with_queries(key, value, split)
    keys = logical_spec(KV_HEADS, key.shape)

    # After the all-to-all a shard holds its tensor shard's heads split again
    # over the sequence axis; a mask, a bias or sinks with a head dimension
    # arrive split the same way, their query and key dimensions whole.
    exchanged_heads = (*tensor, SEQUENCE_AXIS)
    extras = {}
    for name, x in (('mask', mask), ('bias', bias)):
        if x is not None:
            x = _four_dimensional(x)
            # Its rows split as the query's own (`query_split`), its heads as the exchange leaves them.
            rows = _entry(queries, 0) if x.shape[0] == query.shape[0] else None
            extras[name] = (x, P(rows, exchanged_heads if x.shape[1] == heads else None))
    if sinks is not None:
        extras['sinks'] = (sinks, P(exchanged_heads))
    if key_value_seq_lengths is not None:
        extras['key_value_seq_lengths'] = (key_value_seq_lengths, query_split(
            ("activation_batch",), key_value_seq_lengths, queries, query.shape))
    if dropout_seed is not None:
        # Each shard's global rows and heads, split as its query rows and the exchanged heads.
        extras['dropout_seed'] = (dropout_seed, P(None))
        extras['dropout_rows'] = (jnp.arange(query.shape[0]), P(_entry(queries, 0)))
        extras['dropout_heads'] = (jnp.arange(heads), P(exchanged_heads))

    def local(query, key, value, *arrays):
        query, key, value = (jax.lax.all_to_all(x, SEQUENCE_AXIS, 2, 1, tiled=True)
                             for x in (query, key, value))
        arguments = dict(zip(extras, arrays, strict=True))
        if dropout_seed is not None:
            rows, heads = arguments.pop('dropout_rows'), arguments.pop('dropout_heads')
            arguments.update(dropout_rate=dropout_rate, dropout_places=template.DropPlaces(
                rows, heads, jnp.arange(query.shape[1]), jnp.arange(key.shape[1])))
        out = kernel(query, key, value, causal=causal, sliding_window=sliding_window, **arguments)
        return jax.lax.all_to_all(out, SEQUENCE_AXIS, 1, 2, tiled=True)

    exchanged = manual_map(
        local, (queries, keys, keys, *(spec for _, spec in extras.values())), queries)
    return exchanged(query, key, value, *(x for x, _ in extras.values()))


def gathered_keys_attention(kernel, query, key, value, shards: int, *, causal,
                            sliding_window, mask, bias, sinks, key_value_seq_lengths=None,
                            dropout_rate=0.0, dropout_seed=None):
    """Run `kernel` on each shard's slice of the queries against the whole
    keys and values, which every shard gathers over the sequence axis.

    Any head count and length, at the cost of every shard holding every key and
    value. Batch rows and heads split as `exchanged_heads_attention` splits
    them where the query heads divide by the tensor axis, and heads stay whole
    otherwise. A query length the shards do not divide is padded and the
    padding dropped; such a key length arrives whole. A causal, windowed or
    masked call stripes its queries for equal work and carries their positions
    in the mask, which the kernel reads in place of its causal flag, so the mask
    and the caller's rotary angles stay exact. A dropping call draws each
    shard's weights at their global rows, heads and queries, its keys whole
    (`template.DropPlaces`), so it drops what one device would.
    """
    q_len, kv_len = query.shape[1], key.shape[1]
    positions = jnp.arange(q_len)
    key, value = split_with_queries(key, value, tensor_shards(query))

    reordered = causal or sliding_window is not None or mask is not None
    if reordered:
        def rows_in_order(x: jax.Array | None) -> jax.Array | None:
            # A broadcast query row has the same value in either order.
            if x is not None and x.ndim >= 2 and x.shape[-2] == q_len:
                return stripe(x, shards, axis=-2)
            return x

        query = stripe(query, shards)
        positions = stripe(positions, shards, axis=0)
        mask, bias = rows_in_order(mask), rows_in_order(bias)
        structural = structural_mask(
            stripe(jnp.arange(q_len), shards, axis=0), kv_len, causal, sliding_window)
        if structural is not None:
            mask = structural if mask is None else jnp.logical_and(mask, structural)
    else:
        padding = -q_len % shards
        query = _pad_rows(query, padding)
        positions = jnp.pad(positions, (0, padding))
        if bias is not None and bias.ndim >= 2 and bias.shape[-2] == q_len:
            bias = jnp.pad(bias, ((0, 0),) * (bias.ndim - 2) + ((0, padding), (0, 0)))

    # A key length the shards do not divide takes no sequence axis, and
    # arrives whole instead of gathered; so do no keys, whose all-gather
    # does not lower.
    queries, keys = logical_spec(HEADS, query.shape), logical_spec(KV_HEADS, key.shape)
    if not kv_len:
        keys = P(_entry(keys, 0), None, *keys[2:])
    gathered = SEQUENCE_AXIS in mesh_axes(_entry(keys, 1))
    extras = {}
    for name, x in (('mask', mask), ('bias', bias)):
        if x is not None:
            x = _four_dimensional(x)
            extras[name] = (x, query_split(SCORES, x, queries, query.shape))
    if sinks is not None:
        extras['sinks'] = (sinks, query_split(("activation_heads",), sinks, queries, query.shape))
    if key_value_seq_lengths is not None:
        extras['key_value_seq_lengths'] = (key_value_seq_lengths, query_split(
            ("activation_batch",), key_value_seq_lengths, queries, query.shape))
    if dropout_seed is not None:
        # Each shard's global rows, heads and (striped) queries, split as its query's own.
        extras['dropout_seed'] = (dropout_seed, P(None))
        extras['dropout_rows'] = (jnp.arange(query.shape[0]), P(_entry(queries, 0)))
        extras['dropout_heads'] = (jnp.arange(query.shape[2]), P(_entry(queries, 2)))
        extras['dropout_queries'] = (positions, P(_entry(queries, 1)))

    def local(query, key, value, *arrays):
        if gathered:
            key, value = (jax.lax.all_gather(x, SEQUENCE_AXIS, axis=1, tiled=True)
                          for x in (key, value))
        arguments = dict(zip(extras, arrays, strict=True))
        if dropout_seed is not None:
            rows, heads, queries = (arguments.pop(name) for name in
                                    ('dropout_rows', 'dropout_heads', 'dropout_queries'))
            arguments.update(dropout_rate=dropout_rate, dropout_places=template.DropPlaces(
                rows, heads, queries, jnp.arange(key.shape[1])))
        return kernel(query, key, value, causal=False, sliding_window=None, **arguments)

    attended = manual_map(
        local, (queries, keys, keys, *(spec for _, spec in extras.values())), queries)
    out = attended(query, key, value, *(x for x, _ in extras.values()))
    return unstripe(out, shards) if reordered else out[:, :q_len]


def cudnn_runs(query, softcap=None) -> bool:
    """Whether cudnn's fused kernel runs a query of this dtype and width here: the target takes it
    (`CUDNN_SUPPORTS`) and this process places it on its own GPU (`gpu_placed`), which a CUDA
    library call needs."""
    case = attention_case(query, query, softcap=softcap)
    return gpu_placed() and not ATTENTION.kernel('cudnn').failed(case)


def flash_attention(query, key, value, causal: bool, sliding_window=None, sinks=None, segment_ids=None,
                    max_segments: int | None = None):
    """FlashAttention (dew_flash_attn, kernels/flash_attn) over `[B, S, H, D]`
    arrays at jax.nn's default scale: FlashAttention-3 on Hopper and
    FlashAttention-2 elsewhere. A window keeps the keys `window_sides` counts,
    and `sinks` `[H]` puts one value-free logit a head in each row's
    denominator. Packed rows, `segment_ids` `[B, S]` with at most
    `max_segments` documents a row, run as the kernel's varlen sequences: each
    document, and each row's padding after them, one sequence over the
    flattened tokens. Its causal mask is aligned to the last key, where
    jax.nn's is aligned to the first, so a causal call is square, or reads
    another sequence's documents, each document's queries aligned to its last
    keys (`CrossDocuments`), which it orders by id. On a mesh it
    runs per shard (`_per_shard`). What it takes is its requirements'
    (`attention_kernels.FLASH_SUPPORTS`), which the selector holds a call to before it runs here."""
    flash = importlib.import_module('dew_flash_attn')
    window = (-1, -1) if sliding_window is None else window_sides(causal, sliding_window)

    def local(query, key, value, sinks=None, segment_ids=None, key_segment_ids=None):
        if segment_ids is None or max_segments is None:
            return flash.flash_mha(query, key, value, is_causal=causal, window_size=window, sinks=sinks)
        rows, length = segment_ids.shape
        order = None
        if key_segment_ids is not None:
            # Another sequence's documents run as packed sequences in id order, padding after them:
            # the keys arrive so (a cache's), and the queries go so and come back in their own.
            order = query_order(segment_ids)
            query = jnp.take_along_axis(query, order[:, :, None, None], axis=1)
            segment_ids = jnp.take_along_axis(segment_ids, order, axis=1)
        flat = (query.reshape(rows * length, *query.shape[2:]),
                *(x.reshape(-1, *x.shape[2:]) for x in (key, value)))
        if key_segment_ids is None:
            # A call's own documents are its runs, whatever their ids.
            changes = segment_ids[:, 1:] != segment_ids[:, :-1]
            begins = jnp.concatenate([jnp.ones((rows, 1), bool), changes], axis=1)
            offsets, own_keys = cu_seqlens(begins, rows * (max_segments + 1)), {}
        else:
            # Another sequence's documents pair with the queries' by id, sequence i each side's
            # document i. FA3 takes a positive bound on the keys' lengths, which an empty key array
            # has none of.
            offsets, key_offsets = (document_offsets(ids, max_segments)
                                    for ids in (segment_ids, key_segment_ids))
            own_keys = {"key_cu_seqlens": key_offsets, "key_max_seqlen": max(1, key_segment_ids.shape[1])}
        out = flash.flash_mha_varlen(*flat, offsets, length, is_causal=causal, window_size=window,
                                     sinks=sinks, **own_keys)
        out = out.reshape(rows, length, *out.shape[1:])
        if order is None:
            return out
        back = jnp.zeros_like(order).at[jnp.arange(rows)[:, None], order].set(jnp.arange(length)[None, :])
        return jnp.take_along_axis(out, back[:, :, None, None], axis=1)

    queries, keys = (None, None) if segment_ids is None else document_ids(segment_ids)
    if queries is not None and keys is not None and (
            query.shape[:-2] != queries.shape or key.shape[:-2] != keys.shape
            or value.shape[:-2] != keys.shape):
        # Packed rows run flattened into one run of tokens, where rows of another count or length can
        # flatten to as many tokens and be read as the next row's.
        raise ValueError(
            f"packed flash attention reads each token by its document's id: queries {query.shape[:-2]} "
            f"beside ids {queries.shape}, keys {key.shape[:-2]} and values {value.shape[:-2]} beside ids "
            f"{keys.shape}")
    return _per_shard(local, query, key, value, sinks=(sinks, ("activation_heads",)),
                      segment_ids=(queries, ("activation_batch", None)),
                      key_segment_ids=(keys if isinstance(segment_ids, CrossDocuments) else None,
                                       ("activation_batch", None)))


def triton_attention(query, key, value, causal: bool):
    """tokamax's Pallas-Triton flash kernel over `[B, S, H, D]` arrays, at
    tokamax's heuristic config, with jax.nn's default scale of 1/sqrt(D).

    It takes grouped key heads and any length itself. A Pallas call has no
    partitioning rule, so on a mesh it runs per shard (`_per_shard`).

    The queries reach it scaled once, at its scale 1: its forward and dQ
    sweep scale the queries but its dK/dV sweep the keys (tokamax 0.0.15,
    pallas_triton_vjp), so at large logits that sweep's scores round apart
    from the forward's and the weights it rebuilds from the forward's
    residuals go off by their difference's exponential (row 1676: e^76 at
    logits near 1e4, a MeanFlow step's key and value gradients 1e35 times
    xla's error).
    """

    return _per_shard(lambda query, key, value: tokamax.dot_product_attention(
        query * query.shape[-1] ** -0.5, key, value, scale=1.0, is_causal=causal,
        implementation='triton'), query, key, value)


def template_attention(query, key, value, causal: bool, sliding_window, *, sinks, softcap, segment_ids,
                       lengths, starts, block_groups=None, image_groups=None, key_valid=None, key_runs=None,
                       places=None, dropout_rate=0.0, dropout_seed=None, dropout_places=None, bias=None):
    """The attention template's route (`attention_route.attend`) over the call's visibility as spans
    (`key_spans`): its causality, window, documents (`segment_ids`, or `CrossDocuments` aligned to their last
    keys), key lengths and starts, block and image groups, and key validity and counted places, with
    probability dropout at `dropout_rate` drawn from `dropout_seed`. A softcap is a score hook, a `bias` one
    term a key `[B or 1, H or 1, 1, K]` (ALiBi's, over the keys' logical places) a key score hook added after
    it (`template.key_biased`), and sinks are the softmax's initial state. On a mesh it runs per shard
    (`_per_shard`), as the decode kernel does: the key heads repeat until the tensor axes split them, the
    sinks split with their heads, and the ids, lengths, starts, groups, validity and places with their rows.
    Its requirements (`attention_kernels.TEMPLATE_SUPPORTS`) keep it off a sharded sequence axis.
    """
    if sinks is not None and sinks.shape != (query.shape[2],):
        raise ValueError(f"sinks must have one logit per query head, got {sinks.shape}")

    def local(query, key, value, sinks=None, segment_ids=None, lengths=None, starts=None,
              block_groups=None, image_groups=None, key_valid=None, key_runs=None, places=None,
              global_rows=None, global_heads=None, seed=None, bias=None):
        batch, queries, keys = query.shape[0], query.shape[1], key.shape[1]
        spans = key_spans(queries, keys, causal, sliding_window, segment_ids, lengths, starts,
                          block_groups=block_groups, image_groups=image_groups, key_valid=key_valid,
                          key_runs=key_runs,
                          places=places, rows=batch)
        drop = None
        if seed is not None and dropout_places is not None:
            # A sequence exchange's slice draws at the call's global places (`template.DropPlaces`).
            drop = template.Dropout(seed, dropout_places.rows, dropout_places.heads, dropout_rate,
                                    dropout_places.queries, dropout_places.keys)
        elif seed is not None and global_rows is not None and global_heads is not None:
            drop = template.Dropout(seed, global_rows, global_heads, dropout_rate)
        capped = None if softcap is None else functools.partial(template.capped, softcap)
        hooks = template.Hooks(
            score_mod=capped if bias is None else functools.partial(template.key_biased, capped),
            score_params=() if bias is None else (template.Param(bias[:, :, 0], template.Kind.KEY),),
            initial=None if sinks is None else template.sinks(sinks, batch, queries, value.shape[-1]),
            drop=drop)
        return attention_route.attend(query, key, value, spans, hooks)

    # The GPU arms are Pallas launches, which have no partitioning rule, so each shard runs them on its own
    # rows and heads. A dropping call's draw takes each shard's global rows and heads from the query's own
    # split (`_per_shard`'s coordinates), so it does not depend on the mesh or on the rules of the other axes.
    rows = ("activation_batch",)
    return _per_shard(local, query, key, value, coordinates=dropout_seed is not None,
                      sinks=(sinks, ("activation_heads",)),
                      segment_ids=(segment_ids, ("activation_batch", None)), lengths=(lengths, rows),
                      starts=(starts, rows), block_groups=(block_groups, ("activation_batch", None)),
                      image_groups=(image_groups, ("activation_batch", None)),
                      key_valid=(key_valid, ("activation_batch", None)),
                      key_runs=(key_runs, ("activation_batch", None, None, None)),
                      places=(places, ("activation_batch", None)), seed=(dropout_seed, (None,)),
                      bias=(None if bias is None else jnp.broadcast_to(
                          bias, (query.shape[0], query.shape[2], 1, key.shape[1])),
                            ("activation_batch", "activation_heads", None, None)))


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

    `equation` is flax's value product, '...hqk,...khd->...qhd'; a CPU under
    Dew's tests sums an fp32 one in float64 and rounds once (`precision.accumulated`).
    """
    if equation != '...hqk,...khd->...qhd':
        raise ValueError(f"weighted_values takes flax's value product, got {equation!r}")
    return float32_einsum(equation, weights.astype(value.dtype), value, precision=precision)



def reference_attention(query, key, value, *, mask=None, bias=None, sinks=None, softcap=None, dtype=None,
                        precision=None, force_fp32_for_softmax=True, dropout_rate=0.0, dropout_seed=None,
                        dropout_places=None, wide_logits=False):
    """Attend over [B, S, H, D] in plain XLA ops, every score policy on one pipeline.

    It is flax's reference attention, op for op: the query scaled by
    1/sqrt(D) in its dtype, the logits, the additive `bias`, the structural
    `mask`, the softmax (in at least fp32 under `force_fp32_for_softmax`), a
    probability draw at `dropout_rate` from `dropout_seed`, each weight its own
    (at its global places where a sequence exchange holds a slice of the call,
    `dropout_places`), and the value product (`weighted_values`). Gemma 2's `softcap` squashes the
    scaled logits into (-softcap, softcap) before the bias and mask
    (modeling_gemma2.py:192-208), and `sinks` `[H]` put one value-free logit
    per head in the softmax's denominator (`sink_softmax`). Keys and values
    may carry fewer heads. Without sinks they repeat to the query's, as flax
    repeats them, which keeps flax's rounding; with sinks the query heads
    group over their key head and no key or value repeats, as the decode
    selector (`attention_kernels.decodes`) needs where it routes a sink
    layer here rather than to the fused decode kernel, which takes sinks on a
    GPU. `wide_logits` computes the logits from the query and keys widened to
    at least fp32 and scales them after the product, as DeepSeek V4 does
    (modeling_deepseek_v4.py:708-736); the value product stays in the inputs'
    dtype.
    """
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype
    *rows, length, heads, width = query.shape
    kv_heads, keys = key.shape[-2], key.shape[-3]
    if wide_logits:
        scaled_query, scoring = query.astype(at_least_fp32(dtype)), key.astype(at_least_fp32(dtype))
    else:
        scaled_query, scoring = query / jnp.sqrt(width).astype(dtype), key
    if sinks is None:
        scoring, value = repeat_kv_heads(scoring, heads), repeat_kv_heads(value, heads)
        logits = float32_einsum('...qhd,...khd->...hqk', scaled_query, scoring, precision=precision)
    elif kv_heads == 1:
        # One key/value head (DeepSeek V4's): the query heads are a free dim beside the rows, and no
        # size-one head axis rides as a batch dim, which had XLA transpose the fp32 query for the
        # product (a V4 layer's training step on an H100, batch 4 by 2048: 2 GiB more and 5% slower).
        logits = float32_einsum('...tgd,...sd->...gts', scaled_query, scoring[..., 0, :], precision=precision)
    else:
        grouped = scaled_query.reshape(*rows, length, kv_heads, heads // kv_heads, width)
        logits = float32_einsum('...tkgd,...skd->...kgts', grouped, scoring, precision=precision).reshape(
            *rows, heads, length, keys)
    if wide_logits:
        logits = logits * width ** -0.5
    if softcap is not None:
        logits = capped_logits(logits, softcap)
    if bias is not None:
        logits = logits + bias
    if mask is not None:
        # Beside a sink a masked key is -inf, so it weighs nothing whatever the sink's logit: the sink
        # keeps every row's denominator finite. Without one the finite floor keeps a row every key
        # masks a number, as flax's does.
        logits = jnp.where(mask, logits, -jnp.inf if sinks is not None else jnp.finfo(dtype).min)
    if sinks is not None:
        if sinks.shape != (heads,):
            raise ValueError(f"sinks must have one logit per query head, got {sinks.shape}")
        wide = jnp.promote_types(logits.dtype, sinks.dtype)
        weights = sink_softmax(logits, sinks, at_least_fp32(wide) if force_fp32_for_softmax else None)
    elif force_fp32_for_softmax and dtype != jnp.float32:
        weights = jax.nn.softmax(logits.astype(at_least_fp32(dtype)))
    else:
        # A bias wider than the logits widens them; the softmax runs there and its weights round after.
        weights = jax.nn.softmax(logits).astype(dtype)
    if dropout_rate:
        if dropout_seed is None:
            raise ValueError("training attention dropout requires dropout_seed")
        # The keep bits are the draw's, shared with the kernels; the scale 1 / (1 - rate) takes the product's
        # dtype, rounded once from its double (ledger 1533: a bf16 1 / (1 - rate) biases every kept weight).
        # The weights round once, to their dtype.
        wide = at_least_fp32(dtype)
        keep = template.dropout_keep(dropout_seed, dropout_rate, weights.shape, dropout_places)
        scale = jnp.asarray(1.0 / (1.0 - dropout_rate), wide)
        weights = (weights.astype(wide) * jnp.where(keep, scale, 0)).astype(dtype)
    if sinks is None:
        return weighted_values('...hqk,...khd->...qhd', weights, value, precision=precision)
    if kv_heads == 1:
        return float32_einsum('...gts,...sd->...tgd', weights.astype(value.dtype), value[..., 0, :],
                              precision=precision)
    groups = weights.astype(value.dtype).reshape(*rows, kv_heads, heads // kv_heads, length, keys)
    return float32_einsum('...kgts,...skd->...tkgd', groups, value, precision=precision).reshape(
        *rows, length, heads, value.shape[-1])


def scaled_dot_product_attention(query, key, value, dtype=None, precision=None,
                                 force_fp32_for_softmax=True, implementation='auto',
                                 causal=False, sliding_window=None, mask=None, bias=None,
                                 sinks=None, softcap=None, segment_ids=None, max_segments=None,
                                 key_value_seq_lengths=None, dropout_rate=0.0,
                                 dropout_rng=None, deterministic=True, key_segment_ids=None,
                                 key_value_seq_starts=None, block_groups=None, image_groups=None,
                                 key_valid=None, key_runs=None, places=None):
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
    keeps each packed document to itself, 0 as padding (`document_mask`), and
    `max_segments`, static, bounds the documents a row holds
    (`Coordinates.max_segments`), which flash's packed sequences need.
    `key_segment_ids` `[B, K]` are the keys' own where they are not the
    queries (`CrossDocuments`): each query reads its document's keys, and
    under causality or a window a document's queries are its last keys, in
    order (a cache call's tokens, appended after the keys their row held).
    `key_value_seq_lengths` `[B]` ends each row's keys, which cuDNN skips; a
    mask that only ends rows early costs cuDNN a dense bias per head, so pass
    lengths for right padding. `key_value_seq_starts` `[B]`, below the
    lengths, starts each row's keys there: a windowed decode step's cache,
    which the decode kernel and the template read from the window's first block
    (`decode_attention.attend`) and every other path as the mask they mean.
    `block_groups` and `image_groups` `[B, K]` number the keys' blocks, -1
    outside any: under causality a block's query also sees its block whole,
    an image's its later keys within the window (`template.spans`).
    `key_valid` `[B, K]` marks the real keys, `key_runs` `[B, Q, R, 2]` each
    query's key intervals, whose union it may see, and `places` `[B, S]` the places a
    window counts (`counted_places`). The template reads these as its spans
    (`key_spans`); every other path takes the mask they mean (`visibility_mask`).

    A mesh sequence axis above one runs `sequence_parallel_attention`. Active dropout draws each weight
    from `dropout_rng` on the route the selector takes for it (`reference`, or `template` where named),
    at its global place on a sequence axis above one, and refuses explicit fused kernels. The
    result carries the checkpoint name 'attention_output', so a remat policy
    can save it instead of replaying the kernel (`remat_block` in dit.py).
    """
    shards = sequence_shards()
    if not 0 <= dropout_rate < 1:
        raise ValueError(f"dropout_rate must be within [0, 1), got {dropout_rate}")
    dropping = bool(dropout_rate) and not deterministic
    if dropping and dropout_rng is None:
        raise ValueError("training attention dropout requires dropout_rng")
    if key_segment_ids is not None:
        if segment_ids is None:
            raise ValueError("key_segment_ids are the keys' documents beside the queries' segment_ids")
        segment_ids = CrossDocuments(segment_ids, key_segment_ids)
    if not causal:
        block_groups = image_groups = None
    if sliding_window is None:
        places = None
    spanned = {"block_groups": block_groups, "image_groups": image_groups, "key_valid": key_valid,
               "key_runs": key_runs,
               "places": places}
    if any(x is not None for x in spanned.values()) and shards > 1:
        # Only a single device's template reads groups, key validity and places; across sequence shards the
        # call's spans are the mask they mean, its causality, window, documents, key lengths and starts among
        # them. On one device a route other than the template's takes them so (`_attention_kernel`).
        mask = visibility_mask(query.shape[-3], key.shape[-3], causal, sliding_window, mask, segment_ids,
                               key_value_seq_lengths, key_value_seq_starts, **spanned)
        causal, sliding_window, segment_ids, block_groups, image_groups = False, None, None, None, None
        key_value_seq_lengths = key_value_seq_starts = key_valid = key_runs = places = None
        implementation = kernel_for_materialized_mask(
            implementation, query, key, dtype=dtype, precision=precision,
            force_fp32_for_softmax=force_fp32_for_softmax)
    if (isinstance(segment_ids, CrossDocuments) and (causal or sliding_window is not None)
            and shards > 1):
        # Only a single device's kernels read documents aligned to their last keys; elsewhere they
        # are the mask they mean.
        aligned = cross_document_mask(segment_ids, causal=causal, sliding_window=sliding_window)
        mask = aligned if mask is None else jnp.logical_and(aligned, mask)
        causal, sliding_window, segment_ids = False, None, None
    if key_value_seq_starts is not None and shards > 1:
        # Only a single device's decode kernel reads starts; elsewhere they are the mask they mean.
        mask = with_key_starts(mask, key_value_seq_starts, key.shape[-3])
        key_value_seq_starts = None
    words = template.dropout_seed(dropout_rng) if dropping else None
    if shards > 1 and segment_ids is not None:
        # The exchanges split an explicit mask along the sequence and have no
        # split for the ids, so the documents travel as the mask they stand
        # for, with causality and the window folded in beside them.
        mask = combined_attention_mask(query.shape[-3], key.shape[-3], causal, sliding_window,
                                       with_documents(mask, segment_ids))
        causal, sliding_window, segment_ids = False, None, None
        implementation = kernel_for_materialized_mask(
            implementation, query, key, dtype=dtype, precision=precision,
            force_fp32_for_softmax=force_fp32_for_softmax)
    kernel = functools.partial(
        attention_kernel, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, implementation=implementation,
        softcap=softcap)
    if shards > 1:
        out = sequence_parallel_attention(
            kernel, query, key, value, shards, causal=causal,
            sliding_window=sliding_window, mask=mask, bias=bias, sinks=sinks,
            key_value_seq_lengths=key_value_seq_lengths,
            dropout_rate=dropout_rate if dropping else 0.0, dropout_seed=words)
    else:
        out = kernel(query, key, value, causal=causal, sliding_window=sliding_window,
                     mask=mask, bias=bias, sinks=sinks, segment_ids=segment_ids,
                     max_segments=max_segments, key_value_seq_lengths=key_value_seq_lengths,
                     key_value_seq_starts=key_value_seq_starts, block_groups=block_groups,
                     image_groups=image_groups, key_valid=key_valid, key_runs=key_runs, places=places,
                     dropout_rate=dropout_rate if dropping else 0.0, dropout_seed=words)
    return checkpoint_name(out, 'attention_output')


def _read_nothing(out: jax.Array, live: jax.Array | None) -> jax.Array:
    """`out` with each row that sees no key (`visible_rows`) exactly zero. Its
    cotangent is zero there too, so no kernel's backward reads those rows."""
    return out if live is None else jnp.where(live, out, jnp.zeros((), out.dtype))


def _fused_attention(query, key, value, bias, mask, causal, sliding_window, implementation, *,
                     softcap, sinks, segment_ids, key_value_seq_lengths, max_segments=None,
                     key_value_seq_starts=None, block_groups=None, image_groups=None, key_valid=None,
                     key_runs=None,
                     places=None, dropout_rate=0.0, dropout_seed=None, dropout_places=None):
    """Run the kernel `implementation` names, at the query's head width: the executor of what the
    attention selector chose (`resolve_implementation`, `_attention_kernel`), which held the call to
    the kernel's requirements (`attention_kernels.ATTENTION`) before it reaches here.

    A narrower value rides in padded and its columns come back out, at no runtime branch, except that the
    template takes its own value width. `key_value_seq_lengths` reaches cudnn, xla and the template as
    lengths and 'tpu' as the mask it means; `key_value_seq_starts` reaches the template and the decode
    kernel alone, and the groups, key validity and places the template alone (`_attention_kernel` folds them
    into the mask elsewhere).
    """
    v_head_dim = value.shape[-1]
    if v_head_dim != query.shape[-1] and implementation != 'template':
        value = widen_value_heads(query, value)

    if implementation == 'cudnn':
        out = cudnn_attention(query, key, value, bias, mask, causal, sliding_window,
                              key_value_seq_lengths)
    elif implementation == 'triton':
        out = triton_attention(query, key, value, causal)
    elif implementation == 'flash':
        out = flash_attention(query, key, value, causal, sliding_window, sinks, segment_ids, max_segments)
    elif implementation == 'decode':
        starts = key_value_seq_starts

        def decoded(query, key, value, *sunk):
            # A Pallas call has no partitioning rule: each shard decodes its own rows and heads.
            def attend(query, key, value, lengths, starts=None, sinks=None):
                return decode_attention.attend(query, key, value, lengths, starts, softcap=softcap,
                                               sinks=sinks)

            rows = ("activation_batch",)
            return _per_shard(attend, query, key, value, lengths=(key_value_seq_lengths, rows),
                              starts=(starts, rows), sinks=(sunk[0] if sunk else None, ("activation_heads",)))

        def folded(query, key, value, *sunk):
            visible = with_key_starts(with_key_lengths(None, key_value_seq_lengths, key.shape[-3]), starts,
                                      key.shape[-3])
            if not sunk and softcap is None:
                return folded_attention(query, key, value, visible)
            # The folded form has neither; the reference path computes both over the same keys.
            return reference_attention(query, key, value, mask=visible, sinks=sunk[0] if sunk else None,
                                       softcap=softcap)

        # The kernel has no derivative, so both modes differentiate the attention it stands in for, the
        # sinks among its operands, at the kernel's arithmetic: IEEE products for fp32 operands
        # (`decode_attention`), its 16-bit products summed in fp32 otherwise.
        out = forward_differentiable(decoded, folded, query, key, value, *([] if sinks is None else [sinks]),
                                     precision="highest" if query.dtype == jnp.float32 else None)
    elif implementation == 'xla' and folds(query, key, bias, mask, causal, sliding_window):
        # Key lengths and the prefix mask they mean take one path here, so where no lengths-only
        # kernel takes the call the two forms agree bitwise.
        out = folded_attention(query, key, value, mask if key_value_seq_lengths is None
                               else with_key_lengths(mask, key_value_seq_lengths, key.shape[-3]))
    elif implementation == 'xla':
        # A window of (l, r) keeps the l keys before the query and the r after
        # it on both the xla and the cudnn path, the sides `window_sides` counts.
        # jax.nn's products take the precision their dtype does (`float32_products`) through the
        # matmul config, which it reads for float32 operands.
        with configured_precision(float32_products(query.dtype)):
            out = jax.nn.dot_product_attention(
                query, key, value, bias=bias, mask=mask, is_causal=causal,
                key_value_seq_lengths=key_value_seq_lengths,
                local_window_size=None if sliding_window is None else window_sides(causal, sliding_window),
                implementation='xla')
    elif implementation == 'template':
        out = template_attention(query, key, value, causal, sliding_window, sinks=sinks, softcap=softcap,
                                 segment_ids=segment_ids, lengths=key_value_seq_lengths,
                                 starts=key_value_seq_starts, block_groups=block_groups,
                                 image_groups=image_groups, key_valid=key_valid, key_runs=key_runs,
                                 places=places, dropout_rate=dropout_rate, dropout_seed=dropout_seed,
                                 dropout_places=dropout_places, bias=bias)
    else:  # 'tpu'
        if key_value_seq_lengths is not None:
            mask = with_key_lengths(mask, key_value_seq_lengths, key.shape[-3])
        out = tpu_attention(query, key, value, bias, mask, causal, sliding_window,
                            softcap=softcap, sinks=sinks, segment_ids=segment_ids,
                            interpret=generation.backend() != 'tpu')
    return out if v_head_dim == out.shape[-1] else out[..., :v_head_dim]


FORWARD_MODE = contextvars.ContextVar("forward_mode_attention", default=False)


@contextlib.contextmanager
def forward_mode_attention():
    """Trace the attention calls inside for `jax.jvp`.

    cuDNN's and the TPU's fused kernels are reverse-mode only (`jax.custom_vjp`).
    Inside this context each such call keeps the fused kernel for its value and
    takes its tangent from the reference path's JVP, which materializes the
    `[B, H, Q, K]` probabilities once. Consistency models (sCM, rCM) run their
    JVP inside it and train outside it, on the fused backward.
    """
    token = FORWARD_MODE.set(True)
    try:
        yield
    finally:
        FORWARD_MODE.reset(token)


def attention_kernel(query, key, value, dtype=None, precision=None,
                     force_fp32_for_softmax=True, implementation='auto',
                     causal=False, sliding_window=None, mask=None, bias=None, sinks=None,
                     softcap=None, segment_ids=None, max_segments=None, key_value_seq_lengths=None,
                     key_value_seq_starts=None, block_groups=None, image_groups=None, key_valid=None,
                     key_runs=None,
                     places=None, dropout_rate=0.0, dropout_seed=None, dropout_places=None):
    """Dispatch one whole-sequence attention call to the kernel it names,
    differentiable in forward mode under `forward_mode_attention`.

    'auto' resolves here, once, against this call's shapes and backend, and
    `_attention_kernel` runs the call on the kernel it resolved to. A query
    row that sees no key comes back zero whichever kernel ran
    (`visible_rows`), inside a sequence exchange too.
    """
    if sliding_window is not None and sliding_window < 1:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")
    if (key_value_seq_lengths is not None and query.shape[-3] == 1 and not causal
            and sliding_window is None and mask is None and bias is None and sinks is None
            and segment_ids is None and query.shape[-2] > key.shape[-2]
            and query.shape[-2] % key.shape[-2] == 0 and dropout_seed is None
            and key_valid is None and key_runs is None and block_groups is None and image_groups is None
            and places is None):
        # The grouped call carries key lengths and starts alone, so a call masking its keys otherwise
        # (key validity, runs, block or image groups, counted places) takes the whole path.
        # The key heads repeated as each shard would hold them (`_per_shard`), so the tensor axes
        # split the groups as they split the query heads.
        key, value = split_with_queries(key, value, tensor_shards(query))
        return decode_attention.grouped_query(functools.partial(
            attention_kernel, dtype=dtype, precision=precision, force_fp32_for_softmax=force_fp32_for_softmax,
            implementation=implementation, softcap=softcap, key_value_seq_lengths=key_value_seq_lengths,
            key_value_seq_starts=key_value_seq_starts), query, key, value)
    lengths = (None if key_value_seq_lengths is None
               else jnp.asarray(key_value_seq_lengths, jnp.int32))
    starts = None if key_value_seq_starts is None else jnp.asarray(key_value_seq_starts, jnp.int32)
    masked = mask if lengths is None else with_key_lengths(mask, lengths, key.shape[-3])
    masked = with_key_starts(masked, starts, key.shape[-3])
    resolved = resolve_implementation(
        implementation, query, key, dtype=dtype, precision=precision,
        force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap, sinks=sinks,
        causal=causal, sliding_window=sliding_window, mask=mask, bias=bias,
        segment_ids=segment_ids, max_segments=max_segments, key_value_seq_lengths=lengths,
        key_value_seq_starts=starts, block_groups=block_groups, image_groups=image_groups,
        key_valid=key_valid, key_runs=key_runs, places=places, dropout=dropout_seed is not None)
    call = functools.partial(
        _attention_kernel, dtype=dtype, precision=precision, force_fp32_for_softmax=force_fp32_for_softmax,
        causal=causal, sliding_window=sliding_window, mask=mask, masked=masked, softcap=softcap,
        segment_ids=segment_ids, max_segments=max_segments, lengths=lengths, starts=starts,
        block_groups=block_groups, image_groups=image_groups, key_valid=key_valid, key_runs=key_runs,
        places=places, dropout_rate=dropout_rate, dropout_seed=dropout_seed, dropout_places=dropout_places)
    live = visible_rows(query.shape[-3], key.shape[-3], causal, sliding_window, mask, segment_ids, lengths,
                        starts, block_groups=block_groups, image_groups=image_groups, key_valid=key_valid,
                        key_runs=key_runs,
                        places=places)
    if not FORWARD_MODE.get() or resolved not in ('cudnn', 'triton', 'flash', 'tpu', 'template'):
        return _read_nothing(call(query, key, value, bias=bias, sinks=sinks, implementation=resolved), live)

    def fused(query, key, value, bias, sinks):
        return call(query, key, value, bias=bias, sinks=sinks, implementation=resolved)

    def reference(query, key, value, bias, sinks):
        return call(query, key, value, bias=bias, sinks=sinks, implementation='reference')

    return _read_nothing(forward_differentiable(fused, reference, query, key, value, bias, sinks), live)


def forward_differentiable(value, tangent, *arrays, precision: str | None = None):
    """`value(*arrays)`, whose JVP is `tangent`'s: a kernel that defines only
    its reverse-mode derivative, or none, takes its derivatives from a
    reference computing the same function, reverse mode as the transposed JVP.
    `precision` is the kernel's own arithmetic, which the reference's products
    take too (`jax.default_matmul_precision`), so a kernel of IEEE fp32 products
    is not differentiated by TF32 ones; None leaves the products at the
    caller's. Only the arrays a derivative moves are the reference's tangents: a known zero stays
    out of its derivative, where a product's own derivative (`dew.nn.precision.accumulated`'s on a
    CPU) takes its moving operands alone."""
    from jax.custom_derivatives import SymbolicZero

    @jax.custom_jvp
    def attend(*arrays):
        return value(*arrays)

    def rule(primals, tangents):
        live = [index for index, direction in enumerate(tangents) if not isinstance(direction, SymbolicZero)]
        output = attend(*primals)
        if not live:
            return output, jax.tree.map(jnp.zeros_like, output)

        def moved(*moving):
            arrays = list(primals)
            for index, array in zip(live, moving, strict=True):
                arrays[index] = array
            return tangent(*arrays)

        with contextlib.nullcontext() if precision is None else jax.default_matmul_precision(precision):
            differentiated = jax.jvp(moved, tuple(primals[index] for index in live),
                                     tuple(tangents[index] for index in live))[1]
        return output, differentiated

    attend.defjvp(rule, symbolic_zeros=True)
    return attend(*arrays)


def _attention_kernel(query, key, value, *, bias, sinks, implementation, dtype, precision,
                      force_fp32_for_softmax, causal, sliding_window, mask, masked, softcap,
                      segment_ids, max_segments, lengths, starts, block_groups=None, image_groups=None,
                      key_valid=None, key_runs=None, places=None, dropout_rate=0.0, dropout_seed=None,
                      dropout_places=None):
    """Run one whole-sequence attention call on the kernel `attention_kernel` resolved.

    `masked` is `mask` with the key `lengths` and `starts` in it. Splash takes sinks, a
    softcap and packed segment ids itself, flash takes the ids as its packed sequences given
    `max_segments`, and the template reads the ids (`CrossDocuments` included) as its spans;
    elsewhere segment ids become the document mask, and sinks and a softcap run
    `reference_attention`, since `jax.nn.dot_product_attention` has neither (a fused route this
    host runs as xla included, `_hosted`). The rest goes to `_fused_attention`. Key lengths reach
    cudnn, xla and the template as they are and every other path as the mask they mean, which
    splash cannot describe, so 'auto' never picks 'tpu' for them. Key starts reach the decode
    kernel and the template as they are and every other path as the mask they mean. Block and image
    groups, key validity and counted places reach the template alone; every other path takes the call's
    spans as the mask they mean (`visibility_mask`).
    """
    spanned = (block_groups, image_groups, key_valid, key_runs, places)
    if any(x is not None for x in spanned) and implementation != 'template':
        mask = masked = visibility_mask(query.shape[-3], key.shape[-3], causal, sliding_window, mask,
                                        segment_ids, lengths, starts, block_groups=block_groups,
                                        image_groups=image_groups, key_valid=key_valid, key_runs=key_runs,
                                        places=places)
        causal, sliding_window, segment_ids, lengths, starts = False, None, None, None, None
        block_groups = image_groups = key_valid = key_runs = places = None
    # Another sequence's documents under causality or a window align each document's queries with
    # its last keys (`CrossDocuments`), which splash does not; elsewhere the mask carries them.
    aligned = isinstance(segment_ids, CrossDocuments) and (causal or sliding_window is not None)
    packed = (implementation == 'template' or (implementation == 'tpu' and not aligned)
              or (implementation == 'flash' and max_segments is not None))
    if segment_ids is not None and not packed:
        if aligned:
            inside = cross_document_mask(segment_ids, causal=causal, sliding_window=sliding_window)
            mask, masked = (inside if m is None else jnp.logical_and(m, inside) for m in (mask, masked))
            causal, sliding_window = False, None
        else:
            mask = with_documents(mask, segment_ids)
            masked = with_documents(masked, segment_ids)
        # cuDNN takes the documents as their mask, an additive bias (`kernel_for_materialized_mask`):
        # a cache's aligned ones as it took a cache's mask, a call's own where cuDNN is named.
    if implementation == 'reference' or (implementation == 'xla' and (
            sinks is not None or softcap is not None or dropout_seed is not None)):
        mask = combined_attention_mask(query.shape[-3], key.shape[-3], causal, sliding_window, masked)
        return reference_attention(query, key, value, mask=mask, bias=bias, sinks=sinks, softcap=softcap,
                                   dtype=dtype, precision=precision,
                                   force_fp32_for_softmax=force_fp32_for_softmax,
                                   dropout_rate=dropout_rate, dropout_seed=dropout_seed,
                                   dropout_places=dropout_places)

    # The decode kernel and the template read key starts as their own spans; elsewhere they are the mask.
    if implementation not in ('decode', 'template'):
        mask = with_key_starts(mask, starts, key.shape[-3])
        starts = None
    return _fused_attention(query, key, value, bias, mask, causal, sliding_window,
                            implementation, softcap=softcap, sinks=sinks,
                            segment_ids=segment_ids, max_segments=max_segments,
                            key_value_seq_lengths=lengths, key_value_seq_starts=starts,
                            block_groups=block_groups, image_groups=image_groups, key_valid=key_valid,
                            key_runs=key_runs,
                            places=places, dropout_rate=dropout_rate, dropout_seed=dropout_seed,
                            dropout_places=dropout_places)


def reference_only(query, dtype, precision, force_fp32_for_softmax) -> bool:
    """Whether a call asks for arithmetic only the reference path performs.

    The three are what every fused kernel's requirements refuse
    (`attention_kernels`' arithmetic requirement): a matmul precision other
    than DEFAULT (a higher one, a dot algorithm or a preset of one), a
    softmax outside fp32, and a compute dtype other than the inputs'.
    """
    return (not asks_default_precision(precision)
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
        query.dtype == jnp.bfloat16 and generation.backend() == 'gpu' and not bf16_dot_runs())


def _xla_kernel_widens_less() -> bool:
    """Whether jax.nn's xla attention would run this call below the arithmetic its process asks of a CPU,
    so 'auto' and 'xla' take the reference path: a CPU under Dew's test setting (`WIDE_CPU_ARITHMETIC`),
    which sums fp32 products in float64 (`precision.accumulated`) where jax's sums them in fp32, and whose
    reference path differentiates the softmax by its own derivative, exactly zero for a query that reads
    one key, where jax's takes rowsum(dO * O) and leaves it a rounding. A process without the setting runs
    jax's kernel on a CPU as on any platform."""
    return wide_cpu_arithmetic() and generation.backend() == 'cpu'


def attention_case(query, key, *, dtype=None, precision=None, force_fp32_for_softmax=True, softcap=None,
                   sinks=None, causal=False, sliding_window=None, mask=None, bias=None, segment_ids=None,
                   max_segments=None, key_value_seq_lengths=None, key_value_seq_starts=None,
                   block_groups=None, image_groups=None, key_valid=None, key_runs=None, places=None,
                   dropout: bool = False, named: str = 'auto') -> AttentionCase:
    """The `AttentionCase` of a call `scaled_dot_product_attention` takes under the `named`
    `AttentionImpl`, which decides whether splash might run it."""
    cross = isinstance(segment_ids, CrossDocuments)
    q_len, kv_len = query.shape[-3], key.shape[-3]
    case = AttentionCase(
        queries=q_len, keys=kv_len, heads=query.shape[-2], kv_heads=key.shape[-2],
        head_dim=query.shape[-1], dtype=jnp.dtype(query.dtype), key_dtype=jnp.dtype(key.dtype),
        causal=bool(causal), window=sliding_window,
        documents=None if segment_ids is None else 'cross' if cross else 'own',
        aligned=cross and (causal or sliding_window is not None), max_segments=max_segments,
        lengths=key_value_seq_lengths is not None, starts=key_value_seq_starts is not None,
        blocks=bool(causal) and (block_groups is not None or image_groups is not None),
        spans_only=((bool(causal) and (block_groups is not None or image_groups is not None))
                    or key_valid is not None or key_runs is not None
                    or (places is not None and sliding_window is not None)),
        mask=mask is not None, bias=bias is not None,
        key_bias=(bias is not None and jnp.ndim(bias) == 4 and jnp.shape(bias)[-2] == 1
                  and jnp.dtype(bias.dtype).itemsize <= 4),
        sinks=sinks is not None, softcap=softcap,
        arithmetic=reference_only(query, dtype, precision, force_fp32_for_softmax),
        narrows=_xla_kernel_narrows(query) or _xla_kernel_widens_less(),
        folds=folds(query, key, bias, mask, causal, sliding_window),
        splash_mask=False, shards=sequence_shards(), dropout=dropout)
    # Splash's descriptor reads a concrete mask on the host, so it is built only where splash may run,
    # under a name that may choose it on a TPU (`TPU_BACKEND`); its tiling and every other requirement
    # read the case (`TPU_SUPPORTS`).
    if named in ('auto', 'tpu') and TPU_BACKEND.holds(case):
        described = splash_mask_descriptor(q_len, kv_len, shard_heads(query), causal, sliding_window, mask)
        case = dataclasses.replace(case, splash_mask=described is not None)
    return case


def resolve_implementation(implementation, query, key, *, dtype=None, precision=None,
                           force_fp32_for_softmax=True, softcap=None, sinks=None, causal=False,
                           sliding_window=None, mask=None, bias=None, segment_ids=None, max_segments=None,
                           key_value_seq_lengths=None, key_value_seq_starts=None, block_groups=None,
                           image_groups=None, key_valid=None, key_runs=None, places=None,
                           dropout: bool = False) -> str:
    """The kernel an `AttentionImpl` names for this call on this host, as the attention selector
    reports it (`select`, over `attention_case`): 'auto' takes the kernel the target's row measured
    fastest for the call where it takes it, else the first that does; a named kernel is forced, and
    raises where it does not take the call (xla handing what it cannot honour to the reference path).
    A call of few positions over key lengths is the decode selector's (`DECODE`), whose decode kernel
    runs as 'decode' (`_fused_attention`); a named xla lowers it there too, as jax.nn's own lowering
    of a decode step, unless `generation.chosen(decode='xla')` names the folded attention. What runs
    is what this host runs (`_hosted`): xla for a CUDA kernel a step traced for an assumed GPU on a
    CPU reports."""
    if implementation not in get_args(AttentionImpl):
        raise ValueError(f"Unknown attention implementation: {implementation}")
    case = attention_case(query, key, dtype=dtype, precision=precision,
                          force_fp32_for_softmax=force_fp32_for_softmax, softcap=softcap, sinks=sinks,
                          causal=causal, sliding_window=sliding_window, mask=mask, bias=bias,
                          segment_ids=segment_ids, max_segments=max_segments,
                          key_value_seq_lengths=key_value_seq_lengths,
                          key_value_seq_starts=key_value_seq_starts, block_groups=block_groups,
                          image_groups=image_groups, key_valid=key_valid, key_runs=key_runs, places=places,
                          dropout=dropout, named=implementation)
    if not (decodes(case) and implementation in ('auto', 'cudnn', 'xla')):
        return select(ATTENTION, case, None if implementation == 'auto' else implementation).ran
    named = {'auto': None, 'cudnn': 'cudnn'}.get(implementation)
    if implementation == 'xla':
        named = 'xla' if forced(DECODE.choice) == 'xla' or DECODE.kernel('decode').failed(case) else 'decode'
    return select(DECODE, case, named).ran


def kernel_for_materialized_mask(implementation: str, query, key, *, dtype=None, precision=None,
                                 force_fp32_for_softmax=True) -> str:
    """Send a call that built an explicit mask to the xla kernel.

    cuDNN has no mask argument: jax hands it a bool mask as an additive bias of
    -2**41 in the compute dtype (`combine_bias_and_mask` in
    jax/_src/cudnn/fused_attention_stablehlo.py), and `check_is_flash_attention`
    then refuses an odd length while training. The xla kernel masks by
    exclusion with the same fp32 softmax, at 83.6 ms and 5.80 GiB a step against
    cuDNN's fixed window at 75.8 ms and 4.99 GiB
    (docs/concepts/language_models.md). A call the xla kernel hands off goes to
    the reference path as its requirements say (`attention_kernels.XLA_SUPPORTS`),
    an explicit 'cudnn' among them, which runs here as xla.
    """
    masked = 'xla' if implementation in ('auto', 'cudnn') else implementation
    if masked != 'xla':
        return masked
    case = dataclasses.replace(attention_case(query, key, dtype=dtype, precision=precision,
                                              force_fp32_for_softmax=force_fp32_for_softmax, named='xla'),
                               mask=True, folds=False)
    return selected(ATTENTION, case, 'xla').name


def cudnn_takes_mask(implementation: str, query, *, softcap=None, sinks: bool = False) -> bool:
    """Whether cuDNN may take a call's materialized mask as its additive bias:
    the call names cudnn or chooses ('auto'), cuDNN runs it (`cudnn_runs`,
    which also keeps the deterministic-ops lane off it) and it has no sinks,
    which cuDNN cannot honor. A caller adds why its mask suits cuDNN's bias,
    where `kernel_for_materialized_mask` sends the rest to xla: a forward-only
    call, or a mask a window bounds."""
    return implementation in ('auto', 'cudnn') and not sinks and cudnn_runs(query, softcap)


# How much of an explicit boolean mask splash will carry. The array is read
# on the host while the executable is built and its unresolved blocks are
# stored dense inside it, so this bounds host and executable bytes, not
# device memory that grows with the batch. 4 Mi cells is a 16-head 512x512.
SPLASH_DENSE_MASK_CELLS = 1 << 22



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

    A Mosaic kernel cannot be partitioned automatically, so on a mesh both run
    per shard (`_per_shard`), on its rows and heads. Splash's descriptor then
    covers one shard's heads, so a mask that differs by head where the mesh
    splits the heads is flash's.
    """
    q_len, kv_len = query.shape[-3], key.shape[-3]
    descriptor = None
    if bias is None and not (q_len % SPLASH_LANES or kv_len % SPLASH_LANES):
        descriptor = splash_mask_descriptor(
            q_len, kv_len, shard_heads(query), causal, sliding_window, mask)
    if descriptor is None and (softcap is not None or sinks is not None):
        raise ValueError(
            "attention implementation 'tpu' takes a softcap and sinks only on "
            "the splash kernel, and this call has a bias, a mask splash cannot "
            "describe or a length that is not a multiple of "
            f"{SPLASH_LANES}. Use attention_impl 'xla'.")

    def local(query, key, value, bias=None, mask=None, sinks=None, segment_ids=None):
        if descriptor is None:
            return pallas_flash_attention(query, key, value, bias, mask, causal,
                                          sliding_window, segment_ids)
        return splash_attention(query, key, value, descriptor, softcap=softcap, sinks=sinks,
                                segment_ids=segment_ids, interpret=interpret)

    def attend(query, key, value, bias, sinks):
        # Splash holds the mask in its descriptor; flash takes it as an array.
        held = None if descriptor is not None or mask is None else _four_dimensional(mask)
        return _per_shard(local, query, key, value,
                          bias=(None if bias is None else _four_dimensional(bias), SCORES),
                          mask=(held, SCORES), sinks=(sinks, ("activation_heads",)),
                          segment_ids=(segment_ids, ("activation_batch", None)))

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
    `softcap` is applied to the scaled logits before the mask and `sinks` `[H]`
    seed each head's running maximum and denominator, with their gradient, as
    in `reference_attention`. The kernel
    lets equal segment ids attend, so padding (id 0) reads other padding where
    `document_mask` reads nothing; no document's query reads padding either way.
    The kernel is JAX's, vendored with its backward's delta formed from the
    weights (`dew.nn.kernels.splash`).
    """
    from dew.nn.kernels import splash

    kernel = splash.make_splash_mha(
        descriptor, block_sizes=splash.block_sizes(query.shape[-3], key.shape[-3], width=query.shape[-1],
                                                   itemsize=query.dtype.itemsize,
                                                   softcap=softcap is not None),
        attn_logits_soft_cap=softcap, interpret=interpret)
    scale = jnp.asarray(1.0 / math.sqrt(query.shape[-1]), query.dtype)
    segments = None if segment_ids is None else splash.SegmentIds(*document_ids(segment_ids))

    def example(q, k, v, ids):
        return kernel(q, k, v, segment_ids=ids, sinks=sinks)

    attended = jax.vmap(example)(jnp.moveaxis(query, -2, -3) * scale,
                                 jnp.moveaxis(key, -2, -3), jnp.moveaxis(value, -2, -3),
                                 segments)
    return jnp.moveaxis(attended, -3, -2)


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
    segments = None if segment_ids is None else SegmentIds(*document_ids(segment_ids))
    return jnp.moveaxis(
        flash_attention(q, k, v, ab=combined, segment_ids=segments, causal=causal), -3, -2)


def projected_heads(x, name: str, heads: int, head_dim: int, *, flat: bool, bias: bool = True,
                    dtype: Dtype | None = None, precision: PrecisionLike = None,
                    norm: Callable[[jax.Array], jax.Array] | None = None, across_heads: bool = False
                    ) -> jax.Array:
    """`x` `[..., C]` projected by the matrix `name` to `[..., heads, head_dim]`
    in the calling module, then normalized by `norm`.

    `flat` stores the matrix as the published diffusion and vision attentions
    do, `[C, heads * head_dim]`, and splits its product into heads; otherwise
    it is `[C, heads, head_dim]` (`DenseGeneral`), Dew's own, whose heads a
    tensor axis splits. A checkpoint's reader maps one layout or the other, so
    each attention keeps its own. `norm` normalizes each head's channels, or
    with `across_heads` each token's whole projection before it splits, as
    Wan's `rms_norm_across_heads` does: another function, not another order.
    """
    width = heads * head_dim
    if flat:
        projected = Dense(width, use_bias=bias, dtype=dtype, precision=precision, name=name,
                          dot_general=float32_dot_general)(x)
    else:
        projected = DenseGeneral((heads, head_dim), use_bias=bias, dtype=dtype, precision=precision,
                                 name=name, dot_general=float32_dot_general)(x)
    if norm is not None and across_heads:
        projected = norm(projected.reshape(*x.shape[:-1], width))
    projected = projected.reshape(*x.shape[:-1], heads, head_dim)
    return norm(projected) if norm is not None and not across_heads else projected


def projected_output(attended, name: str, features: int, *, flat: bool, bias: bool = True,
                     dtype: Dtype | None = None, precision: PrecisionLike = None) -> jax.Array:
    """Heads `[..., heads, head_dim]` joined by the matrix `name` to `[..., features]`
    in the calling module, the matrix stored as `projected_heads` stores its
    own: `[heads * head_dim, features]` flat, else `[heads, head_dim, features]`."""
    if flat:
        # The width is named, not inferred: an empty stream (no context tokens) has nothing to infer it from.
        return Dense(features, use_bias=bias, dtype=dtype, precision=precision, name=name,
                     dot_general=float32_dot_general)(
            attended.reshape(*attended.shape[:-2], math.prod(attended.shape[-2:])))
    return DenseGeneral(features, axis=(-2, -1), use_bias=bias, dtype=dtype, precision=precision,
                        name=name, dot_general=float32_dot_general)(attended)


@dataclasses.dataclass(frozen=True)
class QKNorm:
    """How an attention RMS-normalizes its queries and keys: the norms' module `names`, their
    `epsilon`, whether over each head's channels or, `across_heads`, each token's whole projection
    before it splits (Wan's `rms_norm_across_heads`: another function, not another order), and
    whether the normalized row rounds to the weight's dtype before it scales (`scale_after_cast`,
    diffusers' `RMSNorm`)."""

    names: tuple[str, str] = ("q_norm", "k_norm")
    epsilon: float = 1e-6
    across_heads: bool = False
    scale_after_cast: bool = False


CROSS_KEY = "cross_key"
"""The cache variable of a cached call's context keys, `[rows, context_length, kv_heads, head_dim]`."""
CROSS_VALUE = "cross_value"
"""The cache variable of a cached call's context values, shaped as its keys."""


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
    `flat` stores each projection as the published attentions do,
    `[C, heads * head_dim]`, rather than Dew's headed `[C, heads, head_dim]`
    (`projected_heads`), under `names` (query, key, value, output): Wan's
    are the defaults, the AutoencoderKL mid-block's `query/key/value/
    proj_attn`. `qk_norm` normalizes the queries and keys (`QKNorm`), and
    `rotary_pairs` names the channels each angle of `freqs_cis` turns
    (`dew.nn.rope.rotate`).
    `mask`, boolean and broadcasting to `[B, H, S, S_context]`, keeps the keys
    each query may read. `key_value_seq_lengths` `[B]` ends each row's keys
    instead, for a right-padded context, which a fused kernel skips rather
    than reading as a dense bias per head. `segment_ids` `[B, S]` keeps each
    query to its packed document's keys, `key_segment_ids` `[B, S_context]`
    numbering the context's (`CrossDocuments`), at most `max_segments` a
    row, which a fused kernel runs as packed sequences rather than a mask.

    It is every cross-attention's attention, the diffusion blocks' and Wan's
    and a decoder's gated block's alike (`dew.nn.cross_attention`): `kv_heads`
    groups the queries over fewer key-value heads, and with `cache_length` a
    `cached` call keeps a context's keys and values (`CROSS_KEY`,
    `CROSS_VALUE`), so a decode step without the context reads them.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_bias: bool = True
    qkv_bias: bool | None = None
    force_fp32_for_softmax: bool = True
    qk_norm: QKNorm | None = None
    attention_impl: str = "auto"  # an AttentionImpl
    flat: bool = False
    names: tuple[str, str, str, str] = ("to_q", "to_k", "to_v", "to_out_0")
    rotary_pairs: Literal["half", "adjacent"] = "half"
    kv_heads: int | None = None
    """The keys' and values' heads, which the query heads share in equal groups; None gives each
    query head its own."""
    cache_length: int | None = None
    """The most context tokens a row's cached call keeps the keys and values of; None keeps none."""

    @property
    def key_heads(self) -> int:
        return self.heads if self.kv_heads is None else self.kv_heads

    @nn.nowrap
    def projections(self, context_features: int | None = None) -> dict[str, tuple[int, int]]:
        inner, keys = self.heads * self.dim_head, self.key_heads * self.dim_head
        context = self.query_dim if context_features is None else context_features
        query, key, value, output = self.names
        return {query: (self.query_dim, inner), key: (context, keys), value: (context, keys),
                output: (inner, self.query_dim)}

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Self-attention maps and its full QK/AV products on real rows."""
        return self_attention_flops(lengths, self.projections(), self.heads, self.dim_head)

    @nn.nowrap
    def cross_flops(self, lengths, context_lengths, context_features: int) -> float:
        """Cross-attention with the context geometry supplied by its owner."""
        maps = self.projections(context_features)
        query, key, value, output = self.names
        return (sum(projection_flops(lengths, *maps[name]) for name in (query, output))
                + sum(projection_flops(context_lengths, *maps[name]) for name in (key, value))
                + cross_attention_flops(lengths, context_lengths, self.heads, self.dim_head, self.dim_head))

    def _heads(self, name: str, x, norm: str | None = None, *, keyed: bool = False):
        """`x` projected to `[B, S, heads, head_dim]`, the key-value heads where `keyed`."""
        normed = self.qk_norm if norm is not None else None
        rms = None if normed is None else RMSNorm(
            epsilon=normed.epsilon, scale_after_cast=normed.scale_after_cast, dtype=self.dtype, name=norm)
        projected = projected_heads(
            x, name, self.key_heads if keyed else self.heads, self.dim_head, flat=self.flat,
            bias=self.use_bias if self.qkv_bias is None else self.qkv_bias, dtype=self.dtype,
            precision=self.precision, across_heads=normed is not None and normed.across_heads, norm=rms)
        return constrain(projected, KV_HEADS if keyed and self.kv_heads is not None else HEADS)

    def _keys_values(self, context):
        """The context's keys and values, its positions split over a sequence axis where its link
        pays for it (`split_positions`)."""
        _, k_name, v_name, _ = self.names
        k_norm = self.qk_norm.names[1] if self.qk_norm is not None else None
        return split_positions(context, (self.key_heads, self.dim_head),
                               lambda tokens: (self._heads(k_name, tokens, k_norm, keyed=True),
                                               self._heads(v_name, tokens, keyed=True)))

    def _cached(self, rows: int, dtype, context):
        """The keys and values a cached call reads. A call that carries a context computes them
        from it and, once the cache is allocated, keeps them; a call without one reads what the
        cache keeps. As with a decoder's own cache, the first cached call allocates the cache and
        writes nothing. The call reads the keys and values as the cache keeps them, in `dtype`."""
        if self.cache_length is None:
            raise ValueError("a cached call keeps a context's keys and values, which this attention has "
                             "no cache_length for")
        allocated = self.has_variable("cache", CROSS_KEY)
        shape = (rows, self.cache_length, self.key_heads, self.dim_head)
        keys = self.variable("cache", CROSS_KEY, jnp.zeros, shape, dtype)
        values = self.variable("cache", CROSS_VALUE, jnp.zeros, shape, dtype)
        if context is None:
            return keys.value, values.value
        width = context.shape[1]
        if width > self.cache_length:
            raise ValueError(f"a cached call keeps at most {self.cache_length} context tokens a row "
                             f"(CrossAttention.context_length), got {width}")
        key, value = (projected.astype(dtype) for projected in self._keys_values(context))
        if allocated:
            padding = ((0, 0), (0, self.cache_length - width), (0, 0), (0, 0))
            keys.value = jnp.pad(key, padding)
            values.value = jnp.pad(value, padding)
        return key, value

    @nn.compact
    def __call__(self, x, context=None, freqs_cis=None, mask=None, key_value_seq_lengths=None,
                 segment_ids=None, key_segment_ids=None, max_segments=None, *, cached: bool = False,
                 cache_dtype=None):
        """`cached` reads and keeps the context's keys and values in the cache (`_cached`), in
        `cache_dtype` (the module's dtype by default); the context is then None on a step that reads
        the keys and values its prefill kept."""
        orig_x_shape = x.shape
        if len(x.shape) == 4:
            x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        if not cached:
            context = x if context is None else context
        if context is not None and len(context.shape) == 4:
            context = context.reshape(
                (context.shape[0], context.shape[1] * context.shape[2], context.shape[3]))
        # [B, S, heads, head_dim], column-parallel under a tensor axis.
        q_name, _, _, out_name = self.names
        q_norm = self.qk_norm.names[0] if self.qk_norm is not None else None
        query = self._heads(q_name, x, q_norm)
        if cached:
            dtype = cache_dtype or self.dtype or x.dtype
            key, value = self._cached(x.shape[0], dtype, context)
        else:
            key, value = self._keys_values(context)
        if freqs_cis is not None:
            query, key = (rotate(part, *freqs_cis, pairs=self.rotary_pairs) for part in (query, key))

        hidden_states = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            implementation=self.attention_impl, mask=mask, key_value_seq_lengths=key_value_seq_lengths,
            segment_ids=segment_ids, key_segment_ids=key_segment_ids, max_segments=max_segments,
        )
        proj = projected_output(constrain(hidden_states, HEADS), out_name, self.query_dim, flat=self.flat,
                                bias=self.use_bias, dtype=self.dtype, precision=self.precision)
        return proj.reshape(orig_x_shape)


class BasicTransformerBlock(nn.Module):
    """Run a first attention, cross-attention over `context`, then feed-forward.

    Each is pre-normed (`norm`, RMS or affine layer norm, at `norm_epsilon`)
    with a residual. `first_attention` names what the first attention reads:
    the tokens themselves (`self`), the `context` too (`context`, the SD
    checkpoints with no self-attention), or nothing, when the block has none
    (`none`). `only_pure_attention` runs the cross-attention alone, with no
    norm and no residual, which the UNets' stages do by default.
    `qkv_bias` is the attentions' (`NormalAttention`). `dropout_rate` drops
    each attention's output, inside the feed-forward and the block's output,
    as diffusers' Flax `FlaxBasicTransformerBlock` does in training (its
    PyTorch block drops no residual stream); `approximate_gelu` is the
    feed-forward's GELU (`geglu_feed_forward`).
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    use_bias: bool = True
    qkv_bias: bool | None = None
    first_attention: Literal["self", "context", "none"] = "self"
    only_pure_attention:bool = False
    force_fp32_for_softmax: bool = True
    norm: Literal["rms", "layer"] = "rms"
    norm_epsilon: float = 1e-4
    dropout_rate: float = 0.0
    approximate_gelu: bool = True
    attention_impl: str = "auto"  # an AttentionImpl

    @nn.nowrap
    def cross_flops(self, lengths, context_lengths, context_features: int) -> float:
        """The configured attention branches and optional GEGLU maps."""
        from .mlp import geglu_feed_forward  # mlp's blocks import this module's norms

        attention = NormalAttention(self.query_dim, self.heads, self.dim_head, parent=None)
        crossed = attention.cross_flops(lengths, context_lengths, context_features)
        count = crossed
        if not self.only_pure_attention:
            first = {"self": attention.model_flops(lengths), "context": crossed, "none": 0}
            count += first[self.first_attention]
            count += geglu_feed_forward(self.query_dim, parent=None).model_flops(lengths)
        return count

    def setup(self):
        attention = functools.partial(
            NormalAttention,
            query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            precision=self.precision,
            use_bias=self.use_bias,
            qkv_bias=self.qkv_bias,
            dtype=self.dtype,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            attention_impl=self.attention_impl,
        )
        self.attention1 = attention(name='Attention1')
        self.attention2 = attention(name='Attention2')

        from .mlp import geglu_feed_forward  # mlp's blocks import this module's norms

        self.ff = geglu_feed_forward(self.query_dim, approximate=self.approximate_gelu, dtype=self.dtype,
                                     precision=self.precision, dropout_rate=self.dropout_rate)
        norm = RMSNorm if self.norm == "rms" else LayerNorm
        self.norm1, self.norm2, self.norm3 = (norm(epsilon=self.norm_epsilon, dtype=self.dtype)
                                              for _ in range(3))
        self.dropout = Dropout(self.dropout_rate)

    def _dropped(self, x, train: bool):
        return self.dropout(x, deterministic=not train)

    def __call__(self, x, context=None, *, train: bool = False):
        if self.only_pure_attention:
            return self.attention2(x, context)

        if self.first_attention != "none":
            first = context if self.first_attention == "context" else None
            x = x + self._dropped(self.attention1(self.norm1(x), first), train)
        x = x + self._dropped(self.attention2(self.norm2(x), context), train)
        x = x + self.ff(self.norm3(x), train=train)
        return self._dropped(x, train)


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

    @nn.nowrap
    def cross_flops(self, lengths, input_features: int, context_lengths=None,
                    context_features: int | None = None) -> float:
        """Stage projections and its self/cross/FFN branches at actual widths."""
        inner = self.heads * self.dim_head if self.use_projection else input_features
        context_lengths = lengths if context_lengths is None else context_lengths
        context_features = inner if context_features is None else context_features
        block = BasicTransformerBlock(inner, self.heads, self.dim_head,
                                       first_attention="self" if self.use_self_and_cross else "none",
                                       only_pure_attention=self.only_pure_attention, parent=None)
        count = block.cross_flops(lengths, context_lengths, context_features)
        return count + (2 * projection_flops(lengths, input_features, inner) if self.use_projection else 0)

    @nn.compact
    def __call__(self, x, context=None):
        channels = x.shape[-1]
        if self.norm_inputs:
            x = RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)(x)

        def project(features: int, name: str):
            return Dense(features=features, use_bias=False, precision=self.precision,
                         dtype=self.dtype, name=name, dot_general=float32_dot_general)

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
            first_attention="self" if self.use_self_and_cross else "none",
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
