# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX's Splash attention TPU kernels, vendored, with the backward's delta formed from the weights.

The body of `jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py` in JAX's source
tree as of jax-v0.12.0 (license header above), carried for the one call Dew makes of it
(`dew.nn.attention.splash_attention`): multi-head attention over grouped key heads, the head dimension
minor, a static mask (`splash_attention_mask`, processed by JAX's public `splash_attention_mask_info`),
segment ids, sinks and a logit softcap, its backward in separate dQ and dK/dV kernels. Upstream's
references, its multi-query and sequence-minor layouts, its dynamic masks, its fused backward kernel,
saved residuals and manual sharding spec are left out; its long functions are cut into named parts and
its style moved to this tree's lint, every operation and its order as upstream's.

One change of logic, marked `# Dew:`: the backward's delta (D, the row sums the softmax's derivative
subtracts) is each query's sum_k p * dp, taken in a pass of the dQ kernel's own before the dQ and dK/dV
kernels from the same weights and products the dQ kernel forms, where upstream takes rowsum(o * do). The
two are equal in exact arithmetic; only the first cancels exactly where a query's weight is all on one key
(a causal first query, a document of one token), so that query's dQ is exactly zero, as the softmax's
derivative has it (1.2e-6 from a causal first query on a TPU v5e upstream's way). It reads each logits
block three times where upstream's fused kernel reads it once. It was Dew's JAX build's commit e963f3739,
whose `BlockSizes.delta_from_weights` it carries.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
    splash_attention_mask_info as mask_info_lib,
)

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # a @ b
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # a @ b.T

MaskFunction = Callable[..., jax.Array]


class SegmentIds(NamedTuple):
    """Segment ids along the query and key sequences: only a query and a key of equal ids attend. The
    static mask is and-ed with them, so no row of their product may be all false."""

    q: jax.Array  # [q_seq_len]
    kv: jax.Array  # [kv_seq_len]


@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
    """The kernels' tiles: the forward's query and key blocks and its key compute block, the dK/dV
    kernel's, and the dQ kernel's, which its delta pass takes too. They change no number, only speed."""

    block_q: int
    block_kv: int
    block_kv_compute: int
    block_q_dkv: int
    block_kv_dkv: int
    block_kv_dkv_compute: int
    block_q_dq: int
    block_kv_dq: int


def _kernel_name(segmented: bool, phase: str, residuals: bool = False) -> str:
    """A kernel's name, upstream's: its phase ('fwd', 'delta', 'dq', 'dkv') and variant."""
    segments = "_segmented" if segmented else ""
    return f"splash_mha_{phase}{segments}{'_residuals' if residuals else '_no_residuals'}"


def _next_nonzero(h, i, j, data_next_ref, block_mask_ref, m_next_ref, next_i=False):
    """The block a grid step reads (`data_next`), its partial mask's block, whether it runs, and whether it
    reads the whole block unmasked; every block where the mask has no block structure."""
    assert (data_next_ref is None) == (block_mask_ref is None)
    if data_next_ref is None and block_mask_ref is None:
        # No masking nor next data information: fetch the next data and mask every block.
        assert m_next_ref is None
        return i if next_i else j, None, True, False
    assert data_next_ref.shape == block_mask_ref.shape
    assert m_next_ref is None or data_next_ref.shape[0] == m_next_ref.shape[0]
    # One head's mask: force the head index to 0.
    if data_next_ref.shape[0] == 1:
        h = 0

    # Scalar memory narrower than int32 is upcast to be read in the kernel.
    def to_i32(x):
        return x.astype(jnp.int32)

    is_nonzero = to_i32(block_mask_ref[h, i, j]) > 0
    if m_next_ref is None:
        should_not_mask = True
        next_m = None
    else:
        should_not_mask = to_i32(block_mask_ref[h, i, j]) != 1
        next_m = to_i32(m_next_ref[h, i, j])
    next_j = to_i32(data_next_ref[h, i, j])
    return next_j, next_m, is_nonzero, should_not_mask


def _computed_mask(
    q_sequence_ref,
    mask_function: MaskFunction,
    *,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    k_in_lanes: bool,
) -> jax.Array:
    """A block's mask from its queries' places (`q_sequence_ref`) and its keys', which only a query
    sequence's sharding leaves to compute in the kernel."""
    if k_in_lanes:
        assert q_sequence_ref.shape == (bq, NUM_LANES)
        k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (bq, k_slice.size), 1)
        repeats, rem = divmod(k_slice.size, NUM_LANES)
        assert rem == 0
        q_sequence = jnp.tile(q_sequence_ref[...], (1, repeats))  # [bq, k_slice.size]
    else:
        assert q_sequence_ref.shape == (NUM_SUBLANES, bq)
        k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (k_slice.size, bq), 0)
        q_sequence = jnp.broadcast_to(q_sequence_ref[:1, :], (k_slice.size, bq))
    assert q_sequence.shape == k_sequence.shape
    computed = mask_function(q_sequence, k_sequence)
    if computed.dtype != jnp.dtype(jnp.bool_):
        raise ValueError(f"Mask function must return a boolean-valued array, but got: {computed.dtype}")
    return computed


def _segment_mask(
    q_segment_ids_ref, kv_segment_ids_ref, *, k_slice: pl.Slice, bq: int, k_in_lanes: bool
) -> jax.Array:
    """Whether each query and key of a block share a segment."""
    if k_in_lanes:
        kv_ids = kv_segment_ids_ref[:1, k_slice]  # [1, k_slice]
        repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
        if rem:
            raise NotImplementedError(f"block_kv must be a multiple of {NUM_LANES}")
        q_ids = jnp.tile(q_segment_ids_ref[:], (1, repeats))  # [bq, bkv]
    else:
        assert bq == q_segment_ids_ref.shape[-1]
        repeats, rem = divmod(bq, NUM_LANES)
        if rem:
            raise NotImplementedError(f"block_q must be a multiple of {NUM_LANES}")
        kv_ids = jnp.tile(kv_segment_ids_ref[k_slice, :], (1, repeats))  # [k_slice, bq]
        q_ids = q_segment_ids_ref[:1, :]  # [1, bq]
    return q_ids == kv_ids


def _apply_mask_and_soft_cap(
    qk: jax.Array,
    mask_value: float,
    should_not_mask,
    mask_ref,
    q_sequence_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    *,
    attn_logits_soft_cap: float | None,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    k_in_lanes=True,
    mask_function=None,
) -> jax.Array:
    """A block's logits capped (`attn_logits_soft_cap`), then `mask_value` wherever its partial mask, its
    computed mask or its segments keep a query from a key; its keys along its lanes or its sublanes."""
    assert mask_ref is None or q_sequence_ref is None
    assert (q_sequence_ref is None) == (mask_function is None)
    masks = []
    if mask_ref is not None:
        mask = mask_ref[:, k_slice] if k_in_lanes else mask_ref[k_slice, :]
        masks.append(jnp.bitwise_or(mask, jnp.broadcast_to(should_not_mask, mask.shape)))
    if mask_function is not None:
        masks.append(
            _computed_mask(
                q_sequence_ref,
                mask_function,
                k_slice=k_slice,
                k_offset=k_offset,
                bq=bq,
                k_in_lanes=k_in_lanes,
            )
        )
    if q_segment_ids_ref is not None:
        masks.append(
            _segment_mask(
                q_segment_ids_ref, kv_segment_ids_ref, k_slice=k_slice, bq=bq, k_in_lanes=k_in_lanes
            )
        )
    if attn_logits_soft_cap is not None:
        qk = jnp.tanh(qk / attn_logits_soft_cap) * attn_logits_soft_cap
    if masks:
        qk = jnp.where(functools.reduce(jnp.logical_and, masks), qk, mask_value)
    return qk


def _through_soft_cap(ds: jax.Array, qk_uncapped: jax.Array, attn_logits_soft_cap: float | None) -> jax.Array:
    """Score cotangents `ds` taken back through the softcap, from the uncapped logits."""
    if attn_logits_soft_cap is None:
        return ds
    d = jnp.tanh(qk_uncapped / attn_logits_soft_cap)
    g = ds * (1 - d)
    return g + g * d


def _div(dividend: int, divisor: int):
    """`dividend // divisor` in an index map, no division where the divisor is 1."""
    if divisor == 1:
        return dividend
    return lax.div(dividend, divisor)


class _Maps(NamedTuple):
    """A kernel's index maps over its grid: its queries' blocks (and the output's, the cotangent's), its
    keys' and values' (and their gradients'), the partial mask's, the queries' rows of segment ids and
    places, the keys' segment ids, and the queries' statistics (the log-sum-exp, the delta)."""

    query: Callable
    key: Callable
    mask: Callable
    query_rows: Callable
    key_columns: Callable
    stats: Callable


def _query_grid(groups: int) -> _Maps:
    """The index maps of a grid of (head, query block, key block) steps, which the forward and the dQ
    kernel walk: each step reads its key block where the mask has one (`_next_nonzero`), its key head the
    query head's group's."""

    def query(h, i, *_):
        return h, i, 0

    def key(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None, *_):
        next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
        return _div(h, groups), next_j, 0

    def mask(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None, *_):
        _, next_m, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
        return next_m, 0, 0

    def query_rows(h, i, j, *_):
        del h, j
        return i, 0

    def key_columns(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None, *_):
        next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
        return 0, next_j

    def stats(h, i, *_):
        return h, 0, i

    return _Maps(query, key, mask, query_rows, key_columns, stats)


def _key_grid(groups: int) -> _Maps:
    """The index maps of a grid of (key block, head, query block) steps, which the dK/dV kernel walks: each
    step reads its query block where the mask has one (`_next_nonzero`'s `next_i`)."""

    def next_query(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None):
        return _next_nonzero(
            head_index, q_index, kv_index, data_next_ref, block_mask_ref, mask_next_ref, next_i=True
        )

    def query(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None, *_):
        next_i, *_ = next_query(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref)
        return head_index, next_i, 0

    def key(kv_index, head_index, *_):
        return _div(head_index, groups), kv_index, 0

    def mask(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None, *_):
        _, next_m, *_ = next_query(
            kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref
        )
        return next_m, 0, 0

    def query_rows(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None, *_):
        next_i, *_ = next_query(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref)
        return 0, next_i

    def key_columns(kv_index, *_):
        return kv_index, 0

    def stats(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None, *_):
        next_i, *_ = next_query(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref)
        return head_index, 0, next_i

    return _Maps(query, key, mask, query_rows, key_columns, stats)


def _geometry(q: jax.Array, k: jax.Array, v: jax.Array, bq: int, bkv: int) -> tuple[int, int, int]:
    """A call's query heads, its keys and its query heads a key head, its shapes and blocks checked."""
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    num_kv_heads, kv_seq_len, _ = k.shape
    if k.shape[-1] != head_dim_qk:
        raise ValueError(f"Expected 'key' head dimension to be: {head_dim_qk}. Instead got: {k.shape[-1]}.")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
            f" multiple of the number of 'query' heads ({num_q_heads})"
        )
    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError(
            f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same leading dimensions."
        )
    if bq > q_seq_len:
        raise ValueError(f"{bq=} should not be greater than {q_seq_len=}")
    if bkv > kv_seq_len:
        raise ValueError(f"{bkv=} should not be greater than {kv_seq_len=}")
    return num_q_heads, kv_seq_len, num_q_heads // num_kv_heads


def _shared_operands(
    maps: _Maps,
    mask_info: mask_info_lib.MaskInfo,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    *,
    bq: int,
    bkv: int,
    q_seq_len: int,
    kv_seq_len: int,
    heads: int,
    keys_in_lanes: bool,
) -> tuple[list, list, list]:
    """The segment ids', the sinks', the partial mask's and the query places' specs and arrays every kernel
    reads, its keys along its lanes (the forward's and the dQ kernel's) or its sublanes (the dK/dV
    kernel's): the specs of the first three, the arrays of the first three, and the last's spec and array."""
    if segment_ids is None:
        segment_specs, segments = [None, None], [None, None]
    elif keys_in_lanes:
        segment_specs = [
            pl.BlockSpec((bq, NUM_LANES), maps.query_rows),
            pl.BlockSpec((NUM_SUBLANES, bkv), maps.key_columns),
        ]
        segments = [
            jax.lax.broadcast_in_dim(segment_ids.q, (q_seq_len, NUM_LANES), (0,)),
            jax.lax.broadcast_in_dim(segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,)),
        ]
    else:
        segment_specs = [
            pl.BlockSpec((NUM_SUBLANES, bq), maps.query_rows),
            pl.BlockSpec((bkv, NUM_LANES), maps.key_columns),
        ]
        segments = [
            jax.lax.broadcast_in_dim(segment_ids.q, (NUM_SUBLANES, q_seq_len), (1,)),
            jax.lax.broadcast_in_dim(segment_ids.kv, (kv_seq_len, NUM_LANES), (0,)),
        ]
    sinks_spec = None
    if sinks is not None:
        assert sinks.shape == (heads,)
        # Aligned to the sublanes, so the kernel vmaps and shard_maps.
        sinks_spec = pl.BlockSpec((NUM_SUBLANES, heads), lambda *_: (0, 0), memory_space=pltpu.SMEM)
        sinks = jnp.broadcast_to(sinks.astype(jnp.float32)[None, :], (NUM_SUBLANES, heads))
    mask_block = (None, bq, bkv) if keys_in_lanes else (None, bkv, bq)
    mask_spec = None if mask_info.partial_mask_blocks is None else pl.BlockSpec(mask_block, maps.mask)
    assert mask_info.partial_mask_blocks is None or mask_info.q_sequence is None
    q_sequence = q_sequence_spec = None
    if mask_info.q_sequence is not None:
        if keys_in_lanes:
            q_sequence = jax.lax.broadcast_in_dim(mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,))
            q_sequence_spec = pl.BlockSpec((bq, NUM_LANES), maps.query_rows)
        else:
            q_sequence = jax.lax.broadcast_in_dim(mask_info.q_sequence, (NUM_SUBLANES, q_seq_len), (1,))
            q_sequence_spec = pl.BlockSpec((NUM_SUBLANES, bq), maps.query_rows)
    return (
        [*segment_specs, sinks_spec],
        [*segments, sinks],
        [mask_spec, q_sequence_spec, mask_info.partial_mask_blocks, q_sequence],
    )


def _forward_step(
    kv_compute_index,
    _,
    *,
    refs: tuple,
    should_not_mask,
    global_kv_index,
    mask_value: float,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    attn_logits_soft_cap: float | None,
    mask_function: MaskFunction | None,
):
    """One key compute block of the forward's online softmax: its logits, the running maximum and
    denominator, and the output's running sum, rescaled."""
    (
        q_ref,
        k_ref,
        v_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        mask_ref,
        q_sequence_ref,
        m_scratch_ref,
        l_scratch_ref,
        o_scratch_ref,
    ) = refs
    slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    assert m_prev.shape == (bq, NUM_LANES)
    assert l_prev.shape == (bq, NUM_LANES)
    qk = lax.dot_general(q_ref[...], k_ref[slice_k, :], NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
    assert qk.shape == (bq, bkv_compute)
    assert isinstance(slice_k, pl.Slice)
    # Where the iteration space is shrunk (local attention), a key block's program id is not its place:
    # the mask reads the unshrunk index, `data_next`'s.
    qk = _apply_mask_and_soft_cap(
        qk,
        mask_value,
        should_not_mask,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=global_kv_index * bkv + kv_compute_index * bkv_compute,
        bq=bq,
        mask_function=mask_function,
    )
    m_curr = qk.max(axis=-1)[:, None]
    assert m_curr.shape == (bq, 1)
    m_next = jnp.maximum(m_prev, m_curr)
    assert m_next.shape == (bq, NUM_LANES)
    bkv_repeats, rem = divmod(bkv_compute, NUM_LANES)
    if rem != 0:
        raise NotImplementedError(f"{bkv_compute=} should be a multiple of {NUM_LANES}")
    s_curr = jnp.exp(qk - jnp.tile(m_next, (1, bkv_repeats)))
    assert s_curr.shape == (bq, bkv_compute)
    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
    assert l_curr.shape == (bq, NUM_LANES)
    alpha = jnp.exp(m_prev - m_next)
    l_next = l_curr + alpha * l_prev
    m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next
    o_curr = lax.dot_general(s_curr, v_ref[slice_k, :].astype(jnp.float32), NN_DIM_NUMBERS)
    alpha_o = jnp.tile(alpha, (1, pl.cdiv(head_dim_v, NUM_LANES)))[..., : o_scratch_ref.shape[-1]]
    o_scratch_ref[:] = alpha_o * o_scratch_ref[:] + o_curr


def _forward_kernel(
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    sinks_ref,
    mask_ref,
    q_sequence_ref,
    o_ref,
    logsumexp_ref,
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    *,
    grid_width: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    **step,
):
    """The forward over a (head, query block, key block) grid: the running state from the sinks or empty,
    each key block's compute blocks (`_forward_step`), then the output normalized and the log-sum-exp."""
    h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    @pl.when(j == 0)
    def init():
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
        if sinks_ref is not None:
            # max = sinks, so exp(sinks - max) = 1.
            m_scratch_ref[...] = sinks_ref[0, h].astype(m_scratch_ref.dtype) * jnp.ones_like(m_scratch_ref)
            l_scratch_ref[...] = jnp.ones_like(l_scratch_ref)
        else:
            m_scratch_ref[...] = jnp.full_like(m_scratch_ref, step["mask_value"])
            l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

    global_kv_index, _, should_run, should_not_mask = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )
    refs = (
        q_ref,
        k_ref,
        v_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        mask_ref,
        q_sequence_ref,
        m_scratch_ref,
        l_scratch_ref,
        o_scratch_ref,
    )
    body = functools.partial(
        _forward_step,
        refs=refs,
        should_not_mask=should_not_mask,
        global_kv_index=global_kv_index,
        bkv=bkv,
        bkv_compute=bkv_compute,
        head_dim_v=head_dim_v,
        **step,
    )

    @pl.when(should_run)
    def run():
        assert bkv % bkv_compute == 0
        lax.fori_loop(0, k_ref.shape[0] // bkv_compute, body, None, unroll=True)

    @pl.when(j == grid_width - 1)
    def end():
        total = l_scratch_ref[...]
        l_inv = jnp.tile(1.0 / total, (1, pl.cdiv(head_dim_v, NUM_LANES)))[..., : o_scratch_ref.shape[-1]]
        o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
        if logsumexp_ref is not None:
            assert logsumexp_ref.shape == (step["bq"], NUM_LANES)
            logsumexp_ref[...] = (jnp.log(total) + m_scratch_ref[...]).astype(logsumexp_ref.dtype)


class _Call(NamedTuple):
    """What every kernel of one call takes: its mask value, softcap, mask function and interpreter flag."""

    mask_value: float
    attn_logits_soft_cap: float | None
    mask_function: MaskFunction | None
    interpret: bool


def _forward(
    mask_info: mask_info_lib.MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    *,
    blocks: BlockSizes,
    call: _Call,
    residuals: bool,
) -> tuple[jax.Array, jax.Array | None]:
    """The forward kernel's output `[H, Q, Dv]`, and with `residuals` each query's log-sum-exp `[H, Q]`."""
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    head_dim_v = v.shape[-1]
    bq, bkv, bkv_compute = blocks.block_q, blocks.block_kv, blocks.block_kv_compute
    partial_mask_blocks = mask_info.partial_mask_blocks
    if partial_mask_blocks is not None and jnp.dtype(partial_mask_blocks.dtype) != np.bool_:
        raise ValueError(f"partial_mask_blocks must be of type np.bool_ but got {partial_mask_blocks.dtype}")
    heads, kv_seq_len, groups = _geometry(q, k, v, bq, bkv)
    if bkv % bkv_compute:
        raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
    if bkv_compute % NUM_LANES:
        raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")
    if segment_ids is not None and segment_ids.q.shape != (q_seq_len,):
        raise ValueError(f"Invalid shape for q segment_ids: {segment_ids.q.shape}. Expected: {(q_seq_len,)}")
    if segment_ids is not None and segment_ids.kv.shape != (kv_seq_len,):
        raise ValueError(
            f"Invalid shape for kv segment_ids: {segment_ids.kv.shape}. Expected: {(kv_seq_len,)}"
        )
    maps = _query_grid(groups)
    specs, arrays, (mask_spec, q_sequence_spec, mask_blocks, q_sequence) = _shared_operands(
        maps,
        mask_info,
        segment_ids,
        sinks,
        bq=bq,
        bkv=bkv,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        heads=heads,
        keys_in_lanes=True,
    )
    in_specs = [
        pl.BlockSpec((None, bq, head_dim_qk), maps.query),
        pl.BlockSpec((None, bkv, head_dim_qk), maps.key),
        pl.BlockSpec((None, bkv, head_dim_v), maps.key),
        *specs,
        mask_spec,
        q_sequence_spec,
    ]
    out_shapes = [
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim_v), q.dtype),
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32) if residuals else None,
    ]
    out_specs = [
        pl.BlockSpec((None, bq, head_dim_v), maps.query),
        pl.BlockSpec((None, bq, NUM_LANES), maps.query) if residuals else None,
    ]
    scratch_shapes = [
        pltpu.VMEM((bq, NUM_LANES), jnp.float32),
        pltpu.VMEM((bq, NUM_LANES), jnp.float32),
        pltpu.VMEM((bq, head_dim_v), jnp.float32),
    ]
    name = _kernel_name(segment_ids is not None, "fwd", residuals)
    grid_width = kv_seq_len // bkv if mask_info.data_next is None else mask_info.data_next.shape[-1]
    sizes = {"block_q": bq, "block_kv": bkv, "block_kv_compute": bkv_compute}
    kernel = functools.partial(
        _forward_kernel,
        grid_width=grid_width,
        bq=bq,
        bkv=bkv,
        bkv_compute=bkv_compute,
        head_dim_v=head_dim_v,
        mask_value=call.mask_value,
        attn_logits_soft_cap=call.attn_logits_soft_cap,
        mask_function=call.mask_function,
    )
    with jax.named_scope(name):
        out, logsumexp = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                in_specs=in_specs,
                out_specs=out_specs,
                grid=(num_q_heads, q_seq_len // bq, grid_width),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary")),
            out_shape=out_shapes,
            name=name,
            interpret=call.interpret,
            metadata={"xprof_metadata": json.dumps(sizes)},
        )(
            mask_info.data_next,
            mask_info.block_mask,
            mask_info.mask_next,
            q,
            k,
            v,
            *arrays,
            mask_blocks,
            q_sequence,
        )
    return out, (logsumexp[..., 0] if residuals else None)


def _dq_kernel(
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    sinks_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    dq_ref,
    dq_scratch_ref,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    attn_logits_soft_cap: float | None,
    mask_function: MaskFunction | None,
    delta: bool,
):
    """dQ over a (head, query block, key block) grid, or with `delta` each query's delta (`_dq`)."""
    del sinks_ref  # its gradient is the backward's own (`_backward`)
    h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    @pl.when(j == 0)
    def init():
        dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

    global_kv_index, _, should_run, should_not_mask = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )

    @pl.when(should_run)
    def run():
        k, v = k_ref[...], v_ref[...]
        logsumexp = jnp.expand_dims(logsumexp_ref[0], -1)
        do = do_ref[...]
        qk_uncapped = lax.dot_general(q_ref[...], k, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
        k_slice = pl.ds(0, bkv)
        assert isinstance(k_slice, pl.Slice)
        qk = _apply_mask_and_soft_cap(
            qk_uncapped,
            mask_value,
            should_not_mask,
            mask_ref,
            q_sequence_ref,
            q_segment_ids_ref,
            kv_segment_ids_ref,
            attn_logits_soft_cap=attn_logits_soft_cap,
            k_slice=k_slice,
            k_offset=global_kv_index * bkv,
            bq=bq,
            mask_function=mask_function,
        )
        p = jnp.exp(qk - logsumexp)
        dp = lax.dot_general(do.astype(v.dtype), v, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
        if delta:
            # Dew: the block's share of the query's delta, sum_k p * dp, lane-broadcast.
            dq_scratch_ref[...] += jnp.broadcast_to(
                jnp.sum(p * dp, axis=-1, keepdims=True), dq_scratch_ref.shape
            )
            return
        ds = _through_soft_cap((dp - jnp.expand_dims(di_ref[0], -1)) * p, qk_uncapped, attn_logits_soft_cap)
        dq_scratch_ref[...] += lax.dot_general(
            ds.astype(k.dtype), k, NN_DIM_NUMBERS, preferred_element_type=jnp.float32
        )

    @pl.when(j == grid_width - 1)
    def end():
        dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)


def _dq(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    logsumexp: jax.Array,
    do: jax.Array,
    di: jax.Array | None,
    *,
    blocks: BlockSizes,
    mask_info: mask_info_lib.MaskInfo,
    call: _Call,
) -> jax.Array:
    """dQ `[H, Q, D]`, or where `di` is None each query's delta `[H, Q]` fp32, sum_k p * dp, from the same
    weights and products (Dew's: the delta pass)."""
    delta = di is None
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    head_dim_v = v.shape[-1]
    bq, bkv = blocks.block_q_dq, blocks.block_kv_dq
    heads, kv_seq_len, groups = _geometry(q, k, v, bq, bkv)
    if bkv % NUM_LANES:
        raise ValueError(f"{bkv=} must be a multiple of {NUM_LANES}.")
    maps = _query_grid(groups)
    specs, arrays, (mask_spec, q_sequence_spec, mask_blocks, q_sequence) = _shared_operands(
        maps,
        mask_info,
        segment_ids,
        sinks,
        bq=bq,
        bkv=bkv,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        heads=heads,
        keys_in_lanes=True,
    )
    logsumexp = jnp.expand_dims(logsumexp, axis=-2)
    stats_spec = pl.BlockSpec((None, 1, bq), maps.stats)
    di_spec = None if di is None else stats_spec
    di = None if di is None else jnp.expand_dims(di, axis=-2)
    in_specs = [
        pl.BlockSpec((None, bq, head_dim_qk), maps.query),
        pl.BlockSpec((None, bkv, head_dim_qk), maps.key),
        pl.BlockSpec((None, bkv, head_dim_v), maps.key),
        *specs,
        stats_spec,
        pl.BlockSpec((None, bq, head_dim_v), maps.query),
        di_spec,
        mask_spec,
        q_sequence_spec,
    ]
    # Dew: the result is dQ, or each query's delta broadcast across the lanes; the accumulator as wide.
    width = NUM_LANES if delta else head_dim_qk
    out_shape = (
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, width), jnp.float32)
        if delta
        else jax.ShapeDtypeStruct.like(q)
    )
    grid_width = kv_seq_len // bkv if mask_info.data_next is None else mask_info.data_next.shape[-1]
    kernel = functools.partial(
        _dq_kernel,
        grid_width=grid_width,
        mask_value=call.mask_value,
        bq=bq,
        bkv=bkv,
        attn_logits_soft_cap=call.attn_logits_soft_cap,
        mask_function=call.mask_function,
        delta=delta,
    )
    name = _kernel_name(segment_ids is not None, "delta" if delta else "dq")
    with jax.named_scope(name):
        (dq,) = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                in_specs=in_specs,
                out_specs=[pl.BlockSpec((None, bq, width), maps.query)],
                grid=(num_q_heads, q_seq_len // bq, grid_width),
                scratch_shapes=[pltpu.VMEM((bq, width), jnp.float32)],
            ),
            out_shape=[out_shape],
            compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary", "arbitrary")),
            name=name,
            interpret=call.interpret,
            metadata={"xprof_metadata": json.dumps({"block_q_dq": bq, "block_kv_dq": bkv})},
        )(
            mask_info.data_next,
            mask_info.block_mask,
            mask_info.mask_next,
            q,
            k,
            v,
            *arrays,
            logsumexp,
            do,
            di,
            mask_blocks,
            q_sequence,
        )
    return dq[..., 0] if delta else dq


def _dkv_step(
    i,
    _,
    *,
    refs: tuple,
    should_not_mask,
    kv_index,
    mask_value: float,
    bq: int,
    bkv: int,
    bkv_compute: int,
    attn_logits_soft_cap: float | None,
    mask_function: MaskFunction | None,
):
    """One key compute block of the dK/dV kernel: its logits, keys along its sublanes, its weights from the
    log-sum-exp, and dV's and dK's products added into their running sums."""
    (
        q_ref,
        k_ref,
        v_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        logsumexp_ref,
        do_ref,
        di_ref,
        mask_ref,
        q_sequence_ref,
        dk_scratch_ref,
        dv_scratch_ref,
    ) = refs
    slice_k = pl.ds(i * bkv_compute, bkv_compute)
    q = q_ref[...]
    k, v = k_ref[slice_k, :], v_ref[slice_k, :]
    logsumexp = logsumexp_ref[:1, :]
    do = do_ref[...]
    di = di_ref[:1, :]
    qk_uncapped = lax.dot_general(k, q, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
    assert isinstance(slice_k, pl.Slice)
    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        should_not_mask,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=kv_index * bkv + i * bkv_compute,
        bq=bq,
        k_in_lanes=False,
        mask_function=mask_function,
    )
    p = jnp.exp(qk - logsumexp)
    dv = lax.dot(p.astype(do.dtype), do, preferred_element_type=jnp.float32)
    dv_scratch_ref[slice_k, :] = dv.astype(dv_scratch_ref.dtype) + dv_scratch_ref[slice_k, :]
    dp = lax.dot_general(v, do, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
    ds = _through_soft_cap((dp - di) * p, qk_uncapped, attn_logits_soft_cap)
    dk = lax.dot_general(ds.astype(do.dtype), q, NN_DIM_NUMBERS, preferred_element_type=jnp.float32)
    dk_scratch_ref[slice_k, :] = dk.astype(dk_scratch_ref.dtype) + dk_scratch_ref[slice_k, :]


def _dkv_kernel(
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    sinks_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    dk_ref,
    dv_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    grid_width: int,
    bkv_compute: int,
    **step,
):
    """dK and dV over a (key block, head, query block) grid: each key head's sums start at the first query
    head of its group to read it and are written at its last (query heads 0, 2, 4, 6 start key heads 0 to 3
    of eight query heads over four, and 1, 3, 5, 7 write them)."""
    del sinks_ref  # its gradient is the backward's own (`_backward`)
    kv_index, q_head_index, q_index = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    should_initialize = q_index == 0
    q_heads_per_kv_heads = q_head_index_per_kv_head = None
    if num_kv_heads < num_q_heads:
        q_heads_per_kv_heads = num_q_heads // num_kv_heads
        q_head_index_per_kv_head = lax.rem(q_head_index, q_heads_per_kv_heads)
        should_initialize = jnp.logical_and(should_initialize, q_head_index_per_kv_head == 0)

    @pl.when(should_initialize)
    def init():
        dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
        dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

    _, _, should_run, should_not_mask = _next_nonzero(
        q_head_index, q_index, kv_index, data_next_ref, block_mask_ref, mask_next_ref, next_i=True
    )
    refs = (
        q_ref,
        k_ref,
        v_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        logsumexp_ref,
        do_ref,
        di_ref,
        mask_ref,
        q_sequence_ref,
        dk_scratch_ref,
        dv_scratch_ref,
    )
    body = functools.partial(
        _dkv_step,
        refs=refs,
        should_not_mask=should_not_mask,
        kv_index=kv_index,
        bkv_compute=bkv_compute,
        **step,
    )

    @pl.when(should_run)
    def run():
        lax.fori_loop(0, k_ref.shape[0] // bkv_compute, body, None, unroll=True)

    should_write = q_index == grid_width - 1
    if num_kv_heads < num_q_heads:
        assert q_heads_per_kv_heads is not None
        should_write = jnp.logical_and(should_write, q_head_index_per_kv_head == q_heads_per_kv_heads - 1)

    @pl.when(should_write)
    def end():
        dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
        dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)


def _dkv(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    logsumexp: jax.Array,
    do: jax.Array,
    di: jax.Array,
    *,
    blocks: BlockSizes,
    mask_info: mask_info_lib.MaskInfo,
    call: _Call,
) -> tuple[jax.Array, jax.Array]:
    """dK `[Hk, K, D]` and dV `[Hk, K, Dv]`."""
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    head_dim_v = v.shape[-1]
    bq, bkv, bkv_compute = blocks.block_q_dkv, blocks.block_kv_dkv, blocks.block_kv_dkv_compute
    heads, kv_seq_len, groups = _geometry(q, k, v, bq, bkv)
    if bkv_compute > bkv:
        raise ValueError(f"{bkv_compute=} should not be greater than {bkv=}")
    if bkv % bkv_compute:
        raise ValueError(f"{bkv=} should be a multiple of {bkv_compute=}")
    maps = _key_grid(groups)
    specs, arrays, (mask_spec, q_sequence_spec, mask_blocks, q_sequence) = _shared_operands(
        maps,
        mask_info,
        segment_ids,
        sinks,
        bq=bq,
        bkv=bkv,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        heads=heads,
        keys_in_lanes=False,
    )
    assert logsumexp.shape == di.shape == (num_q_heads, q_seq_len)
    # Expanded across the sublanes until Mosaic has every retiling (upstream's).
    stats_shape = (num_q_heads, NUM_SUBLANES, q_seq_len)
    logsumexp = jnp.broadcast_to(jnp.expand_dims(logsumexp, -2), stats_shape)
    di = jnp.broadcast_to(jnp.expand_dims(di, -2), stats_shape)
    stats_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), maps.stats)
    in_specs = [
        pl.BlockSpec((None, bq, head_dim_qk), maps.query),
        pl.BlockSpec((None, bkv, head_dim_qk), maps.key),
        pl.BlockSpec((None, bkv, head_dim_v), maps.key),
        *specs,
        stats_spec,
        pl.BlockSpec((None, bq, head_dim_v), maps.query),
        stats_spec,
        mask_spec,
        q_sequence_spec,
    ]
    out_specs = [
        pl.BlockSpec((None, bkv, head_dim_qk), maps.key),
        pl.BlockSpec((None, bkv, head_dim_v), maps.key),
    ]
    scratch_shapes = [pltpu.VMEM((bkv, head_dim_qk), jnp.float32), pltpu.VMEM((bkv, head_dim_v), jnp.float32)]
    grid_width = q_seq_len // bq if mask_info.data_next is None else mask_info.data_next.shape[-2]
    kernel = functools.partial(
        _dkv_kernel,
        mask_value=call.mask_value,
        num_q_heads=num_q_heads,
        num_kv_heads=num_q_heads // groups,
        grid_width=grid_width,
        bq=bq,
        bkv_compute=bkv_compute,
        attn_logits_soft_cap=call.attn_logits_soft_cap,
        bkv=bkv,
        mask_function=call.mask_function,
    )
    name = _kernel_name(segment_ids is not None, "dkv")
    sizes = {"block_q_dkv": bq, "block_kv_dkv": bkv, "block_kv_dkv_compute": bkv_compute}
    with jax.named_scope(name):
        dk, dv = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                in_specs=in_specs,
                out_specs=out_specs,
                grid=(kv_seq_len // bkv, num_q_heads, grid_width),
                scratch_shapes=scratch_shapes,
            ),
            out_shape=[jax.ShapeDtypeStruct.like(k), jax.ShapeDtypeStruct.like(v)],
            # Every dimension arbitrary: the prefetch schedule assumes no megacore along the keys, and the
            # heads and the queries are reduced over.
            compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary", "arbitrary")),
            name=name,
            interpret=call.interpret,
            metadata={"xprof_metadata": json.dumps(sizes)},
        )(
            mask_info.data_next,
            mask_info.block_mask,
            mask_info.mask_next,
            q,
            k,
            v,
            *arrays,
            logsumexp,
            do,
            di,
            mask_blocks,
            q_sequence,
        )
    return dk, dv


_NONDIFF = ("blocks", "call")


@functools.partial(jax.custom_vjp, nondiff_argnames=_NONDIFF)
def _attention(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dq_mask_info: mask_info_lib.MaskInfo,
    dkv_mask_info: mask_info_lib.MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    blocks: BlockSizes,
    call: _Call,
) -> jax.Array:
    """The output `[H, Q, Dv]`. The backward's mask infos ride as arguments, so a sharded call's backward
    reads its own device's slice of them, as its forward does."""
    del dq_mask_info, dkv_mask_info
    return _forward(fwd_mask_info, q, k, v, segment_ids, sinks, blocks=blocks, call=call, residuals=False)[0]


def _attention_fwd(
    fwd_mask_info, dq_mask_info, dkv_mask_info, q, k, v, segment_ids, sinks, blocks: BlockSizes, call: _Call
):
    out, logsumexp = _forward(
        fwd_mask_info, q, k, v, segment_ids, sinks, blocks=blocks, call=call, residuals=True
    )
    return out, (q, k, v, segment_ids, sinks, out, logsumexp, dq_mask_info, dkv_mask_info)


def _attention_bwd(blocks: BlockSizes, call: _Call, residuals: tuple, do: jax.Array):
    """dQ, dK and dV from the delta pass, the dK/dV kernel and the dQ kernel; the sinks' gradient."""
    q, k, v, segment_ids, sinks, o, logsumexp, dq_mask_info, dkv_mask_info = residuals
    # Dew: di is each query's sum_k p * dp from the dQ kernel's own pass, not rowsum(o * do).
    di = _dq(
        q, k, v, segment_ids, sinks, logsumexp, do, None, blocks=blocks, mask_info=dq_mask_info, call=call
    )
    dk, dv = _dkv(
        q, k, v, segment_ids, sinks, logsumexp, do, di, blocks=blocks, mask_info=dkv_mask_info, call=call
    )
    dq = _dq(q, k, v, segment_ids, sinks, logsumexp, do, di, blocks=blocks, mask_info=dq_mask_info, call=call)
    dsinks = None
    if sinks is not None:
        sinks_exp = -jnp.exp(
            sinks[..., None, None].astype(jnp.float32) - logsumexp[..., None].astype(jnp.float32)
        )
        dsinks = jnp.sum(sinks_exp * o.astype(jnp.float32) * do.astype(jnp.float32), axis=(-1, -2)).astype(
            sinks.dtype
        )
    return None, None, None, dq, dk, dv, None, dsinks


_attention.defvjp(_attention_fwd, _attention_bwd)


def _collapsed(mask_info: mask_info_lib.MaskInfo) -> mask_info_lib.MaskInfo:
    """`mask_info` with its partial mask blocks' leading dimensions (heads, query blocks, key blocks)
    collapsed into one, as the kernels take them."""
    if mask_info.partial_mask_blocks is None:
        return mask_info
    blocks = mask_info.partial_mask_blocks
    return mask_info._replace(partial_mask_blocks=blocks.reshape(-1, *blocks.shape[-2:]))


@functools.partial(jax.jit, static_argnames=_NONDIFF)
def _splash(
    fwd_mask_info,
    dq_mask_info,
    dkv_mask_info,
    q,
    k,
    v,
    segment_ids,
    sinks,
    *,
    blocks: BlockSizes,
    call: _Call,
) -> jax.Array:
    return _attention(
        _collapsed(fwd_mask_info),
        _collapsed(dq_mask_info),
        _collapsed(dkv_mask_info),
        q,
        k,
        v,
        segment_ids,
        sinks,
        blocks,
        call,
    )


@tree_util.register_pytree_node_class
class SplashAttentionKernel:
    """A Splash kernel over one mask: its forward's, dQ kernel's and dK/dV kernel's mask infos (its leaves,
    so a vmap or a shard_map carries them) and its call's static facts."""

    def __init__(
        self,
        fwd_mask_info: mask_info_lib.MaskInfo,
        dq_mask_info: mask_info_lib.MaskInfo,
        dkv_mask_info: mask_info_lib.MaskInfo,
        *,
        blocks: BlockSizes,
        call: _Call,
    ):
        self.fwd_mask_info, self.dq_mask_info, self.dkv_mask_info = fwd_mask_info, dq_mask_info, dkv_mask_info
        self.blocks, self.call = blocks, call

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        segment_ids: SegmentIds | None = None,
        sinks: jax.Array | None = None,
    ) -> jax.Array:
        """Attention of `q` `[H, Q, D]` over `k` `[Hk, K, D]` and `v` `[Hk, K, Dv]`, `sinks` `[H]`."""
        with jax.named_scope("static_sparsity_splash"):
            return _splash(
                self.fwd_mask_info,
                self.dq_mask_info,
                self.dkv_mask_info,
                q,
                k,
                v,
                segment_ids,
                sinks,
                blocks=self.blocks,
                call=self.call,
            )

    def tree_flatten(self):
        return (self.fwd_mask_info, self.dq_mask_info, self.dkv_mask_info), (self.blocks, self.call)

    @classmethod
    def tree_unflatten(cls, facts, values):
        # A NamedTuple's type does not survive a pytree's serialization.
        fwd, dq, dkv = (mask_info_lib.MaskInfo(*leaves) for leaves in values)
        blocks, call = facts
        return cls(fwd, dq, dkv, blocks=blocks, call=call)


def make_splash_mha(
    mask: np.ndarray | mask_lib.MultiHeadMask,
    *,
    block_sizes: BlockSizes,
    attn_logits_soft_cap: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    interpret: bool = False,
) -> SplashAttentionKernel:
    """A Splash kernel over the static `mask` `[H, Q, K]`, one device's, at `block_sizes`."""
    if len(mask.shape) != 3:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    if isinstance(mask, np.ndarray):
        mask = mask_lib.MultiHeadMask([mask_lib.NumpyMask(head_mask) for head_mask in mask])
    fwd_mask_info, mask_function_fwd = mask_info_lib.process_mask(
        mask, (block_sizes.block_q, block_sizes.block_kv), head_shards=1, q_seq_shards=1
    )
    dq_mask_info, mask_function_dq = mask_info_lib.process_mask(
        mask, (block_sizes.block_q_dq, block_sizes.block_kv_dq), head_shards=1, q_seq_shards=1
    )
    dkv_mask_info, mask_function_dkv = mask_info_lib.process_mask_dkv(
        mask,
        (block_sizes.block_q_dkv, block_sizes.block_kv_dkv),
        head_shards=1,
        q_seq_shards=1,
        shrink_grid=True,
    )
    assert (mask_function_fwd is None) == (mask_function_dq is None) == (mask_function_dkv is None)
    infos = (
        tree_util.tree_map(jnp.array, blocked) for blocked in (fwd_mask_info, dq_mask_info, dkv_mask_info)
    )
    return SplashAttentionKernel(
        *infos, blocks=block_sizes, call=_Call(mask_value, attn_logits_soft_cap, mask_function_fwd, interpret)
    )
