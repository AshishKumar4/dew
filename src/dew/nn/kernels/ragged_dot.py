# Copyright 2026 The JAX Authors.
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

"""JAX's Pallas/Triton grouped matmul kernels, `gmm` and `tgmm`, vendored.

`gmm` is `[m, k] x [g, k, n] -> [m, n]` over rows sorted by group (with
`trans_rhs`, `[g, n, k]`), and `tgmm` the kernel gradient, `[m, k] x [m, n]
-> [g, k, n]`. They are the body of `jax/_src/lax/pallas_lowerings/gpu/
ragged_dot.py` in the jax-v0.11.2 source tree (tag jax-v0.11.2, 32544801;
license header above), which no jax wheel ships: `DEFAULT_BLOCK_M` through
`_hyperparam_selection_rule`, restyled to this tree's lint gate and moved
from `jax._src` imports to their public equivalents, with lines of logic
changed and marked `# Dew:`: the prologue and epilogue hooks with their side
operands (`_side_tile`), tgmm's output cast, and gmm's zeroing when every
group is empty. The file's `ragged_dot_general`
adapter is left out: Dew calls the kernels itself, through the GEMM template
(`templates.gemm.template`), which differentiates them, the way MaxText calls megablox.

Every global load and store is masked on every dimension, rows included, and
each output past the routed rows is the epilogue of a zero accumulator, zero
without one. The kernels multiply in
`compute_dtype`, accumulate in `acc_dtype` and ignore any precision: with
16-bit operands that is exact products summed in fp32, with fp32 operands
the GPU's TF32 rate.
"""

import math
import typing
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jax.typing import DTypeLike

DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 64
DEFAULT_BLOCK_K = 64

# In the simplest strategy, each of our kernel invocation would attempt to
# handle one full group. However, for very large groups (or when there are only
# a few groups), each kernel invocation would do a lot of work. We chunk within
# group to remedy this and increase the number of SMs that can work on each
# group. This "chunk" is not really tunable and it'll be further tiled with
# block_m. It solves the problem of large groups or very few groups.
CHUNK_M = 512  # block size to chunk rows into to increase SM participation for very large groups

arange = partial(jnp.arange, dtype=np.int32)


class RaggedDotSizes(typing.NamedTuple):
  m: int
  k: int
  n: int
  g: int


class BlockSizes(typing.NamedTuple):
  m: int
  k: int
  n: int


class GMMGroupLookupMetadata(typing.NamedTuple):
  """Metadata for splitting large groups into smaller chunks for ragged_dot,
  trans_ragged_dot doesn't use this."""

  group_sizes: jax.Array  # the regular group sizes
  group_offsets: jax.Array  # the group offsets
  row_it2group: jax.Array  # mapping between the first grid dim and group id
  row_it2it_in_group: jax.Array  # mapping between the first grid dim and which
                                 # chunk in the group we're computing
  total_row_its: jax.Array       # first grid dim upper bound / number of chunks


def _make_gmm_group_metadata(
  group_sizes: jax.Array, m: int, chunk_m: int
) -> GMMGroupLookupMetadata:
  """Split large groups into chunks to let several SMs work on one group. For
  ragged_dot, not trans_ragged_dot."""
  group_sizes = group_sizes.astype(np.int32)
  g: int = group_sizes.size
  blocks_per_group = (
    group_sizes + chunk_m - 1
  ) // chunk_m  # ceil div of group sizes into chunks

  # temporary variables to ragged concat group metadata into a single flat
  # vector via 2D shifted construction + reduce
  _blocks_offset = jnp.cumsum(blocks_per_group) - blocks_per_group
  _blocks_upper_bound = (m + chunk_m - 1) // chunk_m + g
  _group_iota = arange(_blocks_upper_bound)[None, :] - _blocks_offset[:, None]
  _group_write_mask = (_group_iota >= 0) & (
    _group_iota < blocks_per_group[:, None]
  )

  row_it2group = jnp.sum(
      jnp.where(_group_write_mask, arange(g)[:, None], 0), 0, dtype=np.int32
  )  # block row i to group id
  row_it2it_in_group = jnp.sum(
      jnp.where(_group_write_mask, _group_iota, 0), 0, dtype=np.int32
  )  # block row i to within group chunk

  return GMMGroupLookupMetadata(
      group_sizes=group_sizes,
      group_offsets=jnp.cumsum(group_sizes) - group_sizes,
      row_it2group=row_it2group,
      row_it2it_in_group=row_it2it_in_group,
      total_row_its=jnp.sum(blocks_per_group, dtype=np.int32),
  )


def _side_tile(ref, axes, origins: dict, masks: dict):
  """Dew: a side operand's tile, loaded at the launch's own origins: each of its dimensions follows the
  launch axis `axes` names ('rows', 'depth', 'columns'), its start, extent and mask from `origins` and
  `masks`. A vector is one axis's, which the hook broadcasts itself."""
  index = tuple(pl.ds(*origins[axis]) for axis in axes)
  first, *rest = (masks[axis] for axis in axes)
  mask = first[:, None] & rest[0][None, :] if rest else first
  return plgpu.load(ref.at[index], mask=mask, other=0)


def _gpu_ragged_dot_kernel(
  # inputs
  x_ref,  # [m, k]
  A_ref,  # [k, n] or [n, k]
  group_metadata_ref: GMMGroupLookupMetadata,
  # Dew: the prologue's and the epilogue's side operands, then the outputs
  *refs,
  # static problem shapes
  chunk_m: int,
  trans_rhs: bool,
  size: RaggedDotSizes,
  block: BlockSizes,  # hyperparameters
  compute_dtype: DTypeLike | None = None,
  acc_dtype: DTypeLike = np.float32,
  prologue=None,
  prologue_axes: tuple = (),
  epilogue=None,
  epilogue_axes: tuple = (),
  exact_sums: bool = False,
):
  prologue_refs = refs[:len(prologue_axes)]
  epilogue_refs = refs[len(prologue_axes):len(prologue_axes) + len(epilogue_axes)]
  out_refs = refs[len(prologue_axes) + len(epilogue_axes):]
  i, j = pl.program_id(0), pl.program_id(1)
  pid = SimpleNamespace(i=i, j=j, gi=group_metadata_ref.row_it2group[i])
  size = jax.tree.map(partial(np.array, dtype=np.int32), size)

  local_inc = chunk_m * group_metadata_ref.row_it2it_in_group[pid.i]
  valid = pid.i < group_metadata_ref.total_row_its[...]
  group_sz = jnp.where(valid, jnp.clip(group_metadata_ref.group_sizes[pid.gi]
                                       - local_inc, min=0, max=chunk_m), 0)
  compute_dtype = compute_dtype if compute_dtype is not None else x_ref.dtype

  # row index into lhs and output
  start_ridx = group_metadata_ref.group_offsets[pid.gi] + local_inc

  def outer_compute(r_offset, _):
    ridx = (
      start_ridx + r_offset * block.m
    )  # r_offset is 0,1,2,... need to map it to actual row indices
    lhs_rows_mask = (r_offset * block.m + arange(block.m)) < group_sz
    lhs_rows_idx = pl.ds(ridx, block.m)
    rhs_cols_idx = pl.ds(0, block.n)
    rhs_cols_mask = (block.n * pid.j + arange(block.n)) < size.n

    def inner_compute(k, acc):
      inner_idx = pl.ds(k * block.k, block.k)
      inner_mask = (k * block.k + arange(block.k)) < size.k
      _load = partial(plgpu.load, other=0)
      mask = lhs_rows_mask[:, None] & inner_mask[None, :]
      x = _load(x_ref.at[lhs_rows_idx, inner_idx], mask=mask)
      if prologue is not None:
        # Dew: the prologue, on the loaded tile and its side operands' tiles.
        tiles = [_side_tile(ref, axes, {'rows': (ridx, block.m), 'depth': (k * block.k, block.k)},
                            {'rows': lhs_rows_mask, 'depth': inner_mask})
                 for ref, axes in zip(prologue_refs, prologue_axes, strict=True)]
        # Dew: the padding a hook made something of (1 / 0) is zero again before the product.
        x = jnp.where(mask, prologue(x, *tiles), 0)
      if not trans_rhs:
        mask = inner_mask[:, None] & rhs_cols_mask[None, :]
        A = _load(A_ref.at[pid.gi, inner_idx, rhs_cols_idx], mask=mask)
      else:
        mask = rhs_cols_mask[:, None] & inner_mask[None, :]
        A = _load(A_ref.at[pid.gi, rhs_cols_idx, inner_idx], mask=mask)
      x, A = x.astype(compute_dtype), A.astype(compute_dtype)
      dim_nums = (
          (((1,), (0,)), ((), ()))  # normal matmul product x @ y
          if not trans_rhs else
          (((1,), (1,)), ((), ()))  # transposed matmul product x @ y.T
      )
      # Dew: the default precision named, so a configured one does not reach the kernel (TF32 for fp32).
      xA = lax.dot_general(
          x, A, dimension_numbers=dim_nums, precision=lax.Precision.DEFAULT, preferred_element_type=acc_dtype
      )
      return _summed(acc, xA.astype(acc_dtype), group_sz > 0, exact_sums=exact_sums)

    acc = jnp.zeros((block.m, block.n), dtype=acc_dtype)
    acc = lax.fori_loop(0, pl.cdiv(size.k, block.k), inner_compute, acc)
    stored(acc, ridx, lhs_rows_mask)

  # Dew: the epilogue on each fp32 accumulator tile and its side operands' tiles, its outputs stored.
  stored = partial(_store_epilogued, out_refs, epilogue, epilogue_refs, epilogue_axes, pid.j, block, size)
  lax.fori_loop(0, pl.cdiv(group_sz, block.m), outer_compute, None)
  _store_past_routed(stored, group_metadata_ref, pid, block, size, acc_dtype)


def _summed(acc, partial_sum, live, *, exact_sums: bool):
  """Dew: the running fp32 sum plus a step's: with `exact_sums`, an IEEE add of the step's product summed
  apart, which a tensor core's accumulation (truncating each add) leaves less exact; `live` (true for
  every computed step, unknown to the compiler) keeps the add from folding into the MMA's accumulator."""
  if exact_sums:
    return acc + jnp.where(live, partial_sum, 0)
  return acc + partial_sum


def _store_epilogued(out_refs, epilogue, refs, axes_of, column_block, block, size, acc, row, rows_mask):
  """Dew: each output of the epilogue on an fp32 accumulator tile at `row` and its side operands' tiles
  (the accumulator itself without one), stored at its rows and the program's columns."""
  cols_mask = (column_block * block.n + arange(block.n)) < size.n
  outs = (acc,)
  if epilogue is not None:
    tiles = [_side_tile(ref, axes, {'rows': (row, block.m), 'columns': (column_block * block.n, block.n)},
                        {'rows': rows_mask, 'columns': cols_mask})
             for ref, axes in zip(refs, axes_of, strict=True)]
    outs = epilogue(acc, *tiles)
    outs = outs if isinstance(outs, tuple) else (outs,)
  mask = rows_mask[:, None] & cols_mask[None, :]
  for y_ref, out in zip(out_refs, outs, strict=True):
    plgpu.store(y_ref.at[pl.ds(row, block.m), pl.ds(0, block.n)], out.astype(y_ref.dtype), mask=mask)


def _store_past_routed(stored, group_metadata_ref, pid, block, size, acc_dtype):
  """Dew: the rows past sum(group_sizes) sum nothing: by the last program along m, each output there is
  the epilogue of a zero accumulator, zero without one (`stored`)."""
  last_offset = (
    group_metadata_ref.group_offsets[size.g - 1]
    + group_metadata_ref.group_sizes[size.g - 1]
  )

  @pl.when(
    # Dew: upstream zeroes in the last program that computed rows, which
    # none is when every group is empty; the first program zeroes then.
    (pid.i == lax.max(group_metadata_ref.total_row_its[...], np.int32(1)) - 1)
    & (last_offset < size.m)
  )
  def _():
    def store_empty(i, _):
      row = last_offset + i * block.m
      stored(jnp.zeros((block.m, block.n), acc_dtype), row, (row + arange(block.m)) < size.m)

    lax.fori_loop(0, pl.cdiv(size.m - last_offset, block.m), store_empty, None)


@partial(jax.jit, static_argnames=(
  "block_m", "block_k", "block_n", "trans_rhs", "interpret", "compute_dtype", "acc_dtype", "num_warps",
  "num_stages", "chunk_m", "out_dtype", "prologue", "prologue_axes", "epilogue", "epilogue_axes",
  "out_dtypes", "exact_sums"))
def gmm(
  x: jax.Array,  # [m, k]
  A: jax.Array,  # [g, k, n]
  group_sizes: jax.Array,  # [g]
  block_m: int = DEFAULT_BLOCK_M,
  block_k: int = DEFAULT_BLOCK_K,
  block_n: int = DEFAULT_BLOCK_N,
  trans_rhs: bool = False,
  interpret: bool = False,
  compute_dtype: DTypeLike | None = None,
  acc_dtype: DTypeLike | None = np.float32,
  num_warps: int | None = None,
  num_stages: int | None = None,
  chunk_m: int = CHUNK_M,
  out_dtype: DTypeLike | None = None,
  *,
  prologue=None,
  prologue_operands: tuple[jax.Array, ...] = (),
  prologue_axes: tuple = (),
  epilogue=None,
  epilogue_operands: tuple[jax.Array, ...] = (),
  epilogue_axes: tuple = (),
  out_dtypes: tuple | None = None,
  exact_sums: bool = False,
) -> jax.Array | tuple[jax.Array, ...]:
  """Compute grouped matmul on GPU via a Pallas lowering.

  Dew: `prologue(tile, *side_tiles)` transforms each loaded `[block_m, block_k]` tile of `x` and
  `epilogue(acc, *side_tiles)` each fp32 accumulator tile, returning one output or a tuple of them
  (`out_dtypes`). Each side operand's dimensions follow the launch axes its `*_axes` entry names
  (`_side_tile`): 'rows' and 'depth' for the prologue's, 'rows' and 'columns' for the epilogue's.
  `exact_sums` adds each depth step's product into the fp32 sum by an IEEE add (`_summed`)."""

  msg = "This gmm kernel only supports either (m, k) x (g, k, n) -> (m, n) "
  msg += f"or (m, k) x (g, n, k) -> (m, n), but got {x.shape=} {A.shape=}"
  if not (A.ndim == 3 and x.ndim == 2):
    raise ValueError(msg)
  msg = f"Group sizes {group_sizes.shape=} must match first dimension of "
  msg += f"{A.shape=}"
  if not A.shape[:1] == group_sizes.shape:
    raise ValueError(msg)
  n = A.shape[-1] if not trans_rhs else A.shape[-2]
  Ak = A.shape[-2] if not trans_rhs else A.shape[-1]
  assert Ak == x.shape[1], msg
  size = RaggedDotSizes(m=x.shape[0], k=x.shape[1], n=n, g=A.shape[0])

  # normalize the block sizes for GPU
  block_m, block_k, block_n = (
    pl.next_power_of_2(min(b, s))
    for b, s in zip([block_m, block_k, block_n], [size.m, size.k, size.n], strict=True)
  )
  block_k, block_n = max(block_k, 16), max(block_n, 16)

  A_spec = pl.BlockSpec((size.g, size.k, block_n), lambda i, j: (0, 0, j))
  if trans_rhs:  # transposed spec
    A_spec = pl.BlockSpec((size.g, block_n, size.k), lambda i, j: (0, j, 0))

  group_metadata = _make_gmm_group_metadata(
    group_sizes=group_sizes, m=size.m, chunk_m=chunk_m
  )
  sides = (*prologue_operands, *epilogue_operands)
  in_specs = [
    pl.BlockSpec((size.m, size.k), lambda i, j: (0, 0)),
    A_spec,
    jax.tree.map(
      lambda x: pl.BlockSpec(x.shape, lambda *args: (0,) * x.ndim),
      group_metadata,
    ),
    # Dew: each side operand whole; the kernel loads its tiles at its own origins.
    *(pl.BlockSpec(side.shape, lambda i, j, ndim=side.ndim: (0,) * ndim) for side in sides),
  ]
  dtypes = out_dtypes or (out_dtype or x.dtype,)
  out_shape = tuple(jax.ShapeDtypeStruct((size.m, size.n), dtype=dtype) for dtype in dtypes)
  out_specs = tuple(pl.BlockSpec((size.m, block_n), lambda i, j: (0, j)) for _ in dtypes)
  grid_upper_bound = (size.m + chunk_m - 1) // chunk_m + size.g
  grid = (grid_upper_bound, pl.cdiv(size.n, block_n))
  block_sizes = BlockSizes(m=block_m, k=block_k, n=block_n)
  other_kws = {
    "compute_dtype": compute_dtype,
    "acc_dtype": acc_dtype,
    "trans_rhs": trans_rhs,
    "chunk_m": chunk_m,
    "prologue": prologue,
    "prologue_axes": prologue_axes,
    "epilogue": epilogue,
    "epilogue_axes": epilogue_axes,
    "exact_sums": exact_sums,
  }
  with jax.named_scope("pallas_triton_ragged_dot"):
    outs = pl.pallas_call(
      partial(
        _gpu_ragged_dot_kernel, size=size, block=block_sizes,
        **other_kws  # pyrefly: ignore[bad-argument-type]
      ),
      out_shape=out_shape,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      interpret=interpret,
      compiler_params=plgpu.CompilerParams(
        num_warps=num_warps, num_stages=num_stages
      ),
      name="pallas_triton_ragged_dot",
    )(x, A, group_metadata, *sides)
  return outs if out_dtypes else outs[0]


def _tgmm_ragged_dot_kernel(
  # inputs
  x_ref,  # [m, k]
  y_ref,  # [k, n]
  group_sizes_ref,  # [g]
  group_offset_ref,  # [g]
  # Dew: the prologue's side operands, then the output [g, k, n]
  *refs,
  # static problem shapes
  size: RaggedDotSizes,
  block: BlockSizes,  # hyperparameters
  compute_dtype: DTypeLike | None = None,
  acc_dtype: DTypeLike = np.float32,
  prologue=None,
  prologue_axes: tuple = (),
  exact_sums: bool = False,
):
  *prologue_refs, A_bar_ref = refs
  assert A_bar_ref.shape == (block.k, block.n)
  # Dew: a group's output tiles launch together, its column tiles fastest, so a wave reads one group's
  # rows from the L2 rather than each group's same tile (upstream put the group on the fastest axis).
  pid = SimpleNamespace(c=pl.program_id(0), r=pl.program_id(1), gi=pl.program_id(2))
  size = jax.tree.map(partial(np.array, dtype=np.int32), size)
  group_sz = group_sizes_ref[pid.gi]
  compute_dtype = compute_dtype if compute_dtype is not None else x_ref.dtype

  @pl.when(group_sz > 0)
  def _():
    # row index into lhs and output
    start_ridx = jnp.where(pid.gi == 0, 0, group_offset_ref[pid.gi])

    k_idx = pl.ds(0, block.k)
    k_mask = (pid.r * block.k + arange(block.k)) < size.k
    cols_idx = pl.ds(0, block.n)
    cols_mask = (pid.c * block.n + arange(block.n)) < size.n

    def inner_compute(r_offset, acc):
      # r_offset is 0,1,2,... need to map it to actual row indices
      ridx = start_ridx + r_offset * block.m
      xy_rows_mask = (r_offset * block.m + arange(block.m)) < group_sz
      xy_rows_idx = pl.ds(ridx, block.m)

      mask = xy_rows_mask[:, None] & k_mask[None, :]
      x = plgpu.load(x_ref.at[xy_rows_idx, k_idx], mask=mask, other=0)
      if prologue is not None:
        # Dew: the prologue, on the loaded rows of `x` and its side operands' tiles, then the rows
        # past the group zeroed again, which no hook may bring back.
        tiles = [_side_tile(ref, axes, {'rows': (ridx, block.m), 'depth': (pid.r * block.k, block.k)},
                            {'rows': xy_rows_mask, 'depth': k_mask})
                 for ref, axes in zip(prologue_refs, prologue_axes, strict=True)]
        x = jnp.where(mask, prologue(x, *tiles), 0)
      mask = xy_rows_mask[:, None] & cols_mask[None, :]
      y = plgpu.load(y_ref.at[xy_rows_idx, cols_idx], mask=mask, other=0)
      x, y = x.astype(compute_dtype), y.astype(compute_dtype)
      dim_nums = (((0,), (0,)), ((), ()))
      partial_sum = lax.dot_general(x, y, dimension_numbers=dim_nums, precision=lax.Precision.DEFAULT,
                                    preferred_element_type=acc_dtype)
      return _summed(acc, partial_sum.astype(acc.dtype), group_sz > 0, exact_sums=exact_sums)

    acc = jnp.zeros((block.k, block.n), dtype=acc_dtype)
    acc = lax.fori_loop(0, pl.cdiv(group_sz, block.m), inner_compute, acc)
    # Dew: upstream casts to y_ref's dtype, the second input's, which fails
    # to store whenever out_dtype differs from it (bf16 inputs, fp32 grad).
    acc = acc.astype(A_bar_ref.dtype)
    mask = k_mask[:, None] & cols_mask[None, :]
    plgpu.store(A_bar_ref.at[k_idx, cols_idx], acc, mask=mask)

  @pl.when(group_sz == 0)
  def _():
    rmask = (pid.r * block.k + arange(block.k)) < size.k
    cmask = (pid.c * block.n + arange(block.n)) < size.n
    plgpu.store(A_bar_ref, jnp.zeros_like(A_bar_ref),
                mask=rmask[:, None] & cmask[None, :])


@partial(jax.jit, static_argnames=(
  "block_m", "block_n", "block_k", "interpret", "compute_dtype", "acc_dtype", "num_warps", "num_stages",
  "chunk_m", "out_dtype", "prologue", "prologue_axes", "exact_sums"))
def tgmm(
  x: jax.Array,  # [m, k]
  y: jax.Array,  # [m, n]
  group_sizes: jax.Array,  # [g]
  block_m: int = DEFAULT_BLOCK_M,  # shape[0] of A_i tile (block_m, block_n)
  block_n: int = DEFAULT_BLOCK_N,  # shape[1] of A_i tile (block_m, block_n)
  block_k: int = DEFAULT_BLOCK_K,  # how many rows in the acc loop over block_m
  interpret: bool = False,
  compute_dtype: DTypeLike | None = None,
  acc_dtype: DTypeLike | None = np.float32,
  num_warps: int | None = None,
  num_stages: int | None = None,
  chunk_m: int = CHUNK_M,
  out_dtype: DTypeLike | None = None,
  *,
  prologue=None,
  prologue_operands: tuple[jax.Array, ...] = (),
  prologue_axes: tuple = (),
  exact_sums: bool = False,
) -> jax.Array:
  """Compute grouped matmul on GPU via a Pallas lowering.

  Dew: `prologue(rows, *side_tiles)` transforms each loaded `[block_m, block_k]` tile of `x`'s rows
  before its transposed product; side operands as `gmm`'s, along 'rows' and 'depth'. `exact_sums` as
  `gmm`'s."""
  del chunk_m
  msg = "This tgmm kernel only supports (m, k) x (m, n) -> (g, k, n), but got "
  msg += f"{x.shape=} {y.shape=}"
  assert y.ndim == 2 and x.ndim == 2 and x.shape[0] == y.shape[0], msg
  (m, k), n = x.shape, y.shape[-1]
  size = RaggedDotSizes(m=m, k=k, n=n, g=group_sizes.size)

  block_m, block_n = min(block_m, m), min(block_n, n)

  # normalize the block sizes for GPU
  block_m, block_k, block_n = (
    max(pl.next_power_of_2(min(b, s)), 16)
    for b, s in zip([block_m, block_k, block_n], [size.m, size.k, size.n], strict=True)
  )

  group_offsets = jnp.cumsum(group_sizes) - group_sizes
  in_specs = [
    pl.BlockSpec((size.m, block_k), lambda c, r, i: (0, r)),
    pl.BlockSpec((size.m, block_n), lambda c, r, i: (0, c)),
    pl.BlockSpec((size.g,), lambda c, r, i: (0,)),
    pl.BlockSpec((size.g,), lambda c, r, i: (0,)),
    *(pl.BlockSpec(side.shape, lambda c, r, i, ndim=side.ndim: (0,) * ndim) for side in prologue_operands),
  ]

  out_shape = jax.ShapeDtypeStruct(
    (size.g, size.k, size.n), dtype=out_dtype or x.dtype)
  out_specs = pl.BlockSpec((None, block_k, block_n), lambda c, r, i: (i, r, c))
  grid = (pl.cdiv(size.n, block_n), pl.cdiv(size.k, block_k), size.g)

  block_sizes = BlockSizes(m=block_m, k=block_k, n=block_n)
  dtype_spec = {"compute_dtype": compute_dtype, "acc_dtype": acc_dtype, "prologue": prologue,
                "prologue_axes": prologue_axes, "exact_sums": exact_sums}
  with jax.named_scope("tgmm_ragged_dot"):
    return pl.pallas_call(
      partial(_tgmm_ragged_dot_kernel, size=size, block=block_sizes,
              **dtype_spec),  # pyrefly: ignore[bad-argument-type]
      out_shape=out_shape,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      interpret=interpret,
      compiler_params=plgpu.CompilerParams(
        num_warps=num_warps, num_stages=num_stages
      ),
      name="tgmm_ragged_dot",
    )(x, y, group_sizes, group_offsets, *prologue_operands)


def _hyperparam_selection_rule(dtype: np.dtype):
  smem_size = 100 * 1024  # 100 KiB
  ideal_operand_size = smem_size / dtype.itemsize / 4
  tile_m = 2 ** round(math.ceil(math.log2(math.sqrt(ideal_operand_size))))
  tile_k, tile_n = tile_m // 2, tile_m
  while tile_m > 16:
    if tile_m * tile_k * 4 * dtype.itemsize <= smem_size:
      return {"block_m": tile_m, "block_k": tile_k, "block_n": tile_n}
    tile_m //= 2
    tile_k, tile_n = tile_m // 2, tile_m
  return {"block_m": tile_m, "block_k": tile_k, "block_n": tile_n}

