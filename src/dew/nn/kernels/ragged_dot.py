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
`_hyperparam_selection_rule`, restyled to this tree's lint gate, with two
lines of logic changed and marked `# Dew:`: tgmm's output cast, and gmm's
zeroing when every group is empty. The file's `ragged_dot_general`
adapter is left out: Dew calls the kernels itself, through
`dew.nn.moe.expert_projection`'s custom VJP, the way MaxText calls megablox.

Every global load and store is masked on every dimension, rows included, and
rows past the routed ones are written as zeros. The kernels multiply in
`compute_dtype`, accumulate in `acc_dtype` and ignore any precision: with
16-bit operands that is exact products summed in fp32, with fp32 operands
the GPU's TF32 rate.
"""

import math
import typing
from functools import partial
from types import SimpleNamespace

import numpy as np
from jax._src import api, core, tree_util
from jax._src.lax import lax
from jax._src.lax.control_flow import loops
from jax._src.numpy import array_creation, lax_numpy as jnp, reductions
from jax._src.typing import Array, DTypeLike
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

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

def cdiv(a, b):
  return (a + b - 1) // b


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

  group_sizes: Array  # the regular group sizes
  group_offsets: Array  # the group offsets
  row_it2group: Array  # mapping between the first grid dim and group id
  row_it2it_in_group: Array  # mapping between the first grid dim and which
                                 # chunk in the group we're computing
  total_row_its: Array       # first grid dim upper bound / number of chunks


def _make_gmm_group_metadata(
  group_sizes: Array, m: int, chunk_m: int
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
  _blocks_offset = reductions.cumsum(blocks_per_group) - blocks_per_group
  _blocks_upper_bound = (m + chunk_m - 1) // chunk_m + g
  _group_iota = arange(_blocks_upper_bound)[None, :] - _blocks_offset[:, None]
  _group_write_mask = (_group_iota >= 0) & (
    _group_iota < blocks_per_group[:, None]
  )

  row_it2group = reductions.sum(
      jnp.where(_group_write_mask, arange(g)[:, None], 0), 0, dtype=np.int32
  )  # block row i to group id
  row_it2it_in_group = reductions.sum(
      jnp.where(_group_write_mask, _group_iota, 0), 0, dtype=np.int32
  )  # block row i to within group chunk

  return GMMGroupLookupMetadata(
      group_sizes=group_sizes,
      group_offsets=reductions.cumsum(group_sizes) - group_sizes,
      row_it2group=row_it2group,
      row_it2it_in_group=row_it2it_in_group,
      total_row_its=reductions.sum(blocks_per_group, dtype=np.int32),
  )


def _gpu_ragged_dot_kernel(
  # inputs
  x_ref,  # [m, k]
  A_ref,  # [k, n] or [n, k]
  group_metadata_ref: GMMGroupLookupMetadata,
  # outputs
  y_ref,  # [k, n]
  # static problem shapes
  chunk_m: int,
  trans_rhs: bool,
  size: RaggedDotSizes,
  block: BlockSizes,  # hyperparameters
  compute_dtype: DTypeLike | None = None,
  acc_dtype: DTypeLike = np.float32,
):
  i, j = pl.program_id(0), pl.program_id(1)
  pid = SimpleNamespace(i=i, j=j, gi=group_metadata_ref.row_it2group[i])
  size = tree_util.tree_map(partial(np.array, dtype=np.int32), size)

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
      xA = lax.dot_general(
          x, A, dimension_numbers=dim_nums, preferred_element_type=acc_dtype
      )
      return acc + xA.astype(acc_dtype)

    acc = array_creation.zeros((block.m, block.n), dtype=acc_dtype)
    acc = loops.fori_loop(0, cdiv(size.k, block.k), inner_compute, acc)
    acc = acc.astype(y_ref.dtype)
    mask = lhs_rows_mask[:, None] & rhs_cols_mask[None, :]
    plgpu.store(y_ref.at[lhs_rows_idx, rhs_cols_idx], acc, mask=mask)
    return

  loops.fori_loop(0, cdiv(group_sz, block.m), outer_compute, None)

  # zero out memory past sum(group_sizes) if we're the last kernel along m
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
    col_mask = (block.n * pid.j + arange(block.n)) < size.n

    def set_zero(i, _):
      row_mask = (last_offset + i * block.m + arange(block.m)) < size.m
      window = (pl.ds(last_offset + i * block.m, block.m), pl.ds(0, block.n))
      mask = row_mask[:, None] & col_mask[None, :]
      zero = array_creation.zeros((block.m, block.n), dtype=y_ref.dtype)
      plgpu.store(y_ref.at[*window], zero, mask=mask)

    loops.fori_loop(0, cdiv(size.m - last_offset, block.m), set_zero, None)


@api.jit(static_argnums=list(range(3, 14)))
def gmm(
  x: Array,  # [m, k]
  A: Array,  # [g, k, n]
  group_sizes: Array,  # [g]
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
) -> Array:
  """Compute grouped matmul on GPU via a Pallas lowering."""

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
  in_specs = [
    pl.BlockSpec((size.m, size.k), lambda i, j: (0, 0)),
    A_spec,
    tree_util.tree_map(
      lambda x: pl.BlockSpec(x.shape, lambda *args: (0,) * x.ndim),
      group_metadata,
    ),
  ]
  out_shape = core.ShapeDtypeStruct(
    (size.m, size.n), dtype=out_dtype or x.dtype)
  out_specs = pl.BlockSpec((size.m, block_n), lambda i, j: (0, j))
  grid_upper_bound = (size.m + chunk_m - 1) // chunk_m + size.g
  grid = (grid_upper_bound, pl.cdiv(size.n, block_n))
  block_sizes = BlockSizes(m=block_m, k=block_k, n=block_n)
  other_kws = {
    "compute_dtype": compute_dtype,
    "acc_dtype": acc_dtype,
    "trans_rhs": trans_rhs,
    "chunk_m": chunk_m,
  }
  with api.named_scope("pallas_triton_ragged_dot"):
    return pl.pallas_call(
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
    )(x, A, group_metadata)


def _tgmm_ragged_dot_kernel(
  # inputs
  x_ref,  # [m, k]
  y_ref,  # [k, n]
  group_sizes_ref,  # [g]
  group_offset_ref,  # [g]
  # outputs
  A_bar_ref,  # [g, k, n]
  # static problem shapes
  size: RaggedDotSizes,
  block: BlockSizes,  # hyperparameters
  compute_dtype: DTypeLike | None = None,
  acc_dtype: DTypeLike = np.float32,
):
  assert A_bar_ref.shape == (block.k, block.n)
  pid =  SimpleNamespace(gi=pl.program_id(0), r=pl.program_id(1),
                         c=pl.program_id(2))
  size = tree_util.tree_map(partial(np.array, dtype=np.int32), size)
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
      mask = xy_rows_mask[:, None] & cols_mask[None, :]
      y = plgpu.load(y_ref.at[xy_rows_idx, cols_idx], mask=mask, other=0)
      x, y = x.astype(compute_dtype), y.astype(compute_dtype)
      dim_nums = (((0,), (0,)), ((), ()))
      return acc + lax.dot_general(
          x, y, dimension_numbers=dim_nums, preferred_element_type=acc_dtype
      ).astype(acc.dtype)

    acc = array_creation.zeros((block.k, block.n), dtype=acc_dtype)
    acc = loops.fori_loop(0, cdiv(group_sz, block.m), inner_compute, acc)
    # Dew: upstream casts to y_ref's dtype, the second input's, which fails
    # to store whenever out_dtype differs from it (bf16 inputs, fp32 grad).
    acc = acc.astype(A_bar_ref.dtype)
    mask = k_mask[:, None] & cols_mask[None, :]
    plgpu.store(A_bar_ref.at[k_idx, cols_idx], acc, mask=mask)

  @pl.when(group_sz == 0)
  def _():
    rmask = (pid.r * block.k + arange(block.k)) < size.k
    cmask = (pid.c * block.n + arange(block.n)) < size.n
    plgpu.store(A_bar_ref, array_creation.zeros_like(A_bar_ref),
                mask=rmask[:, None] & cmask[None, :])


@api.jit(static_argnums=list(range(3, 13)))
def tgmm(
  x: Array,  # [m, k]
  y: Array,  # [m, n]
  group_sizes: Array,  # [g]
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
) -> Array:
  """Compute grouped matmul on GPU via a Pallas lowering."""
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

  group_offsets = reductions.cumsum(group_sizes) - group_sizes
  in_specs = [
    pl.BlockSpec((size.m, block_k), lambda i, r, c: (0, r)),
    pl.BlockSpec((size.m, block_n), lambda i, r, c: (0, c)),
    pl.BlockSpec((size.g,), lambda i, r, c: (0,)),
    pl.BlockSpec((size.g,), lambda i, r, c: (0,)),
  ]

  out_shape = core.ShapeDtypeStruct(
    (size.g, size.k, size.n), dtype=out_dtype or x.dtype)
  out_specs = pl.BlockSpec((None, block_k, block_n), lambda i, r, c: (i, r, c))
  grid = (size.g, pl.cdiv(size.k, block_k), pl.cdiv(size.n, block_n))

  block_sizes = BlockSizes(m=block_m, k=block_k, n=block_n)
  dtype_spec = {"compute_dtype": compute_dtype, "acc_dtype": acc_dtype}
  with api.named_scope("tgmm_ragged_dot"):
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
    )(x, y, group_sizes, group_offsets)


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


def block_sizes(dtype) -> dict[str, int]:
  """The tile the vendored rule picks for operands of `dtype`."""
  return _hyperparam_selection_rule(np.dtype(dtype))
