"""The GEMM template's TPU mainloops, on Megablox's schedule (`jax.experimental.pallas.ops.tpu.megablox.gmm`,
jax-v0.11.2): `gmm`, `[rows, depth] x [groups, depth, width]` over rows sorted by group (with
`transpose_rhs`, `[groups, width, depth]`), and `tgmm`, each group's rows transposed times their cotangent
rows, `[groups, depth, width]`.

Megablox's group metadata (`make_group_metadata`) lists a grid slot for each row block of each group: a
block two groups share runs once for each, consecutively, so an output block stays in VMEM across its
visits and each visit stores its group's rows alone. Each grid slot's blocks are DMA'd into VMEM by
Pallas's pipeline, double-buffered, and multiplied by the MXUs, which take bf16 operands and accumulate
in fp32. Dew's, beside upstream's:

- the template's hooks: `prologue(tile, *side_tiles)` on each `lhs` block before its product, rounded
  to `lhs`'s dtype, and `epilogue(acc, *side_tiles)` on each fp32 accumulator block before its one
  rounding, returning one output or a tuple of them (`out_dtypes`). A side operand's dimensions follow
  the launch axes its `*_axes` entry names ('rows' and 'depth' for the prologue's, 'rows' and 'columns'
  for the epilogue's), its blocks DMA'd beside the operands';
- each writes the output dtype it is asked for (fp32 for a kernel gradient), not the operands';
- the rows are padded to a multiple of the row block, and the rows past the routed ones, the padding
  among them, are one more group, whose blocks multiply nothing: each output there is the epilogue of
  a zero accumulator (zero without one), where upstream required the groups to cover every row.
"""

import dataclasses

import jax
from jax import lax, numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.megablox.gmm import make_group_metadata

from dew.nn.kernels.templates.gemm import cost, rows_padded
from dew.nn.sharding import per_shard

LANES = 128
"""A vreg's lanes: Mosaic TPU tiles a block's minor dimensions by them (a block of two dimensions or more
is of whole multiples), and a v5e MXU's side."""


def _exact(dtype) -> lax.Precision:
    """The precision whose MXU product of `dtype` operands sums them exactly into fp32: a 16-bit operand's
    one pass (Mosaic TPU refuses an fp32 contraction of bf16 operands, which a process's default precision
    of 'highest' would ask), a wider one's six."""
    return lax.Precision.DEFAULT if jnp.dtype(dtype).itemsize == 2 else lax.Precision.HIGHEST


@dataclasses.dataclass(frozen=True)
class Blocks:
    """One launch's output block (`block_m` x `block_n`) and contraction step (`block_k`); a kernel
    gradient's `block_m` and `block_n` tile its depth and width and `block_k` its rows, as
    `mosaic_gpu.Tiles`'s do."""
    block_m: int
    block_n: int
    block_k: int


def _compiler_params() -> pltpu.CompilerParams:
    # Upstream's: the column blocks are independent, the row blocks and depth steps revisit an output
    # block, so they run in order.
    return pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary"))


def gmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, blocks: Blocks, transpose_rhs: bool,
        out_dtype, interpret: bool | pltpu.InterpretParams = False, prologue=None,
        prologue_operands: tuple = (), prologue_axes: tuple = (), epilogue=None,
        epilogue_operands: tuple = (), epilogue_axes: tuple = (),
        out_dtypes: tuple | None = None) -> jax.Array | tuple[jax.Array, ...]:
    """Each group's rows of `lhs` `[rows, depth]` times its matrix of `rhs`, `[groups, depth, width]` or
    with `transpose_rhs` `[groups, width, depth]`: `[rows, width]` in `out_dtype` (or a tuple in
    `out_dtypes`), each row past the routed ones the epilogue of a zero accumulator. The depth must divide
    by `blocks.block_k` and the width by `blocks.block_n`."""
    block_m, block_n, block_k = blocks.block_m, blocks.block_n, blocks.block_k
    rows, depth = lhs.shape
    groups = rhs.shape[0]
    rhs_depth, width = (rhs.shape[2], rhs.shape[1]) if transpose_rhs else rhs.shape[1:]
    if depth != rhs_depth or depth % block_k or width % block_n:
        raise ValueError(f"a [{rows}, {depth}] x {list(rhs.shape)} product does not take {blocks}")
    lhs = rows_padded(lhs, 0, block_m)
    m = lhs.shape[0]
    sizes = group_sizes.astype(jnp.int32)
    lengths = jnp.concatenate([sizes, jnp.maximum(m - jnp.sum(sizes), 0)[None]])
    metadata, active = make_group_metadata(group_sizes=lengths, m=m, tm=block_m, start_group=jnp.int32(0),
                                           num_nonzero_groups=groups + 1, visit_empty_groups=False)
    prologue_operands = tuple(_side_padded(side, axes, block_m)
                              for side, axes in zip(prologue_operands, prologue_axes, strict=True))
    epilogue_operands = tuple(_side_padded(side, axes, block_m)
                              for side, axes in zip(epilogue_operands, epilogue_axes, strict=True))
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    steps = depth // block_k
    sides = len(prologue_operands) + len(epilogue_operands)

    def kernel(group_offsets, group_ids, m_tile_ids, lhs_ref, rhs_ref, *refs):
        prologue_refs, epilogue_refs = refs[:len(prologue_operands)], refs[len(prologue_operands):sides]
        out_refs, acc_ref = refs[sides:-1], refs[-1]
        slot, step = pl.program_id(1), pl.program_id(2)
        group = group_ids[slot]

        @pl.when(step == 0)
        def _():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        # Dew: the unrouted rows' group multiplies nothing; its accumulator stays zero.
        @pl.when(group < groups)
        def _():
            tile = lhs_ref[...]
            if prologue is not None:
                tile = prologue(tile, *(side[...] for side in prologue_refs)).astype(lhs_ref.dtype)
            acc_ref[...] += lax.dot_general(tile, rhs_ref[...], (((1,), (int(transpose_rhs),)), ((), ())),
                                            precision=_exact(tile.dtype), preferred_element_type=jnp.float32)

        @pl.when(step == steps - 1)
        def _():
            acc = acc_ref[...]
            outs = (acc,) if epilogue is None else _as_tuple(
                epilogue(acc, *(side[...] for side in epilogue_refs)))
            start, end = group_offsets[group], group_offsets[group + 1]
            row = m_tile_ids[slot] * block_m + lax.broadcasted_iota(jnp.int32, (block_m, block_n), 0)
            mine = (row >= start) & (row < end)
            for out_ref, out in zip(out_refs, outs, strict=True):
                out_ref[...] = jnp.where(mine, out.astype(out_ref.dtype), out_ref[...])

    def lhs_index(column, slot, step, group_offsets, group_ids, m_tile_ids):
        return m_tile_ids[slot], step

    def rhs_index(column, slot, step, group_offsets, group_ids, m_tile_ids):
        group = jnp.minimum(group_ids[slot], groups - 1)
        return (group, column, step) if transpose_rhs else (group, step, column)

    def side_spec(axes):
        extent = {'rows': block_m, 'depth': block_k, 'columns': block_n}

        def index(column, slot, step, group_offsets, group_ids, m_tile_ids):
            origin = {'rows': m_tile_ids[slot], 'depth': step, 'columns': column}
            return tuple(origin[axis] for axis in axes)

        return pl.BlockSpec(tuple(extent[axis] for axis in axes), index)

    def out_index(column, slot, step, group_offsets, group_ids, m_tile_ids):
        return m_tile_ids[slot], column

    out_shape = tuple(jax.ShapeDtypeStruct((m, width), dtype) for dtype in dtypes)
    per_shard("dew_gmm")
    outs = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[pl.BlockSpec((block_m, block_k), lhs_index),
                      pl.BlockSpec((None, block_n, block_k) if transpose_rhs else (None, block_k, block_n),
                                   rhs_index),
                      *(side_spec(axes) for axes in (*prologue_axes, *epilogue_axes))],
            out_specs=tuple(pl.BlockSpec((block_m, block_n), out_index) for _ in dtypes),
            grid=(width // block_n, active, steps),
            scratch_shapes=[pltpu.VMEM((block_m, block_n), jnp.float32)]),
        compiler_params=_compiler_params(),
        interpret=interpret,
        cost_estimate=cost(2 * rows * depth * width, lhs, rhs, *prologue_operands, *epilogue_operands,
                           *out_shape),
        name="dew_gmm",
    )(*metadata, lhs, rhs, *prologue_operands, *epilogue_operands)
    outs = tuple(out[:rows] for out in outs)
    return outs if out_dtypes else outs[0]


def tgmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, blocks: Blocks, out_dtype,
         interpret: bool | pltpu.InterpretParams = False, prologue=None, prologue_operands: tuple = (),
         prologue_axes: tuple = ()) -> jax.Array:
    """Each group's rows of `prologue(lhs)` (`[rows, depth]`) transposed times its rows of `rhs`
    (`[rows, width]`): `[groups, depth, width]` in `out_dtype`, an empty group's zero. The depth must
    divide by `blocks.block_m` and the width by `blocks.block_n`; `blocks.block_k` is the rows a step
    takes. The prologue sees each step's `[block_k, block_m]` rows before the group's are selected, so a
    neighbour's rows reach no sum, whatever the prologue makes of them."""
    block_m, block_n, block_k = blocks.block_m, blocks.block_n, blocks.block_k
    rows, depth = lhs.shape
    width = rhs.shape[1]
    groups = group_sizes.shape[0]
    if rhs.shape[0] != rows or depth % block_m or width % block_n:
        raise ValueError(f"a [{rows}, {depth}] by {list(rhs.shape)} kernel gradient does not take {blocks}")
    lhs, rhs = rows_padded(lhs, 0, block_k), rows_padded(rhs, 0, block_k)
    m = lhs.shape[0]
    metadata, active = make_group_metadata(group_sizes=group_sizes.astype(jnp.int32), m=m, tm=block_k,
                                           start_group=jnp.int32(0), num_nonzero_groups=groups,
                                           visit_empty_groups=True)
    prologue_operands = tuple(_side_padded(side, axes, block_k)
                              for side, axes in zip(prologue_operands, prologue_axes, strict=True))

    def kernel(group_offsets, group_ids, m_tile_ids, lhs_ref, rhs_ref, *refs):
        *side_refs, out_ref, acc_ref = refs
        slot = pl.program_id(2)
        group = group_ids[slot]
        first = (slot == 0) | (group_ids[jnp.maximum(slot - 1, 0)] != group)
        final = pl.num_programs(2) - 1
        last = (slot == final) | (group_ids[jnp.minimum(slot + 1, final)] != group)
        start, end = group_offsets[group], group_offsets[group + 1]

        @pl.when(first)
        def _():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        @pl.when(end > start)
        def _():
            tile = lhs_ref[...]
            if prologue is not None:
                tile = prologue(tile, *(side[...] for side in side_refs)).astype(lhs_ref.dtype)
            row = m_tile_ids[slot] * block_k + lax.broadcasted_iota(jnp.int32, (block_k, 1), 0)
            mine = (row >= start) & (row < end)
            # Upstream's selection, in fp32: a neighbour's rows, a NaN or an infinity among them, are zeros.
            tile = jnp.where(mine, tile.astype(jnp.float32), 0.0).astype(lhs_ref.dtype)
            cotangent = jnp.where(mine, rhs_ref[...].astype(jnp.float32), 0.0).astype(rhs_ref.dtype)
            acc_ref[...] += lax.dot(tile.swapaxes(0, 1), cotangent, precision=_exact(tile.dtype),
                                    preferred_element_type=jnp.float32)

        @pl.when(last)
        def _():
            out_ref[...] = acc_ref[...].astype(out_ref.dtype)

    def rows_index(column, block, slot, group_offsets, group_ids, m_tile_ids):
        return m_tile_ids[slot], block

    def cotangent_index(column, block, slot, group_offsets, group_ids, m_tile_ids):
        return m_tile_ids[slot], column

    def side_spec(axes):
        extent = {'rows': block_k, 'depth': block_m}

        def index(column, block, slot, group_offsets, group_ids, m_tile_ids):
            origin = {'rows': m_tile_ids[slot], 'depth': block}
            return tuple(origin[axis] for axis in axes)

        return pl.BlockSpec(tuple(extent[axis] for axis in axes), index)

    def out_index(column, block, slot, group_offsets, group_ids, m_tile_ids):
        return group_ids[slot], block, column

    out_shape = jax.ShapeDtypeStruct((groups, depth, width), jnp.dtype(out_dtype))
    per_shard("dew_tgmm")
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[pl.BlockSpec((block_k, block_m), rows_index),
                      pl.BlockSpec((block_k, block_n), cotangent_index),
                      *(side_spec(axes) for axes in prologue_axes)],
            out_specs=pl.BlockSpec((None, block_m, block_n), out_index),
            grid=(width // block_n, depth // block_m, active),
            scratch_shapes=[pltpu.VMEM((block_m, block_n), jnp.float32)]),
        compiler_params=_compiler_params(),
        interpret=interpret,
        cost_estimate=cost(2 * rows * depth * width, lhs, rhs, *prologue_operands, out_shape),
        name="dew_tgmm",
    )(*metadata, lhs, rhs, *prologue_operands)


def _as_tuple(outs) -> tuple:
    return outs if isinstance(outs, tuple) else (outs,)


def _side_padded(side: jax.Array, axes: tuple, block: int) -> jax.Array:
    """A side operand padded as the rows are along the dimension that follows them, wherever it is."""
    return rows_padded(side, axes.index('rows'), block) if 'rows' in axes else side
