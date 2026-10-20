# Copyright 2025 The JAX Authors.
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

"""Grouped matmuls for Hopper in Pallas Mosaic GPU: `gmm`, `[rows, depth] x [groups, depth, width]` over
rows sorted by group (with `transpose_rhs`, `[groups, width, depth]`), and `tgmm`, each group's rows
transposed times their cotangent rows, `[groups, depth, width]`.

Derived from the jax-v0.11.2 source tree (tag jax-v0.11.2, 32544801; license header above), which ships
these as examples: `pallas/ops/gpu/hopper_matmul_mgpu.py`'s kernel (one memory warpgroup feeding two
compute warpgroups through `emit_pipeline_warp_specialized`, persistent over the multiprocessors, a
planar snake over the output tiles) with the grouping of `ragged_dot_mgpu.py` and
`transposed_ragged_dot_mgpu.py`. Changed for Dew:

- A group's tiles start at its first row, so only its last tile (or contraction step) is partial and
  its valid rows are a prefix. Which group and rows each tile holds is computed in XLA before the
  launch (`tile_map`, as MaxText's megablox computes its group metadata), where upstream searched
  every group from each tile.
- Rows past the routed ones form one more group, whose tiles store zeros, so the output's unrouted
  rows are zero.
- `gmm` stores a partial tile's valid rows in a power-of-two ladder of TMA stores through untiled
  64 x 64 epilogue slots.
- `tgmm` zeroes the rows past its group in both operands (a select, so a non-finite neighbour stays
  out): each compute warpgroup in its own half of the cotangent tile, and warpgroup 0 in the shared
  row tile, which warpgroup 1 waits for on a barrier. It writes any output dtype, an fp32 gradient
  above all.
- The program count is the device's multiprocessors, where upstream fixed an H100 SXM's 132.

Each sums exact products of its 16-bit operands in fp32 (wgmma's accumulator) and rounds once to the
output dtype.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

EPILOGUE = 64
"""The rows and columns of an epilogue slot: a tile's accumulator is stored 64 x 64 at a time."""


@dataclasses.dataclass(frozen=True)
class Tiles:
    """One launch's tiles: each compute warpgroup's accumulator (`tile_m` x `tile_n`), the contraction
    step (`tile_k`), the steps in flight (`stages`), the output dimension the two compute warpgroups
    split (`split`, 'm' or 'n'), and the output tiles in each band of the planar snake (`width`)."""
    tile_m: int
    tile_n: int
    tile_k: int
    stages: int
    split: str
    width: int

    @property
    def cta_m(self) -> int:
        """The rows of one program's output tile."""
        return self.tile_m * (2 if self.split == 'm' else 1)

    @property
    def cta_n(self) -> int:
        """The columns of one program's output tile."""
        return self.tile_n * (2 if self.split == 'n' else 1)


def capacity(rows: int, groups: int, cta_m: int) -> int:
    """The most tiles `tile_map` lays over `rows` rows in `groups` groups and the unrouted rest: each
    of the `groups + 1` parts adds at most one partial tile to `ceil(rows / cta_m)`."""
    return pl.cdiv(rows, cta_m) + groups


def tile_map(group_sizes: jax.Array, rows: int, cta_m: int):
    """Each of `capacity` tiles' group, first row and valid rows, and how many tiles are live.

    A group's tiles cover its rows from its first, `cta_m` at a time; the rows past the routed ones
    are group `groups`, whose tiles store zeros. Slots past the live tiles name group `groups` and
    no rows."""
    groups = group_sizes.shape[0]
    sizes = group_sizes.astype(jnp.int32)
    routed = jnp.sum(sizes)
    parts = jnp.concatenate([sizes, jnp.maximum(rows - routed, 0)[None]])
    starts = jnp.cumsum(parts) - parts
    counts = (parts + cta_m - 1) // cta_m
    ends = jnp.cumsum(counts)
    slots = jnp.arange(capacity(rows, groups, cta_m), dtype=jnp.int32)
    part = jnp.minimum(jnp.searchsorted(ends, slots, side='right'), groups).astype(jnp.int32)
    index = slots - (ends[part] - counts[part])
    first = starts[part] + index * cta_m
    valid = jnp.clip(parts[part] - index * cta_m, 0, cta_m)
    return part, first, valid, ends[-1:]


def _transforms(dtype, columns: int):
    """hopper_matmul_mgpu's tiling and swizzle for a block `columns` wide."""
    swizzle = plgpu.find_swizzle(columns * jnp.dtype(dtype).itemsize * 8)
    return (plgpu.TilingTransform((8, swizzle // jnp.dtype(dtype).itemsize)), plgpu.SwizzleTransform(swizzle))


def _ignore(*_, **__) -> None:
    """A pipeline body or compute context that does nothing: the allocations a pipeline takes do not
    depend on them (hopper_matmul_mgpu's `ignore`)."""


def _compiler_params():
    return plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup)


def gmm(tokens: jax.Array, kernel: jax.Array, group_sizes: jax.Array, *, tiles: Tiles, transpose_rhs: bool,
        out_dtype, multiprocessors: int, interpret: bool = False) -> jax.Array:
    """Each group's rows of `tokens` times its matrix of `kernel`, or of its transpose with
    `transpose_rhs`, `[rows, width]` in `out_dtype`; unrouted rows are zero. The depth must divide by
    `tiles.tile_k` and the width by `tiles.cta_n`."""
    rows, depth = tokens.shape
    groups = kernel.shape[0]
    width = kernel.shape[1] if transpose_rhs else kernel.shape[2]
    out_dtype = jnp.dtype(out_dtype)
    t = tiles
    k_steps, n_tiles = depth // t.tile_k, width // t.cta_n
    transforms = _transforms(tokens.dtype, t.tile_k)
    part, first, valid, live = tile_map(group_sizes, rows, t.cta_m)
    epi_m, epi_n = min(EPILOGUE, t.tile_m), min(EPILOGUE, t.tile_n)

    def pipeline(step, compute_context):
        rhs = (plgpu.BlockSpec((t.cta_n, t.tile_k), lambda k: (0, k), transforms=transforms)
               if transpose_rhs else
               plgpu.BlockSpec((t.tile_k, t.cta_n), lambda k: (k, 0), transforms=transforms))
        return plgpu.emit_pipeline_warp_specialized(
            step, grid=(k_steps,), memory_registers=40,
            in_specs=[plgpu.BlockSpec((t.cta_m, t.tile_k), lambda k: (0, k), transforms=transforms), rhs],
            wg_axis="wg", num_compute_wgs=2, max_concurrent_steps=t.stages,
            compute_context=compute_context)

    def body(part_ref, first_ref, valid_ref, live_ref, x_ref, w_ref, out_ref):
        @functools.partial(pl.run_scoped,
                           allocations=pipeline(_ignore, _ignore).get_allocations(x_ref, w_ref.at[0]),
                           slots=plgpu.SMEM((2, 2, epi_m, epi_n), out_dtype),
                           collective_axes="wg")
        def scope(allocations, slots):
            wg = lax.axis_index("wg")
            wg_rows = wg * t.tile_m if t.split == 'm' else 0
            wg_columns = wg * t.tile_n if t.split == 'n' else 0

            @plgpu.nd_loop((live_ref[0] * n_tiles,), collective_axes="sm")
            def _(info):
                (index,) = info.index
                tile, column = plgpu.planar_snake(index, (live_ref[0], n_tiles), 1, t.width)
                group, start, count = part_ref[tile], first_ref[tile], valid_ref[tile]
                # This warpgroup's rows of the tile that are the group's: a prefix of its own.
                own = jnp.clip(count - wg_rows, 0, t.tile_m)
                out = out_ref.at[pl.ds(start + wg_rows, t.tile_m),
                                 pl.ds(column * t.cta_n + wg_columns, t.tile_n)]

                def store(value):
                    _ladder(value, own, out, slots.at[wg], epi_m, epi_n)

                def compute_context(run):
                    @functools.partial(pl.run_scoped, acc=plgpu.ACC((t.tile_m, t.tile_n), jnp.float32))
                    def _(acc):
                        run(acc)
                        store(acc[...].astype(out_dtype))

                def step(_, x_smem, w_smem, acc):
                    lhs = x_smem.at[pl.ds(wg_rows, t.tile_m)] if t.split == 'm' else x_smem
                    if transpose_rhs:
                        rhs = (w_smem.at[pl.ds(wg_columns, t.tile_n)] if t.split == 'n' else w_smem).T
                    else:
                        rhs = w_smem.at[:, pl.ds(wg_columns, t.tile_n)] if t.split == 'n' else w_smem
                    plgpu.wgmma(acc, lhs, rhs)
                    plgpu.wgmma_wait(0)
                    return acc

                @pl.when(group < groups)
                def _():
                    columns = pl.ds(column * t.cta_n, t.cta_n)
                    pipeline(step, compute_context)(
                        x_ref.at[pl.ds(start, t.cta_m)],
                        w_ref.at[group].at[columns] if transpose_rhs else w_ref.at[group].at[:, columns],
                        allocations=allocations)

                # The unrouted rows: the compute warpgroups store zeros, the memory warpgroup idles.
                @pl.when((group == groups) & (wg < 2))
                def _():
                    store(_zeros((t.tile_m, t.tile_n), out_dtype))

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    launch = plgpu.kernel(
        body, out_type=jax.ShapeDtypeStruct((rows, width), out_dtype), grid=(multiprocessors,),
        grid_names=("sm",), num_threads=3, thread_name="wg", compiler_params=_compiler_params(),
        interpret=interpret)
    return launch(part, first, valid, live, tokens, kernel)


def _ladder(value, rows, out, slots, epi_m: int, epi_n: int) -> None:
    """Store the first `rows` rows of `value`, one warpgroup's tile, to `out` at its own place: 64 x 64
    at a time through `slots` (two, double-buffered), each slot's valid rows in TMA stores of 64, 32, ...,
    1 rows, which every count below 128 sums to. A slot is written again only once its last store has
    read it."""
    tile_m, tile_n = value.shape
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
    for epi_row in range(tile_m // epi_m):
        for epi_column in range(tile_n // epi_n):
            slot = (epi_row * (tile_n // epi_n) + epi_column) % 2
            plgpu.wait_smem_to_gmem(1, wait_read_only=True)
            slots[slot] = value[epi_row * epi_m:(epi_row + 1) * epi_m,
                                epi_column * epi_n:(epi_column + 1) * epi_n]
            plgpu.commit_smem()
            stored = jnp.clip(rows - epi_row * epi_m, 0, epi_m)
            offset = jnp.int32(0)
            size = epi_m
            while size:
                plgpu.copy_smem_to_gmem(
                    slots.at[slot].at[pl.ds(offset, size)],
                    out.at[pl.ds(epi_row * epi_m + offset, size), pl.ds(epi_column * epi_n, epi_n)],
                    predicate=(stored & size) != 0, commit_group=False)
                offset = offset + (stored & size)
                size //= 2
            plgpu.commit_smem_to_gmem_group()


def tgmm(tokens: jax.Array, cotangent: jax.Array, group_sizes: jax.Array, *, tiles: Tiles, out_dtype,
         multiprocessors: int, interpret: bool = False) -> jax.Array:
    """Each group's rows of `tokens` `[rows, depth]` transposed times its rows of `cotangent`
    `[rows, width]`, `[groups, depth, width]` in `out_dtype`; an empty group's is zero. The two compute
    warpgroups split the width (`tiles.split` is 'n'); `tiles.tile_m` is the depth of an output tile
    and `tiles.tile_k` the rows a contraction step takes. The depth must divide by `tiles.tile_m` and the
    width by `tiles.cta_n`."""
    rows, depth = tokens.shape
    width = cotangent.shape[1]
    groups = group_sizes.shape[0]
    out_dtype = jnp.dtype(out_dtype)
    t = tiles
    if t.split != 'n':
        raise ValueError(f"tgmm's warpgroups split the width, not {t.split!r}")
    m_tiles, n_tiles = depth // t.tile_m, width // t.cta_n
    sizes = group_sizes.astype(jnp.int32)
    starts = jnp.cumsum(sizes) - sizes
    x_transforms = _transforms(tokens.dtype, t.tile_m)
    dy_transforms = _transforms(cotangent.dtype, t.tile_n)
    out_transforms = _transforms(out_dtype, t.tile_n)

    def pipeline(step, compute_context, steps):
        return plgpu.emit_pipeline_warp_specialized(
            step, grid=(steps,), memory_registers=40,
            in_specs=[plgpu.BlockSpec((t.tile_k, t.tile_m), lambda r: (r, 0), transforms=x_transforms),
                      plgpu.BlockSpec((t.tile_k, t.cta_n), lambda r: (r, 0), transforms=dy_transforms)],
            wg_axis="wg", num_compute_wgs=2, max_concurrent_steps=t.stages,
            compute_context=compute_context)

    def body(sizes_ref, starts_ref, x_ref, dy_ref, out_ref):
        @functools.partial(pl.run_scoped,
                           # A static grid of fewer steps than stages would shrink the buffers.
                           allocations=pipeline(_ignore, _ignore, t.stages).get_allocations(x_ref, dy_ref),
                           slots=plgpu.SMEM((2, t.tile_m, t.tile_n), out_dtype, transforms=out_transforms),
                           masked=plgpu.Barrier(num_arrivals=1), seen=plgpu.Barrier(num_arrivals=1),
                           collective_axes="wg")
        def scope(allocations, slots, masked, seen):
            wg = lax.axis_index("wg")
            columns = wg * t.tile_n

            @plgpu.nd_loop((groups, m_tiles * n_tiles), collective_axes="sm")
            def _(info):
                group, flat = info.index
                m_tile, n_tile = plgpu.planar_snake(flat, (m_tiles, n_tiles), 1, t.width)
                size, start = sizes_ref[group], starts_ref[group]
                steps = pl.cdiv(size, t.tile_k)
                out = out_ref.at[group, pl.ds(m_tile * t.tile_m, t.tile_m),
                                 pl.ds(n_tile * t.cta_n + columns, t.tile_n)]

                def store(value):
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                    slots[wg] = value
                    plgpu.commit_smem()
                    plgpu.copy_smem_to_gmem(slots.at[wg], out)

                def compute_context(run):
                    @functools.partial(pl.run_scoped, acc=plgpu.ACC((t.tile_m, t.tile_n), jnp.float32))
                    def _(acc):
                        run(acc)
                        store(acc[...].astype(out_dtype))

                def step(indices, x_smem, dy_smem, acc):
                    (r,) = indices
                    kept = size - r * t.tile_k
                    own = dy_smem.at[:, pl.ds(columns, t.tile_n)]

                    # The group's last step may hold the next group's rows: zero them in both
                    # operands, each warpgroup its own cotangent columns, warpgroup 0 the shared rows,
                    # which warpgroup 1 reads once `masked` says so. Warpgroup 0 waits for `seen` in
                    # turn, so it never arrives on `masked` twice before warpgroup 1 waits once (a
                    # barrier's waits tell phases apart only by their parity).
                    @pl.when(kept < t.tile_k)
                    def _():
                        _zero_rows(own, kept)

                        @pl.when(wg == 0)
                        def _():
                            _zero_rows(x_smem, kept)
                            plgpu.commit_smem()
                            plgpu.barrier_arrive(masked)
                            plgpu.barrier_wait(seen)

                        @pl.when(wg == 1)
                        def _():
                            plgpu.commit_smem()
                            plgpu.barrier_wait(masked)
                            plgpu.barrier_arrive(seen)

                    plgpu.wgmma(acc, x_smem.T, own)
                    plgpu.wgmma_wait(0)
                    return acc

                @pl.when(size > 0)
                def _():
                    pipeline(step, compute_context, steps)(
                        x_ref.at[pl.ds(start, rows), pl.ds(m_tile * t.tile_m, t.tile_m)],
                        dy_ref.at[pl.ds(start, rows), pl.ds(n_tile * t.cta_n, t.cta_n)],
                        allocations=allocations)

                @pl.when((size == 0) & (wg < 2))
                def _():
                    store(_zeros((t.tile_m, t.tile_n), out_dtype))

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    launch = plgpu.kernel(
        body, out_type=jax.ShapeDtypeStruct((groups, depth, width), out_dtype), grid=(multiprocessors,),
        grid_names=("sm",), num_threads=3, thread_name="wg", compiler_params=_compiler_params(),
        interpret=interpret)
    return launch(sizes, starts, tokens, cotangent)


def _zeros(shape, dtype) -> jax.Array:
    """Zeros in the layout of a warpgroup's accumulator, which the stores take."""
    return plgpu.layout_cast(jnp.zeros(shape, dtype), plgpu.Layout.WGMMA)


def _zero_rows(block, kept) -> None:
    """Zero the rows of the SMEM `block` from `kept` on, by selection, as transposed_ragged_dot_mgpu
    masks its first and last blocks."""
    values = block[...]
    rows = plgpu.layout_cast(lax.broadcasted_iota(jnp.int32, values.shape, 0), plgpu.Layout.WGMMA)
    block[...] = jnp.where(rows < kept, values, jnp.zeros_like(values))
