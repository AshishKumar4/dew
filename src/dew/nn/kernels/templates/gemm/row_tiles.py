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

"""Dew: a grouped GEMM's row tiles (`mosaic_gpu.gmm`, `tgmm`): its call's tile table (`grouped_table`),
each tile's group read from it (`GroupInfo`), the rows its schedule reads (`row_extent`), and a kernel
gradient's runs of each group's rows (`sliced`)."""

import dataclasses
from typing import NamedTuple

import jax
from jax import lax, numpy as jnp
from jax.experimental import pallas as pl

from dew.nn.kernels import tables

TMA_ROWS = 8
"""The rows a TMA descriptor tiles by, to which `gmm` pads its rows and a group's first tile starts."""


class TileGroup(NamedTuple):
    """Dew: a row tile's entry in a grouped call's tile table (`grouped_table`): its group (the sizes'
    index), the tile's first row before its reader rounds it down to the rows its copies start at
    (`GroupInfo.create`), its group's first and past-last row, and the matrix it multiplies (its group's, or
    a stack's from `first`); a tile no group reaches, zeros. Its one declaration, which the table's writer
    and its reader take their columns from."""

    group: tables.Column
    block_start: tables.Column
    group_start: tables.Column
    group_end: tables.Column
    matrix: tables.Column


@dataclasses.dataclass(frozen=True)
class GroupInfo:
    """Information regarding the group being processed in a block."""

    group_id: jax.Array
    block_start: jax.Array
    actual_start: jax.Array
    actual_end: jax.Array
    start_within_block: jax.Array
    actual_size: jax.Array
    matrix: jax.Array
    """Dew: the matrix the tile multiplies (`TileGroup.matrix`)."""

    @classmethod
    def create(cls, table, tile, tid, unit: int = 8):
        """Dew: the group info of row tile `tid`, read from its call's tile table (`TileGroup`) rather than
        walked over every group's length in each tile."""
        tile = jnp.int32(tile)
        entry = tables.unpacked(TileGroup, lambda column: table[tid, column])
        # Mosaic's copies and stores read a slice's start as a multiple of the rows its tiling takes
        # (`row_unit`, a power of two), to which the tile's first row rounds down (`TileGroup`).
        block_start = pl.multiple_of(lax.bitwise_and(entry.block_start, jnp.int32(-unit)), unit)
        actual_start = jnp.maximum(entry.group_start, block_start)
        # No rows where its group ends before the tile's first (the unrouted rows' where the sizes reach
        # them, `_one_tile`), and such a tile at row 0, as a slot no group reaches: an empty tile's finish
        # reads its side rows from its origin unmasked on an A100 (head review of afa1aedcd8).
        actual_end = jnp.maximum(jnp.minimum(entry.group_end, block_start + tile), actual_start)
        empty = actual_end == actual_start
        block_start = pl.multiple_of(jnp.where(empty, 0, block_start), unit)
        actual_start, actual_end = (jnp.where(empty, 0, bound) for bound in (actual_start, actual_end))
        start_within_block = actual_start - block_start
        actual_size = actual_end - actual_start
        return cls(group_id=entry.group, block_start=block_start, actual_start=actual_start,
                   actual_end=actual_end, start_within_block=start_within_block, actual_size=actual_size,
                   matrix=entry.matrix)

    @classmethod
    def dense(cls, tile: int, tid, rows: int):
        """A dense row tile's bounds, with a final tile clipped to the padded operand's rows."""
        start = pl.multiple_of(tid * tile, TMA_ROWS)
        size = jnp.minimum(tile, rows - start)
        return cls(group_id=jnp.int32(0), block_start=start, actual_start=start,
                   actual_end=start + size, start_within_block=jnp.int32(0), actual_size=size,
                   matrix=jnp.int32(0))


def grouped_table(group_sizes: jax.Array, m: int, row_tile: int, first: jax.Array | None,
                  unit: int = TMA_ROWS) -> jax.Array:
    """A grouped call's tile table (`TileGroup`'s columns, int32 `[cdiv(m, row_tile) + groups, 5]`) over its
    `m` padded rows at `row_tile`-row tiles. A group's tiles start at its first row (rounded down to the
    `unit` rows its family's copies start at, `row_unit`), whole tiles after it, as CUTLASS's grouped GEMM
    tiles each group: a group of up to a tile's rows reads its matrix once, where tiles on the product's own
    row grid read it once a tile the group touches (two for most of Kimi Linear's 128-row experts at 128-row
    tiles). Each entry is one masked sum over the groups, its group's terms alone, so XLA forms the table in
    as few kernels as its sums nest: where every row lies in one tile (`m <= row_tile`: a decode step's),
    a group's one tile is its own slot and each column a sum of the sizes, one reduction (`_one_tile`);
    else its tiles are packed in the groups' order, the slots each group starts at a prefix sum of the
    tiles before it, three (`_packed`). Upstream's `GroupInfo.create` walked every group's length in every
    tile, unrolled: at 129 groups (Qwen3's 128 experts and the unrouted rows) two thirds of a down
    projection's instructions, and its global loads a third of its stalls (NCU-full); XLA's table with an
    argmax, gathers and a stack took six kernels a call (row 14177: a third of an MoE decode layer's
    launches). The rows past the routed ones, a last group of the packed table's, whose tiles walk nothing
    (`GroupInfo`'s group is the call's own)."""
    sizes = group_sizes.astype(jnp.int32)
    matrix = jnp.int32(0) if first is None else jnp.asarray(first, jnp.int32)
    if m <= row_tile and sizes.shape[0]:
        return _one_tile(sizes, m, matrix)
    return _packed(sizes, m, row_tile, pl.cdiv(m, row_tile) + sizes.shape[0], unit, matrix)


def _one_tile(sizes: jax.Array, m: int, first: jax.Array) -> jax.Array:
    """`grouped_table`'s where every routed row lies in one row tile: each group's one tile at the group's
    own slot, then the unrouted rows' (to `m`, whose tile stores zeros), its first row its group's (rounded
    down by its reader), each column `TileGroup`'s summed over the sizes, `[groups + 1, 5]`: zeros where a
    group routes no row; the unrouted rows' none where the sizes reach `m` (its end at or before its start,
    which its reader takes as no rows)."""
    groups = sizes.shape[0]
    slot = jnp.arange(groups + 1, dtype=jnp.int32)[:, None, None]
    term = jnp.arange(groups, dtype=jnp.int32)[None, None, :]
    column = jnp.arange(len(TileGroup._fields), dtype=jnp.int32)[None, :, None]
    unrouted = slot == groups
    held = jnp.concatenate([sizes, jnp.zeros((1,), jnp.int32)])[:, None, None]
    before = jnp.where(term < slot, sizes[None, None, :], 0)
    once = jnp.where(term == 0, 1, 0)
    through = before + jnp.where(term == slot, sizes[None, None, :], 0)
    terms = _columned(column, TileGroup(
        group=slot * once, block_start=before, group_start=before,
        group_end=jnp.where(unrouted, m * once, through),
        matrix=(jnp.minimum(slot, groups - 1) + first) * once))
    return jnp.sum(jnp.where((held > 0) | unrouted, terms, 0), axis=2, dtype=jnp.int32)


def _packed(sizes: jax.Array, m: int, tile: int, tiles: int, unit: int, first: jax.Array) -> jax.Array:
    """`grouped_table`'s in general, `[tiles, 5]`: the groups' ends one prefix sum (`prefix_sums`); each
    group's tiles' count and the slots it starts and ends at two more over its tiles; each slot's entry its
    owning group's, summed over the groups (none owning it: zeros). The groups past `sizes` one more, the
    rows to `m` past the routed ones."""
    routed = prefix_sums(sizes)
    total = routed[-1:] if sizes.shape[0] else jnp.zeros((1,), jnp.int32)
    lengths = jnp.concatenate([sizes, jnp.maximum(m - total, 0)])
    ends = jnp.concatenate([routed, jnp.maximum(m, total)])
    starts = ends - lengths
    first_rows = lax.bitwise_and(starts, jnp.int32(-unit))
    counts = _ceil_divided(ends - first_rows, tile) * (lengths > 0)
    group = jnp.arange(lengths.shape[0], dtype=jnp.int32)
    # Each group's last slot's next and its first: the tiles through it, and before it.
    through = prefix_sums(counts)
    before = through - counts
    slot = jnp.arange(tiles, dtype=jnp.int32)[:, None, None]
    owns = (before[None, None, :] <= slot) & (slot < through[None, None, :])
    column = jnp.arange(len(TileGroup._fields), dtype=jnp.int32)[None, :, None]
    entry = _columned(column, TileGroup(
        group=group[None, None, :], block_start=first_rows + (slot - before[None, None, :]) * tile,
        group_start=starts[None, None, :], group_end=ends[None, None, :],
        matrix=jnp.minimum(group, lengths.shape[0] - 2)[None, None, :] + first))
    return jnp.sum(jnp.where(owns, entry, 0), axis=2, dtype=jnp.int32)


def _columned(column: jax.Array, values: TileGroup) -> tables.Column:
    """Each of `values`' columns where `column` (an iota over `TileGroup`'s columns) is its own: selections by
    comparison alone, which XLA fuses into the sum reading them (`jnp.select` picks by an argmax over its
    conditions, a reduction and a selection of its own: two more kernels, row 14177's census)."""
    *rest, last = values
    for at in range(len(rest) - 1, -1, -1):
        last = jnp.where(column == at, rest[at], last)
    return last


def _ceil_divided(values: jax.Array, divisor: int) -> jax.Array:
    """Non-negative int32 `values` over `divisor`, rounded up: by a shift where `divisor` is a power of two,
    which XLA fuses into the sums reading it, where it keeps an integer division apart (row 14177)."""
    raised = values + (divisor - 1)
    if divisor & (divisor - 1) == 0:
        return lax.shift_right_logical(raised, jnp.int32(divisor.bit_length() - 1))
    return lax.div(raised, jnp.int32(divisor))


def prefix_sums(values: jax.Array) -> jax.Array:
    """`values`' inclusive prefix sums, as one sum over a lower-triangular mask: a group count's square is
    a few hundred thousand integers at most, which XLA fuses into the table's own kernels, where
    `jnp.cumsum` lowers to reduce-window kernels of its own. The table so took 6.4 to 7.2 us against 18 to
    20 at 129 and 257 groups, decode to training rows (row 3622, FS-3622-TABLE1-h100), the same table."""
    count = values.shape[0]
    held = jnp.arange(count)[None, :] <= jnp.arange(count)[:, None]
    return jnp.sum(jnp.where(held, values[None, :], 0), axis=1, dtype=values.dtype)


def sliced(group_sizes: jax.Array, block_k: int, splits: int) -> jax.Array:
    """Dew: each group's rows as `splits` consecutive runs (group-major, a run's size each): the group's
    `block_k`-row blocks, from its first row's block, shared out in whole blocks as evenly as they divide, the
    first run starting at the group's first row and the last ending at its last; a run of a group with
    fewer blocks than runs is empty."""
    starts, ends, blocks, block_starts = group_blocks(group_sizes, block_k)
    shares = jnp.arange(1, splits)[None, :] * blocks[:, None] // splits
    inner = jnp.clip(block_starts[:, None] + shares * block_k, starts[:, None], ends[:, None])
    bounds = jnp.concatenate([starts[:, None], inner, ends[:, None]], axis=1)
    return jnp.diff(bounds, axis=1).reshape(-1).astype(jnp.int32)


def group_blocks(group_sizes: jax.Array, block_k: int) -> tuple:
    """Each group's first row, end, count of `block_k`-row blocks, and its first block's first row, from the
    groups' one prefix sum (`prefix_sums`, as `grouped_table` reads them)."""
    group_sizes = group_sizes.astype(jnp.int32)
    group_ends = prefix_sums(group_sizes)
    group_starts = group_ends - group_sizes
    group_block_starts = group_starts // block_k * block_k
    group_block_ends = -(group_ends // -block_k) * block_k
    return group_starts, group_ends, (group_block_ends - group_block_starts) // block_k, group_block_starts


def row_extent(rows: int, row_tile: int, unit: int = TMA_ROWS) -> int:
    """Dew: the rows `gmm`'s tile schedule reads of `rows` rows (a multiple of `unit`, `row_unit`) at tiles
    of `row_tile` rows: a tile starts at most `unit` before the rows end (`tile_groups`) and reads
    `row_tile` rows on."""
    return rows + row_tile - unit
