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

"""Dew: a grouped GEMM's row tiles (`mosaic_gpu.gmm`, `tgmm`): its call's tile table (`tile_groups`,
`grouped_table`), each tile's group read from it (`GroupInfo`), the rows its schedule reads
(`row_extent`), and a kernel gradient's runs of each group's rows (`sliced`)."""

import dataclasses
from typing import NamedTuple

import jax
from jax import lax, numpy as jnp
from jax.experimental import pallas as pl

from dew.nn.kernels import tables


class TileGroup(NamedTuple):
    """Dew: a row tile's entry in a grouped call's tile table (`grouped_table`, `tables.packed`): its group
    (the lengths' index), the tile's first row, its group's first and past-last row, and the matrix it
    multiplies (its group's, the unrouted rows' the last's, or a stack's from `first`); a tile no group
    reaches, zeros. Its one declaration, which the table's writer and its reader (`GroupInfo.create`) take
    their columns from."""

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
        # (`row_unit`).
        block_start = pl.multiple_of(entry.block_start, unit)
        actual_start = jnp.maximum(entry.group_start, block_start)
        actual_end = jnp.minimum(entry.group_end, block_start + tile)
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


def tile_groups(lengths: jax.Array, tile: int, tiles: int, unit: int = 8,
                first: jax.Array | None = None) -> TileGroup:
    """Dew: each row tile's group info for `GroupInfo.create`, in XLA before the launch (`TileGroup`, packed
    by `grouped_table`): of tiles `0` to `tiles`, the group (`lengths`' index), the tile's first row, its
    group's first and past-last row, and its matrix (its group's, the last group's for the rows past the
    routed ones `lengths` ends with, or from `first` a stack's); a tile no group reaches, zeros. A group's
    tiles start at its first row (rounded down to the `unit` rows its family's copies start at, `row_unit`),
    whole tiles after it, as CUTLASS's grouped GEMM tiles each group: a group of up to a tile's rows reads its
    matrix once, where tiles on the product's own row grid read it once a tile the group touches (two for most
    of Kimi Linear's 128-row experts at 128-row tiles). The tiles number at most `cdiv(rows, tile)` plus the
    groups less one (a ceiling's sum exceeds the sum's ceiling by at most one a part). Upstream's
    `GroupInfo.create` walked every group's length in every tile, unrolled: at 129 groups (Qwen3's 128
    experts and the unrouted rows) two thirds of a down projection's instructions, and its global loads a
    third of its stalls (NCU-full)."""
    lengths = lengths.astype(jnp.int32)
    ends = prefix_sums(lengths)
    starts = ends - lengths
    rows = jnp.int32(tile)
    first_rows = starts // unit * unit
    counts = lax.div(ends - first_rows + rows - 1, rows) * (lengths > 0)
    firsts = prefix_sums(counts) - counts
    tids = jnp.arange(tiles, dtype=jnp.int32)[:, None]
    ours = (firsts[None] <= tids) & (tids < (firsts + counts)[None])
    group = jnp.argmax(ours, axis=1).astype(jnp.int32)
    found = jnp.any(ours, axis=1)
    block_start = first_rows[group] + (tids[:, 0] - firsts[group]) * rows
    matrix = jnp.minimum(group, lengths.shape[0] - 2) + (0 if first is None else first)
    return TileGroup(*(jnp.where(found, column, 0)
                       for column in (group, block_start, starts[group], ends[group], matrix)))


def prefix_sums(values: jax.Array) -> jax.Array:
    """`values`' inclusive prefix sums, as one sum over a lower-triangular mask: a group count's square is
    a few hundred thousand integers at most, which XLA fuses into the table's own kernels, where
    `jnp.cumsum` lowers to reduce-window kernels of its own. The table so took 6.4 to 7.2 us against 18 to
    20 at 129 and 257 groups, decode to training rows (row 3622, FS-3622-TABLE1-h100), the same table."""
    count = values.shape[0]
    held = jnp.arange(count)[None, :] <= jnp.arange(count)[:, None]
    return jnp.sum(jnp.where(held, values[None, :], 0), axis=1, dtype=values.dtype)


TMA_ROWS = 8
"""The rows a TMA descriptor tiles by, to which `gmm` pads its rows and a group's first tile starts."""



def grouped_table(group_sizes: jax.Array, m: int, row_tile: int, first: jax.Array | None,
                   unit: int = TMA_ROWS) -> jax.Array:
    """A grouped call's tile table (`tile_groups`) over its `m` padded rows at `row_tile`-row tiles: its
    groups', then the rows past the routed ones a last group, whose tiles store zeros."""
    g = group_sizes.shape[0]
    lengths = jnp.concatenate([group_sizes.astype(jnp.int32),
                               jnp.maximum(m - jnp.sum(group_sizes, dtype=jnp.int32), 0)[None]])
    # Each tile's matrix is its group's, of the stack from `first` where given (`TileGroup.matrix`): the
    # group itself stays the call's, which its validity and its side operands read.
    return tables.packed(tile_groups(lengths, row_tile, pl.cdiv(m, row_tile) + g, unit, first))


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
    groups' one prefix sum (`prefix_sums`, as `tile_groups` reads them)."""
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
