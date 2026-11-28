"""A grouped call's tile table (`row_tiles.grouped_table`): each group's row tiles from its first row, in its
one-tile form (every routed row in one tile: a decode step's) and its packed form, read as `gmm` reads them
(`row_tiles.GroupInfo.create`), and none reading past its operands' allocation."""

import jax.numpy as jnp
import pytest

from dew.nn.kernels.templates.gemm import row_tiles

TABLE_CASES = [([120, 8], 64, None), ([3, 200, 1, 0, 57], 128, None), ([0, 0, 13], 256, None),
               ([64, 64], 64, None), ([7] * 9, 128, None), ([0, 1, 0, 0, 1, 1, 0, 5], 64, None),
               ([5, 0, 6], 64, 3), ([5, 0, 6, 40], 32, 2), ([1] * 8, 8, None)]
"""Group sizes, a row tile and a stack's first matrix: decode steps' (every row in one tile: the table's
one-tile form) and prefills' (its packed form), groups empty, unaligned, and filling whole tiles."""


def _table_tiles(sizes, row_tile, first):
    """The tiles a grouped call's table gives (`row_tiles.grouped_table`), each slot read as `gmm` reads it
    (`row_tiles.GroupInfo.create`): its slots, and each tile that walks its group's rows `(group, first row,
    first and past-last row walked, matrix)`."""
    rows = -(-sum(sizes) // row_tiles.TMA_ROWS) * row_tiles.TMA_ROWS
    table = row_tiles.grouped_table(jnp.asarray(sizes, jnp.int32), rows, row_tile,
                                    None if first is None else jnp.int32(first))
    infos = [row_tiles.GroupInfo.create(table, row_tile, slot) for slot in range(table.shape[0])]
    walked = {(int(info.group_id), int(info.block_start), int(info.actual_start), int(info.actual_end),
               int(info.matrix)) for info in infos if int(info.actual_size) > 0}
    return rows, table.shape[0], infos, walked


@pytest.mark.parametrize("sizes, row_tile, first", TABLE_CASES)
def test_a_grouped_tables_tiles_are_each_groups_from_its_first_row(sizes, row_tile, first):
    """A grouped call's tile table (`row_tiles.grouped_table`, in its one-tile or its packed form) gives each
    group of rows its tiles from its first row rounded down to 8, whole tiles after it, each over the
    group's rows within it at the group's matrix (a stack's from `first`), the rows past the routed ones a
    last group's (its tiles store zeros, at the last matrix), and no other tile, over as many slots as the
    walk takes (`cdiv(rows, row_tile)` plus the groups), as CUTLASS's grouped GEMM tiles each group (row
    14177: the table now one masked sum a column, its tiles the same)."""
    rows, slots, _, walked = _table_tiles(sizes, row_tile, first)
    expected, start = set(), 0
    for group, size in enumerate([*sizes, rows - sum(sizes)]):
        if not size:
            continue
        row = start // row_tiles.TMA_ROWS * row_tiles.TMA_ROWS
        while row < start + size:
            expected.add((group, row, max(start, row), min(start + size, row + row_tile),
                          min(group, len(sizes) - 1) + (first or 0)))
            row += row_tile
        start += size
    assert slots == -(-rows // row_tile) + len(sizes)
    assert walked == expected


@pytest.mark.parametrize("sizes, row_tile, first", TABLE_CASES)
def test_no_row_tile_reads_past_its_operands_allocation(sizes, row_tile, first):
    """Every row tile `gmm`'s schedule takes (`row_tiles.grouped_table`, a group's tiles from its first row)
    reads within the rows its operands are allocated to (`row_tiles.row_extent`): its first row plus the
    tile's rows, unpredicated on an A100, stay inside the extent (row 1866 K3: groups [120, 8] at 64-row
    tiles read rows 120 to 183 of a 128-row lhs); an empty slot's too, whose finish reads its side rows from
    its origin (an empty unrouted tail's at row 0: the head review of afa1aedcd8)."""
    rows, _, infos, _ = _table_tiles(sizes, row_tile, first)
    reads = max(int(info.block_start) + row_tile for info in infos)
    assert reads <= row_tiles.row_extent(rows, row_tile)
    if sizes == [120, 8]:
        assert reads > -(-rows // row_tile) * row_tile  # past the row-tile padding that held before


SIDE_STRIDES = {"uint8 columns first": (jnp.uint8, ("columns", "rows"), (128, 8), 16),
                "bf16 columns first": (jnp.bfloat16, ("columns", "rows"), (128, 4), 8),
                "fp32 depth first": (jnp.float32, ("depth", "rows"), (64, 3), 8),
                "uint8 rows first": (jnp.uint8, ("rows", "columns"), (8, 128), 8),
                "uint8 rows alone": (jnp.uint8, ("rows",), (8,), 8)}
"""A hook's side, its axes, shape and the rows a TMA copy of it takes past a decode's 8 (`strided_rows`)."""


@pytest.mark.parametrize("dtype, axes, shape, rows", SIDE_STRIDES.values(), ids=SIDE_STRIDES)
def test_a_side_on_its_rows_is_held_where_a_tma_copy_takes_its_strides(dtype, axes, shape, rows):
    """A grouped call's hook side held at a decode's rows (`gemm.sides_extended` at 8 of them, a bounded
    family's `held`): one whose rows follow another of its dimensions takes the fewest rows past them whose
    strides are each a whole 16 bytes, but the last, as a TMA copy reads them (`copies.ROW_BYTES`), its
    values kept and zeros past them; one whose rows lead its dimensions stays at the rows (train 1499's
    review K-F1: a uint8 side of 128 columns by 8 rows held at those rows had an 8-byte stride, which
    Mosaic GPU's copy refused)."""
    import math

    import numpy as np

    from dew.nn.kernels.templates import copies, gemm

    side = jnp.ones(shape, dtype)
    (held,) = gemm.sides_extended((side,), (axes,), 8)
    axis = axes.index("rows")
    assert held.shape[axis] == rows
    strides = [math.prod(held.shape[at + 1:]) * jnp.dtype(dtype).itemsize for at in range(held.ndim - 1)]
    assert all(stride % copies.ROW_BYTES == 0 for stride in strides)
    kept = tuple(slice(0, extent) for extent in shape)
    np.testing.assert_array_equal(np.asarray(held[kept]), np.asarray(side))
    assert not np.asarray(held).sum() - np.asarray(side).sum()
