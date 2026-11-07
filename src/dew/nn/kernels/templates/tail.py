"""A persistent launch's walk over its output tiles: the planar snake that orders them (`planar_snake`);
its tail, the output tiles past its whole waves, their steps balanced over the programs (`balanced`,
`Tail`), each tile fixed up from the partials its programs leave by the last of them to arrive
(`mosaic.Family.fixes_up`, `mosaic.Family.fixed_up`); and the order an interpreter, with no semaphores,
walks them in (`ordered`)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


@dataclasses.dataclass(frozen=True)
class Tail:
    """A dense kernel gradient's output tiles beyond its whole waves, balanced over the programs
    (`balanced`): the whole waves each program walks first, a tile a wave, as the snake orders them; then
    the remaining tiles' steps, `iterations` of them, shared out as `partitions` contiguous ranges, each
    program whose range crosses into a tile leaving its partial, the last of them to arrive adding them
    (CUTLASS's data-parallel and stream-K hybrid)."""
    waves: int
    iterations: int
    partitions: int
    steps: int

    @property
    def segments(self) -> int:
        """The tiles (whole or in part) a program walks at most: its waves' and its range's."""
        return self.waves + -(-self.iterations // self.partitions // self.steps) + 1

    def segment(self, index, programs: int, tiles: int) -> tuple:
        """A program's `index`-th segment (`(segment, program)`): its tile's place in the snake, the first
        of the tile's steps it walks and how many, and the programs that walk a part of the tile, the first
        (which walks its first step) and the exclusive end (`mosaic.Family.fixed_up`'s); a segment past the
        program's range walks none, at a tile its descriptors can take."""
        segment, program = index
        whole = segment < self.waves
        lo = jnp.minimum(program, self.partitions) * self.iterations // self.partitions
        hi = jnp.minimum(program + 1, self.partitions) * self.iterations // self.partitions
        tail_tile = lo // self.steps + jnp.maximum(segment - self.waves, 0)
        start = jnp.where(whole, 0, jnp.maximum(lo, tail_tile * self.steps) - tail_tile * self.steps)
        end = jnp.where(whole, self.steps,
                        jnp.minimum(hi, (tail_tile + 1) * self.steps) - tail_tile * self.steps)
        # The programs whose ranges reach into the tile: those whose first step `p * iterations //
        # partitions` lies before its end, the first the last of them whose first step is at or before its
        # start.
        first = jnp.where(whole, program,
                          -(-((tail_tile * self.steps + 1) * self.partitions) // self.iterations) - 1)
        gathered = jnp.where(whole, program + 1,
                             -(-((tail_tile + 1) * self.steps * self.partitions) // self.iterations))
        tile = jnp.where(whole, segment * programs + program, self.waves * programs + tail_tile)
        return (jnp.minimum(tile, tiles - 1), start, jnp.maximum(end - start, 0),
                (first, jnp.minimum(gathered, self.partitions)))


@dataclasses.dataclass(frozen=True)
class Traffic:
    """What a product's output tiles read a step and how fast the device serves it, which a tail's pricing
    reads (`balanced`): the tiles' grid as the walk orders them (`planar_snake`'s `shape`, its columns the
    minor dimension, `band` tiles wide), a step's operand block of a tile row (`row_bytes`, the left
    operand's) and of a tile column (`column_bytes`, the right's), one MMA step on a multiprocessor alone
    (`step_seconds`), the multiprocessors the programs share (`multiprocessors`: programs beyond them take
    their steps in turn), the device memory's rate (`bandwidth`) and the L2's bytes (`l2_bytes`), within
    whose rounds a block read again is a hit; none where unknown."""
    grid: tuple[int, int]
    band: int
    row_bytes: int
    column_bytes: int
    step_seconds: float
    multiprocessors: int
    bandwidth: float
    l2_bytes: int | None

    def issued(self, programs: int) -> float:
        """The seconds `programs` programs' one MMA step each takes, sharing the multiprocessors."""
        return self.step_seconds * -(-programs // self.multiprocessors)


def balanced(tiles: int, steps: int, programs: int, traffic: Traffic, *,
             handoff_steps: float = 0.0) -> Tail | None:
    """A dense product's tail (`Tail`) where balancing its last wave is faster than walking it, each priced
    from the tiles' traffic (`Traffic`); None elsewhere, and where the whole product is one wave. Walked,
    the last wave's `tiles` mod `programs` tiles take their `steps` together, each step reading its rows'
    and columns' blocks once (`_walked`). Balanced, the programs share those steps in contiguous ranges, a
    round of them reading each program's blocks, a block read again within the rounds the L2 holds read
    from it (`_rounds`): programs on tiles of a row or column at nearby steps share their reads, as the
    ranges' alignment with the tiles leaves them (row 3798: an H100 PCIe balanced Llama 3 8B's output
    projection's kernel gradient 3% faster, its ranges a tile's half, against Qwen3.5-4B's query-key-value
    one 6% slower); and each tile's partials reach the program adding them through a hand-off
    (`handoff_steps`, the part's in its steps, row 3613). Each round takes its MMA step, or its missed
    bytes at the memory's rate where longer."""
    waves, tail = divmod(tiles, programs)
    if waves == 0 or tail == 0 or steps == 0:
        return None
    iterations = tail * steps
    candidate = Tail(waves, iterations, min(programs, iterations), steps)
    with jax.ensure_compile_time_eval():
        rows, columns = (np.asarray(axis) for axis in planar_snake(
            jnp.arange(waves * programs, tiles, dtype=jnp.int32), traffic.grid, 1, traffic.band))
    shared = _rounds(candidate, rows, columns, traffic) + handoff_steps * traffic.step_seconds
    return candidate if shared < _walked(steps, rows, columns, traffic) else None


def _walked(steps: int, rows: np.ndarray, columns: np.ndarray, traffic: Traffic) -> float:
    """The seconds the tail's tiles (at `rows`, `columns`) take walked together: each of their `steps`
    reading every distinct row's and column's block once, its MMA step or those bytes, whichever longer."""
    read = len(np.unique(rows)) * traffic.row_bytes + len(np.unique(columns)) * traffic.column_bytes
    return steps * max(traffic.issued(len(rows)), read / traffic.bandwidth)


def _rounds(tail: Tail, rows: np.ndarray, columns: np.ndarray, traffic: Traffic) -> float:
    """The seconds a balanced tail's rounds take: in each, every program walks the next step of its range
    (tile `iteration // steps`, step `iteration % steps`), reading its tile's row and column blocks at that
    step; a block missed where no program read it within the rounds the L2 holds (a round asks at most
    every program's two blocks), each round its MMA step or its missed bytes, whichever longer."""
    bounds = np.arange(tail.partitions + 1) * tail.iterations // tail.partitions
    program = np.repeat(np.arange(tail.partitions), np.diff(bounds))
    iteration = np.arange(tail.iterations)
    turn = iteration - bounds[program]
    tile, step = np.divmod(iteration, tail.steps)
    asked = tail.partitions * (traffic.row_bytes + traffic.column_bytes)
    window = 0 if traffic.l2_bytes is None else traffic.l2_bytes // asked
    rounds = int(turn.max()) + 1
    missed = np.zeros(rounds)
    for line, size in ((rows[tile], traffic.row_bytes), (columns[tile], traffic.column_bytes)):
        block = line * tail.steps + step
        order = np.lexsort((turn, block))
        block, read = block[order], turn[order]
        # A block's first read misses, and one after a gap longer than the L2 holds.
        first = np.ones(len(block), dtype=bool)
        first[1:] = (block[1:] != block[:-1]) | (read[1:] - read[:-1] > window)
        missed += np.bincount(read[first], minlength=rounds) * size
    active = np.bincount(turn, minlength=rounds)
    issued = traffic.step_seconds * -(-active // traffic.multiprocessors)
    return float(np.maximum(issued, missed / traffic.bandwidth).sum())


def ordered(tail: Tail | None, index: tuple, programs: int, interpreted: bool) -> tuple:
    """A persistent walk's index as its program walks it: interpreted, a balanced tail's programs counted
    from the last. The interpreter has no semaphores, and its one program walks the segments in turn, each
    segment's programs from the first (`gemm.mosaic_gpu`'s kernel gradient); a program leaving a tile its
    partial (`mosaic.Family.fixed_up`) walks it in its first tail segment, no later than the tile's first
    program, so with the programs reversed each tile's partials are there before that program, which
    interpreted adds them, reads them."""
    if tail is None or not interpreted:
        return index
    return (*index[:-1], programs - 1 - index[-1])


# `planar_snake` is JAX's `jax._src.pallas.mosaic_gpu.helpers.planar_snake` (jax-ml/jax 38dc6dd2f), with Dew's
# full bands kept as a shift and a mask and its coordinates selected without branching where the band is a
# power of two (the fork's 44eda26b6 and cd1f7681a), under JAX's license:
#
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
def planar_snake(index: jax.Array, shape: tuple[int | jax.Array, int | jax.Array], minor_dim: int,
                 width: int) -> tuple[jax.Array, jax.Array]:
    """The tile of `shape` a persistent walk's linear `index` takes: the minor dimension (`minor_dim`) split
    into bands `width` tiles wide, each band walked along the major dimension a line of its tiles at a time,
    every other band's major dimension reversed, so the walk snakes from one band into the next and a wave
    of programs shares its operands' blocks in the L2 (`resources.band` sizes the band). Over `(8, 8)`,
    `minor_dim=0` and `width=2`, each pair of rows a band:

         0   2   4   6   8  10  12  14
         1   3   5   7   9  11  13  15
        30  28  26  24  22  20  18  16
        31  29  27  25  23  21  19  17
        32  34  36  38  40  42  44  46
        33  35  37  39  41  43  45  47
        62  60  58  56  54  52  50  48
        63  61  59  57  55  53  51  49

    A last band narrower than `width` holds the minor dimension's remainder."""
    def divmod_(a, b):
        return lax.div(a, b), lax.rem(a, b)

    width_is_power_of_2 = width & (width - 1) == 0
    major_extent = shape[1 - minor_dim]
    major_is_power_of_2 = isinstance(major_extent, int) and major_extent & (major_extent - 1) == 0
    shift = width.bit_length() - 1
    band_width = jnp.int32(width)
    major_size = jnp.int32(major_extent)
    minor_size = jnp.int32(shape[minor_dim])
    band_size = band_width * major_size
    full_bands, last_width = divmod_(minor_size, band_width)
    in_full_band = index < full_bands * band_size

    if width_is_power_of_2:
        band, within = divmod_(index, band_size)
        # The full band's shift and mask apart from the last band's division: with two divisions LLVM
        # selects their divisors before dividing, turning the constant power-of-two division into a
        # variable one for every tile.
        major, minor_within = jax.tree.map(
            functools.partial(lax.select, in_full_band),
            (lax.shift_right_logical(within, jnp.int32(shift)), lax.bitwise_and(within, band_width - 1)),
            divmod_(within, lax.max(last_width, jnp.int32(1))))
    else:
        def full():
            # Divided by the width first, so a power-of-two major size makes one of the divisions cheap.
            line, minor = divmod_(index, band_width)
            band, major = divmod_(line, major_size)
            return band, major, minor

        def last():
            return full_bands, *divmod_(index - full_bands * band_size, last_width)

        if major_is_power_of_2:
            band, major, minor_within = lax.cond(in_full_band, full, last)
        else:
            band, major, minor_within = jax.tree.map(functools.partial(lax.select, in_full_band), full(),
                                                     last())

    major = lax.select(lax.rem(band, jnp.int32(2)) == 0, major, major_size - 1 - major)
    minor = band * band_width + minor_within
    return (minor, major) if minor_dim == 0 else (major, minor)
