"""A persistent launch's walk over its output tiles: the planar snake that orders them (`planar_snake`);
its tail, the output tiles past its whole waves, their steps balanced over the programs (`balanced`,
`Tail`), each tile fixed up by the program that walks its first step from the partials the others leave it
(`mosaic.Family.fixes_up`, `mosaic.RingFamily.fixed_up`); and the order an interpreter, with no
semaphores, walks them in (`ordered`)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax


@dataclasses.dataclass(frozen=True)
class Tail:
    """A dense kernel gradient's output tiles beyond its whole waves, balanced over the programs
    (`balanced`): the whole waves each program walks first, a tile a wave, as the snake orders them; then
    the remaining tiles' steps, `iterations` of them, shared out as `partitions` contiguous ranges, a
    program's range crossing a tile's boundary leaving its partial tile to the program that walks the tile's
    first step, which adds it (CUTLASS's data-parallel and stream-K hybrid)."""
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
        of the tile's steps it walks and how many, and the exclusive end of the programs whose ranges
        start inside the tile (which leave it their partials); a segment past the program's range walks
        none, at a tile its descriptors can take."""
        segment, program = index
        whole = segment < self.waves
        lo = jnp.minimum(program, self.partitions) * self.iterations // self.partitions
        hi = jnp.minimum(program + 1, self.partitions) * self.iterations // self.partitions
        tail_tile = lo // self.steps + jnp.maximum(segment - self.waves, 0)
        start = jnp.where(whole, 0, jnp.maximum(lo, tail_tile * self.steps) - tail_tile * self.steps)
        end = jnp.where(whole, self.steps,
                        jnp.minimum(hi, (tail_tile + 1) * self.steps) - tail_tile * self.steps)
        gathered = jnp.where(whole, program + 1,
                             -(-((tail_tile + 1) * self.steps * self.partitions) // self.iterations))
        tile = jnp.where(whole, segment * programs + program, self.waves * programs + tail_tile)
        return (jnp.minimum(tile, tiles - 1), start, jnp.maximum(end - start, 0),
                jnp.minimum(gathered, self.partitions))


def balanced(tiles: int, steps: int, programs: int, *, handoff_steps: float = 0.0) -> Tail | None:
    """A dense kernel gradient's tail (`Tail`) where balancing its last wave is no slower than walking it,
    priced in MMA steps; None elsewhere. Walked, the last wave's `tiles` mod `programs` tiles take their
    `steps` whole while the other programs idle. Balanced, the programs share those steps, but the programs
    sharing a balanced tile read its operands apart, without the reuse a wave's neighbouring tiles share,
    at about half the rate, and a tile's partials reach the program that adds them through a hand-off
    (`handoff_steps`, the part's measured semaphore hand-off in the tile's steps, row 3613): so a tail
    balances where it is at most half a wave less its hand-off, and none where the whole product is one
    wave: TGHYBRID1-h100 and
    BALANCED1-h100 (an H100 PCIe's 114 programs) measured every tail at most half a wave faster or level
    (Qwen3.5-4B's up and down projections' kernel gradients 3% to 7%, Llama 3 8B's output projection's 3%,
    their 36, 36 and 56 tiles) and every fuller one slower (Qwen3.5-4B's query-key-value projection's 6%,
    its output projection's 30% at 92 tiles, its key projection's 87% at 80 tiles and no whole wave)."""
    waves, tail = divmod(tiles, programs)
    if waves == 0 or tail == 0 or steps == 0 or 2 * tail * steps / programs + handoff_steps > steps:
        return None
    iterations = tail * steps
    return Tail(waves, iterations, min(programs, iterations), steps)


def ordered(tail: Tail | None, index: tuple, programs: int, interpreted: bool) -> tuple:
    """A persistent walk's index as its program walks it: interpreted, a balanced tail's programs counted
    from the last. The interpreter has no semaphores, and its one program walks the segments in turn, each
    segment's programs from the first (`gemm.mosaic_gpu`'s kernel gradient); a program leaving a tile its
    partial (`mosaic.RingFamily.fixed_up`) walks it in its first tail segment, no later than the tile's
    first program, so with the programs reversed each tile's partials are there before that program adds
    them."""
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
