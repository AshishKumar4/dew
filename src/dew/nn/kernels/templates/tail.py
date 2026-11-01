"""A persistent launch's tail: the output tiles past its whole waves, their steps balanced over the
programs (`balanced`, `Tail`), each tile fixed up by the program that walks its first step from the
partials the others leave it (`mosaic.Family.fixes_up`, `mosaic.RingFamily.fixed_up`); and the order an
interpreter, with no semaphores, walks them in (`ordered`)."""

import dataclasses

import jax.numpy as jnp


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


def balanced(tiles: int, steps: int, programs: int) -> Tail | None:
    """A dense kernel gradient's tail (`Tail`) where its `tiles` of `steps` each leave a last wave at
    most half full over `programs`; None elsewhere. A short last wave's programs idle while a few walk
    whole tiles, which balancing them saves; but the programs sharing a balanced tile read its operands
    apart, without the reuse a wave's neighbouring tiles share, at about half the rate, so a tail
    balances where it is at most half a wave, and none where the whole product is one: TGHYBRID1-h100 and
    BALANCED1-h100 (an H100 PCIe's 114 programs) measured every tail at most half a wave faster or level
    (Qwen3.5-4B's up and down projections' kernel gradients 3% to 7%, Llama 3 8B's output projection's 3%,
    their 36, 36 and 56 tiles) and every fuller one slower (Qwen3.5-4B's query-key-value projection's 6%,
    its output projection's 30% at 92 tiles, its key projection's 87% at 80 tiles and no whole wave)."""
    waves, tail = divmod(tiles, programs)
    if waves == 0 or tail == 0 or steps == 0 or 2 * tail > programs:
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
