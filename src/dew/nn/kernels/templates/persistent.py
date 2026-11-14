"""A persistent kernel's tile loop (`mosaic.Family.persist`'s): its tiles spread over the launch's programs,
the step its streams are at and the accumulator turn each tile takes counted across them (`tiled`), a
consumer role's tiles issued and finished in turn (`consumed`), and a tile's walk of no steps (`unwalked`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

if TYPE_CHECKING:
    from dew.nn.kernels.templates.mosaic import Family


def unwalked(family: Family) -> Callable:
    """A walk of no steps (`Family.persist`'s, where a tile streams nothing): its carry and accumulators as
    they start."""
    def walk(step, start, shapes, *, initial=None):
        del step
        return start, family.started(shapes, initial)

    return walk


def tiled(tiles: tuple[int, ...], schedule: Callable, visit: Callable, *, axis: str,
          turns: Callable | None = None) -> tuple[jax.Array, ...]:
    """A persistent kernel's tiles (`Family.persist`'s): the grid `tiles` spread over the launch's programs
    along `axis`, each tile's `schedule(index)`, `(active, steps, views, tile)`, visited as `visit(counter,
    active, steps, views, tile, done)` at the step its streams are at across the tiles (`counter`: the steps
    of the active tiles before it) and its turn round accumulators (`done`: the turns the active tiles before
    it took, one a tile, or `turns(steps)` where given: an exactly summed tile's windows). Returns both past
    the last tile."""
    @plgpu.nd_loop(tiles, collective_axes=axis, init_carry=(jnp.int32(0), jnp.int32(0)))
    def ended(loop: plgpu.NDLoopInfo, carry):  # nd_loop passes its carry by this name
        counter, done = carry
        active, steps, views, tile = schedule(loop.index)
        visit(counter, active, steps, views, tile, done)
        taken = 1 if turns is None else turns(steps)
        return counter + jnp.where(active, steps, 0), done + jnp.where(active, taken, 0)

    return ended


def consumed(family: Family, tiles: tuple[int, ...], schedule: Callable, issue: Callable, finish: Callable,
             store: Callable, index, walking: Callable, *, axis: str, turns=None) -> jax.Array:
    """Consumer role `index`'s tiles of a persistent kernel (`tiled`'s), issued and finished in turn
    (`Family.persist`'s `issue` and `finish`): each tile's accumulators `issue(index, tile, walk,
    active=...)`, `walk` its steps' (`walking(counter, active, steps, views)`, built where the tile is
    visited) where it streams any, a walk of none (`unwalked`) where not, then `finish(index, tile, accs,
    store, active=...)`. Returns the counter past the last tile.

    With `turns` (`Family.meeting`'s two barriers of two consumers' arrivals: ping-pong, `persist`'s
    `pingpong`), each tile is its owner's alone, the active tiles' turn counted (`done`) alternating between
    the two consumers, so each walks its own tiles at the shared counter and passes over its partner's. Its
    products take the tensor cores in turn: an owner arrives at its own barrier and waits for its partner's
    arrival there, issues the tile, then arrives at its partner's barrier, and finishes the tile while its
    partner issues the next (FlashAttention-3's alternation; consumer 1 arrives once at consumer 0's
    barrier before its first tile, so consumer 0 goes first). No arrival overruns a phase: a consumer's next
    arrival at a barrier comes after its own wait at its other barrier, which needs its partner's arrival
    there, which comes only after the partner's wait at this barrier completed the phase before. Past the
    last tile, the consumer the next tile would be its arrives at its own barrier once more and waits there,
    completing and observing the phase its partner's last arrival opened (consumer 1's first, where no tile
    ran), so no barrier ends with an arrival pending or a phase its waiter has not seen."""
    if turns is not None:
        pl.when(index == 1)(lambda: family.turn(turns.at[0], 2, wait=False))

    def visit(counter, active, steps, views, tile, done):
        walk = walking(counter, active, steps, views)

        def run(walk, *, active: bool):
            finish(index, tile, issue(index, tile, walk, active=active), store, active=active)

        if turns is None:
            pl.when(active)(lambda: run(walk, active=True))
            pl.when(~active)(lambda: run(unwalked(family), active=False))
            return
        mine, alternating = done % 2 == index, turns

        def owned():
            family.turn(alternating.at[index], 2)
            accumulators = issue(index, tile, walk, active=True)
            family.turn(alternating.at[1 - index], 2, wait=False)
            finish(index, tile, accumulators, store, active=True)

        pl.when(active & mine)(owned)
        pl.when(~active & mine)(lambda: run(unwalked(family), active=False))

    counter, done = tiled(tiles, schedule, visit, axis=axis)
    if turns is not None:
        pl.when(done % 2 == index)(lambda: family.turn(turns.at[index], 2))
    return counter
