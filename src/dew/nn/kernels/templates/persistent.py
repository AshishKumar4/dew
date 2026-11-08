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
             store: Callable, index, walking: Callable, *, axis: str) -> jax.Array:
    """Consumer role `index`'s tiles of a persistent kernel (`tiled`'s), issued and finished in turn
    (`Family.persist`'s `issue` and `finish`): each tile's accumulators `issue(index, tile, walk,
    active=...)`, `walk` its steps' (`walking(counter, active, steps, views)`, built where the tile is
    visited) where it streams any, a walk of none (`unwalked`) where not, then `finish(index, tile, accs,
    store, active=...)`. Returns the counter past the last tile."""
    def visit(counter, active, steps, views, tile, done):
        del done
        walk = walking(counter, active, steps, views)

        def run(walk, *, active: bool):
            finish(index, tile, issue(index, tile, walk, active=active), store, active=active)

        pl.when(active)(lambda: run(walk, active=True))
        pl.when(~active)(lambda: run(unwalked(family), active=False))

    return tiled(tiles, schedule, visit, axis=axis)[0]
