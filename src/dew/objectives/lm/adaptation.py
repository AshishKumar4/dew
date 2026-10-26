"""Test-time training over the task loss: a continuum level stepped in context, a window at a time.

Nested Learning's continuum memory (Behrouz et al., arXiv 2512.24695, section 7.1, equation 71)
steps a block's parameters every `C` tokens by the gradient of the model's own objective over them:
theta_{i+1} = theta_i - eta sum_{t in the window} grad L(theta_i; x_t), at each window's end, the
window's tokens read at the theta it opened with. Equation 71 sums from `i - C` to `i`, `C + 1` terms,
so two windows would share their boundary token; here each token is counted once, in its own window,
as `dew.optim.scale_by_m3` counts its chunks.

The level is `dew.nn.continuum.Continuum`, read on a decoder's final states before its head
(`CausalTransformer.continuum`), so each token's next-token loss reads the level through its own state
alone and a window's gradient is its tokens' own, equation 71's per-token term exactly. The trunk runs
once, and each window's states go through the level at that window's parameters and the model's own
head (`continuum_states`, `head_logits`); the next-token losses there are the objective's, and their
gradient, weighted as the objective weights its targets, steps the level for the next window. The
outer loss differentiates through every step, the second-order meta-gradient of the level's start.

Each row adapts by itself, and each document of a packed row by itself: a run of a document's
real tokens, padding inside it read through and a document number used again after another's a run
of its own, counts its windows from its first real token, and its first window reads the level's
start, so a packed document reads as it would alone. A padding slot weighs nothing and belongs to the
window open at it. The row is cut into blocks of `C` slots (`sequence_blocks`), which a run's windows
straddle where its start or padding falls inside one: a block's slots read the window open from the
block before, the run's next window, or a new run's first, so the scan carries one window per row.
A row with no padding and one document has every window a block, and its blocks read one level.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.nn.inputs import sequence_blocks
from dew.nn.protocols import SteppedInContext


@dataclasses.dataclass(frozen=True)
class ChunkAdaptation:
    """How an objective steps a model's continuum level in context (`LMObjective(adaptation=...)`).

    `window` is equation 71's `C`, the tokens of a document between two steps; `rate` is its `eta`,
    each token's gradient weighted by it and by its target's weight, summed over the window, not
    averaged; `select` names the level's parts that step (`down`, `up`), every part where empty, the
    others held at their trained values.
    """

    window: int = 16
    rate: float = 0.01
    select: tuple[str, ...] = ()

    def __post_init__(self):
        if self.window < 1:
            raise ValueError(f"a window holds at least one token, not {self.window}")
        if not math.isfinite(self.rate) or self.rate < 0:
            raise ValueError(f"a window's step takes a finite rate of at least zero, not {self.rate}")
        object.__setattr__(self, "select", tuple(self.select))


@dataclasses.dataclass(frozen=True)
class Adapted:
    """What the adapted head scores a row's targets: each target's next-token `losses`, its top-1
    column where asked (`predicted`), its log partition (`log_z`), and the whole logits where asked."""

    losses: jax.Array
    predicted: jax.Array | None
    log_z: jax.Array
    logits: jax.Array | None


def _stepped_mask(level: Mapping, select: tuple[str, ...]) -> Mapping:
    """Whether each leaf of the level's parameters steps in context (`ChunkAdaptation.select`)."""
    unknown = sorted(set(select) - set(level))
    if unknown:
        raise ValueError(f"the continuum level has no part {unknown}; it has {sorted(level)}")
    return {name: jax.tree.map(lambda _, name=name: not select or name in select, part)
            for name, part in level.items()}


def window_keys(valid, documents, rows: int, length: int, window: int) -> tuple[jax.Array, jax.Array]:
    """Each slot's window as `(run, window)` `[B, S]` each: the run of its document's real tokens it
    belongs to, numbered from 1 along the row (a padding slot the run before it, 0 before any), and
    its window within the run, its real tokens counted from the run's first (-1 before any)."""
    real = jnp.ones((rows, length), bool) if valid is None else jnp.asarray(valid, bool)
    documents = (jnp.zeros((rows, length), jnp.int32) if documents is None
                 else jnp.asarray(documents, jnp.int32))
    index = jnp.broadcast_to(jnp.arange(length), (rows, length))
    # The last real slot before each slot, -1 before any.
    last = jax.lax.cummax(jnp.where(real, index, -1), axis=1)
    before = jnp.concatenate([jnp.full((rows, 1), -1), last[:, :-1]], axis=1)
    previous = jnp.take_along_axis(documents, jnp.maximum(before, 0), axis=1)
    starts = real & ((before < 0) | (previous != documents))
    runs = jnp.cumsum(starts, axis=1)
    counted = jnp.cumsum(real, axis=1)
    opened = jax.lax.cummax(jnp.where(starts, counted - 1, -1), axis=1)
    return runs, jnp.where(runs > 0, (counted - 1 - opened) // window, -1)


def _rows(tree, rows: int):
    """Each leaf of `tree` once per row."""
    return jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (rows, *leaf.shape)), tree)


def _where(choice: jax.Array, first, second):
    """Per row, `first`'s leaf where `choice` holds and `second`'s otherwise."""
    return jax.tree.map(lambda one, other: jnp.where(choice.reshape(-1, *[1] * (one.ndim - 1)), one, other),
                        first, second)


def adapted_scores(model: nn.Module, variables, hidden, targets, weights, layout,
                   adaptation: ChunkAdaptation, *, predict: bool, logits: bool) -> Adapted:
    """A row's targets scored through its continuum level stepped in context (module docstring):
    `hidden` the final states `[B, S, features]`, `targets` `[B, S]`, `weights` each target's weight in
    the loss, and `layout` the rows' validity and documents (`dew.nn.inputs.token_layout`). The model
    owns the level's place in its variables (`SteppedInContext`)."""
    from dew.objectives.lm.chunked import logits_cross_entropy

    start = model.in_context_level(variables) if isinstance(model, SteppedInContext) else None
    if start is None:
        raise ValueError("the model has no continuum level to step in context")
    steps = _stepped_mask(start, adaptation.select)
    rows, length = targets.shape
    window = adaptation.window
    valid, documents = layout
    runs, windows = window_keys(valid, documents, rows, length, window)
    blocks = -(-length // window)
    # The last block's slots past the row weigh nothing and belong to the row's last window.
    edge = ((0, 0), (0, blocks * window - length))
    inputs = (*(sequence_blocks(value, window, leading=True) for value in (hidden, targets, weights)),
              *(sequence_blocks(jnp.pad(key, edge, mode="edge"), window, leading=True)
                for key in (runs, windows)))
    # Only padding or several documents let a window open inside a block; else every window is one.
    straddles = valid is not None or documents is not None

    def level_states(level, states) -> jax.Array:
        return jnp.asarray(model.apply(model.with_in_context_level(variables, level), states,
                                       method="continuum_states"))

    def scored(levels, chosen, states, block_targets):
        """Each slot's losses, its states read at its own of `levels` (`chosen`, per slot)."""
        read = [jax.vmap(level_states)(level, states) for level in levels]
        passed = read[0]
        for index, alternative in enumerate(read[1:], start=1):
            passed = jnp.where((chosen == index)[..., None], alternative, passed)
        block_logits = jnp.asarray(model.apply(variables, passed, method="head_logits"))
        losses, top, log_z = logits_cross_entropy(block_logits, block_targets, predict=predict)
        return losses, (top, log_z, block_logits)

    def stepped(level, gradient):
        return jax.tree.map(lambda leaf, grad, moves: leaf - adaptation.rate * grad if moves else leaf,
                            level, gradient, steps)

    started = _rows(start, rows)

    def block(carry, block_inputs):
        level, partial, run, place = carry
        states, block_targets, block_weights, block_runs, block_windows = block_inputs
        same = block_runs == run[:, None]
        tail = same & (block_windows == place[:, None])
        following = same & (block_windows == place[:, None] + 1)
        if straddles:
            # The open window's tail, read at the window's own level, finishes its gradient.
            def tail_loss(at):
                losses, _ = scored((at,), jnp.zeros_like(tail, jnp.int32), states, block_targets)
                return jnp.sum(jnp.where(tail, block_weights * losses, 0.0))
            closed = jax.tree.map(jnp.add, partial, jax.grad(tail_loss)(level))
            levels = (level, stepped(level, closed), started)
            chosen = jnp.where(tail, 0, jnp.where(following, 1, 2))
        else:
            closed = partial
            levels = (stepped(level, closed),)
            chosen = jnp.zeros_like(tail, jnp.int32)
        last_run, last_place = block_runs[:, -1], block_windows[:, -1]
        last = (block_runs == last_run[:, None]) & (block_windows == last_place[:, None])

        def last_loss(at):
            losses, held = scored(at, chosen, states, block_targets)
            return jnp.sum(jnp.where(last, block_weights * losses, 0.0)), (losses, held)

        (_, (losses, (top, log_z, block_logits))), gradients = jax.value_and_grad(
            last_loss, has_aux=True)(levels)
        # The last slot's window, one level's slots, takes its gradient from that level alone.
        reached = gradients[0]
        for gradient in gradients[1:]:
            reached = jax.tree.map(jnp.add, reached, gradient)
        opened = levels[0]
        for index, alternative in enumerate(levels[1:], start=1):
            opened = _where(chosen[:, -1] == index, alternative, opened)
        continuing = (last_run == run) & (last_place == place)
        carried = (opened, _where(continuing, closed, reached), last_run, last_place)
        return carried, (losses, top, log_z, block_logits if logits else None)

    zeros = jax.tree.map(jnp.zeros_like, started)
    # Before a row's first real token no run is open: run 0, window -1.
    first = (started, zeros, jnp.zeros((rows,), runs.dtype), jnp.full((rows,), -1, windows.dtype))
    _, (losses, top, log_z, whole) = jax.lax.scan(jax.checkpoint(block), first, inputs)

    def joined(value: jax.Array) -> jax.Array:
        """A scan's `[blocks, B, window, ...]` output as the row's `[B, S, ...]`."""
        return jnp.moveaxis(value, 0, 1).reshape(rows, blocks * window, *value.shape[3:])[:, :length]

    return Adapted(joined(losses), None if top is None else joined(top), joined(log_z),
                   None if whole is None else joined(whole))
