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

Each row adapts by itself, and each document of a packed row by itself: its windows are counted from
its first token (`counted_places`), and its first window reads the level's start, so a packed
document reads as it would alone. The row is cut into blocks of `C` slots, which a document's windows
straddle where its start falls inside one: a block's slots read the window open from the block before,
the document's next window, or a new document's first, so the scan carries one window per row.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from dew.nn.inputs import counted_places


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


def _blocks(value: jax.Array, window: int, blocks: int, fill) -> jax.Array:
    """`value` `[B, S, ...]` padded with `fill` to `blocks * window` slots, as `[blocks, B, window, ...]`."""
    padding = blocks * window - value.shape[1]
    padded = jnp.pad(value, [(0, 0), (0, padding)] + [(0, 0)] * (value.ndim - 2), constant_values=fill)
    return jnp.moveaxis(padded.reshape(value.shape[0], blocks, window, *value.shape[2:]), 1, 0)


def _rows(tree, rows: int):
    """Each leaf of `tree` once per row."""
    return jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (rows, *leaf.shape)), tree)


def _where(choice: jax.Array, first, second):
    """Per row, `first`'s leaf where `choice` holds and `second`'s otherwise."""
    return jax.tree.map(lambda one, other: jnp.where(choice.reshape(-1, *[1] * (one.ndim - 1)), one, other),
                        first, second)


def adapted_scores(model, variables, hidden, targets, weights, layout, adaptation: ChunkAdaptation, *,
                   predict: bool, logits: bool) -> Adapted:
    """A row's targets scored through its continuum level stepped in context (module docstring):
    `hidden` the final states `[B, S, features]`, `targets` `[B, S]`, `weights` each target's weight in
    the loss, and `layout` the rows' validity and documents (`dew.nn.inputs.token_layout`)."""
    from dew.objectives.lm.chunked import logits_cross_entropy

    params = variables["params"]
    start = params["continuum"]
    steps = _stepped_mask(start, adaptation.select)
    rows, length = targets.shape
    window = adaptation.window
    blocks = -(-length // window)
    valid, documents = layout
    places = counted_places(segments=documents, valid=valid)
    places = jnp.broadcast_to(jnp.arange(length), (rows, length)) if places is None else places
    documents = (jnp.zeros((rows, length), jnp.int32) if documents is None
                 else jnp.asarray(documents, jnp.int32))
    # A block's padding slots are a document of their own, which no real one numbers, and weigh nothing.
    alone = jnp.min(documents) - 1
    inputs = (_blocks(hidden, window, blocks, 0), _blocks(targets, window, blocks, 0),
              _blocks(weights, window, blocks, 0), _blocks(documents, window, blocks, alone),
              _blocks(places // window, window, blocks, 0))
    # Whether a document's window can open inside a block: only where a row holds several documents or
    # padding; else every window is a block, and the next window is the block's every slot.
    straddles = layout[1] is not None or layout[0] is not None

    def with_level(level):
        return {**variables, "params": {**params, "continuum": level}}

    def level_states(level, states):
        return model.apply(with_level(level), states, method="continuum_states")

    def scored(levels, chosen, states, block_targets):
        """Each slot's losses, its states read at its own of `levels` (`chosen`, per slot)."""
        read = [jax.vmap(level_states)(level, states) for level in levels]
        passed = read[0]
        for index, alternative in enumerate(read[1:], start=1):
            passed = jnp.where((chosen == index)[..., None], alternative, passed)
        block_logits = model.apply(variables, passed, method="head_logits")
        losses, top, log_z = logits_cross_entropy(block_logits, block_targets, predict=predict)
        return losses, (top, log_z, block_logits)

    def stepped(level, gradient):
        return jax.tree.map(lambda leaf, grad, moves: leaf - adaptation.rate * grad if moves else leaf,
                            level, gradient, steps)

    started = _rows(start, rows)

    def block(carry, block_inputs):
        level, partial, document, place = carry
        states, block_targets, block_weights, block_documents, block_places = block_inputs
        same = block_documents == document[:, None]
        tail = same & (block_places == place[:, None])
        following = same & (block_places == place[:, None] + 1)
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
        last_document, last_place = block_documents[:, -1], block_places[:, -1]
        last = (block_documents == last_document[:, None]) & (block_places == last_place[:, None])

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
        continuing = (last_document == document) & (last_place == place)
        carried = (opened, _where(continuing, closed, reached), last_document, last_place)
        return carried, (losses, top, log_z, block_logits if logits else None)

    zeros = jax.tree.map(jnp.zeros_like, started)
    first = (started, zeros, documents[:, 0], jnp.full((rows,), -1, places.dtype))
    _, (losses, top, log_z, whole) = jax.lax.scan(jax.checkpoint(block), first, inputs)

    def joined(value: jax.Array) -> jax.Array:
        """A scan's `[blocks, B, window, ...]` output as the row's `[B, S, ...]`."""
        return jnp.moveaxis(value, 0, 1).reshape(rows, blocks * window, *value.shape[3:])[:, :length]

    return Adapted(joined(losses), None if top is None else joined(top), joined(log_z),
                   None if whole is None else joined(whole))
