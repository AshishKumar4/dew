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

from dew.batch import document_ranks, document_starts, prefix_sum, real_slots, token_layout
from dew.nn.inputs import sequence_blocks
from dew.nn.precision import at_least_fp32
from dew.nn.protocols import SteppedInContext


@dataclasses.dataclass(frozen=True)
class ChunkAdaptation:
    """How an objective steps a model's continuum level in context (`LMObjective(adaptation=...)`).

    `window` is equation 71's `C`, the tokens of a document between two steps; `rate` is its `eta`,
    each token's gradient weighted by it and by its target's weight, summed over the window, not
    averaged; `select` names the level's parts that step (`down`, `up`), every part where empty, the
    others held at their trained values.

    `damped` steps the selected parts' kernels alone (`up`'s, `down`'s, or both, every part where
    `select` is empty; an adapter's factors on them hold) at rate / (1 + rate L), L a bound on the curvature
    of the window's loss in them along the step (`_damping`): rate times L stays below 1, so a step never
    overshoots, whatever the rate. The level reads s = x + U f, f = act(D x); with Ls and Gs the head's
    bounds on a slot loss's curvature and slope in s (`SteppedInContext.head_bounds`), and p1, p2 the
    activation's on |act'| and |act''| (`level_terms`), a slot's curvature is at most
    Ls (|U|^2 p1^2 |x|^2 [D steps] + |f|^2 [U steps]) + Gs (|U| p2 |x|^2 [D steps] + p1 |x| [both step]),
    |U| and |f| taken at their largest along the window's step (`stepped`), summed over the window by its
    weights, x and f as `down` and `up` multiply them (a checkpoint's quantizer of their inputs applied),
    L a constant of the step, not differentiated. Under an affine head the loss is convex in `up`, and a
    step below 1 / L of a convex loss is non-expansive, so stepping `up` alone moves two nearby levels no
    further apart, whatever the rate: their rounding grows at most by what each window adds. A softcap,
    a head transform or a step of `down` leaves the loss non-convex, where a step below 1 / L expands two
    levels' distance by at most 1 + rate L, under 2, a window, not by the rate times curvature past 2 of
    an undamped step, whose rounding grows by 10^4 over three windows of 0.5 N(0, 1) weights at a rate
    of a half (ledger 1714).
    An activation whose act'' no number bounds (ReLU's, a point mass) bounds no step of `down`: refused,
    as are a step of a kernel its product rounds and a step of `down` through an `up` that rounds its
    input or kernel (`Continuum._step_terms`), the loss piecewise constant in what steps.
    """

    window: int = 16
    rate: float = 0.01
    select: tuple[str, ...] = ()
    damped: bool = False

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


def _stepped_mask(level: Mapping, adaptation: ChunkAdaptation) -> Mapping:
    """Whether each leaf of the level's parameters steps in context (`ChunkAdaptation.select`), in the
    level's own container (a dict or a `FrozenDict`), as every tree the scan maps beside it is: a
    damped step's selected parts' kernels alone (`_damped_parts`), any adapter's factors beside them held."""
    select = adaptation.select
    unknown = sorted(set(select) - set(level))
    if unknown:
        raise ValueError(f"the continuum level has no part {unknown}; it has {sorted(level)}")
    parts = _damped_parts(level, adaptation)

    def keys(path) -> tuple[str, ...]:
        return tuple(str(part.key) if isinstance(part, jax.tree_util.DictKey) else str(part) for part in path)

    def stepped(path, _) -> bool:
        if adaptation.damped:
            return keys(path) in {(part, "kernel") for part in parts}
        return not select or keys(path)[0] in select
    return jax.tree_util.tree_map_with_path(stepped, level)


def _damped_parts(level: Mapping, adaptation: ChunkAdaptation) -> tuple[str, ...]:
    """The parts whose kernels a damped step steps (`ChunkAdaptation.damped`): the selected ones of the
    level's `down` and `up`, both where `select` is empty; none for an undamped step."""
    if not adaptation.damped:
        return ()
    parts = tuple(part for part in ("down", "up") if not adaptation.select or part in adaptation.select)
    if not parts:
        raise ValueError(f"a damped step steps the level's `down` or `up` kernel, not {adaptation.select}")
    for part in parts:
        if "kernel" not in level[part]:
            raise ValueError(f"a damped step steps the level's `{part}` kernel, and this level's `{part}` "
                             f"holds none of its own ({sorted(level[part])}): a frozen or quantized one is "
                             "the model's")
    return parts


def window_keys(valid, documents, rows: int, length: int, window: int) -> tuple[jax.Array, jax.Array]:
    """Each slot's window as `(run, window)` `[B, S]` each: the run of its document's real tokens it
    belongs to, numbered from 1 along the row (a padding slot the run before it, 0 before any), and
    its window within the run, its document rank over `window` (`document_starts`,
    `document_ranks`, the row's readers' own; -1 before any)."""
    real = jnp.ones((rows, length), bool) if valid is None else jnp.asarray(valid, bool)
    documents = (jnp.zeros((rows, length), jnp.int32) if documents is None
                 else jnp.asarray(documents, jnp.int32))
    starts = document_starts(documents, real)
    # A row's first real token opens its first run, which `document_starts` leaves to what precedes.
    runs = prefix_sum(starts | (real & (prefix_sum(real, dtype=jnp.int32) == 1)), dtype=jnp.int32)
    return jnp.asarray(runs), jnp.where(runs > 0, document_ranks(real, starts) // window, -1)


def _distance(leaf: jax.Array, start: jax.Array) -> jax.Array:
    """Per row, the Frobenius norm of `leaf` `[B, ...]` less `start` `[B, ...]`, in at least fp32."""
    moved = (leaf - start).astype(at_least_fp32(leaf.dtype))
    return jnp.sqrt(jnp.sum(jnp.square(moved), axis=tuple(range(1, moved.ndim))))


def _rows(tree, rows: int):
    """Each leaf of `tree` once per row."""
    return jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (rows, *leaf.shape)), tree)


def _where(choice: jax.Array, first, second):
    """Per row, `first`'s leaf where `choice` holds and `second`'s otherwise."""
    return jax.tree.map(lambda one, other: jnp.where(choice.reshape(-1, *[1] * (one.ndim - 1)), one, other),
                        first, second)


_TERMS = 4
"""What each slot adds to its window's damping (`_damping`): weighted |x|^2, |f|^2, |f| |x| and |x|."""


def _damping(model: nn.Module, variables, start, hidden, weights, adaptation: ChunkAdaptation):
    """A damped step's bounds (`ChunkAdaptation.damped`): the head's on a slot loss's curvature and slope
    in the states (`SteppedInContext.head_bounds`), the norm of the level's start's output map and its
    activation's p1, p2, and each slot's weighted terms `[B, S, _TERMS]` at the level's start
    (`level_terms`, of the parts that step, which refuses what no curvature bounds), whose window sums
    set its rate (`_step_rate`); zeros for an undamped step."""
    if not adaptation.damped:
        zero = jnp.zeros((), jnp.float32)
        return zero, zero, zero, zero, zero, jnp.zeros((*weights.shape, _TERMS), jnp.float32)
    parts = _damped_parts(start, adaptation)
    curvature, slope = _arrays(model.apply(variables, method="head_bounds"))
    held = model.with_in_context_level(variables, start)
    squares, inputs, norm, p1, p2 = _arrays(model.apply(held, hidden, parts, method="level_terms"))
    lengths, fed = jnp.sqrt(squares), jnp.sqrt(inputs)
    terms = jnp.stack([squares, inputs, fed * lengths, lengths], axis=-1)
    return jax.lax.stop_gradient((curvature, slope, norm, p1, p2, weights[..., None] * terms))


def _arrays(applied) -> tuple[jax.Array, ...]:
    """A method's tuple of arrays, as `nn.Module.apply` returns it untyped."""
    return tuple(jnp.asarray(value) for value in applied)


def _step_rate(adaptation: ChunkAdaptation, parts: tuple[str, ...], bounds, sums, moved, gradient):
    """A damped window's rate per row, rate / (1 + rate L), L its loss's curvature bound along the step
    (`ChunkAdaptation.damped`): the window's `sums` `[B, _TERMS]` of its slots' terms, `bounds` the head's
    curvature and slope, the start's output-map norm and its activation's p1, p2 (`_damping`), `moved` per
    row the Frobenius norms `down` and `up` have moved from the start before the window, and `gradient`
    the window's. A part's largest norm along the step is its start's plus what it moved and what this
    step moves it at most, the rate times its gradient's norm. The rate is a constant of the step, as its
    bound is: the outer gradient flows through the step's gradient, not through the bound it reads (whose
    distances are zero at a run's start, where their root has no derivative)."""
    curvature, slope, norm, p1, p2 = bounds
    squares, inputs, crossed, lengths = (sums[:, index] for index in range(_TERMS))
    rate = adaptation.rate

    def reach(part: str) -> jax.Array:
        if part not in parts:
            return jnp.zeros_like(squares)
        leaf = gradient[part]["kernel"]
        step = jnp.sqrt(jnp.sum(jnp.square(leaf.astype(jnp.float32)), axis=tuple(range(1, leaf.ndim))))
        return moved[part] + rate * step

    # D's distance along the step bounds how far f = act(D x) moves, p1 |x| per unit; U's, its norm.
    spread, output = reach("down"), norm + reach("up")
    bend = jnp.zeros_like(squares)
    stretch = jnp.zeros_like(squares)
    if "down" in parts:
        stretch = stretch + output ** 2 * p1 ** 2 * squares
        bend = bend + output * p2 * squares
    if "up" in parts:
        stretch = stretch + inputs + 2 * p1 * spread * crossed + p1 ** 2 * spread ** 2 * squares
        if "down" in parts:
            bend = bend + p1 * lengths
    return jax.lax.stop_gradient(rate / (1 + rate * (curvature * stretch + slope * bend)))


def adapted_scores(model: nn.Module, variables, hidden, targets, weights, fields,
                   adaptation: ChunkAdaptation, *, predict: bool, logits: bool) -> Adapted:
    """A row's targets scored through its continuum level stepped in context (module docstring):
    `hidden` the final states `[B, S, features]`, `targets` `[B, S]`, `weights` each target's weight in
    the loss, and `fields` the rows' token fields, their documents and real slots read as every
    reader of rows reads them (`token_layout`, `real_slots`: a validity mask, and a packed row's
    document 0, are padding). A row's places (`positions`) move no window: each run counts its own.
    The model owns the level's place in its variables (`SteppedInContext`)."""
    from dew.objectives.lm.chunked import logits_cross_entropy

    start = model.in_context_level(variables) if isinstance(model, SteppedInContext) else None
    if start is None:
        raise ValueError("the model has no continuum level to step in context")
    steps = _stepped_mask(start, adaptation)
    rows, length = targets.shape
    window = adaptation.window
    straddles, weights, runs, windows = _laid_out(fields, weights, rows, length, adaptation)
    *bounds, energies = _damping(model, variables, start, hidden, weights, adaptation)
    parts = _damped_parts(start, adaptation)
    blocks = -(-length // window)
    # The last block's slots past the row weigh nothing and belong to the row's last window.
    edge = ((0, 0), (0, blocks * window - length))
    inputs = (*(sequence_blocks(value, window, leading=True)
                for value in (hidden, targets, weights, energies)),
              *(sequence_blocks(jnp.pad(key, edge, mode="edge"), window, leading=True)
                for key in (runs, windows)))
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

    def stepped(level, gradient, energy):
        """`level` stepped by its window's `gradient`, at the rate a damped step takes from the
        window's summed terms `energy` and how far its kernels moved from the start (`_step_rate`),
        one rate a row."""
        rate = None
        if adaptation.damped:
            moved = {part: _distance(level[part]["kernel"], started[part]["kernel"]) for part in parts}
            rate = _step_rate(adaptation, parts, bounds, energy, moved, gradient)

        def step(leaf, grad, moves):
            if not moves:
                return leaf
            if rate is None:
                return leaf - adaptation.rate * grad
            return leaf - rate.reshape(-1, *[1] * (leaf.ndim - 1)).astype(leaf.dtype) * grad
        return jax.tree.map(step, level, gradient, steps)

    started = _rows(start, rows)

    def block(carry, block_inputs):
        level, partial, energy, run, place = carry
        states, block_targets, block_weights, block_energies, block_runs, block_windows = block_inputs
        same = block_runs == run[:, None]
        tail = same & (block_windows == place[:, None])
        following = same & (block_windows == place[:, None] + 1)
        if straddles:
            # The open window's tail, read at the window's own level, finishes its gradient.
            def tail_loss(at):
                losses, _ = scored((at,), jnp.zeros_like(tail, jnp.int32), states, block_targets)
                return jnp.sum(jnp.where(tail, block_weights * losses, 0.0))
            closed = jax.tree.map(jnp.add, partial, jax.grad(tail_loss)(level))
            closed_energy = energy + jnp.sum(jnp.where(tail[..., None], block_energies, 0), axis=1)
            levels = (level, stepped(level, closed, closed_energy), started)
            chosen = jnp.where(tail, 0, jnp.where(following, 1, 2))
        else:
            closed, closed_energy = partial, energy
            levels = (stepped(level, closed, closed_energy),)
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
        reached_energy = jnp.sum(jnp.where(last[..., None], block_energies, 0), axis=1)
        carried = (opened, _where(continuing, closed, reached),
                   jnp.where(continuing[:, None], closed_energy, reached_energy), last_run, last_place)
        return carried, (losses, top, log_z, block_logits if logits else None)

    zeros = jax.tree.map(jnp.zeros_like, started)
    # Before a row's first real token no run is open: run 0, window -1.
    first = (started, zeros, jnp.zeros((rows, _TERMS), energies.dtype), jnp.zeros((rows,), runs.dtype),
             jnp.full((rows,), -1, windows.dtype))
    _, (losses, top, log_z, whole) = jax.lax.scan(
        jax.checkpoint(block), first, inputs)

    def joined(value: jax.Array) -> jax.Array:
        """A scan's `[blocks, B, window, ...]` output as the row's `[B, S, ...]`."""
        return jnp.moveaxis(value, 0, 1).reshape(rows, blocks * window, *value.shape[3:])[:, :length]

    return Adapted(joined(losses), None if top is None else joined(top), joined(log_z),
                   None if whole is None else joined(whole))


def _laid_out(fields, weights, rows: int, length: int, adaptation: ChunkAdaptation):
    """Where the rows' windows run (`window_keys`): whether a window can open inside a block (padding
    or several documents in a row), the weights with each padding slot's at nothing, and each slot's run
    and window."""
    marked, documents = token_layout(fields)
    straddles = marked is not None or documents is not None
    valid = real_slots(fields, (rows, length)) if straddles else None
    if valid is not None:
        weights = jnp.where(valid, weights, 0)
    runs, windows = window_keys(valid, documents, rows, length, adaptation.window)
    return straddles, weights, runs, windows
