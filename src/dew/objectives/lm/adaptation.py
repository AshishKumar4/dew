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

A run that carries its level (`ChunkAdaptation.carry`, "run") keeps what its rows' steps moved it by:
each row's documents step one level, its first window reading the level the run carried to it, and
each committed update carries the mean over its rows of how far their windows moved the level, every
window's step applied, into the `CARRIED` collection, which the run saves and resumes as it does its
parameters. The next update's rows start from the trained level plus what is carried.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.batch import document_ranks, document_starts, prefix_sum, real_slots, token_layout
from dew.nn.continuum import HeadTerms, LevelTerms
from dew.nn.inputs import sequence_blocks
from dew.nn.precision import at_least_fp32
from dew.nn.protocols import SteppedInContext


@dataclasses.dataclass(frozen=True)
class ChunkAdaptation:
    """How an objective steps a model's continuum level in context (`LMObjective(adaptation=...)`).

    `window` is equation 71's `C`, the tokens of a document between two steps; `rate` is its `eta`,
    each token's gradient weighted by it and by its target's weight, summed over the window, not
    averaged; `select` names the level's parts that step (`down`, `up`), every part where empty, the
    others held at their trained values. A chain of levels (`dew.nn.continuum`, section 7.1's
    multi-frequency memory) takes one window and rate a level, in its order, or one for every level:
    its windows, ordered, each divide the next, so a slower level's boundaries fall on a faster one's.

    `damped` steps the selected parts' kernels alone (`up`'s, `down`'s, or both, every part where
    `select` is empty; an adapter's factors on them hold) at a rate d with d <= rate / (1 + rate L(d)),
    L(d) a bound on the curvature of the window's loss in them along a step of rate d (`_step_rate`, which
    settles d toward the largest such): d L(d) stays below 1, so a step never overshoots, whatever the
    rate. The level reads s = x + U f, f = act(D x); with Ls and Gs the head's bounds on a slot loss's
    curvature and slope in s (`SteppedInContext.head_bounds`, a head transform's read over the states
    the step can reach from the window's), and p1, p2 the activation's on |act'| and |act''|
    (`level_terms`), a slot's curvature is at most
    Ls (|U|^2 p1^2 |x|^2 [D steps] + |f|^2 [U steps]) + Gs (|U| p2 |x|^2 [D steps] + p1 |x| [both step]),
    |U| and |f| taken at their largest along the window's step, summed over the window by its weights, x,
    f and every kernel as the products multiply them (a checkpoint's quantizer of their inputs applied,
    Qwix's rounding bounded, `dew.quantization.rounding_growth`), L a constant of the step, not
    differentiated. Where Qwix quantizes a product, a step follows its straight-through gradient: the
    bound holds for the surrogate's smooth part, and a step that crosses a rounding cell adds the
    quantization's own jump. Under an affine head the loss is convex in `up`, and a step below 1 / L of a
    convex loss is non-expansive, so stepping `up` alone moves two nearby levels no further apart,
    whatever the rate: their rounding grows at most by what each window adds. A softcap, a head
    transform or a step of `down` leaves the loss non-convex, where a step below 1 / L expands two levels'
    distance by at most 1 + rate L, under 2, a window, not by the rate times curvature past 2 of an
    undamped step, whose rounding grows by 10^4 over three windows of 0.5 N(0, 1) weights at a rate of a
    half (ledger 1714). An activation whose act'' no number bounds (ReLU's, a point mass) bounds no step
    of `down`, a head map or an `up` a method interceptor computes other than as a linear map (a
    checkpoint's quantizer of a head map's input, a frozen FP8 product, an adapter's dropout) leaves the
    loss piecewise constant in what steps, and a head transform's norm that bounds no derivatives
    (`BoundedNorm`) and a Qwix weight calibration whose clip no bound reads bound nothing: each refused.

    `carry` says what a document's first window reads: "document", the level the run trains, each
    document afresh, Titans' and TTT's reset; "run", the level the run carried, each row's documents
    stepping one level in turn and each update carrying its rows' mean step on (module docstring),
    Nested Learning's continuum memory kept across documents and steps.

    `layers` names the decoder layers whose own feed-forward blocks are the levels instead, one window
    and rate a layer in its order or one for all (`dew.objectives.lm.stacking`, section 7.3's ad-hoc
    level stacking): a pretrained decoder's blocks stepped in context from their trained values, which
    continued pre-training trains through the steps; empty, the model's continuum levels.
    """

    window: int | tuple[int, ...] = 16
    rate: float | tuple[float, ...] = 0.01
    select: tuple[str, ...] = ()
    damped: bool = False
    carry: str = "document"
    layers: tuple[int, ...] = ()

    def __post_init__(self):
        windows = self.window if isinstance(self.window, tuple) else (self.window,)
        rates = self.rate if isinstance(self.rate, tuple) else (self.rate,)
        if any(window < 1 for window in windows):
            raise ValueError(f"a window holds at least one token, not {self.window}")
        if any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError(f"a window's step takes a finite rate of at least zero, not {self.rate}")
        ordered = sorted(windows)
        if any(slower % faster for faster, slower in itertools.pairwise(ordered)):
            raise ValueError(f"a chain's windows, ordered, each divide the next, so a slower level steps on "
                             f"a faster one's boundary: not {self.window}")
        if self.carry not in ("document", "run"):
            raise ValueError(f"a level carries across a 'document' or a 'run', not {self.carry!r}")
        object.__setattr__(self, "select", tuple(self.select))
        object.__setattr__(self, "layers", tuple(self.layers))
        if len(set(self.layers)) != len(self.layers):
            raise ValueError(f"each layer's feed-forward block is one level, not {self.layers}")

    def windows(self, levels: int) -> tuple[int, ...]:
        """Each of `levels` levels' window, a chain's in its order: `window`'s own, or one for all."""
        return _each(self.window, levels, "window")

    def rates(self, levels: int) -> tuple[float, ...]:
        """Each of `levels` levels' rate, a chain's in its order: `rate`'s own, or one for all."""
        return _each(self.rate, levels, "rate")


def _each(value, levels: int, name: str) -> tuple:
    """`value`, one for each of `levels` levels or one for all, as a tuple a level (`ChunkAdaptation`)."""
    if not isinstance(value, tuple):
        return (value,) * levels
    if len(value) != levels:
        raise ValueError(f"this model's continuum holds {levels} levels, and the adaptation names a {name} "
                         f"for {len(value)}: one a level, or one for all")
    return value


CARRIED = "carried"
"""The collection holding what a run carries of its level's in-context steps (`ChunkAdaptation.carry`),
the level's tree, zero where the run starts (`_carried_start`)."""


def _carried_start(level) -> Mapping:
    """What a run that carries its level holds at its start (`CARRIED`): nothing moved, the level's
    tree in zeros."""
    return jax.tree.map(jnp.zeros_like, level)


def _carried_update(carried, moved: Mapping) -> Mapping:
    """What a run carries after an update (`CARRIED`): `carried` plus the mean over its counted rows of
    how far their steps moved the level, `moved` the rows' summed moves (`moves`) and their count
    (`rows`), summed over the update's microbatches."""
    rows = jnp.maximum(moved["rows"], 1)
    return jax.tree.map(lambda held, move: held + (move / rows).astype(held.dtype), carried, moved["moves"])


@dataclasses.dataclass(frozen=True)
class Adapted:
    """What the adapted head scores a row's targets: each target's next-token `losses`, its top-1
    column where asked (`predicted`), its log partition (`log_z`), and the whole logits where asked;
    where the run carries its level, how far the rows' steps moved it (`moved`: the moves summed over the
    rows that weigh anything, `moves`, and their count, `rows`), every window's step applied."""

    losses: jax.Array
    predicted: jax.Array | None
    log_z: jax.Array
    logits: jax.Array | None
    moved: Mapping | None = None


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
    runs, ranks = _run_ranks(valid, documents, rows, length)
    return runs, jnp.where(runs > 0, ranks // window, -1)


def _run_ranks(valid, documents, rows: int, length: int) -> tuple[jax.Array, jax.Array]:
    """Each slot's run of its document's real tokens, numbered from 1 along the row (`window_keys`), and
    its rank within its document (`document_ranks`)."""
    real = jnp.ones((rows, length), bool) if valid is None else jnp.asarray(valid, bool)
    documents = (jnp.zeros((rows, length), jnp.int32) if documents is None
                 else jnp.asarray(documents, jnp.int32))
    starts = document_starts(documents, real)
    # A row's first real token opens its first run, which `document_starts` leaves to what precedes.
    runs = prefix_sum(starts | (real & (prefix_sum(real, dtype=jnp.int32) == 1)), dtype=jnp.int32)
    return jnp.asarray(runs), jnp.asarray(document_ranks(real, starts))


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


_SUMS, _MOST = 4, 5
"""What each slot adds to its window's damping (`_slot_terms`): its weighted |x|^2, |f|^2, |f| |x| and |x|,
summed over the window; beside them its head norm's input's spread (`HeadTerms.spread`), least over the
window's slots that weigh anything; and its |f|, |x|, the norm of the kernel `up` it read, and the
Jacobian and second-derivative bounds of the levels after it (`LevelTerms.jacobian`, `bend`), largest
over them (`_gathered`)."""

_WIDTH = _SUMS + 1 + _MOST


def _slot_terms(read: list, spread: jax.Array, weights: jax.Array) -> list:
    """Each level's slots' terms `[B, W, _WIDTH]` (`_SUMS`), `read` each level's `LevelTerms` at the states
    and versions its slots read, in the chain's order, `spread` the head's at the slots' final states and
    `weights` their targets'. The levels after a level compose: through `g` after `h`, a Jacobian of at
    most J_g J_h and a second derivative of at most H_g J_h^2 + J_g H_h. A slot that weighs nothing bounds
    nothing of its window's."""
    counted = weights > 0
    after, bent = jnp.ones_like(spread), jnp.zeros_like(spread)
    terms = []
    for level in reversed(read):
        lengths, fed = jnp.sqrt(level.squares), jnp.sqrt(level.inputs)
        terms.append(jnp.stack([
            *(weights * term for term in (level.squares, level.inputs, fed * lengths, lengths)),
            jnp.where(counted, spread, jnp.inf),
            *(jnp.where(counted, term, 0) for term in (fed, lengths, jnp.broadcast_to(level.up, spread.shape),
                                                      after, bent))], axis=-1))
        jacobian, bend = jnp.broadcast_to(level.jacobian(), spread.shape), jnp.broadcast_to(level.bend(),
                                                                                             spread.shape)
        after, bent = after * jacobian, bent * jnp.square(jacobian) + after * bend
    return jax.lax.stop_gradient(terms[::-1])


def _gathered(mask: jax.Array, terms: jax.Array) -> jax.Array:
    """A window's terms `[B, _WIDTH]` over the slots of `terms` `[B, W, _WIDTH]` that `mask` holds
    (`_SUMS`): the sums, the least spread and the largest of the rest."""
    sums = jnp.sum(jnp.where(mask[..., None], terms[..., :_SUMS], 0), axis=1)
    least = jnp.min(jnp.where(mask, terms[..., _SUMS], jnp.inf), axis=1)
    most = jnp.max(jnp.where(mask[..., None], terms[..., _SUMS + 1:], 0), axis=1)
    return jnp.concatenate([sums, least[:, None], most], axis=-1)


def _joined(one: jax.Array, other: jax.Array) -> jax.Array:
    """Two parts' window terms `[B, _WIDTH]` as one window's (`_gathered`)."""
    return jnp.concatenate([one[:, :_SUMS] + other[:, :_SUMS],
                            jnp.minimum(one[:, _SUMS:_SUMS + 1], other[:, _SUMS:_SUMS + 1]),
                            jnp.maximum(one[:, _SUMS + 1:], other[:, _SUMS + 1:])], axis=-1)


def _arrays(applied) -> tuple[jax.Array, ...]:
    """A method's tuple of arrays, as `nn.Module.apply` returns it untyped."""
    return tuple(jnp.asarray(value) for value in applied)


_SETTLINGS = 4
"""The rounds of T(T(d)) that settle a damped window's rate toward its bound's fixed point (`_step_rate`)."""


def _step_rate(rate: float, parts: tuple[str, ...], head: HeadTerms, level: LevelTerms, sums, gradient):
    """A damped window's rate per row (`ChunkAdaptation.damped`): a d with d <= T(d) = rate / (1 + rate
    L(d)), L(d) its loss's curvature bound along a step of rate d from the window's own level, so
    d L(d) < 1, settled toward T's fixed point. The window's `sums` `[B, _WIDTH]` of its slots' terms
    (`_gathered`: |x| and |f| as the level read them, its kernel `up`'s norm, the head norm's least
    input spread and the Jacobian J and second derivative H of the levels after it), `head` the head's
    bounds (`HeadTerms`), `level` the level's activation bounds and its kernels' rounding growth
    (`LevelTerms`), and `gradient` the window's set L(d).

    Along a step of rate d each part moves by d times its gradient's norm; a slot's state y = x + U f
    moves by at most |dU| |f| + |U| |df|, |df| at most p1 |x| |dD|, and the states the head reads by J
    times that, where the head's bounds are read (`HeadTerms.at`); the levels after this one bound the
    loss's curvature in y by J^2 Ls + Gs H and its slope by J Gs, Ls and Gs the head's. With them, a
    slot's curvature in the level's kernels is at most
    Ls (|U|^2 p1^2 |x|^2 [D steps] + |f|^2 [U steps]) + Gs (|U| p2 |x|^2 [D steps] + p1 |x| [both step]),
    |U| and |f| at their largest along the step, summed over the window by its weights.

    L grows with d, so T falls, and its fixed point d* is the largest such d: every d <= d* has
    T(d) >= T(d*) = d* >= d. T(rate), the bound at the plain step's length, is at most d*, and so is
    T(T(d)) for every d <= d*, which T(T) moves no lower than d (T(d) <= rate, so T(T(d)) >= T(rate)):
    `_SETTLINGS` rounds of T(T) rise toward d* from below, each continuous in what the bound reads, so
    the rate moves with its inputs' rounding by as much as they do. The rate is a constant of the step, as
    its bound is: the outer gradient flows through the step's gradient, not through the bound it reads."""
    squares, inputs, crossed, lengths = (sums[:, index] for index in range(_SUMS))
    least, fed, longest, norm, after, bent = (sums[:, index] for index in range(_SUMS, _WIDTH))
    p1, p2, growth = level.p1, level.p2, level.up_growth
    steps = {part: jnp.sqrt(jnp.sum(jnp.square(gradient[part]["kernel"].astype(jnp.float32)),
                                    axis=tuple(range(1, gradient[part]["kernel"].ndim))))
             for part in parts}

    def curvature_along(length: jax.Array) -> jax.Array:
        """L for a step of rate `length`, per row."""
        def reach(part: str) -> jax.Array:
            return length * steps[part] if part in parts else jnp.zeros_like(squares)

        # D's reach bounds how far f = act(D x) moves, p1 |x| per unit; U's, its norm, grown by its rounding.
        spread, output = reach("down"), growth * (norm + reach("up"))
        turned = p1 * spread * longest
        read, slope = head.at(least, after * (reach("up") * (fed + turned) + growth * norm * turned))
        curvature, slope = jnp.square(after) * read + slope * bent, after * slope
        bend = jnp.zeros_like(squares)
        stretch = jnp.zeros_like(squares)
        if "down" in parts:
            stretch = stretch + output ** 2 * p1 ** 2 * squares
            bend = bend + output * p2 * squares
        if "up" in parts:
            stretch = stretch + inputs + 2 * p1 * spread * crossed + p1 ** 2 * spread ** 2 * squares
            if "down" in parts:
                bend = bend + p1 * lengths
        return curvature * stretch + slope * bend

    def settled(length: jax.Array) -> jax.Array:
        return rate / (1 + rate * curvature_along(length))

    damped = settled(jnp.full_like(squares, rate))
    for _ in range(_SETTLINGS):
        damped = settled(settled(damped))
    return jax.lax.stop_gradient(damped)


def adapted_scores(model: nn.Module, variables, hidden, targets, weights, fields,
                   adaptation: ChunkAdaptation, *, predict: bool, logits: bool) -> Adapted:
    """A row's targets scored through its continuum levels stepped in context (module docstring):
    `hidden` the final states `[B, S, features]`, `targets` `[B, S]`, `weights` each target's weight in
    the loss, and `fields` the rows' token fields, their documents and real slots read as every
    reader of rows reads them (`token_layout`, `real_slots`: a validity mask, and a packed row's
    document 0, are padding). A row's places (`positions`) move no window: each run counts its own.
    The model owns the levels' place in its variables (`SteppedInContext`).

    A chain's levels (`dew.nn.continuum`) each step at their own window and rate (`ChunkAdaptation.
    windows`, `rates`), equation 71 a level: a token reads every level at the version its own window of
    that level opened with, and a window's gradient in its level is its tokens' own, each read through
    the other levels as it read them (`_Chain`). A damped window's bound reads its slots' terms at the
    states and versions they read (`_slot_terms`), gathered over the window as the scan reads them."""
    start = _start(model, variables, adaptation)
    names = _level_names(model, start)
    starts = [start] if not names else [start[name] for name in names]
    windows = adaptation.windows(len(starts))
    rows, length = targets.shape
    block = min(windows)
    straddles, weights, runs, keys = _laid_out(fields, weights, rows, length, adaptation, windows)
    chain = _Chain(model, variables, names, tuple(starts), adaptation, straddles, predict, rows)
    if adaptation.damped:
        chain = dataclasses.replace(chain, damping=chain.bounds(hidden))
    blocks = -(-length // block)
    # The last block's slots past the row weigh nothing and belong to the row's last window.
    edge = ((0, 0), (0, blocks * block - length))
    inputs = (*(sequence_blocks(value, block, leading=True) for value in (hidden, targets, weights)),
              sequence_blocks(jnp.pad(runs, edge, mode="edge"), block, leading=True),
              tuple(sequence_blocks(jnp.pad(key, edge, mode="edge"), block, leading=True) for key in keys))
    zeros = tuple(jax.tree.map(jnp.zeros_like, level) for level in chain.started)
    nothing = _gathered(jnp.zeros((rows, 1), bool), jnp.zeros((rows, 1, _WIDTH), jnp.float32))
    # Before a row's first real token no run is open: run 0, window -1.
    first = (chain.started, zeros, (nothing,) * len(starts), jnp.zeros((rows,), runs.dtype),
             tuple(jnp.full((rows,), -1, key.dtype) for key in keys))
    (levels, partials, energies, _, _), (losses, top, log_z, whole) = jax.lax.scan(
        jax.checkpoint(functools.partial(chain.block, logits=logits)), first, inputs)

    joined = functools.partial(_rejoined, length=length)
    moved = None
    if adaptation.carry == "run":
        # Each row's levels with their open windows' steps applied.
        reached = [chain.stepped(index, level, partial, energy) for index, (level, partial, energy)
                   in enumerate(zip(levels, partials, energies, strict=True))]
        moved = _moved(chain.tree(reached), chain.tree(chain.started), weights)
    return Adapted(joined(losses), None if top is None else joined(top), joined(log_z),
                   None if whole is None else joined(whole), moved)


@dataclasses.dataclass(frozen=True)
class _Chain:
    """The scan `adapted_scores` runs over a row's blocks of its least window, for a model's levels
    `starts` (a chain's, by their `names`, or one level's). A level whose window closes inside a block
    finishes its gradient over the block's slots before the boundary (its tail) at its open version, the
    levels of smaller windows read there as their slots read them, as their boundaries fall on its own,
    so the levels finish in order of their windows. A damped step (`damping`) reads the head's bounds and
    each level's activation bounds and rounding growth (`LevelTerms`) once, and its slots' terms as the
    scan reads them."""

    model: nn.Module
    variables: object
    names: tuple[str, ...]
    starts: tuple
    adaptation: ChunkAdaptation
    straddles: bool
    predict: bool
    rows: int
    damping: tuple[HeadTerms, tuple[LevelTerms, ...]] | None = None

    @functools.cached_property
    def started(self) -> tuple:
        return tuple(_rows(level, self.rows) for level in self.starts)

    @functools.cached_property
    def steps(self) -> tuple:
        return tuple(_stepped_mask(level, self.adaptation) for level in self.starts)

    @functools.cached_property
    def parts(self) -> tuple[str, ...]:
        return _damped_parts(self.starts[0], self.adaptation)

    def tree(self, levels):
        """The model's level tree holding `levels`, one a level of the chain."""
        return levels[0] if not self.names else dict(zip(self.names, levels, strict=True))

    def held(self, index: int, level):
        """The variables holding level `index` at `level`, the others at their starts."""
        held = list(self.starts)
        held[index] = level
        return self.model.with_in_context_level(self.variables, self.tree(held))

    def level_states(self, index: int, level, states) -> jax.Array:
        """`states` past level `index` at `level`."""
        return jnp.asarray(self.model.apply(self.held(index, level), states,
                                            *([index] if self.names else []), method="continuum_states"))

    def level_terms(self, index: int, level, states) -> LevelTerms:
        """What bounds a damped step through level `index` at `level` and `states` (`LevelTerms`): of its
        parts that step, read after a level before it (`passed`)."""
        return self.model.apply(self.held(index, level), states, self.parts, index, index > 0,
                                method="level_terms")

    def bounds(self, hidden) -> tuple[HeadTerms, tuple]:
        """The head's bounds at the states past the levels' starts (`SteppedInContext.head_bounds`) and
        each level's terms there, which refuse what no curvature bounds."""
        passed, read = hidden, []
        for index, level in enumerate(self.starts):
            read.append(self.level_terms(index, level, passed))
            passed = self.level_states(index, level, passed)
        head = self.model.apply(self.variables, passed, method="head_bounds")
        if not isinstance(head, HeadTerms):
            raise TypeError(f"a model stepped in context returns its head's bounds as `HeadTerms` "
                            f"(`SteppedInContext.head_bounds`), not {type(head).__name__}")
        return jax.lax.stop_gradient(head), tuple(jax.lax.stop_gradient(level) for level in read)

    def scored(self, versions, chosen, states, block_targets, block_weights):
        """Each slot's losses, its states read through every level at its own of that level's `versions`
        (`chosen`, per slot), in the chain's order; and, damped, each level's slot terms there
        (`_slot_terms`)."""
        from dew.objectives.lm.chunked import logits_cross_entropy

        passed, read = states, []
        for index in range(len(self.starts)):
            if self.adaptation.damped:
                each = [jax.vmap(functools.partial(self.level_terms, index))(version, passed)
                        for version in versions[index]]
                read.append(jax.tree.map(lambda *values, index=index: _chosen_by(chosen[index], values),
                                         *each))
            states_read = [jax.vmap(functools.partial(self.level_states, index))(version, passed)
                           for version in versions[index]]
            passed = _chosen_by(chosen[index], states_read)
        block_logits = jnp.asarray(self.model.apply(self.variables, passed, method="head_logits"))
        losses, top, log_z = logits_cross_entropy(block_logits, block_targets, predict=self.predict)
        terms = None
        if self.adaptation.damped:
            spread = self.model.apply(self.variables, passed, method="head_bounds").spread
            terms = _slot_terms(read, spread, block_weights)
        return losses, (top, log_z, block_logits, terms)

    def stepped(self, index: int, level, gradient, energy):
        """Level `index` at `level` stepped by its window's `gradient` at its rate, or at the rate a damped
        step takes from the window's gathered terms `energy` (`_step_rate`), one rate a row."""
        plain = self.adaptation.rates(len(self.starts))[index]
        rate = None
        if self.damping is not None:
            head, read = self.damping
            rate = _step_rate(plain, self.parts, head, read[index], energy, gradient)

        def step(leaf, grad, moves):
            if not moves:
                return leaf
            if rate is None:
                return leaf - plain * grad
            return leaf - rate.reshape(-1, *[1] * (leaf.ndim - 1)).astype(leaf.dtype) * grad
        return jax.tree.map(step, level, gradient, self.steps[index])

    def opened(self, carry, block_inputs):
        """Each level's versions a block reads and each slot's choice among them, and each level's
        window gradient and terms closed at its tail (`_Chain`)."""
        opens, partials, energies, run, places = carry
        states, block_targets, block_weights, block_runs, block_keys = block_inputs
        count = len(self.starts)
        versions: list = [None] * count
        chosen: list = [None] * count
        closed: list = [None] * count
        closing = list(energies)
        nothing = jnp.zeros_like(block_runs, jnp.int32)
        if not self.straddles:
            # One document a row and no padding: every level's windows open at blocks' starts.
            for index in range(count):
                opening = block_keys[index][:, 0] != places[index]
                closed[index] = partials[index]
                stepped = self.stepped(index, opens[index], partials[index], energies[index])
                versions[index] = (_where(opening, stepped, opens[index]),)
                chosen[index] = nothing
            return versions, chosen, closed, closing
        same = block_runs == run[:, None]
        windows = self.adaptation.windows(count)
        for index in sorted(range(count), key=lambda at: windows[at]):
            tail = same & (block_keys[index] == places[index][:, None])
            following = same & (block_keys[index] == places[index][:, None] + 1)

            # The open window's tail, read at the window's own level, finishes its gradient and terms.
            def tail_loss(at, index=index, tail=tail):
                held = [((opens[m],), nothing) if versions[m] is None else (versions[m], chosen[m])
                        for m in range(count)]
                held[index] = ((at,), nothing)
                losses, (*_, terms) = self.scored([version for version, _ in held],
                                                  [which for _, which in held], states, block_targets,
                                                  block_weights)
                return jnp.sum(jnp.where(tail, block_weights * losses, 0.0)), terms
            gradient, terms = jax.grad(tail_loss, has_aux=True)(opens[index])
            closed[index] = jax.tree.map(jnp.add, partials[index], gradient)
            if self.adaptation.damped:
                closing[index] = _joined(energies[index], _gathered(tail, terms[index]))
            stepped = self.stepped(index, opens[index], closed[index], closing[index])
            versions[index] = (opens[index], stepped, self.started[index])
            chosen[index] = jnp.where(tail, 0, jnp.where(following, 1, 2))
        return versions, chosen, closed, closing

    def block(self, carry, block_inputs, *, logits: bool):
        """One block of the scan (`adapted_scores`): its losses, and each level's version, window gradient
        and terms and window carried to the next block."""
        _, partials, energies, run, places = carry
        states, block_targets, block_weights, block_runs, block_keys = block_inputs
        versions, chosen, closed, closing = self.opened(carry, block_inputs)
        last_run = block_runs[:, -1]
        losses, pull, (top, log_z, block_logits, terms) = jax.vjp(
            lambda held: self.scored(held, chosen, states, block_targets, block_weights), versions,
            has_aux=True)
        carried = []
        for index in range(len(self.starts)):
            last_place = block_keys[index][:, -1]
            last = (block_runs == last_run[:, None]) & (block_keys[index] == last_place[:, None])
            # The last slot's window, one version's slots, takes its gradient from that version alone.
            pulled = pull(jnp.where(last, block_weights, 0.0).astype(losses.dtype))[0][index]
            reached = functools.reduce(lambda one, other: jax.tree.map(jnp.add, one, other), pulled)
            level = versions[index][0]
            for at, alternative in enumerate(versions[index][1:], start=1):
                level = _where(chosen[index][:, -1] == at, alternative, level)
            continuing = (last_run == run) & (last_place == places[index])
            following = closed[index] if self.straddles else jax.tree.map(jnp.add, partials[index], reached)
            energy = energies[index]
            if self.adaptation.damped:
                arrived = _gathered(last, terms[index])
                kept = closing[index] if self.straddles else _joined(energies[index], arrived)
                energy = jnp.where(continuing[:, None], kept, arrived)
            carried.append((level, _where(continuing, following, reached), energy, last_place))
        levels, gradients, energies, places = (tuple(parts) for parts in zip(*carried, strict=True))
        state = (levels, gradients, energies, last_run, places)
        return state, (losses, top, log_z, block_logits if logits else None)


def _rejoined(value: jax.Array, length: int) -> jax.Array:
    """A scan's `[blocks, B, block, ...]` output over a row's blocks as the row's `[B, length, ...]`."""
    blocks, rows, block = value.shape[:3]
    return jnp.moveaxis(value, 0, 1).reshape(rows, blocks * block, *value.shape[3:])[:, :length]


def _chosen_by(chosen: jax.Array, values) -> jax.Array:
    """Per slot, the value of `values` that `chosen` `[B, W]` names, each `[B, W, ...]`, or `[B]`, a row's
    for each of its slots."""
    def slotted(value) -> jax.Array:
        value = jnp.asarray(value)
        return jnp.broadcast_to(value[:, None], chosen.shape) if value.ndim < chosen.ndim else value

    values = [slotted(value) for value in values]
    kept = values[0]
    for at, alternative in enumerate(values[1:], start=1):
        kept = jnp.where((chosen == at).reshape(*chosen.shape, *[1] * (alternative.ndim - chosen.ndim)),
                         alternative, kept)
    return kept


def _level_names(model: nn.Module, start) -> tuple[str, ...]:
    """A chain's levels' names in its order (`SteppedInContext.in_context_levels`), none for one level."""
    count = model.in_context_levels if isinstance(model, SteppedInContext) else 1
    return () if count == 1 else tuple(f"continuum_{index}" for index in range(count))


def _laid_out(fields, weights, rows: int, length: int, adaptation: ChunkAdaptation,
              windows: tuple[int, ...]):
    """Where the rows' windows run (`window_keys`): whether a window can open inside a block (padding
    or several documents in a row), the weights with each padding slot's at nothing, each slot's run, and
    its window of each level (`windows`). A carried level steps through a row's documents in turn, one
    run a row."""
    marked, documents = token_layout(fields)
    straddles = marked is not None or documents is not None
    valid = real_slots(fields, (rows, length)) if straddles else None
    if valid is not None:
        weights = jnp.where(valid, weights, 0)
    documents = None if adaptation.carry == "run" else documents
    runs, ranks = _run_ranks(valid, documents, rows, length)
    return straddles, weights, runs, tuple(jnp.where(runs > 0, ranks // window, -1) for window in windows)


def _start(model: nn.Module, variables, adaptation: ChunkAdaptation) -> Mapping:
    """The level a document's first window reads (`ChunkAdaptation.carry`): the trained level, plus what
    the run carried of its steps on top (`CARRIED`), which its loss differentiates through as the start
    it is."""
    start = model.in_context_level(variables) if isinstance(model, SteppedInContext) else None
    if start is None:
        raise ValueError("the model has no continuum level to step in context")
    if adaptation.carry != "run":
        return start
    if CARRIED not in variables:
        raise ValueError(f"a run that carries its level holds what it carried in `{CARRIED}`, and these "
                         "variables hold none: a state built for this objective (`LMObjective.init`)")
    return jax.tree.map(lambda leaf, held: leaf + jax.lax.stop_gradient(held).astype(leaf.dtype),
                        start, variables[CARRIED])


def _moved(reached: Mapping, started: Mapping, weights: jax.Array) -> Mapping:
    """How far the rows' steps moved the level (`Adapted.moved`): each row's `reached` less its
    `started`, summed over the rows whose targets weigh anything, and their count."""
    counted = jnp.sum(weights, axis=1) > 0
    moves = jax.tree.map(lambda end, began: jnp.sum(jnp.where(
        counted.reshape(-1, *[1] * (end.ndim - 1)), end - began, 0), axis=0), reached, started)
    return jax.lax.stop_gradient({"moves": moves, "rows": jnp.sum(counted).astype(jnp.float32)})
