"""A scanned run of layers that updates its own weights in its backward (row 4796).

A scan's backward makes every layer's gradient before an optimizer reads any, so a scanned step holds
a gradient the size of its parameters that a step running a layer at a time never does: XLA reads each
layer's gradient into its update as it makes it. A training step whose update reads each parameter's
gradient alone (`dew.training.transaction.leaf_local`) opens an `InBackward` plan around its forward and
backward; a decoder's scanned run (`dew.nn.backbones.decoder_stack`) then runs as `scanned_run`, one
custom VJP over the whole run, whose backward walks the layers top down, takes each layer's gradient from
the residuals its forward kept (the layer's own remat keeps what it keeps), and writes the layer's update
into the run's weights and optimizer moments where they lie, so no gradient of the run is ever held whole.

The plan's `update` is the step's own optimizer update on one layer's slice (the transaction's
`optimizer_update`): the weights, moments and masters it updates reach the backward as the run's inputs
(variable collections a step adds, `MOMENTS`, `MASTERS`, `NORMS`), and leave it as their cotangents,
which the step takes for the run's new state in place of an update of its own. Whether the step commits
at all (`gated`) is decided after its forward, and the backward reads it as the step takes its gradient.
"""

from __future__ import annotations

import collections
import contextlib
import contextvars
import dataclasses
import functools
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import ClassVar, NamedTuple

import jax
import jax.extend.core as jcore
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray

from dew import _cache
from dew.nn.derivatives import (
    Surrogate,
    contracted,
    declared_derivative,
    marked_call_p,
    with_gradients_formed,
)
from dew.nn.finite import zeroed_unless
from dew.nn.programs import called, computed, holes

type Tree = Mapping[str, object]

MOMENTS = "update_moments"
"""The prefix of the collections a step adds for its optimizer's moments, one each (`moments_0`, ...), each
holding the copies of the run's weights its state keeps (Adam's first and second moments), so the run's
backward reads and writes them where the step holds them."""

MASTERS = "update_masters"
"""The collection holding the parameters the step updates where its forward reads narrow copies of them
(`dew.training.narrow`): the float32 masters the update moves, which the copies are made from again."""

NORMS = "update_norms"
"""The collection holding, per run, `[L]` zeros whose cotangents are each layer's gradient's sum of squares
(`sumsq`) and its count of non-finite entries (`nonfinite`): the step's norm and finiteness, read whole
from layers no step holds together."""


@dataclasses.dataclass(frozen=True)
class RunRule(Surrogate):
    """The rule of a run updating or pooling in its backward (`scanned_run`): a surrogate (its cotangents
    the step's new state), which a step's gate tells from every other custom rule by its kind
    (`dew.training.transaction._run_reads`): an outer rule wrapping it, which may scale what it pools, is
    no run's. Never held in `derivatives._SPANS`: its forward mutates `_Traced` as it traces and its
    backward reads the untraced `_GATE` and `_FINITE` contexts."""

    held: ClassVar[bool] = False


RUN_RULE = RunRule()


class RunRead(NamedTuple):
    """A run's variable in a loss's program (`read_once`): its run's path, the shapes of what the run's
    call reads as its trace saw them (`InBackward.shapes`), where each operand of that call reads from
    (`InBackward.sources`), and the variable's own collection and leaf's path within the run, which its
    weights, masters and moments share."""

    run: tuple[str, ...]
    called: tuple[tuple[int, ...], ...]
    sources: tuple[tuple[str, tuple[str, ...]] | None, ...]
    collection: str
    leaf: tuple[str, ...]


_PLACED = frozenset({"sharding_constraint", "optimization_barrier", "copy"})
"""The primitives whose results are their operands, value for value, placed (a sharding constraint), held
from fusion (a barrier) or copied: a run's variables pass through them as themselves (`read_once`). A cast
rounds what a rule would update in place (the step's narrow copies are inputs of their own)."""

_LAID_OUT = frozenset({"reshape", "squeeze", "expand_dims"})
"""The primitives that lay their operand out anew, its coordinates in row-major order (a stage-local
stack's layers reshaped to a run's): a run's variables pass through them where each leaf's weights,
masters and moments pass through the same ones, so the state its rule writes is laid out as the weights
it reads (`read_once`). A transpose, or a reshape that permutes its dimensions first, moves coordinates:
no such layout."""

_DRAWS = frozenset({"random_bits", "threefry2x32", "rng_bit_generator"})
"""The primitives that draw random bits: a run whose layers draw them (dropout) keys its masks otherwise
than Linen's scan does (`dew.nn.backbones.decoder_stack._updated_in_backward`), so no plan updates it."""


def read_once(program: jcore.Jaxpr, roots: Mapping[jcore.Var, RunRead], runs: Sequence[tuple[str, ...]]
              ) -> bool:
    """Whether each of `runs` is read in `program`, a loss's, once by its own rule (`runs_rule`) at the
    loss's own level, its variables (`roots`) read by nothing else: through calls (but a nested jit, whose
    cached program a later trace reads without calling the run), placements (`_PLACED`) and layouts every
    variable of the run takes alike (`_LAID_OUT`); its rule's operands shaped as its trace saw them (no
    batching transform's lanes beside them) and drawing no random bits; none of its variables among the
    loss's results (its statistics, which a reduction may read, outside the program)."""
    reads: collections.Counter[tuple[str, ...]] = collections.Counter()
    refused: set[tuple[str, ...]] = set()
    held = _reads(program, {value: (root, ()) for value, root in roots.items()}, reads, refused)
    refused.update(held[value][0].run for value in program.outvars if computed(value) and value in held)
    return all(reads[run] == 1 and run not in refused for run in runs)


def _reads(program: jcore.Jaxpr, roots: Mapping[jcore.Var, tuple[RunRead, tuple]],
           reads: collections.Counter[tuple[str, ...]], refused: set[tuple[str, ...]]
           ) -> dict[jcore.Var, tuple[RunRead, tuple]]:
    """`read_once`'s walk of `program`, each run variable held with the layouts it passed through: counts
    each run's rule reading it in `reads`, puts in `refused` each run any other equation reads; the run
    variables among `program`'s results."""
    held = dict(roots)
    for equation in program.eqns:
        read = [held[value] for value in equation.invars if computed(value) and value in held]
        if not read:
            continue
        runs = {root.run for root, _ in read}
        name = equation.primitive.name
        if name.startswith("custom_vjp") and runs_rule(equation):
            reads.update(runs)
            root = read[0][0]
            operands = equation.invars[-len(root.called):]
            given = tuple(tuple(value.aval.shape) if isinstance(value.aval, ShapedArray) else ()
                          for value in operands)
            # Each operand read from the source its call states (no weight swapped, repeated or missing),
            # each leaf laid out alike in every collection reading it, its operands its trace's, no draws.
            leaves = {(root.run, root.leaf) for root, _ in read}
            if (len(runs) != 1 or not _sourced(operands, held, root.sources)
                    or len({(root.run, root.leaf, layout) for root, layout in read}) != len(leaves)
                    or given != root.called
                    or any(_draws(inner) for inner in jcore.jaxprs_in_params(equation.params))):
                refused |= runs
            continue
        if name == "reshape" and equation.params.get("dimensions") is not None:
            refused |= runs
            continue
        if name in _PLACED or name in _LAID_OUT:
            step = () if name in _PLACED else ((name, repr(sorted(equation.params.items()))),)
            held.update({value: (held[given][0], held[given][1] + step)
                         for value, given in zip(equation.outvars, equation.invars, strict=False)
                         if computed(given) and given in held})
            continue
        body = None if name.startswith(("custom", "pjit", "jit")) else called(equation)
        if body is None:
            refused |= runs
            continue
        inner = {inside: held[value] for value, inside in zip(equation.invars, body.invars, strict=True)
                 if computed(value) and value in held}
        through = _reads(body, inner, reads, refused)
        held.update({value: through[inside]
                     for value, inside in zip(equation.outvars, body.outvars, strict=True)
                     if computed(inside) and inside in through})
    return held


def _sourced(operands: Sequence[jcore.Var | jcore.Literal], held: Mapping[jcore.Var, tuple[RunRead, tuple]],
             sources: Sequence[tuple[str, tuple[str, ...]] | None]) -> bool:
    """Whether each of a run's rule's `operands` reads the source its call states (`InBackward.sources`): a
    run variable (`held`) of that collection and leaf where it states one (the step's narrow copies among
    the variables its gate roots), never a value the loss made in its place; none where it states none."""
    for operand, source in zip(operands, sources, strict=True):
        root = held[operand][0] if computed(operand) and operand in held else None
        read = None if root is None else (root.collection, root.leaf)
        if read != source:
            return False
    return True


def runs_rule(equation: jcore.JaxprEqn) -> bool:
    """Whether `equation`, a custom derivative's call, is a run's own rule (`RUN_RULE`): its function a
    marked call of that kind (`dew.nn.derivatives.declared_derivative`), no rule around it."""
    return any(inner.primitive is marked_call_p and isinstance(inner.params["kind"], RunRule)
               for program in jcore.jaxprs_in_params(equation.params) for inner in program.eqns)


def _draws(program: jcore.Jaxpr) -> bool:
    """Whether `program`, or any program it holds, draws random bits (`_DRAWS`)."""
    return any(equation.primitive.name in _DRAWS
               or any(_draws(inner) for inner in jcore.jaxprs_in_params(equation.params))
               for equation in program.eqns)


@dataclasses.dataclass(frozen=True)
class InBackward:
    """A step's plan for the runs that update in their backward.

    `update(path, index, gradient, weights, moments)` is the step's optimizer update of one layer: the
    layer `index` of the run at `path` (the run's module path in the model), its `gradient` and
    `weights` (the masters where the step keeps narrow copies), and its `moments` by collection; it returns
    the moved weights and moments. `moments` names the collections of the optimizer's moments the step
    added. `handled` records each run's path as the model traces it in the plan, so the step reads those
    runs' state off the backward."""

    update: Callable[[tuple[str, ...], jax.Array, Tree, Tree, Mapping[str, Tree]],
                     tuple[Tree, dict[str, Tree]]]
    moments: tuple[str, ...]
    handled: list[tuple[str, ...]] = dataclasses.field(default_factory=list)
    shapes: dict[tuple[str, ...], tuple[tuple[int, ...], ...]] = dataclasses.field(default_factory=dict)
    """The shapes of what each run's call reads, by its path, as its trace sees them (`_registered`): under
    a batching transform (`jax.vmap`) its lanes' axis hidden from them, which the call's program states,
    one call standing for each lane's (`dew.training.transaction._read_once`)."""
    sources: dict[tuple[str, ...], tuple[tuple[str, tuple[str, ...]] | None, ...]] = dataclasses.field(
        default_factory=dict)
    """Where each operand of each run's call reads from, by the run's path, in the call's order
    (`_registered`): the collection and the leaf's path within the run of each of its weights, masters,
    moments and norms, None for what it reads beside them (its input, its rows' values)."""


_PLAN: contextvars.ContextVar[InBackward | None] = _cache.untraced(
    contextvars.ContextVar("dew_in_backward", default=None))
_GATE: contextvars.ContextVar[jax.Array | None] = _cache.untraced(
    contextvars.ContextVar("dew_in_backward_gate", default=None))
_FINITE: contextvars.ContextVar[jax.Array | None] = _cache.untraced(
    contextvars.ContextVar("dew_in_backward_finite", default=None))
"""Per-step tracing contexts for `scanned_run`, marked untraced (`_cache.untraced`) so global trace caches
keyed by `options.trace_context()` (`derivatives._SPANS`, `projections._DECISIONS`, `algebra.fusion`) neither
thrash per trace nor retain `InBackward.update` closures, transactions, trainers, executables, or tracers."""


@contextlib.contextmanager
def planned(plan: InBackward) -> Iterator[None]:
    """Open `plan` for the forward a step traces in it."""
    token = _PLAN.set(plan)
    try:
        yield
    finally:
        _PLAN.reset(token)


@contextlib.contextmanager
def gated(due: jax.Array, finite: jax.Array | None = None) -> Iterator[None]:
    """Let the backward a step takes in it write the runs' updates where `due`, and leave each run's state
    as it was elsewhere: whether the step commits, known once its forward is. Given `finite`, whether the
    step's loss is finite, a layer whose loss or own gradient is not takes the update of a zero gradient:
    its weights as they were, its optimizer one step on (training.md, a step that is not finite), as every
    layer of a step with no loss scale that is not finite does; a layer the walk reached before is moved.
    Once a layer is held, every layer after it in walk order is, across the step's runs: a run that held one
    hands the run below a cotangent that is not finite (`_passed_down`), whose layers its gate then holds."""
    tokens = _GATE.set(due), _FINITE.set(finite)
    try:
        yield
    finally:
        _GATE.reset(tokens[0])
        _FINITE.reset(tokens[1])


PERIOD_NORMS = "layers_0"
"""Where a run of a repeated period holds its norms (`NORMS`): under its period's first position, the key
its other collections hold that position's layers under, so a reader of a stage-local state's first layer
reads it as one of them (`CausalTransformer.first_layer_shapes`)."""


def norms_of(tree) -> dict[str, jax.Array]:
    """A run's norms (`NORMS`) as `scanned_run` writes them: a periodic run's from under its first position
    (`PERIOD_NORMS`)."""
    held = tree.get(PERIOD_NORMS, tree)
    return {"sumsq": held["sumsq"], "nonfinite": held["nonfinite"]}


def plan() -> InBackward | None:
    """The open plan, None outside one."""
    return _PLAN.get()


_Structure = tuple[jax.tree_util.PyTreeDef]
"""A tree's structure in a one-element tuple: JAX declares `PyTreeDef` as a value, which a type admits only
inside another (`dew.nn.kernels.options._Structure`)."""


@dataclasses.dataclass
class _Traced:
    """What a run's forward traced of one layer's pullback: its `structure`, and where each residual is read
    again (`slots`: the layer's weights, its input or what every layer reads, by position; None for one
    the forward stacked)."""

    structure: _Structure | None = None
    """The pullback's structure (`_Structure`)."""
    slots: list[tuple[str, int] | None] = dataclasses.field(default_factory=list)


def _zeros_like_input(leaf: jax.Array) -> jax.Array | np.ndarray:
    """An input's zero cotangent: a float leaf's zeros, an integer or boolean leaf's float0."""
    if jnp.issubdtype(leaf.dtype, jnp.inexact):
        return jnp.zeros_like(leaf)
    return np.zeros(np.shape(leaf), jax.dtypes.float0)


def _sliced(tree, index):
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _written(tree, layer, index):
    return jax.tree.map(lambda leaf, row: jax.lax.dynamic_update_index_in_dim(leaf, row.astype(leaf.dtype),
                                                                              index, 0), tree, layer)


def _flowing(shared: tuple, *, flows: tuple[bool, ...]) -> tuple:
    """What every layer reads that the layers differentiate (`flows`), the floating arrays among `shared`."""
    return tuple(leaf for leaf, flow in zip(shared, flows, strict=True) if flow)


def _merged(shared: tuple, flowed: tuple, *, flows: tuple[bool, ...]) -> tuple:
    """`shared` with its differentiated arrays (`flows`) taken from `flowed` in order."""
    given = iter(flowed)
    return tuple(next(given) if flow else leaf for leaf, flow in zip(shared, flows, strict=True))


def _norms(gradient) -> tuple[jax.Array, jax.Array]:
    """A layer's gradient's sum of squares and whether it is not finite, 1 or 0 (`NORMS`): a value not finite
    anywhere makes the sum so, read with no pass of its own."""
    leaves = jax.tree.leaves(gradient)
    squares = jnp.sum(jnp.stack([jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves]))
    return squares, (~jnp.isfinite(squares)).astype(jnp.float32)


def _passed_down(cotangent, tripped: jax.Array):
    """A cotangent a run hands what it read (its input and every differentiated shared input, an encoder's
    context among them): not finite where it held a layer (`tripped`), so the walk's latch holds across
    runs as data the backward carries, every layer reached below gated on its own gradient, which is then
    not finite (`gated`)."""
    return jax.tree.map(lambda leaf: jnp.where(tripped, jnp.full_like(leaf, jnp.nan), leaf), cotangent)


def _layer_update(update: Callable, finite: jax.Array | None, tripped: jax.Array, gradient, held, moments,
                  bad: jax.Array):
    """One layer's update as the walk takes it: as it is where no step's finiteness gates the walk
    (`gated`: a run that takes its steps as formed), else gated (`_gated_layer`) on the loss's finiteness,
    the walk's latch (`tripped`) and the layer's own gradient's (`bad`, from its norm). The layer's weights,
    moments, non-finite count and the latch after it."""
    if finite is None:
        moved, moved_moments = update(gradient, held, moments)
        return moved, moved_moments, bad, tripped
    ok = finite & (bad == 0) & ~tripped
    moved, moved_moments = _gated_layer(update, ok, gradient, held, moments)
    return moved, moved_moments, bad, tripped | ~ok


def _gated_layer(update: Callable, ok: jax.Array, gradient, held, moments):
    """One layer's update in a step that skips what is not finite (`gated`): where `ok` (the loss finite,
    the layer's gradient finite) by its gradient, else by a zero gradient, its weights kept as they were and
    its moments stepped as a commit's are. The layer's moved weights and moments."""
    moved, moved_moments = update(zeroed_unless(ok, gradient), held, moments)
    return jax.tree.map(lambda new, old: jnp.where(ok, new, old), moved, held), moved_moments


def _pulled(back: Callable, dx):
    """`back(dx)`, a layer's VJP: its parameters' gradients, the layer below's cotangent and the flowed
    state's, the cotangent returned only with the gradients formed (`with_gradients_formed`, row 6027), which
    the update reads at once."""
    gradient, dx, dflowed = back(dx)
    dx, (gradient, dflowed) = with_gradients_formed(dx, (gradient, dflowed))
    return gradient, dx, dflowed


def scanned_run(path: tuple[str, ...], layer: Callable, weights: Tree, masters: Tree | None,
                moments: Mapping[str, Tree], norms: Tree, x: jax.Array, per_layer: tuple | jax.Array | None,
                shared: Sequence[object]) -> jax.Array:
    """`layer(weights_i, x, per_layer_i, *shared)` over the run's layers in order, as a scan computes it,
    whose backward updates the run in place (`InBackward`): its new weights, masters and moments, and each
    layer's gradient's norm and finiteness (`NORMS`), are the cotangents of `weights`, `masters`,
    `moments` and `norms`. `per_layer` holds what each layer reads of its own (a scan's xs: each layer's
    dropout key, its read-only collections), `shared` what every layer reads (attention metadata), passed
    in, not closed over. A floating array `shared` holds (a cross-attention's context, which a trainable
    encoder computes) takes the sum of every layer's cotangent, as a scan's closed-over input does; the
    others none."""
    current = plan()
    assert current is not None, "a run updates in its backward only inside a step's plan"
    traced = _Traced()
    # What every layer reads crosses the rule as its arrays; its other leaves (a static field of the
    # attention metadata) are the rule's own.
    arrays, shared_of = holes(tuple(shared), lambda leaf: isinstance(leaf, jax.Array | np.ndarray))
    # Which of them the layers differentiate: the floating ones.
    flows = tuple(bool(jnp.issubdtype(jnp.result_type(leaf), jnp.inexact)) for leaf in arrays)

    flowing, merged = functools.partial(_flowing, flows=flows), functools.partial(_merged, flows=flows)

    def forward_layer(x, slices, shared):
        weights_i, per_layer_i = slices
        return layer(weights_i, x, per_layer_i, *shared_of(shared))

    def run(weights, masters, moments, norms, x, per_layer, shared):
        def body(x, slices):
            return forward_layer(x, slices, shared), None
        y, _ = jax.lax.scan(body, x, (weights, per_layer))
        return y

    def run_forward(weights, masters, moments, norms, x, per_layer, shared):
        def body(x, slices):
            weights_i, per_layer_i = slices
            y, back = jax.vjp(lambda weights_i, x, flowed: layer(weights_i, x, per_layer_i,
                                                                   *shared_of(merged(shared, flowed))),
                              weights_i, x, flowing(shared))
            leaves, treedef = jax.tree_util.tree_flatten(back)
            # A residual that is the layer's weights, its own input or what every layer reads is read again
            # where it lies in the backward, not stacked a second time.
            given = {id(leaf): ("weights", position)
                     for position, leaf in enumerate(jax.tree.leaves(weights_i))}
            given.update({id(leaf): ("per_layer", position)
                          for position, leaf in enumerate(jax.tree.leaves(per_layer_i))})
            given.update({id(leaf): ("shared", position)
                          for position, leaf in enumerate(jax.tree.leaves(shared))})
            given.update({id(leaf): ("x", position) for position, leaf in enumerate(jax.tree.leaves(x))})
            traced.structure = (treedef,)
            traced.slots = [given.get(id(leaf)) for leaf in leaves]
            return y, ([leaf for leaf in leaves if id(leaf) not in given], x)
        y, (kept, inputs) = jax.lax.scan(body, x, (weights, per_layer))
        kept = list(kept)
        return y, (weights, masters, moments, norms, kept, inputs, per_layer, shared)

    def run_backward(residuals, cotangent):
        weights, masters, moments, norms, kept, inputs, per_layer, shared = residuals
        assert traced.structure is not None, "the run's forward traces its layer before its backward"
        (treedef,), slots = traced.structure, traced.slots
        due, finite = _GATE.get(), _FINITE.get()
        count = jax.tree.leaves(weights)[0].shape[0]
        shared_leaves = jax.tree.leaves(shared)

        # The tree the update moves, written in place as the walk goes: the weights, or the masters where the
        # step keeps narrow copies, which the layers below read as the forward did (the step casts its new
        # masters to them again as it commits); the copies are read, not carried.
        narrow = masters is not None

        def body(carry, index):
            dx, moving, moments, sumsq, nonfinite, dshared, tripped = carry
            weights_i = _sliced(weights if narrow else moving, index)
            given = {"weights": jax.tree.leaves(weights_i), "shared": shared_leaves,
                     "x": jax.tree.leaves(_sliced(inputs, index)),
                     "per_layer": jax.tree.leaves(_sliced(per_layer, index))}
            stored = iter([leaf[index] for leaf in kept])
            back = jax.tree_util.tree_unflatten(treedef, [given[slot[0]][slot[1]] if slot is not None
                                                          else next(stored) for slot in slots])
            gradient, dx, dflowed = _pulled(back, dx)
            dshared = jax.tree.map(jnp.add, dshared, dflowed)
            held = _sliced(moving, index)
            held_moments = {name: _sliced(tree, index) for name, tree in moments.items()}
            squares, bad = _norms(gradient)
            moved, moved_moments, bad, tripped = _layer_update(
                functools.partial(current.update, path, index), finite, tripped, gradient, held,
                held_moments, bad)
            if due is not None:
                moved, moved_moments = jax.tree.map(lambda new, old, due=due: jnp.where(due, new, old),
                                                    (moved, moved_moments), (held, held_moments))
            # One barrier over the layer's new rows keeps XLA from fusing their producers (an Adam's new
            # moments feed both its new weight and the moments' writes) across the stacks' separate writes,
            # which held slices of the stacks past a row's write on the measured CUDA builds (row 8389).
            moved, moved_moments = jax.lax.optimization_barrier((moved, moved_moments))
            moving = _written(moving, moved, index)
            moments = {name: _written(tree, moved_moments[name], index) for name, tree in moments.items()}
            sumsq = sumsq.at[index].set(squares)
            nonfinite = nonfinite.at[index].set(bad)
            return (dx, moving, moments, sumsq, nonfinite, dshared, tripped), None

        start = (cotangent, masters if narrow else weights, moments, jnp.zeros((count,), jnp.float32),
                 jnp.zeros((count,), jnp.float32), tuple(jnp.zeros_like(leaf) for leaf in flowing(shared)),
                 jnp.asarray(a=False))
        (dx, moved, moments, sumsq, nonfinite, dshared, tripped), _ = jax.lax.scan(
            body, start, jnp.arange(count), reverse=True)
        dx, dshared = _passed_down((dx, dshared), tripped)
        written_norms = {"sumsq": sumsq.astype(norms["sumsq"].dtype),
                         "nonfinite": nonfinite.astype(norms["nonfinite"].dtype)}
        # The copies' cotangent is nothing the step reads (`Transaction.split_planned` takes the masters').
        weights_out = jax.tree.map(jnp.zeros_like, weights) if narrow else moved
        return (weights_out, moved if narrow else None, moments, written_norms, dx,
                jax.tree.map(_zeros_like_input, per_layer),
                merged(tuple(_zeros_like_input(leaf) for leaf in shared), dshared))

    # A rule whose cotangents are the step's new state, Dew's choice, never the run's derivative
    # (`Surrogate`): the step reads them as its update (`InBackward`), its gate (`gated`) a value of the
    # step's own trace the backward reads where it runs.
    updating = contracted("test_a_run_updating_in_its_backward_moves_as_its_step_would")(
        declared_derivative(RUN_RULE, run, run_forward, run_backward))
    called = (weights, masters, dict(moments), norms, x, per_layer, tuple(arrays))
    _registered(current, path, called)
    return updating(*called)


def _registered(current: InBackward, path: tuple[str, ...], called: tuple) -> None:
    """Record a run's `path` as the model traces it in the plan (`InBackward.handled`), with the shapes of
    what its call reads as the trace sees them (`InBackward.shapes`)."""
    current.handled.append(path)
    current.shapes[path] = tuple(jnp.shape(leaf) for leaf in jax.tree.leaves(called))
    weights, masters, moments, norms, *rest = called

    def sourced(collection: str, tree) -> list[tuple[str, tuple[str, ...]]]:
        return [(collection, tuple(str(key.key) for key in leaf if isinstance(key, jax.tree_util.DictKey)))
                for leaf, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]

    current.sources[path] = (*sourced("params", weights), *sourced(MASTERS, masters),
                             *(source for name in sorted(moments) for source in sourced(name, moments[name])),
                             *sourced(NORMS, norms), *([None] * len(jax.tree.leaves(rest))))
