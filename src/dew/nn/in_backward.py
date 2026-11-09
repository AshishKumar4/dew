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

import contextlib
import contextvars
import dataclasses
from collections.abc import Callable, Iterator, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.derivatives import Surrogate, contracted, declared_derivative
from dew.nn.programs import holes

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


_PLAN: contextvars.ContextVar[InBackward | None] = contextvars.ContextVar("dew_in_backward", default=None)
_GATE: contextvars.ContextVar[jax.Array | None] = contextvars.ContextVar("dew_in_backward_gate", default=None)


@contextlib.contextmanager
def planned(plan: InBackward) -> Iterator[None]:
    """Open `plan` for the forward a step traces in it."""
    token = _PLAN.set(plan)
    try:
        yield
    finally:
        _PLAN.reset(token)


@contextlib.contextmanager
def gated(due: jax.Array) -> Iterator[None]:
    """Let the backward a step takes in it write the runs' updates where `due`, and leave each run's state
    as it was elsewhere: whether the step commits, known once its forward is."""
    token = _GATE.set(due)
    try:
        yield
    finally:
        _GATE.reset(token)


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


def scanned_run(path: tuple[str, ...], layer: Callable, weights: Tree, masters: Tree | None,
                moments: Mapping[str, Tree], norms: Tree, x: jax.Array, per_layer: jax.Array | None,
                shared: Sequence[object]) -> jax.Array:
    """`layer(weights_i, x, per_layer_i, *shared)` over the run's layers in order, as a scan computes it,
    whose backward updates the run in place (`InBackward`): its new weights, masters and moments, and each
    layer's gradient's norm and finiteness (`NORMS`), are the cotangents of `weights`, `masters`,
    `moments` and `norms`. `per_layer` holds what each layer reads of its own (a scan's xs: each layer's
    dropout key), `shared` what every layer reads (attention metadata), passed in, not closed over."""
    current = plan()
    assert current is not None, "a run updates in its backward only inside a step's plan"
    current.handled.append(path)
    traced = _Traced()
    # What every layer reads crosses the rule as its arrays; its other leaves (a static field of the
    # attention metadata) are the rule's own.
    arrays, shared_of = holes(tuple(shared), lambda leaf: isinstance(leaf, jax.Array | np.ndarray))

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
            y, back = jax.vjp(lambda weights_i, x: layer(weights_i, x, per_layer_i, *shared_of(shared)),
                              weights_i, x)
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
        due = _GATE.get()
        count = jax.tree.leaves(weights)[0].shape[0]
        shared_leaves = jax.tree.leaves(shared)

        # The tree the update moves, written in place as the walk goes: the weights, or the masters where the
        # step keeps narrow copies, which the layers below read as the forward did (the step casts its new
        # masters to them again as it commits); the copies are read, not carried.
        narrow = masters is not None

        def body(carry, index):
            dx, moving, moments, sumsq, nonfinite = carry
            weights_i = _sliced(weights if narrow else moving, index)
            given = {"weights": jax.tree.leaves(weights_i), "shared": shared_leaves,
                     "x": jax.tree.leaves(_sliced(inputs, index)),
                     "per_layer": jax.tree.leaves(_sliced(per_layer, index))}
            stored = iter([leaf[index] for leaf in kept])
            back = jax.tree_util.tree_unflatten(treedef, [given[slot[0]][slot[1]] if slot is not None
                                                          else next(stored) for slot in slots])
            gradient, dx = back(dx)
            held = _sliced(moving, index)
            held_moments = {name: _sliced(tree, index) for name, tree in moments.items()}
            moved, moved_moments = current.update(path, index, gradient, held, held_moments)
            leaves = jax.tree.leaves(gradient)
            squares = jnp.sum(jnp.stack([jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves]))
            bad = jnp.sum(jnp.stack([jnp.sum(~jnp.isfinite(leaf)) for leaf in leaves])).astype(jnp.float32)
            if due is not None:
                moved = jax.tree.map(lambda new, old, due=due: jnp.where(due, new, old), moved, held)
                moved_moments = jax.tree.map(lambda new, old, due=due: jnp.where(due, new, old),
                                             moved_moments, held_moments)
            moving = _written(moving, moved, index)
            moments = {name: _written(tree, moved_moments[name], index) for name, tree in moments.items()}
            sumsq = sumsq.at[index].set(squares)
            nonfinite = nonfinite.at[index].set(bad)
            return (dx, moving, moments, sumsq, nonfinite), None

        start = (cotangent, masters if narrow else weights, moments, jnp.zeros((count,), jnp.float32),
                 jnp.zeros((count,), jnp.float32))
        (dx, moved, moments, sumsq, nonfinite), _ = jax.lax.scan(body, start, jnp.arange(count), reverse=True)
        written_norms = {"sumsq": sumsq.astype(norms["sumsq"].dtype),
                         "nonfinite": nonfinite.astype(norms["nonfinite"].dtype)}
        # The copies' cotangent is nothing the step reads (`Transaction.split_planned` takes the masters').
        weights_out = jax.tree.map(jnp.zeros_like, weights) if narrow else moved
        return (weights_out, moved if narrow else None, moments, written_norms, dx,
                jax.tree.map(_zeros_like_input, per_layer), jax.tree.map(_zeros_like_input, shared))

    # A rule whose cotangents are the step's new state, Dew's choice, never the run's derivative
    # (`Surrogate`): the step reads them as its update (`InBackward`), its gate (`gated`) a value of the
    # step's own trace the backward reads where it runs.
    updating = contracted("test_a_run_updating_in_its_backward_moves_as_its_step_would")(
        declared_derivative(Surrogate(), run, run_forward, run_backward))
    return updating(weights, masters, dict(moments), norms, x, per_layer,
                    tuple(arrays))
