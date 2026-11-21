"""An accelerator realization over immutable banks of a CPU-owned state.

Only objective evaluation and its pullback cross this boundary. The optimizer,
accumulation and checkpoint trees retain their original logical leaf identities.
Each snapshot bank is assembled and transferred before the next is built; no
whole parameter tree is ever materialized on the accelerator.

The CPU owns what moves. A leaf the objective's trainable filter froze never
changes, so the state holds it where a realization reads it (`resident`):
a layer's leaf in the pinned host memory its bank streams from, anything
outside the stack in the accelerator's own memory. Its snapshot is itself,
one array for the whole run, and the per-step copy is of the moving leaves
alone. A scanned run stacks its frozen rows once, the first step that reads
them, and keeps that bank for as long as the rows it was built from stay.

The pullback is over the moving leaves alone. A resident leaf, or a bank
stacked from resident rows only, enters the objective as a value the vjp
does not differentiate, the way the resident transaction passes `frozen`:
the forward keeps nothing for a cotangent no optimizer reads, and the
cotangent of a frozen row is never traced rather than dropped after the
fact. What the backward needs of a resident bank is the bank itself, read
from where it sits.
"""
from __future__ import annotations

import dataclasses
import functools
import math
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P

from dew.batch import FROZEN, PROBES, at_path, merge, placed_at, select, thaw
from dew.distributed.batches import batch_shardings
from dew.distributed.host import transfer
from dew.distributed.placement import placed_shapes
from dew.inference.banks import (
    assembled,
    bank_sharding,
    bank_sites,
    banked,
    entry_tree,
    in_namespace,
    logical,
    narrowed,
    one_layer,
    without_banks,
)
from dew.mesh import batch_axes
from dew.nn.backbones.decoder_stack import DecoderBank
from dew.nn.backbones.layer_names import layer_index
from dew.nn.inspection import unscaled, unscaling
from dew.nn.kernels.options import Option, owned_program, program
from dew.training.memory import compiled_if_it_fits, device_tokens, step_compiler_options
from dew.training.transaction import Realization, discovered_probes, placed_as_rows, refuse_unread_probes

if TYPE_CHECKING:
    from dew.data.pipeline import DeviceStep
    from dew.distributed.placement import Placement
    from dew.state import TrainState

BANK_MEMORY = "pinned_host"
"""Where a layer's bank sits beside the accelerator: the stack fetches one
layer of it at a time as it reaches it (`run_stack`). The CPU backend has
the space too, so a CPU-only run exercises the same fetching loop."""


def _without_banks(variables, sites):
    """`variables` without the banks the stacks of `sites` store, at every root (`without_banks`)."""
    return {**without_banks(variables, _own(sites)),
            **{collection: _without_banks(variables[collection], _within(sites, collection))
               for collection in _nested(sites) if collection in variables}}


def _in_stack(path: tuple[str, ...], sites) -> bool:
    """Whether a `params`-relative leaf path is one of a declared stack's layers."""
    return any(path[:len(site.namespace)] == site.namespace
               and len(path) > len(site.namespace)
               and layer_index(path[len(site.namespace)]) is not None
               for site in sites)


def stack_sites(objective, variables) -> tuple[DecoderBank, ...]:
    """The objective's declared layer stacks over `variables` as its model reads them, the frozen leaves
    back in `params` (`thaw`), which a host layout streams as banks: their runs end where the model's
    rows differ (`CausalTransformer.bank_sites`), so the frozen banks, the execution's snapshot and the
    model's own scan agree on every run."""
    store = thaw(variables)
    return bank_sites(objective, store) if objective.bank_sites(store) else ()


def _own(sites) -> tuple[DecoderBank, ...]:
    """The `sites` in the run's own tree, at no nested root (`DecoderBank.root`)."""
    return tuple(site for site in sites if not site.root)


def _nested(sites) -> tuple[str, ...]:
    """The collections holding a nested root's stacks of `sites`, each once."""
    return tuple(dict.fromkeys(site.root[0] for site in sites if site.root))


def _within(sites, collection: str) -> tuple[DecoderBank, ...]:
    """The `sites` below `collection`, seen from the variables it holds."""
    return tuple(dataclasses.replace(site, root=site.root[1:])
                 for site in sites if site.root[:1] == (collection,))


def frozen_trees(variables, sites):
    """The frozen trees of `variables` a host layout places beside the accelerator, each as its path, the
    tree, and the `sites` over it seen from it: the run's frozen collection, and both collections of each
    nested model that trains nothing at its root (`DecoderBank.root`), a distillation's teacher whole,
    its own frozen split (a LoRA base) beside its params. A root `variables` does not hold has none: an
    EMA averages the student alone."""
    for root in dict.fromkeys(((), *(site.root for site in sites))):
        held = tuple(dataclasses.replace(site, root=()) for site in sites if site.root == root)
        for path in ((*root, "params"), (*root, FROZEN)) if root else ((FROZEN,),):
            tree = at_path(variables, path)
            if tree is not None:
                yield path, tree, held


def bank_bytes(tree, sites) -> int:
    """How many bytes of a frozen collection a host layout keeps in bank memory.

    The leaves of every declared stack's layers land there, banked or not;
    the rest stays on the device. `resident` places by the same rule.
    """
    return sum(leaf.nbytes for path, leaf in jax.tree_util.tree_leaves_with_path(tree)
               if _in_stack(tuple(entry.key for entry in path), sites))


HOST_LIMIT = "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"


def check_bank_pool(nbytes: int, mesh) -> None:
    """Refuse a GPU pinned-host pool that cannot hold `nbytes` of banks.

    XLA grows the pool by regions, each a power of two at least as large as
    the request and the one before it, capped at the process's host memory
    limit; a region is never returned. Banks placed one at a time therefore
    reserve up to the next power of two above their size unless the limit
    stops them: 69 GB for 49 GB of banks on an A100 at the 72 GB limit that
    run set. Set `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` to the banks plus a few
    gigabytes for transfers before the backend starts, and the pool ends
    there. The limit is read here because it is the only lever there is;
    the backend is already running.
    """
    if mesh.devices.flat[0].platform != "gpu":
        return
    limit = os.environ.get(HOST_LIMIT)
    if limit is not None and float(limit) * 1e9 < nbytes:
        raise ValueError(
            f"the frozen banks need {nbytes / 1e9:.1f} GB of pinned host memory, more than "
            f"{HOST_LIMIT}={limit} allows; set it to at least {nbytes / 1e9 + 4:.0f} before "
            "the JAX backend starts")


def unscanned(sites) -> str | None:
    """Why a host layout cannot hold the stacks `sites` in bank memory, None where it can: a stack whose
    decoder runs a plain loop, whose layer fetches the compiler hoists together (`resident`)."""
    loops = [".".join(site.namespace) or "<root>" for site in sites if not site.scanned]
    if not loops:
        return None
    return (f"a host layout streams a scanned stack; the decoder at {', '.join(loops)} runs a plain loop "
            "(scan_layers=False), whose layer fetches the compiler hoists together, so set "
            "scan_layers=True (and bank_layers to bound a run) on it")


def resident(placement, sites, accelerator):
    """Return where the frozen collection sits beside the accelerator.

    A stack's layers go to bank memory and the rest to device memory, each
    leaf on the shard the layout named. The leaves a scanned run holds in
    every row are placed as that run's bank, layer axis in front (`banked`).

    Only a scanned stack is placed in bank memory: its scan fetches one row
    per iteration, so the device holds one layer of the bank at a time. A
    plain loop's fetches have no order between them, so the scheduler hoists
    them all to the front and the whole stack lands on the device. That is
    the memory the layout was asked to avoid, so it is refused by name (`unscanned`)."""
    refused = unscanned(sites)
    if refused is not None:
        raise ValueError(refused)
    def leaf(path, sharding):
        keys = tuple(entry.key for entry in path)
        kind = BANK_MEMORY if _in_stack(keys, sites) else None
        return NamedSharding(accelerator, sharding.spec, memory_kind=kind)
    placed = jax.tree_util.tree_map_with_path(leaf, placement)
    return banked(placed, sites, lambda rows, path: bank_sharding(rows[0], memory_kind=BANK_MEMORY))


def _bank_shardings(placed, accelerator, count):
    """Place one bank in bank memory, layer axis in front when it stacks rows."""
    def place(sharding):
        row = NamedSharding(accelerator, sharding.spec, memory_kind=BANK_MEMORY)
        return bank_sharding(row, memory_kind=BANK_MEMORY) if count > 1 else row

    return jax.tree.map(place, placed)


def _after(device: tuple[DeviceStep, ...], loss, variables, batch, execution):
    """`loss` over the batch the dataset's `device` steps make of `batch`."""
    for step in device:
        batch = step(batch)
    return loss(variables, batch, execution)


def _forward(loss, primal, frozen, store, batch, execution, scale=None, *, layout):
    """`loss` of the moving leaves `primal` beside the `frozen` ones, the
    store's other collections, the batch and the step's clock and keys, with
    its pullback. Compiled once for its objective (`HostExecution`), it hands
    back a pullback whose static part is the same program each realization,
    so `_trainable_cotangent` traces once rather than once an attempt. A step
    that watches its outputs' cotangents (`Step.watched`) differentiates its
    watched modules' probes beside the moving leaves, laid out as its model
    calls read them (`discovered_probes`), its pullback giving both, each
    probe's cotangent read unscaled by the loss `scale` the backward is
    seeded with, an argument of the program (`dew.nn.inspection.unscaled`)."""
    treedef, held = layout

    def merged(trainable):
        moved, kept = iter(trainable), iter(frozen)
        return treedef.unflatten([next(kept) if known else next(moved) for known in held])

    watched = execution.watched
    if watched is None or not watched.cotangents:
        return jax.vjp(lambda trainable: loss({**store, "params": merged(trainable)}, batch, execution),
                       primal, has_aux=True)
    probes = discovered_probes(loss, {**store, "params": merged(primal)}, batch, execution)

    def evaluate(trainable, probes):
        return loss({**store, "params": merged(trainable), PROBES: probes}, batch, execution)
    with unscaled(scale):
        stats, back, aux = jax.vjp(evaluate, primal, probes, has_aux=True)
    refuse_unread_probes(probes, aux)
    return stats, back, aux


def _probed_cotangent(back, cotangent, *, sites, paths, layout):
    """`_trainable_cotangent` of a watching step's pullback, and its probes' cotangents beside it: the
    pullback run once for both (`_forward`), as the device path runs it."""
    pulled = back(cotangent)
    return _selected(pulled[0], sites=sites, paths=paths, layout=layout), pulled[1]


def _trainable_cotangent(back, cotangent, *, sites, paths, layout):
    """Run the pullback, the unstack and the selection as one computation.

    `back` crosses this boundary as its own pytree, so the residuals are
    arguments rather than captured constants. It returns one cotangent per
    moving store leaf (`_selected`).
    """
    return _selected(back(cotangent)[0], sites=sites, paths=paths, layout=layout)


def _selected(moving, *, sites, paths, layout):
    """The pullback's cotangents of the moving store leaves (`moving`), in the store's shape: `layout` (the
    store's treedef and which of its leaves are held) puts them back, a held leaf's place empty, so a mixed
    bank's frozen rows are the dead values the compiler drops and its moving rows are selected under their
    logical paths."""
    treedef, held = layout
    cotangents = iter(moving)
    placed = treedef.unflatten([None if known else next(cotangents) for known in held])
    gradient = logical({"params": placed}, sites)["params"]
    trainable = frozenset(paths)
    return select(gradient, lambda path: path in trainable)


def spec_on(mesh: Mesh):
    """Where `HostExecution`'s moves to `mesh` place a leaf: the partition spec it has, or replicated where
    it states none (a scalar clock, a key crossing the boundary)."""
    def placement(leaf):
        sharding = getattr(leaf, "sharding", None)
        return NamedSharding(mesh, sharding.spec if isinstance(sharding, NamedSharding) else P())
    return placement


def _stack(*rows):
    """A bank of `rows`, stacked on the CPU (`HostExecution._stack_rows`)."""
    return jnp.stack(rows)


class _Abstract:
    """A realization's moves of shapes (`HostExecution.inputs`' `moved`): each leaf the shape it would
    take, placed where the move would place it (`placed_shapes`, its weak type kept, so the step's programs
    over shapes are keyed as its calls over arrays are, `OuterProgram.abstract_of`), and a bank of rows
    its stack's shape."""

    def placed(self, tree, placement):
        return placed_shapes(tree, placement, weak=True)

    def _stack_rows(self, rows, sharding):
        return jax.ShapeDtypeStruct((len(rows), *rows[0].shape), rows[0].dtype, sharding=sharding)

    def _moved(self, tree, mesh):
        return self.placed(tree, jax.tree.map(spec_on(mesh), tree))

    def rows(self, index, batch):
        return placed_as_rows(index, batch)


class _Collected(_Abstract):
    """A realization's moves of shapes within `options.collecting` (`HostExecution.collected`): as
    `_Abstract`'s, a bank's stack its program's, compiled over its rows' shapes on the CPU as a
    realization over arrays runs it (`HostExecution._stacked`)."""

    def __init__(self, execution: HostExecution):
        self.execution = execution

    def _stack_rows(self, rows, sharding):
        stacked = self.execution._stacked(*self._moved(rows, self.execution.cpu))
        return self.placed(stacked, sharding)


def on_mesh(mesh: Mesh | AbstractMesh):
    """`mesh` in context, as a program traced on it reads it: an abstract mesh (a target's devices this
    process holds none of, `generation.gpu_mesh`) as JAX's abstract mesh, a mesh of devices as `set_mesh`."""
    return jax.sharding.use_abstract_mesh(mesh) if isinstance(mesh, AbstractMesh) else jax.set_mesh(mesh)


class HostExecution:
    """Runs one objective evaluation and its pullback on the accelerator.

    The CPU owns the state. This class places a snapshot of it on the
    accelerator, runs the loss there, and hands back a pullback. It keeps no
    state of its own between attempts.
    """
    def __init__(self, objective, layout, accelerator, cpu, sites: tuple[DecoderBank, ...],
                 device: tuple[DeviceStep, ...] = (), gathered: int | None = None):
        self.objective, self.layout = objective, layout
        # The bytes its step's all-gathers combine up to (`layer_gathers`), its plan's.
        self.gathered = gathered
        self.accelerator, self.cpu = accelerator, cpu
        # The trainer's banks over the state it owns (`stack_sites`), the runs its frozen banks keep;
        # a gradient reaches the run's own stacks alone, a nested root (a teacher) training nothing.
        self.sites, self.own = sites, _own(sites)
        self.device = device
        self.unstack = functools.partial(program(logical, static_argnames="sites"), sites=self.own)
        self._stacked = program(_stack)

    def forward(self, step: Mapping[str, Option]):
        """The loss and its pullback, compiled for the objective's modules as they are now, under the
        `step`'s XLA options (`step_options`): the objective owns the program, which its fit ladder
        replaces with its model. The dataset's device steps run in it, on the accelerator, before the
        loss, so it holds them (`holds`)."""
        return self.objective.program(
            "host_forward",
            functools.partial(_forward, functools.partial(_after, self.device, self.objective._loss)),
            static_argnames=("layout",), holds=self.device, step=step)

    def step_options(self, variables, batch) -> dict[str, Option]:
        """The XLA options the resident step compiles under (`step_compiler_options`), which the host
        step's forward and pullback take too, so the two differ by the host boundary alone: the
        tokens each accelerator device steps of `batch`, whether the step trains next to frozen
        weights, and the accelerator's devices."""
        shards = math.prod(self.accelerator.shape[axis] for axis in batch_axes(self.accelerator))
        options = step_compiler_options(self.objective, device_tokens(self.objective, batch, shards),
                                        FROZEN in variables, self.accelerator.size,
                                        gathered=self.gathered)
        return dict(options or {})

    def snapshot(self, variables, *, moved=None):
        """The declared stacks as banks, everything else as it is stored, each moved by `moved` (this
        execution, where None): `_Abstract()`, the shapes and placements it would hold, of `variables`'
        shapes, nothing moved, what a plan reads before the state is placed (`compiled_whole`); `_Collected`,
        those shapes with its banks' stacks compiled (`collected`).

        The frozen split is undone by reference, so a bank holds every one of
        a layer's parameter leaves and a partially frozen run does not stack
        two sparse trees of different shape. The canonical state keeps its
        own `params` and `frozen` collections; this merged view exists only
        for the duration of one realization. A frozen leaf already sits
        where the snapshot puts it, so `transfer` hands it back as it is; a
        frozen bank (`banked`) is read as the state holds it, and the rows
        stacked here are the moving ones and the frozen leaves only some
        rows hold.
        """
        if variables is None:
            return None
        moved = self if moved is None else moved
        if not self.sites:
            return moved.placed(variables, self.layout.shardings(self.accelerator, variables))
        # The layout names a leaf's axes as its module declares them, which a
        # bank's leading layer axis is not; a bank is already placed, so the
        # placement is asked about everything but the banks.
        placement = self.layout.shardings(self.accelerator, _without_banks(variables, self.sites))
        return self._assembled(variables, placement, self.sites, moved)

    @staticmethod
    def placed(tree, placement):
        """`tree` moved where `placement` places it (`transfer`)."""
        return transfer(tree, placement)

    def _assembled(self, variables, placement, sites, moved):
        """`snapshot` of `variables` placed by `placement`: the stacks of `sites` in their own tree as
        banks, each nested root's (`DecoderBank.root`) assembled the same way in the collection holding
        it, and everything else as it is stored, each moved by `moved` (`placed`, `_stack_rows`)."""
        nested, own = _nested(sites), _own(sites)
        whole, placement = thaw(variables), thaw(placement)
        weights = {"params": whole["params"]}
        places = {"params": placement["params"]}
        # Objective-owned mutable collections retain their logical paths:
        # router effects, for example, match per-layer sows to per-layer bias.
        # Only model weights adopt execution banks.
        entries = {**{collection: tree for collection, tree in whole.items()
                      if collection != "params" and collection not in nested}, **entry_tree(weights, own)}
        store = moved.placed(entries, narrowed(placement, entries))

        def bank(site, first: int, count: int, name: str):
            """A group's bank from the state: its rows stacked on the CPU into the bank's placement
            (`_stack_rows`) over the frozen bank the state holds, or its one layer placed as it is."""
            banks = in_namespace(weights, site.namespace)["params"]
            rows = [one_layer(weights, index, namespace=site.namespace)
                    for index in range(first, first + count)]
            held = {"params": banks[name]} if count > 1 and name in banks else {}
            if not rows[0]:
                return held or None
            spread = _bank_shardings(one_layer(places, first, namespace=site.namespace),
                                     self.accelerator, count)
            if count == 1:
                return moved.placed(rows[0], spread)
            return merge(held, jax.tree_util.tree_map_with_path(
                lambda path, sharding, *leaves: moved._stack_rows(leaves, sharding), spread, *rows))

        return {**merge(store, assembled(own, bank)),
                **{collection: self._assembled(variables[collection], placement[collection],
                                               _within(sites, collection), moved)
                   for collection in nested if collection in variables}}

    def _stack_rows(self, rows, sharding):
        """Stack the rows of one bank leaf into the bank's placement.

        The stack runs on the CPU backend whatever the rows' placement: a
        moving row is the CPU's already, and a frozen row sits in the
        accelerator's pinned host memory, where a stack traced against the
        state mesh cannot reach it and a stack on the accelerator would
        materialise the whole bank in device memory, which is exactly what
        the bank's placement avoids. One bank crosses host memory at a time.
        """
        return transfer(self._stacked(*self.on_cpu(rows)), sharding)

    @staticmethod
    def _moved(tree, mesh):
        """Move `tree` to `mesh`, each leaf keeping the partition spec it has (`spec_on`)."""
        return transfer(tree, jax.tree.map(spec_on(mesh), tree))

    @staticmethod
    def rows(index, batch):
        """`index`, an entry for each of `batch`'s rows, placed as its rows are (`placed_as_rows`)."""
        return placed_as_rows(index, batch)

    def on_cpu(self, tree):
        """Move `tree` to the CPU mesh, where the state lives."""
        return self._moved(tree, self.cpu)

    def on_accelerator(self, tree):
        """Move `tree` to the accelerator mesh, where the loss runs."""
        return self._moved(tree, self.accelerator)

    def inputs(self, variables, batch, step, *, moved=None):
        """What one realization hands the accelerator (`realize`): the moving leaves of the state's snapshot
        (`snapshot`), the vjp's primal; its frozen leaves, rows and banks, the state's own arrays, which the
        vjp reads as values; its other collections; the batch placed on the accelerator; and the step, which
        crosses whole, so its examples' draws are the resident step's (`Step.sample`), its averaged weights
        the snapshot's and its places the batch's rows'; with the layout `_trainable_cotangent` reads.
        Each moved by `moved` (this execution, where None; `snapshot`): `_Abstract()`, their shapes, placed
        where the moves would place them, nothing moved, what a plan compiles before anything is placed
        (`compiled_whole`)."""
        moved = self if moved is None else moved
        with jax.set_mesh(self.cpu):
            store = self.snapshot(variables, moved=moved)
            ema = self.snapshot(step.ema, moved=moved)
        assert store is not None, "a realization always receives model variables"
        # A frozen leaf by its path: the frozen collection's leaves stand at their own paths in the
        # snapshot's parameters (`thaw`), a stack's banked rows at their bank's, which the vjp differentiates
        # as its moving rows (`snapshot`): one answer over arrays and over their shapes alike, which a run's
        # startup pass reads (`collected`).
        frozen_paths = {path for path, _ in
                        jax.tree_util.tree_flatten_with_path(variables.get(FROZEN, {}))[0]}
        named, treedef = jax.tree_util.tree_flatten_with_path(store["params"])
        leaves = [leaf for _, leaf in named]
        held = tuple(path in frozen_paths for path, _ in named)
        primal = [leaf for leaf, known in zip(leaves, held, strict=True) if not known]
        frozen = [leaf for leaf, known in zip(leaves, held, strict=True) if known]
        with on_mesh(self.accelerator):
            batch = moved.placed(batch, batch_shardings(self.accelerator, batch))
            here = moved._moved(step.replace(ema=None), self.accelerator).replace(ema=ema)
            if here.example_index is not None:
                here = here.replace(example_index=moved.rows(here.example_index, batch))
        rest = {name: tree for name, tree in store.items() if name != "params"}
        return primal, frozen, rest, batch, here, (treedef, held)

    def compiled_whole(self, variables, batch, step) -> tuple[jax.stages.Compiled | None, tuple]:
        """The accelerator's work of one realization and its pullback as one program over the shapes of
        `variables`, `batch` and `step` as a realization hands them over (`inputs`), compiled for the
        accelerator under the step's options (`step_options`) as a plan's candidates compile
        (`compiled_if_it_fits`: None where XLA refuses it for memory): what a plan reads of a step that
        streams its parameters from the host before anything is placed
        (`dew.training.memory.streamed_price`), its FLOPs the forward's and the pullback's and its memory the
        forward's residuals held into the pullback, as the two programs a realization runs hold them; and
        those inputs, which an admission counts."""
        primal, frozen, rest, placed, here, layout = self.inputs(variables, batch, step, moved=_Abstract())
        moving = tuple(tuple(entry.key for entry in path) for path, _ in
                       jax.tree_util.tree_leaves_with_path(variables["params"]))
        loss = functools.partial(_after, self.device, self.objective._loss)

        # A watching step's pullback hands its probes' cotangents beside the moving leaves', as a sampled
        # step's realization does (`_realized`'s `probed`): one program holding both, its outputs counted.
        pulled = (_probed_cotangent if step.watched is not None and step.watched.cotangents
                  else _trainable_cotangent)

        def whole(primal, frozen, rest, batch, step):
            stats, back, aux = _forward(loss, primal, frozen, rest, batch, step, layout=layout)
            return stats, aux, pulled(back, jax.tree.map(jnp.ones_like, stats), sites=self.own, paths=moving,
                                      layout=layout)

        with on_mesh(self.accelerator):
            compiled = owned_program(whole, step=self.step_options(variables, batch))
            arguments = (primal, frozen, rest, placed, here)
            refusals: list[RuntimeError] = []
            executable = compiled_if_it_fits(compiled.lower(*arguments), compiled.options, refusals,
                                             relower=lambda: compiled.lower(*arguments))
        return executable, arguments

    def realize(self, variables, batch, step):
        """Evaluate the objective on the accelerator, over a snapshot of the state (`inputs`).

        The moving leaves are the vjp's primal; every frozen leaf, row or
        bank enters as a value. The forward then keeps nothing for a
        cotangent no optimizer reads. The statistics and the report come home
        to the CPU, while the pullback stays here and runs on the accelerator
        when it is called.
        """
        # A host step's pass over its state's shapes (its run's startup, `options.collecting`) realizes
        # over shapes: every program it runs compiled there (`collected`).
        if isinstance(jax.tree.leaves(variables)[0], jax.ShapeDtypeStruct):
            return self.collected(variables, batch, step)
        return self._realized(variables, batch, step, self)

    def collected(self, variables, batch, step) -> Realization:
        """`realize` over the shapes of `variables`, `batch` and `step` within `options.collecting`: every
        program a realization runs (its banks' stacks, the forward, the unstack of its collections and,
        once called, the pullback) compiled over them as a realization over arrays of those shapes calls
        it, the run then stepping on them (`OuterProgram`'s `_shaped`), its results' shapes in place of
        values (`_Collected`): a host step's startup pass (`dew.training.compiler`)."""
        return self._realized(variables, batch, step, _Collected(self))

    def _realized(self, variables, batch, step, moved) -> Realization:
        """`realize`'s body, each move made by `moved` (`inputs`): one path over arrays and over shapes."""
        moving = tuple(tuple(entry.key for entry in path) for path, _ in
                       jax.tree_util.tree_leaves_with_path(variables["params"]))
        primal, frozen, rest, batch, here, layout = self.inputs(variables, batch, step, moved=moved)
        watched = step.watched
        # A watching step's loss scale, which its probes unscale their cotangents by, crosses with it
        # (`_forward`); a step that watches none reads none, its program the one an uninspected run runs.
        scale = None
        if watched is not None and watched.cotangents and (held := unscaling()) is not None:
            scale = moved._moved(held, self.accelerator)
        with on_mesh(self.accelerator):
            options = self.step_options(variables, batch)
            stats, back, aux = self.forward(options)(primal, frozen, rest, batch, here, scale, layout=layout)
        stats, aux = moved._moved((stats, aux), self.cpu)
        if aux.variables is not None:
            with jax.set_mesh(self.cpu):
                aux = dataclasses.replace(aux, variables=self.unstack(aux.variables))

        def pulled_by(cotangent_of):
            def pulled(cotangent):
                # A shared owner is one bank input, so native reverse mode already
                # sums every use's contribution into it.
                with on_mesh(self.accelerator):
                    # The pullback runs its int8 dots here, apart from the forward's program, with an
                    # int8 backward (`Quantization.bwd_qtype`) on the options that ask for it.
                    compiled = program(cotangent_of, step=options,
                                       static_argnames=("sites", "paths", "layout"))
                    found = compiled(back, moved._moved(cotangent, self.accelerator), sites=self.own,
                                     paths=moving, layout=layout)
                return moved._moved(found, self.cpu)
            return pulled

        probed = (None if watched is None or not watched.cotangents
                  else pulled_by(_probed_cotangent))
        return Realization(stats, aux, pulled_by(_trainable_cotangent), probed)


def banked_frozen(objective, variables, stack, release=None):
    """`variables` with each scanned run's shared leaves of every frozen tree (`frozen_trees`) stacked
    into one bank (`dew.inference.banks.banked`), the runs `objective`'s stacks declare (`stack_sites`):
    `stack(rows, path)` is given each bank's path from the root of `variables`, and
    `release(tree, namespace, index, keys, bank)` the path of the frozen tree a row left.

    That is the shape a placement and a checkpoint template take, and the
    arrays the state holds."""
    for path, tree, sites in tuple(frozen_trees(variables, stack_sites(objective, variables))):
        variables = placed_at(variables, path, banked(
            tree, sites, lambda rows, keys, path=path: stack(rows, (*path, *keys)),
            None if release is None else functools.partial(release, path)))
    return variables


def placed_host(objective, fresh: Callable[[], TrainState], state_mesh: Mesh, device_mesh: Mesh,
                shardings: Placement[TrainState]) -> TrainState:
    """A fresh CPU-owned state, `fresh()` built on `state_mesh`'s first device, its leaves streamed into
    place one at a time.

    The tree is built eagerly on the CPU, where the held leaves the
    objective hands over are read by reference and nothing of size is
    computed. Each leaf is then moved to its placement and the source let
    go as it lands (`host.stream`): a moving leaf to the companion, a
    frozen one to where it stays resident (`resident`). The transient is
    one leaf, not the tree, and one JIT could not have returned to the two
    device sets anyway.
    """
    from dew.distributed.host import place_leaf, stream
    from dew.distributed.pages import evict
    held = objective.held_variables()
    with jax.default_device(state_mesh.local_devices[0]):
        state = fresh()
        trees = tuple(frozen_trees(state.variables, stack_sites(objective, state.variables)))
        if trees:
            check_bank_pool(sum(bank_bytes(tree, sites) for _, tree, sites in trees), device_mesh)
            # A scanned run's frozen rows become its bank here and the bank lands where it stays
            # resident before the next is stacked, so the host holds one bank in transit, not the
            # stack; the rows' pages are let go as it lands and the held tree holds the placed bank
            # where each row was. An objective placed this way holds banks at its rows afterwards and
            # does not seed a second trainer.
            def stack(rows, path):
                target = at_path(shardings.variables, path)
                assert isinstance(target, jax.sharding.Sharding), path
                bank = place_leaf(jnp.stack(rows), target)
                for row in rows:
                    evict(np.asarray(row))
                return bank

            def release(tree, namespace, index, keys, bank):
                # The rows sit under the held tree's own frozen split when it came split, or under its
                # params when init split it; a nested model's, under its own params (`frozen_trees`).
                node = (None if held is None else held.get(FROZEN, held.get("params")) if tree == (FROZEN,)
                        else at_path(held, tree))
                for component in (*namespace, f"layers_{index}", *keys[:-1]):
                    node = node.get(component) if isinstance(node, dict) else None
                if isinstance(node, dict) and keys[-1] in node:
                    node[keys[-1]] = bank
            state = dataclasses.replace(
                state, variables=banked_frozen(objective, state.variables, stack, release))
    params = stream(state.variables, shardings.variables, held)
    ema = None if state.ema is None else stream(state.ema, shardings.ema)
    rest = dataclasses.replace(state, variables=None, ema=None)
    placed = transfer(rest, dataclasses.replace(shardings, variables=None, ema=None))
    return dataclasses.replace(placed, variables=params, ema=ema)
