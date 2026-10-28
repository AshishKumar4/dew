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
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.batch import FROZEN, merge, select, thaw
from dew.distributed.batches import batch_shardings
from dew.distributed.host import transfer
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
from dew.nn.backbones.layer_plan import layer_index
from dew.nn.kernels.options import Option, program
from dew.training.memory import device_tokens, step_compiler_options
from dew.training.transaction import Realization, placed_as_rows

if TYPE_CHECKING:
    from dew.data.pipeline import DeviceStep
    from dew.distributed.placement import Placement
    from dew.state import TrainState

BANK_MEMORY = "pinned_host"
"""Where a layer's bank sits beside the accelerator: the stack fetches one
layer of it at a time as it reaches it (`run_stack`). The CPU backend has
the space too, so a CPU-only run exercises the same fetching loop."""


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


def resident(placement, sites, accelerator):
    """Return where the frozen collection sits beside the accelerator.

    A stack's layers go to bank memory and the rest to device memory, each
    leaf on the shard the layout named. The leaves a scanned run holds in
    every row are placed as that run's bank, layer axis in front (`banked`).

    Only a scanned stack is placed in bank memory: its scan fetches one row
    per iteration, so the device holds one layer of the bank at a time. A
    plain loop's fetches have no order between them, so the scheduler hoists
    them all to the front and the whole stack lands on the device. That is
    the memory the layout was asked to avoid, so it is refused by name."""
    unscanned = [".".join(site.namespace) or "<root>" for site in sites if not site.scanned]
    if unscanned:
        raise ValueError(
            "a host layout streams a scanned stack; the decoder at "
            f"{', '.join(unscanned)} runs a plain loop (scan_layers=False), whose layer "
            "fetches the compiler hoists together, so set scan_layers=True (and bank_layers "
            "to bound a run) on it")
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


def _forward(loss, primal, frozen, store, batch, execution, *, layout):
    """`loss` of the moving leaves `primal` beside the `frozen` ones, the
    store's other collections, the batch and the step's clock and keys, with
    its pullback. Compiled once for its objective (`HostExecution`), it hands
    back a pullback whose static part is the same program each realization,
    so `_trainable_cotangent` traces once rather than once an attempt."""
    treedef, held = layout

    def evaluate(trainable):
        moved, kept = iter(trainable), iter(frozen)
        merged = treedef.unflatten([next(kept) if known else next(moved) for known in held])
        return loss({**store, "params": merged}, batch, execution)
    return jax.vjp(evaluate, primal, has_aux=True)


def _trainable_cotangent(back, cotangent, *, sites, paths, layout):
    """Run the pullback, the unstack and the selection as one computation.

    `back` crosses this boundary as its own pytree, so the residuals are
    arguments rather than captured constants. It returns one cotangent per
    moving store leaf; `layout` (the store's treedef and which of its leaves
    are held) puts them back in the store's shape, a held leaf's place
    empty, so a mixed bank's frozen rows are the dead values the compiler
    drops and its moving rows are selected under their logical paths.
    """
    treedef, held = layout
    cotangents = iter(back(cotangent)[0])
    placed = treedef.unflatten([None if known else next(cotangents) for known in held])
    gradient = logical({"params": placed}, sites)["params"]
    trainable = frozenset(paths)
    return select(gradient, lambda path: path in trainable)


class HostExecution:
    """Runs one objective evaluation and its pullback on the accelerator.

    The CPU owns the state. This class places a snapshot of it on the
    accelerator, runs the loss there, and hands back a pullback. It keeps no
    state of its own between attempts.
    """
    def __init__(self, objective, layout, accelerator, cpu, sites: tuple[DecoderBank, ...],
                 device: tuple[DeviceStep, ...] = ()):
        self.objective, self.layout = objective, layout
        self.accelerator, self.cpu = accelerator, cpu
        # The trainer's banks over the state it owns (`stack_sites`), the runs its frozen banks keep.
        self.sites = sites
        self.device = device
        self.unstack = functools.partial(program(logical, static_argnames="sites"), sites=self.sites)

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
                                        FROZEN in variables, self.accelerator.devices.size)
        return dict(options or {})

    def snapshot(self, variables):
        """The declared stacks as banks, everything else as it is stored.

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
        if not self.sites:
            return transfer(variables, self.layout.shardings(self.accelerator, variables))
        # The layout names a leaf's axes as its module declares them, which a
        # bank's leading layer axis is not; a bank is already placed, so the
        # placement is asked about everything but the banks.
        placement = self.layout.shardings(self.accelerator, without_banks(variables, self.sites))
        whole, placement = thaw(variables), thaw(placement)
        weights = {"params": whole["params"]}
        places = {"params": placement["params"]}
        # Objective-owned mutable collections retain their logical paths:
        # router effects, for example, match per-layer sows to per-layer bias.
        # Only model weights adopt execution banks.
        entries = {**{collection: tree for collection, tree in whole.items()
                      if collection != "params"}, **entry_tree(weights, self.sites)}
        store = transfer(entries, narrowed(placement, entries))

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
                return transfer(rows[0], spread)
            return merge(held, jax.tree_util.tree_map_with_path(
                lambda path, sharding, *leaves: self._stack_rows(leaves, sharding), spread, *rows))

        return merge(store, assembled(self.sites, bank))

    def _stack_rows(self, rows, sharding):
        """Stack the rows of one bank leaf into the bank's placement.

        The stack runs on the CPU backend whatever the rows' placement: a
        moving row is the CPU's already, and a frozen row sits in the
        accelerator's pinned host memory, where a stack traced against the
        state mesh cannot reach it and a stack on the accelerator would
        materialise the whole bank in device memory, which is exactly what
        the bank's placement avoids. One bank crosses host memory at a time.
        """
        return transfer(jnp.stack(self.on_cpu(rows)), sharding)

    def _moved(self, tree, mesh):
        """Move `tree` to `mesh`, each leaf keeping the partition spec it has.

        A leaf with no named sharding is replicated, which is what a scalar
        clock or a key crossing the boundary asks for.
        """
        def placement(leaf):
            sharding = getattr(leaf, "sharding", None)
            spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
            return NamedSharding(mesh, spec)
        return transfer(tree, jax.tree.map(placement, tree))

    def on_cpu(self, tree):
        """Move `tree` to the CPU mesh, where the state lives."""
        return self._moved(tree, self.cpu)

    def on_accelerator(self, tree):
        """Move `tree` to the accelerator mesh, where the loss runs."""
        return self._moved(tree, self.accelerator)

    def realize(self, variables, batch, step):
        """Evaluate the objective on the accelerator, over a snapshot of the state.

        The moving leaves are the vjp's primal; every frozen leaf, row or
        bank enters as a value. The forward then keeps nothing for a
        cotangent no optimizer reads. The statistics and the report come home
        to the CPU, while the pullback stays here and runs on the accelerator
        when it is called.
        """
        with jax.set_mesh(self.cpu):
            store = self.snapshot(variables)
            # A frozen leaf, row or bank, is the state's own array: those the
            # vjp reads as values.
            resident = {id(leaf) for leaf in jax.tree.leaves(variables.get(FROZEN, {}))}
            ema = self.snapshot(step.ema)
        assert store is not None, "a realization always receives model variables"
        moving = tuple(tuple(entry.key for entry in path) for path, _ in
                       jax.tree_util.tree_leaves_with_path(variables["params"]))
        leaves, treedef = jax.tree.flatten(store["params"])
        held = tuple(id(leaf) in resident for leaf in leaves)
        primal = [leaf for leaf, known in zip(leaves, held, strict=True) if not known]
        frozen = [leaf for leaf, known in zip(leaves, held, strict=True) if known]
        layout = (treedef, held)
        with jax.set_mesh(self.accelerator):
            batch = transfer(batch, batch_shardings(self.accelerator, batch))
            options = self.step_options(variables, batch)
            # The step crosses whole, so its examples' draws are the resident step's (`Step.sample`);
            # its averaged weights are the snapshot's, and its places the batch's rows'.
            here = self.on_accelerator(step.replace(ema=None)).replace(ema=ema)
            if here.example_index is not None:
                here = here.replace(example_index=placed_as_rows(here.example_index, batch))
            rest = {name: tree for name, tree in store.items() if name != "params"}
            stats, back, aux = self.forward(options)(primal, frozen, rest, batch, here, layout=layout)
        stats, aux = self.on_cpu((stats, aux))
        if aux.variables is not None:
            with jax.set_mesh(self.cpu):
                aux = dataclasses.replace(aux, variables=self.unstack(aux.variables))

        def pullback(cotangent):
            # A shared owner is one bank input, so native reverse mode already
            # sums every use's contribution into it.
            with jax.set_mesh(self.accelerator):
                # The pullback runs its int8 dots here, apart from the forward's program, with an
                # int8 backward (`Quantization.bwd_qtype`) on the options that ask for it.
                pulled = program(_trainable_cotangent, *self.objective.towers(), step=options,
                                 static_argnames=("sites", "paths", "layout"))
                gradient = pulled(back, self.on_accelerator(cotangent), sites=self.sites, paths=moving,
                                  layout=layout)
            return self.on_cpu(gradient)
        return Realization(stats, aux, pullback)


def banked_frozen(objective, variables, stack, release=None):
    """Stack each scanned run's shared leaves of the frozen collection of `variables` into one bank
    (`dew.inference.banks.banked`), the runs `objective`'s stack declares (`stack_sites`).

    That is the shape a placement and a checkpoint template take, and the
    arrays the state holds."""
    return banked(variables[FROZEN], stack_sites(objective, variables), stack, release)


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
    from dew.distributed.host import evict, place_leaf, stream
    held = objective.held_variables()
    with jax.default_device(state_mesh.local_devices[0]):
        state = fresh()
        if FROZEN in state.variables:
            sites = stack_sites(objective, state.variables)
            check_bank_pool(bank_bytes(state.variables[FROZEN], sites), device_mesh)
            # A scanned run's frozen rows become its bank here and the bank lands where it stays
            # resident before the next is stacked, so the host holds one bank in transit, not the
            # stack; the rows' pages are let go as it lands and the held tree holds the placed bank
            # where each row was. An objective placed this way holds banks at its rows afterwards and
            # does not seed a second trainer.
            targets = shardings.variables[FROZEN]

            def stack(rows, path):
                target = targets
                for component in path:
                    target = target[component]
                bank = place_leaf(jnp.stack(rows), target)
                for row in rows:
                    evict(np.asarray(row))
                return bank

            def release(namespace, index, keys, bank):
                # The rows sit under the held tree's own frozen split
                # when it came split, or under its params when init split it.
                node = None if held is None else held.get(FROZEN, held.get("params"))
                for component in (*namespace, f"layers_{index}", *keys[:-1]):
                    node = node.get(component) if isinstance(node, dict) else None
                if isinstance(node, dict) and keys[-1] in node:
                    node[keys[-1]] = bank
            frozen = banked_frozen(objective, state.variables, stack, release)
            state = dataclasses.replace(state, variables={**state.variables, FROZEN: frozen})
    params = stream(state.variables, shardings.variables, held)
    ema = None if state.ema is None else stream(state.ema, shardings.ema)
    rest = dataclasses.replace(state, variables=None, ema=None)
    placed = transfer(rest, dataclasses.replace(shardings, variables=None, ema=None))
    return dataclasses.replace(placed, variables=params, ema=ema)
