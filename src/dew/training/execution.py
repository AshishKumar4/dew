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
import os
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.inference.banks import bank_sites, entry_tree, in_namespace, layer_index, narrowed, one_layer
from dew.nn.backbones.causal_transformer import group_name
from dew.objectives.base import FROZEN, Step, merge, thaw
from dew.training.distributed import batch_shardings
from dew.training.host import transfer
from dew.training.transaction import Realization

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
    return banked(placed, sites, lambda rows, path: NamedSharding(
        accelerator, P(None, *rows[0].spec), memory_kind=BANK_MEMORY))


def banked(tree, sites, stack, release=None):
    """Replace a scanned run's per-layer rows with one stacked bank.

    `tree` is a frozen collection keyed per layer. Every leaf that all rows
    of a run hold moves under the run's bank name, stacked by
    `stack(rows, path)`. Arrays stack into one bank, shapes into one shape,
    shardings into the bank's. A leaf only some rows hold stays per layer, as
    does every run of one, and `run_stack` stacks those with the moving rows
    at each snapshot.

    `stack` is called with the rows and `path`, the bank leaf's keys from the
    root, so a caller can place each bank as it makes it. A frozen bank then
    exists once, from the moment it is placed, rather than as its rows in
    pinned memory and a stacked copy beside them, which for a stack that
    fills the host is the second copy that does not fit.

    `release(namespace, index, keys, bank)` is told each row leaf the bank
    replaced, and the bank itself. The tree the rows came from can then hold
    the bank where the row was and let the row go.
    """
    tree = dict(tree)
    for site in sites:
        local = tree
        for component in site.namespace:
            if not isinstance(local.get(component), Mapping):
                local = None
                break
            local[component] = dict(local[component])
            local = local[component]
        if local is None:
            continue
        for first, count in site.view.groups:
            if count == 1:
                continue
            held = [local.get(f"layers_{first + offset}") for offset in range(count)]
            if not all(isinstance(row, Mapping) for row in held):
                continue
            rows = [dict(row) for row in held if isinstance(row, Mapping)]
            leaves = [{tuple(entry.key for entry in path): leaf
                       for path, leaf in jax.tree_util.tree_leaves_with_path(row)} for row in rows]
            shared = set(leaves[0]).intersection(*leaves[1:])
            if not shared:
                continue
            name = group_name(first, count)
            bank = local.setdefault(name, {})
            for keys in sorted(shared):
                node = bank
                for key in keys[:-1]:
                    node = node.setdefault(key, {})
                node[keys[-1]] = stack([held[keys] for held in leaves], (*site.namespace, name, *keys))
                for offset, row in enumerate(rows):
                    _drop(row, keys)
                    if release is not None:
                        release(site.namespace, first + offset, keys, node[keys[-1]])
            for offset, row in enumerate(rows):
                name = f"layers_{first + offset}"
                if row:
                    local[name] = row
                else:
                    del local[name]
    return tree


def _without_banks(variables, sites):
    """Return `variables` without the banks a scanned run stores under its name."""
    banks = {(*site.namespace, name) for site in sites
             for (first, count), name in zip(site.view.groups, site.view.bank_names(), strict=True)
             if count > 1}

    def prune(tree, path):
        pruned = {}
        for name, value in tree.items():
            current = (*path, name)
            if current in banks:
                continue
            pruned[name] = prune(value, current) if isinstance(value, Mapping) else value
        return pruned

    return {collection: prune(tree, ()) for collection, tree in variables.items()}


def _drop(tree, keys):
    """Delete the leaf at `keys` from `tree`, and any node that emptied with it."""
    node, parents = tree, []
    for key in keys[:-1]:
        parents.append((node, key))
        # The copy is written into the parent before the walk moves on; a
        # chained assignment would bind `node` first and write into the copy.
        child = dict(node[key])
        node[key] = child
        node = child
    del node[keys[-1]]
    for parent, key in reversed(parents):
        if parent[key]:
            break
        del parent[key]


def _bank_shardings(placed, accelerator, count):
    """Place one bank in bank memory, layer axis in front when it stacks rows."""
    return jax.tree.map(
        lambda s: NamedSharding(accelerator, P(None, *s.spec) if count > 1 else s.spec,
                                memory_kind=BANK_MEMORY), placed)


def _replaced(tree, namespace, subtrees):
    """`tree` with each collection's namespace subtree replaced, spines copied."""
    replaced = dict(tree)
    for collection, subtree in subtrees.items():
        if not namespace:
            replaced[collection] = subtree
            continue
        spine = dict(replaced[collection])
        replaced[collection] = spine
        node = spine
        for name in namespace[:-1]:
            node[name] = dict(node[name])
            node = node[name]
        node[namespace[-1]] = subtree
    return replaced


def _logical(tree, sites):
    """Put every declared stack's banks back under their stored `layers_N` paths."""
    for site in sites:
        local = in_namespace(tree, site.namespace)
        if local:
            tree = _replaced(tree, site.namespace, site.view.unstack(local))
    return tree


def _selected(tree, paths):
    """Pick out the leaves at `paths`, which are the canonical trainable ones."""
    selected: dict = {}
    for path in paths:
        node, target = tree, selected
        for name in path[:-1]:
            node, target = node[name], target.setdefault(name, {})
        target[path[-1]] = node[path[-1]]
    return selected


@functools.partial(jax.jit, static_argnames=("sites", "paths", "layout"))
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
    gradient = _logical({"params": placed}, sites)["params"]
    return _selected(gradient, paths)


class HostExecution:
    """Runs one objective evaluation and its pullback on the accelerator.

    The CPU owns the state. This class places a snapshot of it on the
    accelerator, runs the loss there, and hands back a pullback. It keeps no
    state of its own between attempts.
    """
    def __init__(self, objective, layout, accelerator, cpu):
        self.objective, self.layout = objective, layout
        self.accelerator, self.cpu = accelerator, cpu
        self.sites = bank_sites(objective) if objective.bank_sites else ()
        self.loss = jax.jit(objective.loss)
        self.unstack = jax.jit(functools.partial(_logical, sites=self.sites))

    def resident(self, placement):
        """Where the frozen collection lives for the run, given its layout specs."""
        return resident(placement, self.sites, self.accelerator)

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
        placement = self.layout.shardings(self.accelerator, _without_banks(variables, self.sites))
        whole, placement = thaw(variables), thaw(placement)
        weights = {"params": whole["params"]}
        places = {"params": placement["params"]}
        # Objective-owned mutable collections retain their logical paths:
        # router effects, for example, match per-layer sows to per-layer bias.
        # Only model weights adopt execution banks.
        entries = {**{collection: tree for collection, tree in whole.items()
                      if collection != "params"}, **entry_tree(weights, self.sites)}
        store = transfer(entries, narrowed(placement, entries))
        for site in self.sites:
            banks = in_namespace(weights, site.namespace)["params"]
            for (first, count), name in zip(site.view.groups, site.view.bank_names(), strict=True):
                rows = [one_layer(weights, index, namespace=site.namespace)
                        for index in range(first, first + count)]
                held = {"params": banks[name]} if count > 1 and name in banks else {}
                if not rows[0] and not held:
                    continue
                if count == 1:
                    bank = transfer(rows[0], _bank_shardings(
                        one_layer(places, first, namespace=site.namespace), self.accelerator, count))
                elif rows[0]:
                    spread = _bank_shardings(one_layer(places, first, namespace=site.namespace),
                                             self.accelerator, count)
                    bank = jax.tree_util.tree_map_with_path(
                        lambda path, sharding, *leaves: self._stack_rows(leaves, sharding), spread, *rows)
                    bank = merge(held, bank)
                else:
                    bank = held
                bank = jax.block_until_ready(bank)
                for collection, values in bank.items():
                    branch = store.setdefault(collection, {})
                    for component in site.namespace:
                        branch = branch.setdefault(component, {})
                    branch[name] = values
        return store

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
        layout = (treedef, held)
        with jax.set_mesh(self.accelerator):
            batch = transfer(batch, batch_shardings(self.accelerator, batch))
            clock, key = self.on_accelerator((step.step, step.key))
            execution_info = Step(clock, key, ema)

            def loss(trainable):
                moved = iter(trainable)
                merged = treedef.unflatten([leaf if known else next(moved)
                                            for leaf, known in zip(leaves, held, strict=True)])
                return self.loss({**store, "params": merged}, batch, execution_info)
            stats, back, aux = jax.vjp(loss, primal, has_aux=True)
        stats, aux = self.on_cpu((stats, aux))
        if aux.variables is not None:
            with jax.set_mesh(self.cpu):
                aux = dataclasses.replace(aux, variables=self.unstack(aux.variables))

        def pullback(cotangent):
            # A shared owner is one bank input, so native reverse mode already
            # sums every use's contribution into it.
            with jax.set_mesh(self.accelerator):
                gradient = _trainable_cotangent(
                    back, self.on_accelerator(cotangent), sites=self.sites, paths=moving,
                    layout=layout)
            return self.on_cpu(gradient)
        return Realization(stats, aux, pullback)
