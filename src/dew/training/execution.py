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


def resident(placement, sites, accelerator):
    """The frozen collection's placement beside the accelerator, from the
    specs the layout gave it: a stack's layers in bank memory, the rest in
    device memory, each leaf the shard the layout named; and the leaves a
    scanned run holds in every row placed as that run's bank, with the
    layer axis in front (`banked`).

    Only a scanned stack is placed in bank memory: its scan fetches one row
    per iteration, so the device holds one layer of the bank at a time. A
    plain loop's fetches have no order between them, the scheduler hoists
    them all to the front, and the whole stack lands on the device, which
    is the memory the layout was asked to avoid; it is refused by name."""
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
    """`tree`, a frozen collection keyed per layer, with every leaf that all
    rows of a scanned run hold moved under the run's bank name as
    `stack(rows, path)`: arrays stack into one bank, shapes into one shape,
    shardings into the bank's, `path` the bank leaf's keys from the root so
    a caller can place the bank as it makes it. A leaf only some rows hold
    stays per layer, as does every run of one, and `run_stack` stacks those
    with the moving rows at each snapshot.

    So a frozen bank exists once, as the bank the scan reads, from the
    moment it is placed: not as its rows in pinned memory and a stacked
    copy beside them, which for a stack that fills the host is the second
    copy that does not fit. `release(namespace, index, keys, bank)` is told
    each row leaf the bank replaced and the bank itself, so the tree the rows
    came from can hold the bank where the row was and let the row go.
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
    """`variables` without the banks a scanned run stores under its name."""
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
    """`tree` without the leaf at `keys`, and without the nodes that emptied."""
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
    """Every declared stack's banks back under their stored `layers_N` paths."""
    for site in sites:
        local = in_namespace(tree, site.namespace)
        if local:
            tree = _replaced(tree, site.namespace, site.view.unstack(local))
    return tree


def _selected(tree, paths):
    """The leaves at `paths`, which are the canonical trainable ones."""
    selected: dict = {}
    for path in paths:
        node, target = tree, selected
        for name in path[:-1]:
            node, target = node[name], target.setdefault(name, {})
        target[path[-1]] = node[path[-1]]
    return selected


@functools.partial(jax.jit, static_argnames=("sites", "paths", "layout"))
def _trainable_cotangent(back, cotangent, *, sites, paths, layout):
    """One accelerator computation: the pullback, the unstack and the selection.

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
    """Own snapshot placement and attempt-local tapes, not another TrainState."""
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
        """The rows of one bank leaf stacked into the bank's placement.

        The stack runs on the CPU backend whatever the rows' placement: a
        moving row is the CPU's already, and a frozen row sits in the
        accelerator's pinned host memory, where a stack traced against the
        state mesh cannot reach it and a stack on the accelerator would
        materialise the whole bank in device memory, which is exactly what
        the bank's placement avoids. One bank crosses host memory at a time.
        """
        return transfer(jnp.stack(self.on_cpu(rows)), sharding)

    def on_cpu(self, tree):
        def placement(leaf):
            sharding = getattr(leaf, "sharding", None)
            spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
            return NamedSharding(self.cpu, spec)
        return transfer(tree, jax.tree.map(placement, tree))

    def on_accelerator(self, tree):
        def placement(leaf):
            sharding = getattr(leaf, "sharding", None)
            spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
            return NamedSharding(self.accelerator, spec)
        return transfer(tree, jax.tree.map(placement, tree))

    def realize(self, variables, batch, step):
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
