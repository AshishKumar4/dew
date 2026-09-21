"""An accelerator realization over immutable banks of a CPU-owned state.

Only objective evaluation and its pullback cross this boundary. The optimizer,
accumulation and checkpoint trees retain their original logical leaf identities.
Each snapshot bank is assembled and transferred before the next is built; no
whole parameter tree is ever materialized on the accelerator.
"""
from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.inference.banks import bank_sites, entry_tree, in_namespace, narrowed, one_layer
from dew.objectives.base import Step, thaw
from dew.training.distributed import batch_shardings
from dew.training.host import transfer
from dew.training.transaction import Realization

_stack = jax.jit(lambda *rows: jax.tree.map(lambda *leaves: jnp.stack(leaves), *rows))


def _replaced(tree, namespace, subtrees):
    """`tree` with each collection's namespace subtree replaced, spines copied."""
    result = dict(tree)
    for collection, subtree in subtrees.items():
        if not namespace:
            result[collection] = subtree
            continue
        spine = dict(result[collection])
        result[collection] = spine
        node = spine
        for name in namespace[:-1]:
            node[name] = dict(node[name])
            node = node[name]
        node[namespace[-1]] = subtree
    return result


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


@functools.partial(jax.jit, static_argnames=("sites", "paths"))
def _trainable_cotangent(back, cotangent, *, sites, paths):
    """One accelerator computation: the pullback, the unstack and the selection.

    `back` crosses this boundary as its own pytree, so the residuals are
    arguments rather than captured constants, and the cotangents of frozen
    rows are dead values the compiler can drop instead of arrays a caller
    receives and throws away.
    """
    gradient = _logical({"params": back(cotangent)[0]}, sites)["params"]
    return _selected(gradient, paths)


class HostExecution:
    """Own snapshot placement and attempt-local tapes, not another TrainState."""
    def __init__(self, objective, layout, accelerator, cpu):
        self.objective, self.layout = objective, layout
        self.accelerator, self.cpu = accelerator, cpu
        self.sites = bank_sites(objective) if objective.bank_sites else ()
        self.loss = jax.jit(objective.loss)
        self.unstack = jax.jit(functools.partial(_logical, sites=self.sites))

    def snapshot(self, variables):
        """The declared stacks as banks, everything else as it is stored.

        The frozen split is undone by reference, so a bank holds every one of
        a layer's parameter leaves and a partially frozen run does not stack
        two sparse trees of different shape. The canonical state keeps its
        own `params` and `frozen` collections; this merged view exists only
        for the duration of one realization.
        """
        if variables is None:
            return None
        placement = self.layout.shardings(self.accelerator, variables)
        if not self.sites:
            return transfer(variables, placement)
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
            for (first, count), name in zip(site.view.groups, site.view.bank_names(), strict=True):
                rows = [one_layer(weights, index, namespace=site.namespace)
                        for index in range(first, first + count)]
                if not rows[0]:
                    continue
                placed = one_layer(places, first, namespace=site.namespace)
                bank = rows[0] if count == 1 else _stack(*rows)
                spread = jax.tree.map(
                    lambda s: NamedSharding(
                        self.accelerator, P(None, *s.spec) if count > 1 else s.spec,
                        memory_kind="pinned_host"), placed)
                bank = jax.block_until_ready(transfer(bank, spread))
                for collection, values in bank.items():
                    branch = store.setdefault(collection, {})
                    for component in site.namespace:
                        branch = branch.setdefault(component, {})
                    branch[name] = values
        return store

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

    def realize(self, variables, batch, info):
        with jax.set_mesh(self.cpu):
            store = self.snapshot(variables)
            ema = self.snapshot(info.ema)
        assert store is not None, "a realization always receives model variables"
        moving = tuple(tuple(entry.key for entry in path) for path, _ in
                       jax.tree_util.tree_leaves_with_path(variables["params"]))
        with jax.set_mesh(self.accelerator):
            batch = transfer(batch, batch_shardings(self.accelerator, batch))
            clock, key = self.on_accelerator((info.step, info.key))
            execution_info = Step(clock, key, ema)

            def loss(trainable):
                return self.loss({**store, "params": trainable}, batch, execution_info)
            stats, back, aux = jax.vjp(loss, store["params"], has_aux=True)
        stats, aux = self.on_cpu((stats, aux))
        if aux.variables is not None:
            with jax.set_mesh(self.cpu):
                aux = dataclasses.replace(aux, variables=self.unstack(aux.variables))

        def pullback(cotangent):
            # A shared owner is one bank input, so native reverse mode already
            # sums every use's contribution into it.
            with jax.set_mesh(self.accelerator):
                if not self.sites:
                    return self.on_cpu(back(self.on_accelerator(cotangent))[0])
                gradient = _trainable_cotangent(
                    back, self.on_accelerator(cotangent), sites=self.sites, paths=moving)
            return self.on_cpu(gradient)
        return Realization(stats, aux, pullback)
