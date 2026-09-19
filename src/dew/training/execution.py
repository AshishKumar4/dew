"""An accelerator realization over immutable banks of a CPU-owned state.

Only objective evaluation and its pullback cross this boundary. The optimizer,
accumulation and checkpoint trees retain their original logical leaf identities.
Each snapshot bank is assembled and transferred before the next is built; no
whole parameter tree is ever materialized on the accelerator.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.inference.banks import entry_names, named, one_layer
from dew.nn.backbones.causal_transformer import StackView
from dew.objectives.base import FROZEN, Step
from dew.training.distributed import batch_shardings
from dew.training.host import transfer
from dew.training.transaction import Realization

_stack = jax.jit(lambda *rows: jax.tree.map(lambda *leaves: jnp.stack(leaves), *rows))


class HostExecution:
    """Own snapshot placement and attempt-local tapes, not another TrainState."""
    def __init__(self, objective, layout, accelerator, cpu):
        self.objective, self.layout = objective, layout
        self.accelerator, self.cpu = accelerator, cpu
        self.view = StackView(objective.layer_groups)
        self.loss = jax.jit(objective.loss)
        self.unstack = jax.jit(self.view.unstack)

    def snapshot(self, variables):
        if variables is None:
            return None
        placement = self.layout.shardings(self.accelerator, variables)
        if not self.view.groups:
            return transfer(variables, placement)
        # Objective-owned mutable collections retain their logical paths:
        # router effects, for example, match per-layer sows to per-layer bias.
        # Only trainable and frozen model weights adopt execution banks.
        weights = {collection: tree for collection, tree in variables.items()
                   if collection in ("params", FROZEN)}
        entries = {**{collection: tree for collection, tree in variables.items()
                       if collection not in weights}, **named(weights, entry_names(weights))}
        entry_placement = {collection: {name: placement[collection][name] for name in tree}
                           for collection, tree in entries.items()}
        store = transfer(entries, entry_placement)
        for (first, count), name in zip(self.view.groups, self.view.bank_names(), strict=True):
            rows = [one_layer(weights, index) for index in range(first, first + count)]
            placed = one_layer({name: placement[name] for name in weights}, first)
            if not rows[0]:
                continue
            bank = rows[0] if count == 1 else _stack(*rows)
            places = jax.tree.map(
                lambda s: NamedSharding(
                    self.accelerator, P(None, *s.spec) if count > 1 else s.spec,
                    memory_kind="pinned_host"), placed)
            bank = jax.block_until_ready(transfer(bank, places))
            for collection, values in bank.items():
                store.setdefault(collection, {})[name] = values
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
            with jax.set_mesh(self.accelerator):
                gradient = back(self.on_accelerator(cotangent))[0]
            gradient = self.on_cpu(gradient)
            with jax.set_mesh(self.cpu):
                return self.unstack({"params": gradient})["params"]
        return Realization(stats, aux, pullback)
