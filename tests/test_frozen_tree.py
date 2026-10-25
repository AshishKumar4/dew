"""A run holds one frozen tree, shared with its objective: the compiled step takes the frozen tree
(`variables[FROZEN]`) as an argument of its own, never donated and never returned
(`dew.state.apart`), so the frozen leaves a LoRA run starts from are the objective's held
arrays, fresh or placed, through every step and through the plan's timing of its candidates; a
frozen leaf an objective's init builds is the state's own, placed as it starts fresh."""

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.core import freeze

from dew.batch import FROZEN
from dew.interop import Pretrained
from dew.lora import LoRA
from dew.objectives.lm import TEXT_KEY, LMObjective
from dew.training import Layout, MeshSpec, Trainer

LLAMA = Path(__file__).resolve().parent / "fixtures" / "hf" / "llama-tiny"


def adapted(placed: bool = False) -> LMObjective:
    """A language objective over the tiny Llama with a LoRA adapter, its base the frozen tree, loaded
    onto the mesh where `placed` (each leaf where the trainer's state keeps it), else on one device."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", max_seq_len=16,
                             mesh=MeshSpec() if placed else None)
    return LMObjective(source.adapt(LoRA(rank=2, modules=("q_proj", "v_proj")), key=0), 8, ema_decay=None)


def batch():
    return {TEXT_KEY: jnp.asarray(np.random.default_rng(0).integers(0, 256, (8, 9)), jnp.int32)}


def frozen_ids(tree) -> set[int]:
    return {id(leaf) for leaf in jax.tree.leaves(tree)}


@pytest.mark.parametrize("placed", [False, True])
def test_a_runs_frozen_tree_is_its_objectives_held_arrays_through_every_step(placed):
    """The frozen leaves of a state built from the objective's held tree, fresh (`initial_state`) or
    placed on a mesh (`place`) from a source loaded onto it, are the held arrays themselves; two
    compiled steps leave them so,
    none deleted, while the adapter's factors move; and the compiled step's outputs hold no frozen
    leaf, its output bytes less than the frozen tree's alone."""
    objective = adapted(placed)
    trainer = (Trainer(objective, optax.sgd(0.1), key=jax.random.key(5), mesh=MeshSpec()) if placed
               else Trainer(objective, optax.sgd(0.1), key=jax.random.key(5)))
    state = trainer.place()[0] if placed else trainer.initial_state()
    held = objective.held_variables()
    assert held is not None and FROZEN in held
    assert frozen_ids(state.variables[FROZEN]) == frozen_ids(held[FROZEN])
    rows = batch()
    step = trainer.compile(state, rows)
    started = jax.tree.map(np.asarray, state.variables["params"])
    for _ in range(2):
        state = step(state, rows)[0]
    assert frozen_ids(state.variables[FROZEN]) == frozen_ids(held[FROZEN])
    assert not any(leaf.is_deleted() for leaf in jax.tree.leaves(held[FROZEN]))
    assert any(not np.array_equal(np.asarray(after), before) for after, before in zip(
        jax.tree.leaves(state.variables["params"]), jax.tree.leaves(started), strict=True))
    assert trainer.executable is not None
    frozen_bytes = sum(leaf.nbytes for leaf in jax.tree.leaves(held[FROZEN]))
    analysis = trainer.executable.memory_analysis()
    assert analysis is not None and analysis.output_size_in_bytes < frozen_bytes


def test_timing_a_frontier_of_plans_leaves_the_frozen_tree_whole():
    """A plan timed over two candidates deletes the state it timed the first on and restores a copy
    (`planning.Timer`): the moving state alone, so the frozen tree the run and its objective share
    is the same arrays after the timing, none deleted."""
    objective = adapted(placed=True)
    trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(5), mesh=MeshSpec())
    state = trainer.place()[0]
    held = objective.held_variables()
    assert held is not None
    from dew.training.planning import Timer

    rows = batch()
    trainer.compile(state, rows)
    timer = Timer(state, rows)
    step = trainer.compile(state, rows)
    timer([step, step])
    assert frozen_ids(timer.state.variables[FROZEN]) == frozen_ids(held[FROZEN])
    assert not any(leaf.is_deleted() for leaf in jax.tree.leaves(held[FROZEN]))


class Rebuilt(LMObjective):
    """An objective whose init builds its first frozen leaf, `change` of the held one, and passes the
    rest through."""

    def __init__(self, change: Callable[[jax.Array], jax.Array], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.change = change

    def init(self, key, variables=None):
        tree = super().init(key, variables)
        leaves, structure = jax.tree.flatten(tree[FROZEN])
        return {**tree, FROZEN: jax.tree.unflatten(structure, [self.change(leaves[0]), *leaves[1:]])}


@pytest.mark.parametrize("change", ["values", "dtype"])
def test_a_frozen_leaf_an_init_builds_is_the_states_own_placed_as_fresh(change):
    """An init that builds a frozen leaf, same-shaped values of its own or another dtype, and passes
    the rest through, from a source loaded onto the mesh: the placed start's frozen tree is the fresh
    start's, value for value, the built leaf its own array and every other the held array itself, the
    held arrays left whole."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", max_seq_len=16,
                             mesh=MeshSpec())
    built = {"values": lambda leaf: leaf + 1, "dtype": lambda leaf: leaf.astype(jnp.float16)}[change]
    rebuilt = Rebuilt(built, source.adapt(LoRA(rank=2, modules=("q_proj", "v_proj")), key=0), 8,
                      ema_decay=None)
    placed = Trainer(rebuilt, optax.sgd(0.1), key=jax.random.key(5), mesh=MeshSpec()).place()[0]
    fresh = Trainer(rebuilt, optax.sgd(0.1), key=jax.random.key(5)).initial_state()
    held = rebuilt.held_variables()
    assert held is not None
    first, *rest = jax.tree.leaves(placed.variables[FROZEN])
    original, *passed = jax.tree.leaves(held[FROZEN])
    assert first is not original and [id(leaf) for leaf in rest] == [id(leaf) for leaf in passed]
    for mine, theirs in zip(jax.tree.leaves(placed.variables[FROZEN]),
                            jax.tree.leaves(fresh.variables[FROZEN]), strict=True):
        assert mine.dtype == theirs.dtype and np.array_equal(np.asarray(mine), np.asarray(theirs))
    assert not any(leaf.is_deleted() for leaf in jax.tree.leaves(held[FROZEN]))


def test_a_frozen_tree_an_init_moves_is_the_held_arrays_where_the_state_keeps_them():
    """An init that nests its held frozen tree under a key of its own, as an objective that holds a
    policy's tree does: the placed start's frozen leaves at their new paths are the held arrays
    themselves, placed where the state keeps them once, from a source loaded onto the mesh, every leaf
    sharded (`min_shard=1`) so no path's placement is the replicated one."""

    class Moved(LMObjective):
        def init(self, key, variables=None):
            tree = super().init(key, variables)
            return {**tree, FROZEN: {"policy": tree[FROZEN]}}

    sharded = Layout(min_shard=1)
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", max_seq_len=16,
                             mesh=MeshSpec(), layout=sharded)
    objective = Moved(source.adapt(LoRA(rank=2, modules=("q_proj", "v_proj")), key=0), 8, ema_decay=None)
    trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(5), mesh=MeshSpec(), layout=sharded)
    state, shardings = trainer.place()[:2]
    held = objective.held_variables()
    assert held is not None
    assert frozen_ids(state.variables[FROZEN]["policy"]) == frozen_ids(held[FROZEN])
    for leaf, sharding in zip(jax.tree.leaves(state.variables[FROZEN]),
                              jax.tree.leaves(shardings.variables[FROZEN]), strict=True):
        assert leaf.sharding == sharding


def test_a_placed_start_shares_a_frozen_tree_held_in_flax_containers():
    """An objective that holds its tree in Flax's FrozenDict starts placed as one holding dicts: its
    frozen leaves are the held arrays."""

    class Frozen(LMObjective):
        def held_variables(self):
            held = super().held_variables()
            return None if held is None else freeze(held)

    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", max_seq_len=16,
                             mesh=MeshSpec())
    objective = Frozen(source.adapt(LoRA(rank=2, modules=("q_proj", "v_proj")), key=0), 8, ema_decay=None)
    state = Trainer(objective, optax.sgd(0.1), key=jax.random.key(5), mesh=MeshSpec()).place()[0]
    held = objective.held_variables()
    assert held is not None
    assert frozen_ids(state.variables[FROZEN]) == frozen_ids(held[FROZEN])


def test_a_source_on_one_device_starts_placed_on_the_mesh_as_it_starts_fresh():
    """A source loaded onto one device, trained on a mesh of every device: its frozen leaves move to
    where the state keeps them, value for value the fresh start's, and the held arrays stay whole."""
    objective = adapted(placed=False)
    trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(5), mesh=MeshSpec())
    state, shardings = trainer.place()[:2]
    fresh = Trainer(objective, optax.sgd(0.1), key=jax.random.key(5)).initial_state()
    for leaf, sharding in zip(jax.tree.leaves(state.variables[FROZEN]),
                              jax.tree.leaves(shardings.variables[FROZEN]), strict=True):
        assert leaf.sharding == sharding
    for mine, theirs in zip(jax.tree.leaves(state.variables[FROZEN]),
                            jax.tree.leaves(fresh.variables[FROZEN]), strict=True):
        assert np.array_equal(np.asarray(mine), np.asarray(theirs))
    held = objective.held_variables()
    assert held is not None and not any(leaf.is_deleted() for leaf in jax.tree.leaves(held[FROZEN]))
