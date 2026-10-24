"""A run holds one frozen tree, shared with its objective: the compiled step takes the frozen tree
(`variables[FROZEN]`) as an argument of its own, never donated and never returned
(`dew.training.state.apart`), so the frozen leaves a LoRA run starts from are the objective's held
arrays, fresh or placed, through every step and through the plan's timing of its candidates."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.interop import Pretrained
from dew.lora import LoRA
from dew.objectives.base import FROZEN
from dew.objectives.lm import TEXT_KEY, LMObjective
from dew.training import MeshSpec, Trainer

LLAMA = Path(__file__).resolve().parent / "fixtures" / "hf" / "llama-tiny"


def adapted() -> LMObjective:
    """A language objective over the tiny Llama with a LoRA adapter, its base the frozen tree."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", max_seq_len=16)
    return LMObjective(source.adapt(LoRA(rank=2, modules=("q_proj", "v_proj")), key=0), 8, ema_decay=None)


def batch():
    return {TEXT_KEY: jnp.asarray(np.random.default_rng(0).integers(0, 256, (2, 9)), jnp.int32)}


def frozen_ids(tree) -> set[int]:
    return {id(leaf) for leaf in jax.tree.leaves(tree)}


@pytest.mark.parametrize("placed", [False, True])
def test_a_runs_frozen_tree_is_its_objectives_held_arrays_through_every_step(placed):
    """The frozen leaves of a state built from the objective's held tree, fresh (`initial_state`) or
    placed on a mesh (`place`), are the held arrays themselves; two compiled steps leave them so,
    none deleted, while the adapter's factors move; and the compiled step's outputs hold no frozen
    leaf, its output bytes less than the frozen tree's alone."""
    objective = adapted()
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
    objective = adapted()
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


def test_an_objective_whose_init_rebuilds_its_frozen_tree_is_refused_on_the_mesh():
    """The trainer shares an objective's held frozen tree as its state's (`Objective.held_variables`):
    an objective whose init returns a frozen tree of another dtype than the one it holds breaks that
    contract, and its placed start is refused rather than holding two."""

    class Widened(LMObjective):
        def init(self, key, variables=None):
            tree = super().init(key, variables)
            return {**tree, FROZEN: jax.tree.map(lambda leaf: leaf.astype(jnp.float16), tree[FROZEN])}

    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", max_seq_len=16)
    rebuilt = Widened(source.adapt(LoRA(rank=2, modules=("q_proj", "v_proj")), key=0), 8, ema_decay=None)
    with pytest.raises(ValueError):
        Trainer(rebuilt, optax.sgd(0.1), key=jax.random.key(5), mesh=MeshSpec()).place()
