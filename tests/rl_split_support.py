"""Shared helpers for control PPO and Dreamer filled-batch, split-step and mid-window resume tests."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.checkpoints import Checkpoints
from dew.distributed.placement import MeshSpec
from dew.objectives.base import taken_record
from dew.state import apart, together
from dew.training import Trainer


def sequence_learned_bits(state) -> list[np.ndarray]:
    """Host byte views of `state`'s learned and optimizer-held trees (`variables`, `opt_state`, `ema`,
    `updates`), apart from attempt counters (`step`, `microstep`) and transient window buffers."""
    from recording import bits

    checked = bits((state.variables, state.opt_state, state.ema, state.updates))
    assert all(not jnp.issubdtype(leaf.dtype, jnp.inexact) or bool(jnp.all(jnp.isfinite(leaf)))
               for leaf in checked)
    return checked


def nudged_tree(tree, seed: int, scale: float = 0.4):
    """`tree` with normal perturbations on every floating leaf so zero-initialized heads vary."""
    leaves, structure = jax.tree.flatten(tree)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    moved = [leaf + scale * jax.random.normal(key, leaf.shape, leaf.dtype)
             if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf
             for leaf, key in zip(leaves, keys, strict=True)]
    return jax.tree.unflatten(structure, moved)


def assert_sequence_filled_invariants(objective, filled, corrupted, repeats, *, splits=(1, 2),
                                      nudge: bool = False) -> None:
    """Assert through `Trainer.compile` at each `split` that `filled` and `corrupted` (differing in filler
    rows, including NaNs) step to bit-identical learned state, loss and metrics, and `repeats` (`VALID_ROWS`
    all False) commits no update while advancing attempt clocks."""
    mesh = MeshSpec().build(jax.devices()[:1])
    for split in splits:
        trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(3), split=split,
                          checkpoints=None, tracker=None)
        trainer.device_mesh = mesh
        initial = trainer.initial_state()
        if nudge:
            initial = initial.replace(variables={**initial.variables,
                                                 "params": nudged_tree(initial.variables["params"], 17)})
        step = trainer.compile(initial, filled)
        first_state, first_loss, first_metrics, *_ = step(jax.tree.map(jnp.copy, initial), filled)
        other_state, other_loss, other_metrics, *_ = step(jax.tree.map(jnp.copy, initial), corrupted)
        for left, right in zip(sequence_learned_bits(first_state), sequence_learned_bits(other_state),
                               strict=True):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(np.asarray(first_loss), np.asarray(other_loss))
        jax.tree.map(np.testing.assert_array_equal, jax.device_get(first_metrics),
                     jax.device_get(other_metrics))
        idle_state, idle_loss, *_ = step(jax.tree.map(jnp.copy, initial), repeats)
        assert int(idle_state.updates) == 0 and float(idle_loss) == 0.0
        assert int(idle_state.step) == 1 and int(idle_state.microstep) == split
        for left, right in zip(sequence_learned_bits(idle_state), sequence_learned_bits(initial),
                               strict=True):
            np.testing.assert_array_equal(left, right)


def recorded_step_tapes(objective, batch, *, key: int = 7, split: int,
                        variables=None) -> tuple[object, dict[str, np.ndarray]]:
    """Run one recorded step at `split` and return `(stepped_state, ordered_tapes_by_example)`."""
    rows = jax.tree.leaves(batch)[0].shape[0]
    mesh = MeshSpec().build(jax.devices()[:1])
    trainer = Trainer(objective, optax.sgd(0.05), key=jax.random.key(key), split=split, record=True,
                      checkpoints=None, tracker=None)
    trainer.device_mesh = mesh
    state = trainer.initial_state()
    if variables is not None:
        state = state.replace(variables=jax.tree.map(jnp.copy, variables))
    step = trainer.compile(state, batch)
    stepped, _, metrics, *_ = step(state, batch)
    record = jax.device_get(taken_record(metrics)[0])
    assert record is not None
    tapes = {}
    for name, values in record["values"].items():
        ordered = np.zeros((rows, *values.shape[2:]), values.dtype)
        for indices, part_values in zip(record["example_index"], values, strict=True):
            ordered[indices] = part_values
        tapes[name] = ordered
    return stepped, tapes


def assert_mid_window_resume(objective, batch_a, batch_b, directory: str, *, nudge: bool = False):
    """Verify a `split=2` step saved after part 0 and restored resumes part 1 with the stored
    `batch_statistics`, matching an uninterrupted step bit for bit; also return the compiled plan."""
    mesh = MeshSpec().build(jax.devices()[:1])
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(10), split=2,
                      checkpoints=None, tracker=None)
    trainer.device_mesh = mesh
    initial = trainer.initial_state()
    if nudge:
        initial = initial.replace(variables={**initial.variables,
                                             "params": nudged_tree(initial.variables["params"], 29)})
    planned = trainer._compiled_plan(initial, batch_a, None, None)
    runner = trainer._compiler.runner(trainer._stepping(), planned, mesh)
    warmed, *_ = runner(initial, batch_a)
    expected, *_ = runner(jax.tree.map(jnp.copy, warmed), batch_b)
    parts = planned.parts(mesh)
    placed_b = jax.device_put(batch_b, jax.devices()[0])
    moving, frozen = apart(jax.tree.map(jnp.copy, warmed))
    moving, *_ = planned.executable(moving, frozen, placed_b, parts[0])
    mid = together(moving, frozen)
    saver = Checkpoints(directory)
    saver.save(1, mid)
    saver.wait()
    saver.close()
    loader = Checkpoints(directory)
    template = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding), mid)
    restored, _ = loader.restore(template, 1)
    loader.close()
    moving_r, frozen_r = apart(restored)
    moving_r, *_ = planned.executable(moving_r, frozen_r, placed_b, parts[1])
    resumed = together(moving_r, frozen_r)
    for left, right in zip(sequence_learned_bits(resumed), sequence_learned_bits(expected), strict=True):
        np.testing.assert_array_equal(left, right)
    return planned, mesh


def prepare_split_whole_steps(objective, rate: float, run_key: jax.Array, template, filled_template):
    """Compile paired `split=2` (on `filled_template`) and `split=1` (on `template`) SGD steps and return
    `(split_start, whole_start, split_step, whole_step, update_key)`."""
    from dew.objectives.base import EXAMPLE_STREAM

    mesh = MeshSpec().build(jax.devices()[:1])
    split_trainer = Trainer(objective, optax.sgd(rate), key=run_key, split=2, checkpoints=None, tracker=None)
    split_trainer.device_mesh = mesh
    whole_trainer = Trainer(objective, optax.sgd(rate), key=run_key, split=1, checkpoints=None, tracker=None)
    whole_trainer.device_mesh = mesh
    split_start, whole_start = split_trainer.initial_state(), whole_trainer.initial_state()
    split_step = split_trainer.compile(split_start, filled_template)
    whole_step = whole_trainer.compile(whole_start, template)
    update_key = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(split_start.key, EXAMPLE_STREAM),
                                                       0), 0)
    return split_start, whole_start, split_step, whole_step, update_key
