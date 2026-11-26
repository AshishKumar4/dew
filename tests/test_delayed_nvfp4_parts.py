"""A delayed-scaling NVFP4 run's amax histories beside the parts of a run that hold weights of their own
(row 12721): an average of the weights, which reads the committed histories, and a critic drawn apart."""

import dataclasses
import functools
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from delayed_nvfp4 import DELAYED, Referenced, data, delayed_lm, histories, rolled_once

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.dense import NVFP4_HISTORY
from dew.nn.kernels import generation
from dew.training import Trainer


def test_an_averaged_forward_reads_the_committed_histories():
    """A run that keeps an average of its weights averages no amax history: a forward over the averaged
    weights (`TrainState.averaged`) reads the histories the run committed, bit for bit, and the run's
    histories roll as a run keeping no average rolls them (review of row 12721: the average smoothed them,
    and an averaged forward read a fraction of the step's largest)."""
    with generation.run_precision(DELAYED):
        averaged = delayed_lm(1, ema_decay=0.9).fit(data(), steps=1)
        plain = delayed_lm(1).fit(data(), steps=1)
    for one, other, read in zip(histories(averaged), histories(plain),
                                jax.tree.leaves(averaged.averaged[NVFP4_HISTORY]), strict=True):
        np.testing.assert_array_equal(one, other)
        np.testing.assert_array_equal(np.asarray(read), one)


def test_a_reference_keeps_the_histories_it_started_with():
    """A run whose average is a frozen reference (`Objective.ema_is_reference`: DPO's, GRPO's) holds its
    start's amax histories with its start's weights: a forward over the reference (`TrainState.averaged`)
    reads the histories the run began with, bit for bit, while the run's own roll on (row 12993: the
    reference read the live histories beside its frozen weights)."""
    with generation.run_precision(DELAYED):
        trainer = delayed_lm(1, ema_decay=1.0, objective=Referenced)
        started = histories(trainer.place()[0])
        trained = trainer.fit(data(), steps=2)
    for start, read, now in zip(started, jax.tree.leaves(trained.averaged[NVFP4_HISTORY]), histories(trained),
                                strict=True):
        np.testing.assert_array_equal(np.asarray(read), start)
        assert now[..., 0].min() > 0, now


def _every_second() -> optax.GradientTransformation:
    return optax.MultiSteps(optax.adamw(1e-3), 2).gradient_transformation()


def test_an_optimizer_holding_updates_back_rolls_the_largest_of_every_call_it_held():
    """Under an optimizer that applies an update every second call (`optax.MultiSteps`), the call that
    applies it rolls the largest magnitude of both calls into each history, once: the first call's records
    wait (`NVFP4_PENDING`) where it applies nothing, so the history after two calls is the history a run
    reading both calls' rows as one step's two microbatches rolls (row 12994: the second rolled its own
    alone)."""
    with generation.run_precision(DELAYED):
        held = delayed_lm(1, optimizer=_every_second()).fit(data(), steps=2)
        joined = delayed_lm(2).fit(data(32), steps=1)
    for one, other in zip(histories(held), histories(joined), strict=True):
        np.testing.assert_array_equal(one, other)
        assert one[..., 0].min() > 0 and not one[..., 1:].any(), one


def test_what_waits_rolls_once_an_update_applies_and_a_skipped_input_waits_for_none():
    """What waits to roll (`rolled_waiting`): a call the optimizer skips adds nothing, even its largest
    magnitude; the call that applies an update rolls the larger of its own and what waited, a projection an
    earlier call ran and it did not among them, a recorded zero as any magnitude; nothing waits after, and a
    history nothing waited for keeps as it was (row 12994's review)."""
    from dew.nn.dense import pending_amax, rolled_waiting

    started = {"ran": jnp.zeros((3,)), "earlier": jnp.zeros((3,)), "zero": jnp.asarray([1.0, 2.0, 3.0]),
               "never": jnp.asarray([5.0, 4.0, 3.0])}
    yes, no = jnp.asarray(a=True), jnp.asarray(a=False)
    waiting = pending_amax(started)
    _, waiting = rolled_waiting(started, waiting, {"ran": 2.0, "earlier": 7.0, "zero": 0.0}, no, yes)
    _, waiting = rolled_waiting(started, waiting, {"ran": 9.0}, no, no)
    rolled, after = rolled_waiting(started, waiting, {"ran": 3.0}, yes, yes)
    assert {name: np.asarray(value).tolist() for name, value in rolled.items()} == {
        "ran": [3.0, 0.0, 0.0], "earlier": [7.0, 0.0, 0.0], "zero": [0.0, 1.0, 2.0], "never": [5.0, 4.0, 3.0]}
    assert all(float(value) == -1.0 for value in jax.tree.leaves(after))


def test_a_run_resumed_partway_through_a_held_update_rolls_what_its_taken_calls_recorded(tmp_path):
    """A run whose optimizer holds updates back, resumed from a checkpoint written partway through one,
    restores what its taken call recorded (`NVFP4_PENDING`) and rolls the histories the uninterrupted run
    rolls; a checkpoint an earlier Dew wrote, which holds none, resumes where an update applied, nothing
    waiting, and is refused partway through one, whose records it lost (row 12994's review)."""
    from dew.checkpoints import Checkpoints
    from dew.nn.dense import NVFP4_PENDING

    with generation.run_precision(DELAYED):
        whole = delayed_lm(1, optimizer=_every_second()).fit(data(), steps=4)
        delayed_lm(1, Checkpoints(str(tmp_path / "held")), optimizer=_every_second()).fit(
            data(), steps=3, checkpoint_every=1)
        # Step 3's checkpoint, partway through the second update, read again below as it was written.
        shutil.copytree(tmp_path / "held", tmp_path / "run")
        resumed = delayed_lm(1, Checkpoints(str(tmp_path / "run")), optimizer=_every_second()).fit(
            data(), steps=4, checkpoint_every=1)
        for one, other in zip(histories(resumed), histories(whole), strict=True):
            np.testing.assert_array_equal(one, other)
        # A run's checkpoint where its update applied (step 2): one keeps its latest step alone.
        delayed_lm(1, Checkpoints(str(tmp_path / "applied")), optimizer=_every_second()).fit(
            data(), steps=2, checkpoint_every=2)

        def resaved(step: int, name: str, waiting) -> Trainer:
            """The run's checkpoint at `step` (step 2's run's, else the resumed one's step 3) written again at
            `name` with `waiting` as what waits to roll (`NVFP4_PENDING`): None, as an earlier Dew wrote it,
            holds none."""
            source = delayed_lm(1, Checkpoints(str(tmp_path / ("applied" if step == 2 else "held"))),
                                optimizer=_every_second())
            # Its layers on `layers_N`, as a checkpoint holds them (`Trainer.stored`).
            state = source.stored(source.checkpoints.restore(source.place()[0], step)[0])
            variables = {key: value for key, value in state.variables.items() if key != NVFP4_PENDING}
            if waiting is not None:
                variables[NVFP4_PENDING] = waiting(state.variables[NVFP4_PENDING])
            written = Checkpoints(str(tmp_path / name))
            written.save(step, dataclasses.replace(state, variables=variables))
            written.wait()
            return delayed_lm(1, Checkpoints(str(tmp_path / name)), optimizer=_every_second())

        nothing = functools.partial(jax.tree.map, lambda leaf: jnp.full_like(leaf, -1.0))
        # Where an update applied, an earlier Dew's checkpoint resumes, nothing waiting.
        waits = resaved(2, "legacy-applied", None).place()[0].variables[NVFP4_PENDING]
        assert all(float(leaf.max()) == -1.0 for leaf in jax.tree.leaves(waits))
        # Partway through a held update, a checkpoint holding nothing waiting resumes as written; the same
        # state as an earlier Dew wrote it, the collection alone missing, is refused.
        resaved(3, "current-held", nothing).place()
        legacy = resaved(3, "legacy-held", None)
        with pytest.raises(ValueError):
            legacy.place()


def test_a_reference_an_earlier_dew_saved_is_refused_and_one_this_dew_saved_resumes(tmp_path):
    """A frozen reference's checkpoint an earlier Dew wrote, which kept no histories of the reference (its
    average held none, row 13057), is refused: the reference's start histories, a trained start's among them,
    are recorded nowhere, and no live history stands for them; one this Dew wrote restores its own."""
    from dew.checkpoints import Checkpoints

    with generation.run_precision(DELAYED):
        def run(name: str) -> Trainer:
            return delayed_lm(1, Checkpoints(str(tmp_path / name)), ema_decay=1.0, objective=Referenced)

        run("current").fit(data(), steps=2, checkpoint_every=2)
        source = run("current")
        state = source.stored(source.checkpoints.restore(source.place()[0], 2)[0])
        # A current checkpoint whose reference histories hold values of their own, which a restore reads back.
        seeded = dataclasses.replace(state, ema={**state.ema, NVFP4_HISTORY: jax.tree.map(
            lambda leaf: leaf + 1.0, state.ema[NVFP4_HISTORY])})
        unkept = {key: value for key, value in state.ema.items() if key != NVFP4_HISTORY}
        for name, ema in (("legacy", unkept), ("seeded", seeded.ema)):
            written = Checkpoints(str(tmp_path / name))
            written.save(2, dataclasses.replace(state, ema=ema))
            written.wait()
        earlier = run("legacy")
        with pytest.raises(ValueError):
            earlier.place()
        again = run("seeded")
        kept = again.stored(again.place()[0]).ema[NVFP4_HISTORY]
    for read, written in zip(jax.tree.leaves(kept), jax.tree.leaves(seeded.ema[NVFP4_HISTORY]), strict=True):
        np.testing.assert_array_equal(np.asarray(read), np.asarray(written))


def test_a_ppo_steps_records_are_its_own_passes_its_critics_among_them():
    """A PPO objective draws its critic apart from its actor (`PPOObjective.init`), and the critic's draw
    keeps nothing its initializing pass recorded: a step's records are each part's own call's alone, one
    a projection, never an initializer's activations beside them (review of row 12721: the critic's draw
    kept its init's records, which every step's largest then took in), and its step rolls every history,
    the actor's and the critic's."""
    from dew.nn.dense import NVFP4_AMAX
    from dew.objectives.base import Step
    from dew.objectives.rl import PPOObjective, ValueHead

    model = CausalTransformer(vocab_size=16, emb_features=32, num_layers=1, num_heads=2, mlp_features=64,
                              max_seq_len=8)
    objective = PPOObjective(model, 5, critic=ValueHead(model))
    ids = np.tile(np.arange(6, dtype=np.int32), (jax.device_count(), 1))
    mask = np.broadcast_to(np.arange(6) >= 3, ids.shape).astype(np.float32)
    batch = {"input_ids": ids, "text_segment_ids": np.ones_like(ids),
             "text_positions": np.broadcast_to(np.arange(6), ids.shape).astype(np.int32),
             "response_mask": mask, "advantages": mask, "behavior_log_probs": np.full(ids.shape, -3.0),
             "old_values": np.zeros_like(mask), "returns": mask}
    with generation.run_precision(DELAYED):
        trainer = Trainer(objective, optax.adamw(1e-3), key=0)
        initial = trainer.initial_state()
        observing = Step(np.int32(0), jax.random.key(1), None, amax=True)
        _, aux = objective.loss(initial.variables, batch, observing)
        state, *_ = trainer.compile(initial, batch)(initial, batch)
    assert NVFP4_AMAX not in initial.variables, sorted(initial.variables)
    calls = jax.tree.leaves(aux.amax["critic"], is_leaf=lambda node: isinstance(node, tuple))
    assert calls and all(len(call) == 1 for call in calls), [len(call) for call in calls]
    rolled_once(state)
