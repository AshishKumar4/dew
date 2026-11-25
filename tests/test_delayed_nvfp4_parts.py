"""A delayed-scaling NVFP4 run's amax histories beside the parts of a run that hold weights of their own
(row 12721): an average of the weights, which reads the committed histories, and a critic drawn apart."""

import jax
import numpy as np
import optax
from delayed_nvfp4 import DELAYED, data, delayed_lm, histories, rolled_once

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
    """A run whose average is a frozen reference (unit decay: DPO's, GRPO's) holds its start's amax histories
    with its start's weights: a forward over the reference (`TrainState.averaged`) reads the histories the
    run began with, bit for bit, while the run's own roll on (row 12993: the reference read the live
    histories beside its frozen weights)."""
    with generation.run_precision(DELAYED):
        trainer = delayed_lm(1, ema_decay=1.0)
        started = histories(trainer.place()[0])
        trained = trainer.fit(data(), steps=2)
    rolled_on = histories(trained)
    for start, read, now in zip(started, jax.tree.leaves(trained.averaged[NVFP4_HISTORY]), rolled_on,
                                strict=True):
        np.testing.assert_array_equal(np.asarray(read), start)
        assert now[..., 0].min() > 0, now


def test_an_optimizer_holding_updates_back_rolls_the_largest_of_every_call_it_held():
    """Under an optimizer that applies an update every second call (`optax.MultiSteps`), the call that
    applies it rolls the largest magnitude of both calls into each history, once: the first call's records
    wait (`NVFP4_PENDING`) where it applies nothing, so the history after two calls is the history a run
    reading both calls' rows as one step's two microbatches rolls (row 12994: the second rolled its own
    alone)."""
    from delayed_nvfp4 import SEQ

    from dew.objectives.lm import LMObjective

    def trainer(optimizer, split: int) -> Trainer:
        model = CausalTransformer(vocab_size=64, emb_features=128, num_layers=2, num_heads=2,
                                  mlp_features=256, max_seq_len=SEQ)
        return Trainer(LMObjective(model, seq_len=SEQ), optimizer, key=0, split=split)

    with generation.run_precision(DELAYED):
        held = trainer(optax.MultiSteps(optax.adamw(1e-3), 2), 1).fit(data(), steps=2)
        joined = trainer(optax.adamw(1e-3), 2).fit(data(32), steps=1)
    for one, other in zip(histories(held), histories(joined), strict=True):
        np.testing.assert_array_equal(one, other)
        assert one[..., 0].min() > 0 and not one[..., 1:].any(), one


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
