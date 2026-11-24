"""A delayed-scaling NVFP4 run's amax histories as its training steps move them (row 12721): each projection
the recipe takes reads its history and records its rows' largest magnitude a call; the trainer takes the
largest across a step's calls and microbatches and rolls it into the history once per committed update."""

import jax
import numpy as np
import optax

from dew.checkpoints import Checkpoints
from dew.data import Memory
from dew.data.pipeline import Pipeline
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.dense import NVFP4_HISTORY
from dew.nn.kernels import generation
from dew.nn.precision import DTYPE_PRESETS, LowPrecision, Numerics
from dew.objectives.lm import LMObjective
from dew.training import Trainer

SEQ = 16
DELAYED = Numerics(DTYPE_PRESETS["bf16-mixed"], low_precision=LowPrecision(recipe="nvfp4",
                                                                           tensor_scaling="delayed"))
"""A run's numerics in the delayed-scaling NVFP4 recipe, which takes the feed-forward's projections."""


def delayed_lm(split: int, checkpoints: Checkpoints | None = None, ema_decay: float | None = None) -> Trainer:
    """A two-layer LM whose feed-forward projections the delayed recipe takes, initialized under it (so its
    variables hold their histories), its batch of sixteen rows read in `split` microbatches, keeping an
    average of its weights at `ema_decay` where given."""
    model = CausalTransformer(vocab_size=64, emb_features=128, num_layers=2, num_heads=2, mlp_features=256,
                              max_seq_len=SEQ)
    return Trainer(LMObjective(model, seq_len=SEQ, ema_decay=ema_decay), optax.adamw(1e-3), key=0,
                   split=split, checkpoints=checkpoints)


def data() -> Pipeline:
    """Rows of sixteen, each step's its own: every microbatch of a split in two whole rows on each of a
    CPU mesh's eight data shards."""
    text = (np.arange(128 * (SEQ + 1)).reshape(128, SEQ + 1) * 7 % 64).astype(np.int32)
    return Pipeline(Memory({"text": text})).batch(16)


def histories(state) -> list[np.ndarray]:
    """Each projection's amax history the state holds, in tree order."""
    return [np.asarray(leaf) for leaf in jax.tree.leaves(state.variables[NVFP4_HISTORY])]


def test_a_split_step_rolls_the_largest_its_microbatches_recorded_into_each_history_once_an_update():
    """A step read whole and the same step read in two microbatches roll the same largest magnitude into
    each history, newest first (the window's largest is the largest of its microbatches', so the history
    does not depend on the split), once an update: after one step each history holds one recorded value and
    zeros, and after the next, the first step's value moved one place on, behind the second's."""
    with generation.run_precision(DELAYED):
        whole, halves = delayed_lm(split=1), delayed_lm(split=2)
        started = histories(whole.place()[0])
        first = [whole.fit(data(), steps=1), halves.fit(data(), steps=1)]
        second = delayed_lm(split=2).fit(data(), steps=2)
    assert started and all(not history.any() for history in started), started
    for one, other in zip(histories(first[0]), histories(first[1]), strict=True):
        np.testing.assert_array_equal(one, other)
        assert one[0] > 0 and not one[1:].any(), one
    for after, before in zip(histories(second), histories(first[1]), strict=True):
        np.testing.assert_array_equal(after[1], before[0])
        assert after[0] > 0 and not after[2:].any(), after


def test_a_resumed_run_restores_its_amax_histories(tmp_path):
    """A delayed-scaling NVFP4 run's histories are state its checkpoint keeps: a run resumed from it reads
    them as the run that saved them held them, and its next step rolls them on."""
    with generation.run_precision(DELAYED):
        saved = delayed_lm(1, Checkpoints(str(tmp_path))).fit(data(), steps=2, checkpoint_every=1)
        fresh = delayed_lm(1, Checkpoints(str(tmp_path)))
        restored = fresh.place()[0]
        resumed = fresh.fit(data(), steps=3, checkpoint_every=1)
    for one, other in zip(histories(saved), histories(restored), strict=True):
        np.testing.assert_array_equal(one, other)
    for after, before in zip(histories(resumed), histories(saved), strict=True):
        np.testing.assert_array_equal(after[1:3], before[:2])
        assert after[0] > 0, after


def rolled_once(state) -> None:
    """Every amax history `state` holds took a recorded largest magnitude newest: each projection the step
    ran was recorded at its history's path, none left at its start."""
    held = histories(state)
    assert held and all(history[..., 0].min() > 0 and not history[..., 1:].any() for history in held), held


def test_a_decision_step_rolls_its_backbones_histories_where_they_lie():
    """A decision objective's step, its backbone and head applied apart (`DecisionModel.beside_head`), rolls
    each history its backbone holds under its own part (`nvfp4/backbone/...`): its records nest where its
    histories lie, every scored exit's beside the last (review of row 12721: they came back unnested, and
    the histories stayed at their start)."""
    from decision_support import decision_rows
    from test_decision_objective import EXAMPLES, LAYOUT, SPECIALS, TOKENIZER, tiny_backbone

    from dew.decision import DecisionObjective

    with generation.run_precision(DELAYED):
        objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT)
        state = Trainer(objective, optax.adamw(1e-3), key=0).fit(
            decision_rows(EXAMPLES, 8, objective=objective.data()), steps=1)
    rolled_once(state)


def test_a_block_diffusion_step_rolls_the_largest_of_every_pass():
    """A block diffusion step runs its projections in three passes (the encoder's, the conditioning pass's
    and the denoiser's): its histories roll the largest any of them recorded, the step taken, not refused
    (review of row 12721: the objective dropped its records and the step was refused)."""
    from dew.nn.diffusion_gemma import DiffusionGemma
    from dew.objectives.diffusion.block import BlockDiffusionObjective

    text = CausalTransformer(vocab_size=16, emb_features=32, num_layers=2, num_heads=2, mlp_features=64,
                             max_seq_len=8, layer_scalar="trainable")
    objective = BlockDiffusionObjective(DiffusionGemma(text=text, canvas_length=4), prompt_length=4)
    rows = 8 * jax.device_count()
    batch = {"text": np.random.default_rng(12721).integers(1, 16, size=(rows, 8)).astype(np.int32)}
    with generation.run_precision(DELAYED):
        trainer = Trainer(objective, optax.adamw(1e-3), key=0, split=1)
        initial = trainer.initial_state()
        state, *_ = trainer.compile(initial, batch)(initial, batch)
    rolled_once(state)


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
