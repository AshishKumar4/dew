"""The general trainer on a small objective: the loop, checkpoints, resume,
validation, the rejected mixed-precision step and the custom step.

Nothing here knows a modality. The objective is a two-output affine map, the
data is synthetic, and what is asserted is what the trainer owns: the step
count, what lands on disk and when, what a resume restores, what reaches the
tracker, and what a failure does to the run.

Local resume combines Dew's run and data-position policies; Orbax oracles the stored state.
"""

import contextlib
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from affine_run import BATCH, FEATURES, Counting, Features, Regression, Spread, dataset, raw_leaf, val_batches
from recording import RecordingTracker
from steady_state import steady_state

from dew import position
from dew.artifacts import Representations
from dew.checkpoints import Ranking
from dew.config import OptimConfig
from dew.config.data import loader
from dew.data.dataset import DataPartition
from dew.objectives.base import Aux, EMASpec, Objective, merge, under
from dew.training import Checkpoints, Layout, MeshSpec, Trainer, TrainState, ema_update
from dew.training.optim import Cosine, Power
from dew.training.state import CHECKPOINTED


def endless():
    """A stream without get_state."""
    source = Counting()
    while True:
        yield next(source)


def make_trainer(tmp_path=None, objective=None, optimizer=None, keep=3, **kwargs):
    checkpoints = None if tmp_path is None else Checkpoints(str(tmp_path / "run"), keep=keep)
    return Trainer(
        Regression() if objective is None else objective,
        optax.sgd(0.1) if optimizer is None else optimizer,
        key=jax.random.key(0),
        layout=Layout(min_shard=1, tolerance=1.0),
        checkpoints=checkpoints,
        **kwargs,
    )


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

class TokenRegression(Regression):
    counts_tokens = True

    def real_lengths(self, batch):
        return batch["index"] + 1


@pytest.mark.parametrize("log_every", [1, 4])
@pytest.mark.parametrize("accumulation", [1, 2])
def test_a_real_token_budget_counts_each_microbatch_without_padding(log_every, accumulation):
    tracker = RecordingTracker()
    trainer = make_trainer(objective=TokenRegression(), tracker=tracker, accumulation=accumulation)
    state = trainer.fit(dataset(), tokens=30, log_every=log_every)
    assert int(state.step) == 3  # Eight rows of lengths 1, 2, 3 cross 30 at 48.
    if log_every == 1:
        assert [scalars["train/tokens"] for _, scalars in tracker.scalars if "train/tokens" in scalars] == [
            8, 16, 24]


def test_a_token_budget_and_step_cap_stop_at_the_first_reached():
    state = make_trainer(objective=TokenRegression()).fit(dataset(), steps=2, tokens=30)
    assert int(state.step) == 2


@pytest.mark.parametrize("schedule", [None, Power(peak=.1, warmup_steps=2, a=.1, b=0),
                                       Cosine(peak=.1, warmup_steps=1, decay_steps=8)])
def test_a_token_only_run_uses_the_schedules_own_horizon(schedule):
    config = OptimConfig(schedule=schedule)
    state = make_trainer(objective=TokenRegression(), optimizer=config).fit(dataset(), tokens=30)
    assert int(state.step) == 3


def test_a_token_budget_resumes_its_count_from_the_checkpoint(tmp_path):
    first = make_trainer(tmp_path, objective=TokenRegression())
    assert int(first.fit(dataset(), steps=2, tokens=30, log_every=1).step) == 2
    assert first.checkpoints.control(2)["tokens"] == 24
    resumed = make_trainer(tmp_path, objective=TokenRegression())
    assert int(resumed.fit(dataset(), tokens=30, log_every=1).step) == 3
    assert resumed.checkpoints.control(3)["tokens"] == 48
    complete = make_trainer(tmp_path, objective=TokenRegression())
    assert int(complete.fit(dataset(), tokens=30).step) == 3


def test_an_explicit_state_starts_a_token_count_with_its_own_reader(tmp_path):
    make_trainer(tmp_path, objective=TokenRegression()).fit(dataset(), steps=2, tokens=30)
    trainer = make_trainer(tmp_path, objective=TokenRegression())
    state = trainer.initial_state()
    assert int(trainer.fit(dataset(), tokens=30, state=state).step) == 3


def test_a_step_only_prefix_keeps_lifetime_real_tokens(tmp_path):
    first = make_trainer(tmp_path, objective=TokenRegression())
    first.fit(dataset(), steps=2, log_every=1)
    assert first.checkpoints.control(2)["tokens"] == 24
    resumed = make_trainer(tmp_path, objective=TokenRegression())
    assert int(resumed.fit(dataset(), tokens=30, log_every=1).step) == 3


class LargeTokenRegression(Regression):
    counts_tokens = True

    def real_lengths(self, batch):
        return batch["real_length"]


class LargeTokenRows(Counting):
    def __next__(self):
        batch = super().__next__()
        lengths = np.full(BATCH, 32768, np.int64)
        lengths[-1] -= 1
        return {**batch, "real_length": lengths}


@pytest.mark.parametrize("cadence", [1, 100])
def test_a_token_budget_stays_exact_above_float32_integer_resolution(cadence):
    trainer = make_trainer(objective=LargeTokenRegression())
    state = trainer.fit(dataset(LargeTokenRows), tokens=65 * 262143 + 1, log_every=cadence)
    assert int(state.step) == 66


@pytest.mark.parametrize("cadence", [1, 100])
def test_resumed_tokens_stay_exact_above_float32_integer_resolution(tmp_path, cadence):
    first = make_trainer(tmp_path, objective=LargeTokenRegression())
    first.fit(dataset(LargeTokenRows), steps=65, log_every=cadence)
    assert first.checkpoints.control(65)["tokens"] == 65 * 262143
    resumed = make_trainer(tmp_path, objective=LargeTokenRegression())
    state = resumed.fit(dataset(LargeTokenRows), tokens=65 * 262143 + 1, log_every=cadence)
    assert int(state.step) == 66


@pytest.mark.parametrize("variant", ["ema", "accumulation", "schedule", "dynamic_scale", "checkpoints"])
def test_steps_after_the_first_logs_neither_compile_nor_move_data_unasked(variant, tmp_path):
    """Past its first two logging intervals and checkpoints the loop reruns
    the programs it compiled, and the only data that crosses is the batches
    it places and what it reads, by name, at the logging and checkpoint
    cadences (`steady_state`). A float() of the loss in the loop, a fresh
    counter built on the host, or a counter on another device than the
    loss's would each fail it."""
    window = contextlib.ExitStack()

    class Steady(RecordingTracker):
        def log(self, scalars, step):
            super().log(scalars, step)
            if step == 8:
                window.enter_context(steady_state())
            elif step == 24:
                window.close()

    options = {"ema": {}, "accumulation": {"accumulation": 2}, "dynamic_scale": {"dynamic_scale": True},
               "schedule": {"optimizer": optax.inject_hyperparams(optax.adam)(
                   learning_rate=optax.cosine_decay_schedule(1e-2, 24))},
               "checkpoints": {"tmp_path": tmp_path}}[variant]
    tracker = Steady()
    try:
        make_trainer(tracker=tracker, **options).fit(
            dataset(), steps=24, log_every=4, checkpoint_every=4 if variant == "checkpoints" else None)
    finally:
        window.close()
    assert [step for step, scalars in tracker.scalars if "train/loss" in scalars] == [4, 8, 12, 16, 20, 24]


def test_integer_root_key_matches_a_typed_key_bit_exactly():
    integer = Trainer(Regression(), optax.adam(1e-3), key=0)
    typed = Trainer(Regression(), optax.adam(1e-3), key=jax.random.key(0))
    left = integer.fit(dataset(), steps=3, log_every=1)
    right = typed.fit(dataset(), steps=3, log_every=1)
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for actual, expected in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(np.asarray(raw_leaf(actual)), np.asarray(raw_leaf(expected)))


def test_typed_root_key_round_trips_and_resumes_bit_exactly(tmp_path):
    def build(path=None):
        return Trainer(Regression(), optax.adam(1e-3), key=0,
                       checkpoints=None if path is None else Checkpoints(str(path)))

    baseline = build().fit(dataset(), steps=4, log_every=1)
    split = build(tmp_path / "typed")
    prefix = split.fit(dataset(), steps=2, checkpoint_every=1, log_every=1)
    assert jnp.issubdtype(prefix.key.dtype, jax.dtypes.prng_key)
    fresh = build(tmp_path / "typed")
    restored, _, position = fresh.place()
    assert restored.key.dtype == prefix.key.dtype
    assert int(prefix.step) == int(restored.step) == 2
    for field in ("variables", "opt_state", "ema"):
        for before, after in zip(jax.tree.leaves(getattr(prefix, field)),
                                 jax.tree.leaves(getattr(restored, field)), strict=True):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after), err_msg=field)
    assert json.loads(position)["index"] == 2
    np.testing.assert_array_equal(jax.random.key_data(restored.key), jax.random.key_data(prefix.key))
    resumed = fresh.fit(dataset(), steps=4, checkpoint_every=1, log_every=1)
    assert int(resumed.step) == 4 and fresh.checkpoints.latest == 4
    assert (tmp_path / "typed" / "4").is_dir()
    assert not (tmp_path / "typed" / "0").exists()
    for left, right in zip(jax.tree.leaves(resumed), jax.tree.leaves(baseline), strict=True):
        np.testing.assert_array_equal(np.asarray(raw_leaf(left)), np.asarray(raw_leaf(right)))


class Keyed(Regression):
    """Loss scaled by the step key, so the key stream is observable in the parameters.

    Scaling multiplies the gradients, which a constant offset would not, so two
    runs that draw different keys land in different places.
    """

    def loss(self, variables, batch, step):
        base, aux = super().loss(variables, batch, step)
        factor = 1.0 + 0.01 * jax.random.normal(step.key, ())
        return base * factor, aux


def test_a_resumed_run_continues_the_key_stream(tmp_path):
    """A resume continues fold_in(run_key, step) where it stopped.

    The data position already decides which batches the resumed run sees, so an
    unkeyed loss cannot tell a resumed run from a restarted key stream. Scaling
    the loss by the step key makes the keys observable: a resume that restarted
    its keys, or restored the step but not the run key, lands where the
    uninterrupted run does not.
    """
    make_trainer(tmp_path, objective=Keyed()).fit(dataset(), steps=2, log_every=1)
    resumed = make_trainer(tmp_path, objective=Keyed()).fit(dataset(), steps=4, log_every=1)
    whole = make_trainer(objective=Keyed()).fit(dataset(), steps=4, log_every=1)

    for expected, actual in zip(jax.tree.leaves(whole.variables),
                                jax.tree.leaves(resumed.variables), strict=True):
        np.testing.assert_allclose(np.asarray(expected), np.asarray(actual), rtol=1e-6)


# --------------------------------------------------------------------------
# Checkpoints: what lands, when, and what a resume gets back
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cadence, steps, expected", [(None, 5, set()), (2, 5, {2, 4}), (2, 6, {2, 4, 6})])
def test_final_checkpoint_policy_is_independent_of_the_cadence(tmp_path, cadence, steps, expected):
    trainer = make_trainer(tmp_path, keep=4)
    trainer.fit(dataset(), steps=steps, checkpoint_every=cadence, checkpoint_final=False)
    assert set(trainer.checkpoints._open().all_steps()) == expected


def held_lm_trainer(container=dict, **settings):
    """A one-layer LM trainer over held weights, which the objective holds
    in `container` (Flax's FrozenDict, say)."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8)
    weights = jax.jit(LMObjective(model, seq_len=4).init)(jax.random.key(0))
    objective = LMObjective(model, seq_len=4, variables=container(weights), ema_decay=0.999)
    return Trainer(objective, optax.adam(1e-3), key=jax.random.key(0),
                   layout=Layout(min_shard=1, tolerance=1.0), **settings), objective, weights


def test_native_frozen_variables_train_save_and_resume_as_plain_ones(tmp_path):
    """A FrozenDict of variables, Flax's own container, trains, saves and
    resumes as the same tree of dicts does, to the same numbers, and a
    supplied state holding one fits: the trainer holds every tree as dicts."""
    from flax.core import FrozenDict

    batch = {"text": jnp.tile(jnp.arange(1, 6, dtype=jnp.int32)[None], (jax.device_count(), 1))}

    def trained(container, directory, steps):
        trainer, _, _ = held_lm_trainer(container, checkpoints=Checkpoints(str(directory)))
        state, _, _ = trainer.place()
        step = trainer.compile(state, batch)
        for _ in range(steps):
            state, *_ = step(state, batch)
        trainer.checkpoints.save(int(state.step), state)
        trainer.checkpoints.wait()
        return trainer, state

    _, frozen = trained(FrozenDict, tmp_path / "frozen", 2)
    assert type(frozen.variables) is dict and type(frozen.variables["params"]) is dict
    _, resumed = trained(FrozenDict, tmp_path / "frozen", 2)
    _, plain = trained(dict, tmp_path / "plain", 4)
    assert int(resumed.step) == 4
    for got, want in zip(jax.tree.leaves((resumed.variables, resumed.opt_state)),
                         jax.tree.leaves((plain.variables, plain.opt_state)), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    regression = make_trainer(objective=Regression())
    supplied = regression.initial_state()
    assert int(regression.fit(dataset(), steps=1, state=supplied.replace(
        variables=FrozenDict(supplied.variables))).step) == 1


def test_a_runs_checkpoint_opens_with_orbax_alone(tmp_path):
    """A step is one Orbax PyTree item: Orbax's own manager reads the state's
    fields by name, the PRNG key as a key, and an EMA leaf the metadata
    lists as byte planes decodes to the run's average bit for bit, as the
    checkpoint guide says."""
    import orbax.checkpoint as ocp

    trainer, _, _ = held_lm_trainer(checkpoints=Checkpoints(str(tmp_path)))
    state, _, _ = trainer.place()
    batch = {"text": jnp.tile(jnp.arange(1, 6, dtype=jnp.int32)[None], (jax.device_count(), 1))}
    step = trainer.compile(state, batch)
    for _ in range(2):
        state, *_ = step(state, batch)
    trainer.checkpoints.save(int(state.step), state)
    trainer.checkpoints.wait()

    manager = ocp.CheckpointManager(str(tmp_path), item_handlers=ocp.PyTreeCheckpointHandler())
    tree = manager.restore(manager.latest_step(), args=ocp.args.PyTreeRestore())
    deltas = manager.metadata(manager.latest_step()).custom_metadata["ema_deltas"]
    assert int(tree["step"]) == int(state.step) == 2
    assert jnp.issubdtype(tree["key"].dtype, jax.dtypes.prng_key)
    np.testing.assert_array_equal(jax.random.key_data(tree["key"]), jax.random.key_data(state.key))
    for got, want in zip(jax.tree.leaves(tree["variables"]), jax.tree.leaves(state.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    weights = {jax.tree_util.keystr(path): np.asarray(leaf)
               for path, leaf in jax.tree_util.tree_flatten_with_path(tree["variables"])[0]}
    averages = dict(jax.tree_util.tree_flatten_with_path(state.ema)[0])
    decoded = 0
    for path, stored in jax.tree_util.tree_flatten_with_path(tree["ema"])[0]:
        name, want = jax.tree_util.keystr(path), np.asarray(averages[path])
        if name in deltas:
            live = weights[name]
            unsigned = np.dtype(f"uint{8 * live.dtype.itemsize}")
            planes = np.asarray(stored)
            bits = sum(planes[k].astype(unsigned) << unsigned.type(8 * k) for k in range(len(planes)))
            stored, decoded = (live.view(unsigned) ^ bits).view(live.dtype), decoded + 1
        np.testing.assert_array_equal(np.asarray(stored), want)
    assert decoded == len(deltas) > 0


def test_overlapping_token_windows_resume_in_a_fresh_trainer_bit_exactly(tmp_path):
    from dew.config.data import TextData
    from dew.data import Loading
    from dew.data.dataset import Forwarding

    tokens = np.arange(29, dtype=np.uint16)
    tokens.tofile(tmp_path / "train.bin")
    tokens.tofile(tmp_path / "val.bin")

    class WindowsRegression(Regression):
        def loss(self, variables, batch, step):
            x = batch["text"][:, :FEATURES].astype(jnp.float32) / 32
            return super().loss(variables, {"x": x, "y": 2 * x[:, :2]}, step)

    class Observed(Forwarding):
        def __init__(self, source, seen):
            self._source, self.seen = source, seen

        def __iter__(self):
            return self

        def __next__(self):
            batch = next(self._source)
            self.seen.append(batch["text"].copy())
            return batch

        def get_state(self):
            return self._source.get_state()

        def set_state(self, state):
            self._source.set_state(state)

    def dataset(seen):
        data = loader(TextData(source=str(tmp_path), seq_len=4, stride=1), batch=BATCH, read="memory", seed=7,
            loading=Loading(workers=0, threads=1, read_buffer=1), validation_records=4 * (BATCH))
        return dataclasses.replace(data, train=lambda partition: Observed(data.train(partition), seen))

    def trainer(directory):
        return make_trainer(directory, objective=WindowsRegression(), optimizer=optax.adam(1e-3))

    whole_seen, prefix_seen, resumed_seen = [], [], []
    whole = trainer(tmp_path / "whole").fit(dataset(whole_seen), steps=4, checkpoint_every=1)
    prefix = trainer(tmp_path / "split").fit(dataset(prefix_seen), steps=2, checkpoint_every=1)
    fresh = trainer(tmp_path / "split")
    restored, _, place = fresh.place()
    assert position.read(place).records == 2 * BATCH
    assert jax.tree.structure(restored) == jax.tree.structure(prefix)
    for left, right in zip(jax.tree.leaves(restored), jax.tree.leaves(prefix), strict=True):
        actual, expected = np.asarray(raw_leaf(left)), np.asarray(raw_leaf(right))
        assert (actual.dtype, actual.shape, actual.tobytes()) == (
            expected.dtype,
            expected.shape,
            expected.tobytes(),
        )
    resumed = fresh.fit(dataset(resumed_seen), steps=4, checkpoint_every=1)
    assert resumed_seen[0].tobytes() == whole_seen[2].tobytes(), "the first batch after resume"
    assert jax.tree.structure(resumed) == jax.tree.structure(whole)
    for left, right in zip(jax.tree.leaves(resumed), jax.tree.leaves(whole), strict=True):
        actual, expected = np.asarray(raw_leaf(left)), np.asarray(raw_leaf(right))
        assert (actual.dtype, actual.shape, actual.tobytes()) == (
            expected.dtype,
            expected.shape,
            expected.tobytes(),
        )
    _, resumed_place = Checkpoints(str(tmp_path / "split/run")).restore(share=DataPartition())
    _, whole_place = Checkpoints(str(tmp_path / "whole/run")).restore(share=DataPartition())
    assert resumed_place == whole_place
    assert position.read(resumed_place).records == 4 * BATCH


@pytest.mark.gpu(reason="CUDA embedding-gradient reductions")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_a_cuda_lm_repeats_and_resumes_bit_exactly_with_deterministic_ops(tmp_path, dtype):
    """Repeated token IDs share embedding-gradient updates. CUDA's default
    scatter-add order is not repeatable; conftest enables deterministic ops
    before the backend opens. Check every state leaf, not just parameters.
    bf16 at the default precision runs the head that rounds its logits and
    their gradient to bf16, and resumes bit-exactly too."""
    from dew.config.data import TextData
    from dew.data import Loading
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    tokens = np.tile(np.arange(8, dtype=np.uint8), 256)
    tokens.tofile(tmp_path / "train.bin")
    tokens[:128].tofile(tmp_path / "val.bin")
    (tmp_path / "meta.json").write_text(json.dumps({
        "tokenizer": "symbols", "vocab_size": 8, "dtype": "uint8",
        "train_tokens": len(tokens), "val_tokens": 128, "eos_id": None,
    }))
    data = loader(TextData(source=str(tmp_path), seq_len=16), batch=8, read="memory",
        loading=Loading(workers=0), validation_records=4 * (8))
    model = CausalTransformer(vocab_size=8, emb_features=16, num_layers=1,
                              num_heads=2, mlp_features=32, max_seq_len=32, dtype=dtype)
    objective = LMObjective(model, seq_len=16, ema_decay=None)

    def trainer(checkpoints=None):
        return Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0),
                       checkpoints=checkpoints)

    baseline = trainer().fit(data, steps=3)
    repeated = trainer().fit(data, steps=3)
    split = trainer(Checkpoints(str(tmp_path / "checkpoints")))
    prefix = split.fit(data, steps=2, checkpoint_every=1)
    restarted = trainer(Checkpoints(str(tmp_path / "checkpoints")))
    restored, _, position = restarted.place()
    assert position is not None
    resumed = restarted.fit(data, steps=3, checkpoint_every=1)
    for actual, expected in ((restored, prefix), (repeated, baseline), (resumed, baseline)):
        assert jax.tree.structure(actual) == jax.tree.structure(expected)
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(np.asarray(raw_leaf(left)), np.asarray(raw_leaf(right)))


REPEATED_CONV_STEPS = """
import sys

import jax
import numpy as np
import optax

from dew.diffusion import presets
from dew.inputs import Field, InputSpec
from dew.objectives.diffusion import DiffusionObjective
from dew.nn.backbones import SimpleDiT
from dew.sampling import Euler
from dew.training import Trainer

model = SimpleDiT(patch_size=2, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2)
objective = DiffusionObjective(model, presets.Flow()(), InputSpec(Field("image", (16, 16, 3))), guidance=None,
                               solver=Euler(), steps=2, ema_decay=None)
trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0))
state = trainer.initial_state()
batch = {"image": np.random.default_rng(0).integers(0, 256, (32, 16, 16, 3)).astype(np.uint8)}
step = trainer.compile(state, batch)
for _ in range(2):
    state, *_ = step(state, batch)
leaves = jax.tree.leaves(jax.tree.map(lambda leaf: jax.random.key_data(leaf)
                                      if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
                                      else leaf, state))
np.savez(sys.argv[1], *[np.asarray(leaf) for leaf in leaves])
"""


@pytest.mark.gpu(reason="cuDNN convolution autotuning")
def test_two_processes_train_a_conv_model_to_the_bit_under_the_repeatable_flags(tmp_path):
    """Autotuning picks a convolution's cuDNN algorithm per process, so only
    a second process shows whether it agrees. Two fresh processes, without a
    compilation cache, train a DiT (a convolution embeds its patches) two
    Adam steps under the cuda lane's flags; every state leaf agrees."""
    from lane_environment import REPEATABLE_GPU_FLAGS

    root = Path(__file__).resolve().parents[1]
    flags = [flag for flag in os.environ.get("XLA_FLAGS", "").split()
             if not flag.startswith(("--xla_gpu_deterministic_ops", "--xla_gpu_autotune_level"))]
    environment = {**os.environ, "JAX_PLATFORMS": "cuda", "PYTHONPATH": str(root / "src"),
                   "XLA_FLAGS": " ".join([*flags, *REPEATABLE_GPU_FLAGS]),
                   "JAX_ENABLE_COMPILATION_CACHE": "false"}
    runs = []
    for index in range(2):
        out = tmp_path / f"state-{index}.npz"
        done = subprocess.run([sys.executable, "-c", REPEATED_CONV_STEPS, str(out)], env=environment,
                              capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, done.stderr
        runs.append(np.load(out))
    assert runs[0].files == runs[1].files
    for name in runs[0].files:
        np.testing.assert_array_equal(runs[0][name], runs[1][name])


def test_the_compiled_step_consumes_the_state_it_is_given():
    """`new = step(old, batch)` runs the update in place: the old state's
    buffers move into the new one, so peak memory holds one copy of the
    parameters and optimizer state. Reading the old state afterwards is
    the caller's error, and JAX names it."""
    trainer = make_trainer()
    state, _, _ = trainer.place()
    batch = next(Counting())
    step = trainer.compile(state, batch)
    stale = jax.tree.leaves(state.variables)[0]
    advanced, _loss, _, finite, _, _ = step(state, batch)
    assert bool(finite) and int(advanced.step) == 1
    assert stale.is_deleted()
    again, _, _, _, _, _ = step(advanced, batch)
    assert int(again.step) == 2


def test_restore_preserves_the_optimizer_state_the_ema_and_the_key(tmp_path):
    trainer = make_trainer(tmp_path, optimizer=optax.adam(1e-3))
    trained = trainer.fit(dataset(), steps=3, log_every=1)

    resumed = make_trainer(tmp_path, optimizer=optax.adam(1e-3))
    state, _, position = resumed.place()

    assert int(state.step) == 3, "the step counter was reset"
    for field in ("variables", "opt_state", "ema"):
        for before, after in zip(jax.tree.leaves(getattr(trained, field)),
                                 jax.tree.leaves(getattr(state, field)), strict=True):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after), err_msg=field)
    assert jnp.array_equal(jax.random.key_data(trained.key), jax.random.key_data(state.key))
    assert json.loads(position)["index"] == 3


def _rewrite_position(trainer, step, rows, shares):
    """Save `step` again with `rows` as the checkpoint's position table, each
    row read for the share `shares` names."""
    restored, _ = trainer.checkpoints.restore()
    written = [np.frombuffer(row, np.uint8) for row in rows]
    table = {"rows": np.stack(written),
             "lengths": np.array([len(row) for row in written], np.int64),
             "shares": np.array(shares, np.int64)}
    manager = trainer.checkpoints._open()
    manager.save(step, args=ocp.args.PyTreeSave({**restored, "position": table}), force=True)
    manager.wait_until_finished()


def test_a_global_position_is_read_by_any_process_count(tmp_path):
    """A global position is a record count over an order every process count
    reads the same way, so the table two processes wrote is this one
    process's position as well. Only a shard offset ties a resume to the
    count that wrote it."""
    trainer = make_trainer(tmp_path)
    trainer.fit(dataset(), steps=1, log_every=1)
    global_position = position.encode(position.Global(records=16, order="Counting"))
    _rewrite_position(trainer, 2, [global_position, global_position], shares=[[0, 2], [1, 2]])

    assert Checkpoints(str(tmp_path / "run")).restore(step=2, share=DataPartition())[1] == global_position


def held_state(params, ema):
    """A train state that holds `params` and `ema` and scalars beside them."""
    count = jnp.zeros((), jnp.int32)
    return TrainState(step=count, microstep=count, updates=count, variables=params,
                      opt_state={"count": count}, ema=ema,
                      key=jax.random.key_data(jax.random.key(0)), scale=None,
                      window_size=jnp.ones((), jnp.int32), accumulation=None)


def bit_patterns(shape, dtype, seed):
    """Values drawn bit by bit, so NaN payloads and subnormals turn up, with
    both zeros, a NaN of every payload bit set and the smallest subnormal first."""
    kind = np.dtype(f"u{np.dtype(dtype).itemsize}")
    bits = np.random.default_rng(seed).integers(0, np.iinfo(kind).max, size=shape,
                                                dtype=kind, endpoint=True)
    bits.flat[:4] = [1 << (8 * kind.itemsize - 1), 0, np.iinfo(kind).max, 1]
    return bits.view(dtype)


def same_bits(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    return (expected.dtype == actual.dtype and expected.shape == actual.shape
            and np.array_equal(expected.view(np.uint8), actual.view(np.uint8)))


def awkward_state():
    """Weights in fp32 and bf16, and an EMA of each dtype, one of them
    narrower than the weight it follows."""
    params = {"params": {"kernel": bit_patterns((16, 8), np.float32, 0),
                         "scale": bit_patterns((8,), jnp.bfloat16, 1),
                         "bias": bit_patterns((8,), np.float32, 2)}}
    ema = {"params": {"kernel": bit_patterns((16, 8), np.float32, 3),
                      "scale": bit_patterns((8,), jnp.bfloat16, 4),
                      "bias": bit_patterns((8,), jnp.bfloat16, 5)}}
    return held_state(jax.tree.map(jnp.asarray, params), jax.tree.map(jnp.asarray, ema))


def test_the_ema_comes_back_bit_for_bit_however_it_is_read(tmp_path):
    """Every NaN payload, signed zero and subnormal of the EMA survives a save,
    read whole, typed, without the weights beside it, into pinned host memory
    and as the shapes the checkpoint reports."""
    state = awkward_state()
    checkpoints = Checkpoints(str(tmp_path / "run"))
    checkpoints.save(0, state, None)
    checkpoints.wait()
    checkpoints = Checkpoints(str(tmp_path / "run"))

    def typed(tree, memory_kind="device"):
        where = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind=memory_kind)
        return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=where),
                            tree)

    whole, _ = checkpoints.restore()
    ema_alone, _ = checkpoints.restore({"ema": typed(state.ema)})
    pinned, _ = checkpoints.restore({"variables": typed(state.variables),
                                     "ema": typed(state.ema, "pinned_host")})
    resumed, _ = checkpoints.restore(held_state(typed(state.variables), typed(state.ema)))
    for restored in (whole["ema"], ema_alone["ema"], pinned["ema"], resumed.ema):
        assert jax.tree.all(jax.tree.map(same_bits, state.ema, restored))
    assert jax.tree.all(jax.tree.map(same_bits, state.variables, resumed.variables))
    assert all(leaf.sharding.memory_kind == "pinned_host" for leaf in jax.tree.leaves(pinned["ema"]))
    assert jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), checkpoints.stored()["ema"]) == \
        jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), state.ema)


def test_an_ema_held_in_lists_comes_back_bit_for_bit(tmp_path):
    """Variables may nest sequences; the EMA's leaves inside them are found by
    their key paths, read whole, alone, and beside weights asked for in
    another dtype, which sends the weights they undo to a read of their own."""
    kernels = [bit_patterns((4, 3), np.float32, seed) for seed in (0, 1)]
    params = {"params": {"layers": [{"w": jnp.asarray(kernel)} for kernel in kernels]}}
    ema = {"params": {"layers": [{"w": jnp.asarray(bit_patterns((4, 3), np.float32, seed))}
                                 for seed in (2, 3)]}}
    checkpoints = Checkpoints(str(tmp_path / "run"))
    checkpoints.save(0, held_state(params, ema), None)
    checkpoints.wait()
    checkpoints = Checkpoints(str(tmp_path / "run"))
    where = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def typed(tree, dtype=None):
        return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, dtype or leaf.dtype,
                                                              sharding=where), tree)

    whole, _ = checkpoints.restore()
    alone, _ = checkpoints.restore({"ema": typed(ema)})
    beside, _ = checkpoints.restore({"variables": typed(params, jnp.bfloat16), "ema": typed(ema)})
    for restored in (whole["ema"], alone["ema"], beside["ema"]):
        assert jax.tree.all(jax.tree.map(same_bits, ema, restored))


def test_a_checkpoint_that_stores_the_ema_as_itself_still_restores(tmp_path):
    """Checkpoints written before the EMA was stored as its difference from
    the weights hold it as plain arrays, and read back as they were written."""
    state = awkward_state()
    written = ocp.CheckpointManager(str(tmp_path / "run"), item_handlers=ocp.PyTreeCheckpointHandler())
    written.save(0, args=ocp.args.PyTreeSave({name: getattr(state, name) for name in CHECKPOINTED}))
    written.wait_until_finished()

    checkpoints = Checkpoints(str(tmp_path / "run"))
    template = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype,
                                                              sharding=leaf.sharding), state.ema)
    for restored in (checkpoints.restore()[0]["ema"], checkpoints.restore({"ema": template})[0]["ema"]):
        assert jax.tree.all(jax.tree.map(same_bits, state.ema, restored))


def test_the_best_step_is_the_lowest_loss(tmp_path):
    """The loss rides along with the save, and the lowest one is what orbax
    keeps and reports however the run wanders; a metric-less save is newer
    than the best and must not displace it."""
    checkpoints = Checkpoints(str(tmp_path / "best"), keep=1)
    state = make_trainer().initial_state()
    for step, loss in ((1, 0.9), (2, 0.3), (3, 0.7)):
        checkpoints.save(step, state.replace(step=jnp.asarray(step)), None,
                         ranking=Ranking("train/loss", loss))
    checkpoints.wait()

    assert checkpoints.best == 2
    assert set(checkpoints._open().all_steps()) == {2, 3}

    checkpoints.save(4, state.replace(step=jnp.asarray(4)), None)
    checkpoints.wait()
    assert checkpoints.best == 2
    assert set(checkpoints._open().all_steps()) == {2, 4}
    assert Checkpoints(str(tmp_path / "best")).best == 2, "the metric did not survive a reopen"


# --------------------------------------------------------------------------
# Local checkpoints
# --------------------------------------------------------------------------

def local_trainer(tmp_path, **kwargs):
    return Trainer(
        Regression(), optax.sgd(0.1), key=jax.random.key(0),
        layout=Layout(min_shard=1, tolerance=1.0),
        checkpoints=Checkpoints(str(tmp_path / "run"), keep=4,
                                local_directory=str(tmp_path / "local"), local_every=2),
        **kwargs)


def test_terminal_local_cadence_does_not_need_a_final_persistent_save(tmp_path):
    trainer = local_trainer(tmp_path)
    trainer.fit(dataset(), steps=2, checkpoint_final=False)
    restored, _, _ = local_trainer(tmp_path).place()
    assert int(restored.step) == 2
    assert not os.path.exists(tmp_path / "run" / "2")


def test_the_local_checkpoint_is_written_on_its_own_cadence_and_keeps_the_newest(tmp_path):
    """Local every two, persistent every three: after eight steps the
    persistent directory holds 3, 6 and the final 8 with their losses, the
    local one holds 8 alone (2, 4 and 6 replaced), and the persistent directory holds nothing of the
    local cadence."""
    trainer = local_trainer(tmp_path)
    trainer.fit(dataset(), steps=8, log_every=1, checkpoint_every=3)
    checkpoints = trainer.checkpoints

    assert sorted(checkpoints._open().all_steps()) == [3, 6, 8]
    assert sorted(checkpoints._open_local().all_steps()) == [8]
    assert checkpoints.best in (3, 6, 8)
    assert checkpoints.local_path == str(tmp_path / "local" / "process0")
    assert (tmp_path / "local" / "process0" / "8" / "commit_success.txt").exists()
    assert not (tmp_path / "local" / "process0" / "4").exists()
    assert not (tmp_path / "run" / "4").exists()


class ExplodingManager:
    """Orbax when the filesystem refuses the write.

    Stubbed, not provoked with a read-only directory: a real failed async
    orbax write leaves a background thread that never joins, which hangs
    interpreter exit and with it the whole test session.
    """

    def latest_step(self):
        return None

    def save(self, *args, **kwargs):
        raise OSError("No space left on device")

    def wait_until_finished(self):
        pass


def test_a_checkpoint_that_does_not_land_fails_the_run(tmp_path):
    """A checkpoint that did not get written is data loss, not a log line."""
    trainer = make_trainer(tmp_path)
    trainer.checkpoints._manager = ExplodingManager()
    with pytest.raises(OSError):
        trainer.fit(dataset(), steps=1, log_every=1)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_eval_every_scores_the_validation_split_and_logs_the_artifacts():
    seen = []
    tracker = RecordingTracker()
    objective = Features()
    trainer = make_trainer(objective=objective, tracker=tracker)
    state = trainer.fit(dataset(val=val_batches(3)), steps=4, log_every=2, eval_every=2,
                        metrics=(Spread(seen),), preview=True)

    # Two passes: at step 2, and at the end of the run.
    assert seen == [((BATCH, 2), (BATCH, FEATURES))] * 6
    scored = [(step, s) for step, s in tracker.scalars if "val/spread" in s]
    assert [step for step, _ in scored] == [2, 4]
    # The pass at the end scores the averaged weights over the whole split, so
    # its spread is the mean of the three batches' own.
    expected = np.mean([float(jnp.std(objective.model.apply(state.averaged, batch["x"])))
                        for batch in val_batches(3)()])
    assert scored[-1][1]["val/spread"] == pytest.approx(expected, rel=1e-6)
    # The first batch's artifact of each pass reaches the tracker.
    assert [step for step, value in tracker.artifacts if isinstance(value, Representations)] == [2, 4]


def test_each_suite_is_scored_on_its_own_cadence_with_its_own_metrics():
    """Two named suites: one every 2 steps with fit's metric, one every 3 with
    its own; each at its multiples and at the end, under its own name."""
    from dew.training import EvalSuite

    tracker = RecordingTracker()
    trainer = make_trainer(objective=Features(), tracker=tracker)
    trainer.fit(dataset(val=None), steps=6, log_every=2, eval_every=2, metrics=(Spread([]),),
                validation={"often": dataset(val=val_batches(1)).val,
                            "rare": EvalSuite(dataset(val=val_batches(2)).val, (Spread([]),), 3)})
    steps = {name: [step for step, scores in tracker.scalars if f"{name}/spread" in scores]
             for name in ("often", "rare")}
    assert steps == {"often": [2, 4, 6], "rare": [3, 6]}


def test_the_validation_split_is_the_one_suite_val():
    """`dataset.val` alone and a one-suite mapping of it are the same run."""
    from dew.training import EvalSuite

    runs = []
    held = dataset(val=val_batches(3)).val
    for validation in (None, {"val": held}, {"val": EvalSuite(held, (Spread([]),), 3)}):
        tracker = RecordingTracker()
        make_trainer(objective=Features(), tracker=tracker).fit(
            dataset(val=val_batches(3)), steps=6, log_every=2, eval_every=3, metrics=(Spread([]),),
            validation=validation)
        runs.append([(step, scores["val/spread"]) for step, scores in tracker.scalars
                     if "val/spread" in scores])
    assert runs[0] == runs[1] == runs[2] and [step for step, _ in runs[0]] == [3, 6]


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def test_the_log_tick_carries_the_loss_the_objective_metrics_and_the_throughput():
    tracker = RecordingTracker()
    make_trainer(tracker=tracker).fit(dataset(endless), steps=4, log_every=2)

    ticks = [(step, scalars) for step, scalars in tracker.scalars if "train/loss" in scalars]
    assert [step for step, _ in ticks] == [2, 4]
    for _, scalars in ticks:
        assert scalars["train/probe"] == 1.0
        assert np.isfinite(scalars["train/loss"])
        assert scalars["train/step_time_ms"] > 0
        assert scalars["train/samples_per_sec"] > 0


# --------------------------------------------------------------------------
# Divergence
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Rejected mixed-precision steps
# --------------------------------------------------------------------------

class ScaledObjective(Objective):
    """loss = scale * sum(w^2), with the scale carried by the batch so that
    one batch overflows the scaled float32 loss while the params stay sane."""

    def __init__(self):
        self.ema = EMASpec(decay=lambda step: 0.5)

    def init(self, key, variables=None):
        return {"params": {"w": jnp.ones((2,))}}

    def loss(self, variables, batch, step):
        return jnp.sum(variables["params"]["w"] ** 2) * batch["scale"][0], Aux({})


def host(state):
    return np.array(state.variables["params"]["w"]), np.array(state.ema["params"]["w"])


@pytest.mark.parametrize("accum", [1, 2])
def test_a_rejected_dynamic_scale_step_leaves_no_trace(accum):
    trainer = Trainer(ScaledObjective(), optax.sgd(.1), key=jax.random.key(0),
                      accumulation=accum, dynamic_scale=True)
    state = trainer.initial_state()
    good = {"scale": jnp.ones((jax.device_count(),), jnp.float32)}
    bad = {"scale": jnp.full((jax.device_count(),), 1e35, jnp.float32)}
    step = trainer.compile(state, good)
    for _ in range(2 * accum - 1):
        state, *_ = step(state, good)
    w, ema = host(state)
    np.testing.assert_allclose(w, .8, rtol=1e-6)
    np.testing.assert_allclose(ema, .9, rtol=1e-6)
    # the step consumes the state
    counted, advanced, scaled = int(state.step), int(state.microstep), float(state.scale.scale)
    state, _, _, finite, accepted, _ = step(state, bad)
    assert bool(finite) and not bool(accepted)
    assert int(state.step) == counted + 1
    assert int(state.microstep) == advanced
    assert float(state.scale.scale) == scaled / 2
    np.testing.assert_array_equal(state.variables["params"]["w"], w)
    np.testing.assert_array_equal(state.ema["params"]["w"], ema)
    state, *_ = step(state, good)
    w, ema = host(state)
    np.testing.assert_allclose(w, .64, rtol=1e-6)
    np.testing.assert_allclose(ema, .77, rtol=1e-6)
    assert int(state.microstep) == 2 * accum


# --------------------------------------------------------------------------
# The collections an objective writes back
# --------------------------------------------------------------------------

class Counted(Objective):
    """A `stats` collection the loss advances by one every step."""

    ema = None

    def init(self, key, variables=None):
        return {"params": {"w": jnp.ones((2,))}, "stats": {"seen": jnp.zeros(())}}

    def loss(self, variables, batch, step):
        loss = jnp.sum(variables["params"]["w"] ** 2)
        return loss, Aux({}, variables={"stats": {"seen": variables["stats"]["seen"] + 1}})


def test_aux_variables_are_written_back_into_the_state_and_checkpointed(tmp_path):
    trainer = make_trainer(tmp_path, objective=Counted(), optimizer=optax.adamw(1e-2, weight_decay=1.0))
    state = trainer.fit(dataset(), steps=3, log_every=1)
    assert float(state.variables["stats"]["seen"]) == 3.0
    assert set(state.opt_state[0].mu) == {"w"}

    restored, _ = trainer.checkpoints.restore()
    assert float(restored["variables"]["stats"]["seen"]) == 3.0


# --------------------------------------------------------------------------
# The EMA over a subtree
# --------------------------------------------------------------------------

class TwoTrees(Objective):
    """Two independent parameter subtrees, so EMA scoping is observable."""

    def __init__(self, ema):
        self.ema = ema

    def init(self, key, variables=None):
        return {"params": {"tracked": {"w": jnp.ones((2,))},
                           "untracked": {"w": jnp.ones((2,))}}}

    def loss(self, variables, batch, step):
        total = sum(jnp.sum(leaf ** 2) for leaf in jax.tree.leaves(variables))
        return total, Aux({})


def test_the_ema_stores_only_the_selected_subtree_and_merges_over_the_rest():
    objective = TwoTrees(EMASpec(decay=optax.constant_schedule(0.5),
                                 select=under("params", "tracked")))
    state = make_trainer(objective=objective).fit(dataset(endless), steps=2, log_every=1)

    assert set(state.ema["params"]) == {"tracked"}
    live = state.variables["params"]
    assert not np.allclose(state.ema["params"]["tracked"]["w"], live["tracked"]["w"])
    merged = merge(state.variables, state.ema)
    assert merged["params"]["untracked"] is live["untracked"]
    np.testing.assert_array_equal(merged["params"]["tracked"]["w"], state.ema["params"]["tracked"]["w"])
    params = {"params": {"tracked": {"w": jnp.full((2,), 3.0)}, "untracked": {"w": jnp.zeros((2,))}}}
    average = {"params": {"tracked": {"w": jnp.ones((2,))}}}
    updated = ema_update(average, params, 0.5)
    assert set(updated["params"]) == {"tracked"}
    np.testing.assert_allclose(updated["params"]["tracked"]["w"], 2.0)


# --------------------------------------------------------------------------
# The custom step
# --------------------------------------------------------------------------

class TwoPlayers(Objective):
    """A generator that moves a scalar toward a target and a discriminator
    that tracks the generator: two losses, two parameter groups."""

    ema = None

    def init(self, key, variables=None):
        return {"params": {"gen": {"g": jnp.zeros(())}, "disc": {"d": jnp.zeros(())}}}

    def loss(self, variables, batch, step):
        raise AssertionError("the custom step never calls the single loss")

    def generator_loss(self, params, batch):
        return (params["gen"]["g"] - 1.0) ** 2

    def discriminator_loss(self, params, batch):
        return (params["disc"]["d"] - jax.lax.stop_gradient(params["gen"]["g"])) ** 2


def two_optimizers(gen, disc):
    """Both optimizers' states under one init; the step updates one at a time."""
    def init(params):
        return {"gen": gen.init(params["gen"]), "disc": disc.init(params["disc"])}

    def update(*_):
        raise AssertionError("the alternating step drives the two optimizers itself")

    return optax.GradientTransformation(init, update)


def alternating(gen, disc):
    def make_step(objective, optimizer):
        def step(state, batch):
            trainable = state.variables["params"]

            def generator(operand):
                loss, grads = jax.value_and_grad(objective.generator_loss)(trainable, batch)
                updates, gen_state = gen.update(grads["gen"], state.opt_state["gen"], trainable["gen"])
                params = {**trainable, "gen": optax.apply_updates(trainable["gen"], updates)}
                return params, {**state.opt_state, "gen": gen_state}, loss

            def discriminator(operand):
                loss, grads = jax.value_and_grad(objective.discriminator_loss)(trainable, batch)
                updates, disc_state = disc.update(grads["disc"], state.opt_state["disc"], trainable["disc"])
                params = {**trainable, "disc": optax.apply_updates(trainable["disc"], updates)}
                return params, {**state.opt_state, "disc": disc_state}, loss

            params, opt_state, loss = jax.lax.cond(state.microstep % 2 == 0, generator, discriminator, None)
            new_state = state.replace(
                microstep=state.microstep + 1,
                updates=state.updates + 1,
                opt_state=opt_state,
                variables={**state.variables, "params": params},
            )
            return new_state, loss, Aux({"player": (state.microstep % 2).astype(jnp.float32)})
        return step
    return make_step


def test_a_custom_step_alternates_two_optimizers_on_the_same_checkpoints_and_tracker(tmp_path):
    """The escape hatch: a GAN-style step with two optimizers, checkpointed
    and logged by the same trainer, resumed from the same directory."""
    gen, disc = optax.sgd(0.25), optax.sgd(0.25)
    tracker = RecordingTracker()

    def trainer():
        return Trainer(TwoPlayers(), two_optimizers(gen, disc), key=jax.random.key(0),
                       layout=Layout(min_shard=1, tolerance=1.0),
                       checkpoints=Checkpoints(str(tmp_path / "gan"), keep=2),
                       tracker=tracker, step=alternating(gen, disc))

    state = trainer().fit(dataset(), steps=2, log_every=1)
    # Step 0 moved the generator half way to 1 (lr 0.25 on a gradient of 2(g - 1));
    # step 1 moved the discriminator half way to the generator.
    assert float(state.variables["params"]["gen"]["g"]) == pytest.approx(0.5)
    assert float(state.variables["params"]["disc"]["d"]) == pytest.approx(0.25)
    assert [s["train/player"] for _, s in tracker.scalars if "train/player" in s] == [0.0, 1.0]

    resumed = trainer().fit(dataset(), steps=4, log_every=1)
    assert float(resumed.variables["params"]["gen"]["g"]) == pytest.approx(0.75)
    assert float(resumed.variables["params"]["disc"]["d"]) == pytest.approx(0.5)
    assert Checkpoints(str(tmp_path / "gan")).latest == 4


@pytest.mark.mesh
def test_a_fresh_state_is_built_in_the_buffers_its_held_checkpoint_arrives_in(monkeypatch):
    """A held checkpoint reaches the state's JIT as a copy placed where the
    state keeps each variable, sharded as it is, and the JIT takes those
    buffers over. Handed over as they are, the arrays were placed below the
    state and freed after it was built, a hole as large as the checkpoint:
    on an A100 Qwen3-1.7B's 'minimal' rung then found no block for its
    temporaries. The objective's own arrays stay as they were."""
    from jax.tree_util import Partial

    held = {}
    put = jax.device_put

    def recorded(x, *args, **kwargs):
        out = put(x, *args, **kwargs)
        if isinstance(out, Partial):
            held.update(jax.tree_util.tree_leaves_with_path(out.keywords["variables"]))
        return out

    monkeypatch.setattr(jax, "device_put", recorded)
    trainer, _objective, weights = held_lm_trainer(mesh=MeshSpec(fsdp=jax.device_count()))
    state, shardings, _ = trainer.place()
    params = dict(jax.tree_util.tree_leaves_with_path(shardings.variables))
    sharded = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(weights):
        copy = held[path]
        assert copy.is_deleted(), jax.tree_util.keystr(path)
        assert copy.sharding == params[path], jax.tree_util.keystr(path)
        sharded += not copy.sharding.is_fully_replicated
        assert not leaf.is_deleted()
    assert sharded
    jax.tree.map(np.testing.assert_array_equal, state.variables, weights)


def test_a_step_compiles_from_its_arrays_shapes_before_they_are_placed():
    """Compiling reads the parameters', optimizer state's and key's shapes
    and shardings, not their values, so a run compiles its step ahead of
    time, into the persistent cache, before it holds them on devices; the
    trainer's own settings (the window, the scaler) stay values. The loss's
    shape once folded the step into the key eagerly, which a key known only
    by its shape cannot do (74761deb)."""
    trainer, _, _ = held_lm_trainer()
    state, shardings, _ = trainer.place()

    def shape(leaf, sharding):
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding)

    abstract = dataclasses.replace(
        state, variables=jax.tree.map(shape, state.variables, shardings.variables),
        opt_state=jax.tree.map(shape, state.opt_state, shardings.opt_state),
        key=shape(state.key, shardings.key))

    from dew.objectives.base import Step

    compiled = trainer.compile(abstract, {"text": jax.ShapeDtypeStruct((8, 5), jnp.int32)})
    batch = {"text": jnp.zeros((8, 5), jnp.int32)}
    expected, _ = trainer.objective.scalar_loss(state.variables, batch,
                              Step(state.microstep, jax.random.fold_in(state.key, state.step), None))
    advanced, loss, _, finite, _, _ = jax.block_until_ready(compiled(state, batch))
    assert loss == pytest.approx(float(expected), rel=1e-6)
    assert int(advanced.step) == 1 and bool(finite)


def test_a_field_no_device_holds_stays_on_the_host_for_the_code_that_reads_it():
    """A batch may carry a field no device holds, a caption: the step and the
    objective's evaluation take the rest, and a metric reads the caption on
    the host, for the rows it scores."""

    def captioned(source):
        for batch in source:
            yield {**batch, "caption": np.array([f"row {index}" for index in batch["index"]])}

    captions = []

    class Captions(Spread):
        def __call__(self, artifact, batch):
            captions.extend(batch["caption"].tolist())
            return super().__call__(artifact, batch)

    data = dataset(lambda: captioned(Counting()), val=lambda: captioned(val_batches(1)()))
    state = Trainer(Features(), optax.sgd(.1), key=jax.random.key(0)).fit(
        data, steps=2, eval_every=2, metrics=(Captions([]),), checkpoint_final=False)
    scored = next(val_batches(1)())["index"]
    assert int(state.step) == 2 and captions == [f"row {index}" for index in scored]


def test_a_failing_metric_fails_the_validation_pass():
    class Broken:
        name = "broken"
        reads = Representations

        def __call__(self, artifact, batch):
            raise ZeroDivisionError("metric over an empty batch")

        def merge(self, accumulated, contribution):
            return accumulated + contribution

        def finalize(self, accumulated):
            return accumulated

    with pytest.raises(ZeroDivisionError):
        make_trainer(objective=Features()).fit(dataset(val=val_batches(1)), steps=1,
                                               log_every=1, eval_every=1, metrics=(Broken(),))
