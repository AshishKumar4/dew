"""Effective-loss and checkpoint transactions over tiny deterministic objectives."""
import dataclasses
import json

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.checkpoints import Checkpoints
from dew.objectives import Aux, EMASpec, Mean, Objective, mean_loss, scalar_loss
from dew.training import Trainer
from dew.objectives.base import under


@struct.dataclass
class Terms:
    prediction: Mean
    rows: Mean
    scores: jax.Array
    counts: jax.Array
    positions: jax.Array


class Tiny(Objective):
    ema = EMASpec(optax.constant_schedule(.5), select=under("params"))

    def __init__(self, composite=False):
        self.composite = composite

    def init(self, key):
        return {"params": {"w": jnp.array(.2)}, "stats": {"seen": jnp.array(0.)}}

    def loss(self, variables, batch, step):
        w = variables["params"]["w"]
        noise = jax.random.uniform(step.key, batch["y"].shape) * .03
        prediction = (w + noise + variables["stats"]["seen"] * .01
                      + step.ema["stats"]["seen"] * .1 + step.step * .02)
        errors = (prediction - batch["y"]) ** 2
        bad = jax.lax.cond(batch["bad"],
                           lambda x: jnp.sqrt(x - jax.lax.stop_gradient(x)),
                           lambda x: x * 0, w)
        main = Mean(jnp.sum(errors * batch["mask"]) + bad, jnp.sum(batch["mask"]))
        aux = Aux({}, variables={"stats": {"seen": variables["stats"]["seen"] + 1}})
        if not self.composite:
            return main, aux
        logits = jnp.stack((prediction, -prediction, prediction * .3), axis=-1)
        scores = jax.nn.softmax(logits, axis=-1)
        counts = jnp.bincount(jnp.argmax(scores, axis=-1), length=3)
        row = Mean(jnp.sum((prediction + batch["y"]) ** 2) * batch["active"],
                   jnp.asarray(batch["y"].size) * batch["active"])
        return Terms(main, row, jnp.sum(scores, axis=0) * batch["active"],
                     counts * batch["active"], jnp.asarray(scores.shape[0]) * batch["active"]), aux

    def reduce_loss(self, stats):
        if isinstance(stats, Mean):
            return mean_loss(stats)
        main, a = mean_loss(stats.prediction)
        row, b = mean_loss(stats.rows)
        positions = jnp.where(stats.positions > 0, stats.positions, 1)
        auxiliary = .2 * 3 * jnp.vdot(jax.lax.stop_gradient(stats.counts), stats.scores) / positions ** 2
        return main + .1 * row + auxiliary, a | b | (stats.positions > 0)

def batches():
    return [dict(y=jnp.tile(jnp.array([1., 2.]), jax.device_count()),
                 mask=jnp.tile(jnp.array(mask), jax.device_count()),
                 bad=jnp.array(bad), active=jnp.array(1))
            for mask, bad in [([1., 0.], False), ([1., 1.], True), ([.5, 1.], False), ([1., 1.], False)]]

class ShortScaleTrainer(Trainer):
    def initial_state(self):
        state = super().initial_state()
        return dataclasses.replace(state, scale=dataclasses.replace(state.scale, growth_interval=1))


def trainer(composite=False, k=2, checkpoints=None):
    return ShortScaleTrainer(Tiny(composite), optax.adamw(.03, weight_decay=.1), key=jax.random.PRNGKey(9),
                   accumulation=k, dynamic_scale=True, checkpoints=checkpoints)


@pytest.mark.parametrize("composite", [False, True])
def test_boundary_rejection_preserves_accepted_prefix_and_mutable_reads(composite):
    train = trainer(composite)
    initial = train.initial_state()
    data = batches()
    run = train.compile(initial, data[0])
    prefix, *_ = run(initial, data[0])
    rejected, _, _, finite, accepted = run(prefix, data[1])
    assert bool(finite) and not bool(accepted)
    assert int(rejected.step) == 2 and int(rejected.microstep) == 1
    assert float(rejected.scale.scale) == float(prefix.scale.scale) / 2
    for before, after in zip(jax.tree.leaves(prefix.params), jax.tree.leaves(rejected.params), strict=True):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(jax.tree.leaves(prefix.accumulation), jax.tree.leaves(rejected.accumulation), strict=True):
        np.testing.assert_array_equal(before, after)
    final, _, _, _, accepted = run(rejected, data[2])
    assert bool(accepted) and int(final.updates) == 1
    assert float(final.params["stats"]["seen"]) == 2
    assert float(initial.params["stats"]["seen"]) == 0

    def combined(w):
        stats = []
        for attempt, seen in [(0, 0.), (2, 1.)]:
            variables = {"params": {"w": w}, "stats": {"seen": jnp.array(seen)}}
            from dew.objectives import Step
            info = Step(jnp.asarray(int(seen)), jax.random.fold_in(initial.key, attempt), variables)
            value, _ = train.objective.loss(variables, data[attempt], info)
            stats.append(value)
        pooled = jax.tree.map(lambda a, b: a + b, *stats)
        return train.objective.reduce_loss(pooled)[0]
    gradient = jax.grad(combined)(initial.params["params"]["w"])
    update, expected_opt = train.optimizer.update({"w": gradient}, initial.opt_state, initial.params["params"])
    expected = optax.apply_updates(initial.params["params"], update)
    np.testing.assert_allclose(final.params["params"]["w"], expected["w"], rtol=1e-6, atol=1e-7)
    for want, got in zip(jax.tree.leaves(expected_opt), jax.tree.leaves(final.opt_state), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)


class Stream:
    def __init__(self):
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        result = batches()[self.index % 4]
        self.index += 1
        return result
    def get_state(self):
        return json.dumps(self.index).encode()
    def set_state(self, state):
        self.index = json.loads(state)


class Data:
    batch = 2 * jax.device_count()
    def train(self):
        return Stream()


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("cut", [1, 2])
def test_partial_checkpoint_replays_exact_realized_records(tmp_path, composite, cut):
    checkpoints = Checkpoints(str(tmp_path / "run"))
    trainer(composite, checkpoints=checkpoints).fit(Data(), steps=cut)
    resumed = trainer(composite, checkpoints=checkpoints).fit(Data(), steps=4)
    assert float(resumed.scale.scale) == 65536.
    assert int(resumed.scale.fin_steps) == 0
    uninterrupted = trainer(composite).fit(Data(), steps=4)
    for want, got in zip(jax.tree.leaves(uninterrupted), jax.tree.leaves(resumed), strict=True):
        np.testing.assert_array_equal(got, want)
    class Unused(Data):
        def train(self):
            raise AssertionError("same-target resume opened the data stream")
    same = trainer(composite, checkpoints=checkpoints).fit(Unused(), steps=4)
    assert int(same.step) == 4
    with pytest.raises(ValueError, match="window_size"):
        trainer(composite, k=4, checkpoints=checkpoints).fit(Unused(), steps=5)


@pytest.mark.parametrize("composite,aux_active,updates", [(False, 0, 0), (True, 0, 0), (True, 1, 1)])
def test_activity_uses_all_loss_terms(composite, aux_active, updates):
    train = trainer(composite)
    initial = train.initial_state()
    data = batches()[0]
    data["mask"] = jnp.zeros_like(data["mask"])
    data["active"] = jnp.asarray(aux_active)
    run = train.compile(initial, data)
    state, *_ = run(initial, data)
    state, *_ = run(state, data)
    assert int(state.microstep) == 2
    assert int(state.updates) == updates
    if not updates:
        np.testing.assert_array_equal(state.params["params"]["w"], initial.params["params"]["w"])
        for a, b in zip(jax.tree.leaves(initial.opt_state), jax.tree.leaves(state.opt_state), strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("auxiliary", [None, "rows", "global"])
def test_real_lm_mtp_router_and_qk_update_matches_combined_batch(auxiliary):
    from dew.data.chat import Role
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives import Step
    from dew.objectives.lm import LMObjective
    from dew.training.optim import scale_by_qk_clip

    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1,
                              num_heads=2, mlp_features=16, max_seq_len=16,
                              num_nextn_predict_layers=2,
                              mixture={"experts": 4, "top_k": 2, "bias": True,
                                       "score_function": "softmax"})
    objective = LMObjective(model, 10, head_chunks=1, mtp_weight=.3,
                            loss_role=Role.ASSISTANT, balance_rate=.02,
                            aux_loss_alpha=None if auxiliary is None else .2,
                            seq_aux=auxiliary != "global", qk_stats=True)
    optimizer = optax.chain(optax.clip_by_global_norm(.5), optax.sgd(.03), scale_by_qk_clip(.05))
    train = Trainer(objective, optimizer, key=jax.random.PRNGKey(7), accumulation=2,
                    dynamic_scale=True)
    initial = train.initial_state()
    rows = jax.device_count()
    rng = np.random.default_rng(4)
    data = []
    for targets in (1, 9):
        roles = np.full((rows, 11), int(Role.USER), np.int32)
        roles[:, -targets:] = int(Role.ASSISTANT)
        data.append({"text": jnp.asarray(rng.integers(0, 16, size=(rows, 11)), jnp.int32),
                     "text_roles": jnp.asarray(roles)})
    run = train.compile(initial, data[0])
    partial, *_ = run(initial, data[0])
    for before, after in zip(jax.tree.leaves(initial.params), jax.tree.leaves(partial.params), strict=True):
        np.testing.assert_array_equal(before, after)
    actual, actual_loss, _, _, accepted = run(partial, data[1])
    assert bool(accepted) and int(actual.updates) == 1
    combined = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), *data)
    info = Step(jnp.array(0), jax.random.fold_in(initial.key, 0), initial.params)
    (expected_loss, aux), gradient = jax.value_and_grad(
        lambda p: scalar_loss(objective, {**initial.params, "params": p}, combined, info),
        has_aux=True)(initial.params["params"])
    updates, _ = optimizer.update(gradient, initial.opt_state, initial.params["params"], qk_stats=aux.qk_stats)
    expected = optax.apply_updates(initial.params["params"], updates)
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-6, atol=2e-6)
    for want, got in zip(jax.tree.leaves(expected), jax.tree.leaves(actual.params["params"]), strict=True):
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)
    expected_moe = objective.apply_effects(initial.params, aux.effects)["moe"]
    for want, got in zip(jax.tree.leaves(expected_moe), jax.tree.leaves(actual.params["moe"]), strict=True):
        np.testing.assert_array_equal(got, want)

    # An independent role-mask equation catches a shared bug in both batching paths.
    if auxiliary is None:
        scores = objective.token_scores(initial.params, combined["text"], depths=True)
        roles = combined["text_roles"] == int(Role.ASSISTANT)
        numerator = jnp.sum(scores.losses * roles[:, 1:])
        for depth, (losses, _) in enumerate(scores.depths, 1):
            numerator += .3 / 2 * jnp.sum(losses * roles[:, depth + 1:])
        np.testing.assert_allclose(expected_loss, numerator / jnp.sum(roles[:, 1:]), rtol=1e-6)


def test_replay_preserves_half_precision_cotangents_and_integer_support():
    class Half(Tiny):
        def loss(self, variables, batch, step):
            w = variables["params"]["w"]
            count = jnp.asarray(batch["y"].size)
            prediction = Mean(((w - 1) ** 2 * count).astype(jnp.float16), count)
            row = Mean(((w + 3) ** 2).astype(jnp.float16), jnp.asarray(1))
            return Terms(prediction, row, jnp.zeros(3, jnp.float16),
                         jnp.zeros(3, jnp.int32), jnp.asarray(0)), Aux({})
    train = Trainer(Half(), optax.sgd(.1), key=jax.random.PRNGKey(1), accumulation=2)
    initial = train.initial_state()
    data = batches()[0]
    step = train.compile(initial, data)
    state, *_ = step(initial, data)
    state, *_ = step(state, data)
    w = initial.params["params"]["w"]
    expected = w - .1 * (2 * (w - 1) + .2 * (w + 3))
    # Original fp16 statistic pullbacks round their coefficients to fp16.
    np.testing.assert_allclose(state.params["params"]["w"], expected, rtol=0, atol=5e-5)


def test_local_partial_snapshot_survives_continued_training_and_weight_restore(tmp_path):
    checkpoints = Checkpoints(str(tmp_path / "persistent"),
                              local_directory=str(tmp_path / "local"), local_every=1)
    train = trainer(True, checkpoints=checkpoints)
    initial = train.initial_state()
    data = batches()
    step = train.compile(initial, data[0])
    prefix, *_ = step(initial, data[0])
    checkpoints.save_local(1, prefix, b"1")
    final, *_ = step(prefix, data[1])
    final, *_ = step(final, data[2])
    checkpoints.wait()
    resumed_trainer = trainer(True, checkpoints=checkpoints)
    restored, _, position = resumed_trainer.place()
    assert position == b"1"
    step = resumed_trainer.compile(restored, data[1])
    resumed, *_ = step(restored, data[1])
    resumed, *_ = step(resumed, data[2])
    for want, got in zip(jax.tree.leaves(final), jax.tree.leaves(resumed), strict=True):
        np.testing.assert_array_equal(got, want)
    template = {"params": jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), restored.params)}
    selected, _ = checkpoints.restore(template, 1)
    for want, got in zip(jax.tree.leaves(prefix.params), jax.tree.leaves(selected["params"]), strict=True):
        np.testing.assert_array_equal(got, want)

