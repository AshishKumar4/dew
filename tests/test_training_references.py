"""The trainer's commit and retention rules against the libraries that define them.

Each test runs the published code beside Dew on the same events: Orbax's own
`CheckpointManager` and preservation policies for what `Keep` and a ranking
retain, PyTorch's `ReduceLROnPlateau` for when `Plateau` stops, Flax's
`DynamicScale` for the loss scale's recurrence, optax's `MultiSteps` for an
accumulation window over equal microbatches and optax's own update on the
pooled batch for unequal ones, optax's own updates for weights held on the
host, and Orbax's manager, with no Dew in the read, for what a checkpoint
holds.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
import torch
from flax.training.dynamic_scale import DynamicScale
from orbax.checkpoint import checkpoint_managers
from test_checkpoint_ranking import Overfit, data
from test_host_training import HOST, Coupled, centered

from dew.artifacts import TokenScores
from dew.checkpoints import Checkpoints, Keep, Ranking
from dew.objectives.base import Aux, Objective, Ratio, Shown
from dew.training import Trainer
from dew.training.trainer import Plateau

ROWS = 2 * jax.device_count()


def test_keep_and_a_ranking_retain_the_steps_orbaxs_own_policies_retain(tmp_path):
    """Twenty saves with distinct validation losses, under the latest two,
    every fifth step and the three lowest losses, leave the steps Orbax's
    manager keeps under `LatestN`, `EveryNSteps` and `BestN` together."""
    losses = np.random.default_rng(0).permutation(20).astype(float) / 4
    dew = Checkpoints(str(tmp_path / "dew"), keep=Keep(latest=2, every=5))
    policy = checkpoint_managers.AnyPreservationPolicy([
        checkpoint_managers.LatestN(2), checkpoint_managers.EveryNSteps(5),
        checkpoint_managers.BestN(get_metric_fn=lambda metrics: metrics["val/loss"], reverse=True, n=3,
                                  keep_checkpoints_without_metrics=False)])
    reference = ocp.CheckpointManager(
        tmp_path / "orbax", options=ocp.CheckpointManagerOptions(
            preservation_policy=policy, best_fn=lambda metrics: metrics["val/loss"], best_mode="min",
            create=True, enable_async_checkpointing=False),
        item_handlers=ocp.PyTreeCheckpointHandler())
    for step, loss in enumerate(losses, 1):
        tree = {"w": np.full((2,), step, np.float32)}
        dew.save(step, tree, metrics={"val/loss": loss}, ranking=Ranking("val/loss", loss, "min", top=3))
        dew.wait()
        reference.save(step, args=ocp.args.PyTreeSave(tree), metrics={"val/loss": loss})
    reference.wait_until_finished()
    assert [kept.step for kept in dew.kept()] == list(reference.all_steps())


@dataclasses.dataclass
class Scripted:
    """A validation score read off a fixed sequence, one value an evaluation."""

    values: list[float]
    shown: Shown
    name = "scripted"
    reads = TokenScores

    def __call__(self, artifact, batch):
        return self.values.pop(0)

    def merge(self, left, right):
        return left

    def finalize(self, value):
        return value


def reductions(scores, mode, evals, min_delta):
    """The evaluation at which PyTorch's `ReduceLROnPlateau` first cuts the
    rate, with `evals - 1` evaluations of patience and an absolute threshold
    of `min_delta`, or None."""
    weight = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([weight], lr=1.0)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=0.5, patience=evals - 1, threshold=min_delta, threshold_mode="abs",
        cooldown=0)
    for index, score in enumerate(scores, 1):
        plateau.step(score)
        if optimizer.param_groups[0]["lr"] < 1.0:
            return index
    return None


@pytest.mark.parametrize(("scores", "better", "evals", "min_delta"), [
    ([1.0, 0.75, 0.75, 0.875, 0.5, 0.5, 0.5, 0.5, 0.25], "lower", 3, 0.0),
    ([1.0, 0.75, 0.625, 0.5, 0.5, 0.375, 0.25, 0.25], "lower", 2, 0.125),
    ([0.25, 0.5, 0.5, 0.75, 0.625, 1.0, 1.0, 1.0, 1.0], "higher", 3, 0.0),
    ([0.5, 0.625, 0.75, 1.0, 1.125, 1.375, 1.5, 1.75], "higher", 2, 0.25),
    ([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125], "lower", 1, 0.0),
])
def test_plateau_stops_where_torchs_reduce_on_plateau_first_cuts_the_rate(tmp_path, scores, better, evals,
                                                                          min_delta):
    """Both count evaluations since the best one, an improvement being a
    move past it by more than `min_delta`: Dew stops at the `evals`-th such
    evaluation, where PyTorch, with one fewer of patience, first cuts the
    rate. Ties and moves of exactly `min_delta` count as no improvement."""
    metric = Scripted(list(scores), Shown(better=better))
    trainer = Trainer(Overfit(), optax.sgd(0.1), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(tmp_path / "run")))
    state = trainer.fit(data(), steps=len(scores), log_every=1, eval_every=1, checkpoint_every=len(scores),
                        metrics=[metric], stop=Plateau(metric, evals=evals, min_delta=min_delta))
    stopped = reductions(scores, "min" if better == "lower" else "max", evals, min_delta)
    assert int(state.step) == (len(scores) if stopped is None else stopped)
    assert len(scores) - len(metric.values) == int(state.step)


class Flagged(Objective):
    """A loss whose gradient is infinite on a batch flagged bad."""

    def init(self, key, variables=None):
        return {"params": {"w": jnp.ones((2,), jnp.float32)}}

    def loss(self, variables, batch, step):
        return jnp.sum(variables["params"]["w"] * batch["x"]) / ROWS, Aux({})


def test_the_loss_scale_follows_flaxs_dynamic_scale_through_finite_and_rejected_steps():
    """Flax's `DynamicScale.value_and_grad` and the trainer's loss scale, on
    the same run of finite and infinite gradients, halve on each rejection,
    double after every `growth_interval` finite steps in a row, and agree on
    the scale and the streak after every step."""
    pattern = [False, False, False, True, False, False, False, False, True, True, False, False, False]
    trainer = Trainer(Flagged(), optax.sgd(0.01), key=jax.random.key(0), dynamic_scale=True)
    state = trainer.initial_state()
    state = dataclasses.replace(state, scale=dataclasses.replace(state.scale, growth_interval=2))
    reference = DynamicScale(growth_interval=2)
    weights = jnp.ones((2,), jnp.float32)
    run = None
    for bad in pattern:
        x = np.tile(np.asarray([[np.inf if bad else 1.0, 1.0]], np.float32), (ROWS, 1))
        batch = {"x": jnp.asarray(x)}
        run = trainer.compile(state, batch) if run is None else run
        state, *_ = run(state, batch)
        reference, finite, _, _ = reference.value_and_grad(lambda w, x=x: jnp.sum(w * x) / ROWS)(weights)
        assert bool(finite) != bad
        assert (float(state.scale.scale), int(state.scale.fin_steps)) == (
            float(reference.scale), int(reference.fin_steps))


class Linear(Objective):
    """A masked least-squares loss pooled as the ratio of its sums."""

    def init(self, key, variables=None):
        return {"params": {"w": jnp.asarray([0.3, -0.2], jnp.float32)}}

    def loss(self, variables, batch, step):
        errors = (batch["x"] @ variables["params"]["w"] - batch["y"]) ** 2
        return Ratio(jnp.sum(errors * batch["mask"]), jnp.sum(batch["mask"])), Aux({})


def microbatches(masks):
    rng = np.random.default_rng(1)
    return [{"x": jnp.asarray(rng.normal(size=(ROWS, 2)), jnp.float32),
             "y": jnp.asarray(rng.normal(size=(ROWS,)), jnp.float32),
             "mask": jnp.asarray(np.resize(mask, ROWS), jnp.float32)} for mask in masks]


def windowed(batches, tx, accumulation):
    trainer = Trainer(Linear(), tx, key=jax.random.key(0), accumulation=accumulation)
    state = trainer.initial_state()
    run = trainer.compile(state, batches[0])
    for batch in batches:
        state, *_ = run(state, batch)
    return np.asarray(state.variables["params"]["w"])


def gradient(w, batches):
    """The gradient of the pooled ratio over `batches` at `w`."""
    def pooled(w):
        total = sum(jnp.sum((batch["x"] @ w - batch["y"]) ** 2 * batch["mask"]) for batch in batches)
        return total / sum(jnp.sum(batch["mask"]) for batch in batches)
    return jax.grad(pooled)(w)


def test_a_window_of_equal_microbatches_commits_what_optaxs_multisteps_commits():
    """Over microbatches of equal mass, optax's `MultiSteps` averaging each
    microbatch's own gradient and the trainer pooling the window's sums are
    one update, window after window of Adam."""
    batches = microbatches([[1.0, 1.0]] * 6)
    tx = optax.adam(0.05)
    steps = optax.MultiSteps(tx, every_k_schedule=3)
    w = jnp.asarray([0.3, -0.2], jnp.float32)
    held = steps.init(w)
    for batch in batches:
        updates, held = steps.update(gradient(w, [batch]), held, w)
        w = optax.apply_updates(w, updates)
    np.testing.assert_allclose(windowed(batches, tx, 3), np.asarray(w), rtol=2e-6, atol=1e-7)


def test_a_window_of_unequal_microbatches_commits_optaxs_update_on_the_pooled_batch():
    """Over microbatches of unequal mass the window is the pooled batch: the
    gradient of its total over its mass, which a mean of the microbatches'
    means (`MultiSteps`) is not. Two windows of SGD with momentum land where
    optax's own updates on the two pooled batches do."""
    batches = microbatches([[1.0, 0.0], [1.0, 1.0], [0.5, 1.0], [0.0, 1.0], [1.0, 1.0], [0.25, 0.0]])
    tx = optax.sgd(0.1, momentum=0.9)
    w = jnp.asarray([0.3, -0.2], jnp.float32)
    held = tx.init(w)
    for window in (batches[:3], batches[3:]):
        updates, held = tx.update(gradient(w, window), held, w)
        w = optax.apply_updates(w, updates)
    np.testing.assert_allclose(windowed(batches, tx, 3), np.asarray(w), rtol=2e-6, atol=1e-7)


def test_a_checkpoint_reads_back_through_orbaxs_own_manager(tmp_path):
    """A checkpoint is an Orbax step directory: Orbax's manager, with no Dew
    in the read, restores its weights, optimizer state and step bit for bit.
    The average is stored as its difference from the weights, which Dew's
    reader undoes."""
    trainer = Trainer(Linear(), optax.adam(0.05), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(tmp_path / "run")))
    batches = microbatches([[1.0, 1.0]] * 3)
    state = trainer.initial_state()
    run = trainer.compile(state, batches[0])
    for batch in batches:
        state, *_ = run(state, batch)
    expected = jax.tree.map(np.asarray, {"variables": state.variables, "opt_state": state.opt_state,
                                         "step": state.step})
    trainer.checkpoints.save(3, state, None)
    trainer.checkpoints.wait()
    stock = ocp.CheckpointManager(tmp_path / "run", item_handlers=ocp.PyTreeCheckpointHandler())
    restored = stock.restore(3, args=ocp.args.PyTreeRestore())
    for name, tree in expected.items():
        leaves = jax.tree.leaves(tree)
        held = jax.tree.leaves(restored[name])
        assert len(held) == len(leaves)
        for want, got in zip(leaves, held, strict=True):
            np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("optimizer", [
    optax.chain(optax.clip_by_global_norm(.25), optax.adam(optax.linear_schedule(.02, .01, 3))),
    centered(),
    optax.partition({"a": optax.adam(.01), "b": optax.adamw(.02)}, {"first": "a", "second": "b"}),
], ids=["global-clip-scheduled-adam", "cross-leaf-mean", "masked-partition-state"])
def test_weights_held_on_the_host_take_optaxs_own_updates(optimizer):
    """With the weights and the optimizer state on the host, three steps land
    where optax's own `update` lands from the same weights and gradients:
    the gradient leaves the device and the update comes back whole, a global
    clip, a mean across leaves and a partitioned state included."""
    data = {"small": jnp.asarray(.5), "large": jnp.asarray(40.)}
    trainer = Trainer(Coupled(), optimizer, key=jax.random.key(4), layout=HOST)
    state, _, _ = trainer.place()
    step = trainer.compile(state, data)
    for _ in range(3):
        state, *_ = step(state, data)
    params = Coupled().init(None)["params"]
    gradients = jax.grad(lambda p: Coupled().loss({"params": p}, data, None)[0])(params)
    held = optimizer.init(params)
    for _ in range(3):
        updates, held = optimizer.update(gradients, held, params)
        params = optax.apply_updates(params, updates)
    for name, value in params.items():
        np.testing.assert_allclose(np.asarray(state.variables["params"][name]), np.asarray(value),
                                   rtol=1e-6, atol=1e-7)
