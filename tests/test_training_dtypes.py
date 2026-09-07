"""Training and checkpoint precision against native Linen/Optax updates."""
from flax import linen as nn, struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.checkpoints import Checkpoints
from dew.objectives import Aux, EMASpec, Mean, Objective, mean_loss
from dew.training import Trainer


@struct.dataclass
class Moments:
    errors: Mean
    predictions: jax.Array
    rows: jax.Array


class MixedDense(nn.Module):
    output_dtype: jax.typing.DTypeLike

    @nn.compact
    def __call__(self, x):
        hidden = nn.Dense(3, param_dtype=jnp.bfloat16)(x)
        return nn.Dense(1, param_dtype=self.output_dtype)(jnp.tanh(hidden))


class DenseObjective(Objective):
    def __init__(self, parameter_kind, loss_kind):
        self.loss_kind = loss_kind
        self.compute_dtype = jnp.float64 if "64" in parameter_kind else jnp.float32
        self.model = (MixedDense(self.compute_dtype) if parameter_kind.startswith("mixed") else
                      nn.Dense(1, param_dtype=jnp.float64 if parameter_kind == "float64" else jnp.bfloat16))

    def init(self, key, variables=None):
        return self.model.init(key, jnp.ones((1, 2), self.compute_dtype))

    def reference_loss(self, params, batch):
        prediction = self.model.apply({"params": params}, batch["x"])
        value = jnp.mean((prediction - batch["y"]) ** 2)
        return value + .125 * jnp.mean(prediction) ** 2 if self.loss_kind == "composite" else value

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        errors = (prediction - batch["y"]) ** 2
        if self.loss_kind == "scalar":
            return jnp.mean(errors), Aux({})
        result = Mean(jnp.sum(errors), jnp.asarray(errors.size, jnp.int32))
        if self.loss_kind == "mean":
            return result, Aux({})
        return Moments(result, jnp.sum(prediction), jnp.asarray(prediction.size, jnp.int32)), Aux({})

    def reduce_loss(self, stats):
        if isinstance(stats, Moments):
            value, active = mean_loss(stats.errors)
            return value + .125 * (stats.predictions / stats.rows) ** 2, active
        return super().reduce_loss(stats)


def compare(actual, expected, *, exact=False, tolerance=2e-6, moment_rounding=False):
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert got.dtype == want.dtype
        if moment_rounding and not exact and got.dtype == jnp.bfloat16:
            # A materialized accumulation boundary can round the gradient before
            # Adam, unlike the fused reference. Allow one adjacent bf16 moment,
            # not a relative tolerance on parameters, EMA, or resumed state.
            # These fixtures differ by at most one neighbor on JAX 0.11.1/CUDA
            # (four accumulation cases); CPU matches the compiled reference.
            reference = np.asarray(want)
            lower = np.nextafter(reference, np.full_like(reference, -np.inf))
            upper = np.nextafter(reference, np.full_like(reference, np.inf))
            assert np.all(np.asarray(got) >= lower) and np.all(np.asarray(got) <= upper)
        elif exact or got.dtype == jnp.bfloat16:
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance * .1)


def exercise_updates_and_resume(tmp_path, parameter_kind, loss_kind, k, *, dynamic_scale=False, ema_decay=None):
    objective = DenseObjective(parameter_kind, loss_kind)
    if ema_decay is not None:
        objective.ema = EMASpec(optax.constant_schedule(ema_decay))
    optimizer = optax.adam(.01)
    checkpoints = Checkpoints(str(tmp_path / "run"))
    def trainer():
        return Trainer(objective, optimizer, key=jax.random.PRNGKey(17),
                       accumulation=k, dynamic_scale=dynamic_scale, checkpoints=checkpoints)
    train = trainer()
    state, _, _ = train.place()
    expected_params, expected_opt, expected_ema = state.params["params"], state.opt_state, state.ema
    dtype = objective.compute_dtype
    x = jnp.array([[1.1234567890123, 2.2345678901234], [2.3456789012345, -1.4567890123456]], dtype)
    y = jnp.array([[.01234567890123], [1.2345678901234]], dtype)
    batch = {"x": jnp.tile(x, (jax.device_count(), 1)), "y": jnp.tile(y, (jax.device_count(), 1))}
    step = train.compile(state, batch)
    tolerance = 2e-13 if parameter_kind == "float64" else 2e-6
    # Compile the independent Linen loss and native Optax update together, as a
    # real training loop does. Eager Adam rounds every bf16 intermediate; GPU
    # fusion need not, so eager parameter bits are not the compiled contract.
    @jax.jit
    def reference_update(params, opt_state, batch):
        loss, grads = jax.value_and_grad(objective.reference_loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return loss, optax.apply_updates(params, updates), opt_state

    for window in range(2):
        expected_loss, expected_params, expected_opt = reference_update(expected_params, expected_opt, batch)
        if expected_ema is not None:
            def average(old, new):
                work = np.float64 if old.dtype == jnp.float64 or new.dtype == jnp.float64 else np.float32
                weight = np.asarray(ema_decay, work)
                value = weight * np.asarray(old, work) + (np.asarray(1, work) - weight) * np.asarray(new, work)
                return value.astype(old.dtype)
            expected_ema = jax.tree.map(average, expected_ema, {"params": expected_params})
        for micro in range(k):
            state, loss, _, finite, accepted = step(state, batch)
            assert bool(finite) and bool(accepted)
            if window == 0 and micro == 0:
                checkpoints.save(1, state, None, {"loss": float(loss)})
            if micro < k - 1 and state.accumulation.gradient is not None:
                for gradient, parameter in zip(jax.tree.leaves(state.accumulation.gradient),
                                                jax.tree.leaves(state.params["params"]), strict=True):
                    assert gradient.dtype == jnp.promote_types(parameter.dtype, jnp.float32)
        np.testing.assert_allclose(loss, expected_loss, rtol=tolerance, atol=tolerance * .1)
        compare(state.params["params"], expected_params, tolerance=tolerance)
        compare(state.opt_state, expected_opt, tolerance=tolerance, moment_rounding=True)
        compare(state.ema, expected_ema, tolerance=tolerance)
    assert int(state.updates) == 2
    checkpoints.wait()
    resumed_trainer = trainer()
    restored, _, _ = resumed_trainer.place()
    step = resumed_trainer.compile(restored, batch)
    for _ in range(2 * k - 1):
        restored, *_ = step(restored, batch)
    compare(restored, state, exact=True)


@pytest.mark.parametrize("parameter_kind,loss_kind,k", [
    ("bfloat16", "scalar", 1),
    ("bfloat16", "scalar", 2),
    ("bfloat16", "mean", 1),
    ("bfloat16", "mean", 2),
    ("mixed", "mean", 2),
    ("mixed", "composite", 2),
])
def test_adam_native_parameter_dtypes_survive_updates_and_checkpoints(tmp_path, parameter_kind, loss_kind, k):
    exercise_updates_and_resume(tmp_path, parameter_kind, loss_kind, k)


@pytest.mark.parametrize("parameter_kind,loss_kind,k", [
    ("float64", "scalar", 1),
    ("float64", "mean", 2),
    ("float64", "composite", 2),
    ("mixed64", "composite", 2),
])
def test_float64_statistics_and_updates_preserve_requested_precision(tmp_path, parameter_kind, loss_kind, k):
    with jax.enable_x64():
        total, mass = 1 + 2. ** -40, 3 + 2. ** -35
        reduced, _ = mean_loss(Mean(jnp.array(total, jnp.float64), jnp.array(mass, jnp.float64)))
        np.testing.assert_allclose(reduced, total / mass, rtol=0, atol=1e-15)
        exercise_updates_and_resume(tmp_path, parameter_kind, loss_kind, k)


@pytest.mark.parametrize("parameter_kind", ["bfloat16", "float64"])
def test_scaled_native_gradients_preserve_update_and_restart(tmp_path, parameter_kind):
    with jax.enable_x64():
        exercise_updates_and_resume(tmp_path, parameter_kind, "composite", 2, dynamic_scale=True)


@pytest.mark.parametrize("seq_aux", [False, True])
def test_router_reductions_preserve_float64_scores(seq_aux):
    from dew.nn.moe import deepseek_v2_aux_loss
    with jax.enable_x64():
        values = np.array([[[.8 + 2.**-35, .2 - 2.**-35], [.65, .35], [.6, .4]],
                           [[.9, .1], [.15, .85], [.2, .8]]], np.float64)
        indices = np.argmax(values, axis=-1)[..., None]
        counts = np.eye(2)[indices[..., 0]].sum(axis=1)
        if seq_aux:
            coefficients = .2 * 2 * counts[:, None, :] / (2 * 3**2)
        else:
            coefficients = .2 * 2 * counts.sum(axis=0) / 6**2
        expected_gradient = np.broadcast_to(coefficients, values.shape)
        expected = np.sum(values * expected_gradient)
        loss, gradient = jax.value_and_grad(
            lambda scores: deepseek_v2_aux_loss(scores, jnp.asarray(indices), .2, seq_aux))(jnp.asarray(values))
        np.testing.assert_allclose(loss, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=1e-15)


def test_optimizer_dtype_overflow_backs_off_without_losing_the_prefix():
    import dataclasses

    class Cancellation(Objective):
        def init(self, key, variables=None):
            return {"params": {"w": jnp.array(0., jnp.float16)}}
        def loss(self, variables, batch, step):
            value = variables["params"]["w"].astype(jnp.float32)
            first = batch["first"]
            left_mass = jnp.where(first, 100., 1.)
            right_mass = jnp.where(first, 1., 100.)
            left = Mean(value * jnp.where(first, 60000., -50000.) * left_mass, left_mass)
            right = Mean(value * jnp.where(first, -50000., 60000.) * right_mass, right_mass)
            return (left, right), Aux({})
        def reduce_loss(self, stats):
            left, active_left = mean_loss(stats[0])
            right, active_right = mean_loss(stats[1])
            return left + right, active_left | active_right

    trainer = Trainer(Cancellation(), optax.sgd(.01), key=jax.random.PRNGKey(2),
                      accumulation=2, dynamic_scale=True)
    initial = trainer.initial_state()
    initial = dataclasses.replace(initial, scale=dataclasses.replace(initial.scale, scale=jnp.array(1.)))
    batch = {"first": jnp.array(True)}
    step = trainer.compile(initial, batch)
    prefix, _, _, _, accepted = step(initial, batch)
    assert bool(accepted)
    # Each finalized contribution fits fp16; their fp32 sum does not.
    rejected, loss, _, finite, accepted = step(prefix, {"first": jnp.array(False)})
    assert bool(finite) and float(loss) == 0 and not bool(accepted)
    assert int(rejected.step) == 2 and int(rejected.microstep) == 1 and int(rejected.updates) == 0
    assert float(rejected.scale.scale) == .5
    compare(rejected.accumulation, prefix.accumulation, exact=True)
    compare(rejected.params, prefix.params, exact=True)


@pytest.mark.parametrize("parameter_kind", ["bfloat16", "mixed", "float64", "mixed64"])
def test_ema_storage_dtype_and_precision_survive_real_updates_and_restart(tmp_path, parameter_kind):
    with jax.enable_x64():
        exercise_updates_and_resume(tmp_path, parameter_kind, "mean", 2,
                                    ema_decay=jnp.array(.5, jnp.float32))


def test_unit_decay_preserves_mixed_frozen_leaves_bitwise():
    from dew.training import ema_update
    with jax.enable_x64():
        average = {"low": jnp.array([1., jnp.nan], jnp.bfloat16),
                   "single": jnp.array([2., -3.], jnp.float32),
                   "double": jnp.array([1. + 2.**-40, jnp.nan], jnp.float64)}
        live = jax.tree.map(lambda x: jnp.full_like(x, jnp.inf), average)
        result = ema_update(average, live, jnp.array(1., jnp.float64))
        for got, want in zip(jax.tree.leaves(result), jax.tree.leaves(average), strict=True):
            assert got.dtype == want.dtype
            assert np.asarray(got).tobytes() == np.asarray(want).tobytes()


def test_router_bias_direction_keeps_large_integer_count_differences():
    from dew.nn.moe import load_balance_update
    with jax.enable_x64():
        counts = jnp.array([2**55, 2**55 + 1], jnp.int64)
        np.testing.assert_array_equal(load_balance_update(counts, .02), [.02, -.02])


@pytest.mark.parametrize("bias_dtype", [jnp.bfloat16, jnp.float32, jnp.float64])
def test_x64_router_bias_storage_survives_updates_and_restart(tmp_path, bias_dtype):
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives import Step
    from dew.objectives.lm import LMObjective
    with jax.enable_x64():
        model = CausalTransformer(vocab_size=8, emb_features=8, num_layers=1,
                                  num_heads=2, mlp_features=16, max_seq_len=8,
                                  mixture={"experts": 2, "top_k": 1, "bias": True})
        initial = model.init(jax.random.PRNGKey(1), jnp.ones((1, 4), jnp.int32))
        initial["moe"] = jax.tree.map(lambda x: x.astype(bias_dtype), initial["moe"])
        objective = LMObjective(model, 4, pretrained=initial, head_chunks=1,
                                balance_rate=.02, aux_loss_alpha=.1, seq_aux=False)
        checkpoints = Checkpoints(str(tmp_path / "router"))
        def build():
            return Trainer(objective, optax.sgd(.01), key=jax.random.PRNGKey(2),
                           accumulation=2, checkpoints=checkpoints)
        trainer = build()
        state, _, _ = trainer.place()
        batch = {"text": jnp.ones((jax.device_count(), 5), jnp.int32)}
        _, aux = objective.loss(state.params, batch, Step(state.microstep, state.key, state.params))
        counts = jax.tree.map(np.asarray, aux.effects)
        assert all(x.dtype == np.int64 for x in jax.tree.leaves(counts))
        def expected_bias(bias, count):
            total = sum(int(x) for x in count)
            direction = np.array([np.sign(total - count.size * int(x)) for x in count])
            return (np.asarray(bias, np.float64) + .02 * direction).astype(bias.dtype)
        expected = jax.tree.map(expected_bias, state.params["moe"], counts)
        step = trainer.compile(state, batch)
        prefix, *_ = step(state, batch)
        checkpoints.save(1, prefix, None, {"loss": 0.})
        actual, *_ = step(prefix, batch)
        compare(actual.params["moe"], expected, exact=True)
        for _ in range(2):
            actual, *_ = step(actual, batch)
        checkpoints.wait()
        resumed_trainer = build()
        restored, _, _ = resumed_trainer.place()
        step = resumed_trainer.compile(restored, batch)
        for _ in range(3):
            restored, *_ = step(restored, batch)
        compare(restored, actual, exact=True)

