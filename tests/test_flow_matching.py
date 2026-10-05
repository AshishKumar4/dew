"""Flow matching on the linear (rectified flow) path.

Covers the schedule invariants, the exact velocity round-trip, the claim that
the DDIM and Euler solvers already integrate the flow ODE, DiffusionObjective's
loss and gradient against Diffusers' Flux training loss on the same draws
(tools/flow_loss_reference.py), and a toy end-to-end run proving the
objective actually learns a distribution.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference

from dew.diffusion import FlowMatchPredictionTransform, Process, broadcast_rates, expand, presets
from dew.diffusion.schedules import FlowMatchingScheduler
from dew.sampling import DDIM, Euler, sample

STEPS = jnp.array([0.05, 0.3, 0.6, 0.95])


def test_linear_path_rates():
    schedule = FlowMatchingScheduler()
    alpha, sigma = schedule.rates(STEPS)
    assert jnp.allclose(alpha + sigma, 1.0, atol=1e-6)
    assert jnp.allclose(sigma, STEPS, atol=1e-6)
    # No input preconditioning on the linear path
    assert FlowMatchPredictionTransform().get_input_scale((alpha, sigma)) == 1


def test_endpoints_are_data_and_noise(rng):
    schedule = FlowMatchingScheduler()
    key0, key1 = jax.random.split(rng)
    x0 = jax.random.normal(key0, (4, 8, 8, 3))
    noise = jax.random.normal(key1, (4, 8, 8, 3))
    transform = FlowMatchPredictionTransform()
    at_zero = transform.forward_diffusion(x0, noise, broadcast_rates(schedule, jnp.zeros((4,)), x0))[0]
    at_one = transform.forward_diffusion(x0, noise, broadcast_rates(schedule, jnp.ones((4,)), x0))[0]
    assert jnp.allclose(at_zero, x0, atol=1e-6)
    assert jnp.allclose(at_one, noise, atol=1e-6)


DENSITIES = dict(np.load(Path(__file__).parent / "fixtures" / "flow" / "densities.npz"))


@pytest.mark.parametrize("case,fields", [
    ("logit_normal", {"density": "logit_normal"}),
    ("logit_normal_shifted", {"density": "logit_normal", "logit_mean": 0.5, "logit_std": 0.8}),
    ("mode", {"density": "mode", "mode_scale": 1.29}),
    ("mode_negative", {"density": "mode", "mode_scale": -0.5}),
])
def test_training_times_are_diffusers_sd3_densities(case, fields):
    """SD3's training-time densities (Esser et al. 2024, section 3.1) as
    Diffusers 0.34.0's `compute_density_for_timestep_sampling` computes them
    (tools/flow_density_reference.py), on the draw `sample_t` makes from the
    same key, held to the float64 rule: logit-normal at its default and
    shifted, and the mode density on both sides of uniform."""
    drawn = FlowMatchingScheduler(**fields).sample_t(jax.random.key(0), DENSITIES[case].shape[0])
    assert_as_exact_as_the_reference(drawn, DENSITIES[case], DENSITIES[f"{case}_f64"], case)


def test_uniform_training_times_are_the_draw_itself():
    """Diffusers' fallback density is the uniform draw unchanged, as Dew's is."""
    count = DENSITIES["uniform"].shape[0]
    drawn = FlowMatchingScheduler(density="uniform").sample_t(jax.random.key(0), count)
    np.testing.assert_array_equal(drawn, DENSITIES["uniform"])


def test_cosmap_times_follow_the_papers_density(rng):
    """Eq. 21's density 2 / (pi (1 - 2t + 2t^2)) integrates to the CDF
    (2 / pi) atan(t / (1 - t)); the draws' empirical CDF stays within the
    Kolmogorov-Smirnov bound for 200k samples (1.63 / sqrt(n), p = 0.01)."""
    count = 200_000
    drawn = np.sort(np.asarray(FlowMatchingScheduler(density="cosmap").sample_t(rng, count),
                               np.float64))
    cdf = 2 / np.pi * np.arctan(drawn / (1 - drawn))
    empirical = np.arange(1, count + 1) / count
    assert np.max(np.abs(empirical - cdf)) < 1.63 / np.sqrt(count)


def test_mode_scale_outside_the_monotone_range_is_refused():
    with pytest.raises(ValueError, match="monotone"):
        FlowMatchingScheduler(density="mode", mode_scale=2.0)


def test_resolution_shift_is_identity_at_one():
    schedule = FlowMatchingScheduler(shift=1.0)
    assert jnp.allclose(schedule.shift_timesteps(STEPS), STEPS, atol=1e-7)


@pytest.mark.parametrize("shift", [0.5, 1.0, 3.0])
def test_resolution_shift_is_monotonic_and_fixes_endpoints(shift):
    schedule = FlowMatchingScheduler(shift=shift)
    t = jnp.linspace(0.0, 1.0, 101)
    shifted = schedule.shift_timesteps(t)
    assert jnp.all(jnp.diff(shifted) > 0)
    assert float(shifted[0]) == pytest.approx(0.0, abs=1e-7)
    assert float(shifted[-1]) == pytest.approx(1.0, abs=1e-7)
    # A shift above 1 moves every interior timestep towards higher noise
    if shift >= 1:
        assert jnp.all(shifted >= t - 1e-7)
    else:
        assert jnp.all(shifted <= t + 1e-7)


def test_timestep_conditioning_is_scaled_to_the_embedding_range():
    schedule = FlowMatchingScheduler(shift=2.0)
    assert jnp.allclose(schedule.model_time(STEPS), schedule.shift_timesteps(STEPS) * 1000)


def test_velocity_roundtrip_is_exact(rng):
    """The linear path round-trips to 1e-5; the observed difference is 3.6e-7 on CPU."""
    schedule = FlowMatchingScheduler()
    transform = FlowMatchPredictionTransform()
    key0, key1 = jax.random.split(rng)
    x0 = jax.random.normal(key0, (4, 8, 8, 3))
    noise = jax.random.normal(key1, (4, 8, 8, 3))
    rates = broadcast_rates(schedule, STEPS, x0)

    xt, _, target = transform.forward_diffusion(x0, noise, rates)
    assert jnp.allclose(target, noise - x0, atol=1e-6)

    recovered_x0, recovered_noise = transform.backward_diffusion(xt, target, rates)
    assert jnp.max(jnp.abs(recovered_x0 - x0)) < 1e-5
    assert jnp.max(jnp.abs(recovered_noise - noise)) < 1e-5


############################################################################################################
# The existing solvers already integrate the flow ODE
############################################################################################################

@pytest.mark.parametrize("solver", [Euler(), DDIM()], ids=lambda s: type(s).__name__)
def test_solver_step_is_the_flow_euler_step(solver, rng):
    """x_{t+dt} = x_t + u * dt exactly, for the unmodified solvers."""
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    key0, key1 = jax.random.split(rng)
    x_t = jax.random.normal(key0, (4, 8, 8, 3))
    velocity = jax.random.normal(key1, (4, 8, 8, 3))
    t = jnp.full((4,), 0.8)
    t_next = jnp.full((4,), 0.6)

    rates = broadcast_rates(process.schedule, t, x_t)
    x0, eps = process.prediction.backward_diffusion(x_t, velocity, rates)
    stepped, _ = solver.step(x_t, t, t_next, x0, eps, (), rng, process, None)
    expected = x_t + velocity * (0.6 - 0.8)
    assert jnp.max(jnp.abs(stepped - expected)) < 1e-5


############################################################################################################
# Toy end-to-end: a two-mode gaussian mixture in the plane
############################################################################################################

MODE_CENTERS = jnp.array([[-0.5, -0.5], [0.5, 0.5]])
MODE_STD = 0.08


def sample_mixture(key, n):
    """Two well-separated modes in the leading two channels; the third channel
    carries no mode information, so the sampler's fixed channel count does not
    turn this into a harder problem."""
    mode_key, noise_key = jax.random.split(key)
    modes = jax.random.bernoulli(mode_key, 0.5, (n,)).astype(jnp.int32)
    centers = jnp.concatenate([MODE_CENTERS[modes], jnp.zeros((n, 1))], axis=-1)
    return (centers + MODE_STD * jax.random.normal(noise_key, (n, 3))).reshape(n, 1, 1, 3)


class ToyVelocityMLP(nn.Module):
    features: int = 128

    @nn.compact
    def __call__(self, x, temb):
        t = jnp.reshape(temb, (-1, 1)) / 1000.0
        freqs = jnp.arange(1, 5, dtype=jnp.float32) * jnp.pi
        h = jnp.concatenate([x.reshape(x.shape[0], -1), t, jnp.sin(t * freqs), jnp.cos(t * freqs)], axis=-1)
        h = nn.swish(nn.Dense(self.features)(h))
        h = nn.swish(nn.Dense(self.features)(h))
        return nn.Dense(3)(h).reshape(x.shape)


def test_flow_matching_learns_a_two_mode_mixture():
    process = presets.Flow()()
    schedule, transform = process.schedule, process.prediction
    model = ToyVelocityMLP()
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 1, 1, 3)), jnp.zeros((1,)))
    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(params)

    def loss_fn(params, x0, noise, steps):
        rates = broadcast_rates(schedule, steps, x0)
        x_t, c_in, target = transform.forward_diffusion(x0, noise, rates)
        preds = model.apply(params, x_t * c_in, schedule.model_time(steps))
        weights = expand(process.weight(steps), x0)
        return jnp.mean(weights * (preds - target) ** 2)

    @jax.jit
    def train_step(params, opt_state, key):
        data_key, noise_key, time_key = jax.random.split(key, 3)
        x0 = sample_mixture(data_key, 512)
        noise = jax.random.normal(noise_key, x0.shape)
        steps = schedule.sample_t(time_key, 512)
        loss, grads = jax.value_and_grad(loss_fn)(params, x0, noise, steps)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    run_key = jax.random.PRNGKey(1)
    for step in range(1500):
        params, opt_state, loss = train_step(params, opt_state, jax.random.fold_in(run_key, step))
    assert float(loss) < 1.0, "flow matching loss did not come down"

    denoise = process.denoiser(model, params, {})
    x_T = process.noise(jax.random.PRNGKey(2), (2048, 1, 1, 3))
    samples = sample(denoise, x_T, 64, solver=Euler(), key=jax.random.PRNGKey(3)).reshape(-1, 3)

    assignment = samples[:, 0] > 0
    fraction = float(jnp.mean(assignment))
    assert 0.35 < fraction < 0.65, f"modes not balanced: {fraction:.2f}"

    for mode, center in enumerate(MODE_CENTERS):
        members = samples[assignment == bool(mode)]
        assert jnp.max(jnp.abs(jnp.mean(members[:, :2], axis=0) - center)) < 0.04
        assert abs(float(jnp.std(members[:, :2])) - MODE_STD) < 0.04
    # The third channel is a single zero-centred gaussian, not a mixture
    assert abs(float(jnp.mean(samples[:, 2]))) < 0.04
    assert abs(float(jnp.std(samples[:, 2])) - MODE_STD) < 0.04


LOSS = np.load(Path(__file__).resolve().parent / "fixtures" / "flow" / "loss.npz")


class Velocity(nn.Module):
    """tools/flow_loss_reference.py's stand-in velocity network."""

    @nn.compact
    def __call__(self, x, timestep, train=False):
        time = timestep.reshape(-1, 1, 1, 1) / 1000
        weights = self.param("weights", nn.initializers.zeros, (4, *x.shape[1:]))
        return (jnp.tanh(x) * weights[0] + jnp.sin(2 * time) * x * weights[1]
                + jnp.cos(x) * time * weights[2] + time * weights[3])


def test_the_static_shift_is_diffusers_schedulers():
    """`FlowMatchEulerDiscreteScheduler(shift=3)`'s published sigma grid is
    Dew's shifted noise rate at the scheduler's own unshifted grid."""
    schedule = presets.Flow(shift=3.0)().schedule
    np.testing.assert_allclose(np.asarray(schedule.rates(jnp.asarray(LOSS["grid"]))[1]),
                               LOSS["grid_shifted_3"], rtol=2e-7, atol=0)


@pytest.mark.parametrize("case", [0, 1], ids=["shift_1", "shift_3"])
def test_the_flow_loss_and_its_gradient_are_diffusers_flux_trainings(case):
    """DiffusionObjective's flow loss on its own draws against Diffusers'
    Flux DreamBooth loss statements run as published on the same sigmas and
    noise (its "logit_normal" weighting, ones): Dew's is half of it, as
    Dew's L2 halves the squared error, within 1e-6 of the float64 run, and
    twice its gradient in the network's 192 weights is held to the float64
    run by the float64 rule. At shift 3 the drawn times move to the
    shifted sigmas the static shift gives."""
    from dew.inputs import Field, InputSpec
    from dew.objectives.base import Step
    from dew.objectives.diffusion import DiffusionObjective

    shift = float(LOSS["shifts"][case])
    pixels = LOSS["pixels"]
    objective = DiffusionObjective(Velocity(), presets.Flow(shift=shift)(),
                                   InputSpec(Field("image", pixels.shape[1:])), guidance=None, solver=Euler(),
                                   steps=2, unconditional_prob=0.0, ema_decay=None)
    variables = objective.init(jax.random.PRNGKey(0))
    step = Step(step=jnp.asarray(0), key=jax.random.key(int(LOSS["key"])), ema=None)
    times = objective.process.schedule.sample_t(jax.random.split(step.key, 5)[2], pixels.shape[0])
    np.testing.assert_array_equal(np.asarray(objective.process.schedule.rates(times)[1]),
                                  LOSS[f"sigmas_{case}"])

    def loss(weights):
        tree = {**variables, "params": {"weights": weights}}
        return objective.scalar_loss(tree, {"image": pixels}, step)[0]

    value, gradient = jax.value_and_grad(loss)(jnp.asarray(LOSS["weights"]))
    np.testing.assert_allclose(2 * float(value), float(LOSS[f"loss_{case}_f64"]), rtol=1e-6)
    assert_as_exact_as_the_reference(2 * gradient, LOSS[f"grad_{case}"], LOSS[f"grad_{case}_f64"], "gradient")
