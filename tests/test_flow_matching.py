"""Flow matching on the linear (rectified flow) path.

Covers the schedule invariants, the exact velocity round-trip, the claim that
the DDIM and Euler solvers already integrate the flow ODE, and a toy
end-to-end run proving the objective actually learns a distribution.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn

from dew.diffusion import FlowMatchPredictionTransform, Process, broadcast_rates, expand, presets
from dew.diffusion.schedules import FlowMatchingScheduler
from dew.diffusion.schedules.flow import compute_resolution_shift
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


def test_timesteps_are_logit_normal(rng):
    schedule = FlowMatchingScheduler(logit_mean=-0.3, logit_std=1.4)
    steps = schedule.sample_t(rng, 50000)
    assert jnp.all((steps > 0) & (steps < 1))
    logits = jnp.log(steps) - jnp.log1p(-steps)
    assert abs(float(jnp.mean(logits)) - (-0.3)) < 0.05
    assert abs(float(jnp.std(logits)) - 1.4) < 0.05


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


def test_resolution_shift_grows_with_sequence_length():
    shifts = [compute_resolution_shift(n) for n in (256, 1024, 4096)]
    assert shifts == sorted(shifts)
    assert shifts[0] == pytest.approx(np.exp(0.5))
    assert shifts[-1] == pytest.approx(np.exp(1.15))


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


def test_preset_wires_flow_matching():
    process = presets.Flow(shift=2.0)()
    assert isinstance(process.schedule, FlowMatchingScheduler)
    assert process.sampling is None
    assert isinstance(process.prediction, FlowMatchPredictionTransform)
    assert process.schedule.shift == 2.0


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
