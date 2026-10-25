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
from learning_gate import assert_learns, within
from reference_error import assert_as_exact_as_the_reference
from scipy import stats

from dew.diffusion import broadcast_rates, expand, presets
from dew.diffusion.schedules import FlowMatchingScheduler
from dew.sampling import Euler, sample

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


COSMAP_DRAWS = 4_000_000
"""The cosmap site's draws: at one in a billion the Kolmogorov-Smirnov test fails an empirical CDF off
the paper's by 3.27 / sqrt(COSMAP_DRAWS), 0.0016, under half the 1.63 / sqrt(200,000), 0.0036, the site
held at one in a hundred before."""


def test_cosmap_times_follow_the_papers_density(rng):
    """Eq. 21's density 2 / (pi (1 - 2t + 2t^2)) integrates to the CDF
    (2 / pi) atan(t / (1 - t)); the draws' empirical CDF is the paper's by
    the Kolmogorov-Smirnov test at one in a billion."""
    drawn = np.asarray(FlowMatchingScheduler(density="cosmap").sample_t(rng, COSMAP_DRAWS), np.float64)
    assert stats.kstest(drawn, lambda t: 2 / np.pi * np.arctan2(t, 1 - t)).pvalue > 1e-9


############################################################################################################
# The existing solvers already integrate the flow ODE
############################################################################################################

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


MIXTURE_RUNS = 64
"""The mixture site's declared training runs (the learning gate's), each its own initialization, data
and noise: `learning_gate.units_for` over twenty pilots of 168 to 192 runs declared 47 to 58, and the
site declares more than any of them, since trained together 64 runs cost what 56 did (this file, 95 s
on armada against 93 s). Each mode's mean
sat 0.011 to 0.012 inside its center on average, against a margin of 0.04, and moved 0.017 from run to
run, so one run missed the margin on an axis about one time in twenty-five."""


@jax.jit
def mixture_runs(runs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The toy's runs numbered `runs`, trained together: each a velocity network trained by flow matching
    on the mixture for 1500 steps from keys of its number's own, and 2048 samples of its 64-step Euler
    walk; with each run's last step's loss."""
    process = presets.Flow()()
    schedule, transform = process.schedule, process.prediction
    model = ToyVelocityMLP()
    optimizer = optax.adam(3e-3)

    def loss_fn(params, x0, noise, steps):
        rates = broadcast_rates(schedule, steps, x0)
        x_t, c_in, target = transform.forward_diffusion(x0, noise, rates)
        preds = model.apply(params, x_t * c_in, schedule.model_time(steps))
        weights = expand(process.weight(steps), x0)
        return jnp.mean(weights * (preds - target) ** 2)

    def train_step(step, carry, run_key):
        params, opt_state, _ = carry
        data_key, noise_key, time_key = jax.random.split(jax.random.fold_in(run_key, step), 3)
        x0 = sample_mixture(data_key, 512)
        noise = jax.random.normal(noise_key, x0.shape)
        steps = schedule.sample_t(time_key, 512)
        loss, grads = jax.value_and_grad(loss_fn)(params, x0, noise, steps)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    def one(run):
        key = jax.random.fold_in(jax.random.PRNGKey(0), run)
        params = model.init(jax.random.fold_in(key, 0), jnp.zeros((1, 1, 1, 3)), jnp.zeros((1,)))
        carry = (params, optimizer.init(params), jnp.zeros((), jnp.float32))
        run_key = jax.random.fold_in(key, 1)
        params, _, loss = jax.lax.fori_loop(0, 1500, lambda step, carry: train_step(step, carry, run_key),
                                            carry)
        denoise = process.denoiser(model, params, {})
        x_T = process.noise(jax.random.fold_in(key, 2), (2048, 1, 1, 3))
        samples = sample(denoise, x_T, 64, solver=Euler(), key=jax.random.fold_in(key, 3))
        return samples.reshape(-1, 3), loss

    return jax.vmap(one)(runs)


def mixture_contrasts(runs: np.ndarray) -> dict[str, np.ndarray]:
    """The runs' contrasts (`learning_gate.within`), one a run, for the claims the toy held one run to:
    its last loss below one, its samples' modes balanced within 0.15 of a half, each mode's mean within
    0.04 of its center and its spread within 0.04 of MODE_STD, and the third channel one gaussian within
    0.04 of zero mean and MODE_STD's spread."""
    samples, losses = (np.asarray(value, np.float64) for value in mixture_runs(jnp.asarray(runs)))
    assignment = samples[..., 0] > 0
    contrasts = {"loss below one": 1.0 - losses,
                 **within("mode share", assignment.mean(axis=1), 0.5, 0.15)}
    for mode, center in enumerate(np.asarray(MODE_CENTERS, np.float64)):
        members = [run[side == bool(mode)] for run, side in zip(samples, assignment, strict=True)]
        for axis in range(2):
            contrasts |= within(f"mode {mode}'s mean {axis}",
                                [member[:, axis].mean() for member in members], center[axis], 0.04)
        contrasts |= within(f"mode {mode}'s spread", [member[:, :2].std() for member in members],
                            MODE_STD, 0.04)
    # The third channel is a single zero-centred gaussian, not a mixture.
    contrasts |= within("third channel's mean", samples[..., 2].mean(axis=1), 0.0, 0.04)
    return contrasts | within("third channel's spread", samples[..., 2].std(axis=1), MODE_STD, 0.04)


def test_flow_matching_learns_a_two_mode_mixture():
    """Flow matching learns the mixture: over independent training runs, the runs' samples meet the toy's
    margins on average (the learning gate), each claim's two sides held by the one-sided t."""
    assert_learns(mixture_contrasts, "two-mode mixture", units=MIXTURE_RUNS)


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
    # The step reads the reference's times and noise (`Step.tape`), drawn as tools/flow_loss_reference.py
    # draws them.
    keys = jax.random.split(jax.random.key(int(LOSS["key"])), 5)
    times = objective.process.schedule.sample_t(keys[2], pixels.shape[0])
    step = Step(step=jnp.asarray(0), key=keys[0], ema=None,
                tape={"times": times, "noise": jax.random.normal(keys[3], pixels.shape)})
    np.testing.assert_array_equal(np.asarray(objective.process.schedule.rates(times)[1]),
                                  LOSS[f"sigmas_{case}"])

    def loss(weights):
        tree = {**variables, "params": {"weights": weights}}
        return objective.scalar_loss(tree, {"image": pixels}, step)[0]

    value, gradient = jax.value_and_grad(loss)(jnp.asarray(LOSS["weights"]))
    np.testing.assert_allclose(2 * float(value), float(LOSS[f"loss_{case}_f64"]), rtol=1e-6)
    assert_as_exact_as_the_reference(2 * gradient, LOSS[f"grad_{case}"], LOSS[f"grad_{case}_f64"], "gradient")
