"""Dreamer 4's world model at small scale (`dew.objectives.world_model`): shortcut forcing's loss against
Eqs. 4, 6-8 of arXiv 2509.24527 transcribed in NumPy float64, its step sampling's grid, and the
dynamics model's block-causal reading of time."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.objectives.world_model import Dynamics, sampled_steps, shortcut_forcing_loss


def test_shortcut_forcing_is_the_papers_bootstrap():
    """Each frame's loss is its x-prediction's squared error where d is the smallest step, and else
    (1 - tau)^2 times the squared error of its velocity against the mean of the two half-step
    velocities the predictor takes itself, every frame times 0.9 tau + 0.1, within float64 rounding
    of Eq. 7 run in NumPy over a predictor with a closed form."""
    rng = np.random.default_rng(0)
    clean, noise = rng.normal(size=(3, 5, 4, 2)), rng.normal(size=(3, 5, 4, 2))
    d = 1 / 2.0 ** rng.integers(0, 4, (3, 5))
    tau = np.floor(rng.uniform(size=(3, 5)) / d) * d
    weights = rng.normal(size=(4, 2))

    def predictor(xp):
        def f(noisy, tau, d):
            shape = (*tau.shape, 1, 1)
            return xp.tanh(noisy * weights + tau.reshape(shape) - 2 * d.reshape(shape))
        return f

    with jax.enable_x64(new_val=True):
        found = np.asarray(shortcut_forcing_loss(predictor(jnp), clean, noise, tau, d, 1 / 8))
        roundings = chain_roundings(jax.make_jaxpr(
            lambda *a: shortcut_forcing_loss(predictor(jnp), *a, 1 / 8))(clean, noise, tau, d))
    f = predictor(np)
    expected = np.zeros(tau.shape)
    for b in range(3):
        for t in range(5):
            z1, z0, level, step = clean[b, t], noise[b, t], tau[b, t], d[b, t]
            noisy = (1 - level) * z0 + level * z1
            one = np.asarray([[level]]), np.asarray([[step]])
            predicted = f(noisy[None, None], *one)[0, 0]
            if step <= 1 / 8:
                loss = np.sum((predicted - z1) ** 2)
            else:
                half = np.asarray([[step / 2]])
                first = (f(noisy[None, None], one[0], half)[0, 0] - noisy) / (1 - level)
                between = noisy + first * step / 2
                later = f(between[None, None], one[0] + step / 2, half)[0, 0]
                second = (later - between) / (1 - level - step / 2)
                velocity = (predicted - noisy) / (1 - level)
                loss = (1 - level) ** 2 * np.sum((velocity - (first + second) / 2) ** 2)
            expected[b, t] = (0.9 * level + 0.1) * loss
    assert_computes_the_oracle(found, expected, "shortcut forcing", roundings=roundings)


def test_step_sizes_are_powers_of_two_and_levels_lie_on_their_grid():
    """Eq. 4: every step size is the reciprocal of a power of two up to K_max, every one is drawn, and
    each signal level is a multiple of its frame's step size below 1."""
    tau, d = (np.asarray(x) for x in sampled_steps(jax.random.key(1), (4096,), 64))
    exponents = -np.log2(d)
    np.testing.assert_array_equal(exponents, np.round(exponents))
    assert set(exponents.astype(int)) == set(range(7))
    np.testing.assert_array_equal(tau / d, np.round(tau / d))
    assert (tau >= 0).all() and (tau <= 1 - d).all()


def test_the_dynamics_reads_no_frame_after_its_own():
    """A frame's prediction is unchanged, bit for bit, by any change to a later frame's latents, level
    or step: the transformer attends block-causally over time."""
    model = Dynamics(latents=8, bottleneck=4, spatial=4, width=32, depth=2, heads=2, levels=4, frames=6)
    rng = np.random.default_rng(2)
    noisy = jnp.asarray(rng.normal(size=(2, 6, 8, 4)), jnp.float32)
    tau, d = jnp.full((2, 6), 0.5), jnp.full((2, 6), 0.25)
    params = model.init(jax.random.key(3), noisy, tau, d)
    changed = noisy.at[:, 4:].set(jnp.asarray(rng.normal(size=(2, 2, 8, 4)), jnp.float32))
    apply = jax.jit(model.apply)
    later = apply(params, changed, tau.at[:, 4:].set(0.125), d.at[:, 4:].set(0.5))
    np.testing.assert_array_equal(apply(params, noisy, tau, d)[:, :4], later[:, :4])
