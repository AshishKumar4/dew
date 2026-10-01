"""Process mutation witnesses: conditioning, sampling and raw model arithmetic."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from dew.diffusion.process import DenoisingCondition, Process, aligned_conditions
from dew.diffusion.schedules import CosineNoiseScheduler, FlowMatchingScheduler, KarrasVENoiseScheduler
from dew.diffusion.transforms import EpsilonPredictionTransform, KarrasPredictionTransform


class Conditioned(nn.Module):
    @nn.compact
    def __call__(self, x, time, conditioning):
        caption = jnp.mean(conditioning.context, axis=(1, 2))[:, None]
        return 2 * x + caption + conditioning.guidance[:, None] + time[:, None] / 1000


class Doubled(nn.Module):
    @nn.compact
    def __call__(self, x, time):
        return 2 * x


def test_aligned_records_keep_the_null_caption_and_the_given_rows_guidance():
    given = DenoisingCondition(jnp.ones((2, 3, 4)), guidance=jnp.array([2., 4.]))
    null = DenoisingCondition(jnp.zeros((1, 3, 4)), guidance=jnp.array([-9.]))
    spatial = jnp.arange(6).reshape(1, 2, 3)
    aligned = aligned_conditions({"conditioning": given, "spatial": jnp.ones((2, 2, 3))},
                                 {"conditioning": null, "spatial": spatial})
    assert aligned["conditioning"].context is null.context
    np.testing.assert_array_equal(aligned["conditioning"].guidance, given.guidance)
    assert aligned["spatial"] is spatial
    np.testing.assert_array_equal(null.guidance, [-9.])


@pytest.mark.parametrize("given_record,null_record", [(True, False), (False, True)])
def test_mismatched_record_kinds_are_refused_on_either_side(given_record, null_record):
    array = jnp.zeros((1, 2, 3))
    record = DenoisingCondition(array)
    with pytest.raises(ValueError, match="record on one side only"):
        aligned_conditions({"conditioning": record if given_record else array},
                           {"conditioning": record if null_record else array})


def test_a_two_entry_table_caps_the_grid_but_continuous_domains_do_not():
    table = Process(CosineNoiseScheduler(2), EpsilonPredictionTransform())
    np.testing.assert_array_equal(table.times(5), np.float32([2, 0]))
    continuous = KarrasVENoiseScheduler(sigma_min=1, sigma_max=3, rho=1)
    continuous.T = 0.5
    process = Process(FlowMatchingScheduler(), EpsilonPredictionTransform(), sampling=continuous)
    np.testing.assert_array_equal(process.times(5), np.float32([.5, .375, .25, .125, 0]))


def test_sampling_noise_uses_the_sampling_priors_scale_and_the_exact_key():
    with jax.enable_x64():
        sampling = KarrasVENoiseScheduler(sigma_min=1, sigma_max=3, rho=1)
        process = Process(FlowMatchingScheduler(), EpsilonPredictionTransform(), sampling=sampling)
        key, shape = jax.random.key(11), (2, 3, 4)
        # The schedule stores rates in fp32; the Gaussian and multiplication
        # are fp64 here, so the exact reference uses that stored prior scale.
        scale = np.float64(np.sqrt(np.float32(10)))
        expected = np.asarray(jax.random.normal(key, shape), np.float64) * scale
        np.testing.assert_array_equal(process.noise(key, shape), expected)


def test_edm_raw_input_is_preconditioned_before_the_model_is_called():
    with jax.enable_x64():
        process = Process(KarrasVENoiseScheduler(sigma_min=0, sigma_max=1, rho=1),
                          KarrasPredictionTransform(sigma_data=2))
        denoiser = process.denoiser(Doubled(), {}, {})
        x = jnp.array([[1., 2., 3.], [4., 5., 6.]], jnp.float64)
        # At sigma=0, EDM's c_in=1/sigma_data=1/2. The model doubles its
        # input, giving x exactly in fp64; division or exponentiation do not.
        np.testing.assert_array_equal(denoiser.raw(x, jnp.zeros((2,))), np.asarray(x))


def test_guided_raw_batch_preserves_caption_axes_and_per_row_guidance():
    with jax.enable_x64():
        process = Process(FlowMatchingScheduler(), EpsilonPredictionTransform())
        context = jnp.broadcast_to(jnp.array([1., 2.])[:, None, None], (2, 3, 4))
        given = DenoisingCondition(context, guidance=jnp.array([2., 4.]))
        null = DenoisingCondition(jnp.full((1, 3, 4), -3.), guidance=jnp.array([-9.]))
        denoiser = process.denoiser(Conditioned(), {}, {"conditioning": given}, {"conditioning": null})
        x = jnp.array([[1., 2., 3.], [4., 5., 6.]], jnp.float64)
        t = jnp.array([.25, .5])
        conditional, unconditional = denoiser.raw_both(x, t)
        common = 2 * np.asarray(x) + np.float64([2., 4.])[:, None] + np.float64([.25, .5])[:, None]
        np.testing.assert_array_equal(conditional, common + np.float64([1., 2.])[:, None])
        np.testing.assert_array_equal(unconditional, common - 3)


@pytest.mark.parametrize("guided", [False, True])
def test_interval_duration_uses_a_difference_on_both_immutable_branches(guided):
    process = Process(FlowMatchingScheduler(), EpsilonPredictionTransform(), interval=True)
    conditions = {"label": jnp.array([1., 2.])}
    null = {"label": jnp.zeros((1,))} if guided else None
    denoiser = process.denoiser(Doubled(), {}, conditions, null)
    spanned = denoiser.spanning(jnp.array([.75, .5]), jnp.array([.25, .125]))
    np.testing.assert_array_equal(spanned.conditions["duration"], [500., 375.])
    if guided:
        np.testing.assert_array_equal(spanned.unconditional["duration"], [500., 375.])
    else:
        assert spanned.unconditional is None
    assert "duration" not in denoiser.conditions
    with pytest.raises(FrozenInstanceError):
        denoiser.process = Process(FlowMatchingScheduler(), EpsilonPredictionTransform())
    with pytest.raises(FrozenInstanceError):
        process.interval = False
