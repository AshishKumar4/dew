"""Process mutation witnesses: conditioning, sampling and raw model arithmetic."""


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from dew.diffusion.process import DenoisingCondition, Process, aligned_conditions
from dew.diffusion.schedules import FlowMatchingScheduler, KarrasVENoiseScheduler
from dew.diffusion.transforms import EpsilonPredictionTransform, KarrasPredictionTransform


class Conditioned(nn.Module):
    @nn.compact
    def __call__(self, x, time, conditioning):
        caption = jnp.mean(conditioning.context, axis=(1, 2))[:, None]
        return 2 * x + caption + conditioning.guidance[:, None] + time[:, None] / 1000


class Sized(nn.Module):
    """A denoiser that reads its row's size and crop (SDXL's `time_ids`) beside its caption."""

    @nn.compact
    def __call__(self, x, time, conditioning):
        caption = jnp.mean(conditioning.context, axis=(1, 2))[:, None]
        return 2 * x + caption + jnp.sum(conditioning.time_ids, axis=-1)[:, None]


class Doubled(nn.Module):
    @nn.compact
    def __call__(self, x, time):
        return 2 * x


@pytest.mark.parametrize("given_record,null_record", [(True, False), (False, True)])
def test_mismatched_record_kinds_are_refused_on_either_side(given_record, null_record):
    array = jnp.zeros((1, 2, 3))
    record = DenoisingCondition(array)
    with pytest.raises(ValueError):
        aligned_conditions({"conditioning": record if given_record else array},
                           {"conditioning": record if null_record else array})


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


def test_a_blank_caption_keeps_its_rows_size_and_crop():
    """SDXL's `time_ids` hold the image's size and crop, the row's and not its caption's, so the
    unconditional branch reads the blank caption at its row's own geometry: the two branches differ by
    the caption alone, as classifier-free guidance compares them."""
    with jax.enable_x64():
        process = Process(FlowMatchingScheduler(), EpsilonPredictionTransform())
        context = jnp.broadcast_to(jnp.array([1., 2.])[:, None, None], (2, 3, 4))
        sizes = jnp.array([[512., 512., 0., 0., 512., 512.], [768., 1024., 64., 0., 768., 768.]])
        given = DenoisingCondition(context, time_ids=sizes)
        null = DenoisingCondition(jnp.full((1, 3, 4), -3.), time_ids=jnp.full((1, 6), 1024.))
        denoiser = process.denoiser(Sized(), {}, {"conditioning": given}, {"conditioning": null})
        x = jnp.array([[1., 2., 3.], [4., 5., 6.]], jnp.float64)
        conditional, unconditional = denoiser.raw_both(x, jnp.array([.25, .5]))
        np.testing.assert_array_equal(np.asarray(conditional) - np.asarray(unconditional),
                                      np.float64([1., 2.])[:, None] + 3)
