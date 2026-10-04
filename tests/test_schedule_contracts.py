"""NoiseScheduler's own time intervals, prior and variance-exploding equations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.diffusion.schedules.common import GeneralizedNoiseScheduler, NoiseScheduler


class LinearSigma(GeneralizedNoiseScheduler):
    def sigmas(self, t):
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)

    def t_of_sigma(self, sigma):
        return (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)


def test_base_time_and_runge_kutta_intervals_are_the_declared_fp32_values():
    schedule = LinearSigma()
    t, following = np.float64([.75, .5, .125]), np.float64([.25, .375, .0625])
    with jax.enable_x64():
        np.testing.assert_array_equal(schedule.step_interval(t, following), t - following)
        np.testing.assert_array_equal(schedule.half_interval(t, following), (t - following) / 2)
    assert schedule.step_interval(t, following).dtype == jnp.float32
    assert schedule.half_interval(t, following).dtype == jnp.float32


def test_generalized_defaults_reach_the_papers_sigma_endpoints_and_data_scale():
    schedule = LinearSigma()
    alpha, sigma = schedule.rates(jnp.array([0., 1.]))
    np.testing.assert_array_equal(alpha, np.float32([1., 1.]))
    np.testing.assert_array_equal(sigma, np.float32([.002, 80.]))
    assert schedule.sigma_data == .5
    expected = np.sqrt(np.float32(1 + 80 ** 2))
    np.testing.assert_array_equal(schedule.prior_scale(), expected)


def test_generalized_weight_is_the_fp64_edm_lambda_with_two_fp32_roundings():
    schedule = LinearSigma(sigma_min=1, sigma_max=3, sigma_data=.5)
    times = np.float64([0., .5, 1.])
    sigma = np.float64([1., 2., 3.])
    reference = (sigma ** 2 + .5 ** 2) / (sigma * .5) ** 2
    # Dyadic parameters make every input exact. The reciprocal and addition
    # in the stored fp32 formula each round once; the oracle is fp64.
    np.testing.assert_allclose(schedule.weight(times), reference, rtol=2 * np.finfo(np.float32).eps, atol=0)


def test_model_time_is_log_sigma_over_four_without_an_epsilon_guard():
    schedule = LinearSigma(sigma_min=0, sigma_max=1)
    times = np.float64([.5, 1.])
    # The supplied times are dyadic and produce exact stored fp32 sigmas.
    # The logarithm rounds in fp32; division by four is exact binary scaling.
    expected = np.log(np.float64([.5, 1.])) / 4
    np.testing.assert_allclose(schedule.model_time(times), expected,
                               rtol=2 * np.finfo(np.float32).eps, atol=0)
    assert np.isneginf(np.asarray(schedule.model_time(np.float32(0))))


def test_base_prior_scale_squares_both_nonunit_signal_and_noise_rates():
    class PartialSignal(NoiseScheduler):
        T = 1.

        def rates(self, t):
            return 1 - t / 2, 3 * t / 2

        def sample_t(self, key, n):
            return jax.random.uniform(key, (n,))

        def weight(self, t):
            return jnp.ones_like(t)

    # alpha(T)=1/2 and sigma(T)=3/2 are exactly representable; the fp64
    # reference rounds only its square root back to the production dtype.
    expected = np.float32(np.sqrt(np.float64(.5 ** 2 + 1.5 ** 2)))
    np.testing.assert_array_equal(PartialSignal().prior_scale(), expected)


@pytest.mark.parametrize("missing", ["rates", "sample_t", "weight"])
def test_each_required_forward_schedule_operation_prevents_incomplete_construction(missing):
    operations = {
        "T": 1.,
        "rates": lambda self, t: (jnp.ones_like(t), t),
        "sample_t": lambda self, key, n: jax.random.uniform(key, (n,)),
        "weight": lambda self, t: jnp.ones_like(t),
    }
    del operations[missing]
    incomplete = type("IncompleteForwardSchedule", (NoiseScheduler,), operations)
    with pytest.raises(TypeError):
        incomplete()


@pytest.mark.parametrize("missing", ["sigmas", "t_of_sigma"])
def test_each_required_sigma_schedule_operation_prevents_incomplete_construction(missing):
    operations = {"sigmas": lambda self, t: 1 + t, "t_of_sigma": lambda self, sigma: sigma - 1}
    del operations[missing]
    incomplete = type("IncompleteSigmaSchedule", (GeneralizedNoiseScheduler,), operations)
    with pytest.raises(TypeError):
        incomplete()



