"""Invariant tests for the noise schedulers.

These encode the properties the rest of the library relies on:
variance preservation for VP schedules, alpha=1 for the generalized (VE)
schedules, monotone SNR along the trajectory, rates that broadcast against
image and video batches, and exact invertibility of the forward diffusion.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dew.diffusion.schedules as schedulers
from dew.diffusion import (
    EpsilonPredictionTransform,
    KarrasPredictionTransform,
    MinSNR,
    Process,
    VPredictionTransform,
    broadcast_rates,
    expand,
    presets,
)
from dew.diffusion.schedules import (
    CosineContinuousNoiseScheduler,
    CosineGeneralNoiseScheduler,
    CosineNoiseScheduler,
    EDMNoiseScheduler,
    ExpNoiseScheduler,
    FlowMatchingScheduler,
    KarrasVENoiseScheduler,
    LinearNoiseScheduler,
    SqrtContinuousNoiseScheduler,
)

# Timesteps ascending from low to high noise, in each schedule's own domain:
# an index into the beta table for the discrete schedules, [0, 1] for the
# continuous ones.
DISCRETE_STEPS = jnp.array([10, 300, 600, 900])
CONTINUOUS_STEPS = jnp.array([0.05, 0.3, 0.6, 0.95])

# (class, factory, probe steps, family); family picks the extra rate
# identity: 'vp' is variance preserving, 've' keeps alpha=1 and scales the
# input, 'flow' is the rectified-flow linear path.
SCHEDULES = [
    (CosineNoiseScheduler, partial(CosineNoiseScheduler, 1000), DISCRETE_STEPS, 'vp'),
    (LinearNoiseScheduler, partial(LinearNoiseScheduler, 1000), DISCRETE_STEPS, 'vp'),
    (ExpNoiseScheduler, partial(ExpNoiseScheduler, 1000), DISCRETE_STEPS, 'vp'),
    (CosineContinuousNoiseScheduler, CosineContinuousNoiseScheduler, CONTINUOUS_STEPS, 'vp'),
    (SqrtContinuousNoiseScheduler, SqrtContinuousNoiseScheduler, CONTINUOUS_STEPS, 'vp'),
    (CosineGeneralNoiseScheduler, CosineGeneralNoiseScheduler, CONTINUOUS_STEPS, 've'),
    (KarrasVENoiseScheduler, partial(KarrasVENoiseScheduler, sigma_max=80, rho=7, sigma_data=0.5), CONTINUOUS_STEPS, 've'),
    (EDMNoiseScheduler, partial(EDMNoiseScheduler, sigma_max=80, sigma_data=0.5), CONTINUOUS_STEPS, 've'),
    (FlowMatchingScheduler, FlowMatchingScheduler, CONTINUOUS_STEPS, 'flow'),
]

ALL_CASES = SCHEDULES
ALL_IDS = [cls.__name__ for cls, *_ in SCHEDULES]
VP_CASES = [case for case in ALL_CASES if case[3] == 'vp']
VP_IDS = [case[0].__name__ for case in VP_CASES]
VE_CASES = [case for case in ALL_CASES if case[3] == 've']
VE_IDS = [case[0].__name__ for case in VE_CASES]


@pytest.mark.parametrize("cls,make,steps,family", ALL_CASES, ids=ALL_IDS)
def test_snr_decreases_along_the_trajectory(cls, make, steps, family):
    """t runs from clean to noisy, so the signal-to-noise ratio must fall."""
    snr = make().snr(steps)
    assert jnp.all(jnp.diff(snr) < 0), snr


@pytest.mark.parametrize("cls,make,steps,family", ALL_CASES, ids=ALL_IDS)
@pytest.mark.parametrize("sample_shape", [(8, 8, 3), (2, 8, 8, 3)], ids=['image', 'video'])
def test_rates_broadcast_against_the_sample(cls, make, steps, family, sample_shape):
    """broadcast_rates is how every caller shapes the rates: the result must
    broadcast against the batch it came from, for images and for video."""
    x = jnp.zeros((len(steps), *sample_shape))
    alpha, sigma = broadcast_rates(make(), steps, x)
    assert alpha.shape == sigma.shape == (len(steps),) + (1,) * (x.ndim - 1)
    assert (alpha * x).shape == x.shape
    assert jnp.array_equal(alpha, expand(make().rates(steps)[0], x))


@pytest.mark.parametrize("cls,make,steps,family", ALL_CASES, ids=ALL_IDS)
@pytest.mark.parametrize("sample_shape", [(8, 8, 3), (2, 8, 8, 3)], ids=['image', 'video'])
def test_forward_diffusion_invertible(cls, make, steps, family, sample_shape, rng):
    """The noised sample gives x_0 back exactly at the same timestep, for the
    epsilon parameterization on every schedule."""
    schedule = make()
    key0, key1 = jax.random.split(rng)
    full_shape = (len(steps), *sample_shape)
    x0 = jax.random.normal(key0, full_shape)
    noise = jax.random.normal(key1, full_shape)
    rates = broadcast_rates(schedule, steps, x0)
    xt, _, target = EpsilonPredictionTransform().forward_diffusion(x0, noise, rates)
    recovered, _ = EpsilonPredictionTransform().backward_diffusion(xt, target, rates)
    assert xt.shape == x0.shape
    assert jnp.max(jnp.abs(recovered - x0)) < 1e-4


@pytest.mark.parametrize("cls,make,steps,family", ALL_CASES, ids=ALL_IDS)
def test_training_times_stay_in_the_domain(cls, make, steps, family, rng):
    """sample_t draws what rates accepts: indices below T for a table, [0, 1)
    for a continuous schedule, any real for EDM's log-normal sigmas."""
    schedule = make()
    t = schedule.sample_t(rng, 1000)
    assert t.shape == (1000,)
    alpha, sigma = schedule.rates(t)
    assert jnp.all(jnp.isfinite(alpha)) and jnp.all(sigma > 0)
    assert jnp.all(jnp.isfinite(schedule.weight(t)))
    if isinstance(schedule, schedulers.DiscreteNoiseScheduler):
        assert jnp.issubdtype(t.dtype, jnp.integer)
        assert int(t.min()) >= 0 and int(t.max()) < schedule.T
    elif not isinstance(schedule, EDMNoiseScheduler):
        assert float(t.min()) >= 0 and float(t.max()) < schedule.T


@pytest.mark.parametrize("cls,make,steps,family", VP_CASES, ids=VP_IDS)
def test_vp_variance_preserving(cls, make, steps, family):
    alpha, sigma = make().rates(steps)
    assert jnp.allclose(alpha**2 + sigma**2, 1.0, atol=1e-5)


@pytest.mark.parametrize("cls,make,steps,family", VE_CASES, ids=VE_IDS)
def test_ve_signal_rate_is_one(cls, make, steps, family):
    alpha, sigma = make().rates(steps)
    assert jnp.allclose(alpha, 1.0)
    # Noise grows monotonically with t
    assert jnp.all(jnp.diff(sigma) > 0)


@pytest.mark.parametrize("cls,make,steps,family", VE_CASES, ids=VE_IDS)
def test_ve_schedules_invert_their_sigmas(cls, make, steps, family):
    """The sigma integrators step in sigma and read the model back at the
    time that sigma belongs to."""
    schedule = make()
    assert jnp.allclose(schedule.t_of_sigma(schedule.sigmas(steps)), steps, atol=1e-5)


@pytest.mark.parametrize("cls,make,steps,family", VE_CASES, ids=VE_IDS)
def test_ve_schedules_condition_the_model_on_log_sigma_over_four(cls, make, steps, family):
    """c_noise of Karras et al. 2022 Table 1, the one input the EDM
    preconditioned oracle reads its sigma from."""
    schedule = make()
    assert jnp.allclose(schedule.model_time(steps), jnp.log(schedule.sigmas(steps)) / 4)


def test_flow_matching_stays_on_the_linear_path():
    """Rectified flow: x_t = (1 - t) x_0 + t eps, so the rates sum to one."""
    alpha, sigma = FlowMatchingScheduler().rates(CONTINUOUS_STEPS)
    assert jnp.allclose(alpha + sigma, 1.0, atol=1e-6)


def test_sqrt_schedule_matches_the_diffusion_lm_formula():
    """Li et al. 2022: alpha = sqrt(1 - t), sigma = sqrt(t), the plain x_0 loss."""
    schedule = SqrtContinuousNoiseScheduler()
    alpha, sigma = schedule.rates(CONTINUOUS_STEPS)
    assert jnp.allclose(alpha, jnp.sqrt(1 - CONTINUOUS_STEPS), atol=1e-6)
    assert jnp.allclose(sigma, jnp.sqrt(CONTINUOUS_STEPS), atol=1e-6)
    assert jnp.allclose(schedule.weight(CONTINUOUS_STEPS), 1.0)


def test_discrete_table_reaches_its_last_entry_at_t_equal_T():
    """A sampling grid starts at T itself, one past the last index, and reads
    the noisiest entry; an out-of-range gather would return something else."""
    schedule = CosineNoiseScheduler(1000)
    top = schedule.rates(jnp.array([1000.0]))
    last = schedule.rates(jnp.array([999]))
    assert jnp.array_equal(top[0], last[0]) and jnp.array_equal(top[1], last[1])


@pytest.mark.parametrize("cls,make,steps,family", VE_CASES, ids=VE_IDS)
def test_generalized_weights_are_the_edm_lambda(cls, make, steps, family):
    """Karras et al. 2022 Eq. 8: lambda(sigma) = (sigma^2 + sigma_data^2) /
    (sigma sigma_data)^2, for every variance exploding schedule and its own
    sigma_data."""
    schedule = make()
    sigma = schedule.sigmas(steps)
    expected = (sigma**2 + schedule.sigma_data**2) / ((sigma * schedule.sigma_data) ** 2)
    assert jnp.allclose(schedule.weight(steps), expected, rtol=1e-4)


def test_karras_weights_at_sigma_min():
    """At sigma_min, where (sigma sigma_data)^2 is 1e-6, the weight is the
    formula's 1e6 and not what an epsilon in the denominator would halve it to."""
    schedule = KarrasVENoiseScheduler(sigma_min=0.002, sigma_max=80, rho=7, sigma_data=0.5)
    sigma = jnp.array([0.002])
    expected = (sigma**2 + 0.5**2) / ((sigma * 0.5) ** 2)
    # steps=0 maps to sigma_min under the karras rho spacing
    assert jnp.allclose(schedule.weight(jnp.array([0.0])), expected, rtol=1e-2)


def test_cosine_general_weights_read_its_sigma_data():
    """The EDM lambda depends on sigma_data, so two values of it are two
    weightings and not one."""
    wide = CosineGeneralNoiseScheduler(sigma_data=1.0).weight(CONTINUOUS_STEPS)
    narrow = CosineGeneralNoiseScheduler(sigma_data=0.5).weight(CONTINUOUS_STEPS)
    assert jnp.allclose(narrow - wide, 1 / 0.5**2 - 1 / 1.0**2, rtol=1e-5)


@pytest.mark.parametrize("P_mean,P_std", [(-0.4, 1.0), (-1.2, 1.2)])
def test_edm_lognormal_sigma_distribution(rng, P_mean, P_std):
    """EDM training sigmas follow exp(N(P_mean, P_std^2)), defaulting to EDM2."""
    schedule = EDMNoiseScheduler(sigma_max=80, sigma_data=0.5, P_mean=P_mean, P_std=P_std)
    log_sigma = jnp.log(schedule.sigmas(schedule.sample_t(rng, 20000)))
    assert abs(float(jnp.mean(log_sigma)) - P_mean) < 0.05
    assert abs(float(jnp.std(log_sigma)) - P_std) < 0.05


def test_cosine_table_is_nichol_and_dhariwals_cumulative_alpha():
    """Nichol and Dhariwal 2021, Eq. 17: at index t the cumulative alpha is
    f(t + 1) / f(0) with f(u) = cos^2((u / T + s) / (1 + s) pi / 2), where the
    table is the product of one minus each beta. Checked away from the top of
    the table, where the clip on each beta rounds the last few entries."""
    T, s = 1000, 0.008
    schedule = CosineNoiseScheduler(T, beta_start=s)
    index = jnp.array([0, 10, 300, 600, 900])
    def f(u):
        return jnp.cos((u / T + s) / (1 + s) * jnp.pi / 2) ** 2
    expected = f(index + 1.0) / f(0.0)
    assert jnp.allclose(schedule.rates(index)[0] ** 2, expected, rtol=1e-4)


def test_discrete_p2_default_makes_the_v_loss_an_x0_loss():
    """The P2 weight at k = 1, gamma = 1 is 1 / (1 + SNR), and the v error is
    (1 + SNR) times the x_0 error, so their product is the unweighted x_0 loss."""
    schedule = CosineNoiseScheduler(1000)
    snr = schedule.snr(DISCRETE_STEPS)
    assert jnp.allclose(schedule.weight(DISCRETE_STEPS) * VPredictionTransform().target_error_scale(snr),
                        1.0, rtol=1e-4)


def test_small_beta_p2_weights_agree_with_the_reported_snr():
    """A rounded alpha-bar of one must not erase a nonzero training weight."""
    schedule = schedulers.DiscreteNoiseScheduler(np.array([1e-10, 1e-7, .01], np.float32))
    steps = jnp.arange(3)
    expected = 1 / (1 + np.asarray(schedule.snr(steps), np.float64))
    # The independently rounded rates, SNR and weight allow a few fp32 operations.
    np.testing.assert_allclose(schedule.weight(steps), expected,
                               rtol=8 * np.finfo(np.float32).eps, atol=0)

############################################################################################################
# min-SNR-gamma loss weighting (Hang et al. 2023), through Process
############################################################################################################

MIN_SNR_STEPS = jnp.array([10, 200, 400, 600, 800, 990])


def min_snr_process(transform, gamma):
    return Process(CosineNoiseScheduler(1000), transform, weighting=MinSNR(gamma))


def test_min_snr_epsilon_weights_match_the_paper():
    process = min_snr_process(EpsilonPredictionTransform(), 5.0)
    snr = process.schedule.snr(MIN_SNR_STEPS)
    expected = jnp.minimum(snr, 5.0) / snr
    assert jnp.allclose(process.weight(MIN_SNR_STEPS), expected, rtol=1e-5)


def test_min_snr_v_weights_match_the_paper():
    process = min_snr_process(VPredictionTransform(), 5.0)
    snr = process.schedule.snr(MIN_SNR_STEPS)
    expected = jnp.minimum(snr, 5.0) / (snr + 1)
    assert jnp.allclose(process.weight(MIN_SNR_STEPS), expected, rtol=1e-5)


def test_min_snr_karras_weights_match_the_paper():
    """On the EDM preconditioning the x_0 error is c_out times the raw error,
    so the x_0-space min-SNR weight divides by 1 / sigma_data^2 + SNR."""
    process = Process(KarrasVENoiseScheduler(sigma_data=0.5), KarrasPredictionTransform(0.5),
                      weighting=MinSNR(5.0))
    snr = process.schedule.snr(CONTINUOUS_STEPS)
    expected = jnp.minimum(snr, 5.0) / (1 / 0.5**2 + snr)
    assert jnp.allclose(process.weight(CONTINUOUS_STEPS), expected, rtol=1e-5)


def test_min_snr_weights_are_capped_and_non_increasing_in_snr():
    """High-SNR (low noise) steps stop dominating the gradient."""
    process = min_snr_process(EpsilonPredictionTransform(), 5.0)
    # ascending timesteps are descending SNR, so weights must be non-decreasing
    weights = process.weight(jnp.arange(1, 1000, 10))
    assert jnp.all(jnp.diff(weights) >= -1e-6)
    assert jnp.all(weights <= 1.0 + 1e-6)


def test_min_snr_gamma_infinity_is_the_unweighted_case():
    process = min_snr_process(EpsilonPredictionTransform(), float('inf'))
    assert jnp.allclose(process.weight(MIN_SNR_STEPS), 1.0, atol=1e-6)


def test_the_schedule_weight_is_the_default():
    process = Process(CosineNoiseScheduler(1000), VPredictionTransform())
    snr = process.schedule.snr(MIN_SNR_STEPS)
    assert jnp.allclose(process.weight(MIN_SNR_STEPS), 1 / (1 + snr), rtol=1e-5)


############################################################################################################
# Presets
############################################################################################################

@pytest.mark.parametrize("preset, scale, ordinary", [
    (presets.Cosine, lambda snr: snr + 1, lambda snr: 1 / (snr + 1)),
    (presets.EDM, lambda snr: snr + 4, lambda snr: snr + 4),
    (presets.Karras, lambda snr: snr + 4, lambda snr: snr + 4),
    (presets.Flow, lambda snr: (1 + jnp.sqrt(snr)) ** 2, jnp.ones_like),
    (presets.Sqrt, jnp.ones_like, jnp.ones_like),
], ids=["cosine", "edm", "karras", "flow", "sqrt"])
def test_preset_weights_the_training_schedule_with_min_snr(preset, scale, ordinary):
    """The capped x_0 loss is converted to each preset's prediction space.

    Without the cap: P2 for cosine, EDM lambda for Karras preconditioning,
    and the unweighted velocity/x_0 loss for flow/square-root.
    """
    fields = {"regime": "pixel"} if preset is presets.EDM else {}
    capped, standard = preset(min_snr_gamma=5.0, **fields)(), preset(**fields)()
    times = jnp.array([0.05, 0.25, 0.5, 0.75, 0.95]) * capped.schedule.T
    snr = capped.schedule.snr(times)
    assert jnp.allclose(capped.weight(times), jnp.minimum(snr, 5.0) / scale(snr), rtol=1e-5)
    assert jnp.allclose(standard.weight(times), ordinary(snr), rtol=1e-5)


@pytest.mark.parametrize("fields, mean, std", [
    ({"regime": "pixel"}, -1.2, 1.2),
    ({"regime": "latent"}, -0.4, 1.0),
    ({"regime": "pixel", "P_mean": 0.2}, 0.2, 1.2),
    ({"regime": "latent", "P_std": 0.7}, -0.4, 0.7),
    ({"regime": "pixel", "P_mean": -0.8, "P_std": 0.7}, -0.8, 0.7),
], ids=["pixel", "latent", "mean_override", "std_override", "both_override"])
def test_edm_draws_the_sigmas_of_the_space_it_denoises(rng, fields, mean, std):
    schedule = presets.EDM(**fields)().schedule
    log_sigma = jnp.log(schedule.sigmas(schedule.sample_t(rng, 20000)))
    assert abs(float(jnp.mean(log_sigma)) - mean) < 0.05
    assert abs(float(jnp.std(log_sigma)) - std) < 0.05


def test_edm_refuses_an_unspecified_training_distribution():
    with pytest.raises(ValueError):
        presets.EDM()()


def test_a_run_config_draws_pixel_sigmas_without_an_autoencoder_and_latent_ones_with(rng):
    from dew.objectives.diffusion.config import DiffusionRunConfig, PretrainedAutoencoder

    for config, mean, std in (
        (DiffusionRunConfig(), -1.2, 1.2),
        (DiffusionRunConfig(autoencoder=PretrainedAutoencoder()), -0.4, 1.0),
        (DiffusionRunConfig(preset=presets.EDM(P_mean=-0.8, P_std=0.7)), -0.8, 0.7),
    ):
        schedule = config.preset().schedule
        log_sigma = jnp.log(schedule.sigmas(schedule.sample_t(rng, 20000)))
        assert abs(float(jnp.mean(log_sigma)) - mean) < 0.05
        assert abs(float(jnp.std(log_sigma)) - std) < 0.05


def test_edm_preset_samples_on_the_karras_grid():
    """Training draws log-normal sigmas; inference walks the rho-spaced grid
    with the same sigma range and sigma_data."""
    process = presets.EDM(sigma_min=0.01, sigma_max=40.0, rho=5.0, sigma_data=0.7, regime="pixel")()
    # Eq. 5 at rho 5: the endpoints are the preset's sigma range and the
    # midpoint pins the spacing, 2.99 here against 2.02 at the default rho 7.
    midpoint = ((40.0 ** 0.2 + 0.01 ** 0.2) / 2) ** 5
    assert process.sampler_schedule.sigmas(jnp.array([0.0, 0.5, 1.0])).tolist() == (
        pytest.approx([0.01, midpoint, 40.0], rel=1e-5))
    sigmas = jnp.array([0.01, midpoint, 40.0])
    times = process.schedule.t_of_sigma(sigmas)
    assert jnp.allclose(process.weight(times), 1 / 0.7**2 + 1 / sigmas**2, rtol=1e-5)
    assert jnp.allclose(process.prediction.get_input_scale(process.schedule.rates(times)),
                        1 / jnp.sqrt(sigmas**2 + 0.7**2), rtol=1e-5)


def test_presets_rebuild_from_their_fields(rng):
    """Recorded flow fields control the rebuilt rates and training draws."""
    import dataclasses

    from dew.registry import presets as registry
    preset = registry.Flow(shift=3.0, logit_mean=0.5, logit_std=0.7)
    process = registry.build("flow", **dataclasses.asdict(preset))()
    times = jnp.array([0.05, 0.5, 0.95])
    alpha, sigma = process.schedule.rates(times)
    expected = 3 * times / (1 + 2 * times)
    assert jnp.allclose(sigma, expected, rtol=1e-6)
    assert jnp.allclose(alpha, 1 - expected, rtol=1e-6)
    draws = process.schedule.sample_t(rng, 20000)
    logits = jnp.log(draws / (1 - draws))
    assert abs(float(jnp.mean(logits)) - 0.5) < 0.05
    assert abs(float(jnp.std(logits)) - 0.7) < 0.05
    with pytest.raises(ValueError):
        registry.build("flow", shfit=3.0)
