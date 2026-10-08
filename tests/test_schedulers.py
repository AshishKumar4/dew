"""Invariant tests for the noise schedulers.

These encode the properties the rest of the library relies on:
variance preservation for VP schedules, alpha=1 for the generalized (VE)
schedules, monotone SNR along the trajectory, rates that broadcast against
image and video batches, and exact invertibility of the forward diffusion.
"""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

import dew.diffusion.schedules as schedulers
from dew.diffusion import (
    DirectPredictionTransform,
    EpsilonPredictionTransform,
    FlowMatchPredictionTransform,
    KarrasPredictionTransform,
    MinSNR,
    Process,
    VPredictionTransform,
    broadcast_rates,
    expand,
    presets,
)
from dew.diffusion.schedules import (
    CosineNoiseScheduler,
    EDMNoiseScheduler,
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
    (CosineNoiseScheduler, partial(CosineNoiseScheduler, 1000), DISCRETE_STEPS, "vp"),
    (LinearNoiseScheduler, partial(LinearNoiseScheduler, 1000), DISCRETE_STEPS, "vp"),
    (SqrtContinuousNoiseScheduler, SqrtContinuousNoiseScheduler, CONTINUOUS_STEPS, "vp"),
    (
        KarrasVENoiseScheduler,
        partial(KarrasVENoiseScheduler, sigma_max=80, rho=7, sigma_data=0.5),
        CONTINUOUS_STEPS,
        "ve",
    ),
    (EDMNoiseScheduler, partial(EDMNoiseScheduler, sigma_max=80, sigma_data=0.5), CONTINUOUS_STEPS, "ve"),
    (FlowMatchingScheduler, FlowMatchingScheduler, CONTINUOUS_STEPS, "flow"),
]

ALL_CASES = SCHEDULES
ALL_IDS = [cls.__name__ for cls, *_ in SCHEDULES]
VP_CASES = [case for case in ALL_CASES if case[3] == 'vp']
VP_IDS = [case[0].__name__ for case in VP_CASES]
VE_CASES = [case for case in ALL_CASES if case[3] == 've']
VE_IDS = [case[0].__name__ for case in VE_CASES]

BETAS = dict(np.load(Path(__file__).parent / "fixtures" / "schedules" / "betas.npz"))


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


@pytest.mark.parametrize("make", [SqrtContinuousNoiseScheduler,
                                  partial(KarrasVENoiseScheduler, sigma_max=80, rho=7, sigma_data=0.5)],
                         ids=["continuous", "generalized"])
def test_uniform_training_times_follow_the_uniform_law(make):
    """A continuous schedule (`ContinuousNoiseScheduler`) and a generalized
    one (`GeneralizedNoiseScheduler`) train on times drawn uniformly over
    [0, T). Over 2^16 draws, scipy's one-sample Kolmogorov-Smirnov test
    against U(0, T) must not reject at one in a billion, which fails any law
    whose CDF strays from the uniform's by more than 0.013 anywhere; every
    draw lies in [0, T)."""
    from scipy import stats

    schedule = make()
    t = np.asarray(schedule.sample_t(jax.random.key(17), 1 << 16), np.float64)
    assert t.min() >= 0 and t.max() < schedule.T
    assert stats.kstest(t, stats.uniform(0, schedule.T).cdf).pvalue > 1e-9


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


@pytest.mark.parametrize("steps", [2000, 1000])
def test_sqrt_schedule_is_diffusion_lms_table(steps):
    """Diffusion-LM's own `get_named_beta_schedule("sqrt", T)` (Li et al.
    2022, XiangLi1999/Diffusion-LM at 759889d, run by
    tools/beta_schedule_reference.py): at its step k the cumulative alpha is
    Dew's alpha^2 at t = (k + 1) / T. Dew evaluates in fp32 from a rounded t,
    so each rate squared carries at most 8 fp32 roundings of quantities no
    larger than one: t and t + s (each moving alpha^2 by at most half a
    rounding), the square root, the subtraction, the division and its
    constant (a rounding each), and the final square root, twice once
    squared. The last step is excluded: Diffusion-LM clips its beta at
    0.999 where the cumulative alpha crosses zero, and Dew stops at (0, 1)."""
    table = np.cumprod(1 - BETAS[f"diffusion_lm_sqrt_{steps}"])
    schedule = SqrtContinuousNoiseScheduler()
    alpha, sigma = schedule.rates(np.arange(1, steps + 1) / steps)
    bound = 8 * 2.0 ** -24
    assert np.max(np.abs(np.asarray(alpha, np.float64)[:-1] ** 2 - table[:-1])) <= bound
    assert np.max(np.abs(np.asarray(sigma, np.float64)[:-1] ** 2 - (1 - table[:-1]))) <= bound
    assert schedule.rates(jnp.asarray(1.0)) == (0.0, 1.0)
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


def test_generalized_weights_read_their_sigma_data():
    """The EDM lambda depends on sigma_data, so two values of it are two
    weightings and not one."""
    wide = KarrasVENoiseScheduler(sigma_data=1.0).weight(CONTINUOUS_STEPS)
    narrow = KarrasVENoiseScheduler(sigma_data=0.5).weight(CONTINUOUS_STEPS)
    assert jnp.allclose(narrow - wide, 1 / 0.5**2 - 1 / 1.0**2, rtol=1e-5)


@pytest.mark.parametrize("name,steps", [("linear", 1000), ("linear", 250), ("linear", 4000),
                                        ("cosine", 1000), ("cosine", 4000)])
def test_beta_tables_are_improved_diffusions(name, steps):
    """openai/improved-diffusion's own `get_named_beta_schedule` at 1bc7bbb
    (Nichol and Dhariwal 2021; tools/beta_schedule_reference.py): the linear
    table of Ho et al. scaled to the step count, and the cosine table with
    its 0.999 clip. Both are computed in float64 the way the authors do, so
    the betas are theirs to the bit, and the fp32 rates are their cumulative
    alphas rounded once."""
    reference = BETAS[f"improved_diffusion_{name}_{steps}"]
    schedule = (LinearNoiseScheduler if name == "linear" else CosineNoiseScheduler)(steps)
    np.testing.assert_array_equal(np.asarray(schedule.betas), reference)
    alpha, sigma = schedule.rates(jnp.arange(steps))
    cumulative = np.cumprod(1 - reference)
    np.testing.assert_array_equal(alpha, np.sqrt(cumulative).astype(np.float32))
    np.testing.assert_array_equal(sigma, np.sqrt(1 - cumulative).astype(np.float32))


def test_small_beta_p2_weights_agree_with_the_reported_snr():
    """A rounded alpha-bar of one must not erase a nonzero training weight."""
    schedule = schedulers.DiscreteNoiseScheduler(np.array([1e-10, 1e-7, .01], np.float32))
    steps = jnp.arange(3)
    expected = 1 / (1 + np.asarray(schedule.snr(steps), np.float64))
    # The independently rounded rates, SNR and weight allow a few fp32 operations.
    np.testing.assert_allclose(schedule.weight(steps), expected,
                               rtol=8 * np.finfo(np.float32).eps, atol=0)

WEIGHTS = dict(np.load(Path(__file__).parent / "fixtures" / "weighting" / "weights.npz"))
TABLES = ("linear", "zero_terminal_cosine")


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("k,gamma", [(1, 1), (1, 0.5), (2, 1)])
def test_p2_weights_are_the_authors(k, gamma, table):
    """P2 (Choi et al. 2022) as jychoi118/P2-weighting@3da0947 computes it:
    its own `training_losses` on an epsilon model at every step of the
    table (tools/weighting_reference.py), against the discrete table's
    weight, held to the float64 rule."""
    key = f"p2_epsilon_p2_k{k}_p2_gamma{gamma}_{table}".replace(".", "p")
    schedule = schedulers.DiscreteNoiseScheduler(WEIGHTS[f"betas_{table}"], p2_loss_weight_k=k,
                                                 p2_loss_weight_gamma=gamma)
    weight = Process(schedule, EpsilonPredictionTransform()).weight(jnp.arange(schedule.T))
    assert_as_exact_as_the_reference(weight, WEIGHTS[key], WEIGHTS[f"{key}_f64"], key)


############################################################################################################
# min-SNR-gamma loss weighting (Hang et al. 2023), through Process
############################################################################################################

MIN_SNR_STEPS = jnp.array([10, 200, 400, 600, 800, 990])


def min_snr_process(transform, gamma):
    return Process(CosineNoiseScheduler(1000), transform, weighting=MinSNR(gamma))


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("transform,name", [
    (EpsilonPredictionTransform(), "epsilon_mse_loss_weight_typemin_snr_5"),
    (VPredictionTransform(), "velocity_mse_loss_weight_typevmin_snr_5"),
    (DirectPredictionTransform(), "start_x_mse_loss_weight_typemin_snr_5"),
], ids=["epsilon", "v", "x0"])
def test_min_snr_weights_are_the_authors(transform, name, table):
    """min-SNR-5 as TiankaiHang/Min-SNR-Diffusion-Training@5189997 computes
    it, per parameterization (`min_snr_5` on epsilon and x_0, `vmin_snr_5` on
    v): its own `training_losses` at every step (tools/weighting_reference.py),
    held to the float64 rule. On the zero-terminal table the last step has
    zero SNR, which the authors weight as one."""
    key = f"min_snr_{name}_{table}"
    process = Process(schedulers.DiscreteNoiseScheduler(WEIGHTS[f"betas_{table}"]), transform,
                      weighting=MinSNR(5.0))
    weight = process.weight(jnp.arange(process.schedule.T))
    assert_as_exact_as_the_reference(weight, WEIGHTS[key], WEIGHTS[f"{key}_f64"], key)


@pytest.mark.parametrize("transform", [KarrasPredictionTransform(0.5),
                                       KarrasPredictionTransform(0.5, velocity=True),
                                       DirectPredictionTransform()],
                         ids=["karras", "karras_velocity", "direct"])
def test_min_snr_weights_the_same_clean_image_prediction_the_same_whatever_the_parameterization(transform):
    """Min-SNR-gamma weights the x_0 loss by min(SNR, gamma), the
    reference's START_X case (TiankaiHang/Min-SNR-Diffusion-Training
    @5189997, guided_diffusion/gaussian_diffusion.py:881-884). The EDM
    preconditioning computes its loss on the x_0 it reads out of the model,
    as a direct x_0 prediction does, so two parameterizations whose outputs
    read out the same clean image take the same weighted loss."""
    schedule = KarrasVENoiseScheduler(sigma_data=0.5)
    process = Process(schedule, transform, weighting=MinSNR(5.0))
    x_0, epsilon = (jax.random.normal(key, (4, 3, 3, 1)) for key in jax.random.split(jax.random.key(0)))
    estimate = x_0 + 0.1 * jax.random.normal(jax.random.key(1), x_0.shape)
    rates = broadcast_rates(schedule, CONTINUOUS_STEPS, x_0)
    x_t, _, target = transform.forward_diffusion(x_0, epsilon, rates)
    # The raw output whose read-out is `estimate`: x_0 = c_skip x_t + c_out F.
    sigma = rates[1]
    c_out = sigma * 0.5 / jnp.sqrt(0.25 + sigma ** 2)
    c_skip = 0.25 / (0.25 + sigma ** 2)
    raw = ((estimate - c_skip * x_t) / c_out * (-1 if getattr(transform, "velocity", False) else 1)
           if isinstance(transform, KarrasPredictionTransform) else estimate)
    read = transform.pred_transform(x_t, raw, rates, CONTINUOUS_STEPS)
    error = jnp.mean(jnp.square(read - target), axis=(1, 2, 3))
    snr = schedule.snr(CONTINUOUS_STEPS)
    np.testing.assert_allclose(process.weight(CONTINUOUS_STEPS), jnp.minimum(snr, 5.0), rtol=1e-5)
    np.testing.assert_allclose(process.weight(CONTINUOUS_STEPS) * error,
                               jnp.minimum(snr, 5.0) * jnp.mean(jnp.square(estimate - x_0), axis=(1, 2, 3)),
                               rtol=1e-4)


def test_min_snr_weights_a_flow_velocity_as_the_clean_image_it_reads_out():
    """Min-SNR-gamma's START_X weight, min(SNR, gamma) on the x_0 loss, on
    the rectified-flow path x_t = (1 - t) x_0 + t eps, where SNR is
    ((1 - t) / t)^2 and a velocity v reads out x_0 = x_t - t v: the weighted
    velocity loss of a model whose read-out is `estimate` is min(SNR, gamma)
    times the clean image's squared error, as a direct x_0 prediction of
    the same image on the same path is weighted."""
    schedule = FlowMatchingScheduler()
    t = CONTINUOUS_STEPS
    x_0, epsilon = (jax.random.normal(key, (4, 3, 3, 1)) for key in jax.random.split(jax.random.key(2)))
    estimate = x_0 + 0.1 * jax.random.normal(jax.random.key(3), x_0.shape)
    path_t = np.asarray(t, np.float64)
    snr = ((1 - path_t) / path_t) ** 2
    want = np.minimum(snr, 5.0) * np.mean(np.square(np.asarray(estimate - x_0, np.float64)), axis=(1, 2, 3))
    rates = broadcast_rates(schedule, t, x_0)
    for transform in (FlowMatchPredictionTransform(), DirectPredictionTransform()):
        process = Process(schedule, transform, weighting=MinSNR(5.0))
        x_t, _, target = transform.forward_diffusion(x_0, epsilon, rates)
        velocity = isinstance(transform, FlowMatchPredictionTransform)
        raw = (x_t - estimate) / expand(t, x_0) if velocity else estimate
        read = transform.pred_transform(x_t, raw, rates, t)
        error = jnp.mean(jnp.square(read - target), axis=(1, 2, 3))
        np.testing.assert_allclose(process.weight(t) * error, want, rtol=2e-5,
                                   err_msg=type(transform).__name__)


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
    (presets.EDM, jnp.ones_like, lambda snr: snr + 4),
    (presets.Karras, jnp.ones_like, lambda snr: snr + 4),
    (presets.Flow, lambda snr: (1 + jnp.sqrt(snr)) ** 2, jnp.ones_like),
    (presets.Sqrt, jnp.ones_like, jnp.ones_like),
], ids=["cosine", "edm", "karras", "flow", "sqrt"])
def test_preset_weights_the_training_schedule_with_min_snr(preset, scale, ordinary):
    """The capped x_0 loss is converted to the space each preset computes its
    loss in; the EDM preconditioning computes it on x_0, so it takes the cap
    as is.

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
    preset = presets.Flow(shift=3.0, logit_mean=0.5, logit_std=0.7)
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
