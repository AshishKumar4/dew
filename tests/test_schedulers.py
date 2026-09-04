"""Invariant tests for the noise schedulers.

These encode the properties the rest of the library relies on:
variance preservation for VP schedules, alpha=1 for the generalized (VE)
schedules, monotone SNR along the trajectory, rates that broadcast against
image and video batches, and exact invertibility of the forward diffusion.

SCHEDULES below is the single table every shared invariant runs over, and
test_every_exported_scheduler_is_covered fails if a scheduler is exported
without being added to it, so new schedulers inherit the invariants.
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

import dew.diffusion.schedules as schedulers
from dew.diffusion import (
    EpsilonPredictionTransform, KarrasPredictionTransform, MinSNR, Process, ScheduleWeighting,
    VPredictionTransform, broadcast_rates, expand, presets,
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

# (class, factory, probe steps, family, a keyword of the constructor); family
# picks the extra rate identity: 'vp' is variance preserving, 've' keeps
# alpha=1 and scales the input, 'flow' is the rectified-flow linear path.
SCHEDULES = [
    (CosineNoiseScheduler, partial(CosineNoiseScheduler, 1000), DISCRETE_STEPS, 'vp', 'beta_end'),
    (LinearNoiseScheduler, partial(LinearNoiseScheduler, 1000), DISCRETE_STEPS, 'vp', 'beta_end'),
    (ExpNoiseScheduler, partial(ExpNoiseScheduler, 1000), DISCRETE_STEPS, 'vp', 'beta_end'),
    (CosineContinuousNoiseScheduler, CosineContinuousNoiseScheduler, CONTINUOUS_STEPS, 'vp', None),
    (SqrtContinuousNoiseScheduler, SqrtContinuousNoiseScheduler, CONTINUOUS_STEPS, 'vp', None),
    (CosineGeneralNoiseScheduler, CosineGeneralNoiseScheduler, CONTINUOUS_STEPS, 've', 'sigma_data'),
    (KarrasVENoiseScheduler, partial(KarrasVENoiseScheduler, sigma_max=80, rho=7, sigma_data=0.5), CONTINUOUS_STEPS, 've', 'sigma_data'),
    (EDMNoiseScheduler, partial(EDMNoiseScheduler, sigma_max=80, sigma_data=0.5), CONTINUOUS_STEPS, 've', 'sigma_data'),
    (FlowMatchingScheduler, FlowMatchingScheduler, CONTINUOUS_STEPS, 'flow', 'shift'),
]

# Base classes: no rates of their own, so nothing to assert invariants against
ABSTRACT_SCHEDULERS = {
    'NoiseScheduler',
    'GeneralizedNoiseScheduler',
    'DiscreteNoiseScheduler',
    'ContinuousNoiseScheduler',
}
# Exported helpers that are not schedulers
EXPORTED_HELPERS = {
    'expand',
    'compute_resolution_shift',
    'linear_beta_schedule',
    'cosine_beta_schedule',
    'exp_beta_schedule',
}

ALL_CASES = [case[:4] for case in SCHEDULES]
ALL_IDS = [cls.__name__ for cls, *_ in SCHEDULES]
VP_CASES = [case for case in ALL_CASES if case[3] == 'vp']
VP_IDS = [case[0].__name__ for case in VP_CASES]
VE_CASES = [case for case in ALL_CASES if case[3] == 've']
VE_IDS = [case[0].__name__ for case in VE_CASES]
KEYWORD_CASES = [(make, keyword) for _, make, _, _, keyword in SCHEDULES if keyword]
KEYWORD_IDS = [cls.__name__ for cls, _, _, _, keyword in SCHEDULES if keyword]


def test_every_exported_scheduler_is_covered():
    """Exporting a scheduler without adding it to SCHEDULES fails here."""
    exported = set(schedulers.__all__) - EXPORTED_HELPERS - ABSTRACT_SCHEDULERS
    assert exported == {cls.__name__ for cls, *_ in SCHEDULES}


def test_abstract_schedulers_stay_exported():
    """They are the documented subclassing surface."""
    assert ABSTRACT_SCHEDULERS <= set(schedulers.__all__)


@pytest.mark.parametrize("make,keyword", KEYWORD_CASES, ids=KEYWORD_IDS)
def test_misspelled_keyword_is_rejected(make, keyword):
    """A typo in a keyword must raise at construction. Swallowing it means the
    run silently trains with the default."""
    typo = keyword[:-1] + "_"
    with pytest.raises(TypeError, match=typo):
        make(**{typo: 5.0})


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
    x = jnp.zeros((len(steps),) + sample_shape)
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
    full_shape = (len(steps),) + sample_shape
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
    the noisiest entry rather than whatever an out-of-range gather returns."""
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
    """No epsilon guard: the old one halved the weight at sigma_min, where
    (sigma sigma_data)^2 is 1e-6."""
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


def test_edm_defaults_to_edm2_distribution():
    schedule = EDMNoiseScheduler()
    assert (schedule.P_mean, schedule.P_std) == (-0.4, 1.0)


def test_discrete_p2_default_makes_the_v_loss_an_x0_loss():
    """The P2 weight at k = 1, gamma = 1 is 1 / (1 + SNR), and the v error is
    (1 + SNR) times the x_0 error, so their product is the unweighted x_0 loss."""
    schedule = CosineNoiseScheduler(1000)
    snr = schedule.snr(DISCRETE_STEPS)
    assert jnp.allclose(schedule.weight(DISCRETE_STEPS) * VPredictionTransform().target_error_scale(snr),
                        1.0, rtol=1e-4)


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
    assert process.weighting == ScheduleWeighting()
    assert jnp.allclose(process.weight(MIN_SNR_STEPS), process.schedule.weight(MIN_SNR_STEPS))


############################################################################################################
# Presets
############################################################################################################

@pytest.mark.parametrize("preset", [presets.Cosine, presets.EDM, presets.Karras, presets.Flow, presets.Sqrt],
                         ids=lambda cls: cls.__name__)
def test_preset_weights_the_training_schedule_with_min_snr(preset):
    """min_snr_gamma on a preset is the MinSNR weighting of its process; left
    unset, the process keeps the schedule's own weight."""
    assert preset(min_snr_gamma=5.0)().weighting == MinSNR(5.0)
    assert preset()().weighting == ScheduleWeighting()


def test_edm_preset_samples_on_the_karras_grid():
    """Training draws log-normal sigmas; inference walks the rho-spaced grid
    with the same sigma range and sigma_data."""
    process = presets.EDM(sigma_min=0.01, sigma_max=40.0, rho=5.0, sigma_data=0.7)()
    assert isinstance(process.schedule, EDMNoiseScheduler)
    assert isinstance(process.sampler_schedule, KarrasVENoiseScheduler)
    assert (process.sampler_schedule.sigma_min, process.sampler_schedule.sigma_max,
            process.sampler_schedule.rho, process.sampler_schedule.sigma_data) == (0.01, 40.0, 5.0, 0.7)
    assert process.prediction.sigma_data == 0.7


def test_presets_rebuild_from_their_fields():
    """What the manifest stores is the preset's fields; building the registry
    member from them is the same process."""
    import dataclasses
    from dew.registry import presets as registry
    preset = registry.Flow(shift=3.0, logit_mean=0.5)
    rebuilt = registry.build("flow", **dataclasses.asdict(preset))
    assert rebuilt == preset
    assert rebuilt().schedule.shift == 3.0
    with pytest.raises(ValueError, match="no field"):
        registry.build("flow", shfit=3.0)
