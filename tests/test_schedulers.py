"""Invariant tests for the noise schedulers.

These encode the properties the rest of the library relies on:
variance preservation for VP schedules, alpha=1 for the generalized (VE)
schedules, monotone SNR along the trajectory, rates that broadcast against
image and video batches, and invertibility of the forward diffusion. Each
rate, weight and round trip is on par with its closed form, or its authors'
code, from the float64 evaluation over drawn inputs (check 1,
tests/reference_error.py); a training draw's law is held by a
Kolmogorov-Smirnov test; what is exact is held exactly.
"""

import math
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact, declared_seeds, indexed_recording, recording
from scipy import stats

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
from tools.weighting_reference import RECORDED, TABLES, closed_form, label_of, table_inputs

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
    (ExpNoiseScheduler, partial(ExpNoiseScheduler, 1000), DISCRETE_STEPS, "vp"),
    (CosineContinuousNoiseScheduler, CosineContinuousNoiseScheduler, CONTINUOUS_STEPS, "vp"),
    (CosineGeneralNoiseScheduler, CosineGeneralNoiseScheduler, CONTINUOUS_STEPS, "ve"),
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
BETAS = dict(np.load(Path(__file__).parent / "fixtures" / "schedules" / "betas.npz"))
SEEDS = {'invertible': 64, 'diffusion-lm sqrt': 128, 'clean image weighting': 256,
         'flow clean image weighting': 32, 'karras grid': 32, 'edm weight': 8, 'edm time': 8, 'flow rates': 8}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada)."""
ROWS = 64
"""A drawn batch's rows: enough that a few seeds resolve a site."""


def drawn(seed: int, site: int):
    """A seed's generator for the site numbered `site`, the same on every run."""
    return np.random.default_rng([0x5C, site, seed])


def assert_follows(draws, law):
    """Training draws follow `law` (a scipy distribution): over 2^18 or more draws, the one-sample
    Kolmogorov-Smirnov test does not reject at one in a billion. Its critical distance there is 0.0064,
    so a normal's deviation off by 0.05 at 1.2 (a distance of 0.0099) fails it all but surely."""
    draws = np.asarray(draws, np.float64)
    assert draws.size >= 1 << 18
    assert stats.kstest(draws, law.cdf).pvalue > 1e-9


@pytest.mark.parametrize("cls,make,steps,family", ALL_CASES, ids=ALL_IDS)
@pytest.mark.parametrize("sample_shape", [(8, 8, 3), (2, 8, 8, 3)], ids=['image', 'video'])
def test_forward_diffusion_invertible(cls, make, steps, family, sample_shape):
    """The noised sample gives x_0 back at the same timestep, for the epsilon
    parameterization on every schedule: the round trip on par with the same
    arithmetic in NumPy from x_0 itself over drawn samples, noise and times
    in the probe steps' range (check 1)."""
    schedule, transform = make(), EpsilonPredictionTransform()
    low, high = float(steps.min()), float(steps.max())

    def draw(seed):
        rng = drawn(seed, 0)
        times = (rng.integers(int(low), int(high) + 1, 16) if jnp.issubdtype(steps.dtype, jnp.integer)
                 else rng.uniform(low, high, 16).astype(np.float32))
        x0, noise = rng.standard_normal((2, 16, *sample_shape)).astype(np.float32)
        alpha, sigma = (np.asarray(rate) for rate in broadcast_rates(schedule, times, x0))
        return x0, noise, alpha, sigma

    def round_trip(x0, noise, alpha, sigma):
        xt, _, target = transform.forward_diffusion(x0, noise, (alpha, sigma))
        assert xt.shape == x0.shape
        return transform.backward_diffusion(xt, target, (alpha, sigma))[0]

    def written(x0, noise, alpha, sigma):
        return (alpha * x0 + sigma * noise - noise * sigma) / alpha

    assert_as_exact(jax.jit(round_trip), as_given(written), draw, "invertible", seeds=SEEDS['invertible'],
                    truth_fn=lambda x0, noise, alpha, sigma: x0)


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


@pytest.mark.parametrize("steps", [2000, 1000])
def test_sqrt_schedule_is_diffusion_lms_table(steps):
    """Diffusion-LM's own `get_named_beta_schedule("sqrt", T)` (Li et al.
    2022, XiangLi1999/Diffusion-LM at 759889d, run by
    tools/beta_schedule_reference.py): at its step k the cumulative alpha is
    Dew's alpha^2 at t = (k + 1) / T. Over drawn steps, Dew's rates at that t
    rounded to fp32 are on par with Diffusion-LM's own alpha_bar,
    1 - sqrt(t + 1e-4), at the same fp32 t in NumPy, normalized as its table
    telescopes, from the table itself, the step's exact rates (check 1). The
    last step is excluded: Diffusion-LM clips its beta at 0.999 where the
    cumulative alpha crosses zero, and Dew stops at (0, 1). The paper trains
    the plain x_0 loss: the weight is one."""
    table = np.cumprod(1 - BETAS[f"diffusion_lm_sqrt_{steps}"])
    schedule = SqrtContinuousNoiseScheduler()

    def time(step):
        # Step k's time (k + 1) / T, rounded once to the arms' float32.
        return ((step + 1) / steps).astype(np.float32)

    @as_given
    def theirs(step):
        def alpha_bar(t):
            return 1 - np.sqrt(t + 0.0001)
        cumulative = alpha_bar(time(step)) / alpha_bar(np.float32(0))
        return {"alpha": np.sqrt(cumulative), "sigma": np.sqrt(1 - cumulative)}

    def tabled(step):
        return {"alpha": np.sqrt(table[step]), "sigma": np.sqrt(1 - table[step])}

    def rates(step):
        return dict(zip(("alpha", "sigma"), schedule.rates(time(step)), strict=True))

    assert_as_exact(rates, theirs, lambda seed: (drawn(seed, 1).integers(0, steps - 1, ROWS),),
                    "diffusion-lm sqrt", seeds=SEEDS['diffusion-lm sqrt'], truth_fn=tabled)
    assert schedule.rates(jnp.asarray(1.0)) == (0.0, 1.0)
    np.testing.assert_array_equal(schedule.weight(CONTINUOUS_STEPS), np.ones(4, np.float32))


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


def assert_weights_are_the_authors(name, mean, options, table, process_of):
    """Check 1: Dew's weight at every step of a drawn beta table (`process_of(betas)`, a Process) on par
    with its authors' own `training_losses` in fp32, recorded over the seeds the recording declares
    (tools/weighting_reference.py), from their float64 run, which the authors' formula in float64
    recomputes (`closed_form`)."""
    label = label_of(name, mean, options, table)
    seeds = declared_seeds(RECORDED)[label]
    draw, recorded, truth = indexed_recording(
        recording(RECORDED), label, seeds, inputs_of=partial(table_inputs, table),
        truth_fn=jax.jit(partial(closed_form, name, mean, options, dtype=jnp.float64)))

    @as_given
    def weighted(index, inputs):
        process = process_of(np.asarray(inputs["betas"]))
        return {"weight": process.weight(jnp.arange(process.schedule.T))}

    assert_as_exact(weighted, recorded, draw, label, seeds=seeds, truth_fn=truth)


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("k,gamma", [(1, 1), (1, 0.5), (2, 1)])
def test_p2_weights_are_the_authors(k, gamma, table):
    """P2 (Choi et al. 2022) as jychoi118/P2-weighting@3da0947 computes it:
    its own `training_losses` on an epsilon model at every step of drawn
    tables (tools/weighting_reference.py), against the discrete table's
    weight (check 1)."""
    def process_of(betas):
        schedule = schedulers.DiscreteNoiseScheduler(betas, p2_loss_weight_k=k, p2_loss_weight_gamma=gamma)
        return Process(schedule, EpsilonPredictionTransform())

    assert_weights_are_the_authors("p2", "EPSILON", {"p2_k": k, "p2_gamma": gamma}, table, process_of)


############################################################################################################
# min-SNR-gamma loss weighting (Hang et al. 2023), through Process
############################################################################################################

@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("transform,mean,weighting", [
    (EpsilonPredictionTransform(), "EPSILON", "min_snr_5"),
    (VPredictionTransform(), "VELOCITY", "vmin_snr_5"),
    (DirectPredictionTransform(), "START_X", "min_snr_5"),
], ids=["epsilon", "v", "x0"])
def test_min_snr_weights_are_the_authors(transform, mean, weighting, table):
    """min-SNR-5 as TiankaiHang/Min-SNR-Diffusion-Training@5189997 computes
    it, per parameterization (`min_snr_5` on epsilon and x_0, `vmin_snr_5` on
    v): its own `training_losses` at every step of drawn tables
    (tools/weighting_reference.py), against Dew's (check 1). On a
    zero-terminal table the last step has zero SNR, which the authors weight
    as one."""
    def process_of(betas):
        return Process(schedulers.DiscreteNoiseScheduler(betas), transform, weighting=MinSNR(5.0))

    assert_weights_are_the_authors("min_snr", mean, {"mse_loss_weight_type": weighting}, table, process_of)


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
    read out the same clean image take the same weighted loss: the weight
    and the weighted loss of a raw output reading out a drawn estimate are on
    par with min(SNR, 5) and the x_0 loss of that output's read-out in NumPy,
    Eq. 5's sigma at the drawn times, from their float64 evaluation (check 1)."""
    schedule = KarrasVENoiseScheduler(sigma_data=0.5)
    process = Process(schedule, transform, weighting=MinSNR(5.0))
    sign = -1 if getattr(transform, "velocity", False) else 1

    def karras_sigma(t):
        ramp = 1 - t
        return (80 ** (1 / 7) + ramp * (0.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7

    def draw(seed):
        rng = drawn(seed, 2)
        t = rng.uniform(0.05, 0.95, ROWS).astype(np.float32)
        x_0, epsilon, offset = rng.standard_normal((3, ROWS, 3, 3, 1))
        # The raw output whose read-out is a drawn estimate: x_0 = c_skip x_t + c_out F, in float64,
        # held at float32 as a model's output is.
        sigma = karras_sigma(t.astype(np.float64))[:, None, None, None]
        x_t, estimate = x_0 + sigma * epsilon, x_0 + 0.1 * offset
        c_out, c_skip = sigma * 0.5 / np.sqrt(0.25 + sigma**2), 0.25 / (0.25 + sigma**2)
        preconditioned = isinstance(transform, KarrasPredictionTransform)
        raw = (estimate - c_skip * x_t) / c_out * sign if preconditioned else estimate
        return t, *(value.astype(np.float32) for value in (x_0, epsilon, raw))

    def weighted(t, x_0, epsilon, raw):
        rates = broadcast_rates(schedule, t, x_0)
        x_t, _, target = transform.forward_diffusion(x_0, epsilon, rates)
        error = jnp.mean(jnp.square(transform.pred_transform(x_t, raw, rates, t) - target), axis=(1, 2, 3))
        weight = process.weight(t)
        return {"weight": weight, "loss": weight * error}

    @as_given
    def clean(t, x_0, epsilon, raw):
        sigma = karras_sigma(t)[:, None, None, None]
        x_t = x_0 + sigma * epsilon
        if isinstance(transform, KarrasPredictionTransform):
            raw = sign * (sigma * 0.5 / np.sqrt(0.25 + sigma**2)) * raw + 0.25 / (0.25 + sigma**2) * x_t
        weight = np.minimum(1 / sigma**2, 5.0).reshape(-1)
        return {"weight": weight, "loss": weight * np.mean(np.square(raw - x_0), axis=(1, 2, 3))}

    assert_as_exact(jax.jit(weighted), clean, draw, "clean image weighting",
                    seeds=SEEDS['clean image weighting'])


def test_min_snr_weights_a_flow_velocity_as_the_clean_image_it_reads_out():
    """Min-SNR-gamma's START_X weight, min(SNR, gamma) on the x_0 loss, on
    the rectified-flow path x_t = (1 - t) x_0 + t eps, where SNR is
    ((1 - t) / t)^2 and a velocity v reads out x_0 = x_t - t v: the weighted
    velocity loss of a model whose read-out is `estimate` is min(SNR, gamma)
    times the clean image's squared error, as a direct x_0 prediction of
    the same image on the same path is weighted: each parameterization's
    weighted loss of a raw output reading out a drawn estimate on par with
    min(SNR, 5) times the squared error of that output's read-out in NumPy,
    from its float64 evaluation over drawn times and samples (check 1)."""
    schedule = FlowMatchingScheduler()

    def draw(seed):
        rng = drawn(seed, 3)
        t = rng.uniform(0.05, 0.95, ROWS).astype(np.float32)
        x_0, epsilon, offset = rng.standard_normal((3, ROWS, 3, 3, 1))
        path = t.astype(np.float64)[:, None, None, None]
        estimate = x_0 + 0.1 * offset
        # The velocity whose read-out x_t - t v is the estimate, in float64, held at float32 as a
        # model's output is; the direct prediction's raw output is the estimate itself.
        velocity = ((1 - path) * x_0 + path * epsilon - estimate) / path
        return t, *(value.astype(np.float32) for value in (x_0, epsilon, velocity, estimate))

    def weighted(t, x_0, epsilon, velocity, estimate):
        rates = broadcast_rates(schedule, t, x_0)
        losses = {}
        for transform, raw in ((FlowMatchPredictionTransform(), velocity),
                               (DirectPredictionTransform(), estimate)):
            x_t, _, target = transform.forward_diffusion(x_0, epsilon, rates)
            read = transform.pred_transform(x_t, raw, rates, t)
            weight = Process(schedule, transform, weighting=MinSNR(5.0)).weight(t)
            losses[type(transform).__name__] = weight * jnp.mean(jnp.square(read - target), axis=(1, 2, 3))
        return losses

    @as_given
    def clean(t, x_0, epsilon, velocity, estimate):
        path = t[:, None, None, None]
        weight = np.minimum(((1 - t) / t) ** 2, 5.0)
        read = (1 - path) * x_0 + path * epsilon - path * velocity
        return {"FlowMatchPredictionTransform": weight * np.mean(np.square(read - x_0), axis=(1, 2, 3)),
                "DirectPredictionTransform": weight * np.mean(np.square(estimate - x_0), axis=(1, 2, 3))}

    assert_as_exact(jax.jit(weighted), clean, draw, "flow clean image weighting",
                    seeds=SEEDS['flow clean image weighting'])


############################################################################################################
# Presets
############################################################################################################

@pytest.mark.parametrize("fields, mean, std", [
    ({"regime": "pixel"}, -1.2, 1.2),
    ({"regime": "latent"}, -0.4, 1.0),
    ({"regime": "pixel", "P_mean": 0.2}, 0.2, 1.2),
    ({"regime": "latent", "P_std": 0.7}, -0.4, 0.7),
    ({"regime": "pixel", "P_mean": -0.8, "P_std": 0.7}, -0.8, 0.7),
], ids=["pixel", "latent", "mean_override", "std_override", "both_override"])
def test_edm_draws_the_sigmas_of_the_space_it_denoises(rng, fields, mean, std):
    """The log of a training sigma is normal at the preset's mean and deviation (`assert_follows`)."""
    schedule = presets.EDM(**fields)().schedule
    assert_follows(jnp.log(schedule.sigmas(schedule.sample_t(rng, 1 << 18))), stats.norm(mean, std))


def test_a_run_config_draws_pixel_sigmas_without_an_autoencoder_and_latent_ones_with(rng):
    from dew.config.diffusion import DiffusionRunConfig
    from dew.nn.autoencoders.pretrained import PretrainedAutoencoder

    for config, mean, std in (
        (DiffusionRunConfig(), -1.2, 1.2),
        (DiffusionRunConfig(autoencoder=PretrainedAutoencoder()), -0.4, 1.0),
        (DiffusionRunConfig(preset=presets.EDM(P_mean=-0.8, P_std=0.7)), -0.8, 0.7),
    ):
        schedule = config.preset().schedule
        assert_follows(jnp.log(schedule.sigmas(schedule.sample_t(rng, 1 << 18))), stats.norm(mean, std))


def test_edm_preset_samples_on_the_karras_grid():
    """Training draws log-normal sigmas; inference walks the rho-spaced grid
    with the same sigma range and sigma_data. Over drawn times and sigmas, on
    par with each closed form in NumPy from its float64 evaluation (check 1):
    the sampling grid with Eq. 5 at rho 5 between the preset's sigma range,
    its ends included; the training weight, EDM's lambda at sigma_data 0.7,
    and the input scale with the pixel regime's sigma at the training times of sigmas in that range;
    and the time of a sigma with that sigma's inverse."""
    process = presets.EDM(sigma_min=0.01, sigma_max=40.0, rho=5.0, sigma_data=0.7, regime="pixel")()

    def times(seed):
        rng = drawn(seed, 4)
        return (np.concatenate([[0.0, 1.0], rng.uniform(0, 1, ROWS - 2)]).astype(np.float32),)

    def karras(t):
        return (40 ** 0.2 + (1 - t) * (0.01 ** 0.2 - 40 ** 0.2)) ** 5

    assert_as_exact(process.sampler_schedule.sigmas, as_given(karras), times, "karras grid",
                    seeds=SEEDS['karras grid'])

    def weighted(t):
        return {"weight": process.weight(t),
                "scale": process.prediction.get_input_scale(process.schedule.rates(t))}

    @as_given
    def edm(t):
        sigma = np.exp(t * 1.2 + -1.2)
        return {"weight": 1 / 0.7**2 + 1 / sigma**2, "scale": 1 / np.sqrt(0.7**2 + sigma**2)}

    def training_times(seed):
        # The times of sigmas in the preset's range, uniform in log sigma: a normal draw's far tail
        # makes lambda's error one value's, which no count of seeds resolves.
        low, high = ((np.log(sigma) + 1.2) / 1.2 for sigma in (0.01, 40.0))
        return (drawn(seed, 5).uniform(low, high, (ROWS, ROWS)).astype(np.float32),)

    assert_as_exact(weighted, edm, training_times, "edm weight", seeds=SEEDS['edm weight'])

    def sigmas(seed):
        return (np.exp(drawn(seed, 6).uniform(np.log(0.01), np.log(40.0), ROWS)).astype(np.float32),)

    assert_as_exact(process.schedule.t_of_sigma, as_given(lambda sigma: (np.log(sigma) + 1.2) / 1.2), sigmas,
                    "edm time", seeds=SEEDS['edm time'])


def test_a_dynamic_shift_keeps_its_ends_and_their_derivatives_and_a_static_file_maps_statically():
    """A resolution's shift maps as Diffusers' `time_shift` does and keeps its ends: 0 and 1 map to
    themselves, and its derivatives there are finite, shift at 0 to the bit, where the reciprocal form
    would multiply zero by infinity. A scheduler file without dynamic shifting maps as the static
    preset does, to the bit, and one with it as the resolution's preset does."""
    from dew.diffusion.presets import Flow, ResolutionShift
    from dew.diffusion.schedules.source import SourceSchedule

    resolution = ResolutionShift(tokens=1024)
    schedule = Flow(resolution_shift=resolution)().schedule
    assert schedule.dynamic
    ends = jnp.asarray([0.0, 1.0])
    np.testing.assert_array_equal(schedule.rates(ends)[1], ends)

    def sigma(t):
        return schedule.rates(t)[1]

    first = jax.vmap(jax.grad(sigma))(ends)
    second = jax.vmap(jax.grad(jax.grad(sigma)))(ends)
    assert np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    assert first[0] == np.float32(schedule.shift)
    grid = jnp.linspace(0.0, 1.0, 101)
    static = SourceSchedule.from_config({"_class_name": "FlowMatchEulerDiscreteScheduler",
                                         "num_train_timesteps": 1000, "shift": 3.0,
                                         "use_dynamic_shifting": False}).training_process().schedule
    assert not static.dynamic
    np.testing.assert_array_equal(static.rates(grid)[1], Flow(shift=3.0)().schedule.rates(grid)[1])
    dynamic = SourceSchedule.from_config({"_class_name": "FlowMatchEulerDiscreteScheduler",
                                          "num_train_timesteps": 1000, "use_dynamic_shifting": True,
                                          "base_shift": 0.5, "max_shift": 1.15, "base_image_seq_len": 256,
                                          "max_image_seq_len": 4096}).training_process(1024).schedule
    np.testing.assert_array_equal(dynamic.rates(grid)[1], schedule.rates(grid)[1])


def test_presets_rebuild_from_their_fields(rng):
    """Recorded flow fields control the rebuilt rates, on par with the shift's closed form in NumPy
    from its float64 evaluation over drawn times (check 1), and the training draws, whose logits are
    normal at the recorded mean and deviation (`assert_follows`)."""
    import dataclasses

    from dew.registry import presets as registry
    preset = presets.Flow(shift=3.0, logit_mean=0.5, logit_std=0.7)
    process = registry.build("flow", **dataclasses.asdict(preset))()
    def shifted(t):
        sigma = 3 * t / (1 + 2 * t)
        return {"alpha": 1 - sigma, "sigma": sigma}

    assert_as_exact(lambda t: dict(zip(("alpha", "sigma"), process.schedule.rates(t), strict=True)),
                    as_given(shifted),
                    lambda seed: (drawn(seed, 7).uniform(0, 1, ROWS).astype(np.float32),), "flow rates",
                    seeds=SEEDS['flow rates'])
    draws = process.schedule.sample_t(rng, 1 << 18)
    assert_follows(jnp.log(draws / (1 - draws)), stats.norm(0.5, 0.7))
    with pytest.raises(ValueError):
        registry.build("flow", shfit=3.0)


@pytest.mark.parametrize("schedule", [EDMNoiseScheduler(), KarrasVENoiseScheduler(),
                                      CosineGeneralNoiseScheduler(kappa=0.5)],
                         ids=["edm", "karras", "flaxdiff-tangent"])
def test_a_sigma_schedule_keeps_a_float64_runs_times_in_float64(schedule):
    """A float64 run's times stay float64 through a Karras schedule's sigmas and back, and FlaxDiff's
    tangent grid's (`schedules.common.times`), as every other schedule keeps them: a float64 truth's
    noise rates, loss weights and conditioning times are then float64's, not float32's widened."""
    with jax.enable_x64(new_val=True):
        t = np.linspace(0.05, 0.95, 64) * schedule.T
        sigma = schedule.sigmas(jnp.asarray(t, jnp.float64))
        assert sigma.dtype == jnp.float64
        assert schedule.t_of_sigma(sigma).dtype == jnp.float64
        assert schedule.weight(jnp.asarray(t, jnp.float64)).dtype == jnp.float64
        # Computed in float64, not in float32 and widened.
        narrow = schedule.sigmas(jnp.asarray(t, jnp.float32))
        assert narrow.dtype == jnp.float32
        assert not np.array_equal(np.asarray(sigma), np.asarray(narrow, np.float64))


T_OF_SIGMA_SEEDS = 8
"""Check 1's declared seeds for the EDM schedule's time of a sigma, sized by `seeds_for`."""


def test_the_edm_schedules_time_of_a_sigma_is_on_par_with_its_correctly_rounded_quotient():
    """An EDM1 schedule's time of a sigma, the centred log-sigma over P_std = 1.2, is on par with the
    same quotient correctly rounded in float32 (check 1, float64 the truth), however XLA lowers its
    division by the constant."""
    from reference_error import assert_as_exact

    schedule = EDMNoiseScheduler(P_mean=-1.2, P_std=1.2)
    t_of_sigma = jax.jit(schedule.t_of_sigma)

    def draw(seed):
        return (np.exp(np.random.default_rng(seed).normal(-1.2, 1.2, 4096)).astype(np.float32),)

    @as_given
    def reference(sigma):
        centred = np.log(sigma.astype(np.float64)).astype(sigma.dtype) - sigma.dtype.type(-1.2)
        return centred / sigma.dtype.type(1.2)

    assert_as_exact(t_of_sigma, reference, draw, "EDM t_of_sigma", seeds=T_OF_SIGMA_SEEDS)



# --------------------------------------------------------------- FlaxDiff's schedules


FLAXDIFF = dict(np.load(Path(__file__).parent / "fixtures" / "flaxdiff_schedules.npz"))
"""FlaxDiff's own tables, rates, sigmas and weights, in float32 (tools/flaxdiff_schedules_reference.py)."""


def flaxdiff_case(case: str) -> dict:
    """A case of the FlaxDiff fixture, its arguments as the tool built it."""
    import json

    meta = json.loads(bytes(FLAXDIFF["meta"]))
    return {**meta["exp"], **meta["general"]}[case]


FLAXDIFF_SEEDS = Path(__file__).parent / "fixtures" / "flaxdiff_schedules_seeds.npz"
FLAXDIFF_SEEDED = recording(FLAXDIFF_SEEDS)
"""Each FlaxDiff site over check 1's seeds: FlaxDiff's float32 outputs and their float64 truth
(`tools/flaxdiff_schedules_reference.py seeds`)."""


def assert_on_par_with_flaxdiff(site: str, dew, *, summarized: bool = False) -> None:
    """Check 1 of `dew(inputs)` against FlaxDiff's recorded outputs at `site`, each leaf on its own,
    over the seeds its recording declares; or, `summarized`, against its recorded evidence alone, the
    times drawn again (`flaxdiff_times`) and the truth Dew's own float64 path, held to the recorded
    truth's sketch."""
    from diffusion_support import flaxdiff_times

    seeds = declared_seeds(FLAXDIFF_SEEDS)[site]
    again = {"inputs_of": flaxdiff_times, "truth_fn": dew} if summarized else {}
    draw, recorded, truth = indexed_recording(FLAXDIFF_SEEDED, site, seeds, **again)
    assert_as_exact(lambda index, inputs: dew(inputs), recorded, draw, f"FlaxDiff {site}", seeds=seeds,
                    truth_fn=truth)


@pytest.mark.parametrize("case", ["exp/default", "exp/clipped"])
def test_the_exponential_table_is_flaxdiffs_with_its_rates_and_weights(case):
    """FlaxDiff's exponential beta table, bit for bit as its arithmetic makes it (`exp_beta_schedule`,
    clipped at the schedule's own beta_end, not Diffusers' fixed 0.999), and the rates and P2 weights
    Dew tabulates from it, each on par with FlaxDiff's, which accumulates its table in float32, at
    the steps each seed draws."""
    arguments = flaxdiff_case(case)
    schedule = ExpNoiseScheduler(**arguments)
    np.testing.assert_array_equal(np.asarray(schedule.betas, np.float32), FLAXDIFF[f"{case}/betas"])
    names = ("sqrt_alpha_cumprod", "sqrt_one_minus_alpha_cumprod", "p2_loss_weights")
    tables = {name: jnp.asarray(getattr(schedule, name)) for name in names}
    assert_on_par_with_flaxdiff(case, lambda inputs: {name: table[inputs["index"]]
                                                      for name, table in tables.items()})


def test_flaxdiffs_continuous_cosine_keeps_a_float64_runs_times_in_float64():
    """FlaxDiff's continuous cosine computes a float64 run's rates in float64, as FlaxDiff's own follow
    their times' dtype, not in float32 widened."""
    schedule = CosineContinuousNoiseScheduler()
    with jax.enable_x64(new_val=True):
        t = np.linspace(0.05, 0.95, 64)
        wide = schedule.rates(jnp.asarray(t, jnp.float64))
        narrow = schedule.rates(jnp.asarray(t, jnp.float32))
        assert {rate.dtype for rate in wide} == {jnp.dtype(jnp.float64)}
        assert not np.array_equal(np.asarray(wide[1]), np.asarray(narrow[1], np.float64))


def test_the_continuous_cosine_is_flaxdiffs_with_its_snr_weight():
    """FlaxDiff's continuous cosine, alpha = cos(pi t / 2) and sigma = sin(pi t / 2), weighted
    1 / (1 + SNR): the rates and the weight each on par with FlaxDiff's, and at the ends exactly
    float64's, no noise and a weight of 0 at t = 0, all noise and a weight of 1 at t = 1, as
    FlaxDiff's are."""
    schedule = CosineContinuousNoiseScheduler()

    def rates(inputs):
        alpha, sigma = schedule.rates(inputs["t"])
        return {"alpha": alpha, "sigma": sigma, "weight": schedule.weight(inputs["t"])}

    assert_on_par_with_flaxdiff("cosine_continuous", rates)
    ends = jnp.asarray([0.0, 1.0], jnp.float32)
    alpha, sigma = schedule.rates(ends)
    assert alpha[0] == 1 and sigma[0] == 0 and sigma[1] == 1
    np.testing.assert_array_equal(schedule.weight(ends), [0.0, 1.0])
    np.testing.assert_array_equal(FLAXDIFF["cosine_continuous/weight"][[0, -1]], [0.0, 1.0])


FLAXDIFF_GRIDS = ["cosine_general/default", "cosine_general/bounded", "cosine_general/steep"]


@pytest.mark.parametrize("case", FLAXDIFF_GRIDS)
def test_the_tangent_grid_is_flaxdiffs_with_its_weight_and_conditioning(case):
    """FlaxDiff's tangent grid of sigmas, from sigma_min / kappa^2 as its divides the tangent by kappa
    again, at its own bounds and at others: the sigmas and the weight of its `GeneralizedNoiseScheduler`,
    1 + sigma^2 / sigma_max^2, each on par with FlaxDiff's; alpha exactly 1; and the model conditioned
    on FlaxDiff's whole thousandths of t, exactly its own at 33 times and 4096 drawn, the same
    float32 product truncated as its int32 cast truncates it."""
    schedule = CosineGeneralNoiseScheduler(**flaxdiff_case(case))
    assert schedule.sigma_data == 1
    def grid(inputs):
        return {"sigma": schedule.sigmas(inputs["t"]), "weight": schedule.weight(inputs["t"])}

    assert_on_par_with_flaxdiff(case, grid, summarized=True)
    t = FLAXDIFF["times"]
    signal, _ = schedule.rates(t)
    np.testing.assert_array_equal(signal, FLAXDIFF[f"{case}/signal"])
    for times, name in ((t, "conditioning"), (FLAXDIFF["conditioning_times"], "conditioning_drawn")):
        np.testing.assert_array_equal(schedule.model_time(times), FLAXDIFF[f"{case}/{name}"])


TANGENT_INVERSE_SEEDS = {"cosine_general/default": 8, "cosine_general/bounded": 8, "cosine_general/steep": 8}
"""Check 1's declared seeds for each tangent grid's time of a sigma, sized by `seeds_for`."""


@pytest.mark.parametrize("case", FLAXDIFF_GRIDS)
def test_the_tangent_grids_time_of_a_sigma_is_on_par_with_its_formula_rounded_in_float32(case):
    """The tangent grid's inverse, (atan(kappa sigma) - theta_min) / (theta_max - theta_min), which
    FlaxDiff leaves unwritten, on par with the same formula in NumPy's float32 at sigmas drawn
    log-uniform over the grid (check 1, float64 the truth)."""
    schedule = CosineGeneralNoiseScheduler(**flaxdiff_case(case))
    t_of_sigma = jax.jit(schedule.t_of_sigma)
    low, high = math.log(schedule.sigma_min), math.log(schedule.sigma_max)

    def draw(seed):
        return (np.exp(np.random.default_rng(seed).uniform(low, high, 4096)).astype(np.float32),)

    @as_given
    def reference(sigma):
        kind = sigma.dtype.type
        return ((np.arctan(sigma * kind(schedule.kappa)) - kind(schedule.theta_min))
                / kind(schedule.theta_max - schedule.theta_min))

    assert_as_exact(t_of_sigma, reference, draw, f"FlaxDiff {case} t_of_sigma",
                    seeds=TANGENT_INVERSE_SEEDS[case])


@pytest.mark.parametrize("kappa", [0.5, 2.0])
def test_heuns_churn_stays_inside_the_tangent_grid_whatever_its_kappa(kappa):
    """The tangent grid of nominal bounds 0.02 and 80 ends at 80 / kappa^2 (`sigma_max`, 320 at kappa
    0.5 and 20 at kappa 2), the top a churn may raise a sigma to: a churn of the grid's top sigma is
    refused, and one of the sigma at t = 0.99, raised by a tenth to about 102 at kappa 0.5 (past the
    nominal 80) and 14 at kappa 2, runs."""
    from types import SimpleNamespace

    from dew.sampling import Heun

    schedule = CosineGeneralNoiseScheduler(kappa=kappa)
    assert schedule.nominal == (0.02, 80.0)
    process, x = SimpleNamespace(sampler_schedule=schedule), jnp.zeros((1, 4), jnp.float32)
    with pytest.raises(ValueError):
        Heun(s_churn=0.2).init(x, jnp.asarray([1.0, 0.5, 0.0]), process, key=jax.random.key(0))
    below = Heun(s_churn=0.2).init(x, jnp.asarray([0.99, 0.5, 0.0]), process, key=jax.random.key(0))
    assert float(below) == np.float32(0.1)
