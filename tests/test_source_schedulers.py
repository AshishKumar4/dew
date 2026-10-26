"""Published scheduler files, reconstructed by `SourceSchedule` from their own configs.

Each case's trajectory and gradient are held to the ones the source scheduler
class walks (tools/diffusers_source_reference.py), and a grid's own limits, a
row walked alone and a seeded Brownian path, to what the source does there.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffusion_support import SourceOracle, relative_gap, source_final, walk
from reference_error import (
    assert_as_exact,
    assert_as_exact_as_the_reference,
    assert_as_exact_over_orders,
    declared_seeds,
    distance,
    indexed_recording,
    recording,
    without_ynn,
)

from dew.sampling import DPMSolverSDE, sample

DIFFUSERS_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "diffusers"
SOURCE = json.loads((DIFFUSERS_FIXTURES / "source_schedulers.json").read_text())
SOURCE_ARRAYS = np.load(DIFFUSERS_FIXTURES / "source_schedulers.npz")
SOURCE_ROUNDING = np.load(DIFFUSERS_FIXTURES / "source_rounding.npz")
SOURCE_SEEDS = DIFFUSERS_FIXTURES / "source_seeds.npz"
SOURCE_SEEDED = recording(SOURCE_SEEDS)
"""The seeded pointwise cases' trajectory gradients over check 1's seeds (`tools/diffusers_source_reference.py
seeds`), each name's arrays under its own prefix."""


def source_case(name: str):
    """The reconstructed process, grid and solver of a fixture case, with the
    initial latent the reference walked from."""
    from dew.diffusion.schedules.source import SourceSchedule

    config = json.loads(str(SOURCE_ARRAYS[f"{name}.config"]))
    schedule = SourceSchedule.from_config(config)
    process, times = schedule.sampling(SOURCE["cases"][name]["steps"])
    return schedule, process, times, jnp.asarray(SOURCE_ARRAYS[f"{name}.x_T"])


def source_gradient_over_orders(final, initial, cotangent) -> None:
    """Pixel permutations move CFG's variance reductions alone.

    The pointwise oracle and the per-sample quantile commute with the same
    permutation of input and cotangent. The generating tool checks the
    float64 VJP's invariance and records each reference distance. A single
    draw concentrates on the few unclipped entries; 52 orders average the
    reference's reduction roundings before the factor-two comparison. The
    measured RMS ratios are 0.90/0.60/0.90 on default/AVX/Ice Lake CPU codegen.
    """
    gradient = jax.jit(lambda value, cot: jax.vjp(final, value)[1](cot)[0])
    truth = SOURCE_ARRAYS["cfg.epsilon.grad"]
    distances = []
    for order in SOURCE_ROUNDING["orders"]:
        def moved(value, order=order):
            return np.asarray(value).reshape(initial.shape[0], -1)[:, order].reshape(initial.shape)
        actual = gradient(jnp.asarray(moved(initial)), jnp.asarray(moved(cotangent)))
        distances.append(distance(actual, moved(truth)))
    assert_as_exact_over_orders(distances, SOURCE_ROUNDING["cfg.epsilon.distances"], "cfg.epsilon")


@pytest.mark.parametrize("name", sorted(name for name in SOURCE["cases"] if not name.startswith("dpm_sde.")))
def test_source_config_rebuilds_its_scheduler_trajectory_and_gradient(name):
    """Every published scheduler file the reference tool saved, rebuilt from
    that file alone: its beta table, the standard deviation its
    `init_noise_sigma` draws, the model time at each grid point, the latent
    after every interval and the vector-Jacobian product of the last one
    through the whole trajectory.

    The reference calls the actual scheduler class, saves its config before
    `set_timesteps` and walks every call a pipeline makes, so a spacing,
    sigma transformation, terminal sigma, clipping order or stage placement
    that this reconstruction read differently would move the trajectory.
    A stochastic class integrates the exact draws `sample` folds per step, and
    `DPMSolverSDEScheduler`'s walk integrates the draws its own `torchsde`
    tree answered, whose interval the reconstruction is checked to prepare.
    Every native walk and VJP runs in float32.

    The VJP is held to the source's own rounding from its float64
    evaluation, which widens every scheduler table and keeps that gradient
    in float64, so its truth shares no float32 arithmetic with the reference
    whose rounding it measures. In pointwise cases pixel permutations
    reproduce one rounding draw. Exponential v-prediction with thresholding
    is on par over the seeds its recording declares (check 1,
    source_seeds.npz): its one draw's ratio was 1.70, 0.99 over 128 seeds.
    Four others, whose few unclipped entries carry too heavy a tail for
    seeds to resolve, average 32 independent copies (768 gradient entries)
    in source_rounding.npz, held to twice the source's own distance; a
    256-copy pilot's disjoint original-size subsets reached ratio 3.52,
    while every 32-copy subset stayed below 1.40 on default/AVX/Ice Lake
    CPU codegen. Thresholding's maximum is 1 in these cases, so the
    quantile is clamped to a constant and the limit is a pointwise clip.
    CFG epsilon instead averages 52 pixel orders of its per-sample variance
    reductions; the rest are held to twice the source's own distance on
    their one recorded draw.
    """
    schedule, process, times, x_T = source_case(name)
    rescale = SOURCE["cases"][name]["guidance"]
    if f"{name}.betas" in SOURCE_ARRAYS:
        np.testing.assert_allclose(schedule.betas, SOURCE_ARRAYS[f"{name}.betas"],
                                   atol=1e-5, rtol=1e-5)
    expected_times = SOURCE_ARRAYS[f"{name}.grid_times"]
    model_times = np.asarray(process.sampler_schedule.model_time(times[:-1]))
    assert model_times.shape == expected_times.shape
    np.testing.assert_allclose(model_times, expected_times, atol=1e-3, rtol=1e-6)
    prior = float(process.sampler_schedule.prior_scale())
    assert abs(prior - float(SOURCE_ARRAYS[f"{name}.prior"])) < 1e-5 * max(prior, 1.0)
    expected = SOURCE_ARRAYS[f"{name}.latents"]
    cotangent = jnp.asarray(SOURCE_ARRAYS[f"{name}.cotangent"])
    final = source_final(schedule, process, times, rescale, SOURCE["label"], SOURCE["guidance_scale"])
    if rescale is None:
        latents = walk(schedule.solver, process, SourceOracle(), x_T, times)
        assert latents.shape == expected.shape
        assert relative_gap(latents, expected) < 1e-4
    else:
        assert relative_gap(final(x_T)[None], expected[-1:]) < 1e-4
    if name == "cfg.epsilon":
        source_gradient_over_orders(final, x_T, cotangent)
    elif f"{name}.seeds" in SOURCE_SEEDED:
        seeds = declared_seeds(SOURCE_SEEDS)[name]
        draw, recorded, truth = indexed_recording(SOURCE_SEEDED, name, seeds)
        gradient = jax.jit(lambda value, cot: {"grad": jax.vjp(final, value)[1](cot)[0]})
        assert_as_exact(lambda index, inputs: gradient(inputs["x_T"], inputs["cotangent"]), recorded, draw,
                        name, seeds=seeds, truth_fn=truth)
    elif f"{name}.x_T" in SOURCE_ROUNDING:
        gradient = jax.jit(lambda value, cot: jax.vjp(final, value)[1](cot)[0])(
            jnp.asarray(SOURCE_ROUNDING[f"{name}.x_T"]), jnp.asarray(SOURCE_ROUNDING[f"{name}.cotangent"]))
        assert_as_exact_as_the_reference(
            gradient, SOURCE_ROUNDING[f"{name}.grad_float32"], SOURCE_ROUNDING[f"{name}.grad"], name)
    else:
        (gradient,) = jax.vjp(final, x_T)[1](cotangent)
        assert_as_exact_as_the_reference(
            gradient, SOURCE_ARRAYS[f"{name}.grad_float32"], SOURCE_ARRAYS[f"{name}.grad"], name)


@pytest.mark.parametrize("config", [
    {"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": 3.0},
    "euler.default", "kdpm2.default", "dpm_multi.default"],
    ids=lambda config: config if isinstance(config, str) else "flow.shift3")
def test_one_row_walks_each_source_grid_as_it_walks_in_a_batch(config):
    """A prompt alone walks where its row of a batch walks, on the flow, sigma,
    stage and VP grids.

    One row is the walk XLA compiles differently: each grid read becomes a
    dynamic slice rather than a gather. On a GPU (the cuda lane) XLA's
    DynamicSliceAnnotator failed that compile while the lookups searched
    the grid with jnp.interp's bisection loop inside the scan, which is how
    every published flow-matching checkpoint first failed on one prompt.
    """
    from dew.diffusion.schedules.source import SourceSchedule

    if isinstance(config, str):
        config = json.loads(str(SOURCE_ARRAYS[f"{config}.config"]))
    schedule = SourceSchedule.from_config(config)
    process, times = schedule.sampling(6)
    model = SourceOracle()
    x_T = jax.random.normal(jax.random.PRNGKey(3), (2, 3, 4))
    denoise = process.denoiser(model, model.init(jax.random.PRNGKey(1), x_T[:1], jnp.ones((1,))), {})
    run = jax.jit(lambda value: sample(denoise, value, solver=schedule.solver, key=jax.random.PRNGKey(0),
                                       times=times, final_denoise=False))
    batch = run(x_T)
    for row in range(2):
        np.testing.assert_allclose(run(x_T[row:row + 1])[0], batch[row], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kind, extra", [
    ("EulerDiscrete", {"sigma_min": 30000, "sigma_max": 80000}),
    ("KDPM2Discrete", {}),
    ("DPMSolverMultistep", {"solver_order": 1}),
])
def test_source_grids_walk_repeated_initial_model_times(kind, extra):
    """Repeated model times keep their own sigma rows, as an explicitly indexed source walk does:
    each walked latent is as near the source's float64 walk as its own float32 walk is. The float64
    walk builds its tables in float64 (`float64_scheduler`), at the float32 walk's integer times."""
    import contextlib

    import torch
    from diffusers.schedulers import (
        scheduling_dpmsolver_multistep,
        scheduling_euler_discrete,
        scheduling_k_dpm_2_discrete,
    )
    from tool_support import load

    from dew.diffusion.schedules.source import SourceSchedule

    modules = {"EulerDiscrete": scheduling_euler_discrete,
               "KDPM2Discrete": scheduling_k_dpm_2_discrete,
               "DPMSolverMultistep": scheduling_dpmsolver_multistep}
    module = modules[kind]
    config = {"_class_name": kind + "Scheduler", "beta_schedule": "squaredcos_cap_v2",
              "use_karras_sigmas": True, **extra}
    schedule = SourceSchedule.from_config(config)
    process, times = schedule.sampling(6)
    model_times = np.asarray(process.sampler_schedule.model_time(times[:-1]))
    assert np.count_nonzero(model_times == model_times[0]) > 1
    sigmas = process.sampler_schedule.sigmas(times)
    assert np.all(np.diff(sigmas) < 0)
    initial = np.random.default_rng(7).normal(size=(2, 3, 64)).astype(np.float32)
    model = SourceOracle()
    denoise = process.denoiser(model, {}, {})
    _, walked = sample(denoise, jnp.asarray(initial), solver=schedule.solver, key=0,
                       times=times, final_denoise=False, record=lambda x, state, denoised: x)

    precision = load("diffusers_wan_reference")
    reference = []
    published = None
    for dtype in (torch.float32, torch.float64):
        context = precision.float64_scheduler(module) if dtype == torch.float64 else contextlib.nullcontext()
        with context:
            source = getattr(module, kind + "Scheduler").from_config(config)
            source.set_timesteps(6)
            # A pipeline can choose its first sigma by index, without a model-time lookup.
            source.set_begin_index(0)
            if published is None:
                published = source.timesteps.clone()
            elif not published.is_floating_point():
                # Integer conditioning times are the published ones, not a float64 rounding of them.
                source.timesteps = published
            x = torch.tensor(initial, dtype=dtype)
            latents = []
            for index, time in enumerate(source.timesteps):
                scaled = source.scale_model_input(x, time)
                prediction = torch.sin(scaled) * 0.07 + time * 0.001
                x = source.step(prediction, time, x).prev_sample
                if kind != "KDPM2Discrete" or index % 2 == 1 or index == len(source.timesteps) - 1:
                    latents.append(x.numpy().copy())
        reference.append(np.stack(latents))
    for index, actual in enumerate(walked):
        assert_as_exact_as_the_reference(actual, reference[0][index], reference[1][index],
                                         f"{kind} repeated model time, step {index}")


def walked(config: dict, steps: int) -> jax.Array:
    """Each interval's new state as the source `config`'s own solver walks `steps` of its grid,
    from one fixed latent (`walk`)."""
    from dew.diffusion.schedules.source import SourceSchedule

    schedule = SourceSchedule.from_config(config)
    process, times = schedule.sampling(steps)
    return walk(schedule.solver, process, SourceOracle(), jax.random.normal(jax.random.PRNGKey(3), (2, 3, 4)),
                times)


@pytest.mark.parametrize("config", [
    {"_class_name": "UniPCMultistepScheduler", "num_train_timesteps": 4, "timestep_spacing": "leading"},
    {"_class_name": "DPMSolverMultistepScheduler", "num_train_timesteps": 4, "timestep_spacing": "leading"},
    {"_class_name": "UniPCMultistepScheduler", "use_flow_sigmas": True, "flow_shift": 1e12,
     "prediction_type": "flow_prediction"},
], ids=["unipc.leading", "dpm_multi.leading", "unipc.flow_collapsed"])
def test_a_multistep_walk_over_an_interval_of_no_length_is_refused(config):
    """A multistep solver's coefficients divide by the gaps between the levels its history spans.
    Four leading-spaced steps over a four-entry table put every step at one model time and one
    sigma, and a flow shift of 1e12 gives four sigmas that differ as the grid is built and are all
    one in its float32 table: the walk refuses either before it runs, where it would otherwise end
    in NaN. Three steps over the four-entry table walk distinct levels, to finite latents."""
    with pytest.raises(ValueError):
        walked(config, 4)
    if "num_train_timesteps" in config:
        assert np.isfinite(np.asarray(walked(config, 3))).all()


def test_an_euler_walk_stays_put_over_a_plateau_of_its_table():
    """Euler takes an interval of no length as staying where it is, as the source's
    `x + derivative * dt` does: betas that stop at zero hold one sigma over the whole table, so
    three steps over it keep the initial latent through their two plateau intervals."""
    config = {"_class_name": "EulerDiscreteScheduler", "num_train_timesteps": 4,
              "trained_betas": [0.01, 0.0, 0.0, 0.0]}
    latents = np.asarray(walked(config, 3))
    initial = np.asarray(jax.random.normal(jax.random.PRNGKey(3), (2, 3, 4)))
    np.testing.assert_array_equal(latents[:2], np.stack([initial, initial]))
    assert np.isfinite(latents).all()


def test_a_source_noise_sampler_seed_pins_the_brownian_path():
    """`noise_sampler_seed` makes the tree's entropy the checkpoint's, so two
    walks with different keys integrate one path; without it the walk's own
    key decides, as the source's unseeded tree draws its entropy per run."""

    _, process, times, x_T = source_case("dpm_sde.default")
    model = SourceOracle()
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, *x_T.shape[1:])), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})

    def run(solver, seed):
        return sample(denoise, x_T, solver=solver, key=jax.random.PRNGKey(seed), times=times,
                      final_denoise=False)

    pinned = DPMSolverSDE(seed=7)
    np.testing.assert_array_equal(run(pinned, 0), run(pinned, 1))
    free = DPMSolverSDE()
    assert not np.allclose(run(free, 0), run(free, 1), atol=1e-3)


BROWNIAN_SEEDS = 8
"""Check 1's declared seeds for the Brownian walk against Levy's construction unrolled (`seeds_for`)."""


def test_the_brownian_walk_descends_every_level_as_levys_construction_unrolled():
    """The walk reads its point off Levy's construction to its depth (`_brownian_walk`), one level a
    loop iteration: on par with the construction unrolled level by level in Python (check 1), at the
    full depth and at one level, over drawn keys and queries of a grid's interval, its ends, its
    middle and points between. A loop and its unrolled levels fuse apart, so they round apart. The
    truth is the construction in float64 on the same float32 draws, its position and the branch
    each level takes read in float64 too: the path is continuous, so where float32's rounding of the
    position crosses a cell's end, the cell beside it meets it at the node they share."""
    from dew.sampling.solvers.brownian import MAX_BROWNIAN_DEPTH, _Brownian, _brownian_walk

    low, high, shape = np.float32(0.03), np.float32(14.6), (3, 4)

    def draw(seed):
        between = np.random.default_rng(seed).uniform(low, high, 9)
        points = np.concatenate([[low, high, (low + high) / 2], between]).astype(np.float32)
        return jax.random.PRNGKey(seed), points

    def walked(depth):
        def walk(key, points):
            state = _Brownian(key, jnp.float32(low), jnp.float32(high))
            return jax.vmap(lambda point: _brownian_walk(state, point, shape, depth))(points)
        return walk

    def unrolled(depth):
        def construction(key, points):
            # The values in the points' dtype, float32 for the reference and float64 for the truth.
            dtype = points.dtype
            span = jnp.asarray(high, dtype) - jnp.asarray(low, dtype)
            root = jax.random.normal(jax.random.fold_in(key, 0), shape, dtype=jnp.float32)

            def one(point):
                position = jnp.clip((point - jnp.asarray(low, dtype)) / span, 0.0, 1.0)
                start, end = jnp.zeros(shape, dtype), jnp.sqrt(span) * root.astype(dtype)
                index = jnp.zeros((), jnp.int32)
                for level in range(1, depth + 1):
                    node = jax.random.fold_in(jax.random.fold_in(key, level), 2 * index + 1)
                    deviation = 0.5 * jnp.sqrt(span * 2.0 ** (1 - level))
                    middle = 0.5 * (start + end) + deviation * jax.random.normal(
                        node, shape, dtype=jnp.float32).astype(dtype)
                    position = position * 2.0
                    # The next binary digit of the position, exact for a doubled value in [0, 2].
                    right = jnp.floor(position).astype(bool)
                    position = jnp.subtract(position, right.astype(dtype))
                    start, end = jnp.where(right, middle, start), jnp.where(right, end, middle)
                    index = 2 * index + right.astype(jnp.int32)
                return start + position * (end - start)

            return jax.vmap(one)(points)
        return construction

    for depth in (1, MAX_BROWNIAN_DEPTH):
        assert_as_exact(without_ynn(walked(depth)), without_ynn(unrolled(depth)), draw,
                        f"the Brownian walk at depth {depth}", seeds=BROWNIAN_SEEDS,
                        truth_fn=jax.jit(unrolled(depth)))
