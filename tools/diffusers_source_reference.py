"""Diffusers scheduler files reconstructed from their own configs, at the release the test extra pins.

`tools/diffusers_reference.py` proves the solver updates over grids Dew's own
processes supply. This tool proves the other half: that
`SourceSchedule.from_config` rebuilds what a published `scheduler_config.json`
means. Every case constructs the actual scheduler class, saves its authentic
config with `save_pretrained` *before* `set_timesteps` (DPM singlestep rewrites
its own `lower_order_final` there, and a config saved afterwards would hide
that policy), calls the actual `set_timesteps`, and then walks every scheduler
call a pipeline would make: `scale_model_input`, the model, `step`, including
the repeated stage rows the two-evaluation classes take.

What lands per case: the saved config, the beta table, the source timesteps and
sigmas, its `init_noise_sigma`, the latent after every grid interval and the
vector-Jacobian product of the last one against a fixed cotangent through the
whole trajectory. The float64 walk is the class's independent oracle's
(tools/float64_schedulers.py), every table its own in float64, the library
untouched; its gradient is stored in float64. Both precisions start from the
same float32 input and cotangent.

The model is the same closed form on both sides, a function of the scaled model
input and the model time, so the comparison is of scheduler policy alone. A
stochastic class is fed the exact draws Dew's solver makes at that step;
`DPMSolverSDEScheduler`'s own `torchsde` draws are recorded and replayed
in both precisions and in Dew. A separate record checks the native bridge's
identities against the source tree.

Run on CPU in an isolated reference environment with Dew, the test extra's diffusers
and torchsde==0.2.6:

    PYTHONPATH=src python tools/diffusers_source_reference.py
    PYTHONPATH=src python tools/diffusers_source_reference.py rounding \
        tests/fixtures/diffusers/source_rounding.npz
    PYTHONPATH=src python tools/diffusers_source_reference.py seeds \
        tests/fixtures/diffusers/source_seeds.npz
    PYTHONPATH=src python tools/diffusers_source_reference.py sde \
        tests/fixtures/diffusers/oracle_sde.npz

`cases NAME...` records those cases alone into the stored record and keeps
every other array as it is, so a case added later leaves the cases another
machine's float32 rounding recorded untouched.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import torch
from diffusers_reference_helpers import (
    fed_noise as fed_noise,
    restore_transformers_names,
    scheduler_module,
    step_noise as step_noise,
)
from float64_oracle import threshold_bound
from float64_schedulers import ORACLES
from float64_tables import PNDM
from scheduler_cases import CASES, GUIDANCE_SCALE, LABEL, SEED, STEPS, VP, Case, calls, guided, walked

from dew.diffusion.schedules.common import GeneralizedNoiseScheduler
from dew.diffusion.schedules.source import SourceSchedule

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "diffusers"
SHAPE = (2, 3, 4)

TWO_STAGE = ("HeunDiscreteScheduler", "KDPM2DiscreteScheduler",
             "KDPM2AncestralDiscreteScheduler", "DPMSolverSDEScheduler")


def walk(scheduler, module: ModuleType | None, case: Case, x: torch.Tensor, noises: list[np.ndarray]
         ) -> tuple[list[torch.Tensor], list[np.ndarray]]:
    """Every scheduler call the walk makes (`calls`), its model the case's, under the pipeline's
    classifier-free guidance where it states one (`guided`: Diffusers' own `rescale_noise_cfg`), keeping the
    latent at each grid interval's end: one call per interval, two for the classes that evaluate the model
    at an interpolated stage, and PNDM's own warmup timetable. A library class's draws are `noises`, fed to
    its `module` (`fed_noise`); an oracle's (no module) its own tape."""
    times = scheduler.timesteps
    last = len(times) - 1
    latents, inputs = [], []
    model = guided(case, rescale_noise_cfg)
    with fed_noise(module, noises) if module is not None else contextlib.nullcontext():
        for call in calls(scheduler, case, x, model):
            inputs.append(call.model_input.detach().numpy().astype(np.float32))
            if case.scheduler not in TWO_STAGE or call.at % 2 == 1 or call.at == last:
                latents.append(call.after)
    if case.scheduler == "PNDMScheduler":
        latents = (latents[1:] if scheduler.config.skip_prk_steps
                   else [latents[3], latents[7], latents[11]] + latents[12:])
    return latents, inputs


def rescale_noise_cfg():
    """Diffusers' own guidance rescale (`rescale_noise_cfg`), from its pipelines, which import the
    transformers names `restore_transformers_names` puts back."""
    restore_transformers_names()
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg

    return rescale_noise_cfg


def native_bounds(config: Mapping[str, object], steps: int) -> tuple[float, float]:
    """The positive sigma domain Dew prepares for this config, which is the
    interval the source builds its Brownian tree over."""
    schedule = SourceSchedule.from_config(config)
    process, _ = schedule.sampling(steps)
    sigma_schedule = process.sampler_schedule
    assert isinstance(sigma_schedule, GeneralizedNoiseScheduler)
    return float(sigma_schedule.sigma_min), float(sigma_schedule.sigma_max)


def grid_times(scheduler, case: Case) -> np.ndarray:
    """The model times at the grid points the outer walk visits, out of the
    call list the source runs: PNDM keeps its own ascending grid beside the
    warmup timetable, Heun repeats each point for its corrector, and the
    stage classes interleave a stage row between points."""
    times = scheduler.timesteps.numpy().astype(np.float64)
    if case.scheduler == "PNDMScheduler":
        # Its grid's own points, re-derived from its config by its oracle, whose timetable is the library's.
        rederived = PNDM(torch.float64, **case.config)
        rederived.set_timesteps(case.steps)
        assert torch.equal(rederived.timesteps, scheduler.timesteps.to(torch.int64)), case
        return rederived.grid_points[::-1].astype(np.float64)
    if case.scheduler == "HeunDiscreteScheduler":
        return np.concatenate([times[:1], times[1::2]])
    if case.scheduler in TWO_STAGE:
        return np.concatenate([times[:1], times[2::2]])
    return times


def built(module: ModuleType, case: Case, dtype: torch.dtype):
    """The scheduler a walk at `dtype` steps: the unmodified library class at float32, through its public
    interface alone, and at float64 the class's independent oracle (tools/float64_schedulers.py, held to the
    library by tests/test_float64_schedulers.py), every table its own in float64 (testing.md, row 13738)."""
    if dtype == torch.float64:
        return ORACLES[case.scheduler](torch.float64, **case.config)
    return getattr(module, case.scheduler)(**case.config)


def trajectory(module: ModuleType, case: Case, initial: torch.Tensor, cotangent: torch.Tensor,
               dtype: torch.dtype, noises: list[np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """One walk on identical inputs, cotangent and random draws: its latent at each grid interval's end
    (`after <i>`), or a guided walk's last alone (`final`), as the guided pipeline returns it, and the
    VJP of its last latent against `cotangent` (`grad`), the scheduler `built` for `dtype`."""
    scheduler = built(module, case, dtype)
    scheduler.set_timesteps(case.steps)
    x = initial.detach().to(dtype)
    x.requires_grad_(True)
    # The library's draws fed to its module; an oracle's its own tape, no module touched.
    library = module if dtype != torch.float64 else None
    if case.scheduler == "DPMSolverSDEScheduler":
        # The library walks its own seeded tree untouched; the oracle the answers that tree gives the same
        # walk (`brownian`), so both follow one path and the comparison is of their arithmetic.
        if library is None:
            scheduler.draws = list(brownian(module, case, initial.detach().float()).draws)
        latents, _ = walk(scheduler, None, case, x, [])
    else:
        noises = step_noise(SHAPE, len(scheduler.timesteps), np.float64) if noises is None else noises
        scheduler.draws = list(noises)
        latents, _ = walk(scheduler, library, case, x, noises)
    (gradient,) = torch.autograd.grad((latents[-1] * cotangent.to(dtype)).sum(), x)
    kept = ({"final": latents[-1]} if case.guidance is not None
            else {f"after {index}": latent for index, latent in enumerate(latents)})
    return {**{leaf: latent.detach().numpy() for leaf, latent in kept.items()}, "grad": gradient.numpy()}


def threshold_clearance(module: ModuleType, case: Case, initial: torch.Tensor,
                        noises: list[np.ndarray]) -> float:
    """How far the float64 walk from `initial` keeps clear of its dynamic thresholding's discontinuities,
    relative to the bound's scale: at every step, each entry of the predicted sample (the oracle's own,
    before it thresholds it: `float64_oracle.Oracle.observed`) from the per-sample clip bound `s`, and
    the unclamped quantile from the clamp's bounds, 1 and `sample_max_value`. An entry within rounding of
    either is clipped, or its bound clamped, in one precision and not the other."""
    seen: list[torch.Tensor] = []
    with torch.no_grad():
        scheduler = built(module, case, torch.float64)
        scheduler.set_timesteps(case.steps)
        scheduler.draws = list(noises)
        scheduler.observed = seen
        walk(scheduler, None, case, initial.detach().to(torch.float64), noises)
    ratio, top = scheduler.config.dynamic_thresholding_ratio, scheduler.config.sample_max_value
    margins = [1.0]
    for sample in seen:
        flat = sample.reshape(sample.shape[0], -1).abs().double()
        raw, bound = threshold_bound(flat, ratio, top)
        margins.append(float(((flat - bound).abs() / bound).min()))
        margins.append(float(torch.minimum((raw - 1).abs(), (raw - top).abs()).min() / top))
    return min(margins)


SEEDED_SHAPE = (8, 3, 32)
"""A seeded case's latent: at SHAPE's 24 entries DDIM's and DDPM's few unclipped gradient entries did not
resolve within 512 seeds (armada, 11-06)."""
SEEDED_STEPS = {"ddim.threshold": 2}
"""A seeded case's walk where it is shorter than its trajectory's: DDIM's dynamic thresholding over the
case's 5 steps contracts its clipped entries onto the next step's clip bound, so from step 2 on dozens sit
within rounding of it whatever the draw (float64 walk, 3 seeds: step 2 holds 2 to 10 entries within 1e-5 of
the bound, step 3 22 to 58, step 4 33 to 69, the closest 3e-7, where steps 0 and 1 keep every entry 9e-4
clear). Two steps keep both regimes, its quantile clamped at `sample_max_value` and inside the clamp, and
their draws clear (`threshold_clearance`); the long walk's arithmetic is the unthresholded DDIM cases'."""
SEEDED_CASES = tuple(name for name in CASES if not name.startswith("dpm_sde."))
"""Every case whose walk check 1 holds over seeds (`seeds`), an independent initial state and cotangent each:
all but DPMSolverSDE, whose walk integrates the draws its own Brownian tree answers. CFG epsilon's seeds hold
its walk alone: its gradient's rounding sits in its per-sample reductions and is averaged over pixel orders
(`rounding`)."""
FORWARD_ONLY = ("cfg.epsilon",)
"""The seeded cases whose seeds hold their walk and not its gradient (SEEDED_CASES)."""


def seeds(destination: Path) -> None:
    """Each seeded case's walk over the seeds check 1 draws (`record_case`): a seed's initial state of
    SEEDED_SHAPE at the scheduler's `init_noise_sigma` and its cotangent, the float32 walk from it and the
    float64 one (`trajectory`: its latents and, outside FORWARD_ONLY, its VJP), each kept whole: Dew's
    float64 walk reads its schedule's float32 tables, so it is no truth of the source's. The per-step draws
    Dew's
    solver makes at its key 0 fed to both, the count confirmed paired with Dew's own walk
    (tests/diffusion_support.py's `source_walked`), over the case's steps or SEEDED_STEPS', which
    `<case>.steps` records. A thresholding case's draw
    clears its clip by `selection_margins.MARGIN` (`threshold_clearance`): a seed takes its first draw that
    does (`selection_margins.clearing`)."""
    import diffusers
    import jax

    sys.path.insert(0, str(FIXTURES.parents[1]))
    from diffusion_support import source_walked
    from recorded_references import provenance, record_case, save
    from selection_margins import MARGIN, clearing

    torch.set_num_threads(2)
    noises = step_noise(SEEDED_SHAPE, STEPS, np.float64)
    arrays: dict[str, np.ndarray] = {}
    for name in SEEDED_CASES:
        case, module = CASES[name], scheduler_module(CASES[name].scheduler)
        case = replace(case, steps=SEEDED_STEPS.get(name, case.steps))
        scheduler = getattr(module, case.scheduler)(**case.config)
        with tempfile.TemporaryDirectory(prefix="dew-source-scheduler-") as saved:
            scheduler.save_pretrained(saved)
            config = json.loads((Path(saved) / "scheduler_config.json").read_text())
        scheduler.set_timesteps(case.steps)
        prior = float(scheduler.init_noise_sigma)
        schedule = SourceSchedule.from_config(config)
        process, times = schedule.sampling(case.steps)
        walked = source_walked(schedule, process, times, case.guidance, LABEL, GUIDANCE_SCALE,
                               gradient=name not in FORWARD_ONLY)

        def seeded(seed, prior=prior):
            generator = torch.Generator().manual_seed(seed)
            initial = (torch.randn(SEEDED_SHAPE, generator=generator, dtype=torch.float64) * prior).float()
            cotangent = torch.randn(SEEDED_SHAPE, generator=generator, dtype=torch.float64).float()
            return {"x_T": initial.numpy(), "cotangent": cotangent.numpy()}

        draw = seeded
        if case.config.get("thresholding"):
            def clears(inputs, case=case, module=module):
                return threshold_clearance(module, case, torch.from_numpy(inputs["x_T"]), noises) >= MARGIN

            draw = clearing(seeded, clears)

        def evaluate(inputs, case=case, module=module, name=name):
            initial, cotangent = (torch.from_numpy(inputs[key]) for key in ("x_T", "cotangent"))
            single = trajectory(module, case, initial, cotangent, torch.float32, noises)
            wide = trajectory(module, case, initial, cotangent, torch.float64, noises)
            if name in FORWARD_ONLY:
                single.pop("grad"), wide.pop("grad")
            return single, wide

        # Sized paired, as check 1's verdict pairs Dew with the reference: the reference compared with itself
        # across draws did not resolve DDIM's thresholding within 512 seeds, its per-draw errors spanning 36x.
        arrays.update(record_case(name, draw, evaluate, dew=jax.jit(walked)))
        arrays[f"{name}.steps"] = np.asarray(case.steps)
    version = diffusers.__version__
    command = f"python tools/diffusers_source_reference.py seeds {destination}"
    save(destination, {**arrays, "provenance": provenance("diffusers", version, "cpu", command,
                                                          torch=torch.__version__)},
         reference=f"huggingface/diffusers {version}", license="Apache-2.0")


SDE_CASES = tuple(name for name in CASES if name.startswith("dpm_sde."))
"""The DPMSolverSDE cases, whose walks draw from torchsde's Brownian tree, which the test extra does not
install: their oracle is held to these recordings (`sde`)."""


class Brownian(NamedTuple):
    """DPMSolverSDE's Brownian path over a walk (`brownian`): each query's ends, its answers, the tree's
    bounds."""

    intervals: np.ndarray
    draws: np.ndarray
    bounds: tuple[float, float]


def brownian(module: ModuleType, case: Case, x_T: torch.Tensor) -> Brownian:
    """The answers DPMSolverSDE's Brownian tree gives its walk from `x_T`, drawn apart from the walk: the
    module's public `BrownianTreeNoiseSampler` built as the scheduler builds its own (the walk's first
    sample, the least positive and the greatest of its published sigmas, the config's
    `noise_sampler_seed`, the identity transform) and asked at each step's ends as its `step` asks, in its
    arithmetic: a first stage to the midpoint of its sigmas' logs, a second from the interval's start to its
    end, none onto sigma 0. The library's own walk, untouched, asks its own tree the same: an oracle walking
    these answers follows its path (tests/test_float64_schedulers.py holds it to that walk)."""
    scheduler = getattr(module, case.scheduler)(**case.config)
    scheduler.set_timesteps(case.steps)
    sigmas = scheduler.sigmas
    bounds = (sigmas[sigmas > 0].min().item(), sigmas.max().item())
    tree = module.BrownianTreeNoiseSampler(x_T, *bounds, case.config["noise_sampler_seed"])
    intervals, answers, second = [], [], False
    for index in range(len(scheduler.timesteps)):
        sigma, after = (sigmas[index - 1], sigmas[index]) if second else (sigmas[index], sigmas[index + 1])
        if after == 0:
            continue
        start, end = sigma.log().neg(), after.log().neg()
        reached = end if second else start + (end - start) * 0.5
        asked = (start.neg().exp(), reached.neg().exp())
        intervals.append(tuple(float(end) for end in asked))
        answers.append(tree(*asked).numpy().astype(np.float64))
        second = not second
    return Brownian(np.asarray(intervals, np.float64), np.stack(answers), bounds)


def sde(destination: Path) -> None:
    """Each DPMSolverSDE case's unmodified float32 walk over the seeds check 1 draws (`record_case`, judged:
    the arm its oracle's check 1 judges in tests/test_float64_schedulers.py), from a seed's x_T at the
    scheduler's `init_noise_sigma` (`recorded_inputs.normals`), its latent after each step
    (`scheduler_cases.walked`), the library run untouched; beside the inputs, the answers its Brownian tree
    gives that walk, drawn apart from it (`brownian`): the oracle's float32 walk on them the reference,
    its float64 walk the truth (row 13738)."""
    import diffusers

    sys.path.insert(0, str(FIXTURES.parents[1]))
    from recorded_inputs import normals
    from recorded_references import provenance, record_case, save

    torch.set_num_threads(2)
    arrays: dict[str, np.ndarray] = {}
    for name in SDE_CASES:
        case, module = CASES[name], scheduler_module(CASES[name].scheduler)
        probe = getattr(module, case.scheduler)(**case.config)
        probe.set_timesteps(case.steps)
        prior = float(probe.init_noise_sigma)

        def draw(seed, prior=prior):
            return {"x_T": normals([0x5DE, seed], SHAPE) * np.float32(prior)}

        def replayed(inputs, dtype, case=case):
            scheduler = ORACLES[case.scheduler](dtype, **case.config)
            scheduler.draws = list(inputs["draws"])
            return walked(scheduler, case, torch.from_numpy(inputs["x_T"]).to(dtype))

        def evaluate(inputs, case=case, module=module, replayed=replayed):
            x_T = torch.from_numpy(inputs["x_T"])
            with torch.no_grad():
                latents = walked(getattr(module, case.scheduler)(**case.config), case, x_T)
            made = {"draws": brownian(module, case, x_T).draws}
            return latents, replayed({**inputs, **made}, torch.float64), made

        arrays.update(record_case(f"{name} oracle", draw, evaluate, judged=True,
                                  dew=lambda inputs, replayed=replayed: replayed(inputs, torch.float32)))
    version = diffusers.__version__
    command = f"python tools/diffusers_source_reference.py sde {destination}"
    save(destination, {**arrays, "provenance": provenance("diffusers", version, "cpu", command,
                                                          torch=torch.__version__)},
         reference=f"huggingface/diffusers {version}", license="Apache-2.0")


def rounding(destination: Path) -> None:
    """Pixel orders for CFG's reduction.

    CFG with rescaling reduces over pixels, so its 52 permutations do move
    rounding. Both its input and cotangent move, and its float64 VJP must
    commute with each permutation before a reference distance is recorded.
    The original source_schedulers arrays are read, never replaced.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from reference_error import ORDERS, distance
    from residual_orders import orders

    torch.set_num_threads(2)
    arrays: dict[str, np.ndarray] = {}
    name = "cfg.epsilon"
    case, module = CASES[name], scheduler_module(CASES[name].scheduler)
    drawn = orders(int(np.prod(SHAPE[1:])), ORDERS, SEED)
    distances = []
    with np.load(FIXTURES / "source_schedulers.npz") as stored:
        truth = stored[f"{name}.grad"]
        for index, order in enumerate(drawn):
            def moved(value, order=order):
                return value.reshape(SHAPE[0], -1)[:, order].reshape(SHAPE)
            initial, cotangent = (torch.tensor(moved(stored[f"{name}.{field}"]))
                                  for field in ("x_T", "cotangent"))
            single = trajectory(module, case, initial, cotangent, torch.float32)["grad"]
            wide = trajectory(module, case, initial, cotangent, torch.float64)["grad"]
            scale = float(np.sqrt(np.mean(np.square(truth))))
            assert distance(wide, moved(truth)) <= 1e-12 * scale, (
                "the float64 VJP must commute with the order")
            if index == 0:
                np.testing.assert_array_equal(single, stored[f"{name}.grad_float32"])
            distances.append(distance(single, moved(truth)))
    arrays["orders"] = drawn.astype(np.uint8)
    arrays[f"{name}.distances"] = np.asarray(distances)
    np.savez_compressed(destination, allow_pickle=False, **arrays)
    print(f"{destination}: {ORDERS} pixel orders")


def run(name: str, case: Case) -> dict[str, np.ndarray]:
    module = scheduler_module(case.scheduler)
    scheduler = getattr(module, case.scheduler)(**case.config)
    with tempfile.TemporaryDirectory(prefix="dew-source-scheduler-") as saved:
        scheduler.save_pretrained(saved)
        config = json.loads((Path(saved) / "scheduler_config.json").read_text())
    scheduler.set_timesteps(case.steps)
    times = scheduler.timesteps.tolist()
    # The source finds its starting step index by matching the first model
    # time, and takes the second match when that time repeats elsewhere in the
    # grid, which shifts its whole walk by one. A grid that repeats its first
    # model time is not a policy this reconstruction models, so a case must
    # not produce one; the deliberate repeats of PNDM's warmup and of the
    # two-evaluation classes are counted from there and are fine.
    assert times.count(times[0]) == 1, f"{name}: the first model time repeats in {times}"
    generator = torch.Generator().manual_seed(SEED)
    prior = float(scheduler.init_noise_sigma)
    x_T = (torch.randn(SHAPE, generator=generator, dtype=torch.float64) * prior).float().double()
    cotangent = torch.randn(SHAPE, generator=generator, dtype=torch.float64).float().double()
    arrays: dict[str, np.ndarray] = {}
    if case.scheduler == "DPMSolverSDEScheduler":
        # The library walks its own seeded tree untouched; its path's queries and answers drawn apart
        # (`brownian`), the ones Dew's walk integrates.
        latents, inputs = walk(scheduler, None, case, x_T, [])
        path = brownian(module, case, x_T)
        low, high = native_bounds(config, case.steps)
        assert abs(path.bounds[0] - low) < 1e-5 * low, (
            f"{name}: the source tree starts at {path.bounds[0]}, Dew prepares {low}")
        assert abs(path.bounds[1] - high) < 1e-5 * high, (
            f"{name}: the source tree ends at {path.bounds[1]}, Dew prepares {high}")
        arrays["intervals"] = path.intervals
        arrays["noise"] = path.draws
        arrays["bounds"] = np.asarray(path.bounds, np.float64)
    else:
        noises = step_noise(SHAPE, len(scheduler.timesteps), np.float64)
        latents, inputs = walk(scheduler, module, case, x_T, noises)
    assert len(latents) == case.steps, (name, len(latents), case.steps)
    single = trajectory(module, case, x_T, cotangent, torch.float32)["grad"]
    gradient = trajectory(module, case, x_T, cotangent, torch.float64)["grad"]
    arrays.update({
        "x_T": x_T.detach().numpy().astype(np.float32),
        "latents": np.stack([latent.detach().numpy() for latent in latents]).astype(np.float32),
        "cotangent": cotangent.numpy().astype(np.float32),
        "grad": gradient,
        "grad_float32": single,
        "times": scheduler.timesteps.numpy().astype(np.float64),
        "prior": np.asarray(prior, np.float64),
        "inputs": np.stack(inputs),
        "grid_times": grid_times(scheduler, case),
        "config": np.asarray(json.dumps(config)),
    })
    if hasattr(scheduler, "betas"):
        arrays["betas"] = scheduler.betas.numpy().astype(np.float32)
    if hasattr(scheduler, "sigmas"):
        arrays["sigmas"] = np.asarray(scheduler.sigmas, np.float64)
    return arrays


def brownian_record() -> dict[str, np.ndarray]:
    """The actual `torchsde` tree the source builds, over the interval a
    published DPMSolverSDE grid prepares.

    Endpoint values, the increments of a nested query set and the same set
    queried in the reverse order land here, so the identities Dew's bridge is
    built on are checked against a real one rather than assumed.
    """
    case = CASES["dpm_sde.default"]
    module = scheduler_module(case.scheduler)
    scheduler = module.DPMSolverSDEScheduler(**case.config)
    with tempfile.TemporaryDirectory(prefix="dew-source-brownian-") as saved:
        scheduler.save_pretrained(saved)
        config = json.loads((Path(saved) / "scheduler_config.json").read_text())
    scheduler.set_timesteps(case.steps)
    levels = np.asarray(scheduler.sigmas, np.float64)
    low, high = float(levels[levels > 0].min()), float(levels.max())
    sampler = module.BrownianTreeNoiseSampler
    # Interior points: a query that starts exactly on the tree's own end warns
    # about its float boundary and says nothing extra about the identities.
    points = np.linspace(low, high, 7)[1:-1]
    queries = [(points[0], points[2]), (points[2], points[4]), (points[0], points[4]),
               (points[1], points[3]), (points[0], points[1])]

    def normalized(order):
        tree = sampler(torch.zeros(SHAPE, dtype=torch.float64), low, high, seed=SEED)
        values = {}
        for index in order:
            first, second = queries[index]
            values[index] = tree(torch.as_tensor(first), torch.as_tensor(second)).numpy()
        return np.stack([values[index] for index in range(len(queries))])

    forward = normalized(range(len(queries)))
    reversed_order = normalized(list(reversed(range(len(queries)))))
    tree = sampler(torch.zeros(SHAPE, dtype=torch.float64), low, high, seed=SEED)
    swapped = np.stack([tree(torch.as_tensor(second), torch.as_tensor(first)).numpy()
                        for first, second in queries])
    widths = np.asarray([abs(second - first) for first, second in queries], np.float64)
    return {
        "config": np.asarray(json.dumps(config)),
        "bounds": np.asarray([low, high], np.float64),
        "queries": np.asarray(queries, np.float64),
        "widths": widths,
        "normalized": forward.astype(np.float64),
        "reordered": reversed_order.astype(np.float64),
        "swapped": swapped.astype(np.float64),
    }


REFUSED: Mapping[str, Case] = {
    # The largest sigmas of a cosine table all recover the same model index, so
    # this grid repeats its first model time and the source cannot walk it.
    "cosine_karras": Case("DPMSolverMultistepScheduler", dict(
        VP, beta_schedule="squaredcos_cap_v2", use_karras_sigmas=True)),
}


def refused_record() -> dict[str, np.ndarray]:
    """Configurations Diffusers itself cannot run, with the failure it raises.

    A refusal is only honest against the real behaviour, so each of these is
    walked here and the exception it produces is recorded beside its config.
    """
    arrays: dict[str, np.ndarray] = {}
    for name, case in REFUSED.items():
        module = scheduler_module(case.scheduler)
        scheduler = getattr(module, case.scheduler)(**case.config)
        with tempfile.TemporaryDirectory(prefix="dew-source-refused-") as saved:
            scheduler.save_pretrained(saved)
            config = json.loads((Path(saved) / "scheduler_config.json").read_text())
        scheduler.set_timesteps(case.steps)
        times = scheduler.timesteps.tolist()
        x = torch.zeros(SHAPE, dtype=torch.float64)
        try:
            walk(scheduler, module, case, x, step_noise(SHAPE, len(times), np.float64))
        except Exception as failure:
            reason = f"{type(failure).__name__}: {failure}"
        else:
            raise AssertionError(f"{name} walked to the end; it is not a refused configuration")
        arrays[f"refused.{name}.config"] = np.asarray(json.dumps(config))
        arrays[f"refused.{name}.times"] = np.asarray(times, np.float64)
        arrays[f"refused.{name}.failure"] = np.asarray(reason)
        arrays[f"refused.{name}.steps"] = np.asarray(case.steps, np.int64)
        print(f"refused {name}: {reason} at times {times}")
    return arrays


def main(only: Sequence[str] = ()) -> None:
    import diffusers
    from reference_releases import require

    require("diffusers")
    torch.set_num_threads(2)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(only) - set(CASES))
    if unknown:
        raise SystemExit(f"no cases named {unknown}")
    arrays: dict[str, np.ndarray] = {}
    cases: dict[str, dict[str, object]] = {}
    record: dict[str, Any] = {"diffusers": diffusers.__version__, "shape": list(SHAPE),
                                 "guidance_scale": GUIDANCE_SCALE,
                                 "label": LABEL, "cases": cases}
    if only:
        record = json.loads((FIXTURES / "source_schedulers.json").read_text())
        if record["diffusers"] != diffusers.__version__:
            raise SystemExit(f"the stored record is diffusers {record['diffusers']}, "
                             f"this is {diffusers.__version__}")
        cases = record["cases"]
        with np.load(FIXTURES / "source_schedulers.npz") as stored:
            arrays = {key: stored[key] for key in stored.files
                      if not any(key.startswith(f"{name}.") for name in only)}
    for name, case in CASES.items():
        if only and name not in only:
            continue
        result = run(name, case)
        for key, value in result.items():
            arrays[f"{name}.{key}"] = value
        cases[name] = {"scheduler": case.scheduler, "steps": case.steps,
                       "guidance": case.guidance,
                       "latent_scale": float(np.abs(result["latents"]).max())}
        print(f"{name}: {case.steps} intervals, |latent| <= {cases[name]['latent_scale']:.4g}")
    if not only:
        arrays.update(refused_record())
        record["refused"] = sorted(REFUSED)
        for key, value in brownian_record().items():
            arrays[f"brownian.{key}"] = value
        print(f"brownian: torchsde tree over {arrays['brownian.bounds']}")
    np.savez_compressed(FIXTURES / "source_schedulers.npz", allow_pickle=False, **arrays)
    (FIXTURES / "source_schedulers.json").write_text(json.dumps(record, indent=1) + "\n")
    size = (FIXTURES / "source_schedulers.npz").stat().st_size
    print(f"{FIXTURES / 'source_schedulers.npz'}: {size / 1e3:.0f} kB, {len(CASES)} cases")


if __name__ == "__main__":
    if sys.argv[1:2] == ["rounding"]:
        rounding(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["seeds"]:
        seeds(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["sde"]:
        sde(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["cases"]:
        main(sys.argv[2:])
    else:
        main()
