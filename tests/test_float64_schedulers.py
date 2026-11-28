"""The schedulers' independent oracles (tools/float64_schedulers.py) held to Diffusers' own: check 1 of the
gate over each case tools/scheduler_cases.py states (the ones the source references walk, `CASES` and
`FLOW_CASES`, and `ORACLE_CASES`), the library's float32 walk as the arm judged, the oracle's float32 walk
(its tables built as the library builds them) its reference, the oracle's float64 walk the truth. A formula
the oracle gets wrong leaves its float32 walk beside its wrong truth and the library's far from it; one it
gets right puts both float32 walks at one rounding from it (row 13738, Main's ruling of 2026-11-27)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from reference_error import as_given, assert_as_exact, declared_seeds, indexed_recording, recording
from tool_support import load, scheduler_oracles

oracles = scheduler_oracles()
cases = load("scheduler_cases")
helpers = load("diffusers_reference_helpers")
scheduler_module = helpers.scheduler_module
WALKED, walked = {**cases.CASES, **cases.FLOW_CASES, **cases.ORACLE_CASES}, cases.walked

ORACLES = oracles.ORACLES

UNWALKED = {"DPMSolverSDEScheduler": "its steps draw from torchsde's Brownian tree, which the test extra "
                                     "does not install: its oracle is held to its recorded walks"}
"""The classes whose library no case walks here, each with why: any other class's case fails."""
SDE = Path(__file__).resolve().parent / "fixtures" / "diffusers" / "oracle_sde.npz"
"""DPMSolverSDE's library walks with their Brownian trees' answers (tools/diffusers_source_reference.py's
`sde`)."""

SHAPE = (2, 3, 4)
SEEDS = 8
"""The declared seeds of each case's site."""

COVERED = sorted(name for name, case in WALKED.items() if case.scheduler not in UNWALKED)
"""Every case a library class walks, guided ones under their pipeline's guidance (`model`)."""


DRAWS = 16
"""The unit draws any case's walk takes at most."""


def draw(seed: int) -> tuple[dict[str, np.ndarray]]:
    """A case's prior draw, and the seed of the torch generator its steps draw from (`generated`)."""
    rng = np.random.default_rng(seed)
    return ({"x": rng.standard_normal(SHAPE).astype(np.float32), "seed": np.asarray(seed, np.int64)},)


def generated(seed) -> torch.Generator:
    """The torch generator a walk's steps draw their noise from, seeded as the inputs state."""
    return torch.Generator().manual_seed(int(np.asarray(seed)))


def model(name: str) -> cases.Model:
    """The model case `name`'s walks run: the oracle, or under a guided case's pipeline guidance Diffusers'
    own `rescale_noise_cfg` (`scheduler_cases.guided`), the one the source recorder walks."""
    def rescaling():
        helpers.restore_transformers_names()
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg

        return rescale_noise_cfg

    return cases.guided(WALKED[name], rescaling)


def library(name: str):
    """The unmodified library's float32 walk of case `name`, a step that draws given the seeded generator
    (its public `generator`)."""
    case = WALKED[name]
    module = scheduler_module(case.scheduler)
    draws = hasattr(ORACLES[case.scheduler], "drawn")

    @as_given
    def run(inputs):
        scheduler = getattr(module, case.scheduler)(**case.config)
        drawing = {"generator": generated(inputs["seed"])} if draws else {}
        return walked(scheduler, case, torch.as_tensor(np.asarray(inputs["x"])), model(name), **drawing)

    return run


def oracle_walk(name: str, dtype: torch.dtype):
    """The oracle's walk of case `name` at `dtype`, its draws its own (`Drawn.draws`)."""
    case = WALKED[name]

    @as_given
    def run(inputs):
        scheduler = ORACLES[case.scheduler](dtype, **case.config)
        # The library's draws, in its order: float32 unit normals of the generator (`randn_tensor`).
        generator = generated(inputs["seed"])
        scheduler.draws = [torch.randn(SHAPE, generator=generator, dtype=torch.float32).numpy()
                           for _ in range(DRAWS)]
        return walked(scheduler, case, torch.as_tensor(np.asarray(inputs["x"])).to(dtype), model(name))

    return run


@pytest.mark.parametrize("name", COVERED)
def test_an_oracle_walks_as_the_library_does(name):
    """The library's float32 walk of each case is on par with the oracle's float32 walk, against the
    oracle's float64 walk (check 1)."""
    assert_as_exact(library(name), oracle_walk(name, torch.float32), draw, f"{name} oracle", seeds=SEEDS,
                    truth_fn=oracle_walk(name, torch.float64))


def test_edm_refuses_a_sample_prediction_as_the_library_does():
    """EDM's oracle steps no `sample` prediction, which the library's signature names but its step refuses
    (`precondition_outputs`): a config the library walks no further, the oracle walks no further."""
    config = {**cases.EDM, "prediction_type": "sample"}
    library = scheduler_module("EDMDPMSolverMultistepScheduler").EDMDPMSolverMultistepScheduler(**config)
    oracle = ORACLES["EDMDPMSolverMultistepScheduler"](torch.float64, **config)
    for scheduler in (library, oracle):
        scheduler.set_timesteps(4)
        x = torch.ones(SHAPE, dtype=torch.float64 if scheduler is oracle else torch.float32)
        with pytest.raises(ValueError):
            scheduler.step(x, scheduler.timesteps[0], x)


def held(build, case, dtype: torch.dtype, stepping=lambda scheduler, seed: {}):
    """The walk of what `build(dtype)` makes over `case` from a draw's x at `dtype`, each step given what
    `stepping(scheduler, seed)` hands it."""
    @as_given
    def run(inputs):
        scheduler = build(dtype)
        x = torch.as_tensor(np.asarray(inputs["x"])).to(dtype)
        return walked(scheduler, case, x, **stepping(scheduler, inputs["seed"]))
    return run


def test_an_oracle_of_a_scheduler_built_from_another_classes_config_walks_as_it():
    """A library scheduler built from another class's config (`from_config`, as a pipeline swaps its
    scheduler: PNDM from DDIM's), its config keeping the options its class ignores: its oracle (`oracle_of`)
    reads its own class's alone and walks as it does (check 1)."""
    module = scheduler_module("PNDMScheduler")
    given = scheduler_module("DDIMScheduler").DDIMScheduler(**cases.VP).config
    case = cases.Case("PNDMScheduler")

    def library_built(dtype):
        return module.PNDMScheduler.from_config(given)

    def oracle_built(dtype):
        return oracles.oracle_of(library_built(dtype), dtype)

    assert_as_exact(held(library_built, case, torch.float32), held(oracle_built, case, torch.float32), draw,
                    "PNDM from DDIM's config", seeds=SEEDS, truth_fn=held(oracle_built, case, torch.float64))


def test_ddim_takes_a_pipelines_eta_through_its_step_dispatch():
    """DDIM's oracle takes the eta and generator a pipeline hands its step by the step's own signature
    (`prepare_extra_step_kwargs`, which passes `eta` to a step naming it alone): at eta 1, on the
    generator's draws, it walks as the library does (check 1)."""
    helpers.restore_transformers_names()
    from diffusers import StableDiffusionPipeline

    module = scheduler_module("DDIMScheduler")
    case = cases.Case("DDIMScheduler", {**cases.VP, "clip_sample": False})

    def stepping(scheduler, seed):
        return StableDiffusionPipeline.prepare_extra_step_kwargs(SimpleNamespace(scheduler=scheduler),
                                                                 generated(seed), 1.0)

    def library_built(dtype):
        return module.DDIMScheduler(**case.config)

    def oracle_built(dtype):
        return ORACLES["DDIMScheduler"](dtype, **case.config)

    assert_as_exact(held(library_built, case, torch.float32, stepping),
                    held(oracle_built, case, torch.float32, stepping), draw, "DDIM's eta from a pipeline",
                    seeds=SEEDS, truth_fn=held(oracle_built, case, torch.float64, stepping))

@pytest.mark.parametrize("name", sorted(name for name, case in cases.CASES.items()
                                        if case.scheduler in UNWALKED and case.guidance is None))
def test_the_sde_oracle_walks_as_its_recorded_library(name):
    """DPMSolverSDE's unmodified float32 walks over the seeds its recording holds, each beside the answers
    its Brownian tree gave, on par with the oracle's float32 walk on those answers against the oracle's
    float64 walk (check 1; the recording is the arm judged)."""
    case, label = WALKED[name], f"{name} oracle"
    seeds = declared_seeds(SDE)[label]
    draw, recorded, _ = indexed_recording(recording(SDE), label, seeds)

    def replayed(dtype: torch.dtype):
        @as_given
        def run(index, inputs):
            scheduler = ORACLES[case.scheduler](dtype, **case.config)
            scheduler.draws = list(np.asarray(inputs["draws"]))
            return walked(scheduler, case, torch.as_tensor(np.asarray(inputs["x_T"])).to(dtype))
        return run

    assert_as_exact(recorded, replayed(torch.float32), draw, label, seeds=seeds,
                    truth_fn=replayed(torch.float64))

