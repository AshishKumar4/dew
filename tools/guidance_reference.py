"""Guided walks against their published references, for tests/fixtures/guidance.

- APG: Diffusers' `AdaptiveProjectedGuidance` guider
  (`src/diffusers/guiders/adaptive_projected_guidance.py` and the
  `BaseGuidance` it extends in `guider_utils.py`, read at a pinned commit):
  its `__init__`, `set_state`, `prepare_inputs`, `forward` and
  `_is_apg_enabled`, with `normalized_guidance`, `MomentumBuffer` and
  `rescale_noise_cfg`, run as published. The guider is driven as a modular
  pipeline drives it, `set_state` and `prepare_inputs` then `forward` at
  every step of an Euler walk on the linear path, combining a closed-form
  flow model's conditional and unconditional velocities; its `start` and
  `stop` count steps, [int(start N), int(stop N)). Its batching helper
  (`_prepare_batch`) and output container (`GuiderOutput`) are stand-ins:
  neither touches the numbers.
- Autoguidance: NVlabs/edm2's `edm_sampler` (`generate_images.py`, pinned,
  run as published, not vendored: CC BY-NC-SA 4.0) walks EDM's Heun sampler
  with a closed-form main network and a weaker one as `gnet`.
- Guidance interval: Kynkaanniemi et al.'s own `edm_sampler`
  (kynkaat/guidance-interval `sampling/edm_sampler.py`, pinned, run as
  published, not vendored: CC BY-NC-SA 4.0) walks EDM's Heun sampler with
  classifier-free guidance on the steps whose index lies in
  `guidance_interval`, closed at both ends, both of a step's evaluations
  taking its decision. The sampler names float64 for its walk and float32
  for what it returns; each run reads its own precision for both.

Each runs in float64 and in float32, over the seeds check 1 draws
(`tools/recorded_references.py`): one initial state's rounding is a single
draw, which a pointwise walk's spatial orders do not move. APG's walks land
with their initial states (`apg-seeds`); the APG projection, the autoguided
walk and the interval-guided walks land as their two results a seed, their
inputs drawn again by the test from the same seeded torch generator
(`walk-seeds`, `projection_draw`, `noise_draw`).

    PYTHONPATH=src python tools/guidance_reference.py apg-seeds tests/fixtures/guidance/apg_seeds.npz
    PYTHONPATH=src python tools/guidance_reference.py walk-seeds tests/fixtures/guidance/walk_seeds.npz
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
from reference_source import definitions, fetched, selected

DIFFUSERS = "5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2"
GUIDERS = f"https://raw.githubusercontent.com/huggingface/diffusers/{DIFFUSERS}/src/diffusers/guiders/"
EDM2 = ("https://raw.githubusercontent.com/NVlabs/edm2/"
        "4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc/generate_images.py")
INTERVAL = ("https://raw.githubusercontent.com/kynkaat/guidance-interval/"
            "be70a2334f7ebf4e55fec981d518d5036a2ca19b/sampling/edm_sampler.py")
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "guidance"
SHAPE = (2, 3, 4)
# Each case's walk takes `steps` Euler steps; `start` and `stop` are the
# guider's own, fractions of those steps.
APG_CASES = {
    "plain": {"scale": 4.0, "eta": 1.0, "norm_threshold": 0.0, "momentum": 0.0,
              "start": 0.0, "stop": 1.0, "steps": 5},
    "projected": {"scale": 6.0, "eta": 0.0, "norm_threshold": 0.8, "momentum": 0.0,
                  "start": 0.0, "stop": 1.0, "steps": 5},
    "momentum": {"scale": 6.0, "eta": 0.2, "norm_threshold": 0.8, "momentum": -0.5,
                 "start": 0.0, "stop": 1.0, "steps": 5},
    "interval": {"scale": 6.0, "eta": 0.2, "norm_threshold": 0.8, "momentum": -0.5,
                 "start": 0.25, "stop": 0.75, "steps": 10},
}
LABEL = 0.7
# The paper's sampler over `num_steps` Heun steps, guided on the step indices
# of `guidance_interval`; the conditional and unconditional networks are the
# closed-form denoiser at two strengths.
INTERVAL_CASES = {
    "middle": {"num_steps": 10, "guidance_interval": [3, 6], "G": 2.5},
    "tail": {"num_steps": 10, "guidance_interval": [7, 9], "G": 2.5},
}
CONDITIONAL, UNCONDITIONAL = 0.5, 0.35
PROJECTION_CASES = {"ordinary": (1.0, 1.0), "large": (1e30, 1e30), "tiny": (1e-20, 1e-20),
                    "different_scales": (1.0, 1e30)}
"""The APG projection's cases: the scales of the conditional and the unconditional prediction, so the
orthogonal component's rounding shows at every magnitude float32 holds."""
PROJECTION_ROWS = 32
AUTOGUIDANCE = {"num_steps": 8, "guidance": 2.2, "strength": 0.5, "guide_strength": 0.35}


def members(url: str, owner: str, names: set[str]) -> list[str]:
    """The source of the named members of class `owner` in the file at
    `url`, decorators kept but for `register_to_config`, which only records
    the arguments for a config file."""
    found = selected(fetched(url), names, within=owner, filename=url)
    for member in found:
        if isinstance(member, ast.FunctionDef):
            member.decorator_list = [decorator for decorator in member.decorator_list
                                     if ast.unparse(decorator) != "register_to_config"]
    return [ast.unparse(member) for member in found]


def guider() -> type:
    """Diffusers' `AdaptiveProjectedGuidance` as published, on a
    `BaseGuidance` holding its own published `__init__` and `set_state`."""
    scope = definitions(GUIDERS + "adaptive_projected_guidance.py", ("MomentumBuffer", "normalized_guidance"),
                      {"torch": torch, "math": math})
    definitions(GUIDERS + "guider_utils.py", ("rescale_noise_cfg",), scope)
    scope["logger"] = types.SimpleNamespace(warning=lambda *_: None)
    scope["GuiderOutput"] = types.SimpleNamespace
    base = members(GUIDERS + "guider_utils.py", "BaseGuidance", {"__init__", "set_state"})
    exec("class BaseGuidance:\n" + "\n".join(
        "\n".join("    " + line for line in member.splitlines()) for member in base), scope)
    published = members(GUIDERS + "adaptive_projected_guidance.py", "AdaptiveProjectedGuidance",
                        {"_input_predictions", "__init__", "prepare_inputs", "forward", "num_conditions",
                         "_is_apg_enabled"})
    stand_in = ["def _prepare_batch(self, data, tuple_idx, input_prediction):\n    return None"]
    exec("class AdaptiveProjectedGuidance(BaseGuidance):\n" + "\n".join(
        "\n".join("    " + line for line in member.splitlines()) for member in published + stand_in),
         scope)
    return scope["AdaptiveProjectedGuidance"]


def velocity(x, time, label):
    """The flow model the APG walk and `tests` run: the scaled input's
    nonlinearity, the model time and the condition."""
    return torch.sin(x) * 0.07 + time * 0.001 + label


def apg_walk(published: type, case: dict, x: torch.Tensor) -> torch.Tensor:
    """An Euler walk of `case["steps"]` steps, the guider told each step's
    index before it combines that step's two velocities."""
    guidance = published(guidance_scale=case["scale"],
                         adaptive_projected_guidance_momentum=case["momentum"] or None,
                         adaptive_projected_guidance_rescale=case["norm_threshold"], eta=case["eta"],
                         start=case["start"], stop=case["stop"])
    sigmas = torch.linspace(1.0, 0.0, case["steps"] + 1, dtype=x.dtype)
    for step, (sigma, following) in enumerate(itertools.pairwise(sigmas)):
        guidance.set_state(step, case["steps"], sigma * 1000)
        guidance.prepare_inputs({})
        conditional = velocity(x, sigma * 1000, LABEL)
        unconditional = velocity(x, sigma * 1000, 0.0)
        x = x + (following - sigma) * guidance.forward(conditional, unconditional).pred
    return x


def apg_draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's initial state for the APG walks, a unit normal of SHAPE from torch's generator."""
    return {"x_T": torch.randn(*SHAPE, generator=torch.Generator().manual_seed(seed)).numpy()}


def apg_seeds(destination: Path) -> None:
    """Every APG case over the seeds check 1 draws (`record_case`): each seed's initial state, the
    published guider's float32 walk from it and its float64 one, the count planned on Dew's own walk
    (tests/diffusion_support.py's `apg_walked`)."""
    sys.path.insert(0, str(FIXTURE.parents[1]))
    from diffusion_support import apg_walked
    from recorded_references import provenance, record_case, save

    published = guider()
    arrays = {"cases": np.asarray(json.dumps(APG_CASES)), "label": np.asarray(LABEL)}
    for name, case in APG_CASES.items():
        def evaluate(inputs, case=case):
            x = torch.from_numpy(inputs["x_T"])
            return ({"walked": apg_walk(published, case, x).numpy()},
                    {"walked": apg_walk(published, case, x.double()).numpy()})

        def walked(inputs, case=case):
            return {"walked": apg_walked(case, LABEL, inputs["x_T"])}

        arrays.update(record_case(name, apg_draw, evaluate, walked))
    save(destination, {**arrays, "provenance": provenance(
        "diffusers", DIFFUSERS, "cpu", f"python tools/guidance_reference.py apg-seeds {destination}",
        torch=torch.__version__)}, reference=f"huggingface/diffusers {DIFFUSERS}", license="Apache-2.0")


def projection_draw(name: str):
    """A seed's conditional and unconditional raw predictions for the APG projection's case `name`:
    nearly parallel, at the case's scales (PROJECTION_CASES), from torch's generator."""
    scale, blank_scale = PROJECTION_CASES[name]

    def draw(seed: int) -> dict[str, np.ndarray]:
        generator = torch.Generator().manual_seed(seed)
        conditional = 1.0 + 0.1 * torch.randn(PROJECTION_ROWS, *SHAPE[1:], generator=generator)
        unconditional = 0.3 * conditional + 0.03 * torch.randn(conditional.shape, generator=generator)
        return {"conditional": (conditional * scale).numpy(),
                "unconditional": (unconditional * blank_scale).numpy()}
    return draw


def noise_draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's unit normal noise of SHAPE for the EDM samplers' walks, from torch's generator."""
    return {"noise": torch.randn(*SHAPE, generator=torch.Generator().manual_seed(seed)).numpy()}


def walk_seeds(destination: Path) -> None:
    """The APG projection (Diffusers' guider on raw predictions, without a norm clip or a solver's
    prediction conversion), NVlabs/edm2's autoguided walk and Kynkaanniemi et al.'s interval-guided
    walks over the seeds check 1 draws (`record_case`), each seed's float32 result and its float64 one;
    the test draws the inputs again (`projection_draw`, `noise_draw`)."""
    from recorded_references import provenance, record_case, save

    published = guider()
    arrays = {"cases": np.asarray(json.dumps(INTERVAL_CASES)),
              "strengths": np.asarray([CONDITIONAL, UNCONDITIONAL]),
              "settings": np.asarray(json.dumps(AUTOGUIDANCE))}
    for name in PROJECTION_CASES:
        def projected(inputs):
            results = []
            for dtype in (torch.float32, torch.float64):
                guidance = published(guidance_scale=6.0, adaptive_projected_guidance_rescale=0.0, eta=0.0)
                guidance.set_state(0, 1, 1000)
                guidance.prepare_inputs({})
                conditional, unconditional = (torch.from_numpy(inputs[key]).to(dtype)
                                              for key in ("conditional", "unconditional"))
                results.append({"result": guidance.forward(conditional, unconditional).pred.numpy()})
            return tuple(results)

        arrays.update(record_case(f"projection.{name}", projection_draw(name), projected, store_inputs=False))

    edm2 = definitions(EDM2, ("edm_sampler",), {"torch": torch, "np": np})["edm_sampler"]

    def autoguided(inputs):
        noise = torch.from_numpy(inputs["noise"])
        return tuple({"walked": edm2(Denoiser(AUTOGUIDANCE["strength"]), noise.to(dtype),
                                     gnet=Denoiser(AUTOGUIDANCE["guide_strength"]),
                                     num_steps=AUTOGUIDANCE["num_steps"], guidance=AUTOGUIDANCE["guidance"],
                                     dtype=dtype).numpy()} for dtype in (torch.float32, torch.float64))

    arrays.update(record_case("autoguidance", noise_draw, autoguided, store_inputs=False))
    samplers = {}
    for dtype in (torch.float32, torch.float64):
        # The run's precision wherever the sampler names one.
        names = types.SimpleNamespace(**{**vars(torch), "float64": dtype, "float32": dtype})
        samplers[dtype] = definitions(INTERVAL, ("edm_sampler",), {"torch": names, "np": np, "Any": object,
                                                                 "Optional": object})["edm_sampler"]
    for name, case in INTERVAL_CASES.items():
        def interval(inputs, case=case):
            noise = torch.from_numpy(inputs["noise"])
            return tuple({"walked": samplers[dtype](Denoiser(CONDITIONAL), noise.to(dtype),
                                                    uncond_net=Denoiser(UNCONDITIONAL), **case).numpy()}
                         for dtype in (torch.float32, torch.float64))

        arrays.update(record_case(f"interval.{name}", noise_draw, interval, store_inputs=False))
    save(destination, {**arrays, "provenance": provenance(
        "diffusers, edm2, guidance-interval", f"{DIFFUSERS}, {EDM2}, {INTERVAL}", "cpu",
        f"python tools/guidance_reference.py walk-seeds {destination}", torch=torch.__version__)},
        reference=f"huggingface/diffusers {DIFFUSERS}; NVlabs/edm2; kynkaat/guidance-interval",
        license="Apache-2.0; CC BY-NC-SA 4.0 (results only)")


class Denoiser:
    """EDM's closed-form D(x; sigma) at a strength: the main network and its
    weaker guide differ in it."""

    sigma_min, sigma_max = 0.0, float("inf")

    def __init__(self, strength: float):
        self.strength = strength

    def __call__(self, x, sigma, labels=None):
        return torch.tanh(x / torch.sqrt(1 + sigma ** 2)) * self.strength + 0.1 * torch.sin(sigma)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "apg-seeds":
        apg_seeds(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "walk-seeds":
        walk_seeds(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
