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

Each runs in float64 and in float32; what lands per case is the initial
state, the settings and both results.

    PYTHONPATH=src python tools/guidance_reference.py

    PYTHONPATH=src python tools/guidance_reference.py apg-ensemble OUTPUT.npz
    PYTHONPATH=src python tools/guidance_reference.py apg-projection OUTPUT.npz

The ensemble mode leaves the existing fixtures alone, checks every APG
array regenerates bit for bit, and records plain APG over 1024 independent
initial states. Spatial orders leave this pointwise case's fp32 result
unchanged, so more states are what resolve its RMS error.
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch

GUIDERS = ("https://raw.githubusercontent.com/huggingface/diffusers/"
           "5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2/src/diffusers/guiders/")
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
APG_SAMPLES = 1024


def extracted(url: str, names: tuple[str, ...], scope: dict) -> dict:
    """The named top-level definitions of the file at `url`, run as published."""
    text = urllib.request.urlopen(url).read().decode()
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            exec(ast.get_source_segment(text, node), scope)
    return scope


def members(url: str, owner: str, names: set[str]) -> list[str]:
    """The source of the named members of class `owner` in the file at
    `url`, decorators kept but for `register_to_config`, which only records
    the arguments for a config file."""
    text = urllib.request.urlopen(url).read().decode()
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == owner:
            for member in node.body:
                name = (member.name if isinstance(member, ast.FunctionDef)
                        else member.targets[0].id if isinstance(member, ast.Assign)
                        and isinstance(member.targets[0], ast.Name) else None)
                if name in names:
                    if isinstance(member, ast.FunctionDef):
                        member.decorator_list = [decorator for decorator in member.decorator_list
                                                 if ast.unparse(decorator) != "register_to_config"]
                    found[name] = ast.unparse(member)
    assert set(found) == names, names - set(found)
    return list(found.values())


def guider() -> type:
    """Diffusers' `AdaptiveProjectedGuidance` as published, on a
    `BaseGuidance` holding its own published `__init__` and `set_state`."""
    scope = extracted(GUIDERS + "adaptive_projected_guidance.py", ("MomentumBuffer", "normalized_guidance"),
                      {"torch": torch, "math": math})
    extracted(GUIDERS + "guider_utils.py", ("rescale_noise_cfg",), scope)
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


def apg_ensemble(destination: Path) -> None:
    """Plain APG has no order-dependent reductions. Independent initial
    states instead resolve its RMS rounding error against the same reference."""
    published = guider()
    with np.load(FIXTURE / "apg.npz") as stored:
        original = torch.from_numpy(stored["x_T"])
        for name, case in APG_CASES.items():
            for dtype, suffix in ((torch.float64, ""), (torch.float32, "32")):
                assert np.array_equal(apg_walk(published, case, original.to(dtype)).numpy(),
                                      stored[f"{name}.result{suffix}"]), name
    x = torch.randn(APG_SAMPLES, *SHAPE[1:], generator=torch.Generator().manual_seed(23))
    case = APG_CASES["plain"]
    np.savez_compressed(destination, x_T=x.numpy(), case=np.asarray(json.dumps(case)),
                        label=np.asarray(LABEL),
                        result=apg_walk(published, case, x.double()).numpy(),
                        result32=apg_walk(published, case, x).numpy())


def apg_projection(destination: Path) -> None:
    """Diffusers' projection on fixed conditional and unconditional raw
    predictions, without a norm clip or a solver's prediction conversion."""
    published = guider()
    generator = torch.Generator().manual_seed(29)
    conditional = 1.0 + 0.1 * torch.randn(1024, *SHAPE[1:], generator=generator)
    unconditional = 0.3 * conditional + 0.03 * torch.randn(conditional.shape, generator=generator)
    arrays = {}
    for name, scale, blank_scale in (("ordinary", 1.0, 1.0), ("large", 1e30, 1e30),
                                     ("tiny", 1e-20, 1e-20), ("different_scales", 1.0, 1e30)):
        conditioned, blank = conditional * scale, unconditional * blank_scale
        arrays.update({f"{name}.conditional": conditioned.numpy(), f"{name}.unconditional": blank.numpy()})
        for dtype, suffix in ((torch.float64, ""), (torch.float32, "32")):
            guidance = published(guidance_scale=6.0, adaptive_projected_guidance_rescale=0.0, eta=0.0)
            guidance.set_state(0, 1, 1000)
            guidance.prepare_inputs({})
            result = guidance.forward(conditioned.to(dtype), blank.to(dtype)).pred
            arrays[f"{name}.result{suffix}"] = result.numpy()
    if destination.is_file():
        with np.load(destination) as stored:
            assert all(np.array_equal(stored[name], arrays[name]) for name in stored.files)
    np.savez_compressed(destination, **arrays)


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


def main() -> None:
    FIXTURE.mkdir(parents=True, exist_ok=True)
    published = guider()
    x = torch.randn(*SHAPE, generator=torch.Generator().manual_seed(3))
    arrays = {"x_T": x.numpy(), "cases": np.asarray(json.dumps(APG_CASES)), "label": np.asarray(LABEL)}
    for name, case in APG_CASES.items():
        arrays[f"{name}.result"] = apg_walk(published, case, x.double()).numpy()
        arrays[f"{name}.result32"] = apg_walk(published, case, x).double().numpy()
    np.savez(FIXTURE / "apg.npz", **arrays)

    noise = torch.randn(*SHAPE, generator=torch.Generator().manual_seed(5))
    arrays = {"noise": noise.numpy(), "cases": np.asarray(json.dumps(INTERVAL_CASES)),
              "strengths": np.asarray([CONDITIONAL, UNCONDITIONAL])}
    for dtype, tail in ((torch.float64, ""), (torch.float32, "32")):
        # The run's precision wherever the sampler names one.
        names = types.SimpleNamespace(**{**vars(torch), "float64": dtype, "float32": dtype})
        sampler = extracted(INTERVAL, ("edm_sampler",), {"torch": names, "np": np, "Any": object,
                                                         "Optional": object})["edm_sampler"]
        for name, case in INTERVAL_CASES.items():
            walked = sampler(Denoiser(CONDITIONAL), noise.to(dtype), uncond_net=Denoiser(UNCONDITIONAL),
                             **case)
            arrays[f"{name}.result{tail}"] = walked.double().numpy()
    np.savez(FIXTURE / "interval.npz", **arrays)

    sampler = extracted(EDM2, ("edm_sampler",), {"torch": torch, "np": np})["edm_sampler"]
    noise = torch.randn(*SHAPE, generator=torch.Generator().manual_seed(4))
    settings = {"num_steps": 8, "guidance": 2.2, "strength": 0.5, "guide_strength": 0.35}
    walks = {}
    for dtype in (torch.float64, torch.float32):
        walks[dtype] = sampler(Denoiser(settings["strength"]), noise.to(dtype),
                               gnet=Denoiser(settings["guide_strength"]),
                               num_steps=settings["num_steps"], guidance=settings["guidance"],
                               dtype=dtype).double().numpy()
    np.savez(FIXTURE / "autoguidance.npz", noise=noise.numpy(), result=walks[torch.float64],
             result32=walks[torch.float32], settings=np.asarray(json.dumps(settings)))
    print(f"{FIXTURE}: {len(APG_CASES)} APG walks, one autoguided walk, "
          f"{len(INTERVAL_CASES)} interval-guided walks")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "apg-ensemble":
        apg_ensemble(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "apg-projection":
        apg_projection(Path(sys.argv[2]))
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit(__doc__)
