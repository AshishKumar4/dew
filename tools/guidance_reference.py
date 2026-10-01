"""Guided walks against their published references, for tests/fixtures/guidance.

- APG: Diffusers' `normalized_guidance` and `MomentumBuffer`
  (`src/diffusers/guiders/adaptive_projected_guidance.py`, read at a pinned
  commit, the two definitions extracted and run as published) combine a
  closed-form flow model's conditional and unconditional velocities at every
  step of an Euler walk on the linear path.
- Autoguidance: NVlabs/edm2's `edm_sampler` (`generate_images.py`, pinned,
  run as published, not vendored: CC BY-NC-SA 4.0) walks EDM's Heun sampler
  with a closed-form main network and a weaker one as `gnet`.

Both run in float64 and in float32; what lands per case is the initial
state, the settings and both results.

    PYTHONPATH=src python tools/guidance_reference.py
"""

from __future__ import annotations

import ast
import itertools
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

DIFFUSERS = ("https://raw.githubusercontent.com/huggingface/diffusers/"
             "5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2/src/diffusers/guiders/adaptive_projected_guidance.py")
EDM2 = ("https://raw.githubusercontent.com/NVlabs/edm2/"
        "4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc/generate_images.py")
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "guidance"
SHAPE = (2, 3, 4)
APG_CASES = {
    "plain": {"scale": 4.0, "eta": 1.0, "norm_threshold": 0.0, "momentum": 0.0},
    "projected": {"scale": 6.0, "eta": 0.0, "norm_threshold": 0.8, "momentum": 0.0},
    "momentum": {"scale": 6.0, "eta": 0.2, "norm_threshold": 0.8, "momentum": -0.5},
}
STEPS = 6
LABEL = 0.7


def extracted(url: str, names: tuple[str, ...], scope: dict) -> dict:
    """The named top-level definitions of the file at `url`, run as published."""
    text = urllib.request.urlopen(url).read().decode()
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            exec(ast.get_source_segment(text, node), scope)
    return scope


def velocity(x, time, label):
    """The flow model the APG walk and `tests` run: the scaled input's
    nonlinearity, the model time and the condition."""
    return torch.sin(x) * 0.07 + time * 0.001 + label


def apg_walk(reference: dict, case: dict, x: torch.Tensor) -> torch.Tensor:
    buffer = reference["MomentumBuffer"](case["momentum"]) if case["momentum"] else None
    sigmas = torch.linspace(1.0, 0.0, STEPS, dtype=x.dtype)
    for sigma, following in itertools.pairwise(sigmas):
        conditional = velocity(x, sigma * 1000, LABEL)
        unconditional = velocity(x, sigma * 1000, 0.0)
        guided = reference["normalized_guidance"](
            conditional, unconditional, case["scale"], buffer, case["eta"], case["norm_threshold"])
        x = x + (following - sigma) * guided
    return x


class Denoiser:
    """EDM's closed-form D(x; sigma) at a strength: the main network and its
    weaker guide differ in it."""

    sigma_min, sigma_max = 0.0, float("inf")

    def __init__(self, strength: float):
        self.strength = strength

    def __call__(self, x, sigma, labels=None):
        return torch.tanh(x / torch.sqrt(1 + sigma ** 2)) * self.strength + 0.1 * torch.sin(sigma)


def main() -> None:
    FIXTURE.mkdir(parents=True, exist_ok=True)
    reference = extracted(DIFFUSERS, ("MomentumBuffer", "normalized_guidance"), {"torch": torch})
    x = torch.randn(*SHAPE, generator=torch.Generator().manual_seed(3))
    arrays = {"x_T": x.numpy(), "cases": np.asarray(json.dumps(APG_CASES)),
              "steps": np.asarray(STEPS), "label": np.asarray(LABEL)}
    for name, case in APG_CASES.items():
        arrays[f"{name}.result"] = apg_walk(reference, case, x.double()).numpy()
        arrays[f"{name}.result32"] = apg_walk(reference, case, x).double().numpy()
    np.savez(FIXTURE / "apg.npz", **arrays)

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
    print(f"{FIXTURE}: {len(APG_CASES)} APG walks, one autoguided walk")


if __name__ == "__main__":
    main()
