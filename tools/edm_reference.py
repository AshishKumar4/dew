"""EDM's own sampler (Karras et al. 2022, Algorithm 2), for tests/fixtures/edm.

The reference is `edm_sampler` in NVlabs/edm's `generate.py`, read at a
pinned commit and run as published: its code is not vendored here (the
repository's licence is CC BY-NC-SA 4.0), only the function is extracted and
executed. The network is a closed-form denoiser both sides compute, so the
trajectory depends on the sampler alone. The fresh noise each step draws is
the draw Dew's `sample` makes at that step, `jax.random.normal` under the
walk's key folded with the step index, fed to the reference as its
`randn_like`, so both integrate one path.

What lands per case: the sampler's arguments, the initial latent, and the
result computed in float64 and in float32.

    PYTHONPATH=src python tools/edm_reference.py
"""

from __future__ import annotations

import ast
import json
import os
import urllib.request
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np
import torch

COMMIT = "008a4e5316c8e3bfe61a62f874bddba254295afb"
SOURCE = f"https://raw.githubusercontent.com/NVlabs/edm/{COMMIT}/generate.py"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "edm" / "heun.npz"
SHAPE = (2, 3, 4)
KEY = 7
CASES = {
    "deterministic": {"num_steps": 8},
    # The ImageNet-64 settings of the paper's Table 5.
    "churn": {"num_steps": 8, "S_churn": 40, "S_min": 0.05, "S_max": 50, "S_noise": 1.003},
    # Churn down to sigma_min, below the level whose raise would pass sigma_max.
    "churn_to_the_end": {"num_steps": 6, "S_churn": 10, "S_max": 40, "S_noise": 1.0},
}


class Net:
    sigma_min, sigma_max = 0.0, float("inf")

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def __call__(self, x, sigma, class_labels=None):
        return torch.tanh(x / torch.sqrt(1 + sigma ** 2)) * 0.5 + 0.1 * torch.sin(sigma)


def edm_sampler(dtype: str):
    """The published function, extracted from the pinned `generate.py`,
    computing in `dtype`: as published at float64, and its float32 twin with
    the two casts it makes retargeted, whose distance from the first is the
    rounding error a float32 walk of the same algorithm makes."""
    text = urllib.request.urlopen(SOURCE).read().decode()
    tree = ast.parse(text)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "edm_sampler")
    scope = {"torch": torch, "np": np}
    exec(ast.get_source_segment(text, function).replace("torch.float64", f"torch.{dtype}"), scope)
    return scope["edm_sampler"]


def main() -> None:
    key = jax.random.PRNGKey(KEY)
    latents = np.asarray(jax.random.normal(jax.random.fold_in(key, 10_000), SHAPE), np.float32)
    arrays: dict[str, np.ndarray] = {"latents": latents}
    for dtype, suffix in (("float64", "result"), ("float32", "result32")):
        sampler = edm_sampler(dtype)
        for name, arguments in CASES.items():
            noises = iter([np.asarray(jax.random.normal(jax.random.fold_in(key, index), SHAPE))
                           for index in range(arguments["num_steps"])])
            result = sampler(Net(), torch.from_numpy(latents),
                             randn_like=lambda x: torch.from_numpy(next(noises)).to(x.dtype),
                             **arguments)
            arrays[f"{name}.{suffix}"] = result.numpy()
            arrays[f"{name}.arguments"] = np.asarray(json.dumps(arguments))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"{FIXTURE}: {len(CASES)} cases")


if __name__ == "__main__":
    main()
