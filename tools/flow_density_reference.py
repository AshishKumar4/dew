"""Diffusers' rectified-flow training-time densities, for
tests/fixtures/flow/densities.npz.

The reference is `compute_density_for_timestep_sampling` in Diffusers
0.34.0's `training_utils`, the SD3 densities of Esser et al. 2024 (section
3.1) as Diffusers' SD3, Flux and Lumina training scripts draw them:
`logit_normal`, `mode` and the uniform fallback. It is called as published,
except that its one random draw is the draw Dew's
`FlowMatchingScheduler.sample_t` makes from `jax.random.key(0)`: a standard
normal for `logit_normal`, which `torch.normal(mean, std)` returns as
mean + std * normal, and a uniform for the others. Each case runs on that
draw in float32 and in float64 (the same draw, widened), whose distance
from the first is the float32 run's rounding.

    PYTHONPATH=src python tools/flow_density_reference.py
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import diffusers
import jax
import numpy as np
import torch
from diffusers.training_utils import compute_density_for_timestep_sampling

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "flow" / "densities.npz"
DRAWS = 4096
CASES = {
    "logit_normal": {"weighting_scheme": "logit_normal", "logit_mean": 0.0, "logit_std": 1.0},
    "logit_normal_shifted": {"weighting_scheme": "logit_normal", "logit_mean": 0.5, "logit_std": 0.8},
    "mode": {"weighting_scheme": "mode", "mode_scale": 1.29},
    "mode_negative": {"weighting_scheme": "mode", "mode_scale": -0.5},
    "uniform": {"weighting_scheme": "none"},
}
"""Each case's arguments, Diffusers' names for FlowMatchingScheduler's
density, logit_mean, logit_std and mode_scale."""


@contextlib.contextmanager
def drawn(base: torch.Tensor):
    """`torch.normal(mean, std, size)` and `torch.rand(size)` return `base`'s values."""
    normal, rand = torch.normal, torch.rand
    torch.normal = lambda mean, std, size, **_: mean + std * base
    torch.rand = lambda size, **_: base
    try:
        yield
    finally:
        torch.normal, torch.rand = normal, rand


def main() -> None:
    key = jax.random.key(0)
    bases = {"normal": np.asarray(jax.random.normal(key, (DRAWS,), dtype=jax.numpy.float32)),
             "uniform": np.asarray(jax.random.uniform(key, (DRAWS,), dtype=jax.numpy.float32))}
    arrays = {f"base/{name}": value for name, value in bases.items()}
    for case, arguments in CASES.items():
        base = bases["normal" if arguments["weighting_scheme"] == "logit_normal" else "uniform"]
        for dtype, tail in ((torch.float32, ""), (torch.float64, "_f64")):
            with drawn(torch.tensor(base, dtype=dtype)):
                arrays[f"{case}{tail}"] = compute_density_for_timestep_sampling(
                    batch_size=DRAWS, **arguments).numpy()
    arrays["diffusers_version"] = np.array(diffusers.__version__)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"{FIXTURE}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
