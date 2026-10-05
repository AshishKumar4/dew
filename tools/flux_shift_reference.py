"""Flux's resolution shift as Diffusers computes it, for
tests/fixtures/flow/resolution_shift.npz.

Two functions of Diffusers 0.34.0, run as published: `calculate_shift` of
`diffusers/pipelines/flux/pipeline_flux.py`, mu linear in the packed
latent's token count, read out of the installed module's source (importing
the pipeline module needs a transformers release older than Dew's), and
`FlowMatchEulerDiscreteScheduler._time_shift_exponential`, the noise level
a time maps to under exp(mu). Each set of constants is a pipeline's
scheduler configuration: Flux's (0.5 at 256 tokens to 1.15 at 4096) and one
with another slope and span (0.5 at 256 to 0.9 at 8192). mu is Python's
double; the shifted noise levels are computed on a float32 grid of times as
the scheduler computes them, and on the same grid widened to float64.

    python tools/flux_shift_reference.py
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import diffusers
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines import flux

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "flow" / "resolution_shift.npz"
CONSTANTS = {"flux": {"base_seq_len": 256, "max_seq_len": 4096, "base_shift": 0.5, "max_shift": 1.15},
             "long": {"base_seq_len": 256, "max_seq_len": 8192, "base_shift": 0.5, "max_shift": 0.9}}
TOKENS = (1, 4, 16, 64, 256, 1024, 2304, 4096, 9216, 16384)


def calculate_shift():
    """The published `calculate_shift`, defined from the installed source."""
    path = Path(inspect.getfile(flux)).parent / "pipeline_flux.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "calculate_shift"]
    scope: dict = {}
    exec(compile(tree, str(path), "exec"), scope)
    return scope["calculate_shift"]


def main() -> None:
    shift = calculate_shift()
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    times = np.arange(1, 100, dtype=np.float32) / np.float32(100)
    arrays = {"tokens": np.asarray(TOKENS), "times": times,
              "diffusers_version": np.array(diffusers.__version__)}
    for name, constants in CONSTANTS.items():
        arrays[f"{name}/constants"] = np.asarray([constants[key] for key in sorted(constants)])
        mus = np.asarray([shift(tokens, **constants) for tokens in TOKENS], np.float64)
        arrays[f"{name}/mu"] = mus
        for tail, grid in (("", times), ("_f64", times.astype(np.float64))):
            arrays[f"{name}/sigmas{tail}"] = np.stack([scheduler._time_shift_exponential(mu, 1.0, grid)
                                                      for mu in mus])
    arrays["constant_names"] = np.asarray(sorted(CONSTANTS["flux"]))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"{FIXTURE}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
