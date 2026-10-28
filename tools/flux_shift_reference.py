"""Flux's resolution shift as Diffusers computes it, for
tests/fixtures/flow/resolution_shift.npz.

`calculate_shift` of Diffusers 0.34.0's
`diffusers/pipelines/flux/pipeline_flux.py`, run as published: mu linear in
the packed latent's token count, read out of the installed module's source
(importing the pipeline module needs a transformers release older than
Dew's). Each set of constants is a pipeline's scheduler configuration:
Flux's (0.5 at 256 tokens to 1.15 at 4096) and one with another slope and
span (0.5 at 256 to 0.9 at 8192). mu is Python's double. The noise level a
time maps to under exp(mu) is Diffusers' scheduler's own `time_shift`, which
the test runs over drawn times.

    python tools/flux_shift_reference.py
"""

from __future__ import annotations

import inspect
from pathlib import Path

import diffusers
import numpy as np
from diffusers.pipelines import flux
from reference_source import defined_in

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "flow" / "resolution_shift.npz"
CONSTANTS = {"flux": {"base_seq_len": 256, "max_seq_len": 4096, "base_shift": 0.5, "max_shift": 1.15},
             "long": {"base_seq_len": 256, "max_seq_len": 8192, "base_shift": 0.5, "max_shift": 0.9}}
TOKENS = (1, 4, 16, 64, 256, 1024, 2304, 4096, 9216, 16384)


def calculate_shift():
    """The published `calculate_shift`, defined from the installed source."""
    path = Path(inspect.getfile(flux)).parent / "pipeline_flux.py"
    return defined_in(path.read_text(), {"calculate_shift"}, {}, filename=str(path))["calculate_shift"]


def main() -> None:
    shift = calculate_shift()
    arrays = {"tokens": np.asarray(TOKENS), "diffusers_version": np.array(diffusers.__version__)}
    for name, constants in CONSTANTS.items():
        arrays[f"{name}/constants"] = np.asarray([constants[key] for key in sorted(constants)])
        arrays[f"{name}/mu"] = np.asarray([shift(tokens, **constants) for tokens in TOKENS], np.float64)
    arrays["constant_names"] = np.asarray(sorted(CONSTANTS["flux"]))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"{FIXTURE}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
