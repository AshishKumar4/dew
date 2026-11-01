"""EDM's own sampler (Karras et al. 2022, Algorithm 2), for tests/fixtures/edm/heun.npz.

The reference is `edm_sampler` in NVlabs/edm's `generate.py`, read at a
pinned commit and run as published: its code is not vendored here (the
repository's licence is CC BY-NC-SA 4.0), only the function is extracted and
executed. The network is a closed-form denoiser both sides compute, so the
trajectory depends on the sampler alone. The fresh noise each step draws is
the draw Dew's `sample` makes at that step, `jax.random.normal` under the
walk's key folded with the step index, fed to the reference as its
`randn_like`, so both integrate one path.

What lands per case, over the seeds check 1 draws (`record_case`,
tools/recorded_references.py): the sampler's arguments and its result, as
published in float64 (the truth) and by its float32 twin (the reference). A
seed's inputs, the walk's key and the initial latent, are a seeded draw
(`draw`) the test makes again.

    PYTHONPATH=src python tools/edm_reference.py
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import jax
import numpy as np
import torch
from recorded_draws import Tape
from reference_source import defined_in, fetched

COMMIT = "008a4e5316c8e3bfe61a62f874bddba254295afb"
SOURCE = f"https://raw.githubusercontent.com/NVlabs/edm/{COMMIT}/generate.py"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "edm" / "heun.npz"
SHAPE = (2, 3, 4)
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
    text = fetched(SOURCE).replace("torch.float64", f"torch.{dtype}")
    return defined_in(text, {"edm_sampler"}, {"torch": torch, "np": np}, filename=SOURCE)["edm_sampler"]


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's inputs: the walk's key, `jax.random.PRNGKey(seed)`, and the initial latent at unit
    sigma, a seeded NumPy draw the walk scales by its top sigma."""
    return {"key": np.asarray(jax.random.PRNGKey(seed)),
            "latents": np.random.default_rng([0xED, seed]).standard_normal(SHAPE).astype(np.float32)}


def main() -> None:
    jax.config.update("jax_platforms", "cpu")
    from recorded_references import provenance, record_case, save

    samplers = {dtype: edm_sampler(dtype) for dtype in ("float32", "float64")}

    def evaluated(inputs, arguments):
        """The float32 twin's result and the published float64 one, both walking the noise Dew's
        `sample` draws at each step under the seed's key."""
        noises = [np.asarray(jax.random.normal(jax.random.fold_in(inputs["key"], index), SHAPE))
                  for index in range(arguments["num_steps"])]
        results = []
        for dtype in ("float32", "float64"):
            tape = Tape(noises)
            result = samplers[dtype](Net(), torch.from_numpy(inputs["latents"]),
                                     randn_like=lambda x, tape=tape: torch.from_numpy(
                                         tape.next(shape=x.shape)).to(x.dtype), **arguments)
            results.append({"result": result.numpy()})
        return tuple(results)

    arrays: dict[str, np.ndarray] = {}
    for name, arguments in CASES.items():
        arrays.update(record_case(name, draw, functools.partial(evaluated, arguments=arguments),
                                  store_inputs=False))
        arrays[f"{name}.arguments"] = np.asarray(json.dumps(arguments))
    written = save(FIXTURE, {**arrays, "provenance": provenance(
        "NVlabs/edm", COMMIT, "cpu", "PYTHONPATH=src python tools/edm_reference.py",
        torch=torch.__version__)}, reference="NVlabs/edm", license="CC BY-NC-SA 4.0 (results only)")
    print(f"{written}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
