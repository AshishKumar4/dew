#!/usr/bin/env python3
"""Write the FlaxDiff noise schedules fixture with FlaxDiff's own schedulers.

FlaxDiff (github.com/AshishKumar4/FlaxDiff) is the project Dew grew out of.
Its scheduler modules, fetched at commit 15c55b001304604147a2a8002a76cf5d4c32092f
and checked against their SHA-256, build the exponential beta table
(`ExpNoiseScheduler`), the continuous cosine (`CosineContinuousNoiseScheduler`)
and the tangent grid of sigmas (`CosineGeneralNoiseScheduler`), whose tables,
rates, sigmas and weights the tool records in float32, as FlaxDiff computes
them (tests/test_diffusion_schedules_flaxdiff.py measures Dew's against them
and float64).

Cases:
- `exp/default`: 1000 steps at the default beta_end and P2 weights;
- `exp/clipped`: 8 steps, every beta clipped at a beta_end of 0.5, P2 weights
  of k = 2 and gamma = 0.5;
- `cosine_continuous`: 33 times from 0 to 1;
- `cosine_general/default` and `cosine_general/bounded`: the same times, at
  the defaults and at sigma_min 0.002, sigma_max 160 and kappa 0.5.

Run with the Dew test environment, on CPU:

    python tools/flaxdiff_schedules_reference.py OUTPUT.npz
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from reference_source import loaded, package, published_files, saved_fixture

REPO, COMMIT = "AshishKumar4/FlaxDiff", "15c55b001304604147a2a8002a76cf5d4c32092f"
# In import order: each module's own imports come before it.
FILES = {
    "common": "2b7e643c25a09dedd5489b7c62c7c18fb48a5dafb277b1d7aebdd34b7a9748b0",
    "discrete": "a5d033246451b754e93a06b81cd6c11ac2d5ae4af74036cc031110853f58ae7d",
    "continuous": "b4678c37406a6d18bc3e4eeb062d5f3aeb733f7d6040c8e4b7a7617252bf5ad4",
    "cosine": "80f0b571caf1b6f1197106bf7a9010be167a9ec9faee661b4b6e117fb801209a",
    "exp": "42439005dd48a876710af7cae67c98fdbd07e7aea8d59ac828e34f721d86d683",
}
TIMES = np.linspace(0, 1, 33, dtype=np.float32)
EXP = {"exp/default": {"timesteps": 1000},
       "exp/clipped": {"timesteps": 8, "beta_end": 0.5, "p2_loss_weight_k": 2, "p2_loss_weight_gamma": 0.5}}
GENERAL = {"cosine_general/default": {}, "cosine_general/bounded": {"sigma_min": 0.002, "sigma_max": 160.0,
                                                                     "kappa": 0.5}}


def schedulers() -> dict[str, types.ModuleType]:
    """FlaxDiff's scheduler modules at `COMMIT`, each under its own name in the package `flaxdiff`,
    their one import from beyond them (`..utils.RandomMarkovState`, a type their timestep draws name)
    a stand-in, as none of the tables reads it."""
    root = published_files(REPO, COMMIT, {f"flaxdiff/schedulers/{name}.py": digest
                                          for name, digest in FILES.items()})
    package("flaxdiff", root / "flaxdiff")
    utils = types.ModuleType("flaxdiff.utils")
    utils.RandomMarkovState = type("RandomMarkovState", (), {})
    sys.modules["flaxdiff.utils"] = utils
    return {name: loaded(f"flaxdiff.schedulers.{name}", root / "flaxdiff" / "schedulers" / f"{name}.py")
            for name in FILES}


def main(destination: Path) -> None:
    modules = schedulers()
    arrays: dict[str, np.ndarray] = {"times": TIMES}
    for case, arguments in EXP.items():
        scheduler = modules["exp"].ExpNoiseScheduler(**arguments)
        for name in ("betas", "sqrt_alpha_cumprod", "sqrt_one_minus_alpha_cumprod", "p2_loss_weights"):
            arrays[f"{case}/{name}"] = np.asarray(getattr(scheduler, name))
    continuous = modules["cosine"].CosineContinuousNoiseScheduler()
    steps = jnp.asarray(TIMES)
    alpha, sigma = continuous.get_rates(steps, shape=(-1,))
    weight = continuous.get_schedule_weights(steps, shape=(-1,))
    arrays.update({"cosine_continuous/alpha": np.asarray(alpha), "cosine_continuous/sigma": np.asarray(sigma),
                   "cosine_continuous/weight": np.asarray(weight)})
    for case, arguments in GENERAL.items():
        general = modules["cosine"].CosineGeneralNoiseScheduler(**arguments)
        signal, noise = general.get_rates(steps, shape=(-1,))
        arrays.update({f"{case}/sigma": np.asarray(general.get_sigmas(steps)),
                       f"{case}/signal": np.asarray(signal), f"{case}/noise": np.asarray(noise)})
    saved_fixture(destination, arrays, {"repo": REPO, "commit": COMMIT, "files": FILES, "exp": EXP,
                                        "general": GENERAL})


if __name__ == "__main__":
    main(Path(sys.argv[1]))
