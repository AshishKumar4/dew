#!/usr/bin/env python3
"""Write the FlaxDiff noise schedules fixture with FlaxDiff's own schedulers.

FlaxDiff (github.com/AshishKumar4/FlaxDiff) is the project Dew grew out of.
Its scheduler modules, fetched at commit 15c55b001304604147a2a8002a76cf5d4c32092f
and checked against their SHA-256, build the exponential beta table
(`ExpNoiseScheduler`), the continuous cosine (`CosineContinuousNoiseScheduler`)
and the tangent grid of sigmas (`CosineGeneralNoiseScheduler`). tests/test_schedulers.py measures Dew's
against them and float64.

`python tools/flaxdiff_schedules_reference.py OUTPUT.npz` writes what the
tests compare bit for bit: the beta tables, and each case's rates, sigmas and
conditioning (`transform_inputs`, the model's integer thousandths) at 33
times from 0 to 1, with the conditioning also at CONDITIONED random times.

`python tools/flaxdiff_schedules_reference.py seeds OUTPUT.npz` records each
site over the seeds check 1 declares (`record_case`): FlaxDiff's float32
outputs and their float64 truth. The truth is FlaxDiff's own program run on
the inputs widened where it computes in its inputs' dtype (the continuous
cosine, the tangent grid), its weight's function 1 + sigma^2 / sigma_max^2 on
that grid (FlaxDiff's expression for it cancels as sigma grows, in float64
too), and the same formulas in float64 on FlaxDiff's own float64 betas where
it fixes float32 (a beta table's rates and P2 weights).

Cases:
- `exp/default`: 1000 steps at the default beta_end and P2 weights;
- `exp/clipped`: 8 steps, every beta clipped at a beta_end of 0.5, P2 weights
  of k = 2 and gamma = 0.5;
- `cosine_continuous`;
- `cosine_general/default`, `cosine_general/bounded` and `cosine_general/steep`:
  the defaults, sigma_min 0.002, sigma_max 160 and kappa 0.5, and kappa 2.

Run with the Dew test environment, on CPU.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import jax
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
                                                                     "kappa": 0.5},
           "cosine_general/steep": {"kappa": 2.0}}
CONDITIONED = 4096
"""The random times each tangent grid's conditioning is also recorded at, drawn from seed 0."""
ROWS = 64
"""The times, or a beta table's steps, of one seed's draw."""


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
    conditioned = np.random.default_rng(0).uniform(0, 1, CONDITIONED).astype(np.float32)
    arrays["conditioning_times"] = conditioned
    for case, arguments in GENERAL.items():
        general = modules["cosine"].CosineGeneralNoiseScheduler(**arguments)
        signal, noise = general.get_rates(steps, shape=(-1,))
        arrays.update({f"{case}/sigma": np.asarray(general.get_sigmas(steps)),
                       f"{case}/signal": np.asarray(signal), f"{case}/noise": np.asarray(noise),
                       f"{case}/conditioning": np.asarray(general.transform_inputs(None, steps)[1]),
                       f"{case}/conditioning_drawn": np.asarray(
                           general.transform_inputs(None, jnp.asarray(conditioned))[1])})
    saved_fixture(destination, arrays, {"repo": REPO, "commit": COMMIT, "files": FILES, "exp": EXP,
                                        "general": GENERAL})


def time_draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's ROWS times, uniform on [0, 1)."""
    return {"t": np.random.default_rng(seed).uniform(0, 1, ROWS).astype(np.float32)}


def evaluated(evaluate):
    """`evaluate(steps)`'s outputs at a draw's times as FlaxDiff computes them, and their float64
    truth, the same program on the times widened."""
    def outputs(inputs):
        reference = jax.tree.map(np.asarray, evaluate(jnp.asarray(inputs["t"])))
        with jax.enable_x64(new_val=True):
            truth = jax.tree.map(np.asarray, evaluate(jnp.asarray(inputs["t"].astype(np.float64))))
        return reference, truth

    return outputs


def table_site(module, arguments: dict):
    """An exponential beta table's site: a draw of ROWS of its steps, FlaxDiff's float32 rates and P2
    weights at them, and the float64 truth, its formulas on FlaxDiff's own float64 betas."""
    scheduler = module.ExpNoiseScheduler(**arguments)
    timesteps = arguments["timesteps"]
    k, gamma = arguments.get("p2_loss_weight_k", 1), arguments.get("p2_loss_weight_gamma", 1)
    bars = np.cumprod(1 - module.exp_beta_schedule(timesteps, 0.008, arguments.get("beta_end", 0.999)))
    truth = {"sqrt_alpha_cumprod": np.sqrt(bars), "sqrt_one_minus_alpha_cumprod": np.sqrt(1 - bars),
             "p2_loss_weights": (k + bars / (1 - bars)) ** -gamma}

    def draw(seed):
        # Every step of the table, as the table's comparison read it whole, and a seed's ROWS more.
        drawn = np.random.default_rng(seed).integers(0, timesteps, ROWS)
        return {"index": np.concatenate([np.arange(timesteps), drawn]).astype(np.int32)}

    def evaluate(inputs):
        index = inputs["index"]
        return ({name: np.asarray(getattr(scheduler, name))[index] for name in truth},
                {name: values[index] for name, values in truth.items()})

    return draw, evaluate


def seeds(destination: Path) -> None:
    """Every site over the seeds check 1 declares (`record_case`), with its provenance."""
    from recorded_references import provenance, record_case, save

    modules = schedulers()
    # tests/ is on the path once recorded_references is imported.
    from diffusion_support import flaxdiff_times

    arrays = {}
    for case, arguments in EXP.items():
        arrays.update(record_case(case, *table_site(modules["exp"], arguments)))
    continuous = modules["cosine"].CosineContinuousNoiseScheduler()

    def rates(steps):
        alpha, sigma = continuous.get_rates(steps, shape=(-1,))
        return {"alpha": alpha, "sigma": sigma, "weight": continuous.get_schedule_weights(steps, shape=(-1,))}

    arrays.update(record_case("cosine_continuous", time_draw, evaluated(rates)))
    for case, arguments in GENERAL.items():
        general = modules["cosine"].CosineGeneralNoiseScheduler(**arguments)

        def grid(inputs, general=general):
            """FlaxDiff's float32 sigmas and weights at a draw's times, and their float64 truth: its grid
            in float64, and its weight's function there, 1 + sigma^2 / sigma_max^2, without the
            cancellation its own expression has (1 / (1 + (1 - sigma^2) / sigma^2) for sigma^2)."""
            steps = jnp.asarray(inputs["t"])
            reference = {"sigma": np.asarray(general.get_sigmas(steps)),
                         "weight": np.asarray(general.get_schedule_weights(steps, shape=(-1,)))}
            with jax.enable_x64(new_val=True):
                sigma = np.asarray(general.get_sigmas(jnp.asarray(inputs["t"].astype(np.float64))))
            return reference, {"sigma": sigma, "weight": 1 + sigma ** 2 / general.sigma_max ** 2}

        # The evidence alone, over draws the test makes again (`flaxdiff_times`), as their outputs would
        # take megabytes.
        arrays.update(record_case(case, flaxdiff_times, grid, store_inputs=False, summary=True))
    save(destination, {**arrays, "provenance": provenance(
        REPO, COMMIT, "cpu", f"python tools/flaxdiff_schedules_reference.py seeds {destination}")},
        reference=f"{REPO} {COMMIT}", license="MIT")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "seeds":
        seeds(Path(sys.argv[2]))
    else:
        main(Path(sys.argv[1]))
