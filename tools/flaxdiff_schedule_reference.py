"""FlaxDiff's exponential and cosine noise schedules, recorded for tests/fixtures/flaxdiff_schedules.

Dew's `ExpNoiseScheduler`, `CosineContinuousNoiseScheduler` and
`CosineGeneralNoiseScheduler` are FlaxDiff's. The reference is
AshishKumar4/FlaxDiff at a pinned commit: `exp_beta_schedule` and
`ExpNoiseScheduler` (flaxdiff/schedulers/exp.py), the two cosine schedules of
flaxdiff/schedulers/cosine.py, and the `NoiseScheduler`,
`GeneralizedNoiseScheduler`, `ContinuousNoiseScheduler` and
`DiscreteNoiseScheduler` they extend, read out of the published files and run
as written. Each case runs twice: as FlaxDiff runs, JAX's float32 with x64 off
(its beta tables' cumulative products then in float32), and wide, x64 on and
the source's `jnp.float32` read as `jnp.float64`, which is the float64 truth.

Each case's rates and loss weights land over the seeds check 1 draws
(`record_case`, tools/recorded_references.py): an exponential table's
indices, and the continuous schedules' times in [0, 1]. The generalized
cosine's draws are large (`WIDE_ROWS`), so its cases keep FlaxDiff's evidence
alone (`summary`), their times drawn again by `general_times` and their
float64 truth Dew's own float64 path, held to the recorded truth's sketch
(`indexed_recording`). Beside them land
each exponential table's betas, which FlaxDiff computes in float64 on the
host, and the generalized cosine's model input at a fixed grid of times,
FlaxDiff's `transform_inputs`, an integer.

    PYTHONPATH=src python tools/flaxdiff_schedule_reference.py tests/fixtures/flaxdiff_schedules/recorded.npz
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

COMMIT = "15c55b001304604147a2a8002a76cf5d4c32092f"
FLAXDIFF = f"https://raw.githubusercontent.com/AshishKumar4/FlaxDiff/{COMMIT}/flaxdiff/"
ROWS = 64
"""A drawn batch's rows."""
WIDE_ROWS = 1 << 16
"""A generalized cosine's drawn rows: its tangent's conditioning grows toward t = 1, so each seed's float32
error is a few rows' near there, which a small batch leaves too varied for check 1's seeds to resolve, and
with 4096 rows the kappa case asked 128 seeds, whose evidence passed what git holds."""
EXP = {"exp-1000": {"timesteps": 1000},
       "exp-250-clipped": {"timesteps": 250, "beta_end": 0.02, "p2_loss_weight_k": 2,
                           "p2_loss_weight_gamma": 0.5}}
"""The exponential cases, FlaxDiff's `ExpNoiseScheduler` arguments: its defaults at 1000 steps, and 250
steps whose betas, 1 - exp(-12 / 250) = 0.0469, a `beta_end` of 0.02 clips, with a nondefault P2 weight."""
GENERAL = {"general-cosine": {},
           "general-cosine-kappa": {"sigma_min": 0.002, "sigma_max": 160.0, "kappa": 0.5}}
"""The generalized cosine cases, FlaxDiff's `CosineGeneralNoiseScheduler` arguments: its defaults, and
nondefault ends and kappa."""
GRID = np.linspace(0, 1, 33, dtype=np.float32)
"""The times the generalized cosine's model input is recorded at."""
NARROW: dict = {}
WIDE: dict = {}
"""FlaxDiff's classes as it runs them and in float64, read when a recording starts (`published`)."""


def general_times(site: str, seed: int) -> dict[str, np.ndarray]:
    """A generalized cosine case's times for `seed`, `WIDE_ROWS` of them uniform in [0, 1)."""
    drawn = np.random.default_rng([0xC0, 1 + list(GENERAL).index(site), seed]).uniform(0, 1, WIDE_ROWS)
    return {"steps": drawn.astype(np.float32)}


def statements(path: str, names: set[str], *, wide: bool):
    """The definitions `names` of FlaxDiff's `path` (`selected`), in float64 where `wide`."""
    from reference_source import fetched, selected

    text = fetched(FLAXDIFF + path)
    if wide:
        text = text.replace("jnp.float32", "jnp.float64")
    return selected(text, names, filename=f"flaxdiff/{path}")


def published(*, wide: bool) -> dict:
    """FlaxDiff's schedule classes, run as published in one namespace, in float64 where `wide`."""
    from reference_source import ran

    scope: dict = {"jax": jax, "jnp": jnp, "np": np, "math": math, "struct": struct, "Union": object}
    ran(statements("utils.py", {"MarkovState", "RandomMarkovState"}, wide=wide), scope, "flaxdiff/utils.py")
    sources = {"common": {"get_coeff_shapes_tuple", "reshape_rates", "NoiseScheduler",
                          "GeneralizedNoiseScheduler"},
               "continuous": {"ContinuousNoiseScheduler"}, "discrete": {"DiscreteNoiseScheduler"},
               "cosine": {"CosineGeneralNoiseScheduler", "CosineContinuousNoiseScheduler"},
               "exp": {"exp_beta_schedule", "ExpNoiseScheduler"}}
    for stem, names in sources.items():
        path = f"schedulers/{stem}.py"
        ran(statements(path, names, wide=wide), scope, f"flaxdiff/{path}")
    return scope


def evaluated(build, read, inputs) -> tuple[dict, dict]:
    """`read(schedule, inputs)` as FlaxDiff runs, x64 off, and wide, x64 on and its source in float64,
    each schedule built by `build(scope)` in its own run."""
    with jax.enable_x64(new_val=False):
        narrow = read(build(NARROW), inputs)
        reference = {name: np.asarray(value) for name, value in narrow.items()}
    wide_inputs = {name: value.astype(np.float64) if value.dtype == np.float32 else value
                   for name, value in inputs.items()}
    truth = {name: np.asarray(value, np.float64) for name, value in read(build(WIDE), wide_inputs).items()}
    return reference, truth


def rates_and_weight(schedule, inputs) -> dict:
    """A schedule's rates and loss weight at the drawn steps, as FlaxDiff's loss reads them."""
    steps = jnp.asarray(inputs["steps"])
    alpha, sigma = schedule.get_rates(steps, shape=(-1,))
    return {"alpha": alpha, "sigma": sigma, "weight": schedule.get_schedule_weights(steps, shape=(-1,))}


def sigma_and_weight(schedule, inputs) -> dict:
    """A generalized schedule's sigma and loss weight at the drawn times."""
    steps = jnp.asarray(inputs["steps"])
    return {"sigma": schedule.get_sigmas(steps),
            "weight": schedule.get_schedule_weights(steps, shape=(-1,))}


def record(destination: Path) -> None:
    """Every case over the seeds check 1 draws, its betas or model inputs beside them."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from recorded_references import provenance, record_case, save

    jax.config.update("jax_enable_x64", val=True)
    NARROW.update(published(wide=False))
    WIDE.update(published(wide=True))

    arrays: dict[str, np.ndarray] = {}
    for site, (name, arguments) in enumerate(EXP.items()):
        def indices(seed, site=site, steps=arguments["timesteps"]):
            drawn = np.random.default_rng([0xE7, site, seed]).integers(0, steps, ROWS)
            return {"steps": drawn.astype(np.int32)}

        def build(scope, arguments=arguments):
            return scope["ExpNoiseScheduler"](**arguments)

        arrays.update(record_case(name, indices,
                                  lambda inputs, build=build: evaluated(build, rates_and_weight, inputs)))
        with jax.enable_x64(new_val=False):
            arrays[f"{name}.betas"] = np.asarray(NARROW["exp_beta_schedule"](
                arguments["timesteps"], 0.008, arguments.get("beta_end", 0.999)))

    def times(seed):
        return {"steps": np.random.default_rng([0xC0, 0, seed]).uniform(0, 1, ROWS).astype(np.float32)}

    arrays.update(record_case("continuous-cosine", times, lambda inputs: evaluated(
        lambda scope: scope["CosineContinuousNoiseScheduler"](), rates_and_weight, inputs)))
    for name, arguments in GENERAL.items():
        def build(scope, arguments=arguments):
            return scope["CosineGeneralNoiseScheduler"](**arguments)

        arrays.update(record_case(name, lambda seed, name=name: general_times(name, seed),
                                  lambda inputs, build=build: evaluated(build, sigma_and_weight, inputs),
                                  store_inputs=False, summary=True))
        with jax.enable_x64(new_val=False):
            _, chunks = build(NARROW).transform_inputs(None, jnp.asarray(GRID))
        arrays[f"{name}.model_time"] = np.asarray(chunks)
    arrays["grid"] = GRID
    arrays["cases"] = np.asarray(json.dumps({**EXP, **GENERAL}))
    save(destination, {**arrays, "provenance": provenance(
        "flaxdiff", COMMIT, "cpu", f"python tools/flaxdiff_schedule_reference.py {destination}",
        jax=jax.__version__)}, reference=f"AshishKumar4/FlaxDiff {COMMIT}", license="MIT")


if __name__ == "__main__":
    record(Path(sys.argv[1]))
