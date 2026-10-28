"""FlaxDiff's RK4 and multistep samplers, for tests/fixtures/flaxdiff_solvers/solvers.npz.

Dew's `RK4` and `MultiStepDPM` are FlaxDiff's `RK4Sampler` and
`MultiStepDPM`. The reference is AshishKumar4/FlaxDiff at a pinned commit:
`DiffusionSampler.sample_step` from flaxdiff/samplers/common.py, the two
samplers' classes, and `KarrasVENoiseScheduler` with the
`GeneralizedNoiseScheduler` and `NoiseScheduler` it extends, read out of the
published files and executed as written. The package's own imports (Flax
models, autoencoders, input encoders) are not needed by those classes and
are not loaded.

The model is a stand-in denoiser of (x, sigma), the same function on both
sides, so what is compared is the samplers' stages and history and the
schedule they read sigma and time through. Each sampler walks FlaxDiff's own
grid (`get_steps(1, 0, STEPS)`) from x_T as `generate_samples` walks it,
every interval but the closing denoise, and `generate_samples` itself walks
it with the closing denoise, its output before and through the clip to
[-1, 1] of the pipeline's `post_process`. What lands, over the seeds check 1
draws (`record_case`, tools/recorded_references.py), per sampler: the
latent after every interval and the vector-Jacobian product of the last one
through the whole walk (`<sampler> walk`), and the closing output unclipped
and clipped (`<sampler> closed`), as published in float32 and by its float64
twin, summarized (each leaf's error from its truth and its truth's sketch,
`record_case`'s summary), and FlaxDiff's grid. Each draw walks its own
grid, FlaxDiff's with every point moved within JITTER of a spacing, given to
the samplers as `generate_samples` takes a grid it is given
(`steps_override`). The float64 twin widens every
`jnp.float32` of the schedule and the sampler's common steps, each a dtype
argument or cast at the pinned commit (schedulers/common.py:18, a dtype
default; karras.py:49, an astype, and :71, a random draw's dtype the walk
does not make; samplers/common.py:144 and :206, the step array's dtype), so
its sigma arithmetic runs in float64 on the float32 grid's own times. A
seed's inputs, x_T, the cotangent and the grid, are a seeded NumPy draw
(`draw`) the test makes again.

    python tools/flaxdiff_solver_reference.py
"""

from __future__ import annotations

import ast
import functools
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from reference_source import fetched, ran, selected

COMMIT = "15c55b001304604147a2a8002a76cf5d4c32092f"
FLAXDIFF = f"https://raw.githubusercontent.com/AshishKumar4/FlaxDiff/{COMMIT}/flaxdiff/"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "flaxdiff_solvers" / "solvers.npz"
STEPS, SIGMA_MAX, RHO, SIGMA_DATA = 12, 80.0, 7.0, 0.5
SHAPE = (32, 64)
JITTER = 0.01
"""How far a draw moves each point of the grid, in its spacings. A walk's rounding is dominated by the
sigmas its grid fixes, which both sides compute by one formula: on one grid each side rounds each of them
once, a coin flip no count of draws resolves (RK4's latent after its fourth interval at a pooled 1.59 times
FlaxDiff's over 80 draws of one grid, each interval alone at 0.92 to 1.05 of it, armada 10-28). Each draw
rounds its own grid's."""
SAMPLERS = {"rk4": ("rk4_sampler.py", "RK4Sampler"), "multistep": ("multistep_dpm.py", "MultiStepDPM")}


def statements(path: str, names: set[str], *, wide: bool = False,
               within: str | None = None) -> list[ast.stmt]:
    """The definitions `names` of FlaxDiff's `path` (`selected`), in float64 where `wide`."""
    text = fetched(FLAXDIFF + path)
    if wide:
        text = text.replace("jnp.float32", "jnp.float64")
    return selected(text, names, within=within, filename=f"flaxdiff/{path}")


def published(*, wide: bool = False) -> dict:
    """The schedule classes, a `DiffusionSampler` holding its published
    `sample_step`, `get_steps` and `generate_samples`, and the two samplers,
    in one namespace."""
    scope: dict = {"jax": jax, "jnp": jnp, "Union": object, "List": list, "Tuple": tuple, "Dict": dict,
                   "struct": struct, "tqdm": types.SimpleNamespace(tqdm=lambda steps: steps)}
    ran(statements("utils.py", {"MarkovState", "RandomMarkovState", "clip_images"}), scope,
        "flaxdiff/utils.py")
    schedules = statements("schedulers/common.py", {"get_coeff_shapes_tuple", "reshape_rates",
                                                    "NoiseScheduler", "GeneralizedNoiseScheduler"},
                           wide=wide)
    schedules += statements("schedulers/karras.py", {"KarrasVENoiseScheduler"}, wide=wide)
    ran(schedules, scope, "flaxdiff/schedulers")
    methods = statements("samplers/common.py", {"sample_step", "get_steps", "generate_samples"}, wide=wide,
                         within="DiffusionSampler")
    holder = ast.parse("class DiffusionSampler:\n"
                       "    def __init__(self, noise_schedule):\n"
                       "        self.noise_schedule = noise_schedule\n").body[0]
    holder.body.extend(methods)
    ran([holder], scope, "flaxdiff/samplers/common.py")
    for path, name in SAMPLERS.values():
        ran(statements(f"samplers/{path}", {name}), scope, f"flaxdiff/samplers/{path}")
    return scope


def denoised(x, sigma):
    """The stand-in's clean estimate: the Gaussian-data optimum bent by a
    tanh, so the ODE is not linear in x."""
    bent = SIGMA_DATA ** 2 * x + sigma * SIGMA_DATA * jnp.tanh(x / SIGMA_DATA)
    return bent / (SIGMA_DATA ** 2 + sigma ** 2)


def sampler_of(scope: dict, name: str):
    """The sampler `name` over the Karras VE schedule, and the stand-in as its model, which takes
    `x_t` at `t` to the clean estimate, the noise it implies and no state."""
    schedule = scope["KarrasVENoiseScheduler"](timesteps=1.0, sigma_max=SIGMA_MAX, rho=RHO,
                                                sigma_data=SIGMA_DATA)

    def model(x_t, t):
        _, sigma = schedule.get_rates(t, scope["get_coeff_shapes_tuple"](x_t))
        x_0 = denoised(x_t, sigma)
        return x_0, (x_t - x_0) / sigma, None

    return scope[SAMPLERS[name][1]](schedule), model


def walk(scope: dict, name: str, x_T, steps):
    """`generate_samples`' loop over every interval of `steps`, as written,
    with the stand-in as the sampler's model."""
    sampler, model = sampler_of(scope, name)
    latents, samples = [], x_T
    for i in range(len(steps) - 1):
        samples, _ = sampler.sample_step(model, samples, steps[i], (),
                                         next_step=steps[i + 1], state=None)
        latents.append(samples)
    return jnp.stack(latents)


def closed(scope: dict, name: str, x_T, steps, clipped: bool):
    """`generate_samples` as written from `x_T` (its `priors`) over the
    grid `steps` (its `steps_override`): every interval but the last grid point, then
    the clean prediction there. The instance holds what the published
    constructor would have bound: the stand-in as `sample_model`, no
    autoencoder and no conditions, and `post_process`, with no autoencoder
    `clip_images` to [-1, 1], when `clipped`; without it the result is the
    walk's own closing prediction, before the output's clip."""
    sampler, model = sampler_of(scope, name)
    sampler.sample_model = lambda params, x_t, t: model(x_t, t)
    sampler.post_process = scope["clip_images"] if clipped else (lambda samples: samples)
    sampler.autoencoder = sampler.input_config = None
    return sampler.generate_samples(None, x_T.shape[0], x_T.shape[1], diffusion_steps=STEPS, priors=x_T,
                                    steps_override=steps)


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's inputs: x_T, standard normals at the schedule's top sigma, the cotangent the last
    latent is pulled back by, and its grid, FlaxDiff's with each point moved within JITTER of a spacing
    and both ends kept inside [0, 1]."""
    generator = np.random.default_rng([0xF1, seed])
    shift = JITTER / (STEPS - 1) * generator.uniform(-1, 1, STEPS)
    shift[0], shift[-1] = -abs(shift[0]), abs(shift[-1])
    return {"x_T": (generator.standard_normal(SHAPE) * SIGMA_MAX).astype(np.float32),
            "cotangent": generator.standard_normal(SHAPE).astype(np.float32),
            "times": (np.linspace(1.0, 0.0, STEPS) + shift).astype(np.float32)}


def runs(name: str, dtype) -> dict:
    """Sampler `name` as published in `dtype` (float64 widened, `published`), compiled over a draw's
    grid: its walk's latent after every interval and the last one's pullback of the cotangent, and its
    closing output before and through the clip."""
    scope = published(wide=dtype == jnp.float64)

    @jax.jit
    def walked(x_T, cotangent, steps):
        latents = walk(scope, name, x_T, steps)
        (gradient,) = jax.vjp(lambda x: walk(scope, name, x, steps)[-1], x_T)[1](cotangent)
        return {**{f"after {index}": latent for index, latent in enumerate(latents)}, "gradient": gradient}

    @jax.jit
    def closing(x_T, cotangent, steps):
        return {label: closed(scope, name, x_T, steps, clipped) for clipped, label in
                ((False, "closed"), (True, "clipped"))}

    return {"walk": walked, "closed": closing}


def evaluated(inputs, compiled, part: str):
    """`part` of a seed's float32 run and its float64 twin's, each leaf checked to be its run's dtype."""
    results = []
    for dtype in (jnp.float32, jnp.float64):
        drawn = (jnp.asarray(inputs[key], dtype) for key in ("x_T", "cotangent", "times"))
        result = compiled[dtype][part](*drawn)
        assert all(leaf.dtype == dtype for leaf in result.values()), (part, dtype)
        results.append(jax.tree.map(np.asarray, result))
    return tuple(results)


def main() -> None:
    # Set here, not on import: the test loads this file for `draw` alone.
    jax.config.update("jax_enable_x64", val=True)
    from recorded_references import provenance, record_case, save

    assert published()["KarrasVENoiseScheduler"](timesteps=1.0).max_timesteps == 1.0
    arrays = {"times": np.asarray(jnp.linspace(1.0, 0.0, STEPS, dtype=jnp.float32))}
    for name in SAMPLERS:
        compiled = {dtype: runs(name, dtype) for dtype in (jnp.float32, jnp.float64)}
        for part in ("walk", "closed"):
            arrays.update(record_case(f"{name} {part}", draw,
                                      functools.partial(evaluated, compiled=compiled, part=part),
                                      store_inputs=False, summary=True))
    written = save(FIXTURE, {**arrays, "provenance": provenance(
        "AshishKumar4/FlaxDiff", COMMIT, "cpu", "python tools/flaxdiff_solver_reference.py")},
        reference=f"AshishKumar4/FlaxDiff {COMMIT}", license="MIT")
    print(f"{written}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
