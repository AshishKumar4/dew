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
every interval but the closing denoise. What lands: the grid, x_T, a
cotangent, and per sampler the latent after every interval and the
vector-Jacobian product of the last one through the whole walk, in float32
and in float64.

    python tools/flaxdiff_solver_reference.py
    python tools/flaxdiff_solver_reference.py rounding tests/fixtures/flaxdiff_solvers/rounding.npz 4
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from reference_source import fetched, ran, selected

jax.config.update("jax_enable_x64", val=True)

COMMIT = "15c55b001304604147a2a8002a76cf5d4c32092f"
FLAXDIFF = f"https://raw.githubusercontent.com/AshishKumar4/FlaxDiff/{COMMIT}/flaxdiff/"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "flaxdiff_solvers" / "solvers.npz"
STEPS, SIGMA_MAX, RHO, SIGMA_DATA = 12, 80.0, 7.0, 0.5
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


def closed(scope: dict, name: str, x_T, steps: int, clipped: bool):
    """`generate_samples` as written from `x_T` (its `priors`) over
    `get_steps(1, 0, steps)`: every interval but the last grid point, then
    the clean prediction there. The instance holds what the published
    constructor would have bound: the stand-in as `sample_model`, no
    autoencoder and no conditions, and `post_process`, with no autoencoder
    `clip_images` to [-1, 1], when `clipped`; without it the result is the
    walk's own closing prediction, before the output's clip."""
    sampler, model = sampler_of(scope, name)
    sampler.sample_model = lambda params, x_t, t: model(x_t, t)
    sampler.post_process = scope["clip_images"] if clipped else (lambda samples: samples)
    sampler.autoencoder = sampler.input_config = None
    return sampler.generate_samples(None, x_T.shape[0], x_T.shape[1], diffusion_steps=steps, priors=x_T)


def rounding(destination: Path, copies: int) -> None:
    """RK4's clipped output over independent initial states.

    Its denoiser and stages are pointwise, so permutations repeat the same
    rounding draw. More independent states average their per-input variance
    without changing the reference rule. The float64 namespace widens the
    source's explicit float32 timestep casts as well as the input; otherwise
    its sigma calculations retain float32 roundings in the oracle. Every
    `jnp.float32` the widening replaces is a dtype argument or cast:
    schedulers/common.py:18 (a dtype default), karras.py:49 (astype) and :71
    (a random draw's dtype, unused by the walk), samplers/common.py:144 and
    :206 (the step array's dtype), at the pinned commit.
    """
    generator = np.random.default_rng(4)
    initial = (generator.standard_normal((32 * copies, 64)) * SIGMA_MAX).astype(np.float32)
    single = closed(published(), "rk4", jnp.asarray(initial), STEPS, clipped=True)
    truth = closed(published(wide=True), "rk4", jnp.asarray(initial, jnp.float64), STEPS, clipped=True)
    np.savez_compressed(destination, copies=np.asarray(copies), x_T=initial,
                        clipped=np.asarray(single), clipped_f64=np.asarray(truth))
    print(f"{destination}: {copies} independent initial states per original entry")


def main() -> None:
    scope = published()
    grid = scope["KarrasVENoiseScheduler"](timesteps=1.0)
    steps32 = jnp.linspace(1.0, 0.0, STEPS, dtype=jnp.float32)
    generator = np.random.default_rng(4)
    arrays = {"commit": np.array(COMMIT), "times": np.asarray(steps32),
              "x_T": (generator.standard_normal((32, 64)) * SIGMA_MAX).astype(np.float32),
              "cotangent": generator.standard_normal((32, 64)).astype(np.float32)}
    assert grid.max_timesteps == 1.0
    for name in SAMPLERS:
        for dtype, tail in ((jnp.float32, ""), (jnp.float64, "_f64")):
            x_T = jnp.asarray(arrays["x_T"], dtype)
            steps = jnp.asarray(steps32, dtype)

            def final(x_T, name=name, steps=steps):
                return walk(scope, name, x_T, steps)[-1]

            latents = walk(scope, name, x_T, steps)
            _, pullback = jax.vjp(final, x_T)
            (gradient,) = pullback(jnp.asarray(arrays["cotangent"], dtype))
            assert latents.dtype == dtype and gradient.dtype == dtype, (name, latents.dtype)
            arrays[f"{name}/latents{tail}"] = np.asarray(latents)
            arrays[f"{name}/grad{tail}"] = np.asarray(gradient)
            for clipped, label in ((False, "closed"), (True, "clipped")):
                closing = closed(scope, name, x_T, STEPS, clipped)
                assert closing.dtype == dtype, closing.dtype
                arrays[f"{name}/{label}{tail}"] = np.asarray(closing)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"{FIXTURE}: {sorted(arrays)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["rounding"]:
        rounding(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
