"""FlaxDiff's RK4 and multistep samplers, for tests/fixtures/flaxdiff/solvers.npz.

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
"""

from __future__ import annotations

import ast
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", val=True)

COMMIT = "15c55b001304604147a2a8002a76cf5d4c32092f"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "flaxdiff" / "solvers.npz"
STEPS, SIGMA_MAX, RHO, SIGMA_DATA = 12, 80.0, 7.0, 0.5
SAMPLERS = {"rk4": ("rk4_sampler.py", "RK4Sampler"), "multistep": ("multistep_dpm.py", "MultiStepDPM")}


def source(path: str) -> str:
    url = f"https://raw.githubusercontent.com/AshishKumar4/FlaxDiff/{COMMIT}/flaxdiff/{path}"
    return urllib.request.urlopen(url).read().decode()


def definitions(path: str, names: set[str]) -> list[ast.stmt]:
    tree = ast.parse(source(path))
    found = [node for node in tree.body
             if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in names]
    assert {node.name for node in found} == names, names - {node.name for node in found}
    return found


def published() -> dict:
    """The schedule classes, a `DiffusionSampler` holding only its published
    `sample_step`, and the two samplers, in one namespace."""
    scope: dict = {"jax": jax, "jnp": jnp, "Union": object, "RandomMarkovState": object,
                   "MarkovState": object}
    schedules = definitions("schedulers/common.py", {"get_coeff_shapes_tuple", "reshape_rates",
                                                     "NoiseScheduler", "GeneralizedNoiseScheduler"})
    schedules += definitions("schedulers/karras.py", {"KarrasVENoiseScheduler"})
    exec(compile(ast.Module(body=schedules, type_ignores=[]), "flaxdiff/schedulers", "exec"), scope)
    sampler = next(node for node in definitions("samplers/common.py", {"DiffusionSampler"})[0].body
                   if isinstance(node, ast.FunctionDef) and node.name == "sample_step")
    holder = ast.parse("class DiffusionSampler:\n"
                       "    def __init__(self, noise_schedule):\n"
                       "        self.noise_schedule = noise_schedule\n").body[0]
    holder.body.append(sampler)
    exec(compile(ast.Module(body=[holder], type_ignores=[]), "flaxdiff/samplers/common.py", "exec"),
         scope)
    for path, name in SAMPLERS.values():
        exec(compile(ast.Module(body=definitions(f"samplers/{path}", {name}), type_ignores=[]),
                     f"flaxdiff/samplers/{path}", "exec"), scope)
    return scope


def denoised(x, sigma):
    """The stand-in's clean estimate: the Gaussian-data optimum bent by a
    tanh, so the ODE is not linear in x."""
    bent = SIGMA_DATA ** 2 * x + sigma * SIGMA_DATA * jnp.tanh(x / SIGMA_DATA)
    return bent / (SIGMA_DATA ** 2 + sigma ** 2)


def walk(scope: dict, name: str, x_T, steps):
    """`generate_samples`' loop over every interval of `steps`, as written,
    with the stand-in as the sampler's model."""
    schedule = scope["KarrasVENoiseScheduler"](timesteps=1.0, sigma_max=SIGMA_MAX, rho=RHO,
                                                sigma_data=SIGMA_DATA)
    sampler = scope[SAMPLERS[name][1]](schedule)

    def sample_model_fn(x_t, t):
        _, sigma = schedule.get_rates(t, scope["get_coeff_shapes_tuple"](x_t))
        x_0 = denoised(x_t, sigma)
        return x_0, (x_t - x_0) / sigma, None

    latents, samples = [], x_T
    for i in range(len(steps) - 1):
        samples, _ = sampler.sample_step(sample_model_fn, samples, steps[i], (),
                                         next_step=steps[i + 1], state=None)
        latents.append(samples)
    return jnp.stack(latents)


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
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"{FIXTURE}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
