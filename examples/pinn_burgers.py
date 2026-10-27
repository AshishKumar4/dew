"""A physics-informed network to float64 precision: the self-similar blowup profile of 1D Burgers.

    python examples/pinn_burgers.py --out runs/burgers
    python examples/pinn_burgers.py --model mlp --out runs/burgers-mlp
    JAX_PLATFORMS=cpu python examples/pinn_burgers.py --smoke --out /tmp/burgers-smoke

The inviscid Burgers equation u_t + u u_x = 0 blows up in finite time along the self-similar profile
u = tau^lambda U(x tau^-(1 + lambda)), whose profile solves

    F(y) = -lambda U + ((1 + lambda) y + U) U_y = 0,   lambda = 0.5,

on y in [0, inf). As Zhao et al. (arXiv 2605.24278, Appendix J, after Wang et al. 2026) set it up, the
network learns U~ over eta in [-c, c], c = 30, y = sinh(eta), and the profile is odd with the known
tail built in:

    U(eta) = U~(eta) - U~(-eta) - (y / (1 + y))^15 y^(lambda / (1 + lambda)).

Each step draws a batch of eta uniform in [0, c) from its key and minimizes the mean of F^2 + (dF/deta)^2
in float64, with Adam at 1e-4 decayed to zero by a cosine over the run (epsilon 1e-15). `--model pyramid`
(the default) reads U~ through a 12-level Fourier feature pyramid, grids of 2 to 4096 points of 4
channels, at the coordinate (eta + c) / (2c), into a tanh MLP of width 20 and four layers; `--model mlp`
is that MLP alone, on eta. `--damping` steps by Gauss-Newton instead of Adam (`Residuals`), damped as
Levenberg-Marquardt's.

The run writes `metrics.json` to `--out`: on the grid eta_i = c (i + 1/2) / 1000, the mean of F^2 and
log10 of the largest |F|, which the paper reports (its Table 2) as 6.29e-19 and -8.46 for the pyramid
after 20,000 steps of 4096 points, and 6.21e-7 and -2.41 for the MLP alone. On one H100 this example's
default run reaches 2.26e-18 and -7.91 in 26 minutes, and `--model mlp` 5.68e-8 and -2.95 in 15.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax import linen as nn

from dew.data import Memory
from dew.nn.fourier import FourierPyramid
from dew.objectives.base import Step
from dew.objectives.residuals import Residuals
from dew.training import Trainer

# The run is float64 throughout: its parameters, its points and the residuals they square.
jax.config.update("jax_enable_x64", val=True)

LAMBDA, C, TAIL = 0.5, 30.0, 15


def torch_uniform(key, shape, dtype=jnp.float64):
    """PyTorch's default `nn.Linear` draw, uniform within 1 / sqrt(shape[0]), as the paper's MLP starts."""
    limit = math.sqrt(1.0 / shape[0])
    return jax.random.uniform(key, shape, dtype, -limit, limit)


class Profile(nn.Module):
    """U~ at eta: a tanh MLP of `width` and `depth` layers, reading eta itself or, with `pyramid` levels, a
    Fourier feature pyramid at (eta + c) / (2c)."""

    pyramid: int = 12
    width: int = 20
    depth: int = 4

    @nn.compact
    def __call__(self, eta):
        x = eta[..., None]
        if self.pyramid:
            grids = tuple(min(2 ** (level + 1), 4096) for level in range(self.pyramid))
            x = FourierPyramid(grids, features=4, global_scale=1000.0, level_scale=0.30, init_noise=1e-5,
                               param_dtype=jnp.float64)((eta[..., None] + C) / (2 * C))
        for _ in range(self.depth - 1):
            x = jnp.tanh(nn.Dense(self.width, kernel_init=torch_uniform, bias_init=torch_uniform,
                                  dtype=jnp.float64, param_dtype=jnp.float64)(x))
        return nn.Dense(1, kernel_init=torch_uniform, bias_init=torch_uniform, dtype=jnp.float64,
                        param_dtype=jnp.float64)(x)[..., 0]


class BurgersProfile(Residuals):
    """The profile equation's residual F and its derivative in eta at points each step draws in [0, c)."""

    def __init__(self, model: Profile, damping: float | None = None):
        self.model = self.bind_model(model)
        self.damping = damping

    def fresh_variables(self, key, held):
        return self.model.init(key, jnp.zeros((1,), jnp.float64))

    def profile(self, variables, eta):
        """U at one eta: the learned part made odd, and the known tail."""
        y = jnp.sinh(eta)
        tail = -(y ** (LAMBDA / (1 + LAMBDA))) * (y / (1 + y)) ** TAIL
        learned = self.model.apply(variables, jnp.stack([eta, -eta]))
        return learned[0] - learned[1] + tail

    def equation(self, variables, eta):
        """F at one eta, U_y read as U_eta / cosh(eta)."""
        profile, along = jax.value_and_grad(lambda point: self.profile(variables, point))(eta)
        return -LAMBDA * profile + ((1 + LAMBDA) * jnp.sinh(eta) + profile) * along / jnp.cosh(eta)

    def residuals(self, variables, batch, step: Step):
        eta = C * jax.random.uniform(step.key, (batch["point"].shape[0],), jnp.float64)
        return jax.vmap(lambda point: jnp.stack(jax.value_and_grad(
            lambda at: self.equation(variables, at))(point)))(eta)

    def diagnostics(self, variables) -> dict[str, float]:
        """The paper's Table 2 figures, on its grid eta_i = c (i + 1/2) / 1000."""
        eta = C * (jnp.arange(1000, dtype=jnp.float64) + 0.5) / 1000
        residual = np.asarray(jax.vmap(lambda point: self.equation(variables, point))(eta))
        return {"pde_mse": float(np.mean(residual ** 2)),
                "log10_max_residual": float(np.log10(np.max(np.abs(residual))))}


@dataclasses.dataclass
class Config:
    out: Path = Path("runs/burgers")
    """Where metrics.json goes."""
    model: str = "pyramid"
    """`pyramid` (the Fourier feature pyramid and MLP) or `mlp` (the MLP alone)."""
    steps: int = 20_000
    batch: int = 4096
    learning_rate: float = 1e-4
    damping: float | None = None
    """Gauss-Newton's Levenberg-Marquardt damping in place of Adam; None trains with Adam."""
    seed: int = 0
    smoke: bool = False
    """Twenty steps of a four-level pyramid on 64 points, to check the path."""


def main(config: Config) -> dict[str, float]:
    levels = 0 if config.model == "mlp" else 4 if config.smoke else 12
    steps, batch = (20, 64) if config.smoke else (config.steps, config.batch)
    objective = BurgersProfile(Profile(pyramid=levels), damping=config.damping)
    optimizer = (optax.sgd(1.0) if config.damping is not None else
                 optax.adam(optax.cosine_decay_schedule(config.learning_rate, steps), eps=1e-15))
    points = Memory([{"point": np.zeros((), np.float64)}] * batch).repeat().batch(batch)
    trainer = Trainer(objective, optimizer, key=config.seed)
    state = trainer.fit(points, steps=steps, log_every=max(1, steps // 20))
    metrics = {"steps": steps, "batch": batch, "model": config.model,
               **objective.diagnostics(state.variables)}
    config.out.mkdir(parents=True, exist_ok=True)
    (config.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    main(tyro.cli(Config))
