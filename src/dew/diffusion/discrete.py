"""Masked (absorbing-state) discrete diffusion, shaped like the Gaussian one.

The forward process replaces each token by a mask id independently, with a
probability that grows along t in [0, 1]; `MaskingSchedule.alpha(t)` is the
fraction of tokens still visible. Training is the continuous-time negative
ELBO of MDLM (Sahoo et al. 2024, "Simple and Effective Masked Diffusion
Language Models"): the cross entropy of the model's prediction at the masked
positions, weighted by -alpha'(t) / (1 - alpha(t)). Sampling reverses the
process one interval at a time: a masked token is revealed with probability
(alpha(s) - alpha(t)) / (1 - alpha(t)) and, when revealed, drawn from the
model's categorical, which is MDLM's `_ddpm_update`.

`DiscreteProcess` has the surface `dew.sampling.sample` walks: a time grid,
an initial state, and a denoiser whose two outputs are the model's argmax
fill of the masked positions and the log-probabilities the solver draws
from, in the slots a Gaussian denoiser puts x_0 and epsilon.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from dew.registry import presets, samplers


class MaskingSchedule(ABC):
    """alpha(t) in (0, 1]: the fraction of tokens left unmasked at t, with
    alpha(0) = 1."""

    @abstractmethod
    def alpha(self, t) -> jax.Array: ...

    @abstractmethod
    def alpha_prime(self, t) -> jax.Array:
        """d alpha / dt, negative."""


@dataclass(frozen=True)
class LogLinear(MaskingSchedule):
    """MDLM's log-linear schedule: alpha(t) = 1 - (1 - eps) t, so the masking
    rate -log alpha is linear in log space and the NELBO weight is 1 / t."""

    eps: float = 1e-3

    def alpha(self, t):
        return 1 - (1 - self.eps) * jnp.asarray(t, jnp.float32)

    def alpha_prime(self, t):
        return jnp.full_like(jnp.asarray(t, jnp.float32), -(1 - self.eps))


@dataclass(frozen=True)
class DiscreteProcess:
    """The masking process over a vocabulary whose mask token is `mask_id`."""

    schedule: MaskingSchedule
    mask_id: int
    T = 1.0

    def sample_t(self, key, n: int) -> jax.Array:
        """`n` times stratified over [0, 1), MDLM's antithetic draw: one
        uniform offset shared by the batch, so the weights 1 / t of one batch
        cover the trajectory instead of clustering."""
        offset = jax.random.uniform(key, (), minval=0.0, maxval=1.0)
        return (jnp.arange(n, dtype=jnp.float32) + offset) / n

    def corrupt(self, key, tokens, t) -> tuple[jax.Array, jax.Array]:
        """`(masked tokens, is_masked)` at `t`, one t per row."""
        move_chance = 1 - self.schedule.alpha(t)
        is_masked = jax.random.uniform(key, tokens.shape) < move_chance[:, None]
        return jnp.where(is_masked, self.mask_id, tokens), is_masked

    def weight(self, t) -> jax.Array:
        """The NELBO weight -alpha'(t) / (1 - alpha(t)) on the masked cross
        entropy, and exactly zero at t = 0: nothing is masked there, so no
        token contributes, and the quotient itself is undefined."""
        t = jnp.asarray(t, jnp.float32)
        return jnp.where(t > 0, -self.schedule.alpha_prime(t) / (1 - self.schedule.alpha(t)), 0.0)

    def times(self, steps: int) -> jax.Array:
        return jnp.linspace(self.T, 0.0, steps, dtype=jnp.float32)

    def noise(self, key, shape) -> jax.Array:
        """x_T: every position masked. `key` is unused, the fully masked state
        is one point."""
        return jnp.full(shape, self.mask_id, jnp.int32)

    def denoiser(self, model, params, conditions: dict[str, Any] | None = None,
                 unconditional=None) -> DiscreteDenoiser:
        if conditions or unconditional is not None:
            raise ValueError("the masked diffusion LM takes no conditions")
        return DiscreteDenoiser(self, model, params)


@dataclass(frozen=True)
class DiscreteDenoiser:
    """`(x_t, t) -> (argmax fill, log-probabilities)` for `model` under `params`.

    The model's own logits at an unmasked position are irrelevant: the
    position keeps its token, which is MDLM's carry-over parameterization.
    The mask token itself carries no mass: it marks corruption, so the
    categorical a reveal draws from never offers it, however the model
    scores it.
    """

    process: DiscreteProcess
    model: Any
    params: Any

    def __call__(self, x_t, t):
        logits = self.model.apply(self.params, x_t)
        logits = logits.at[..., self.process.mask_id].set(-jnp.inf)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        masked = x_t == self.process.mask_id
        filled = jnp.where(masked, jnp.argmax(log_probs, axis=-1), x_t)
        return filled, log_probs


@samplers("unmask")
@dataclass(frozen=True)
class Unmask:
    """MDLM's reverse step from t to s < t: each masked position is revealed
    with probability (alpha(s) - alpha(t)) / (1 - alpha(t)), with a token drawn
    from the model's categorical; the rest stay masked. Integrates a
    `DiscreteProcess`."""

    State = tuple

    def init(self, x) -> tuple:
        return ()

    def step(self, x, t, t_next, denoised, log_probs, state, key, process, denoise):
        if not isinstance(process, DiscreteProcess):
            raise ValueError(
                f"Unmask reveals masked tokens, so it needs a DiscreteProcess and not "
                f"{type(process).__name__}")
        alpha_t = process.schedule.alpha(t)
        alpha_s = process.schedule.alpha(t_next)
        reveal_chance = (alpha_s - alpha_t) / (1 - alpha_t)
        reveal_key, token_key = jax.random.split(key)
        masked = x == process.mask_id
        reveal = jax.random.uniform(reveal_key, x.shape) < reveal_chance[:, None]
        drawn = jax.random.categorical(token_key, log_probs, axis=-1)
        return jnp.where(masked & reveal, drawn, x), state


@presets("mdlm")
@dataclass(frozen=True)
class MDLM:
    """Sahoo et al. 2024 with the log-linear schedule."""

    mask_id: int
    eps: float = 1e-3

    def __call__(self) -> DiscreteProcess:
        return DiscreteProcess(schedule=LogLinear(eps=self.eps), mask_id=self.mask_id)
