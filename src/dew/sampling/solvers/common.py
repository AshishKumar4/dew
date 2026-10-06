"""The solver step contract, endpoint checks and history arithmetic shared by concrete solvers."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from typing_extensions import TypeVar

from dew.diffusion.process import Process
from dew.diffusion.schedules import GeneralizedNoiseScheduler

# tests/test_samplers.py checks the Diffusers-named solvers' trajectories and
# trajectory gradients against fixtures recorded by tools/diffusers_reference.py;
# tools/diffusers_limits_reference.py verifies the finite endpoint limits.

# A solver whose state nobody names is a solver over any state: `StateT` is
# covariant, so `Solver` written bare is the type every concrete solver
# satisfies, and a call site that carries the state names it.
StateT = TypeVar("StateT", covariant=True, default=object)


class Solver(Protocol[StateT]):
    """The interface for one diffusion solver step and the state it keeps between steps.

    `StateT` is that state: nothing for a one-step solver, the previous model
    outputs for a multistep one. It is a type parameter, so each solver's own
    state type is checked at its call sites.
    """

    def init(self, x, times, process, *, key) -> StateT:
        """Return the initial state for sampling from `x` over the concrete grid `times`.

        It checks the grid's endpoint domains at compile time. `key` is the
        root key of the whole sampling run; only `DPMSolverSDE` reads it, for
        its whole-trajectory Brownian tree. Each step draws from the folded key
        passed to it.
        """
        ...

    def step(self, x, t, t_next, denoised, eps, state, key, process,
             denoise, /) -> tuple[jax.Array, StateT]:
        """Return the sample at `t_next` and the new state, from `x` and the model's `(denoised, eps)` at `t`.

        `sample` passes every argument by position, so a solver over another
        algebra can name the pair for what it reads: the discrete solver takes
        log-probabilities where a Gaussian one takes eps.
        """
        ...


def _check_endpoint_domain(process: Process, times: jax.Array, *,
                           source: bool = False, target: bool = False, reason: str) -> None:
    """Reject undefined endpoint integrals while the sampling grid is concrete.

    This runs at initialization, including compile-time initialization in
    sample(), rather than introducing host callbacks into a solver step.
    """
    if times.shape[0] < 2:
        return
    with jax.ensure_compile_time_eval():
        alpha, sigma = process.sampler_schedule.rates(times)
        if source and bool(jnp.any(alpha[:-1] == 0)):
            raise ValueError("alpha=0 source has no finite update: " + reason)
        if target and bool(jnp.any(sigma[1:] == 0)):
            raise ValueError("sigma=0 target has no finite update: " + reason)


def _sigma_integrator(name: str, process: Process) -> GeneralizedNoiseScheduler:
    schedule = process.sampler_schedule
    if not isinstance(schedule, GeneralizedNoiseScheduler):
        raise ValueError(
            f"{name} integrates dx/dsigma = eps, which holds only when alpha is 1, "
            f"so it needs a GeneralizedNoiseScheduler and not {type(schedule).__name__}")
    return schedule

def _push(buffer, value):
    """`buffer` with its oldest entry dropped and `value` appended, so the
    most recent entry is last."""
    return jnp.concatenate([buffer[1:], value[None]], axis=0)


class Multistep(NamedTuple):
    """What a multistep solver carries: the last `depth` converted model
    outputs with the signal and noise rates they were read at, most recent
    last, the number of steps taken and the number the walk has. Slots not
    yet filled hold zeros at unit rates, which no update reads."""

    outputs: jax.Array
    alphas: jax.Array
    sigmas: jax.Array
    taken: jax.Array
    steps: jax.Array

    @property
    def lambdas(self) -> jax.Array:
        return _half_log_snr(self.alphas, self.sigmas)

    def push(self, output, alpha, sigma) -> Multistep:
        return Multistep(_push(self.outputs, output), _push(self.alphas, alpha),
                         _push(self.sigmas, sigma), self.taken, self.steps)

    def advance(self) -> Multistep:
        return self._replace(taken=self.taken + 1)


def _multistep(x, times, depth: int) -> Multistep:
    rates = jnp.ones((depth, x.shape[0]) + (1,) * (x.ndim - 1), jnp.float32)
    return Multistep(jnp.zeros((depth, *x.shape), x.dtype), rates, rates,
                     jnp.zeros((), jnp.int32), jnp.asarray(times.shape[0] - 1, jnp.int32))


def _half_log_snr(alpha, sigma):
    """lambda = log(alpha) - log(sigma), the variable DPM-Solver integrates in."""
    return jnp.log(alpha) - jnp.log(sigma)


def _lambda_step(alpha_s, sigma_s, alpha_t, sigma_t):
    """Return the target-zero mask and a finite coefficient placeholder at
    either endpoint. Each caller substitutes its own proved endpoint limit;
    higher-order DEIS and corrected UniPC epsilon need more than clean x_0.
    """
    terminal = sigma_t <= 0
    endpoint = terminal | (alpha_s == 0)
    safe_sigma_t = jnp.where(terminal, sigma_s, sigma_t)
    safe_alpha_s = jnp.where(alpha_s == 0, 1.0, alpha_s)
    h = _half_log_snr(alpha_t, safe_sigma_t) - _half_log_snr(safe_alpha_s, sigma_s)
    return terminal, jnp.where(endpoint, 1.0, h)


def _tapered_order(order: int, history: Multistep, lower_order_final: bool,
                   euler_at_final: bool):
    """The order a Diffusers multistep DPM-Solver takes at this step: it grows
    with the history up to `order`; the last step is first order under
    `euler_at_final`, and in a walk under 15 steps `lower_order_final` makes
    the last step first order and caps the one before it at second."""
    taken, steps = history.taken, history.steps
    current = jnp.minimum(order, taken + 1)
    short = jnp.logical_and(lower_order_final, steps < 15)
    final = jnp.logical_and(taken == steps - 1, jnp.logical_or(euler_at_final, short))
    current = jnp.where(final, 1, current)
    penultimate = jnp.logical_and(taken == steps - 2, short)
    return jnp.where(penultimate, jnp.minimum(current, 2), current)
