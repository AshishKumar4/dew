"""Post-hoc EMA: weight averages of any length, reconstructed after training.

This follows Karras et al., "Analyzing and Improving the Training Dynamics
of Diffusion Models" (CVPR 2024, arXiv:2312.02696), Section 3 and Appendix
C. After a run has made T updates, a power-function EMA weighs the weights
of update t in proportion to t^γ. Its length is given by `std`, the
standard deviation of that profile relative to T (σ_rel in the paper). The
profile stretches with T, which an exponential EMA's fixed decay does not,
so one σ_rel means the same thing at every point of a run.

A run tracks a few such averages (`dew.training.optim.power_profiles`,
`OptimConfig.ema_profiles`), and every checkpoint save also keeps a
snapshot of them (`Checkpoints.profile_steps`). Each snapshot is a linear
combination of the weights along the trajectory, with a known profile. So
the average for any σ_rel at any snapshot's step is approximately a
weighted sum of snapshots, with weights from a least-squares fit of the
profiles. `coefficients` solves for those weights (Algorithm 3), and
`Checkpoints.posthoc_ema` sums the snapshots.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
import optax


def exponent(std: float) -> float:
    """Return the exponent γ of the power-function profile whose relative
    standard deviation is `std`.

    γ is the largest real root of γ³ + 7γ² + (16 - std⁻²)γ + (12 - std⁻²)
    (Eq. 126 and Algorithm 2). `std` must lie in (0, 12^-0.5); any other
    value raises `ValueError`.
    """
    if not 0 < std < 12 ** -0.5:
        raise ValueError(f"a relative standard deviation lies in (0, {12 ** -0.5:.4f}); got {std}")
    tail = std ** -2
    return float(np.roots([1, 7, 16 - tail, 12 - tail]).real.max())


def power_decay(std: float) -> optax.Schedule:
    """Return the decay schedule of the power-function EMA whose relative
    standard deviation is `std`, in the form an `EMASpec` reads.

    After `count` completed updates, update count + 1 keeps
    (1 - 1/(count + 1))^(γ + 1) of the average (Eq. 127). So the first
    update replaces the average with the weights entirely.

    The decay is computed as exp((γ + 1) log1p(-1/t)). That keeps
    1 - decay, which is of order (γ + 1)/t, accurate in fp32 late in a run.
    """
    power = exponent(std) + 1

    def decay(count):
        steps = jnp.asarray(count, jnp.float32) + 1
        return jnp.exp(power * jnp.log1p(-1 / steps))
    return decay


def _correlation(t_a, gamma_a, t_b, gamma_b):
    """The inner product of the profiles of the averages at `t_a` with
    exponent `gamma_a` and at `t_b` with `gamma_b` (Eq. 151, Algorithm 3)."""
    ratio = t_a / t_b
    power = np.where(t_a < t_b, gamma_b, -gamma_a)
    return ((gamma_a + 1) * (gamma_b + 1) * ratio ** power
            / ((gamma_a + gamma_b + 1) * np.maximum(t_a, t_b)))


def coefficients(snapshots: Sequence[tuple[int, float]], updates: int, std: float) -> np.ndarray:
    """Return the weights that combine the averages in `snapshots` into the
    average of `std` after `updates` updates (Algorithm 3).

    Each snapshot is a pair (updates it had seen, its relative standard
    deviation). The weights are the least-squares fit of the target profile
    by the snapshots' profiles, normalized to sum to one. An empty
    `snapshots`, or a snapshot taken after no updates or after more than
    `updates`, raises `ValueError`.
    """
    times = np.array([seen for seen, _ in snapshots], np.float64)
    if times.size == 0 or np.any(times <= 0) or np.any(times > updates):
        raise ValueError(f"reconstructing the average after {updates} updates takes snapshots "
                         f"after 1 to {updates} updates; got {times.tolist()}")
    gammas = np.array([exponent(deviation) for _, deviation in snapshots])
    gram = _correlation(times[:, None], gammas[:, None], times[None, :], gammas[None, :])
    target = _correlation(times, gammas, np.float64(updates), exponent(std))
    weights = np.linalg.solve(gram, target)
    return weights / weights.sum()


__all__ = ["coefficients", "exponent", "power_decay"]
