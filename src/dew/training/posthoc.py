"""Post-hoc EMA: averages of any length, reconstructed after training.

Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion
Models" (CVPR 2024, arXiv:2312.02696), Section 3 and Appendix C. A
power-function EMA weighs the weights of update t, out of the T a run has
made, in proportion to t^γ; its length is named by `std`, the profile's
standard deviation relative to T (their σ_rel). Unlike an exponential EMA
of fixed decay, its profile scales with T, so one σ_rel means the same
thing at every point of a run.

A run tracks a few such averages (`dew.training.optim.power_profiles`,
`OptimConfig.ema_profiles`), and every checkpoint save also keeps a
snapshot of them (`Checkpoints.profile_steps`). Each snapshot is a linear
image of the weight trajectory with a known profile, so the average of
any σ_rel at any snapshot's step is, to a least-squares fit of profiles,
a weighted sum of snapshots: `coefficients` solves for the weights
(Algorithm 3) and `reconstruct` sums the snapshots.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.objectives.base import Variables


def exponent(std: float) -> float:
    """The exponent γ of the power-function profile whose relative standard
    deviation is `std` (Eq. 126 and Algorithm 2: the largest real root of
    γ³ + 7γ² + (16 - std⁻²)γ + (12 - std⁻²))."""
    if not 0 < std < 12 ** -0.5:
        raise ValueError(f"a relative standard deviation lies in (0, {12 ** -0.5:.4f}); got {std}")
    tail = std ** -2
    return float(np.roots([1, 7, 16 - tail, 12 - tail]).real.max())


def relative_std(gamma: float) -> float:
    """The relative standard deviation of the profile of exponent `gamma` (Eq. 123)."""
    return float(np.sqrt((gamma + 1) / (gamma + 2) ** 2 / (gamma + 3)))


def power_decay(std: float) -> optax.Schedule:
    """The decay of the power-function EMA of relative standard deviation
    `std`, as an `EMASpec` reads it: after `count` completed updates, the
    update that makes count + 1 keeps (1 - 1/(count + 1))^(γ + 1) of the
    average (Eq. 127), so the first update takes the weights whole.

    Computed as exp((γ + 1) log1p(-1/t)), which keeps 1 - decay, of order
    (γ + 1)/t, accurate in fp32 late in a run."""
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
    """The weights that sum the averages `snapshots`, each (updates it had
    seen, its relative standard deviation), into the average of `std` after
    `updates` updates (Algorithm 3): the least-squares fit of its profile by
    theirs, normalized so the weights sum to one."""
    times = np.array([seen for seen, _ in snapshots], np.float64)
    if times.size == 0 or np.any(times <= 0) or np.any(times > updates):
        raise ValueError(f"reconstructing the average after {updates} updates takes snapshots "
                         f"after 1 to {updates} updates; got {times.tolist()}")
    gammas = np.array([exponent(deviation) for _, deviation in snapshots])
    gram = _correlation(times[:, None], gammas[:, None], times[None, :], gammas[None, :])
    target = _correlation(times, gammas, np.float64(updates), exponent(std))
    weights = np.linalg.solve(gram, target)
    return weights / weights.sum()


def reconstruct(directory: str, std: float, step: int | None = None) -> Variables:
    """The average of relative standard deviation `std` at checkpoint `step`
    (default: the latest snapshot) of the run in `directory`, as host arrays
    in the params' structure and dtypes.

    Sums every snapshot up to `step`, of every tracked profile, with the
    weights `coefficients` solves for, one snapshot read at a time and
    accumulated in fp32 or wider. The result goes where the run's params
    go: `merge(params, {"params": reconstruct(...)})`.
    """
    from dew.checkpoints import Checkpoints

    checkpoints = Checkpoints(directory)
    steps = checkpoints.profile_steps()
    if not steps:
        raise FileNotFoundError(f"{directory} holds no EMA profile snapshots; train with "
                                f"OptimConfig.ema_profiles to keep them")
    step = steps[-1] if step is None else step
    if step not in steps:
        raise ValueError(f"{directory} holds EMA profile snapshots at steps {steps}, not {step}")
    held = [(each, *checkpoints.profile_metadata(each)) for each in steps if each <= step]
    # A snapshot before the first update holds the initial weights, with no
    # profile to fit.
    held = [(each, updates, stds) for each, updates, stds in held if updates > 0]
    if not held or held[-1][0] != step:
        raise ValueError(f"the snapshot at step {step} was taken before the first update")
    weights = iter(coefficients([(updates, deviation) for _, updates, stds in held for deviation in stds],
                                held[-1][1], std))
    total, dtypes = None, None
    for each, _, _ in held:
        for average in checkpoints.restore_profiles(each):
            weight = next(weights)
            if total is None:
                dtypes = jax.tree.map(lambda leaf: leaf.dtype, average)
                total = jax.tree.map(
                    lambda leaf: weight * leaf.astype(np.promote_types(leaf.dtype, np.float32)), average)
            else:
                total = jax.tree.map(lambda sum_, leaf: sum_ + weight * leaf, total, average)
    return jax.tree.map(lambda leaf, dtype: leaf.astype(dtype), total, dtypes)
