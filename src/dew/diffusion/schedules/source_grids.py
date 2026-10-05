"""The schedules that sampling on a published diffusion checkpoint's grid uses.

`source.py` reads one scheduler file into a policy, and the policy builds
its `Process` from one of these schedules. Each is an ordinary Dew schedule,
so a solver reads its rates, model time and prior the same way as any
other's.

There are four kinds: the training beta table indexed by t (`TabulatedVP`);
a paired sigma and model-time table, in variance-exploding (`SigmaGrid`) or
normalized-VP (`VPGrid`) coordinates; the variance-exploding table with the
stage rows a two-evaluation solver reads (`StageSigmaGrid`); and a
rectified-flow table whose signal and noise sum to one (`FlowGrid`). A
paired table is read at a coordinate t that counts down from
T = len(sigmas) - 1, so a non-integer t interpolates between two prepared
rows.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dew.diffusion.schedules.common import GeneralizedNoiseScheduler, NoiseScheduler
from dew.diffusion.schedules.discrete import DiscreteNoiseScheduler


class TabulatedVP(DiscreteNoiseScheduler):
    """The training beta table used as the sampling schedule, indexed by t.

    `stride` is the fixed training transfer that DDIM and PNDM step over,
    whatever their evaluation grid is. None keeps the grid's own interval,
    which DDPM's previous-timestep policy and the distilled schedules use. A
    t below zero is the source's "no previous alpha" end, where the rates
    come from `final_alpha_cumprod`.
    """

    def __init__(self, betas: np.ndarray, *, final_alpha_cumprod: float, stride: int | None):
        super().__init__(betas, p2_loss_weight_gamma=0)
        self.final_rates = tuple(np.float32(np.sqrt(value))
                                 for value in (final_alpha_cumprod, 1 - final_alpha_cumprod))
        self.stride = stride

    def rates(self, t):
        t = jnp.asarray(t, jnp.float32)
        index = jnp.clip(t.astype(jnp.int32), 0, self.T - 1)
        return (jnp.where(t < 0, self.final_rates[0], self.sqrt_alpha_cumprod[index]),
                jnp.where(t < 0, self.final_rates[1], self.sqrt_one_minus_alpha_cumprod[index]))

    def model_time(self, t):
        return jnp.maximum(jnp.asarray(t, jnp.float32), 0.0)

    def step_interval(self, t, t_next):
        """Return the published DDIM and PNDM transfer stride, independent of the evaluation spacing.

        Without a `stride`, this is the grid's own interval.
        """
        if self.stride is None:
            return super().step_interval(t, t_next)
        return jnp.full_like(jnp.asarray(t, jnp.float32), self.stride)

    def half_interval(self, t, t_next):
        """Return half the integer transfer stride, rounded down, as published PRK uses it."""
        return jnp.floor(self.step_interval(t, t_next) / 2)


def _interpolate(x, xp, fp):
    """`jnp.interp(x, xp, fp)` in its own arithmetic, with the interval found
    by comparing x against every point of xp.

    jnp.interp finds it by bisection, a loop of log2(len(xp)) steps. Inside
    `sample`'s scan with one row, XLA rewrites the table reads into dynamic
    slices whose offsets come out of that loop, and XLA:GPU's
    DynamicSliceAnnotator (jax 0.11.1) evaluates each offset without running
    the loop, then fails the whole compile on the unknown value instead of
    skipping the slice (openxla/xla#49299). A grid holds a few dozen points,
    so comparing against all of them costs nothing and leaves no loop.
    """
    x = jnp.asarray(x, jnp.float32)
    i = jnp.clip(jnp.searchsorted(xp, x, side="right", method="compare_all"), 1, len(xp) - 1)
    df = fp[i] - fp[i - 1]
    dx = xp[i] - xp[i - 1]
    flat = jnp.abs(dx) <= np.spacing(np.finfo(np.float32).eps)
    f = jnp.where(flat, fp[i - 1], fp[i - 1] + (x - xp[i - 1]) / jnp.where(flat, 1, dx) * df)
    return jnp.where(x > xp[-1], fp[-1], jnp.where(x < xp[0], fp[0], f))


class _PairedGrid:
    """Reads a prepared grid of paired sigmas and model times by coordinate.

    Coordinate t counts down from `T = len(sigmas) - 1`, so t and the grid
    index run opposite ways and a non-integer t interpolates between two
    prepared rows. Subclasses say what the sigmas mean as rates.
    """

    def __init__(self, sigmas: np.ndarray, model_times: np.ndarray, prior: float):
        self.table = jnp.asarray(sigmas, jnp.float32)
        self.times = jnp.asarray(model_times, jnp.float32)
        self.rows = jnp.arange(len(sigmas), dtype=jnp.float32)
        self.prior = jnp.asarray(prior, jnp.float32)
        self.T = float(len(sigmas) - 1)

    def sigmas(self, t):
        return _interpolate(self.T - jnp.asarray(t, jnp.float32), self.rows, self.table)

    def t_of_sigma(self, sigma):
        return self.T - _interpolate(sigma, self.table[::-1], self.rows[::-1])

    def model_time(self, t):
        return _interpolate(self.T - jnp.asarray(t, jnp.float32), self.rows, self.times)

    def prior_scale(self):
        return self.prior


class SigmaGrid(_PairedGrid, GeneralizedNoiseScheduler):
    """A variance-exploding schedule on a prepared grid of paired sigmas and model times.

    `sigma_min` and `sigma_max` are the grid's smallest positive sigma and
    its largest sigma, which is the domain a source's noise sampler is built
    over.
    """

    def __init__(self, sigmas: np.ndarray, model_times: np.ndarray, prior: float):
        levels = np.asarray(sigmas, np.float64)
        GeneralizedNoiseScheduler.__init__(self, sigma_min=float(np.min(levels[levels > 0])),
                                           sigma_max=float(np.max(levels)))
        _PairedGrid.__init__(self, sigmas, model_times, prior)


class StageSigmaGrid(SigmaGrid):
    """A source's variance-exploding grid that includes the stage rows of a two-evaluation solver.

    Even coordinates are the grid points the outer loop visits. The odd
    coordinate between each pair is the source's own interpolated
    evaluation, at the sigma the source places there and the model time it
    computes for that sigma. `t_of_sigma` maps a sigma to a stage
    coordinate. That is the only inversion these solvers need from a
    schedule, because KDPM2's midpoint and DPMSolverSDE's proposal both land
    on a stage row.
    """

    def __init__(self, sigmas: np.ndarray, model_times: np.ndarray, prior: float):
        super().__init__(sigmas, model_times, prior)
        stages = np.asarray(sigmas, np.float64)[1::2]
        self.stages = jnp.asarray(stages[::-1].copy(), jnp.float32)
        self.stage_positions = jnp.asarray(
            np.arange(1, len(sigmas), 2, dtype=np.float32)[::-1].copy())

    def t_of_sigma(self, sigma):
        return self.T - _interpolate(sigma, self.stages, self.stage_positions)


class _UniformGrid(_PairedGrid, NoiseScheduler):
    """A paired grid whose source trains at uniform times and weights nothing.

    The flow and normalized-VP grids differ only in what their sigmas mean
    as rates; the draw and the loss weight are the same for both.
    """

    def sample_t(self, key, n: int):
        return jax.random.uniform(key, (n,), minval=0, maxval=self.T)

    def weight(self, t):
        return jnp.ones_like(jnp.asarray(t, jnp.float32))


class FlowGrid(_UniformGrid):
    """A rectified-flow grid, where alpha is 1 - sigma.

    The source's forward process is x_t = (1 - sigma) x_0 + sigma eps, so
    signal and noise sum to one; in a normalized VP pair, their squares do.
    Model times are the sigmas times the training step count, which is where
    the source's `timesteps` come from. The prior at sigma 1 is the unit
    Gaussian. Training draws times uniformly with a constant loss weight.
    """

    def rates(self, t):
        sigma = self.sigmas(t)
        return 1.0 - sigma, sigma


class VPGrid(_UniformGrid):
    """A paired sigma and model-time grid in normalized VP coordinates.

    A grid sigma gives alpha = 1 / sqrt(1 + sigma^2) and a noise rate of
    sigma alpha. Training draws times uniformly with a constant loss weight.
    """

    def rates(self, t):
        sigma = self.sigmas(t)
        # The source rounds sqrt before its reciprocal. Fusing to rsqrt moves
        # stiff cosine-grid VJPs beyond the float32 source-parity bound.
        alpha = 1 / jax.lax.optimization_barrier(jnp.sqrt(1 + sigma ** 2))
        return alpha, sigma * alpha


__all__ = ["FlowGrid", "SigmaGrid", "StageSigmaGrid", "TabulatedVP", "VPGrid"]
