"""The schedule objects a published diffusion checkpoint's grid is walked on.

`source.py` reads one scheduler file into a policy, and these are the
schedules that policy hands to `Process`. Each is an ordinary Dew schedule,
so a solver reads its rates, its model time and its prior the way it reads
any other.

The four are the training beta table indexed by t, a paired
sigma/model-time table in variance-exploding or normalized-VP coordinates,
the same table refined with the stage row a two-evaluation solver reads, and
a rectified-flow table whose signal and noise sum to one.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dew.diffusion.schedules.common import GeneralizedNoiseScheduler, NoiseScheduler
from dew.diffusion.schedules.discrete import DiscreteNoiseScheduler


class TabulatedVP(DiscreteNoiseScheduler):
    """The training beta table as the sampling schedule, indexed by t.

    `stride` is the fixed training transfer DDIM and PNDM step over, whatever
    their evaluation grid is. None leaves the grid's own interval, which is
    what DDPM's previous-timestep policy and the distilled schedules take. A
    t below zero is the source's "no previous alpha" end.
    """

    def __init__(self, betas: np.ndarray, *, final_alpha_cumprod: float, stride: int | None):
        super().__init__(betas, p2_loss_weight_gamma=0)
        self.final_alpha_cumprod = jnp.asarray(final_alpha_cumprod, jnp.float32)
        self.stride = stride

    def rates(self, t):
        t = jnp.asarray(t, jnp.float32)
        index = jnp.clip(t.astype(jnp.int32), 0, self.T - 1)
        alpha = jnp.where(t < 0, self.final_alpha_cumprod, self.alpha_cumprod[index])
        return jnp.sqrt(alpha), jnp.sqrt(1 - alpha)

    def model_time(self, t):
        return jnp.maximum(jnp.asarray(t, jnp.float32), 0.0)

    def step_interval(self, t, t_next):
        """Published DDIM/PNDM transfer stride, independent of evaluation spacing."""
        if self.stride is None:
            return super().step_interval(t, t_next)
        return jnp.full_like(jnp.asarray(t, jnp.float32), self.stride)

    def half_interval(self, t, t_next):
        """Published PRK uses the integer transfer stride divided by two."""
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
    """A VE process with paired continuous sigma and model-time coordinates.

    `sigma_min` and `sigma_max` are the prepared grid's own positive
    extremes, which is the domain a source noise sampler is built over.
    """

    def __init__(self, sigmas: np.ndarray, model_times: np.ndarray, prior: float):
        levels = np.asarray(sigmas, np.float64)
        GeneralizedNoiseScheduler.__init__(self, sigma_min=float(np.min(levels[levels > 0])),
                                           sigma_max=float(np.max(levels)))
        _PairedGrid.__init__(self, sigmas, model_times, prior)


class StageSigmaGrid(SigmaGrid):
    """A source VE grid that carries the stage rows of a two-evaluation solver.

    Even coordinates are the grid points the outer walk visits. The odd one
    between each pair is the source's own interpolated evaluation, at the
    sigma it places there and the model time it reads back for that sigma.
    `t_of_sigma` resolves a sigma to a stage coordinate, the only inversion
    these solvers ask of a schedule, since KDPM2's midpoint and
    DPMSolverSDE's proposal both land on a stage row.
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
    """A rectified-flow grid: alpha is 1 - sigma, not a normalized VP pair.

    The source's forward process is x_t = (1 - sigma) x_0 + sigma eps, so
    signal and noise sum to one rather than their squares. Model times are
    the sigmas times the training count, which is where the source's
    `timesteps` come from. The prior at sigma 1 is the unit Gaussian.
    """

    def rates(self, t):
        sigma = self.sigmas(t)
        return 1.0 - sigma, sigma


class VPGrid(_UniformGrid):
    """The same paired coordinates in normalized VP latent space."""

    def rates(self, t):
        sigma = self.sigmas(t)
        # The source rounds sqrt before its reciprocal. Fusing to rsqrt moves
        # stiff cosine-grid VJPs beyond the float32 source-parity bound.
        alpha = 1 / jax.lax.optimization_barrier(jnp.sqrt(1 + sigma ** 2))
        return alpha, sigma * alpha


__all__ = ["FlowGrid", "SigmaGrid", "StageSigmaGrid", "TabulatedVP", "VPGrid"]
