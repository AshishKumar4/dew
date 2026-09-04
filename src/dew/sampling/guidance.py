"""Classifier-free guidance as a wrapper around a denoiser."""

from dataclasses import dataclass

import jax.numpy as jnp

from dew.diffusion.process import Denoiser
from dew.diffusion.schedules import expand


@dataclass(frozen=True)
class CFG:
    """Interval-limited classifier-free guidance (Kynkaanniemi et al. 2024).

    The guided prediction is uncond + scale (cond - uncond). Guidance hurts at
    high noise and buys nothing at low noise, so outside `interval` the scale
    drops to 1, which is exactly the plain conditional prediction. The
    interval is in trajectory progress, 0 at pure noise and 1 at the clean
    sample; the default covers all of it.
    """

    scale: float
    interval: tuple[float, float] = (0.0, 1.0)

    def __call__(self, denoise: Denoiser):
        T = denoise.process.sampler_schedule.T
        start, stop = self.interval

        def guided(x, t):
            (x_0, eps), (x_0_uncond, eps_uncond) = denoise.both(x, t)
            progress = 1.0 - t / T
            inside = (progress >= start) & (progress <= stop)
            scale = expand(jnp.where(inside, self.scale, 1.0), x)
            return (x_0_uncond + scale * (x_0 - x_0_uncond),
                    eps_uncond + scale * (eps - eps_uncond))

        return guided
