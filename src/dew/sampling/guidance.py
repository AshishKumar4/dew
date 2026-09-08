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

    `rescale` is the guidance rescaling of Lin et al. 2023 ("Common Diffusion
    Noise Schedules and Sample Steps are Flawed", section 3.4), Diffusers'
    `guidance_rescale`: the guided output is rescaled to the per-sample
    standard deviation of the conditional one and mixed back at that weight,
    so 0 leaves the guided output alone and 1 takes the rescaled one. The
    standard deviation is over everything but the batch axis, with the
    unbiased correction the reference's `Tensor.std` applies.

    Guidance combines the model's raw outputs and lets the denoiser convert
    once, so a source's clipping, dynamic thresholding or consistency
    boundary sees the guided output rather than each branch separately.
    """

    scale: float
    interval: tuple[float, float] = (0.0, 1.0)
    rescale: float = 0.0

    def __post_init__(self):
        # A record's interval arrives as a list, from a run's json or a
        # command line; a tuple keeps the value hashable, so it can ride
        # into a jit as a static argument.
        object.__setattr__(self, "interval", tuple(float(edge) for edge in self.interval))
        object.__setattr__(self, "rescale", float(self.rescale))

    def __call__(self, denoise: Denoiser):
        T = denoise.process.sampler_schedule.T
        start, stop = self.interval

        def guided(x, t):
            output, unconditional = denoise.raw_both(x, t)
            # Progress is the fraction of the trajectory walked, so it lives in
            # [0, 1]: t = T is 0 and the terminal point past the grid's end is
            # 1. Clamping it there is that definition, and it also keeps the
            # top of a walk inside the default interval, which a fused
            # 1 - t / T can miss by an ulp.
            progress = jnp.clip(1.0 - jnp.asarray(t, jnp.float32) / T, 0.0, 1.0)
            inside = (progress >= start) & (progress <= stop)
            scale = expand(jnp.where(inside, self.scale, 1.0), x)
            combined = unconditional + scale * (output - unconditional)
            if self.rescale:
                axes = tuple(range(1, combined.ndim))
                deviation = jnp.std(output, axis=axes, keepdims=True, ddof=1)
                guided_deviation = jnp.std(combined, axis=axes, keepdims=True, ddof=1)
                rescaled = combined * (deviation / guided_deviation)
                combined = self.rescale * rescaled + (1.0 - self.rescale) * combined
            return denoise.convert(x, t, combined)

        return guided
