"""Guidance as a wrapper around a denoiser.

Every guidance reads a second prediction beside the conditional one and
combines the two. `CFG` and `APG` read the same model without its condition,
`CFGPlusPlus` renoises with that unconditional prediction, and `Autoguidance`
reads a weaker model under the same condition.

A guidance applied to a denoiser, `guidance(denoise)`, is the guided
`(x_t, t) -> (x_0, epsilon)`. `sample` walks it through `walk(denoise)`, a
`Walk` that also carries whatever the guidance keeps from one step to the
next: nothing for all but `APG` with momentum, whose running average of the
guidance direction rides in the walk's state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.diffusion.process import Denoiser
from dew.diffusion.schedules import expand

Predict = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]


@dataclass(frozen=True)
class Walk:
    """A guided denoiser over one trajectory and the state it carries.

    `init(x_T)` is the state before the first step. `step(x, t, state)` is
    the guided prediction a step starts from and the state after it. `at`
    is the guided prediction under a state without advancing it, which a
    solver's own extra evaluations (a corrector, a midpoint) read.
    """

    init: Callable[[jax.Array], Any]
    step: Callable[[jax.Array, jax.Array, Any], tuple[tuple[jax.Array, jax.Array], Any]]
    at: Callable[[Any], Predict]

    @classmethod
    def stateless(cls, predict: Predict) -> Walk:
        return cls(lambda x: (), lambda x, t, state: (predict(x, t), state), lambda state: predict)


def _scale(denoise: Denoiser, scale: float, interval: tuple[float, float], x, t) -> jax.Array:
    """`scale` inside `interval` and 1 outside it, shaped against `x`.

    Progress is the fraction of the trajectory walked, so it lives in
    [0, 1]: t = T is 0 and the terminal point past the grid's end is 1.
    Clamping it there is that definition, and it also keeps the top of a walk
    inside the default interval, which a fused 1 - t / T can miss by an ulp.
    """
    progress = jnp.clip(1.0 - jnp.asarray(t, jnp.float32) / denoise.process.sampler_schedule.T,
                        0.0, 1.0)
    start, stop = interval
    return expand(jnp.where((progress >= start) & (progress <= stop), scale, 1.0), x)


def _interval(interval) -> tuple[float, float]:
    # A record's interval arrives as a list, from a run's json or a command
    # line; a tuple keeps the value hashable, so it can ride into a jit as a
    # static argument.
    start, stop = (float(edge) for edge in interval)
    return start, stop


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
        object.__setattr__(self, "interval", _interval(self.interval))
        object.__setattr__(self, "rescale", float(self.rescale))

    def __call__(self, denoise: Denoiser) -> Predict:
        def guided(x, t):
            output, unconditional = denoise.raw_both(x, t)
            scale = _scale(denoise, self.scale, self.interval, x, t)
            combined = unconditional + scale * (output - unconditional)
            if self.rescale:
                axes = tuple(range(1, combined.ndim))
                deviation = jnp.std(output, axis=axes, keepdims=True, ddof=1)
                guided_deviation = jnp.std(combined, axis=axes, keepdims=True, ddof=1)
                rescaled = combined * (deviation / guided_deviation)
                combined = self.rescale * rescaled + (1.0 - self.rescale) * combined
            return denoise.convert(x, t, combined)

        return guided

    def walk(self, denoise: Denoiser) -> Walk:
        return Walk.stateless(self(denoise))


@dataclass(frozen=True)
class CFGPlusPlus:
    """CFG++ (Chung et al. 2025, "CFG++: Manifold-constrained Classifier Free
    Guidance for Diffusion Models", Algorithm 1): the clean prediction is
    guided, uncond + scale (cond - uncond) with `scale` in [0, 1], and the
    noise a step renoises with is the unconditional prediction's.

    The pair it returns is what DDIM's update, alpha x_0 + sigma epsilon,
    reads; a solver that integrates from x_t alone rather than from that
    pair takes plain CFG's trajectory instead.
    """

    scale: float
    interval: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        if not 0.0 <= self.scale <= 1.0:
            raise ValueError(f"CFG++ interpolates with a scale in [0, 1], got {self.scale}")
        object.__setattr__(self, "interval", _interval(self.interval))

    def __call__(self, denoise: Denoiser) -> Predict:
        def guided(x, t):
            output, unconditional = denoise.raw_both(x, t)
            scale = _scale(denoise, self.scale, self.interval, x, t)
            clean, _ = denoise.convert(x, t, unconditional + scale * (output - unconditional))
            _, noise = denoise.convert(x, t, unconditional)
            return clean, noise

        return guided

    def walk(self, denoise: Denoiser) -> Walk:
        return Walk.stateless(self(denoise))


@dataclass(frozen=True)
class APG:
    """Adaptive projected guidance (Sadat et al. 2025, "Eliminating
    Oversaturation and Artifacts of High Guidance Scales in Diffusion
    Models"), as Diffusers' `AdaptiveProjectedGuidance` combines raw outputs.

    The guidance direction cond - uncond is averaged with `momentum` over the
    walk (the paper's beta, negative to push away from earlier steps' update;
    0 keeps none), its norm over everything but the batch axis is clipped at
    `norm_threshold` (0 clips nothing), and its component parallel to the
    conditional output is scaled by `eta`: uncond + scale (orthogonal + eta
    parallel). eta 1 without clipping or momentum is CFG. Where guidance is
    off (outside `interval`, or at scale 1) the average rests, as Diffusers'
    buffer does; Diffusers' interval counts steps, [int(start N), int(stop
    N)), where this one is closed in progress, so the two agree without one.
    """

    scale: float
    eta: float = 1.0
    norm_threshold: float = 15.0
    momentum: float = 0.0
    interval: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        object.__setattr__(self, "interval", _interval(self.interval))

    def _guided(self, denoise: Denoiser, x, t, average):
        output, unconditional = denoise.raw_both(x, t)
        direction = output - unconditional + self.momentum * average
        axes = tuple(range(1, direction.ndim))
        if self.norm_threshold > 0:
            norm = jnp.sqrt(jnp.sum(jnp.square(direction), axis=axes, keepdims=True))
            direction = direction * jnp.minimum(1.0, self.norm_threshold / norm)
        unit = output / jnp.maximum(
            jnp.sqrt(jnp.sum(jnp.square(output), axis=axes, keepdims=True)), 1e-12)
        parallel = jnp.sum(direction * unit, axis=axes, keepdims=True) * unit
        update = direction - parallel + self.eta * parallel
        scale = _scale(denoise, self.scale, self.interval, x, t)
        off = scale == 1.0
        combined = jnp.where(off, output, unconditional + scale * update)
        running = jnp.where(off, average, output - unconditional + self.momentum * average)
        return denoise.convert(x, t, combined), running

    def __call__(self, denoise: Denoiser) -> Predict:
        if self.momentum:
            raise ValueError("APG's momentum runs over a walk; sample it with `sample`, which "
                             "carries the running average")
        return lambda x, t: self._guided(denoise, x, t, 0.0)[0]

    def walk(self, denoise: Denoiser) -> Walk:
        if not self.momentum:
            return Walk.stateless(self(denoise))

        def step(x, t, average):
            pair, running = self._guided(denoise, x, t, average)
            return pair, running.astype(average.dtype)

        return Walk(jnp.zeros_like, step,
                    lambda average: lambda x, t: self._guided(denoise, x, t, average)[0])


@dataclass(frozen=True)
class Autoguidance:
    """Autoguidance (Karras et al. 2024, "Guiding a diffusion model with a bad
    version of itself"): guide + scale (model - guide), where the guide is a
    weaker model of the same task, smaller or less trained, read under the
    same condition. NVlabs/edm2's `edm_sampler(gnet=...)` is the reference.

    `model` is the guide's module; its variables ride in the denoiser's under
    `guide`, so they reach a compiled walk as arguments beside the model's.
    Both share the process, and their raw outputs combine before the one
    conversion, as `CFG`'s do.
    """

    scale: float
    model: nn.Module
    interval: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        object.__setattr__(self, "interval", _interval(self.interval))

    def __call__(self, denoise: Denoiser) -> Predict:
        if "guide" not in denoise.params:
            raise ValueError("autoguidance reads its guide's variables under `guide` in the "
                             "denoiser's variables")
        main = Denoiser(denoise.process, denoise.model,
                        {name: tree for name, tree in denoise.params.items() if name != "guide"},
                        denoise.conditions)
        guide = Denoiser(denoise.process, self.model, denoise.params["guide"], denoise.conditions)

        def guided(x, t):
            output, weak = main.raw(x, t), guide.raw(x, t)
            scale = _scale(denoise, self.scale, self.interval, x, t)
            return denoise.convert(x, t, weak + scale * (output - weak))

        return guided

    def walk(self, denoise: Denoiser) -> Walk:
        return Walk.stateless(self(denoise))


Guidance = CFG | CFGPlusPlus | APG | Autoguidance
"""Every guidance `sample` walks, a union `isinstance` reads."""


__all__ = ["APG", "CFG", "Autoguidance", "CFGPlusPlus", "Guidance", "Walk"]
