"""Guidance as a wrapper around a denoiser.

Every guidance reads a second prediction beside the conditional one and
combines the two. `CFG` and `APG` read the same model without its condition,
`CFGPlusPlus` renoises with that unconditional prediction, and `Autoguidance`
reads a weaker model under the same condition.

A guidance applied to a denoiser over a walk of N steps,
`guidance.walk(denoise, N)`, is a `Walk`: each step's guided
`(x_t, t) -> (x_0, epsilon)`, and whatever the guidance keeps from one step
to the next: nothing for all but `APG` with momentum, whose running average
of the guidance direction rides in the walk's state.

Every guidance's `interval` is the part of the walk it guides, in fractions
of the walk, closed at both ends: step i of N is guided when start <= i / N
<= stop, and the closing denoise is step N. A step decides once and every
evaluation within it (a corrector, a midpoint, a stage) takes that decision,
as Kynkaanniemi et al.'s own sampler and Diffusers' guiders decide. The
paper's `guidance_interval=[a, b]` over N steps is `interval=(a / N,
b / N)`, and a Diffusers guider's `start` and `stop`, which guide its steps
[int(start N), int(stop N)), are `interval=(int(start N) / N,
(int(stop N) - 1) / N)`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.diffusion.process import Denoiser

Predict = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
# A step's index in its walk: traced inside the walk's scan, a Python int at
# its closing denoise.
Index = jax.Array | int


@dataclass(frozen=True)
class Walk:
    """A guided denoiser over one walk and the state it carries.

    `init(x_T)` is the state before the first step. `step(x, t, index,
    state)` is the guided prediction step `index` starts from and the state
    after it. `at(state, index)` is step `index`'s guided prediction under a
    state without advancing it, which a solver's own evaluations within the
    step (a corrector, a midpoint) and the closing denoise read.
    """

    init: Callable[[jax.Array], Any]
    step: Callable[[jax.Array, jax.Array, Index, Any], tuple[tuple[jax.Array, jax.Array], Any]]
    at: Callable[[Any, Index], Predict]

    @classmethod
    def stateless(cls, predict: Callable[[Index], Predict]) -> Walk:
        """The walk of `predict`, which maps a step's index to its guided prediction."""
        return cls(lambda x: (), lambda x, t, index, state: (predict(index)(x, t), state),
                   lambda state, index: predict(index))

    @classmethod
    def over(cls, denoise: Predict, guidance: Guidance | None, steps: int | jax.Array) -> Walk:
        """`guidance`'s walk of `denoise` over `steps` steps, or the unguided walk."""
        if guidance is None:
            return cls.stateless(lambda index: denoise)
        if not isinstance(denoise, Denoiser):
            raise TypeError("guidance needs a continuous Denoiser; the masked diffusion LM takes none")
        return guidance.walk(denoise, steps)


# An interval's edges are read as the nearest fractions with denominators up
# to this, so a step's fraction that an edge rounds is that edge (0.3 of ten
# steps is step 3, though 0.3 * 10 is 3.0000000000000004 in binary), and the
# integer products a step's decision compares stay inside int32 for walks of
# up to 2**16 steps.
_DENOMINATOR = 1 << 15


def _scales(scale: float, interval: tuple[float, float],
            steps: int | jax.Array) -> Callable[[Index], jax.Array]:
    """Each step's scale over a walk of `steps` steps: `scale` on the steps i
    with start <= i / steps <= stop, 1 on the rest.

    The comparison is exact: i q >= p steps in integers against an edge p / q,
    so no rounding of a time or of i / steps moves an edge. `steps` may be
    traced, as a recorded rollout's is. A walk of no steps is its closing
    denoise alone, step 0.
    """
    start, stop = (Fraction(edge).limit_denominator(_DENOMINATOR) for edge in interval)
    steps = jnp.maximum(steps, 1)

    def at(index: Index) -> jax.Array:
        inside = ((index * start.denominator >= start.numerator * steps)
                  & (index * stop.denominator <= stop.numerator * steps))
        return jnp.where(inside, scale, 1.0)

    return at


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

    def _guided(self, denoise: Denoiser, scale: jax.Array) -> Predict:
        def guided(x, t):
            output, unconditional = denoise.raw_both(x, t)
            combined = unconditional + scale * (output - unconditional)
            if self.rescale:
                axes = tuple(range(1, combined.ndim))
                deviation = jnp.std(output, axis=axes, keepdims=True, ddof=1)
                guided_deviation = jnp.std(combined, axis=axes, keepdims=True, ddof=1)
                rescaled = combined * (deviation / guided_deviation)
                combined = self.rescale * rescaled + (1.0 - self.rescale) * combined
            return denoise.convert(x, t, combined)

        return guided

    def walk(self, denoise: Denoiser, steps: int | jax.Array) -> Walk:
        scales = _scales(self.scale, self.interval, steps)
        return Walk.stateless(lambda index: self._guided(denoise, scales(index)))


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

    def _guided(self, denoise: Denoiser, scale: jax.Array) -> Predict:
        def guided(x, t):
            output, unconditional = denoise.raw_both(x, t)
            clean, _ = denoise.convert(x, t, unconditional + scale * (output - unconditional))
            _, noise = denoise.convert(x, t, unconditional)
            return clean, noise

        return guided

    def walk(self, denoise: Denoiser, steps: int | jax.Array) -> Walk:
        scales = _scales(self.scale, self.interval, steps)
        return Walk.stateless(lambda index: self._guided(denoise, scales(index)))


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
    buffer does.
    """

    scale: float
    eta: float = 1.0
    norm_threshold: float = 15.0
    momentum: float = 0.0
    interval: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        object.__setattr__(self, "interval", _interval(self.interval))

    def _guided(self, denoise: Denoiser, scale: jax.Array, x, t, average):
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
        off = scale == 1.0
        combined = jnp.where(off, output, unconditional + scale * update)
        running = jnp.where(off, average, output - unconditional + self.momentum * average)
        return denoise.convert(x, t, combined), running

    def walk(self, denoise: Denoiser, steps: int | jax.Array) -> Walk:
        scales = _scales(self.scale, self.interval, steps)
        if not self.momentum:
            return Walk.stateless(
                lambda index: lambda x, t: self._guided(denoise, scales(index), x, t, 0.0)[0])

        def step(x, t, index, average):
            pair, running = self._guided(denoise, scales(index), x, t, average)
            return pair, running.astype(average.dtype)

        return Walk(jnp.zeros_like, step, lambda average, index: lambda x, t: self._guided(
            denoise, scales(index), x, t, average)[0])


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

    def walk(self, denoise: Denoiser, steps: int | jax.Array) -> Walk:
        if "guide" not in denoise.params:
            raise ValueError("autoguidance reads its guide's variables under `guide` in the "
                             "denoiser's variables")
        main = Denoiser(denoise.process, denoise.model,
                        {name: tree for name, tree in denoise.params.items() if name != "guide"},
                        denoise.conditions)
        guide = Denoiser(denoise.process, self.model, denoise.params["guide"], denoise.conditions)
        scales = _scales(self.scale, self.interval, steps)

        def guided(scale):
            def predict(x, t):
                output, weak = main.raw(x, t), guide.raw(x, t)
                return denoise.convert(x, t, weak + scale * (output - weak))

            return predict

        return Walk.stateless(lambda index: guided(scales(index)))


Guidance = CFG | CFGPlusPlus | APG | Autoguidance
"""Every guidance `sample` walks, a union `isinstance` reads."""


__all__ = ["APG", "CFG", "Autoguidance", "CFGPlusPlus", "Guidance", "Walk"]
