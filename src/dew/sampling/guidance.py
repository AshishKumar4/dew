"""Guidance as a wrapper around a denoiser.

Every guidance computes a second prediction next to the conditional one and
combines the two. `CFG` and `APG` evaluate the same model without its
condition, `CFGPlusPlus` also renoises with that unconditional prediction,
and `Autoguidance` evaluates a weaker model under the same condition.

A walk is one pass of the sampler over its N steps.
`guidance.walk(denoise, N)` applies a guidance to a denoiser over such a
walk and returns a `Walk`. The `Walk` gives each step's guided prediction
`(x_t, t) -> (x_0, epsilon)` and holds whatever the guidance keeps from one
step to the next. Only `APG` with momentum keeps anything: the walk's state
holds its running average of the guidance direction.

Every guidance's `interval` is the part of the walk it guides, as fractions
of the walk. The interval is closed at both ends, so step i of N is guided
when start <= i / N <= stop, and the closing denoise counts as step N. Each
step decides once whether it is guided, and every evaluation within the step
(a corrector, a midpoint, a stage) uses that decision. Kynkaanniemi et al.'s
own sampler and Diffusers' guiders decide the same way.

The paper's `guidance_interval=[a, b]` over N steps converts to
`interval=(a / N, b / N)`. A Diffusers guider's `start` and `stop` guide its
steps [int(start N), int(stop N)), which converts to
`interval=(int(start N) / N, (int(stop N) - 1) / N)`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.diffusion.process import Denoiser
from dew.nn.linear import _compensated_add

Predict = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
# A step's index in its walk: traced inside the walk's scan, a Python int at
# its closing denoise.
Index = jax.Array | int


@dataclass(frozen=True)
class Walk:
    """A guided denoiser for one walk, with the state the guidance keeps between steps.

    `init(x_T)` returns the state before the first step.
    `step(x, t, index, state)` returns the guided prediction at the start of
    step `index` and the state after that step. `at(state, index)` returns
    step `index`'s guided prediction function under a given state, without
    advancing the state. The solver calls it for its own evaluations within a
    step (a corrector, a midpoint) and for the closing denoise.
    """

    init: Callable[[jax.Array], Any]
    step: Callable[[jax.Array, jax.Array, Index, Any], tuple[tuple[jax.Array, jax.Array], Any]]
    at: Callable[[Any, Index], Predict]

    @classmethod
    def stateless(cls, predict: Callable[[Index], Predict]) -> Walk:
        """Return a walk that keeps no state between steps.

        `predict` maps a step's index to that step's guided prediction.
        """
        return cls(lambda x: (), lambda x, t, index, state: (predict(index)(x, t), state),
                   lambda state, index: predict(index))

    @classmethod
    def over(cls, denoise: Predict, guidance: Guidance | None, steps: int | jax.Array) -> Walk:
        """Return `guidance`'s walk of `denoise` over `steps` steps.

        When `guidance` is None, the walk is unguided. Raises `TypeError` when
        `guidance` is given and `denoise` is not a continuous `Denoiser`, as
        with the masked diffusion LM.
        """
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
    """Interval-limited classifier-free guidance.

    The guided prediction is uncond + scale (cond - uncond). Kynkaanniemi et
    al. 2024 found that guidance hurts at high noise and does not help at
    low noise, so outside `interval` the scale is 1, which gives exactly the
    plain conditional prediction. The interval is measured in progress along
    the trajectory, from 0 at pure noise to 1 at the clean sample, and the
    default covers all of it.

    `rescale` applies the guidance rescaling of Lin et al. 2023 ("Common
    Diffusion Noise Schedules and Sample Steps are Flawed", section 3.4),
    which Diffusers calls `guidance_rescale`. The guided output is rescaled
    to the per-sample standard deviation of the conditional output, then
    mixed with the unrescaled guided output at weight `rescale`. So 0 leaves
    the guided output alone and 1 uses the rescaled one. The standard
    deviation is taken over every axis but the batch axis, with the unbiased
    correction that the reference's `Tensor.std` applies.

    The guidance combines the model's raw outputs, and the denoiser then
    converts the combined output once. So a source's clipping, dynamic
    thresholding or consistency boundary is applied once, to the guided
    output, and never to the conditional or unconditional output alone.
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
    """Guides the clean prediction and renoises with the unconditional noise (CFG++).

    This is Algorithm 1 of Chung et al. 2025, "CFG++: Manifold-constrained
    Classifier Free Guidance for Diffusion Models". The clean prediction is
    uncond + scale (cond - uncond) with `scale` in [0, 1], and the noise a
    step renoises with is the unconditional prediction's noise. A `scale`
    outside [0, 1] raises `ValueError`.

    The guided pair (x_0, epsilon) is what DDIM's update, alpha x_0 + sigma
    epsilon, reads. A solver that integrates from x_t alone, without that
    pair, follows plain CFG's trajectory.
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


def _product_pair(left, right):
    """Dekker's two-float product keeps the bits an fp32 projection discards."""
    splitter = (1 << 27) + 1 if left.dtype == jnp.float64 else (1 << 12) + 1
    split_left, split_right = left * splitter, right * splitter
    left_hi, right_hi = split_left - (split_left - left), split_right - (split_right - right)
    left_lo, right_lo = left - left_hi, right - right_hi
    high = left * right
    low = ((left_hi * right_hi - high) + left_hi * right_lo + left_lo * right_hi) + left_lo * right_lo
    return high, low


def _projected_components(direction, output):
    """Diffusers projects in float64 and rounds the parallel and orthogonal
    components separately. Two-float products, sums and a corrected quotient
    keep the small orthogonal component through cancellation. The projected
    Euler walk matches its float64-projection twin bitwise over 52 spatial
    orders on CPU (default, AVX and SDE-ICX). Raw predictions meet the
    reference-error rule on an RTX 4080. TPU arithmetic has not been measured."""
    dtype = jnp.result_type(direction, output, jnp.float32)
    vector, conditioned = direction.astype(dtype), output.astype(dtype)
    axes = tuple(range(1, output.ndim))
    shape = (output.shape[0],) + (1,) * (output.ndim - 1)
    zero = jnp.asarray(0.0, dtype)
    # Powers of two move the exponent exactly, so finite inputs cannot
    # overflow either dot product before the quotient removes their scale.
    _, vector_exponent = jnp.frexp(jnp.max(jnp.abs(vector), axis=axes, keepdims=True))
    _, output_exponent = jnp.frexp(jnp.max(jnp.abs(conditioned), axis=axes, keepdims=True))
    vector_exponent, output_exponent = jnp.maximum(vector_exponent, 0), jnp.maximum(output_exponent, 0)
    vector, conditioned = jnp.ldexp(vector, -vector_exponent), jnp.ldexp(conditioned, -output_exponent)
    floor = jnp.ldexp(jnp.asarray(1e-24, dtype), -2 * output_exponent)

    def summed(left, right):
        high, low = jax.lax.reduce(_product_pair(left, right), (zero, zero), _compensated_add, axes)
        return high.reshape(shape), low.reshape(shape)

    numerator, numerator_lo = summed(vector, conditioned)
    denominator, denominator_lo = summed(conditioned, conditioned)
    denominator_lo = jnp.where(denominator < floor, 0.0, denominator_lo)
    denominator = jnp.maximum(denominator, floor)
    quotient = numerator / denominator
    product, product_lo = _product_pair(quotient, denominator)
    residual, residual_lo = _compensated_add(
        (numerator, numerator_lo), (-product, -product_lo - quotient * denominator_lo))
    correction = (residual + residual_lo) / denominator
    parallel, parallel_lo = _product_pair(quotient, conditioned)
    parallel_lo = parallel_lo + correction * conditioned
    orthogonal, orthogonal_lo = _compensated_add((vector, jnp.zeros_like(vector)), (-parallel, -parallel_lo))
    orthogonal = jnp.ldexp(orthogonal + orthogonal_lo, vector_exponent).astype(output.dtype)
    parallel = jnp.ldexp(parallel + parallel_lo, vector_exponent).astype(output.dtype)
    return orthogonal, parallel


@dataclass(frozen=True)
class APG:
    """Adaptive projected guidance, combining raw outputs as Diffusers' `AdaptiveProjectedGuidance` does.

    The method is from Sadat et al. 2025, "Eliminating Oversaturation and
    Artifacts of High Guidance Scales in Diffusion Models". The guided output
    is uncond + scale (orthogonal + eta parallel), where the guidance
    direction cond - uncond goes through three changes:

    - `momentum` (the paper's beta) keeps a running average of the direction
      over the walk. A negative value pushes away from earlier steps'
      updates, and 0 keeps no average.
    - The direction's norm, taken over every axis but the batch axis, is
      clipped at `norm_threshold`. 0 clips nothing.
    - The direction's component parallel to the conditional output is scaled
      by `eta`.

    With eta 1, no clipping and no momentum, this is CFG. Where guidance is
    off (outside `interval`, or at scale 1), the running average stays as it
    is, like Diffusers' buffer.
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
        orthogonal, parallel = _projected_components(direction, output)
        update = orthogonal + self.eta * parallel
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
    """Guides a model with a weaker version of itself (autoguidance).

    The method is from Karras et al. 2024, "Guiding a diffusion model with a
    bad version of itself", and NVlabs/edm2's `edm_sampler(gnet=...)` is the
    reference. The guided output is guide + scale (main - guide), where main
    is the output of the denoiser's own model. The guide is a weaker model
    for the same task, smaller or less trained, and it is evaluated under the
    same condition as the main model.

    `model` is the guide's module. Its variables go in the denoiser's
    variables under `guide`, so a compiled walk receives them as arguments
    together with the main model's; `walk` raises `ValueError` when they are
    missing. Both models use the same process, and their raw outputs are
    combined before the single conversion, as in `CFG`.
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


@runtime_checkable
class Guidance(Protocol):
    """What `sample` guides with: a walk over the steps for one denoiser.
    CFG, CFG++, APG and autoguidance are four; a run records any by class."""

    def walk(self, denoise: Denoiser, steps: int | jax.Array) -> Walk: ...


__all__ = ["APG", "CFG", "Autoguidance", "CFGPlusPlus", "Guidance", "Walk"]
