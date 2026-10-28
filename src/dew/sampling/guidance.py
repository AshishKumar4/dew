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

Each guidance is a pytree: its scale (and APG's eta) a leaf, so a compiled
walk takes another value as data; its interval and the switches that shape the
program (CFG's rescale, APG's clipping and momentum, a guide's module) are
static.

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
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import linen as nn, struct

from dew.diffusion.process import Denoiser
from dew.diffusion.schedules import expand
from dew.nn.precision import compensated_add, two_product

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


def _inside(interval: tuple[float, float], steps: int | jax.Array) -> Callable[[Index], jax.Array]:
    """Whether each step of a walk of `steps` steps guides: the steps i with start <= i / steps <=
    stop.

    The comparison is exact: i q >= p steps in integers against an edge p / q,
    so no rounding of a time or of i / steps moves an edge. `steps` may be
    traced, as a recorded rollout's is. A walk of no steps is its closing
    denoise alone, step 0.
    """
    start, stop = (Fraction(edge).limit_denominator(_DENOMINATOR) for edge in interval)
    steps = jnp.maximum(steps, 1)

    def at(index: Index) -> jax.Array:
        return ((index * start.denominator >= start.numerator * steps)
                & (index * stop.denominator <= stop.numerator * steps))

    return at


def _scales(scale: float, interval: tuple[float, float],
            steps: int | jax.Array) -> Callable[[Index], jax.Array]:
    """Each step's scale over a walk of `steps` steps: `scale` on the steps that guide (`_inside`),
    1 on the rest."""
    inside = _inside(interval, steps)
    return lambda index: jnp.where(inside(index), scale, 1.0)


def _interval(interval) -> tuple[float, float]:
    # A record's interval arrives as a list, from a run's json or a command
    # line; a tuple keeps the value hashable, so it can ride into a jit as a
    # static argument.
    start, stop = (float(edge) for edge in interval)
    return start, stop


def guide(conditional: jax.Array, unconditional: jax.Array, scale) -> jax.Array:
    """Classifier-free guidance's affine combination, unconditional + scale (conditional -
    unconditional), of two predictions of one sample: `scale` one value, or one per row. Every
    guidance here combines through it: the samplers' (CFG, CFG++, autoguidance's weaker model in
    the unconditional's place) and the distillations' teacher targets. At a unit scale, which a step
    outside its guidance interval takes, it is the conditional prediction exactly, as the sources
    return it unguided there: the affine form would round it through their difference (float32's
    1 against 2**30 gives 0)."""
    # The scale in the predictions' dtype, so an integer scale (CFG++ at 1) has a tangent.
    return _guided(conditional, unconditional, expand(jnp.asarray(scale, conditional.dtype), conditional))


@jax.custom_jvp
def _guided(conditional: jax.Array, unconditional: jax.Array, scale: jax.Array) -> jax.Array:
    """`guide`'s value: the conditional prediction exactly at a unit scale, the affine combination
    elsewhere."""
    return jnp.where(scale == 1, conditional, unconditional + scale * (conditional - unconditional))


@_guided.defjvp
def _guided_jvp(primals, tangents):
    """The affine combination's derivative at every scale, a unit one's included, where the exact
    conditional the value selects would leave the scale's out."""
    conditional, unconditional, scale = primals
    along_conditional, along_unconditional, along_scale = tangents
    # The predictions' direction in the affine order, the conditional's own at a unit scale.
    predicted = jnp.where(scale == 1, along_conditional,
                          along_unconditional + scale * (along_conditional - along_unconditional))
    # The scale's: the predictions' difference, halved first only where it would pass float32's largest,
    # so two finite predictions of opposite sign stay finite where the scale does not move.
    wide = jnp.maximum(jnp.abs(conditional), jnp.abs(unconditional)) > jnp.finfo(conditional.dtype).max / 2
    spread = jnp.where(wide, 0.5 * conditional - 0.5 * unconditional, conditional - unconditional)
    return _guided(*primals), predicted + jnp.where(wide, 2, 1) * (spread * along_scale)


@struct.dataclass
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
    interval: tuple[float, float] = struct.field(pytree_node=False, default=(0.0, 1.0))
    rescale: float = struct.field(pytree_node=False, default=0.0)

    def __post_init__(self):
        object.__setattr__(self, "interval", _interval(self.interval))
        object.__setattr__(self, "rescale", float(self.rescale))

    def _guided(self, denoise: Denoiser, scale: jax.Array) -> Predict:
        def guided(x, t):
            output, unconditional = denoise.raw_both(x, t)
            combined = guide(output, unconditional, scale)
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


@struct.dataclass
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
    interval: tuple[float, float] = struct.field(pytree_node=False, default=(0.0, 1.0))

    def __post_init__(self):
        if not isinstance(self.scale, jax.Array) and not 0.0 <= self.scale <= 1.0:
            raise ValueError(f"CFG++ interpolates with a scale in [0, 1], got {self.scale}")
        object.__setattr__(self, "interval", _interval(self.interval))

    def _guided(self, denoise: Denoiser, active: jax.Array) -> Predict:
        """The guided pair on a step that guides, and the conditional denoiser's own pair on one
        outside the interval, where the unconditional noise would still steer the walk."""
        def guided(x, t):
            output, unconditional = denoise.raw_both(x, t)
            renoising = jnp.where(active, unconditional, output)
            clean, _ = denoise.convert(x, t, guide(output, renoising, self.scale))
            _, noise = denoise.convert(x, t, renoising)
            return clean, noise

        return guided

    def walk(self, denoise: Denoiser, steps: int | jax.Array) -> Walk:
        inside = _inside(self.interval, steps)
        return Walk.stateless(lambda index: self._guided(denoise, inside(index)))


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
        high, low = jax.lax.reduce(two_product(left, right), (zero, zero), compensated_add, axes)
        return high.reshape(shape), low.reshape(shape)

    numerator, numerator_lo = summed(vector, conditioned)
    denominator, denominator_lo = summed(conditioned, conditioned)
    denominator_lo = jnp.where(denominator < floor, 0.0, denominator_lo)
    denominator = jnp.maximum(denominator, floor)
    quotient = numerator / denominator
    product, product_lo = two_product(quotient, denominator)
    residual, residual_lo = compensated_add(
        (numerator, numerator_lo), (-product, -product_lo - quotient * denominator_lo))
    correction = (residual + residual_lo) / denominator
    parallel, parallel_lo = two_product(quotient, conditioned)
    parallel_lo = parallel_lo + correction * conditioned
    orthogonal, orthogonal_lo = compensated_add((vector, jnp.zeros_like(vector)), (-parallel, -parallel_lo))
    orthogonal = jnp.ldexp(orthogonal + orthogonal_lo, vector_exponent).astype(output.dtype)
    parallel = jnp.ldexp(parallel + parallel_lo, vector_exponent).astype(output.dtype)
    return orthogonal, parallel


@struct.dataclass
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
    norm_threshold: float = struct.field(pytree_node=False, default=15.0)
    momentum: float = struct.field(pytree_node=False, default=0.0)
    interval: tuple[float, float] = struct.field(pytree_node=False, default=(0.0, 1.0))

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


@struct.dataclass
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
    model: nn.Module = struct.field(pytree_node=False)
    interval: tuple[float, float] = struct.field(pytree_node=False, default=(0.0, 1.0))

    def __post_init__(self):
        object.__setattr__(self, "interval", _interval(self.interval))

    def walk(self, denoise: Denoiser, steps: int | jax.Array) -> Walk:
        if "guide" not in denoise.params:
            raise ValueError("autoguidance reads its guide's variables under `guide` in the "
                             "denoiser's variables")
        main = replace(denoise, unconditional=None,
                       params={name: tree for name, tree in denoise.params.items() if name != "guide"})
        weaker = replace(denoise, model=self.model, params=denoise.params["guide"], unconditional=None)
        scales = _scales(self.scale, self.interval, steps)

        def guided(scale):
            def predict(x, t):
                output, weak = main.raw(x, t), weaker.raw(x, t)
                return denoise.convert(x, t, guide(output, weak, scale))

            return predict

        return Walk.stateless(lambda index: guided(scales(index)))


@runtime_checkable
class Guidance(Protocol):
    """What `sample` guides with: a walk over the steps for one denoiser.
    CFG, CFG++, APG and autoguidance are four; a run records any by class."""

    def walk(self, denoise: Denoiser, steps: int | jax.Array) -> Walk: ...


def guidance_modules(guidance: Guidance | None) -> tuple[nn.Module, ...]:
    """The modules `guidance` runs besides its denoiser's model, autoguidance's guide, which a
    program that walks it compiles for as it does for the denoiser (`compiler_options`)."""
    return (guidance.model,) if isinstance(guidance, Autoguidance) else ()


__all__ = ["APG", "CFG", "Autoguidance", "CFGPlusPlus", "Guidance", "Walk"]
