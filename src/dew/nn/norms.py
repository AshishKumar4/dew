"""The normalizations Dew's modules share: RMS (`rms_normalized`, a residual stream's
`stream_rms_normalized`, a gated one), group, L2 and unweighted RMS forms, each computed in at least
float32, and their modules (`RMSNorm`, `LayerNorm`, `GroupNorm`).
"""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import canonicalize_dtype
from flax.typing import Dtype

from .kernels import norm_swiglu
from .precision import at_least_fp32, rms_statistics, rounded_to


def normalized_in_fp32(normalize, static_argnums: tuple[int, ...] = ()):
    """Wrap a norm body in `jax.checkpoint` so nothing fp32 is retained.

    The norms reduce in fp32 for bf16 stability, and differentiated as written
    the backward pass would keep two fp32 copies of the activations per norm
    (2.35 GiB on one SimpleDiT step). Saving nothing keeps the arguments (the
    bf16 input, the weight and the bias) and leaves the residuals to the policy
    that recomputes the block (`mixer_base.RESIDUALS`). The wrapped function
    takes arrays first and static arguments last, so no flax lifting sits
    between the checkpoint and the module; forward values are unchanged.
    """
    return jax.checkpoint(normalize, policy=jax.checkpoint_policies.nothing_saveable,
                          static_argnums=static_argnums)


@functools.partial(normalized_in_fp32, static_argnums=(2, 3, 4, 5, 6, 7))
def _xla_rms_normalized(x, scale, epsilon: float, dtype, scale_offset: bool, scale_after_cast: bool,
                        fp32_statistics: bool, activation=None):
    """Normalize `x` by its root mean square, then apply `scale`.

    The families differ on which side of the cast the weight goes, which
    `scale_after_cast` picks. `scale` of None is the weightless norm.
    `fp32_statistics` of False computes the square, the mean and the
    normalization in the input's dtype. `activation` (None for none) runs
    on the result before its last cast to `dtype`, as `GroupNorm`'s does.
    """
    activated = (lambda y: y) if activation is None else activation
    if fp32_statistics:
        y = x.astype(at_least_fp32(x.dtype))
        y = y * rms_statistics(y, epsilon)[1]
    else:
        # The reference rounds every step to the input dtype. XLA fuses the
        # chain and carries fp32 between the ops, so each rounding is explicit.
        y = rounded_to(x * rounded_to(jax.lax.rsqrt(rounded_to(
            rounded_to(jnp.mean(rounded_to(jnp.square(x), x.dtype), axis=-1, keepdims=True), x.dtype)
            + epsilon, x.dtype)), x.dtype), x.dtype)
    if scale is None:
        # A pure normalization with no learned weight, as Gemma 4 norms
        # its values (modeling_gemma4.py, Gemma4RMSNorm with_scale=False).
        return activated(y).astype(dtype)
    weight = (1.0 + scale) if scale_offset else scale
    if scale_after_cast:
        # The reference casts the activations, then multiplies by its weight:
        # a bf16 product for a bf16 weight, an fp32 one for an fp32 master
        # that the next layer rounds. Rounding in place keeps both roundings
        # under jit, where XLA drops a narrowing cast that a widening one
        # follows, and keeps the weight's product and gradient in fp32.
        return rounded_to(activated(rounded_to(y, dtype) * weight), dtype).astype(dtype)
    return activated(y * weight).astype(dtype)


def rms_normalized(x, scale, epsilon: float, dtype, scale_offset: bool, scale_after_cast: bool,
                   fp32_statistics: bool, *, implementation: str = 'auto', axes=None):
    """Normalize the last axis and apply an optional vector weight.

    `scale_offset` adds one to the stored weight; `scale_after_cast` rounds
    the normalized activation before multiplying and rounds the product
    again. `fp32_statistics=False` retains the input-dtype arithmetic.
    `implementation` selects the generation's measured kernel by default;
    off a GPU, 'pallas' too runs XLA's arithmetic. The fused VJP retains only the
    narrow input, weight and per-row mean square, not fp32 activations.
    Unsupported variants run the checkpointed XLA reference instead
    (`norm_swiglu.rms_kernel`), including float64 and input-dtype statistics.
    `axes`, `x`'s logical axes, place the kernel's rows on a mesh; without
    them a split mesh runs XLA.
    """
    return stream_rms_normalized(x, None, scale, epsilon, dtype, scale_offset, scale_after_cast,
                                 fp32_statistics, implementation=implementation, axes=axes)[1]


def stream_rms_normalized(x, branch, scale, epsilon: float, dtype, scale_offset: bool,
                          scale_after_cast: bool, fp32_statistics: bool, *, implementation: str = 'auto',
                          axes=None, activation=None):
    """A pre-norm block's seam: the residual stream `x` plus `branch` (None
    adds nothing) and that stream's `rms_normalized`, as `(stream, normed)`.

    The caller carries the returned stream on, so the fused kernel
    (`norm_swiglu.stream_rms_norm`) owns both ends of the norm: its forward
    reads the two addends once and writes the sum beside the norm, and its
    reverse adds the stream's own cotangent to the norm's in the same pass.
    XLA's add and norm are the portable path and the oracle. `activation`
    (None for none) runs on the norm before its last cast, on XLA's path.
    """
    chosen = norm_swiglu.rms_kernel(implementation, x, scale, dtype, fp32_statistics, branch=branch,
                                    axes=axes, activated=activation is not None)
    def portable(x, branch, scale):
        stream = x if branch is None else x + branch
        return stream, _xla_rms_normalized(stream, scale, epsilon, dtype, scale_offset, scale_after_cast,
                                           fp32_statistics, activation)

    if chosen == 'pallas':
        return norm_swiglu.stream_rms_norm(x, branch, scale, epsilon, dtype, scale_offset, scale_after_cast,
                                           portable, axes=axes)
    return portable(x, branch, scale)


def gated_rms_normalized(x, gate, scale, epsilon: float, dtype, activation: str, *,
                         implementation: str = 'auto', axes=None):
    """Qwen3-Next's gated norm (`Qwen3NextRMSNormGated`,
    modeling_qwen3_next.py:57-74): `x` normalized in fp32, times `scale`,
    times `activation(gate)` ('silu' or 'sigmoid'), cast once to `dtype`.
    The fused kernel (`norm_swiglu.gated_rms_norm`) reads both inputs once a
    pass; XLA's arithmetic is the portable path and the oracle. `axes`, the
    inputs' logical axes, place the kernel's rows on a mesh.
    """
    def portable(x, gate, scale):
        wide = at_least_fp32(x.dtype)
        y = unweighted_rmsnorm(x.astype(wide), epsilon, implementation='xla')
        y = y if scale is None else y * scale.astype(wide)
        gate = gate.astype(wide)
        return (y * (jax.nn.silu(gate) if activation == 'silu' else jax.nn.sigmoid(gate))).astype(dtype)

    chosen = norm_swiglu.gated_rms_kernel(implementation, x, gate, scale, dtype, activation, axes=axes)
    if chosen == 'pallas':
        return norm_swiglu.gated_rms_norm(x, gate, scale, epsilon, dtype, activation, portable, axes=axes)
    return portable(x, gate, scale)


def _moment_normalized(x, scale, bias, axes: tuple[int, ...], epsilon: float, requested):
    """`x` normalized over `axes`, reduced in the module's `requested` dtype,
    else `x`'s, at least fp32, as flax lays the computation out
    (flax.linen.normalization._compute_stats) but for the variance: the
    centered mean square E[(x - E[x])^2], as torch's GroupNorm takes it, where
    flax takes E[x^2] - E[x]^2. That difference cancels where a group's mean
    dwarfs its spread, and its gradient carries the loss: FLUX.2's published
    VAE's input gradient sat twice torch's distance from float64, its worst
    seeds 6 to 10 times. The centered form has no such cancellation, at one
    more pass over the group. The weight is folded into the inverse deviation
    before it meets the centered activations. `scale` and `bias` (None for
    none) broadcast against `x`'s trailing axes."""
    statistics = at_least_fp32(x.dtype if requested is None else requested)
    reduced = x.astype(statistics)
    mean = jnp.mean(reduced, axis=axes, keepdims=True)
    variance = jnp.mean(jax.lax.square(reduced - mean), axis=axes, keepdims=True)
    scaling = jax.lax.rsqrt(variance + epsilon)
    if scale is not None:
        scaling = scaling * scale
    # flax centers the input as it arrived, promoted with the statistics' dtype.
    y = (x.astype(jnp.promote_types(x.dtype, statistics)) - mean) * scaling
    return y if bias is None else y + bias


@functools.partial(normalized_in_fp32, static_argnums=(3, 4, 5, 6, 7))
def group_normalized(x, scale, bias, groups: int, epsilon: float, activation, requested, dtype):
    """`x` `[batch, ..., channels]` normalized over each of `groups` channel
    groups and every axis but the batch, as flax's `GroupNorm` lays it out, its
    variance centered as torch's (`_moment_normalized`). `activation` (None for none) runs on the result
    before its one cast to `dtype`."""
    grouped = x.reshape(*x.shape[:-1], groups, x.shape[-1] // groups)
    scale, bias = (None if value is None else value.reshape(grouped.shape[-2:]) for value in (scale, bias))
    y = _moment_normalized(grouped, scale, bias, (*range(1, x.ndim - 1), x.ndim), epsilon, requested)
    y = y.reshape(x.shape)
    return (y if activation is None else activation(y)).astype(dtype)


def l2_normalized(x, eps: float = 1e-12):
    """torch's `F.normalize`: `x` over its L2 norm on the last axis, the norm
    held at `eps` at least.

    The norm is the root of the squared norm held at `eps`², the same value,
    so a zero vector's gradient is zero: the root of a sum that is zero has
    an infinite slope, which the floor's zero gradient would turn into NaN.
    The norm is then held at `eps` again, a no-op that keeps the division off
    a bare square root, which XLA rewrites to a product with rsqrt, so the
    quotient rounds as torch's does."""
    norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(x), axis=-1, keepdims=True), eps * eps))
    return x / jnp.maximum(norm, eps)


def unweighted_rmsnorm(x, eps: float, *, implementation: str = 'auto'):
    """Normalize the last axis by its root mean square, with no learned weight.

    The reduction runs in fp32 and the inverse deviation is cast back
    before it multiplies, which is what `DeepseekV4UnweightedRMSNorm`
    (modeling_deepseek_v4.py:66-72) and its GLM twin do; the fused RMSNorm
    takes it where the target's row names it, rounding the inverse as they do
    (`norm_swiglu.rms_norm`'s `inverse_rounded`).
    """
    if norm_swiglu.rms_kernel(implementation, x, None, x.dtype, fp32_statistics=True) == 'pallas':
        def portable(x, scale):
            return unweighted_rmsnorm(x, eps, implementation='xla')

        return norm_swiglu.rms_norm(x, None, eps, x.dtype, offset=False, after_cast=False, portable=portable,
                                    inverse_rounded=True)
    fp32 = x.astype(at_least_fp32(x.dtype))
    inverse = rms_statistics(fp32, eps)[1]
    return x * inverse.astype(x.dtype)


class RMSNorm(nn.Module):
    """Normalize the last axis by its root mean square, reducing in fp32 by default.

    `scale_offset` stores the weight as Gemma does, (1 + w), initialized to
    zeros, so a Gemma checkpoint's weights land unchanged. Gemma multiplies in
    fp32 and casts the product (modeling_gemma3.py:147-150); Llama and Qwen3
    cast first and multiply by the weight (modeling_qwen3.py:61-64), which
    `scale_after_cast` reproduces. The two agree at fp32 and differ under bf16.
    `fp32_statistics=False` with `scale_after_cast` is timm's `RmsNorm2d`
    order, x * rsqrt(mean(x^2) + eps) * w in the input dtype. The backward pass
    keeps the input in its own dtype (`normalized_in_fp32`). `activation`
    (None for none) runs on the norm before its last cast, as `GroupNorm`'s.
    `groups` takes the statistics over each of that many equal, contiguous
    parts of the axis, under one weight over the whole axis, as
    `K2HorizonRMSNorm` does (modeling_k2_horizon.py:613-636).
    """
    epsilon: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    with_scale: bool = True
    fp32_statistics: bool = True
    dtype: Dtype | None = None
    implementation: str = 'auto'
    activation: Callable[[jax.Array], jax.Array] | None = None
    groups: int = 1

    def __call__(self, x, *, axes=None):
        """The norm of `x`. `axes`, `x`'s logical axes, place the fused kernel on a mesh."""
        return self.seam(x, axes=axes)[1]

    @nn.compact
    def seam(self, x, branch=None, *, axes=None):
        """A pre-norm block's residual seam: `(stream, normed)` for the stream
        `x + branch` (`x` for a None branch) and its norm, whose fused kernel
        owns the add and the stream's cotangent (`stream_rms_normalized`)."""
        dtype = self.dtype if self.dtype is not None else x.dtype
        scale = self.param(
            'scale',
            nn.initializers.zeros if self.scale_offset else nn.initializers.ones,
            (x.shape[-1],), jnp.float32) if self.with_scale else None
        if self.groups == 1:
            return stream_rms_normalized(x, branch, scale, self.epsilon, dtype, self.scale_offset,
                                         self.scale_after_cast, self.fp32_statistics,
                                         implementation=self.implementation, axes=axes,
                                         activation=self.activation)
        if x.shape[-1] % self.groups:
            raise ValueError(f"a {x.shape[-1]}-wide norm splits into no {self.groups} equal groups")

        def grouped(value):
            return None if value is None else value.reshape(*value.shape[:-1], self.groups, -1)

        # The residual summed first, as it broadcasts; then the groups' own last axis, which the one
        # norm body reduces over under the weight laid out alike, the width's logical axis naming
        # the groups and each group's own span none.
        stream = x if branch is None else x + branch
        _, normed = stream_rms_normalized(
            grouped(stream), None, grouped(scale), self.epsilon, dtype, self.scale_offset,
            self.scale_after_cast, self.fp32_statistics, implementation=self.implementation,
            axes=None if axes is None else (*axes, None), activation=self.activation)
        return stream, normed.reshape(stream.shape)


class LayerNorm(nn.remat(nn.LayerNorm, policy=jax.checkpoint_policies.nothing_saveable)):
    """flax's `nn.LayerNorm`, rematerialized saving nothing: its backward pass keeps the input as
    it arrived and its parameters, not the two fp32 copies flax's own differentiation keeps
    (2.35 GiB of them on one SimpleDiT step). The parameter tree, the arithmetic
    and the module's name are flax's, so checkpoints and scopes move both ways."""


class GroupNorm(nn.Module):
    """Flax's `nn.GroupNorm` over the last axis, with its parameter tree, its
    variance centered as torch's GroupNorm takes it (`_moment_normalized`),
    and an `activation` applied to its fp32 output before the one cast to
    `dtype`.

    Diffusers' norms keep their fp32 affine output into the activation and
    the next layer casts its result (vae_flax.py:154 and its UNet), so a
    norm that rounded first and activated after would round where they do
    not. The backward pass keeps the input as it arrived
    (`normalized_in_fp32`), so the activation's fp32 input is recomputed,
    not stored.
    """
    num_groups: int = 32
    epsilon: float = 1e-6
    activation: Callable[[jax.Array], jax.Array] | None = None
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        width = (x.shape[-1],)
        scale = self.param('scale', nn.initializers.ones, width, self.param_dtype)
        bias = self.param('bias', nn.initializers.zeros, width, self.param_dtype)
        return group_normalized(x, scale, bias, self.num_groups, self.epsilon, self.activation, self.dtype,
                                canonicalize_dtype(x, scale, bias, dtype=self.dtype))
