"""Flax's dense layers, with their bias gradients summed at fp32.

`Dense` and `DenseGeneral` are flax 0.12's layers, fields, parameter names,
shapes, initializers, promotion and product hooks alike, so checkpoints,
LoRA and Qwix see the layers they did. Their call is their own: the same
product, then the bias added by `biased`, which sums its gradient at fp32
and rounds the sum once to the dtype flax's addition gives it. The bias is
added inside the layer's own managed call, so a method interceptor (an
adapter, an input quantization) and an intermediate capture see the
rounded output as they saw flax's.

flax's own calls add the bias in its storage dtype, after the product, and
offer no step between the two; its Dense, DenseGeneral and Conv carry no
`preferred_element_type` (its Einsum and ConvTranspose do).
"""

from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import meta
from flax.typing import Array

from .precision import biased


def _product(layer: nn.Dense | nn.DenseGeneral):
    """The layer's product: its `dot_general_cls`, its `dot_general`, or lax's, as flax picks it."""
    if layer.dot_general_cls is not None:
        return layer.dot_general_cls()
    return layer.dot_general if layer.dot_general is not None else jax.lax.dot_general


def _reshaped(init, flat_shape):
    """`init` drawn at `flat_shape` and reshaped to the shape asked of it, a boxed draw (a
    partitioned initializer's) reshaped inside its box, as flax's DenseGeneral draws."""
    def drawn(rng, shape, dtype=jnp.float32):
        value = init(rng, tuple(int(size) for size in flat_shape(shape)), dtype)
        if isinstance(value, meta.AxisMetadata):
            return meta.replace_boxed(value, jnp.reshape(value.unbox(), shape))
        return jnp.reshape(value, shape)
    return drawn


def _promoted(layer: nn.Dense | nn.DenseGeneral, inputs: Array, kernel: Array, bias: Array | None):
    """The layer's operands through its `promote_dtype`, as flax promotes them."""
    inputs, kernel, bias = layer.promote_dtype(inputs, kernel, bias, dtype=layer.dtype)
    if inputs is None or kernel is None:
        raise ValueError("a dense layer's promotion must return its input and kernel")
    return inputs, kernel, bias


def _axes(axes: Iterable[int] | int, ndim: int | None = None) -> tuple[int, ...]:
    """`axes` as a tuple, as flax canonicalizes them, and with `ndim` given, a negative axis
    counted from the end and the whole sorted; an axis out of range stays out of range."""
    axes = tuple(axes) if isinstance(axes, Iterable) else (axes,)
    return axes if ndim is None else tuple(sorted(axis if axis >= 0 else ndim + axis for axis in axes))


class Dense(nn.Dense):
    """Flax's Dense, its bias added by `biased` (the module docstring)."""

    @nn.compact
    def __call__(self, inputs: Array) -> jax.Array:
        width = jnp.shape(inputs)[-1]
        kernel = self.param('kernel', self.kernel_init, (width, self.features), self.param_dtype)
        bias = (self.param('bias', self.bias_init, (self.features,), self.param_dtype)
                if self.use_bias else None)
        inputs, kernel, bias = _promoted(self, inputs, kernel, bias)
        product = _product(self)(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())),
                                 precision=self.precision)
        return product if bias is None else biased(product, bias)


class DenseGeneral(nn.DenseGeneral):
    """Flax's DenseGeneral, its bias added by `biased` (the module docstring)."""

    @nn.compact
    def __call__(self, inputs: Array) -> jax.Array:
        features, batch_dims = _axes(self.features), _axes(self.batch_dims)
        if batch_dims and set(batch_dims) != set(range(max(batch_dims) + 1)):
            raise ValueError(f'batch_dims {batch_dims} must be consecutive leading dimensions '
                             'starting from 0.')
        axis, batch_dims = _axes(self.axis, inputs.ndim), _axes(batch_dims, inputs.ndim)
        batched = len(batch_dims)
        batch_shape = tuple(inputs.shape[dim] for dim in batch_dims)
        kernel = self.param(
            'kernel',
            _reshaped(self.kernel_init, lambda shape: (np.prod(shape[:batched + len(axis)]),
                                                       np.prod(shape[-len(features):]))),
            batch_shape + tuple(inputs.shape[dim] for dim in axis) + features, self.param_dtype)
        bias = self.param(
            'bias', _reshaped(self.bias_init, lambda shape: (np.prod(shape[:batched])
                                                             * np.prod(shape[-len(features):]),)),
            batch_shape + features, self.param_dtype) if self.use_bias else None
        inputs, kernel, bias = _promoted(self, inputs, kernel, bias)
        contracted = tuple(range(batched, batched + len(axis)))
        product = _product(self)(inputs, kernel, ((axis, contracted), (batch_dims, tuple(range(batched)))),
                                 precision=self.precision)
        if bias is None:
            return product
        # The product is [batch dims, kept dims, features]; the bias broadcasts over the kept dims.
        kept = tuple(inputs.shape[dim] if dim in batch_dims else 1
                     for dim in range(inputs.ndim) if dim not in axis)
        return biased(product, jnp.reshape(bias, kept + features))
