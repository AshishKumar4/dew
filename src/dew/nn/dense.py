"""Flax's dense layers, with their bias gradients summed at fp32 (in float64 on a CPU under Dew's tests).

`Dense` and `DenseGeneral` are flax 0.12's layers, fields, parameter names,
shapes, initializers, promotion and product hooks alike, so checkpoints,
LoRA and Qwix see the layers they did, but for a batched `DenseGeneral`'s
kernel, which draws each batch member alone (`_members`). Their call is their own: the same
product, rounded to its dtype as flax's is, then the bias added by `biased`,
which adds it in the dtype flax's addition gives them and broadcasts it at
fp32, so its gradient sums at fp32 as torch's addmm sums it, and in float64 on a CPU under
Dew's tests, as such a CPU sums every float32 product (`dew.nn.precision.accumulated`). The bias is
added inside the layer's own managed call, so a method interceptor (an
adapter, an input quantization) and an intermediate capture see the
output as they saw flax's.

flax's own calls broadcast the bias in its storage dtype, where its gradient
then sums, and offer no step to widen it.
"""

import functools
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import meta
from flax.typing import Array

from .precision import accumulated, biased


def _product(layer: nn.Dense | nn.DenseGeneral):
    """The layer's product: its `dot_general_cls` or its `dot_general`, as flax picks it, else lax's at
    its stated precision (`accumulated`: the plain product, its fp32 sum taken in float64 on a CPU under
    Dew's tests)."""
    if layer.dot_general_cls is not None:
        return layer.dot_general_cls()
    return layer.dot_general if layer.dot_general is not None else _accumulated_dot_general


def _accumulated_dot_general(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None,
                             **kwargs) -> jax.Array:
    """`jax.lax.dot_general` as stated, its fp32 sum taken in float64 on a CPU under Dew's tests
    (`accumulated`)."""
    return accumulated(functools.partial(jax.lax.dot_general, dimension_numbers=dimension_numbers,
                                         precision=precision, **kwargs),
                       lhs, rhs, preferred_element_type=preferred_element_type)


def _biased(layer: nn.Dense | nn.DenseGeneral, inputs: Array, kernel: Array, dimensions, bias: Array | None,
            shaped):
    """The layer's product of `inputs` and `kernel` over `dimensions` as flax takes it, plus `bias`
    reshaped by `shaped` to broadcast over it (`biased`: added as flax adds it, its gradient summed
    at fp32, in float64 on a CPU under Dew's tests)."""
    product = _product(layer)(inputs, kernel, dimensions, precision=layer.precision)
    if bias is None:
        return product
    return biased(product, shaped(bias))


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
        return _biased(self, inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())), bias, lambda bias: bias)


def _members(init, batched: int, flat_shape):
    """`init` drawn for each member of a kernel's `batched` leading batch axes alone, at `flat_shape`
    of the rest, and stacked: each member's own fan-in, its inputs', where flax draws the whole kernel
    at `[batch * inputs, features]` and a variance-scaling initializer counts the batch members as
    inputs (DeepSeek V4's grouped `o_a_proj`, eight groups, would start sqrt(8) narrower than one
    group's lecun init). A boxed draw (a partitioned initializer's) is reshaped inside its box, as
    `_reshaped` reshapes it."""
    def drawn(rng, shape, dtype=jnp.float32):
        member = tuple(int(size) for size in flat_shape(shape[batched:]))
        keys = jax.random.split(rng, int(np.prod(shape[:batched])))
        stacked = jax.vmap(lambda key: init(key, member, dtype))(keys)
        return jax.tree.map(lambda value: jnp.reshape(value, shape), stacked)
    return drawn


class DenseGeneral(nn.DenseGeneral):
    """Flax's DenseGeneral, its bias added by `biased` (the module docstring), and a batched kernel
    drawn a batch member at a time (`_members`)."""

    @nn.compact
    def __call__(self, inputs: Array) -> jax.Array:
        features, batch_dims = _axes(self.features), _axes(self.batch_dims)
        if batch_dims and set(batch_dims) != set(range(max(batch_dims) + 1)):
            raise ValueError(f'batch_dims {batch_dims} must be consecutive leading dimensions '
                             'starting from 0.')
        axis, batch_dims = _axes(self.axis, inputs.ndim), _axes(batch_dims, inputs.ndim)
        batched = len(batch_dims)
        batch_shape = tuple(inputs.shape[dim] for dim in batch_dims)

        def flat(shape):
            """`shape` as `[inputs, features]`, the contracted axes and the features each flattened."""
            return np.prod(shape[:len(shape) - len(features)]), np.prod(shape[-len(features):])

        drawn = _members(self.kernel_init, batched, flat) if batched else _reshaped(self.kernel_init, flat)
        shape = batch_shape + tuple(inputs.shape[dim] for dim in axis) + features
        kernel = self.param('kernel', drawn, shape, self.param_dtype)
        bias = self.param(
            'bias', _reshaped(self.bias_init, lambda shape: (np.prod(shape[:batched])
                                                             * np.prod(shape[-len(features):]),)),
            batch_shape + features, self.param_dtype) if self.use_bias else None
        inputs, kernel, bias = _promoted(self, inputs, kernel, bias)
        contracted = tuple(range(batched, batched + len(axis)))
        # The product is [batch dims, kept dims, features]; the bias broadcasts over the kept dims.
        kept = tuple(inputs.shape[dim] if dim in batch_dims else 1
                     for dim in range(inputs.ndim) if dim not in axis)
        return _biased(self, inputs, kernel, ((axis, contracted), (batch_dims, tuple(range(batched)))), bias,
                       lambda bias: jnp.reshape(bias, kept + features))
