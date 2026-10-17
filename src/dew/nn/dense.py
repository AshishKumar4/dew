"""Flax's dense layers, with their bias gradients accumulated at fp32.

`Dense` and `DenseGeneral` are flax's, parameters, names, initializers and
the product's interception points alike, so checkpoints, LoRA and Qwix see
the layers they did. Only the bias reaches the sum at fp32 (`fp32_bias`,
through flax's `promote_dtype` field), which sums its gradient at fp32, and
the sum rounds back to the product's dtype. A layer given its own
`dot_general` keeps that product's result, which Dew's fp32-result products
(`fp32_result_dot_general`, `head_dot_general`) already widen, the bias
summed at fp32 with them.

flax 0.12's Dense and DenseGeneral carry no `preferred_element_type`, which
would widen the product instead; its Einsum and ConvTranspose do.
"""

import jax
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import PromoteDtypeFn
from flax.typing import Array

from .precision import fp32_bias, product_dtype


def _rounded_back(layer: nn.Dense | nn.DenseGeneral, inputs: Array, summed: jax.Array) -> jax.Array:
    """`summed` rounded to the dtype `layer`'s own product gives it, unless it multiplies by a
    product of its own, whose result stands."""
    if layer.dot_general is not None or layer.dot_general_cls is not None:
        return summed
    return summed.astype(product_dtype(inputs, layer.param_dtype, layer.dtype))


class Dense(nn.Dense):
    """Flax's Dense, its bias gradient accumulated at fp32 (the module docstring)."""

    promote_dtype: PromoteDtypeFn = fp32_bias(promote_dtype)

    # flax's own call manages the layer's state and runs its method interceptors once; this one
    # only rounds its result, outside them (`nn.nowrap`, for an overridden core method).
    @nn.nowrap
    def __call__(self, inputs: Array) -> jax.Array:
        return _rounded_back(self, inputs, super().__call__(inputs))


class DenseGeneral(nn.DenseGeneral):
    """Flax's DenseGeneral, its bias gradient accumulated at fp32 (the module docstring)."""

    promote_dtype: PromoteDtypeFn = fp32_bias(promote_dtype)

    @nn.nowrap
    def __call__(self, inputs: Array) -> jax.Array:
        return _rounded_back(self, inputs, super().__call__(inputs))
