"""FP32 split-product and compensated-reduction candidates for isolated dots."""
import os

import jax
import jax.numpy as jnp
from jax.custom_derivatives import linear_call

from dew.nn.precision import compensated_add, two_product


def raw(left, right):
    return jnp.matmul(left, right, precision="highest")


def split_value(left, right):
    if left.dtype != jnp.float32:
        return raw(left, right)
    # A 12-significant-bit head leaves a residual about 2^-12 of each operand.
    lh, rh = (jax.lax.optimization_barrier(jax.lax.reduce_precision(x, 8, 11))
              for x in (left, right))
    ll, rl = left - lh, right - rh
    head = raw(lh, rh)
    tail = raw(lh, rl) + raw(ll, rh)
    if os.environ.get("DEW_PROBE_FOUR_PRODUCTS"):
        tail = tail + raw(ll, rl)
    hi, lo = compensated_add((head, jnp.zeros_like(head)), (tail, jnp.zeros_like(tail)))
    return hi + lo


def reduced_value(left, right):
    if left.dtype != jnp.float32:
        return raw(left, right)
    lhs, rhs = left[..., :, :, None], right[..., None, :, :]
    if os.environ.get("DEW_PROBE_NO_PRODUCT_RESIDUAL"):
        product = lhs * rhs
        products = product, jnp.zeros_like(product)
    else:
        products = two_product(lhs, rhs, rounding=jax.lax.optimization_barrier)
    zero = jnp.zeros((), left.dtype)
    hi, lo = jax.lax.reduce(products, (zero, zero), compensated_add, (left.ndim - 1,))
    return hi + lo


@jax.custom_jvp
def dot(left, right):
    return (reduced_value if os.environ.get("DEW_PROBE_REDUCE") else split_value)(left, right)


@dot.defjvp
def tangent(primals, tangents):
    left, right = primals
    dleft, dright = tangents
    lhs = linear_call(lambda right, dl: dot(dl, right),
                      lambda right, ct: dot(ct, jnp.swapaxes(right, -1, -2)), right, dleft)
    rhs = linear_call(lambda left, dr: dot(left, dr),
                      lambda left, ct: dot(jnp.swapaxes(left, -1, -2), ct), left, dright)
    return dot(left, right), lhs + rhs
