"""The residual storage a pullback names before compiling, not a claim about XLA's scheduled peak."""

import jax
import jax.numpy as jnp

from dew.training.footprint import residual_bytes


def test_a_pullback_counts_new_residuals_not_input_storage():
    """The cosine needed to differentiate sine is new storage. Squaring needs its input, which is
    already resident and must not count again as residual storage."""
    shape = jax.ShapeDtypeStruct((8, 16), jnp.float32)

    def sine(params):
        return jnp.sin(params).sum(), ()

    def square(params):
        return jnp.square(params).sum(), ()

    assert residual_bytes(sine, shape, ()) == 8 * 16 * 4
    assert residual_bytes(square, shape, ()) == 0


def test_a_custom_pullback_counts_one_buffer_shared_by_two_residual_slots():
    """A custom VJP retains the same intermediate twice and its input once. Only the one new
    intermediate's storage counts; neither repeating its reference nor retaining an input doubles it."""
    @jax.custom_vjp
    def summed(params):
        return jnp.sin(params).sum()

    def forward(params):
        cosine = jnp.cos(params)
        return summed(params), (cosine, cosine, params)

    def backward(saved, cotangent):
        first, second, inputs = saved
        return ((first + second) * cotangent / 2 + inputs * 0,)

    summed.defvjp(forward, backward)

    def loss(params):
        return summed(params), ()

    shape = jax.ShapeDtypeStruct((8, 16), jnp.float32)
    assert residual_bytes(loss, shape, ()) == 8 * 16 * 4
