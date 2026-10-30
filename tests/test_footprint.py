"""The residual storage a pullback names before compiling, not a claim about XLA's scheduled peak."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.training.footprint import device_bytes, residual_bytes


def test_resident_device_bytes_do_not_include_host_state():
    """A state leaf placed on pinned host memory is not HBM, even though it has the same global
    shape and partition as a device leaf. Counting the two as resident would charge offload twice."""
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",))
    sharding = NamedSharding(mesh, P())
    device = jax.ShapeDtypeStruct((8, 16), jnp.float32, sharding=sharding)
    host = jax.ShapeDtypeStruct((8, 16), jnp.float32, sharding=sharding.with_memory_kind("pinned_host"))
    assert device_bytes({"params": device, "optimizer": host}) == 8 * 16 * 4


def test_a_pullback_counts_new_residuals_not_input_storage():
    """The exponential needed to differentiate exp is new storage, an expensive value XLA keeps. A dot
    product needs the other operand, already resident as an input, which must not count again as
    residual storage; nor does the cosine sine's derivative needs, which XLA recomputes from its input
    in the backward's fusion, a sine's cosine being cheap (XLA's `InstructionFusion::IsExpensive`)."""
    shape = jax.ShapeDtypeStruct((8, 16), jnp.float32)

    def exponential(params):
        return jnp.exp(params).sum(), ()

    def sine(params):
        return jnp.sin(params).sum(), ()

    def product(params, values):
        return jnp.vdot(params, values), ()

    assert residual_bytes(exponential, shape, ()) == 8 * 16 * 4
    assert residual_bytes(sine, shape, ()) == 0
    assert residual_bytes(product, shape, (shape,)) == 0


def test_a_pullback_counts_a_cheap_chain_as_the_product_it_is_recomputed_from():
    """Residuals XLA recomputes in the backward's fusions, each a cheap elementwise value (a select, a
    multiply) of a product, count as that product, once: a rectified, scaled product squared keeps the
    product's bytes, not the two cheap values JAX's pullback closes over."""
    weights, rows = jax.ShapeDtypeStruct((16, 32), jnp.float32), jax.ShapeDtypeStruct((8, 16), jnp.float32)

    def loss(weights, rows):
        return jnp.sum(jnp.square(jnp.maximum(rows @ weights, 0.0) * 2.0 + 1.0)), ()

    assert residual_bytes(loss, weights, (rows,)) == 8 * 32 * 4


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
