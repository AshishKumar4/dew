"""Marked calls (docs/design/algebra.md §6.10): a span of a computation carried as one call of its kind
(`dew.nn.derivatives.marked`), which every transform keeps a call of that kind, so the algebra pass reads a
span's kind from the program it rewrites, its forward, tangent and cotangent alike."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify
from jax.sharding import Mesh, PartitionSpec
from reference_error import equations

from dew.nn.derivatives import Algebra, KernelGlue, Protected, Surrogate, marked, marked_call_p

KINDS = [Protected(), Surrogate(), KernelGlue(), Algebra()]


def kinds_of(closed) -> list:
    """The kind of every marked call a program holds, its bodies' included."""
    return [equation.params["kind"] for equation in equations(closed) if equation.primitive is marked_call_p]


def body(x, y):
    return jnp.sin(x) * y + jnp.sum(x * y)


@pytest.mark.parametrize("kind", KINDS, ids=lambda kind: type(kind).__name__)
def test_a_marked_span_computes_as_its_body_and_stays_one_call_of_its_kind_through_every_transform(kind):
    """A marked span computes what its body does, and its program holds it as one call of its kind; so do
    its derivatives forward and backward (`jax.jvp`, `jax.linearize`, `jax.grad`), its batched form
    (`jax.vmap`), its checkpointed one (`jax.checkpoint`), its sharded one (`jax.shard_map`) and its
    checked one (`checkify.checkify`, as a sampler's decode step checks its program), which runs as its body
    does."""
    spanned = marked(kind, body)
    x, y = jnp.linspace(0.1, 1.0, 8), jnp.linspace(1.0, 2.0, 8)
    np.testing.assert_array_equal(np.asarray(jax.jit(spanned)(x, y)), np.asarray(jax.jit(body)(x, y)))
    error, checked = jax.jit(checkify.checkify(spanned))(x, y)
    error.throw()
    np.testing.assert_array_equal(np.asarray(checked), np.asarray(jax.jit(body)(x, y)))

    def loss(x, y):
        return jnp.sum(spanned(x, y))

    transformed = {
        "call": lambda x, y: spanned(x, y),
        "jvp": lambda x, y: jax.jvp(spanned, (x, y), (y, x))[1],
        "linearize": lambda x, y: jax.linearize(spanned, x, y)[1](y, x),
        "grad": jax.grad(loss, argnums=(0, 1)),
        "vmap": jax.vmap(spanned),
        "checkpoint": jax.grad(jax.checkpoint(loss)),
        "shard_map": jax.shard_map(spanned, mesh=Mesh(np.array(jax.devices()[:1]), ("rows",)),
                                   in_specs=PartitionSpec("rows"), out_specs=PartitionSpec("rows")),
        "checkify": checkify.checkify(spanned),
    }
    for name, function in transformed.items():
        found = kinds_of(jax.make_jaxpr(function)(x, y) if name != "vmap"
                         else jax.make_jaxpr(function)(x[None], y[None]))
        assert found and all(seen == kind for seen in found), name


def test_a_marked_spans_gradient_is_its_bodys():
    """A marked span differentiates as its body does: its gradient on par with the body's, exactly, the
    marker computing nothing of its own."""
    x, y = jnp.linspace(0.1, 1.0, 8), jnp.linspace(1.0, 2.0, 8)
    spanned = jax.grad(lambda x, y: jnp.sum(marked(Protected(), body)(x, y)), argnums=(0, 1))
    plain = jax.grad(lambda x, y: jnp.sum(body(x, y)), argnums=(0, 1))
    for got, expected in zip(jax.jit(spanned)(x, y), jax.jit(plain)(x, y), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
