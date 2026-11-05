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

from dew.nn import derivatives
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


def _scaled_backward(scale: float):
    """A function `sin` whose declared backward scales its cotangent by `scale`, so a test sees the rule ran,
    its static `scale` passed beside the arrays."""
    def primal(x, factor):
        return jnp.sin(x) * factor

    def forward(x, factor):
        return primal(x, factor), (x, factor)

    def backward(saved, cotangent):
        x, factor = saved
        return scale * cotangent * jnp.cos(x) * factor, jnp.zeros_like(factor)

    return primal, forward, backward


@pytest.mark.parametrize("declare, kind", [
    (lambda *rule: derivatives.kernel_derivative(*rule), KernelGlue()),
    (lambda *rule: derivatives.numerics_derivative(*rule), Protected())],
    ids=["kernel_derivative", "numerics_derivative"])
def test_a_declared_derivative_is_one_marked_call_and_differentiates_as_its_rule(declare, kind):
    """A custom derivative declared through its class's API (`kernel_derivative`, `numerics_derivative`: no
    bare `jax.custom_vjp`) is one marked call of its kind, its primal, forward and backward together, which
    every transform keeps: its value is its primal's, reverse mode runs its declared backward (a cotangent
    scaled by 3) in calls of its kind, a second order differentiates that backward, and forward mode refuses
    it exactly as a bare `jax.custom_vjp` does."""
    primal, forward, backward = _scaled_backward(3.0)
    declared = declare(primal, forward, backward)
    bare = jax.custom_vjp(primal)
    bare.defvjp(forward, backward)
    x, factor = jnp.linspace(0.1, 1.0, 8), jnp.full(8, 2.0)
    np.testing.assert_array_equal(np.asarray(declared(x, factor)), np.asarray(primal(x, factor)))
    assert kinds_of(jax.make_jaxpr(declared)(x, factor)) == [kind]

    def loss(function):
        return lambda x: jnp.sum(function(x, factor))

    np.testing.assert_array_equal(np.asarray(jax.grad(loss(declared))(x)),
                                  np.asarray(jax.grad(loss(bare))(x)))
    assert not np.array_equal(np.asarray(jax.grad(loss(bare))(x)), np.asarray(jax.grad(loss(primal))(x)))
    # Reverse mode splits the call into its forward and its transpose, each a call of its kind.
    found = kinds_of(jax.make_jaxpr(jax.grad(loss(declared)))(x))
    assert found and all(seen == kind for seen in found)
    np.testing.assert_array_equal(np.asarray(jax.grad(lambda x: jnp.sum(jax.grad(loss(declared))(x)))(x)),
                                  np.asarray(jax.grad(lambda x: jnp.sum(jax.grad(loss(bare))(x)))(x)))
    with pytest.raises(TypeError) as marked_error:
        jax.jvp(lambda x: declared(x, factor), (x,), (x,))
    with pytest.raises(TypeError) as bare_error:
        jax.jvp(lambda x: bare(x, factor), (x,), (x,))
    assert str(marked_error.value) == str(bare_error.value)
    np.testing.assert_array_equal(np.asarray(jax.vmap(declared)(x[:, None], factor[:, None])),
                                  np.asarray(jax.vmap(bare)(x[:, None], factor[:, None])))


@pytest.mark.parametrize("declare, kind", [(lambda *rule: derivatives.numerics_tangent(*rule), Protected()),
                                           (lambda *rule: derivatives.kernel_tangent(*rule), KernelGlue())],
                         ids=["numerics_tangent", "kernel_tangent"])
def test_a_declared_tangent_is_one_marked_call_and_differentiates_as_its_rule(declare, kind):
    """A tangent rule declared through its class's function (`numerics_tangent`, `kernel_tangent`: no bare
    `jax.custom_jvp`) is one marked call of its kind, which forward mode and reverse mode (its tangent
    transposed) keep, each from the declared tangent (doubled here, so a test sees it ran)."""
    def primal(x):
        return jnp.sin(x)

    def tangent(primals, tangents):
        [x], [dx] = primals, tangents
        return jnp.sin(x), 2.0 * jnp.cos(x) * dx

    declared = declare(primal, tangent)
    bare = jax.custom_jvp(primal)
    bare.defjvp(tangent)
    x = jnp.linspace(0.1, 1.0, 8)
    assert kinds_of(jax.make_jaxpr(declared)(x)) == [kind]
    for function in (lambda f: jax.jvp(f, (x,), (jnp.ones_like(x),))[1],
                     lambda f: jax.grad(lambda x: jnp.sum(f(x)))(x)):
        np.testing.assert_array_equal(np.asarray(function(declared)), np.asarray(function(bare)))
    assert not np.array_equal(np.asarray(jax.grad(lambda x: jnp.sum(bare(x)))(x)), np.asarray(jnp.cos(x)))
    found = kinds_of(jax.make_jaxpr(jax.grad(lambda x: jnp.sum(declared(x))))(x))
    assert found and all(seen == kind for seen in found)


def test_a_surrogate_is_one_surrogate_call():
    """A surrogate (`surrogate`: a value and its declared derivative rule) is one marked call of its kind,
    which reverse mode keeps, its derivative its rule's."""
    rounded = derivatives.surrogate(lambda x: jnp.round(x), lambda primals, tangents, value: tangents[0])
    x = jnp.linspace(0.1, 4.0, 8)
    assert kinds_of(jax.make_jaxpr(rounded)(x)) == [Surrogate()]
    np.testing.assert_array_equal(np.asarray(jax.grad(lambda x: jnp.sum(rounded(x)))(x)), np.ones(8))


def test_a_declared_derivatives_nondifferentiable_array_is_bound_not_traced():
    """An array at a declared derivative's nondifferentiable position (`nondiff_argnums`, a table the rule
    reads as a constant) reaches the rule as the bare `jax.custom_vjp` takes it, concrete, eagerly and under a
    jit, not as a traced operand of the marked call, which the custom derivative would refuse."""
    def primal(table, x):
        return jnp.sin(x) * table[0]

    def forward(table, x):
        return primal(table, x), x

    def backward(table, x, cotangent):
        return (3.0 * cotangent * jnp.cos(x) * table[0],)

    declared = derivatives.kernel_derivative(primal, forward, backward, nondiff_argnums=(0,))
    bare = jax.custom_vjp(primal, nondiff_argnums=(0,))
    bare.defvjp(forward, backward)
    table, x = jnp.asarray([2.0]), jnp.linspace(0.1, 1.0, 8)

    def gradient(function, run):
        return run(jax.grad(lambda x: jnp.sum(function(table, x))))(x)

    for run in (lambda f: f, jax.jit):
        np.testing.assert_array_equal(np.asarray(gradient(declared, run)), np.asarray(gradient(bare, run)))
