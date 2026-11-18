"""One statement of a user's code and the values it reads (`dew._stated.stated`), which every identity built
on it reads alike (a data step's exact resume, a program's digest, a rung's key, the accuracy gate's
arms): a function `jax.jit` wraps, an array of JAX's and a function made inside another are stated by what
they are."""

import jax
import jax.numpy as jnp
import numpy as np

from dew._stated import stated


def times(x, factor):
    return x * factor


def plus(x, factor):
    return x + factor


def test_a_jitted_function_is_what_it_wraps_and_how_it_is_wrapped():
    """Two wrappings of one function under the same options state alike, a static argument named or
    placed; another function, no static argument, or a donated one, states otherwise. A jitted function of the user's was refused, so a program or an arm
    that called one had no identity (train 1015's review)."""
    one = stated(jax.jit(times, static_argnums=(1,)), "jit")
    # A static argument named or placed is one option: JAX reads each from the other by the signature.
    assert one == stated(jax.jit(times, static_argnums=(1,)), "jit") == stated(
        jax.jit(times, static_argnames=("factor",)), "jit")
    assert one != stated(jax.jit(plus, static_argnums=(1,)), "jit")
    assert one != stated(jax.jit(times), "jit")
    assert one != stated(jax.jit(times, static_argnums=(1,), donate_argnums=(0,)), "jit")


def test_an_array_of_jaxs_is_its_values_and_sharding():
    """An array of JAX's states its dtype, shape and bytes, fetched to the host: equal values alike, one
    changed element or another dtype otherwise. It was refused as a constant of the program."""
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    changed = values.copy()
    changed[1, 2] = -1
    one = stated(jnp.asarray(values), "array")
    assert one == stated(jnp.asarray(values.copy()), "array")
    assert one != stated(jnp.asarray(changed), "array")
    assert one != stated(jnp.asarray(values, jnp.bfloat16), "array")


def made(factor):
    def inner(x):
        return x * factor
    return inner


def made_otherwise(factor):
    def inner(x):
        return x + factor
    return inner


def test_a_function_made_inside_another_is_its_code_and_what_it_closes_over():
    """Two functions one maker made over the same value state alike; over another value, or made by a maker
    whose code differs, otherwise. A function made inside another was refused, however plainly its code and
    closure stated it."""
    one = stated(made(2), "local")
    assert one == stated(made(2), "local")
    assert one != stated(made(3), "local")
    assert one != stated(made_otherwise(2), "local")
