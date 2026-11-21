"""`rounding_bound.rounding_scale`: a program's rounding scale, its walk at the formats a program states."""

import jax
import jax.numpy as jnp
import numpy as np
from rounding_bound import rounding_scale


def test_a_bit_cast_and_a_step_run_at_the_formats_they_were_traced_at():
    """A primitive whose meaning is a float format's own (`arm_arithmetic.FORMATTED`: a bit cast, a step to a
    neighbour) runs in the walk at the dtypes it was traced at, as the float64 replay runs it, its float
    results widened after: a float32 read as uint32 is one word an element, so a selection on those words
    keeps its shape (bound on float64 it was two words an element, and the selection refused the two shapes:
    a Gemma 3n weighed site's walk, row 9780's debugger); each bit cast round trip is its value, rounding
    nothing; and float32's step to its neighbour is one float32 ulp, not float64's."""
    x = jnp.asarray([0.0, 1.5, -2.25, 3.0], jnp.float32)
    value, _ = rounding_scale(
        lambda x: jnp.where(jax.lax.bitcast_convert_type(x, jnp.uint32) == 0, x, x * 2.0), x)
    np.testing.assert_array_equal(value, np.asarray(x, np.float64) * 2.0)
    for float_type, word in ((jnp.float32, jnp.uint32), (jnp.bfloat16, jnp.uint16),
                             (jnp.float16, jnp.uint16)):
        sample = x.astype(float_type)
        back, scale = rounding_scale(lambda x, f=float_type, u=word: jax.lax.bitcast_convert_type(
            jax.lax.bitcast_convert_type(x, u), f), sample)
        np.testing.assert_array_equal(back, np.asarray(sample, np.float64))
        np.testing.assert_array_equal(scale, np.zeros_like(scale))
    # Each a normal's neighbour (zero's is a subnormal, which XLA's CPU flushes).
    normal = x[1:]
    stepped, _ = rounding_scale(lambda x: jax.lax.nextafter(x, jnp.full_like(x, jnp.inf)), normal)
    np.testing.assert_array_equal(
        stepped, np.nextafter(np.asarray(normal), np.float32(np.inf)).astype(np.float64))


def test_a_platforms_branch_walks_the_branch_its_platform_takes_alone():
    """A `platform_dependent` branch, its selector the platform the walk evaluates on and so known, walks the
    branch that platform takes alone: every other branch is dead there, one holding a primitive no rule of the
    walk evaluates (a Cholesky factor) among them, where the walk took the selector's operands as traced and
    walked every branch (row 9780's debugger)."""
    dead = "tpu" if jax.default_backend() != "tpu" else "cpu"
    x = jnp.asarray([[4.0, 1.0], [1.0, 4.0]], jnp.float32)
    value, _ = rounding_scale(lambda a: jax.lax.platform_dependent(
        a, **{dead: jax.lax.linalg.cholesky}, default=lambda z: z + 1.0), x)
    np.testing.assert_array_equal(value, np.asarray(x, np.float64) + 1.0)
