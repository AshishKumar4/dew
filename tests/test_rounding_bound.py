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


def test_a_program_traced_under_one_context_is_not_served_under_another():
    """A rounding scale's program, held once traced (`rounding_bound.TRACED`), is held by the context its
    trace reads (`options.trace_context`): a function reading a context variable set anew is traced again
    under it, never served the program traced before it was set (row 10878: a hand-picked key missed Dew's
    contexts, its kernel choices among them)."""
    import contextvars

    factor = contextvars.ContextVar("factor", default=1.0)
    x = jnp.asarray([1.0, 2.0, 3.0], jnp.float32)

    def scaled(x):
        return x * np.float32(factor.get())

    before, _ = rounding_scale(scaled, x)
    token = factor.set(2.0)
    try:
        after, _ = rounding_scale(scaled, x)
    finally:
        factor.reset(token)
    np.testing.assert_array_equal(after, 2.0 * before)


def _held_whole_in_spans(monkeypatch, program) -> None:
    """`program`'s values and rounding scale at three values walked in spans of 8 steps
    (`float64_terms.SPAN_STEPS`) as compiled whole: each rounding's sign keyed by its place in the program,
    not by the span it falls in."""
    import float64_terms
    import rounding_bound

    x = jnp.asarray([1.0, -2.0, 3.5], jnp.float32)
    whole = rounding_scale(program, x)
    rounding_bound.released_scales()
    monkeypatch.setattr(float64_terms, "SPAN_STEPS", 8)
    monkeypatch.setattr(rounding_bound, "SPAN_STEPS", 8)
    spanned = rounding_scale(program, x)
    rounding_bound.released_scales()
    np.testing.assert_array_equal(spanned[0], whole[0])
    np.testing.assert_array_equal(spanned[1], whole[1])


def test_a_program_of_more_steps_than_a_span_is_scaled_as_compiled_whole(monkeypatch):
    """A static program of more steps than a compiled span holds (`float64_terms.SPAN_STEPS`) is walked a
    span at a time (row 9780: a Gemma 3n site compiled whole peaked at 9.1 GiB), its values and its rounding
    scale each element's as the program compiled whole gives them: each rounding's sign keyed by its place
    in the program, not by the span it falls in. Here spans of 8 steps over a program of 40."""
    def chained(x):
        for step in range(20):
            x = x * np.float32(1.0 + 1e-3 * (step % 7)) + np.float32(0.25)
        return x

    _held_whole_in_spans(monkeypatch, chained)


def test_a_call_of_more_steps_than_a_span_is_walked_as_its_bodys_spans(monkeypatch):
    """A call whose body holds more steps than a compiled span (`float64_terms.SPAN_STEPS`; a jit-wrapped
    reference, the gate's common arm, is one such step) is walked as its body's spans, no compiled span
    holding more steps than a span may, its neighbours' and its body's counted (row 10950: counted as one
    step, it was compiled whole with its neighbours; row 11004: the values alone cannot show it), and its
    values and rounding scale those the program compiled whole gives."""
    import float64_terms

    @jax.jit
    def body(x):
        for _ in range(10):
            x = x * np.float32(1.5) + np.float32(0.25)
        return x

    def program(x):
        return body(x + np.float32(1.0)) * np.float32(2.0)

    _held_whole_in_spans(monkeypatch, program)
    rounding_scale(program, jnp.asarray([1.0, -2.0, 3.5], jnp.float32))
    compiled = [sum(float64_terms._steps(equation) for equation in jaxpr.eqns[start:stop])
                for jaxpr, start, stop, *_ in float64_terms.RUNS]
    assert compiled and max(compiled) <= float64_terms.SPAN_STEPS
