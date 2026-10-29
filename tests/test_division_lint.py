"""SLOP024: in the library, a JAX array is never divided by a literal that is no power of two, which XLA
computes as a product with the literal's rounded reciprocal, not correctly rounded (rows 1402, 1403);
`dew.nn.precision.divided` divides as IEEE does. The rule reports only an operand it proves is a JAX
array: a call of jax.numpy, jax.lax, jax.nn, jax.random or jax.scipy, a method or a subscript of one,
arithmetic with one, or a parameter annotated as one. A Python number, or a name it cannot prove, is no
finding."""

import pytest
from tool_support import lint_codes, load


@pytest.fixture(scope="module")
def lint():
    return load("lint_slop")


DIVISIONS = {
    "a jax.numpy call's result": ("import jax.numpy as jnp\ndef f(x):\n    return jnp.mean(x) / 3.0", 1),
    "jax.numpy imported from jax": ("from jax import numpy as jnp\ndef f(x):\n    return jnp.sum(x) / 10", 1),
    "a lax call, through jax": ("import jax\ndef f(x):\n    return jax.lax.rsqrt(x) / 255.0", 1),
    "a method of one": ("import jax.numpy as jnp\ndef f(x):\n"
                        "    return jnp.asarray(x).astype(jnp.float32) / 6", 1),
    "arithmetic with one": ("import jax.numpy as jnp\ndef f(x, y):\n    return (y - jnp.exp(x)) / 1000.0", 1),
    "a subscript of one": ("import jax.numpy as jnp\ndef f(x):\n    return jnp.cumsum(x)[1:] / 3", 1),
    "a parameter annotated as an array": ("import jax\ndef f(x: jax.Array):\n    return x / 7.0", 1),
    "an annotated parameter divided in place": ("from jax import Array\ndef f(x: Array):\n    x /= 3\n"
                                                "    return x", 1),
    "a negative literal": ("import jax.numpy as jnp\ndef f(x):\n    return jnp.abs(x) / -3.0", 1),
    "a power of two": ("import jax.numpy as jnp\ndef f(x):\n    return jnp.mean(x) / 4.0 + jnp.sum(x) / 0.5",
                       0),
    "a Python number": ("def f(seconds: float):\n    return seconds / 60 + float(seconds) / 3 + 1 / 3", 0),
    "a name the rule cannot prove": ("def f(x):\n    return x / 3.0", 0),
    "an array's dtype facts": ("import jax.numpy as jnp\ndef f(x):\n"
                               "    return jnp.finfo(x.dtype).eps / 3", 0),
    "a non-literal divisor": ("import jax.numpy as jnp\ndef f(x, n):\n    return jnp.sum(x) / n", 0),
    "the correctly rounded division": (
        "import jax.numpy as jnp\nfrom dew.nn.precision import divided\n"
        "def f(x):\n    return divided(jnp.mean(x), 3.0)", 0),
}


@pytest.mark.parametrize("case", list(DIVISIONS))
def test_a_jax_array_divided_by_a_literal_no_power_of_two_is_reported_in_the_library(lint, case, tmp_path):
    body, count = DIVISIONS[case]
    assert lint_codes(lint, tmp_path, ("src", "dew", "case.py"), body).count("SLOP024") == count


def test_the_rule_reads_the_library_alone(lint, tmp_path):
    body = "import jax.numpy as jnp\ndef f(x):\n    return jnp.mean(x) / 3.0"
    assert "SLOP024" not in lint_codes(lint, tmp_path, ("tests", "test_case.py"), body)
    assert "SLOP024" not in lint_codes(lint, tmp_path, ("tools", "case.py"), body)
