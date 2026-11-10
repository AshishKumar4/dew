"""SLOP024: in the library, a JAX array is never divided by a compile-time constant no power of two
divides exactly by, which XLA computes as a product with the constant's rounded reciprocal, not correctly
rounded (rows 1402, 1403): a literal, a named constant, one computed from others or a format's, a module's
attribute or a dtype's scalar; by `/` or jax.numpy's division. `dew.nn.precision.divided` divides as IEEE
does. The rule reports only an operand it proves is a JAX array: a call of jax.numpy, jax.lax, jax.nn,
jax.random or jax.scipy, a method or a subscript of one, arithmetic with one, a parameter annotated as one,
or a name its function assigns one. A Python number, or a name it cannot prove, is no finding."""

import pytest
from tool_support import lint_codes

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
    "an import in one function alone": ("def f(x):\n    import jax.numpy as jnp\n"
                                        "    return jnp.mean(x) / 3.0\n"
                                        "def g(jnp, x):\n    return jnp.mean(x) / 3.0", 1),
    "a power of two": ("import jax.numpy as jnp\ndef f(x):\n    return jnp.mean(x) / 4.0 + jnp.sum(x) / 0.5",
                       0),
    "a Python number": ("def f(seconds: float):\n    return seconds / 60 + float(seconds) / 3 + 1 / 3", 0),
    "a name the rule cannot prove": ("def f(x):\n    return x / 3.0", 0),
    "an array's dtype facts": ("import jax.numpy as jnp\ndef f(x):\n"
                               "    return jnp.finfo(x.dtype).eps / 3", 0),
    "a non-literal divisor": ("import jax.numpy as jnp\ndef f(x, n):\n    return jnp.sum(x) / n", 0),
    "a named constant": ("import jax.numpy as jnp\nSCALE = 6.0\ndef f(x):\n    return jnp.abs(x) / SCALE", 1),
    "a constant computed from others": ("import jax.numpy as jnp\nBASE = 3\nSCALE = float(BASE * 2)\n"
                                        "def f(x):\n    return jnp.abs(x) / SCALE", 1),
    "a constant of a format": ("import jax.numpy as jnp\nTOP = float(jnp.finfo(jnp.float4_e2m1fn).max)\n"
                               "def f(x):\n    return jnp.abs(x) / TOP", 1),
    "a module's attribute": ("import math\nimport jax.numpy as jnp\ndef f(x):\n"
                             "    return jnp.abs(x) / math.log(2) + jnp.abs(x) / math.pi", 2),
    "a scalar of a dtype": ("import jax.numpy as jnp\ndef f(x):\n    return jnp.abs(x) / jnp.float32(3)", 1),
    "a name assigned an array": ("import jax.numpy as jnp\nTOP = 6.0\ndef f(x):\n"
                                 "    floored = jnp.maximum(x, 1.0)\n    return floored / TOP", 1),
    "jax.numpy's division": ("import jax.numpy as jnp\ndef f(x):\n    return jnp.divide(jnp.abs(x), 3.0)", 1),
    "a power of two named": ("import jax.numpy as jnp\nHALF = 0.5\ndef f(x):\n    return jnp.abs(x) / HALF",
                             0),
    "a local name shadowing a constant": ("import jax.numpy as jnp\nSCALE = 6.0\ndef f(x, SCALE):\n"
                                          "    return jnp.abs(x) / SCALE", 0),
    "a name rebound past its constant": ("import jax.numpy as jnp\nSCALE = 3\nSCALE += 1\ndef f(x):\n"
                                         "    return jnp.abs(x) / SCALE", 0),
    "an enclosing function's parameter": ("import jax.numpy as jnp\nSCALE = 6\ndef outer(SCALE):\n"
                                          "    return lambda x: jnp.abs(x) / SCALE", 0),
    "a comprehension's name": ("import jax.numpy as jnp\nSCALE = 6\ndef f(x):\n"
                               "    return [jnp.abs(x) / SCALE for SCALE in (2, 4)]", 0),
    "an array name later a scalar": ("import jax.numpy as jnp\ndef f(x):\n    value = jnp.sum(x)\n"
                                     "    value = value.item()\n    return value / 3", 0),
    "a call no constant": ("import numpy as np\nimport jax.numpy as jnp\ndef f(x):\n"
                           "    return jnp.abs(x) / np.random.uniform(2, 3)", 0),
    "typed scalars' arithmetic": ("import numpy as np\nimport jax.numpy as jnp\ndef f(x):\n"
                                  "    return jnp.abs(x) / (np.float32(0.1) * np.float32(10))", 0),
    "a module's own float": ("import jax.numpy as jnp\ndef float(value):\n    return 4\ndef f(x):\n"
                             "    return jnp.abs(x) / float(3)", 0),
    "a class's name in its method": ("import jax.numpy as jnp\nSCALE = 6\nclass C:\n    SCALE = 4\n"
                                     "    def f(self, x):\n        return jnp.abs(x) / SCALE", 1),
    "a parameter's default reads the module": ("import jax.numpy as jnp\nSCALE = 6\n"
                                               "def f(x, SCALE=jnp.abs(1.0) / SCALE):\n    return x", 1),
    "a comprehension's numerator": ("import jax.numpy as jnp\ndef f(x):\n    value = jnp.abs(x)\n"
                                    "    return [value / 3 for value in (1, 2)]", 0),
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


@pytest.mark.parametrize("read", ["from dew.formats import TOP\n", "import dew.formats as formats\n"])
def test_a_constant_another_library_module_defines_is_read_there(lint, tmp_path, read):
    """A divisor another module of the library defines (a format's limit), imported by name or read as an
    attribute of its module, is that module's constant, read in its own source: an array divided by it is
    reported."""
    formats = tmp_path / "src" / "dew" / "formats.py"
    formats.parent.mkdir(parents=True)
    formats.write_text("import jax.numpy as jnp\nTOP = float(jnp.finfo(jnp.float4_e2m1fn).max)\n")
    divisor = "TOP" if read.startswith("from") else "formats.TOP"
    body = f"import jax.numpy as jnp\n{read}def f(x):\n    return jnp.abs(x) / {divisor}"
    assert lint_codes(lint, tmp_path, ("src", "dew", "case.py"), body).count("SLOP024") == 1


@pytest.mark.parametrize("first", ["a.py", "b.py"])
def test_a_constant_through_an_import_cycle_is_read_in_either_order(lint, tmp_path, first):
    """Two library modules importing each other's constants, read in either order, give each its
    constants: a division by one computed across the cycle is reported."""
    package = tmp_path / "src" / "dew"
    package.mkdir(parents=True)
    a = ("import jax.numpy as jnp\nBASE = 3\nfrom dew.b import SCALE\nSIX = SCALE\n"
         "def f(x):\n    return jnp.abs(x) / SIX")
    b = "from dew.a import BASE\nSCALE = BASE * 2\n"
    (package / "a.py").write_text(a)
    (package / "b.py").write_text(b)
    lint_codes(lint, tmp_path, ("src", "dew", first), (package / first).read_text())
    assert lint_codes(lint, tmp_path, ("src", "dew", "a.py"), a).count("SLOP024") == 1
