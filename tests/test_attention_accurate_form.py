"""The accurate form's outcomes on the default route (row 2556): a call that names the accurate form
(`generation.chosen(attention_form='accurate')`), whichever kernel the route takes, is on par with the
reference path's IEEE float32 products, and on a GPU the fast form is not, over the inputs that tell them
apart. Apart from the route's selection and refusals (`test_attention_route`), which another lane edits."""

import functools

import jax
import jax.numpy as jnp
import pytest
from reference_error import assert_as_exact, with_gradients
from test_attention_route import drawing
from test_attention_template import KEYS, QUERIES, SEEDS

from dew.nn.attention import scaled_dot_product_attention


def routed_call(form: str):
    """A causal call on the default route ('auto') under `form` (`generation.chosen`), at JAX's own default
    matmul precision, as a run's default numerics multiply (the suite's environment states 'highest'), as
    a function of `(query, key, value, sinks, dout)`: its output, then its gradients (`with_gradients`)."""
    from dew.nn.kernels.generation import chosen

    def call(query, key, value, sinks):
        with chosen(attention_form=form), jax.default_matmul_precision("default"):
            return scaled_dot_product_attention(query, key, value, causal=True, sinks=sinks)

    return with_gradients(call)


ieee_reference = with_gradients(functools.partial(scaled_dot_product_attention, causal=True,
                                                  implementation="reference",
                                                  precision=jax.lax.Precision.HIGHEST))
"""The causal call on the reference path at IEEE float32 products (`Precision.HIGHEST`), the arithmetic
an accurate run's form states: its output and its gradients (`with_gradients`)."""


def accurate_draws():
    """The inputs that tell an accurate call from a fast one, each a check-1 draw: float32 rows; bfloat16
    rows, whose fast kernels keep less than IEEE fp32 products (on an H100 the fast form's error is 8
    times the reference's); and float32 products past the largest float before the scale, which a fast
    path multiplying first overflows (`test_attention_stability`)."""
    from test_attention_stability import _product_past_largest

    return {"float32": drawing(QUERIES, KEYS, jnp.float32, sinks=False),
            "bfloat16": drawing(QUERIES, KEYS, jnp.bfloat16, sinks=False),
            "products past the largest float": _product_past_largest}


@pytest.mark.kernel
@pytest.mark.parametrize("case", ["float32", "bfloat16", "products past the largest float"])
def test_an_accurate_call_on_the_default_route_is_on_par_with_ieee_fp32(case):
    """Under `generation.chosen(attention_form='accurate')`, a call on the default route ('auto'), whichever
    kernel it takes, is on par with the reference path's IEEE float32 products forward and backward,
    measured from float64 (check 1; row 2350: an accurate run's calls ran fast on FlashAttention, cuDNN,
    xla and splash), over inputs a fast kernel fails on (`accurate_draws`); the kernel lab's GPU tier runs
    it on each target GPU."""
    assert_as_exact(jax.jit(routed_call("accurate")), jax.jit(ieee_reference), accurate_draws()[case],
                    f"the accurate form on the default route, {case}", seeds=SEEDS)


@pytest.mark.gpu(reason="a GPU's default route takes fast kernels for the fast form, which a CPU never does")
@pytest.mark.parametrize("case", ["bfloat16", "products past the largest float"])
def test_the_fast_form_on_a_gpus_default_route_is_not_on_par_with_ieee_fp32(case):
    """On a GPU the fast form's default route fails check 1 against IEEE fp32 products where the accurate
    form passes it (`accurate_draws`): the accurate test's inputs tell the two routes apart (row 2556)."""
    with pytest.raises(AssertionError):
        assert_as_exact(jax.jit(routed_call("fast")), jax.jit(ieee_reference), accurate_draws()[case],
                        f"the fast form on the default route, {case}", seeds=SEEDS)
