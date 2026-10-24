"""Attention's probability dropout: a dropping call on the template is on par with the reference path
on the same draw, forward and backward, both measured from float64 (check 1), and a dropping call
reports its Route like every other attention call. The reference's draw is unbiased in bf16.
"""

import jax
import jax.numpy as jnp
import pytest
from reference_error import assert_as_exact
from test_attention_route import CASES, HEADS, KV_HEADS, WIDTH, drawing, keywords
from test_attention_template import DTYPES, KEYS, QUERIES, ROWS, SEEDS, geometry

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels import generation

RATE = 0.25
"""The dropout rate the check-1 tests drop at: a quarter of the weights."""

DROPPED = ("full-grouped", "causal-window", "packed-documents", "sinks-softcap")


def dropping(case: dict, implementation: str):
    """`case`'s call of `scaled_dot_product_attention` on `implementation`, dropping at `RATE` from
    `rng`, as a function of `(query, key, value, sinks, rng)`."""
    def call(query, key, value, sinks, rng):
        return scaled_dot_product_attention(query, key, value, implementation=implementation, sinks=sinks,
                                            dropout_rate=RATE, dropout_rng=rng, deterministic=False,
                                            **keywords(case))

    return call


def pulled(case: dict, implementation: str):
    """`case`'s dropping call on `implementation`: its output, then its gradients of the query, key,
    value and sinks under the cotangent `dout`, the sinks' gradient None where the case has none."""
    call = dropping(case, implementation)

    def run(query, key, value, sinks, dout, rng):
        out, pull = jax.vjp(lambda q, k, v, s: call(q, k, v, s, rng), query, key, value, sinks)
        return (out, *pull(dout))

    return run


def dropping_draw(queries: int, keys: int, dtype, *, sinks: bool):
    """A draw `(query, key, value, sinks, dout, rng)`: the route tests' draw, with the dropout key from
    its seed."""
    base = drawing(queries, keys, dtype, sinks=sinks)

    def draw(seed: int):
        return (*base(seed), jax.random.key(seed))

    return draw


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize("name", DROPPED)
def test_a_dropping_template_call_is_on_par_with_the_reference_path_forward_and_backward(name, dtype):
    """The template's output and its gradients of the query, key, value and sinks, on a dropping call,
    on par with the reference path (`implementation='reference'`) on the same draw, both measured from
    float64 (check 1)."""
    case = CASES[name]
    queries, keys = geometry(case)
    assert_as_exact(jax.jit(pulled(case, "template")), jax.jit(pulled(case, "reference")),
                    dropping_draw(queries, keys, dtype, sinks=bool(case.get("sinks"))),
                    f"dropping template {name} {jnp.dtype(dtype).name}", seeds=SEEDS)


def test_a_dropping_call_reports_its_route_and_a_chosen_template_runs_it():
    """A dropping call reports the route it runs (the reference path on a CPU, where the template is
    not preferred), and `generation.chosen(attention='template')` runs it on the template instead."""
    query = jnp.zeros((ROWS, QUERIES, HEADS, WIDTH), jnp.bfloat16)
    key = jnp.zeros((ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16)
    rng = jax.random.key(0)

    def dropping_call():
        return scaled_dot_product_attention(query, key, key, dropout_rate=RATE, dropout_rng=rng,
                                            deterministic=False)

    with generation.routes() as found:
        dropping_call()
    (route,) = [route for route in found if route.choice == "attention"]
    assert route.kernel == "reference"
    with generation.chosen(attention="template"), generation.routes() as chosen:
        out = dropping_call()
    (named,) = [route for route in chosen if route.choice == "attention"]
    assert named.kernel == "template"
    assert out.shape == (ROWS, QUERIES, HEADS, WIDTH)


def test_the_reference_draw_keeps_a_bf16_call_unbiased():
    """Each of 128 keys of equal weight 1/128 (exact in bf16) keeps with probability 0.9 and scales by
    1/0.9, so each query row's output has mean 1 over the draws. The 16384 rows are independent draws:
    a row's output has standard deviation 0.0295, so the mean of the rows has standard error 0.00023,
    and the bound is 3.5 of those. A multiplier rounded to bf16 (1.109375 for 1/0.9) would bias the
    mean by -0.00156."""
    query = jnp.zeros((1, 16384, 1, 8), jnp.bfloat16)
    key = jnp.zeros((1, 128, 1, 8), jnp.bfloat16)
    value = jnp.ones((1, 128, 1, 8), jnp.bfloat16)
    out = scaled_dot_product_attention(query, key, value, implementation="reference", dropout_rate=0.1,
                                       dropout_rng=jax.random.key(0), deterministic=False)
    assert abs(float(jnp.mean(out.astype(jnp.float32))) - 1) < 0.0008
