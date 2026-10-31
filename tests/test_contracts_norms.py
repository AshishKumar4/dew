"""Float-range and nested-derivative contracts of the norms' and precision's seams: the exact GELU, the
precision seams and default-precision calls, the fused norms and products (ledger 2641; the harness is
tests/contract_support.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import EDGES, OWN_FORWARD_ORDERS, assert_contract, joined, normal_draw

from dew.nn import precision
from dew.nn.activations import exact_gelu


@pytest.mark.parametrize("edge", EDGES)
def test_an_exact_gelu_holds_every_order(edge):
    """`activations.exact_gelu` (rounded once to its dtype in both directions): every order finite wherever
    float64 is a normal float32 and the plain formula, x (1 + erf(x / sqrt 2)) / 2, is."""
    (x,) = normal_draw(2641, (4, 7), scale=EDGES[edge])
    (direction,) = normal_draw(2642, x.shape, scale=EDGES[edge])
    (cotangent,) = normal_draw(2643, x.shape)

    def plain(x):
        return 0.5 * x * (1 + jax.scipy.special.erf(x / np.sqrt(2.0)))

    assert_contract(exact_gelu, (x,), (direction,), cotangent, f"an exact gelu {edge}", truth_fn=plain,
                    plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_the_precision_seams_hold_every_order(edge):
    """`precision`'s correctly rounded division, compensated sum and rounded operand (each a custom
    derivative): every order finite wherever float64 is a normal float32 and the plain formula, the
    division, the sum and the identity as JAX writes them, is."""
    numerator, denominator = normal_draw(2648, (5, 4), (5, 4), scale=EDGES[edge])
    numerator_direction, denominator_direction = normal_draw(2649, (5, 4), (5, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(2650, (5, 4))
    # Denominators a quarter of the scale or more from zero, so the quotient is a normal float32 wherever
    # float64's is.
    denominator = jnp.copysign(jnp.abs(denominator) + 0.25 * EDGES[edge], denominator)
    assert_contract(precision.correctly_rounded_divide, (numerator, denominator),
                    (numerator_direction, denominator_direction), cotangent,
                    f"a correctly rounded division {edge}", truth_fn=jnp.divide, plain_fn=jnp.divide)
    (summed,) = normal_draw(2651, (3, 64), scale=EDGES[edge])
    (summed_direction,) = normal_draw(2652, summed.shape, scale=EDGES[edge])
    (summed_cotangent,) = normal_draw(2653, (3,))

    def plain_sum(values):
        return jnp.sum(values, axis=1)

    assert_contract(lambda values: precision.compensated_sum(values, (1,)), (summed,), (summed_direction,),
                    summed_cotangent, f"a compensated sum {edge}", truth_fn=plain_sum, plain_fn=plain_sum)
    assert_contract(lambda x: precision.rounded_operand(x, jnp.bfloat16), (numerator,),
                    (numerator_direction,), cotangent, f"a rounded operand {edge}",
                    plain_fn=lambda x: x.astype(jnp.bfloat16).astype(x.dtype))


def plain_rms(x, epsilon: float = 1e-6):
    """x over its root mean square, the RMS norm as written."""
    return x / jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + epsilon)


@pytest.mark.parametrize("edge", EDGES)
def test_the_fused_norms_and_products_hold_every_order(edge):
    """`norm_swiglu`'s fused pre-norm seam, RMS norm, gated norm, SiLU product and packed SiLU product (each
    through `generation.linear_tangent`), interpreted on a CPU as the norm tests call them: every order finite
    wherever float64 is a normal float32 and the plain formula, the norm and the product as written, is."""
    import test_norm_swiglu as norms

    x, branch, gate, up = normal_draw(2661, (3, 16), (3, 16), (3, 16), (3, 16), scale=EDGES[edge])
    (scale,) = normal_draw(2662, (16,))
    dx, dbranch, dgate, dup = normal_draw(2663, (3, 16), (3, 16), (3, 16), (3, 16), scale=EDGES[edge])
    (dscale,) = normal_draw(2664, (16,))
    (cotangent,) = normal_draw(2665, (3, 16))
    sites = {
        "a pre-norm seam": (joined(norms.fused_seam(1e-6, jnp.float32, offset=False, after_cast=False)),
                            joined(lambda x, branch, scale: (x + branch, plain_rms(x + branch) * scale)),
                            (x, branch, scale), (dx, dbranch, dscale),
                            jnp.concatenate([cotangent, cotangent], axis=-1)),
        "an unweighted RMS norm": (norms.fused_unweighted(1e-6), plain_rms, (x,), (dx,), cotangent),
        "a gated RMS norm": (norms.fused_gated(1e-6, jnp.float32, "silu"),
                             lambda x, gate, scale: plain_rms(x) * scale * jax.nn.silu(gate),
                             (x, gate, scale), (dx, dgate, dscale), cotangent),
        "a SiLU product": (norms.fused_swiglu(), lambda gate, up: jax.nn.silu(gate) * up, (gate, up),
                           (dgate, dup), cotangent),
        "a packed SiLU product": (norms.fused_packed(),
                                  lambda gate_up: jax.nn.silu(gate_up[..., 0, :]) * gate_up[..., 1, :],
                                  (jnp.stack([gate, up], axis=-2),), (jnp.stack([dgate, dup], axis=-2),),
                                  cotangent),
    }
    if edge == "near the largest":
        # The gated norm's reverse mode is NaN there where the plain formula is finite (ledger 2692); that
        # row's head adds the edge with its fix.
        del sites["a gated RMS norm"]
    for name, (fn, plain, primals, direction, pulled_back) in sites.items():
        assert_contract(fn, primals, direction, pulled_back, f"{name} {edge}", truth_fn=plain, plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_a_default_precision_call_holds_every_order_it_takes(edge):
    """A function traced at the default matmul precision (`precision.at_default_precision`, a custom VJP):
    every order it takes finite wherever float64 is a normal float32 and the product as written is; it takes
    no forward derivative of itself, nor a second (ledger 2658), every forward derivative of its gradient
    running."""
    def product(x, y):
        return x @ y

    primals = normal_draw(1208, (5, 16), (16, 3), scale=EDGES[edge])
    direction = normal_draw(1209, (5, 16), (16, 3), scale=EDGES[edge])
    (cotangent,) = normal_draw(1210, (5, 3))
    assert_contract(precision.at_default_precision(product), primals, direction, cotangent,
                    f"a default-precision call {edge}", truth_fn=product, plain_fn=product,
                    without=OWN_FORWARD_ORDERS)
