"""Float-range and nested-derivative contracts of the norms' and precision's seams: the exact GELU, the
precision seams and default-precision calls, the fused norms and products, and the declared derivative
(`dew.nn.derivatives.surrogate`; ledger 2641; the harness is tests/contract_support.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import EDGES, OWN_FORWARD_ORDERS, REVERSE_ONLY, assert_contract, normal_draw

from dew.nn import ngpt, precision
from dew.nn.activations import exact_gelu, gated_product


def joined(seam):
    """`seam`'s two outputs, the stream and its norm, side by side along the features: one output a contract
    pulls back."""
    return lambda *inputs: jnp.concatenate(seam(*inputs), axis=-1)


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


def _declared_cube(declare: str):
    """The cube declared through `dew.nn.derivatives`' builder `declare` as a rule its numerics need as
    written (`Protected`), its rule the cube's own derivative (a backward over residuals, or a tangent), so
    every order it takes is the plain formula's."""
    from dew.nn import derivatives

    def cube(x):
        return x * x * x

    if declare.endswith("tangent"):
        def tangent(primals, tangents):
            [x], [dx] = primals, tangents
            return cube(x), 3 * x * x * dx
        return getattr(derivatives, declare)(derivatives.Protected(), cube, tangent)

    def forward(x):
        return cube(x), x

    def backward(x, cotangent):
        return (3 * x * x * cotangent,)
    return getattr(derivatives, declare)(derivatives.Protected(), cube, forward, backward)


@pytest.mark.parametrize("declare", ["declared_derivative", "declared_tangent"])
@pytest.mark.parametrize("edge", EDGES)
def test_a_declared_derivative_holds_every_order(declare, edge):
    """`dew.nn.derivatives`' declaring builders (a reverse-mode derivative and a tangent, each of a kind): a
    cube declared with its own derivative holds every order it takes finite wherever float64 is a normal
    float32 and the plain formula is; a reverse-mode rule takes no forward order, as a bare `jax.custom_vjp`
    takes none (`REVERSE_ONLY`)."""
    declared = _declared_cube(declare)
    (x,) = normal_draw(4012, (5,), scale=EDGES[edge] ** (1 / 3))
    (direction,) = normal_draw(4013, (5,), scale=EDGES[edge] ** (1 / 3))
    (cotangent,) = normal_draw(4014, (5,))
    assert_contract(declared, (x,), (direction,), cotangent, f"{declare} {edge}",
                    plain_fn=lambda x: x * x * x,
                    without=() if declare.endswith("tangent") else REVERSE_ONLY)


@pytest.mark.parametrize("edge", EDGES)
def test_a_surrogate_holds_every_order(edge):
    """`dew.nn.derivatives.surrogate` (a custom JVP: the value one function's, the derivative a declared
    rule's): a cube rounded to bfloat16, differentiated as the cube's float32 derivative, every order finite
    wherever float64 is a normal float32 and the plain formula is, its value the rounded cube's bit for bit
    and its gradient the declared rule's, the same arithmetic, bit for bit; static arguments named out of
    order reach the value in place and the rule in their positions' order; and a derivative of a
    derivative meets the declared rule again: the straight-through rounding's (`precision.rounded_operand`)
    forward derivative of the gradient of its square is twice the direction, unrounded, as the rule
    declares at every order."""
    from dew.nn.derivatives import surrogate

    def rounded_cube(x, dtype):
        return (x * x * x).astype(dtype).astype(x.dtype)

    cube = surrogate(rounded_cube, lambda dtype, primals, tangents, value: 3 * primals[0] ** 2 * tangents[0],
                     nondiff_argnums=(1,))
    (x,) = normal_draw(3271, (5,), scale=EDGES[edge] ** (1 / 3))
    (direction,) = normal_draw(3272, (5,), scale=EDGES[edge] ** (1 / 3))
    (cotangent,) = normal_draw(3273, (5,))
    np.testing.assert_array_equal(np.asarray(cube(x, jnp.bfloat16)),
                                  np.asarray(rounded_cube(x, jnp.bfloat16)))
    np.testing.assert_array_equal(np.asarray(jax.grad(lambda x: cube(x, jnp.bfloat16).sum())(x)),
                                  np.asarray(3 * x**2))
    assert_contract(lambda x: cube(x, jnp.bfloat16), (x,), (direction,), cotangent, f"a surrogate {edge}",
                    truth_fn=lambda x: x * x * x, plain_fn=lambda x: x * x * x)

    scaled = surrogate(lambda scale, x, dtype: rounded_cube(scale * x, dtype),
                       lambda scale, dtype, primals, tangents, value: scale * tangents[0],
                       nondiff_argnums=(2, 0))
    value, moved = jax.jvp(lambda x: scaled(2.0, x, jnp.bfloat16), (x,), (direction,))
    np.testing.assert_array_equal(np.asarray(value), np.asarray(rounded_cube(2.0 * x, jnp.bfloat16)))
    np.testing.assert_array_equal(np.asarray(moved), np.asarray(2.0 * direction))

    def square(x):
        return jnp.sum(precision.rounded_operand(x, jnp.bfloat16) ** 2)

    unrounded = jnp.full((5,), 1 + 2.0**-12, jnp.float32)
    np.testing.assert_array_equal(np.asarray(jax.jvp(jax.grad(square), (x,), (unrounded,))[1]),
                                  np.asarray(2 * unrounded))


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
    for name, (fn, plain, primals, direction, pulled_back) in sites.items():
        assert_contract(fn, primals, direction, pulled_back, f"{name} {edge}", truth_fn=plain, plain_fn=plain)


def plain_stepped(residual, branch, alpha):
    """nGPT's join as its formula reads, each unit norm `precision.l2norm`'s: the reference its two-pass
    reverse (`ngpt.stepped`) is held to."""
    wide = precision.at_least_fp32(jnp.result_type(residual, branch, alpha))
    start, end = (precision.l2norm(x).astype(wide) for x in (residual, branch))
    rate = jnp.abs(alpha.astype(wide) * STEP)
    return precision.l2norm(start + rate * (end - start)).astype(residual.dtype)


def plain_hidden(gate, up, gate_gain, up_gain):
    """nGPT's normed hidden as its formula reads (`GatedMLP._ngpt_hidden` before its two-pass reverse):
    the gained gate and up rounded to their dtype, their SiLU product in at least fp32, unit-normed."""
    wide = precision.at_least_fp32(jnp.result_type(gate, up))
    gated = (gate.astype(wide) * ngpt.gain(gate_gain, HIDDEN, wide)).astype(gate.dtype)
    upped = (up.astype(wide) * ngpt.gain(up_gain, HIDDEN, wide)).astype(up.dtype)
    product = gated_product("swiglu", implementation="xla")
    return precision.l2norm(product(gated.astype(wide), upped.astype(wide)).astype(gate.dtype))


STEP, HIDDEN = 0.05, 4.0
"""A join's step scale and the hidden gains' factor, at a model's magnitudes."""


@pytest.mark.parametrize("edge", EDGES)
def test_ngpts_join_and_normed_hidden_hold_every_order(edge):
    """nGPT's join (`ngpt.stepped`) and normed hidden (`ngpt.normed_hidden`), each through
    `generation.linear_tangent` with a reverse of its own: every order finite wherever float64 is a normal
    float32 and the plain formula, the norms, the step, the gains and the product as written, is."""
    residual, branch, gate, up = normal_draw(2671, (3, 16), (3, 16), (3, 16), (3, 16), scale=EDGES[edge])
    alpha, gate_gain, up_gain = normal_draw(2672, (16,), (16,), (16,))
    directions = normal_draw(2673, (3, 16), (3, 16), (16,), (3, 16), (3, 16), (16,), (16,),
                             scale=EDGES[edge])
    (cotangent,) = normal_draw(2674, (3, 16))
    product = gated_product("swiglu", implementation="xla")
    sites = {
        "nGPT's join": (lambda residual, branch, alpha: ngpt.stepped(residual, branch, alpha, STEP),
                        plain_stepped, (residual, branch, alpha), directions[:3]),
        "nGPT's normed hidden": (lambda gate, up, gate_gain, up_gain: ngpt.normed_hidden(
                                     gate, up, gate_gain, up_gain, HIDDEN, product),
                                 plain_hidden, (gate, up, gate_gain, up_gain), directions[3:]),
    }
    for name, (fn, plain, primals, direction) in sites.items():
        assert_contract(fn, primals, direction, cotangent, f"{name} {edge}", truth_fn=plain, plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_a_default_precision_call_holds_every_order_it_takes(edge):
    """A function traced at the default matmul precision (`precision.at_precision('default')`, a custom VJP):
    every order it takes finite wherever float64 is a normal float32 and the product as written is; it takes
    no forward derivative of itself, nor a second (ledger 2658), every forward derivative of its gradient
    running."""
    def product(x, y):
        return x @ y

    primals = normal_draw(1208, (5, 16), (16, 3), scale=EDGES[edge])
    direction = normal_draw(1209, (5, 16), (16, 3), scale=EDGES[edge])
    (cotangent,) = normal_draw(1210, (5, 3))
    assert_contract(precision.at_precision('default')(product), primals, direction, cotangent,
                    f"a default-precision call {edge}", truth_fn=product, plain_fn=product,
                    without=OWN_FORWARD_ORDERS)
