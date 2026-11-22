"""Float-range and nested-derivative contracts of the norms' and precision's seams: the exact GELU, the
precision seams and default-precision calls, the fused norms and products, and the declared derivative
(`dew.nn.derivatives.surrogate`; ledger 2641; the harness is tests/contract_support.py)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import EDGES, OWN_FORWARD_ORDERS, REVERSE_ONLY, assert_contract, normal_draw, on_a_tpu

from dew.nn import activations, ngpt, precision
from dew.nn.activations import exact_gelu, gated_product
from dew.nn.aligned import aligned_sum


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
def test_silu_holds_every_order(edge):
    """`activations.silu` (`x / (1 + e^-x)`, Dew's one logistic: its exp accurate and a true division, its
    tangent declared, row 7062): every order finite wherever float64 is a normal float32 and JAX's silu
    is."""
    (x,) = normal_draw(2671, (4, 7), scale=EDGES[edge])
    (direction,) = normal_draw(2672, x.shape, scale=EDGES[edge])
    (cotangent,) = normal_draw(2673, x.shape)
    assert_contract(activations.silu, (x,), (direction,), cotangent, f"silu {edge}",
                    truth_fn=jax.nn.silu, plain_fn=jax.nn.silu)


@pytest.mark.parametrize("edge", EDGES)
def test_accurate_exp_holds_every_order(edge):
    """`precision.accurate_exp` as a TPU takes it (Dew's own exp, `e^x` and `2^x`, its tangent declared, row
    7062), traced for an assumed v5e: every order finite wherever float64 is a normal float32 and JAX's exp
    is."""
    (x,) = normal_draw(2681, (4, 7), scale=EDGES[edge])
    (direction,) = normal_draw(2682, x.shape, scale=EDGES[edge])
    (cotangent,) = normal_draw(2683, x.shape)
    for base2, plain in ((False, jnp.exp), (True, jnp.exp2)):
        assert_contract(on_a_tpu(functools.partial(precision.accurate_exp, base2=base2)), (x,), (direction,),
                        cotangent, f"{plain.__name__} {edge}", truth_fn=plain, plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_an_aligned_sum_holds_every_order(edge):
    """`aligned.aligned_sum` of three-factor terms summed along an axis (a forked call's entry write, its
    terms aligned to the largest before the sum, its tangent the terms' own): every order finite wherever
    float64 is a normal float32 and the plain sum of products is."""
    primals = normal_draw(7951, (2, 1, 1), (2, 6, 1), (2, 1, 5), scale=EDGES[edge])
    directions = normal_draw(7952, (2, 1, 1), (2, 6, 1), (2, 1, 5), scale=EDGES[edge])
    (cotangent,) = normal_draw(7953, (1, 6, 5))

    def aligned(weight, value, key):
        return aligned_sum([(weight, value, key)], axis=0)

    def plain(weight, value, key):
        return jnp.sum(weight * value * key, axis=0, keepdims=True)

    assert_contract(aligned, primals, directions, cotangent, f"an aligned sum {edge}", truth_fn=plain,
                    plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_an_exact_widening_holds_every_order(edge):
    """`float_bits.widened_exactly` of bfloat16 to float32 (a subnormal set in float32's bits, a linear map
    transposed as the cast back, row 8966): every order finite wherever float64 is a normal float32 and the
    plain cast's is, and on training's mesh each order of a seed's draw on par with the float64 identity,
    its own truth (base review 20ad463a7 F4)."""
    from dew.nn.float_bits import widened_exactly

    (x,) = normal_draw(8966, (4, 7), scale=EDGES[edge], dtype=jnp.bfloat16)
    (direction,) = normal_draw(8967, x.shape, scale=EDGES[edge], dtype=jnp.bfloat16)
    (cotangent,) = normal_draw(8968, x.shape)

    def plain(x):
        return x.astype(jnp.float32)

    def draw(seed):
        """A seed's bfloat16 operand and direction and its float32 cotangent at unit scale."""
        return (normal_draw([8966, seed], (4, 7), dtype=jnp.bfloat16),
                normal_draw([8967, seed], (4, 7), dtype=jnp.bfloat16), normal_draw([8968, seed], (4, 7))[0])

    assert_contract(lambda x: widened_exactly(x, jnp.float32), (x,), (direction,), cotangent,
                    f"an exact widening {edge}", truth_fn=lambda x: x, plain_fn=plain,
                    draw=draw if edge == "normal" else None)


@pytest.mark.parametrize("edge", EDGES)
def test_softplus_holds_every_order(edge):
    """`precision.softplus` (its log1p by its own series, its derivative the logistic, row 7062): every
    order finite wherever float64 is a normal float32 and the plain formula, log(1 + exp(x)) as JAX writes
    it, is."""
    (x,) = normal_draw(2661, (4, 7), scale=EDGES[edge])
    (direction,) = normal_draw(2662, x.shape, scale=EDGES[edge])
    (cotangent,) = normal_draw(2663, x.shape)
    assert_contract(precision.softplus, (x,), (direction,), cotangent, f"softplus {edge}",
                    truth_fn=jax.nn.softplus, plain_fn=jax.nn.softplus)


@pytest.mark.parametrize("edge", EDGES)
def test_the_precision_seams_hold_every_order(edge):
    """`precision`'s correctly rounded division (a primitive of its own), compensated sum, rounded operand
    and sum of products rounded once (each a custom derivative): every order finite wherever float64 is a
    normal float32 and the plain formula, the division, the sum, the identity and the sum of products as JAX
    writes them, is."""
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

    def plain_terms(x, eps, dt, slope):
        return x + eps * dt + slope * dt ** 2

    def terms_once(x, eps, dt, slope):
        return precision.summed_once((x,), ((eps, dt), (slope, dt ** 2)))

    (scale,) = normal_draw(2657, (5, 1), scale=EDGES[edge] ** 0.25)
    (scale_direction,) = normal_draw(2658, (5, 1), scale=EDGES[edge] ** 0.25)
    assert_contract(terms_once, (numerator, denominator, scale, numerator),
                    (numerator_direction, denominator_direction, scale_direction, numerator_direction),
                    cotangent, f"a sum of products rounded once {edge}", truth_fn=plain_terms,
                    plain_fn=plain_terms)


@pytest.mark.parametrize("edge", EDGES)
def test_a_straight_through_rounding_holds_every_order(edge):
    """`fake_quant.straight_through` (a declared surrogate): its value the rounded operand's bit for bit, a
    negative zero kept, and every order finite wherever float64 is a normal float32 and the plain formula,
    the rounding as JAX writes it, is."""
    from dew.nn.fake_quant import straight_through

    (x,) = normal_draw(2654, (5, 4), scale=EDGES[edge])
    (direction,) = normal_draw(2655, x.shape, scale=EDGES[edge])
    (cotangent,) = normal_draw(2656, x.shape)
    rounded = x.at[0, 0].set(-0.0).astype(jnp.bfloat16)
    value = np.asarray(straight_through(x, rounded))
    np.testing.assert_array_equal(value, np.asarray(rounded.astype(x.dtype)))
    assert np.signbit(value[0, 0])
    np.testing.assert_array_equal(np.asarray(jax.grad(lambda x: straight_through(x, rounded).sum())(x)),
                                  np.ones(x.shape, np.float32))
    assert_contract(lambda x: straight_through(x, x.astype(jnp.bfloat16)), (x,), (direction,), cotangent,
                    f"a straight-through rounding {edge}",
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
    """`norm_swiglu`'s fused pre-norm seam, RMS norm, RMS norm with an activation after it (its tangent on
    the tile declared), gated norm, SiLU product and packed SiLU product (each through
    `generation.linear_tangent`), interpreted on a CPU as the norm tests call them, each launched as its
    selector's choice launches it, on the logical axes production names (`RESIDUAL` a norm's rows,
    `MLP_HIDDEN` a product's), so a mesh runs it per shard: every order finite wherever float64 is a normal
    float32 and the plain formula, the norm and the product as written, is."""
    import test_norm_swiglu as norms

    from dew.nn.sharding import MLP_HIDDEN, RESIDUAL

    rows = (2, 3, 16)
    x, branch, gate, up = normal_draw(2661, rows, rows, rows, rows, scale=EDGES[edge])
    (scale,) = normal_draw(2662, (16,))
    dx, dbranch, dgate, dup = normal_draw(2663, rows, rows, rows, rows, scale=EDGES[edge])
    (dscale,) = normal_draw(2664, (16,))
    (cotangent,) = normal_draw(2665, rows)
    sites = {
        "a pre-norm seam": (joined(norms.fused_seam(1e-6, jnp.float32, offset=False, after_cast=False,
                                                    axes=RESIDUAL)),
                            joined(lambda x, branch, scale: (x + branch, plain_rms(x + branch) * scale)),
                            (x, branch, scale), (dx, dbranch, dscale),
                            jnp.concatenate([cotangent, cotangent], axis=-1)),
        "an unweighted RMS norm": (norms.fused_unweighted(1e-6, axes=RESIDUAL), plain_rms, (x,), (dx,),
                                   cotangent),
        "an RMS norm with an activation": (norms.fused_activated(1e-6, jax.nn.swish, axes=RESIDUAL),
                                           lambda x, scale: jax.nn.swish(plain_rms(x) * scale), (x, scale),
                                           (dx, dscale), cotangent),
        "a gated RMS norm": (norms.fused_gated(1e-6, jnp.float32, "silu", axes=RESIDUAL),
                             lambda x, gate, scale: plain_rms(x) * scale * jax.nn.silu(gate),
                             (x, gate, scale), (dx, dgate, dscale), cotangent),
        "a SiLU product": (norms.fused_swiglu(axes=MLP_HIDDEN), lambda gate, up: jax.nn.silu(gate) * up,
                           (gate, up), (dgate, dup), cotangent),
        "a packed SiLU product": (norms.fused_packed(axes=MLP_HIDDEN),
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


@pytest.mark.parametrize("edge", EDGES)
def test_a_norms_xla_path_holds_every_order(edge):
    """An RMS norm's XLA path (`norms.stream_rms_normalized` at 'xla': its reverse mode returning the rows'
    cotangent only with the weight's gradient formed, `norms._formed`): every order, forward and reverse,
    under a nested jit and a scan, in the rows and the weight, finite wherever float64 is a normal float32
    and wherever its plain norm (`norms._xla_rms_normalized`, its float64 run the truth) is, at rows near
    float32's ends."""
    from dew.nn import norms

    primals = normal_draw(6178, (4, 8, 32), (32,), scale=EDGES[edge])
    direction = normal_draw(6179, (4, 8, 32), (32,), scale=EDGES[edge])
    (cotangent,) = normal_draw(6180, (4, 8, 32))

    offset, after_cast, fp32_statistics, weight_first = False, False, True, False

    def normed(rows, scale):
        return norms.stream_rms_normalized(rows, None, scale, 1e-6, None, offset, after_cast, fp32_statistics,
                                           implementation="xla")[1]

    def plain(rows, scale):
        return norms._xla_rms_normalized(rows, scale, 1e-6, None, offset, after_cast, fp32_statistics, None,
                                         weight_first)

    assert_contract(normed, primals, direction, cotangent, f"a norm's xla path {edge}", plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_a_module_reading_its_parameters_formed_together_holds_every_order(edge):
    """An input and the parameters a module reads through `derivatives.formed_together` (identity, its
    tangents tied by `with_tangents_formed`, row 11713): every order of a product of them, in the input and
    each parameter, finite wherever float64 is a normal float32 and wherever the plain product's (the same
    without the tie) is, at values near float32's ends."""
    from dew.nn.derivatives import formed_together

    primals = normal_draw(11713, (4, 8), (8,), (8,), scale=EDGES[edge])
    direction = normal_draw(11714, (4, 8), (8,), (8,), scale=EDGES[edge])
    (cotangent,) = normal_draw(11715, (4, 8))

    def read(x, gain, shift, fixed, count):
        return (x * gain + shift) * fixed + count.astype(x.dtype)

    def plain(x, gain, shift):
        return read(x, gain, shift, jnp.ones(8), jnp.arange(8))

    def tied(x, gain, shift):
        # Beside the moving parameters a fixed float one and an integer one, whose tangents are symbolic zeros
        # (an integer's JAX's `float0`): they pass as they came (row 11713's review).
        x, parameters = formed_together(x, (gain, shift, jnp.ones(8), jnp.arange(8)))
        return read(x, *parameters)

    assert_contract(tied, primals, direction, cotangent, f"parameters formed together {edge}", plain_fn=plain)

