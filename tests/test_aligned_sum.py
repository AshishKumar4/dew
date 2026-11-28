"""`dew.nn.aligned.aligned_sum`, the exact sum of exact products rounded once, and its cotangents: its value
at the range's edges, its derivatives of every order, its placement on a mesh and its lowering in kernels."""

import contextlib
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import CANCELLING_LEVELS, MOST
from reference_error import assert_as_exact

from dew.nn.aligned import aligned_segment_sum, aligned_sum
from dew.telemetry.instrumentation import hlo_instructions

ALIGNED_CASES = {
    "cancelling": ([(2.0 ** 100, 2.0 ** -20, 2.0 ** 60), (2.0 ** 100, 2.0 ** -20, -(2.0 ** 60))], 0.0),
    "in range though its terms are not": ([(1 + 2.0 ** -20, 2.0 ** 70, 2.0 ** 70),
                                           (-1.0, 2.0 ** 70, 2.0 ** 70)], 2.0 ** 120),
    "past the largest": ([(1.0, 2.0 ** 70, 2.0 ** 70), (1.0, 2.0 ** 70, 2.0 ** 70)], float("inf")),
    "a unit below cancelling terms": ([(2.0 ** 100, 2.0 ** 27, 1.0), (2.0 ** 100, -(2.0 ** 27), 1.0),
                                       (1.0, 1.0, 1.0)], 1.0),
    "a term past the normals below cancelling terms": ([(2.0 ** 100, 2.0 ** 60, 2.0 ** 60),
                                                        (-(2.0 ** 100), 2.0 ** 60, 2.0 ** 60),
                                                        (2.0 ** -50, 2.0 ** -50, 2.0 ** -20)], 2.0 ** -120),
    "a tie below cancelling terms": ([(2.0 ** 100, 2.0 ** 27, 1.0), (2.0 ** 100, -(2.0 ** 27), 1.0),
                                      (2.0 ** 25, 1.0, 1.0), (-(2.0 ** 24), 1.0, 1.0),
                                      (-(2.0 ** 24), 1.0, 1.0), (-2.0, 1.0, 1.0)], -2.0),
    "a band below two that cancel": ([(2.0 ** 100, 2.0 ** 27, 1.0), (2.0 ** 100, -(2.0 ** 27), 1.0),
                                      (2.0 ** 25, 1.0, 1.0), (-(2.0 ** 24), 1.0, 1.0),
                                      (-(2.0 ** 24), 1.0, 1.0), (2.0 ** -50, 2.0 ** -50, 1.0)], 2.0 ** -100),
    "three levels cancelling beside the largest": ([(key, value, 1.0) for key, value in CANCELLING_LEVELS],
                                                   MOST),
    "cancelling past twice the precision": ([(2.0 ** 80, 1.0, 1.0), (2.0 ** 50, 1.0, 1.0), (1.0, 1.0, 1.0),
                                             (-(2.0 ** 50), 1.0, 1.0), (-1.0, 1.0, 1.0),
                                             (-(2.0 ** 80), 1.0, 1.0)], 0.0),
}
"""`aligned_sum`'s cases: three-factor terms whose products leave fp32's range, and their sum, exact in any
order of summing (their sum a float32): a term more than float32's exponent range below the largest kept,
where the largest cancel (base review 224fcf53a F1), and a lower band's rounding, -2^25 - 2 a tie, kept
where its rounded sum cancels a higher band's (base review 644ef5535 F1), a third band's 2^-100 kept
where the two above it cancel (base review f5e6e678d F1), and three levels of cancelling terms beside the
largest (`contract_support.CANCELLING_LEVELS`) the largest, not inf, and
`2^80 + 2^50 + 1 - 2^50 - 1 - 2^80` zero, which floats summed in some order lose (row 8757)."""


@pytest.mark.parametrize("case", list(ALIGNED_CASES))
def test_an_aligned_sum_is_exact_where_its_terms_leave_fp32s_range(case):
    """`aligned_sum` of fp32 terms each past fp32's range where their sum is in it or past it: exactly the
    sum, as a float32 holds it (a subnormal sum is no case: XLA:CPU flushes subnormals to zero). Summed as
    formed, two terms of +-2^140 were inf - inf, NaN (base review ead11c83a F1)."""
    terms, total = ALIGNED_CASES[case]
    found = aligned_sum([tuple(jnp.float32(factor) for factor in term) for term in terms])
    assert float(found) == total


ROUNDED_CASES = {
    "a tie to even": ([1.0, 2.0 ** -24], 1.0),
    "past a tie by a sticky bit": ([1.0, 2.0 ** -24, 2.0 ** -60], 1 + 2.0 ** -23),
    "below a tie by a sticky bit": ([1.0, 2.0 ** -24, -(2.0 ** -60)], 1.0),
    "the largest past by half its unit": ([float(np.finfo(np.float32).max), 2.0 ** 103], float("inf")),
    "a cancelled zero": ([1.0, -1.0], 0.0),
    "negative zeros": ([-0.0, -0.0], -0.0),
    "infinities of both signs": ([float("inf"), float("-inf")], float("nan")),
    "an infinity beside a finite term": ([float("inf"), 1.0], float("inf")),
    "a NaN": ([float("nan"), 1.0], float("nan")),
}
"""`aligned_sum`'s rounding cases: terms of one factor and the float their sum rounds to, nearest and ties
to even, every bit below the rounding place sticky, its zeros' and non-finite terms' as IEEE's."""


@pytest.mark.parametrize("case", list(ROUNDED_CASES))
def test_an_aligned_sum_rounds_its_exact_sum_once_to_nearest_ties_to_even(case):
    """`aligned_sum` of one-factor fp32 terms (`ROUNDED_CASES`): the float their exact sum rounds to, to
    nearest, ties to even, a bit far below the tie deciding it, past the largest infinite, a cancelled sum
    positive zero and negative zeros' negative, infinities' and NaNs' as IEEE sums them (row 8757)."""
    terms, total = ROUNDED_CASES[case]
    found = np.asarray(aligned_sum([(jnp.float32(term),) for term in terms]))
    np.testing.assert_array_equal(found, np.float32(total))
    # A zero's sign is IEEE's; a NaN's is none it states (base review 6c68b90ad F5).
    assert total != 0 or np.signbit(found) == np.signbit(np.float32(total))


def test_an_aligned_sum_of_fp32_terms_under_x64_keeps_fp32():
    """`aligned_sum` of three fp32 ones with x64 on: fp32 3, where its integer digits' sums widened to int64
    and their bits read as fp32 pairs (base review 6c68b90ad F1)."""
    with jax.enable_x64(new_val=True):
        found = aligned_sum([(jnp.ones((3,), jnp.float32),)], axis=0)
    assert found.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(found), [3.0])


def test_an_aligned_sum_of_mixed_factors_reads_a_subnormal_from_its_bits():
    """`aligned_sum` of an fp32 subnormal times an fp64 factor, with x64 on: `2^-149 2^149` is fp64's 1 and
    `-(0x007fffff) 2^-149 2^149` its `-(2^23 - 1)`, each factor widened from its bits, where cast to fp64
    first XLA:CPU flushed the subnormal and the sum was 0 (row 8966)."""
    with jax.enable_x64(new_val=True):
        subnormal = jnp.asarray(np.asarray([1, 0x807FFFFF], np.uint32).view(np.float32))
        found = aligned_sum([(subnormal, jnp.full((2,), 2.0 ** 149, jnp.float64))])
    assert found.dtype == jnp.float64
    np.testing.assert_array_equal(np.asarray(found), [1.0, -(2.0 ** 23 - 1)])


def test_an_aligned_sum_of_mixed_factors_differentiates_each_factor():
    """`aligned_sum([(a, b)])` of an fp32 `a` and an fp64 `b`, with x64 on: its tangent and gradient in `a`
    at `a = 0` and at the subnormal `2^-149` are `b`, 2, each factor's own type's, where its exact widening
    read `a`'s bits outside the declared rule and its tangent was 0 (base review ed11c8e4e F1)."""
    with jax.enable_x64(new_val=True):
        a = jnp.asarray([0.0, 2.0 ** -149], jnp.float32)
        b = jnp.full((2,), 2.0, jnp.float64)
        _, tangent = jax.jvp(lambda a: aligned_sum([(a, b)]), (a,), (jnp.ones_like(a),))
        gradient = jax.grad(lambda a: jnp.sum(aligned_sum([(a, b)])))(a)
    assert tangent.dtype == jnp.float64 and gradient.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(tangent), [2.0, 2.0])
    np.testing.assert_array_equal(np.asarray(gradient), [2.0, 2.0])


def test_an_aligned_sums_tangent_keeps_its_float64_sum_where_no_direction_moves_its_float64_term():
    """The tangent of `aligned_sum([(a, 2^70), (1,)])` of fp32 `a` and `2^70` and a float64 `1`, with x64
    on, at `a = 0` in the direction `2^70`: float64's `2^140`, the sum's type, where the float64 term no
    direction moves was dropped from the tangent's terms and they were summed in fp32, infinite (base review
    a8a1fdea7 F1)."""
    with jax.enable_x64(new_val=True):
        big = jnp.float32(2.0 ** 70)
        value, tangent = jax.jvp(lambda a: aligned_sum([(a, big), (jnp.float64(1.0),)]),
                                 (jnp.float32(0.0),), (big,))
    assert value.dtype == tangent.dtype == jnp.float64
    assert float(tangent) == 2.0 ** 140 and float(value) == 1.0


def test_an_aligned_sum_of_a_weak_float64_factor_keeps_float32():
    """`aligned_sum` of one weakly typed float64 factor (`jnp.asarray(1.0)` with x64 on) is float32's 1, the
    factor narrowed as promotion narrows it, where its exact widening read float64's bits as float32's and
    shifted by a negative count (base review ed11c8e4e F2)."""
    with jax.enable_x64(new_val=True):
        found = aligned_sum([(jnp.asarray(1.0),)])
    assert found.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(found), 1.0)


def test_an_aligned_sums_subnormal_direction_widens_exactly():
    """The tangent of `aligned_sum([(a, 2^149)])` of a float32 `a` and a float64 `2^149`, with x64 on, at
    `a = 0` in the direction `2^-149` (built from its bits): float64's 1, the direction widened from its
    bits, where XLA:CPU's cast flushed it to zero (base review 20ad463a7 F2)."""
    direction = jnp.asarray(np.asarray(1, np.uint32).view(np.float32))
    with jax.enable_x64(new_val=True):
        _, tangent = jax.jvp(lambda a: aligned_sum([(a, jnp.float64(2.0 ** 149))]), (jnp.float32(0.0),),
                             (direction,))
    assert tangent.dtype == jnp.float64 and float(tangent) == 1.0


@pytest.mark.parametrize("axis", [None, 0], ids=["elementwise", "along the rows"])
def test_aligned_cotangents_are_the_aligned_sums_vjp(axis):
    """`aligned.aligned_cotangents` of an entry write's three-factor term (a weight near 2^60, keys and
    values) and a two-factor term of factors of their own, elementwise and summed along an axis: each
    distinct factor's cotangent bit for bit what `jax.vjp` of `aligned_sum` gives it, compiled, so a
    backward that takes no autodiff (a Pallas TPU kernel's) forms an aligned sum's transpose as its reverse
    mode does (row 8666)."""
    from dew.nn.aligned import aligned_cotangents

    rng = np.random.default_rng(8666)
    weight, key, value, left, right = (jnp.asarray(rng.standard_normal(shape) * scale, jnp.float32)
                                       for shape, scale in (((2, 1, 1), 2.0 ** 60), ((2, 6, 1), 1.0),
                                                            ((2, 1, 5), 1.0), ((2, 6, 5), 2.0 ** 40),
                                                            ((2, 6, 5), 2.0 ** 30)))
    shape = aligned_sum([(weight, key, value), (left, right)], axis).shape
    cotangent = jnp.asarray(rng.standard_normal(shape), jnp.float32)

    def summed(*factors):
        return aligned_sum([factors[:3], factors[3:]], axis)

    _, pullback = jax.vjp(summed, weight, key, value, left, right)
    want = jax.jit(pullback)(cotangent)
    found = jax.jit(lambda *factors: aligned_cotangents([factors[:3], factors[3:]], cotangent, axis))(
        weight, key, value, left, right)
    for got, wanted in zip(found, want, strict=True):
        assert got.dtype == wanted.dtype and got.shape == wanted.shape
        np.testing.assert_array_equal(np.asarray(got), np.asarray(wanted))


def test_aligned_cotangents_sum_a_shared_factors_terms_before_they_round():
    """`aligned_cotangents` of terms `(w, 2^70, 2^70)`, `(w, -2^70, 2^70)` and `(w, 1, 1)` sharing one
    float32 `w` (an entry write's weight) at a cotangent of one: `w`'s one cotangent, 1, as `jax.vjp` gives
    it, its terms' products summed together before they round, where each term's rounded alone was
    `+inf`, `-inf` and 1, summed NaN (review of bba02dcf8 F2); and a float64 factor's beside a float32 one
    under x64 at a weak cotangent of `2^100`: `2^200`, the cotangent in the sum's float64, where taken
    against the float32 factor it overflowed (F4)."""
    from dew.nn.aligned import aligned_cotangents

    w, big, one = jnp.float32(3.0), jnp.float32(2.0 ** 70), jnp.float32(1.0)
    terms = [(w, big, big), (w, -big, big), (w, one, one)]
    (shared, *_) = jax.jit(lambda w: aligned_cotangents([(w, *term[1:]) for term in terms], one))(w)
    _, pullback = jax.vjp(lambda w: aligned_sum([(w, *term[1:]) for term in terms]), w)
    assert float(shared) == float(pullback(one)[0]) == 1.0
    with jax.enable_x64(new_val=True):
        a, b = jnp.float32(2.0 ** 100), jnp.float64(1.0)
        _, of_b = aligned_cotangents([(a, b)], 2.0 ** 100)
    assert of_b.dtype == jnp.float64 and float(of_b) == 2.0 ** 200


def test_an_aligned_sum_of_three_factors_over_a_long_axis_is_its_exact_sum():
    """`aligned_sum([(x, x, x)], axis=0)` of 2^20 fp32 ones: 2^20, its digits as narrow as its count of limbs
    needs, where four million pieces were refused (base review 6c68b90ad F2)."""
    x = jnp.ones((2 ** 20,), jnp.float32)
    np.testing.assert_array_equal(np.asarray(aligned_sum([(x, x, x)], axis=0)), [2.0 ** 20])


@pytest.mark.parametrize(("term", "total"), [((2.0 ** -75, 2.0 ** -75, float("inf")), float("inf")),
                                             ((0.0, float("inf"), 1.0), float("nan"))],
                         ids=["an infinity beside a product that underflows", "an infinity times zero"])
def test_an_aligned_sums_non_finite_product_is_ieees(term, total):
    """`aligned_sum` of one three-factor fp32 term with an infinite factor: `2^-75 2^-75 inf` infinite, where
    the finite factors' rounded product, zero, met the infinity, NaN; and `0 inf` NaN (base review 6c68b90ad
    F3)."""
    found = aligned_sum([tuple(jnp.float32(factor) for factor in term)])
    np.testing.assert_array_equal(np.asarray(found), np.float32(total))


@pytest.mark.parametrize("x64", [False, True], ids=["x32", "x64"])
def test_an_aligned_sum_in_a_tpu_kernel_lowers_for_a_tpu(x64):
    """A Pallas launch summing a `[16, 16]` tile's rows aligned (as the forked SSD's entry-write gradient
    does), traced for an assumed v5p, with x64 off and on: it lowers on any host to a Mosaic kernel, where
    under x64 its digit loop's index was 64-bit and Mosaic's 32 (base review d710d7ee9 F2, row 8757)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import AbstractDevice, AbstractMesh, AxisType

    from dew.nn.kernels import launches
    from dew.nn.kernels.generation import TARGETS, assumed
    from dew.nn.kernels.lowered import custom_calls

    def body(x_ref, y_ref, out_ref):
        out_ref[...] = aligned_sum([(x_ref[...], y_ref[...])], axis=0)

    spec = pl.BlockSpec((16, 16), lambda: (0, 0), memory_space=pltpu.VMEM)
    call = launches.pallas_call(body, grid=(), in_specs=[spec, spec],
                                out_specs=pl.BlockSpec((1, 16), lambda: (0, 0), memory_space=pltpu.VMEM),
                                out_shape=jax.ShapeDtypeStruct((1, 16), jnp.float32), interpret=False,
                                name="aligned_sum_lowered")
    tile = jax.ShapeDtypeStruct((16, 16), jnp.float32)
    device = AbstractDevice(device_kind=TARGETS["v5p"].device_kind, num_cores=None, platform="tpu")
    mesh = AbstractMesh((1,), ("data",), axis_types=(AxisType.Auto,), abstract_device=device)
    with jax.enable_x64(new_val=x64), assumed("v5p"), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(call).trace(tile, tile).lower(lowering_platforms=("tpu",))
    assert any(found.target == "tpu_custom_call" for found in custom_calls(lowered))


def test_an_aligned_sums_tangent_over_four_factors_and_2_28_terms_traces():
    """The tangent of `aligned_sum([(x, x, x, x)], axis=0)` along `x` for fp32 `x` of 2^28 terms, lowered:
    its four terms' 2^30 products take digits the integers hold beside their carries, each term carried as
    it is added, where one sum of all 2^30 was refused (base review 5a5cb5c98 F2, row 8757)."""
    x = jax.ShapeDtypeStruct((2 ** 28,), jnp.float32)

    def tangent(x):
        return jax.jvp(lambda x: aligned_sum([(x, x, x, x)], axis=0), (x,), (x,))[1]

    lowered = jax.jit(tangent).trace(x).lower()
    assert lowered.out_info.shape == (1,)


def test_an_aligned_sums_tangent_moves_only_the_factors_it_reaches():
    """The tangent along `x` of `aligned_sum([(x,), (inf, 1)])` on fp32 scalars, and the gradient of `x`:
    one each, the constant term's infinity moving nothing, where its zero direction met the other constant's
    infinity, NaN (base review ee7d1e761 F1, row 8757)."""
    infinite, one = jnp.float32(jnp.inf), jnp.float32(1.0)

    def summed(x):
        return aligned_sum([(x,), (infinite, one)])

    assert float(jax.jvp(summed, (one,), (one,))[1]) == 1.0
    assert float(jax.grad(summed)(one)) == 1.0


def test_an_aligned_sums_tangent_takes_the_values_layout():
    """The tangent along a scalar `x` of `aligned_sum([(x,), (v,)])` beside a constant `v` of `[10, 20]`:
    `[1, 1]`, the value's layout, where the moving term alone made a scalar (base review 0c6ef9c11 F1)."""
    v = jnp.asarray([10.0, 20.0], jnp.float32)
    tangent = jax.jvp(lambda x: aligned_sum([(x,), (v,)]), (jnp.float32(1.0),), (jnp.float32(1.0),))[1]
    np.testing.assert_array_equal(np.asarray(tangent), [1.0, 1.0])


def test_an_aligned_sums_mixed_derivative_holds_a_constant_infinity_still():
    """The tangent along `y` of the tangent along `x` of `aligned_sum([(x, y), (x, inf, 1)])` at `x = y =
    1`: one, the constant infinity held still, where its residual's zero tangent met it, NaN (base review
    0c6ef9c11 F2)."""
    infinite, one = jnp.float32(jnp.inf), jnp.float32(1.0)

    def along_x(y):
        return jax.jvp(lambda x: aligned_sum([(x, y), (x, infinite, one)]), (one,), (one,))[1]

    assert float(jax.jvp(along_x, (one,), (one,))[1]) == 1.0


def test_an_aligned_sums_tangent_keeps_the_values_placement():
    """The tangent along a scalar `x` of `aligned_sum([(x,), (v,)])` beside a constant `v` of `[10, 20]`
    split over an Explicit two-device data axis: `[1, 1]` placed as `v`'s value is, where the moving term
    alone broadcast whole and JAX refused its placement (base review c439a41f3 F1, row 8757)."""
    mesh = jax.make_mesh((2,), ("data",), (jax.sharding.AxisType.Explicit,), devices=jax.devices()[:2])
    with jax.set_mesh(mesh):
        split = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
        v = jax.device_put(jnp.asarray([10.0, 20.0], jnp.float32), split)
        tangent = jax.jvp(lambda x: aligned_sum([(x,), (v,)]), (jnp.float32(1.0),), (jnp.float32(1.0),))[1]
    np.testing.assert_array_equal(np.asarray(tangent), [1.0, 1.0])
    assert tangent.sharding.is_equivalent_to(split, tangent.ndim)


def test_an_aligned_sum_over_two_terms_of_2_30_lowers():
    """`aligned_sum([(x,), (y,)], axis=0)` of fp32 `x` of 2^30 terms and `y` of `2^30 - 1`, lowered: one-bit
    digits whose partial sums and carry, in `[-n, n - 1]`, int32 holds, where a looser carry bound refused
    them (base review c439a41f3 F2, row 8757)."""
    x, y = (jax.ShapeDtypeStruct((length,), jnp.float32) for length in (2 ** 30, 2 ** 30 - 1))
    lowered = jax.jit(lambda x, y: aligned_sum([(x,), (y,)], axis=0)).trace(x, y).lower()
    assert lowered.out_info.shape == (1,)


def test_an_aligned_sums_second_order_holds_a_constant_direction_still():
    """The tangent along `db` of the tangent of `aligned_sum([(a, b)])` at `a` one and `b` infinite, `a`'s
    direction a constant one: `a`, one, where the constant direction's zero second-order tangent met the
    infinity, NaN (row 8757)."""
    one, infinite = jnp.float32(1.0), jnp.float32(jnp.inf)

    def tangent(db):
        return jax.jvp(lambda a, b: aligned_sum([(a, b)]), (one, infinite), (one, db))[1]

    assert float(jax.jvp(tangent, (one,), (one,))[1]) == 1.0


def test_an_aligned_sums_second_order_is_one_exact_sum():
    """The tangent along `(b, db)` of the tangent of `aligned_sum([(a, b)])` along `(a, db)` at `a` 2^70, `da`
    `a` itself, outer direction `(-2^70, 2^70)`: `a vb + a vdb`, its parts `-2^140` and `2^140` summed to
    zero, where each, rounded apart, overflowed and added to inf - inf, NaN (row 8757)."""
    a = jnp.float32(2.0 ** 70)

    def tangent(b, db):
        return jax.jvp(lambda a, b: aligned_sum([(a, b)]), (a, b), (a, db))[1]

    direction = (jnp.float32(-(2.0 ** 70)), jnp.float32(2.0 ** 70))
    assert float(jax.jvp(tangent, (jnp.float32(1.0), jnp.float32(0.0)), direction)[1]) == 0.0


def test_an_aligned_sums_second_order_keeps_a_lone_parts_sign():
    """The tangent along `db` alone of the tangent of `aligned_sum([(a, b)])` at `a` one, `db`'s direction a
    negative zero: a negative zero, where its one part's plain sum added it to a positive zero (row 8757)."""
    one = jnp.float32(1.0)

    def tangent(db):
        return jax.jvp(lambda a, b: aligned_sum([(a, b)]), (one, one), (one, db))[1]

    found = np.asarray(jax.jvp(tangent, (one,), (jnp.float32(-0.0),))[1])
    assert found == 0.0 and np.signbit(found)


def test_an_aligned_sums_tangents_value_and_gradient_keeps_its_joint_tangent():
    """The tangent along `(0, -2^70, 2^70 (1 + k 2^-23), 0)` at `(2^70, 2^70, 2^70, 0)`, `k` each of 0 to 8
    in check 1's first stage (the seed's modulo `offsets`, its seeds; rows 9087, 9091), of the value of
    `aligned_sum([(a, b)])`'s tangent's value-and-gradient, `da vb + vda b`: `k 2^117`, its two products each
    2^140 past float32's largest, cancelling exactly at `k` zero, held by check 1 against the formula in
    float64 rounded once (exact, so Dew must be) where an untraced map's parts, rounded apart, added to inf -
    inf, NaN (rows 8757, 8975)."""
    from arm_forms import as_given

    big, offsets = 2.0 ** 70, 9

    def draw(seed):
        k = seed % offsets
        point, direction = (big, big, big, 0.0), (0.0, -big, big * (1 + k * 2.0 ** -23), 0.0)
        return tuple(np.float32(x) for x in (*point, *direction))

    def tangent(a, b, da, db):
        return jax.jvp(lambda a, b: aligned_sum([(a, b)]), (a, b), (da, db))[1]

    def ours(*inputs):
        def value(*point):
            return jax.value_and_grad(tangent)(*point)[0]

        return jax.jvp(value, inputs[:4], inputs[4:])[1]

    def formula(a, b, da, db, va, vb, vda, vdb):
        return vda * b + da * vb + va * db + a * vdb

    rounded = as_given(lambda *inputs: np.float32(formula(*(np.float64(x) for x in inputs))))
    assert_as_exact(jax.jit(ours), rounded, draw, "an aligned sum's tangent's value and gradient",
                    seeds=offsets, truth_fn=jax.jit(formula))


def test_an_aligned_sums_tangent_pulls_back_inside_its_sum():
    """The gradient in `b` of `aligned_sum([(a, b), (v,)])`'s tangent along `a`'s direction a quarter, `v`
    a constant `[0, 0]`, under a cotangent of `[fp32's largest] * 2`: half the largest, where the tangent
    broadcast to the value's layout outside the sum, its transpose summing the cotangents first, inf (base
    review 89fc6bccd F1, row 8757)."""
    largest, v = float(np.finfo(np.float32).max), jnp.zeros((2,), jnp.float32)

    def tangent(b):
        return jax.jvp(lambda a: aligned_sum([(a, b), (v,)]), (jnp.float32(1.0),), (jnp.float32(0.25),))[1]

    _, pulled = jax.vjp(tangent, jnp.float32(1.0))
    assert float(pulled(jnp.full((2,), largest, jnp.float32))[0]) == np.float32(largest / 2)


def test_an_aligned_sum_in_an_interpreted_kernel_keeps_its_derivative():
    """The tangent of a Pallas launch, interpreted, writing `aligned_sum([(x,)], axis=0)` of a `[16, 16]`
    tile: the direction's own sum, where an interpreted body took the bare bits, which carry none, zero
    (base review 89fc6bccd F2, row 8757)."""
    from jax.experimental import pallas as pl

    from dew.nn.kernels import launches

    def body(x_ref, out_ref):
        out_ref[...] = aligned_sum([(x_ref[...],)], axis=0)

    call = launches.pallas_call(body, grid=(), in_specs=[pl.BlockSpec((16, 16), lambda: (0, 0))],
                                out_specs=pl.BlockSpec((1, 16), lambda: (0, 0)),
                                out_shape=jax.ShapeDtypeStruct((1, 16), jnp.float32), interpret=True,
                                name="aligned_sum_interpreted")
    x, direction = jnp.ones((16, 16), jnp.float32), jnp.arange(256, dtype=jnp.float32).reshape(16, 16)
    tangent = jax.jvp(call, (x,), (direction,))[1]
    np.testing.assert_array_equal(np.asarray(tangent), np.asarray(direction).sum(0, keepdims=True))


@pytest.mark.parametrize("length", [2 ** 30, 2 ** 31 - 2, 2 ** 31 - 1], ids=["2^30", "2^31 - 2", "2^31 - 1"])
def test_an_aligned_sum_over_one_long_axis_lowers(length):
    """`aligned_sum([(x,)], axis=0)` over `length` fp32 terms, up to its cap of `2^31 - 1` products a value,
    lowered: one-bit digits whose one term's sum int32 holds, its quotient carried before its remainder
    adds and the carry in `[-n, n - 1]`, where they were refused (base reviews ee7d1e761 F2, 0c6ef9c11 F3,
    43f8e75e7 F3, row 8757)."""
    lowered = jax.jit(lambda x: aligned_sum([(x,)], axis=0)).trace(
        jax.ShapeDtypeStruct((length,), jnp.float32)).lower()
    assert lowered.out_info.shape == (1,)


def test_an_aligned_sums_tangent_leaks_no_tracer():
    """The tangent and the gradient of `aligned_sum([(a, b)])` under `jax.checking_leaks`: no tracer kept,
    where its tangent's map, a primitive's static parameter, closed over its traced value (row 8757)."""
    one = jnp.float32(1.0)
    with jax.checking_leaks():
        jax.jvp(lambda a: aligned_sum([(a, one)]), (one,), (one,))
        jax.grad(lambda a: aligned_sum([(a, one)]))(one)


@pytest.mark.parametrize("axis", [2, -3, (0, 0), (1, -1), 2 ** 31, (0, 2 ** 31)],
                         ids=["past the rank", "before it", "repeated", "repeated by its sign",
                              "past a C int", "one of them past a C int"])
def test_an_aligned_sum_refuses_an_axis_its_term_has_not(axis):
    """`aligned_sum([(x,)], axis)` of a `[2, 3]` term over an axis past its rank or repeated: a ValueError, as
    `jnp.sum`'s, where the axis wrapped by the rank reduced another (row 8837), and one past a C int's range,
    where numpy's conversion of it raised an OverflowError (row 9086)."""
    with pytest.raises(ValueError):
        aligned_sum([(jnp.ones((2, 3), jnp.float32),)], axis=axis)


def test_an_aligned_sum_over_an_empty_axis_is_positive_zero():
    """`aligned_sum` over an empty fp32 axis: positive zero, IEEE's sum of no terms, where a sum of none
    counted as of negative zeros alone, negative zero (base review 6c68b90ad F4)."""
    found = np.asarray(aligned_sum([(jnp.zeros((0,), jnp.float32),)], axis=0))
    np.testing.assert_array_equal(found, [0.0])
    assert not np.signbit(found).any()


@pytest.mark.parametrize(("terms", "total"), [
    ([(2.0 ** -75, 2.0 ** -75)] * 2, 2.0 ** -149),
    ([(2.0 ** -75, 2.0 ** -75)], 0.0),
    ([(2.0 ** -75, 2.0 ** -75)] * 3, 2.0 ** -148),
    ([(1 + 2.0 ** -12, 1 + 2.0 ** -12), (-(1 + 2.0 ** -11), 1.0)], 2.0 ** -24),
], ids=["two products a subnormal", "a product half the least subnormal", "three such halves",
        "a product's bit below its rounding"])
def test_an_aligned_sum_sums_its_products_every_bit(terms, total):
    """`aligned_sum` of two-factor fp32 terms whose products round in fp32 or are subnormal: their exact sum
    rounded once, `2^-150` twice the least subnormal, once a tie to zero, three times `2^-148`, and `(1 +
    2^-12)^2 - (1 + 2^-11)` its `2^-24`, where its products were rounded first and subnormals flushed (row
    8757)."""
    found = aligned_sum([tuple(jnp.float32(factor) for factor in term) for term in terms])
    np.testing.assert_array_equal(np.asarray(found), np.float32(total))


def test_an_aligned_sum_over_a_long_axis_is_its_exact_sum():
    """`aligned_sum([(x,)], axis=0)` of 65,536 ones but `2^100` and `-2^100` first: 65,534, its digits'
    sums within their integers however many terms an axis holds (row 8757)."""
    x = jnp.ones((65536, 2), jnp.float32).at[0].set(2.0 ** 100).at[1].set(-(2.0 ** 100))
    np.testing.assert_array_equal(np.asarray(aligned_sum([(x,)], axis=0)), [[65534.0, 65534.0]])


@pytest.mark.parametrize("terms", [[(0.0, 0.5, 0.5)], [(0.0, 0.0, 0.0), (0.0, 2.0 ** 100, 2.0 ** 100)]],
                         ids=["one zero term", "zero terms past fp32's range"])
def test_an_aligned_sum_of_zero_terms_is_zero(terms):
    """`aligned_sum` of terms whose products are all zero, one beside factors past fp32's range: zero, where
    its largest binade's sentinel aligned a zero term by 2^896 and wrapped it to -inf, its sum NaN (base
    review 820eaf429 F1)."""
    found = aligned_sum([tuple(jnp.float32(factor) for factor in term) for term in terms])
    assert float(found) == 0.0


def test_an_aligned_sum_of_float64_terms_keeps_float64():
    """`aligned_sum` of float64 terms sums in float64: an entry weight of `1 + 2^-30` written at unit key and
    value is itself, where cast to fp32 it was 1 (base review 820eaf429 F2)."""
    with jax.enable_x64(new_val=True):
        found = aligned_sum([(jnp.float64(1 + 2.0 ** -30), jnp.float64(1.0), jnp.float64(1.0))])
        assert found.dtype == jnp.float64 and float(found) == 1 + 2.0 ** -30


def test_an_aligned_sums_derivative_reduces_each_term_before_it_adds():
    """`aligned_sum` along an axis of terms of different shapes (`x` `[2, 1]` and `y` `[1, 1]`, unit
    values and directions): its tangent is 3, and `y`'s gradient 1, each term reduced over the axis before
    the terms add, as its value is; summed broadcast first, `y`'s direction counted on both rows gave 4, its
    gradient 2 (base review 675f2c056 F1)."""
    x, y = jnp.ones((2, 1), jnp.float32), jnp.ones((1, 1), jnp.float32)

    def summed(x, y):
        return aligned_sum([(x,), (y,)], axis=0)

    _, tangent = jax.jvp(summed, (x, y), (x, y))
    assert float(tangent[0, 0]) == 3.0
    assert float(jax.grad(lambda y: summed(x, y)[0, 0])(y)[0, 0]) == 1.0


def test_an_aligned_sums_derivatives_hold_a_cancelling_shared_factors_range():
    """`aligned_sum` of an entry write's two terms sharing their weight and key, a key of `(2^80, 0, 0)` and
    values of 2^80 and -2^80 (each term 2^160, their sum zero): its tangent along the shared factors and
    every factor's gradient float64's exactly, where its tangent summed each term's 2^160 plainly, inf - inf,
    and a shared factor's gradient took each term's apart, the same (row 8501)."""
    weight, key = np.ones((1, 1)), np.asarray([[2.0 ** 80], [0.0], [0.0]])
    up = np.full((1, 4), 2.0 ** 80)
    factors = (weight, key, up, -up)
    directions = (np.ones((1, 1)), np.ones((3, 1)), np.zeros((1, 4)), np.zeros((1, 4)))
    cotangent = np.arange(12.0).reshape(3, 4) - 5.0

    def summed(weight, key, up, down):
        return aligned_sum([(weight, key, up), (weight, key, down)])

    narrow = [tuple(jnp.asarray(x, jnp.float32) for x in group) for group in (factors, directions)]
    _, tangent = jax.jvp(summed, *narrow)
    gradients = jax.grad(lambda *given: jnp.sum(summed(*given) * jnp.asarray(cotangent, jnp.float32)),
                         argnums=(0, 1, 2, 3))(*narrow[0])
    # float64's: the tangent each factor's direction in its place, the gradients the cotangent times the
    # other factors summed over each one's broadcast, each exact here.
    weight_d, key_d, up_d, down_d = directions
    exact_tangent = (weight_d * key * (up - up) + weight * key_d * (up - up)
                     + weight * key * (up_d + down_d))
    exact = (np.sum(cotangent * key * (up - up), keepdims=True).reshape(1, 1),
             np.sum(cotangent * weight * (up - up), axis=1, keepdims=True),
             np.sum(cotangent * weight * key, axis=0, keepdims=True),
             np.sum(cotangent * weight * key, axis=0, keepdims=True))
    np.testing.assert_array_equal(np.asarray(tangent, np.float64), exact_tangent)
    for found, expected in zip(gradients, exact, strict=True):
        np.testing.assert_array_equal(np.asarray(found, np.float64), expected)


@pytest.mark.parametrize("smaller", [[1.0], [2.0 ** -100], [2.0 ** 25, -(2.0 ** 24), -(2.0 ** 24), -2.0],
                                     [2.0 ** 25, -(2.0 ** 24), -(2.0 ** 24), 2.0 ** -100]],
                         ids=["a unit", "2^-100", "a tie", "a band below two that cancel"])
def test_an_aligned_sums_derivatives_keep_smaller_terms_below_cancelling_ones(smaller):
    """`aligned_sum([(w, a), (w, -a), *((w, s) for s in smaller)])` of `a` 2^127 at `w` zero along one: its
    tangent and `w`'s gradient the sum of `smaller`, and at `w` one its value, float64's, where aligned to the
    largest term, a unit or 2^-100 more than float32's exponent range below it was dropped before the largest
    cancelled, zero (base review 224fcf53a F1), where each band rounded its sum alone, -2^25 - 2 tied to
    -2^25 beside 2^25, zero (base review 644ef5535 F1), and where the bands' totals were aligned at once,
    2^-100 dropped before 2^25 and -2^25 cancelled (base review f5e6e678d F1)."""
    terms = [jnp.float32(2.0 ** 127), jnp.float32(-(2.0 ** 127)), *map(jnp.float32, smaller)]
    small = sum(smaller)

    def summed(w):
        return aligned_sum([(w, term) for term in terms])

    _, tangent = jax.jvp(summed, (jnp.float32(0.0),), (jnp.float32(1.0),))
    assert float(tangent) == small
    assert float(jax.grad(summed)(jnp.float32(0.0))) == small
    assert float(summed(jnp.float32(1.0))) == small


def test_an_aligned_sum_reduces_a_term_before_its_value_broadcasts_it():
    """`aligned_sum([(x,), (y,)], axis=0)` of `x` `[1, 2]` and `y` `[[10, 20]]`: `x` summed to one value
    broadcast across the value's two, `[[13, 23]]`, its tangent along `x`'s direction `[1, 2]` 3 at each, and
    under a cotangent of `[[1, 2]]` `x`'s gradient 3 at each element and `y`'s the cotangent, where the
    transpose laid `x` against the value's axis and gave it `[1, 2]` (base review f5e6e678d F2), and the
    value and tangent reduced it against a top broadcast to the value's shape, `[[11, 22]]` and `[[1, 2]]`
    (base review db30f3e80 F2)."""
    x, y = jnp.asarray([1.0, 2.0], jnp.float32), jnp.asarray([[10.0, 20.0]], jnp.float32)

    def summed(x, y):
        return aligned_sum([(x,), (y,)], axis=0)

    value, pulled = jax.vjp(summed, x, y)
    np.testing.assert_array_equal(np.asarray(value), [[13.0, 23.0]])
    tangent = jax.jvp(summed, (x, y), (jnp.asarray([1.0, 2.0], jnp.float32), jnp.zeros_like(y)))[1]
    np.testing.assert_array_equal(np.asarray(tangent), [[3.0, 3.0]])
    dx, dy = pulled(jnp.asarray([[1.0, 2.0]], jnp.float32))
    np.testing.assert_array_equal(np.asarray(dx), [3.0, 3.0])
    np.testing.assert_array_equal(np.asarray(dy), [[1.0, 2.0]])


def test_an_aligned_sums_shared_factors_gradient_holds_its_groups_scaled_until_they_sum():
    """`aligned_sum([(f, x), (f, y)], axis=1)` of `f` ones `[2, 3]` beside `x` `[2, 3]` of 2^125 and `y`
    `[4, 2, 3]` of -1.25 2^125, each reduced over its own second axis (`f`'s second in the first, its first
    in the second), under a unit cotangent: every element of `f`'s gradient 2^126, where its two terms'
    gradients, laid apart, were scaled back apart, 3 2^127 and -2.5 2^127 opposite infinities, and summed to
    zero (base review db30f3e80 F1)."""
    f = jnp.ones((2, 3), jnp.float32)
    x = jnp.full((2, 3), 2.0 ** 125, jnp.float32)
    y = jnp.full((4, 2, 3), -1.25 * 2.0 ** 125, jnp.float32)
    value, pulled = jax.vjp(lambda f: aligned_sum([(f, x), (f, y)], axis=1), f)
    np.testing.assert_array_equal(np.asarray(pulled(jnp.ones_like(value))[0]), np.full((2, 3), 2.0 ** 126))


def _plain_sum(terms, axis=None):
    """`aligned_sum`'s value in plain float32 arithmetic: each term's product, reduced over `axis` where one
    is given, the terms added."""
    products = [functools.reduce(jnp.multiply, term) for term in terms]
    return functools.reduce(jnp.add, [p if axis is None else jnp.sum(p, axis=axis) for p in products])


@pytest.mark.parametrize("order", ["forward over reverse", "reverse over reverse"])
def test_an_aligned_sums_second_order_keeps_its_losss_curvature(order):
    """The Hessian along `d` of `sum(aligned_sum([(a, a, c)])^2)`, its loss nonlinear in the sum (`12 a^2 c^2
    d`), and of a mixed-rank sum's `sum(aligned_sum([(x, y), (y,)], axis=0)^2)` its cross curvature along
    `d` in `x`, the `y`-gradient's (`4 y (x + 1) d`), each by `jax.jvp` of a gradient and by the gradient of
    a gradient's product with the direction, on par with the same losses' in plain float32 arithmetic
    against float64 (check 1), over normal draws, where the sum's tangent rule formed its value from bits,
    which carry no tangent, and both orders lost the loss's curvature (at `a` `[1, 2]`, `c` `[3, 4]`, `d`
    `[1, 0]`: `[36, 0]` and `[0, 0]` for `[108, 0]`; base review d2d45bd6f F2, row 8701)."""
    def draw(seed):
        rng = np.random.default_rng([2026_1120, seed])
        return tuple(jnp.asarray(rng.standard_normal(shape), jnp.float32)
                     for shape in ((4096,), (4096,), (4096,), (1, 4096)))

    def curvatures(summed):
        def run(a, c, direction, y):
            def loss(a):
                return jnp.sum(summed([(a, a, c)]) ** 2)

            def mixed(x, y):
                return jnp.sum(summed([(x, y), (y,)], axis=0) ** 2)

            if order == "forward over reverse":
                curvature = jax.jvp(jax.grad(loss), (a,), (direction,))[1]
                crossed = jax.jvp(lambda x: jax.grad(mixed, argnums=1)(x, y), (a,), (direction,))[1]
            else:
                curvature = jax.grad(lambda a: jnp.vdot(jax.grad(loss)(a), direction))(a)
                crossed = jax.grad(lambda y: jnp.vdot(jax.grad(mixed)(a, y), direction))(y)
            return {"curvature": curvature, "mixed": crossed}

        return run

    assert_as_exact(jax.jit(curvatures(aligned_sum)), jax.jit(curvatures(_plain_sum)), draw,
                    f"an aligned sum's second order, {order}", seeds=8)


def test_an_aligned_sum_over_an_axis_a_mesh_splits_sums_it_shard_by_shard():
    """`aligned_sum([(x,)], axis=0)` of ones `[6, 2]` split over an Explicit two-device data axis: `[[6, 6]]`,
    and its gradient ones placed as `x` is, where its axis summed in halves sliced the split axis at three,
    which two shards do not divide (base review 9e56c760b F1); and of a whole `x` of ones beside a split `y`
    of twos, `x`'s gradient twos placed whole as `x` is, where it kept `y`'s split and JAX's accumulator
    refused it (base review 8d59803fd F1)."""
    mesh = jax.make_mesh((2,), ("data",), (jax.sharding.AxisType.Explicit,), devices=jax.devices()[:2])
    with jax.set_mesh(mesh):
        rows = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
        whole = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        x = jax.device_put(jnp.ones((6, 2), jnp.float32), rows)
        value = jax.jit(lambda x: aligned_sum([(x,)], axis=0))(x)
        gradient = jax.jit(jax.grad(lambda x: jnp.sum(aligned_sum([(x,)], axis=0))))(x)
        x = jax.device_put(jnp.ones((6, 2), jnp.float32), whole)
        y = jax.device_put(jnp.full((6, 2), 2.0, jnp.float32), rows)
        beside = jax.jit(jax.grad(lambda x, y: jnp.sum(aligned_sum([(x, y)], axis=0))))(x, y)
    np.testing.assert_array_equal(np.asarray(value), [[6.0, 6.0]])
    np.testing.assert_array_equal(np.asarray(gradient), np.ones((6, 2)))
    assert gradient.sharding.is_equivalent_to(rows, gradient.ndim)
    np.testing.assert_array_equal(np.asarray(beside), np.full((6, 2), 2.0))
    assert beside.sharding.is_equivalent_to(whole, beside.ndim)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis_type, ambient", [(jax.sharding.AxisType.Explicit, True),
                                                (jax.sharding.AxisType.Auto, True),
                                                (jax.sharding.AxisType.Auto, False)],
                         ids=["explicit", "auto", "auto-placed-alone"])
def test_an_elementwise_aligned_sum_split_over_a_mesh_is_computed_shard_by_shard(axis_type, ambient):
    """`aligned_sum([(x, y), (z,)])` elementwise of `[8, 4]` operands split over a two-device data axis,
    Explicit or Auto (the Auto mesh set around the trace, or only placing the operands), is each element's
    exact sum (`3 + 2^-30` rounded, 3, everywhere) and its compiled program holds no collective
    (`dew.telemetry.trace.collective`): each shard's sums are its own, where the digits' walk bounded itself
    by the extrema of every limb across the shards, an all-reduce of independent sums (row 8841)."""
    from dew.telemetry.trace import collective

    mesh = jax.make_mesh((2,), ("data",), (axis_type,), devices=jax.devices()[:2])
    rows = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
    with jax.set_mesh(mesh) if ambient else contextlib.nullcontext():
        x, y, z = (jax.device_put(jnp.full((8, 4), value, jnp.float32), rows)
                   for value in (2.0 ** -15, 2.0 ** -15, 3.0))
        summed = jax.jit(lambda x, y, z: aligned_sum([(x, y), (z,)]), out_shardings=rows)
        compiled = summed.lower(x, y, z).compile()
        value = compiled(x, y, z)
    instructions = hlo_instructions(compiled).values()
    assert instructions
    assert not [each.op for each in instructions if collective(each.op) is not None]
    np.testing.assert_array_equal(np.asarray(value), np.full((8, 4), 3.0, np.float32))
    assert value.sharding.is_equivalent_to(rows, value.ndim)


def test_an_aligned_sum_over_an_empty_axis_is_zero():
    """`aligned_sum([(x, y)])` of a scalar `x` beside an empty `y`: an empty value whose pullback gives `x` a
    zero cotangent, and a sum over an empty axis zero, where the largest binade over no element raised
    (base review 9e56c760b F2)."""
    x, y = jnp.float32(2.0), jnp.zeros((0,), jnp.float32)
    value, pulled = jax.vjp(lambda x, y: aligned_sum([(x, y)]), x, y)
    assert value.shape == (0,) and float(pulled(jnp.zeros_like(value))[0]) == 0.0
    np.testing.assert_array_equal(np.asarray(aligned_sum([(jnp.zeros((0, 3), jnp.float32),)], axis=0)),
                                  np.zeros((1, 3)))


def test_an_aligned_sums_gradient_scales_a_cotangent_before_its_broadcast_sum():
    """`aligned_sum([(x, y), (z,)])` of one-element `x` 2^-20 and `y` 2^-100 beside `z` of two, under a
    cotangent of 1.5 2^127 on each: `x`'s gradient 3 2^27 and `y`'s 3 2^107, float64's, where the cotangent's
    sum across the broadcast came before its products, 3 2^127 past float32's largest (base review d6c1b2c1a
    F1)."""
    x, y, z = (jnp.asarray(v, jnp.float32) for v in ([2.0 ** -20], [2.0 ** -100], [0.0, 0.0]))
    cotangent = jnp.full((2,), 1.5 * 2.0 ** 127, jnp.float32)
    def loss(x, y, z):
        return jnp.sum(aligned_sum([(x, y), (z,)]) * cotangent)

    gradients = jax.grad(loss, argnums=(0, 1))(x, y, z)
    np.testing.assert_array_equal(np.asarray(gradients[0], np.float64), [3.0 * 2.0 ** 27])
    np.testing.assert_array_equal(np.asarray(gradients[1], np.float64), [3.0 * 2.0 ** 107])


def test_an_aligned_sum_of_a_subnormal_factor_raises_it_by_its_bits():
    """`aligned_sum` of a term whose factor is fp32-subnormal (`2^-140`) and whose product is in range
    (`2^-140 2^100`): `2^-40` exactly, the factor normalized by its bits, where lifted by a multiply XLA:CPU
    flushes it to zero (base review 675f2c056 F2)."""
    found = aligned_sum([(jnp.float32(2.0 ** -140), jnp.float32(2.0 ** 100), jnp.float32(1.0))])
    assert float(found) == 2.0 ** -40


def test_an_aligned_sum_along_an_axis_sums_each_rows_terms():
    """`aligned_sum` over an axis: each row's two columns `(1 + 2^-20) 2^(140 + i)` and `-2^(140 + i)`, each
    past fp32's range, summed along the row to `2^(120 + i)` exactly."""
    rows = np.arange(6)
    first = np.stack([np.full(6, 1 + 2.0 ** -20), np.full(6, -1.0)], axis=1).astype(np.float32)
    second = np.broadcast_to(np.float32(2.0 ** 70), (6, 2))
    third = np.broadcast_to((2.0 ** (70 + rows)).astype(np.float32)[:, None], (6, 2))
    found = aligned_sum([(jnp.asarray(first), jnp.asarray(second), jnp.asarray(third))], axis=1)
    np.testing.assert_array_equal(np.asarray(found)[:, 0], 2.0 ** (120 + rows))


def test_an_aligned_segment_sum_sums_each_segments_rows_exactly():
    """`aligned_segment_sum` sums each row's product into the segment its table names, exactly, rounded
    once: 2**240 and -2**240 beside 2**-28 in one segment give 2**-28; 2**128 (as 2**127 times 2) and
    -2**105 give their finite difference, not inf; 256 and -255.5 times 2**-8 give 2**-9; a segment no row
    names gives +0, one of -0 products alone -0 (row 1464)."""
    big = jnp.float32(2.0 ** 120)
    values = jnp.asarray([[1.0], [-1.0], [2.0 ** -28], [2.0], [-(2.0 ** -22)], [256.0], [-255.5]],
                         jnp.float32)
    scales = jnp.asarray([[big], [big], [1.0], [2.0 ** 127], [2.0 ** 127], [2.0 ** -8], [2.0 ** -8]],
                         jnp.float32)
    lifts = jnp.asarray([[2.0 ** 120], [2.0 ** 120], [1.0], [1.0], [1.0], [1.0], [1.0]], jnp.float32)
    table = jnp.asarray([0, 0, 0, 1, 1, 2, 2], jnp.int32)
    got = jax.jit(lambda: aligned_segment_sum([(values, scales, lifts)], table, 4))()
    want = np.asarray([[2.0 ** -28], [np.float32(2.0 ** 128 - 2.0 ** 105)], [2.0 ** -9], [0.0]], np.float32)
    np.testing.assert_array_equal(np.asarray(got), want)
    # The segment no row names sums to +0, an empty sum's; one holding only -0 products to -0.
    assert np.copysign(1.0, np.asarray(got)[3, 0]) == 1.0
    negative = aligned_segment_sum([(jnp.asarray([[-0.0]], jnp.float32),)], jnp.asarray([0], jnp.int32), 2)
    assert [np.copysign(1.0, value) for value in np.asarray(negative)[:, 0]] == [-1.0, 1.0]


@pytest.mark.mesh(devices=2)
def test_an_aligned_segment_sum_sums_its_segments_over_a_checked_maps_devices():
    """`aligned_segment_sum` in a `shard_map` over two CPU devices that checks the axes its values vary
    over, each device its own rows and table: a segment's +2**140 on one device and -2**140 on the other sum
    to 0 beside the first device's 2**-38, every device walking the same digits (row 1464)."""
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    data, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()
    values = jnp.asarray([[2.0 ** 70], [2.0 ** -38], [-(2.0 ** 70)], [0.0]], jnp.float32)
    scales = jnp.asarray([[2.0 ** 70], [1.0], [2.0 ** 70], [1.0]], jnp.float32)
    tables = jnp.zeros((4,), jnp.int32)

    def each_device(v, s, t):
        return aligned_segment_sum([(v, s)], t, 1, ("data",))

    got = jax.jit(jax.shard_map(each_device, mesh=mesh, in_specs=(data, data, data), out_specs=whole,
                                check_vma=True))(values, scales, tables)
    np.testing.assert_array_equal(np.asarray(got), np.asarray([[2.0 ** -38]], np.float32))


def checked_total(a, b, w):
    """A checked `shard_map`'s summed `aligned_sum([(a, b)])` times `w` over two devices, `a` held whole
    beside `b` and `w` split between them."""
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    split, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()

    def local(a, b, w):
        return jax.lax.psum(jnp.sum(aligned_sum([(a, b)]) * w), "data")

    return jax.shard_map(local, mesh=mesh, in_specs=(whole, split, split), out_specs=whole,
                         check_vma=True)(a, b, w)


@pytest.mark.mesh(devices=2)
def test_a_replicated_factors_cotangent_pools_its_digits_over_the_axis_before_it_rounds():
    """The gradient of a checked `shard_map`'s summed `aligned_sum([(a, b)])` times `w` by a replicated `a`
    beside `b` and `w` split over two devices (`checked_total`), `(1 - 2^-23)` and `1` on the first, `-1` and
    `1` on the second (`w` `1 + 2^-23` and `1`): exactly `-2^-46`, the cotangent's terms' digits pooled over
    the axis before the one rounding, where each device rounded first (`1` and `-1`) summed to 0 (train 1461's
    review)."""
    b = jnp.asarray([1 - 2.0 ** -23, -1.0], jnp.float32)
    w = jnp.asarray([1 + 2.0 ** -23, 1.0], jnp.float32)
    found = jax.jit(jax.grad(lambda a: checked_total(a, b, w)))(jnp.ones((1,), jnp.float32))
    np.testing.assert_array_equal(np.asarray(found), np.asarray([-(2.0 ** -46)], np.float32))


@pytest.mark.mesh(devices=2)
def test_a_split_factors_cotangent_in_a_checked_shard_map_is_the_unsplit_calls():
    """The gradient of a checked `shard_map`'s summed `aligned_sum([(a, b)])` times `w` by `b`, split over two
    devices as `w` is beside a replicated `a` (`checked_total`): the unsplit call's bit for bit, no axis in
    excess, its terms summed as they were."""
    a = jnp.asarray([3.0], jnp.float32)
    w = jnp.asarray([1 + 2.0 ** -23, -(2.0 ** 100)], jnp.float32)
    b = jnp.asarray([1 - 2.0 ** -23, -1.0], jnp.float32)
    found = jax.jit(jax.grad(lambda b: checked_total(a, b, w)))(b)
    unsplit = jax.jit(jax.grad(lambda b: jnp.sum(aligned_sum([(a, b)]) * w)))(b)
    np.testing.assert_array_equal(np.asarray(found), np.asarray(unsplit))


@pytest.mark.mesh(devices=2)
def test_a_replicated_factors_pooled_cotangent_is_differentiated_as_it_is_pooled():
    """A replicated `a`'s cotangent (`aligned_cotangents`) of `aligned_sum([(a, b)])` at `w`, `b` and `w`
    split over two devices in a checked `shard_map`, pooled over the axis (train 1461's review), and its
    tangent along `b`: `[1]` beside `b = w = [1, 1]` gives `[2]`, and along `[1, 1]` `[2]`, its pooled sum
    differentiated by its declared rule (train 1470's review: the pooled sum taken bare had no tangent, 0)."""
    from dew.nn.aligned import aligned_cotangents

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    split, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()

    def local(a, b, w, along):
        return jax.jvp(lambda b: aligned_cotangents([(a, b)], w)[0], (b,), (along,))

    run = jax.shard_map(local, mesh=mesh, in_specs=(whole, split, split, split), out_specs=(whole, whole),
                        check_vma=True)
    ones = jnp.ones((2,), jnp.float32)
    value, tangent = jax.jit(run)(jnp.ones((1,), jnp.float32), ones, ones, ones)
    np.testing.assert_array_equal(np.asarray(value), np.asarray([2.0], np.float32))
    np.testing.assert_array_equal(np.asarray(tangent), np.asarray([2.0], np.float32))


@pytest.mark.mesh(devices=2)
def test_a_pooled_cotangents_pullback_to_a_split_factor_beside_a_replicated_weight_is_split():
    """The pullback along a split `b` of a replicated `a`'s pooled cotangent (`aligned_cotangents`) of
    `aligned_sum([(a, b)])` at a replicated `w = [1]`, `b = [1, 1]` split over two devices in a checked
    `shard_map`: the value `[2]`, and `b`'s cotangent of `[1]` each shard's own, `[1, 1]` split, as the
    pooled sum's transpose gives a factor varying where its terms do not (train 1470's review: an invariant
    cotangent handed to a varying accumulator was refused)."""
    from dew.nn.aligned import aligned_cotangents

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    split, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()

    def local(a, b, w):
        value, pullback = jax.vjp(lambda b: aligned_cotangents([(a, b)], w)[0], b)
        return value, pullback(jnp.ones_like(value))[0]

    run = jax.shard_map(local, mesh=mesh, in_specs=(whole, split, whole), out_specs=(whole, split),
                        check_vma=True)
    one = jnp.ones((1,), jnp.float32)
    value, pulled = jax.jit(run)(one, jnp.ones((2,), jnp.float32), one)
    np.testing.assert_array_equal(np.asarray(value), np.asarray([2.0], np.float32))
    np.testing.assert_array_equal(np.asarray(pulled), np.asarray([1.0, 1.0], np.float32))


@pytest.mark.mesh(devices=2)
def test_a_split_factors_varied_cotangent_pulls_back_to_a_replicated_weight_exactly():
    """Reverse over reverse through a split `b`'s cotangent (`aligned_cotangents`) of `aligned_sum([(a, b)],
    0)` at a replicated `w`, in a checked `shard_map` over two devices, its terms `w a` replicated and varied
    to `b`'s placement: pulled back to `w` along `[2^127, 2^127]` on one device and `[-2^127, -2^127]` on the
    other, exactly 0, its transpose pooling the products' digits over the axis before its one rounding
    (train 1473's review: a cast to vary outside the declared rule transposed to a float sum, inf - inf)."""
    from dew.nn.aligned import aligned_cotangents

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    split, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()

    def local(a, b, w, along):
        _, pullback = jax.vjp(lambda w: aligned_cotangents([(a, b)], w, 0)[1], w)
        return pullback(along)[0]

    run = jax.shard_map(local, mesh=mesh, in_specs=(whole, split, whole, split), out_specs=whole,
                        check_vma=True)
    along = jnp.asarray([2.0**127, 2.0**127, -(2.0**127), -(2.0**127)], jnp.float32)
    found = jax.jit(run)(jnp.ones((2,), jnp.float32), jnp.ones((4,), jnp.float32), jnp.ones((), jnp.float32),
                         along)
    np.testing.assert_array_equal(np.asarray(found), np.asarray(0.0, np.float32))
