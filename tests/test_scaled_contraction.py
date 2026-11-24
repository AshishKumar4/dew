"""The scaled contraction (`scaled_contraction_p`): a sum of contractions of products and quotients formed
apart from their binades, its every order the primitive again."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.kernels.templates.scaled_contraction import Form, contract

QUOTIENT = Form(("", ""), ((1, (((0, 1), (1, -1)),)),), "")
"""`x / y` of two scalars."""


def test_a_quotients_product_rule_lands_once():
    """`x / y` at `x = y = 2^-100` moved by `2^100` each: the quotient 1, its tangent 0, where each product
    rule term landed apart was `+inf` and `-inf` (NaN)."""
    tiny = jnp.float32(2.0 ** -100)
    out, tangent = jax.jvp(lambda x, y: contract(QUOTIENT, x, y), (tiny, tiny),
                           (jnp.float32(2.0 ** 100), jnp.float32(2.0 ** 100)))
    np.testing.assert_array_equal(np.asarray(out), 1.0)
    np.testing.assert_array_equal(np.asarray(tangent), 0.0)


def test_a_subnormal_quotient_keeps_its_bits():
    """`2^-127 / 2^-127` is 1, its operands read from their bits, where `frexp` on a process flushing
    subnormals read `0 / 0`."""
    sub = jnp.float32(2.0 ** -127)
    np.testing.assert_array_equal(np.asarray(contract(QUOTIENT, sub, sub)), 1.0)


def test_a_sums_gradient_reaches_the_letters_it_reduces():
    """The gradient of a sum over `i` (`i ->`) is ones along `i`: the transpose takes the letter no other
    operand reads by a unit operand, at first and second order."""
    form = Form(("i",), ((1, (((0, 1),),)),), "")
    x = jnp.arange(1.0, 4.0, dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(jax.grad(lambda v: contract(form, v))(x)), 1.0)
    hessian = jax.jacfwd(jax.grad(lambda v: contract(form, v) ** 2))(x)
    np.testing.assert_array_equal(np.asarray(hessian), 2.0)


def test_a_subscript_naming_a_letter_twice_is_refused():
    """`ii ->`, a trace, is no form of the sum's: refused by name."""
    with pytest.raises(ValueError):
        contract(Form(("ii",), ((1, (((0, 1),),)),), ""), jnp.ones((2, 2), jnp.float32))


def test_a_batched_contraction_is_each_examples():
    """`jax.vmap` of a matrix product's sum over quotients is each example's, a fresh letter on the batch."""
    form = Form(("ik", "i", "kj"), ((1, (((0, 1), (1, -1)), ((2, 1),))),), "ij")
    rng = np.random.default_rng(12388)
    a, d, b = (jnp.asarray(rng.normal(size=shape) + 3.0, jnp.float32)
               for shape in ((3, 2, 4), (3, 2), (3, 4, 5)))
    batched = jax.vmap(lambda *x: contract(form, *x))(a, d, b)
    for example in range(3):
        np.testing.assert_array_equal(np.asarray(batched[example]),
                                      np.asarray(contract(form, a[example], d[example], b[example])))


def test_a_quotients_second_derivatives_are_its_own():
    """`sum_k a_k b_k / d` differentiated twice, reverse over reverse and forward over reverse, is the plain
    formula's, exactly at operands whose products and quotients are exact (its every order the contraction
    again)."""
    form = Form(("k", "", "k"), ((1, (((0, 1), (1, -1)), ((2, 1),))),), "")
    a = jnp.asarray([1.0, 2.0, 3.0], jnp.float32)
    b = jnp.asarray([0.5, -1.0, 2.0], jnp.float32)
    d = jnp.float32(4.0)

    def ours(x, y):
        return contract(form, x, y, b)

    def plain(x, y):
        return jnp.sum(x * b) / y

    for twice in (jax.hessian, jax.jacfwd):
        found = twice(ours, argnums=(0, 1))(a, d) if twice is jax.hessian else jax.jacfwd(
            jax.grad(ours, argnums=(0, 1)), argnums=(0, 1))(a, d)
        truth = twice(plain, argnums=(0, 1))(a, d) if twice is jax.hessian else jax.jacfwd(
            jax.grad(plain, argnums=(0, 1)), argnums=(0, 1))(a, d)
        for got, want in zip(jax.tree.leaves(found), jax.tree.leaves(truth), strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_a_batched_sum_shares_an_unbatched_terms_operand():
    """`x + y` (two terms) batched along `x` alone is each example's `x` plus the shared `y`."""
    form = Form(("i", "i"), ((1, (((0, 1),),)), (1, (((1, 1),),))), "i")
    x = jnp.arange(6.0, dtype=jnp.float32).reshape(3, 2)
    y = jnp.asarray([0.5, -2.0], jnp.float32)
    found = jax.vmap(lambda each: contract(form, each, y))(x)
    np.testing.assert_array_equal(np.asarray(found), np.asarray(x + y))


MEAN = Form(("k", "kd"), ((1, (((0, 1),), ((1, 1),))),), "d", divisor=((1, (((0, 1),),)),), divided="")
"""`sum_k m_k v_kd / sum_k m_k`: a mean whose divisor is its own operand's sum."""


def test_a_means_gradient_lands_its_parts_in_one_sum():
    """The gradient of a mean's channels' sum along its masses, values near fp32's largest over eight
    channels: each mass's part through the values (`8 v / T`, 2^129 and 1.5 x 2^129) and through the total
    (`-8 mean / T`, 1.25 x 2^129) cancel within one sum to `8 (v_k - mean) / T`, `-+2^127`, where formed
    apart they met as `inf - inf`."""
    masses = jnp.asarray([1.0, 1.0], jnp.float32)
    values = jnp.stack([jnp.full(8, 2.0 ** 127), jnp.full(8, 1.5 * 2.0 ** 127)]).astype(jnp.float32)
    found = jax.grad(lambda m: jnp.sum(contract(MEAN, m, values)))(masses)
    np.testing.assert_array_equal(np.asarray(found), np.asarray([-(2.0 ** 127), 2.0 ** 127], np.float32))


def test_a_means_value_and_second_derivatives_are_its_own():
    """A mean with its own divisor is `sum_k m v / sum_k m`, and its Hessian along its masses, reverse over
    reverse (`jacrev` of `grad`) and forward over reverse, the plain formula's at operands whose arithmetic is
    exact."""
    masses = jnp.asarray([1.0, 3.0], jnp.float32)
    values = jnp.asarray([[2.0, -4.0], [6.0, 8.0]], jnp.float32)

    def plain(m):
        return jnp.sum(m[:, None] * values, axis=0) / jnp.sum(m)

    np.testing.assert_array_equal(np.asarray(contract(MEAN, masses, values)), np.asarray(plain(masses)))
    for found, want in ((jax.jacrev(jax.grad(lambda m: contract(MEAN, m, values)[0]))(masses),
                         jax.jacrev(jax.grad(lambda m: plain(m)[0]))(masses)),
                        (jax.jacfwd(jax.grad(lambda m: contract(MEAN, m, values)[1]))(masses),
                         jax.jacfwd(jax.grad(lambda m: plain(m)[1]))(masses))):
        np.testing.assert_array_equal(np.asarray(found), np.asarray(want))


DIVIDED = Form(("", ""), ((1, (((0, 1),),)),), "", divisor=((1, (((1, 1),),)),), divided="")
"""`x / y`, `y` the divisor's own sum."""


def test_a_zero_divisor_read_as_one_does_not_move():
    """`x / y` at `y = 0` is `x`, the zero divisor read as the constant one: moving `y` moves nothing, where
    its divisor's moved term `-x dy / 1` gave -2."""
    out, tangent = jax.jvp(lambda y: contract(DIVIDED, jnp.float32(2.0), y), (jnp.float32(0.0),),
                           (jnp.float32(1.0),))
    np.testing.assert_array_equal(np.asarray(out), 2.0)
    np.testing.assert_array_equal(np.asarray(tangent), 0.0)


def test_a_divisor_past_the_format_divides_its_cotangent_finite():
    """`x / (y + z)` at `x = y = z = 2^127` is 1/2, and its cotangent 2^127 pulls back to `dx = 1/2`: the
    transpose divides by the divisor carried apart from its binades, where the landed `y + z` was infinite
    and gave 0."""
    form = Form(("", "", ""), ((1, (((0, 1),),)),), "", divisor=((1, (((1, 1),),)), (1, (((2, 1),),))),
                divided="")
    big = jnp.float32(2.0 ** 127)
    out, pull = jax.vjp(lambda x: contract(form, x, big, big), big)
    np.testing.assert_array_equal(np.asarray(out), 0.5)
    np.testing.assert_array_equal(np.asarray(pull(big)[0]), 0.5)


BATCHED_MEAN = Form(("bk", "bkd"), ((1, (((0, 1),), ((1, 1),))),), "bd", divisor=((1, (((0, 1),),)),),
                   divided="b")
"""`MEAN` a batch row apiece."""

SPREAD = [[2.0 ** 127], [-(2.0 ** 127)], [2.0 ** -30], [0.0]]
"""Values that cancel past fp32's range beside a part far beneath them (row 12452's witness)."""


def test_a_part_beneath_values_that_cancel_survives_them_at_every_order():
    """A mean of four unit masses over values `[2^127, -2^127, 2^-30, 0]` is `2^-32`: the values contracted a
    band at a time into exponent bins (`scaled_contraction._banded`), where one exponent for the group
    (`2^127`) flushed `2^-30` and gave 0 (row 12452). So is the tangent of a mean of values all `2^127`
    along that direction (the reference's value-only tangent, the row's witness); the gradient along the
    masses is `(v_k - mean) / 4`, and its tangent along the values the same of the direction, at second
    order; each batch row's under `vmap` as alone."""
    masses = jnp.ones(4, jnp.float32)
    spread = jnp.asarray(SPREAD, jnp.float32)
    large = jnp.full((4, 1), 2.0 ** 127, jnp.float32)
    np.testing.assert_array_equal(np.asarray(contract(MEAN, masses, spread)), [2.0 ** -32])
    _, tangent = jax.jvp(lambda v: contract(MEAN, masses, v), (large,), (spread,))
    np.testing.assert_array_equal(np.asarray(tangent), [2.0 ** -32])
    pulled = np.asarray([2.0 ** 125, -(2.0 ** 125), 3 * 2.0 ** -34, -(2.0 ** -34)], np.float32)
    found = jax.grad(lambda m: contract(MEAN, m, spread)[0])(masses)
    np.testing.assert_array_equal(np.asarray(found), pulled)
    _, second = jax.jvp(lambda v: jax.grad(lambda m: contract(MEAN, m, v)[0])(masses), (large,), (spread,))
    np.testing.assert_array_equal(np.asarray(second), pulled)
    rows = jax.vmap(lambda v: contract(MEAN, masses, v))(jnp.stack([spread, large]))
    np.testing.assert_array_equal(np.asarray(rows), [[2.0 ** -32], [2.0 ** 127]])


def test_band_pairs_join_from_the_largest_whatever_their_order():
    """A mean whose products are `+2^121`, `2^-54` and `-2^121` (masses `[2^-6, 2^-126, 2^50]`, values
    `[2^127, 2^72, -2^71]`) is `2^-104`: the band pairs come in mixed-radix order, `2^-54`'s between the two
    that cancel, and each lands in its own exponent bin, the bins joined from the largest, where joined as
    they came `2^-54` was flushed beside `2^121` and the mean was 0 (review of #3967, P0)."""
    masses = jnp.asarray([2.0 ** -6, 2.0 ** -126, 2.0 ** 50], jnp.float32)
    values = jnp.asarray([[2.0 ** 127], [2.0 ** 72], [-(2.0 ** 71)]], jnp.float32)
    np.testing.assert_array_equal(np.asarray(contract(MEAN, masses, values)), [2.0 ** -104])


def test_a_non_finite_row_leaves_the_others_banded():
    """Beside a row whose value is infinite (its mean IEEE's, `inf`), a row whose values cancel past the
    format keeps its `2^-32`: the banded sum reads non-finite values as none and the IEEE result is laid over
    its own outputs alone, where one infinite value anywhere took every row unbanded (review of #3967, P2)."""
    masses = jnp.ones((2, 4), jnp.float32)
    values = jnp.asarray([SPREAD, [[jnp.inf], [1.0], [1.0], [1.0]]], jnp.float32)
    np.testing.assert_array_equal(np.asarray(contract(BATCHED_MEAN, masses, values)),
                                  [[2.0 ** -32], [np.inf]])


def test_a_term_of_more_band_pairs_than_int32_counts_is_summed_unbanded():
    """A product of nine groups, each over values a hundred binades apart, sums to `1 + 2^-900`, 1: its band
    pairs past what int32 counts (`_band` 0), the term is summed unbanded, where its pairs' count wrapped
    negative and the sum stopped before its first (review of #3967, P1)."""
    form = Form(("k",) * 9, ((1, tuple(((index, 1),) for index in range(9))),), "")
    operand = jnp.asarray([1.0, 2.0 ** -100], jnp.float32)
    np.testing.assert_array_equal(np.asarray(contract(form, *(operand,) * 9)), 1.0)


def test_a_banded_sum_holds_its_rows_varying_types_in_a_shard_map():
    """Batch rows split over two devices in a `shard_map` with its varying types checked: each row's banded
    mean its own (`2^-32` for the values that cancel, `2^127` for values all `2^127`), the band loops'
    carries typed as the rows they sum (review of #3967, P1)."""
    from jax.sharding import PartitionSpec

    mesh = jax.make_mesh((2,), ("data",), (jax.sharding.AxisType.Explicit,), devices=jax.devices()[:2])
    masses = jnp.ones((2, 4), jnp.float32)
    values = jnp.asarray([SPREAD, [[2.0 ** 127]] * 4], jnp.float32)
    rows = PartitionSpec("data")
    found = jax.jit(jax.shard_map(lambda m, v: contract(BATCHED_MEAN, m, v), mesh=mesh, in_specs=(rows, rows),
                                  out_specs=rows, check_vma=True))(masses, values)
    np.testing.assert_array_equal(np.asarray(found), [[2.0 ** -32], [2.0 ** 127]])


def test_terms_that_cancel_keep_the_term_beneath_them():
    """`x - y + z` at `x = y = 2^127`, `z = 2^-30` is `2^-30`: the terms joined as double floats at their own
    powers, where joined at the largest's they flushed `z` and gave 0."""
    form = Form(("", "", ""), ((1, (((0, 1),),)), (-1, (((1, 1),),)), (1, (((2, 1),),))), "")
    big = jnp.float32(2.0 ** 127)
    np.testing.assert_array_equal(np.asarray(contract(form, big, big, jnp.float32(2.0 ** -30))), 2.0 ** -30)

