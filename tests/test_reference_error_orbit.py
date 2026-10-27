"""Check 1's rows over an orbit of summation orders (tests/reference_error.py, `assert_as_exact`'s
`orders`): a row that nearly cancels rounds worst at some orders and exactly at others, so both arms are
judged over the same orders, and a fault at any order still fails."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import RESOLUTION, GateFailure, _drawn, as_given, assert_as_exact, seeds_for

ROWS = 64


def _cancelling_draw(seed):
    """Sums of three terms, `[ROWS, 4, 3]`, standard normal but for the first row's, which cancel: 1, a
    quarter of an fp32 unit in the last place of 1 (2^-25), and -1, scaled by a power of two the seed
    picks, small beside the other rows' errors (the pooled ratio is theirs) and far above float64's
    resolution of the leaf, so the row is measured. In the order 1, -1, 2^-25 the exact sum survives; in
    1, 2^-25, -1 the 2^-25 is lost to the first rounding and the row is all error."""
    rng = np.random.default_rng([0x0B, seed])
    terms = rng.normal(size=(ROWS, 4, 3))
    terms[0] = np.array([1.0, 2.0 ** -25, -1.0]) * 2.0 ** int(rng.integers(-10, -6))
    return (terms.astype(np.float32),)


def _in_order(*order):
    """The sums of three terms added in `order`, the first two first."""
    @jax.jit
    def summed(terms):
        first, second, third = (terms[..., index] for index in order)
        return {"sum": (first + second) + third}
    return summed


_left_to_right = _in_order(0, 1, 2)
_outer_first = _in_order(0, 2, 1)
"""The same sums with the last two terms swapped: the first row's comes out exact."""


@jax.jit
def _largest_first(terms):
    """The sums with their two largest terms added first, whatever order they come in: the first row's
    comes out exact at every order, the others rounded twice as the other arms round them."""
    ordered = jnp.take_along_axis(terms, jnp.argsort(-jnp.abs(terms), axis=-1), axis=-1)
    return {"sum": (ordered[..., 0] + ordered[..., 1]) + ordered[..., 2]}


def _swapped(inputs):
    """The orbit of a sum's last two terms: the draw, and its terms with those two swapped."""
    (terms,) = inputs
    return [inputs, (terms[..., [0, 2, 1]],)]


def test_the_cancelling_row_is_measured_and_not_float64_zero():
    (terms,) = _cancelling_draw(0)
    row = np.abs(np.sum(terms[0].astype(np.float64), axis=-1))
    leaf = np.sqrt(np.mean(np.square(np.sum(terms.astype(np.float64), axis=-1))))
    assert np.all(row > 64 * RESOLUTION * leaf)


def test_an_orbit_judges_each_arms_worst_row_over_the_same_orders():
    """A reference whose own order is the exact one fails an equally exact Dew by the worst row alone.
    Over the orbit of the two orders both arms run at each, both worst rows are the lost 2^-25, and the
    site passes; each arm's worst row is its worst at any order, its errors and values the draw's own."""
    seeds = seeds_for(_outer_first, _cancelling_draw, "cancelling sums")
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(_left_to_right, _outer_first, _cancelling_draw, "cancelling sums", seeds=seeds)
    assert raised.type is GateFailure
    assert_as_exact(_left_to_right, _outer_first, _cancelling_draw, "cancelling sums", seeds=seeds,
                    orders=_swapped)
    over = _drawn(_left_to_right, _outer_first, _cancelling_draw, range(8), orders=_swapped)
    # Each order alone, its truth the same sums in float64 (exact: three float32 terms).
    at = [_drawn(_left_to_right, _outer_first, lambda seed, k=k: _swapped(_cancelling_draw(seed))[k],
                 range(8)) for k in range(2)]
    np.testing.assert_array_equal(over.worst, np.maximum(at[0].worst, at[1].worst))
    np.testing.assert_array_equal(over.mine, at[0].mine)
    np.testing.assert_array_equal(over.theirs, at[0].theirs)


def test_an_orbit_fails_a_dew_wrong_only_at_another_order():
    """A reference exact on the cancelling row at every order holds a Dew exact at the draw's own order
    and all error at the other: one order passes it, the orbit fails it."""
    seeds = seeds_for(_largest_first, _cancelling_draw, "largest first")
    assert_as_exact(_outer_first, _largest_first, _cancelling_draw, "largest first", seeds=seeds)
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(_outer_first, _largest_first, _cancelling_draw, "largest first", seeds=seeds,
                        orders=_swapped)
    assert raised.type is GateFailure


def test_an_orbit_judges_the_rows_of_a_leaf_the_reference_holds_exactly():
    """A leaf the reference holds exactly on every seed has no ratio to judge, and its rows still are
    over the orbit: sums of 1, 0 and 1, which a Dew dropping its middle term gets exactly at the draw's
    own order and halves at the order swapping the last two."""
    def draw(seed):
        return (np.tile(np.float32([1, 0, 1]), (2, 1, 1)),)

    @jax.jit
    def whole(terms):
        return {"sum": jnp.sum(terms, axis=-1)}

    @jax.jit
    def middle_dropped(terms):
        return {"sum": terms[..., 0] + terms[..., 2]}

    assert_as_exact(middle_dropped, whole, draw, "exact leaf", seeds=8)
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(middle_dropped, whole, draw, "exact leaf", seeds=8, orders=_swapped)
    assert raised.type is GateFailure


def test_an_orbit_fails_a_value_not_finite_at_another_order():
    """A Dew that turns NaN only at the other order (where the cancelling row's middle term is its
    negative one) passes at the draw's own order and fails over the orbit."""
    @jax.jit
    def broken_when_swapped(terms):
        return {"sum": jnp.where(terms[0, 0, 1] < 0, jnp.nan, _outer_first(terms)["sum"])}

    seeds = seeds_for(_outer_first, _cancelling_draw, "cancelling sums")
    assert_as_exact(broken_when_swapped, _outer_first, _cancelling_draw, "nan", seeds=seeds)
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(broken_when_swapped, _outer_first, _cancelling_draw, "nan", seeds=seeds,
                        orders=_swapped)
    assert raised.type is GateFailure


def test_an_orbit_still_fails_a_sum_that_differs():
    """A lost term fails over the orbit as at one order: the orbit moves rows' rounding, never a sum."""
    @jax.jit
    def lost(terms):
        return {"sum": terms[..., 0] + terms[..., 1]}

    with pytest.raises(AssertionError) as raised:
        assert_as_exact(lost, _outer_first, _cancelling_draw, "lost term",
                        seeds=seeds_for(_outer_first, _cancelling_draw, "cancelling sums"), orders=_swapped)
    assert raised.type is GateFailure


def test_an_orbit_of_a_recorded_reference_or_of_one_order_is_refused():
    """A recorded reference (`indexed_recording`'s, which it marks `recorded`) holds its recorded draws
    alone, and an orbit holds the draw's own order and another."""
    def recorded(terms):
        return _outer_first(terms)
    recorded.recorded = True

    with pytest.raises(ValueError):
        _drawn(_left_to_right, recorded, _cancelling_draw, range(2), orders=_swapped)
    with pytest.raises(ValueError):
        _drawn(_left_to_right, _outer_first, _cancelling_draw, range(2), orders=lambda inputs: [inputs])


def test_narrow_rows_take_their_narrow_factor_and_wider_rows_row_factor():
    """The worst-row rule judged alone (`_judgement` on a leaf whose pooled ratio is one): Dew's worst
    row just inside its width's factor times the reference's passes and just past it fails, at one and
    two values a row (NARROW_FACTORS) and at three and eight (ROW_FACTOR)."""
    from reference_error import NARROW_FACTORS, NO_ZERO_ROWS, ROW_FACTOR, _Draws, _judgement

    for width in (1, 2, 3, 8):
        factor = NARROW_FACTORS.get(width, ROW_FACTOR)
        for worst, fails in ((0.99 * factor, False), (1.01 * factor, True)):
            # The draws hold each worst row by its log (`_log_over`).
            draws = _Draws(["leaf"], np.ones((8, 1)), np.ones((8, 1)),
                           np.tile([[[np.log(worst), 0.0]]], (8, 1, 1)), [[NO_ZERO_ROWS]] * 8,
                           np.ones((8, 1), bool), (width,))
            over, _ = _judgement(draws, "leaf")
            assert bool(over) == fails, (width, worst)


def test_a_narrow_rows_factor_holds_through_check_one():
    """Through `assert_as_exact`: a first row of one value ten times the reference's relative error, too
    small a value for the pooled ratio to see, passes at one value a row and fails at eight."""
    def arms(width):
        def draw(seed):
            values = np.random.default_rng([0x0D, seed]).standard_normal((4096, width))
            values[0] *= 2.0 ** -8
            return (values,)

        @as_given
        def reference(values):
            return {"leaf": values * (1 + 2.0 ** -20)}

        @as_given
        def dew(values):
            moved = values * (1 + 2.0 ** -20)
            moved[0] = values[0] * (1 + 10 * 2.0 ** -20)
            return {"leaf": moved}

        return dew, reference, draw

    def truth(values):
        return {"leaf": values}

    for width, fails in ((1, False), (8, True)):
        dew, reference, draw = arms(width)
        if fails:
            with pytest.raises(AssertionError) as raised:
                assert_as_exact(dew, reference, draw, f"width {width}", seeds=8, truth_fn=truth)
            assert raised.type is GateFailure
        else:
            assert_as_exact(dew, reference, draw, f"width {width}", seeds=8, truth_fn=truth)


def test_the_stages_log_names_a_judgements_narrow_leaves(tmp_path, monkeypatch):
    """Each judgement's stages-log line names its leaves of narrow rows (NARROW_FACTORS), whose local
    errors rest on check 2, and no other."""
    import json

    from reference_error import STAGES_LOG

    log = tmp_path / "stages.jsonl"
    monkeypatch.setenv(STAGES_LOG, str(log))

    def draw(seed):
        rng = np.random.default_rng([0x1E, seed])
        return rng.standard_normal((64, 1)), rng.standard_normal((64, 8))

    def computed(narrow, wide):
        return {"narrow": narrow * (1 + 2.0 ** -20), "wide": wide * (1 + 2.0 ** -20)}

    # A NumPy law over float64 draws, as given; one function as both arms.
    given = as_given(computed)
    assert_as_exact(given, given, draw, "narrow and wide", seeds=8,
                    truth_fn=lambda narrow, wide: {"narrow": narrow, "wide": wide})
    [entry] = [json.loads(line) for line in log.read_text().splitlines()]
    assert entry["narrow_leaves"] == ["['narrow']"]
