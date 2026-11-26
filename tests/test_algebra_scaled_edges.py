"""The scaled construct's and the law's degenerate inputs (`dew.algebra.scaled`, `dew.algebra.means`): zero
and minus zero, a subnormal, the least normal, one, the format's largest, products and quotients past its
range, tangents of one, 2^100 and 2^-100, zero and subnormal totals, mixed scales, outputs an ulp apart, and
products by an inexact constant feeding a compensated sum under a jit (the contraction ledger 7856 found).
The construct's operations are values (`scaled`'s account): each its float64 truth landed once, bit for bit.
The law's are ratios (`ratios.ratio_p`, `means`'s account): its value and the first two orders of its
derivative, each a ratio of its operands, its factors as formed and the order beneath it as landed, held
equation by equation to the ratio's contract by check 2 of the gate (`certified.assert_certified`,
`certified._ratio_holds`: the exact ratio within the compensated sum's bound); its formulas are held at
drawn states by tests/test_algebra_scaled.py's float64 truths. A merged total's sums cancel nothing, so it is
its exact value landed once, bit for bit. A scaled operand is read as it is, a subnormal from its bits, where
XLA:CPU's arithmetic flushes one (the merge P0's head review, F2), and a plain float `applied` multiplies as
that multiply reads it; a subnormal result lands as zero, as XLA:CPU flushes one."""

import contextlib
import functools
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from certified import assert_certified

from dew.algebra import means, ratios, scaled, windows

LARGEST = float(np.finfo(np.float32).max)
LEAST = float(np.finfo(np.float32).tiny)
SUBNORMAL = 2.0 ** -149

POINTS = {"0": 0.0, "-0": -0.0, "subnormal": SUBNORMAL, "least": LEAST, "1": 1.0, "-1.5": -1.5,
          "2^-100": 2.0 ** -100, "2^100": 2.0 ** 100, "largest": LARGEST}
"""Each operand's degenerate values."""

TANGENTS = {"1": 1.0, "2^100": 2.0 ** 100, "2^-100": 2.0 ** -100}
"""Each tangent an operand moves by."""


def _multiplied(x: float) -> float:
    """`x` as XLA:CPU's float multiply reads it, a subnormal as zero: `applied`'s plain operand, which no
    split reads from its bits (its scaled one is read so)."""
    return 0.0 if 0 < abs(x) < LEAST else x


def _land(value: float) -> float:
    """A float64 value landed once into float32, a subnormal result flushed to its signed zero."""
    landed = float(np.float32(value))
    return float(np.copysign(0.0, landed)) if 0 < abs(landed) < LEAST else landed


def _unsigned_land(value: float) -> float:
    """A float64 value landed as the construct and a ratio land it (`scaled`'s account, `ratios.Ratio`): once,
    a zero +0."""
    landed = _land(value)
    return 0.0 if landed == 0 else landed


def assert_bits(got, expected, what: str) -> None:
    """`got` and `expected` (floats, or lists of them) the same float32 bits each, NaN matching NaN: so a
    zero's sign is held, which float equality reads as one."""
    got, expected = np.atleast_1d(np.float32(got)), np.atleast_1d(np.float32(expected))
    same = (got.view(np.uint32) == expected.view(np.uint32)) | (np.isnan(got) & np.isnan(expected))
    assert same.all(), f"{what}: {got.tolist()!r} against {expected.tolist()!r}"


CONSTRUCT = {
    "applied": (lambda x, y: scaled.applied(x, scaled.scaled(y)), lambda x, y: _multiplied(x) * y),
    "mul": (lambda x, y: scaled.landed(scaled.mul(scaled.scaled(x), scaled.scaled(y))), lambda x, y: x * y),
    "add": (lambda x, y: scaled.landed(scaled.add(scaled.scaled(x), scaled.scaled(y))), lambda x, y: x + y),
    "sub": (lambda x, y: scaled.landed(scaled.sub(scaled.scaled(x), scaled.scaled(y))), lambda x, y: x - y),
    "div": (lambda x, y: scaled.landed(scaled.div(scaled.scaled(x), scaled.scaled(y))), lambda x, y: x / y),
}
"""Each binary operation of the construct, floats in and out, and its truth in float64."""

CONSTRUCT_CELLS = [(name, a, b) for name in CONSTRUCT for a, b in itertools.product(POINTS, repeat=2)
                   if name != "div" or POINTS[b] != 0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_base2_exponent_in_the_top_binade_keeps_its_finite_product_and_differentials(dtype):
    factor = math.log2(math.e)
    top = 2.0 ** (np.finfo(dtype).maxexp - 1)
    x = np.asarray([top, -top, top / 2, -top / 2], dtype)
    direction = np.asarray([1, -2, 0, 4], dtype)

    def orders(x, direction):
        value, tangent = jax.jvp(scaled.base2, (x,), (direction,))
        _, pullback = jax.vjp(scaled.base2, x)
        (cotangent,) = pullback((direction, direction))
        return value, tangent, cotangent

    with jax.enable_x64(dtype is np.float64):
        (high, low), (dh, dl), cotangent = jax.jit(orders)(x, direction)
    leading = dtype(factor)
    residual = dtype(factor - float(leading))
    # Normalizing each limb before comparison avoids overflowing the float64 oracle's own product.
    normalized = x / top
    np.testing.assert_array_equal(np.asarray(high) / top, normalized * leading)
    np.testing.assert_array_equal(np.asarray(low) / top, normalized * residual)
    np.testing.assert_array_equal(dh, direction * leading)
    np.testing.assert_array_equal(dl, direction * residual)
    np.testing.assert_array_equal(cotangent, direction * leading)


def test_nonfinite_base2_exponents_have_no_rounding_residual():
    x = np.asarray([-np.inf, np.inf, np.nan, np.finfo(np.float32).max], np.float32)
    high, low = jax.jit(scaled.base2)(x)
    np.testing.assert_array_equal(np.asarray(high), [-np.inf, np.inf, np.nan, np.inf])
    np.testing.assert_array_equal(np.asarray(low), np.zeros_like(x))


@pytest.mark.parametrize(("name", "x", "y"), CONSTRUCT_CELLS,
                         ids=[f"{n}-{a}-{b}" for n, a, b in CONSTRUCT_CELLS])
def test_a_scaled_operation_at_degenerate_inputs_lands_its_float64_truth(name, x, y):
    """The construct's operation `name` at operands `x` and `y`, jitted: its float64 truth landed once, a zero
    +0 (`scaled`'s account)."""
    fn, truth = CONSTRUCT[name]
    got = float(jax.jit(fn)(np.float32(POINTS[x]), np.float32(POINTS[y])))
    assert_bits(got, _unsigned_land(truth(POINTS[x], POINTS[y])), "value")


def _orders(fn, direction):
    """`fn`'s value, its derivative along `direction`, and its second along it twice, one function."""
    def along(moved):
        return jax.jvp(fn, (moved,), (direction,))[1]

    def orders(moved):
        return fn(moved), along(moved), jax.jvp(along, (moved,), (direction,))[1]

    return orders


def _derivatives(fn, operand, direction):
    """`fn`'s value at `operand` and its first two derivatives along `direction` (`_orders`), jitted."""
    return [float(v) for v in jax.jit(_orders(fn, direction))(operand)]


LAW = {
    "finished": lambda x, y: means.finished(jnp.reshape(x, (1,)), y)[0],
    "resumed": lambda x, y: means.resumed(jnp.reshape(x, (1,)), y)[0],
}
"""Each binary operation of the law, a ratio of one state."""

LAW_CELLS = [(name, a, b, t)
             for name in LAW for a, b in itertools.product(POINTS, repeat=2) for t in TANGENTS]


@pytest.mark.parametrize(("name", "x", "y", "tangent"), LAW_CELLS,
                         ids=[f"{n}-{a}-{b}-d{t}" for n, a, b, t in LAW_CELLS])
def test_a_finished_and_a_resumed_state_at_degenerate_inputs_hold_their_contract(name, x, y, tangent):
    """A finished or a resumed state at operands `x` and `y`, both moving by `tangent`: its value and its
    first and second derivatives, each ratio held to its contract (check 2)."""
    direction = np.full(2, TANGENTS[tangent], np.float32)
    orders = _orders(lambda pair: LAW[name](pair[0], pair[1]), direction)
    pair = np.asarray([POINTS[x], POINTS[y]], np.float32)
    assert_certified(orders, lambda seed: (pair,), f"{name} at {x}, {y}, d{tangent}", seeds=1)


MERGES = {
    # Each: outputs, totals, anchors, the units' scale.
    "a zero total": ([[0.5], [-0.5]], [1.0, 0.0], [0.0, 0.0], 1.0),
    "a zero total at a higher anchor": ([[0.5], [-0.5]], [1.0, 0.0], [0.0, 3.0], 1.0),
    "a subnormal total": ([[0.5], [-0.5]], [1.0, SUBNORMAL], [0.0, 0.0], 1.0),
    "totals at the format's largest": ([[0.5], [-0.5]], [LARGEST, LARGEST], [0.0, 0.0], 1.0),
    "totals at the least normal": ([[0.5], [-0.5]], [LEAST, LEAST], [0.0, 0.0], 1.0),
    "mixed scales": ([[1.0], [2.0 ** 100]], [2.0 ** -100, 2.0 ** 100], [-60.0, 0.0], 1.0),
    "a large output's negligible state": ([[0.0], [2.0 ** 127]], [2.0 ** -126, 2.0 ** 127], [0.0, -1000.0],
                                          1.0),
    "outputs an ulp apart": ([[1.0], [1.0 + 2.0 ** -23]], [1.0, 1.0], [0.0, 0.0], 1.0),
    "opposite outputs at the largest": ([[LARGEST], [-LARGEST]], [1.0, 1.0], [0.0, 0.0], 1.0),
    "minus zero outputs": ([[-0.0], [0.0]], [1.0, 3.0], [0.0, 0.0], 1.0),
    "inexact products at an inexact scale": ([[1 / 3], [0.1]], [0.7, 0.3], [0.25, -0.5], 0.3),
    "the largest total beside three of 2^102": ([[0.5], [-0.5], [0.25], [1.0]],
                                                [LARGEST, *[2.0 ** 102] * 3], [0.0] * 4, 1.0),
    "two totals of 2^127 at a near anchor": ([[1.0], [0.5]], [2.0 ** 127] * 2, [0.0] * 2, 1.0),
}
"""Each merge's degenerate states and units: the inexact scale's every term a product by inexact factors
and an inexact constant (its units' scale) feeding a compensated sum under a jit."""

ALONG = ("outputs", "totals", "anchors")
"""The operand a merge moves along."""


MERGE_CELLS = [(case, along, tangent) for case in MERGES for along in ALONG for tangent in TANGENTS]


@pytest.mark.parametrize(("case", "along", "tangent"), MERGE_CELLS,
                         ids=[f"{case}-{along}-d{tangent}" for case, along, tangent in MERGE_CELLS])
def test_a_merge_at_degenerate_states_holds_its_contract(case, along, tangent):
    """The law (`means.merged_output`) at degenerate states, its outputs, totals or anchors moving by
    `tangent`: its value and its first and second derivatives, each ratio held to its contract (check 2)."""
    outputs, totals, anchors = (np.asarray(v, np.float32) for v in MERGES[case][:3])
    units = windows.Units(scale=MERGES[case][3])
    operands = {"outputs": outputs, "totals": totals, "anchors": anchors}

    def fn(moved):
        return means.merged_output(**{**operands, along: moved}, units=units, axis=0)[0]

    orders = _orders(fn, np.full_like(operands[along], TANGENTS[tangent]))
    assert_certified(orders, lambda seed: (operands[along],), f"{case} along {along}, d{tangent}", seeds=1)


def test_a_merge_past_its_anchors_resolution_lands_an_infinite_total():
    """A merge whose mass passes the format at its joined anchor and whose anchor's next value is farther than
    the move its total needs lies outside the law's domain (docs/design/algebra.md 6.12): two totals of 2^127
    at an anchor of 2^31, whose next value 256 past it flushes their mass, merge to an infinite total, as the
    merged total and the merged state each give it, never a finite mass of zero (the base review's F2: a
    later merge with output 1 gave 1 for 1/3); the same totals at an anchor of 0 land finite (the merged
    total cells' "two totals of 2^127 at a near anchor")."""
    outputs = np.asarray([[1.0], [0.5]], np.float32)
    totals, anchors = np.full(2, 2.0 ** 127, np.float32), np.full(2, 2.0 ** 31, np.float32)
    units = windows.Units(scale=1.0)
    total = jax.jit(lambda w, m: means.merged_total(w, m, units, 0)[0])(totals, anchors)
    state = jax.jit(lambda o, w, m: means.merged_state(o, w, m, units=units, axis=0)[1])
    assert [float(np.asarray(total)), float(np.asarray(state(outputs, totals, anchors)))] == [np.inf, np.inf]


def test_a_zero_mass_state_past_its_reach_adds_nothing_to_a_merge():
    """A state of no mass whose anchor lies past the factors' reach from the joined one (its exponential
    refused, NaN) adds nothing to the merge: outputs 1 and 2, totals 1 and 0, anchors 0 and 10^8 merge to 1,
    its total 1 at the anchor 0, as the merged output and the merged state each give it (the base review's
    F3: a zero total times the refused factor was NaN)."""
    outputs = np.asarray([[1.0], [2.0]], np.float32)
    totals, anchors = np.asarray([1.0, 0.0], np.float32), np.asarray([0.0, 1e8], np.float32)
    units = windows.Units(scale=1.0)
    merged = jax.jit(lambda o, w, m: means.merged_output(o, w, m, units=units, axis=0))
    state = jax.jit(lambda o, w, m: means.merged_state(o, w, m, units=units, axis=0))
    merged, state = merged(outputs, totals, anchors), state(outputs, totals, anchors)
    assert [float(np.asarray(merged)[0]), *(float(np.ravel(np.asarray(part))[0]) for part in state)] == [
        1.0, 1.0, 1.0, 0.0]


def test_opposite_channels_gradients_are_their_sums_landed_once():
    """The gradients of a merge whose channels cancel (outputs `+-2^126` opposite across two channels, unit
    totals, zero anchors, the output's cotangent 16 in each) along its totals and anchors are each channel's
    term summed before it lands, exactly zero, the transpose's sum within the ratio (`Ratio.over`): landed a
    channel at a time, `16 2^126 / 2` was `+-inf`, and their sum NaN (the base review's F1)."""
    outputs = np.asarray([[2.0 ** 126, -2.0 ** 126], [-2.0 ** 126, 2.0 ** 126]], np.float32)
    totals, anchors = np.ones(2, np.float32), np.zeros(2, np.float32)
    units = windows.Units(scale=1.0)
    cotangent = np.full(2, 16.0, np.float32)

    def loss(w, m):
        return jnp.sum(means.merged_output(outputs, w, m, units=units, axis=0) * cotangent)

    gradients = jax.jit(jax.grad(loss, argnums=(0, 1)))(totals, anchors)
    assert [np.asarray(gradient).tolist() for gradient in gradients] == [[0.0, 0.0], [0.0, 0.0]]


def test_a_finished_rows_gradient_across_its_broadcast_totals_sums_each_rows_quotient():
    """A finished state's accumulated row broadcast across two totals (`means.finished`, `acc / l` at `l` 1
    and 2): its summed output's gradient along the row is each quotient's derivative summed, `1 + 1/2`, the
    totals varying along the axis it is broadcast over (the review of the base review's fixes: a transpose
    summing the numerators over that axis before one denominator took `sum(N) / D` for `sum(N_c / D_c)`, and
    its result no longer laid to the row)."""
    accumulated, totals = np.ones((1, 3), np.float32), np.asarray([1.0, 2.0], np.float32)
    gradient = jax.jit(jax.grad(lambda a: jnp.sum(means.finished(a, totals))))(accumulated)
    assert np.asarray(gradient).tolist() == [[1.5, 1.5, 1.5]]


def test_a_finished_rows_gradient_past_the_format_sums_its_quotients_before_they_land():
    """A finished state's accumulated row `[[1]]` broadcast across totals 0.25 and 0.5 (`means.finished`),
    pulled back by `2^126` and `-2^127`: its gradient `2^126 / 0.25 - 2^127 / 0.5` is exactly 0, its
    quotients past the format summed before they land (`Ratio.across`), each pulled-back ratio within its
    contract (check 2); landed apart they were `+-inf`, and their sum NaN (the merge P0's base review F1).
    Its derivative along the totals, at a cotangent of ones, is each quotient's own, `-(1 / 0.25^2 + 1 /
    0.5^2)` = -20, never the summed quotient's; past the format, along totals `[x, 2x]`, which keep the
    cancellation at every order, its first three derivatives are exactly 0, each a ratio of its own closed
    form within its contract (check 2): landed, `+-2^128` were infinite operands of its first derivative, and
    the tangents of quotients held in [1, 2) overflowed at its second (the merge P0's base reviews 3 and 4,
    F1)."""
    totals = np.asarray([0.25, 0.5], np.float32)

    def pulled(accumulated, totals, cotangent):
        return jax.vjp(lambda a: means.finished(a, totals), accumulated)[1](cotangent)[0]

    accumulated = np.ones((1, 1), np.float32)
    past = np.asarray([[2.0 ** 126], [-(2.0 ** 127)]], np.float32)
    assert np.asarray(jax.jit(pulled)(accumulated, totals, past)).tolist() == [[0.0]]
    assert_certified(lambda a: pulled(a, totals, past), lambda seed: (accumulated,), "a row past the format",
                     seeds=1)
    ones = np.ones((2, 1), np.float32)
    moved = jax.jit(lambda t: jax.jvp(lambda t: pulled(accumulated, t, ones), (t,), (jnp.ones_like(t),))[1])
    assert np.asarray(moved(totals)).tolist() == [[-20.0]]

    # Along totals [x, 2x] the cotangent's quotients cancel at every order, `(-1)^k k! sum_c c_c d_c^k /
    # t_c^(k+1)` = 0: along the totals themselves, and along `2^126` and `2^127`, past the format at each.
    for along in (totals, np.asarray([2.0 ** 126, 2.0 ** 127], np.float32)):
        def moved_along(k, along=along):
            fn = functools.partial(pulled, accumulated, cotangent=past)
            for _ in range(k):
                fn = functools.partial(lambda inner, t: jax.jvp(inner, (t,), (along,))[1], fn)
            return fn

        for order in (1, 2, 3):
            assert np.asarray(jax.jit(moved_along(order))(totals)).tolist() == [[0.0]], (along, order)
            assert_certified(moved_along(order), lambda seed: (totals,),
                             f"a row past the format moved along {along.tolist()}, order {order}", seeds=1)


def test_a_merge_of_anchors_whose_difference_passes_the_format_weighs_its_scaled_difference():
    """Outputs 0 and 2^127, totals 2^-126 and 2^127, anchors 2^127 and -2^127 at a scale of 2^-120: the
    second state's factor `exp(-256)`, its anchors' difference past the format and its scaled value within
    it, merges as the same states at anchors halved and the scale doubled, whose difference the format
    holds, bit for bit (the merged output and the merged state's output and total; its landed anchor the
    joined one), 1629.34 in float64 (the merge P0's base review F2: the difference overflowed before it was
    scaled, and they merged to 0)."""
    outputs = np.asarray([[0.0], [2.0 ** 127]], np.float32)
    totals = np.asarray([2.0 ** -126, 2.0 ** 127], np.float32)
    merges = []
    for anchor, scale in ((2.0 ** 127, 2.0 ** -120), (2.0 ** 126, 2.0 ** -119)):
        units = windows.Units(scale=scale)
        anchors = np.asarray([anchor, -anchor], np.float32)
        output = jax.jit(lambda o, w, m, units=units: means.merged_output(o, w, m, units, 0))
        state = jax.jit(lambda o, w, m, units=units: means.merged_state(o, w, m, units, 0))
        merged, total, landed = state(outputs, totals, anchors)
        assert float(landed) == anchor
        merges.append([np.asarray(x).tolist() for x in (output(outputs, totals, anchors), merged, total)])
    assert merges[0] == merges[1]


@pytest.mark.parametrize(("outputs", "totals", "anchors"), [((2, 2, 1), (1, 2), (2, 2)),
                                                            ((1, 2, 1), (2, 2), (2, 2))],
                         ids=["totals shared by a batch", "outputs shared by a batch"])
def test_a_merged_state_of_operands_shared_by_a_batch_is_its_parts(outputs, totals, anchors):
    """States merged along their second axis (`means.merged_state`), one operand shared by a batch the
    others hold whole: its output and total on par with those its parts give (`means.merged_output`,
    `means.merged_total`) from float64, two programs held by check 1 (the merge P0's base review 12, F5), and
    its landed anchor theirs bit for bit, jitted (base review F3: totals `[1, 2]` beside anchors `[2, 2]`
    declared a total of one row where its value held two, and its lowering was refused)."""
    from reference_error import assert_as_exact, dew_path

    from dew.parity import FEWEST_SEEDS

    shapes = (outputs, totals, anchors)

    def drawn(seed):
        rng = np.random.default_rng([8, seed])
        outputs, totals, anchors = shapes
        held = (rng.random(totals) + 0.5).astype(np.float32)
        return (rng.standard_normal(outputs).astype(np.float32), held,
                rng.standard_normal(anchors).astype(np.float32))

    units = windows.Units()

    def state(o, w, m):
        return means.merged_state(o, w, m, units, 1)

    def parts(o, w, m):
        return (means.merged_output(o, w, m, units, 1), *means.merged_total(w, m, units, 1))

    assert_as_exact(lambda *xs: state(*xs)[:2], dew_path(lambda *xs: parts(*xs)[:2]), drawn,
                    "a merged state of operands shared by a batch", seeds=FEWEST_SEEDS)
    operands = drawn(0)
    landed = [np.asarray(jax.jit(form)(*operands)[2]).tolist() for form in (state, parts)]
    assert landed[0] == landed[1]


@pytest.mark.mesh(devices=2)
def test_the_laws_operations_vary_where_any_operand_they_read_does():
    """Under `jax.shard_map`, a merged state of replicated outputs and anchors beside each device's own
    totals, a finished state of a replicated accumulation over each device's total, and a replicated value
    scaled by each device's own factor (`scaled.applied`) vary along the devices' axis, as their values do:
    a scan carrying them, its carry varying, traces, and each device's results are its own run's alone,
    bit for bit (the merge P0's base review 8, F1: each result typed by one operand dropped the axis
    another varied along, and the scan's carry and its result disagreed)."""
    from jax.sharding import NamedSharding, PartitionSpec

    mesh = jax.make_mesh((2,), ("rows",), devices=jax.devices()[:2])
    outputs, anchors = np.asarray([[0.0], [1.0]], np.float32), np.zeros(2, np.float32)
    accumulated = np.asarray([[2.0, -3.0]], np.float32)
    totals = np.asarray([[1.0, 1.0], [1.0, 3.0]], np.float32)

    def run(totals):
        def body(carry, _):
            merged, total, _ = means.merged_state(outputs, totals, anchors, windows.Units(), 0)
            done = means.finished(accumulated, totals[:1])
            moved = scaled.applied(outputs[:, 0], scaled.scaled(totals))
            return (merged, total, done, moved), None

        init = jax.tree.map(lambda shape: jnp.zeros(shape.shape, shape.dtype) * totals[0],
                            jax.eval_shape(lambda: body(None, None)[0]))
        return jax.lax.scan(body, init, None, length=1)[0]

    split = PartitionSpec("rows")
    mapped = jax.jit(jax.shard_map(lambda totals: jax.tree.map(lambda x: x[None], run(totals[0])), mesh=mesh,
                                   in_specs=split, out_specs=split))(
        jax.device_put(totals, NamedSharding(mesh, split)))
    # Each device's rows read on the host: an index into a sharded array names no sharding of its own.
    mapped = [np.asarray(x) for x in mapped]
    for device in range(2):
        own = jax.jit(run)(totals[device])
        assert [x[device].tolist() for x in mapped] == [np.asarray(x).tolist() for x in own]


@pytest.mark.mesh(devices=2)
def test_a_scaled_product_varies_where_its_exponent_alone_does():
    """Under `jax.shard_map`, a replicated value and significand scaled by each device's own exponent
    (`scaled.applied`) vary along the devices' axis, as their products do: a scan carrying them from
    `2^exponent` traces, each device's product its own, 1 and 2 (the merge P0's base review 9, F1: the
    product's type read its exponent's axes nowhere, and the scan's carry and its result disagreed)."""
    from jax.sharding import NamedSharding, PartitionSpec

    mesh = jax.make_mesh((2,), ("rows",), devices=jax.devices()[:2])
    split = PartitionSpec("rows")

    def run(exponent):
        def body(carry, _):
            unit = jnp.ones(1, jnp.float32)
            return scaled.applied(unit, scaled.Scaled(unit, exponent, 4)), None

        return jax.lax.scan(body, jnp.exp2(exponent.astype(jnp.float32)), None, length=1)[0]

    exponents = jax.device_put(np.asarray([0, 1], np.int32), NamedSharding(mesh, split))
    made = jax.jit(jax.shard_map(run, mesh=mesh, in_specs=split, out_specs=split))(exponents)
    assert np.asarray(made).tolist() == [1.0, 2.0]


def test_a_finished_states_derivative_along_its_total_keeps_a_quotient_beneath_the_format():
    """float32 `finished([2^-126], 2^30)` moved along its total by `2^120`: its derivative
    `-2^-126 2^120 / 2^60 = -2^-66` exactly, each sum of its closed quotient rule unlanded, and within its
    contract (check 2); through a landed quotient, `2^-156` flushed to zero, it was zero (the merge P0's
    base review 9, F2)."""
    accumulated = np.asarray([[2.0 ** -126]], np.float32)

    def moved(total):
        return jax.jvp(lambda t: means.finished(accumulated, t), (total,), (np.float32([2.0 ** 120]),))[1]

    total = np.float32([2.0 ** 30])
    assert np.asarray(jax.jit(moved)(total)).tolist() == [[-(2.0 ** -66)]]
    assert_certified(moved, lambda seed: (total,), "a finished state moved along its total", seeds=1)


def test_a_sums_derivative_along_one_operand_runs_over_the_whole_domain():
    """`sum(x + y)` over the states' axis, its two terms over `x [1, 1]` and `y [2, 3]`, moved along `x` alone
    by one: its derivative `[2, 2, 2]`, each of the two states and three channels its domain's, forward and
    reverse; with `y`, which no derivative term reads, dropped, it was `[1]` (the merge P0's base review 9,
    F3)."""
    form = ratios.Ratio(((1.0, (("v", 0),)), (1.0, (("v", 1),))), (), 0, reduced=True, units=windows.Units())
    x, y = np.ones((1, 1), np.float32), np.ones((2, 3), np.float32)

    def summed(x):
        return ratios.ratio_p.bind(jnp.asarray(x), jnp.asarray(y), form=form)

    tangent = jax.jit(lambda x: jax.jvp(summed, (x,), (jnp.ones_like(x),))[1])(x)
    pulled = jax.jit(lambda x: jax.vjp(summed, x)[1](jnp.ones(3, jnp.float32))[0])(x)
    assert np.asarray(tangent).tolist() == [2.0, 2.0, 2.0] and np.asarray(pulled).tolist() == [[6.0]]


def test_a_merged_total_past_the_format_lands_at_the_least_finite_anchor_that_holds_it():
    """32 float32 totals of `2^127` at anchors 0, at a scale of `2^-126`: their mass `2^132` lands at a finite
    anchor that holds it (about `2.77 2^126`), its total finite, within its contract (check 2), where the move
    one binade past it overflowed to an infinite anchor though `3 2^126` holds them (the merge P0's base
    review 9, F4)."""
    totals, anchors = np.full(32, 2.0 ** 127, np.float32), np.zeros(32, np.float32)
    units = windows.Units(scale=2.0 ** -126)
    total, anchor = jax.jit(lambda t: means.merged_total(t, anchors, units, 0))(totals)
    assert np.isfinite(np.asarray(anchor)) and np.isfinite(np.asarray(total))
    assert_certified(lambda t: means.merged_total(t, anchors, units, 0)[0], lambda seed: (totals,),
                     "32 totals past the format", seeds=1)


def test_an_empty_rows_anchor_is_chosen_by_the_law_before_its_mass_and_tangent():
    """A merge of one state of output 0, anchor 1 and total `t`, at `t = 0` a row of no mass: its anchor the
    law's, the largest of its states' maxima, 1, and its total's tangent taken there, `dt`; merged then with
    `(1, 0, 1)`, the output's derivative along `t` at 0 is `-exp(1)`, as the walk's start would hold it. A
    row whose states' anchors are all -inf returns -inf (the merge P0's base review 10, F1: chosen outside
    the law, the row's tangent was taken at 0 and its anchor moved to 1 after, giving -exp(2))."""
    units = windows.Units()

    def merged(t):
        one, total, anchor = means.merged_state(jnp.zeros((1, 1)), jnp.reshape(t, (1,)), jnp.ones(1), units,
                                                0)
        output, _, _ = means.merged_state(jnp.stack([one, jnp.ones(1)]), jnp.stack([total, 1.0]),
                                          jnp.stack([anchor, 0.0]), units, 0)
        return output[0], total, anchor

    zero = jnp.float32(0.0)
    (_, _, anchor), (d_output, d_total, _) = jax.jit(
        lambda t: jax.jvp(merged, (t,), (jnp.ones_like(t),)))(zero)
    assert float(anchor) == 1.0 and float(d_total) == 1.0 and float(d_output) == np.float32(-math.e)
    _, _, empty = jax.jit(lambda: means.merged_state(jnp.zeros((2, 1)), jnp.zeros(2),
                                                     jnp.full(2, -jnp.inf), units, 0))()
    assert float(empty) == -math.inf


@pytest.mark.parametrize("jitted", [True, False], ids=["jitted", "op by op"])
def test_a_merged_totals_anchor_and_move_are_formed_together_in_its_units(jitted):
    """64 float32 totals of `2^127` at anchors `-2^127`, at a scale of `2^-126`: the move their mass `2^133`
    needs, about `4.85 2^126` in the anchor's own units, passes the format beside the anchor, yet the anchor
    and the move together, about `2.85 2^126`, are finite and hold the total, finite and normal, within its
    contract (check 2), jitted and op by op, formed together in the units' scale; formed apart, the move
    overflowed to an infinite anchor and total wherever no compiler fused it into the anchor's sum (the
    merge P0's base review 10, F2: op by op)."""
    totals, anchors = np.full(64, 2.0 ** 127, np.float32), np.full(64, -(2.0 ** 127), np.float32)
    units = windows.Units(scale=2.0 ** -126)
    with contextlib.nullcontext() if jitted else jax.disable_jit():
        total, anchor = jax.jit(lambda t: means.merged_total(t, anchors, units, 0))(totals)
    assert np.isfinite(np.asarray(anchor)) and np.isfinite(np.asarray(total))
    assert abs(float(total)) >= float(jnp.finfo(jnp.float32).tiny)
    assert_certified(lambda t: means.merged_total(t, anchors, units, 0)[0], lambda seed: (totals,),
                     "64 totals past the format beside a far anchor", seeds=1)


def test_a_derivative_along_a_scalar_keeps_the_domains_rank():
    """`sum(x + y)` over the states' axis, `x` of shape `[1, 1]` and `y` a scalar, moved along `y` alone by
    one: its derivative `[1]`, of the domain's rank, forward and reverse; compacted to `y`'s tangent alone,
    a scalar, its states' axis was none (the merge P0's base review 10, F3)."""
    form = ratios.Ratio(((1.0, (("v", 0),)), (1.0, (("v", 1),))), (), 0, reduced=True, units=windows.Units())
    x = jnp.ones((1, 1), jnp.float32)

    def summed(y):
        return ratios.ratio_p.bind(x, y, form=form)

    tangent = jax.jit(lambda y: jax.jvp(summed, (y,), (jnp.ones_like(y),))[1])(jnp.float32(2.0))
    pulled = jax.jit(lambda y: jax.vjp(summed, y)[1](jnp.ones(1, jnp.float32))[0])(jnp.float32(2.0))
    assert np.asarray(tangent).tolist() == [1.0] and float(pulled) == 1.0


def test_a_product_of_a_sum_over_the_states_and_a_sum_over_channels_transposes():
    """`f[i] = sum_k x[k] sum_c y[i, c]` (`ratios.Ratio.products`: one sum over the states, one a state at a
    time over its channels), `y = [[1, 2], [3, 4]]`: the gradient of `sum(f)` along each `x` is `sum_ic y =
    10`, its cotangent contracted over the states against the channel sum, each sum unlanded; refused before
    (the merge P0's base review 10, F4)."""
    over_states = ratios.Summed(((1.0, (("v", 0),)),), reduced=True)
    over_channels = ratios.Summed(((1.0, (("v", 1),)),), reduced=False, over=(1,))
    form = ratios.Ratio((), (), 0, reduced=False, units=windows.Units(), over=(1,),
                       products=((1.0, (over_states, over_channels)),))
    x, y = jnp.ones((2, 1), jnp.float32), jnp.asarray([[1.0, 2.0], [3.0, 4.0]], jnp.float32)
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(ratios.ratio_p.bind(x, y, form=form))))(x)
    assert np.asarray(gradient).tolist() == [[10.0], [10.0]]


def test_products_beneath_their_neighbours_binade_are_summed_before_they_cancel():
    """Three products of one sum each, `2^127`, `2^-126` and `-2^127` a state: their sum `2^-126`, normal,
    summed banded and compensated as a sum's terms are, landing once; joined in turn, the middle one was
    shifted past its precision before the outer two cancelled, and the sum was 0 (the merge P0's base
    review 10, F5)."""
    form = ratios.Ratio((), (), 0, reduced=False, units=windows.Units(), products=tuple(
        (1.0, (ratios.Summed(((1.0, (("v", index),)),), reduced=False),)) for index in range(3)))
    operands = [jnp.full(1, value, jnp.float32) for value in (2.0 ** 127, 2.0 ** -126, -(2.0 ** 127))]
    assert np.asarray(jax.jit(lambda *xs: ratios.ratio_p.bind(*xs, form=form))(*operands)).tolist() == [
        2.0 ** -126]


def test_a_product_of_a_sum_over_the_states_and_two_channel_sums_transposes():
    """`f[i] = sum_k x[k] sum_c y[i, c] sum_c z[i, c]`, two states, `y = z = ones((2, 2))`: the gradient of
    `sum(f)` along each `x` is `sum_i (sum_c y)(sum_c z) = 8`, the two channel sums a product, not one sum of
    their terms' products, each unlanded; refused before (the merge P0's base review 11, F3)."""
    over_states = ratios.Summed(((1.0, (("v", 0),)),), reduced=True)
    over_channels = [ratios.Summed(((1.0, (("v", index),)),), reduced=False, over=(1,)) for index in (1, 2)]
    form = ratios.Ratio((), (), 0, reduced=False, units=windows.Units(), over=(1,),
                       products=((1.0, (over_states, *over_channels)),))
    x, y = jnp.ones((2, 1), jnp.float32), jnp.ones((2, 2), jnp.float32)
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(ratios.ratio_p.bind(x, y, y, form=form))))(x)
    assert np.asarray(gradient).tolist() == [[8.0], [8.0]]


def test_a_channel_sum_beside_a_factor_that_keeps_the_channel_transposes_as_their_product():
    """`f[i, c] = sum_k x[k] sum_d y[i, d] z[i, c]`, two states, `y = z = [[1, 2], [3, 4]]`: the gradient of
    `sum(f)` along each `x` is `sum_ic (sum_d y_id) z_ic = 9 + 49 = 58`, the channel sum a product with the
    factor beside it; the transpose contracted `y_ic z_ic` over the one channel axis, 30 (the merge P0's
    base review 12, F1). Along `y`, `z` held, it is `sum_c (sum_k x_k) z_ic`, the same at every `d`:
    `[[6, 6], [14, 14]]`, where the transpose read `z` at `d`, `[[2, 4], [6, 8]]` (base review 13, F1)."""
    over_states = ratios.Summed(((1.0, (("v", 0),)),), reduced=True)
    over_channels = ratios.Summed(((1.0, (("v", 1),)),), reduced=False, over=(1,))
    kept = ratios.Summed(((1.0, (("v", 2),)),), reduced=False)
    form = ratios.Ratio((), (), 0, reduced=False, units=windows.Units(),
                       products=((1.0, (over_states, over_channels, kept)),))
    x, y = jnp.ones((2, 1), jnp.float32), jnp.asarray([[1.0, 2.0], [3.0, 4.0]], jnp.float32)
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(ratios.ratio_p.bind(x, y, y, form=form))))(x)
    along_y = jax.jit(jax.grad(lambda held: jnp.sum(ratios.ratio_p.bind(x, held, y, form=form))))(y)
    assert np.asarray(gradient).tolist() == [[58.0], [58.0]]
    assert np.asarray(along_y).tolist() == [[6.0, 6.0], [14.0, 14.0]]


def test_a_sum_cancelling_in_two_bands_keeps_the_band_beneath_both():
    """Outputs `[1, -1, 1, -1, 1, 0]` at anchors `[1000, 1000, 200, 200, 0, 0]`, all the mass at the last:
    along totals `[1, 1, 1, 1, 1, 0]` both opposite pairs cancel, `e^1000` beside `e^200` beside `1`, and the
    merged output's derivative is exactly 1. With one band beneath the first, the survivor was placed 288
    binades beneath its band's pair and the derivative was 0 (the merge P0's base review 13, F2)."""
    outputs = np.asarray([1, -1, 1, -1, 1, 0], np.float32)[:, None]
    anchors = np.asarray([1000, 1000, 200, 200, 0, 0], np.float32)
    units = windows.Units(scale=1.0)

    def merged(totals):
        return means.merged_output(outputs, totals, anchors, units=units, axis=0)

    totals, along = np.asarray([0, 0, 0, 0, 0, 1], np.float32), np.asarray([1, 1, 1, 1, 1, 0], np.float32)
    value, tangent = jax.jit(lambda w, t: jax.jvp(merged, (w,), (t,)))(totals, along)
    assert (np.asarray(value).tolist(), np.asarray(tangent).tolist()) == ([0.0], [1.0])


def test_a_ratio_whose_derivatives_products_cancel_has_a_zero_tangent():
    """`sum(x) / sum(x)` at `x = [2]`: its derivative's two products, `dx x` and `-x dx`, are one product of
    opposite coefficients, so its tangent and its gradient are zero; its empty numerator reached its sum's
    bits before (the merge P0's base review 11, F2)."""
    own = ((1.0, (("v", 0),)),)
    form = ratios.Ratio(own, own, 0, reduced=True, units=windows.Units(scale=1.0))
    x = jnp.asarray([2.0], jnp.float32)
    value, tangent = jax.jit(lambda x, t: jax.jvp(lambda x: ratios.ratio_p.bind(x, form=form), (x,), (t,)))(
        x, jnp.ones_like(x))
    gradient = jax.jit(jax.grad(lambda x: ratios.ratio_p.bind(x, form=form)))(x)
    assert (float(value), float(tangent), np.asarray(gradient).tolist()) == (1.0, 0.0, [0.0])


def test_a_band_beneath_the_normals_keeps_a_half_ulp_its_larger_terms_opposite_cancels():
    """Outputs `[2^-97, L, -2^-97, 0, 2^-121, -L]`, `L = 1.5 2^127`, merged at masses `[1/32, 3/8, 1/32, 0,
    1/32, 3/8]` and zero anchors: the pair past the format cancels, and the products beneath the normals,
    `2^-102`, `-2^-102` and `2^-126`, sum to `2^-126`, the merge `(32 / 27) 2^-126`. Summed plainly, the
    band beneath rounded `2^-102 + 2^-126` to `2^-102` before its opposite cancelled it, and the merge was 0
    (the merge P0's base review 11, F1)."""
    big = 1.5 * 2.0 ** 127
    outputs = np.asarray([2.0 ** -97, big, -(2.0 ** -97), 0.0, 2.0 ** -121, -big], np.float32)[:, None]
    totals = np.asarray([1 / 32, 3 / 8, 1 / 32, 0, 1 / 32, 3 / 8], np.float32)
    units = windows.Units(scale=1.0)
    merged = jax.jit(lambda o, w, m: means.merged_output(o, w, m, units=units, axis=0))(
        outputs, totals, np.zeros(6, np.float32))
    assert np.asarray(merged).tolist() == [float(np.float32(2.0 ** -126 * 32 / 27))]


def test_a_ratios_own_sum_transposes_to_its_cotangent_repeated():
    """A ratio summing its numerator over a further axis its one operand alone carries (`Ratio.over`): its
    value the whole sum, and its gradient the cotangent repeated across that axis (the review: the
    transpose's result kept one along it and was refused the operand's shape)."""
    form = ratios.Ratio(((1.0, (("v", 0),)),), (), 0, reduced=True, units=windows.Units(scale=1.0), over=(1,))
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    value = jax.jit(lambda x: ratios.ratio_p.bind(x, form=form))(x)
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(ratios.ratio_p.bind(x, form=form))))(x)
    assert (np.asarray(value).tolist(), np.asarray(gradient).tolist()) == ([15.0], [[1.0] * 3] * 2)


@pytest.mark.parametrize("states", [7, 8, 9, 17, 33])
@pytest.mark.parametrize("case", list(CANCELLING_PAIRS := {
    "a pair past 2^127": 2.0 ** 127 * (1 + 2.0 ** -23), "the largest pair": float(np.finfo(np.float32).max)}))
def test_states_cancelling_across_a_long_row_merge_to_their_truth(case, states):
    """A row of many states, all but three of no mass, whose two largest outputs cancel exactly at its first
    and middle states beside a least normal at its last: the merged output and the merged state's are the
    least normal, bit for bit, however the law reads the row's states (`means._streamed`, a chunk at a
    time: the pair and the least normal in different chunks, where a merge of chunks' landed merges lost
    the least normal to the first chunk's landing, the decode merge's design question)."""
    big = CANCELLING_PAIRS[case]
    outputs, totals = np.zeros((states, 1), np.float32), np.zeros(states, np.float32)
    for place, value in ((0, big), (states // 2, -big), (states - 1, 3 * LEAST)):
        outputs[place, 0], totals[place] = value, 1.0
    anchors = np.zeros(states, np.float32)
    units = windows.Units(scale=1.0)
    merged = jax.jit(lambda o, w, m: means.merged_output(o, w, m, units=units, axis=0))
    state = jax.jit(lambda o, w, m: means.merged_state(o, w, m, units=units, axis=0)[0])
    merged, state = merged(outputs, totals, anchors), state(outputs, totals, anchors)
    assert [float(np.asarray(merged)[0]), float(np.asarray(state)[0])] == [LEAST, LEAST]


CANCELLING = {
    "a pair past 2^127": 2.0 ** 127 * (1 + 2.0 ** -23),
    "the largest pair": LARGEST,
    "a pair at 2^126": 2.0 ** 126 * (1 + 2.0 ** -23),
}
"""Each pair of outputs that cancel exactly beside three least normals (the cancelling states' cells)."""


@pytest.mark.parametrize("case", list(CANCELLING))
@pytest.mark.parametrize("order", list(itertools.permutations(range(3))), ids=str)
def test_states_cancelling_beside_a_least_normal_merge_to_their_truth(case, order):
    """States of equal mass whose two largest outputs cancel exactly beside one of three least normals
    (attention's slices near fp32's largest, row 6909's class), in every order: the merged output and the
    merged state's are the least normal, the remaining output over the total mass, bit for bit, as fp32's
    plain merge keeps it (a sum placed beneath the pair's magnitudes flushed the least-normal output, and the
    merge came to 0: attention's check 1 at the law's 513aad8eb, which this sweep's contract cells passed)."""
    big = CANCELLING[case]
    outputs = np.asarray([[big], [-big], [3 * LEAST]], np.float32)[list(order)]
    totals, anchors = np.ones(3, np.float32), np.zeros(3, np.float32)
    units = windows.Units(scale=1.0)
    merged = jax.jit(lambda o, w, m: means.merged_output(o, w, m, units=units, axis=0))
    state = jax.jit(lambda o, w, m: means.merged_state(o, w, m, units=units, axis=0)[0])
    merged, state = merged(outputs, totals, anchors), state(outputs, totals, anchors)
    assert [float(np.asarray(merged)[0]), float(np.asarray(state)[0])] == [LEAST, LEAST]


@pytest.mark.parametrize("case", list(MERGES))
@pytest.mark.parametrize("tangent", list(TANGENTS))
def test_a_merged_total_at_degenerate_states_holds_its_contract(case, tangent):
    """The merged total (`means.merged_total`) at degenerate states, its totals moving by `tangent`: its
    value the sum of its masses at the anchor it lands at, its factors to it as formed, landed once, and
    finite, the anchor moved where the sum at the joined one passes the format (the largest total beside
    three of 2^102 landed past it where its landing read a rounded sum that lost them, the law's review);
    and its derivatives (linear in the totals) the factors' sum along them, then none."""
    _, totals, anchors = (np.asarray(v, np.float32) for v in MERGES[case][:3])
    units = windows.Units(scale=MERGES[case][3])
    along = np.full_like(totals, TANGENTS[tangent])
    got = _derivatives(lambda w: means.merged_total(w, anchors, units, 0)[0], totals, along)
    landing = float(means.merged_total(jnp.asarray(totals), jnp.asarray(anchors), units, 0)[1])
    significand, exponent = windows._factors_value(jnp.asarray(anchors), jnp.full((1,), landing, jnp.float32),
                                                  units=units)
    factors = np.ldexp(np.asarray(significand, np.float64), np.asarray(exponent))
    value = _unsigned_land(np.sum(totals.astype(np.float64) * factors))
    assert_bits(got, [value, _unsigned_land(np.sum(factors * along)), 0.0], "value, first, second")
    assert np.isfinite(got[0]), f"{case}: a total landed past the format"


def test_applied_preserves_single_rounding_under_gradual_underflow():
    """`scaled.applied` uses `precision.power2`'s single-rounding path so an in-range negative exponent
    applies one exact power of two before the significand (never splitting `2^-4` into two `2^-2` factors
    that double-round a subnormal on a gradual-underflow backend)."""
    x = np.uint32(0x00800009).view(np.float32)
    y = np.float32(15.0 / 256.0)
    got = np.asarray(scaled.applied(jnp.asarray(x), scaled.scaled(jnp.asarray(y))), np.float32)
    probe = np.asarray(jnp.asarray(x) * jnp.asarray(np.float32(2.0 ** -4)), np.float32)
    if probe != 0.0:
        assert int(got.view(np.uint32)) == 0x00078001
    else:
        assert int(got.view(np.uint32)) == 0x00000000


def _unprotected_primitives(jaxpr):
    """Primitives reachable from `jaxpr` recursing through all nested bodies except `Protected` spans
    (preserving precision-transform barriers while checking that no window loop or barrier leaks out)."""
    from dew.nn.derivatives import Protected, marked_call_p
    from dew.nn.programs import bodies

    seen = set()
    stack = [jaxpr]
    while stack:
        body = stack.pop()
        for eq in body.eqns:
            seen.add(eq.primitive)
            if eq.primitive is marked_call_p and isinstance(eq.params["kind"], Protected):
                continue
            stack.extend(bodies(eq))
    return seen


def test_singleton_ratios_emit_no_window_loop_and_hold_degenerate_values():
    """Any 1-state ratio (`means.finished`, `means.resumed`, 1-state `merged_output`, 1-state `merged_state`,
    and an arbitrary signed 1-state `ratios.Ratio`) lowers through `_streamed`'s general singleton path
    without a `while` loop anywhere or any unprotected window `optimization_barrier`, and matches float64
    truth at zero, negative zero, subnormals, and extreme magnitudes (rows 12223, 12269, 12331)."""
    from dew.nn.programs import equations

    signed_form = ratios.Ratio(numerator=((-1.0, (("v", 0),)),), denominator=((1.0, (("v", 1),)),),
                               axis=0, reduced=True, units=windows.Units())
    a = np.asarray([[0.0, -0.0, SUBNORMAL, LEAST, 1.5, -2.0 ** 100, LARGEST]], np.float32)
    b = np.asarray([[2.0]], np.float32)
    for sum_fn, args in (
        (lambda x, y: ratios._ratio_sums((x, y), means._FINISHED), (a, b[:, 0])),
        (lambda x, y: ratios._ratio_sums((x, y), means._RESUMED), (a, b[:, 0])),
        (lambda x, y: ratios._ratio_sums((x, y), signed_form), (a, b)),
    ):
        traced = jax.make_jaxpr(sum_fn)(*args).jaxpr
        all_prims = {eq.primitive for eq in equations(traced)}
        assert jax.lax.while_p not in all_prims
        assert jax.lax.optimization_barrier_p not in _unprotected_primitives(traced)
    merged_traced = jax.make_jaxpr(
        lambda o, w, m: means._merged_state_value(o, w, m, units=windows.Units(), axis=0))(
            a, b[:, 0], np.asarray([-3.0], np.float32)).jaxpr
    assert jax.lax.while_p not in {eq.primitive for eq in equations(merged_traced)}
    expected_signed = np.asarray(
        [_unsigned_land(-float(v) / 2.0) for v in a[0].astype(np.float64)], np.float32)
    got_signed = np.asarray(jax.jit(lambda x, y: ratios.ratio([x, y], signed_form))(a, b))
    assert_bits(got_signed, expected_signed, "signed singleton ratio")
    expected_id = np.asarray([_unsigned_land(float(v)) for v in a[0].astype(np.float64)], np.float32)
    one_out = jax.jit(means.merged_output, static_argnums=(3, 4))(
        a, np.asarray([2.0], np.float32), np.asarray([0.0], np.float32), windows.Units(), 0)
    assert_bits(np.asarray(one_out), expected_id, "1-state merged_output")
    one_state = jax.jit(means.merged_state, static_argnums=(3, 4))(
        a, np.asarray([2.0], np.float32), np.asarray([-3.0], np.float32), windows.Units(), 0)
    assert_bits(np.asarray(one_state[0]), expected_id, "1-state merged_state output")
    assert_bits(np.asarray(one_state[1]), np.float32(2.0), "1-state merged_state total")
    assert_bits(np.asarray(one_state[2]), np.float32(-3.0), "1-state merged_state anchor")
