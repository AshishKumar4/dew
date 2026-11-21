"""The scaled value and the convex mean over it (docs/design/algebra.md §6.12), by what each holds exactly:
the construct's arithmetic lands the float the reals' result is where its intermediates pass fp32's range; the
merge of finished states at fp32's edges (the cases of the five review rounds, the attention lane's
handoff/merge_cases.md) stays within its outputs' hull and finite where its value is, keeps what exactly
opposite terms leave, and its second derivative is the law's own, jitted and op by op. Its accuracy against
float64 is held through the gate (check 1: the finished merge's cells of tests/test_attention_algebra.py) and
every order's finiteness by its contract (tests/test_contracts_algebra.py)."""

import functools
import itertools

import hypothesis
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import strategies as st
from reference_error import ahead_arm, as_given, assert_as_exact, dew_path, ran_here

from dew.algebra import means, scaled
from dew.algebra.forms import Refused
from dew.parity import FEWEST_SEEDS

LARGEST = float(np.finfo(np.float32).max)


NATURAL = means.Units()
"""Natural anchors, compensated: the accurate form's units (`means.Units`)."""


def _merge(outputs, totals, anchors):
    """The law over natural anchors along the states' first axis."""
    return means.merged_output(outputs, totals, anchors, NATURAL, 0)


def _ways(fn, label: str):
    """`fn` each way a cell runs the law (`WAYS`): op by op, eagerly, and jitted as an arm a GPU tier part's
    prepare compiles ahead at its host inputs and its run runs from that program (`reference_error.ahead_arm`,
    `fn` jitted outside a part), under `label`, one program a label. A cell of several arms runs each before
    it judges any: a prepare's arm gives zeros, and a check between two would fail there, the arms after it
    never compiled ahead."""
    def eager(*args):
        with jax.disable_jit():
            return fn(*args)
    jitted = ahead_arm(jax.jit(fn), label)
    return {"jitted": jitted, "op by op": eager, "on a gpu": jitted, "on a tpu": jitted}


CPU_LANE = pytest.mark.cpu(reason="eager dispatch and the CPU lane's jit: a GPU's tier runs the law on a gpu")
"""The ways the CPU lane alone takes (`WAYS`): op by op, whose eager dispatch no prepare compiles ahead, and
jitted here, which a GPU's tier takes as its own way."""

WAYS = [pytest.param("jitted", marks=CPU_LANE), pytest.param("op by op", marks=CPU_LANE),
        pytest.param("on a gpu", marks=pytest.mark.gpu(reason="the primitives on a GPU")),
        pytest.param("on a tpu", marks=pytest.mark.tpu(reason="the primitives on a TPU"))]
"""Each way a cell runs the law: jitted and op by op on the CPU lane, jitted on a GPU and on a TPU on their
tiers, each from the programs its tier's prepare compiled ahead (`_ways`)."""


def test_a_scaled_value_lands_as_the_float_it_holds():
    values = np.asarray([0.0, 1.0, -1.5, 3.0, 2.0 ** -126, -(2.0 ** -100), 2.0 ** 100, LARGEST, -LARGEST],
                        np.float32)
    landings = ran_here(lambda x: scaled.landed(scaled.scaled(x)), "scaled values landed")
    found = [run(values) for run in landings]
    for landed in found:
        np.testing.assert_array_equal(np.asarray(landed), values)


def test_a_product_and_a_quotient_past_the_range_land_their_value():
    """(2^100 2^100) / 2^150 and (2^-100 2^-100) 2^150 exactly: their products are past fp32's range on the
    way, as the masses of the merge's cases are."""
    def landed(*values):
        big, small, step = (scaled.scaled(value) for value in values)
        return (scaled.landed(scaled.div(scaled.mul(big, big), scaled.mul(step, step))),
                scaled.landed(scaled.mul(scaled.mul(small, small), scaled.mul(step, step))))

    values = tuple(np.float32(2.0 ** e) for e in (100, -100, 75))
    found = [run(*values) for run in ran_here(landed, "products and quotients past the range")]
    assert all((float(high), float(low)) == (2.0 ** 50, 2.0 ** -50) for high, low in found)


def test_a_product_by_a_scaled_coefficient_moves_one_way_to_its_value():
    """fp32's largest by 3/4 is 3/4 of it, where the coefficient's significand first (1.5) would pass the
    format; 2^-120 by 2^200 / 2^150 is 2^-70, where 2^-120 by 2^-150 first would flush."""
    def applied(three, four, largest, high, low, least):
        three_quarters = scaled.div(scaled.scaled(three), scaled.scaled(four))
        coefficient = scaled.div(scaled.div(scaled.scaled(high), scaled.scaled(low)), scaled.scaled(high))
        return scaled.applied(largest, three_quarters), scaled.applied(least, coefficient)

    largest = np.float32(LARGEST)
    values = (np.float32(3.0), np.float32(4.0), largest, *(np.float32(2.0 ** e) for e in (100, -50, -120)))
    found = [run(*values) for run in ran_here(applied, "a product by a scaled coefficient")]
    assert all((float(near), float(far)) == (largest * np.float32(0.75), 2.0 ** -70) for near, far in found)


def test_a_form_past_its_exponents_reach_is_refused():
    """Squaring a value 21 times takes its exponent's bound past `REACH`: refused, never emitted."""
    def squared(x):
        value = scaled.scaled(x)
        for _ in range(21):
            value = scaled.mul(value, value)
        return scaled.landed(value)

    # Refused as it is traced, nothing computing (a GPU tier part's prepare computes nothing).
    with pytest.raises(Refused):
        jax.eval_shape(squared, jax.ShapeDtypeStruct((), jnp.float32))


def test_the_compensated_sum_keeps_what_cancels_in_every_order():
    """+-1.7 2^126 / 3 beside 1.3 2^-126: the residual exactly, whatever the order of the terms."""
    large, tiny = np.float32(1.7 * 2.0 ** 126 / 3), np.float32(1.3 * 2.0 ** -126)

    def kept(terms):
        summed, errors = scaled.compensated(list(jnp.asarray(terms, jnp.float32)))
        return summed + errors

    sums = ran_here(kept, "a compensated sum of cancelling terms")
    found = [run(np.asarray(order, np.float32)) for order in itertools.permutations([large, -large, tiny])
             for run in sums]
    assert all(float(value) == tiny for value in found)


L = LARGEST
CASES = {
    "V1, outputs of both signs near fp32's largest, masses 1 and 1/3": ([[0.9 * L], [-0.8 * L]], [1.0, 1 / 3],
                                                                        [0.0, 0.0]),
    "V2, sixteen masses near 2^124": ([[1.0 + k / 16] for k in range(16)], [2.0 ** 124] * 16, [0.0] * 16),
    "V3, two masses near 2^127": ([[1.0], [1.2]], [1.5 * 2.0 ** 126, 1.9 * 2.0 ** 126], [0.0, 0.0]),
    "V5, masses in [2^126, 2^127]": ([[1.0], [1.0 + 2.0 ** -10]], [2.0 ** 126, 1.7 * 2.0 ** 126], [0.0, 0.0]),
    "V6, an anchor 100 below the joined one, its total 2^120": ([[1.0], [2.0]], [1.0, 2.0 ** 120],
                                                                [0.0, -100.0]),
    "V7, an empty state beside a mass of 2^127": ([[5.0], [1.0]], [0.0, 1.5 * 2.0 ** 126], [3.0, 0.0]),
    "two outputs at fp32's largest, shares 29/30 and 1/30": ([[L], [L]], [29.0, 1.0], [0.0, 0.0]),
}


def _hull(outputs, totals):
    """The outputs of positive mass' least and largest, channel by channel."""
    held = outputs[totals > 0]
    return held.min(0), held.max(0)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("way", WAYS)
def test_a_merge_at_fp32s_edges_stays_in_its_hull(case, way):
    """Each case's merged output is finite, within the hull of its outputs of positive mass (a state of no
    mass weighed nothing, V7's 5 never read), where the forms of the five rounds each left it for an
    overflow, a flush or a cancellation."""
    outputs, totals, anchors = (np.asarray(x, np.float32) for x in CASES[case])
    mean = np.asarray(_ways(_merge, "a merge at fp32's edges")[way](outputs, totals, anchors))
    least, most = _hull(outputs, totals)
    assert np.all(np.isfinite(mean)) and np.all(mean >= least) and np.all(mean <= most)


def test_exactly_opposite_terms_leave_the_rest_exactly():
    """V4: outputs +-1.7 2^126 beside 3.9 2^-126 at equal masses: the mean is the third output's term alone,
    its output by a third (`scaled.applied`), jitted and op by op, whatever the states' order."""
    large, tiny = 1.7 * 2.0 ** 126, 3.9 * 2.0 ** -126

    def a_third(x, three):
        return scaled.applied(x, scaled.div(scaled.one_like(three), scaled.scaled(three)))

    # Every arm run before any is judged (`ran_here`): each way's third and its merges, judged by its own.
    thirds = [run(np.float32(tiny), np.float32(3.0)) for run in ran_here(a_third, "a third of the rest")]
    merges = ran_here(_merge, "opposite terms beside a third")
    found = [[merged(np.asarray(order, np.float32)[:, None], np.ones(3, np.float32), np.zeros(3, np.float32))
              for order in itertools.permutations([large, -large, tiny])] for merged in merges]
    for third, merged in zip(thirds, found, strict=True):
        assert all(float(mean[0]) == float(third) for mean in merged)


def test_a_window_a_sum_completed_is_not_summed_again():
    """Five states of masses `2^-120` at anchors `[0, 0, -200, -200, -400]`: a channel of `2^-120` in its
    first state alone, and one of `±2^120` in turn, whose pairs that cancel lie in windows apart and keep the
    windows going (`means._windows`). The first channel's sole term is summed in its first window and its
    next finds none; a window after that read the term again, its ceiling raised by the empty one (`2^-120`,
    the merge P0's head review, F1). Its mean is its output over the two states at the largest anchor,
    `2^-121` exactly; the second's, `2^120 e^-400` over them, beneath float32's least, jitted and op by
    op."""
    outputs = np.asarray([[2.0 ** -120, 2.0 ** 120], [0.0, -(2.0 ** 120)], [0.0, 2.0 ** 120],
                          [0.0, -(2.0 ** 120)], [0.0, 2.0 ** 120]], np.float32)
    masses = np.full(5, 2.0 ** -120, np.float32)
    anchors = np.asarray([0.0, 0.0, -200.0, -200.0, -400.0], np.float32)
    for merged in ran_here(_merge, "windows a sum completed"):
        np.testing.assert_array_equal(np.asarray(merged(outputs, masses, anchors)), [2.0 ** -121, 0.0])


def test_a_subnormal_output_weighs_in_the_merge_and_its_derivative():
    """Outputs `2^-127` and `2^-125` at equal anchors and unit masses merge to their mean, `1.25 2^-126`, a
    normal float the subnormal output is part of; and the merge of outputs `0` and `2^-130` moves along a
    mass direction `[2^127, 0]` by `-2^-130 2^127 / 4 = -2^-5`. A subnormal read as zero gave `2^-126` and
    `0` (`scaled.scaled`, the merge P0's head review, F2), jitted and op by op."""
    unit, level = np.ones(2, np.float32), np.zeros(2, np.float32)
    outputs = np.asarray([[0.0], [2.0 ** -130]], np.float32)
    along = np.asarray([2.0 ** 127, 0.0], np.float32)

    def moved(totals):
        return jax.jvp(lambda totals: _merge(outputs, totals, level), (totals,), (along,))[1]

    # Every arm run before any is judged (`_ways`).
    merged_means = [merged(np.asarray([[2.0 ** -127], [2.0 ** -125]], np.float32), unit, level)
              for merged in ran_here(_merge, "subnormal outputs merged")]
    tangents = [run(unit) for run in ran_here(moved, "a subnormal output's tangent")]
    assert all(float(mean[0]) == 1.25 * 2.0 ** -126 for mean in merged_means)
    assert all(float(tangent[0]) == -(2.0 ** -5) for tangent in tangents)


def test_a_subnormal_total_holds_its_mass_and_a_negative_one_none():
    """A state's mass read from its stored sign and magnitude (`means._mass`), where a comparison on a lane
    that flushes subnormal operands (XLA's CPU) reads a subnormal total as a zero of its sign: a finished
    state of contraction and total `2^-149` is 1 (it divided by one); states of equal subnormal masses and
    outputs 1 and 3 merge to 2, alone and as one operation with their total (it was 0, their hull empty);
    and totals `2^-125` beside `-2^-127` at equal anchors merge to `2^-125`, the negative one weighing none
    (it gave `3 2^-127`, read as negative by the split): the merge P0's head review 2, jitted and op by op."""
    least = np.float32(2.0 ** -149)
    level = np.zeros(2, np.float32)
    outputs, masses = np.asarray([[1.0], [3.0]], np.float32), np.full(2, least, np.float32)
    totals = np.asarray([2.0 ** -125, -(2.0 ** -127)], np.float32)
    # Every arm run before any is judged (`_ways`).
    finished = [run(np.full(1, least, np.float32), least)
                for run in ran_here(lambda x, total: means.finished(x, total), "a subnormal state finished")]
    merged = [run(outputs, masses, level) for run in ran_here(_merge, "subnormal masses merged")]
    states = [run(outputs, masses, level) for run in ran_here(
        lambda o, w, m: means.merged_state(o, w, m, NATURAL, 0)[0], "subnormal masses' merged state")]
    landed = [run(totals, level) for run in ran_here(lambda w, m: means.merged_total(w, m, NATURAL, 0)[0],
                                                  "a negative subnormal total beside a positive one")]
    assert all(float(value[0]) == 1.0 for value in finished)
    assert all(float(value[0]) == 2.0 for value in (*merged, *states))
    assert all(float(value) == 2.0 ** -125 for value in landed)


MIXED = {
    # Each: outputs, totals, a mass direction, then an output direction; the mixed second derivative.
    "D13, equal outputs at unit masses, along 2^60 and 2^67": (
        [[0.0], [0.0]], [1.0, 1.0], [2.0 ** 60, 0.0], [[2.0 ** 67], [0.0]], 2.0 ** 125),
    "D13 at masses 1 and 3": (
        [[0.0], [0.0]], [1.0, 3.0], [2.0 ** 60, 0.0], [[2.0 ** 67], [0.0]], 3 * 2.0 ** 123),
    "a subnormal gap at unit masses": ([[0.0], [2.0 ** -130]], [1.0, 1.0], [1.0, 0.0], [[0.0], [1.0]], -0.25),
}


@pytest.mark.parametrize("case", list(MIXED))
@pytest.mark.parametrize("way", WAYS)
def test_the_merges_mixed_second_derivative_is_its_laws(case, way):
    """Along a mass and then an output, the law's own: D13's 2^125, where a form through `frexp` is
    infinite; at masses 1 and 3, its mean's derivative its own, where the hull's clip, tied at equal outputs,
    split it by halves (the review's D13 at drawn masses, 8% off); at a subnormal gap -1/4, the gap's
    derivative its tangents' difference (`means.gapped_p`), never read off the gap's exponent."""
    outputs, totals, along_mass, along_output, expected = MIXED[case]
    operands = (np.asarray(x, np.float32) for x in (outputs, totals, along_mass, along_output))
    assert float(_ways(_mixed_along, "the merge's mixed second derivative")[way](*operands)[0]) == expected


def _mixed_along(o, w, along_mass, along_output):
    """The merge (anchors 0) differentiated along `along_mass` and then along `along_output`: one program for
    every case of its shapes, its directions arguments, so a jit compiles it once."""
    def along(o, w):
        return jax.jvp(lambda each: _merge(o, each, jnp.zeros_like(each)), (w,), (along_mass,))[1]

    return jax.jvp(lambda moved: along(moved, w), (o,), (along_output,))[1]


def _gradients_derivatives_along(o, w, along_outputs, along_masses):
    """The gradient of the first merged channel as to the totals (anchors 0), differentiated along
    `along_outputs` and along `along_masses` forward over reverse and reverse over reverse: one program for
    every case of its shapes, its directions arguments, so a jit compiles it once."""
    def gradient(o, w):
        return jax.grad(lambda each: _merge(o, each, jnp.zeros_like(each))[0])(w)

    along_o = jax.jvp(lambda moved: gradient(moved, w), (o,), (along_outputs,))[1]
    forward = jax.jvp(lambda moved: gradient(o, moved), (w,), (along_masses,))[1]
    reverse = jax.grad(lambda moved: jnp.vdot(gradient(o, moved), along_masses))(w)
    return along_o, forward, reverse


GRADIENTS = {
    # Each: outputs, totals, an output direction, a mass direction; the jvp of the gradient as to the totals
    # along the outputs, then along the masses forward over reverse, then reverse over reverse.
    "a zero gap": ([[0.5], [0.5]], [1.0, 1.0], [[1.0], [0.0]], [1.0, 1.0], [0.25, -0.25], [0.0, 0.0]),
    "a zero mass": ([[0.5], [-0.5]], [1.0, 0.0], [[1.0], [0.0]], [1.0, 1.0], [0.0, -1.0], [1.0, 3.0]),
    "a subnormal gap": ([[0.0], [2.0 ** -130]], [1.0, 1.0], [[0.0], [1.0]], [0.0, 0.0], [-0.25, 0.25],
                        [0.0, 0.0]),
}


@pytest.mark.parametrize("case", list(GRADIENTS))
@pytest.mark.parametrize("way", WAYS)
def test_the_derivatives_of_the_merges_gradient_are_the_laws(case, way):
    """The gradient as to the totals, differentiated along the outputs and along the masses, forward over
    reverse and reverse over reverse, the law's (`merged_p`, its derivative and its terms' primitives of
    their own, which a reverse pass keeps): at a zero gap 1/4 where a reverse pass that inlined the
    derivative's primal differentiated its frexp to 2; a state of no mass moved by its own total and gap
    (-1, and the Hessian's product 1 and 3); a subnormal gap finite (train 943's review)."""
    outputs, totals, along_outputs, along_masses, mixed, hessian = (np.asarray(x, np.float32)
                                                                    for x in GRADIENTS[case])
    along_o, forward, reverse = (np.asarray(x) for x in _ways(
        _gradients_derivatives_along, "the merge's gradient's derivatives")[way](
        outputs, totals, along_outputs, along_masses))
    assert along_o.tolist() == mixed.tolist()
    assert forward.tolist() == hessian.tolist() and reverse.tolist() == hessian.tolist()


@pytest.mark.parametrize("way", WAYS)
def test_the_merges_derivative_along_a_common_scaling_of_its_masses_is_zero(way):
    """Equal masses under outputs c +- s (s 2^100, c 2^90), along a common scaling of the masses (each mass's
    tangent 2^100 times itself): 0, the mean the same at any common scale, its two terms 2^199 apart past
    fp32 and cancelling at a power of two common to the row (`means.combined_p`); along the outputs by ones,
    1 (the attention template's acceptance cell)."""
    s, c = 2.0 ** 100, 2.0 ** 90
    outputs = np.asarray([[c + s], [c - s]], np.float32)
    totals, anchors = np.ones(2, np.float32), np.zeros(2, np.float32)

    def along(o, w):
        masses = jax.jvp(lambda moved: _merge(o, moved, anchors), (w,), (2.0 ** 100 * w,))[1]
        ones = jax.jvp(lambda moved: _merge(moved, w, anchors), (o,), (jnp.ones_like(o),))[1]
        return masses, ones

    masses, ones = _ways(along, "a common scaling of the masses")[way](outputs, totals)
    assert (float(masses[0]), float(ones[0])) == (0.0, 1.0)


def _relanded(first, second, anchor: float):
    """Two zero outputs' states of totals `first` at `anchor`, merged (`means.merged_state`), then merged
    with a third of output 1 and total `second` at `anchor`: the third's share of the three, its landed
    state's total moved to the anchor the law chose for it."""
    def merged(totals):
        output, total, landed = means.merged_state(jnp.zeros((2, 1)), totals, jnp.full(2, anchor), NATURAL, 0)
        return means.merged_output(jnp.stack([output, jnp.ones(1)]), jnp.stack([total, jnp.float32(second)]),
                                   jnp.stack([landed, jnp.float32(anchor)]), NATURAL, 0)
    return merged


@pytest.mark.parametrize(("total", "anchor"),
                         [(2.0 ** 127, 0.0), (2.0 ** 127, 2.0 ** 30), (2.0 ** 126, 2.0 ** 31)],
                         ids=["past, at zero", "past, at 2^30", "within, at 2^31"])
@pytest.mark.parametrize("way", WAYS)
def test_a_merged_total_lands_at_an_anchor_that_keeps_its_mass(way, total, anchor):
    """Two totals at one anchor, merged (`means.merged_total`), then merged with a third state of output 1
    and the same total at that anchor: the three weigh it a third, as merged at once. Two of 2^127 sum past
    fp32's largest, and the law lands the total at an anchor above it: where the anchor is 2^30, its
    neighbours 128 apart, a move by the powers of two the sum passed by rounded to none and the merged
    state weighed it a half (base review F2). Two of 2^126 at 2^31 sum to 2^127, within the format, kept at
    their anchor: moved by its spacing, 256, the total landed at 0 and the third weighed it whole (the
    re-review's F1). Totals 1 and 2 at one anchor land as 3 at it."""
    # Both arms run before either is judged (`_ways`).
    relanded = _ways(_relanded(total, total, anchor), "a total relanded at its anchor")[way](
        np.full(2, total, np.float32))
    landed, at = _ways(lambda totals, anchors: means.merged_total(totals, anchors, NATURAL, 0),
                       "totals landed at one anchor")[way](
        np.asarray([1.0, 2.0], np.float32), np.full(2, anchor, np.float32))
    assert float(relanded[0]) == float(np.float32(1 / 3))
    assert (float(landed), float(at)) == (3.0, anchor)


@pytest.mark.parametrize("way", WAYS)
def test_a_zero_total_merged_moves_the_next_merge_along_its_total(way):
    """Outputs 0 and 0, totals 1 and `t`, anchors 0, merged (`means.merged_state`), then merged with output
    1, total 1, anchor 0: the mean is 1 / (2 + t), its derivative along `t` at 0 is -1/4, where weighing a
    zero total as none in the merged total moved it nothing (the re-review's F2, row 7124's class)."""
    def composed(t):
        output, total, anchor = means.merged_state(jnp.zeros((2, 1)), jnp.stack([jnp.float32(1.0), t]),
                                                   jnp.zeros(2), NATURAL, 0)
        return means.merged_output(jnp.stack([output, jnp.ones(1)]), jnp.stack([total, jnp.float32(1.0)]),
                                   jnp.stack([anchor, jnp.float32(0.0)]), NATURAL, 0)[0]

    assert float(_ways(jax.grad(composed), "a zero total's merge moved")[way](np.float32(0.0))) == -0.25


@pytest.mark.parametrize("way", WAYS)
def test_a_state_negligible_beside_a_large_output_weighs_nothing(way):
    """Outputs 0 and 2^127, totals 2^-126 and 2^127, anchors 0 and -1000 (`means.merged_output`): the second
    state's mass is 2^-1316 beside the first's 2^-126, its weight in the mean 2^-1190, the mean 2^-1063,
    which rounds to 0; a factor held at a floor of 2^-378 merged them to 4 (base review F1). Its anchor a
    billion below, past `means.FLOOR`, it weighs nothing too."""
    for far in (-1000.0, -1e9):
        merged = _ways(_merge, "a state far below")[way](np.asarray([[0.0], [2.0 ** 127]], np.float32),
                                    np.asarray([2.0 ** -126, 2.0 ** 127], np.float32),
                                    np.asarray([0.0, far], np.float32))
        assert float(merged[0]) == 0.0


def test_a_finished_and_a_resumed_state_land_once():
    """V5 and V8: a contraction of 1.5 2^126 over a total of 2^127, whose reciprocal is subnormal, finishes to
    3/4 where a division through that reciprocal flushed; a mean of 0.8 over a total of 1.5 2^126 resumes to
    its contraction, each one product by a scaled value."""
    total = np.asarray([1.5 * 2.0 ** 126], np.float32)
    # Every arm run before any is judged (`_ways`).
    finished = [finish(np.asarray([[1.5 * 2.0 ** 126]], np.float32), np.asarray([2.0 ** 127], np.float32))
                for finish in ran_here(means.finished, "a state finished past the range")]
    resumed = [resume(np.asarray([[0.8]], np.float32), total)
               for resume in ran_here(means.resumed, "a state resumed past the range")]
    assert all(float(value[0, 0]) == 0.75 for value in finished)
    assert all(float(value[0, 0]) == np.float32(0.8) * total[0] for value in resumed)


@pytest.mark.cpu(reason="Hypothesis draws apart in each process, which no GPU part's prepare compiles "
                        "ahead: the ways' cells hold the hull on a GPU (test_a_merge_at_fp32s_edges_...)")
@hypothesis.settings(max_examples=60, deadline=None)
@hypothesis.given(count=st.integers(1, 12), data=st.data())
def test_any_merge_stays_in_its_hull(count, data):
    """Any states the draw takes, outputs and masses spanning fp32's range, anchors 60 apart, an empty state
    among them at times: the merged output finite and within its outputs' hull."""
    magnitudes = st.floats(-126, 127)
    outputs = np.asarray([[np.sign(data.draw(st.floats(-1, 1))) * 2.0 ** data.draw(magnitudes)]
                          for _ in range(count)], np.float32)
    totals = np.asarray([0.0 if k and data.draw(st.booleans()) else 2.0 ** data.draw(magnitudes)
                         for k in range(count)], np.float32)
    totals[0] = max(totals[0], 2.0 ** -126)
    anchors = np.asarray([data.draw(st.floats(-60, 0)) for _ in range(count)], np.float32)
    mean = np.asarray(jax.jit(_merge)(outputs, totals, anchors))
    least, most = _hull(outputs, totals)
    assert np.all(np.isfinite(mean)) and np.all(mean >= least) and np.all(mean <= most)


def _beside_edges(count: int, signs: str) -> np.ndarray:
    """`count` splits' outputs of two channels, one near fp32's largest and one near its least normal (or
    2^125 and 2^-125 in every split): the largest's signs all alike, alternating, or the first half's
    positive and the rest negative."""
    rng = np.random.default_rng(count)
    if signs == "2^125 beside 2^-125":
        return np.stack([np.full(count, 2.0 ** 125), np.full(count, 2.0 ** -125)], 1).astype(np.float32)
    sign = {"alike": np.ones(count), "alternating": np.where(np.arange(count) % 2, 1.0, -1.0),
            "grouped": np.where(np.arange(count) < count // 2, 1.0, -1.0)}[signs]
    least = float(np.finfo(np.float32).tiny)
    return np.stack([sign * LARGEST * (1 - rng.random(count) * 2.0 ** -10), least * (1 + rng.random(count))],
                    1).astype(np.float32)


@pytest.mark.parametrize("count", [8, 16])
@pytest.mark.parametrize("signs", ["2^125 beside 2^-125", "alike", "alternating", "grouped"])
@pytest.mark.parametrize("way", WAYS)
def test_a_channel_beside_either_edge_keeps_what_an_lse_scaled_combine_keeps(count, signs, way):
    """`count` splits of equal mass (equal LSEs; a power of two of them, so each share is exact), one channel
    near fp32's largest beside one near its least normal: each channel's merge is its mean correctly rounded
    (a channel's values share their exponent within a few, so their float64 mean is exact), each at its own
    power, and its derivative along the totals (each moved by `count / 2`, the grouped gaps' terms past the
    largest together) finite in both, jitted and as written. An LSE-scaled combine (FlashAttention's
    split-KV) rounds each split's product and each partial sum, its grouped derivative past the largest, and
    XLA:CPU flushes its subnormal products, so neither channel loses a bit such a combine keeps: a power
    shared by the row's channels or taken from its coefficients alone left the least normal's terms
    subnormal and lost 20 to 30% of it."""
    outputs = _beside_edges(count, signs)
    along = np.full(count, count / 2, np.float32)

    def merged(outputs, totals):
        return jax.jvp(lambda totals: means.merged_output(outputs, totals, np.zeros(count, np.float32),
                                                          NATURAL, 0), (totals,), (along,))

    mean, moved = _ways(merged, "a channel beside either edge")[way](outputs, np.ones(count, np.float32))
    np.testing.assert_array_equal(mean, outputs.astype(np.float64).mean(0).astype(np.float32))
    assert np.all(np.isfinite(moved))


def _batch(seed: int, count: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`count` merges of four states of eight channels each: outputs, totals and anchors over fp32's range."""
    rng = np.random.default_rng(seed)
    outputs = (rng.standard_normal((count, 4, 8)) * 2.0 ** rng.integers(-60, 60, (count, 4, 1)))
    totals = 2.0 ** rng.integers(-100, 100, (count, 4)) * rng.uniform(1, 2, (count, 4))
    anchors = rng.uniform(-30, 0, (count, 4))
    return tuple(np.asarray(x, np.float32) for x in (outputs, totals, anchors))


def _gradient(merge):
    """The gradient of `merge`'s outputs' sum as to its outputs, totals and anchors."""
    return jax.grad(lambda *state: jnp.sum(merge(*state)), argnums=(0, 1, 2))


def _batched(outputs, totals, anchors):
    """The law over a leading batch of merges, its states along axis 1."""
    return means.merged_output(outputs, totals, anchors, NATURAL, 1)


@functools.cache
def _state(axis: int):
    """The merged state of states along `axis` (`means.merged_state`), one function an axis, so a program of
    it traced and compiled for one cell serves every other at its shapes (rows 9806 and on: the merged
    state's cells compiled each draw's arms again)."""
    return lambda outputs, totals, anchors: means.merged_state(outputs, totals, anchors, NATURAL, axis)


@functools.cache
def _apart(axis: int):
    """The merged output and the merged total and its landed anchor of states along `axis`, apart, one
    function an axis (`_state`)."""
    return lambda outputs, totals, anchors: (means.merged_output(outputs, totals, anchors, NATURAL, axis),
                                             *means.merged_total(totals, anchors, NATURAL, axis))


def _zero_massed(operands):
    """`operands` with each merge's second state of no total."""
    outputs, totals, anchors = operands
    return outputs, totals * np.asarray([1, 0, 1, 1], np.float32), anchors


def _on_par(state, apart, way: str, draw, label: str) -> None:
    """`state` held on par with `apart` from float64 (check 1, the gate's `assert_as_exact`, `apart` one of
    Dew's own paths), each run `way` (`_ways`; op by op both as given): two programs, which XLA promises no
    common bits, held so and not bit for bit (docs/design/testing.md; the merge P0's base review 11, F10).
    Each draw's arms read alike in every draw, so a pair of one value resolves at the fewest seeds."""
    if way == "op by op":
        state, apart = (as_given(_ways(fn, f"{label}, {role}")["op by op"])
                        for fn, role in ((state, "state"), (apart, "apart")))
    assert_as_exact(state, dew_path(apart), draw, f"{label}, {way}", seeds=FEWEST_SEEDS)


def _with_directions(draw):
    """`draw`'s operands and a direction along each, drawn from its seed (the draw holds what the claim
    varies)."""
    def drawn(seed):
        operands = draw(seed)
        rng = np.random.default_rng(seed + 1)
        return (*operands, *(np.asarray(rng.standard_normal(x.shape), np.float32) for x in operands))
    return drawn


@functools.cache
def _value(form):
    """`form`'s merged output and total at a draw's operands, its directions read by none; one function a
    form, as `_state`'s are, so each cell's arms reuse another's programs."""
    return lambda outputs, totals, anchors, *_: form(outputs, totals, anchors)[:2]


@functools.cache
def _moved(form, at: tuple[int, ...]):
    """`form`'s merged output and total moved along its operands `at` by a draw's directions (a forward
    derivative), one function a form and a choice of operands (`_value`)."""
    def ran(*xs):
        operands, along = xs[:3], xs[3:]

        def partial(*ys):
            placed = list(operands)
            for index, y in zip(at, ys, strict=True):
                placed[index] = y
            return form(*placed)[:2]
        return jax.jvp(partial, *(tuple(each[index] for index in at) for each in (operands, along)))[1]
    return ran


@functools.cache
def _second(form):
    """The forward derivative along the totals of the gradient of `form`'s output sum as to its totals, one
    function a form (`_value`)."""
    def ran(outputs, totals, anchors, _, along_totals, __):
        def gradient(totals):
            return jax.grad(lambda totals: jnp.sum(form(outputs, totals, anchors)[0]))(totals)
        return jax.jvp(gradient, (totals,), (along_totals,))
    return ran


@functools.cache
def _batched_form(made):
    """`made(0)` mapped over a batch (`jax.vmap`), one function a form (`_value`)."""
    return jax.vmap(made(0))


MERGED_DRAWS = {"drawn": _batch, "a zero mass": lambda seed: _zero_massed(_batch(seed))}
"""The merged state's draws: three merges of four states over fp32's range (`_batch`), and the same with a
state of no total (the attention merge's zero-mass draw: an operand held is never moved by a zero, whose term
would be a zero mass's factor past the format, row 7124)."""

MERGED_SITES = {
    "its value": lambda: (_value(_state(1)), _value(_apart(1))),
    "along every operand": lambda: (_moved(_state(1), (0, 1, 2)), _moved(_apart(1), (0, 1, 2))),
    "along its outputs": lambda: (_moved(_state(1), (0,)), _moved(_apart(1), (0,))),
    "along its totals": lambda: (_moved(_state(1), (1,)), _moved(_apart(1), (1,))),
    "along its anchors": lambda: (_moved(_state(1), (2,)), _moved(_apart(1), (2,))),
    "its gradient's derivative": lambda: (_second(_state(1)), _second(_apart(1))),
    "batched": lambda: (_value(_batched_form(_state)), _value(_batched_form(_apart))),
}
"""The merged state's sites, each its form and its parts' (`_state`, `_apart`): a cell a site, its draws in
it, so each compiles its own programs once (row 10154: a cell a draw compiled every site's, 250 to 280 s a
cell past the group target, its two draws' programs alike)."""


@pytest.mark.parametrize("site", MERGED_SITES)
@pytest.mark.parametrize("way", WAYS)
def test_a_merged_state_is_on_par_with_its_output_and_its_total_and_their_derivatives(way, site):
    """The merged state (`means.merged_state`, one operation a template may run as one kernel) is on par with
    the merged output and the merged total from float64 (`_on_par`), and its derivatives with theirs: along
    every operand and along each alone, the gradient's derivative along the totals, and batched, each a
    site (`MERGED_SITES`), at each draw (`MERGED_DRAWS`)."""
    state, apart = MERGED_SITES[site]()
    for name, draw in MERGED_DRAWS.items():
        _on_par(state, apart, way, _with_directions(draw),
                f"a merged state against its output and its total, {site}, {name}")


@pytest.mark.parametrize("way", WAYS)
def test_a_merged_states_landed_anchor_is_its_parts_bit_for_bit(way):
    """The merged state's landed anchor (`means.merged_state`), an exact contract, is the merged total's bit
    for bit (`means.merged_total`), at each draw (`MERGED_DRAWS`)."""
    state, apart = (_ways(form, label)[way] for form, label in (
        (_state(1), "a merged state's anchor"), (_apart(1), "its parts' anchor")))
    # Every arm run before any is judged (`reference_error.ran_here`).
    found = [(state(*draw(0))[2], apart(*draw(0))[2]) for draw in MERGED_DRAWS.values()]
    for mine, theirs in found:
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


def _gradients_derivatives(form):
    """The derivatives of a two-state merge's gradient as to its totals (its states' anchors 0, one channel)
    under a cotangent of its output and its total: along the outputs, along the totals forward over
    reverse and reverse over reverse (the attention merge's second-order site's)."""
    def ran(outputs, totals, along_outputs, along_totals):
        def gradient(o, w):
            def objective(each):
                output, total, _ = form(o[:, None], each, jnp.zeros_like(each))
                return 0.75 * output[0] + 0.5 * total
            return jax.grad(objective)(w)

        mixed = jax.jvp(lambda o: gradient(o, totals), (outputs,), (along_outputs,))[1]
        forward = jax.jvp(lambda w: gradient(outputs, w), (totals,), (along_totals,))[1]
        reverse = jax.grad(lambda w: jnp.vdot(gradient(outputs, w), along_totals))(totals)
        return mixed, forward, reverse
    return ran


@pytest.mark.parametrize("totals", [[1.25, 0.0], [1.25, 0.5]], ids=["a zero mass", "two masses"])
@pytest.mark.parametrize("way", WAYS)
def test_a_merged_states_gradients_derivatives_are_on_par_with_its_parts(way, totals):
    """The derivatives of a merged state's gradient (`means.merged_state`), forward and reverse over its
    reverse, on par with the separate merged output's and total's from float64 (`_on_par`), at a state of no
    total beside one of a drawn mass too (the attention merge's zero-mass draw, row 7124, where a rule that
    moved every operand read the anchors' zero tangent as a term and missed by its whole value)."""
    def drawn(seed):
        rng = np.random.default_rng(seed)
        directions = (np.asarray(rng.standard_normal(2), np.float32) for _ in range(2))
        return (np.asarray(rng.uniform(-1, 1, 2), np.float32), np.asarray(totals, np.float32), *directions)

    _on_par(_gradients_derivatives(_state(0)), _gradients_derivatives(_apart(0)), way, drawn,
            f"a merged state's gradient's derivatives against its parts', {totals}")


@pytest.mark.parametrize("way", WAYS)
def test_a_float64_merge_keeps_what_float64_holds(way):
    """A merge of float64 states (a truth's) sums its terms placed beneath float64's largest power, not
    float32's: outputs 2^900, -2^900 and 2^-900 at unit totals and one anchor merge to 2^-900 / 3, bit for
    bit, which a sum placed by float32's range shifted beneath float64's least, to zero (the law's
    review)."""
    with jax.enable_x64(new_val=True):
        outputs = np.asarray([[2.0 ** 900], [-(2.0 ** 900)], [2.0 ** -900]])
        merged = _ways(_merge, "a float64 merge")[way](outputs, np.ones(3), np.zeros(3))
        assert np.asarray(merged).tolist() == [2.0 ** -900 / 3]


@pytest.mark.parametrize("way", WAYS)
def test_a_merged_state_is_the_same_on_whichever_axis_its_states_lie(way):
    """16 states of 256 rows, their anchors drawn 3 apart, merged along their first axis and along their
    second (`means.merged_state`): its output and total on par from float64 (`_on_par`: two layouts, two
    programs, held by check 1, not bit for bit, the merge P0's base review 13, F4), and its landed anchor,
    an exact contract, the same bit for bit (XLA's of a column and a merge kernel's of a row, a Triton tree,
    gave 29 of 64 totals apart on a GPU; XLA's column and row on a CPU 118 of these 256)."""
    def drawn(seed):
        rng = np.random.default_rng([0, seed])
        return (rng.standard_normal((16, 256, 8)).astype(np.float32),
                np.abs(rng.standard_normal((16, 256))).astype(np.float32),
                (3 * rng.standard_normal((16, 256))).astype(np.float32))

    def first(o, w, m):
        return means.merged_state(o, w, m, NATURAL, 0)

    def second(o, w, m):
        return means.merged_state(jnp.swapaxes(o, 0, 1), w.T, m.T, NATURAL, 1)

    _on_par(lambda *xs: first(*xs)[:2], lambda *xs: second(*xs)[:2], way, drawn,
            "a merged state along either axis")
    operands = drawn(0)
    anchors = [np.asarray(_ways(fn, f"a merged state's anchor, {index}")[way](*operands)[2]).view(np.uint32)
               for index, fn in enumerate((first, second))]
    np.testing.assert_array_equal(*anchors)


def test_a_vmapped_merge_and_its_gradient_are_each_merges_own():
    """`jax.vmap` over a batch of merges, the law's value and its gradient as to every operand, each the
    merge's own run alone (`merged_p`'s, `gapped_p`'s, `applied_p`'s and `split_p`'s batching), with the
    totals unbatched as well as batched."""
    outputs, totals, anchors = _batch(0)
    gradients = _gradient(_merge)
    merge_alone, gradient_alone = (ahead_arm(jax.jit(fn), label) for fn, label in (
        (_merge, "a merge alone"), (gradients, "a merge's gradient alone")))
    # Every arm run before any is judged (`ran_here`).
    found = []
    for in_axes in ((0, 0, 0), (0, None, 0)):
        shared = totals if in_axes[1] == 0 else totals[0]
        mapped, derived = (ahead_arm(jax.jit(jax.vmap(fn, in_axes)), f"{label} mapped over {in_axes}")(
            outputs, shared, anchors) for fn, label in ((_merge, "merges"), (gradients, "gradients")))
        alone = [(outputs[k], shared if in_axes[1] is None else shared[k], anchors[k])
                 for k in range(len(outputs))]
        found.append((mapped, derived, [merge_alone(*each) for each in alone],
                      [gradient_alone(*each) for each in alone]))
    for mapped, derived, merged, own_gradients in found:
        for k, (merge, own) in enumerate(zip(merged, own_gradients, strict=True)):
            assert np.asarray(mapped[k]).tolist() == np.asarray(merge).tolist()
            for got, expected in zip(derived, own, strict=True):
                assert np.asarray(got[k]).tolist() == np.asarray(expected).tolist()


@pytest.mark.mesh(devices=2)
def test_a_merge_under_shard_map_and_under_a_sharded_jit_is_each_shards_own():
    """The law and its gradient over a batch of merges split over two devices, in `jax.shard_map` (each
    shard's merges its own, its types' varying axes kept, `merged_p`'s result type) and in a jit of
    operands placed by `NamedSharding`: each merge's value and gradient on par from float64 with the merge's
    own run alone (`_each_alone`)."""
    from jax.sharding import NamedSharding, PartitionSpec

    mesh = jax.make_mesh((2,), ("merges",), devices=jax.devices()[:2])

    def value_and_gradient(o, w, m):
        gradient = jax.grad(lambda *state: jnp.sum(_batched(*state)), argnums=(0, 1, 2))(o, w, m)
        return _batched(o, w, m), gradient

    split = PartitionSpec("merges")
    mapped = jax.jit(jax.shard_map(value_and_gradient, mesh=mesh, in_specs=(split,) * 3,
                                   out_specs=(split, (split,) * 3)))
    for ran, label in ((mapped, "in a shard map"), (jax.jit(value_and_gradient), "in a sharded jit")):
        # Each merge's value and gradient on par from float64 with its own run alone (two programs, held by
        # check 1, the merge P0's base review 13, F4).
        _each_alone(lambda *xs, ran=ran: ran(*(jax.device_put(x, NamedSharding(mesh, split)) for x in xs)),
                    jax.jit(value_and_gradient), lambda seed: _batch(seed + 1, count=4), f"merges {label}")


def _each_alone(placed, alone, draw, label: str) -> None:
    """`placed`, a batch of merges run split over devices, held on par from float64 with each merge's own run
    of `alone` (check 1, `assert_as_exact`; each arm run as given, its placement its own): two programs, held
    so and not bit for bit (docs/design/testing.md; the merge P0's base review 13, F4)."""
    def each(*operands):
        runs = [alone(*(x[k:k + 1] for x in operands)) for k in range(len(operands[0]))]
        return jax.tree.map(lambda *parts: jnp.concatenate(parts), *runs)

    assert_as_exact(as_given(placed), dew_path(as_given(each)), draw, label, seeds=FEWEST_SEEDS)


@pytest.mark.mesh(devices=2)
def test_a_merged_state_under_an_explicit_mesh_drops_its_states_axis_from_its_shardings():
    """The merged state and the law's gradient over merges split over two devices under an explicit mesh
    (each array's sharding a part of its type), its states along their second axis: each result's spec
    without that axis (base review F3: the spec kept whole beside a shape without it was refused), each
    constant and mask the law builds of its operands' sharding (one of a shape alone was refused beside them
    in a select), and each merge's state and gradient on par from float64 with the merge's own run alone
    (`_each_alone`), its landed anchor the merge's own bit for bit."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    mesh = jax.make_mesh((2,), ("merges",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    split = PartitionSpec("merges")
    gradient = _gradient(_batched)

    def merged(o, w, m):
        state = _state(1)(o, w, m)
        return (*state[:2], *gradient(o, w, m)), state[2]

    def placed(*operands):
        with jax.set_mesh(mesh):
            return jax.jit(merged)(*(jax.device_put(x, NamedSharding(mesh, split)) for x in operands))

    # Its state's output and total and the gradient on par with each merge's alone; its landed anchor, an
    # exact contract, each merge's own bit for bit.
    _each_alone(lambda *xs: placed(*xs)[0], lambda *xs: jax.jit(merged)(*xs)[0],
                lambda seed: _batch(seed + 1, count=4), "merged states under an explicit mesh")
    operands = _batch(1, count=4)
    anchors = np.asarray(placed(*operands)[1])
    for k in range(len(operands[0])):
        alone = jax.jit(merged)(*(x[k:k + 1] for x in operands))[1]
        assert anchors[k].tolist() == np.asarray(alone)[0].tolist()


@pytest.mark.mesh(devices=2)
def test_a_long_row_of_states_under_an_explicit_mesh_merges_as_each_merge_alone():
    """Merges of seventeen states each (a row the law reads in several chunks, its loops' carries then a
    mesh's types: `means._streamed`) split over two devices under an explicit mesh: each merge's state on par
    from float64 with its own run alone (`_each_alone`), its landed anchor its own bit for bit (the review of
    the streamed law: carries of a shape alone, replicated, were refused beside the steps' sharded
    reductions)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    def drawn(seed):
        rng = np.random.default_rng([17, seed])
        return (rng.standard_normal((2, 17, 3)).astype(np.float32),
                rng.uniform(0.5, 2.0, (2, 17)).astype(np.float32),
                rng.uniform(-30, 0, (2, 17)).astype(np.float32))

    mesh = jax.make_mesh((2,), ("merges",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    state = jax.jit(_state(1))

    def placed(*operands):
        with jax.set_mesh(mesh):
            return state(*(jax.device_put(x, NamedSharding(mesh, PartitionSpec("merges"))) for x in operands))

    # Its output and total on par with each merge's alone; its landed anchor each merge's own bit for bit.
    _each_alone(lambda *xs: placed(*xs)[:2], lambda *xs: state(*xs)[:2], drawn,
                "long rows under an explicit mesh")
    operands = drawn(0)
    anchors = np.asarray(placed(*operands)[2])
    for k in range(2):
        assert anchors[k].tolist() == np.asarray(state(*(x[k:k + 1] for x in operands))[2])[0].tolist()


@pytest.mark.parametrize("anchors", [np.float32(-0.5), np.linspace(-3.0, 0.0, 5, dtype=np.float32)],
                         ids=["a common scalar", "shared across the batch"])
def test_a_merged_state_of_broadcast_anchors_is_its_output_and_its_total(anchors):
    """A merged state whose anchors are common to its states (a scalar) or shared across its batch (one a
    state) is on par with its merged output and merged total from float64 (`_on_par`; two programs, the
    merge P0's base review 12, F5), which broadcast them as the law weighs them, its landed anchor theirs
    bit for bit (the review of the streamed law: its chunk loader indexed each operand's states axis and
    refused them)."""
    def drawn(seed):
        rng = np.random.default_rng([5, seed])
        return (rng.standard_normal((3, 5, 2)).astype(np.float32),
                rng.uniform(0.5, 2.0, (3, 5)).astype(np.float32), anchors)

    def value(form):
        return lambda *operands: form(*operands)[:2]

    _on_par(value(_state(1)), value(_apart(1)), "jitted", drawn, "a merged state of broadcast anchors")
    operands = drawn(0)
    anchors_of = [np.asarray(ahead_arm(jax.jit(form), f"broadcast anchors' {label}")(*operands)[2]).tolist()
                  for form, label in ((_state(1), "state"), (_apart(1), "parts"))]
    assert anchors_of[0] == anchors_of[1]


def test_the_primitives_result_types_keep_weak_types():
    """A scaled product (`applied_p`) of weakly typed operands is weakly typed, as JAX's own entrywise
    products are, and of a strongly typed one strongly: a Python float's product promotes as a float's
    would."""
    # Types alone, nothing placed (a GPU tier part's prepare computes nothing, not even an array's placing).
    weak, strong = (jax.ShapeDtypeStruct((), jnp.float32, weak_type=weak) for weak in (True, False))
    exponent = jax.ShapeDtypeStruct((), jnp.int32)

    def product(x, significand, exponent):
        return scaled.applied_p.bind(x, significand, exponent)

    assert jax.eval_shape(product, weak, weak, exponent).weak_type
    assert not jax.eval_shape(product, weak, strong, exponent).weak_type


def test_every_primitive_of_the_construct_is_recorded_with_the_value_it_lowers_as():
    """Each operation of the construct's own (`scaled.primitive`), the law's among them, is recorded with
    its value (`scaled.MADE`): a lowering registered from that record computes what its default lowering
    does, so a launch running the construct's functions holds every primitive they reach (a merge
    kernel's Pallas lowering lacked `convex_merged` until it was listed by hand)."""
    outputs, totals, anchors = _batch(0)
    reached = {equation.primitive for equation in jax.make_jaxpr(_state(1))(outputs, totals, anchors).eqns}
    assert reached <= set(scaled.MADE)
    value, several = scaled.MADE[means.merged_state_p]
    assert several

    bound = _state(1)

    def recorded_and_bound(*operands):
        o, w, m = (jnp.asarray(x) for x in operands)
        return value(o, w, m, units=NATURAL, axis=1), bound(o, w, m)

    # Every way run before any is judged (`ran_here`).
    for found, held in [run(outputs, totals, anchors)
                        for run in ran_here(recorded_and_bound, "the merged state's value and its bind")]:
        for got, expected in zip(found, held, strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.cpu(reason="a refusal at bind, alike on every backend: the CPU lane holds it")
def test_a_parameter_made_at_each_call_is_refused_at_bind():
    """A law primitive's parameter keys its compiled programs (`scaled._Primitive`): units made at each call,
    a partial or a closure, are refused at bind, where each call would compile again and hold its executable;
    the units' value (`means.Units`, a frozen dataclass's instance) merges."""
    outputs = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], jnp.float32)
    totals, anchors = jnp.asarray([1.0, 2.0], jnp.float32), jnp.asarray([0.0, -1.0], jnp.float32)
    for made in (functools.partial(jnp.exp), lambda carried, now: jnp.exp(carried - now)):
        with pytest.raises(TypeError):
            means.merged_output(outputs, totals, anchors, made, 0)
    np.testing.assert_array_equal(means.merged_output(outputs, totals, anchors, means.Units(), 0),
                                  means.merged_output(outputs, totals, anchors, NATURAL, 0))

