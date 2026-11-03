"""Check 1's worst-row rule (tests/reference_error.py, `_paired_rows`) on comparisons whose answer is known:
each draw's worst row of Dew held to the reference's on the same draw at PARITY by a sign test, which equally
exact arms pass whatever their rows' tails, and the evidence it reads alike for both arms on every path."""

import math

import numpy as np
import pytest
from reference_error import GateFailure, as_given, assert_as_exact


def test_the_worst_row_is_held_at_parity_at_every_row_width():
    """The worst-row rule judged alone through check 1's stages (`_staged` on a leaf whose pooled ratio is
    one): Dew's worst row just past PARITY times the reference's on every seed fails, its first stage's eight
    seeds screening it (`SCREEN`) and the second's forty refusing it, and just inside PARITY passes, at one
    value a row as at eight (row 3684: a factor of 90 and 11 at narrow rows, and 4 at wider ones, held no
    worst row at PARITY)."""
    from reference_error import NO_ZERO_ROWS, PARITY, _Draws, _staged

    for width in (1, 2, 3, 8):
        for worst, fails in ((0.99 * PARITY, False), (1.01 * PARITY, True)):
            # The draws hold each worst row by its log (`log_over`).
            def drawn(start, stop, worst=worst, width=width):
                count = stop - start
                return _Draws(["leaf"], np.ones((count, 1)), np.ones((count, 1)),
                              np.tile([[[np.log(worst), 0.0]]], (count, 1, 1)), [[NO_ZERO_ROWS]] * count,
                              np.ones((count, 1), bool), (width,))

            over, unresolved, stage = _staged(drawn, 8, "leaf")
            assert (bool(over), stage) == ((True, 2) if fails else (False, 1)), (width, worst)
            assert not unresolved


def test_a_row_ten_times_the_references_fails_at_every_width_through_check_one():
    """Through `assert_as_exact`: a first row ten times the reference's relative error, too small a value
    for the pooled ratio to see, fails at one value a row as at eight: the worst row is held at PARITY
    whatever its width (row 3684)."""
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

    for width in (1, 8):
        dew, reference, draw = arms(width)
        with pytest.raises(AssertionError) as raised:
            assert_as_exact(dew, reference, draw, f"width {width}", seeds=8, truth_fn=truth)
        assert raised.type is GateFailure


def test_the_paired_worst_row_refuses_a_row_far_off_on_some_draws_and_passes_equally_exact_arms():
    """Each draw's worst row is held to the reference's on the same draw (`_paired_rows`, ledger 2469):
    equally exact arms, their worst rows' log ratios symmetric however wide (a spread of 1.0 and 2.0), pass;
    Dew ten times the reference's on a fifth of 512 draws of such spread is refused at LOWER_TAIL; and two
    draws past PARITY, both Dew's, are no finding."""
    from reference_error import LOWER_TAIL, _paired_rows

    rng = np.random.default_rng([2469, 512])
    for spread in (1.0, 2.0):
        null = rng.normal(0.0, spread, 512)
        assert _paired_rows(null) >= LOWER_TAIL
        off = null.copy()
        off[::5] += math.log(10.0)
        assert _paired_rows(off) < LOWER_TAIL
    assert _paired_rows(np.asarray([3.0, 3.0, 0.0])) >= LOWER_TAIL


def test_the_worst_row_passes_heavy_tailed_equal_arms_and_fails_one_twice_the_references_every_seed():
    """Check 1's worst row through its stages (`_staged`), on a leaf whose pooled ratio is one (row 3684):
    equally exact arms whose rows' errors are heavy-tailed, one element a row amplified by 2^k for k uniform
    over 0 to 24 in each arm apart, pass on every one of 64 runs, where the worst row over the seeds held
    at four times the reference's worst failed such pairs on about 6% of runs (one draw's amplified element
    past four times the other arm's every one); and Dew's worst row twice the reference's on every seed fails,
    which four times passed."""
    from reference_error import NO_ZERO_ROWS, _Draws, _staged

    def staged(worst):
        def drawn(start, stop):
            count = stop - start
            return _Draws(["leaf"], np.ones((count, 1)), np.ones((count, 1)), worst[start:stop],
                          [[NO_ZERO_ROWS]] * count, np.ones((count, 1), bool), (8,))
        return _staged(drawn, 8, "leaf")

    rng = np.random.default_rng([3684, 64])
    reds = 0
    for _ in range(64):
        # 40 seeds of 16 rows of 8 values, each arm's errors standard normal with one element a row
        # amplified, over a shared standard normal truth; each seed's worst row by its log over its size.
        truth = np.sqrt(np.mean(np.square(rng.standard_normal((40, 16, 8))), axis=2))
        arms = []
        for _ in range(2):
            errors = rng.standard_normal((40, 16, 8))
            errors[..., 0] *= 2.0 ** rng.integers(0, 25, (40, 16))
            arms.append(np.log((np.sqrt(np.mean(np.square(errors), axis=2)) / truth).max(axis=1)))
        over, unresolved, _ = staged(np.stack(arms, axis=-1)[:, None, :])
        reds += bool(over or unresolved)
    assert reds == 0
    twice = np.zeros((40, 1, 2))
    twice[..., 0] = math.log(2.0)
    over, _, stage = staged(twice)
    assert over and over[0].rule == "row" and stage == 2


def test_equal_arms_whose_worst_rows_take_turns_pass_where_one_seeds_maximum_failed_them():
    """Through `assert_as_exact` (row 3684): two arms each a value off by its own seed's noise, a first row of
    1/64 the leaf's so it is every arm's worst row on every seed, Dew's noise there 40 times its other rows'
    on seed 0 and the reference's 3 times on seeds 1 to 20. Dew's worst row is past PARITY times the
    reference's on fewer seeds than the reverse, so the pair passes; the worst row over every seed held at
    four times the reference's worst failed it by its one seed."""
    def draw(seed):
        rng = np.random.default_rng([3684, seed])
        values = rng.standard_normal((256, 8))
        values[0] = 2.0 ** -6
        mine, theirs = rng.standard_normal((2, 256, 8))
        if seed == 0:
            mine[0] *= 40.0
        elif seed <= 20:
            theirs[0] *= 3.0
        return values, mine, theirs

    @as_given
    def dew(values, mine, theirs):
        return {"leaf": values + 2.0 ** -20 * mine}

    @as_given
    def reference(values, mine, theirs):
        return {"leaf": values + 2.0 ** -20 * theirs}

    assert_as_exact(dew, reference, draw, "turns", seeds=8, truth_fn=lambda values, *_: {"leaf": values})


def test_a_library_arm_never_holds_dews_worst_row_to_less_than_the_references():
    """With a library arm (`library_fn`), the reference's worst row stands where the library arm's error is
    the larger and its worst row the smaller (row 3684): Dew computing the reference's own values, off twice
    a relative unit on a row of 1/64 of the leaf, beside a library arm off one unit on a row of the leaf's
    size, larger in squared error, passes, its worst row the reference's on every seed."""
    def draw(seed):
        rng = np.random.default_rng([3684, 2, seed])
        values = rng.standard_normal((64, 8))
        values[0] *= 2.0 ** -6
        return values, rng.standard_normal((64, 8))

    @as_given
    def reference(values, noise):
        out = values.copy()
        out[0] += 2.0 * 2.0 ** -20 * values[0] * noise[0]
        return {"leaf": out}

    @as_given
    def library(values, noise):
        out = values.copy()
        out[1] += 2.0 ** -20 * values[1] * noise[1]
        return {"leaf": out}

    assert_as_exact(reference, reference, draw, "library budget", seeds=8,
                    truth_fn=lambda values, _: {"leaf": values}, library_fn=library, library_row=1543,
                    library_leaves=["['leaf']"])


def test_a_row_below_resolution_is_measured_only_where_both_arms_measure_it():
    """A row below RESOLUTION of its leaf is judged at its own size only where both arms' errors there are
    under it (`_paired`, row 3684): equally exact arms whose eight such rows each come out twice their own
    size off on half the seeds, each arm apart, and exact on the rest, pass, where the reference alone
    classifying them put a row Dew alone missed into Dew's worst row on nearly every seed, and never the
    reverse."""
    def draw(seed):
        rng = np.random.default_rng([3684, 3, seed])
        values = rng.standard_normal((64, 8))
        values[:8] *= 2.0 ** -60
        return values, rng.standard_normal((2, 64, 8)), rng.integers(0, 2, (2, 8))

    def arm(index):
        @as_given
        def run(values, noises, coins):
            out = values + 2.0 ** -20 * noises[index] * values
            out[:8] = values[:8] * (1 + 2.0 * coins[index])[:, None]
            return {"leaf": out}
        return run

    assert_as_exact(arm(0), arm(1), draw, "tiny rows", seeds=8, truth_fn=lambda values, *_: {"leaf": values})


def test_stacked_examples_hold_each_arms_worst_row_over_every_order():
    """At a site that stacks examples (`examples`) and sweeps an orbit's orders (`orders`), each example's
    worst row is each arm's worst over the orders before the two pair (`_example_worst_rows`, row 3684): a
    first row of 1/64 of its example, every arm's worst, Dew's noise there twice the reference's at the draw's
    own order and the reference's twice Dew's at the other, both arms' worst equal over the orbit, passes,
    where the first order alone paired Dew past PARITY on every example."""
    def draw(seed):
        rng = np.random.default_rng([3684, 4, seed])
        values = rng.standard_normal((32, 16, 8))
        values[:, 0] *= 2.0 ** -6
        return values, rng.standard_normal((32, 16, 8)), np.zeros(())

    def orders(inputs):
        values, noise, _ = inputs
        return [inputs, (values, noise, np.ones(()))]

    def arm(first, second):
        @as_given
        def run(values, noise, order):
            scaled = noise.copy()
            scaled[:, 0] *= second if order else first
            return {"leaf": values + 2.0 ** -20 * scaled}
        return run

    assert_as_exact(arm(2.0, 1.0), arm(1.0, 2.0), draw, "orders", seeds=8, orders=orders, examples=True,
                    truth_fn=lambda values, *_: {"leaf": values})


def test_a_summarized_recording_and_dews_rows_are_sized_alike_at_a_site_with_terms():
    """A summarized recording sizes its reference's rows by its truth alone (`leaf_summaries`), so against it
    Dew's rows are sized so too, the site's terms aside (row 3684): terms of one over a truth of one but a
    first row of 2^-20, the reference off 2^-23 everywhere and Dew off 2^-20 on that row, so its pooled ratio
    is about 1.06 and its worst row eight times the reference's, fails against the recording as against the
    live reference, where Dew's term-sized worst row against the recording's truth-sized one passed."""
    import jax
    from reference_error import indexed_recording
    from tool_support import load

    recorder = load("recorded_references")

    def draw(seed):
        rng = np.random.default_rng([3684, 7, seed])
        values = np.ones((512, 8))
        values[0] = 2.0 ** -20
        return {"values": values, "signs": rng.choice([-1.0, 1.0], (2, 512, 8))}

    def truth(values, signs):
        return {"leaf": values}

    def terms(values, signs):
        return {"leaf": np.ones_like(values)}

    def computed(values, signs, index):
        scale = np.full((512, 1), 2.0 ** -23)
        if index:
            scale[0] = 2.0 ** -20
        return {"leaf": values + scale * signs[index]}

    @as_given
    def dew(values, signs):
        return computed(values, signs, 1)

    @as_given
    def reference(values, signs):
        return computed(values, signs, 0)

    def inputs(seed):
        drawn = draw(seed)
        return drawn["values"], drawn["signs"]

    with pytest.raises(AssertionError) as raised:
        assert_as_exact(dew, reference, inputs, "live", seeds=8, truth_fn=truth, terms=terms)
    assert raised.type is GateFailure and {finding.rule for finding in raised.value.findings} == {"row"}

    with jax.enable_x64(new_val=True):
        recorded = recorder.record_case(
            "site", draw, lambda drawn: (computed(**drawn, index=0), truth(**drawn)), summary=True)
    seeds = int(recorded["site.seeds"])
    drawn, recorded_reference, recorded_truth = indexed_recording(recorded, "site", seeds, inputs_of=draw,
                                                                  truth_fn=lambda inputs: truth(**inputs))

    @as_given
    def dew_at(index, given):
        return computed(**given, index=1)

    with pytest.raises(AssertionError) as raised:
        assert_as_exact(dew_at, recorded_reference, drawn, "summarized", seeds=seeds, truth_fn=recorded_truth,
                        terms=lambda index, given: terms(**given))
    assert raised.type is GateFailure and {finding.rule for finding in raised.value.findings} == {"row"}
