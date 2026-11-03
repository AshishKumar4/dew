"""Check 1's worst-row rule (tests/check1_statistics.py, `_worst_rows`) on comparisons whose answer is known:
each arm's worst rows pooled over the draws, Dew's held to PARITY times the reference's by the pooled ratio's
interval, and the evidence it reads alike for both arms on every path."""

import math

import numpy as np
import pytest
from reference_error import GateFailure, as_given, assert_as_exact


def test_the_worst_row_is_held_at_parity_at_every_row_width():
    """The worst-row rule judged alone through check 1's stages (`_staged` on a leaf whose pooled ratio is
    one): Dew's worst row just past PARITY times the reference's on every seed fails, drawing the second
    stage, and just inside PARITY passes at the first, at one value a row as at eight (row 3684: a factor of
    90 and 11 at narrow rows, and 4 at wider ones, held no worst row at PARITY)."""
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


def staged(worst: np.ndarray, width: int = 8):
    """Check 1's stages (`_staged`) over a leaf whose pooled ratio is one and whose worst rows are `worst`
    ((seeds, 2), each draw's worst row by its log, Dew's and the reference's), 8 seeds and then 40."""
    from reference_error import NO_ZERO_ROWS, _Draws, _staged

    def drawn(start, stop):
        count = stop - start
        return _Draws(["leaf"], np.ones((count, 1)), np.ones((count, 1)), worst[start:stop, None, :],
                      [[NO_ZERO_ROWS]] * count, np.ones((count, 1), bool), (width,))

    return _staged(drawn, 8, "leaf")


def worst_rows(rng, shift: np.ndarray, amplified: bool = False, unpaired: bool = False) -> np.ndarray:
    """40 seeds' worst rows by their logs, Dew's and the reference's, of 16 rows of 8 standard normal errors
    over a standard normal truth, Dew's each `shift` (a seed's factor, by its log) off; `amplified`, one
    element a row times 2^k, k uniform over 0 to 24 in each arm apart; `unpaired`, each arm on a truth of
    its own."""
    def sized():
        return np.sqrt(np.mean(np.square(rng.standard_normal((40, 16, 8))), axis=2))

    truth = sized()
    arms = []
    for _ in range(2):
        errors = rng.standard_normal((40, 16, 8))
        if amplified:
            errors[..., 0] *= 2.0 ** rng.integers(0, 25, (40, 16))
        arms.append(np.log((np.sqrt(np.mean(np.square(errors), axis=2)) / (sized() if unpaired else truth))
                           .max(axis=1)))
    arms[0] = arms[0] + shift
    return np.stack(arms, axis=-1)


@pytest.mark.parametrize(("factor", "share", "fails"),
                         [(1.2, 1.0, False), (2.0, 1.0, True), (10.0, 0.2, True)])
def test_the_worst_row_passes_dew_a_little_past_the_reference_and_fails_it_twice_off_or_far_off_on_some(
        factor, share, fails):
    """Each arm's worst row pooled over the draws as the pooled ratio pools errors (`_worst_rows`, row 3684):
    Dew's worst row 1.2 times the reference's on every seed is on par and passes, where a sign test of
    draws past PARITY either way failed it; twice on every seed fails; ten times on a fifth of the seeds
    fails, the fault ledger 2469 asks the worst row to see, on every one of 16 runs."""
    rng = np.random.default_rng([3684, 8, int(10 * factor)])
    for _ in range(16):
        shift = np.where(np.arange(40) % round(1 / share) == 0, math.log(factor), 0.0)
        over, unresolved, _ = staged(worst_rows(rng, shift))
        assert (bool(over), unresolved) == (fails, []), (factor, share, over, unresolved)


@pytest.mark.parametrize(("amplified", "unpaired"), [(False, True), (True, False), (True, True)])
def test_equally_exact_arms_are_never_over_whatever_their_rows_tails_or_pairing(amplified, unpaired):
    """Equally exact arms, their rows' errors light-tailed but on truths of their own, or one element a row
    amplified by 2^k for k uniform over 0 to 24, so one draw may rule an arm's worst rows, are never over
    on any of 32 runs: where one draw dominates the ratio is inconclusive, its site given more seeds, never
    a fault (row 3684: the worst row over every seed held at four times the reference's worst failed such
    pairs on about 6% of runs)."""
    rng = np.random.default_rng([3684, 9, amplified, unpaired])
    for _ in range(32):
        over, _, _ = staged(worst_rows(rng, np.zeros(40), amplified, unpaired))
        assert not over



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
