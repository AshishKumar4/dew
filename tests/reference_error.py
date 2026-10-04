"""Dew's rounding error held to the reference's own, both measured from float64.

A port that computes what its reference computes, at the same precision,
differs from it only by rounding: the two sum the same terms in other
orders and round at other points. The reference evaluated in float64
(u = 2^-53, nine orders below fp32) is the exact value to every digit an
fp32 or bf16 run resolves, so both runs are measured from it, and Dew's
root-mean-square error over all entries may be at most twice the
reference's:

    rms(dew - f64) <= 2 rms(reference - f64)

Why the root mean square and why twice. Independent roundings add in
variance, so the RMS error of an output is sqrt(sum of the variances of the
roundings that reach it), and over the hundreds to thousands of entries of
a fixture that estimate is stable to a few percent; the largest entry is an
extreme value and is not (the fp32 decoder fixtures put Dew's largest error
at 0.9 to 2.2 times the reference's while the RMS ratios sit at 1.06 to
1.43). A factor of 2 admits four times the reference's error variance, that
is Dew making up to three roundings of the reference's size for each one
the reference makes; a computation that differs rather than rounds (a
dropped term, a wrong scale, a format coarser than the reference's, a mask
one step off) moves the RMS error by more than that and fails.

What the rule cannot see is a difference below that factor: an attention
softmax taken in bf16 instead of fp32 lands at 1.2 to 1.4 times the
reference's error, since the bf16 logits both runs share carry most of it.
Where Dew claims the reference's order of operations, it is held instead to
the reference's own bf16 output:

    rms(dew - reference) <= rms(reference - f64) / 4

Rounding at the same points, the two outputs part only where their fp32
accumulations, summed in different orders, fall on opposite sides of a bf16
rounding boundary: an entry flips with probability about sqrt(K) 2^-24 /
2^-8, K the reduction length, by one bf16 spacing, about twice the typical
rounding error, so the ratio is near 2 sqrt(sqrt(K) 2^-16), 0.04 at K 512.
One rounding at a point the reference does not round is an independent
error of the reference's own size and puts the ratio near one.

Where the truth is an oracle written apart from the reference (a recurrence
or an equation, not the reference run in float64), the rule only measures
rounding if the reference computes the oracle's function: a reference that
shares Dew's mistake would widen the bound to fit it. So the reference's own
arithmetic in float64 is held to the oracle first, within the float64 share
of the reference's rounding (`assert_computes_the_oracle`).
"""

import numpy as np

FACTOR = 2.0


def assert_fp32_reduction_bound(dew, reference, magnitudes, terms: int) -> None:
    """Two fp32 reductions differ by at most 2 gamma_n sum(abs(products)).

    `magnitudes` holds the elementwise sum of absolute products; `terms`
    counts the products and any bias term in each reduction. gamma_n is
    n*u/(1-n*u), with fp32 unit roundoff u=2^-24.
    """
    roundoffs = terms * np.finfo(np.float32).eps / 2
    gamma = roundoffs / (1 - roundoffs)
    error = np.abs(np.asarray(dew, np.float64) - np.asarray(reference, np.float64))
    bound = 2 * gamma * np.asarray(magnitudes, np.float64)
    assert np.all(error <= bound), (float(error.max()), float(bound.max()))


def distance(value, truth) -> float:
    """The root-mean-square difference over every entry, in float64."""
    difference = np.asarray(value, np.float64) - np.asarray(truth, np.float64)
    return float(np.sqrt(np.mean(np.square(difference))))


def assert_as_exact_as_the_reference(dew, reference, truth, label: str) -> None:
    """Dew within FACTOR times the reference's RMS distance from float64."""
    mine, theirs = distance(dew, truth), distance(reference, truth)
    assert theirs > 0, f"{label}: the reference equals float64, so it measures nothing"
    assert mine <= FACTOR * theirs, (
        f"{label}: dew is {mine:.3e} from float64 (rms), the reference {theirs:.3e} "
        f"(ratio {mine / theirs:.2f}, allowed {FACTOR})")


def assert_computes_the_oracle(reference_in_float64, reference, truth, label: str) -> None:
    """The reference run in float64 within FACTOR times its own fp32 error,
    scaled by the float64-to-fp32 ratio of unit roundoffs for each of the two
    float64 runs, of an independent float64 oracle: the two compute one
    function, and any difference in it moves the twin by far more. For a
    rounded computation: one whose fp32 run is exact gives a bound of zero,
    which the reference refuses (`assert_as_exact_as_the_reference` does too)."""
    twin = 2 * float(np.finfo(np.float64).eps / np.finfo(np.float32).eps)
    apart, theirs = distance(reference_in_float64, truth), distance(reference, truth)
    assert theirs > 0, f"{label}: the reference equals float64, so it bounds nothing"
    assert apart <= FACTOR * twin * theirs, (
        f"{label}: the reference in float64 is {apart:.3e} from the oracle (rms), more than float64 "
        f"rounding of its fp32 error {theirs:.3e}")


def assert_rounds_where_the_reference_does(dew, reference, truth, label: str) -> None:
    """Dew within a quarter of the reference's RMS rounding error of the
    reference's own output."""
    apart, theirs = distance(dew, reference), distance(reference, truth)
    assert apart <= theirs / 4, (
        f"{label}: dew is {apart:.3e} from the reference (rms), whose own error is {theirs:.3e} "
        f"(ratio {apart / theirs:.2f}, allowed 0.25)")
