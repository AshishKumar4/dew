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

What the rule cannot see is a difference below that factor. On the tiny
decoder fixtures an RMSNorm reduced in bf16 instead of fp32, or the
attention softmax taken in bf16, moves Dew's bf16 RMS error by at most 1.26
times: at widths of 32 to 64 and 12 positions those orderings round no
more than the matmuls around them do.
"""

import numpy as np

FACTOR = 2.0


def distance(value, truth) -> float:
    """The root-mean-square difference over every entry, in float64."""
    difference = np.asarray(value, np.float64) - np.asarray(truth, np.float64)
    return float(np.sqrt(np.mean(np.square(difference))))


def assert_as_exact_as_the_reference(dew, reference, truth, label: str = "") -> tuple[float, float]:
    """Dew within FACTOR times the reference's RMS distance from float64;
    returns both distances for a caller that reports them."""
    mine, theirs = distance(dew, truth), distance(reference, truth)
    assert theirs > 0, f"{label}: the reference equals float64, so it measures nothing"
    assert mine <= FACTOR * theirs, (
        f"{label}: dew is {mine:.3e} from float64 (rms), the reference {theirs:.3e} "
        f"(ratio {mine / theirs:.2f}, allowed {FACTOR})")
    return mine, theirs
