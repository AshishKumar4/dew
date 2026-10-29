"""The one-draw rule (`assert_as_exact_as_the_reference`, `assert_leaves_as_exact`, `assert_no_less_exact`): a
site's one recorded or drawn comparison, held by its distance from float64 and its rows against the
reference's (`rows_over`), which G02 moves to check 1 site by site (`reference_error.assert_as_exact`). Its
rows' sizes and errors, and the gate's findings, are reference_error's, which re-exports these names."""

import math

import jax
import numpy as np
from reference_error import (
    FACTOR,
    ROW_FACTOR,
    Finding,
    GateFailure,
    _log_over,
    _shown,
    distance,
    log_ratio,
    row_errors,
    within,
)


def rows_over(dew, reference, truth, terms=None, floor: float = 0.0) -> list[tuple[str, float, str]]:
    """Where Dew's rows are further from float64 than ROW_FACTOR times the
    reference's (`row_errors`): its worst row over its own size against the
    reference's worst, at least `floor` (an order floor in the rows' units),
    and each row of size zero, a sum that cancels exactly before anything its
    terms see (a query that keeps one key), against the reference's distance
    on that same row, which is all the evidence such a row has: where the
    reference is exactly the float64 zero, Dew must be too. Each as `(rule,
    measure, message)`, a `Finding`'s rule, measure and message (RULES)."""
    sizes, mine = row_errors(dew, truth, terms)
    _, theirs = row_errors(reference, truth, terms)
    sized = sizes > 0
    over = []
    if sized.any():
        # Compared by their logs, so a row's error over its size orders where the quotient overflows.
        row = float(np.max(_log_over(mine[sized], sizes[sized])))
        their_row = float(np.max(_log_over(theirs[sized], sizes[sized])))
        allowed = max(their_row, float(_log_over(floor, 1.0)))
        if not row <= math.log(ROW_FACTOR) + allowed:
            over.append(("row", row - allowed, f"the worst row is {_shown(row)} from float64 over its size, "
                         f"the reference's {_shown(their_row)}, allowed "
                         f"{_shown(math.log(ROW_FACTOR) + allowed)}"))
    zero = ~sized & ~(mine <= ROW_FACTOR * theirs)
    if zero.any():
        at = int(np.flatnonzero(zero)[0])
        # The measure is the worst of them, the first the example.
        over.append(("zero rows", log_ratio(mine[zero], theirs[zero]),
                     f"{int(zero.sum())} rows of float64 zero are further from it than the reference's own, "
                     f"row {at} by {mine[at]:.3e} against {theirs[at]:.3e}"))
    return over


def assert_as_exact_as_the_reference(dew, reference, truth, label: str, *, terms=None) -> None:
    """Dew within FACTOR times the reference's RMS distance from float64, and
    its rows within ROW_FACTOR times the reference's (`rows_over`), each
    sized by its float64 `terms` where they are given (`float64_terms`)."""
    mine, theirs = distance(dew, truth), distance(reference, truth)
    if not (math.isfinite(mine) and math.isfinite(theirs)):
        raise GateFailure([Finding(label, "not finite", 1, f"{label}: a value of dew, the reference or "
                                   f"float64 is not finite (dew {mine:.3e} and the reference {theirs:.3e} "
                                   f"from float64)")])
    if not theirs > 0:
        # No measure: Dew's own distance from float64 is what a change can make worse, and none at all
        # is a site that cannot be judged (`no measure`), never a pass.
        if mine > 0:
            raise GateFailure([Finding(label, "exact reference", mine, f"{label}: the reference equals "
                                       f"float64 and dew is {mine:.3e} from it (rms)")])
        raise GateFailure([Finding(label, "no measure", 0.0, f"{label}: the reference and dew both equal "
                                   f"float64, so the site measures nothing")])
    found = []
    if not within(mine, FACTOR * theirs):
        found.append(("pooled", log_ratio(mine, theirs), f"dew is {mine:.3e} from float64 (rms), the "
                      f"reference {theirs:.3e} (ratio {mine / theirs:.2f}, allowed {FACTOR})"))
    # Every rule's findings at once, so a change that eases one rule and breaks another shows both.
    found += rows_over(dew, reference, truth, terms)
    if found:
        raise GateFailure([Finding(label, rule, measure, f"{label}: {message}")
                           for rule, measure, message in found],
                          f"{label}: " + "; ".join(message for _, _, message in found))


def assert_no_less_exact(dew, reference, truth, label: str) -> None:
    """An unchanged computation has no larger float64 distance than its source."""
    mine, theirs = distance(dew, truth), distance(reference, truth)
    assert within(mine, theirs), (
        f"{label}: {mine:.3e} from float64 (rms), the unchanged source {theirs:.3e}")


def assert_leaves_as_exact(dew, reference, truth, *, reductions: int = 1, terms=None,
                           rows: bool = True) -> None:
    """Each leaf of the tree `dew` within FACTOR times the matching leaf of
    `reference`'s RMS distance from float64 (`truth`), the three trees alike.

    A port that sums `reductions` terms the reference sums in another order
    rounds by up to sqrt(reductions) float32 units of what it sums in RMS,
    so the reference's distance counts as at least that: one draw of
    rounding below it is luck, not a bar, and a leaf that lost a term still
    fails by orders of magnitude. What a leaf sums is its own float64 size,
    or, where `terms` gives the tree of each leaf's terms (`float64_terms`),
    the larger of that and their root sum of squares, the size of the
    partial sums a reordered sum rounds. A leaf whose sum cancels, as a key
    bias's gradient does under the softmax's shift invariance, is held to
    its terms' rounding, not to its own near-zero size; one whose terms'
    signs are independent keeps about its own. Each leaf's rows are held to
    the reference's too (`rows_over`), unless `rows` is false: after a
    second training step a row inherits the first step's rounding through
    the parameters it reads, which its own terms do not see, so such a
    comparison holds each tensor alone and its first step holds the rows."""
    unit = float(np.finfo(np.float32).eps)
    spread = np.sqrt(reductions)
    sizes = jax.tree.leaves(truth) if terms is None else jax.tree.leaves(terms)
    over = []
    leaves = zip(jax.tree_util.tree_leaves_with_path(dew), jax.tree.leaves(reference), jax.tree.leaves(truth),
                 sizes, strict=True)
    for (path, mine), theirs, true, summed in leaves:
        scale = max(distance(true, 0), distance(summed, 0))
        bar = FACTOR * max(distance(theirs, true), spread * unit * scale)
        if not within(distance(mine, true), bar):
            over.append(f"{jax.tree_util.keystr(path)}: {distance(mine, true):.3e} from float64 (rms), "
                        f"the reference {distance(theirs, true):.3e}, allowed {bar:.3e}")
        # Each row over its own terms where they are given, the order floor in those units.
        over += [f"{jax.tree_util.keystr(path)}: {message}"
                 for _, _, message in (rows_over(mine, theirs, true, None if terms is None else summed,
                                                 spread * unit) if rows else ())]
    assert not over, "\n".join(over)
