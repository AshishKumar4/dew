"""Dew's rounding error held to the reference's own, both measured from float64.

The owner's gate (bible §7, 2026-10-20) is two checks. Check 1, on par (`assert_as_exact`): each
leaf's RMS error from float64, pooled over a site's declared seeds, within PARITY (1.5) times the
reference's on the same inputs, its worst row within ROW_FACTOR of the reference's worst, with no floor
of any kind; a draw may be an orbit's, the same values under an exact symmetry. Check 2, never wrong
(tests/op_certificates.py): every element within its own operation's certified roundoff bound.

A port that computes what its reference computes, at the same precision,
differs from it only by rounding: the two sum the same terms in other
orders and round at other points. The reference evaluated in float64
(u = 2^-53, nine orders below fp32) is the exact value to every digit an
fp32 or bf16 run resolves, so both runs are measured from it. Independent
roundings add in variance, so the RMS error of an output is sqrt(sum of the
variances of the roundings that reach it); a computation that differs
rather than rounds (a dropped term, a wrong scale, a format coarser than
the reference's, a mask one step off) moves it further than PARITY and
fails.

Comparisons not yet converted to check 1 hold the one-draw rule at PARITY
(`assert_as_exact_as_the_reference`), and the K-order and element rules
(`assert_as_exact_over_orders`, `assert_elements_as_exact`) hold
ORDER_FACTOR, the factor their order count is derived for, until they judge
through check 1 over orders (target 2026-10-28). SLOP020 (tools/lint_slop.py)
counts every comparison still outside check 1 until it is gone.

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
arithmetic in float64 is held to the oracle first, within the float64
rounding of the computation itself (`assert_computes_the_oracle`).
"""

import dataclasses
import hashlib
import math

import jax
import jax.numpy as jnp
import numpy as np

from dew.accuracy import distance, reduction_bound, within

PARITY = 1.5
"""The owner's factor (bible §7, 2026-10-20): each leaf's RMS error from float64 within 1.5 times the
reference's, on the same inputs, with no floor of any kind, check 1's (`assert_as_exact`) and every
rule's here."""

ROW_FACTOR = 4.0
"""How far Dew's worst row may sit from float64 against the reference's worst
row (`rows_over`). An RMS over every entry averages a few wrong rows into a
large tensor, so each comparison also holds Dew's worst row to the
reference's. The worst of many rows is an extreme value, not an average,
and parts further by chance: over the suite's 3,660 comparisons whose rows
differ from float64 at all (99 files, 2026-10-14), Dew's worst row over the
reference's sat at a median of 0.21, p99 1.55 and at most 2.42 (Mixtral's
exported logits). A row that differs rather than rounds moves further: the
suite's deliberate failures, a GDN chunk's plain fp32 prefix and a MiMo tower
with its sinks zeroed, put their worst rows at 7.1 and 3,419 times the
reference's."""

PRECISION = "precision"
"""The entry of a reference's array file that records the precision its generator's matmuls took."""


def ieee_fixture(path) -> dict[str, np.ndarray]:
    """The arrays of a reference's array file, refused unless its generator
    recorded IEEE matmuls under `PRECISION`.

    The rule above takes the reference's error as its budget. A reference
    regenerated with TF32 matmuls, ten mantissa bits, sits farther from
    float64 and so widens the budget to fit a coarser Dew, and the float32
    arrays it stores look the same; only the generator's record tells."""
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    recorded = str(arrays.pop(PRECISION)) if PRECISION in arrays else "no precision"
    if recorded != "ieee":
        raise ValueError(f"{path} records {recorded!r} for its matmuls, not 'ieee'; regenerate it "
                         f"with IEEE matmuls")
    return arrays


def widened(tree):
    """`tree` as float64 host arrays, its integer leaves as they are: the
    inputs of a float64 truth."""
    return jax.tree.map(lambda leaf: np.asarray(
        leaf, np.float64 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf.dtype), tree)


def equations(graph):
    """Every equation of a jaxpr, its sub-jaxprs' (scan, cond, custom rules) in turn."""
    if hasattr(graph, "eqns"):
        for equation in graph.eqns:
            yield equation
            yield from equations(equation.params)
    elif hasattr(graph, "jaxpr"):
        yield from equations(graph.jaxpr)
    elif isinstance(graph, dict):
        for value in graph.values():
            yield from equations(value)
    elif isinstance(graph, (tuple, list)):
        for value in graph:
            yield from equations(value)


def assert_fp32_reduction_bound(dew, reference, magnitudes, terms: int) -> None:
    """Two fp32 reductions differ by at most 2 gamma_n sum(abs(products))
    (`dew.accuracy.reduction_bound`): `magnitudes` holds the elementwise sum of
    absolute products and `terms` counts the products and any bias term in each
    reduction."""
    error = np.abs(np.asarray(dew, np.float64) - np.asarray(reference, np.float64))
    bound = reduction_bound(magnitudes, terms)
    assert within(error, bound), (float(error.max()), float(bound.max()))


RESOLUTION = 1024 * float(np.finfo(np.float64).eps)
"""The smallest row float64 resolves, over its leaf's size: a row below it is
float64's own rounding of the leaf (a packed row's padding at 4e-19 of a
leaf at 1e-3), a zero `rows_over` sizes by the reference's row. 1024
roundings is the longest chain the suite's oracles allow
(`assert_computes_the_oracle`); a row at a float32 resolution is ten orders
above it."""


def row_errors(value, truth, terms=None) -> tuple[np.ndarray, np.ndarray]:
    """Each row's float64 size and `value`'s RMS distance from float64 on it,
    a row every axis but the last. A row's size is the RMS of its float64
    values, or, where `terms` gives each entry's float64 terms
    (`float64_terms`), the larger of that and the RMS of what the row sums,
    so a row that cancels is sized by its terms' rounding and not by its
    near-zero result; a row below float64's `RESOLUTION` of its leaf has size
    zero. No other row's size enters: a small row beside large ones has its
    own. A value of rank one or less is one row, which `distance` already
    judges, and has none here."""
    truth = np.asarray(truth, np.float64)
    if truth.ndim < 2:
        return np.zeros(0), np.zeros(0)

    def rms(rows):
        return np.sqrt(np.mean(np.square(np.asarray(rows, np.float64).reshape(-1, truth.shape[-1])), axis=1))

    sizes = rms(truth) if terms is None else np.maximum(rms(truth), rms(terms))
    leaf = max(distance(truth, 0), 0.0 if terms is None else distance(terms, 0))
    sizes = np.where(sizes > RESOLUTION * leaf, sizes, 0.0)
    return sizes, rms(np.asarray(value, np.float64) - truth)


def rows_over(dew, reference, truth, terms=None) -> list[str]:
    """Where Dew's rows are further from float64 than ROW_FACTOR times the
    reference's (`row_errors`): its worst row over its own size against the
    reference's worst, and each row of size zero, a sum that cancels exactly before anything its
    terms see (a query that keeps one key), against the reference's distance
    on that same row, which is all the evidence such a row has: where the
    reference is exactly the float64 zero, Dew must be too."""
    sizes, mine = row_errors(dew, truth, terms)
    _, theirs = row_errors(reference, truth, terms)
    sized = sizes > 0
    over = []
    if sized.any():
        row = float(np.max(mine[sized] / sizes[sized]))
        their_row = float(np.max(theirs[sized] / sizes[sized]))
        if not within(row, ROW_FACTOR * their_row):
            over.append(f"the worst row is {row:.3e} from float64 over its size, the reference's "
                        f"{their_row:.3e}, allowed {ROW_FACTOR * their_row:.3e}")
    zero = ~sized & ~(mine <= ROW_FACTOR * theirs)
    if zero.any():
        at = int(np.flatnonzero(zero)[0])
        over.append(f"{int(zero.sum())} rows of float64 zero are further from it than the reference's own, "
                    f"row {at} by {mine[at]:.3e} against {theirs[at]:.3e}")
    return over


def assert_as_exact_as_the_reference(dew, reference, truth, label: str, *, terms=None) -> None:
    """Dew within PARITY times the reference's RMS distance from float64, and
    its rows within ROW_FACTOR times the reference's (`rows_over`), each
    sized by its float64 `terms` where they are given (`float64_terms`)."""
    mine, theirs = distance(dew, truth), distance(reference, truth)
    assert theirs > 0, f"{label}: the reference equals float64, so it measures nothing"
    assert within(mine, PARITY * theirs), (
        f"{label}: dew is {mine:.3e} from float64 (rms), the reference {theirs:.3e} "
        f"(ratio {mine / theirs:.2f}, allowed {PARITY})")
    over = rows_over(dew, reference, truth, terms)
    assert not over, f"{label}: " + "; ".join(over)



BOOTSTRAP = 2000
"""Resamples of a site's seeds for the pooled ratio's interval (`_interval`), drawn by one fixed key so
an assertion decides alike on every run."""
CONFIDENCE = 0.999
"""Check 1's stated confidence in its ratio's decisions: a pass certifies a leaf's RMS ratio no more
than PARITY², and an over by its ratio that it is above one, each but for a nominal 5e-4 (LOWER_TAIL,
UPPER_TAIL). A leaf failed by a row, a row of float64 zero or a value not finite is failed by that
rule, which certifies no ratio."""
LOWER_TAIL = 5e-4
"""The lower tail of a leaf's bootstrap interval, the over side's whole 5e-4, spent at stage 2 alone, where
an over is decided (`_staged`). Nominal: the rates hold as far as the percentile bootstrap's endpoint
coverage does, and 2,000 resamples interpolate an endpoint this far out."""
UPPER_TAIL = 2.5e-4
"""The upper tail at each of check 1's two stages: the pass side's 5e-4 spent equally over them, a pass
possible at either (nominal, as LOWER_TAIL)."""
FEWEST_SEEDS = 8
"""The fewest seeds check 1 judges by, an engineering minimum and no statistical threshold: over very few
seeds a bootstrap has so few distinct resamples that its interval can sit wholly to one side of PARITY by
the draws' luck, so fewer is Inconclusive whatever their ratio reads; above it the measured interval
decides. `seeds_for` never declares fewer."""
REDRAW = 4
"""A failing site's pooled ratio is drawn again over REDRAW times its seeds before it fails."""


class Inconclusive(AssertionError):
    """Check 1 could not decide a leaf: its seeds do not resolve PARITY. Never a pass; the site takes
    larger inputs or more seeds (`seeds_for`)."""


def _pooled(mine: np.ndarray, theirs: np.ndarray) -> float:
    """sqrt(sum a / sum b): the RMS ratio over every draw and element, each draw's mean squared error
    weighed alike, as the owner's estimand reads it."""
    return float(np.sqrt(np.sum(mine) / np.sum(theirs))) if np.sum(theirs) > 0 else (
        0.0 if np.sum(mine) == 0 else float("inf"))


def _interval(mine: np.ndarray, theirs: np.ndarray | None = None, control: np.ndarray | None = None,
              tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL)) -> tuple[float, float]:
    """The pooled ratio's bootstrap interval over seeds, its lower and upper `tails` left out: Dew's and
    the reference's errors resampled by the same seeds, or, for `control`, the reference's errors on
    two disjoint seed sets resampled apart (the reference compared with itself, an equally exact
    pair)."""
    rng = np.random.default_rng(0)
    count = len(mine)
    picks = rng.integers(0, count, (BOOTSTRAP, count))
    if control is None:
        ratios = np.sqrt(mine[picks].sum(axis=1) / np.maximum(theirs[picks].sum(axis=1), 1e-300))
    else:
        other = rng.integers(0, len(control), (BOOTSTRAP, len(control)))
        ratios = np.sqrt(mine[picks].sum(axis=1) / np.maximum(control[other].sum(axis=1), 1e-300))
    return float(np.quantile(ratios, tails[0])), float(np.quantile(ratios, 1 - tails[1]))


def _resolves(low: float, high: float) -> bool:
    """Whether a pooled ratio's interval is no wider than PARITY, so it tells an equally exact
    computation from one PARITY off."""
    return high <= PARITY * max(low, 1e-300)


def _status(ratio: float, low: float, high: float, *, contained: bool = True) -> str:
    """A leaf's verdict by its pooled ratio and interval: "pass", "over" or "inconclusive". The interval
    is taken to hold the ratio (low and high widened to it, which no endpoint's coverage loses by), so a
    pass certifies the ratio's high end at most PARITY², whether the interval lies wholly at or under
    PARITY or is resolved (no wider than PARITY) with the ratio under it, and an over its low end above
    one. Straddling PARITY and wider than it, the leaf is inconclusive, as is a ratio or end that is not
    a number. Not `contained`, the rule before 2026-10-21: the interval as it is, a leaf over PARITY
    by its ratio wherever the interval does not straddle it unresolved."""
    if not contained:
        if low <= PARITY < high and not _resolves(low, high):
            return "inconclusive"
        return "over" if ratio > PARITY else "pass"
    if not all(np.isfinite(value) for value in (ratio, low, high)):
        return "inconclusive"
    low, high = min(low, ratio), max(high, ratio)
    if high <= PARITY:
        return "pass"
    if low > PARITY:
        return "over"
    if _resolves(low, high):
        return "pass" if ratio <= PARITY else "over"
    return "inconclusive"


@dataclasses.dataclass
class _Draws:
    """Each leaf's errors over a run of seeds: Dew's and the reference's mean squared error from
    float64 (seed by leaf), each seed's worst sized row over its size (Dew's, the reference's), each
    seed's count of rows of float64 zero where Dew is further from it than ROW_FACTOR times the
    reference, and whether every value of the leaf was finite on every seed."""
    names: list[str]
    mine: np.ndarray
    theirs: np.ndarray
    worst: np.ndarray
    zero_rows: np.ndarray
    finite: np.ndarray

    def joined(self, other: "_Draws") -> "_Draws":
        return _Draws(self.names, *(np.concatenate([getattr(self, field), getattr(other, field)])
                                    for field in ("mine", "theirs", "worst", "zero_rows", "finite")))


SKETCH = 32
"""The random sign projections a recording keeps of each leaf of its reference's float64 truth
(`truth_sketch`): a move of the truth independent of the signs, a wrong row or index, moves their root
mean square by about its own norm (a single element's by exactly its size); a move built against the
signs, in the span their complement leaves, is not seen, which no fixed sketch can exclude."""
SKETCH_DEVIATIONS = 2.0
"""How many times the oracle rule's bound on a difference's norm the root mean square of its SKETCH sign
projections may reach. A truth recomputed within the oracle rule, its difference independent of the
signs, is refused with probability under 1e-11 (a Rademacher sum's square is no heavier-tailed than a
normal's: Chernoff's bound on a chi-square of SKETCH degrees past 2² times its mean). A single
element's move past twice the bound and the sketches' own rounding (`sketch_rounding`) is refused
always; a wider move past them in norm is refused but
for its chance of lying near the signs' complement, which is no guarantee: two entries swapped whose
sign words agree move no projection, one pair in 2^32 at a seed, and each seed draws its own signs, so
a wrong truth repeated over a site's seeds hides at all of them only by that chance to their power."""
SKETCH_BLOCK = 64
"""The entries `truth_sketch` sums in float64 in one block, each block's sum then summed exactly: its
rounding at most gamma(SKETCH_BLOCK) of the magnitudes', in whatever order a host takes a block."""


def _sign_bits(size: int, seed: int) -> np.ndarray:
    """SKETCH random signs for each of `size` entries at a recorded seed, one bit a sign, the same on every
    host and run, and drawn apart at every seed."""
    return np.random.default_rng([0x5E7C4, size, seed]).integers(0, 1 << SKETCH, size, dtype=np.uint64)


def truth_sketch(truth, seed: int) -> np.ndarray:
    """A float64 truth leaf's sketch `[SKETCH]` at a recorded seed: its entries summed under SKETCH random
    signs of that seed's own (`_sign_bits`), each sign's
    product exact, in blocks of SKETCH_BLOCK whose float64 sums are summed exactly (`math.fsum`), so a
    truth recomputed apart is held to the recorded one without the recording keeping it, every host
    rounding the sketch alike within `sketch_rounding` (`indexed_recording`)."""
    flat = np.asarray(truth, np.float64).reshape(-1)
    if not flat.size:
        return np.zeros(SKETCH)
    bits, shifts, partials = _sign_bits(flat.size, seed), np.arange(SKETCH, dtype=np.uint64), []
    for start in range(0, flat.size, 1 << 16):
        chunk = flat[start:start + (1 << 16)]
        signs = ((bits[start:start + chunk.size, None] >> shifts) & 1).astype(np.float64) * 2 - 1
        products = chunk[:, None] * signs
        padded = -chunk.size % SKETCH_BLOCK
        products = np.concatenate([products, np.zeros((padded, SKETCH))])
        partials.append(products.reshape(-1, SKETCH_BLOCK, SKETCH).sum(axis=1))
    stacked = np.concatenate(partials)
    return np.asarray([math.fsum(stacked[:, index].tolist()) for index in range(SKETCH)])


def sketch_rounding(magnitude: float) -> float:
    """The most `truth_sketch` of a leaf whose magnitudes sum to `magnitude` may round by, on any host:
    gamma(SKETCH_BLOCK) of it within the blocks, and the exact sum's one rounding."""
    eps = float(np.finfo(np.float64).eps)
    return (SKETCH_BLOCK * eps / 2 / (1 - SKETCH_BLOCK * eps / 2) + eps / 2) * magnitude


@dataclasses.dataclass(frozen=True)
class RecordedLeaf:
    """A recorded reference's evidence for one leaf at one seed, all check 1 reads of it (`_drawn`): its
    mean squared error from its own float64 truth, its worst sized row over its size, its RMS error on
    each row of float64 zero by the row's index (the zero-row rule), and whether it and its truth were
    finite."""
    mse: float
    worst: float
    zero_rows: np.ndarray
    zero_errors: np.ndarray
    finite: bool
    dtype: str


def leaf_summary(reference, truth) -> RecordedLeaf:
    """`RecordedLeaf` of a leaf against its float64 truth: check 1's evidence of the reference's leaf, or
    of Dew's (`_drawn` reads both so, a recorded reference's as recorded). An empty leaf is exact."""
    dtype = str(jnp.dtype(getattr(reference, "dtype", np.asarray(reference).dtype)))
    reference, truth = np.asarray(reference, np.float64), np.asarray(truth, np.float64)
    if not reference.size:
        assert truth.shape == reference.shape, (f"an empty leaf {reference.shape} against a truth "
                                                f"{truth.shape}")
        return RecordedLeaf(0.0, 0.0, np.zeros(0, np.int64), np.zeros(0), finite=True, dtype=dtype)
    sizes, rows = row_errors(reference, truth)
    sized = sizes > 0
    return RecordedLeaf(float(np.mean(np.square(reference - truth))),
                        float(np.max(rows[sized] / sizes[sized])) if sized.any() else 0.0,
                        np.flatnonzero(~sized), rows[~sized],
                        finite=bool(np.isfinite(reference).all() and np.isfinite(truth).all()), dtype=dtype)


def _drawn(dew_fn, reference_fn, draw, seeds, truth_fn=None, reference_seeds=None) -> _Draws:
    """Every leaf's errors (`_Draws`) over `seeds`: both functions run on `draw(seed)`'s inputs, truth
    `truth_fn` on them (by default `reference_fn` on them widened to float64). With `reference_seeds`,
    the reference runs on those draws instead, each held to its own draw's truth (`null_runs`), and a
    row of float64 zero is paired only where the two truths are the same values (an orbit's draws)."""
    mine, theirs, worst, zero_rows, finite, names = [], [], [], [], [], None

    def truth_of(inputs):
        with jax.enable_x64(new_val=True):
            return (truth_fn or reference_fn)(*widened(inputs))

    for at, seed in enumerate(seeds):
        inputs = draw(seed)
        dew, truth = dew_fn(*inputs), truth_of(inputs)
        if reference_seeds is None:
            reference, their_truth = reference_fn(*inputs), truth
        else:
            their_inputs = draw(reference_seeds[at])
            reference, their_truth = reference_fn(*their_inputs), truth_of(their_inputs)
        paths = jax.tree_util.tree_leaves_with_path(dew)
        names = names or [jax.tree_util.keystr(path) for path, _ in paths]
        row = []
        for (path, leaf), their_leaf, true, their_true in zip(paths, jax.tree.leaves(reference),
                                                              jax.tree.leaves(truth),
                                                              jax.tree.leaves(their_truth), strict=True):
            recorded = isinstance(their_leaf, RecordedLeaf)
            their = their_leaf if recorded else leaf_summary(their_leaf, their_true)
            if np.size(leaf) == 0:
                # An empty leaf (a one-tap convolution's history) is exact by construction: held to the
                # reference's shape and dtype (a recorded one's shape its truth's), and the truth empty
                # alike (in float64, its dtype is its own), it adds no error and resolves nothing.
                kept = (np.shape(leaf), str(jnp.dtype(leaf.dtype)))
                held = (np.shape(true) if recorded else np.shape(their_leaf), their.dtype)
                assert kept == held, (jax.tree_util.keystr(path), kept, held)
                assert np.shape(true) == np.shape(leaf), (jax.tree_util.keystr(path), np.shape(true))
                row.append((0.0, 0.0, 0.0, 0.0, 0, True))
                continue
            # Rows pair where the two truths are the same values: the same draw, or an orbit's draws.
            paired = (recorded or their_truth is truth
                      or np.array_equal(np.asarray(true), np.asarray(their_true)))
            row.append(_paired(jax.tree_util.keystr(path), leaf_summary(leaf, true), their, paired=paired))
        mine.append([entry[0] for entry in row])
        theirs.append([entry[1] for entry in row])
        worst.append([(entry[2], entry[3]) for entry in row])
        zero_rows.append([entry[4] for entry in row])
        finite.append([entry[5] for entry in row])
    return _Draws(names, np.asarray(mine), np.asarray(theirs), np.asarray(worst).reshape(len(mine), -1, 2),
                  np.asarray(zero_rows), np.asarray(finite))


def _paired(name: str, mine: RecordedLeaf, theirs: RecordedLeaf, *, paired: bool = True) -> tuple:
    """One seed's evidence of a leaf for check 1 (`_Draws`), Dew's and the reference's alike from
    `leaf_summary`: their mean squared errors, their worst sized rows, how many rows of float64 zero Dew
    is further from than ROW_FACTOR times the reference is on that same row (`rows_over`'s rule for
    them), and whether every value was finite. A row of zero in Dew's truth the reference's lacks (a
    recorded truth recomputed otherwise) is refused. Not `paired`, the two held to truths of their own
    draws (`null_runs`), no row is paired."""
    if not paired:
        return mine.mse, theirs.mse, mine.worst, theirs.worst, 0, mine.finite and theirs.finite
    at_zero = dict(zip(theirs.zero_rows.tolist(), theirs.zero_errors.tolist(), strict=True))
    unrecorded = [at for at in mine.zero_rows.tolist() if at not in at_zero]
    if unrecorded:
        raise AssertionError(f"{name}: the truth holds rows of float64 zero the reference's did not, "
                             f"{unrecorded[:4]}")
    zero = sum(1 for at, error in zip(mine.zero_rows.tolist(), mine.zero_errors.tolist(), strict=True)
               if not error <= ROW_FACTOR * at_zero[at])
    return mine.mse, theirs.mse, mine.worst, theirs.worst, zero, mine.finite and theirs.finite


def _judged(draws: _Draws, index: int, tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL)
            ) -> tuple[float, float, float]:
    """A leaf's pooled ratio and its interval over the draws, its `tails` left out."""
    a, b = draws.mine[:, index], draws.theirs[:, index]
    return (_pooled(a, b), *_interval(a, b, tails=tails))


def _judgement(draws: _Draws, label: str, tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL), *,
               contained: bool = True) -> tuple[list[str], list[str]]:
    """Check 1's findings over `draws` (`assert_as_exact`): each leaf over PARITY (`_status`), a row or
    a value it fails, and each leaf its draws leave inconclusive."""
    over, unresolved = [], []
    for index, name in enumerate(draws.names):
        leaf, count = label + name, len(draws.mine)
        a, b = draws.mine[:, index], draws.theirs[:, index]
        if not draws.finite[:, index].all():
            nonfinite = int(np.sum(~draws.finite[:, index]))
            over.append(f"{leaf}: a value is not finite on {nonfinite} of {count} seeds")
            continue
        if np.sum(b) == 0:
            if np.sum(a) > 0:
                over.append(f"{leaf}: the reference equals float64 on every seed and dew does not")
            continue
        ratio, low, high = _judged(draws, index, tails)
        status = _status(ratio, low, high, contained=contained)
        if status == "inconclusive":
            unresolved.append(f"{leaf}: inconclusive: the pooled ratio {ratio:.3f}'s interval over {count} "
                              f"seeds is [{low:.3f}, {high:.3f}], wider than {PARITY}; give the site larger "
                              f"inputs or more seeds (`seeds_for`)")
        elif status == "over":
            over.append(f"{leaf}: dew's RMS error from float64 is {ratio:.3f} times the reference's over "
                        f"{count} seeds, interval [{low:.3f}, {high:.3f}] (allowed {PARITY})")
        my_row, their_row = float(np.max(draws.worst[:, index, 0])), float(np.max(draws.worst[:, index, 1]))
        if not within(my_row, ROW_FACTOR * their_row):
            over.append(f"{leaf}: the worst row over {count} seeds is {my_row:.3e} from float64 over its "
                        f"size, the reference's {their_row:.3e}, allowed {ROW_FACTOR * their_row:.3e}")
        if draws.zero_rows[:, index].any():
            over.append(f"{leaf}: {int(draws.zero_rows[:, index].sum())} rows of float64 zero over {count} "
                        f"seeds are further from it than ROW_FACTOR times the reference's own")
    return over, unresolved


PREVIOUS_TAILS = (5e-4, 5e-4)
"""The interval's tails before check 1's two stages (`_staged`'s `two_stage=False`)."""


def _staged(drawn, seeds: int, label: str, *, two_stage: bool = True) -> tuple[list[str], list[str], int]:
    """Check 1's two pre-declared stages over `drawn(start, stop)`, the draws of seeds start to stop: at
    `seeds`, a site whose every leaf passes (`_judgement`: its ratio, rows and values) passes; any leaf
    that does not draws the second stage, REDRAW times as many further seeds, and every leaf is judged
    again over all of them, that verdict final. Its findings, the leaves left inconclusive and the stage
    reached. With `two_stage` false, the procedure before it (2026-10-21), for comparison: tails
    PREVIOUS_TAILS, the interval as it is, only a leaf whose ratio is over PARITY drawing again."""
    tails = (LOWER_TAIL, UPPER_TAIL) if two_stage else PREVIOUS_TAILS
    draws = drawn(0, seeds)
    if two_stage:
        further = any(_judgement(draws, label, tails))
    else:
        further = any(np.sum(draws.theirs[:, index]) > 0 and _judged(draws, index, tails)[0] > PARITY
                      for index in range(len(draws.names)))
    if further:
        draws = draws.joined(drawn(seeds, (1 + REDRAW) * seeds))
    return (*_judgement(draws, label, tails, contained=two_stage), 2 if further else 1)


STAGES_LOG = "DEW_CHECK1_STAGES"
"""The environment variable naming a file each check-1 judgement appends its stage and verdict to, one
JSON line (`_logged`): what a train reads every site's stage-two rate from, every attempt counted."""


def _logged(label: str, stage: int, verdict: str) -> None:
    import json
    import os

    path = os.environ.get(STAGES_LOG)
    if path:
        with open(path, "a", encoding="utf-8") as log:
            log.write(json.dumps({"site": os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0],
                                  "label": label, "stage": stage, "verdict": verdict}) + "\n")


def assert_as_exact(dew_fn, reference_fn, draw, label: str, *, seeds: int, truth_fn=None) -> None:
    """Check 1 of the owner's gate (bible §7, 2026-10-20): on par. Each leaf's RMS error from float64,
    pooled over the site's declared seeds, within PARITY times the reference's, and its worst row over
    those seeds within ROW_FACTOR times the reference's worst; each row of float64 zero no further from
    it than ROW_FACTOR times the reference on that row; every value finite.

    `draw(seed)` returns the inputs of one draw, the same for a seed on every run; `dew_fn(*inputs)`
    and `reference_fn(*inputs)` return the same tree, and truth is `reference_fn` on the inputs
    widened to float64, or `truth_fn` on them where the reference fixes a precision of its own (an
    output dtype it closes over). A draw may be an orbit's: the same values under an exact symmetry
    the seed picks. `seeds` is the site's declared count, sized by `seeds_for` so the reference
    compared with itself resolves PARITY; jit the two functions so every seed reuses one compiled
    program.

    A leaf passes when its pooled ratio (`_pooled`) is at most PARITY and its seeds resolve it: the
    ratio's bootstrap interval over them (tails LOWER_TAIL and UPPER_TAIL, widened to hold the ratio)
    lying wholly at or under PARITY, or no wider than PARITY. Check 1 is two pre-declared stages
    (`_staged`): where every leaf passes at the declared seeds the site passes; where any leaf does not
    (over PARITY, inconclusive, failing a row or a value), the site draws REDRAW times as many further
    seeds and every leaf is judged again over all of them, its ratio, its interval and its rows. Over
    PARITY there, it fails, a finding in the code or the reference arm; still inconclusive there, it is
    Inconclusive, never a pass, and its site is sized again (`seeds_for`). A pass certifies the ratio at
    most PARITY² and an over by its ratio above one, each at a nominal 5e-4 (CONFIDENCE). Every
    judgement's stage is logged (STAGES_LOG). A leaf the reference holds exactly on every seed, Dew
    must too.

    This is an engineering gate, deterministic and reproducible, not a guarantee against every error
    law: "never wrong" is check 2's, each element within its operation's certified bound
    (tests/op_certificates.py)."""
    if seeds < FEWEST_SEEDS:
        raise Inconclusive(f"{label}: {seeds} seeds cannot be judged; check 1 takes at least {FEWEST_SEEDS} "
                           f"(`seeds_for`)")
    over, unresolved, stage = _staged(
        lambda start, stop: _drawn(dew_fn, reference_fn, draw, range(start, stop), truth_fn), seeds, label)
    _logged(label, stage, "over" if over else "inconclusive" if unresolved else "pass")
    if over:
        raise AssertionError("\n".join(over + unresolved))
    if unresolved:
        raise Inconclusive("\n".join(unresolved))


def seeds_for(reference_fn, draw, label: str, *, start: int = FEWEST_SEEDS, most: int = 512,
              truth_fn=None) -> int:
    """The seeds a site declares: the fewest, doubling from `start`, for which the reference compared
    with itself (its errors on two disjoint seed sets, `_interval`'s control) resolves PARITY in every
    leaf: the pooled ratio's interval no wider than PARITY. Fails past `most`, where the
    site needs larger inputs or fewer shared factors, not more seeds."""
    count = max(start, FEWEST_SEEDS)
    while count <= most:
        first = _drawn(reference_fn, reference_fn, draw, range(count), truth_fn)
        second = _drawn(reference_fn, reference_fn, draw, range(count, 2 * count), truth_fn)
        wide = [name for index, name in enumerate(first.names)
                if np.sum(first.theirs[:, index]) > 0 and np.sum(second.theirs[:, index]) > 0
                and not _resolves(*_interval(first.theirs[:, index], control=second.theirs[:, index]))]
        if not wide:
            return count
        count *= 2
    raise AssertionError(f"{label}: the reference compared with itself does not resolve {PARITY} within "
                         f"{most} seeds at {wide[:4]}; give the site larger inputs or fewer shared factors")


def recording(path) -> dict[str, np.ndarray]:
    """A recording of a reference over seeds (`indexed_recording`'s `stored`), read whole: where a pin
    holds it, from the fetched store's local copy, checked against the pin; else from the file at
    `path`, as git holds one of at most 1 MiB (tools/fetched_fixtures.py)."""
    from tool_support import load

    with np.load(load("fetched_fixtures").local(path), allow_pickle=False) as stored:
        return dict(stored)


def null_runs(reference_fn, draw, label: str, *, seeds: int, runs: int, first: int = 0,
              truth_fn=None, two_stage: bool = True, dew_fn=None) -> list[tuple[str, int]]:
    """Check 1 run `runs` times on an equally exact pair, the reference against itself on disjoint
    draws: each run's verdict, "pass", "over" or "inconclusive", and the stage it reached, a run judged
    as `assert_as_exact` judges a site (`_staged`: its declared `seeds`, its second stage, every leaf's
    ratio, interval and rows), or as the procedure before it with `two_stage` false. Run r of
    a call takes draws no other run of any call with the same `first` grid takes: seeds from
    (first + r)·2·(1 + REDRAW)·seeds on, the first half its "Dew"'s and the second its reference's, so
    the runs are independent repetitions as far as `draw` makes its draws so. The bible-§10 null
    control of a site's operation class (orders-through-check1.md): its false-red rate, every
    nonpassing verdict counted, bounds the gate's own on that class. With `dew_fn` a fault of the
    reference (a dropped term, a wrong scale), the runs measure the procedure's power on that class,
    both procedures on the same draws."""
    verdicts = []
    span = (1 + REDRAW) * seeds
    for run in range(first, first + runs):
        base = run * 2 * span

        def drawn(start, stop, base=base):
            return _drawn(dew_fn or reference_fn, reference_fn, draw, range(base + start, base + stop),
                          truth_fn, reference_seeds=range(base + span + start, base + span + stop))

        over, unresolved, stage = _staged(drawn, seeds, label, two_stage=two_stage)
        verdicts.append(("over" if over else "inconclusive" if unresolved else "pass", stage))
    return verdicts


def null_rate_bound(reds: int, runs: int, confidence: float = CONFIDENCE) -> float:
    """The one-sided Clopper-Pearson upper bound on a false-red rate at `confidence` after `reds` red
    runs of `runs`: 1 - (1 - confidence)^(1/runs) at none, so 0 of 1,400 bounds it at 0.49%."""
    from scipy import stats

    return 1.0 if reds >= runs else float(stats.beta.ppf(confidence, reds + 1, runs - reds))


def declared_seeds(path) -> dict[str, int]:
    """Each site's declared seed count in a recording of a reference over seeds (`indexed_recording`):
    its `<site>.seeds`, written at record time by `seeds_for` on the recorded reference."""
    return {name.removesuffix(".seeds"): int(value) for name, value in recording(path).items()
            if name.endswith(".seeds")}


def input_digest(inputs) -> np.uint64:
    """A seed's inputs (a dict of NumPy arrays, a torch draw's converted) as 64 bits of the SHA-256 of
    each array's name, dtype by name, shape and bytes in C order: what a recording of outputs alone
    keeps, so that inputs drawn again are checked to be the recorded ones. The dtype goes by its name, as
    some narrow formats share a byte code (float8_e4m3fn's and float8_e4m3fnuz's), and the shape as the
    array's own, a scalar's included."""
    digest = hashlib.sha256()
    for name in sorted(inputs):
        value = np.asarray(inputs[name])
        digest.update(f"{name}:{value.dtype}:{value.shape};".encode())
        digest.update(value.tobytes())
    return np.frombuffer(digest.digest()[:8], np.uint64)[0]


def indexed_recording(stored, site: str, seeds: int, *, inputs_of=None, truth_fn=None):
    """`assert_as_exact`'s draw, reference and truth for a reference recorded over seeds, where it cannot
    run in the test (a CUDA-only library, a fetched source): `stored` holds `<site>.reference.<name>` and
    `<site>.truth.<name>`, and `<site>.input.<name>` unless `inputs_of(seed)` draws a seed's inputs again
    (a seeded NumPy or torch draw the generator made them by), every array stacked over the seeds on its
    first axis, the truth in float64, and `<site>.redraw`, check 1's REDRAW it was recorded for. Inputs
    drawn again are checked against the recording's `<site>.input_digest` (`input_digest`), so a
    generator that changed since is named, not judged. A draw is `(index, inputs)`, the index an
    integer `widened` keeps; the reference and the truth read the recorded outputs at it, and Dew's
    function takes the same two arguments and ignores the index. The recording covers check 1's redraw,
    (1 + REDRAW) times the declared seeds, so a numerical failure stays a finding rather than running
    out of recorded seeds.

    A recording of the reference's evidence alone (`record_case(..., summary=True)`: `<site>.mse.<name>`,
    `<site>.worst.<name>`, `<site>.finite.<name>` and `<site>.sketch.<name>` over the seeds, the truth's
    shape `<site>.shape.<name>`, and its
    rows of float64 zero's indices and errors, `<site>.zero_rows.<name>` and `<site>.zero_errors.<name>`,
    each seed's span of them in `<site>.zero_offsets.<name>`) keeps no outputs: its reference arm gives
    check 1 that evidence (`RecordedLeaf`), and its truth is `truth_fn(inputs)`, Dew's own float64 path,
    held at every seed to the recorded truth's sketch (`truth_sketch`) within float64's rounding of the
    computation itself, the oracle rule's bound (`assert_computes_the_oracle`, its roundings counted on
    `truth_fn`'s program, `chain_roundings`): a truth recomputed wrong moves its sketch by far more, so
    check 1 never judges Dew against its own float64 path where that path is not the reference's, as
    far as a wrong truth's move is independent of the sketch's signs (SKETCH). A seed whose recomputed
    truth has a row of float64 zero the recording's had not is refused. Its reference and truth are
    host functions, called as check 1 calls them, not traced."""
    if any(name.startswith(f"{site}.sketch.") for name in stored):
        return _summarized(stored, site, seeds, inputs_of, truth_fn)

    def tree(kind):
        prefix = f"{site}.{kind}."
        return {name.removeprefix(prefix): value for name, value in stored.items() if name.startswith(prefix)}

    inputs, reference, truth = tree("input"), tree("reference"), tree("truth")
    narrow = [name for name, value in truth.items()
              if jnp.issubdtype(value.dtype, jnp.floating) and value.dtype != np.float64]
    if narrow:
        raise AssertionError(f"{site}: the recorded truth {narrow} is not float64; record it in float64, as "
                             f"no later widening recovers what a narrower recording dropped")
    if inputs_of is None and not inputs:
        raise AssertionError(f"{site}: no inputs recorded and no `inputs_of` to draw them again")
    if int(stored.get(f"{site}.redraw", -1)) != REDRAW:
        raise AssertionError(f"{site}: the recording holds no `{site}.redraw` of {REDRAW}, which "
                             f"record_case writes and the fetched store's rule reads it by")
    digests = stored.get(f"{site}.input_digest")
    if inputs_of is not None and digests is None:
        raise AssertionError(f"{site}: `inputs_of` draws the inputs again and the recording keeps no "
                             f"digest to check them by; record the outputs alone (`store_inputs=False`)")
    recorded = [*reference.values(), *truth.values(),
                *(inputs.values() if inputs_of is None else [digests])]
    count = min(len(value) for value in recorded)
    if count < (1 + REDRAW) * seeds:
        raise AssertionError(f"{site}: {count} recorded seeds, and check 1 over {seeds} may draw "
                             f"{(1 + REDRAW) * seeds}; record (1 + REDRAW) times the declared seeds")

    def draw(seed):
        if inputs_of is None:
            return np.int32(seed), {name: value[seed] for name, value in inputs.items()}
        drawn = inputs_of(seed)
        if input_digest(drawn) != digests[seed]:
            raise AssertionError(f"{site}: seed {seed}'s inputs drawn again are not the recorded ones (their "
                                 f"digest differs); the generator's draw changed since the recording")
        return np.int32(seed), drawn

    def lookup(values):
        return jax.jit(lambda index, inputs: {name: jnp.asarray(value)[index]
                                            for name, value in values.items()})

    return draw, lookup(reference), lookup(truth)


def _summarized(stored, site: str, seeds: int, inputs_of, truth_fn):
    """`indexed_recording` of a recording of the reference's evidence alone."""
    def tree(kind):
        prefix = f"{site}.{kind}."
        return {name.removeprefix(prefix): np.asarray(value) for name, value in stored.items()
                if name.startswith(prefix)}

    kinds = ("mse", "worst", "finite", "sketch", "zero_rows", "zero_errors", "zero_offsets", "shape", "dtype")
    mse, worst, finite, sketch, zero_rows, zero_errors, zero_offsets, shapes, dtypes = (
        tree(kind) for kind in kinds)
    if truth_fn is None or inputs_of is None:
        raise AssertionError(f"{site}: a recording of the reference's evidence alone takes `truth_fn`, "
                             f"Dew's float64 path, and `inputs_of`, its inputs drawn again")
    if int(stored.get(f"{site}.redraw", -1)) != REDRAW:
        raise AssertionError(f"{site}: the recording holds no `{site}.redraw` of {REDRAW}")
    digests = stored[f"{site}.input_digest"]
    count = min([len(value) for part in (mse, worst, finite, sketch) for value in part.values()]
                + [len(value) - 1 for value in zero_offsets.values()])
    if min(count, len(digests)) < (1 + REDRAW) * seeds:
        raise AssertionError(f"{site}: {min(count, len(digests))} recorded seeds, and check 1 over {seeds} "
                             f"may draw {(1 + REDRAW) * seeds}")
    roundings = {}

    def draw(seed):
        drawn = inputs_of(seed)
        if input_digest(drawn) != digests[seed]:
            raise AssertionError(f"{site}: seed {seed}'s inputs drawn again are not the recorded ones (their "
                                 f"digest differs); the generator's draw changed since the recording")
        return np.int32(seed), drawn

    def reference(index, inputs):
        index = int(index)

        def leaf(name):
            start, stop = zero_offsets[name][index], zero_offsets[name][index + 1]
            return RecordedLeaf(float(mse[name][index]), float(worst[name][index]),
                                zero_rows[name][start:stop], zero_errors[name][start:stop],
                                finite=bool(finite[name][index]), dtype=str(dtypes[name]))

        return {name: leaf(name) for name in mse}

    def truth(index, inputs):
        index = int(index)
        computed = truth_fn(inputs)
        if not roundings:
            roundings["count"] = chain_roundings(jax.make_jaxpr(truth_fn)(inputs))
        eps = float(np.finfo(np.float64).eps)
        if set(computed) != set(sketch):
            raise AssertionError(f"{site}: `truth_fn` gives the leaves {sorted(computed)}, the recording "
                                 f"{sorted(sketch)}")
        for name, value in computed.items():
            value = np.asarray(value, np.float64)
            if value.shape != tuple(shapes[name].tolist()):
                raise AssertionError(f"{site}: the truth of {name!r} recomputed is {value.shape}, the "
                                     f"recorded reference's {tuple(shapes[name].tolist())}")
            recorded = sketch[name][index]
            if not np.isfinite(value).all():
                raise AssertionError(f"{site}: seed {index}'s float64 truth of {name!r} recomputed is not "
                                     f"finite")
            # The leaf's norm, scaled by its largest magnitude first so no square overflows.
            largest = float(np.max(np.abs(value))) if value.size else 0.0
            norm = largest * float(np.linalg.norm(value / largest)) if largest > 0 else 0.0
            # The oracle rule's bound on the difference's norm, and the rounding of both sketches, the
            # recorded truth's magnitudes no more than this one's and the difference's (`sketch_rounding`).
            oracle = 2 * roundings["count"] * eps * norm
            magnitude = float(np.sum(np.abs(value))) * (1 + value.size * eps) + np.sqrt(value.size) * oracle
            bound = (SKETCH_DEVIATIONS * oracle + 2 * sketch_rounding(magnitude)) * (1 + 1e-6)
            apart = truth_sketch(value, index) - recorded
            # Its root mean square, scaled by its largest first so no square overflows.
            widest = float(np.max(np.abs(apart))) if apart.shape == recorded.shape else np.inf
            away = (widest * float(np.sqrt(np.mean(np.square(apart / widest)))) if 0 < widest < np.inf
                    else widest)
            if not (np.isfinite(bound) and away <= bound):
                raise AssertionError(
                    f"{site}: seed {index}'s float64 truth of {name!r} recomputed is not the recorded "
                    f"reference's: its sketch is {away:.3e} away (root mean square), past float64's "
                    f"rounding of the computation "
                    f"({roundings['count']} roundings); `truth_fn` does not compute what the reference's "
                    f"truth did")
        return computed

    return draw, reference, truth


def assert_no_less_exact(dew, reference, truth, label: str) -> None:
    """An unchanged computation has no larger float64 distance than its source."""
    mine, theirs = distance(dew, truth), distance(reference, truth)
    assert within(mine, theirs), (
        f"{label}: {mine:.3e} from float64 (rms), the unchanged source {theirs:.3e}")


SUMS = frozenset({"dot_general", "ragged_dot_general", "reduce_sum", "conv_general_dilated", "cumsum",
                  "scatter-add"})
"""The operations that sum terms: a leaf's terms are those its own sums add,
taken at their float64 values (`float64_terms`). A ragged dot, an expert
layer's grouped matmul, sums as a dot does within each group, and its
transpose, which the kernel's gradient runs, is one too."""

CARRIES = frozenset({"add", "add_any", "sub", "neg", "convert_element_type", "reshape", "transpose",
                     "broadcast_in_dim", "squeeze", "expand_dims", "copy", "copy_p", "reduce_precision",
                     "slice", "dynamic_slice", "concatenate", "pad", "select_n", "optimization_barrier",
                     "name", "psum", "psum_invariant", "reduce_scatter", "all_gather", "all_gather_invariant",
                     "all_to_all", "ppermute", "pvary"})
"""The operations that carry the terms of sums into a leaf without making
new ones: sums combined (tied weights, accumulated microbatches, an update
added to a parameter, devices' partials) and moved, by a remat name
(`checkpoint_name`) or a collective too.
A product or quotient of a sum by a value that is none carries them too,
scaled element by element (`_applied`), as `s_i sum_k x_ik` sums the terms
`s_i x_ik`; a product of two sums ends them."""

CALLS = frozenset({"pjit", "jit", "closed_call", "core_call", "remat", "remat2", "checkpoint",
                   "custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr"})
"""The calls whose bodies are evaluated op by op, as the computation's own:
a jit, a remat (`jax.checkpoint` binds 'remat2'), a custom rule's primal. A
shard_map's per-device program is too, on each device's blocks (`_mapped`).
A body that runs elsewhere, a Pallas kernel, is one operation, its outputs
their own size."""


def float64_forward(module, variables, *args, **kwargs):
    """`module`'s own forward of `args` in float64, and each output's terms
    (`float64_terms`): the module computing in float64 over its variables
    widened, integer inputs as they are, `kwargs` given as they are. A
    reference's float64 truth written elsewhere (a fixture's) is sized by
    these terms, what Dew's own float64 sums add."""
    with jax.enable_x64(new_val=True):
        wide = module.clone(dtype=jnp.float64)
        return float64_terms(lambda held, *given: wide.apply(held, *given, **kwargs),
                             *widened((variables, *args)))


def float64_terms(fn, *args):
    """`fn(*args)` evaluated op by op, as float64 as `args` are, and the tree of
    each output leaf's terms, alike: the root sum of squares of the terms its
    final sums add (`SUMS`), at their operands' values, carried into the leaf
    (`CARRIES`). That is the size of the sum's partial sums, which its
    rounding scales with in any order: about the leaf's own size where the
    terms' signs are independent, far above it where they cancel by design
    (a softmax's shift invariance). A leaf made by any other operation is
    its own size, and an input carries no terms, so a parameter's terms
    after an update are the update's. Nothing propagates through the layers
    behind a sum, which would bound the whole computation."""
    closed = jax.make_jaxpr(fn)(*args)
    values, energies = _evaluated(closed.jaxpr, closed.consts, jax.tree.leaves(args))
    tree = jax.tree.structure(jax.eval_shape(fn, *args))
    return tree.unflatten(values), tree.unflatten([np.sqrt(np.asarray(_or_size(energy, value)))
                                                   for energy, value in zip(energies, values, strict=True)])


def _evaluated(jaxpr, consts, args, given=None):
    """`jaxpr`'s outputs and their terms' energies, the sums of their squares
    (`float64_terms`), from `args` and the energies `given` for them (none for
    a computation's own inputs). A value no sum made has none."""
    from jax.extend.core import Literal

    wanted = _carrying(jaxpr)
    values, energies = {}, {}
    for var, value, energy in zip(jaxpr.constvars + jaxpr.invars, [*consts, *args],
                                  [None] * len(consts) + list(given or [None] * len(args)), strict=True):
        values[var], energies[var] = value, energy

    def read(var):
        return var.val if isinstance(var, Literal) else values[var]

    def energy(var):
        return None if isinstance(var, Literal) else energies.get(var)

    # Each value and energy goes once its last reader has run, as an execution
    # frees it: a chunked scan's float64 intermediates would otherwise all stay.
    last = {var: index for index, equation in enumerate(jaxpr.eqns)
            for var in equation.invars if not isinstance(var, Literal)}
    for var in jaxpr.outvars:
        if not isinstance(var, Literal):
            last.pop(var, None)
    for index, equation in enumerate(jaxpr.eqns):
        inputs = [read(var) for var in equation.invars]
        with equation.ctx.manager:
            outputs, made = _applied(equation, inputs, [energy(var) for var in equation.invars],
                                     any(var in wanted for var in equation.outvars))
        for var in equation.invars:
            if not isinstance(var, Literal) and last.get(var) == index:
                values.pop(var, None)
                energies.pop(var, None)
        for var, value, made_energy in zip(equation.outvars, outputs, made, strict=True):
            values[var], energies[var] = value, made_energy
    return [read(var) for var in jaxpr.outvars], [energy(var) for var in jaxpr.outvars]


def _carrying(jaxpr) -> set:
    """The variables whose terms reach an output of `jaxpr`."""
    from jax.extend.core import Var

    wanted = {var for var in jaxpr.outvars if isinstance(var, Var)}
    for equation in reversed(jaxpr.eqns):
        if any(var in wanted for var in equation.outvars):
            wanted.update(equation.invars[index] for index in _carried(equation)
                          if isinstance(equation.invars[index], Var))
    return wanted


def _real(var) -> bool:
    return jnp.issubdtype(var.aval.dtype, jnp.floating)


def _carried(equation) -> list[int]:
    """The operands whose terms `equation` carries to its outputs: real floats
    alone, as a complex value's terms have no order to sum in."""
    name, operands = equation.primitive.name, equation.invars
    if not all(_real(var) or not jnp.issubdtype(var.aval.dtype, jnp.inexact) for var in equation.outvars):
        return []
    floats = [index for index, var in enumerate(operands) if _real(var)]
    if name in CARRIES or _body(equation) is not None or name in ("scan", "while", "cond", "shard_map"):
        return floats
    if name == "reduce_sum":
        # A sum over unit axes adds nothing: it moves its operand, as a broadcast's transpose does.
        return floats if all(operands[0].aval.shape[axis] == 1 for axis in equation.params["axes"]) else []
    if name in ("mul", "div"):
        return floats
    return []


def _body(equation):
    """The jaxpr and constants a call runs (`CALLS`), or None."""
    if equation.primitive.name not in CALLS:
        return None
    for name in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        body = equation.params.get(name)
        if body is not None:
            return (body.jaxpr, body.consts) if hasattr(body, "consts") else (body, [])
    return None


def _squared(value):
    return jnp.square(jnp.abs(value)) if jnp.issubdtype(jnp.result_type(value), jnp.inexact) else None


def _applied(equation, inputs, given, wanted: bool):
    """`equation`'s outputs from `inputs`, and their energies where `wanted`."""
    name, params = equation.primitive.name, equation.params
    body = _body(equation)
    if wanted and body is not None:
        return _evaluated(*body, inputs, given)
    if wanted and name == "scan":
        return _scanned(params, inputs, given)
    if wanted and name == "while":
        return _looped(params, inputs, given)
    if wanted and name == "shard_map":
        return _mapped(params, inputs, given)
    if wanted and name == "cond":
        branch = params["branches"][int(inputs[0])]
        return _evaluated(branch.jaxpr, branch.consts, inputs[1:], given[1:])
    outputs = equation.primitive.bind(*inputs, **equation.primitive.get_bind_params(params))
    outputs = list(outputs) if equation.primitive.multiple_results else [outputs]
    if not wanted:
        return outputs, [None] * len(outputs)
    carried = _carried(equation)
    if name in SUMS and not carried and all(_real(var) for var in equation.outvars):
        squares = [_squared(value) if _real(var) or jnp.issubdtype(var.aval.dtype, jnp.complexfloating)
                   else value for var, value in zip(equation.invars, inputs, strict=True)]
        return outputs, [equation.primitive.bind(*squares, **equation.primitive.get_bind_params(params))]
    if not carried:
        return outputs, [None] * len(outputs)
    if name in ("mul", "div"):
        summed = [index for index in carried if given[index] is not None]
        if len(summed) != 1 or (name == "div" and summed != [0]):
            return outputs, [None] * len(outputs)
        (index,) = summed
        scale = jnp.square(jnp.abs(inputs[1 - index]))
        return outputs, [given[index] * scale if name == "mul" else given[index] / scale]
    if all(given[index] is None for index in carried):
        return outputs, [None] * len(outputs)
    operands = [_or_zero(given[index], value) if index in carried else value
                for index, value in enumerate(inputs)]
    primitive = jax.lax.add_p if name == "sub" else equation.primitive
    made = (primitive.bind(*operands) if name == "sub" else
            operands[0] if name == "neg" else
            primitive.bind(*operands, **primitive.get_bind_params(params)))
    made = list(made) if primitive.multiple_results else [made]
    return outputs, [energy if jnp.issubdtype(jnp.result_type(value), jnp.inexact) else None
                     for energy, value in zip(made, outputs, strict=True)]


def _scanned(params, inputs, given):
    """A scan's outputs and energies (`float64_terms`), its body evaluated a step at a time."""
    body = params["jaxpr"]
    constants, carries, _ = (len(group) for group in params["ft_in"].update(range(len(inputs))).unpack())
    consts, carry, xs = (inputs[:constants], inputs[constants:constants + carries],
                         inputs[constants + carries:])
    held, carry_terms, xs_terms = (given[:constants], given[constants:constants + carries],
                                   given[constants + carries:])
    ys, ys_terms = [], []
    steps = range(params["length"])
    for step in reversed(steps) if params["reverse"] else steps:
        outputs, made = _evaluated(body.jaxpr, body.consts, [*consts, *carry, *(x[step] for x in xs)],
                                   [*held, *carry_terms, *(None if t is None else t[step] for t in xs_terms)])
        carry, carry_terms = outputs[:carries], made[:carries]
        ys.append(outputs[carries:])
        ys_terms.append(made[carries:])
    if params["reverse"]:
        ys.reverse()
        ys_terms.reverse()
    stacked = [jnp.stack(column) for column in zip(*ys, strict=True)] if ys else []
    stacked_terms = [None if any(t is None for t in column) else jnp.stack(column)
                     for column in zip(*ys_terms, strict=True)] if ys_terms else []
    return [*carry, *stacked], [*carry_terms, *stacked_terms]


def _looped(params, inputs, given):
    """A while loop's carries and energies (`float64_terms`), its body evaluated a step at a time."""
    condition, body = params["cond_jaxpr"], params["body_jaxpr"]
    tested, run = params["cond_nconsts"], params["body_nconsts"]
    condition_consts, body_consts, carry = inputs[:tested], inputs[tested:tested + run], inputs[tested + run:]
    body_terms, carry_terms = given[tested:tested + run], given[tested + run:]
    while bool(_evaluated(condition.jaxpr, condition.consts, [*condition_consts, *carry])[0][0]):
        carry, carry_terms = _evaluated(body.jaxpr, body.consts, [*body_consts, *carry],
                                        [*body_terms, *carry_terms])
    return carry, carry_terms


def _mapped(params, inputs, given):
    """A shard_map's outputs and energies (`float64_terms`), its program evaluated
    op by op on each device's blocks, as the map runs it: an expert layer's
    dispatch (`dew.nn.moe`), whose grouped matmul and its transpose sum."""
    body, specs, out_specs = params["jaxpr"], tuple(params["in_specs"]), tuple(params["out_specs"])

    def program(values, energies):
        return tuple(map(tuple, _evaluated(body.jaxpr, body.consts, values, energies)))

    return jax.shard_map(program, mesh=params["mesh"], in_specs=(specs, specs),
                         out_specs=(out_specs, out_specs), axis_names=params["newly_manual_axes"],
                         check_vma=params["check_vma"])(tuple(inputs), tuple(given))


def _or_zero(energy, value):
    return jnp.zeros_like(value) if energy is None else energy


def _or_size(energy, value):
    """An output's energy: the one its sums carried, else its own size's square."""
    if not jnp.issubdtype(jnp.result_type(value), jnp.inexact):
        return np.zeros(np.shape(value))
    return _squared(value) if energy is None else energy


FALSE_FAILURE = 1e-6
"""The chance a port exactly as exact as its reference fails the K-order
rule. The suite holds about 150 rules and CI runs it some tens of times a
day, so this is one false failure in a few hundred days."""

ORDER_FACTOR = 2.0
"""The K-order and element rules' factor (`assert_as_exact_over_orders`,
`assert_elements_as_exact`), the one ORDERS is derived for: no parity, which
is PARITY's alone. These rules judge by it until they judge through check 1
over orders sized once and declared, validated against null controls and
injected faults first (bible §10); target 2026-10-28."""

ORDERS = 52
"""The rounding orders the K-order rule takes on each side: the fewest K
for which a mean of K squared distances over another such mean exceeds
ORDER_FACTOR^2 with probability at most FALSE_FAILURE when both are equally exact.
A squared RMS of Gaussian roundings is a weighted sum of one-degree
chi-squares, whose spread is widest when one direction carries all of it,
as a training step's can (Kimi Linear's updated logits spread like 3 to 5
degrees, where one draw against one exceeds the factor about 10% of the
time). At one degree the ratio is F(K, K), and F(52, 52) passes 4 with
probability 8e-7; any larger spread in degrees is more concentrated still,
so K holds for any computation (tests/test_bf16_reference.py derives it)."""


def assert_as_exact_over_orders(dew, reference, label: str) -> None:
    """Dew's RMS over ORDERS rounding orders within ORDER_FACTOR times the
    reference's RMS over the same orders.

    `dew` and `reference` hold one distance from float64 per order (each a
    `distance`), where an order is an exact symmetry of the computation, a
    permutation of the residual stream (tests/residual_orders.py), so only
    the rounding moves."""
    dew, reference = np.asarray(dew, np.float64), np.asarray(reference, np.float64)
    assert dew.shape == reference.shape == (ORDERS,), f"{label}: {dew.shape} and {reference.shape} orders"
    mine, theirs = float(np.sqrt(np.mean(dew ** 2))), float(np.sqrt(np.mean(reference ** 2)))
    assert within(mine, ORDER_FACTOR * theirs), (
        f"{label}: dew is {mine:.3e} from float64 (rms over {ORDERS} orders), the reference "
        f"{theirs:.3e} (allowed {ORDER_FACTOR} times it)")


def assert_rows_as_exact_over_orders(dew, reference, label: str, sizes=None) -> None:
    """Each row held over ORDERS rounding orders, beside `assert_as_exact_over_orders`'s global RMS:
    `dew` and `reference` `[ORDERS, rows]` hold each order's distance of each row from float64, and
    `sizes` `[rows]` each row's float64 size (its terms', `row_errors`), one where none is given. Each
    row's RMS over the orders over its size, Dew's worst within ROW_FACTOR of the reference's worst,
    as `rows_over` holds one draw."""
    _assert_rows_over_orders(dew, np.asarray(reference, np.float64)[None], label, sizes)


def _assert_rows_over_orders(dew, members, label: str, sizes=None) -> None:
    """`assert_rows_as_exact_over_orders` against the worst of `members` `[members, ORDERS, rows]`
    at each row."""
    dew, members = np.asarray(dew, np.float64), np.asarray(members, np.float64)
    assert members.shape[1:] == dew.shape and dew.shape[0] == ORDERS, (
        f"{label}: {dew.shape} and {members.shape[1:]} rows over orders, the rule takes {ORDERS} orders")
    sizes = np.ones(dew.shape[1]) if sizes is None else np.asarray(sizes, np.float64)
    sized = sizes > 0
    if not sized.any():
        return  # rows of float64 zero, which the element rule holds (`assert_elements_as_exact`)
    mine = np.sqrt(np.mean(np.square(dew), axis=0))[sized] / sizes[sized]
    theirs = np.max(np.sqrt(np.mean(np.square(members), axis=1)), axis=0)[sized] / sizes[sized]
    worst, their_worst = float(np.max(mine)), float(np.max(theirs))
    assert within(worst, ROW_FACTOR * their_worst), (
        f"{label}: dew's worst row is {worst:.3e} from float64 (rms over {ORDERS} orders, over its size), "
        f"the reference's {their_worst:.3e} (allowed {ROW_FACTOR:g} times it)")


STAGES = 4
"""The batches of ORDERS fresh orders an element is measured over before it fails
(`assert_elements_as_exact`): the first, and a confirmation of each element still flagged in each of
the rest. Provisional, from the review of the rule (2026-10-19): with errors independent across
elements and orders, K = ORDERS a side needs three stages for a familywise FALSE_FAILURE at Dory's
43.5M elements where the errors are Gaussian, and four where they are Laplace. Shapes with heavier
or rarer tails (a Student t of 3 degrees, rare bf16 flips) are certified at none of these: the stage
count comes from calibrating the rule on held-out real order traces at the real element count, and
until it has, the rule's false-failure rate on such shapes is unmeasured. Nor is it calibrated
where a sum's order no symmetry changes: that rounding is the same at every stage, so more stages
do not redraw it, and an element it parts past the factor stays flagged."""


REFERENCE = "reference"
"""The member of an ensemble that is the reference itself (`assert_elements_as_exact`)."""


def _inflation(bar, reference) -> dict:
    """How far the worst admitted member's sums `bar` raise the bar over the reference's alone, over a
    leaf's elements: the ratio's mean of sums, its median, 99th percentile and largest at elements
    where the reference is not exact, and the count of elements it makes positive where the
    reference is exact."""
    bar, reference = (np.asarray(side, np.float64).ravel() for side in (bar, reference))
    moving = reference > 0
    ratios = bar[moving] / reference[moving]
    total = float(reference.sum())
    return {"inflation": float(bar.sum()) / total if total > 0 else 1.0,
            "median": float(np.median(ratios)) if ratios.size else 1.0,
            "p99": float(np.percentile(ratios, 99)) if ratios.size else 1.0,
            "largest": float(ratios.max()) if ratios.size else 1.0,
            "positive_where_exact": int(np.count_nonzero(~moving & (bar > 0)))}


def assert_elements_as_exact(batch, label: str, *, reference: str = REFERENCE, admitted=None) -> dict:
    """Every element on par with the reference at that element, measured over fresh batches of ORDERS
    orders. `batch(stage)` gives `(dew, members)` for stage 0, 1, ...: Dew's sums and the reference's
    ensemble's, by member, each member's tree of Dew's shape, arrays or trees of them holding each
    element's absolute distance from float64 summed over the stage's ORDERS orders, the same orders
    for every side. The member `reference` is the reference itself; every other is the reference's
    math under another declared summation tree (tests/reference_trees.py), declared before Dew runs
    and computing every element at every order and stage. `admitted` maps each leaf's path to the
    members admitted there, those no less exact than the reference over the leaf
    (`assert_leaves_as_exact_over_orders` measures it), every member where it is None. `batch` is
    asked for a stage only while elements stay flagged, up to STAGES.

    An element is flagged at a stage where Dew's mean absolute error over its orders is more than
    ORDER_FACTOR times the worst admitted member's there: no floor, no error borrowed from its row or
    from a model of rounding. Two equally exact trees cross a sum's rounding boundaries at different rates, so
    at an element the reference's own tree can be lucky in every order; the worst admitted member says
    what the reference's accuracy allows there, and a member is admitted only where it is no less exact
    than the reference over its leaf, so the ensemble moves the reference's accuracy between elements
    and never loosens it overall. Where every admitted member measures zero and Dew does not, the
    element is flagged; where all measure zero, it passes. An element fails only where every stage
    flags it, each re-measuring it over fresh orders on every side. Any non-finite value fails at
    once. The mean absolute error rather than the RMS: a bf16 rounding's flips give errors
    heavier-tailed than a Gaussian's, whose squares' mean varies with their kurtosis.

    Returns, by leaf, the members admitted and the bar's inflation over the reference's alone
    (`_inflation`): the worst admitted member's mean absolute error over the leaf's elements, over the
    reference's, and its spread over the elements."""
    flagged, report = None, {}
    for stage in range(STAGES):
        dew, members = batch(stage)
        assert reference in members, f"{label}: the ensemble {sorted(members)} holds no {reference!r}"
        paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(dew)]
        mines = [np.asarray(leaf, np.float64) for leaf in jax.tree.leaves(dew)]
        named = {member: [np.asarray(leaf, np.float64) for leaf in jax.tree.leaves(tree)]
                 for member, tree in members.items()}
        for member, leaves in named.items():
            assert len(leaves) == len(mines), (
                f"{label}: member {member!r} has {len(leaves)} leaves, Dew {len(mines)}")
        bars = []
        for index, (name, mine) in enumerate(zip(paths, mines, strict=True)):
            held = list(members) if admitted is None else list(admitted[name])
            assert reference in held, f"{label} {name}: the admitted members {held} leave out {reference!r}"
            for member, leaves in named.items():
                theirs = leaves[index]
                assert theirs.shape == mine.shape, (
                    f"{label} {name}: member {member!r} {theirs.shape}, Dew {mine.shape}")
                broken = int(np.count_nonzero(~np.isfinite(theirs)))
                assert broken == 0, (
                    f"{label} {name}: {broken} non-finite sums of member {member!r} at stage {stage}")
            broken = int(np.count_nonzero(~np.isfinite(mine)))
            assert broken == 0, f"{label} {name}: {broken} non-finite element sums of Dew's at stage {stage}"
            worst = np.max(np.stack([named[member][index] for member in held]), axis=0)
            bars.append(worst)
            if stage == 0:
                report[name] = {"admitted": held, **_inflation(worst, named[reference][index])}
        over = [mine > ORDER_FACTOR * bar for mine, bar in zip(mines, bars, strict=True)]
        # A confirmation clears an element it does not flag again; one flagged before stays so only if it is.
        flagged = over if flagged is None else [now & then for now, then in zip(over, flagged, strict=True)]
        if not any(leaf.any() for leaf in flagged):
            return report
    failing = [(name, leaf, mine, bar)
               for name, leaf, mine, bar in zip(paths, flagged, mines, bars, strict=True) if leaf.any()]
    name, leaf, mine, bar = failing[0]
    at = np.unravel_index(int(np.argmax(np.where(leaf, mine - ORDER_FACTOR * bar, -np.inf))), leaf.shape)
    count = sum(int(np.count_nonzero(leaf)) for _, leaf, _, _ in failing)
    raise AssertionError(
        f"{label}: {count} elements past {ORDER_FACTOR:g} times the worst admitted member's error at every "
        f"one of {STAGES} stages of {ORDERS} orders; in {name}, element {tuple(int(i) for i in at)} is "
        f"{mine[at] / ORDERS:.3e} from float64 at the last (mean absolute over its orders), the worst "
        f"admitted member {bar[at] / ORDERS:.3e} ({report[name]['admitted']})")


def assert_graph_as_exact(dew, reference, truth, inputs, orbit, label: str) -> dict:
    """A computation held where it rounds (bible §7's source-aware gate; tests/reference_graph.py).
    `dew`, `reference` and `truth` are `Graph`s (src/dew/nn/kernels/graph.py) of one shape, `truth`
    the reference's in float64;
    `inputs` the case's inputs by name, and `orbit(order)` the case's exact symmetries:
    `(inputs, back)`, order `order`'s permuted inputs and the map that takes outputs back.

    1. Each graph's replay equals its native witness (`Graph.fused`) bit for bit, on order 0's inputs
       and every order's of the end-to-end check.
    2. Every node is held on common parents, both the reference's values and Dew's, the float64 truth
       computed on the same parents widened. A `Sum` or `Online` node is held over ORDERS orders of its
       own summed axis permuted, an exact symmetry of it that redraws both sides' summation order, by
       `assert_leaves_as_exact_over_orders`, its permuted output axes taken back (`Sum.follows`); an
       `Ordered` node so over its declared orbit with its order-preserving members beside the
       reference; a `Pointwise` node, or an `Ordered` one with no orbit, once, each element within
       ORDER_FACTOR of the worst of the reference's node and its admitted members (within
       ADMISSION_MARGIN of its RMS), its leaf within ORDER_FACTOR of the reference's alone. Errors follow
       `reference_graph.errors`: a NaN anywhere fails, and matching infinities count zero only at a
       node's sentinel.
    3. End to end, each output's RMS distance from float64 over ORDERS orders of `orbit` within
       ORDER_FACTOR of the reference's alone.

    Returns each node's report (`assert_elements_as_exact`'s) by node and parent bundle. The element
    rule's calibration at a test's real element count is still open: the report states each node's
    measured inflation, and no false-failure rate is claimed."""
    from dew.nn.kernels.graph import Pointwise

    names = [node.name for node in dew.nodes]
    assert names == [node.name for node in reference.nodes] == [node.name for node in truth.nodes], (
        f"{label}: the graphs name other nodes: {names}, {[node.name for node in reference.nodes]}")
    for graph, side in ((dew, "dew"), (reference, "the reference")):
        _assert_witness(graph, inputs, f"{label} {side}")
    bundles = {"reference's parents": reference.replay(inputs), "dew's parents": dew.replay(inputs)}
    report = {}
    for name in names:
        mine, theirs, true = dew.node(name), reference.node(name), truth.node(name)
        assert type(mine.kind) is type(theirs.kind), f"{label} {name}: {mine.kind} against {theirs.kind}"
        for bundle, values in bundles.items():
            parents = {parent: values[parent] for parent in mine.parents}
            wide = widened(parents)
            where = f"{label} {name} on {bundle}"
            if isinstance(mine.kind, Pointwise) or not mine.kind.axes:
                report[name, bundle] = _assert_node_once(mine, theirs, true, parents, wide, where)
            else:
                report[name, bundle] = assert_leaves_as_exact_over_orders(
                    _node_draw(mine, theirs, true, parents, wide, mine.kind), where, rows=False)
    _assert_end_to_end(dew, reference, truth, inputs, orbit, label)
    return report


def _node_draw(mine, theirs, true, parents, wide, kind):
    """The over-orders draw of a node on common `parents` whose `kind` names an orbit: order `order`
    permutes each named parent's axis (`kind.axes`), an exact symmetry of the node, on Dew's node, the
    reference's, its members (an `Ordered` node's) and the float64 truth's alike, and takes the output
    axes the permutation carries (`kind.follows`) back, so an element is the same element at every
    order. An element where a side and the truth both hold the node's sentinel is no error there; a
    side that misses a sentinel the truth holds is NaN there, which fails."""
    from reference_graph import permuted

    axes = kind.axes
    length = int(np.shape(parents[next(iter(axes))])[axes[next(iter(axes))]])
    sides = {"dew": mine.fn, REFERENCE: theirs.fn, **theirs.members}

    def draw(order):
        units = np.arange(length) if order == 0 else np.random.default_rng(order).permutation(length)
        back = np.argsort(units)

        def restored(value):
            value = np.asarray(value, np.float64)
            for axis in kind.follows:
                value = np.take(value, back, axis=axis)
            return value

        with jax.enable_x64(new_val=True):
            exact = restored(true.fn(**permuted(wide, axes, units)))
        found = {side: restored(fn(**permuted(parents, axes, units))) for side, fn in sides.items()}
        if mine.sentinel is not None:
            held = exact == mine.sentinel
            found = {side: np.where(held, np.where(value == mine.sentinel, 0.0, np.nan), value)
                     for side, value in found.items()}
            exact = np.where(held, 0.0, exact)
        dew = found.pop("dew")
        return {"node": dew}, {side: {"node": value} for side, value in found.items()}, {"node": exact}
    return draw


def _assert_witness(graph, inputs, label: str) -> None:
    """`graph`'s replay equal to its native witness (`Graph.fused`) bit for bit at every value both name."""
    witnessed, replayed = graph.fused(**inputs), graph.replay(inputs)
    for name, value in witnessed.items():
        assert name in replayed, f"{label}: the witness holds {name!r}, which the graph does not name"
        value, again = np.asarray(value), np.asarray(replayed[name])
        # Bit for bit: a NaN the witness holds the graph must hold too, and the node checks fail it.
        assert value.dtype == again.dtype and value.shape == again.shape and np.array_equal(
            value.view(np.uint8), again.view(np.uint8)), (
            f"{label}: {name} replayed from the graph differs from the native witness")


def _assert_node_once(mine, theirs, true, parents, wide, label: str) -> dict:
    """A node no orbit of its own redraws, held once on common `parents` (`assert_graph_as_exact`): one
    draw on each side, so its element rule's confirmations re-read the same values and an element
    flagged once stays flagged. A pointwise node is deterministic there, as it should be; an `Ordered`
    node held once between equally exact sides flags about one element in twenty at ORDER_FACTOR 2, which
    is why an `Ordered` node declares its orbit wherever it has one."""
    from reference_graph import errors

    with jax.enable_x64(new_val=True):
        exact = np.asarray(true.fn(**wide), np.float64)
    ours = errors(mine.fn(**parents), exact, mine.sentinel)
    members = {REFERENCE: errors(theirs.fn(**parents), exact, mine.sentinel)} | {
        member: errors(fn(**parents), exact, mine.sentinel) for member, fn in theirs.members.items()}
    native = float(np.sqrt(np.mean(np.square(members[REFERENCE]))))
    admitted = [member for member, found in members.items()
                if member == REFERENCE
                or float(np.sqrt(np.mean(np.square(found)))) <= ADMISSION_MARGIN * native]
    assert within(float(np.sqrt(np.mean(np.square(ours)))), ORDER_FACTOR * native), (
        f"{label}: dew is {float(np.sqrt(np.mean(np.square(ours)))):.3e} from float64 (rms), the reference "
        f"{native:.3e} (allowed {ORDER_FACTOR} times it)")
    return assert_elements_as_exact(lambda stage: ({"node": ours}, {member: {"node": members[member]}
                                                                    for member in admitted}), label)


def _assert_end_to_end(dew, reference, truth, inputs, orbit, label: str) -> None:
    """Each output's RMS from float64 over ORDERS orders of `orbit` within ORDER_FACTOR of the
    reference's, and each graph's witness equal to its replay at every order (`assert_graph_as_exact`)."""
    distances = {}
    for order in range(ORDERS):
        moved, back = orbit(order)
        for graph, side in ((dew, "dew"), (reference, "the reference")):
            _assert_witness(graph, moved, f"{label} {side} at order {order}")
        outputs = {side: back({name: np.asarray(graph.fused(**moved)[name], np.float64)
                               for name in graph.outputs})
                   for side, graph in (("dew", dew), ("reference", reference))}
        with jax.enable_x64(new_val=True):
            fused = truth.fused(**widened(moved))
            exact = back({name: np.asarray(fused[name], np.float64) for name in truth.outputs})
        for name in dew.outputs:
            for side in ("dew", "reference"):
                distances.setdefault((name, side), []).append(distance(outputs[side][name], exact[name]))
    for name in dew.outputs:
        assert_as_exact_over_orders(distances[name, "dew"], distances[name, "reference"], f"{label} {name}")


def _batch(draw, stage: int, terms, rows: bool) -> tuple[object, dict, list, list, dict]:
    """Batch `stage` of ORDERS draws (`assert_leaves_as_exact_over_orders`), orders `stage` ORDERS on:
    each leaf's absolute distance from float64 at every element summed over them, Dew's in Dew's tree
    and each reference member's so, by member; each order's RMS distance of each leaf, and,
    where `rows`, of each row (`row_errors`), Dew's and each member's; and each leaf's row sizes, from
    the batch's first truth and `terms`."""
    dew_sums, member_sums, leaves, row_distances, sizes = {}, {}, [], [], {}
    for order in range(stage * ORDERS, (stage + 1) * ORDERS):
        dew, members, truth = draw(order)
        summed = jax.tree.leaves(truth) if terms is None else jax.tree.leaves(terms)
        named = {member: jax.tree.leaves(tree) for member, tree in members.items()}
        at_order, rows_at_order = {}, {}
        for index, ((path, mine), true, size) in enumerate(zip(
                jax.tree_util.tree_leaves_with_path(dew), jax.tree.leaves(truth), summed, strict=True)):
            name, true = jax.tree_util.keystr(path), np.asarray(true, np.float64)
            theirs = {member: leaves_of[index] for member, leaves_of in named.items()}

            def add(sums, side, name=name, true=true):
                error = np.abs(np.asarray(side, np.float64) - true)
                sums[name] = error if name not in sums else sums[name] + error

            add(dew_sums, mine)
            for member, leaf in theirs.items():
                add(member_sums.setdefault(member, {}), leaf)
            at_order[name] = (distance(mine, true),
                              {member: distance(leaf, true) for member, leaf in theirs.items()})
            if rows:
                held = None if terms is None else size
                row_sizes, ours = row_errors(mine, true, held)
                sizes.setdefault(name, row_sizes)
                rows_at_order[name] = (ours, {member: row_errors(leaf, true, held)[1]
                                              for member, leaf in theirs.items()})
        leaves.append(at_order)
        row_distances.append(rows_at_order)
    # The sums in Dew's own tree, so the element rule names each leaf by its path.
    structure, names = jax.tree.structure(dew), list(dew_sums)
    return (jax.tree.unflatten(structure, [dew_sums[name] for name in names]),
            {member: jax.tree.unflatten(structure, [sums[name] for name in names])
             for member, sums in member_sums.items()},
            leaves, row_distances, sizes)


ADMISSION_MARGIN = 1.05
"""How much less exact than the reference, in RMS from float64 over a leaf, a member may be and still be
admitted (`admitted_members`): 5%, the owner's (bible §7). With no margin an equally exact member is
admitted about half the time, by its orders' luck alone."""

ADMISSION_CONFIDENCE = 0.99
"""The simultaneous one-sided confidence of `admitted_members`' bound, over every candidate and leaf."""

def admitted_members(native, candidates) -> dict[str, list[str]]:
    """The members admitted at each leaf. `native` maps each leaf's path to the reference's mean squared
    distance from float64 over the leaf at each order of an admission sample (`_admission`), and
    `candidates` maps each other member to its own so, at the same orders. A candidate is admitted at a
    leaf where the one-sided ADMISSION_CONFIDENCE upper bound on the mean over the orders of
    d = its distance - ADMISSION_MARGIN^2 times the reference's is at most zero: a Student t bound on the
    paired differences, the orders the units, simultaneous over every candidate and leaf (Bonferroni).
    The orders are shared, so pairing takes out what every member's rounding at an order has in common.
    The reference is admitted everywhere."""
    from scipy import stats

    tests = max(1, sum(len(leaves) for leaves in candidates.values()))
    admitted = {}
    for leaf, reference in native.items():
        reference = np.asarray(reference, np.float64)
        held = [REFERENCE]
        for member, leaves in candidates.items():
            difference = np.asarray(leaves[leaf], np.float64) - ADMISSION_MARGIN ** 2 * reference
            spread = np.std(difference, ddof=1) / np.sqrt(difference.size)
            quantile = stats.t.ppf(1 - (1 - ADMISSION_CONFIDENCE) / tests, difference.size - 1)
            if float(np.mean(difference) + quantile * spread) <= 0:
                held.append(member)
        admitted[leaf] = held
    return admitted


def _admission(draw) -> dict[str, list[str]]:
    """The members `draw`'s ensemble admits at each leaf (`admitted_members`), over a sample of ORDERS
    orders of its own past the STAGES batches that hold Dew: each member's mean squared distance from
    float64 over each leaf at each order, Dew's draws unread."""
    native, candidates = {}, {}
    for order in range(STAGES * ORDERS, (STAGES + 1) * ORDERS):
        _, members, truth = draw(order)
        assert REFERENCE in members, f"the ensemble {sorted(members)} holds no {REFERENCE!r}"
        paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(truth)]
        for member, tree in members.items():
            found = candidates.setdefault(member, {}) if member != REFERENCE else native
            for name, leaf, true in zip(paths, jax.tree.leaves(tree), jax.tree.leaves(truth), strict=True):
                found.setdefault(name, []).append(distance(leaf, true) ** 2)
    return admitted_members(native, candidates)


def assert_leaves_as_exact_over_orders(draw, label: str, *, terms=None, rows: bool = True) -> dict:
    """The K-order gate over a comparison's rounding draws. `draw(order)` gives one draw `(dew,
    members, truth)`: order 0 the case's own, every other the same input under an exact symmetry of
    the computation that reorders its sums (tests/residual_orders.py), and `members` the reference's
    ensemble, by name, each its run of the same math under one declared summation tree
    (tests/reference_trees.py) or, for an external reference whose tree is fixed, that run beside
    Dew's float32 references. Every order's trees are in order 0's layout (a permuted output's
    cotangent permuted with it, a permuted parameter's gradient permuted back). `terms`, None or
    each leaf's float64 terms in that layout (`float64_terms`, order 0's), size the rows: a symmetry
    moves no row's size, so a draw computes them once, not once an order. Another input is another
    element, not another rounding of this one, so it is no order. A sum no symmetry reorders
    (attention's over keys, where a kernel fixes their order) is reordered only by the trees the
    members sum it under, and the orbit a test draws names the axes it moves.

    Every element of every leaf is held by `assert_elements_as_exact`, over batches of ORDERS orders:
    orders 0 to ORDERS - 1 first, and the next ORDERS for each confirmation while an element stays
    flagged. Over the first batch each leaf's RMS distance is held too, within ORDER_FACTOR of the
    reference's (`assert_as_exact_over_orders`), and, unless `rows` is false, its worst row within
    ROW_FACTOR of the reference's worst (`assert_rows_as_exact_over_orders`): supplements, the leaf's
    rule derived for Gaussian roundings independent across orders and the row's uncalibrated (ORDERS,
    ROW_FACTOR). One draw within its factors is no evidence that a few elements
    are, so the first batch is always taken whole. After a second training step a row inherits the
    first step's rounding through the parameters it reads, which its own terms do not see, so such a
    comparison passes `rows=False` and its first step holds the rows.

    A member other than the reference (`REFERENCE`) is admitted at a leaf where it is no less exact than
    the reference there by ADMISSION_MARGIN, judged over a sample of orders of its own on the ensemble
    alone, Dew's draws unread (`_admission`); the element rule holds Dew to the worst admitted member,
    the leaf and row rules to the reference alone. Returns `assert_elements_as_exact`'s report: by
    leaf, the members admitted and the bar's inflation over the reference alone."""
    batches = {0: _batch(draw, 0, terms, rows)}
    _, members, leaves, row_distances, sizes = batches[0]
    # Eligibility is decided on the reference's ensemble alone, its own sample of orders, never on
    # Dew's; an ensemble of the reference alone has nothing to admit and takes no sample.
    admitted = ({name: [REFERENCE] for name in leaves[0]} if set(members) == {REFERENCE}
                else _admission(draw))
    for name in leaves[0]:
        mine = np.asarray([order[name][0] for order in leaves])
        theirs = {member: np.asarray([order[name][1][member] for order in leaves])
                  for member in leaves[0][name][1]}
        # The leaf and row supplements hold Dew to the reference alone, so admission moves accuracy
        # between elements without loosening the leaf's.
        assert_as_exact_over_orders(mine, theirs[REFERENCE], f"{label} {name}")
        if rows and sizes[name].size:
            ours = np.stack([order[name][0] for order in row_distances])
            reference = np.stack([order[name][1][REFERENCE] for order in row_distances])
            assert_rows_as_exact_over_orders(ours, reference, f"{label} {name}", sizes[name])

    def batch(stage):
        if stage not in batches:
            batches[stage] = _batch(draw, stage, terms, rows=False)
        return batches[stage][0], batches[stage][1]

    return assert_elements_as_exact(batch, label, admitted=admitted)


def assert_computes_the_oracle(reference_in_float64, truth, label: str, *, roundings: int) -> None:
    """The reference run in float64 within the float64 rounding of the
    computation itself of an independent float64 oracle: twice `roundings`
    (the longest chain of roundings an output goes through, its reductions'
    lengths summed along the way) float64 unit roundoffs of the truth's RMS
    scale, each run's rounding of the exact value. Two float64 runs of one
    function part by that much at most; any difference in the function moves the twin by many orders of
    magnitude more. The fp32 error is no measure of it: a near-exact fp32 path
    leaves the float64 runs a few ulps apart all the same."""
    eps = float(np.finfo(np.float64).eps)
    apart, scale = distance(reference_in_float64, truth), float(np.sqrt(np.mean(np.square(truth))))
    assert within(apart, 2 * roundings * eps * scale), (
        f"{label}: the reference in float64 is {apart:.3e} from the oracle (rms), more than float64 "
        f"rounding over {roundings} steps of its scale {scale:.3e}")


def chain_roundings(closed_jaxpr) -> int:
    """An upper bound on the roundings any output of a traced computation
    goes through, for `assert_computes_the_oracle`: each operation rounds
    once and each reduction once per term after its first (a contraction's
    terms, a convolution's window times its input features, a reduction's
    axes), summed over every operation of the jaxpr and of the jaxprs it
    calls, a scan's body once per step. A chain from an input to an output
    visits each operation at most once, so no chain makes more."""

    def length(equation) -> int:
        name, params = equation.primitive.name, equation.params
        shape = equation.invars[0].aval.shape if equation.invars else ()
        if name == "dot_general":
            (contracting, _), _ = params["dimension_numbers"]
            return int(np.prod([shape[axis] for axis in contracting], dtype=np.int64))
        if name == "conv_general_dilated":
            kernel = equation.invars[1].aval.shape
            spec = params["dimension_numbers"].rhs_spec
            return int(np.prod([kernel[axis] for axis in spec[1:]], dtype=np.int64))
        if name.startswith(("reduce_", "argmax", "argmin", "cum")) and "axes" in params:
            return int(np.prod([shape[axis] for axis in params["axes"]], dtype=np.int64))
        return 1

    def walk(jaxpr, times: int) -> int:
        total = 0
        for equation in jaxpr.eqns:
            total += times * length(equation)
            steps = equation.params.get("length", 1) if equation.primitive.name == "scan" else 1
            for value in equation.params.values():
                for inner in value if isinstance(value, (tuple, list)) else (value,):
                    body = getattr(inner, "jaxpr", inner)
                    if hasattr(body, "eqns"):
                        total += walk(body, times * steps)
        return total

    return walk(closed_jaxpr.jaxpr, 1)


def assert_rounds_where_the_reference_does(dew, reference, truth, label: str) -> None:
    """Dew within a quarter of the reference's RMS rounding error of the
    reference's own output."""
    apart, theirs = distance(dew, reference), distance(reference, truth)
    assert within(apart, theirs / 4), (
        f"{label}: dew is {apart:.3e} from the reference (rms), whose own error is {theirs:.3e} "
        f"(ratio {apart / theirs:.2f}, allowed 0.25)")
