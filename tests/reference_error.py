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

Why not once, the reference's own distance plus a floor for summing in
another order. Measured over the suite (2026-10-14: 862 float64
comparisons in 85 files), Dew's distance over the reference's sits at a
median of 1.00, p25 0.94, p75 1.07 and p95 1.55: exactly as exact, the two
part by their own rounding draws, so a ratio above one is as likely as
below. At once, 95 comparisons in 20 files failed with no defect behind
them, and holding them needed a reduction-length floor of half the
reference's own error at each, a weaker bound than twice the reference.
At twice, one comparison failed, a deliberate one (a sinkless MiMo tower).

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

import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dew.accuracy import distance, reduction_bound, within

FACTOR = 2.0

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
"""The smallest row or element float64 resolves, over its leaf's size: one
below it is float64's own rounding of the leaf (a packed row's padding at
4e-19 of a leaf at 1e-3), a zero `rows_over` sizes by the reference's. 1024
roundings is the longest chain the suite's oracles allow
(`assert_computes_the_oracle`); a row at a float32 resolution is ten orders
above it."""


def row_errors(value, truth, terms=None, *, elements: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Each row's float64 size and `value`'s RMS distance from float64 on it,
    a row every axis but the last, or each element alone where `elements`
    asks. A row's size is the RMS of its float64 values, or, where `terms`
    gives each entry's float64 terms (`float64_terms`), the larger of that
    and the RMS of what the row sums, so a row that cancels is sized by its
    terms' rounding and not by its near-zero result; a row below float64's
    `RESOLUTION` of its leaf has size zero. No other row's size enters: a
    small row beside large ones has its own. An element is sized as its row
    is where no terms are given: its own value says nothing of the sum that
    made it, and a sum of terms of either sign lands near zero somewhere in
    any large tensor, where one rounding against another parts by any ratio
    (the dv of a packed attention, a UNet's prediction): sized by its own
    value, two runs rounding alike failed 6.2% of comparisons
    (test_reference_error.py). Its terms size it alone. A value of rank one
    or less is one row, which `distance` already judges, and has none here;
    a scalar has no elements either."""
    truth = np.asarray(truth, np.float64)
    if truth.ndim < (1 if elements else 2):
        return np.zeros(0), np.zeros(0)
    width = 1 if elements else truth.shape[-1]

    def rms(rows):
        return np.sqrt(np.mean(np.square(np.asarray(rows, np.float64).reshape(-1, width)), axis=1))

    if elements and terms is None:
        rows = truth.reshape(-1, truth.shape[-1]) if truth.ndim > 1 else truth.reshape(1, -1)
        sizes = np.repeat(np.sqrt(np.mean(np.square(rows), axis=1)), rows.shape[1])
    else:
        sizes = rms(truth) if terms is None else np.maximum(rms(truth), rms(terms))
    leaf = max(distance(truth, 0), 0.0 if terms is None else distance(terms, 0))
    sizes = np.where(sizes > RESOLUTION * leaf, sizes, 0.0)
    return sizes, rms(np.asarray(value, np.float64) - truth)


def rows_over(dew, reference, truth, terms=None, floor: float = 0.0) -> list[str]:
    """Where Dew's rows, and then its elements, are further from float64 than
    ROW_FACTOR times the reference's (`row_errors`): its worst over its own
    size against the reference's worst, at least `floor` (an order floor in
    those units), and each of size zero, a sum that cancels exactly before
    anything its terms see (a query that keeps one key), against the
    reference's distance on that same row, which is all the evidence it
    has: where the reference is exactly the float64 zero, Dew must be too.
    An element is held as a row is: a wrong element of a long row, or of a
    value of rank one, moves its row's RMS by little and itself by as much as
    it is wrong. An element of size zero is held to the reference's worst
    such element instead of its own: one rounding against another, with no
    row's mean to settle them, part by any ratio, and where the reference
    computes every such element exactly Dew must too."""
    over = []
    for name, elements in (("row", False), ("element", True)):
        sizes, mine = row_errors(dew, truth, terms, elements=elements)
        _, theirs = row_errors(reference, truth, terms, elements=elements)
        sized = sizes > 0
        if sized.any():
            worst = float(np.max(mine[sized] / sizes[sized]))
            their_worst = float(np.max(theirs[sized] / sizes[sized]))
            if not within(worst, ROW_FACTOR * max(their_worst, floor)):
                over.append(f"the worst {name} is {worst:.3e} from float64 over its size, the reference's "
                            f"{their_worst:.3e}, allowed {ROW_FACTOR * max(their_worst, floor):.3e}")
        unsized = theirs[~sized]
        zero = ~sized & ~(mine <= ROW_FACTOR * (np.max(unsized) if elements and unsized.size else theirs))
        if zero.any():
            at = int(np.flatnonzero(zero)[0])
            over.append(f"{int(zero.sum())} {name}s of float64 zero are further from it than the reference's "
                        f"own, {name} {at} by {mine[at]:.3e} against {theirs[at]:.3e}")
    return over


def assert_as_exact_as_the_reference(dew, reference, truth, label: str, *, terms=None) -> None:
    """Dew within FACTOR times the reference's RMS distance from float64, and
    its rows and elements within ROW_FACTOR times the reference's
    (`rows_over`), each sized by its float64 `terms` where they are given
    (`float64_terms`)."""
    mine, theirs = distance(dew, truth), distance(reference, truth)
    assert theirs > 0, f"{label}: the reference equals float64, so it measures nothing"
    assert within(mine, FACTOR * theirs), (
        f"{label}: dew is {mine:.3e} from float64 (rms), the reference {theirs:.3e} "
        f"(ratio {mine / theirs:.2f}, allowed {FACTOR})")
    over = rows_over(dew, reference, truth, terms)
    assert not over, f"{label}: " + "; ".join(over)



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
    signs are independent keeps about its own. Each leaf's rows and elements
    are held to the reference's too (`rows_over`), unless `rows` is false:
    after a second training step a row inherits the first step's rounding
    through the parameters it reads, which its own terms do not see, so
    such a comparison holds each tensor alone and its first step holds the
    rows and elements."""
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
        # Each row and element over its own terms where they are given, the order floor in those units.
        over += [f"{jax.tree_util.keystr(path)}: {row}"
                 for row in (rows_over(mine, theirs, true, None if terms is None else summed, spread * unit)
                             if rows else ())]
    assert not over, "\n".join(over)


def _statistics(dew, reference, truth, terms=None, *,
                rows: bool = True) -> dict[str, tuple[float, float, float]]:
    """Each leaf's distances from float64, Dew's and the reference's, and the factor they are held to:
    the leaf's RMS at FACTOR and, unless `rows` is false, its worst row and element over their own
    sizes at ROW_FACTOR (`row_errors`), keyed by the leaf's path and statistic."""
    sizes = jax.tree.leaves(truth) if terms is None else jax.tree.leaves(terms)
    found = {}
    leaves = zip(jax.tree_util.tree_leaves_with_path(dew), jax.tree.leaves(reference), jax.tree.leaves(truth),
                 sizes, strict=True)
    for (path, mine), theirs, true, summed in leaves:
        name = jax.tree_util.keystr(path)
        found[name] = (distance(mine, true), distance(theirs, true), FACTOR)
        for kind, elements in (("row", False), ("element", True)) if rows else ():
            size, ours = row_errors(mine, true, None if terms is None else summed, elements=elements)
            _, others = row_errors(theirs, true, None if terms is None else summed, elements=elements)
            sized = size > 0
            if sized.any():
                found[f"{name} worst {kind}"] = (float(np.max(ours[sized] / size[sized])),
                                                 float(np.max(others[sized] / size[sized])), ROW_FACTOR)
    return found


def _over_orders(per_order: list[dict[str, tuple[float, float, float]]], label: str) -> list[str]:
    """Each statistic held over ORDERS draws: Dew's RMS over them within its factor of the reference's."""
    over = []
    for name in per_order[0]:
        values = zip(*(order[name] for order in per_order), strict=True)
        mine, theirs, factor = (np.asarray(value, np.float64) for value in values)
        ours, others = float(np.sqrt(np.mean(mine ** 2))), float(np.sqrt(np.mean(theirs ** 2)))
        if not within(ours, factor[0] * others):
            over.append(f"{label} {name}: dew {ours:.3e} from float64 (rms over {len(per_order)} orders), "
                        f"the reference {others:.3e} (ratio {ours / others:.2f}, allowed {factor[0]:g})")
    return over


def assert_leaves_as_exact_over_orders(draws, label: str) -> None:
    """Each leaf held over ORDERS rounding draws: `draws` holds that many `(dew, reference, truth)`
    triples of one tree structure, or `(dew, reference, truth, terms)`, each an independent draw of
    the rounding, another input or an exact symmetry of the computation (tests/residual_orders.py).
    Each leaf's distance from float64, and its worst row and element over their sizes, is held by
    its RMS over the draws within its factor of the reference's (`assert_as_exact_over_orders`): a
    port as exact as its reference passes but for FALSE_FAILURE of the time, one that rounds more
    fails."""
    per_order = [_statistics(*draw) for draw in draws]
    assert len(per_order) == ORDERS, f"{label}: {len(per_order)} draws, the rule takes {ORDERS}"
    over = _over_orders(per_order, label)
    assert not over, "\n".join(over)


ORDERS_REQUESTED = "DEW_ORDERS"
"""The environment variable naming the orders a CI item computes, comma-separated (`record_orders`):
where it is set, a two-stage case records those orders' statistics instead of holding them."""

ORDERS_DIR = "DEW_ORDERS_DIR"
"""The directory a CI item records its orders into, one JSON per case and order (`record_orders`)."""


def _case() -> str:
    """The running test's node id, as pytest names it in PYTEST_CURRENT_TEST."""
    return os.environ["PYTEST_CURRENT_TEST"].rsplit(" ", 1)[0]


def record_orders(label: str, statistics: dict[int, dict[str, tuple[float, float, float]]]) -> None:
    """Each order's statistics into `$DEW_ORDERS_DIR/<case>/<order>.json`, the case a hash of the node
    id: `{"case", "label", "order", "statistics": {name: [dew, reference, factor]}}`, which `reduce`
    holds once every item has run."""
    case = _case()
    directory = Path(os.environ[ORDERS_DIR]) / hashlib.sha256(case.encode()).hexdigest()[:16]
    directory.mkdir(parents=True, exist_ok=True)
    for order, found in statistics.items():
        stored = {name: list(values) for name, values in found.items()}
        (directory / f"{order}.json").write_text(json.dumps(
            {"case": case, "label": label, "order": order, "statistics": stored}))


def assert_leaves_as_exact_in_two_stages(draw, label: str, *, rows: bool = True) -> None:
    """The two-stage rule: `draw(order)` gives one draw `(dew, reference, truth, terms)` of a
    comparison, order 0 the case's own. Where order 0 holds every leaf, row and element within its
    factor, the case passes; where it does not, ORDERS draws hold it (`_over_orders`), so a draw whose
    one rounding parts by luck costs its orders and a port that rounds more still fails. Where the
    run names orders (`ORDERS_REQUESTED`), the case records those draws' statistics instead
    (`record_orders`), for CI's `reduce` over items run apart."""
    if requested := os.environ.get(ORDERS_REQUESTED):
        record_orders(label, {order: _statistics(*draw(order), rows=rows)
                              for order in map(int, requested.split(","))})
        return
    first = _statistics(*draw(0), rows=rows)
    if not _over_orders([first], label):
        return
    over = _over_orders([first, *(_statistics(*draw(order), rows=rows) for order in range(1, ORDERS))], label)
    assert not over, "\n".join(over)


def reduce(directory: Path, *, stage1: bool) -> list[str]:
    """The cases CI recorded under `directory` (`record_orders`): with `stage1`, the node ids whose order
    0 is not within its factors, which need the other orders; else each case's verdict over all
    ORDERS, which must each be present once, as failures."""
    failures, needed = [], []
    for case in sorted(path for path in Path(directory).iterdir() if path.is_dir()):
        records = [json.loads(path.read_text()) for path in sorted(case.glob("*.json"))]
        label, node = records[0]["label"], records[0]["case"]
        by_order = {record["order"]: {name: tuple(values) for name, values in record["statistics"].items()}
                    for record in records}
        if stage1:
            if 0 in by_order and _over_orders([by_order[0]], label):
                needed.append(node)
            continue
        if sorted(by_order) != list(range(ORDERS)) or len(records) != ORDERS:
            failures.append(f"{node}: orders {sorted(by_order)} recorded, "
                            f"the rule takes each of 0 to {ORDERS - 1} once")
            continue
        failures += _over_orders([by_order[order] for order in range(ORDERS)], f"{node} {label}")
    return needed if stage1 else failures


SUMS = frozenset({"dot_general", "ragged_dot_general", "reduce_sum", "conv_general_dilated", "cumsum",
                  "scatter-add"})
"""The operations that sum terms: a leaf's terms are those its own sums add,
taken at their operands' sizes, the terms of the sums chained into them where
there are any (`float64_terms`). A ragged dot, an expert layer's grouped
matmul, sums as a dot does within each group, and its transpose, which the
kernel's gradient runs, is one too."""

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
`s_i x_ik`, and a product of two sums the products of their terms."""

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
    final sums add (`SUMS`), carried into the leaf (`CARRIES`), each term at
    its operands' sizes: an operand's own terms where sums chained into it
    made them, at least its value. That is the size of the sum's partial
    sums and of those behind it, which its rounding scales with in any
    order: about the leaf's own size where the terms' signs are independent,
    far above it where they cancel by design (a softmax's shift invariance),
    upstream too, as a deep gradient's cotangent sums cancelling ones. A
    value made by any other operation (a nonlinearity, a norm's root) is its
    own size, and an input carries no terms, so a parameter's terms after an
    update are the update's."""
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
    """The variables whose terms reach an output of `jaxpr`: those carried into it, and a sum's
    operands, whose terms size its own."""
    from jax.extend.core import Var

    wanted = {var for var in jaxpr.outvars if isinstance(var, Var)}
    for equation in reversed(jaxpr.eqns):
        if any(var in wanted for var in equation.outvars):
            sized = ([index for index, var in enumerate(equation.invars) if _real(var)]
                     if equation.primitive.name in SUMS else [])
            wanted.update(equation.invars[index] for index in {*_carried(equation), *sized}
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
        # Each term at its operands' sizes: their own terms where sums made them (`_sized`).
        squares = [_sized(energy, value) if _real(var) else
                   _squared(value) if jnp.issubdtype(var.aval.dtype, jnp.complexfloating) else value
                   for var, value, energy in zip(equation.invars, inputs, given, strict=True)]
        return outputs, [equation.primitive.bind(*squares, **equation.primitive.get_bind_params(params))]
    if not carried:
        return outputs, [None] * len(outputs)
    if name in ("mul", "div"):
        summed = [index for index in carried if given[index] is not None]
        if not summed or (name == "div" and summed != [0]):
            return outputs, [None] * len(outputs)
        if len(summed) == 1:
            (index,) = summed
            scale = jnp.square(jnp.abs(inputs[1 - index]))
            return outputs, [given[index] * scale if name == "mul" else given[index] / scale]
        # A product of two sums sums the products of their terms.
        return outputs, [_sized(given[0], inputs[0]) * _sized(given[1], inputs[1])]
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


def _sized(energy, value):
    """A sum's operand's square size: its own terms' energy where sums made it, at least its value's
    square, else its value's square."""
    squared = _squared(value)
    return squared if energy is None else jnp.maximum(energy, squared)


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

ORDERS = 52
"""The rounding orders the K-order rule takes on each side: the fewest K
for which a mean of K squared distances over another such mean exceeds
FACTOR^2 with probability at most FALSE_FAILURE when both are equally exact.
A squared RMS of Gaussian roundings is a weighted sum of one-degree
chi-squares, whose spread is widest when one direction carries all of it,
as a training step's can (Kimi Linear's updated logits spread like 3 to 5
degrees, where one draw against one exceeds the factor about 10% of the
time). At one degree the ratio is F(K, K), and F(52, 52) passes 4 with
probability 8e-7; any larger spread in degrees is more concentrated still,
so K holds for any computation (tests/test_bf16_reference.py derives it)."""


def assert_as_exact_over_orders(dew, reference, label: str) -> None:
    """Dew's RMS over ORDERS rounding orders within FACTOR times the
    reference's RMS over the same orders.

    `dew` and `reference` hold one distance from float64 per order (each a
    `distance`), where an order is an exact symmetry of the computation, a
    permutation of the residual stream (tests/residual_orders.py), so only
    the rounding moves."""
    dew, reference = np.asarray(dew, np.float64), np.asarray(reference, np.float64)
    assert dew.shape == reference.shape == (ORDERS,), f"{label}: {dew.shape} and {reference.shape} orders"
    mine, theirs = float(np.sqrt(np.mean(dew ** 2))), float(np.sqrt(np.mean(reference ** 2)))
    assert within(mine, FACTOR * theirs), (
        f"{label}: dew is {mine:.3e} from float64 (rms over {ORDERS} orders), the reference "
        f"{theirs:.3e} (ratio {mine / theirs:.2f}, allowed {FACTOR})")


def assert_rows_as_exact_over_orders(dew, reference, label: str, sizes=None) -> None:
    """Each row held over ORDERS rounding orders, beside `assert_as_exact_over_orders`'s global RMS:
    `dew` and `reference` `[ORDERS, rows]` hold each order's distance of each row from float64, and
    `sizes` `[rows]` each row's float64 size (its terms', `row_errors`), one where none is given. Each
    row's RMS over the orders over its size, Dew's worst within ROW_FACTOR of the reference's worst,
    as `rows_over` holds one draw."""
    dew, reference = np.asarray(dew, np.float64), np.asarray(reference, np.float64)
    assert dew.shape == reference.shape and dew.shape[0] == ORDERS, (
        f"{label}: {dew.shape} and {reference.shape} rows over orders, the rule takes {ORDERS} orders")
    sizes = np.ones(dew.shape[1]) if sizes is None else np.asarray(sizes, np.float64)
    sized = sizes > 0
    mine, theirs = (np.sqrt(np.mean(np.square(rows), axis=0))[sized] / sizes[sized]
                    for rows in (dew, reference))
    worst, their_worst = float(np.max(mine)), float(np.max(theirs))
    assert within(worst, ROW_FACTOR * their_worst), (
        f"{label}: dew's worst row is {worst:.3e} from float64 (rms over {ORDERS} orders, over its size), "
        f"the reference's {their_worst:.3e} (ratio {worst / their_worst:.2f}, allowed {ROW_FACTOR:g})")


def element_factor(elements: int, spread: float | np.ndarray = np.pi / 2,
                   orders: int = ORDERS) -> float | np.ndarray:
    """The factor `assert_elements_as_exact_over_orders` allows each of `elements` elements: the ratio
    of two mean absolute errors over ORDERS orders that a port as exact as its reference passes at
    every element but for FALSE_FAILURE of the time (Bonferroni over the elements), at least FACTOR.
    A mean of `orders` absolute errors varies by (spread - 1) / ORDERS of its square, `spread` the
    errors' E[e^2] / E[|e|]^2 (pi / 2 for a Gaussian's, 2 for a Laplace's), so the log of the ratio
    of two is normal with variance 2 (spread - 1) / orders (the delta method): 3.06 at 43.5M
    Gaussian elements over ORDERS."""
    from scipy import stats

    deviation = np.sqrt(2 * (np.asarray(spread, np.float64) - 1) / orders)
    return np.maximum(FACTOR, np.exp(stats.norm.isf(FALSE_FAILURE / max(elements, 1)) * deviation))


def error_spread(absolutes, squares, orders: int = 2 * ORDERS) -> np.ndarray:
    """Each row's E[e^2] / E[|e|]^2 over its elements' errors: `absolutes` and `squares` `[rows,
    elements]` each element's absolute and squared errors summed over `orders` orders, both sides'
    together. Each element's own over its orders, averaged over the row's (its own draws too few to
    tell; pooled over the row's raw errors, elements of unlike scales would read as heavy tails), a
    Gaussian's pi / 2 where no element moves."""
    first, second = np.asarray(absolutes, np.float64) / orders, np.asarray(squares, np.float64) / orders
    moving = first > 0
    spread = np.where(moving, second / np.where(moving, np.square(first), 1), 0.0)
    counted = np.count_nonzero(moving, axis=1)
    return np.where(counted > 0, spread.sum(axis=1) / np.maximum(counted, 1), np.pi / 2)


def assert_elements_as_exact_over_orders(dew_absolutes, reference_absolutes, label: str, *, dew_squares,
                                         reference_squares, orders: int = ORDERS) -> None:
    """Every element held over ORDERS rounding orders where the orders cannot all be kept:
    `dew_absolutes` and `reference_absolutes` `[rows, ...]` each element's absolute distance from
    float64 summed over the ORDERS orders, and `dew_squares` and `reference_squares` its squares so
    summed (over ORDERS orders, or `orders` a side where a run holds one half of its orders to the
    other). Each element's mean absolute error over the orders within its row's `element_factor` of
    the reference's own at that element, and nothing else: no element borrows its row's error or a
    rounding model's, and where the reference is exactly float64 in every order, Dew must be too.
    The mean absolute error, not the RMS: a bf16 rounding's flips give errors heavier-tailed than a
    Gaussian's, whose squares' mean varies with their kurtosis, the absolute errors' mean with their
    spread alone (`error_spread`), so the rule keeps its power there without a false failure. A
    non-finite element fails, on either side."""
    sides = [np.asarray(x, np.float64) for x in (dew_absolutes, reference_absolutes, dew_squares,
                                                 reference_squares)]
    assert all(side.shape == sides[0].shape for side in sides) and sides[0].ndim >= 2, (
        f"{label}: {[side.shape for side in sides]}, the rule takes [rows, ...] sums of one shape")
    for name, side in zip(("dew's absolutes", "the reference's absolutes", "dew's squares",
                           "the reference's squares"), sides, strict=True):
        broken = int(np.count_nonzero(~np.isfinite(side)))
        assert broken == 0, f"{label}: {broken} of {side.size} of {name} are not finite"
    rows = sides[0].shape[0]
    dew_absolutes, reference_absolutes, dew_squares, reference_squares = (
        side.reshape(rows, -1) for side in sides)
    mine, theirs = dew_absolutes / orders, reference_absolutes / orders
    spread = error_spread(dew_absolutes + reference_absolutes, dew_squares + reference_squares, 2 * orders)
    factor = element_factor(mine.size, spread, orders)[:, None]
    failing = mine > factor * theirs
    if np.any(failing):
        ratios = np.where(theirs > 0, mine / np.where(theirs > 0, theirs, 1), np.inf)
        at = np.unravel_index(int(np.argmax(np.where(failing, ratios / factor, 0.0))), ratios.shape)
        raise AssertionError(
            f"{label}: {int(np.count_nonzero(failing))} of {mine.size} elements past the reference; the "
            f"worst, row {at[0]} element {at[1]}, is {mine[at]:.3e} from float64 (mean over {orders} "
            f"orders), the reference {theirs[at]:.3e} (allowed {factor[at[0], 0]:.2f} times it, its row's "
            f"errors of spread {spread[at[0]]:.2f})")


def assert_computes_the_oracle(reference_in_float64, truth, label: str, *, roundings: int) -> None:
    """The reference run in float64 within the float64 rounding of the
    computation itself of an independent float64 oracle: FACTOR times
    `roundings` (the longest chain of roundings an output goes through, its
    reductions' lengths summed along the way) float64 unit roundoffs of the
    truth's RMS scale, and each entry within that of the largest of the truth's
    RMS scale, its row's (a row every axis but the last) and its own size, so a
    wrong small entry is not diluted by its leaf's others, while a sum that
    cancels to zero keeps the rounding of what it summed. Two float64 runs of
    one function part by that much at most; any difference in the function
    moves the twin by many orders of magnitude more. The fp32 error is no
    measure of it: a near-exact fp32 path leaves the float64 runs a few ulps
    apart all the same."""
    eps = float(np.finfo(np.float64).eps)
    bound = FACTOR * roundings * eps
    apart, scale = distance(reference_in_float64, truth), float(np.sqrt(np.mean(np.square(truth))))
    assert within(apart, bound * scale), (
        f"{label}: the reference in float64 is {apart:.3e} from the oracle (rms), more than float64 "
        f"rounding over {roundings} steps of its scale {scale:.3e}")
    truth = np.asarray(truth, np.float64)
    rows = truth.reshape(-1, truth.shape[-1]) if truth.ndim else truth.reshape(1, 1)
    twin = np.asarray(reference_in_float64, np.float64).reshape(rows.shape)
    row_scales = np.sqrt(np.mean(np.square(rows), axis=1, keepdims=True))
    sizes = np.maximum(np.maximum(row_scales, np.abs(rows)), scale)
    entries = np.abs(twin - rows)
    over = ~(entries <= bound * sizes)
    if np.any(over):
        allowed = bound * sizes
        ratios = np.where(over, np.where(allowed > 0, entries / np.where(allowed > 0, allowed, 1), np.inf),
                          0.0)
        at = np.unravel_index(int(np.argmax(ratios)), rows.shape)
        raise AssertionError(
            f"{label}: {int(np.count_nonzero(over))} of {rows.size} entries of the reference in float64 are "
            f"past float64 rounding over {roundings} steps; the worst, row {at[0]} entry {at[1]}, is "
            f"{entries[at]:.3e} from the oracle {rows[at]:.3e}, of size {sizes[at]:.3e} (its row's rms "
            f"{float(row_scales[at[0], 0]):.3e}, the leaf's {scale:.3e}), "
            f"{ratios[at]:.1f} times the bound")


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


if __name__ == "__main__":
    import sys

    _, command, directory, *flags = sys.argv
    assert command == "reduce", "usage: python tests/reference_error.py reduce <dir> [--stage1]"
    found = reduce(Path(directory), stage1="--stage1" in flags)
    print("\n".join(found))
    sys.exit(0 if "--stage1" in flags or not found else 1)
