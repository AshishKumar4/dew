"""The gate's K-order ensemble rules (tests/reference_error.py's module docstring): every element of
a computation held over exact symmetries of its sums, against the worst admitted member of the
reference's ensemble (`assert_elements_as_exact`, `assert_leaves_as_exact_over_orders`), and a kernel
held where it rounds, as a graph of the values it rounds and holds (`assert_graph_as_exact`)."""


import jax
import numpy as np
from reference_error import (
    FACTOR,
    ORDERS,
    assert_as_exact_over_orders,
    assert_rows_as_exact_over_orders,
    row_errors,
    widened,
)

from dew.accuracy import distance, within

STAGES = 4
"""The batches of ORDERS fresh orders an element is measured over before it fails
(`assert_elements_as_exact`): the first, and a confirmation of each element still flagged in each of
the rest. Provisional, from the review of the rule: with errors independent across
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
    FACTOR times the worst admitted member's there: no floor, no error borrowed from its row or from a
    model of rounding. Two equally exact trees cross a sum's rounding boundaries at different rates, so
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
        over = [mine > FACTOR * bar for mine, bar in zip(mines, bars, strict=True)]
        # A confirmation clears an element it does not flag again; one flagged before stays so only if it is.
        flagged = over if flagged is None else [now & then for now, then in zip(over, flagged, strict=True)]
        if not any(leaf.any() for leaf in flagged):
            return report
    failing = [(name, leaf, mine, bar)
               for name, leaf, mine, bar in zip(paths, flagged, mines, bars, strict=True) if leaf.any()]
    name, leaf, mine, bar = failing[0]
    at = np.unravel_index(int(np.argmax(np.where(leaf, mine - FACTOR * bar, -np.inf))), leaf.shape)
    count = sum(int(np.count_nonzero(leaf)) for _, leaf, _, _ in failing)
    raise AssertionError(
        f"{label}: {count} elements past FACTOR {FACTOR:g} times the worst admitted member's error at every "
        f"one of {STAGES} stages of {ORDERS} orders; in {name}, element {tuple(int(i) for i in at)} is "
        f"{mine[at] / ORDERS:.3e} from float64 at the last (mean absolute over its orders), the worst "
        f"admitted member {bar[at] / ORDERS:.3e} ({report[name]['admitted']})")


def assert_graph_as_exact(dew, reference, truth, inputs, orbit, label: str) -> dict:
    """A computation held where it rounds (the gate's source-aware check; tests/reference_graph.py).
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
       FACTOR of the worst of the reference's node and its admitted members (within ADMISSION_MARGIN
       of its RMS), its leaf within FACTOR of the reference's alone. Errors follow
       `reference_graph.errors`: a NaN anywhere fails, and matching infinities count zero only at a
       node's sentinel.
    3. End to end, each output's RMS distance from float64 over ORDERS orders of `orbit` within FACTOR of
       the reference's alone.

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
    node held once between equally exact sides flags about one element in twenty at FACTOR 2, which
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
    assert within(float(np.sqrt(np.mean(np.square(ours)))), FACTOR * native), (
        f"{label}: dew is {float(np.sqrt(np.mean(np.square(ours)))):.3e} from float64 (rms), the reference "
        f"{native:.3e} (allowed {FACTOR} times it)")
    return assert_elements_as_exact(lambda stage: ({"node": ours}, {member: {"node": members[member]}
                                                                    for member in admitted}), label)


def _assert_end_to_end(dew, reference, truth, inputs, orbit, label: str) -> None:
    """Each output's RMS from float64 over ORDERS orders of `orbit` within FACTOR of the reference's, and
    each graph's witness equal to its replay at every order (`assert_graph_as_exact`)."""
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
admitted (`admitted_members`): 5%. With no margin an equally exact member is
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
    flagged. Over the first batch each leaf's RMS distance is held too, within FACTOR of the
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
