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

import jax
import jax.numpy as jnp
import numpy as np

FACTOR = 2.0

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


def within(error, bound) -> bool:
    """Whether every `error` is at most its `bound`, both finite. Written as the
    passing condition, so a NaN or infinite error never passes, nor does an
    infinite allowance from a reference that diverged."""
    error, bound = np.asarray(error, np.float64), np.asarray(bound, np.float64)
    return bool(np.all(np.isfinite(error) & np.isfinite(bound) & (error <= bound)))


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
    assert within(error, bound), (float(error.max()), float(bound.max()))


def distance(value, truth) -> float:
    """The root-mean-square difference over every entry, in float64."""
    difference = np.asarray(value, np.float64) - np.asarray(truth, np.float64)
    return float(np.sqrt(np.mean(np.square(difference))))


def assert_as_exact_as_the_reference(dew, reference, truth, label: str) -> None:
    """Dew within FACTOR times the reference's RMS distance from float64."""
    mine, theirs = distance(dew, truth), distance(reference, truth)
    assert theirs > 0, f"{label}: the reference equals float64, so it measures nothing"
    assert within(mine, FACTOR * theirs), (
        f"{label}: dew is {mine:.3e} from float64 (rms), the reference {theirs:.3e} "
        f"(ratio {mine / theirs:.2f}, allowed {FACTOR})")



def assert_no_less_exact(dew, reference, truth, label: str) -> None:
    """An unchanged computation has no larger float64 distance than its source."""
    mine, theirs = distance(dew, truth), distance(reference, truth)
    assert within(mine, theirs), (
        f"{label}: {mine:.3e} from float64 (rms), the unchanged source {theirs:.3e}")


def assert_leaves_as_exact(dew, reference, truth, *, reductions: int = 1, terms=None) -> None:
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
    signs are independent keeps about its own."""
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
    assert not over, "\n".join(over)


SUMS = frozenset({"dot_general", "reduce_sum", "conv_general_dilated", "cumsum", "scatter-add"})
"""The operations that sum terms: a leaf's terms are those its own sums add,
taken at their float64 values (`float64_terms`)."""

CARRIES = frozenset({"add", "add_any", "sub", "neg", "convert_element_type", "reshape", "transpose",
                     "broadcast_in_dim", "squeeze", "expand_dims", "copy", "copy_p", "reduce_precision",
                     "slice", "dynamic_slice", "concatenate", "pad", "select_n", "optimization_barrier"})
"""The operations that carry the terms of sums into a leaf without making
new ones: sums combined (tied weights, accumulated microbatches, an update
added to a parameter) and moved. A product or quotient of a sum by a value
that is none carries them too, scaled element by element (`_applied`), as
`s_i sum_k x_ik` sums the terms `s_i x_ik`; a product of two sums ends them."""

CALLS = frozenset({"pjit", "jit", "closed_call", "core_call", "remat", "checkpoint", "custom_jvp_call",
                   "custom_vjp_call", "custom_vjp_call_jaxpr"})
"""The calls whose bodies are evaluated op by op, as the computation's own:
a jit, a remat, a custom rule's primal. A body that runs elsewhere, a
shard_map's per-device program or a Pallas kernel, is one operation, its
outputs their own size."""


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
    if name in CARRIES or _body(equation) is not None or name in ("scan", "while", "cond"):
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


def assert_computes_the_oracle(reference_in_float64, truth, label: str, *, roundings: int) -> None:
    """The reference run in float64 within the float64 rounding of the
    computation itself of an independent float64 oracle: FACTOR times
    `roundings` (the longest chain of roundings an output goes through, its
    reductions' lengths summed along the way) float64 unit roundoffs of the
    truth's RMS scale. Two float64 runs of one function part by that much at
    most; any difference in the function moves the twin by many orders of
    magnitude more. The fp32 error is no measure of it: a near-exact fp32 path
    leaves the float64 runs a few ulps apart all the same."""
    eps = float(np.finfo(np.float64).eps)
    apart, scale = distance(reference_in_float64, truth), float(np.sqrt(np.mean(np.square(truth))))
    assert within(apart, FACTOR * roundings * eps * scale), (
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
