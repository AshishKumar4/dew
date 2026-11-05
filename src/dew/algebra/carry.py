"""A chunk's forms split by extraction into the work a walk does for every chunk at once and the work its
carry does in sequence (docs/design/algebra.md §6.9, `split`).

A function of a carried state and a chunk's operands, traced at one chunk, enters the e-graph as terms
(`terms.region`, Dew's products as their default branch's, `forms.inlined`). The values that read the state
are marked (`reads`), and the products' linear algebra saturates over them (`CARRIED`): a product of a
transposed operand as a plain product of its transpose, a product distributed over a sum or a difference that
reads the state, two products associated so the state is the last operand, two products of one right operand
factored, each where the region's range facts bound the operands its new intermediates read (`rules.tame`): no
form is infinite where the function is finite. Extraction prices the carried outputs' forms by the work that
reads the state first and the rest second (`Priced`), so the carry reads the state through the fewest products
its facts allow. The other outputs, emitted from each chunk's entering state once the carry has run, take the
program's cost (`cost.model`). The forms' largest terms that read no state are the walk's prepared values,
computed for every chunk at once (`Split.prepare`); the carry and the emitted outputs read the state and those
alone (`Split.carry`, `Split.emit`). The association reorders rounding.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import NamedTuple

import jax
from egglog import (
    DagCostModel,
    EGraph,
    String,
    Vec,
    eq,
    function,
    get_callable_args,
    i64,
    rule,
    ruleset,
    seq,
    set_,
    union,
    var,
    vars_,
)

from dew.algebra import cost, forms, program, rules, terms
from dew.algebra.cost import Work
from dew.algebra.extract import Level
from dew.algebra.forms import Refused
from dew.algebra.terms import Term
from dew.cache import ALGEBRA


@function(merge=lambda old, new: old)
def reads(term: Term) -> i64: ...
"""1 for a value that reads the carried state."""


@function(merge=lambda old, new: old)
def plain_of(params: i64) -> i64: ...
"""A product's parameters of `x^T @ y` (both operands' first axes contracted, no batch): the parameters of the
plain product (`matrix_product`) its transpose's operand takes, of the same precision and dtype."""


@function(merge=lambda old, new: old)
def swap_of(params: i64) -> i64: ...
"""For `x^T @ y`'s parameters, the parameters of the transpose that lays its left operand as `x^T`."""


def _carrying():
    """Which values read the state (`reads`)."""
    x, y, z, t = vars_("x y z t", Term)
    name = var("name", String)
    p = var("p", i64)
    one = i64(1)
    marked = []
    for made, children in ((lambda: x.binary(name, y, p), (x, y)), (lambda: x.compare(name, y, p), (x, y)),
                           (lambda: x.unary(name, p), (x,)), (lambda: x.convert(p), (x,)),
                           (lambda: x.shaped(name, p), (x,)), (lambda: x.dot(y, p), (x, y)),
                           (lambda: x.select(y, z), (x, y, z))):
        marked += [rule(eq(t).to(made()), eq(reads(child)).to(one)).then(set_(reads(t)).to(one))
                   for child in children]
    return ruleset(*marked, name="carrying")


def _carried():
    """The products' linear algebra over values that read the state (the module's account), each over
    matrices (`matrix`), products and sums in their operands' dtype (`matrix_product`, `rules.plain`) and
    summands of one shape, each firing only where every intermediate it makes is under a magnitude the
    function as written reaches (`rules.dominated`, §6.3), each stating its own (`rules.made`): each
    reorders rounding."""
    x, y, z, t = vars_("x y z t", Term)
    p, m, s, q = vars_("p m s q", i64)
    left = var("left", Vec[i64])
    length = var("length", i64)
    one = i64(1)
    shape = rules.shape
    # A matrix product's contraction's length's power: its left operand's columns'.
    contracted = (eq(left).to(shape(x)), eq(length).to(rules.log2_of(left[i64(1)])))
    transposed = x.shaped(String("transpose"), s)
    power = var("power", i64)
    sets = [
        # x^T @ y as transpose(x) @ y, so it associates and factors with plain products: exact, a layout of
        # x, whose magnitude it takes where x has one.
        rule(eq(t).to(x.dot(y, p)), eq(m).to(plain_of(p)), eq(s).to(swap_of(p)), rules.matrix(x),
             rules.matrix(y), eq(reads(y)).to(one)).then(union(t).with_(transposed.dot(y, m))),
        rule(eq(t).to(x.shaped(String("transpose"), s)), eq(power).to(rules.log2_bound(x))).then(
            *rules.made(t, power)),
    ]
    # x @ (y @ z) as (x @ y) @ z, the state's product last: x @ y new, and its product with z, each under
    # the program's reach (the new outer product contracting y's columns, as y @ z does).
    facts, power = rules.product_magnitude(rules.magnitude(x, "x"), rules.magnitude(y, "y"), length)
    inner = var("inner", Vec[i64])
    outer = rules.product_magnitude(([], power), rules.magnitude(z, "z"), var("outer_length", i64))
    sets.append(rule(eq(t).to(x.dot(y.dot(z, m), m)), rules.matrix_product(m), rules.matrix(x),
                     rules.matrix(y), rules.matrix(z), eq(reads(z)).to(one), *contracted, *facts,
                     *rules.dominated(power, "new"),
                     eq(inner).to(shape(y)), eq(var("outer_length", i64)).to(rules.log2_of(inner[i64(1)])),
                     *outer[0],
                     *rules.dominated(outer[1], "outer")).then(
        union(t).with_(x.dot(y, m).dot(z, m)), *rules.made(x.dot(y, m), power)))
    for combined in ("add", "sub"):
        plain = (rules.matrix_product(m), rules.plain(q), rules.matrix(x), rules.matrix(y), rules.matrix(z))
        # x @ (y +- z) as x @ y +- x @ z: both new.
        with_y = rules.product_magnitude(rules.magnitude(x, "x"), rules.magnitude(y, "y"), length)
        with_z = rules.product_magnitude(rules.magnitude(x, "x2"), rules.magnitude(z, "z"), length)
        sets.append(rule(eq(t).to(x.dot(y.binary(combined, z, q), m)), *plain, eq(shape(y)).to(shape(z)),
                         eq(reads(y.binary(combined, z, q))).to(one), *contracted, *with_y[0], *with_z[0],
                         *rules.dominated(with_y[1], "y"), *rules.dominated(with_z[1], "z")).then(
            union(t).with_(x.dot(y, m).binary(combined, x.dot(z, m), q)),
            *rules.made(x.dot(y, m), with_y[1]), *rules.made(x.dot(z, m), with_z[1])))
        # x @ z +- y @ z as (x +- y) @ z: x +- y new, and its product with z, each under the reach.
        facts, power = rules.summed_magnitude(rules.magnitude(x, "x"), rules.magnitude(y, "y"))
        added = x.binary(combined, y, q)
        # The new product contracting x's columns, as x @ z does (`contracted`).
        product = rules.product_magnitude(([], power), rules.magnitude(z, "z2"), length)
        sets.append(rule(eq(t).to(x.dot(z, m).binary(combined, y.dot(z, m), q)), *plain,
                         eq(shape(x)).to(shape(y)), eq(reads(z)).to(one), *facts,
                         *rules.dominated(power, "new"), *contracted, *product[0],
                         *rules.dominated(product[1], "summed")).then(
            union(t).with_(added.dot(z, m)), *rules.made(added, power)))
    return ruleset(*sets, name="carried")


CARRYING = _carrying()
CARRIED = rules.Rules("carried", "reorders", _carried(), receipt="docs/design/algebra.md §6.9; "
                      "tests/test_algebra_scan.py")
"""The split's rules, never among the engine's sets (`rules.admitted`)."""


@dataclasses.dataclass(frozen=True, order=True)
class Priced:
    """A carried form's price, compared in order: its work that reads the state (once a chunk, in sequence),
    then the rest (every chunk's at once), each in seconds under XLA's schedule (`cost.fused`), where the
    carry runs (a `lax.scan` step a chunk, recurrence_chunk's derived walk): its span at a loop iteration's
    latency, as `linear_scan`'s; a tile walk prices its span at an in-kernel step's, a dependent reduction's
    (`templates.cost.step_latency`), which the carry, a loop XLA lowers, does not run."""

    sequential: cost.Seconds
    parallel: cost.Seconds

    def __add__(self, other: Priced) -> Priced:
        return Priced(self.sequential + other.sequential, self.parallel + other.parallel)

    def levels(self) -> tuple[Level, ...]:
        """The sequential work's levels, then the parallel's (`extract.Linear`)."""
        return self.sequential.levels() + self.parallel.levels()


def _carry_priced(region: terms.Region, schedule) -> DagCostModel[Priced]:
    """`Priced` per node: its work (`cost.work`) sequential where an operand reads the state (the e-classes
    the extractor hands the callback, which alone it may look up), parallel otherwise."""
    none = cost.Seconds(Work(), schedule)

    def marginal(graph: EGraph, node) -> Priced:
        done = cost.Seconds(cost.work(graph, node, region), schedule)
        operands = next((args for constructor in terms.LANGUAGE
                         if (args := get_callable_args(node, constructor)) is not None), ())
        if any(isinstance(operand, Term) and graph.lookup_function_value(reads(operand)) is not None
               for operand in operands):
            return Priced(done, none)
        return Priced(none, done)

    return DagCostModel(marginal_cost=marginal, identity=Priced(none, none))


class Split(NamedTuple):
    """A function of a carried state split (`split`): `prepare(*operands)` the values its carry and its
    emitted outputs read that read no state, `carry(prepared, state)` its carried outputs and
    `emit(prepared, state)` its others, each a tuple in the function's order."""

    prepare: Callable
    carry: Callable
    emit: Callable


def split(function: Callable, state, operands: Sequence, carried: Sequence[int], *,
          saturation: program.Saturation, ranges: Mapping[int, float] | None = None,
          state_range: float | None = None) -> Split:
    """`function(state, *operands)`, a tuple of outputs, at `state`'s and `operands`' shapes and dtypes, split
    by extraction (the module's account): its outputs at `carried` the carry's, its others emitted, the
    operands' magnitudes bounded by `ranges` (by place among `operands`) and the state's by `state_range`,
    which the reorderings' new intermediates are held under (§6.3). Refused (`Refused`) where an equation
    outside the language reads the state."""
    with jax.disable_jit():
        closed = jax.make_jaxpr(forms.inlined(function))(state, *operands)
    declared = {1 + index: bound for index, bound in (ranges or {}).items()}
    region = terms.region(closed, declared if state_range is None else {**declared, 0: state_range})
    first = len(closed.jaxpr.constvars)
    graph = EGraph()
    program.stated(graph, region)
    _products(graph, region)
    reading = {first}
    for opaque in region.opaque:
        if any(term is not None and _reads(term, reading) for term in opaque.inputs):
            raise Refused(f"{opaque.equation.primitive.name}: an equation outside the language reads the "
                          "carried state")
    graph.register(set_(reads(Term.leaf(i64(first)))).to(i64(1)))
    outputs = list(region.outputs)
    if any(output is None for output in outputs):
        raise Refused("a literal output of a carried function")
    roots = [term for term in outputs if term is not None]
    sequential = Term.bundle(Vec[Term](*(roots[index] for index in carried)))
    emitted = [index for index in range(len(roots)) if index not in carried]
    others = Term.bundle(Vec[Term](*(roots[index] for index in emitted)) if emitted else Vec[Term].empty())
    graph.register(sequential, others)
    facts = seq(rules.FACTS.saturate(), CARRYING.saturate())
    program.saturate(graph, region, (CARRIED,), saturation, facts)
    schedule = cost.fused(cost.target())
    # The extractor oracle where it is on (`Passes.checked`), as every derivation's (`derive._derived`).
    checked = ALGEBRA.value.checked
    carry_forms, _ = program.extracted(graph, sequential, _carry_priced(region, schedule), checked=checked)
    emit_forms, _ = program.extracted(graph, others, cost.model(region, schedule), checked=checked)
    prepared = list({terms.key(term): term for form in (*carry_forms, *emit_forms)
                     for term in _free(form, reading)}.values())
    consts = list(closed.consts)

    def prepare(*values):
        leaves = [*consts, None, *values]
        return tuple(forms.evaluated(region, term, leaves, {}) for term in prepared)

    def reading_forms(chosen: list[Term]) -> Callable:
        def run(given: tuple, h) -> tuple:
            known = {terms.key(term): value for term, value in zip(prepared, given, strict=True)}
            leaves = [*([None] * first), h]
            return tuple(forms.evaluated(region, form, leaves, known) for form in chosen)

        return run

    return Split(prepare, reading_forms(carry_forms), reading_forms(emit_forms))


def _products(graph: EGraph, region: terms.Region) -> None:
    """For each `x^T @ y` of the region's (both operands' first axes contracted, in their dtype,
    `rules.plain`), the plain product and the transpose its rewrite takes (`plain_of`, `swap_of`), their
    parameters added to the region's table where it holds none; the plain products' facts are the program's
    (`program.stated`)."""
    plain = (((1,), (0,)), ((), ()))
    transposed = (((0,), (0,)), ((), ()))
    for term, _ in region.shapes:
        if (args := get_callable_args(term, Term.dot)) is None:
            continue
        index = terms.integer(args[2])
        primitive, params = region.params[index]
        if jax.tree.map(int, params["dimension_numbers"]) != transposed:
            continue
        if not graph.check_bool(rules.plain(i64(index))):
            continue
        made = _parameters(region, primitive, {**params, "dimension_numbers": plain})
        # The plain product multiplies the transposed one's operands (`terms.Region.products`).
        region.products.setdefault(made, region.products.get(index, ()))
        swapped = _parameters(region, jax.lax.transpose_p, {"permutation": (1, 0)})
        graph.register(rules.plain(i64(made)), rules.matrix_product(i64(made)),
                       set_(plain_of(i64(index))).to(i64(made)), set_(swap_of(i64(index))).to(i64(swapped)))


def _parameters(region: terms.Region, primitive, params) -> int:
    """The index of `primitive` at `params` in the region's table, added where it holds none."""
    for index, (held, given) in enumerate(region.params):
        if held is primitive and jax.tree.map(str, dict(given)) == jax.tree.map(str, dict(params)):
            return index
    region.params.append((primitive, dict(params)))
    return len(region.params) - 1


def _reads(term: Term, reading: set[int]) -> bool:
    """Whether `term` reads one of the `reading` leaves."""
    return any((args := get_callable_args(sub, Term.leaf)) is not None and terms.integer(args[0])
               in reading for sub in terms.subterms(term))


def _free(form: Term, reading: set[int]) -> list[Term]:
    """`form`'s largest terms that read none of the `reading` leaves, but its literals, in its order: each
    distinct term walked once, whether it reads them its operands' (a token loop's terms share their
    subterms, and walking every path of them anew held an A10's chunk form past fifteen minutes)."""
    operands: dict[Hashable, list[Term]] = {}
    reads: dict[Hashable, bool] = {}
    pending: list[tuple[Term, bool]] = [(form, False)]
    while pending:
        term, ready = pending.pop()
        named = terms.key(term)
        if ready:
            leaf = get_callable_args(term, Term.leaf)
            reads[named] = (leaf is not None and terms.integer(leaf[0]) in reading) or any(
                reads[terms.key(operand)] for operand in operands[named])
            continue
        if named in operands:
            continue
        operands[named] = [arg for constructor in terms.LANGUAGE
                           if (args := get_callable_args(term, constructor)) is not None
                           for arg in args if isinstance(arg, Term)]
        pending.append((term, True))
        pending.extend((operand, False) for operand in operands[named])
    found: dict[Hashable, Term] = {}
    walked, below = set(), [form]
    while below:
        term = below.pop()
        if (named := terms.key(term)) in walked:
            continue
        walked.add(named)
        if reads[named]:
            below.extend(reversed(operands[named]))
        elif get_callable_args(term, Term.literal) is None:
            found.setdefault(named, term)
    return list(found.values())
