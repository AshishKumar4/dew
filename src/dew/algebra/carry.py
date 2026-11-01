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
from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from egglog import (
    DagCostModel,
    EGraph,
    String,
    Unit,
    Vec,
    eq,
    function,
    get_callable_args,
    get_literal_value,
    i64,
    rewrite,
    rule,
    ruleset,
    seq,
    set_,
    var,
    vars_,
)

from dew.algebra import cost, forms, program, rules, terms
from dew.algebra.forms import Refused
from dew.algebra.terms import Term


@function(merge=lambda old, new: old)
def reads(term: Term) -> i64: ...
"""1 for a value that reads the carried state."""


@function
def matrix_product(params: i64) -> Unit: ...
"""A product's parameters (`Term.dot`) contracting its left operand's last axis with its right's first, no
batch, in its operands' dtype (no wider accumulation it rounds back from): `x @ y` of matrices
(`matrix`)."""


@function
def matrix(term: Term) -> Unit: ...
"""A value of rank two: each operand a rule moves a matrix product past must be one, as a product's
parameters hold whatever its operands' ranks."""


@function(merge=lambda old, new: old)
def plain_of(params: i64) -> i64: ...
"""A product's parameters of `x^T @ y` (both operands' first axes contracted, no batch): the parameters of the
plain product (`matrix_product`) its transpose's operand takes, of the same precision and dtype."""


@function(merge=lambda old, new: old)
def swap_of(params: i64) -> i64: ...
"""For `x^T @ y`'s parameters, the parameters of the transpose that lays its left operand as `x^T`."""


def _carrying():
    """Which values read the state, and the shape of each transpose and product the rules make."""
    x, y, z, t = vars_("x y z t", Term)
    name = var("name", String)
    p = var("p", i64)
    one = i64(1)
    left, right = vars_("left right", Vec[i64])
    marked = []
    for made, children in ((lambda: x.binary(name, y, p), (x, y)), (lambda: x.compare(name, y, p), (x, y)),
                           (lambda: x.unary(name, p), (x,)), (lambda: x.convert(p), (x,)),
                           (lambda: x.shaped(name, p), (x,)), (lambda: x.dot(y, p), (x, y)),
                           (lambda: x.select(y, z), (x, y, z))):
        marked += [rule(eq(t).to(made()), eq(reads(child)).to(one)).then(set_(reads(t)).to(one))
                   for child in children]
    return ruleset(
        *marked,
        rule(eq(left).to(rules.shape(t)), eq(left.length()).to(i64(2))).then(matrix(t)),
        rule(eq(t).to(x.shaped(String("transpose"), p)), eq(left).to(rules.shape(x)),
             eq(left.length()).to(i64(2))).then(
            set_(rules.shape(t)).to(Vec[i64](left[i64(1)], left[i64(0)]))),
        rule(eq(t).to(x.dot(y, p)), matrix_product(p), matrix(x), matrix(y), eq(left).to(rules.shape(x)),
             eq(right).to(rules.shape(y))).then(
            set_(rules.shape(t)).to(Vec[i64](left[i64(0)], right[i64(1)]))),
        name="carrying")


def _carried():
    """The products' linear algebra over values that read the state (the module's account), each over
    matrices (`matrix`), products and sums in their operands' dtype (`matrix_product`, `rules.plain`),
    summands of one shape, and operands its new intermediates read tame (`rules.tame`): each reorders
    rounding."""
    x, y, z = vars_("x y z", Term)
    p, m, s, q = vars_("p m s q", i64)
    one = i64(1)
    shape = rules.shape
    sets = [
        # x^T @ y as transpose(x) @ y, so it associates and factors with plain products.
        rewrite(x.dot(y, p)).to(x.shaped(String("transpose"), s).dot(y, m), eq(m).to(plain_of(p)),
                                eq(s).to(swap_of(p)), matrix(x), matrix(y), eq(reads(y)).to(one)),
        # x @ (y @ z) as (x @ y) @ z, the state's product last.
        rewrite(x.dot(y.dot(z, m), m)).to(x.dot(y, m).dot(z, m), matrix_product(m), matrix(x), matrix(y),
                                          matrix(z), rules.tame(x), rules.tame(y), eq(reads(z)).to(one)),
    ]
    for combined in ("add", "sub"):
        summed = (matrix_product(m), rules.plain(q), matrix(x), matrix(y), matrix(z), rules.tame(x),
                  rules.tame(y))
        sets += [
            rewrite(x.dot(y.binary(combined, z, q), m)).to(
                x.dot(y, m).binary(combined, x.dot(z, m), q), *summed, rules.tame(z),
                eq(shape(y)).to(shape(z)), eq(reads(y.binary(combined, z, q))).to(one)),
            rewrite(x.dot(z, m).binary(combined, y.dot(z, m), q)).to(
                x.binary(combined, y, q).dot(z, m), *summed, eq(shape(x)).to(shape(y)),
                eq(reads(z)).to(one)),
        ]
    return ruleset(*sets, name="carried")


CARRYING = _carrying()
CARRIED = rules.Rules("carried", "reorders", _carried(), receipt="docs/design/algebra.md §6.9; "
                      "tests/test_algebra_scan.py")
"""The split's rules, never among the engine's sets (`rules.admitted`)."""


@dataclasses.dataclass(frozen=True, order=True)
class Priced:
    """A carried form's price, compared in order: the seconds of its work that reads the state (once a chunk,
    in sequence), then the seconds of the rest (every chunk's at once), each `cost.model`'s."""

    sequential: float
    parallel: float

    def __add__(self, other: Priced) -> Priced:
        return Priced(self.sequential + other.sequential, self.parallel + other.parallel)


def _carry_priced(device: cost.Device) -> DagCostModel:
    """`Priced` per node: its seconds (`cost.model`) sequential where an operand reads the state (the
    e-classes the extractor hands the callback, which alone it may look up), parallel otherwise."""
    timed = cost.model(device)

    def marginal(graph: EGraph, node) -> Priced:
        seconds = timed.marginal_cost(graph, node).value
        operands = next((args for constructor in terms.LANGUAGE
                         if (args := get_callable_args(node, constructor)) is not None), ())
        if any(isinstance(operand, Term) and graph.lookup_function_value(reads(operand)) is not None
               for operand in operands):
            return Priced(seconds, 0.0)
        return Priced(0.0, seconds)

    return DagCostModel(marginal_cost=marginal, identity=Priced(0.0, 0.0))


class Split(NamedTuple):
    """A function of a carried state split (`split`): `prepare(*operands)` the values its carry and its
    emitted outputs read that read no state, `carry(prepared, state)` its carried outputs and
    `emit(prepared, state)` its others, each a tuple in the function's order."""

    prepare: Callable
    carry: Callable
    emit: Callable


def split(function: Callable, state, operands: Sequence, carried: Sequence[int], *,
          saturation: program.Saturation) -> Split:
    """`function(state, *operands)`, a tuple of outputs, at `state`'s and `operands`' shapes and dtypes, split
    by extraction (the module's account): its outputs at `carried` the carry's, its others emitted. Refused
    (`Refused`) where an equation outside the language reads the state."""
    with jax.disable_jit():
        closed = jax.make_jaxpr(forms.inlined(function))(state, *operands)
    region = terms.region(closed)
    first = len(closed.jaxpr.constvars)
    graph = EGraph()
    program.stated(graph, region)
    _products(graph, region, closed)
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
    device = cost.Device.of()
    carry_forms, _ = program.extracted(graph, sequential, _carry_priced(device))
    emit_forms, _ = program.extracted(graph, others, cost.model(device))
    prepared = list({str(term): term for form in (*carry_forms, *emit_forms)
                     for term in _free(form, reading)}.values())
    consts = list(closed.consts)

    def prepare(*values):
        leaves = [*consts, None, *values]
        return tuple(forms.evaluated(region, term, leaves, {}) for term in prepared)

    def reading_forms(chosen: list[Term]) -> Callable:
        def run(given: tuple, h) -> tuple:
            known = {str(term): value for term, value in zip(prepared, given, strict=True)}
            leaves = [*([None] * first), h]
            return tuple(forms.evaluated(region, form, leaves, known) for form in chosen)

        return run

    return Split(prepare, reading_forms(carry_forms), reading_forms(emit_forms))


def _products(graph: EGraph, region: terms.Region, closed) -> None:
    """Which of the region's products are plain matrix products (`matrix_product`: in their operands' dtype,
    each equation of `closed` holding its parameters so), and for each `x^T @ y` the plain product and the
    transpose its rewrite takes (`plain_of`, `swap_of`), their parameters added to the region's table where it
    holds none. Each rule asks its operands' ranks of the terms themselves (`matrix`)."""
    plain = (((1,), (0,)), ((), ()))
    transposed = (((0,), (0,)), ((), ()))
    widening: set[int] = set()
    for equation in closed.jaxpr.eqns:
        if equation.primitive is jax.lax.dot_general_p:
            wanted = equation.params.get("preferred_element_type")
            dtypes = {jnp.dtype(atom.aval.dtype) for atom in equation.invars}
            if wanted is not None and dtypes != {jnp.dtype(wanted)}:
                widening.add(_parameters(region, equation.primitive, equation.params))
    for index, (primitive, params) in list(enumerate(region.params)):
        if primitive is not jax.lax.dot_general_p or index in widening:
            continue
        numbers = jax.tree.map(int, params["dimension_numbers"])
        if numbers == plain:
            graph.register(matrix_product(i64(index)))
        elif numbers == transposed:
            made = _parameters(region, primitive, {**params, "dimension_numbers": plain})
            swapped = _parameters(region, jax.lax.transpose_p, {"permutation": (1, 0)})
            graph.register(matrix_product(i64(made)), set_(plain_of(i64(index))).to(i64(made)),
                           set_(swap_of(i64(index))).to(i64(swapped)))


def _parameters(region: terms.Region, primitive, params) -> int:
    """The index of `primitive` at `params` in the region's table, added where it holds none."""
    for index, (held, given) in enumerate(region.params):
        if held is primitive and jax.tree.map(str, dict(given)) == jax.tree.map(str, dict(params)):
            return index
    region.params.append((primitive, dict(params)))
    return len(region.params) - 1


def _reads(term: Term, reading: set[int]) -> bool:
    """Whether `term` reads one of the `reading` leaves."""
    return any((args := get_callable_args(sub, Term.leaf)) is not None and int(get_literal_value(args[0]))
               in reading for sub in terms.subterms(term))


def _free(form: Term, reading: set[int]) -> list[Term]:
    """`form`'s largest terms that read none of the `reading` leaves, but its literals."""
    if not _reads(form, reading):
        return [] if get_callable_args(form, Term.literal) is not None else [form]
    found = []
    for constructor in terms.LANGUAGE:
        if (args := get_callable_args(form, constructor)) is not None:
            for arg in args:
                if isinstance(arg, Term):
                    found += _free(arg, reading)
    return found
