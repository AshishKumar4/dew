"""The program pass (docs/design/algebra.md §6.7): a function's jaxpr held as terms (`terms.region`), the
admitted rule sets run to their fixpoint or the region's limits, and its roots' cheapest forms under the cost
model extracted together and evaluated in place of the program's own. The bodies of its nested programs,
scans, loops, conditionals, checkpoints and custom derivatives' primal calls are rewritten in place, each a
region of its own, and kept bodies, never unrolled. Each rewrite is recorded in the one kernel report
(`dew.nn.kernels.templates.report.rewrites`)."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import time
from collections.abc import Callable, Iterator, Mapping

import jax
import numpy as np
from egglog import EGraph, String, Vec, get_callable_args, i64, set_
from jax._src import config as jax_config
from jax.core import ShapedArray, eval_jaxpr
from jax.extend import core as jcore

from dew.algebra import cost, rules, terms
from dew.cache import ALGEBRA, Passes
from dew.nn.kernels.templates import report as kernel_report


@dataclasses.dataclass(frozen=True)
class Saturation:
    """The limits a region's saturation stops at, from its size (`limits`): nodes at `growth` per equation,
    iterations at `depth` per level of its longest chain of the language's equations."""

    growth: int
    depth: int

    def limits(self, region: terms.Region) -> tuple[int, int]:
        """The nodes and iterations `region` may take."""
        return self.growth * max(1, region.equations), self.depth * max(1, region.depth)


@dataclasses.dataclass(frozen=True)
class AlgebraReport:
    """What one region's rewrite did: the equations it read, the sets admitted and the matches each fired,
    its limits, its iterations and nodes, whether a limit stopped it, the region's price before and after in
    seconds (`cost.model`), and the rewrite's own seconds."""

    equations: int
    admitted: tuple[str, ...]
    fired: Mapping[str, int]
    limits: tuple[int, int]
    iterations: int
    nodes: int
    limited: bool
    before: float
    after: float
    seconds: float


@dataclasses.dataclass(frozen=True)
class Rewrite:
    """A region's terms (`terms.region`), each root's cheapest form by its text (`picked`, empty where none is
    cheaper than the region as traced), and the report of the rewrite that chose them."""

    region: terms.Region
    picked: Mapping[str, terms.Term]
    report: AlgebraReport


@contextlib.contextmanager
def passes(on: bool = True, target: str | None = None) -> Iterator[None]:
    """Within the block, the programs Dew compiles are rewritten (`on`) or not, priced for the GPU `target`
    names (None: the process's own): a context of JAX's trace, so a program held across the switch traces
    afresh (`dew.cache.ALGEBRA`). The accuracy gate holds it off for a reference arm and the float64 truth."""
    with ALGEBRA(Passes(on, target)):
        yield


def rewritten(fn: Callable, *, accurate: bool = False, saturation: Saturation,
              static_argnums: int | tuple[int, ...] = (), static_argnames: str | tuple[str, ...] = ()
              ) -> Callable:
    """`fn` with its program rewritten where the switch is on (`passes`): its static arguments bound as the
    jit around it binds them (`static_argnums`, `static_argnames`), traced over the others, its terms
    saturated under the rule sets its numerics admit (`rules.admitted`) and its values evaluated in their
    cheapest forms. Its arguments, outputs and their structure are `fn`'s."""
    numbers = (static_argnums,) if isinstance(static_argnums, int) else tuple(static_argnums)
    names = (static_argnames,) if isinstance(static_argnames, str) else tuple(static_argnames)

    @functools.wraps(fn)
    def run(*args, **kwargs):
        # Under `jax.disable_jit` a program runs as Python, its values concrete, so nothing is traced to
        # rewrite.
        if not ALGEBRA.value.on or jax_config.disable_jit.value:
            return fn(*args, **kwargs)
        statics = {number % len(args) for number in numbers if -len(args) <= number < len(args)}
        fixed = {name: value for name, value in kwargs.items() if name in names}
        dynamic = [arg for number, arg in enumerate(args) if number not in statics]

        def bound(dynamic_args, dynamic_kwargs):
            given = iter(dynamic_args)
            whole = [args[number] if number in statics else next(given) for number in range(len(args))]
            return fn(*whole, **fixed, **dynamic_kwargs)

        named = {name: value for name, value in kwargs.items() if name not in names}
        closed, shaped = jax.make_jaxpr(bound, return_shape=True)(dynamic, named)
        done = rewrite(closed, jax.tree.leaves((dynamic, named)), accurate=accurate, saturation=saturation)
        return jax.tree.unflatten(jax.tree.structure(shaped), done)
    return run


def rewrite(closed: jcore.Jaxpr, arguments: list, *, accurate: bool, saturation: Saturation) -> list:
    """The outputs of `closed` (a jaxpr and its constants) over `arguments` (flattened), its bodies rewritten
    in place first (`_within`), then its own math (`_windowed`)."""
    rewritten = _windowed(_within(closed, accurate, saturation), accurate, saturation)
    return eval_jaxpr(rewritten, rewritten.consts, *arguments)


WINDOW = 3
"""How deep a window reaches below the head a rule reads (`rules.heads`): a sum of two reductions of
products under a broadcast, the deepest left side the rule sets hold."""

CLUSTER = 64
"""The most equations windows that share one join into, so no term is deeper than a few hundred frames of
egglog's recursion however a program chains its heads: windows past it are held apart, the values one reads
of another read from outside it."""


def _windowed(jaxpr: jcore.Jaxpr, accurate: bool, saturation: Saturation) -> jcore.Jaxpr:
    """`jaxpr` with the values its windows yield (`_windows`) computed in their cheapest forms, where those
    are cheaper than the windows as traced: all windows saturated in one e-graph and extracted together,
    each cheaper value's form traced and spliced in where its equation stood, the rest of the program kept
    as traced, so a body is never traced again."""
    windows = _windows(jaxpr)
    if windows is None:
        return jaxpr
    combined, yielded = windows
    done = _region(combined, accurate=accurate, saturation=saturation)
    kernel_report.record_rewrite(done.report)
    if not done.picked:
        return jaxpr
    leaves = [jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype, weak_type=var.aval.weak_type)
              for var in combined.invars if isinstance(var.aval, ShapedArray)]
    if len(leaves) != len(combined.invars):
        return jaxpr
    made_by = {var: equation for equation in jaxpr.eqns for var in equation.outvars}
    spliced: dict[int, list[jcore.JaxprEqn]] = {}
    for out, root in zip(yielded, done.region.outputs, strict=True):
        form = None if root is None else done.picked.get(str(root))
        if form is None or str(form) == str(root):
            continue
        traced = jax.make_jaxpr(lambda *values, form=form: terms.form_value(done.region, form, list(values)))(
            *leaves)
        made, = traced.jaxpr.outvars
        if traced.consts or not isinstance(made, jcore.Var) or made in traced.jaxpr.invars or (
                traced.out_avals[0].strip_weak_type() != out.aval.strip_weak_type()):
            continue
        renamed = {**dict(zip(traced.jaxpr.invars, combined.invars, strict=True)), made: out}

        def renaming(atom, renamed=renamed):
            return atom if isinstance(atom, jcore.Literal) else renamed.get(atom, atom)

        if id(made_by[out]) in spliced:
            continue
        spliced[id(made_by[out])] = [equation.replace(invars=[renaming(atom) for atom in equation.invars],
                                                      outvars=[renaming(var) for var in equation.outvars])
                                     for equation in traced.jaxpr.eqns]
    if not spliced:
        return jaxpr
    return _live(jaxpr.replace(eqns=[kept for equation in jaxpr.eqns
                                     for kept in spliced.get(id(equation), [equation])]))


def _live(jaxpr: jcore.Jaxpr) -> jcore.Jaxpr:
    """`jaxpr` without the equations nothing reads: those a spliced form left behind, an equation with an
    effect kept whatever reads it."""
    wanted = {atom for atom in jaxpr.outvars if not isinstance(atom, jcore.Literal)}
    kept: list[jcore.JaxprEqn] = []
    for equation in reversed(jaxpr.eqns):
        if equation.effects or any(var in wanted for var in equation.outvars):
            kept.append(equation)
            wanted.update(atom for atom in equation.invars if not isinstance(atom, jcore.Literal))
    return jaxpr.replace(eqns=kept[::-1])


def _windows(jaxpr: jcore.Jaxpr) -> tuple[jcore.Jaxpr, list[jcore.Var]] | None:
    """`jaxpr`'s windows as one jaxpr: each head a rule may lower the price at (`rules.heads`) with the
    equations of the language that make its operands, `WINDOW` deep, windows that share an equation joined up
    to `CLUSTER` equations; its inputs each value a window reads from outside it, its outputs each value a
    window makes that the program reads outside it (the values the rewrite may compute anew), in program
    order; and those values as `jaxpr` names them. None where `jaxpr` holds no head."""
    made = {var: equation for equation in jaxpr.eqns for var in equation.outvars}

    def maker(atom) -> jcore.JaxprEqn | None:
        equation = None if isinstance(atom, jcore.Literal) else made.get(atom)
        return equation if equation is not None and terms.spoken(equation) else None

    heads = [equation for equation in jaxpr.eqns if terms.spoken(equation) and rules.heads(
        equation.primitive.name,
        [None if (operand := maker(atom)) is None else operand.primitive.name for atom in equation.invars])]
    if not heads:
        return None
    order = {id(equation): index for index, equation in enumerate(jaxpr.eqns)}
    clusters: list[dict[int, jcore.JaxprEqn]] = []
    for head in heads:
        window, frontier = {id(head): head}, [(head, 0)]
        while frontier:
            equation, depth = frontier.pop()
            for atom in equation.invars:
                operand = maker(atom)
                if operand is not None and depth + 1 < WINDOW and id(operand) not in window:
                    window[id(operand)] = operand
                    frontier.append((operand, depth + 1))
        sharing = [cluster for cluster in clusters if cluster.keys() & window.keys()]
        if sharing and len(set(window).union(*(cluster.keys() for cluster in sharing))) <= CLUSTER:
            clusters = [cluster for cluster in clusters if all(cluster is not shared for shared in sharing)]
            window = {key: member for cluster in [*sharing, window] for key, member in cluster.items()}
        clusters.append(window)
    read: dict[jcore.Var, set[int]] = {}
    for equation in jaxpr.eqns:
        for atom in equation.invars:
            if not isinstance(atom, jcore.Literal):
                read.setdefault(atom, set()).add(id(equation))
    returned = {atom for atom in jaxpr.outvars if not isinstance(atom, jcore.Literal)}
    outside: dict[jcore.Var, None] = {}
    equations: list[jcore.JaxprEqn] = []
    outputs: list[jcore.Var] = []
    yielded: list[jcore.Var] = []
    for cluster in clusters:
        fresh: dict[jcore.Var, jcore.Var] = {}
        members = sorted(cluster.values(), key=lambda member: order[id(member)])
        for equation in members:
            for atom in equation.invars:
                if not isinstance(atom, jcore.Literal) and atom not in fresh:
                    outside[atom] = None
            invars = [atom if isinstance(atom, jcore.Literal) else fresh.get(atom, atom)
                      for atom in equation.invars]
            fresh.update({var: jcore.Var(var.aval) for var in equation.outvars})
            outvars = [fresh[var] for var in equation.outvars]
            equations.append(equation.replace(invars=invars, outvars=outvars))
        for equation in members:
            out = equation.outvars[0]
            if out in returned or read.get(out, set()) - cluster.keys():
                outputs.append(fresh[out])
                yielded.append(out)
    return jcore.Jaxpr([], list(outside), outputs, equations), yielded


def _region(closed: jcore.Jaxpr, *, accurate: bool, saturation: Saturation) -> Rewrite:
    """`closed`'s roots' forms, its bodies already rewritten, chosen together from its terms saturated under
    the admitted rule sets where they are cheaper than the region as traced, with the report of the
    rewrite."""
    began = time.perf_counter()
    region = terms.region(closed)
    graph, bundle, roots = graphed(region)
    priced = cost.model(cost.Device.of())
    _, before = graph.extract(bundle, include_cost=True, cost_model=priced, extractor="greedy-dag")
    saturated = saturate(graph, region, rules.admitted(accurate), saturation)
    forms, after = extracted(graph, bundle, priced)
    cheaper = after < before.value
    picked = {str(root): form for root, form in zip(roots, forms, strict=True)} if cheaper else {}
    report = AlgebraReport(region.equations, tuple(saturated.fired), saturated.fired, saturated.limits,
                           saturated.iterations, _nodes(graph), saturated.limited, before.value, after,
                           time.perf_counter() - began)
    return Rewrite(region, picked, report)


def graphed(region: terms.Region) -> tuple[EGraph, terms.Term, list[terms.Term]]:
    """An e-graph of `region`'s terms with the facts its rules read (each term's shape, which arithmetic
    computes in its operands' dtype, which broadcasts lay along one axis, which literals are powers of two or
    one, the primitives a stable rule writes), its roots bundled, the facts saturated."""
    graph = EGraph()
    for term, extent in region.shapes:
        graph.register(set_(rules.shape(term)).to(Vec[i64](*(i64(size) for size in extent))))
    arithmetic = (jax.lax.add_p, jax.lax.mul_p, jax.lax.sub_p, jax.lax.div_p)
    for index, (primitive, params) in enumerate(region.params):
        if primitive in arithmetic and params.get("out_dtype") is None:
            graph.register(rules.plain(i64(index)))
        if primitive is jax.lax.dot_general_p and params.get("preferred_element_type") is None:
            graph.register(rules.plain(i64(index)))
        if primitive is jax.lax.broadcast_in_dim_p:
            rank, laid = len(params["shape"]), tuple(params["broadcast_dimensions"])
            missing = [axis for axis in range(rank) if axis not in laid]
            if len(missing) == 1 and laid == tuple(axis for axis in range(rank) if axis != missing[0]):
                graph.register(rules.along(i64(index), i64(missing[0])))
    for name, index in region.stock.items():
        graph.register(set_(rules.stock(String(name))).to(i64(index)))
    for index, value in enumerate(region.literals):
        literal = terms.Term.literal(i64(index))
        if _power_of_two(value):
            graph.register(rules.exact_scale(literal))
        if np.ndim(value) == 0 and float(value) == 1.0:
            graph.register(rules.unit(literal))
    roots = region.roots()
    bundle = terms.Term.bundle(Vec[terms.Term](*roots))
    graph.register(bundle)
    graph.run(rules.FACTS.saturate())
    return graph, bundle, roots


@dataclasses.dataclass(frozen=True)
class Saturated:
    """What a saturation did: the matches each rule set fired, its limits (nodes, iterations), the iterations
    it ran, and whether a limit stopped it."""

    fired: Mapping[str, int]
    limits: tuple[int, int]
    iterations: int
    limited: bool


def saturate(graph: EGraph, region: terms.Region, admitted: tuple[rules.Rules, ...],
             saturation: Saturation) -> Saturated:
    """`graph` run under the `admitted` rule sets to their fixpoint or `region`'s limits (`Saturation`), its
    facts saturated after each set."""
    fired = dict.fromkeys((admitted_rules.name for admitted_rules in admitted), 0)
    nodes, steps = saturation.limits(region)
    iterations, limited = 0, False
    while admitted:
        if iterations == steps or _nodes(graph) > nodes:
            limited = True
            break
        updated = False
        for admitted_rules in admitted:
            graph.run(rules.FACTS.saturate())
            ran = graph.run(admitted_rules.ruleset)
            fired[admitted_rules.name] += sum(ran.num_matches_per_rule.values())
            updated = updated or ran.updated
        iterations += 1
        if not updated:
            break
    graph.run(rules.FACTS.saturate())
    return Saturated(fired, (nodes, steps), iterations, limited)


def extracted(graph: EGraph, bundle: terms.Term, model) -> tuple[list[terms.Term], float]:
    """The roots' forms `model` prices cheapest together, each shared value once, and their price."""
    chosen, price = graph.extract(bundle, include_cost=True, cost_model=model, extractor="greedy-dag")
    forms = get_callable_args(chosen, terms.Term.bundle)
    if forms is None:
        raise ValueError(f"extraction chose no bundle of the program's roots: {chosen}")
    return list(forms[0].value), price.value


BODIES = frozenset({"jit", "pjit", "closed_call", "scan", "while", "cond", "remat2", "checkpoint",
                    "custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr"})
"""The primitives whose bodies the pass rewrites in place, each a region of its own (§1.2): a nested program,
a scan, a loop, a conditional, a checkpoint, and a custom derivative's primal call, its rules kept."""


def _within(jaxpr: jcore.Jaxpr, accurate: bool, saturation: Saturation) -> jcore.Jaxpr:
    """`jaxpr`, its constants kept, with each body (`BODIES`) of its equations rewritten, the equation itself
    kept."""
    equations = [equation.replace(params={name: _body(value, accurate, saturation)
                                          for name, value in equation.params.items()})
                 if equation.primitive.name in BODIES else equation for equation in jaxpr.eqns]
    return jaxpr.replace(eqns=equations)


def _body(value, accurate: bool, saturation: Saturation):
    """A body parameter rewritten: a jaxpr's program, each of a conditional's branches; any other parameter,
    and a body whose constants it does not hold, as it is."""
    if isinstance(value, jcore.Jaxpr):
        return _closed(value, accurate, saturation)
    if isinstance(value, tuple) and value and all(isinstance(branch, jcore.Jaxpr) for branch in value):
        return tuple(_closed(branch, accurate, saturation) for branch in value)
    return value


def _closed(closed: jcore.Jaxpr, accurate: bool, saturation: Saturation) -> jcore.Jaxpr:
    """A body's program rewritten, over the same inputs, weak types included, and the same outputs; a body
    whose constants another holds (an open jaxpr), or whose rewrite would hold constants it did not, or that
    reads a value of no shape, as it is."""
    if len(closed.consts) != len(closed.constvars):
        return closed
    if not all(isinstance(aval, ShapedArray) for aval in closed.in_avals):
        return closed
    return _windowed(_within(closed, accurate, saturation), accurate, saturation)


def _power_of_two(value) -> bool:
    """Whether a literal is a scalar power of two, a multiply by which rounds nothing."""
    if np.ndim(value) != 0 or not np.issubdtype(np.asarray(value).dtype, np.floating):
        return False
    mantissa, _ = np.frexp(float(value))
    return mantissa in (0.5, -0.5)


def _nodes(graph: EGraph) -> int:
    """The rows of every table of `graph`, its nodes and its facts' (egglog states no node count of its
    own)."""
    return sum(size for _, size in graph.all_function_sizes())
