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
import inspect
import math
import time
from collections.abc import Callable, Hashable, Iterator, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from egglog import EGraph, Schedule, String, Vec, get_callable_args, get_literal_value, i64, set_
from jax._src import config as jax_config
from jax.core import ShapedArray, eval_jaxpr
from jax.extend import core as jcore

from dew.algebra import cost, rules, terms
from dew.cache import ALGEBRA, Passes
from dew.nn.kernels import generation
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
    """A region's terms (`terms.region`), each root's cheapest form by its key (`terms.key`; `picked`, empty
    where none is cheaper than the region as traced), the report of the rewrite that chose them, and the roots
    (by key) whose form XLA's own rewrites alone reach (`xla_own`, `Rules.xla`), which XLA's cost analysis
    decides."""

    region: terms.Region
    picked: Mapping[Hashable, terms.Term]
    report: AlgebraReport
    xla_own: frozenset[Hashable] = frozenset()


@contextlib.contextmanager
def passes(on: bool = True, target: str | None = None) -> Iterator[None]:
    """Within the block, the programs Dew compiles are rewritten (`on`) or not, priced for the GPU `target`
    names (None: the process's own): a context of JAX's trace, so a program held across the switch traces
    afresh (`dew.cache.ALGEBRA`). The accuracy gate holds it off for a reference arm and the float64 truth."""
    with ALGEBRA(Passes(on, target)):
        yield


def rewritten(fn: Callable, *, saturation: Saturation, static_argnums: int | tuple[int, ...] = (),
              static_argnames: str | tuple[str, ...] = ()) -> Callable:
    """`fn` with its program rewritten where the switch is on (`passes`): its static arguments bound as the
    jit around it binds them (`static_argnums`, `static_argnames`), traced over the others, its terms
    saturated under the rule sets the run's numerics admit (`rules.admitted`: the `exact` sets alone under
    the accurate forms, `generation.numerics`, which the trace context keys) and its values evaluated in
    their cheapest forms. Its arguments, outputs and their structure are `fn`'s."""
    # The static positions and names as jit resolves them: each given, and its other spelling read from the
    # signature's positional parameters, so a static argument passed either way is bound.
    numbers = {static_argnums} if isinstance(static_argnums, int) else set(static_argnums)
    names = {static_argnames} if isinstance(static_argnames, str) else set(static_argnames)
    try:
        parameters = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        parameters = []
    positional = [parameter.name for parameter in parameters
                  if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]
    numbers |= {number for number, name in enumerate(positional) if name in names}
    names |= {positional[number] for number in numbers if -len(positional) <= number < len(positional)}

    @functools.wraps(fn)
    def run(*args, **kwargs):
        # Under `jax.disable_jit` a program runs as Python, its values concrete, so nothing is traced to
        # rewrite; where the run's numerics admit XLA's own rewrites alone (`Rules.xla`: the accurate forms'
        # commutativity, which XLA's HloCSE ignores already), the pass would leave the program as it is.
        accurate = generation.numerics().forms == "accurate"
        if (not ALGEBRA.value.on or jax_config.disable_jit.value
                or all(sets.xla for sets in rules.admitted(accurate))):
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
    are cheaper than the windows as traced: each cluster saturated in an e-graph of its own (clusters share no
    equation, so none shares a form with another; one e-graph of every window's terms grew its extraction
    with the program, past 500 seconds a gemma3n training step's), each cheaper value's form traced and
    spliced in where its equation stood, the rest of the program kept as traced, so a body is never traced
    again."""
    made_by = {var: equation for equation in jaxpr.eqns for var in equation.outvars}
    spliced: dict[int, list[jcore.JaxprEqn]] = {}
    for window, yielded in _windows(jaxpr):
        done = _region(window, accurate=accurate, saturation=saturation)
        kernel_report.record_rewrite(done.report)
        if done.picked:
            spliced.update(_splices(window, yielded, done, made_by, spliced))
    if not spliced:
        return jaxpr
    return _live(jaxpr.replace(eqns=[kept for equation in jaxpr.eqns
                                     for kept in spliced.get(id(equation), [equation])]))


def _splices(window: jcore.Jaxpr, yielded: list[jcore.Var], done: Rewrite,
             made_by: Mapping[jcore.Var, jcore.JaxprEqn], taken: Mapping[int, list]) -> dict[int, list]:
    """The equations each value `window` yields (`yielded`, as the program names them) is made by in its
    cheapest form (`done`), by the id of the program's equation they stand in for (`made_by`): a value another
    window's form made already (`taken`) kept as that one made it. A form XLA's own rewrites alone reach
    (`Rewrite.xla_own`) is taken only where XLA's cost analysis of the compiled window is lower with it
    (`xla_priced`), whatever else the window rewrites."""
    leaves = [jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype, weak_type=var.aval.weak_type)
              for var in window.invars if isinstance(var.aval, ShapedArray)]
    if len(leaves) != len(window.invars):
        return {}
    in_window: dict = {var: equation for equation in window.eqns for var in equation.outvars}
    spliced: dict[int, list[jcore.JaxprEqn]] = {}
    # The same forms in the window's own names, which XLA's cost analysis compiles alone.
    own: dict[int, list[jcore.JaxprEqn]] = {}
    xla_own: set[int] = set()
    own_xla: set[int] = set()
    for out, mine, root in zip(yielded, window.outvars, done.region.outputs, strict=True):
        form = None if root is None else done.picked.get(terms.key(root))
        made_at = id(made_by[out])
        if form is None or terms.key(form) == terms.key(root) or made_at in taken or made_at in spliced:
            continue
        traced = jax.make_jaxpr(lambda *values, form=form: terms.form_value(done.region, form, list(values)))(
            *leaves)
        made, = traced.jaxpr.outvars
        if traced.consts or not isinstance(made, jcore.Var) or made in traced.jaxpr.invars or (
                traced.out_avals[0].strip_weak_type() != out.aval.strip_weak_type()):
            continue
        inputs = dict(zip(traced.jaxpr.invars, window.invars, strict=True))
        for named, made_as, key in ((spliced, out, id(made_by[out])), (own, mine, id(in_window[mine]))):
            renamed = {**inputs, made: made_as}
            def renaming(atom, renamed=renamed):
                return atom if isinstance(atom, jcore.Literal) else renamed.get(atom, atom)

            named[key] = [equation.replace(invars=[renaming(atom) for atom in equation.invars],
                                           outvars=[renaming(var) for var in equation.outvars])
                          for equation in traced.jaxpr.eqns]
        if terms.key(root) in done.xla_own:
            xla_own.add(id(made_by[out]))
            own_xla.add(id(in_window[mine]))
    if not xla_own:
        return spliced

    def splicing(chosen) -> jcore.Jaxpr:
        return _live(window.replace(eqns=[made for equation in window.eqns
                                          for made in (own[id(equation)] if id(equation) in chosen
                                                       else [equation])]))

    if cost.cheaper(xla_priced(splicing(own)), xla_priced(splicing(own.keys() - own_xla))):
        return spliced
    return {key: value for key, value in spliced.items() if key not in xla_own}


def xla_priced(jaxpr: jcore.Jaxpr | jcore.ClosedJaxpr) -> cost.Seconds:
    """`jaxpr`'s price by XLA's own cost analysis of its compiled program (its FLOPs, transcendentals and
    bytes accessed) at the pass's device's rates (`cost.Fused`): the price of a rewrite XLA would match
    itself, which the e-graph's model cannot see XLA's fusion of."""
    closed = jaxpr if isinstance(jaxpr, jcore.ClosedJaxpr) else jcore.ClosedJaxpr(jaxpr, ())
    avals = list(closed.in_avals)
    analysis = jax.jit(jcore.jaxpr_as_fun(closed)).lower(*avals).compile().cost_analysis()
    analysis = (analysis[0] if isinstance(analysis, list) else analysis) or {}
    work = cost.Work(operations=analysis.get("flops", 0.0), exponentials=analysis.get("transcendentals", 0.0),
                     device_bytes=analysis.get("bytes accessed", 0.0))
    return cost.Seconds(work, cost.fused(cost.target()))


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


def _windows(jaxpr: jcore.Jaxpr) -> list[tuple[jcore.Jaxpr, list[jcore.Var]]]:
    """`jaxpr`'s windows, each cluster a jaxpr of its own: each head a rule may lower the price at
    (`rules.heads`) with the equations of the language that make its operands, `WINDOW` deep, windows that
    share an equation joined up to `CLUSTER` equations; its inputs each value it reads from outside it, its
    outputs each value it makes that the program reads outside it (the values the rewrite may compute anew),
    in program order; and those values as `jaxpr` names them. None where `jaxpr` holds no head."""
    made = {var: equation for equation in jaxpr.eqns for var in equation.outvars}

    def maker(atom) -> jcore.JaxprEqn | None:
        equation = None if isinstance(atom, jcore.Literal) else made.get(atom)
        return equation if equation is not None and terms.spoken(equation) else None

    heads = [equation for equation in jaxpr.eqns if terms.spoken(equation) and rules.heads(
        equation.primitive.name,
        [None if (operand := maker(atom)) is None else operand.primitive.name for atom in equation.invars])]
    if not heads:
        return []
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
    windows = []
    for cluster in clusters:
        outside: dict[jcore.Var, None] = {}
        equations: list[jcore.JaxprEqn] = []
        outputs: list[jcore.Var] = []
        yielded: list[jcore.Var] = []
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
        windows.append((jcore.Jaxpr([], list(outside), outputs, equations), yielded))
    return windows


def _region(closed: jcore.Jaxpr, *, accurate: bool, saturation: Saturation) -> Rewrite:
    """`closed`'s roots' forms, its bodies already rewritten, chosen together from its terms saturated under
    the admitted rule sets where they are cheaper than the region as traced (`cost.cheaper`), with the roots
    whose form XLA's own sets alone reach (`Rewrite.xla_own`: the region saturated under them first) and the
    report of the rewrite. A region no reordering may touch (`rules.reorderable`) builds no e-graph. The
    saturated graph's forms are extracted first; only where one differs from its root are the region as
    traced and XLA's sets' forms extracted too, each from the graph as it stood (egglog's `push`, `pop`): a
    region whose forms are its roots' (commutativity's alone) takes one extraction."""
    began = time.perf_counter()
    region = terms.region(closed)
    admitted = rules.admitted(accurate)
    if not rules.reorderable(region):
        fired = dict.fromkeys((sets.name for sets in admitted), 0)
        report = AlgebraReport(region.equations, tuple(fired), fired, saturation.limits(region),
                               iterations=0, nodes=0, limited=False, before=0.0, after=0.0,
                               seconds=time.perf_counter() - began)
        return Rewrite(region, {}, report)
    priced = cost.model(region, cost.fused(cost.target()))
    own = tuple(sets for sets in admitted if sets.xla)
    graph, bundle, roots = graphed(region)
    graph.push()
    first = saturate(graph, region, own, saturation) if own and len(own) < len(admitted) else None
    graph.push()
    saturated = saturate(graph, region, admitted, saturation)
    nodes = node_count(graph)
    forms, after = extracted(graph, bundle, priced)
    changed = [terms.key(root) for root, form in zip(roots, forms, strict=True)
               if terms.key(form) != terms.key(root)]
    graph.pop()
    theirs = None
    if changed and first is not None and any(count for name, count in first.fired.items()
                                             if rules.RULES[name].rounding != "exact"):
        # The forms XLA's own sets alone reach, from the graph they saturated (an exact set's alone,
        # commutativity's, leave every price as it was).
        theirs, _ = extracted(graph, bundle, priced)
    graph.pop()
    before = after
    if changed:
        _, before = graph.extract(bundle, include_cost=True, cost_model=priced, extractor="greedy-dag")
    picked = ({terms.key(root): form for root, form in zip(roots, forms, strict=True)}
              if changed and cost.cheaper(after, before) else {})
    if theirs is not None:
        xla_own = frozenset(terms.key(root) for root, form, xla_form in zip(roots, forms, theirs, strict=True)
                            if terms.key(form) == terms.key(xla_form) != terms.key(root))
    else:
        xla_own = frozenset(changed) if own and first is None else frozenset()
    if first is not None:
        saturated = saturated.after(first)
    report = AlgebraReport(region.equations, tuple(saturated.fired), saturated.fired, saturated.limits,
                           saturated.iterations, nodes, saturated.limited, before.value, after.value,
                           time.perf_counter() - began)
    return Rewrite(region, picked, report, xla_own)


def graphed(region: terms.Region) -> tuple[EGraph, terms.Term, list[terms.Term]]:
    """An e-graph of `region`'s terms with the facts its rules read (each term's shape, which arithmetic
    computes in its operands' dtype, which broadcasts lay along one axis, which literals are powers of two or
    one, the primitives a stable rule writes), its roots bundled, the facts saturated."""
    graph = EGraph()
    roots = region.roots()
    bundle = terms.Term.bundle(Vec[terms.Term](*roots))
    stated(graph, region, bundle)
    graph.run(rules.FACTS.saturate())
    return graph, bundle, roots


def stated(graph: EGraph, region: terms.Region, *registered) -> None:
    """The facts `region`'s parameters and literals state, registered in `graph` (`graphed`'s) beside
    `registered` in one call (egglog's every call its own program: 200 facts took 98 ms one at a time, 53 at
    once): each term's shape, which arithmetic computes in its operands' dtype, which broadcasts lay along one
    axis, which literals are powers of two or one, the primitives a stable rule writes, each term's number
    bound (`rules.log2_bound`), the program's reach (`rules.reach`), which a reordering's new intermediates
    are held under (§6.3), each extent's power of two (`rules.log2_of`) and each reduction's length's
    (`rules.sum_length`)."""
    facts: list = [*registered]
    facts += [set_(rules.shape(term)).to(Vec[i64](*(i64(size) for size in extent)))
              for term, extent in region.shapes]
    arithmetic = (jax.lax.add_p, jax.lax.mul_p, jax.lax.sub_p, jax.lax.div_p)
    for index, (primitive, params) in enumerate(region.params):
        if primitive in arithmetic and params.get("out_dtype") is None:
            facts.append(rules.plain(i64(index)))
        if primitive is jax.lax.broadcast_in_dim_p:
            rank, laid = len(params["shape"]), tuple(params["broadcast_dimensions"])
            missing = [axis for axis in range(rank) if axis not in laid]
            if len(missing) == 1 and laid == tuple(axis for axis in range(rank) if axis != missing[0]):
                facts.append(rules.along(i64(index), i64(missing[0])))
    facts += [set_(rules.stock(String(name))).to(i64(index)) for name, index in region.stock.items()]
    facts += _products(region)
    sizes = {size for _, extent in region.shapes for size in extent}
    facts += [set_(rules.log2_of(i64(size))).to(i64(max(size - 1, 0).bit_length())) for size in sorted(sizes)]
    for term, _ in region.shapes:
        bound = region.magnitudes.get(terms.key(term))
        if bound is not None:
            facts += rules.made(term, i64(bound))
        if (args := get_callable_args(term, terms.Term.reduce)) is not None:
            _, params = region.params[int(get_literal_value(args[3]))]
            extent = region.shape(args[0])
            length = math.prod(extent[axis] for axis in params["axes"])
            facts.append(set_(rules.sum_length(term)).to(i64(max(length - 1, 0).bit_length())))
    # A bound of numbers alone (the declared ranges', the constants', and what the equations prove of them)
    # holds where the program's values are: a new intermediate under the narrowest float's largest power of
    # two is finite wherever the program is. A bound over an unbounded operand would bound nothing: the
    # program's value may sit far below it while a new intermediate overflows (0.5 x + 0.5 y at 3e38
    # against 0.5 (x + y)), so an unbounded value has none (`terms.Magnitude`).
    floats = [jnp.finfo(dtype).max for dtype in region.dtypes.values() if jnp.issubdtype(dtype, jnp.floating)]
    if floats:
        facts.append(set_(rules.reach()).to(i64(math.floor(math.log2(min(floats))))))
    if facts:
        graph.register(*facts)


def _products(region: terms.Region) -> list:
    """The facts a region's products state: which compute in their operands' dtype (`rules.plain`: no
    accumulation wider than their operands they round back from), which of those are matrix products
    (`rules.matrix_product`), which values are matrices (`rules.matrix`), and the broadcasts of a vector to
    each matrix's extent along either axis (`rules.lifted`), their parameters added to the region's table."""
    plain_numbers = (((1,), (0,)), ((), ()))
    facts: list = []
    for term, extent in region.shapes:
        if len(extent) == 2:
            facts.append(rules.matrix(term))
        if (args := get_callable_args(term, terms.Term.dot)) is None:
            continue
        index = int(get_literal_value(args[2]))
        _, params = region.params[index]
        wanted = params.get("preferred_element_type")
        operands = {region.dtypes.get(terms.key(args[0])), region.dtypes.get(terms.key(args[1]))}
        if wanted is None or operands == {jnp.dtype(wanted)}:
            facts.append(rules.plain(i64(index)))
            if jax.tree.map(int, params["dimension_numbers"]) == plain_numbers:
                facts.append(rules.matrix_product(i64(index)))
    matrices = {extent for _, extent in region.shapes if len(extent) == 2}
    for extent in sorted(matrices):
        for axis in (0, 1):
            region.params.append((jax.lax.broadcast_in_dim_p, {
                "shape": extent, "broadcast_dimensions": (axis,), "sharding": None}))
            facts.append(set_(rules.lifted(Vec[i64](*(i64(size) for size in extent)), i64(axis))).to(
                i64(len(region.params) - 1)))
    for index, value in enumerate(region.literals):
        literal = terms.Term.literal(i64(index))
        if _power_of_two(value):
            facts.append(rules.exact_scale(literal))
        if np.ndim(value) == 0 and np.issubdtype(np.asarray(value).dtype, np.inexact) and value == 1:
            facts.append(rules.unit(literal))
    return facts


@dataclasses.dataclass(frozen=True)
class Saturated:
    """What a saturation did: the matches each rule set fired, its limits (nodes, iterations), the iterations
    it ran, and whether a limit stopped it."""

    fired: Mapping[str, int]
    limits: tuple[int, int]
    iterations: int
    limited: bool

    def after(self, first: Saturated) -> Saturated:
        """This saturation and `first`, which ran before it on the same graph, as one."""
        fired = {name: count + first.fired.get(name, 0) for name, count in self.fired.items()}
        return Saturated(fired, self.limits, first.iterations + self.iterations,
                         first.limited or self.limited)


def saturate(graph: EGraph, region: terms.Region, admitted: tuple[rules.Rules, ...],
             saturation: Saturation, facts: Schedule | None = None) -> Saturated:
    """`graph` run under the `admitted` rule sets to their fixpoint or `region`'s limits (`Saturation`), its
    facts saturated after each set: the engine's (`rules.FACTS`), or `facts` where a caller's rules read
    facts of their own (the derivation's, `dew.algebra.derive`)."""
    facts = rules.FACTS.saturate() if facts is None else facts
    fired = dict.fromkeys((admitted_rules.name for admitted_rules in admitted), 0)
    nodes, steps = saturation.limits(region)
    iterations, limited = 0, False
    while admitted:
        if iterations == steps or node_count(graph) > nodes:
            limited = True
            break
        updated = False
        for admitted_rules in admitted:
            graph.run(facts)
            ran = graph.run(admitted_rules.ruleset)
            fired[admitted_rules.name] += sum(ran.num_matches_per_rule.values())
            updated = updated or ran.updated
        iterations += 1
        if not updated:
            break
    graph.run(facts)
    return Saturated(fired, (nodes, steps), iterations, limited)


def extracted(graph: EGraph, bundle: terms.Term, model):
    """The roots' forms `model` prices cheapest together, each shared value once, and their price (the
    model's cost: `cost.Seconds` for the program's)."""
    chosen, price = graph.extract(bundle, include_cost=True, cost_model=model, extractor="greedy-dag")
    forms = get_callable_args(chosen, terms.Term.bundle)
    if forms is None:
        raise ValueError(f"extraction chose no bundle of the program's roots: {chosen}")
    return list(forms[0].value), price


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


def node_count(graph: EGraph) -> int:
    """The rows of every table of `graph`, its nodes and its facts' (egglog states no node count of its
    own)."""
    return sum(size for _, size in graph.all_function_sizes())
