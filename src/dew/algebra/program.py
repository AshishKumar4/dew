"""The program pass (docs/design/algebra.md §6.7): a function's jaxpr held as terms (`terms.region`), the
admitted rule sets run to their fixpoint or the region's limits, and its roots' cheapest forms under the cost
model extracted together and evaluated in place of the program's own. The bodies of its nested programs,
scans, loops, conditionals, checkpoints and custom derivatives' primal calls are rewritten in place, each a
region of its own, and kept bodies, never unrolled. Each rewrite is recorded in the one kernel report
(`dew.nn.kernels.templates.report.rewrites`)."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import inspect
import math
import threading
import time
import weakref
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from egglog import DagCostModel, EGraph, Schedule, String, Vec, get_callable_args, i64, set_
from jax._src import config as jax_config
from jax.core import ShapedArray, eval_jaxpr
from jax.extend import core as jcore

from dew.algebra import cost, extract, rules, terms
from dew.algebra.extract import Linear
from dew.cache import ALGEBRA
from dew.nn.derivatives import Algebra, KernelGlue, Surrogate, marked_call_p
from dew.nn.kernels import generation
from dew.nn.kernels.templates import report as kernel_report
from dew.nn.programs import inlined


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
    seconds (`cost.model`), the seconds this program's rewrite of it took, the roots whose cheaper form the
    program took (`spliced`, `_splices`: a picked form that does not land, a value another window's form made
    or a form of another type, is none), and whether an earlier window of its structure derived its forms
    (`reused`, `_derived`: its seconds a lookup's)."""

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
    spliced: int = 0
    reused: bool = False


@dataclasses.dataclass(frozen=True)
class Rewrite:
    """A region's terms (`terms.region`), each root's cheapest form by its key (`terms.key`; `picked`, empty
    where none is cheaper than the region as traced), and the report of the rewrite that chose them."""

    region: terms.Region
    picked: Mapping[Hashable, terms.Term]
    report: AlgebraReport


@contextlib.contextmanager
def passes(on: bool = True, target: str | None = None) -> Iterator[None]:
    """Within the block, the programs Dew compiles are rewritten (`on`) or not, priced for the GPU `target`
    names (None: the process's own): a context of JAX's trace, so a program held across the switch traces
    afresh (`dew.cache.ALGEBRA`). The accuracy gate holds it off for a reference arm and the float64 truth."""
    with ALGEBRA(dataclasses.replace(ALGEBRA.value, on=on, target=target)):
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
        # rewrite; where the run's numerics admit saturation aids alone (`Rules.aid`: the accurate forms'
        # commutativity), the pass would leave the program as it is.
        accurate = generation.numerics().forms == "accurate"
        if not _rewriting(accurate) or jax_config.disable_jit.value:
            return fn(*args, **kwargs)
        statics = _statics(args, numbers)
        fixed = {name: value for name, value in kwargs.items() if name in names}
        dynamic, named = _dynamic(args, kwargs, numbers, names)

        def bound(dynamic_args, dynamic_kwargs):
            given = iter(dynamic_args)
            whole = [args[number] if number in statics else next(given) for number in range(len(args))]
            return fn(*whole, **fixed, **dynamic_kwargs)

        closed, shaped = jax.make_jaxpr(bound, return_shape=True)(dynamic, named)
        done = rewrite(closed, jax.tree.leaves((dynamic, named)), accurate=accurate, saturation=saturation)
        return jax.tree.unflatten(jax.tree.structure(shaped), done)
    return run


def rewritten_program(closed: jcore.Jaxpr, *, saturation: Saturation) -> jcore.Jaxpr:
    """`closed`, a jit's own trace of a program, its bodies and its own math rewritten where the switch is on
    (`passes`, `rewrite`'s walks): `closed` itself where no rule changes it, so the trace jit made lowers as
    it is, never staged again (a Dory-4B training step's replay of its unchanged program, 1.8 seconds of
    its compile). The pass's whole seconds, its walks among them, recorded (`report.record_pass`)."""
    accurate = generation.numerics().forms == "accurate"
    if not _rewriting(accurate):
        return closed
    began = time.perf_counter()
    made = _rewritten(closed, accurate, saturation)
    kernel_report.record_pass(time.perf_counter() - began)
    return made


def replaying(closed: jcore.Jaxpr, structure, numbers: Iterable[int],
              names: Iterable[str]) -> Callable:
    """A function of a program's arguments computing `closed` (its rewritten program, `rewritten_program`)
    over its dynamic ones (`_dynamic`, as jit reads them), its outputs in `structure`: what a jit of it
    traces is `closed`'s program."""
    numbers, names = tuple(numbers), frozenset(names)

    def run(*args, **kwargs):
        dynamic, named = _dynamic(args, kwargs, numbers, names)
        return jax.tree.unflatten(structure, eval_jaxpr(closed.jaxpr, closed.consts,
                                                        *jax.tree.leaves((dynamic, named))))
    return run


def _rewriting(accurate: bool) -> bool:
    """Whether the pass rewrites here: the switch on, and the run's numerics admitting a set past saturation
    aids (`Rules.aid`: the accurate forms' commutativity), where the pass would leave a program as it is."""
    return ALGEBRA.value.on and not all(sets.aid for sets in rules.admitted(accurate))


def _statics(args: tuple, numbers: Iterable[int]) -> set[int]:
    """The positions among `args` the static `numbers` name, one from the end where negative, as jit reads
    them."""
    return {number % len(args) for number in numbers if -len(args) <= number < len(args)}


def _dynamic(args: tuple, kwargs: Mapping, numbers: Iterable[int],
             names: Iterable[str]) -> tuple[list, dict]:
    """A call's dynamic arguments, positional and named, its static ones (`numbers`, `names`) dropped: the
    values jit traces over, in its order."""
    statics, names = _statics(args, numbers), frozenset(names)
    return ([arg for number, arg in enumerate(args) if number not in statics],
            {name: value for name, value in kwargs.items() if name not in names})


def rewrite(closed: jcore.Jaxpr, arguments: list, *, accurate: bool, saturation: Saturation) -> list:
    """The outputs of `closed` (a jaxpr and its constants) over `arguments` (flattened), its bodies rewritten
    in place first (`_within`), then its own math (`_windowed`)."""
    rewritten = _rewritten(closed, accurate, saturation)
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
    are cheaper than the windows as traced: each cluster saturated in an e-graph of its own (one e-graph of
    every window's terms grew its extraction with the program, past 500 seconds a gemma3n training step's),
    each cheaper value's form traced and spliced in where its equation stood, the rest of the program kept as
    traced, so a body is never traced again. Two clusters whose union passes `CLUSTER` stay apart though they
    share an equation: a value both yield is made by the first's form (`_splices`' `taken`), each form
    writing the value it replaces under its own name, so readers of either read one value."""
    made_by = {var: equation for equation in jaxpr.eqns for var in equation.outvars}
    spliced: dict[int, list[jcore.JaxprEqn]] = {}
    windows = _windows(jaxpr)
    deriving = _deriving(accurate, saturation) if windows else None
    for window, yielded in windows:
        began = time.perf_counter()
        done, reused = _derived(window, accurate, saturation, deriving)
        seconds = time.perf_counter() - began
        landed = _splices(window, yielded, done, made_by, spliced) if done.picked else {}
        spliced.update(landed)
        # Recorded once its splice lands: what the program took, not what the extraction picked.
        kernel_report.record_rewrite(dataclasses.replace(done.report, spliced=len(landed), seconds=seconds,
                                                         reused=reused))
    if not spliced:
        return jaxpr
    return _live(jaxpr.replace(eqns=[kept for equation in jaxpr.eqns
                                     for kept in spliced.get(id(equation), [equation])]))


WINDOWS_HELD = 4096
"""The windows' derivations a process keeps (`_derived`), its least recently read dropped past them: the
distinct windows of the steps it compiles (161 of the 2,799 a UNet's planner traced over its compiles, each
33 ms on average to derive again; row 3764)."""

_DERIVED: collections.OrderedDict[Hashable, Rewrite] = collections.OrderedDict()
"""Each window's rewrite by what derived it (`_derived`), the least recently read first."""

_HOLDING = threading.Lock()
"""Held while `_DERIVED` is read or written: compiles trace on several threads."""


def structure(window: jcore.Jaxpr) -> Hashable:
    """What a window's derivation (`_region`) reads of it: each input's type (its aval: shape, dtype, weak
    type and placement, the sharding and memory its forms' values take); each equation's primitive, its
    parameters as a region holds them (`terms.parameters_key`), its operands (an input's or an earlier
    output's position, or a literal's value and type) and its outputs' types; and which values it yields.
    Two windows of one structure, their values named apart, derive one rewrite: every layer of a stack's,
    every compile's of one step."""
    position = {var: index for index, var in enumerate(window.invars)}

    def operand(atom) -> Hashable:
        if isinstance(atom, jcore.Literal):
            value = np.asarray(atom.val)
            return value.dtype, value.shape, value.tobytes(), atom.aval
        return position[atom]

    equations = []
    for equation in window.eqns:
        equations.append((terms.parameters_key(equation.primitive, equation.params),
                          tuple(map(operand, equation.invars)), tuple(var.aval for var in equation.outvars)))
        position.update({var: len(position) for var in equation.outvars})
    return tuple(var.aval for var in window.invars), tuple(equations), tuple(map(operand, window.outvars))


def _deriving(accurate: bool, saturation: Saturation) -> Hashable:
    """What a rewrite's derivation reads beside its program, which keeps every rewrite the pass holds
    (`_derived`, `_closed`): the numerics' forms, the saturation, the device it is priced for (`cost.target`,
    resolved: a named target, an assumed generation or the process's own), the matmul precision a product
    of none stated is priced at (`cost._mode`) and the rule sets the registry admits for those numerics
    (`rules.admitted`, by name), so a target assumed, a precision set or a set registered later derives
    afresh."""
    return (accurate, saturation, cost.target(), str(jax.config.jax_default_matmul_precision),
            tuple(sets.name for sets in rules.admitted(accurate)))


def _derived(window: jcore.Jaxpr, accurate: bool, saturation: Saturation,
             deriving: Hashable) -> tuple[Rewrite, bool]:
    """`window`'s rewrite (`_region`) and whether an earlier window derived it: one derivation for each
    structure (`structure`) and what else it reads (`deriving`, `_deriving`), the process's latest
    `WINDOWS_HELD` kept. The planner's candidates and a resumed run's relowering trace one step's program
    again and again, each window of it as the last trace's."""
    key = (structure(window), deriving)
    with _HOLDING:
        done = _DERIVED.get(key)
        if done is not None:
            _DERIVED.move_to_end(key)
            return done, True
    done = _region(window, accurate=accurate, saturation=saturation)
    with _HOLDING:
        _DERIVED[key] = done
        if len(_DERIVED) > WINDOWS_HELD:
            _DERIVED.popitem(last=False)
    return done, False


def _splices(window: jcore.Jaxpr, yielded: list[jcore.Var], done: Rewrite,
             made_by: Mapping[jcore.Var, jcore.JaxprEqn], taken: Mapping[int, list]) -> dict[int, list]:
    """The equations each value `window` yields (`yielded`, as the program names them) is made by in its
    cheapest form (`done`), by the id of the program's equation they stand in for (`made_by`): a value another
    window's form made already (`taken`) kept as that one made it."""
    leaves = [jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype, weak_type=var.aval.weak_type)
              for var in window.invars if isinstance(var.aval, ShapedArray)]
    if len(leaves) != len(window.invars):
        return {}
    spliced: dict[int, list[jcore.JaxprEqn]] = {}
    for out, root in zip(yielded, done.region.outputs, strict=True):
        if root is None:
            continue
        form = done.picked.get(terms.key(root))
        made_at = id(made_by[out])
        if form is None or terms.key(form) == terms.key(root) or made_at in taken or made_at in spliced:
            continue
        traced = jax.make_jaxpr(lambda *values, form=form: terms.form_value(done.region, form, list(values)))(
            *leaves)
        made, = traced.jaxpr.outvars
        if traced.consts or not isinstance(made, jcore.Var) or made in traced.jaxpr.invars or (
                traced.out_avals[0].strip_weak_type() != out.aval.strip_weak_type()):
            continue
        renamed = {**dict(zip(traced.jaxpr.invars, window.invars, strict=True)), made: out}

        def renaming(atom, renamed=renamed):
            return atom if isinstance(atom, jcore.Literal) else renamed.get(atom, atom)

        spliced[made_at] = [equation.replace(invars=[renaming(atom) for atom in equation.invars],
                                             outvars=[renaming(var) for var in equation.outvars])
                            for equation in traced.jaxpr.eqns]
    return spliced


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
    the admitted rule sets where they are cheaper than the region as traced (`cost.cheaper`), and the report
    of the rewrite. A region no reordering may touch (`rules.reorderable`) builds no e-graph. The saturated
    graph's forms are extracted first; only where one differs from its root is the region as traced
    extracted too, from the graph as it stood (egglog's `push`, `pop`): a region whose forms are its roots'
    (commutativity's alone) takes one extraction."""
    began = time.perf_counter()
    admitted = rules.admitted(accurate)
    # A window's structure decides first, its egglog terms built only where a rule may fire.
    skeleton = terms.region(closed, forms=terms.Skeleton)
    if not rules.reorderable(skeleton):
        fired = dict.fromkeys((sets.name for sets in admitted), 0)
        report = AlgebraReport(skeleton.equations, tuple(fired), fired, saturation.limits(skeleton),
                               iterations=0, nodes=0, limited=False, before=0.0, after=0.0,
                               seconds=time.perf_counter() - began)
        return Rewrite(skeleton, {}, report)
    region = terms.region(closed)
    priced = cost.model(region, cost.fused(cost.target()))
    graph, bundle, roots = graphed(region)
    graph.push()
    saturated = saturate(graph, region, admitted, saturation)
    nodes = node_count(graph)
    forms, after = extracted(graph, bundle, priced)
    changed = any(terms.key(form) != terms.key(root) for root, form in zip(roots, forms, strict=True))
    graph.pop()
    before = after
    if changed:
        _, before = extract.cheapest(graph, bundle, priced)
    picked = ({terms.key(root): form for root, form in zip(roots, forms, strict=True)}
              if changed and cost.cheaper(after, before) else {})
    report = AlgebraReport(region.equations, tuple(saturated.fired), saturated.fired, saturated.limits,
                           saturated.iterations, nodes, saturated.limited, before.value, after.value,
                           time.perf_counter() - began)
    return Rewrite(region, picked, report)


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
            _, params = region.params[terms.integer(args[3])]
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
        # The largest power of two under the float's largest, exactly (float64's log2 rounds to 1024).
        facts.append(set_(rules.reach()).to(i64(math.frexp(float(min(floats)))[1] - 1)))
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
        index = terms.integer(args[2])
        _, params = region.params[index]
        # Its contracted length's power, the elements each output sums (`rules.contracted_length`).
        (contracting, _), _ = params["dimension_numbers"]
        length = math.prod(region.shape(args[0])[axis] for axis in contracting)
        facts.append(set_(rules.contracted_length(term)).to(i64(max(length - 1, 0).bit_length())))
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
        if np.ndim(value) == 0 and jnp.issubdtype(np.asarray(value).dtype, jnp.inexact) and value > 0:
            facts.append(rules.positive_scale(literal))
        if np.ndim(value) == 0 and jnp.issubdtype(np.asarray(value).dtype, jnp.inexact) and value == 1:
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


def extracted[Price: Linear](graph: EGraph, bundle: terms.Term, model: DagCostModel[Price], *,
                             exact: bool = False, checked: bool = False) -> tuple[list[terms.Term], Price]:
    """The roots' forms `model` prices cheapest together, each shared value once, and their price (the
    model's cost: `cost.Seconds` for the program's), extracted as `extract.cheapest` extracts them."""
    chosen, price = extract.cheapest(graph, bundle, model, exact=exact, checked=checked)
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
    kept: `jaxpr` itself, and each equation itself, where no body changes. A program's identity is what
    JAX shares its lowering by: every call site of one jitted function holds one body, lowered once (one
    function of the module, its kernels compiled once), so a pass that rebuilt an unchanged body at each
    call site lowered and compiled every layer's apart (a Qwen3.5-4B training step's compile 52 to 487
    seconds on an H100, nothing rewritten)."""
    changed = False
    equations = []
    for equation in jaxpr.eqns:
        if equation.primitive.name in BODIES or _marked(equation, WITHIN) or (
                _marked(equation, (Algebra,)) and not _opens(equation)):
            params = {name: _body(value, accurate, saturation) for name, value in equation.params.items()}
        elif _opens(equation):
            # Its body's own bodies rewritten, its math left to the region it opens into (`_opened`).
            params = {**equation.params, "call_jaxpr": _within(equation.params["call_jaxpr"], accurate,
                                                               saturation)}
        else:
            params = equation.params
        if any(params[name] is not value for name, value in equation.params.items()):
            equation = equation.replace(params=params)
            changed = True
        equations.append(equation)
    return jaxpr.replace(eqns=equations) if changed else jaxpr


WITHIN = (Surrogate, KernelGlue)
"""The marked spans' kinds (`dew.nn.derivatives.marked`) whose body the pass rewrites within its call, as it
rewrites a body (`BODIES`): a surrogate's bodies and a kernel derivative's glue (docs/design/algebra.md §1.2).
A protected span is an opaque leaf, its body untouched; an Algebra span opens into the region around it
(`_opened`)."""


def _marked(equation: jcore.JaxprEqn, kinds: tuple[type, ...]) -> bool:
    """Whether `equation` is a marked span of one of `kinds`."""
    return equation.primitive is marked_call_p and isinstance(equation.params["kind"], kinds)


def _opens(equation: jcore.JaxprEqn) -> bool:
    """Whether `equation` is an Algebra span the pass opens into the region around it (`_opened`): one of no
    effects over arrays alone, whose renamed variables no effect names (a reference's would). Any other
    Algebra span is rewritten within its call."""
    return (_marked(equation, (Algebra,)) and not equation.effects
            and all(isinstance(var.aval, ShapedArray) for var in (*equation.invars, *equation.outvars)))


def _rewritten(closed: jcore.Jaxpr, accurate: bool, saturation: Saturation) -> jcore.Jaxpr:
    """`closed`'s bodies rewritten in place (`_within`), then its own math (`_windowed`) with each Algebra
    span's body opened into it (`_opened`), so a window reaches across the span's boundary; the program
    with its spans kept as traced where no window's form lands (`closed` itself where no body changed
    either)."""
    within = _within(closed, accurate, saturation)
    opened = _opened(within)
    made = _windowed(opened, accurate, saturation)
    return within if made is opened else made


def _opened(jaxpr: jcore.Jaxpr) -> jcore.Jaxpr:
    """`jaxpr` with each Algebra span's call it opens (`_opens`) replaced by its body's equations, nested
    spans opened too, over the call's operands, its body's constants the program's, and each reader of the
    call's results reading what its body computes them as; `jaxpr` itself where it holds none."""
    if not any(_opens(equation) for equation in jaxpr.eqns):
        return jaxpr
    constvars, consts = list(jaxpr.constvars), list(jaxpr.consts)
    computed: dict[jcore.Var, jcore.Var | jcore.Literal] = {}
    equations: list[jcore.JaxprEqn] = []

    def outer(atom: jcore.Var | jcore.Literal) -> jcore.Var | jcore.Literal:
        return atom if isinstance(atom, jcore.Literal) else computed.get(atom, atom)

    for equation in jaxpr.eqns:
        equation = equation.replace(invars=[outer(atom) for atom in equation.invars])
        if not _opens(equation):
            equations.append(equation)
            continue
        body = _opened(equation.params["call_jaxpr"])
        # The body's constants the program's own, each a fresh variable of it (`inlined`).
        fresh = [jcore.Var(var.aval) for var in body.constvars]
        constvars += fresh
        consts += list(body.consts)
        made, returned = inlined(body, equation.invars, fresh)
        equations += made
        computed.update(zip(equation.outvars, returned, strict=True))
    return jaxpr.replace(constvars=constvars, consts=consts, eqns=equations,
                         outvars=[outer(atom) for atom in jaxpr.outvars])


def _body(value, accurate: bool, saturation: Saturation):
    """A body parameter rewritten: a jaxpr's program, each of a conditional's branches (the tuple itself
    where none changes); any other parameter, and a body whose constants it does not hold, as it is."""
    if isinstance(value, jcore.Jaxpr):
        return _closed(value, accurate, saturation)
    if isinstance(value, tuple) and value and all(isinstance(branch, jcore.Jaxpr) for branch in value):
        branches = tuple(_closed(branch, accurate, saturation) for branch in value)
        return value if all(new is old for new, old in zip(branches, value, strict=True)) else branches
    return value


_REWRITTEN: weakref.WeakKeyDictionary[jcore.Jaxpr, dict[Hashable, jcore.Jaxpr | None]] = (
    weakref.WeakKeyDictionary())
"""Each body's rewrites (`_closed`) by what a rewrite read (the numerics' forms, the saturation, the
switch's target), held while the body lives: every call site of one body, and every retrace that holds it
(JAX's tracing cache gives a function's one body), takes one rewritten object, which JAX lowers once. A body
nothing changes holds None, not itself, which would keep it alive."""


def _closed(closed: jcore.Jaxpr, accurate: bool, saturation: Saturation) -> jcore.Jaxpr:
    """A body's program rewritten, over the same inputs, weak types included, and the same outputs, the same
    object for every call site and retrace of one body (`_REWRITTEN`), the body itself where nothing
    changes; a body whose constants another holds (an open jaxpr), or that reads a value of no shape, as it
    is."""
    if len(closed.consts) != len(closed.constvars):
        return closed
    if not all(isinstance(aval, ShapedArray) for aval in closed.in_avals):
        return closed
    rewrites = _REWRITTEN.setdefault(closed, {})
    key = _deriving(accurate, saturation)
    if key not in rewrites:
        made = _rewritten(closed, accurate, saturation)
        rewrites[key] = None if made is closed else made
    made = rewrites[key]
    return closed if made is None else made


def _power_of_two(value) -> bool:
    """Whether a literal is a scalar power of two, a multiply by which rounds nothing: a float's of any dtype,
    ml_dtypes' (bfloat16, float8) among them, which numpy's own type tree does not place among its floats."""
    if np.ndim(value) != 0 or not jnp.issubdtype(np.asarray(value).dtype, jnp.floating):
        return False
    mantissa, _ = np.frexp(float(value))
    return mantissa in (0.5, -0.5)


def node_count(graph: EGraph) -> int:
    """The rows of every table of `graph`, its nodes and its facts' (egglog states no node count of its
    own)."""
    return sum(size for _, size in graph.all_function_sizes())
