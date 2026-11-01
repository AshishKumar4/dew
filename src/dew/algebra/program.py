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
import time
from collections.abc import Callable, Iterator, Mapping

import jax
from egglog import EGraph, Vec, get_callable_args, i64, set_
from jax._src import config as jax_config
from jax.core import ShapedArray
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
    """A program's outputs computed in their cheapest forms, and the report of the rewrite that chose them."""

    outputs: list
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
    in place first (`_within`), then its own math (`_region`) where a rule may lower its price
    (`rules.may_rewrite`); evaluated as traced where none may."""
    closed = _within(closed, accurate, saturation)
    if not rules.may_rewrite(_primitives(closed)):
        return jax.core.eval_jaxpr(closed, closed.consts, *arguments)
    done = _region(closed, arguments, accurate=accurate, saturation=saturation)
    kernel_report.record_rewrite(done.report)
    return done.outputs


def _primitives(jaxpr: jcore.Jaxpr) -> collections.Counter[str]:
    """How many of `jaxpr`'s own equations bind each primitive, by name, its bodies' apart."""
    return collections.Counter(equation.primitive.name for equation in jaxpr.eqns)


def _region(closed: jcore.Jaxpr, arguments: list, *, accurate: bool, saturation: Saturation) -> Rewrite:
    """The outputs of `closed`, whose bodies are rewritten, over `arguments`, its roots' forms chosen together
    from its terms saturated under the admitted rule sets, with the report of the rewrite."""
    began = time.perf_counter()
    region = terms.region(closed)
    graph = EGraph()
    for term, extent in region.shapes:
        graph.register(set_(rules.shape(term)).to(Vec[i64](*(i64(size) for size in extent))))
    for index, (primitive, params) in enumerate(region.params):
        if primitive in (jax.lax.add_p, jax.lax.mul_p) and params.get("out_dtype") is None:
            graph.register(rules.plain(i64(index)))
        if primitive is jax.lax.dot_general_p and params.get("preferred_element_type") is None:
            graph.register(rules.plain(i64(index)))
        if primitive is jax.lax.broadcast_in_dim_p:
            rank, laid = len(params["shape"]), tuple(params["broadcast_dimensions"])
            missing = [axis for axis in range(rank) if axis not in laid]
            if len(missing) == 1 and laid == tuple(axis for axis in range(rank) if axis != missing[0]):
                graph.register(rules.along(i64(index), i64(missing[0])))
    roots = region.roots()
    bundle = terms.Term.bundle(Vec[terms.Term](*roots))
    graph.register(bundle)
    priced = cost.model(cost.Device.of())
    graph.run(rules.FACTS.saturate())
    _, before = graph.extract(bundle, include_cost=True, cost_model=priced, extractor="greedy-dag")
    admitted = rules.admitted(accurate)
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
    chosen, after = graph.extract(bundle, include_cost=True, cost_model=priced, extractor="greedy-dag")
    extracted = get_callable_args(chosen, terms.Term.bundle)
    if extracted is None:
        raise ValueError(f"extraction chose no bundle of the program's roots: {chosen}")
    (forms,) = extracted
    picked = {str(root): form for root, form in zip(roots, forms.value, strict=True)}
    values = [*closed.consts, *arguments] + [None] * (region.leaves - len(closed.consts) - len(arguments))
    outputs = terms.evaluated(region, picked, values)
    report = AlgebraReport(region.equations, tuple(fired), fired, (nodes, steps), iterations, _nodes(graph),
                           limited, before.value, after.value, time.perf_counter() - began)
    return Rewrite(outputs, report)


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
    inner = _within(closed, accurate, saturation)
    if not rules.may_rewrite(_primitives(inner)):
        # Nothing of its own a rule may lower: the body as traced, its own bodies rewritten, never traced
        # again.
        return inner
    shapes = [jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type)
              for aval in inner.in_avals if isinstance(aval, ShapedArray)]

    def run(*arguments):
        done = _region(inner, list(arguments), accurate=accurate, saturation=saturation)
        kernel_report.record_rewrite(done.report)
        return done.outputs

    made = jax.make_jaxpr(run)(*shapes)
    return inner if made.consts and not inner.consts else made


def _nodes(graph: EGraph) -> int:
    """The rows of every table of `graph`, its nodes and its facts' (egglog states no node count of its
    own)."""
    return sum(size for _, size in graph.all_function_sizes())
