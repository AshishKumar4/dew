"""The program pass (docs/design/algebra.md §6.7): a function's jaxpr held as terms (`terms.region`), the
admitted rule sets run to their fixpoint or the region's limits, and its roots' cheapest forms under the cost
model extracted together and evaluated in place of the program's own."""

from __future__ import annotations

import dataclasses
import functools
import time
from collections.abc import Callable

import jax
from egglog import EGraph, Vec, get_callable_args, i64, set_
from jax.extend import core as jcore

from dew.algebra import cost, rules, terms


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
    """What one rewrite did: the equations it read, the sets admitted, its limits, its iterations and nodes,
    whether a limit stopped it, and its seconds."""

    equations: int
    admitted: tuple[str, ...]
    limits: tuple[int, int]
    iterations: int
    nodes: int
    limited: bool
    seconds: float


@dataclasses.dataclass(frozen=True)
class Rewrite:
    """A program's outputs computed in their cheapest forms, and the report of the rewrite that chose them."""

    outputs: list
    report: AlgebraReport


def rewritten(fn: Callable, *, accurate: bool = False, saturation: Saturation) -> Callable:
    """`fn` with its program rewritten: traced over its arguments, its terms saturated under the rule sets its
    numerics admit (`rules.admitted`) and its values evaluated in their cheapest forms. Its arguments, outputs
    and their structure are `fn`'s."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        closed, shaped = jax.make_jaxpr(functools.partial(fn, **kwargs), return_shape=True)(*args)
        done = rewrite(closed, jax.tree.leaves(args), accurate=accurate, saturation=saturation)
        return jax.tree.unflatten(jax.tree.structure(shaped), done.outputs)
    return run


def rewrite(closed: jcore.Jaxpr, arguments: list, *, accurate: bool, saturation: Saturation) -> Rewrite:
    """The outputs of `closed` over `arguments` (flattened), its roots' forms chosen together, with the
    report of the rewrite."""
    began = time.perf_counter()
    region = terms.region(closed)
    graph = EGraph()
    for term, extent in region.shapes:
        graph.register(set_(rules.shape(term)).to(Vec[i64](*(i64(size) for size in extent))))
    for index, (primitive, params) in enumerate(region.params):
        if primitive in (jax.lax.add_p, jax.lax.mul_p) and params.get("out_dtype") is None:
            graph.register(rules.plain(i64(index)))
        if primitive is jax.lax.broadcast_in_dim_p:
            rank, laid = len(params["shape"]), tuple(params["broadcast_dimensions"])
            missing = [axis for axis in range(rank) if axis not in laid]
            if len(missing) == 1 and laid == tuple(axis for axis in range(rank) if axis != missing[0]):
                graph.register(rules.along(i64(index), i64(missing[0])))
    roots = region.roots()
    bundle = terms.Term.bundle(Vec[terms.Term](*roots))
    graph.register(bundle)
    admitted = rules.admitted(accurate)
    nodes, steps = saturation.limits(region)
    iterations, limited = 0, False
    while admitted:
        if iterations == steps or _nodes(graph) > nodes:
            limited = True
            break
        graph.run(rules.FACTS.saturate())
        report = graph.run(functools.reduce(lambda left, right: left | right,
                                            (admitted_rules.ruleset for admitted_rules in admitted)))
        iterations += 1
        if not report.updated:
            break
    graph.run(rules.FACTS.saturate())
    chosen = graph.extract(bundle, cost_model=cost.model(cost.Device.of()), extractor="greedy-dag")
    extracted = get_callable_args(chosen, terms.Term.bundle)
    if extracted is None:
        raise ValueError(f"extraction chose no bundle of the program's roots: {chosen}")
    (forms,) = extracted
    picked = {str(root): form for root, form in zip(roots, forms.value, strict=True)}
    values = [*closed.consts, *arguments] + [None] * (region.leaves - len(closed.consts) - len(arguments))
    outputs = terms.evaluated(region, picked, values)
    report = AlgebraReport(region.equations, tuple(admitted_rules.name for admitted_rules in admitted),
                           (nodes, steps), iterations, _nodes(graph), limited, time.perf_counter() - began)
    return Rewrite(outputs, report)


def _nodes(graph: EGraph) -> int:
    """The rows of every table of `graph`, its nodes and its facts' (egglog states no node count of its
    own)."""
    return sum(size for _, size in graph.all_function_sizes())
