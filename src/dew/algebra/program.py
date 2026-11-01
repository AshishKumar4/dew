"""The program pass (docs/design/algebra.md §6.7): a function's jaxpr held as terms (`terms.region`), the
admitted rule sets run to their fixpoint or the region's limits, and each value's cheapest form under the cost
model extracted and evaluated in place of the program's own."""

from __future__ import annotations

import dataclasses
import functools
import time
from collections.abc import Callable

import jax
from egglog import EGraph, i64
from jax.extend import core as jcore

from dew.algebra import cost, rules, terms


@dataclasses.dataclass(frozen=True)
class Saturation:
    """The limits a region's saturation stops at, from its size: nodes at `growth` per equation, iterations at
    `iterations`. Measured: the largest the registered sets reach at their fixpoint on tests/test_algebra.py's
    programs."""

    growth: int
    iterations: int


MEASURED = Saturation(growth=64, iterations=16)


@dataclasses.dataclass(frozen=True)
class AlgebraReport:
    """What one rewrite did: the equations it read, the sets admitted, its iterations and nodes, whether a
    limit stopped it, and its seconds."""

    equations: int
    admitted: tuple[str, ...]
    iterations: int
    nodes: int
    limited: bool
    seconds: float


REPORTS: list[AlgebraReport] = []
"""This process's rewrites, newest last."""


def rewritten(fn: Callable, *, accurate: bool = False, saturation: Saturation = MEASURED) -> Callable:
    """`fn` with its program rewritten: traced over its arguments, its terms saturated under the rule sets its
    numerics admit (`rules.admitted`) and each value's cheapest form evaluated in its place. Its arguments,
    outputs and their structure are `fn`'s."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        closed, shape = jax.make_jaxpr(functools.partial(fn, **kwargs), return_shape=True)(*args)
        outputs = rewritten_closed(closed, jax.tree.leaves(args), accurate=accurate, saturation=saturation)
        return jax.tree.unflatten(jax.tree.structure(shape), outputs)
    return run


def rewritten_closed(closed: jcore.ClosedJaxpr, arguments: list, *, accurate: bool,
                     saturation: Saturation) -> list:
    """The outputs of `closed` over `arguments` (flattened), each value in its cheapest form."""
    began = time.perf_counter()
    region = terms.region(closed)
    graph = EGraph()
    roots: dict[str, terms.Term] = {}
    for term in [*region.outputs, *(term for opaque in region.opaque for term in opaque.inputs)]:
        if term is not None and str(term) not in roots:
            roots[str(term)] = graph.let(f"root{len(roots)}", term)
    for index, (primitive, params) in enumerate(region.params):
        if primitive is jax.lax.broadcast_in_dim_p:
            rank, laid = len(params["shape"]), tuple(params["broadcast_dimensions"])
            missing = [axis for axis in range(rank) if axis not in laid]
            if len(missing) == 1 and laid == tuple(axis for axis in range(rank) if axis != missing[0]):
                graph.register(rules.along(i64(index), i64(missing[0])))
    admitted = rules.admitted(accurate)
    budget = saturation.growth * max(1, len(closed.jaxpr.eqns))
    iterations, limited = 0, False
    while iterations < saturation.iterations:
        graph.run(rules.FACTS.saturate())
        report = graph.run(sum((admitted_rules.ruleset for admitted_rules in admitted[1:]),
                               admitted[0].ruleset) if admitted else rules.FACTS)
        iterations += 1
        if _nodes(graph) > budget:
            limited = True
            break
        if not report.updated:
            break
    else:
        limited = True
    graph.run(rules.FACTS.saturate())
    priced = cost.model(cost.Device.of())
    chosen = {key: graph.extract(handle, cost_model=priced, extractor="greedy-dag")
              for key, handle in roots.items()}
    values = [*closed.consts, *arguments] + [None] * (region.leaves - len(closed.consts) - len(arguments))
    outputs = terms.evaluated(region, lambda term: chosen[str(term)], values)
    names = tuple(admitted_rules.name for admitted_rules in admitted)
    REPORTS.append(AlgebraReport(len(closed.jaxpr.eqns), names, iterations, _nodes(graph), limited,
                                 time.perf_counter() - began))
    return outputs


def _nodes(graph: EGraph) -> int:
x
    return sum(size for _, size in graph.all_function_sizes())
