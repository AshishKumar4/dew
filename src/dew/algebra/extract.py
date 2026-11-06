"""Extraction (docs/design/algebra.md §6.5): egglog's greedy DAG extraction by default (`cheapest`), and the
exact one (`extracted`), the roots' forms whose nodes together cost least, each shared node priced once, found
as an integer program rather than class by class.

egglog's `greedy-dag` extractor keeps for each e-class the node whose own node set costs least, which can miss
a choice that costs more in one class but shares more with the others. The exact extraction reads the same
nodes and the same model's prices (its `DagCostModel`, through a `greedy-dag` pass that records each node it
prices) as a mixed integer program, solved by HiGHS through SciPy (`scipy.optimize.milp`): one binary variable
a node, a used class choosing one node, a used node's child classes used, and an order variable a class in a
cycle of the graph that a used node's child must precede, so the chosen nodes are a DAG. A price states itself
as levels, compared in order (`Linear`): each the larger of its pieces, every piece a sum over the chosen
nodes, and of the largest of its nodes' peaks (a span, the longest chain's), plus what the nodes add beside it
(a reading's infinite price, a proposal's risk), each solved in turn with the levels before it held at their
optimum. The chosen DAG is then extracted by egglog itself, every chosen node free and every other priced, so
its term is the one the program chose.

Greedy stays the default: over the algebra's and the templates' test derivations (324 extractions) the exact
extraction priced no template's form below greedy's, cheaper only on test declarations (a row's scaling,
0.02%), and took twice greedy's time (15.8 s against 8.0). It is the oracle instead: under `Passes.checked`,
which the test suite sets, each template derivation's greedy choice is held to the exact optimum, and a
declaration greedy prices dearer is extracted exactly (`derive.Reduction.exact`).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Protocol

import numpy as np
from egglog import BaseExpr, DagCostModel, EGraph, get_callable_args, get_callable_fn
from egglog.egraph import ComparableAdd
from scipy.optimize import milp
from scipy.sparse import coo_array, csr_array
from scipy.sparse.csgraph import connected_components

from dew.algebra import terms


@dataclasses.dataclass(frozen=True)
class Level:
    """One level of a node's price (`Linear`): its `summed` pieces, each summed over the chosen nodes, its
    `peak`, the most of which any chosen node holds counts, and what it `added` beside them, summed."""

    summed: tuple[float, ...] = ()
    peak: float = 0.0
    added: float = 0.0


class Linear(ComparableAdd, Protocol):
    """A price extraction composes from its nodes' and compares level by level, in order (`levels`): a DAG's
    price at a level is the largest of its nodes' summed pieces and of their peaks, plus their additions
    (`valued`); egglog's comparable addition (`ComparableAdd`) beside."""

    def levels(self) -> tuple[Level, ...]: ...


@dataclasses.dataclass(frozen=True)
class Node[Price: Linear]:
    """A node the extractor read: its key (`terms.key`), the class it is in, the classes it reads (each
    argument's; a container's elements, a value read through it) and its price under the model."""

    key: Hashable
    owner: Hashable
    children: tuple[Hashable, ...]
    price: Price


class Dearer(ValueError):
    """A greedy extraction priced above the exact optimum where the oracle holds it to it (`cheapest`)."""


def cheapest[Root: BaseExpr, Price: Linear](graph: EGraph, root: Root, model: DagCostModel[Price], *,
                                         exact: bool = False, checked: bool = False) -> tuple[Root, Price]:
    """`root`'s form under `model` and its price: egglog's greedy DAG extraction, or the exact one
    (`extracted`) where the caller asks (`exact`). `checked`, the oracle, holds a greedy choice to the
    exact optimum and refuses one dearer at any level (`Dearer`), which then asks for the exact extraction:
    an infinite level, no form, compares no further."""
    if exact:
        return extracted(graph, root, model)
    form, price = graph.extract(root, include_cost=True, cost_model=model, extractor="greedy-dag")
    if checked:
        _, optimum = extracted(graph, root, model)
        if _dearer(price, optimum):
            raise Dearer(f"greedy extraction prices {root} at {price}, above the exact optimum {optimum}: "
                         "extract it exactly (derive.Reduction.exact)")
    return form, price


def valued(level: Level) -> float:
    """A composed price's `level` as one value: the larger of its pieces and its peak, plus its additions."""
    return max([*level.summed, level.peak]) + level.added


def _dearer(price: Linear, optimum: Linear) -> bool:
    """Whether `price` is above `optimum`, level by level, beyond rounding (a millionth, HiGHS's gap)."""
    for mine, best in zip(price.levels(), optimum.levels(), strict=True):
        ours, theirs = valued(mine), valued(best)
        if math.isinf(ours) and math.isinf(theirs):
            return False
        if ours > theirs + 1e-6 * abs(theirs):
            return True
        if theirs > ours + 1e-6 * abs(ours):
            return False
    return False


def extracted[Root: BaseExpr, Price: Linear](graph: EGraph, root: Root,
                                          model: DagCostModel[Price]) -> tuple[Root, Price]:
    """`root`'s cheapest form under `model` and its price, each shared node once (the module's account)."""
    nodes, owner, greedy = recorded(graph, root, model)
    chosen = solved(nodes, owner, greedy, model.identity)
    keys = {node.key for node in chosen}
    forced = DagCostModel(marginal_cost=lambda _, node: _Outside(int(terms.key(node) not in keys)),
                          identity=_Outside(0))
    return graph.extract(root, cost_model=forced, extractor="greedy-dag"), _priced(chosen, model.identity)


@dataclasses.dataclass(frozen=True, order=True)
class _Outside:
    """A forced extraction's price (`extracted`): the nodes it takes outside the chosen DAG."""

    count: int

    def __add__(self, other: _Outside) -> _Outside:
        return _Outside(self.count + other.count)


def recorded[Price: Linear](graph: EGraph, root: BaseExpr,
                            model: DagCostModel[Price]
                            ) -> tuple[dict[Hashable, Node[Price]], Hashable, Price]:
    """Every node a `greedy-dag` extraction of `root` prices under `model`, by its key, `root`'s class and
    the price that extraction reports:
    the extraction prices the nodes `root` reaches alone, so its class is the one no node reads (a bundle
    of roots), or the class of its row where the DAG reads it back. A node of a table (a constructor's row)
    is in the class its row holds; a value (a literal, a container) is a class of its own."""
    # The root first, so its constructors' tables are among those read: on a graph that does not hold it
    # yet, its rows are made by the extraction, after the tables were listed.
    graph.register(root)
    tables = {function for function, _ in graph.all_function_sizes()}
    nodes: dict[Hashable, Node[Price]] = {}

    def marginal(egraph: EGraph, node: BaseExpr) -> Price:
        price = model.marginal_cost(egraph, node)
        function = get_callable_fn(node)
        found = egraph.lookup_function_value(node) if function in tables else None
        key = terms.key(node)
        nodes[key] = Node(key, key if found is None else terms.key(found),
                          tuple(terms.key(argument) for argument in get_callable_args(node) or ()), price)
        return price

    _, greedy = graph.extract(root, include_cost=True, extractor="greedy-dag",
                              cost_model=DagCostModel(marginal_cost=marginal, identity=model.identity))
    read = {child for node in nodes.values() for child in node.children}
    unread = {node.owner for node in nodes.values()} - read
    if len(unread) == 1:
        return nodes, unread.pop(), greedy
    # A root its own DAG reads back (a pair its projections read) is found by its row; a wrapper is not
    # looked up, as egglog cannot register a container of none (a bundle of no roots) alone.
    found = graph.lookup_function_value(root)
    if found is None:
        raise ValueError(f"the e-graph holds no row of the extracted root {root}")
    return nodes, terms.key(found), greedy


def solved[Price: Linear](nodes: Mapping[Hashable, Node[Price]], root: Hashable, near: Price,
                          identity: Price) -> list[Node[Price]]:
    """The nodes of the cheapest DAG holding `root`'s class (the module's account), each level solved in
    units of `near`'s (a price of the graph's, greedy extraction's): the solver closes a gap of a millionth
    of a unit, so a level of picoseconds against peaks of microseconds is solved to its own digits. A level's
    choice is kept only where its DAG's earlier levels, priced as `identity` and its nodes add them, are no
    dearer than the choice before it: the solver's tolerance admits a dearer earlier level by a hair."""
    classes: dict[Hashable, list[Node[Price]]] = {}
    for node in nodes.values():
        classes.setdefault(node.owner, []).append(node)
    used = _reachable(classes, root)
    candidates = [node for owner in used for node in classes[owner]]
    program = _Program(candidates, {owner: classes[owner] for owner in used}, root)
    kept: list[Node[Price]] | None = None
    for depth, level in enumerate(near.levels()):
        program.level(depth, max([*level.summed, level.peak]) + (level.added if math.isfinite(level.added)
                                                                  else 0.0))
        taken = [node for node, on in zip(candidates, program.solution(), strict=True) if on]
        if kept is not None and _dearer_before(_priced(taken, identity), _priced(kept, identity), depth):
            break
        kept = taken
    return kept if kept is not None else [node for node, on in zip(candidates, program.solution(),
                                                                   strict=True) if on]


def _dearer_before(price: Linear, best: Linear, depth: int) -> bool:
    """Whether `price` is dearer than `best` at a level before `depth`, the first where they differ."""
    for mine, theirs in zip(price.levels()[:depth], best.levels()[:depth], strict=True):
        if valued(mine) != valued(theirs):
            return valued(mine) > valued(theirs)
    return False


def _reachable(classes: Mapping[Hashable, Sequence[Node]], root: Hashable) -> list[Hashable]:
    """The classes `root`'s reads reach, through every node of each, `root` first."""
    seen, order = {root}, [root]
    for owner in order:
        for node in classes[owner]:
            for child in node.children:
                if child in classes and child not in seen:
                    seen.add(child)
                    order.append(child)
    return order


def _cyclic(classes: Mapping[Hashable, Sequence[Node]]) -> dict[Hashable, int]:
    """Each class in a cycle of the graph of classes and their nodes' reads, by its strongly connected
    component (`scipy.sparse.csgraph.connected_components`); a class in no cycle is absent, as no order need
    hold it."""
    owners = list(classes)
    place = {owner: number for number, owner in enumerate(owners)}
    edges = {(place[owner], place[child]) for owner, nodes in classes.items() for node in nodes
             for child in node.children if child in place}
    rows, columns = zip(*edges, strict=True) if edges else ((), ())
    graph = csr_array((np.ones(len(rows)), (rows, columns)), shape=(len(owners), len(owners)))
    _, labels = connected_components(graph, directed=True, connection="strong")
    sizes = np.bincount(labels, minlength=len(owners))
    return {owner: int(labels[number]) for number, owner in enumerate(owners)
            if sizes[labels[number]] > 1 or (number, number) in edges}


FEASIBLE = 1e-6
"""The relative slack a solved level's bound keeps for the next (`_Program.level`): above HiGHS's primal
feasibility tolerance (1e-7), below any price difference a level resolves (a millionth of its unit)."""


class _Program:
    """The integer program over `nodes` (`solved`): its variables each node's choice, then each cyclic
    class's order, then each level's bound; its rows the DAG's constraints, a level's held optimum added as
    each is solved."""

    def __init__(self, nodes: Sequence[Node], classes: Mapping[Hashable, Sequence[Node]], root: Hashable):
        self.nodes = list(nodes)
        position = {id(node): number for number, node in enumerate(self.nodes)}
        cyclic = _cyclic(classes)
        self.order = {owner: len(self.nodes) + number for number, owner in enumerate(cyclic)}
        self.levels = len(self.nodes[0].price.levels())
        self.width = len(self.nodes) + len(self.order) + self.levels
        self.rows: list[tuple[dict[int, float], float, float]] = []
        self.upper = np.ones(self.width)
        # Orders range over the cyclic classes' count, one past which a node's unused order row always holds.
        self.big = float(len(self.order) + 1)
        self.upper[len(self.nodes):len(self.nodes) + len(self.order)] = len(self.order)
        self.upper[len(self.nodes) + len(self.order):] = np.inf
        self.rows.append(({position[id(node)]: 1.0 for node in classes[root]}, 1.0, 1.0))
        for owner, members in classes.items():
            self.rows.append(({position[id(node)]: 1.0 for node in members}, 0.0, 1.0))
            for node in members:
                for child in set(node.children):
                    if child not in classes:
                        continue
                    if child == owner:
                        self.upper[position[id(node)]] = 0.0
                        continue
                    reads = {position[id(member)]: 1.0 for member in classes[child]}
                    reads[position[id(node)]] = reads.get(position[id(node)], 0.0) - 1.0
                    self.rows.append((reads, 0.0, np.inf))
                    if owner in cyclic and cyclic.get(child) == cyclic[owner]:
                        # Used, the node's class comes after its child's: a cycle of used nodes cannot close.
                        self.rows.append(({self.order[owner]: 1.0, self.order[child]: -1.0,
                                           position[id(node)]: -self.big}, 1.0 - self.big, np.inf))
        self.chosen = np.zeros(len(self.nodes))

    def level(self, depth: int, unit: float) -> None:
        """Solve level `depth`'s least value in units of `unit` with the levels before it held, and hold
        it: a level whose additions are infinite on every DAG compares equal on all of them, and holds
        nothing."""
        levels = [node.price.levels()[depth] for node in self.nodes]
        infinite = {number: 1.0 for number, level in enumerate(levels) if math.isinf(level.added)}
        if infinite and self._least(infinite) > 0.5:
            return
        for number in infinite:
            self.upper[number] = 0.0
        bound = len(self.nodes) + len(self.order) + depth
        scale = 1.0 / unit if unit > 0 and math.isfinite(unit) else 1.0
        for piece in range(max((len(level.summed) for level in levels), default=0)):
            row = {number: -level.summed[piece] * scale for number, level in enumerate(levels)
                   if piece < len(level.summed) and level.summed[piece]}
            row[bound] = 1.0
            self.rows.append((row, 0.0, np.inf))
        for number, level in enumerate(levels):
            if level.peak:
                self.rows.append(({bound: 1.0, number: -level.peak * scale}, 0.0, np.inf))
        objective = {number: level.added * scale for number, level in enumerate(levels)
                     if level.added and math.isfinite(level.added)}
        objective[bound] = 1.0
        # Held at its least within the solver's feasibility tolerance: the least it reports is met only
        # within it, so a bound of exactly it may leave the next level no feasible point (the paged decode's
        # forms, HiGHS status 8); a choice the slack lets past is refused by `_dearer_before`.
        least = self._least(objective)
        self.rows.append((objective, -np.inf, least + FEASIBLE * max(1.0, abs(least))))

    def solution(self) -> list[bool]:
        """Each node's choice at the last level solved."""
        if self.levels == 0:
            self._least({})
        return [bool(value > 0.5) for value in self.chosen]

    def _least(self, objective: Mapping[int, float]) -> float:
        """The least value of `objective` under the program's rows, its choice kept (`chosen`)."""
        cost = np.zeros(self.width)
        for column, value in objective.items():
            cost[column] = value
        entries = [(number, column, value) for number, (row, _, _) in enumerate(self.rows)
                   for column, value in row.items()]
        rows, columns, values = zip(*entries, strict=True) if entries else ((), (), ())
        matrix = coo_array((np.asarray(values, float), (np.asarray(rows, int), np.asarray(columns, int))),
                           shape=(len(self.rows), self.width)).tocsr()
        integrality = np.zeros(self.width)
        integrality[:len(self.nodes)] = 1
        lower = np.asarray([low for _, low, _ in self.rows])
        upper = np.asarray([high for _, _, high in self.rows])
        optimum = milp(cost, constraints=(matrix, lower, upper), integrality=integrality,
                       bounds=(np.zeros(self.width), self.upper), options={"mip_rel_gap": 0.0})
        if optimum.status == 2:
            raise ValueError(f"no DAG of the e-graph's nodes holds its root: {optimum.message}")
        if not optimum.success or optimum.x is None:
            raise RuntimeError(f"HiGHS proved no optimum of the extraction (status {optimum.status}): "
                               f"{optimum.message}")
        self.chosen = optimum.x[:len(self.nodes)]
        return float(optimum.fun)


def _priced[Price: Linear](nodes: Iterable[Node[Price]], identity: Price) -> Price:
    """The price of `nodes` together, each once: `identity` and each node's price, added."""
    price = identity
    for node in nodes:
        price = price + node.price
    return price
