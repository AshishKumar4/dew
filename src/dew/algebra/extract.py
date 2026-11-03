"""Exact DAG extraction (docs/design/algebra.md §6.5): the roots' forms whose nodes together cost least, each
shared node priced once, found as an integer program rather than class by class.

egglog's `greedy-dag` extractor keeps for each e-class the node whose own node set costs least, which misses a
choice that costs more in one class but shares more with the others. Here the same nodes and the same model's
prices (its `DagCostModel`, read through a `greedy-dag` pass that records each node it prices) become a mixed
integer program, solved by HiGHS through SciPy (`scipy.optimize.milp`): one binary variable a node, a used
class choosing one node, a used node's child classes used, and an order variable a class in a cycle of the
graph that a used node's child must precede, so the chosen nodes are a DAG. A price states itself as levels,
compared in order (`Linear`): each the larger of its pieces, every piece a sum over the chosen nodes, and of
the largest of its nodes' peaks (a span, the longest chain's), plus what the nodes add beside it (a reading's
infinite price, a proposal's risk), each solved in turn with the levels before it held at their optimum.
The chosen DAG is then extracted by egglog itself, every chosen node free and every other priced, so its term
is the one the program chose.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Protocol, Self

import numpy as np
from egglog import BaseExpr, DagCostModel, EGraph, get_callable_args, get_callable_fn
from scipy.optimize import milp
from scipy.sparse import coo_array

from dew.algebra import terms


@dataclasses.dataclass(frozen=True)
class Level:
    """One level of a node's price (`Linear`): its `summed` pieces, each summed over the chosen nodes, its
    `peak`, the most of which any chosen node holds counts, and what it `added` beside them, summed."""

    summed: tuple[float, ...] = ()
    peak: float = 0.0
    added: float = 0.0


class Linear(Protocol):
    """A price extraction composes from its nodes' and compares level by level, in order (`levels`): a DAG's
    price at a level is the largest of its nodes' summed pieces and of their peaks, plus their additions."""

    def levels(self) -> tuple[Level, ...]: ...

    def __add__(self, other: Self) -> Self: ...

    def __lt__(self, other: Self) -> bool: ...


@dataclasses.dataclass(frozen=True)
class Node[Price: Linear]:
    """A node the extractor read: its key (`terms.key`), the class it is in, the classes it reads (each
    argument's; a container's elements, a value read through it) and its price under the model."""

    key: Hashable
    owner: Hashable
    children: tuple[Hashable, ...]
    price: Price


def extracted[Price: Linear](graph: EGraph, root: BaseExpr,
                             model: DagCostModel[Price]) -> tuple[BaseExpr, Price]:
    """`root`'s cheapest form under `model` and its price, each shared node once (the module's account)."""
    nodes, owner, greedy = recorded(graph, root, model)
    chosen = solved(nodes, owner, greedy)
    keys = {node.key for node in chosen}
    forced = DagCostModel(marginal_cost=lambda _, node: int(terms.key(node) not in keys), identity=0)
    return graph.extract(root, cost_model=forced, extractor="greedy-dag"), _priced(chosen, model.identity)


def recorded[Price: Linear](graph: EGraph, root: BaseExpr,
                            model: DagCostModel[Price]
                            ) -> tuple[dict[Hashable, Node[Price]], Hashable, Price]:
    """Every node a `greedy-dag` extraction of `root` prices under `model`, by its key, `root`'s class and
    the price that extraction reports:
    the extraction prices the nodes `root` reaches alone, so its class is the one no node reads (a bundle
    of roots), or the class of its row where the DAG reads it back. A node of a table (a constructor's row)
    is in the class its row holds; a value (a literal, a container) is a class of its own."""
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


def solved[Price: Linear](nodes: Mapping[Hashable, Node[Price]], root: Hashable,
                          near: Price) -> list[Node[Price]]:
    """The nodes of the cheapest DAG holding `root`'s class (the module's account), each level solved in
    units of `near`'s (a price of the graph's, greedy extraction's): the solver closes a gap of a millionth
    of a unit, so a level of picoseconds against peaks of microseconds is solved to its own digits."""
    classes: dict[Hashable, list[Node[Price]]] = {}
    for node in nodes.values():
        classes.setdefault(node.owner, []).append(node)
    used = _reachable(classes, root)
    chosen = [node for owner in used for node in classes[owner]]
    program = _Program(chosen, {owner: classes[owner] for owner in used}, root)
    for depth, level in enumerate(near.levels()):
        program.level(depth, max([*level.summed, level.peak]) + (level.added if math.isfinite(level.added)
                                                                  else 0.0))
    return [node for node, taken in zip(chosen, program.solution(), strict=True) if taken]


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
    component (Tarjan's, iterative); a class in no cycle is absent, as no order need hold it."""
    index: dict[Hashable, int] = {}
    low: dict[Hashable, int] = {}
    stack: list[Hashable] = []
    on_stack: set[Hashable] = set()
    component: dict[Hashable, int] = {}
    edges = {owner: sorted({child for node in nodes for child in node.children if child in classes},
                           key=repr) for owner, nodes in classes.items()}
    for start in classes:
        if start in index:
            continue
        work = [(start, 0)]
        while work:
            owner, position = work.pop()
            if position == 0:
                index[owner] = low[owner] = len(index)
                stack.append(owner)
                on_stack.add(owner)
            if position < len(edges[owner]):
                work.append((owner, position + 1))
                child = edges[owner][position]
                if child not in index:
                    work.append((child, 0))
                elif child in on_stack:
                    low[owner] = min(low[owner], index[child])
                continue
            if low[owner] == index[owner]:
                members = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    members.append(member)
                    if member == owner:
                        break
                if len(members) > 1 or owner in edges[owner]:
                    component.update(dict.fromkeys(members, len(component)))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[owner])
    return component


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
        least = self._least(objective)
        self.rows.append((objective, -np.inf, least + 1e-9 * max(1.0, abs(least))))

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
        if optimum.x is None:
            raise ValueError(f"no DAG of the e-graph's nodes holds its root: {optimum.message}")
        self.chosen = optimum.x[:len(self.nodes)]
        return float(optimum.fun)


def _priced[Price: Linear](nodes: Iterable[Node[Price]], identity: Price) -> Price:
    """The price of `nodes` together, each once: `identity` and each node's price, added."""
    price = identity
    for node in nodes:
        price = price + node.price
    return price
