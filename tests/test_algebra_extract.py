"""Extraction (`dew.algebra.extract`, docs/design/algebra.md §6.5): egglog's greedy DAG extraction by default,
and the exact one, the roots' forms whose nodes together cost least, each shared node once, where greedy keeps
each class's cheapest node set alone and pays twice for what a costlier choice would have shared; the oracle
holds greedy's choice to the exact optimum."""

from __future__ import annotations

import dataclasses
import math

import pytest
from egglog import (
    DagCostModel,
    EGraph,
    Expr,
    StringLike,
    get_callable_args,
    get_literal_value,
    i64Like,
    union,
)

from dew.algebra import extract


class Toy(Expr):
    """A language of named leaves and one- and two-operand nodes, each priced by its name (`PRICES`)."""

    @classmethod
    def leaf(cls, name: StringLike) -> Toy: ...

    @classmethod
    def one(cls, name: StringLike, operand: Toy) -> Toy: ...

    @classmethod
    def two(cls, left: Toy, right: Toy) -> Toy: ...

    @classmethod
    def scaled(cls, operand: Toy, by: i64Like) -> Toy: ...


@dataclasses.dataclass(frozen=True, order=True)
class Cost:
    """Seconds, then a tie, each summed over the chosen nodes (`extract.Linear`)."""

    seconds: float
    tie: float = 0.0

    def __add__(self, other: Cost) -> Cost:
        return Cost(self.seconds + other.seconds, self.tie + other.tie)

    def levels(self) -> tuple[extract.Level, ...]:
        return (extract.Level(added=self.seconds), extract.Level(added=self.tie))


PRICES = {"big": Cost(10.0), "shared": Cost(6.0), "small": Cost(6.5), "f": Cost(1.0), "g": Cost(1.0),
          "slow": Cost(math.inf), "fast": Cost(3.0), "tied": Cost(1.0, 1.0),
          "untied": Cost(1.0000000005, 0.0)}


def priced(graph: EGraph, node) -> Cost:
    del graph
    for constructor in (Toy.leaf, Toy.one):
        if (args := get_callable_args(node, constructor)) is not None:
            return PRICES[str(get_literal_value(args[0]))]
    return Cost(0.0)


MODEL = DagCostModel(marginal_cost=priced, identity=Cost(0.0))


def sharing() -> tuple[EGraph, Toy]:
    """Two roots' classes: one a big leaf or `f` of a shared leaf, the other a small leaf or `g` of the same
    shared leaf. Alone, each class's cheapest set is `f(shared)` (7) and `small` (6.5); together, `g(shared)`
    shares the leaf `f` already pays for, the pair 8 against 13.5."""
    shared = Toy.leaf("shared")
    first, second = Toy.leaf("big"), Toy.leaf("small")
    graph = EGraph()
    root = Toy.two(first, second)
    graph.register(root, union(first).with_(Toy.one("f", shared)), union(second).with_(Toy.one("g", shared)))
    return graph, root


def test_a_choice_that_shares_a_node_beats_each_class_cheapest_alone():
    """Greedy pays 13.5 where the exact extraction shares the leaf for 8; the oracle refuses greedy's choice
    as dearer, and a caller that asks for the exact extraction gets the shared form."""
    graph, root = sharing()
    shared = Toy.two(Toy.one("f", Toy.leaf("shared")), Toy.one("g", Toy.leaf("shared")))
    assert extract.cheapest(graph, root, MODEL)[1] == Cost(13.5)
    with pytest.raises(extract.Dearer):
        extract.cheapest(graph, root, MODEL, checked=True)
    for form, price in (extract.extracted(graph, root, MODEL),
                        extract.cheapest(graph, root, MODEL, exact=True)):
        assert price == Cost(8.0)
        assert str(form) == str(shared)


def test_the_oracle_admits_a_greedy_choice_at_the_optimum():
    """Where greedy's choice is the cheapest DAG, the oracle passes it as it is."""
    graph = EGraph()
    root = Toy.two(Toy.leaf("small"), Toy.one("f", Toy.leaf("big")))
    graph.register(root)
    form, price = extract.cheapest(graph, root, MODEL, checked=True)
    assert price == Cost(17.5)
    assert str(form) == str(root)


def test_a_class_its_own_cheapest_node_reads_through_a_cycle_takes_a_node_that_closes_none():
    """`a` is a free `scaled(b)` or the leaf `fast`, `b` a free `scaled(a)` or the leaf `small`: the cheapest
    acyclic DAG of a free wrapper of `a` holds `scaled(small)` (6.5) or `fast` (3), never the cycle's two
    free nodes."""
    a, b = Toy.leaf("fast"), Toy.leaf("small")
    root = Toy.scaled(a, 0)
    graph = EGraph()
    graph.register(root, union(a).with_(Toy.scaled(b, 1)), union(b).with_(Toy.scaled(a, 2)))
    form, price = extract.extracted(graph, root, MODEL)
    assert price == Cost(3.0)
    assert str(form) == str(Toy.scaled(Toy.leaf("fast"), 0))


def test_an_infinitely_priced_node_is_avoided_where_a_dag_holds_none():
    """A root that is the infinitely priced `slow` or `f` of `big` (11) takes the finite form; one that is
    only `slow` is extracted at an infinite price, as a form no rule wrote is (`dew.algebra.derive`)."""
    graph = EGraph()
    root = Toy.leaf("slow")
    graph.register(union(root).with_(Toy.one("f", Toy.leaf("big"))))
    form, price = extract.extracted(graph, root, MODEL)
    assert price == Cost(11.0)
    assert str(form) == str(Toy.one("f", Toy.leaf("big")))
    alone = EGraph()
    only = Toy.leaf("slow")
    alone.register(only)
    assert math.isinf(extract.extracted(alone, only, MODEL)[1].seconds)


def test_a_later_level_never_buys_a_dearer_earlier_one():
    """`tied` costs 1.0 then 1.0, `untied` a hair more first and nothing then: compared in order, `tied`
    is cheaper, whatever the solver's tolerance admits of `untied`'s first level, and the oracle passes
    greedy's choice of it."""
    graph = EGraph()
    root = Toy.scaled(Toy.leaf("tied"), 0)
    graph.register(root, union(Toy.leaf("tied")).with_(Toy.leaf("untied")))
    for form, price in (extract.extracted(graph, root, MODEL),
                        extract.cheapest(graph, root, MODEL, checked=True)):
        assert price == Cost(1.0, 1.0)
        assert str(form) == str(root)


def test_a_root_the_graph_does_not_hold_yet_is_extracted_exactly():
    """A root first registered by its extraction is extracted as one already held is."""
    graph = EGraph()
    form, price = extract.extracted(graph, Toy.two(Toy.leaf("small"), Toy.leaf("big")), MODEL)
    assert price == Cost(16.5)
    assert str(form) == str(Toy.two(Toy.leaf("small"), Toy.leaf("big")))
