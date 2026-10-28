"""A kernel's description of the values it rounds and holds, which the float64 gate holds it by
(tests/reference_error.py `assert_graph_as_exact`).

A kernel is checked where it rounds. Each `Node` is one value the kernel rounds and holds (an
accumulator, a stored max, a cast), computed from its parents' values by the kernel's own arithmetic:
a template's stage by construction, or a native probe of a hand kernel with its parents overridden.
A cut never falls inside a fused multiply-add. `Graph.fused` is the kernel as a caller runs it, which
the graph's replay must equal bit for bit.

A node's kind says how its own rounding is redrawn:

- `Sum(axes)`: a genuine sum or sum-product of fixed operands (a dot, p·v, a GEMM accumulator, a row's
  sum of squares); permuting the summed axis of its operands (`axes`, each parent's) is an exact
  symmetry of the node, and the node's orbit;
- `Online(axes)`: an online reduction that rescales its accumulator (flash attention's running max and
  sum over key tiles), whose keys, permuted with their mask, are an exact symmetry of its math;
- `Ordered(axes, follows)`: a recurrence whose terms do not commute; its spread is the reference
  ensemble's, over an orbit of the axes that are an exact symmetry of it where it has one, else once;
- `Pointwise()`: no sum; checked once, element by element.

A node's `sentinel` is the one non-finite value its math defines (attention's empty row holds a max of
-inf): where Dew and float64 both hold it, the element's error is zero. Any other non-finite value,
and any NaN, fails.
"""

import dataclasses
from collections.abc import Callable, Mapping


@dataclasses.dataclass(frozen=True)
class Sum:
    """A sum over one axis of some parents, given as a parent's name to its summed axis. `follows`
    names the output axes the permutation carries, each output axis to the parent whose permuted
    axis it is (a dot's output over a width it does not sum, a state [K, V] over the permuted K):
    the gate maps those back before it compares an element across orders."""
    axes: Mapping[str, int]
    follows: Mapping[int, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Online:
    """An online reduction over one axis of some parents (keys, their values and their mask), and
    the output axes the permutation carries (`Sum.follows`)."""
    axes: Mapping[str, int]
    follows: Mapping[int, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Ordered:
    """A reduction whose terms do not commute. Where an axis of its parents can be permuted as an
    exact symmetry of the whole node (a recurrence's key width), `axes` and `follows` give that orbit
    as `Sum`'s do; the sums it does not reorder keep one order."""
    axes: Mapping[str, int] = dataclasses.field(default_factory=dict)
    follows: Mapping[int, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Pointwise:
    """No sum: arithmetic element by element."""


@dataclasses.dataclass(frozen=True)
class Node:
    """One value the kernel rounds and holds: its name, its parents' names, its kind, `fn` computing it
    from the parents' values (keyword arguments by name), its sentinel (`sentinel`) if its math defines
    one, and, on a reference's `Ordered` node, `members`: other order-preserving ways the reference
    computes the same value (another chunk size of a scan), its ensemble there."""
    name: str
    parents: tuple[str, ...]
    kind: Sum | Online | Ordered | Pointwise
    fn: Callable
    sentinel: float | None = None
    members: Mapping[str, Callable] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Graph:
    """A computation's inputs, its nodes in an order each node's parents come before it, its outputs
    (nodes' names), and `fused`, the native witness: the production path from the inputs (keyword
    arguments) to its outputs and every intermediate and operand it holds that the graph names, by
    name, which the graph's replay must equal bit for bit. A backward is a graph of its own, its
    inputs the common forward's saved values and a fixed cotangent."""
    inputs: tuple[str, ...]
    nodes: tuple[Node, ...]
    outputs: tuple[str, ...]
    fused: Callable

    def node(self, name: str) -> Node:
        return next(node for node in self.nodes if node.name == name)

    def replay(self, inputs: Mapping[str, object]) -> Mapping[str, object]:
        """Every value of the graph, its nodes run in turn from `inputs`."""
        values = dict(inputs)
        for node in self.nodes:
            values[node.name] = node.fn(**{parent: values[parent] for parent in node.parents})
        return values
