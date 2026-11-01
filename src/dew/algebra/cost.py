"""The price of a form (docs/design/algebra.md §6.5): its time on a device, from the device's facts, each
shared value priced once over the extracted DAG (egglog's `DagCostModel` under its `greedy-dag` extractor).

A reduction reads its operand in one pass over memory; elementwise work fuses into its consumer, as XLA fuses
it, and costs its arithmetic alone, over the elements of the value it makes, from its operands' shapes
(`rules.shape`)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from egglog import DagCostModel, EGraph, get_callable_args, get_literal_value

from dew.algebra.rules import shape
from dew.algebra.terms import Term
from dew.cache import ALGEBRA
from dew.nn.kernels import generation


@dataclasses.dataclass(frozen=True)
class Device:
    """The facts a price reads, a GPU target's (`generation.GpuTarget`, their one owner): its memory bandwidth
    (bytes a second) and its fp32 arithmetic rate (operations a second: its CUDA-core lanes' FMAs over its
    multiprocessors and clock)."""

    memory_bandwidth: float
    arithmetic_rate: float

    @classmethod
    def of(cls, target: str | None = None) -> Device:
        """The facts of the target `target` names, else the algebra pass's (`program.passes`'s target), else
        the process's own GPU's (`generation.gpu_target`); a process of no GPU prices as the interpreted
        templates' generation (an H100's), as the gate lowers for."""
        named = target or ALGEBRA.value.target
        if named is None and generation.device_generation() not in generation.TARGETS:
            named = generation.INTERPRETED_GENERATION
        facts = generation.gpu_target(named)
        rate = facts.tensor_rate("fp32")
        if facts.memory_bandwidth is None or rate is None or not (facts.multiprocessors and facts.clock_hz):
            raise ValueError(f"{facts.device_kind}: no memory bandwidth or fp32 rate to price a form by")
        return cls(facts.memory_bandwidth, rate * facts.multiprocessors * facts.clock_hz)


def shape_of(graph: EGraph, term) -> tuple[int, ...] | None:
    """The shape of the value `term`, an e-class the extractor supplied, stands for (`rules.shape`)."""
    known = graph.lookup_function_value(shape(term))
    return None if known is None else tuple(int(get_literal_value(size)) for size in known.value)


def made(graph: EGraph, operands) -> int:
    """The elements an elementwise node makes of `operands` (e-classes the extractor supplied), their shapes
    broadcast together; 0 where one has none known."""
    shapes = [shape_of(graph, operand) for operand in operands]
    if any(known is None for known in shapes):
        return 0
    return math.prod(np.broadcast_shapes(*(known for known in shapes if known is not None)))


@dataclasses.dataclass(frozen=True, order=True)
class Seconds:
    """A form's time, in seconds, as extraction adds and compares it (egglog's `ComparableAdd`)."""

    value: float

    def __add__(self, other: Seconds) -> Seconds:
        return Seconds(self.value + other.value)


def model(device: Device, bytes_per_element: int = 4) -> DagCostModel:
    """The cost of each node alone, in seconds: a reduction its operand's bytes over memory bandwidth, a
    product its operands' bytes, an elementwise node its result's elements over the arithmetic rate, a
    layout change none (XLA lays it out
    within its consumer's fusion); extraction adds them, once a shared node. A node is priced from its
    operands' shapes, the e-classes the extractor hands its callback."""
    def marginal(graph: EGraph, node) -> Seconds:
        if (args := get_callable_args(node, Term.reduce)) is not None:
            return Seconds(made(graph, args[:1]) * bytes_per_element / device.memory_bandwidth)
        if (args := get_callable_args(node, Term.dot)) is not None:
            # A product reads both its operands once.
            read = sum(made(graph, [operand]) for operand in args[:2])
            return Seconds(read * bytes_per_element / device.memory_bandwidth)
        for constructor, operands in ((Term.binary, (0, 2)), (Term.compare, (0, 2)), (Term.select, (0, 1, 2)),
                                      (Term.unary, (0,)), (Term.convert, (0,))):
            if (args := get_callable_args(node, constructor)) is not None:
                return Seconds(made(graph, [args[index] for index in operands]) / device.arithmetic_rate)
        return Seconds(0.0)

    return DagCostModel(marginal_cost=marginal, identity=Seconds(0.0))


RISK = {"log": 1.0, "sub": 1e-3}
"""What a node of the stable proposal (`stable`) costs beside its time: a logarithm whose value is rounded
into what follows it, a difference that may cancel its operands' digits. A proposal, not a choice: dev time
measures every candidate's error (`dew.algebra.error.choose`)."""


def stable(device: Device, bytes_per_element: int = 4) -> DagCostModel:
    """The program model (`model`) with each node of a rounding risk priced up (`RISK`), so its cheapest
    forms are the ones the stable rules wrote: the candidate dev time proposes for its accuracy."""
    timed = model(device, bytes_per_element)

    def marginal(graph: EGraph, node) -> Seconds:
        for constructor in (Term.unary, Term.binary):
            if (args := get_callable_args(node, constructor)) is not None:
                # Each operation's name follows its first operand.
                name = get_literal_value(args[1])
                risk = RISK.get(name if isinstance(name, str) else "", 0.0)
                return timed.marginal_cost(graph, node) + Seconds(risk)
        return timed.marginal_cost(graph, node)

    return DagCostModel(marginal_cost=marginal, identity=Seconds(0.0))
