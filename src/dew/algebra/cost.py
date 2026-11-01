"""The price of a form (docs/design/algebra.md §6.5): its time on a device, from the device's facts, each
shared value priced once over the extracted DAG (egglog's `DagCostModel` under its `greedy-dag` extractor).

A reduction reads its operand in one pass over memory; elementwise work fuses into its consumer, as XLA fuses
it, and costs its arithmetic alone, over the elements of the value it makes, from its operands' shapes
(`rules.shape`)."""

from __future__ import annotations

import dataclasses
import math

import jax
import numpy as np
from egglog import DagCostModel, EGraph, get_callable_args, get_literal_value

from dew.algebra.rules import shape
from dew.algebra.terms import Term


@dataclasses.dataclass(frozen=True)
class Device:
    """The facts a price reads: memory bandwidth (bytes a second) and arithmetic rate (operations a
    second)."""

    memory_bandwidth: float
    arithmetic_rate: float

    @classmethod
    def of(cls, device=None) -> Device:
        """A device's published facts by its kind: an H100 SXM's 3.35 TB/s and 67 TFLOP/s of fp32 vector
        arithmetic, an A100 SXM's 2.04 TB/s and 19.5 TFLOP/s; any other device takes the H100's, whose ratio
        of one to the other is all extraction compares."""
        kind = (device if device is not None else jax.devices()[0]).device_kind
        if "A100" in kind:
            return cls(2.04e12, 1.95e13)
        return cls(3.35e12, 6.7e13)


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
