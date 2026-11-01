"""The price of a form (docs/design/algebra.md §6.5): its time on a device, from the device's facts, each
shared value priced once over the extracted DAG (egglog's `DagCostModel` under its `greedy-dag` extractor).

A reduction reads its operand in one pass over memory; elementwise work fuses into its consumer, as XLA fuses
it, and costs its arithmetic alone, over the elements of the value it makes (`rules.shape`)."""

from __future__ import annotations

import dataclasses
import math

import jax
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


def elements(graph: EGraph, term) -> int:
    """The elements of the value `term` stands for, from its shape (`rules.shape`); 0 where none is known."""
    known = graph.lookup_function_value(shape(term))
    return 0 if known is None else math.prod(int(get_literal_value(size)) for size in known.value)


def model(device: Device, bytes_per_element: int = 4) -> DagCostModel:
    """The cost of each node alone, in seconds: a reduction its operand's bytes over memory bandwidth, every
    other computed node its result's elements over the arithmetic rate; extraction adds them, once a shared
    node."""
    def marginal(graph: EGraph, node) -> float:
        if (args := get_callable_args(node, Term.reduce)) is not None:
            return elements(graph, args[0]) * bytes_per_element / device.memory_bandwidth
        if any(get_callable_args(node, constructor) is not None
               for constructor in (Term.binary, Term.compare, Term.unary, Term.convert, Term.shaped,
                                   Term.select)):
            return elements(graph, node) / device.arithmetic_rate
        return 0.0

    return DagCostModel(marginal_cost=marginal, identity=0.0)
