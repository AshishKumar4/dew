"""The price of a form (docs/design/algebra.md §6.5): each term's work in the one cost vocabulary
(`templates.cost.Work`, `work`), from its operands' shapes (`rules.shape`) and its parameters, turned into
seconds by a schedule over a GPU target's facts (`generation.GpuTarget`, their one owner): XLA's fusion for
the compile-time pass over a program (`fused`), or a template's tile walk for a form a template's algebra
takes (`walk`). Each shared value is priced once over the extracted DAG (egglog's `DagCostModel` under its
`greedy-dag` extractor)."""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable

import numpy as np
from egglog import DagCostModel, EGraph, get_callable_args, get_literal_value

from dew.algebra.rules import shape
from dew.algebra.terms import Region, Term
from dew.cache import ALGEBRA
from dew.nn.kernels import generation
from dew.nn.kernels.templates import cost as kernel_cost
from dew.nn.kernels.templates.cost import Work

EXPONENTIALS = frozenset({"exp", "exp2", "log", "log2", "log1p", "expm1", "tanh", "logistic", "sin", "cos",
                          "erf", "rsqrt", "sqrt", "pow"})
"""The language's functions the special-function units evaluate."""


def target(named: str | None = None) -> generation.GpuTarget:
    """The GPU a price is for: the one `named` names, else the algebra pass's (`program.passes`'s target),
    else the process's own (`generation.gpu_target`); a process of no GPU prices as the interpreted
    templates' generation (an H100's), as the gate lowers for."""
    named = named or ALGEBRA.value.target
    if named is None and generation.device_generation() not in generation.TARGETS:
        named = generation.INTERPRETED_GENERATION
    return generation.gpu_target(named)


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


def work(graph: EGraph, node, region: Region, bytes_per_element: int = 4) -> Work:
    """`node`'s work alone (the cost vocabulary, the module's account): a product's tensor-core FLOPs (its
    result's elements by its contraction's length, twice) and the bytes its operands are read in; a
    reduction's CUDA-core operations, one an element of its operand, and the bytes of the operand it reads;
    a conversion's conversions and a special function's exponentials, one an element of their result; any
    other elementwise node's CUDA-core operations; a layout change none."""
    if (args := get_callable_args(node, Term.reduce)) is not None:
        elements = made(graph, args[:1])
        return Work(operations=elements, device_bytes=elements * bytes_per_element)
    if (args := get_callable_args(node, Term.dot)) is not None:
        left, right = (shape_of(graph, operand) for operand in args[:2])
        if left is None or right is None:
            return Work()
        _, params = region.params[int(get_literal_value(args[2]))]
        (contracted, right_contracted), (batch, right_batch) = params["dimension_numbers"]
        length = math.prod(left[axis] for axis in contracted)
        # The result's elements: the batch, the left's free axes, the right's free axes.
        free = (math.prod(size for axis, size in enumerate(left) if axis not in (*contracted, *batch))
                * math.prod(size for axis, size in enumerate(right)
                            if axis not in (*right_contracted, *right_batch)))
        read = (math.prod(left) + math.prod(right)) * bytes_per_element
        return Work(flops=2.0 * math.prod(left[axis] for axis in batch) * free * length, device_bytes=read,
                    shared_bytes=read)
    if (args := get_callable_args(node, Term.convert)) is not None:
        return Work(conversions=made(graph, args[:1]))
    if (args := get_callable_args(node, Term.unary)) is not None:
        elements = made(graph, args[:1])
        special = get_literal_value(args[1]) in EXPONENTIALS
        return Work(exponentials=elements) if special else Work(operations=elements)
    for constructor, operands in ((Term.binary, (0, 2)), (Term.compare, (0, 2)), (Term.select, (0, 1, 2))):
        if (args := get_callable_args(node, constructor)) is not None:
            return Work(operations=made(graph, [args[index] for index in operands]))
    return Work()


@functools.total_ordering
@dataclasses.dataclass(frozen=True, eq=False)
class Seconds:
    """A form's work and the seconds its schedule gives it, as extraction adds and compares it (egglog's
    `ComparableAdd`): the work summed, compared by the seconds the schedule gives the sum, ties broken by
    the schedule's `tiebreak` where it states one (`Walk`'s: its units' seconds in series); two works of one
    key are equal, so every pair compares."""

    work: Work
    schedule: Callable[[Work], float] = dataclasses.field(repr=False)

    @property
    def value(self) -> float:
        return self.schedule(self.work)

    def __add__(self, other: Seconds) -> Seconds:
        return Seconds(self.work + other.work, self.schedule)

    def key(self) -> tuple[float, float]:
        tiebreak = getattr(self.schedule, "tiebreak", None)
        return self.value, 0.0 if tiebreak is None else tiebreak(self.work)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Seconds) and self.key() == other.key()

    def __lt__(self, other: Seconds) -> bool:
        return self.key() < other.key()

    def __hash__(self) -> int:
        return hash(self.key())


def fused(facts: generation.GpuTarget) -> Callable[[Work], float]:
    """XLA's schedule (the compile-time pass's): a reduction or a product reads its operands in one pass over
    device memory and elementwise work fuses into its consumer, costing its units' work alone, each at the
    whole device's rate, in series."""
    multiprocessors, clock, bandwidth = facts.multiprocessors, facts.clock_hz, facts.memory_bandwidth
    tensor, operations = facts.tensor_rate("16bit"), facts.tensor_rate("fp32")
    if not (multiprocessors and clock and bandwidth and tensor and operations):
        raise ValueError(f"{facts.device_kind}: no multiprocessors, clock, bandwidth or rates to price a "
                         "form by")
    each = multiprocessors * clock

    def seconds(done: Work) -> float:
        return (done.flops / (tensor * each) + (done.operations + done.conversions) / (operations * each)
                + done.exponentials / (kernel_cost.EXPONENTIALS_PER_CLOCK * each)
                + done.device_bytes / bandwidth)

    return seconds


@dataclasses.dataclass(frozen=True)
class Walk:
    """A template's tile walk (`templates.cost.walked`): a tile's work on its program, its units
    overlapped, its operands staged; its device bytes the launch's, which no form of its algebra changes.
    Two forms of one overlapped time are told apart by their units' seconds in series (`tiebreak`): the
    lighter leaves its units more of the step another form's work would take."""

    facts: generation.GpuTarget

    def __call__(self, done: Work) -> float:
        return kernel_cost.walked(dataclasses.replace(done, device_bytes=0.0), self.facts)

    def tiebreak(self, done: Work) -> float:
        return fused(self.facts)(dataclasses.replace(done, device_bytes=0.0))


def walk(facts: generation.GpuTarget) -> Walk:
    """The template's tile walk's schedule over `facts` (`Walk`)."""
    return Walk(facts)


def model(region: Region, schedule: Callable[[Work], float], bytes_per_element: int = 4) -> DagCostModel:
    """Each node's work (`work`) under `schedule`; extraction adds them, once a shared node. A node is priced
    from its operands' shapes, the e-classes the extractor hands its callback, and its parameters
    (`region`'s)."""
    def marginal(graph: EGraph, node) -> Seconds:
        return Seconds(work(graph, node, region, bytes_per_element), schedule)

    return DagCostModel(marginal_cost=marginal, identity=Seconds(Work(), schedule))


RISK = {"log": 1.0, "sub": 1e-3}
"""What a node of the stable proposal (`stable`) costs beside its time, in seconds: a logarithm whose value
is rounded into what follows it, a difference that may cancel its operands' digits. A proposal, not a choice:
dev time measures every candidate's error (`dew.algebra.error.choose`)."""


def stable(region: Region, schedule: Callable[[Work], float], bytes_per_element: int = 4) -> DagCostModel:
    """The program's model (`model`) with each node of a rounding risk priced up (`RISK`), so its cheapest
    forms are the ones the stable rules wrote: the candidate dev time proposes for its accuracy."""
    def marginal(graph: EGraph, node) -> Seconds:
        done = work(graph, node, region, bytes_per_element)
        for constructor in (Term.unary, Term.binary):
            if (args := get_callable_args(node, constructor)) is not None:
                # Each operation's name follows its first operand.
                name = get_literal_value(args[1])
                risk = RISK.get(name if isinstance(name, str) else "", 0.0)
                # A risk is a time the schedule adds: as many CUDA-core operations as take it on one.
                done = done + Work(operations=risk * _per_second(schedule))
        return Seconds(done, schedule)

    return DagCostModel(marginal_cost=marginal, identity=Seconds(Work(), schedule))


def _per_second(schedule: Callable[[Work], float]) -> float:
    """The CUDA-core operations `schedule` takes a second over."""
    return 1.0 / schedule(Work(operations=1.0))
