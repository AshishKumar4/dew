"""The price of a form (docs/design/algebra.md §6.5): each term's work in the one cost vocabulary
(`templates.cost.Work`, `work`), from its operands' shapes (`rules.shape`) and its parameters, turned into
seconds by a schedule over a GPU target's facts (`generation.GpuTarget`, their one owner): XLA's fusion for
the compile-time pass over a program (`fused`), or a template's tile walk for a form a template's algebra
takes (`walk`). Each shared value is priced once over the extracted DAG (egglog's `DagCostModel`, extracted
exactly: `dew.algebra.extract`)."""

from __future__ import annotations

import dataclasses
import functools
import math
import re
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from egglog import DagCostModel, EGraph, get_callable_args

from dew.algebra.extract import Level
from dew.algebra.rules import CONTRACTED, shape
from dew.algebra.terms import Region, Term, integer, key, text
from dew.cache import ALGEBRA
from dew.nn.kernels import generation
from dew.nn.kernels.templates import cost as kernel_cost
from dew.nn.kernels.templates.cost import Work
from dew.nn.kernels.templates.product import EIGHT
from dew.nn.precision import asks_default_precision, precision_names

EXPONENTIALS = frozenset({"exp", "exp2", "log", "log2", "log1p", "expm1", "tanh", "logistic", "sin", "cos",
                          "erf", "rsqrt", "sqrt", "pow"})
"""The language's functions the special-function units evaluate."""


def target(named: str | None = None) -> generation.GpuTarget:
    """The device a price is for: the one `named` names, else the algebra pass's (`program.passes`'s
    target), else the process's own or the one it assumes (`generation.gpu_target`: a GPU no row names its
    family's and its runtime's facts); a process of no accelerator prices as the interpreted templates'
    generation (an H100's), as the gate lowers for."""
    named = named or ALGEBRA.value.target
    if named is None and generation.backend() not in ("gpu", "tpu"):
        named = generation.INTERPRETED_GENERATION
    return generation.gpu_target(named)


def shape_of(graph: EGraph, term, held: dict | None = None) -> tuple[int, ...] | None:
    """The shape of the value `term`, an e-class the extractor supplied, stands for (`rules.shape`), kept in
    `held` (by `terms.key`) where given: an extraction asks each e-class's shape of every node reading it,
    and a lookup in the e-graph is egglog's own program, a millisecond each (46 s of a gemma3n training
    step's pass)."""
    named = key(term)
    if held is not None and named in held:
        return held[named]
    known = graph.lookup_function_value(shape(term))
    found = None if known is None else tuple(integer(size) for size in known.value)
    if held is not None:
        held[named] = found
    return found


def made(graph: EGraph, operands, held: dict | None = None) -> int:
    """The elements an elementwise node makes of `operands` (e-classes the extractor supplied), their shapes
    (`shape_of`, kept in `held`) broadcast together; 0 where one has none known."""
    shapes = [shape_of(graph, operand, held) for operand in operands]
    if any(known is None for known in shapes):
        return 0
    return math.prod(np.broadcast_shapes(*(known for known in shapes if known is not None)))


def work(graph: EGraph, node, region: Region, held: dict | None = None) -> Work:
    """`node`'s work alone (the cost vocabulary, the module's account): an operand's (a leaf's) bytes, at its
    dtype's element size, read from device memory once however many values read it (extraction prices a
    shared node once), and nothing else of device memory, as XLA's fusion reads a program's inputs once (a
    sibling reduction's too) and keeps its fused intermediates on chip; a product's tensor-core FLOPs (its
    result's elements by its contraction's length, twice) and the bytes its operands are staged in (the tile
    walk's, at their dtypes' sizes); a reduction's
    CUDA-core operations, one an element of its operand; a reduction and a product each a kernel of XLA's
    fusion, its root (elementwise work fuses into it); a conversion's conversions and a special
    function's exponentials, one an element of their result; any other elementwise node's CUDA-core
    operations; a layout change none."""
    if (args := get_callable_args(node, Term.leaf)) is not None:
        # The leaf's own shape from its region (a callback may read no table at a node it was not given).
        leaf = key(Term.leaf(args[0]))
        extent = region.extents.get(leaf)
        return Work() if extent is None else Work(
            device_bytes=math.prod(extent) * np.dtype(region.dtypes[leaf]).itemsize)
    if (args := get_callable_args(node, Term.reduce)) is not None:
        return Work(operations=made(graph, args[:1], held), kernels=1.0)
    if (args := get_callable_args(node, Term.dot)) is not None:
        left, right = (shape_of(graph, operand, held) for operand in args[:2])
        if left is None or right is None:
            return Work()
        index = integer(args[2])
        _, params = region.params[index]
        (contracted, right_contracted), (batch, right_batch) = params["dimension_numbers"]
        length = math.prod(left[axis] for axis in contracted)
        # The result's elements: the batch, the left's free axes, the right's free axes.
        free = (math.prod(size for axis, size in enumerate(left) if axis not in (*contracted, *batch))
                * math.prod(size for axis, size in enumerate(right)
                            if axis not in (*right_contracted, *right_batch)))
        operands = region.products[index]
        read = sum(math.prod(extent) * np.dtype(dtype).itemsize
                   for extent, dtype in zip((left, right), operands, strict=True))
        flops = 2.0 * math.prod(left[axis] for axis in batch) * free * length
        mode, passes = _mode(params.get("precision"), set(operands))
        return Work(shared_bytes=read, kernels=1.0, **{_MODES[mode]: flops * passes})
    if (args := get_callable_args(node, Term.convert)) is not None:
        return Work(conversions=made(graph, args[:1], held))
    if (args := get_callable_args(node, Term.unary)) is not None:
        elements = made(graph, args[:1], held)
        special = text(args[1]) in EXPONENTIALS
        return Work(exponentials=elements) if special else Work(operations=elements)
    if (args := get_callable_args(node, Term.binary)) is not None and text(args[1]) == CONTRACTED["mul"]:
        # Its reader's fused multiply-add computes it (`rules._contracted`).
        return Work()
    for constructor, operands in ((Term.binary, (0, 2)), (Term.compare, (0, 2)), (Term.select, (0, 1, 2))):
        if (args := get_callable_args(node, constructor)) is not None:
            return Work(operations=made(graph, [args[index] for index in operands], held))
    return Work()


_MODES = {"16bit": "flops", "tf32": "tf32_flops", "8bit": "fp8_flops", "fp32": "operations"}
"""The `Work` field a product's mode's FLOPs are told in (`generation.Architecture`'s modes; IEEE fp32 is
the CUDA cores')."""

_ALGORITHM = re.compile(r"(BF16|F16|TF32)_\1_F32(?:_X(\d))?")
"""A dot algorithm preset of a tensor-core mode, its passes (`BF16_BF16_F32_X3`: three 16-bit products)."""


def _mode(precision, dtypes: set) -> tuple[str, int]:
    """The mode a product of operands of `dtypes` at `precision` multiplies in, and its passes: its dot
    algorithm's where it names one of the tensor cores' (its passes, an X3's three), by its preset's name
    or, a `jax.lax.DotAlgorithm` (which JAX names no preset), by its fields (16-bit precision types, its
    primitive operations), as JAX lowers both to one algorithm; 16-bit operands' 16-bit mode, FP8 operands'
    8-bit mode (`product.EIGHT`), fp32 operands' TF32 at the default precision (the product declaration's,
    `product.declared`) and IEEE fp32 above it; any other operand's on the CUDA cores."""
    # A dot algorithm is a named tuple itself: one, not a pair of precisions.
    written = precision if isinstance(precision, tuple) and not isinstance(
        precision, jax.lax.DotAlgorithm) else (precision,)
    for algorithm in written:
        if isinstance(algorithm, jax.lax.DotAlgorithm):
            narrow = {jnp.dtype(algorithm.lhs_precision_type), jnp.dtype(algorithm.rhs_precision_type)}
            if narrow <= {jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)}:
                return "16bit", algorithm.num_primitive_operations
            if narrow <= set(EIGHT):
                return "8bit", algorithm.num_primitive_operations
            return "fp32", algorithm.num_primitive_operations
    for name in precision_names(precision):
        if (found := _ALGORITHM.fullmatch(name)) is not None:
            return ("tf32" if found.group(1) == "TF32" else "16bit"), int(found.group(2) or 1)
    if dtypes and all(dtype in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)) for dtype in dtypes):
        return "16bit", 1
    if dtypes and all(dtype in EIGHT for dtype in dtypes):
        return "8bit", 1
    if dtypes == {jnp.dtype(jnp.float32)} and asks_default_precision(precision, configured=True):
        return "tf32", 1
    return "fp32", 1


def product_work(rows: int, depth: int, columns: int, *, count: int = 1, dtype=jnp.float32) -> Work:
    """`count` products of `[rows, depth] @ [depth, columns]` operands of `dtype` at the default precision,
    in their mode (`_mode`), each a kernel of XLA's fusion, its root (`work`'s products): a chunk form's
    products (`dew.algebra.linear_scan`), Dew's fp32 products."""
    mode, passes = _mode(None, {jnp.dtype(dtype)})
    return Work(kernels=float(count), **{_MODES[mode]: 2.0 * rows * depth * columns * count * passes})


@dataclasses.dataclass(frozen=True, eq=False)
class Seconds:
    """A form's work and the seconds its schedule gives it, as extraction adds and compares it
    (`extract.Linear`): the work summed, compared by the seconds the schedule gives the sum and the `extra`
    seconds beside it, which no rate prices (a reading the derivation left, infinite, `derive`; a stable
    proposal's risk, `RISK`), ties broken by the schedule's `tiebreak` (`Walk`'s: its units' seconds in
    series; `Fused`'s none), its `key`, taken where it is compared (extraction adds far more than it
    compares); two works of one key are equal, so every pair compares."""

    work: Work
    schedule: Schedule = dataclasses.field(repr=False)
    extra: float = 0.0

    @functools.cached_property
    def key(self) -> tuple[float, float]:
        return self.schedule(self.work) + self.extra, self.schedule.tiebreak(self.work)

    @property
    def value(self) -> float:
        return self.key[0]

    def __add__(self, other: Seconds) -> Seconds:
        return Seconds(self.work + other.work, self.schedule, self.extra + other.extra)

    def levels(self) -> tuple[Level, ...]:
        """Its seconds and its tie, as extraction composes them node by node (`extract.Linear`): the
        schedule's pieces and span (`Schedule.pieces`, `spanned`) beside its `extra`, then its tiebreak."""
        return (Level(self.schedule.pieces(self.work), self.schedule.spanned(self.work), self.extra),
                Level((self.schedule.tiebreak(self.work),)))

    def __eq__(self, other) -> bool:
        return isinstance(other, Seconds) and self.key == other.key

    def __lt__(self, other: Seconds) -> bool:
        return self.key < other.key

    def __le__(self, other: Seconds) -> bool:
        return self.key <= other.key

    def __gt__(self, other: Seconds) -> bool:
        return self.key > other.key

    def __ge__(self, other: Seconds) -> bool:
        return self.key >= other.key

    def __hash__(self) -> int:
        return hash(self.key)


class Schedule(Protocol):
    """A schedule (the module's account): the seconds it gives a work, the larger of its `pieces` and its
    span's (`spanned`), each piece linear in the work, which extraction sums node by node (`extract.Level`);
    what tells two works of one time apart, linear too; and the work it gives no time, at a rate its target
    states none of (`templates.cost.units`)."""

    def pieces(self, done: Work) -> tuple[float, ...]: ...

    def spanned(self, done: Work) -> float: ...

    def __call__(self, done: Work) -> float: ...

    def tiebreak(self, done: Work) -> float: ...

    def unpriced(self, done: Work) -> tuple[float, ...]: ...


def cheaper(form: Seconds, than: Seconds) -> bool:
    """Whether `form` is cheaper than `than` on every device its schedule's facts describe: fewer seconds,
    and no more work at a rate its target states none of (a TPU's vector and transcendental units), which
    then costs no more whatever that rate is."""
    pairs = zip(form.schedule.unpriced(form.work), than.schedule.unpriced(than.work), strict=True)
    return form.value < than.value and all(mine <= theirs for mine, theirs in pairs)


@dataclasses.dataclass(frozen=True)
class Fused:
    """XLA's schedule (the compile-time pass's): a reduction or a product reads its operands in one pass over
    device memory and elementwise work fuses into its consumer, costing its units' work, each at the whole
    device's rate (`templates.cost.rates`), in series, and each kernel the part's fixed latency (its launch
    and first byte, `Measured.fixed`; row 4081: a small fusion's time is that latency, 0.02 of it priced
    without), or its span at a loop iteration's latency where longer (the work-span model,
    `templates.cost.spanned`); no tie to break."""

    facts: generation.GpuTarget

    def pieces(self, done: Work) -> tuple[float, ...]:
        fixed = self.facts.fixed_seconds
        return (sum(kernel_cost.units(done, kernel_cost.rates(self.facts, device=True))[0])
                + (0.0 if fixed is None else done.kernels * fixed),)

    def spanned(self, done: Work) -> float:
        return kernel_cost.spanned(done, kernel_cost.step_latency(self.facts, device=True))[0]

    def __call__(self, done: Work) -> float:
        return max(*self.pieces(done), self.spanned(done))

    def tiebreak(self, done: Work) -> float:
        del done
        return 0.0

    def unpriced(self, done: Work) -> tuple[float, ...]:
        latency, fixed = kernel_cost.step_latency(self.facts, device=True), self.facts.fixed_seconds
        return (*kernel_cost.units(done, kernel_cost.rates(self.facts, device=True))[1],
                kernel_cost.spanned(done, latency)[1], done.kernels if fixed is None else 0.0)


def fused(facts: generation.GpuTarget) -> Fused:
    """XLA's schedule over `facts` (`Fused`)."""
    return Fused(facts)


@dataclasses.dataclass(frozen=True)
class Walk:
    """A template's tile walk (`templates.cost.walked`): a tile's work on its program, its units
    overlapped, its operands staged; its device bytes the launch's, which no form of its algebra changes.
    Two forms of one overlapped time are told apart by their units' seconds in series (`tiebreak`): the
    lighter leaves its units more of the step another form's work would take."""

    facts: generation.GpuTarget

    def pieces(self, done: Work) -> tuple[float, ...]:
        return tuple(kernel_cost.units(done, kernel_cost.rates(self.facts, device=False))[0])

    def spanned(self, done: Work) -> float:
        return kernel_cost.spanned(done, kernel_cost.step_latency(self.facts, device=False))[0]

    def __call__(self, done: Work) -> float:
        return kernel_cost.walked(done, self.facts)

    def tiebreak(self, done: Work) -> float:
        return sum(kernel_cost.units(done, kernel_cost.rates(self.facts, device=False))[0])

    def unpriced(self, done: Work) -> tuple[float, ...]:
        latency = kernel_cost.step_latency(self.facts, device=False)
        return (*kernel_cost.units(done, kernel_cost.rates(self.facts, device=False))[1],
                kernel_cost.spanned(done, latency)[1])


def walk(facts: generation.GpuTarget) -> Walk:
    """The template's tile walk's schedule over `facts` (`Walk`)."""
    return Walk(facts)


def model(region: Region, schedule: Schedule) -> DagCostModel[Seconds]:
    """Each node's work (`work`) under `schedule`; extraction adds them, once a shared node. A node is priced
    from its operands' shapes, the e-classes the extractor hands its callback, and its parameters
    (`region`'s)."""
    held: dict = {}

    def marginal(graph: EGraph, node) -> Seconds:
        return Seconds(work(graph, node, region, held), schedule)

    return DagCostModel(marginal_cost=marginal, identity=Seconds(Work(), schedule))


PRIMITIVES = {contracted: name for name, contracted in CONTRACTED.items()}
"""The primitive each of a fused multiply-add's names binds (`rules.CONTRACTED`)."""

RISK = {"log": 1.0, "sub": 1e-3}
"""What a node of the stable proposal (`stable`) costs beside its time, in seconds: a logarithm whose value
is rounded into what follows it, a difference that may cancel its operands' digits. A proposal, not a choice:
dev time measures every candidate's error (`dew.algebra.error.choose`)."""


def stable(region: Region, schedule: Schedule) -> DagCostModel[Seconds]:
    """The program's model (`model`) with each node of a rounding risk priced up (`RISK`), so its cheapest
    forms are the ones the stable rules wrote: the candidate dev time proposes for its accuracy."""
    held: dict = {}

    def marginal(graph: EGraph, node) -> Seconds:
        done = work(graph, node, region, held)
        for constructor in (Term.unary, Term.binary):
            if (args := get_callable_args(node, constructor)) is not None:
                # Each operation's name follows its first operand; a fused multiply-add's is its primitive's.
                risk = RISK.get(PRIMITIVES.get(text(args[1]), text(args[1])), 0.0)
                return Seconds(done, schedule, risk)
        return Seconds(done, schedule)

    return DagCostModel(marginal_cost=marginal, identity=Seconds(Work(), schedule))
