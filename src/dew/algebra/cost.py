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
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from egglog import DagCostModel, EGraph, get_callable_args

from dew._cache import ALGEBRA
from dew.algebra.extract import Level
from dew.algebra.rules import CONTRACTED, dtype as dtype_fact, shape
from dew.algebra.terms import RESCALE, Region, Term, integer, key, text
from dew.nn import programs
from dew.nn.kernels import generation
from dew.nn.kernels.templates import cost as kernel_cost
from dew.nn.kernels.templates.cost import Work
from dew.nn.kernels.templates.product import product_mode


def target(named: str | None = None) -> generation.GpuTarget:
    """The device a price is for: the one `named` names, else the algebra pass's (`program.passes`'s
    target), else the process's own or the one it assumes (`generation.gpu_target`: a GPU no row names its
    family's and its runtime's facts); a process of no accelerator prices as the interpreted templates'
    target (`generation.interpreted_target`: an H100's, or the device an interpreted check assumes), as the
    gate lowers for."""
    named = named or ALGEBRA.value.target
    if named is None and generation.backend() not in ("gpu", "tpu"):
        return generation.interpreted_target()
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


def dtype_of(graph: EGraph, term, held: dict | None = None) -> np.dtype | None:
    """The scalar dtype of `term`, an e-class the extractor supplied (`rules.dtype`), kept in `held` where
    given (row 12974)."""
    named = ("dtype", key(term))
    if held is not None and named in held:
        return held[named]
    known = graph.lookup_function_value(dtype_fact(term))
    found = None if known is None else jnp.dtype(text(known))
    if held is not None:
        held[named] = found
    return found


VALUE = jnp.dtype(jnp.float32)
"""The dtype a term's value is priced at where the e-graph holds none (a leaf's and a product's are their
own): a vector lane's fp32, as `LANE_BYTES` keeps it."""


def _shapes(graph: EGraph, operands, held: dict | None) -> list[tuple[int, ...]] | None:
    """The shapes of `operands`, e-classes the extractor supplied (`shape_of`, kept in `held`); None where one
    is unknown."""
    shapes = [shape_of(graph, operand, held) for operand in operands]
    known = [shape for shape in shapes if shape is not None]
    return known if len(known) == len(shapes) else None


def _operation_work(primitive, inputs, outputs, params, on: generation.GpuTarget | None = None,
                    fp32: bool | None = None) -> Work:
    """`emission.operation_work`, the kernels' counter's one rule, imported where a price first asks: the
    emission's module loads the templates' resources, which no process reading the algebra alone imports (the
    test cache keys a row on each module it loads)."""
    from dew.nn.kernels.templates import emission

    return emission.operation_work(primitive, inputs, outputs, params, on, fp32=fp32)


def _priced(primitive, params, shapes: list[tuple[int, ...]], made: tuple[int, ...],
            on: generation.GpuTarget | None = None, dtype=None, *,
            in_dtypes: list[np.dtype | None] | None = None) -> Work:
    """Price the traced widths, without borrowing the synthetic vector dtype as evidence of fp32 work."""
    out_dt = VALUE if dtype is None else dtype
    known = (None,) * len(shapes) if in_dtypes is None else in_dtypes
    inputs = [jax.ShapeDtypeStruct(s, out_dt if dt is None else dt)
              for s, dt in zip(shapes, known, strict=True)]
    return _operation_work(primitive, inputs, [jax.ShapeDtypeStruct(made, out_dt)], params, on,
                           fp32=dtype == VALUE)


def _result_dtype(primitive, params, in_dtypes: list[np.dtype | None],
                  *, compared: bool = False) -> np.dtype | None:
    """The output dtype of an elementwise node from its children's e-graph dtypes and `params`."""
    if primitive is jax.lax.convert_element_type_p and "new_dtype" in params:
        return jnp.dtype(params["new_dtype"])
    if compared or primitive is jax.lax.is_finite_p:
        return jnp.dtype(jnp.bool_)
    if (out_dt := params.get("out_dtype")) is not None:
        return jnp.dtype(out_dt)
    if primitive is jax.lax.select_n_p:
        vals = in_dtypes[1:]
        return vals[0] if len(vals) == 2 and vals[0] == vals[1] else (
            jnp.result_type(*vals) if vals and all(d is not None for d in vals) else None)
    if primitive is jax.lax.clamp_p:
        return in_dtypes[1]
    if not in_dtypes:
        return None
    if len(in_dtypes) == 1 or all(d == in_dtypes[0] for d in in_dtypes):
        return in_dtypes[0]
    return jnp.result_type(*in_dtypes) if all(d is not None for d in in_dtypes) else None


def _elementwise(graph: EGraph, primitive, params, operands, held: dict | None,
                 on: generation.GpuTarget | None = None, *, compared: bool = False) -> Work:
    """An elementwise node's arithmetic (`_priced` on `on`), reading real operand dtypes from the e-graph
    (`dtype_of`), its result its operands' shapes broadcast; nothing where one shape is unknown."""
    shapes = _shapes(graph, operands, held)
    if shapes is None:
        return Work()
    in_dtypes = [dtype_of(graph, operand, held) for operand in operands]
    out_dt = _result_dtype(primitive, params, in_dtypes, compared=compared)
    return _priced(primitive, params, shapes, np.broadcast_shapes(*shapes), on, out_dt, in_dtypes=in_dtypes)


def _product(left: tuple[int, ...], right: tuple[int, ...], params, product,
             on: generation.GpuTarget | None = None) -> tuple[Work, int]:
    """A product's arithmetic as the kernels' counter prices it (`_operation_work`: its FLOPs in the mode
    and passes it multiplies in on `on`, else the price's target, `target`), and its result's elements: the
    batch, the left's free axes, the right's free axes, its operands at their dtypes and its result at the
    one it makes (`terms.Product.made`, JAX's own rule: a bfloat16 by a float16 product is bfloat16, which
    NumPy's promotion refused, base review 0b8b7ed98 F1)."""
    (contracted, right_contracted), (batch, right_batch) = params["dimension_numbers"]
    extents = (*(left[axis] for axis in batch),
               *(size for axis, size in enumerate(left) if axis not in (*contracted, *batch)),
               *(size for axis, size in enumerate(right) if axis not in (*right_contracted, *right_batch)))
    inputs = [jax.ShapeDtypeStruct(extent, jnp.dtype(dtype))
              for extent, dtype in zip((left, right), product.operands, strict=True)]
    outputs = [jax.ShapeDtypeStruct(extents, product.made)]
    done = _operation_work(jax.lax.dot_general_p, inputs, outputs, params, on or target())
    return done, math.prod(extents)


def work(graph: EGraph, node, region: Region, held: dict | None = None,
         on: generation.GpuTarget | None = None) -> Work:
    """`node`'s work alone (the cost vocabulary, the module's account): its arithmetic the kernels' counter's
    for its primitive (`_operation_work`, the one rule: a product's FLOPs in its mode and passes, a
    reduction's operations an operand element, a special function's exponentials and a conversion's
    conversions, any other elementwise operation's operations, a clamp's two, a layout change none, a
    primitive no rule prices its elements unpriced, `Work.unpriced_operations`), an equation outside the
    language a leaf of the region reads (a derivative's `add_any`, `terms.OPERATIONS`) priced so where it is
    entrywise, where it was work no comparison read (the audit of 2026-11-19pm, 8964); and XLA's fusion's
    accounting beside it: an operand's (a leaf's) bytes, at its dtype's element size, read from device memory
    once however many values read it (extraction prices a shared node once), and nothing else of device
    memory, as XLA's fusion reads a program's inputs once (a sibling reduction's too) and keeps its fused
    intermediates on chip; the bytes a product's operands are staged in (the tile walk's, at their dtypes'
    sizes); a reduction and a product each a kernel of XLA's fusion, its root (elementwise work fuses into
    it); a contracted multiply nothing, its reader's fused multiply-add computing it (`rules._contracted`);
    a declaration's own rescale an exponential an element. What XLA's fusion keeps on chip
    (`Work.kept_bytes`, a TPU's VMEM traffic, row 7283): a leaf's bytes read once into the vector registers,
    a reduction's and a product's result written once and read once by its reader (a product's at its
    result's dtype, a reduction's at a vector lane's four bytes an element, `LANE_BYTES`),
    and elementwise work fused into a kernel none, its values in the registers. Priced on `on` (the
    schedule's target), else the process's."""
    if (args := get_callable_args(node, Term.leaf)) is not None:
        # The leaf's own shape from its region (a callback may read no table at a node it was not given).
        leaf = key(Term.leaf(args[0]))
        extent = region.extents.get(leaf)
        if extent is None:
            return Work()
        stored = math.prod(extent) * region.dtypes[leaf].itemsize
        read = Work(device_bytes=stored, kept_bytes=stored)
        outside = region.made_by.get(integer(args[0]))
        if outside is None or not programs.entrywise(outside.equation.primitive):
            return read
        equation = outside.equation
        return read + _operation_work(equation.primitive, [atom.aval for atom in equation.invars],
                                      [variable.aval for variable in equation.outvars], equation.params, on)
    if (args := get_callable_args(node, Term.reduce)) is not None:
        operand = shape_of(graph, args[0], held)
        if operand is None:
            return Work(kernels=1.0)
        primitive, params = region.params[integer(args[3])]
        # Its value: its operand's unreduced extents (an empty reduction's too).
        unreduced = tuple(size for axis, size in enumerate(operand) if axis not in params["axes"])
        in_dt = dtype_of(graph, args[0], held)
        return (_priced(primitive, params, [operand], unreduced, on, in_dt, in_dtypes=[in_dt])
                + Work(kernels=1.0, kept_bytes=kept(math.prod(unreduced))))
    if (args := get_callable_args(node, Term.dot)) is not None:
        left, right = (shape_of(graph, operand, held) for operand in args[:2])
        if left is None or right is None:
            return Work()
        index = integer(args[2])
        _, params = region.params[index]
        product = region.products[index]
        read = sum(math.prod(extent) * np.dtype(dtype).itemsize
                   for extent, dtype in zip((left, right), product.operands, strict=True))
        done, elements = _product(left, right, params, product, on)
        return done + Work(shared_bytes=read, kernels=1.0, kept_bytes=kept(elements, product.made.itemsize))
    if (args := get_callable_args(node, Term.binary)) is not None and text(args[1]) == RESCALE:
        # A declaration's own rescale of two anchors (`derive.Reduction.rescale`): an exponential an element.
        shapes = _shapes(graph, (args[0], args[2]), held)
        return Work() if shapes is None else Work(exponentials=math.prod(np.broadcast_shapes(*shapes)))
    if (args := get_callable_args(node, Term.binary)) is not None and text(args[1]) == CONTRACTED["mul"]:
        # Its reader's fused multiply-add computes it (`rules._contracted`).
        return Work()
    # Each other elementwise operation by its primitive, its parameters' entry (`Region.params`); a
    # selection's and a clamp's their own. A layout change computes nothing.
    if (args := get_callable_args(node, Term.unary)) is not None:
        return _elementwise(graph, *region.params[integer(args[2])], args[:1], held, on)
    if (args := get_callable_args(node, Term.convert)) is not None:
        return _elementwise(graph, *region.params[integer(args[1])], args[:1], held, on)
    for constructor in (Term.binary, Term.compare):
        if (args := get_callable_args(node, constructor)) is not None:
            primitive, params = region.params[integer(args[3])]
            return _elementwise(graph, primitive, params, (args[0], args[2]), held, on,
                                compared=constructor is Term.compare)
    for constructor, primitive in ((Term.select, jax.lax.select_n_p), (Term.clamp, jax.lax.clamp_p)):
        if (args := get_callable_args(node, constructor)) is not None:
            return _elementwise(graph, primitive, {}, args[:3], held, on)
    return Work()


LANE_BYTES = 4
"""The bytes a reduction's result element takes in what XLA's fusion keeps on chip (`work`): a vector lane's,
32 bits, the width a TPU's vector unit reduces in."""


def kept(elements: float, itemsize: float = LANE_BYTES) -> float:
    """The bytes a kernel's result of `elements` of `itemsize` moves on chip (`Work.kept_bytes`): written
    once by the kernel and read once by its reader, the one rule of every kernel's result `work` and the chunk
    forms (`dew.algebra.linear_scan`) price."""
    return 2.0 * elements * itemsize

def product_work(rows: int, depth: int, columns: int, *, count: int = 1, dtype=jnp.float32,
                 precision: jax.lax.PrecisionLike = None, on: generation.GpuTarget | None = None) -> Work:
    """`count` products of `[rows, depth] @ [depth, columns]` operands of `dtype` at `precision` (the default
    where unstated), in the mode and passes they multiply in on `on` (the price's `target` where None;
    `product.product_mode`, the kernels' counter's and `work`'s one rule), each a kernel of XLA's fusion, its
    root, its result of `dtype` kept on chip, written and read (`work`'s products): a chunk form's products
    (`dew.algebra.linear_scan`, at the precision they run at), Dew's fp32 products."""
    field, passes = product_mode(precision, dtype, dtype, on or target())
    held = Work(kernels=float(count), kept_bytes=kept(rows * columns * count, np.dtype(dtype).itemsize))
    return held + kernel_cost.multiplied(field, 2.0 * rows * depth * columns * count * passes)


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
    states none of (`templates.cost.units`). Its `facts`, the target it prices on, a node's work's too
    (`model`)."""

    @property
    def facts(self) -> generation.GpuTarget: ...

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
    device's rate (`templates.cost.rates`), in series, what its kernels keep on chip a TPU's VMEM traffic at
    its rate (`Work.kept_bytes`, row 7283), and each kernel the part's fixed latency (its launch and first
    byte, `Measured.fixed`; row 4081: a small fusion's time is that latency, 0.02 of it priced without), or
    its span at a loop iteration's latency where longer (the work-span model,
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
        # The walk's longest dependent path, the peak its greedy price reads too (`templates.cost.path`).
        return kernel_cost.path(done, self.facts)

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
        return Seconds(work(graph, node, region, held, schedule.facts), schedule)

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
        done = work(graph, node, region, held, schedule.facts)
        for constructor in (Term.unary, Term.binary):
            if (args := get_callable_args(node, constructor)) is not None:
                # Each operation's name follows its first operand; a fused multiply-add's is its primitive's.
                risk = RISK.get(PRIMITIVES.get(text(args[1]), text(args[1])), 0.0)
                return Seconds(done, schedule, risk)
        return Seconds(done, schedule)

    return DagCostModel(marginal_cost=marginal, identity=Seconds(Work(), schedule))
