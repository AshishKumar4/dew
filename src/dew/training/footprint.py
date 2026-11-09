"""A candidate step's device bytes, derived from its shapes at trace cost (rows 2113, 2114).

The fit search admits a step where its temporaries, the new outputs it does not alias and what its
first call builds beside it (`Candidate.held`) take a free block of every device (`step_fits`); the
state it reads is already resident. A footprint states those terms before any compile: the
residuals the loss's pullback keeps between forward and backward at one microbatch, read off the
pullback's traced program (`residual_bytes`), and the trained parameters' gradient, each a device's
share. A step over several devices leaves its residuals' shares to XLA's partitioner, which their
trace does not state, so there the residuals are unknown. XLA's workspace beyond them is no term of
it: what XLA's buffer assignment and scheduling make of a program is no function of its residuals
(tools/benchmark_decoder_remat.py, RTX 4080: 326 MB of residuals against 138 MB of temporaries without
recomputation, 13.5 against 31.9 at 'full'), so the run's one compile confirms the derivation, and the
plan records each term beside what the compile measured.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NamedTuple

import jax
import jax.extend.core as jcore
import numpy as np
from jax.core import ShapedArray

from dew import algebra
from dew._records import JSON
from dew.batch import Batch
from dew.data.dataset import rows_of
from dew.distributed.placement import placed_bytes
from dew.mesh import DATA_AXIS, FSDP_AXIS, row_axes
from dew.nn.kernels.generation import _compute_capability, gpu_target
from dew.nn.programs import bodies, computed, named
from dew.objectives.base import Objective
from dew.state import TrainState
from dew.training.compiled_program import (
    Slicing,
    bitcast,
    cublas_product,
    deviceless,
    dimensions,
    lowering_platform,
    simplified,
)
from dew.training.memory import head_fused_of, head_tile_of, judged
from dew.training.narrow import forward_variables
from dew.training.transaction import attempt_step


@dataclasses.dataclass(frozen=True)
class Footprint:
    """What a candidate step takes of a device beside the state it reads, derived from its shapes:
    `residuals`, the bytes its loss's pullback keeps from forward to backward at one microbatch;
    `gradients`, the trained parameters' gradient; and `held`, what its first call builds beside the
    resident state (`Candidate.held`), each a device's share. `residuals` is None for a step over
    several devices, whose shares of them the partitioner chooses."""

    residuals: int | None
    gradients: int
    held: int
    resident: Mapping[str, int] = dataclasses.field(default_factory=dict)
    """The prepared state's logical bytes by field and device, before buffer assignment aliases
    anything. These are not additional temporary bytes: the state's existing allocations already
    count against allocator headroom, and newly built copies and accumulators are in `held`."""
    peak: int | None = None
    """The most bytes its loss's gradient program holds at once at one microbatch, in its traced order
    (`peak_bytes`): the residuals still live, the backward's cotangents and the gradient as it is
    built. None where `residuals` is."""
    exact: bool = True
    """Whether `peak` is the program's as XLA's rules make it, else a bound from below
    (`bounded_below`): a step whose derived headroom is then an upper bound refuses on it soundly and
    admits on it provisionally, its compile deciding."""
    staged: int = 0
    """The bytes of the state's host-resident fields its commit holds on the device at once, beside the
    whole gradient it reads: the step's own staged cut as it is traced (`dew.training.staging.measured`),
    none for a state the device holds."""
    consumed: bool | None = None
    """Whether `peak` holds each gradient only while it is made, the step reading it there into the
    buffers its state donates (`transaction.consumed_where_made`, asked where the peak is derived), else
    every gradient to the end; None where no peak is derived."""

    @property
    def temporaries(self) -> int | None:
        """The most the step holds at once beside its state: its gradient program's peak, or, where its
        commit stages host-resident fields, its whole gradient and those where they are more. None where
        `peak` is."""
        if self.peak is None or not self.staged:
            return self.peak
        return max(self.peak, self.gradients + self.staged)

    def record(self) -> dict[str, JSON]:
        """The terms traced, not an admission decision: XLA's fit check remains authoritative."""
        return {"residuals": self.residuals, "gradients": self.gradients, "held": self.held,
                "resident": dict(self.resident), "peak": self.peak, "peak_exact": self.exact,
                "staged": self.staged, "consumed": self.consumed}


class Peak(NamedTuple):
    """A gradient program's derived peak (`gradient_peak`): its bytes, and whether they are the
    program's own or a bound from below (`bounded_below`)."""

    bytes: int
    exact: bool


def device_bytes(tree) -> int:
    """The placed leaves in device memory alone: an offloaded optimizer or EMA is host storage,
    not resident HBM. Each device leaf carries the sharding that `placed_bytes` reads."""
    return placed_bytes([leaf for leaf in jax.tree.leaves(tree)
                         if jax.typeof(leaf).memory_space == jax.memory.Space.Device])


CHEAP = frozenset({
    "add", "add_any", "and", "bitcast_convert_type", "broadcast_in_dim", "ceil", "clamp", "clz", "eq", "ne",
    "lt", "le", "gt", "ge", "complex", "concatenate", "convert_element_type", "copy", "copy_p",
    "dynamic_slice", "dynamic_update_slice", "floor", "imag", "iota", "is_finite", "max", "min", "mul", "neg",
    "not", "or", "xor", "pad", "population_count", "real", "reduce_precision", "reshape", "squeeze",
    "expand_dims", "rev", "round", "select_n", "shift_left", "shift_right_arithmetic", "shift_right_logical",
    "slice", "sub", "transpose", "integer_pow", "square", "name"})
"""The primitives XLA's CPU fusion takes for cheap, which it duplicates into each consumer rather than
keep a buffer of (`InstructionFusion::IsExpensive`, xla/service/instruction_fusion.cc: every elementwise
op but division, the transcendentals and their kin, and the layout ops), an integer power and a square
lowering to its multiplies, a cotangent sum, which JAX lowers to one `add` (`add_jaxvals_lowering`,
jax/_src/interpreters/mlir.py), and a checkpoint's name, which computes nothing."""

REAL_CHEAP = frozenset({"abs", "sign", "sin", "cos", "tan", "sinh"})
"""The primitives XLA's CPU fusion takes for cheap over real elements alone (`IsExpensive`)."""

ELEMENTWISE = CHEAP | REAL_CHEAP | frozenset({
    "div", "rem", "exp", "exp2", "log", "log1p", "expm1", "tanh", "logistic", "erf", "erfc", "erf_inv",
    "pow", "sqrt", "rsqrt", "cbrt", "atan2", "asin", "acos", "atan", "cosh", "asinh", "acosh", "atanh"})
"""The elementwise and layout primitives, every one of which XLA's GPU fusion recomputes in each consumer
rather than keep a buffer of: its priority fusion fuses a producer into all its consumers where its cost
model finds the memory traffic saved worth the recompute (xla/backends/gpu/transforms/priority_fusion.cc),
an elementwise one's always on a memory-bound GPU, a transcendental's too (an attention's exponentials and
probabilities, on an H100's compile, recomputed from its scores in the backward)."""


TRANSPOSES = frozenset({"transpose"})
"""The layout primitive XLA's layout assignment folds into the storage of a value the program computes,
laying it out as its reader needs (a returned gradient's), not into an input's, whose layout is fixed."""


LOOPS = frozenset({"scan", "while"})

POINTER_BYTES = 8
"""The bytes of a pointer, which XLA's tuple of a program's results holds one of for each result."""

PRODUCTS = frozenset({"dot_general", "conv_general_dilated", "ragged_dot_general"})
"""The products a weight's gradient is made by, past which a value is a cotangent the backward reads on
(`sunk`)."""
"""The primitives that run their body turn after turn, their results held across the turns."""


def traced_gradient(loss: Callable, params, arguments: Sequence) -> tuple[jcore.Jaxpr, int, list]:
    """One trace of `loss(params, *arguments)`'s forward and backward: the program `jax.vjp`'s pullback
    runs on a cotangent of ones, returning the gradient's leaves and then the pullback's own, its
    residuals; how many gradient leaves lead its outputs; and the residuals' traced shapes. Its math is
    the algebra pass's, as the step that compiles it rewrites its trace (`algebra.rewritten_program`,
    `OuterProgram`, docs/design/algebra.md §6.7): each region it rewrites is recorded as any trace's,
    the derivation the compile then reuses (`program._derived`)."""
    def both(params, *arguments):
        outputs, pullback, _ = jax.vjp(lambda trainable: loss(trainable, *arguments), params, has_aux=True)
        return pullback(jax.tree.map(jax.numpy.ones_like, outputs)), pullback
    closed, (gradients, pullback) = jax.make_jaxpr(both, return_shape=True)(params, *arguments)
    closed = algebra.rewritten_program(closed)
    count = len(jax.tree.leaves(gradients))
    return closed.jaxpr, count, jax.tree.leaves(pullback)


def derived_terms(loss: Callable, params, arguments: Sequence, *, platform: str, workspace: Workspace,
                  lowering: str | None = None, consumed: bool = False,
                  deferred: bool = False) -> tuple[int, Peak]:
    """`residual_bytes` and `gradient_peak` of `loss(params, *arguments)` from one trace of its forward and
    backward (`traced_gradient`), lowered once (`compiled`, its outputs the gradient and the residuals
    alike: an equation's lowering reads no output), as each derives it from its own (row 4261: lowered
    for each, the two lowerings were 16 of a Qwen3.5-4B candidate's 28 seconds)."""
    program, count, shapes = traced_gradient(loss, params, arguments)
    lowered = simplified(program, platform, lowering)
    return (_kept(lowered, count, shapes, platform),
            _peak(lowered, count, platform, workspace, consumed, deferred))


def residual_bytes(loss: Callable, params, arguments: Sequence, *, platform: str,
                   lowering: str | None = None) -> int:
    """The device bytes the pullback of `loss(params, *arguments)` in `params` keeps, a device's share,
    as XLA's fusion for `platform` keeps them: each value its traced program (`jax.vjp`'s pullback, its
    residuals the outputs of `traced_gradient`'s program past the gradient, as JAX lowers it for
    `lowering`, `platform`'s own name where none is given, a GPU's CUDA's: `compiled`) closes over counts
    the values it is computed from without a buffer (`Frontier.reading`), once, but those the program
    reads, which the state and the batch already hold, and literals; a value in pinned host memory counts
    none. A value the pullback closes over keeps its own traced sharding; one whose trace states none, and
    every other value, count whole. Matching an axis's size to the batch size cannot identify its
    sharding: a weight or a scan's layer axis can have that size too."""
    program, count, shapes = traced_gradient(loss, params, arguments)
    return _kept(simplified(program, platform, lowering), count, shapes, platform)


def gradient_peak(loss: Callable, params, arguments: Sequence, *, platform: str,
                  workspace: Workspace, lowering: str | None = None, consumed: bool = False,
                  deferred: bool = False) -> Peak:
    """The most device bytes the gradient of `loss(params, *arguments)` in `params` holds at once, a
    device's share, as XLA's fusion for `platform` keeps its values and each matrix product takes
    `workspace` bytes beside them (`peak_bytes`), and whether that is the program's own or a bound from
    below (`bounded_below`): the program `jax.vjp`'s pullback runs on a cotangent of ones, traced whole,
    forward and backward (`traced_gradient`, its outputs the gradient's), as JAX lowers it for `lowering`
    (`compiled`; `platform`'s own name where none is given, a GPU's CUDA's). Each gradient is held to the
    end, or, `consumed`, while it is made and no longer (`peak_bytes`'s): where the step reads it there
    into the buffers the state donates (`transaction.consumed_where_made`), the optimizer's update or the
    accumulator's sum written in place (the donated state's outputs alias its arguments,
    `HloInputOutputAliasConfig`), and XLA's memory-minimizing schedule runs a consumer that frees its
    operand as soon as it is ready (`ListScheduler::BytesFreedIfScheduled`,
    xla/hlo/transforms/simplifiers/hlo_memory_scheduler.cc; `ScheduleGpuModule`,
    xla/service/gpu/gpu_hlo_schedule.cc). With `deferred`, the gradient's own equations run after the rest
    of the backward (`sunk`)."""
    program, count, _ = traced_gradient(loss, params, arguments)
    return _peak(simplified(program, platform, lowering), count, platform, workspace, consumed, deferred)


def _kept(lowered: jcore.Jaxpr, count: int, shapes: Sequence, platform: str) -> int:
    """`residual_bytes` of `traced_gradient`'s program as it is `lowered` (`compiled`), its outputs past
    the first `count` the residuals."""
    return _residuals(lowered.replace(outvars=lowered.outvars[count:]), shapes, platform)


def _peak(lowered: jcore.Jaxpr, count: int, platform: str, workspace: Workspace, consumed: bool = False,
          deferred: bool = False) -> Peak:
    """`gradient_peak` of `traced_gradient`'s program as it is `lowered` (`compiled`), its first `count`
    outputs the gradient."""
    gradient = lowered.replace(outvars=lowered.outvars[:count])
    # Hoisted last: a constant the gradient returns runs ahead of the work however its weight gradients run.
    ordered = hoisted(sunk(gradient) if deferred else gradient)
    return Peak(peak_bytes(ordered, platform, workspace, consumed=consumed),
                not bounded_below(gradient, platform))


def hoisted(program: jcore.Jaxpr) -> jcore.Jaxpr:
    """`program` with each output computed from literals alone run first, with the equations that compute
    it, in their order: a constant the program returns and no equation of its work reads, which XLA
    schedules ahead of that work, holding it to its reader past the program, each one computed once
    where equal ones repeat (`deduplicated`, which the derivation reads it through). The gradient of a
    parameter the loss reads none of is such a constant, a broadcast zero JAX writes at the backward's
    end, which XLA made during the forward, one a shape: Qwen3.5-4B's step at 4 x 2048 tokens under
    'minimal_with_scan_and_conv', compiled for an H100, held one zero gradient of each of 25 shapes of
    its multi-token head's and its vision tower's 312, 0.48 GiB of 1.69, from step 2,480 of its schedule
    to the update at 15,091, its peak at 4,863 among them (row 4089, its buffer dump). A constant the
    work reads (a loop's zero carry) is made where the work needs it, as XLA makes it."""
    constant: dict[jcore.Var, int] = {}
    for index, equation in enumerate(program.eqns):
        reads = [value for value in equation.invars if computed(value)]
        if not equation.effects and all(value in constant for value in reads):
            constant.update((out, index) for out in equation.outvars if computed(out))
    # A constant equation is hoisted where every reader of its results is a hoisted one or the outputs.
    readers: dict[jcore.Var, set[int]] = collections.defaultdict(set)
    for index, equation in enumerate(program.eqns):
        for value in equation.invars:
            if computed(value):
                readers[value].add(index)
    returned = {value for value in program.outvars if computed(value)}
    early: set[int] = set()
    for index in sorted(set(constant.values()), reverse=True):
        made = [out for out in program.eqns[index].outvars if computed(out)]
        if any(out in returned or readers[out] for out in made) and all(
                reader in early for out in made for reader in readers[out]):
            early.add(index)
    ordered = sorted(range(len(program.eqns)), key=lambda index: index not in early)
    return program.replace(eqns=[program.eqns[index] for index in ordered])


def sunk(program: jcore.Jaxpr) -> jcore.Jaxpr:
    """`program`, a gradient's, with each weight's gradient run after every other equation, in their order:
    the equations whose results reach its outputs through no product (`PRODUCTS`), each weight's gradient
    product and what sums, casts or lays it out, the backward's cotangents, which the products of the
    layers before read, left where they are, and the operands those products read held to the end. XLA
    schedules a step's gradient so where its head takes the backward a tile at a time (`head_tile`,
    unfused) on some GPUs (`GpuTarget.late_weight_gradients`): Qwen3.5-4B's step at 2 x 4096 tokens under
    'full', compiled for an H100, held 29 layers' four [8192, 9216] product operands at its peak, 16.3 of
    its 34.6 GiB, and no layer's gradient (row 3341, its calibration's buffer dumps)."""
    outputs = {value for value in program.outvars if computed(value)}
    readers: dict[jcore.Var, list[int]] = collections.defaultdict(list)
    for index, equation in enumerate(program.eqns):
        for value in {read for read in equation.invars if computed(read)}:
            readers[value].append(index)
    late: set[int] = set()
    for index in range(len(program.eqns) - 1, -1, -1):
        made = [out for out in program.eqns[index].outvars if computed(out)]
        # Every result read by late equations alone, none a product, and one at least read or returned.
        if any(readers[out] or out in outputs for out in made) and all(
                reader in late and named(program.eqns[reader].primitive) not in PRODUCTS
                for out in made for reader in readers[out]):
            late.add(index)
    ordered = sorted(range(len(program.eqns)), key=lambda index: index in late)
    return program.replace(eqns=[program.eqns[index] for index in ordered])


def _residuals(program: jcore.Jaxpr, shapes: Sequence, platform: str) -> int:
    """`residual_bytes` of `program`, a pullback's traced program whose outputs are its residuals, each of
    the traced `shapes`."""
    placed = {value: shape for value, shape in zip(program.outvars, shapes, strict=True)
              if not isinstance(value, jcore.Literal)}
    frontier = Frontier(program, platform, returned=False)
    total = 0
    for value in frontier.reading(program.outvars):
        aval = value.aval
        if not isinstance(aval, ShapedArray):
            raise TypeError(f"a pullback keeps {aval}, which is no array")
        if aval.memory_space != jax.memory.Space.Device:
            continue
        shape = placed.get(value)
        total += (math.prod(aval.shape) * aval.dtype.itemsize if shape is None or shape.sharding is None
                  else placed_bytes(shape))
    return total


def bounded_below(program: jcore.Jaxpr, platform: str) -> bool:
    """Whether `program`'s derived peak bounds what XLA makes of it from below: on a GPU, where it runs a
    ragged product, which holds at least its output while XLA lays the rest out as its target, dtype
    and flags choose: a grouped kernel, or a product over its groups that expands its operands
    (`RaggedDotRewriter`, xla/hlo/transforms/expanders/ragged_dot_rewriter.cc), beside which its layout
    assignment (xla/service/layout_assignment.cc) and fusion (xla/backends/gpu/transforms/
    priority_fusion.cc) keep buffers by their cost models' judgment, no rule the derivation can source.
    Such a step's compile decides its fit, and a miss is logged with the need it compiled to."""
    return platform == "gpu" and any(
        named(equation.primitive) == "ragged_dot_general"
        or any(bounded_below(body, platform) for body in bodies(equation))
        for equation in program.eqns)


def peak_bytes(program: jcore.Jaxpr, platform: str, workspace: Workspace,
               slicing: Slicing | None = None, carried: frozenset = frozenset(), *,
               consumed: bool = False) -> int:
    """The most bytes `program` holds at once beside its inputs, its equations run in their traced
    order. A value XLA keeps a buffer of (`Frontier`) is held from the equation that computes it to the
    last that reads it, an output to the end. An equation that takes its operands from memory
    (`Frontier.fusing` says no: a matrix product, a loop) holds beside them each operand a fused chain
    computes, while it runs, and a cuBLAS matrix product its workspace too (`Workspace`); a loop's or a
    conditional's own peak, its results among them, stands in for its results while it runs, its body
    read with what it slices of the loop it runs in (`slicing`); `program` has its calls inlined
    (`compiled`). A loop's carried
    value is one buffer from its initial value, where that is a buffer of the program's the loop reads
    last, to its result, its body's returned value of it among them (`carry`; `carried`, the values
    `program`, a loop's body, returns into it). With `consumed`, an output is held while the equation
    that makes it runs, as an update that reads it there holds it (`gradient_peak`). The step's own
    program (not a loop's body) holds besides each copy of an input its products read laid out for them
    from the first such product to the last (`laid_copies`), and on a GPU each product the operands it
    may be made beside (`siblings`), and the tuple of its results; on a CPU, as a bound wherever XLA may
    fuse or make apart, the arrays its fusions read apart (`laid_inputs`), what its products read made once
    or copied (`cpu_layouts`), and its input copies from its start. No schedule XLA picks is modelled: a
    compile confirms."""
    frontier = Frontier(program, platform, returned=True)
    last = {kept: index for index, equation in enumerate(program.eqns)
            for kept in frontier.reading(equation.invars)}
    # The copies of the step's inputs its products read laid out for them (`laid_copies`), each held
    # from its first reader to its last, and on a GPU the siblings a product's operands are made beside
    # (`siblings`): the step's own program's, whose parameters' layouts are fixed, not a loop's body's.
    entry = slicing is None and not carried
    beside, spans, spanned = siblings(frontier, program) if entry and platform == "gpu" else ({}, [], {})
    begun: dict = {}
    if platform != "gpu":
        beside, spans, spanned, begun = cpu_layouts(frontier, program)
    laid = laid_copies(frontier, program) if entry else []
    if platform != "gpu":
        # A copy of an input reads nothing XLA must wait on: a CPU's schedule makes it at the program's
        # start (row 5290), so a bound holds it from there.
        laid = [(0, ended, size) for _, ended, size in laid]
    copies = [*laid, *spans]
    if consumed:
        # The program's outputs are read where they are made, by an update that writes the buffers the
        # state donates (the optimizer's, or the accumulator's): each is held while its equation runs.
        made_at = {out: index for index, equation in enumerate(program.eqns) for out in equation.outvars}
        for kept in frontier.reading(program.outvars):
            last.setdefault(kept, made_at.get(kept, 0))
    else:
        last.update({kept: len(program.eqns) for kept in frontier.reading(program.outvars)})
    sliced = (slicing or Slicing()).within(program)
    # The step's results are one tuple, a pointer to each, which XLA holds the whole program
    # (`ShapeUtil::ByteSizeOf` of a tuple).
    live = POINTER_BYTES * len(program.outvars) if entry and len(program.outvars) > 1 else 0
    held, most = set(), live
    for index, equation in enumerate(program.eqns):
        live += sum(size for first, _, size in copies if first == index)
        made = [out for out in equation.outvars
                if out in last and frontier.buffered(out) and out not in carried]
        made_bytes = sum(map(device_share, made))
        # A scatter or a product with a bias writes its result in a buffer it reads last (`in_place`).
        taken = in_place(frontier, equation, held, last, index)
        if made or not frontier.fusing(equation):
            inner = max((peak_bytes(body, platform, workspace, Slicing.entering(equation, body),
                                    returned_carry(equation, body)) for body in bodies(equation)), default=0)
            # An operand made beside an earlier product's is held from there (`siblings`), not made here.
            made_for = unfused(frontier, equation) - spanned.get(index, set())
            operands = sum(map(device_share, made_for)) + beside.get(index, 0)
            if platform != "gpu":
                # The arrays the fusions making its operands read apart (`laid_inputs`), a shared one's
                # where it is made, before its first reader.
                starting = {value for value in spanned.get(index, set()) if begun[value] == index}
                # A reduction's fusion runs on YNNPACK's kernels as a product's does; an elementwise
                # one is XLA's own loop, which reads its broadcasts and transposes in place.
                reduced = [out for out in made if named(equation.primitive).startswith("reduce_")]
                operands += sum(map(device_share, laid_inputs(frontier, [*reduced, *made_for, *starting])))
            products = 0 if not cublas_product(equation) else (
                workspace.sliced if sliced.fused(equation) else workspace.each)
            # A loop's results (its stacked outputs, its carry) are held beside each turn's own; a call's
            # results are what its body returns.
            running = (looped(frontier, last, index, equation, made, carried) + inner
                       if named(equation.primitive) in LOOPS else max(inner, made_bytes) - taken)
            # The fusions that make its operands run before it, each buffer they alone read freed with them
            # (`freed`): the most is theirs, or its own past them.
            freed = sum(map(device_share, dying(frontier, equation, made_for, held, last, index)))
            most = max(most, live + operands, live - freed + running + operands + products)
        held.update(made)
        live += made_bytes
        for done in [value for value in held if last[value] == index]:
            held.discard(done)
            live -= device_share(done)
        live -= sum(size for _, ended, size in copies if ended == index)
    return most


def input_of(frontier: Frontier, value) -> jcore.Var | None:
    """The program input `value` is, or is a bitcast of (`bitcast`: its storage, in its memory order);
    None for any other value."""
    while computed(value) and value not in frontier.inputs:
        equation = frontier.producers.get(value)
        if equation is None or not bitcast(equation):
            return None
        value = equation.invars[0]
    return value if computed(value) else None


def product_order(operand: jcore.Var, contracting: Sequence[int],
                  batch: Sequence[int]) -> tuple[int, ...] | None:
    """The order of `operand`'s dimensions a product reading it with these `contracting` and `batch`
    dimensions reads it in where its storage, row-major, is not one its GEMM takes: its batch, its free
    and its contracting dimensions, each group as the product orders it and dimensions of one element in
    none (XLA's dot canonicalization, `DotDecomposer`). A GEMM takes its operand's groups stored batch,
    rows, columns; batch, columns, rows; rows, batch, columns; or columns, batch, rows (`MatrixLayout::For`,
    xla/service/gpu/matmul_utils.cc), which layout assignment keeps (`SetDotOperandLayout`); None where it
    is one of those."""
    shape = dimensions(operand)
    free = [axis for axis in range(len(shape)) if axis not in contracting and axis not in batch]

    def order(*groups: Sequence[int]) -> tuple[int, ...]:
        return tuple(axis for group in groups for axis in group if shape[axis] != 1)

    taken = [order(batch, free, contracting), order(batch, contracting, free),
             order(free, batch, contracting), order(contracting, batch, free)]
    return None if order(range(len(shape))) in taken else taken[0]


def laid_copies(frontier: Frontier, program: jcore.Jaxpr) -> list[tuple[int, int, int]]:
    """The copies of `program`'s inputs its products read laid out for them, each `(first, last, bytes)`:
    an input, or a bitcast of one, whose storage a product cannot read in the order it takes
    (`product_order`) is copied into that order, which layout assignment fixes for a parameter as it
    stands (xla/service/layout_assignment.cc), one copy for every product that takes it in that order,
    made before the first of them and held to the last (two products reading one input's copy share it:
    an attention's keys read by its scores' product and its queries' gradient, an H100's compile)."""
    spans: dict[tuple, list[int]] = {}
    for index, equation in enumerate(program.eqns):
        if named(equation.primitive) != "dot_general":
            continue
        (contracted, batched) = equation.params["dimension_numbers"]
        for operand, contracting, batch in zip(equation.invars[:2], contracted, batched, strict=True):
            source = input_of(frontier, operand)
            if source is None or not isinstance(operand, jcore.Var):
                continue
            taken = product_order(operand, contracting, batch)
            if taken is not None:
                key = (source, dimensions(operand), taken)
                spans.setdefault(key, [index, index, device_share(operand)])[1] = index
    return [(first, ended, size) for first, ended, size in spans.values()]


def cpu_layouts(frontier: Frontier, program: jcore.Jaxpr) -> tuple[
        dict[int, int], list[tuple[int, int, int]], dict[int, set], dict[jcore.Var, int]]:
    """What a CPU's products hold beside the operands they read made for them (`unfused`): XLA's CPU
    products (YNNPACK's and Eigen's) read every operand from memory in the order they take it, so a
    fused value two products read is made once, held from the first to the last (a span, `(first, last,
    bytes)`, and that operand no longer made for its reads, by product: an attention's scores' cotangent,
    read by its keys' and its queries' gradients); and each value XLA keeps a buffer of (`Frontier`), or
    so made once, that a product reads in an order it does not take (`product_order`) is copied for that
    product (bytes, by product: an attention's probabilities read by its values' gradient laid out
    otherwise). Row 5290: 64 x 128 tokens, 8 heads over 2, compiled 1.375 times the derivation without.
    Last, each shared value's first reader, where it is made."""
    products = [(index, equation) for index, equation in enumerate(program.eqns)
                if named(equation.primitive) == "dot_general"]
    reads: dict = {}
    for index, equation in products:
        for value in unfused(frontier, equation):
            reads.setdefault(value, []).append(index)
    # A value computed from constants alone reads nothing XLA must wait on, so it may be made at the
    # program's start and held to its last reader (a cotangent of ones): held so from the start.
    constant = {value for value in reads if not frontier.roots([value])}
    shared = {value: [0, *readers] if value in constant else readers for value, readers in reads.items()
              if len(readers) > 1 or value in constant}
    spanned: dict[int, set] = {}
    for value, readers in shared.items():
        for index in readers if value not in constant else readers[1:]:
            spanned.setdefault(index, set()).add(value)
    copied: dict[int, int] = {}
    for index, equation in products:
        (contracted, batched) = equation.params["dimension_numbers"]
        for operand, contracting, batch in zip(equation.invars[:2], contracted, batched, strict=True):
            kept = isinstance(operand, jcore.Var) and operand not in frontier.inputs and (
                frontier.buffered(operand) or operand in shared)
            if kept and product_order(operand, contracting, batch) is not None:
                copied[index] = copied.get(index, 0) + device_share(operand)
    return (copied, [(readers[0], readers[-1], device_share(value)) for value, readers in shared.items()],
            spanned, {value: readers[value in constant] for value, readers in shared.items()})


def laid_inputs(frontier: Frontier, values: Sequence[jcore.Var]) -> set[jcore.Var]:
    """The values of the fused chains that make `values` that a CPU's fusion reads as arrays of their
    own: XLA's CPU fusions over YNNPACK's elementwise kernels read their inputs as dense arrays in their
    own order and type, so each broadcast that widens a value, and each transpose or conversion of a
    buffer, in a chain is made apart, beside the chain, as the chain runs (an attention's softmax
    denominator broadcast over its scores, its probabilities' cotangent turned to float32 and transposed
    for its scores' cotangent: 2 x 2048 tokens, 4 heads over 1, in bfloat16, row 5290)."""
    found, pending, seen = set(), list(values), set()
    while pending:
        value = pending.pop()
        if value in seen or not computed(value):
            continue
        seen.add(value)
        equation = frontier.producers.get(value)
        if equation is None or not frontier.fusing(equation):
            # A product or a loop reads its operands from memory: no fusion of the chain runs past it.
            continue
        reads = [read for read in equation.invars if computed(read)]
        name = named(equation.primitive)
        # Every broadcast that widens a value, of constants too (a mask's iota, a cotangent of ones): XLA
        # may fold it into the fusion or make it apart, and a bound counts it apart.
        widened = (name == "broadcast_in_dim"
                   and math.prod(dimensions(value)) > math.prod(dimensions(equation.invars[0])))
        relaid = name in ("transpose", "convert_element_type") and any(
            read in frontier.inputs or frontier.buffered(read) for read in reads)
        if widened or relaid:
            found.add(value)
        pending.extend(read for read in reads if read not in frontier.inputs and not frontier.buffered(read))
    return {value for value in found if value not in values}


def siblings(frontier: Frontier, program: jcore.Jaxpr) -> tuple[dict[int, int], list[tuple[int, int, int]],
                                                                   dict[int, set]]:
    """What XLA's GPU multi-output fusion may make together of `program`'s products' operands: two fused
    values of one element count, each made for its own product's read (`unfused`), read from storage
    both read (`Frontier.roots`, an input's among it) and neither computed from the other through any
    path (`CanFuseSiblings`), are siblings XLA may fuse into one fusion writing both (`FuseSiblings`,
    xla/backends/gpu/transforms/multi_output_fusion.cc), its cost model deciding. The fusion runs once
    both are ready: where that is before the earlier product, the later product's operand is made there
    and held from the earlier product to its own (a span, `(first, last, bytes)`, and that operand no
    longer made for its read, by product); where it is not, the earlier product waits on it, and the
    later holds the earlier's operand beside its own (bytes, by product). An attention's gradient on an
    H100 held its probabilities in bf16, for its values' gradient, beside its scores' cotangent, both
    made from its scores (2 x 2048 tokens, 4 heads over 1, 0.41 GB compiled against 0.34 derived without
    it). Taken wherever the fusion is possible, so a peak it makes holds where XLA fuses them."""
    made = {index: unfused(frontier, equation) for index, equation in enumerate(program.eqns)
            if not frontier.fusing(equation)}
    position = {out: index for index, equation in enumerate(program.eqns) for out in equation.outvars}

    def computed_from(value, other) -> bool:
        """Whether `value` is computed from `other` through any equation."""
        seen, pending = set(), [value]
        while pending:
            equation = frontier.producers.get(pending.pop())
            for read in [] if equation is None else equation.invars:
                if computed(read) and read not in seen:
                    if read is other:
                        return True
                    seen.add(read)
                    pending.append(read)
        return False

    def ready(*values) -> int:
        """The index past which every value the storage `values` read is made."""
        return max((position.get(root, -1) for value in values for root in frontier.roots([value])),
                   default=-1)

    def sibling(left, right) -> bool:
        common = frontier.roots([left]) & frontier.roots([right])
        return (math.prod(dimensions(left)) == math.prod(dimensions(right))
                and any(math.prod(dimensions(stored)) > 1 for stored in common)
                and not computed_from(left, right) and not computed_from(right, left))

    beside: dict[int, set] = {}
    first: dict = {}
    for later, values in made.items():
        for earlier, others in made.items():
            for other, value in itertools.product(others - values, values):
                if earlier >= later or not sibling(value, other):
                    continue
                if ready(value, other) < earlier:
                    # Made before the earlier product, beside its operand, and held to its own reader.
                    first[value] = min(first.get(value, earlier), earlier)
                else:
                    beside.setdefault(later, set()).add(other)
    # A product reading an operand before the fusion that makes it beside its sibling makes its own.
    spanned = {index: kept for index, values in made.items()
               if (kept := {value for value in values & first.keys() if index >= first[value]})}
    last = {value: index for index, values in sorted(spanned.items()) for value in values}
    return ({later: sum(map(device_share, others)) for later, others in beside.items()},
            [(first[value], last[value], device_share(value)) for value in first], spanned)


def unfused(frontier: Frontier, equation: jcore.JaxprEqn) -> set[jcore.Var]:
    """The operands `equation` reads from buffers made for its read, beside the ones it holds: each fused one
    of an equation that takes its operands from memory (`Frontier.fusing` says no), none of one that
    recomputes them, nor of an optimization barrier, whose results are its operands' own storage (the
    buffers made for its fused operands are the ones it holds, not a second set read beside them). A
    scatter fuses its indices and updates but those reading its base, which XLA computes apart, as it
    computes the base apart into the buffer the scatter writes (`CanEmitInputFusedScatter`,
    xla/service/gpu/gpu_fusible.cc): its updates computed from its base hold a buffer of their own."""
    name = named(equation.primitive)
    if name == "optimization_barrier" or not equation.invars:
        return set()
    if name.startswith("scatter"):
        base = frontier.roots(equation.invars[:1])
        return {value for value in equation.invars[1:] if computed(value) and value is not equation.invars[0]
                and frontier.made_for(value) and frontier.roots([value]) & base}
    if frontier.fusing(equation):
        return set()
    return {value for value in equation.invars if computed(value) and frontier.made_for(value)}


def dying(frontier: Frontier, equation: jcore.JaxprEqn, made_for: set[jcore.Var], held: set, last: Mapping,
          index: int) -> set[jcore.Var]:
    """The buffers the program holds (`held`) that the fusions making `equation`'s operands (`made_for`,
    `unfused`) read last, the equation, the program's `index`th, reading none of them itself: each such
    fusion is a kernel of its own XLA runs before the equation, past which a buffer no other instruction
    reads is freed (its live range ending at its last use, `HloLiveRange`,
    xla/hlo/utils/hlo_live_range.cc). A head's weight-gradient product reads the logits' cotangent, which
    XLA computes from the logits a recomputed head holds behind its barrier and frees them: Qwen3.5-4B's
    step at four microbatches was derived holding both, 0.95 GiB over XLA's 7.42 (row 4089)."""
    if not made_for:
        return set()
    direct = frontier.reading([value for value in equation.invars
                               if computed(value) and value not in made_for])
    return {buffer for buffer in frontier.reading(made_for)
            if buffer in held and last.get(buffer) == index and buffer not in direct}


def in_place(frontier: Frontier, equation: jcore.JaxprEqn, held: set, last: Mapping, index: int) -> int:
    """The bytes of the buffer `equation`, the program's `index`th, writes its result in: a scatter's
    base's, or a product's bias's (`biased`), the one buffer the program holds (`held`) that it is or views
    (`Frontier.reading`), where the equation reads that buffer last, its other operands read none of it,
    and it is its result's size, as XLA shares a scatter's first operand's buffer with its result
    (`HloDataflowAnalysis::CanShareOperandBufferWithUser`, xla/hlo/analysis/hlo_dataflow_analysis.cc) and
    a GEMM's bias's that nothing else reads, before it or after (`FuseMatrixBiasAdd`'s `can_overwrite_bias`,
    xla/backends/gpu/transforms/gemm_rewriter.cc); none for any other equation."""
    name = named(equation.primitive)
    if name.startswith("scatter") and equation.invars:
        at = 0
    elif name == "dot_general" and len(equation.invars) == 3:
        at = 2
    else:
        return 0
    base = equation.invars[at]
    if not computed(base) or equation.invars.count(base) != 1 or frontier.made_for(base):
        return 0
    kept = frontier.reading([base])
    if len(kept) != 1:
        return 0
    (buffer,) = kept
    # Its other operands read none of that buffer, which writing it in place would overwrite under them.
    others = [value for number, value in enumerate(equation.invars) if number != at]
    if buffer not in held or last.get(buffer) != index or buffer in frontier.reading(others):
        return 0
    # A GEMM writes in its bias only where the product is the bias's one reader (`can_overwrite_bias`),
    # whatever read it before.
    if at == 2 and {frontier.readers[base], frontier.readers[buffer]} != {1}:
        return 0
    if device_share(buffer) != device_share(equation.outvars[0]):
        return 0
    return device_share(buffer)


def carry(equation: jcore.JaxprEqn) -> list[tuple]:
    """Each value the loop `equation` carries, as its initial value, the loop's result of it, and
    whether its body changes it, returning another value than the parameter it took. XLA keeps the
    initial value, the body's parameter and returned value of it and the loop's result in one buffer
    (`ComputeAliasedValues`, xla/hlo/analysis/hlo_alias_analysis.cc), copying the initial value into a
    buffer of the loop's where it is an input or still read (copy insertion,
    xla/service/copy_insertion.cc)."""
    if named(equation.primitive) == "while":
        start = equation.params["cond_nconsts"] + equation.params["body_nconsts"]
        body = equation.params["body_jaxpr"].jaxpr
        taken = body.invars[equation.params["body_nconsts"]:]
        changed = [returned is not took for returned, took in zip(body.outvars, taken, strict=True)]
        return list(zip(equation.invars[start:], equation.outvars, changed, strict=True))
    if named(equation.primitive) == "scan":
        body = equation.params["jaxpr"].jaxpr
        _, started, _ = equation.params["ft_in"].update(equation.invars).unpack()
        _, taken, _ = equation.params["ft_in"].update(body.invars).unpack()
        ended, _ = equation.params["ft_out"].update(equation.outvars).unpack()
        returned, _ = equation.params["ft_out"].update(body.outvars).unpack()
        changed = [value is not took for value, took in zip(returned, taken, strict=True)]
        return list(zip(started, ended, changed, strict=True))
    return []


def looped(frontier: Frontier, last: Mapping, index: int, equation: jcore.JaxprEqn, made: Sequence,
           reserved: frozenset = frozenset()) -> int:
    """The bytes the loop `equation`, the program's `index`th, holds of its own while it runs, its body's
    aside: each result it makes (`made`) but a carry's, and each carried value its body changes, held in
    a buffer of the loop's whether its result is read after or not, but where it is held in its
    initial value's (`sharing`) or in a buffer an enclosing loop holds it in (`reserved`, the values
    the program it runs in, that loop's body, returns into that loop's carry: the carries' buffers are
    one)."""
    slots = carry(equation)
    ends = {ended for _, ended, _ in slots}
    shared = sharing(frontier, last, index, equation) | reserved
    return (sum(device_share(out) for out in made if out not in ends)
            + sum(device_share(ended) for _, ended, changed in slots if changed and ended not in shared))


def returned_carry(equation: jcore.JaxprEqn, body: jcore.Jaxpr) -> frozenset:
    """The values `body` returns into the buffers the loop `equation` carries, where it is the loop's
    body (`carry`); none of any other program."""
    if named(equation.primitive) == "while" and body is equation.params["body_jaxpr"].jaxpr:
        return frozenset(value for value in body.outvars if computed(value))
    if named(equation.primitive) == "scan" and body is equation.params["jaxpr"].jaxpr:
        returned, _ = equation.params["ft_out"].update(body.outvars).unpack()
        return frozenset(value for value in returned if computed(value))
    return frozenset()


def sharing(frontier: Frontier, last: Mapping, index: int, equation: jcore.JaxprEqn) -> set:
    """The results of the loop `equation`, the program's `index`th, held in their initial value's buffer
    (`carry`): one the program computes, not an input or a literal, whose buffers the loop reads last,
    or which a fused chain computes into a buffer for the loop to read; each buffer one result's, a
    value and a view of it one buffer."""
    found: set = set()
    taken: set = set()
    for start, ended, _ in carry(equation):
        if not computed(start) or start in frontier.inputs:
            continue
        kept = {start} if frontier.made_for(start) else frontier.reading([start])
        if kept and not kept & taken and (frontier.made_for(start) or all(last[value] == index
                                                                           for value in kept)):
            found.add(ended)
            taken |= kept
    return found


class Frontier:
    """Which values of `program` XLA's fusion for `platform` keeps a buffer of: each but one a primitive
    it recomputes in its consumers computes (`recomputed`), which reads those it is computed from in
    its place, and one a layout primitive (`bitcast`) makes of another's buffer, which XLA's layout
    assignment lays that buffer out for; the program's outputs, where it returns them (`returned`),
    always but as such a view; neither its inputs nor its literals."""

    def __init__(self, program: jcore.Jaxpr, platform: str, *, returned: bool):
        self.platform = platform
        self.producers = {out: equation for equation in program.eqns for out in equation.outvars}
        self.inputs = {*program.invars, *program.constvars}
        self.outputs = {out for out in program.outvars if computed(out)} if returned else set()
        self.readers = collections.Counter(value for equation in program.eqns
                                           for value in {read for read in equation.invars if computed(read)})
        self.readers.update({value for value in program.outvars if computed(value)})
        self.reads: dict[jcore.Var, frozenset[jcore.Var]] = {}
        # An optimization barrier defines no value: its results are its operands' own
        # (`HloDataflowAnalysis::UpdateOptimizationBarrierValueSet`,
        # xla/hlo/analysis/hlo_dataflow_analysis.cc).
        self.forwards = {out: operand for equation in program.eqns
                         if named(equation.primitive) == "optimization_barrier"
                         for out, operand in zip(equation.outvars, equation.invars, strict=True)}
        self.barred = {operand for operand in self.forwards.values() if computed(operand)}

    def stored(self, operand) -> bool:
        """Whether `operand` is held in a buffer: an input, a value computed into one, or a view of one."""
        return computed(operand) and (operand in self.inputs or not self.fused(operand)
                                      or self.view(operand))

    def view(self, value) -> bool:
        """Whether `value` is another value's storage: a shape primitive's of a value held or read
        (`bitcast`), or a transpose's of a value the program computes into a buffer and reads nowhere
        else (`TRANSPOSES`)."""
        equation = self.producers.get(value)
        if value in self.forwards:
            return self.stored(self.forwards[value])
        if equation is None or not equation.invars or not computed(equation.invars[0]):
            return False
        operand = equation.invars[0]
        if bitcast(equation):
            return operand in self.inputs or not self.fused(operand) or self.view(operand)
        # XLA lays a value out once, so a transpose is its storage only where it is its one reader.
        return (named(equation.primitive) in TRANSPOSES and operand not in self.inputs
                and self.readers[operand] == 1 and (not self.fused(operand) or self.view(operand)))

    def fused(self, value) -> bool:
        """Whether `value` holds no buffer of its own: recomputed where it is read, or a view; a value an
        optimization barrier reads is computed into a buffer of its own, which the barrier's result is
        and every other reader reads (`forwards`), unless it is a view of another's (`view`)."""
        if not computed(value) or value in self.inputs:
            return False
        if value in self.barred:
            # A barrier's operand holds a buffer unless it is another's storage.
            return self.view(value)
        equation = self.producers.get(value)
        if value in self.forwards:
            return self.stored(self.forwards[value])
        if equation is None or not recomputed(equation, self.platform):
            return False
        return value not in self.outputs or self.view(value)

    def buffered(self, value) -> bool:
        """Whether `value` is computed into a buffer of its own."""
        return computed(value) and value not in self.inputs and not self.fused(value)

    def made_for(self, value) -> bool:
        """Whether reading `value` from memory, as a matrix product or a call does, makes a buffer of it
        for the read: a fused value's, but a view of storage (`view`)."""
        return self.fused(value) and not self.view(value)

    def fusing(self, equation: jcore.JaxprEqn) -> bool:
        """Whether `equation` computes its result within a fusion that recomputes its operands' fused
        chains as it reads them: an elementwise one's (`recomputed`) or a reduction's, a variadic
        `reduce` among them, which XLA fuses into its own kernel whatever its operands' number
        (`PriorityFusion::ChooseKind`, `CanFuseWithNativeEmitter`,
        xla/backends/gpu/transforms/priority_fusion.cc), or a scatter's, an input fusion as a reduction's
        is (`IsInputFusibleScatter`, xla/service/gpu/gpu_fusible.cc), its base computed apart into its
        result and its updates read from its base apart (`unfused`):
        Qwen3.5-4B's tied embedding's gradient, a scatter-add into zeros, was derived holding the zeros
        beside it, 2.37 GiB over XLA's 4.76 at its step's peak under 'full' (row 4089)."""
        name = named(equation.primitive)
        return (recomputed(equation, self.platform) or name == "reduce"
                or name.startswith(("reduce_", "arg", "scatter")))

    def roots(self, values: Iterable) -> set[jcore.Var]:
        """The buffers reading `values` holds (`reading`) and the inputs they are computed from or view: what
        they depend on, the program's inputs among it."""
        found: set[jcore.Var] = set()
        pending = [value for value in values if computed(value)]
        seen: set[jcore.Var] = set()
        while pending:
            value = pending.pop()
            if value in seen:
                continue
            seen.add(value)
            if value in self.inputs or not self.fused(value):
                found.add(value)
                continue
            read = [self.forwards[value]] if value in self.forwards else self.producers[value].invars
            pending.extend(each for each in read if computed(each))
        return found

    def reading(self, values: Iterable) -> set[jcore.Var]:
        """The buffers reading `values` holds: each value's own, or for one fused, those it is
        recomputed from or views."""
        found: set[jcore.Var] = set()
        for value in values:
            if not computed(value) or value in self.inputs:
                continue
            if value not in self.reads:
                # A barrier's result reads its own operand alone, the rest of the barrier's not.
                read = [self.forwards[value]] if value in self.forwards else self.producers[value].invars
                self.reads[value] = (frozenset(self.reading(read)) if self.fused(value)
                                     else frozenset([value]))
            found |= self.reads[value]
        return found


def device_share(value: jcore.Var) -> int:
    """The device bytes `value` holds: none for one in pinned host memory or no array (a token)."""
    aval = value.aval
    if not isinstance(aval, ShapedArray) or aval.memory_space != jax.memory.Space.Device:
        return 0
    return math.prod(aval.shape) * aval.dtype.itemsize


def recomputed(equation: jcore.JaxprEqn, platform: str) -> bool:
    """Whether XLA's fusion on `platform` recomputes what `equation` computes in each consumer rather than
    keep it: on a GPU every elementwise primitive (`ELEMENTWISE`), elsewhere the cheap ones (`CHEAP`, and
    `REAL_CHEAP` over real values). A program it reads has its calls inlined (`compiled`)."""
    name = named(equation.primitive)
    if bitcast(equation):
        # A bitcast holds no buffer of its own, on any platform.
        return True
    if platform == "gpu":
        return name in ELEMENTWISE
    if name in REAL_CHEAP:
        return not any(isinstance(out.aval, ShapedArray) and np.issubdtype(out.aval.dtype, np.complexfloating)
                       for out in equation.outvars)
    return name in CHEAP


class Workspace(NamedTuple):
    """The workspaces a step's cuBLAS matmuls hold, one buffer at a time (`gemm_workspace`): `each`
    matmul's, as its config sets it, and `sliced`, a matmul's that XLA fuses with its loop's slice of its
    operand or its result (`Slicing.fused`), which keeps the one the GEMM rewriter gave it."""

    each: int
    sliced: int


DEFAULT_WORKSPACE = 4 << 20
"""The workspace of cuBLASLt's default config (`CublasLtBackend::GetDefaultConfig`,
xla/backends/gpu/autotuner/cublaslt.cc; `GemmConfig::kDefaultWorkspace`, xla/service/gpu/matmul_utils.h)."""


def gemm_workspace(device) -> Workspace:
    """The workspaces XLA hands the cuBLAS matmuls of a step on `device` (`Workspace`): the GEMM
    rewriter gives each 32 MiB from Hopper on, 4 MiB before (`GemmWorkspaceRewriteVisitor`,
    xla/backends/gpu/transforms/gemm_rewriter.cc; `GemmConfig::kHopperWorkspace`, matmul_utils.h). On a
    device the compile autotunes on its config keeps that (`CublasLtBackend::ApplyConfig`, cublaslt.cc);
    with none (a compile-only GPU, `deviceless`), each matmul takes the
    default config's (`DEFAULT_WORKSPACE`: `GetConfigAssignerOptions`,
    xla/service/gpu/autotuning/config_assigner_pass.cc, turns autotuning off without a device), but one
    in a dynamic-slice fusion, whose computation the config assigner does not visit
    (`ExtractEquivalentInstructions`, xla/backends/autotuner/hlo_extractor.cc). None where the device is
    no CUDA GPU, whose matmuls take none."""
    capability = _compute_capability(device) if device.platform == 'gpu' else None
    if not capability:
        return Workspace(0, 0)
    rewritten = (32 if int(capability.split('.')[0]) >= 9 else 4) << 20
    return Workspace(DEFAULT_WORKSPACE if deviceless(device) else rewritten, rewritten)


def late_weight_gradients(device) -> bool:
    """Whether XLA runs a step's weight gradients on `device` after the whole backward where its head takes
    its gradient a tile at a time, unfused (`GpuTarget.late_weight_gradients`, `sunk`)."""
    capability = _compute_capability(device) if device.platform == 'gpu' else None
    return bool(capability) and gpu_target(f"sm{capability.replace('.', '')}").late_weight_gradients


def traced_loss[LossT, EffectsT](objective: Objective[LossT, EffectsT], state: TrainState, window: int,
                                 split: int) -> tuple[Callable, tuple]:
    """The loss the step evaluates for `objective` over `state`, its update a `window` of microbatches,
    each step read in `split`, with the abstract arguments it is traced over (but the batch): the
    variables, the microstep, the key, the step, the EMA and the updates committed. Shapes
    and dtypes only: a resident frozen leaf sits in another memory space than the moving ones, and
    the loss the realization runs reads a snapshot in one space. The step's key is drawn inside the
    trace, so an abstract state compiles a step as a placed one does."""
    variables = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.variables)

    def loss(variables, batch, microstep, key, step, ema, updates):
        # The step's first attempt of its first part, as the transaction reads it (`attempt_step`).
        current = dataclasses.replace(state, microstep=microstep, key=key, step=step, updates=updates)
        return objective._loss(variables, batch, attempt_step(
            current, variables, ema, batch, split=split, slots=window, attempt=step * split, part=0,
            slot=microstep % window))

    return loss, (variables, state.microstep, state.key, state.step, state.ema, state.updates)


def candidate_footprint[LossT, EffectsT](objective: Objective[LossT, EffectsT],
                                          micro: Batch, window: int, prepared: TrainState,
                                          held: int, target, split: int, consumed: Callable[[], bool],
                                          staged: int, capacity: int | None = None) -> Footprint:
    """What a step over microbatches of `micro`'s shapes, its state `prepared` as placed, takes of a
    device like `target` beside that state (`Footprint`): its loss's residuals and its gradient
    program's peak at one microbatch, as that device's compiler keeps them and its GEMMs' workspace
    (`gemm_workspace`), each gradient held while it is made where the step reads it there (`consumed`,
    `transaction.consumed_where_made`, asked only where the peak is derived), its trained parameters'
    gradient, laid out as they are, and
    `held`, and `staged`, the bytes of its host-resident commit fields its own staged cut holds at
    once (`dew.training.staging.measured`, as the step is traced). Over
    several devices that split the microbatch's rows alone (data and fully sharded data parallelism,
    `row_axes`) the residuals are one device's: the loss traced as one device's step, over a mesh of the
    run's axes each of one device, at that device's share of the microbatch (each leaf's shard,
    `shard_shape`), its parameters whole as those layouts gather them. A layout that splits more than the
    rows (stages, tensor, sequence) derives no residuals, which its compile judges, nor does a step for a
    device whose memory nothing judges at `capacity` (`judged`: a CPU, a compile-only GPU no capacity is
    given), whose walk reads no rung ('unstated', `FitSearch.walked`)."""
    mesh = jax.sharding.get_abstract_mesh()
    spread = set() if mesh.empty else {name for name, size in mesh.shape.items() if size > 1}
    # Data and fully sharded data parallelism split the rows alone; an expert axis splits the experts too.
    rows = set(row_axes(rows_of(micro), mesh=mesh)) & {DATA_AXIS, FSDP_AXIS} if spread else set()
    if not spread <= rows or not judged(target, capacity):
        residuals = peak = reads = None
    else:
        loss, (variables, *rest) = traced_loss(objective, prepared, window, split)
        variables = forward_variables(variables, prepared.compute)

        def trained(params, variables, batch, *rest):
            return loss({**variables, "params": params}, batch, *rest)

        share = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(
            leaf.shape if getattr(leaf, "sharding", None) is None else leaf.sharding.shard_shape(leaf.shape),
            leaf.dtype), micro)
        # The arguments' shapes alone, as one device holds them: no sharding of the run's mesh.
        arguments = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype),
                                 (variables, share, *rest))
        # The run's own first device, which a kernel's choice reads as the run's do.
        placed = jax.sharding.get_mesh()
        device = None if mesh.empty or placed.empty else placed.devices.flat[0]
        with contextlib.nullcontext() if device is None else jax.set_mesh(jax.sharding.Mesh(
                np.asarray([device]).reshape((1,) * len(mesh.axis_names)), mesh.axis_names,
                axis_types=mesh.axis_types)):
            lowering = lowering_platform(target)
            reads = consumed()
            # A head that takes its gradient in the backward a tile at a time, unfused, has XLA run the
            # weight gradients after the backward on some GPUs (`late_weight_gradients`).
            late = (head_tile_of(objective) is not None and not head_fused_of(objective)
                    and late_weight_gradients(target))
            residuals, peak = derived_terms(trained, arguments[0]["params"], arguments,
                                            platform=target.platform, workspace=gemm_workspace(target),
                                            lowering=lowering, consumed=reads, deferred=late)
    resident = {name: device_bytes(value) for name, value in {
        "variables": prepared.variables, "optimizer": prepared.opt_state, "ema": prepared.ema,
        "accumulation": prepared.accumulation, "compute": prepared.compute, "groups": prepared.inner}.items()}
    return Footprint(residuals, device_bytes(prepared.variables["params"]), held, resident,
                     None if peak is None else peak.bytes, peak is None or peak.exact, staged, reads)
