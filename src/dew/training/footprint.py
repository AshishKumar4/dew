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
from dew.devices import xla_flag
from dew.distributed.placement import placed_bytes
from dew.mesh import DATA_AXIS, FSDP_AXIS, row_axes
from dew.nn.kernels.generation import _compute_capability, gpu_target
from dew.nn.programs import bodies, computed
from dew.objectives.base import Objective
from dew.state import TrainState
from dew.training.compiled_program import (
    Slicing,
    bitcast,
    compiled,
    deviceless,
    dimensions,
    lowering_platform,
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

CUBLAS_OUTPUTS = frozenset({"float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32", "float64",
                            "complex64"})
"""The output types cuBLAS computes a product of a matrix and a vector or two matrices in
(`IsCublasSupportedMatMul`)."""

CUBLAS_MATRIX_OUTPUTS = frozenset({"float8_e3m4", "float8_e4m3", "float8_e4m3fnuz", "float8_e5m2fnuz",
                                   "complex128"})
"""The output types cuBLAS computes a product of two matrices alone in (`IsCublasSupportedMatMul`)."""

LOWERED = {"gpu": "cuda"}
"""The platform JAX lowers for where a derivation names only its kind (`lowering_platform`)."""

LOOPS = frozenset({"scan", "while"})

PRODUCTS = frozenset({"dot_general", "conv_general_dilated", "ragged_dot_general"})
"""The products a weight's gradient is made by, past which a value is a cotangent the backward reads on
(`sunk`)."""
"""The primitives that run their body turn after turn, their results held across the turns."""

GEMM_REWRITE_SIZE = 100
"""The least a product's non-contracting sizes summed times its contracting size is for XLA to hand it to
cuBLAS (`xla_gpu_gemm_rewrite_size_threshold`, `kDefaultMinGemmRewriteSize` in
xla/debug_options_flags.cc); below it XLA's own emitters compute it (`IsMatrixMultiplicationTooSmall
ForRewriting`, xla/service/gpu/matmul_utils.cc)."""


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
    lowered = compiled(program, lowering or LOWERED.get(platform, platform))
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
    return _kept(compiled(program, lowering or LOWERED.get(platform, platform)), count, shapes, platform)


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
    return _peak(compiled(program, lowering or LOWERED.get(platform, platform)), count, platform, workspace,
                 consumed, deferred)


def _kept(lowered: jcore.Jaxpr, count: int, shapes: Sequence, platform: str) -> int:
    """`residual_bytes` of `traced_gradient`'s program as it is `lowered` (`compiled`), its outputs past
    the first `count` the residuals."""
    return _residuals(lowered.replace(outvars=lowered.outvars[count:]), shapes, platform)


def _peak(lowered: jcore.Jaxpr, count: int, platform: str, workspace: Workspace, consumed: bool = False,
          deferred: bool = False) -> Peak:
    """`gradient_peak` of `traced_gradient`'s program as it is `lowered` (`compiled`), its first `count`
    outputs the gradient."""
    gradient = hoisted(lowered.replace(outvars=lowered.outvars[:count]))
    return Peak(peak_bytes(sunk(gradient) if deferred else gradient, platform, workspace, consumed=consumed),
                not bounded_below(gradient, platform))


def hoisted(program: jcore.Jaxpr) -> jcore.Jaxpr:
    """`program` with each output computed from literals alone run first, with the equations that compute
    it, in their order: a constant the program returns and no equation of its work reads, which XLA
    schedules ahead of that work and holds to its reader past the program. The gradient of a parameter
    the loss reads none of is such a constant, a broadcast zero JAX writes at the backward's end, which
    XLA made during the forward: Qwen3.5-4B's step at 4 x 2048 tokens under 'minimal_with_scan_and_conv',
    compiled for an H100, held its multi-token head's 25 zero gradients, 0.48 GiB, from step 2,480 of its
    schedule to the update at 15,091, its peak at 4,863 among them (row 4089, its buffer dump). A
    constant the work reads (a loop's zero carry) is made where the work needs it, as XLA makes it."""
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
                reader in late and program.eqns[reader].primitive.name not in PRODUCTS
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
        equation.primitive.name == "ragged_dot_general"
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
    that makes it runs, as an update that reads it there holds it (`gradient_peak`). No schedule XLA
    picks is modelled: a compile confirms."""
    frontier = Frontier(program, platform, returned=True)
    last = {kept: index for index, equation in enumerate(program.eqns)
            for kept in frontier.reading(equation.invars)}
    if consumed:
        # The program's outputs are read where they are made, by an update that writes the buffers the
        # state donates (the optimizer's, or the accumulator's): each is held while its equation runs.
        made_at = {out: index for index, equation in enumerate(program.eqns) for out in equation.outvars}
        for kept in frontier.reading(program.outvars):
            last.setdefault(kept, made_at.get(kept, 0))
    else:
        last.update({kept: len(program.eqns) for kept in frontier.reading(program.outvars)})
    sliced = (slicing or Slicing()).within(program)
    live, held, most = 0, set(), 0
    for index, equation in enumerate(program.eqns):
        made = [out for out in equation.outvars
                if out in last and frontier.buffered(out) and out not in carried]
        made_bytes = sum(map(device_share, made))
        if made or not frontier.fusing(equation):
            inner = max((peak_bytes(body, platform, workspace, Slicing.entering(equation, body),
                                    returned_carry(equation, body)) for body in bodies(equation)), default=0)
            # A barrier's results are its operands' own storage: the buffers made for its fused operands
            # are the ones it holds, not a second set read beside them.
            barrier = equation.primitive.name == "optimization_barrier"
            operands = 0 if frontier.fusing(equation) or barrier else sum(
                device_share(value) for value in {value for value in equation.invars if computed(value)}
                if frontier.made_for(value))
            products = 0 if not cublas_product(equation) else (
                workspace.sliced if sliced.fused(equation) else workspace.each)
            # A loop's results (its stacked outputs, its carry) are held beside each turn's own; a call's
            # results are what its body returns.
            running = (looped(frontier, last, index, equation, made, carried) + inner
                       if equation.primitive.name in LOOPS else max(inner, made_bytes))
            most = max(most, live + running + operands + products)
        held.update(made)
        live += made_bytes
        for done in [value for value in held if last[value] == index]:
            held.discard(done)
            live -= device_share(done)
    return most


def carry(equation: jcore.JaxprEqn) -> list[tuple]:
    """Each value the loop `equation` carries, as its initial value, the loop's result of it, and
    whether its body changes it, returning another value than the parameter it took. XLA keeps the
    initial value, the body's parameter and returned value of it and the loop's result in one buffer
    (`ComputeAliasedValues`, xla/hlo/analysis/hlo_alias_analysis.cc), copying the initial value into a
    buffer of the loop's where it is an input or still read (copy insertion,
    xla/service/copy_insertion.cc)."""
    if equation.primitive.name == "while":
        start = equation.params["cond_nconsts"] + equation.params["body_nconsts"]
        body = equation.params["body_jaxpr"].jaxpr
        taken = body.invars[equation.params["body_nconsts"]:]
        changed = [returned is not took for returned, took in zip(body.outvars, taken, strict=True)]
        return list(zip(equation.invars[start:], equation.outvars, changed, strict=True))
    if equation.primitive.name == "scan":
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
    if equation.primitive.name == "while" and body is equation.params["body_jaxpr"].jaxpr:
        return frozenset(value for value in body.outvars if computed(value))
    if equation.primitive.name == "scan" and body is equation.params["jaxpr"].jaxpr:
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
                         if equation.primitive.name == "optimization_barrier"
                         for out, operand in zip(equation.outvars, equation.invars, strict=True)}

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
        return (equation.primitive.name in TRANSPOSES and operand not in self.inputs
                and self.readers[operand] == 1 and (not self.fused(operand) or self.view(operand)))

    def fused(self, value) -> bool:
        """Whether `value` holds no buffer of its own: recomputed where it is read, or a view."""
        if not computed(value) or value in self.inputs:
            return False
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
        xla/backends/gpu/transforms/priority_fusion.cc)."""
        name = equation.primitive.name
        return recomputed(equation, self.platform) or name == "reduce" or name.startswith(("reduce_", "arg"))

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


def cublas_product(equation: jcore.JaxprEqn) -> bool:
    """Whether XLA hands `equation`, a matrix product, to cuBLAS, which takes a workspace, as the GEMM
    rewriter asks (xla/backends/gpu/transforms/gemm_rewriter.cc's `HandleDot`): one each of whose
    operands contracts a dimension, one at least a matrix (its non-contracting sizes not all one), whose
    output type cuBLAS computes (`IsCublasSupportedMatMul`, xla/service/gpu/ir_emission_utils.cc, its
    dimensions as XLA's dot decomposition leaves them, one of each kind), and that XLA's own emitters
    do not take for being too small (`GEMM_REWRITE_SIZE`, or the run's
    `--xla_gpu_gemm_rewrite_size_threshold`; `IsDotSupportedByClassicalEmitters`, matmul_utils.cc). A
    product XLA's Triton GEMM fusions take instead takes none, which this does not tell apart: it
    holds the workspace."""
    if equation.primitive.name != "dot_general":
        return False
    (left_contracting, right_contracting), (left_batch, right_batch) = equation.params["dimension_numbers"]
    left, right = dimensions(equation.invars[0]), dimensions(equation.invars[1])
    if not left_contracting or not right_contracting:
        return False
    contracting = int(math.prod(left[axis] for axis in left_contracting))
    kept_left = int(math.prod(size for axis, size in enumerate(left)
                              if axis not in (*left_contracting, *left_batch)))
    kept_right = int(math.prod(size for axis, size in enumerate(right)
                               if axis not in (*right_contracting, *right_batch)))
    matrices = (kept_left != 1) + (kept_right != 1)
    produced = str(element_type(equation.outvars[0]))
    operands = tuple(str(element_type(value)) for value in equation.invars)
    supported = (produced in CUBLAS_OUTPUTS or (produced == "int32" and operands == ("int8", "int8"))
                 or (produced in CUBLAS_MATRIX_OUTPUTS and matrices == 2))
    if matrices == 0 or not supported:
        return False
    threshold = int(xla_flag("xla_gpu_gemm_rewrite_size_threshold") or GEMM_REWRITE_SIZE)
    classical = produced in ("float16", "float32", "bfloat16") and emitted_algorithm(equation)
    return (kept_left + kept_right) * contracting >= threshold or not classical


def emitted_algorithm(equation: jcore.JaxprEqn) -> bool:
    """Whether XLA's own emitters compute `equation`'s product at its algorithm
    (`IsSupportedByElementalIrEmitter`, xla/service/algorithm_util.cc): an unset one, or bf16 or f32
    operands accumulated in f32."""
    precision = equation.params.get("precision")
    if not isinstance(precision, jax.lax.DotAlgorithmPreset):
        return True
    return precision.name in ("DEFAULT", "BF16_BF16_F32", "F32_F32_F32")


def element_type(value) -> np.dtype | None:
    """`value`'s element type, none where it is no array."""
    return value.aval.dtype if isinstance(value.aval, ShapedArray) else None


def recomputed(equation: jcore.JaxprEqn, platform: str) -> bool:
    """Whether XLA's fusion on `platform` recomputes what `equation` computes in each consumer rather than
    keep it: on a GPU every elementwise primitive (`ELEMENTWISE`), elsewhere the cheap ones (`CHEAP`, and
    `REAL_CHEAP` over real values). A program it reads has its calls inlined (`compiled`)."""
    name = equation.primitive.name
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
