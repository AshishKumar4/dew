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
from typing import NamedTuple, TypeGuard

import jax
import jax.extend.core as jcore
import numpy as np
from jax.core import ShapedArray

from dew._records import JSON
from dew.batch import Batch
from dew.data.dataset import rows_of
from dew.devices import xla_flag
from dew.distributed.placement import placed_bytes
from dew.mesh import DATA_AXIS, FSDP_AXIS, row_axes
from dew.nn.kernels.generation import _compute_capability
from dew.objectives.base import Objective, Step
from dew.optim import clips_qk
from dew.state import TrainState
from dew.training.narrow import forward_variables
from dew.training.transaction import with_ema


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

    def record(self) -> dict[str, JSON]:
        """The terms traced, not an admission decision: XLA's fit check remains authoritative."""
        return {"residuals": self.residuals, "gradients": self.gradients, "held": self.held,
                "resident": dict(self.resident), "peak": self.peak, "peak_exact": self.exact}


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
    "add", "and", "bitcast_convert_type", "broadcast_in_dim", "ceil", "clamp", "clz", "eq", "ne", "lt", "le",
    "gt", "ge", "complex", "concatenate", "convert_element_type", "copy", "copy_p", "dynamic_slice",
    "dynamic_update_slice", "floor", "imag", "iota", "is_finite", "max", "min", "mul", "neg", "not", "or",
    "xor", "pad", "population_count", "real", "reduce_precision", "reshape", "squeeze", "expand_dims", "rev",
    "round", "select_n", "shift_left", "shift_right_arithmetic", "shift_right_logical", "slice", "sub",
    "transpose", "integer_pow", "square", "name"})
"""The primitives XLA's CPU fusion takes for cheap, which it duplicates into each consumer rather than
keep a buffer of (`InstructionFusion::IsExpensive`, xla/service/instruction_fusion.cc: every elementwise
op but division, the transcendentals and their kin, and the layout ops), an integer power and a square
lowering to its multiplies, and a checkpoint's name, which computes nothing."""

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


VIEWS = frozenset({"reshape", "squeeze", "expand_dims"})
"""The shape primitives, whose result is its operand's storage, its elements in the same order (a
bitcast, in XLA's default row-major layout)."""

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

LOOPS = frozenset({"scan", "while"})
"""The primitives that run their body turn after turn, their results held across the turns."""

GEMM_REWRITE_SIZE = 100
"""The least a product's non-contracting sizes summed times its contracting size is for XLA to hand it to
cuBLAS (`xla_gpu_gemm_rewrite_size_threshold`, `kDefaultMinGemmRewriteSize` in
xla/debug_options_flags.cc); below it XLA's own emitters compute it (`IsMatrixMultiplicationTooSmall
ForRewriting`, xla/service/gpu/matmul_utils.cc)."""


def residual_bytes(loss: Callable, params, arguments: Sequence, *, platform: str) -> int:
    """The device bytes the pullback of `loss(params, *arguments)` in `params` keeps, a device's share,
    as XLA's fusion for `platform` keeps them: each value its traced program (`jax.make_jaxpr` of
    `jax.vjp`'s pullback, as XLA compiles it: `compiled`) closes over counts the values it is computed
    from without a buffer (`Frontier.reading`), once, but those the program reads, which the state and
    the batch already hold, and literals; a value in pinned host memory counts none. A value the
    pullback closes over keeps its own traced sharding; one whose trace states none, and every other
    value, count whole. Matching an axis's size to the batch size cannot identify its sharding: a weight
    or a scan's layer axis can have that size too."""
    def pullback(params, *arguments):
        return jax.vjp(lambda trainable: loss(trainable, *arguments), params, has_aux=True)[1]

    closed, shapes = jax.make_jaxpr(pullback, return_shape=True)(params, *arguments)
    program = compiled(closed.jaxpr, platform)
    placed = {value: shape for value, shape in zip(program.outvars, jax.tree.leaves(shapes), strict=True)
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


def gradient_peak(loss: Callable, params, arguments: Sequence, *, platform: str,
                  workspace: Workspace) -> Peak:
    """The most device bytes the gradient of `loss(params, *arguments)` in `params` holds at once, a
    device's share, as XLA's fusion for `platform` keeps its values and each matrix product takes
    `workspace` bytes beside them (`peak_bytes`), and whether that is the program's own or a bound from
    below (`bounded_below`): the program `jax.vjp`'s pullback runs on a cotangent of ones, traced whole,
    forward and backward."""
    def gradient(params, *arguments):
        outputs, pullback, _ = jax.vjp(lambda trainable: loss(trainable, *arguments), params, has_aux=True)
        return pullback(jax.tree.map(jax.numpy.ones_like, outputs))

    program = jax.make_jaxpr(gradient)(params, *arguments).jaxpr
    return Peak(peak_bytes(compiled(program, platform), platform, workspace),
                not bounded_below(program, platform))


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
               slicing: Slicing | None = None) -> int:
    """The most bytes `program` holds at once beside its inputs, its equations run in their traced
    order. A value XLA keeps a buffer of (`Frontier`) is held from the equation that computes it to the
    last that reads it, an output to the end. An equation that takes its operands from memory
    (`Frontier.fusing` says no: a matrix product, a call, a loop) holds beside them each operand a fused
    chain computes, while it runs, and a cuBLAS matrix product its workspace too (`Workspace`); a
    call's, a loop's or a checkpoint's own peak, its results among them, stands in for its results while
    it runs, its body read with what it slices of the loop it runs in (`slicing`). No schedule XLA picks
    is modelled: a compile confirms."""
    frontier = Frontier(program, platform, returned=True)
    last = {kept: index for index, equation in enumerate(program.eqns)
            for kept in frontier.reading(equation.invars)}
    last.update({kept: len(program.eqns) for kept in frontier.reading(program.outvars)})
    sliced = (slicing or Slicing()).within(program, frontier)
    live, held, most = 0, set(), 0
    for index, equation in enumerate(program.eqns):
        made = [out for out in equation.outvars if out in last and frontier.buffered(out)]
        made_bytes = sum(map(device_share, made))
        if made or not frontier.fusing(equation):
            inner = max((peak_bytes(body, platform, workspace, Slicing.entering(equation, body, frontier))
                         for body in bodies(equation)), default=0)
            operands = 0 if frontier.fusing(equation) else sum(
                device_share(value) for value in {value for value in equation.invars if computed(value)}
                if frontier.made_for(value))
            products = 0 if not cublas_product(equation) else (
                workspace.sliced if sliced.fused(equation) else workspace.each)
            # A loop's results (its stacked outputs, its carry) are held beside each turn's own; a call's
            # results are what its body returns.
            running = made_bytes + inner if equation.primitive.name in LOOPS else max(inner, made_bytes)
            most = max(most, live + running + operands + products)
        held.update(made)
        live += made_bytes
        for done in [value for value in held if last[value] == index]:
            held.discard(done)
            live -= device_share(done)
    return most


ALIGNMENT = 256
"""The alignment of every buffer XLA allocates on a GPU (`kXlaAllocatedBufferAlignBytes`,
xla/service/gpu/gpu_constants.h), which a slice fused into a matmul's call must keep."""


@dataclasses.dataclass(frozen=True)
class Slicing:
    """What a program a loop runs slices of the loop's buffers, which XLA fuses with the cuBLAS matmuls
    that read or write them (`DynamicSliceFusionRewriterV2`, xla/backends/gpu/transforms/
    dynamic_slice_fusion_rewriter_v2.cc: a matmul whose operand is, through bitcasts alone, a slice
    `DynamicSliceAnnotator` annotated, or whose one user updates such a slice into the loop's result;
    `AnalyzeDynamicSlice`, dynamic_slice_analysis.cc, annotates a contiguous slice whose offsets the
    loop's induction variable decides). Each turn's slice of a scan's stacked inputs is such a slice
    (`sliced`), its stacked outputs are updated so (`updated`), and so are the slices a loop's counter
    offsets (`counter`, the values computed from it alone `counted`), contiguous and aligned
    (`contiguous_aligned`)."""

    counter: jcore.Var | None = None
    counted: frozenset = frozenset()
    sliced: frozenset = frozenset()
    updated: frozenset = frozenset()

    def within(self, program: jcore.Jaxpr, frontier: Frontier) -> Slicing:
        """This slicing completed over `program`'s equations: the values computed from the counter and
        literals alone, the slices they offset, and the views of a slice, which XLA lays out as bitcasts
        of it or folds into the matmul that reads it (`TransposeFolding`, xla/service/
        transpose_folding.cc)."""
        counted = {self.counter} if self.counter is not None else set()
        sliced, updated = set(self.sliced), set(self.updated)
        for equation in program.eqns:
            name = equation.primitive.name
            reads = [read for read in equation.invars if computed(read)]
            if name in ("dynamic_slice", "dynamic_update_slice"):
                offsets = [offset for offset in equation.invars[1 if name == "dynamic_slice" else 2:]
                           if computed(offset)]
                if offsets and all(offset in counted for offset in offsets) and contiguous_aligned(equation):
                    if name == "dynamic_slice":
                        sliced.update(equation.outvars)
                    elif computed(update := equation.invars[1]) and frontier.readers[update] == 1:
                        updated.add(update)
            elif name in VIEWS | TRANSPOSES and reads and reads[0] in sliced:
                sliced.update(equation.outvars)
            elif reads and all(read in counted for read in reads) and not bodies(equation):
                counted.update(equation.outvars)
        return dataclasses.replace(self, counted=frozenset(counted), sliced=frozenset(sliced),
                                   updated=frozenset(updated))

    def fused(self, equation: jcore.JaxprEqn) -> bool:
        """Whether XLA fuses `equation`, a cuBLAS matmul, with a slice of its loop's buffers: one of its
        operands is one, or its result is updated as one."""
        return (any(operand in self.sliced for operand in equation.invars)
                or any(out in self.updated for out in equation.outvars))

    @staticmethod
    def entering(equation: jcore.JaxprEqn, body: jcore.Jaxpr, frontier: Frontier) -> Slicing:
        """The slicing `body`, a program `equation` runs, reads of its loop: for a scan, each turn's slice
        of its stacked inputs, each stacked output's update where the body computes it for that alone,
        and its first carry where its turns count it by one (the counter a `fori_loop` over fixed bounds
        carries); nothing for any other program, which no loop of its own slices."""
        if equation.primitive.name != "scan":
            return Slicing()
        _, carried, stacked = equation.params["ft_in"].update(body.invars).unpack()
        returned, outputs = equation.params["ft_out"].update(body.outvars).unpack()
        body_frontier = Frontier(body, frontier.platform, returned=True)
        sliced = {value for value in stacked if device_share(value) % ALIGNMENT == 0}
        updated = {value for value in outputs if computed(value) and body_frontier.readers[value] == 1
                   and device_share(value) % ALIGNMENT == 0}
        counter = None
        if carried and returned and carried[0].aval.dtype == np.int32:
            stepped = any(step.outvars == [returned[0]] and step.primitive.name == "add"
                          and step.invars[0] is carried[0] and isinstance(step.invars[1], jcore.Literal)
                          and step.invars[1].val == 1 for step in body.eqns)
            counter = carried[0] if stepped else None
        return Slicing(counter, sliced=frozenset(sliced), updated=frozenset(updated))


def contiguous_aligned(equation: jcore.JaxprEqn) -> bool:
    """Whether `equation`, a dynamic slice or update, takes a contiguous extent of its buffer whose bytes
    keep its alignment (`IsContiguousSlice`, xla/service/gpu/ir_emission_utils.cc: every dimension after
    the first it cuts whole, its row-major layout's; `IsAlignedSlice`, dynamic_slice_fusion_rewriter_v2.cc:
    a multiple of `ALIGNMENT` bytes)."""
    whole = dimensions(equation.invars[0])
    taken = equation.outvars[0] if equation.primitive.name == "dynamic_slice" else equation.invars[1]
    part = dimensions(taken)
    if not whole or len(whole) != len(part):
        return False
    cut = next((axis for axis, (full, size) in enumerate(zip(whole, part, strict=True)) if size < full), None)
    contiguous = cut is None or all(size == 1 for size in part[:cut])
    element = element_type(equation.invars[0])
    return contiguous and element is not None and math.prod(part) * element.itemsize % ALIGNMENT == 0


class Frontier:
    """Which values of `program` XLA's fusion for `platform` keeps a buffer of: each but one a primitive
    it recomputes in its consumers computes (`recomputed`), which reads those it is computed from in
    its place, and one a layout primitive (`VIEWS`) makes of another's buffer, which XLA's layout
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

    def view(self, value) -> bool:
        """Whether `value` is another value's storage: a shape primitive's of a value held or read
        (`VIEWS`), or a transpose's of a value the program computes into a buffer and reads nowhere
        else (`TRANSPOSES`)."""
        equation = self.producers.get(value)
        if equation is None or not equation.invars or not computed(equation.invars[0]):
            return False
        operand = equation.invars[0]
        if equation.primitive.name in VIEWS:
            return operand in self.inputs or not self.fused(operand) or self.view(operand)
        # XLA lays a value out once, so a transpose is its storage only where it is its one reader.
        return (equation.primitive.name in TRANSPOSES and operand not in self.inputs
                and self.readers[operand] == 1 and (not self.fused(operand) or self.view(operand)))

    def fused(self, value) -> bool:
        """Whether `value` holds no buffer of its own: recomputed where it is read, or a view."""
        if not computed(value) or value in self.inputs:
            return False
        equation = self.producers.get(value)
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
        chains as it reads them: an elementwise one's (`recomputed`) or a reduction's."""
        return recomputed(equation, self.platform) or equation.primitive.name.startswith(("reduce_", "arg"))

    def reading(self, values: Iterable) -> set[jcore.Var]:
        """The buffers reading `values` holds: each value's own, or for one fused, those it is
        recomputed from or views."""
        found: set[jcore.Var] = set()
        for value in values:
            if not computed(value) or value in self.inputs:
                continue
            if value not in self.reads:
                self.reads[value] = (frozenset(self.reading(self.producers[value].invars))
                                     if self.fused(value) else frozenset([value]))
            found |= self.reads[value]
        return found


def compiled(program: jcore.Jaxpr, platform: str) -> jcore.Jaxpr:
    """`program` as JAX lowers it for `platform` and XLA compiles it: each `shard_map` over manual axes
    of one device inline, as its body's equations (`_shard_map_lowering_shardy`, jax/_src/shard_map.py,
    makes no manual computation of it, so XLA's fusion reads its body with the rest), and each
    `platform_dependent` conditional the branch the platform lowers alone (`_cond_lowering`,
    jax/_src/lax/control_flow/conditionals.py keeps one branch, no conditional around it), as a call of
    it; through every program an equation runs but a Pallas kernel's, which XLA never sees (`bodies`)."""
    lowered = {"gpu": ("cuda", "rocm")}.get(platform, (platform,))
    renamed: dict[jcore.Var, jcore.Var | jcore.Literal] = {}
    equations: list[jcore.JaxprEqn] = []

    def read(value):
        return renamed.get(value, value) if computed(value) else value

    for equation in program.eqns:
        name = equation.primitive.name
        if name == "shard_map" and math.prod(equation.params["mesh"].shape[axis]
                                             for axis in equation.params["newly_manual_axes"]) == 1:
            body = compiled(equation.params["jaxpr"], platform)
            # Its own values, fresh, as one body traced once may be mapped more than once.
            local = {inner: read(outer) for inner, outer in zip(body.invars, equation.invars, strict=True)}
            for each in body.eqns:
                fresh = [out if isinstance(out, jcore.DropVar) else jcore.Var(out.aval)
                         for out in each.outvars]
                equations.append(each.replace(invars=[local.get(value, value) if computed(value) else value
                                                      for value in each.invars], outvars=fresh))
                local.update(zip(each.outvars, fresh, strict=True))
            renamed.update({outer: local.get(inner, inner) if computed(inner) else inner
                            for outer, inner in zip(equation.outvars, body.outvars, strict=True)})
            continue
        if name == "pallas_call":
            equations.append(equation.replace(invars=[read(value) for value in equation.invars]))
            continue
        params = dict(equation.params)
        if name == "cond" and params.get("branches_platforms") is not None:
            kept = next(index for index, platforms in enumerate(params["branches_platforms"])
                        if platforms is None or any(each in platforms for each in lowered))
            params = {**params, "branches": (params["branches"][kept],), "branches_platforms": (None,)}
        for key, value in params.items():
            params[key] = within(value, platform)
        equations.append(equation.replace(invars=[read(value) for value in equation.invars], params=params))
    return program.replace(eqns=equations, outvars=[read(value) for value in program.outvars])


def within(value, platform: str):
    """`value`, an equation's parameter, with each program in it as `compiled` makes it."""
    if isinstance(value, jcore.ClosedJaxpr):
        return value.replace(jaxpr=compiled(value.jaxpr, platform))
    if isinstance(value, jcore.Jaxpr):
        return compiled(value, platform)
    # A plain sequence of programs (a conditional's branches); a named tuple (a gather's dimension
    # numbers, a key's implementation) holds none and stays as it is.
    if type(value) in (tuple, list):
        return type(value)(within(each, platform) for each in value)
    return value


def computed(value) -> TypeGuard[jcore.Var]:
    """Whether `value` is a program's variable, not a literal it reads."""
    return isinstance(value, jcore.Var) and not isinstance(value, jcore.Literal)


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


def dimensions(value) -> tuple[int, ...]:
    """`value`'s shape, none where it is no array."""
    aval = value.aval
    return tuple(int(size) for size in aval.shape) if isinstance(aval, ShapedArray) else ()


def bodies(equation: jcore.JaxprEqn) -> list[jcore.Jaxpr]:
    """The programs `equation` runs within it: a call's, a loop's, a branch's or a checkpoint's. A
    Pallas kernel runs none XLA sees: it is one custom call whose results are its outputs alone, its
    body's values in registers and shared memory, its products Triton's, no cuBLAS call's
    (`pallas_call_lowering`, jax/_src/pallas/triton/pallas_call_registration.py)."""
    if equation.primitive.name == "pallas_call":
        return []
    found = []
    for value in equation.params.values():
        for each in value if isinstance(value, tuple | list) else (value,):
            body = each.jaxpr if isinstance(each, jcore.ClosedJaxpr) else each
            if isinstance(body, jcore.Jaxpr):
                found.append(body)
    return found


def recomputed(equation: jcore.JaxprEqn, platform: str) -> bool:
    """Whether XLA's fusion on `platform` recomputes what `equation` computes in each consumer rather than
    keep it: on a GPU every elementwise primitive (`ELEMENTWISE`), elsewhere the cheap ones (`CHEAP`, and
    `REAL_CHEAP` over real values); a call, or a custom derivative's primal, of nothing else."""
    name = equation.primitive.name
    if name in ("pjit", "jit", "custom_jvp_call"):
        inner = equation.params["jaxpr" if name != "custom_jvp_call" else "call_jaxpr"].jaxpr
        return all(recomputed(each, platform) for each in inner.eqns)
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
    with none (a compile-only GPU, `compile_only_gpus`, no device of this run's), each matmul takes the
    default config's (`DEFAULT_WORKSPACE`: `GetConfigAssignerOptions`,
    xla/service/gpu/autotuning/config_assigner_pass.cc, turns autotuning off without a device), but one
    in a dynamic-slice fusion, whose computation the config assigner does not visit
    (`ExtractEquivalentInstructions`, xla/backends/autotuner/hlo_extractor.cc). None where the device is
    no CUDA GPU, whose matmuls take none."""
    capability = _compute_capability(device) if device.platform == 'gpu' else None
    if not capability:
        return Workspace(0, 0)
    rewritten = (32 if int(capability.split('.')[0]) >= 9 else 4) << 20
    return Workspace(rewritten if device in jax.devices() else DEFAULT_WORKSPACE, rewritten)


def traced_loss[LossT, EffectsT](objective: Objective[LossT, EffectsT], state: TrainState, window: int
                                 ) -> tuple[Callable, tuple]:
    """The loss the step evaluates for `objective` over `state`, with the abstract arguments it is
    traced over (but the batch): the variables, the microstep, the key, the step and the EMA. Shapes
    and dtypes only: a resident frozen leaf sits in another memory space than the moving ones, and
    the loss the realization runs reads a snapshot in one space. The step's key is drawn inside the
    trace, so an abstract state compiles a step as a placed one does."""
    variables = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.variables)

    def loss(variables, batch, microstep, key, step, ema):
        return objective._loss(variables, batch, Step(
            microstep, jax.random.fold_in(key, step), with_ema(variables, ema),
            qk_stats=clips_qk(state.opt_state), example_count=window * rows_of(batch) or None))

    return loss, (variables, state.microstep, state.key, state.step, state.ema)


def candidate_footprint[LossT, EffectsT](objective: Objective[LossT, EffectsT],
                                          micro: Batch, window: int, prepared: TrainState,
                                          held: int, target) -> Footprint:
    """What a step over microbatches of `micro`'s shapes, its state `prepared` as placed, takes of a
    device like `target` beside that state (`Footprint`): its loss's residuals and its gradient
    program's peak at one microbatch, as that device's compiler keeps them and its GEMMs' workspace
    (`gemm_workspace`), its trained parameters' gradient, laid out as they are, and `held`. Over
    several devices that split the microbatch's rows alone (data and fully sharded data parallelism,
    `row_axes`) the residuals are one device's: the loss traced as one device's step, over a mesh of the
    run's axes each of one device, at that device's share of the microbatch (each leaf's shard,
    `shard_shape`), its parameters whole as those layouts gather them. A layout that splits more than the
    rows (stages, tensor, sequence) derives no residuals, which its compile judges."""
    mesh = jax.sharding.get_abstract_mesh()
    split = set() if mesh.empty else {name for name, size in mesh.shape.items() if size > 1}
    # Data and fully sharded data parallelism split the rows alone; an expert axis splits the experts too.
    rows = set(row_axes(rows_of(micro), mesh=mesh)) & {DATA_AXIS, FSDP_AXIS} if split else set()
    if not split <= rows:
        residuals = peak = None
    else:
        loss, (variables, *rest) = traced_loss(objective, prepared, window)
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
            residuals = residual_bytes(trained, arguments[0]["params"], arguments, platform=target.platform)
            peak = gradient_peak(trained, arguments[0]["params"], arguments, platform=target.platform,
                                 workspace=gemm_workspace(target))
    resident = {name: device_bytes(value) for name, value in {
        "variables": prepared.variables, "optimizer": prepared.opt_state, "ema": prepared.ema,
        "accumulation": prepared.accumulation, "compute": prepared.compute}.items()}
    return Footprint(residuals, device_bytes(prepared.variables["params"]), held, resident,
                     None if peak is None else peak.bytes, peak is None or peak.exact)
