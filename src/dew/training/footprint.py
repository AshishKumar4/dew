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
import contextvars
import dataclasses
import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NamedTuple

import jax
import jax.extend.core as jcore
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jax.extend.core import primitives

from dew import algebra
from dew._devices import deviceless
from dew._records import JSON
from dew.batch import Batch
from dew.data.dataset import rows_of
from dew.distributed.placement import placed_bytes
from dew.mesh import DATA_AXIS, FSDP_AXIS, row_axes
from dew.nn.kernels.generation import _compute_capability, gpu_target
from dew.nn.programs import (
    bodies,
    computed,
    live as live_equations,
    loop_programs,
    reduction,
    scan_body,
    scatter,
)
from dew.objectives.base import Objective
from dew.state import TrainState
from dew.training.compiled_program import (
    Slicing,
    bitcast,
    cublas_product,
    dimensions,
    lowering_platform,
    simplified,
)
from dew.training.memory import head_fused_of, head_tile_of, judged
from dew.training.narrow import forward_variables
from dew.training.transaction import attempt_step, batch_statistics_shapes


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
    """The gradient program's peak at one microbatch (`peak_bytes`): its residuals, cotangents and
    gradients. A CPU upper bound also holds operands whose matrix gradients may run after the backward;
    `peak_lower` states its interval. None where `residuals` is."""
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
    head: HeadTerms | None = None
    """What `peak` holds of a head that keeps its whole logits (`HeadTerms`), from which the step's peak at
    its head's tile is bounded without a trace of it (`dew.training.planning.FitSearch.walked`); None for a
    step whose head keeps no whole logits, or where no peak is derived."""
    peak_lower: int | None = None
    """A CPU peak's lower bound, beside its upper bound `peak`: the same walk without possible library
    arrays and with gradients consumed in traced order. None where the existing single-bound rules apply."""

    @property
    def temporaries(self) -> int | None:
        """The most the step holds at once beside its state: its gradient program's peak, or, where its
        commit stages host-resident fields, its whole gradient and those where they are more. None where
        `peak` is."""
        if self.peak is None or not self.staged:
            return self.peak
        return max(self.peak, self.gradients + self.staged)

    @property
    def lower_temporaries(self) -> int | None:
        """The least temporaries the derivation admits, including a staged commit's gradient and fields."""
        lower = self.peak if self.peak_lower is None else self.peak_lower
        return lower if lower is None or not self.staged else max(lower, self.gradients + self.staged)

    def record(self) -> dict[str, JSON]:
        """The terms traced, not an admission decision: XLA's fit check remains authoritative."""
        return {"residuals": self.residuals, "gradients": self.gradients, "held": self.held,
                "resident": dict(self.resident), "peak": self.peak, "peak_exact": self.exact,
                "staged": self.staged, "consumed": self.consumed,
                "head": None if self.head is None else {
                    key: value for key, value in self.head._asdict().items() if value is not None},
                **({} if self.peak_lower is None else {"peak_lower": self.peak_lower})}


class HeadTerms(NamedTuple):
    """What a gradient program's peak (`peak_bytes`) holds of its head's whole logits, the values shaped by
    the head's rows and the step's tokens (`logits_shaped`): its peak with none of them held
    (`headless`), the most bytes of them held at once (`held`), the elements of the largest
    (`elements`), and the head's `rows`. A step whose head takes the backward a tile at a time holds the
    rest as the whole-logits step does, and of them a tile's share at most, a fused head's chunk the
    tile's tokens over every row: its peak is at least `headless` and at most `tiled`'s, read off the
    whole-logits step's one trace (row 5556: on Qwen3.5-4B's step, 4 x 12288 tokens, the headless peak of
    its whole-logits trace is its tiled trace's peak to the byte)."""

    headless: int
    held: int
    elements: int
    rows: int
    lower: int | None = None
    """The CPU headless lower bound where its scheduling rules leave an interval."""

    def tiled(self, tile: tuple[int, int], *, fused: bool) -> int:
        """The most the step's gradient program holds at its head's `tile`, fused or not: its peak bounded
        above."""
        share = tile[0] * (self.rows if fused else tile[1])
        return self.headless + math.ceil(share * self.held / self.elements)


class Peak(NamedTuple):
    """A gradient program's derived peak (`gradient_peak`): its bytes, and whether they are the
    program's own or a bound from below (`bounded_below`); a CPU's upper bound and `lower` beside it."""

    bytes: int
    exact: bool
    lower: int | None = None


def device_bytes(tree) -> int:
    """The placed leaves in device memory alone: an offloaded optimizer or EMA is host storage,
    not resident HBM. Each device leaf carries the sharding that `placed_bytes` reads."""
    return placed_bytes([leaf for leaf in jax.tree.leaves(tree)
                         if jax.typeof(leaf).memory_space == jax.memory.Space.Device])


CHEAP = frozenset({jax.lax.add_p, primitives.add_jaxvals_p, jax.lax.and_p, jax.lax.bitcast_convert_type_p,
    jax.lax.broadcast_in_dim_p, jax.lax.ceil_p, jax.lax.clamp_p, jax.lax.clz_p, jax.lax.eq_p, jax.lax.ne_p,
    jax.lax.lt_p, jax.lax.le_p, jax.lax.gt_p, jax.lax.ge_p, jax.lax.complex_p, jax.lax.concatenate_p,
    jax.lax.convert_element_type_p, jax.lax.copy_p, jax.lax.dynamic_slice_p, jax.lax.dynamic_update_slice_p,
    jax.lax.floor_p, jax.lax.imag_p, jax.lax.iota_p, jax.lax.is_finite_p, jax.lax.max_p, jax.lax.min_p,
    jax.lax.mul_p, jax.lax.neg_p, jax.lax.not_p, jax.lax.or_p, jax.lax.xor_p, jax.lax.pad_p,
    jax.lax.population_count_p, jax.lax.real_p, jax.lax.reduce_precision_p, jax.lax.reshape_p,
    jax.lax.squeeze_p, jax.lax.rev_p, jax.lax.round_p, jax.lax.select_n_p, jax.lax.shift_left_p,
    jax.lax.shift_right_arithmetic_p, jax.lax.shift_right_logical_p, jax.lax.slice_p, jax.lax.sub_p,
    jax.lax.transpose_p, jax.lax.integer_pow_p, jax.lax.square_p, primitives.name_p})
"""The primitives XLA's CPU fusion takes for cheap, which it duplicates into each consumer rather than
keep a buffer of (`InstructionFusion::IsExpensive`, xla/service/instruction_fusion.cc: every elementwise
op but division, the transcendentals and their kin, and the layout ops), an integer power and a square
lowering to its multiplies, a cotangent sum, which JAX lowers to one `add` (`add_jaxvals_lowering`,
jax/_src/interpreters/mlir.py), and a checkpoint's name, which computes nothing."""

REAL_CHEAP = frozenset({jax.lax.abs_p, jax.lax.sign_p, jax.lax.sin_p, jax.lax.cos_p, jax.lax.tan_p,
    jax.lax.sinh_p})
"""The primitives XLA's CPU fusion takes for cheap over real elements alone (`IsExpensive`)."""

ELEMENTWISE = CHEAP | REAL_CHEAP | frozenset({jax.lax.div_p, jax.lax.rem_p, jax.lax.exp_p, jax.lax.exp2_p,
    jax.lax.log_p, jax.lax.log1p_p, jax.lax.expm1_p, jax.lax.tanh_p, jax.lax.logistic_p, jax.lax.erf_p,
    jax.lax.erfc_p, jax.lax.erf_inv_p, jax.lax.pow_p, jax.lax.sqrt_p, jax.lax.rsqrt_p, jax.lax.cbrt_p,
    jax.lax.atan2_p, jax.lax.asin_p, jax.lax.acos_p, jax.lax.atan_p, jax.lax.cosh_p, jax.lax.asinh_p,
    jax.lax.acosh_p, jax.lax.atanh_p})
"""The elementwise and layout primitives, every one of which XLA's GPU fusion recomputes in each consumer
rather than keep a buffer of: its priority fusion fuses a producer into all its consumers where its cost
model finds the memory traffic saved worth the recompute (xla/backends/gpu/transforms/priority_fusion.cc),
an elementwise one's always on a memory-bound GPU, a transcendental's too (an attention's exponentials and
probabilities, on an H100's compile, recomputed from its scores in the backward)."""


TRANSPOSES = (jax.lax.transpose_p,)
"""The layout primitive XLA's layout assignment folds into the storage of a value the program computes,
laying it out as its reader needs (a returned gradient's), not into an input's, whose layout is fixed."""


LOOPS = (jax.lax.scan_p, jax.lax.while_p)
"""The primitives that run their body turn after turn, their results held across the turns."""

POINTER_BYTES = 8
"""The bytes of a pointer, which XLA's tuple of a program's results holds one of for each result."""

YNN_MIN_ELEMENTS = 4096
"""The least elements of an operation or an operand YNNPACK prefers over an elementwise loop
(`IsInstructionPreferredByYnn`, xla/backends/cpu/ynn_support.cc), layout operations accepted separately."""

YNN_UNARY = frozenset({jax.lax.abs_p, jax.lax.ceil_p, jax.lax.convert_element_type_p, jax.lax.cos_p,
    jax.lax.erf_p, jax.lax.exp_p, jax.lax.expm1_p, jax.lax.floor_p, jax.lax.log_p, jax.lax.log1p_p,
    jax.lax.logistic_p, jax.lax.neg_p, jax.lax.round_p, jax.lax.rsqrt_p, jax.lax.sign_p, jax.lax.sin_p,
    jax.lax.sqrt_p, jax.lax.tanh_p, jax.lax.tan_p, jax.lax.integer_pow_p, jax.lax.square_p})
"""`GetYnnUnaryOpMap` (xla/backends/cpu/ynn_support.cc), with powers JAX lowers to multiply and divide."""

YNN_BINARY = frozenset({jax.lax.add_p, primitives.add_jaxvals_p, jax.lax.div_p, jax.lax.max_p,
    jax.lax.min_p, jax.lax.mul_p, jax.lax.pow_p, jax.lax.sub_p})
"""`GetYnnBinaryOpMap` (xla/backends/cpu/ynn_support.cc), with JAX's cotangent sum lowered to add."""

YNN_ELEMENT_TYPES = frozenset({"float64", "float32", "bfloat16", "int8", "uint8"})
"""The elementwise types of `IsElementwiseOpSupportedByYnn` (xla/backends/cpu/ynn_support.cc)."""

YNN_ARRAY_TYPES = YNN_ELEMENT_TYPES | frozenset({"int4", "uint4", "float16", "int32"})
"""The array types of `YnnType` (xla/backends/cpu/runtime/ynnpack/ynn_interop.cc)."""

YNN_MAX_RANK = 8
"""`YNN_MAX_TENSOR_RANK`, ynnpack/include/ynnpack.h, checked by `IsLayoutSupportedByYnn`."""

PRODUCTS = (jax.lax.dot_general_p, jax.lax.conv_general_dilated_p, primitives.ragged_dot_general_p)
"""The products a weight's gradient is made by, past which a value is a cotangent the backward reads on
(`sunk`)."""

def _folds(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` folds entries into one value over axes or windows: a reduction that is not the
    place of an extremum, or one over windows."""
    return primitive in WINDOWS or (reduction(primitive)
                                    and primitive not in (jax.lax.argmax_p, jax.lax.argmin_p))


def _ynn_eligible(equation: jcore.JaxprEqn) -> bool:
    """Whether a default CPU reduction library fusion can include `equation`, with row-major arrays:
    `YnnMatcher::IsOpSupported` (xla/backends/cpu/transforms/ynn_matcher.h) and its checks in
    xla/backends/cpu/ynn_support.cc. This is independent of LLVM's cheap-to-recompute operations."""
    values = [*equation.invars, *equation.outvars]
    if len(equation.outvars) != 1:
        return False
    types = []
    for value in values:
        aval = value.aval
        if not isinstance(aval, ShapedArray) or len(aval.shape) > YNN_MAX_RANK:
            return False
        types.append(str(aval.dtype))
    if any(dtype not in YNN_ARRAY_TYPES for dtype in types):
        return False
    primitive = equation.primitive
    result = types[-1]
    if primitive is jax.lax.iota_p:
        return not equation.invars and result in ("float32", "int32")
    # The matcher accepts layout operations before applying the element-count preference.
    if bitcast(equation) or primitive in (
            jax.lax.transpose_p, jax.lax.copy_p, jax.lax.broadcast_in_dim_p, primitives.name_p):
        if len(equation.invars) != 1 or types[0] != result:
            return False
        if primitive is jax.lax.broadcast_in_dim_p:
            axes = equation.params["broadcast_dimensions"]
            return all(left < right for left, right in itertools.pairwise(axes))
        return True
    if primitive is jax.lax.pad_p:
        return (len(equation.invars) == 2 and all(dtype == result for dtype in types)
                and all(interior == 0 and (size != 1 or (low == 0 and high == 0))
                        for size, (low, high, interior) in zip(
                            dimensions(equation.invars[0]), equation.params["padding_config"], strict=True)))
    if not any(math.prod(dimensions(value)) >= YNN_MIN_ELEMENTS for value in values):
        return False
    if primitive in YNN_UNARY or primitive in YNN_BINARY:
        return (len(equation.invars) == (1 if primitive in YNN_UNARY else 2)
                and all(dtype in YNN_ELEMENT_TYPES for dtype in types)
                and (primitive is not jax.lax.round_p or equation.params["rounding_method"]
                     == jax.lax.RoundingMethod.TO_NEAREST_EVEN))
    reducers = {jax.lax.reduce_sum_p: jax.lax.add_p, jax.lax.reduce_max_p: jax.lax.max_p,
                jax.lax.reduce_min_p: jax.lax.min_p, jax.lax.reduce_window_sum_p: jax.lax.add_p,
                jax.lax.reduce_window_max_p: jax.lax.max_p, jax.lax.reduce_window_min_p: jax.lax.min_p}
    reducer = reducers.get(primitive)
    if primitive in (jax.lax.reduce_p, jax.lax.reduce_window_p):
        inner = bodies(equation)
        if len(equation.invars) != 2 or len(inner) != 1 or len(inner[0].eqns) != 1:
            return False
        operation = inner[0].eqns[0]
        if (operation.primitive not in (jax.lax.add_p, jax.lax.max_p, jax.lax.min_p)
                or set(operation.invars) != set(inner[0].invars)
                or operation.outvars != inner[0].outvars or types[1] != result):
            return False
        reducer = operation.primitive
    if reducer is None:
        return False
    if primitive in WINDOWS:
        sizes = equation.params["window_dimensions"]
        strides = equation.params.get("window_strides", (1,) * len(sizes))
        axes = sum(size > 1 or stride > 1 for size, stride in zip(sizes, strides, strict=True))
        if (not axes or len(dimensions(equation.outvars[0])) + axes > YNN_MAX_RANK
                or any(size > 1 and stride != size for size, stride in zip(sizes, strides, strict=True))
                or any(value != 1 for key in ("base_dilation", "window_dilation")
                       for value in equation.params.get(key, (1,) * len(sizes)))):
            return False
    elif not equation.params.get("axes", equation.params.get("dimensions", ())):
        return False
    if reducer is jax.lax.add_p:
        return (types[0], result) in {
            ("float64", "float64"), ("float32", "float32"), ("float16", "float32"),
            ("bfloat16", "float32"), ("int8", "int32"), ("uint8", "int32")}
    return types[0] == result and result in (
        "float64", "float32", "float16", "bfloat16", "int8", "uint8")


def _cpu_dot_inputs(equation: jcore.JaxprEqn) -> frozenset[int]:
    """Operands a scalar/vector dot can loop-fuse: the opposite operand is a vector below 16 KiB
    (`CpuInstructionFusion::ShouldFuse`, xla/service/cpu/cpu_instruction_fusion.cc). Other dots stay
    individual library products. Sizes here are physical, not a head projection's counted bytes."""
    if (equation.primitive is not jax.lax.dot_general_p or len(equation.invars) != 2
            or len(equation.outvars) != 1 or len(dimensions(equation.outvars[0])) > 1):
        return frozenset()
    return frozenset(index for index in (0, 1)
                     if len(dimensions(other := equation.invars[1 - index])) == 1
                     and isinstance(other.aval, ShapedArray)
                     and math.prod(dimensions(other)) * np.dtype(other.aval.dtype).itemsize < 16 * 1024)


def _cpu_output_dot(equation: jcore.JaxprEqn) -> bool:
    """The real, non-batched scalar/vector dots an add may output-fuse (`CanBeOutputFused`,
    xla/service/cpu/cpu_instruction_fusion.cc); the caller checks the one-use condition."""
    return (equation.primitive is jax.lax.dot_general_p and len(equation.outvars) == 1
            and isinstance(equation.outvars[0].aval, ShapedArray)
            and len(dimensions(equation.outvars[0])) <= 1
            and not np.issubdtype(equation.outvars[0].aval.dtype, np.complexfloating)
            and not any(equation.params["dimension_numbers"][1]))


def _cpu_reduction_scratch(equation: jcore.JaxprEqn) -> int:
    """Possible tree-reduction partials, held beside the input of a non-library reduction. CPU's
    AlgebraicSimplifier strength-reduces vector-vector dots to multiply/reduce; TreeReductionRewriter
    splits dimensions into windows of 32 (xla/service/cpu/cpu_compiler.cc,
    xla/hlo/transforms/simplifiers/tree_reduction_rewriter.{h,cc}). Count all possible levels as an upper
    bound; the lower walk need not materialize these partials."""
    if len(equation.outvars) != 1:
        return 0
    if (equation.primitive is jax.lax.dot_general_p and len(equation.invars) == 2
            and all(len(dimensions(value)) == 1 for value in equation.invars)
            and not dimensions(equation.outvars[0])):
        shape, axes = list(dimensions(equation.invars[0])), (0,)
    elif (reduction(equation.primitive) and equation.primitive not in WINDOWS
          and not _ynn_eligible(equation)):
        shape = list(dimensions(equation.invars[0]))
        axes = equation.params.get("axes", equation.params.get("dimensions", ()))
    else:
        return 0
    aval = equation.outvars[0].aval
    if not isinstance(aval, ShapedArray):
        return 0
    size = np.dtype(aval.dtype).itemsize
    if jnp.issubdtype(aval.dtype, jnp.floating):
        size = max(4, size)  # FloatNormalization can promote unsupported low-precision accumulation.
    partials = 0
    while any(shape[axis] > 32 for axis in axes):
        shape = [math.ceil(length / 32) if axis in axes else length
                 for axis, length in enumerate(shape)]
        partials += math.prod(shape) * size
    return partials


WINDOWS = (jax.lax.reduce_window_p, jax.lax.reduce_window_sum_p, jax.lax.reduce_window_max_p,
           jax.lax.reduce_window_min_p)
"""The reductions over windows, which fuse as the reductions over axes do."""


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
                  lowering: str | None = None, consumed: bool = False, deferred: bool = False,
                  table: tuple[int, int] | None = None) -> tuple[int, Peak, HeadTerms | None]:
    """`residual_bytes` and `gradient_peak` of `loss(params, *arguments)` from one trace of its forward and
    backward (`traced_gradient`), lowered once (`compiled`, its outputs the gradient and the residuals
    alike: an equation's lowering reads no output), as each derives it from its own (row 4261: lowered
    for each, the two lowerings were 16 of a Qwen3.5-4B candidate's 28 seconds); and, for a head that keeps
    its whole logits from a table of `table`'s rows and features, what its peak holds of them
    (`head_terms`)."""
    program, count, shapes = traced_gradient(loss, params, arguments)
    lowered = simplified(program, platform, lowering)
    peak = _peak(lowered, count, platform, workspace, consumed, deferred)
    head = (None if table is None
            else head_terms(lowered, count, platform, workspace, table, consumed, deferred))
    return _kept(lowered, count, shapes, platform), peak, head


_COUNTED: contextvars.ContextVar[Callable[[jcore.Var], bool] | None] = contextvars.ContextVar(
    "counted", default=None)
"""Which values `device_share` counts the bytes of, every one where None: `head_terms`'s peaks."""


def shape_of(value: jcore.Var | jcore.Literal) -> tuple[int, ...]:
    """`value`'s array shape, none for one that holds no array (a token)."""
    aval = value.aval
    return tuple(aval.shape) if isinstance(aval, ShapedArray) else ()


def logits_shaped(value: jcore.Var, table: tuple[int, int]) -> bool:
    """Whether `value` is a head's logits, or a value of their shape, of a table of `table`'s rows and
    features: one that holds the rows and more elements than the table, each token's row of them, where the
    table's own shape and its gradient's hold no more."""
    shape = shape_of(value)
    return table[0] in shape and math.prod(shape) > table[0] * table[1]


def head_terms(lowered: jcore.Jaxpr, count: int, platform: str, workspace: Workspace, table: tuple[int, int],
               consumed: bool = False, deferred: bool = False) -> HeadTerms | None:
    """What `_peak` of `lowered` holds of the head's whole logits (`HeadTerms`), its values shaped by
    `table` (`logits_shaped`): its peak with none of them counted and with them alone; None where it holds
    none."""
    def shaped(value) -> bool:
        return logits_shaped(value, table)

    found = [math.prod(shape_of(value)) for equation in lowered.eqns for value in equation.outvars
             if shaped(value)]
    if not found:
        return None
    terms = []
    for counted in (lambda value: not shaped(value), shaped):
        token = _COUNTED.set(counted)
        try:
            terms.append(_peak(lowered, count, platform, workspace, consumed, deferred))
        finally:
            _COUNTED.reset(token)
    headless, held = terms
    return HeadTerms(headless.bytes, held.bytes, max(found), table[0], headless.lower)


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
    upper = peak_bytes(ordered, platform, workspace, consumed=consumed)
    lower = None
    if platform == "cpu":
        # CPU scheduling can leave matrix gradients until their shared backward operands' last reads.
        # Holding those operands to the end bounds that retention without predicting its schedule.
        upper = max(upper, peak_bytes(hoisted(sunk(gradient)), platform, workspace, consumed=consumed))
        if _COUNTED.get() is None:
            minimal = gradient.replace(eqns=live_equations(gradient.eqns, gradient.outvars))
            lower = peak_bytes(hoisted(minimal), platform, workspace, consumed=consumed, lower=True)
            assert lower <= upper
        # A shape-filtered head projection can change which alias of one storage it counts. It is an
        # upper decomposition, not a lower-bound proof for a different head policy.
    return Peak(upper, not bounded_below(gradient, platform), lower)


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
                reader in late and program.eqns[reader].primitive not in PRODUCTS
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
        equation.primitive is primitives.ragged_dot_general_p
        or any(bounded_below(body, platform) for body in bodies(equation))
        for equation in program.eqns)


def peak_bytes(program: jcore.Jaxpr, platform: str, workspace: Workspace,
               slicing: Slicing | None = None, carried: frozenset = frozenset(), *,
               consumed: bool = False, lower: bool = False) -> int:
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
    fuse or make apart, the arrays its library fusions read apart (`laid_inputs`), what its products read
    made once or copied (`cpu_layouts`), and its input copies from its start. No schedule XLA picks is
    modelled here: `_peak` also walks deferred gradients for a CPU upper bound. With `lower`, the CPU's
    possible library arrays and product copies are omitted, leaving the definite traced terms."""
    frontier = Frontier(program, platform, returned=not (platform == "cpu" and lower and consumed),
                        lower=lower)
    last = {kept: index for index, equation in enumerate(program.eqns)
            for kept in frontier.reading(equation.invars)}
    # The copies of the step's inputs its products read laid out for them (`laid_copies`), each held
    # from its first reader to its last, and on a GPU the siblings a product's operands are made beside
    # (`siblings`): the step's own program's, whose parameters' layouts are fixed, not a loop's body's.
    entry = slicing is None and not carried
    beside, spans, spanned = siblings(frontier, program) if entry and platform == "gpu" else ({}, [], {})
    if platform == "cpu" and not lower:
        beside, spans, spanned = cpu_layouts(frontier, program)
    laid = laid_copies(frontier, program) if entry else []
    if platform == "cpu" and not lower:
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
    held, prepared, most = set(), set(), live
    for index, equation in enumerate(program.eqns):
        live += sum(size for first, _, size in copies if first == index)
        made = [out for out in equation.outvars
                if out in last and frontier.buffered(out) and out not in carried]
        made_bytes = sum(map(device_share, made))
        # A scatter or a product with a bias writes its result in a buffer it reads last (`in_place`).
        taken = in_place(frontier, equation, held, last, index)
        if made or not frontier.fusing(equation):
            inner = max((peak_bytes(body, platform, workspace, Slicing.entering(equation, body),
                                    returned_carry(equation, body), lower=lower)
                         for body in bodies(equation)), default=0)
            # An operand made beside an earlier product's is held from there (`siblings`), not made here.
            made_for = unfused(frontier, equation) - spanned.get(index, set())
            operands = sum(map(device_share, made_for)) + beside.get(index, 0)
            reused = set()
            if platform == "cpu" and lower:
                for value in made_for:
                    producer = frontier.producers.get(value)
                    if producer is None:
                        continue
                    base = _lower_reused_storage(frontier, producer, held, last, index)
                    if (base is not None and base not in reused and base not in frontier.reading(
                            read for read in equation.invars if read is not value)):
                        reused.add(base)
                        operands -= device_share(value)
            if platform == "cpu" and not lower:
                # A shared operand's chain runs before its first product, beside its other live buffers.
                starting = spanned.get(index, set()) - prepared
                prepared.update(starting)
                reduced = made if _folds(equation.primitive) else []
                operands += sum(map(device_share, laid_inputs(frontier, [*reduced, *made_for, *starting])))
            products = 0 if not cublas_product(equation) else (
                workspace.sliced if sliced.fused(equation) else workspace.each)
            if platform == "cpu" and not lower:
                products += _cpu_reduction_scratch(equation)
            # A loop's results (its stacked outputs, its carry) are held beside each turn's own; a call's
            # results are what its body returns.
            running = (looped(frontier, last, index, equation, made, carried) + inner
                       if equation.primitive in LOOPS else max(inner, made_bytes) - taken)
            # The fusions that make its operands run before it, each buffer they alone read freed with them
            # (`freed`): the most is theirs, or its own past them.
            freed = sum(map(device_share, dying(frontier, equation, made_for, held, last, index) - reused))
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
        if equation.primitive is not jax.lax.dot_general_p:
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
        dict[int, int], list[tuple[int, int, int]], dict[int, set]]:
    """What a CPU's products hold beside the operands they read made for them (`unfused`): XLA's CPU
    products (YNNPACK's and Eigen's) read every operand from memory in the order they take it, so a
    fused value two products read is made once, held from the first to the last (a span, `(first, last,
    bytes)`, and that operand no longer made for its reads, by product: an attention's scores' cotangent,
    read by its keys' and its queries' gradients); and each value XLA keeps a buffer of (`Frontier`), or
    so made once, that a product reads in an order it does not take (`product_order`) is copied for that
    product (bytes, by product: an attention's probabilities read by its values' gradient laid out
    otherwise). Row 5290: 64 x 128 tokens, 8 heads over 2, compiled 1.375 times the derivation without."""
    products = [(index, equation) for index, equation in enumerate(program.eqns)
                if equation.primitive is jax.lax.dot_general_p]
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
            if not isinstance(operand, jcore.Var) or operand in frontier.inputs:
                continue
            kept = frontier.buffered(operand) or operand in shared
            if kept and product_order(operand, contracting, batch) is not None:
                copied[index] = copied.get(index, 0) + device_share(operand)
    return (copied, [(readers[0], readers[-1], device_share(value)) for value, readers in shared.items()],
            spanned)


def laid_inputs(frontier: Frontier, values: Sequence[jcore.Var]) -> set[jcore.Var]:
    """The arrays a CPU's library fusions read apart while making `values`: the chains a reduction may
    fuse into (`Frontier.reducing`), whose broadcasts, layout changes and conversions YNNPACK may read
    from dense arrays. A transpose followed by a conversion is one array in the outer operation's type
    and order. An input YNNPACK cannot fuse, such as a select, holds its result alone. Elementwise loop
    fusions index their broadcasts and transposes in place (`ElementalIrEmitter::MakeElementGenerator`,
    xla/service/elemental_ir_emitter.cc), reading an upstream library fusion's result from its own array."""
    found, pending, seen = set(), [(value, value in frontier.reducing) for value in values], set()
    while pending:
        value, library = pending.pop()
        if (value, library) in seen or not computed(value):
            continue
        seen.add((value, library))
        equation = frontier.producers.get(value)
        if equation is None:
            # A product or a loop reads its operands from memory: no fusion of the chain runs past it.
            continue
        reads = [read for read in equation.invars if computed(read)]
        if not library:
            if value in frontier.reducing:
                found.add(value)
            elif frontier.fusing(equation):
                pending.extend((read, False) for read in reads
                               if read not in frontier.inputs and not frontier.buffered(read))
            continue
        if not _ynn_eligible(equation):
            found.add(value)
            continue
        primitive = equation.primitive
        # Every broadcast that widens a value, of constants too (a mask's iota, a cotangent of ones): XLA
        # may fold it into the fusion or make it apart, and a bound counts it apart.
        widened = (primitive is jax.lax.broadcast_in_dim_p
                   and math.prod(dimensions(value)) > math.prod(dimensions(equation.invars[0])))
        relaid = primitive in (jax.lax.transpose_p, jax.lax.convert_element_type_p) and any(
            frontier.stored(read) for read in reads)
        if widened or relaid:
            found.add(value)
        if not relaid:
            pending.extend((read, True) for read in reads
                           if read not in frontier.inputs and not frontier.buffered(read))
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
    # The products each value is made for, in their order.
    making: dict[jcore.Var, list[int]] = {}
    for index, values in made.items():
        for value in values:
            making.setdefault(value, []).append(index)

    # Each value's storage, element count and readiness are read once, and its possible siblings found from
    # the storage it reads, never by pairing every operand with every other: a program's products pair
    # their operands by the thousand (a hybrid MoE step asked 1.2 million pairs, each walking its two
    # values' storage anew, 552 s of a 633 s derivation, row 5360; a 4B decoder's step at full remat held
    # its first step 45 minutes, row 5479).
    elements = {value: math.prod(dimensions(value)) for value in making}
    roots = {value: frontier.root_set(value) for value in making}
    ready = {value: max((position.get(root, -1) for root in roots[value]), default=-1) for value in making}
    sharing: dict[tuple[jcore.Var, int], list[jcore.Var]] = {}
    for value in making:
        for root in roots[value]:
            if math.prod(dimensions(root)) > 1:
                sharing.setdefault((root, elements[value]), []).append(value)
    partners = {value: {other for root in roots[value] for other in sharing.get((root, elements[value]), ())
                        if other is not value} for value in making}

    # What each value is computed from, among the values that have a partner: one pass over the program in
    # its order, each value's set the union of its operands' and the operands themselves.
    paired = [value for value, others in partners.items() if others]
    bit = {value: 1 << place for place, value in enumerate(paired)}
    ancestry: dict[jcore.Var, int] = {}
    for equation in program.eqns:
        reached = 0
        for read in equation.invars:
            if computed(read):
                reached |= ancestry.get(read, 0) | bit.get(read, 0)
        for out in equation.outvars:
            ancestry[out] = reached

    def computed_from(value, other) -> bool:
        """Whether `value` is computed from `other` through any equation."""
        return bool(ancestry.get(value, 0) & bit[other])

    beside: dict[int, set] = {}
    first: dict = {}
    for value in paired:
        for other in partners[value]:
            if computed_from(value, other) or computed_from(other, value):
                continue
            at = max(ready[value], ready[other])
            for later in making[value]:
                for earlier in making[other]:
                    if earlier >= later:
                        break
                    if other in made[later]:
                        continue
                    if at < earlier:
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
    primitive = equation.primitive
    if primitive is jax.lax.optimization_barrier_p or not equation.invars:
        return set()
    if scatter(primitive):
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


def _lower_reused_storage(frontier: Frontier, equation: jcore.JaxprEqn, held: set, last: Mapping,
                          index: int) -> jcore.Var | None:
    """A last-read temporary a lower-walk pointwise loop may overwrite, resolving its operand's views
    (`HloDataflowAnalysis::CanShareOperandBufferWithUser`, elementwise case). The same storage remains
    live when such a loop makes an operand for a following product."""
    if (not frontier.lower or frontier.platform != "cpu" or not equation.primitive.elementwise
            or len(equation.outvars) != 1 or equation.effects):
        return None
    output = equation.outvars[0]
    for operand in equation.invars:
        if not computed(operand):
            continue
        storage = frontier.reading([operand])
        if len(storage) != 1:
            continue
        (base,) = storage
        if (base in held and last.get(base) == index
                and isinstance(operand.aval, ShapedArray) and isinstance(output.aval, ShapedArray)
                and dimensions(operand) == dimensions(output) and operand.aval.dtype == output.aval.dtype
                and device_share(base) == device_share(output)
                and base not in frontier.reading(value for value in equation.invars if value is not operand)):
            return base
    return None


def in_place(frontier: Frontier, equation: jcore.JaxprEqn, held: set, last: Mapping, index: int) -> int:
    """The bytes of the buffer `equation`, the program's `index`th, writes its result in: a scatter's
    base's, or a product's bias's (`biased`), the one buffer the program holds (`held`) that it is or views
    (`Frontier.reading`), where the equation reads that buffer last, its other operands read none of it,
    and it is its result's size, as XLA shares a scatter's first operand's buffer with its result
    (`HloDataflowAnalysis::CanShareOperandBufferWithUser`, xla/hlo/analysis/hlo_dataflow_analysis.cc) and
    a GEMM's bias's that nothing else reads, before it or after (`FuseMatrixBiasAdd`'s `can_overwrite_bias`,
    xla/backends/gpu/transforms/gemm_rewriter.cc). The CPU lower walk also permits a last-read
    elementwise operand's storage to supply its result."""
    primitive = equation.primitive
    if (reused := _lower_reused_storage(frontier, equation, held, last, index)) is not None:
        return device_share(reused)
    if scatter(primitive) and equation.invars:
        at = 0
    elif primitive is jax.lax.dot_general_p and len(equation.invars) == 3:
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
    if equation.primitive is jax.lax.while_p:
        start = equation.params["cond_nconsts"] + equation.params["body_nconsts"]
        body = loop_programs(equation)[1].jaxpr
        taken = body.invars[equation.params["body_nconsts"]:]
        changed = [returned is not took for returned, took in zip(body.outvars, taken, strict=True)]
        return list(zip(equation.invars[start:], equation.outvars, changed, strict=True))
    if equation.primitive is jax.lax.scan_p:
        body = scan_body(equation).jaxpr
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
    if equation.primitive is jax.lax.while_p and body is loop_programs(equation)[1].jaxpr:
        return frozenset(value for value in body.outvars if computed(value))
    if equation.primitive is jax.lax.scan_p and body is scan_body(equation).jaxpr:
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

    def __init__(self, program: jcore.Jaxpr, platform: str, *, returned: bool, lower: bool = False):
        self.platform = platform
        self.lower = lower
        self.producers = {out: equation for equation in program.eqns for out in equation.outvars}
        self.inputs = {*program.invars, *program.constvars}
        self.outputs = {out for out in program.outvars if computed(out)} if returned else set()
        self.readers = collections.Counter(value for equation in program.eqns
                                           for value in {read for read in equation.invars if computed(read)})
        self.readers.update({value for value in program.outvars if computed(value)})
        self.reads: dict[jcore.Var, frozenset[jcore.Var]] = {}
        self.rooted: dict[jcore.Var, frozenset[jcore.Var]] = {}
        self.fusions: dict[jcore.Var, bool] = {}
        # An optimization barrier defines no value: its results are its operands' own
        # (`HloDataflowAnalysis::UpdateOptimizationBarrierValueSet`,
        # xla/hlo/analysis/hlo_dataflow_analysis.cc).
        self.forwards = {out: operand for equation in program.eqns
                         if equation.primitive is jax.lax.optimization_barrier_p
                         for out, operand in zip(equation.outvars, equation.invars, strict=True)}
        self.barred = {operand for operand in self.forwards.values() if computed(operand)}
        # YNNPACK grows a reduction through its own supported consumers, including transcendentals,
        # independently of LLVM's cheap-to-recompute classification (`_ynn_eligible`).
        self.reducing: set[jcore.Var] = set()
        self.absorbed: set[jcore.Var] = set()
        self.looping: set[jcore.Var] = set()
        if platform == "cpu":
            for equation in program.eqns:
                if _ynn_eligible(equation) and (_folds(equation.primitive)
                        or any(value in self.reducing for value in equation.invars if computed(value))):
                    self.reducing.update(out for out in equation.outvars if computed(out))
            if lower:
                # Positionwise primitives include JAX's polynomial, which lowers to a loop's
                # multiply/add chain. Shared intermediates can stay inside one loop when every path
                # reaches the same result (`CanFuseOnAllPaths`, xla/service/instruction_fusion.cc).
                # Non-library reductions are loop consumers too (`CanBeLoopFused`,
                # xla/service/cpu/cpu_instruction_fusion.cc).
                looping = {id(equation) for equation in program.eqns if not equation.effects
                           and (equation.primitive in ELEMENTWISE or equation.primitive.elementwise
                                or equation.primitive.positionwise or equation.primitive.rearranges
                                or (reduction(equation.primitive) and equation.primitive not in WINDOWS
                                    and not _ynn_eligible(equation))
                                or _cpu_dot_inputs(equation))}
                self.looping.update(out for equation in program.eqns if id(equation) in looping
                                    for out in equation.outvars if computed(out))
                roots: dict[jcore.Var, int] = {}

                def root(value):
                    if value not in roots:
                        roots[value] = 1 << len(roots)
                    return roots[value]

                reached = {value: root(value) for value in program.outvars if computed(value)}
                for consumer in reversed(program.eqns):
                    terminal = 0
                    for out in consumer.outvars:
                        if computed(out):
                            terminal |= reached.get(out, 0)
                    produced = max((math.prod(dimensions(out)) for out in consumer.outvars), default=0)
                    for number, value in enumerate(consumer.invars):
                        if not computed(value):
                            continue
                        producer = self.producers.get(value)
                        reused = (producer is not None and producer.primitive in ELEMENTWISE
                                  and not recomputed(producer, "cpu")
                                  and math.prod(dimensions(value)) < produced)
                        permitted = (id(consumer) in looping and
                                     (consumer.primitive is not jax.lax.dot_general_p
                                      or number in _cpu_dot_inputs(consumer)))
                        ends = root(value) if not permitted or reused else terminal
                        reached[value] = reached.get(value, 0) | ends
                self.absorbed.update(value for value in self.looping
                                     if value not in self.outputs and (ends := reached.get(value, 0))
                                     and ends & (ends - 1) == 0 and ends != roots.get(value)
                                     and self.producers[value].primitive not in PRODUCTS)
                for consumer in program.eqns:
                    if consumer.primitive not in (jax.lax.add_p, primitives.add_jaxvals_p):
                        continue
                    for value in consumer.invars:
                        if (computed(value) and value not in self.outputs and self.readers[value] == 1
                                and consumer.invars.count(value) == 1
                                and (producer := self.producers.get(value)) is not None
                                and _cpu_output_dot(producer)):
                            self.absorbed.add(value)
                # A single-consumer producer eligible for the same library region needs no separate
                # allocation (`LibraryRewriter::AddFusionCandidates`, `GrowFusion`). The lower walk
                # permits that fusion, including reductions and costly elementwise operations.
                pending = [self.producers[value] for value in self.reducing]
                seen: set[int] = set()
                while pending:
                    consumer = pending.pop()
                    if id(consumer) in seen:
                        continue
                    seen.add(id(consumer))
                    for value in consumer.invars:
                        if not computed(value) or value in self.outputs or self.readers[value] != 1:
                            continue
                        producer = self.producers.get(value)
                        if producer is not None and _ynn_eligible(producer):
                            self.absorbed.add(value)
                            pending.append(producer)
                if not returned:
                    # A leaf-local update can consume an elementwise gradient in the loop that writes
                    # its donated parameter, without a separate gradient array.
                    self.absorbed.update(value for value in program.outvars if computed(value)
                                         and self.producers.get(value) is not None
                                         and value in self.looping)

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
        if self.lower and self.platform == "cpu" and equation.primitive in (
                primitives.name_p, primitives.sharding_constraint_p):
            return self.stored(operand)
        if bitcast(equation):
            return operand in self.inputs or not self.fused(operand) or self.view(operand)
        # XLA lays a value out once, so a transpose is its storage only where it is its one reader.
        return (equation.primitive in TRANSPOSES and operand not in self.inputs
                and self.readers[operand] == 1 and (not self.fused(operand) or self.view(operand)))

    def fused(self, value) -> bool:
        """Whether `value` holds no buffer of its own: recomputed where it is read, or a view; a value an
        optimization barrier reads is computed into a buffer of its own, which the barrier's result is
        and every other reader reads (`forwards`), unless it is a view of another's (`view`). Each value's
        answered once (`fusions`)."""
        if value not in self.fusions:
            self.fusions[value] = self._fused(value)
        return self.fusions[value]

    def _fused(self, value) -> bool:
        if not computed(value) or value in self.inputs:
            return False
        if value in self.barred:
            # A barrier's operand holds a buffer unless it is another's storage.
            return self.view(value)
        if value in self.absorbed:
            return True
        equation = self.producers.get(value)
        if (self.lower and self.platform == "cpu" and equation is not None
                and equation.primitive in (primitives.name_p, primitives.sharding_constraint_p)):
            return True
        if (self.lower and self.platform == "cpu" and value in self.looping
                and equation is not None and equation.primitive not in (*ELEMENTWISE, *PRODUCTS)):
            # An unfamiliar positionwise lowering is not proof that its intermediate is stored.
            return value not in self.outputs or self.view(value)
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
        primitive = equation.primitive
        if self.lower and self.platform == "cpu" and any(
                out in self.looping or out in self.absorbed for out in equation.outvars):
            return True
        return (recomputed(equation, self.platform) or primitive is jax.lax.reduce_p or primitive in WINDOWS
                or reduction(primitive) or scatter(primitive))

    def roots(self, values: Iterable) -> set[jcore.Var]:
        """The buffers reading `values` holds (`reading`) and the inputs they are computed from or view: what
        they depend on, the program's inputs among it."""
        found: set[jcore.Var] = set()
        for value in values:
            if computed(value):
                found |= self.root_set(value)
        return found

    def root_set(self, value: jcore.Var) -> frozenset[jcore.Var]:
        """`roots` of `value` alone, each value's found once (`rooted`) from its operands' own: a fused value
        reads what its operands read, every other its own storage."""
        pending = [value]
        while pending:
            current = pending[-1]
            if current in self.rooted:
                pending.pop()
                continue
            if current in self.inputs or not self.fused(current):
                self.rooted[current] = frozenset([current])
                pending.pop()
                continue
            read = [each for each in ([self.forwards[current]] if current in self.forwards
                                      else self.producers[current].invars) if computed(each)]
            waiting = [each for each in read if each not in self.rooted]
            if waiting:
                pending.extend(waiting)
                continue
            self.rooted[current] = frozenset().union(*(self.rooted[each] for each in read))
            pending.pop()
        return self.rooted[value]

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
    counted = _COUNTED.get()
    if counted is not None and not counted(value):
        return 0
    return math.prod(aval.shape) * aval.dtype.itemsize


def recomputed(equation: jcore.JaxprEqn, platform: str) -> bool:
    """Whether XLA's fusion on `platform` recomputes what `equation` computes in each consumer rather than
    keep it: on a GPU every elementwise primitive (`ELEMENTWISE`), elsewhere the cheap ones (`CHEAP`, and
    `REAL_CHEAP` over real values). A program it reads has its calls inlined (`compiled`)."""
    primitive = equation.primitive
    if bitcast(equation):
        # A bitcast holds no buffer of its own, on any platform.
        return True
    if platform == "gpu":
        return primitive in ELEMENTWISE
    if primitive in REAL_CHEAP:
        return not any(isinstance(out.aval, ShapedArray) and np.issubdtype(out.aval.dtype, np.complexfloating)
                       for out in equation.outvars)
    return primitive in CHEAP


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


def resident_bytes(prepared: TrainState) -> dict[str, int]:
    """The prepared state's device bytes by field (`Footprint.resident`)."""
    return {name: device_bytes(value) for name, value in {
        "variables": prepared.variables, "optimizer": prepared.opt_state, "ema": prepared.ema,
        "accumulation": prepared.accumulation, "compute": prepared.compute, "groups": prepared.inner}.items()}


def row_shards(micro: Batch) -> int | None:
    """How many devices of the abstract mesh split `micro`'s rows, where they split nothing else (data and
    fully sharded data parallelism, `row_axes`; an expert axis splits the experts too): a device's share of
    any value of the rows is that many times smaller. None where the layout splits more than the rows
    (stages, tensor, sequence), whose shares the partitioner chooses."""
    mesh = jax.sharding.get_abstract_mesh()
    spread = set() if mesh.empty else {name for name, size in mesh.shape.items() if size > 1}
    rows = set(row_axes(rows_of(micro), mesh=mesh)) & {DATA_AXIS, FSDP_AXIS} if spread else set()
    return math.prod(mesh.shape[name] for name in rows) if spread <= rows else None


def logits_floor[LossT, EffectsT](objective: Objective[LossT, EffectsT], forward: jcore.Jaxpr,
                                  prepared: TrainState, held: int, micro: Batch) -> Footprint | None:
    """The least a step whose head keeps its whole logits holds beside its state, read off its loss's
    `forward` program before its gradient is traced: those logits in fp32, as the head keeps them for
    the backward pass (`HeadedObjective.keeps_whole_logits`), a row of the head's table for each token of
    the states it contracts the table with (`contracted_tokens`), a device's share of them over the
    devices that split `micro`'s rows (`row_shards`), with `held` beside them; a footprint
    whose headroom is an upper bound (`exact` False), so a step it leaves no room refuses on it untraced
    (`Candidate.floored`). None for a head that keeps no whole logits, a forward whose head reads no
    states, or a layout that splits more than the rows (row 5556: Qwen3.5-4B's 4 x 12288 tokens hold
    48.8 GB of them, where its state holds 51.9 GB of an 80 GB device)."""
    table, shards = objective.whole_head_table(prepared.variables), row_shards(micro)
    if table is None or shards is None:
        return None
    tokens = contracted_tokens(forward, table)
    if not tokens:
        return None
    return Footprint(None, device_bytes(prepared.variables["params"]), held, resident_bytes(prepared),
                     math.ceil(tokens / shards) * table[0] * np.dtype(np.float32).itemsize, exact=False)


def contracted_tokens(program: jcore.Jaxpr, table: tuple[int, int]) -> int:
    """The most tokens of states `program` hands a head's table of `table`'s rows and features with, at
    any depth: for each equation that reads the table (a value of its shape, or its transpose), the
    elements of each other operand the features' width ends, over that width (the final states the table
    contracts, `dew.nn.protocols.OutputTable`); none where the table meets no states."""
    shapes = {table, table[::-1]}
    most = 0
    for equation in program.eqns:
        operands = [shape_of(value) for value in equation.invars]
        if any(shape in shapes for shape in operands):
            most = max([most, *(math.prod(shape[:-1]) for shape in operands
                                if shape not in shapes and len(shape) > 1 and shape[-1] == table[1])])
        for body in bodies(equation):
            most = max(most, contracted_tokens(body, table))
    return most


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
        shared = batch_statistics_shapes(objective, current, batch, window, split)
        stats = None if shared is None else tuple(jnp.zeros(x.shape, x.dtype) for x in shared)
        return objective._loss(variables, batch, attempt_step(
            current, variables, ema, batch, split=split, slots=window, attempt=step * split, part=0,
            slot=microstep % window, batch_statistics=stats))

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
    if row_shards(micro) is None or not judged(target, capacity):
        residuals = peak = reads = head = None
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
            residuals, peak, head = derived_terms(
                trained, arguments[0]["params"], arguments, platform=target.platform,
                workspace=gemm_workspace(target), lowering=lowering, consumed=reads, deferred=late,
                table=objective.whole_head_table(variables))
    return Footprint(residuals, device_bytes(prepared.variables["params"]), held, resident_bytes(prepared),
                     None if peak is None else peak.bytes, peak is None or peak.exact, staged, reads, head,
                     None if peak is None else peak.lower)
