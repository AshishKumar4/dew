"""Row RMSNorm, its residual seam and its gated form, and elementwise SwiGLU,
with hand-written reverse passes.

The kernels follow linkedin/Liger-Kernel's ops/rms_norm.py, ops/fused_add_rms_norm.py
and ops/swiglu.py: save only a row's mean square, accumulate weight gradients
per program, and recompute sigmoid from the two narrow SwiGLU inputs in
reverse. The gated norm is Qwen3-Next's output norm (fla's FusedRMSNormGated
computes the same function).

The norm alone lost to XLA's fusions in a Qwen3.5-0.8B full fine-tune (bf16,
4 x 3072 rows, A100-40, main 2215b81fc): the norms cost 16.3 ms a step
against XLA's 12.5. Its reverse pass ran at 44% of HBM bandwidth, the
residual stream's gradient joined it in a separate 72 MiB pass that XLA had
fused into its own reverse fusion, and a latency-bound third kernel summed
the weight gradient. So the norm owns a pre-norm block's seam
(`stream_rms_norm`): its forward reads the stream and the branch once and
writes their sum beside the norm, and its reverse adds the stream's own
cotangent to the norm's in the one pass. Each program keeps its
weight-gradient sums in a register tile, elementwise, and XLA sums the
programs' compensated pairs.

Device time per call at Qwen3.5-0.8B's shapes (4 x 3072 rows of 1024, bf16;
the gated norm over 16 heads of 128, its bar fla 0.5.2's FusedRMSNormGated in
the Liger column) against XLA's fusions of the same math, and the torch bar
(torch 2.11/2.14, Liger 0.8.4), each in one session: the forward, then its
reverse after "/" with the forward recomputed or after ";" alone, in us.

    A100-SXM4-40GB          XLA            fused          Inductor       Liger (fla)
    seam (add + norm)       103 / 200      76 / 187       76 / 179       76 / 299
    gated norm              123 ; 343      118 ; 224      116 ; 218      114 ; 198
    SwiGLU                  235 ; 361      193 ; 320      196 ; 319      195 ; 324
    H100 80GB HBM3 (SXM)
    seam (add + norm)       44 / 89        36 / 81        36 / 88        35 / 163
    gated norm              59 ; 149       52 ; 101       53 ; 105       51 ; 96
    SwiGLU                  113 ; 171      87 ; 147       88 ; 146       88 ; 147

A packed projection's `[..., 2, H]` output takes SwiGLU's own launch
(`packed_swiglu`), which reads both halves and writes their gradient where
the projection put them, so a joined gate and up GEMM costs no copy.

In that fine-tune's profiled step the norms took 19.0 ms on the A100 and 18.6
ms on the H100 SXM under XLA and 15.4 and 13.7 fused; SwiGLU 12.7 and 16.0
against 12.4 and 13.6. The gated norm's reverse is the one behind its bar
(fla's, by 13% on the A100 and 5% on the H100).

Dew's portable rounded_to also rounds cotangents. These fused VJPs instead
use Liger's numerical contract: fp32 intermediate gradients and one cast at
each gradient output, the stream's cotangent included. Forwards retain the
model's cast sites, including cast-before-scale RMSNorm and SiLU rounded
before the up product, and the seam's sum rounds as the block's own add does.
RMSNorm's weight gradient reads that forward's rounded activation; SwiGLU's
backward recomputes SiLU in fp32, as Liger does. At c53's [257,127] bf16
shape, Liger's norm dx was 1.26e-3 RMS from float64 against the rounded VJP's
1.78e-3; SwiGLU's gate gradient was 1.00e-3 against 1.45e-3 (A100).

The fp32 activation/input-gradient tests hold relative differences below
1e-5. Every output and gradient keeps the project's 2x-XLA float64 budget.
Weight-gradient accumulation is separately judged by the exact float64 sum
of the identical rounded factors, within XLA's sum error plus one final
rounding, and at 24,576 rows within 2 ulp outright: every product is kept
exactly as an fp32 pair and compensated (Neumaier) into its program's tile,
the tile's rows meet pairwise with their roundings kept, and the programs'
pairs meet by Knuth's TwoSum (an A100's earlier left fold was 4.91e-3 RMS
from float64 at 24,576 rows against XLA's 1.22e-3). The bf16 tests compare
outputs and every gradient against float64, not bitwise against XLA's
differently rounded autodiff.
"""
import dataclasses
import functools
import math
from typing import ClassVar, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from ..precision import at_least_fp32, compensated_add, compensated_total, rounded_to, two_product
from ..sharding import logical_spec, manual_map
from . import generation, tuning
from .generation import INHERENT, measured_kernel
from .selection import Kernel, Op, Requirement, select

IMPLEMENTATIONS = ('auto', 'xla', 'pallas')
DTYPES = (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16))
ROW_ELEMENTS = 2048
"""The heuristic forward's and reverse's register tile, in elements a program holds of a step's rows."""
ROW_BLOCKS = 12
"""The heuristic reverse's row blocks a program walks at least, so the two fp32
partial rows it writes stay a small share of what it reads."""
MAX_PROGRAMS = 1024
ELEMENT_BLOCK = 1024
PACKED_BLOCK = 4096
"""The packed SwiGLU's heuristic columns a program. Of the configs measured at Qwen3.5-0.8B's
24576 rows of 2 x 3584 (bf16, 2026-10-15), a whole row of 4096 with four warps was the fastest
forward on an A100 80GB and an H100 and within 1% of the fastest reverse, level with the apart
kernel's best (forward 0.323 against 0.315 ms on the A100, 0.181 against 0.178 on the H100), where
512 columns, the largest power of two dividing the width, lost 3 to 10%."""


def _neighbours[C: (RowsConfig, ReverseConfig, BlockConfig)](config: C,
                                                               **steps: tuple[int, int]) -> tuple[C, ...]:
    """`config` first, then each field of `steps` halved and doubled alone, within its `(least, most)`:
    a launch's variants (`tuning.KernelOp.variants`), the configs an opt-in tuning run times beside it."""
    found = [config]
    for name, (least, most) in steps.items():
        value = getattr(config, name)
        found += [dataclasses.replace(config, **{name: other}) for other in (value // 2, value * 2)
                  if least <= other <= most]
    return tuple(dict.fromkeys(found))


WARPS = (1, 8)
"""The warps a program's variants range over: at least one, at most eight."""


def _row_block(width: int) -> int:
    """The heuristic's rows a program holds: at most sixteen, the tile at most
    `ROW_ELEMENTS` elements except when a single row is wider. A 4096-wide
    norm takes one row rather than spilling copies of every backward
    intermediate."""
    return min(16, max(1, ROW_ELEMENTS // pl.next_power_of_2(width)))


@dataclasses.dataclass(frozen=True)
class RowsConfig:
    """The forward's rows a program normalizes at once, and its warps."""
    rows: int
    warps: int


@dataclasses.dataclass(frozen=True)
class ReverseConfig:
    """The reverse's rows a step, its programs and their warps: rows and
    programs first, which order the weight gradient's sums."""
    rows: int
    programs: int
    warps: int


@dataclasses.dataclass(frozen=True)
class BlockConfig:
    """SwiGLU's elements a program and its warps."""
    block: int
    warps: int


def _round(x, dtype, interpret: bool):
    """Triton's explicit narrowing casts survive widening, unlike XLA's
    excess-precision casts. Interpretation uses Dew's explicit rounding."""
    return rounded_to(x, dtype) if interpret else x.astype(dtype).astype(x.dtype)


def _neumaier_add(total, correction, value, *, interpret: bool) -> tuple[jax.Array, jax.Array]:
    """Neumaier's fp32 sum: keep the lost low bits even under cancellation.

    Explicit add/sub rounding protects the residual from compiler fusion
    and cancellation rewrites. CPU interpretation uses optimizer barriers;
    Triton uses round-to-nearest instructions, not a compiler-only option.
    """
    if not interpret:
        shape = jax.ShapeDtypeStruct(total.shape, jnp.float32)
        high = plgpu.elementwise_inline_asm(
            "add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=(total, value),
            result_shape_dtypes=(shape,), pack=1)[0]
        low = plgpu.elementwise_inline_asm(
            """{
            .reg .f32 a, b, e, f, s, c;
            .reg .pred p;
            add.rn.f32 s, $1, $3;
            abs.f32 a, $1;
            abs.f32 b, $3;
            setp.ge.f32 p, a, b;
            sub.rn.f32 e, $1, s;
            add.rn.f32 e, e, $3;
            sub.rn.f32 f, $3, s;
            add.rn.f32 f, f, $1;
            selp.f32 e, e, f, p;
            add.rn.f32 c, $2, e;
            mov.f32 $0, c;
            }""", constraints="=f,f,f,f", args=(total, correction, value),
            result_shape_dtypes=(shape,), pack=1)[0]
        return high, low
    value = jax.lax.optimization_barrier(value)
    merged = jax.lax.optimization_barrier(total + value)
    lost = jnp.where(jnp.abs(total) >= jnp.abs(value),
                     (total - merged) + value, (value - merged) + total)
    return merged, correction + lost


def _product_parts(left, right, *, interpret: bool) -> tuple[jax.Array, jax.Array]:
    """The rounded fp32 product and its lost bits, without a float64 dtype."""
    if not interpret:
        shape = jax.ShapeDtypeStruct(left.shape, jnp.float32)
        high = plgpu.elementwise_inline_asm(
            "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=(left, right),
            result_shape_dtypes=(shape,), pack=1)[0]
        low = plgpu.elementwise_inline_asm(
            """{
            .reg .f32 e;
            neg.f32 e, $3;
            fma.rn.f32 e, $1, $2, e;
            mov.f32 $0, e;
            }""", constraints="=f,f,f,f", args=(left, right, high),
            result_shape_dtypes=(shape,), pack=1)[0]
        return high, low
    return two_product(left, right, bit_split=True, rounding=jax.lax.optimization_barrier)


def _selected[C](op: Op[C], case: C, implementation: str) -> str:
    """The kernel `op` runs for `case`, reported as the target's route (`selection.select`): the one a
    per-call `implementation` names, as the call's preference, a refused one handing the call to XLA
    with its refusal; for 'auto', the one `chosen` names, else the target's preference."""
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")
    return select(op, case, None if implementation == 'auto' else implementation, insist=False).kernel


QUALIFY_FP16 = 'ledger 823'
"""The row that tracks qualifying the fused norm and SwiGLU kernels on fp16 operands."""


def _within_fp32(dtype) -> bool:
    """Whether a dtype's accuracy is no finer than fp32's, which the kernels compute in; a dtype that
    is no float falls to the `dtype` requirement."""
    return not jnp.issubdtype(dtype, jnp.floating) or jnp.finfo(dtype).bits <= 32


class _Typed(Protocol):
    """A case that names every dtype its call reads and writes."""

    @property
    def dtypes(self) -> tuple[np.dtype, ...]: ...


def _dtype_requirements[C: _Typed](name: str, case_type: type[C]) -> tuple[Requirement[C], ...]:
    """What the norm and SwiGLU kernels need of every dtype a call of `case_type` reads and writes (its
    `dtypes`): none wider than fp32, whose accuracy the kernels compute at; none fp16, which they are
    not yet qualified on (ledger 823); and none but fp32, bf16 and fp16, the floats they compute."""
    return (
        Requirement[C]('precision', lambda case: all(_within_fp32(dtype) for dtype in case.dtypes),
                    lambda case: f"{name} reads or writes {', '.join(map(str, case.dtypes))}; the kernel "
                                 "computes in fp32, which a wider dtype's accuracy needs more than",
                    row=INHERENT),
        Requirement[C]('fp16', lambda case: jnp.float16 not in case.dtypes,
                    f"{name}'s kernel is not yet qualified on float16 operands", row=QUALIFY_FP16),
        Requirement[C]('dtype', lambda case: all(dtype in (*DTYPES, jnp.float16) for dtype in case.dtypes),
                    lambda case: f"{name} reads or writes {', '.join(map(str, case.dtypes))}; the kernel "
                                 "computes float32, bfloat16 and float16 alone", row=INHERENT),
    )


def _runs(case) -> bool:
    """Whether the Pallas kernels run for `case` here: a GPU Dew's Triton kernels run on, or on a CPU
    interpreted where the call names them (`case.named`)."""
    return generation.triton_runs() or (case.named == 'pallas' and generation.backend() == 'cpu')


def _split() -> bool:
    """Whether a mesh axis the context leaves to the partitioner splits anything."""
    mesh = jax.sharding.get_abstract_mesh()
    return any(mesh.shape[name] > 1 for name in mesh.axis_names if name not in mesh.manual_axes)


def _placed(axes, operands, *, whole_last: bool) -> bool:
    """Whether the kernel can run on its `operands`' shards: none varies in a
    `shard_map` that checks varying axes, and on a split mesh the caller
    names the first one's logical axes (`axes`), which with `whole_last`
    leave its last axis whole."""
    if any(jax.typeof(operand).mat.varying for operand in operands if operand is not None):
        return False
    if not _split():
        return True
    first = operands[0]
    return axes is not None and (not whole_last or len(logical_spec(axes, first.shape)) < first.ndim)


def _per_shard(local, axes, operands, replicated):
    """`local(*operands)` as it runs: whole where no mesh axis splits, else on
    each shard (`manual_map`), the operands placed as `axes` names and those
    marked `replicated` (a norm's weight) whole on every shard, so the weight's
    gradient sums over the shards. Every output takes the first operand's place."""
    if not _split():
        return local(*operands)
    spec = logical_spec(axes, operands[0].shape)
    specs = tuple(None if operand is None else jax.sharding.PartitionSpec() if whole else spec
                  for operand, whole in zip(operands, replicated, strict=True))
    return manual_map(local, specs, spec)(*operands)


@dataclasses.dataclass(frozen=True)
class NormCase:
    """A norm call as the norm selector reads it (`NORM`): the `dtypes` it reads and writes (its input,
    its output, its weight's and its gate's where it has them), its `width` (0 without a feature axis)
    and whether it is `empty`; whether its weight is a vector over the features (`scale_fits`), a
    residual branch is the input's shape and dtype (`branch_fits`) and a gate its shape
    (`gate_fits`), and its operands' mesh placement fits (`placed`); whether an activation follows it
    (`activated`), its statistics are fp32's, and a gated norm's `activation` (None for a plain one);
    and the implementation the call `named`, None for 'auto'."""
    dtypes: tuple[np.dtype, ...]
    width: int
    empty: bool
    scale_fits: bool
    branch_fits: bool
    gate_fits: bool
    placed: bool
    activated: bool
    fp32_statistics: bool
    activation: str | None
    named: str | None


def norm_case(implementation: str, x, scale, dtype, axes, *, branch=None, gate=None, activated=False,
              fp32_statistics=True, activation=None) -> NormCase:
    """The `NormCase` of a norm over `x` and its weight `scale`, a residual `branch` and a `gate`. A
    branch adds in `x`'s dtype, as the stream's own add does; a gate is read in fp32 whatever its float
    dtype, as the XLA arithmetic widens it."""
    return NormCase(
        dtypes=tuple(jnp.dtype(dtype_of) for dtype_of in (x.dtype, dtype, *(
            operand.dtype for operand in (scale, gate) if operand is not None))),
        width=x.shape[-1] if x.shape else 0, empty=not x.shape or x.size == 0,
        scale_fits=scale is None or scale.shape == x.shape[-1:],
        branch_fits=branch is None or (branch.shape == x.shape and branch.dtype == x.dtype),
        gate_fits=gate is None or gate.shape == x.shape,
        placed=_placed(axes, (x, scale, branch, gate), whole_last=True), activated=activated,
        fp32_statistics=fp32_statistics, activation=activation,
        named=None if implementation == 'auto' else implementation)


NORM_SUPPORTS: tuple[Requirement[NormCase], ...] = (
    Requirement('capability', _runs, "the Pallas kernels need a GPU of sm80 or later", row=INHERENT),
    Requirement('activation', lambda case: not case.activated,
                "the norm is followed by an activation, which the kernel does not apply",
                row='ledger 821'),
    Requirement('statistics', lambda case: case.fp32_statistics,
                "input-dtype statistics round each reduction step; the kernel reduces in fp32",
                row=INHERENT),
    Requirement('gate_activation', lambda case: case.activation in (None, 'silu', 'sigmoid'),
                lambda case: f"the gated kernel gates by silu or sigmoid, not {case.activation!r}",
                row=INHERENT),
    *_dtype_requirements('RMSNorm', NormCase),
    Requirement('empty', lambda case: not case.empty, "RMSNorm needs nonempty rows with a feature axis",
                row=INHERENT),
    Requirement('width', lambda case: 0 < case.width <= 8192,
                "RMSNorm's register tile holds at most 8192 features", row='ledger 820'),
    Requirement('scale', lambda case: case.scale_fits,
                "the kernel takes a vector scale over the features, not a broadcast affine tensor",
                row=INHERENT),
    Requirement('branch', lambda case: case.branch_fits,
                "a residual branch must match the normalized input's shape and dtype", row=INHERENT),
    Requirement('gate', lambda case: case.gate_fits, "a gate must match the normalized input's shape",
                row=INHERENT),
    Requirement('placement', lambda case: case.placed,
                "a mesh splits RMSNorm's operands by axes the caller does not name or across its features, "
                "or a shard_map that checks varying axes holds them, and a Pallas call states no varying "
                "type", row='ledger 822'),
)
"""The fused norms' requirements: a plain, stream and gated RMSNorm over fp32 or bf16 rows of at most
8192 features, with fp32 statistics, a vector weight, and operands placed as the kernel reads them."""

NORM: Op[NormCase] = Op('rms_norm', (Kernel('pallas', NORM_SUPPORTS), Kernel('xla', (), generic=True)),
                        lambda case: measured_kernel('rms_norm', 'xla'))
"""The norm selector: the fused kernel where its requirements hold and the target's row names it,
else XLA's arithmetic."""


def rms_kernel(implementation: str, x, scale, dtype, fp32_statistics: bool, *,
               branch=None, axes=None, activated: bool = False) -> str:
    """Resolve RMSNorm without changing its arithmetic or collecting a split
    feature axis. Float64 and input-dtype statistics belong to XLA, and so
    do a residual `branch` the stream's own add would broadcast or cast and
    an activation after the norm. `axes`, `x`'s logical axes, places the
    kernel on a mesh (`_placed`)."""
    case = norm_case(implementation, x, scale, dtype, axes, branch=branch, activated=activated,
                     fp32_statistics=fp32_statistics)
    return _selected(NORM, case, implementation)


def gated_rms_kernel(implementation: str, x, gate, scale, dtype, activation: str, *, axes=None) -> str:
    """Resolve the gated RMSNorm (`gated_rms_norm`), the one kernel family
    `KERNELS` names 'rms_norm'; other gates keep their XLA arithmetic."""
    return _selected(NORM, norm_case(implementation, x, scale, dtype, axes, gate=gate, activation=activation),
                     implementation)


@dataclasses.dataclass(frozen=True)
class SwigluCase:
    """A SiLU product as the SwiGLU selector reads it (`SWIGLU`): its operands' `dtypes` and the dtype
    their product `promotes` to (weak types honoured, as the operands' own promotion is), whether they
    share a shape the kernel reads (`shape_fits`), whether their mesh placement fits (`placed`), and the
    implementation the call `named`, None for 'auto'."""
    dtypes: tuple[np.dtype, ...]
    promotes: np.dtype
    shape_fits: bool
    placed: bool
    named: str | None


SWIGLU_SUPPORTS: tuple[Requirement[SwigluCase], ...] = (
    Requirement('capability', _runs, "the Pallas kernels need a GPU of sm80 or later", row=INHERENT),
    *_dtype_requirements('SwiGLU', SwigluCase),
    Requirement('matching', lambda case: len(set(case.dtypes)) == 1,
                lambda case: f"SwiGLU of {' and '.join(map(str, case.dtypes))} operands; the kernel reads "
                             "one dtype, and XLA's product promotes mixed ones", row=INHERENT),
    Requirement('shape', lambda case: case.shape_fits,
                "SwiGLU needs nonempty equal shapes, not broadcasting, or a packed [..., 2, H] gate and up",
                row=INHERENT),
    Requirement('placement', lambda case: case.placed,
                "a mesh splits SwiGLU's operands by axes the caller does not name, or a shard_map that "
                "checks varying axes holds them, and a Pallas call states no varying type",
                row='ledger 822'),
)
"""The fused SiLU products' requirements: fp32 or bf16 operands of one shape, apart or packed, placed
as the kernel reads them."""


def _swiglu_preferred(case: SwigluCase) -> str:
    """The kernel measured fastest for a SiLU product of the case's dtype (its operands' common dtype,
    so a bf16 gate over an fp32 up, Gemma-4's per-layer input, is an fp32 product): the target's row
    for a bf16 one, which every row measured, and XLA for an fp32 one, which fuses the product's
    converts and neighbours into one pass the kernel's launch splits (8 layers of Dory's 1 x 2048 nGPT
    block on an A100: 5.4 ms a step through the kernel, 2.6 through XLA). A kernel named per call or
    by `chosen` runs as named."""
    return 'xla' if case.promotes == jnp.float32 else measured_kernel('swiglu', 'xla')


SWIGLU: Op[SwigluCase] = Op('swiglu', (Kernel('pallas', SWIGLU_SUPPORTS), Kernel('xla', (), generic=True)),
                            _swiglu_preferred)
"""The SwiGLU selector: the fused product where its requirements hold and the target prefers it, else
XLA's arithmetic."""


def swiglu_kernel(implementation: str, gate, up, *, axes=None) -> str:
    """Resolve the SiLU product only. GeGLU and SiTU keep their own XLA
    arithmetic; broadcasting or mixed operand dtypes also stay there.
    `axes`, the operands' logical axes, places the kernel on a mesh (`_placed`)."""
    case = SwigluCase(dtypes=(jnp.dtype(gate.dtype), jnp.dtype(up.dtype)),
                      promotes=jnp.dtype(jnp.result_type(gate, up)),
                      shape_fits=gate.shape == up.shape and bool(gate.shape) and gate.size > 0,
                      placed=_placed(axes, (gate, up), whole_last=False),
                      named=None if implementation == 'auto' else implementation)
    return _selected(SWIGLU, case, implementation)


def packed_swiglu_kernel(implementation: str, gate_up, *, axes=None) -> str:
    """Resolve the SiLU product of one packed projection's `[..., 2, H]` output (`packed_swiglu`),
    the kernel `KERNELS` names 'swiglu', which reads the halves where the projection wrote them.
    `axes` are the product's logical axes, `[..., H]`'s."""
    case = SwigluCase(dtypes=(jnp.dtype(gate_up.dtype),), promotes=jnp.dtype(gate_up.dtype),
                      shape_fits=gate_up.ndim >= 2 and gate_up.shape[-2] == 2 and gate_up.size > 0,
                      placed=_placed(None if axes is None else (*axes[:-1], None, axes[-1]), (gate_up,),
                                     whole_last=False),
                      named=None if implementation == 'auto' else implementation)
    return _selected(SWIGLU, case, implementation)


def _weight(scale_ref, columns, width: int, offset: bool, stored, interpret: bool):
    weight = plgpu.load(scale_ref.at[columns], mask=columns < width, other=0).astype(jnp.float32)
    # The offset adds in the stored weight's dtype, including a bf16 weight.
    return _round(weight + 1.0, stored, interpret) if offset else weight


def activation_and_slope(gate, activation: str) -> tuple[jax.Array, jax.Array]:
    """An fp32 gate's `act(gate)` and its derivative, for 'silu' or 'sigmoid'.

    silu's derivative `s + g s (1 - s)`, with `s = sigmoid(g)`, cancels to
    zero near g = -1.28; formed as the product rule forms it, `s + g (s (1 - s))`,
    its worst element there is half `(g s)(1 - s) + s`'s, as XLA's is."""
    sigmoid = jax.nn.sigmoid(gate)
    slope = sigmoid * (1.0 - sigmoid)
    if activation == 'sigmoid':
        return sigmoid, slope
    return gate * sigmoid, sigmoid + gate * slope


def _rms_forward(x_ref, branch_ref, scale_ref, gate_ref, y_ref, stream_ref, mean_square_ref, *,
                 row_block: int, epsilon: float, offset: bool, stored, after_cast: bool,
                 inverse_rounded: bool, activation: str, interpret: bool):
    rows, width = x_ref.shape
    columns = jnp.arange(pl.next_power_of_2(width))
    row = pl.program_id(0) * row_block + jnp.arange(row_block)
    at = (row[:, None], columns[None, :])
    mask = (row[:, None] < rows) & (columns[None, :] < width)
    x = plgpu.load(x_ref.at[at], mask=mask, other=0).astype(jnp.float32)
    if branch_ref is not None:
        # The residual sum rounds to the stream's dtype, as the block's own add does.
        branch = plgpu.load(branch_ref.at[at], mask=mask, other=0)
        x = _round(x + branch.astype(jnp.float32), stream_ref.dtype, interpret)
        plgpu.store(stream_ref.at[at], x.astype(stream_ref.dtype), mask=mask)
    # XLA's mean multiplies by the fp32 reciprocal, not an fp32 division.
    mean_square = jnp.sum(x * x, axis=1) * (1.0 / width) + epsilon
    inverse = jax.lax.rsqrt(mean_square)
    y = x * _rounded_inverse(inverse, inverse_rounded, y_ref.dtype, interpret)[:, None]
    if scale_ref is not None:
        if after_cast:
            y = _round(y, y_ref.dtype, interpret)
        y = y * _weight(scale_ref, columns, width, offset, stored, interpret)[None, :]
    if gate_ref is not None:
        # The gate multiplies the weighted norm in fp32, before the one cast.
        y = y * activation_and_slope(plgpu.load(gate_ref.at[at], mask=mask, other=0).astype(jnp.float32),
                                     activation)[0]
    plgpu.store(y_ref.at[at], y.astype(y_ref.dtype), mask=mask)
    plgpu.store(mean_square_ref.at[row], mean_square, mask=row < rows)


def _rounded_inverse(inverse, rounded: bool, dtype, interpret: bool):
    """The fp32 inverse RMS a row multiplies by: rounded to `dtype` first where the model rounds it
    (`dew.nn.attention.unweighted_rmsnorm`), whose product with the input then rounds once."""
    return _round(inverse, dtype, interpret) if rounded else inverse


def _rsqrt_slope(inverse, mean_square):
    """The derivative of each row's inverse RMS by its mean square (with epsilon), as XLA's rsqrt
    transposes it: `-0.5 * (inverse / mean_square)`, both the forward's own values. Where the direct
    and reduced cotangents cancel, this keeps the inverse's one rounding in both; the cube of the
    inverse took it three times (a row of 0.5 under cotangents of 1.5 * 2**119 came out 4 times XLA's
    distance from float64, TIER13 on an A100 and an H100)."""
    return -0.5 * (inverse / mean_square)


def _rounded_inverse_cotangent(m, x, multiplier, slope, narrow, width: int, interpret: bool):
    """The input's cotangent, `[rows, width]` in fp32, of `x * inverse.astype(narrow)` for the
    cotangent `m`, where XLA and torch round it: the direct product and the inverse's cotangent, its
    products summed, each round to `narrow`, and the reduction's contribution rounds apart before the
    two join (the store rounds the join). A rounding the inverse took then recurs in both, so where
    they cancel, it cancels too. `slope` is `_rsqrt_slope`'s; `width` is the row's own."""
    direct = _round(m * multiplier[:, None], narrow, interpret)
    sensed = _round(jnp.sum(_round(m * x, narrow, interpret), axis=1), narrow, interpret)
    # rsqrt's half in its coefficient and the square's two apart, as XLA's transposes keep them, so no
    # product overflows, nor a small cotangent's halving rounds, where XLA's does not.
    correction = sensed * slope * (1.0 / width)
    reduced = _round((2.0 * x) * correction[:, None], narrow, interpret)
    return direct + reduced


def _pairwise(values, *, interpret: bool) -> tuple[jax.Array, jax.Array]:
    """A power-of-two tile's rows summed pairwise, each level's rounding
    kept: the fp32 sum per column and the low bits it lost."""
    lost = jnp.zeros_like(values)
    while values.shape[0] > 1:
        left, right = jnp.split(values, 2, axis=0)
        left_lost, right_lost = jnp.split(lost, 2, axis=0)
        values, lost = _neumaier_add(left, left_lost + right_lost, right, interpret=interpret)
    return values.reshape(values.shape[1:]), lost.reshape(lost.shape[1:])


def _rms_backward(x_ref, scale_ref, gate_ref, mean_square_ref, dy_ref, stream_ref, dx_ref, dgate_ref,
                  partial_ref, correction_ref, *, row_block: int, offset: bool, stored, after_cast: bool,
                  inverse_rounded: bool, activation: str, interpret: bool):
    rows, width = x_ref.shape
    columns = jnp.arange(pl.next_power_of_2(width))
    groups = pl.num_programs(0)
    group = pl.program_id(0)
    weight = _weight(scale_ref, columns, width, offset, stored, interpret) if scale_ref is not None else 1.0

    def block(i, accumulated):
        row = (i * groups + group) * row_block + jnp.arange(row_block)
        at = (row[:, None], columns[None, :])
        mask = (row[:, None] < rows) & (columns[None, :] < width)

        def streamed(ref):
            # Read once and never again: first out of L2, so the weight and
            # the partial sums stay. All three are issued before the row
            # reduction waits on any of them.
            return plgpu.load(ref.at[at], mask=mask, other=0, eviction_policy="evict_first")

        x = streamed(x_ref).astype(jnp.float32)
        dy = streamed(dy_ref).astype(jnp.float32)
        ds = None if stream_ref is None else streamed(stream_ref)
        # The forward's mean square, and its inverse root as the forward took it.
        mean_square = plgpu.load(mean_square_ref.at[row], mask=row < rows, other=1)
        inverse = jax.lax.rsqrt(mean_square)
        # The rounded inverse multiplies the input; the rounding passes its cotangent on unchanged.
        multiplier = _rounded_inverse(inverse, inverse_rounded, dy_ref.dtype, interpret)
        normalized = x * multiplier[:, None]
        if after_cast:
            normalized = _round(normalized, dy_ref.dtype, interpret)
        if gate_ref is not None:
            # The gate's cotangent reads the weighted norm; the norm's, the gated one.
            activated, slope = activation_and_slope(streamed(gate_ref).astype(jnp.float32), activation)
            dgate = dy * (normalized * weight) * slope
            plgpu.store(dgate_ref.at[at], dgate.astype(dgate_ref.dtype), mask=mask,
                        eviction_policy="evict_first")
            dy = dy * activated
        m = dy * weight
        slope = _rsqrt_slope(inverse, mean_square)
        if inverse_rounded:
            dx = _rounded_inverse_cotangent(m, x, multiplier, slope, dy_ref.dtype, width, interpret)
        else:
            # The two x contributions are kept separate, as XLA's square VJP is.
            correction = jnp.sum(m * x, axis=1) * slope * (1.0 / width)
            dx = m * multiplier[:, None] + (2.0 * x) * correction[:, None]
        if ds is not None:
            # The stream's own cotangent joins in fp32, one rounding for the sum.
            dx = dx + ds.astype(jnp.float32)
        plgpu.store(dx_ref.at[at], dx.astype(dx_ref.dtype), mask=mask, eviction_policy="evict_first")
        if scale_ref is None:
            return accumulated
        # Every product exact as a pair, compensated (Neumaier) into the
        # program's own tile of sums: elementwise, nothing crosses threads
        # until the tile's rows meet pairwise once, after the loop.
        product, low = _product_parts(dy, normalized, interpret=interpret)
        total, kept = _neumaier_add(accumulated[0], accumulated[1], product, interpret=interpret)
        return total, kept + low

    shape = (row_block, columns.shape[0])
    partial, correction = jax.lax.fori_loop(0, pl.cdiv(rows, groups * row_block), block,
                                           (jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32)))
    if partial_ref is not None:
        partial, lost = _pairwise(partial, interpret=interpret)
        plgpu.store(partial_ref.at[group, columns], partial, mask=columns < width)
        plgpu.store(correction_ref.at[group, columns], lost + jnp.sum(correction, axis=0),
                    mask=columns < width)


def _scale_sum(partial, correction):
    """The programs' compensated parts summed over programs as compensated
    pairs (`compensated_add`), one XLA column reduction, read as every
    compensated sum is (`compensated_total`): the pair rounded once, or the
    programs' plain sum where that reaches an infinity, whose residual is
    inf - inf."""
    zero = jnp.zeros((), jnp.float32)
    pair = jax.lax.reduce((partial, correction), (zero, zero), compensated_add, (0,))
    return compensated_total(pair, jnp.sum(partial, axis=0))


class _Norm(NamedTuple):
    """One fused norm's static arithmetic: everything but its operands. The
    weight reaches the kernels in fp32, `stored` its own dtype, in which the
    offset adds; the activation gates only a norm called with a gate."""
    epsilon: float
    dtype: str
    offset: bool
    stored: str
    after_cast: bool
    activation: str
    interpret: bool
    inverse_rounded: bool = False


@dataclasses.dataclass(frozen=True)
class RmsForward(tuning.KernelOp):
    """The norm's forward launch over `[rows, width]` (`_rms_forward`): the
    normed rows, the stream `x + branch` where a branch joins, and each
    row's mean square with epsilon, whose inverse root the norm multiplies by."""

    config_cls: ClassVar[type[RowsConfig]] = RowsConfig

    ROWS: ClassVar[tuple[str, ...]] = ('x', 'branch', 'gate')

    def _fwd(self, x: jax.Array, branch: jax.Array | None, scale: jax.Array | None, gate: jax.Array | None, *,
             epsilon: float, dtype: str, offset: bool, stored: str, after_cast: bool, activation: str,
             interpret: bool, inverse_rounded: bool, config: RowsConfig, return_residuals: bool = False):
        rows, width = x.shape
        return pl.pallas_call(
            functools.partial(_rms_forward, row_block=config.rows, epsilon=epsilon, offset=offset,
                              stored=jnp.dtype(stored), after_cast=after_cast,
                              inverse_rounded=inverse_rounded, activation=activation, interpret=interpret),
            grid=(pl.cdiv(rows, config.rows),),
            out_shape=(jax.ShapeDtypeStruct((rows, width), jnp.dtype(dtype)),
                       None if branch is None else jax.ShapeDtypeStruct((rows, width), x.dtype),
                       jax.ShapeDtypeStruct((rows,), jnp.float32)),
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=1),
            interpret=interpret, name="rms_norm_forward",
        )(x, branch, scale, gate), None

    def _get_heuristics_config(self, ba) -> RowsConfig:
        return RowsConfig(_row_block(ba.arguments['x'].shape[1]), 4)

    def variants(self, ba) -> tuple[RowsConfig, ...]:
        # A program's rows within twice the heuristic's register tile.
        rows = max(1, 2 * ROW_ELEMENTS // pl.next_power_of_2(ba.arguments['x'].shape[1]))
        return _neighbours(ba.heuristics_config, rows=(1, rows), warps=WARPS)


@dataclasses.dataclass(frozen=True)
class RmsBackward(tuning.KernelOp):
    """The norm's reverse launch (`_rms_backward`): the input's gradient with
    the stream's own cotangent `ds` added, the gate's, and each program's
    compensated partial sums of the weight's."""

    config_cls: ClassVar[type[ReverseConfig]] = ReverseConfig

    ROWS: ClassVar[tuple[str, ...]] = ('x', 'gate', 'mean_square', 'dy', 'ds')

    def _fwd(self, x: jax.Array, scale: jax.Array | None, gate: jax.Array | None, mean_square: jax.Array,
             dy: jax.Array, ds: jax.Array | None, *, offset: bool, stored: str, after_cast: bool,
             inverse_rounded: bool, activation: str, interpret: bool, config: ReverseConfig,
             return_residuals: bool = False):
        rows, width = x.shape
        groups = max(1, min(config.programs, pl.cdiv(rows, config.rows)))
        partial = None if scale is None else jax.ShapeDtypeStruct((groups, width), jnp.float32)
        return pl.pallas_call(
            functools.partial(_rms_backward, row_block=config.rows, offset=offset, stored=jnp.dtype(stored),
                              after_cast=after_cast, inverse_rounded=inverse_rounded, activation=activation,
                              interpret=interpret),
            grid=(groups,),
            out_shape=(jax.ShapeDtypeStruct((rows, width), x.dtype),
                       None if gate is None else jax.ShapeDtypeStruct((rows, width), gate.dtype),
                       partial, partial),
            compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=1),
            interpret=interpret, name="rms_norm_backward",
        )(x, scale, gate, mean_square, dy, ds), None

    def _get_heuristics_config(self, ba) -> ReverseConfig:
        rows, width = ba.arguments['x'].shape
        block = _row_block(width)
        return ReverseConfig(block, max(1, min(MAX_PROGRAMS, pl.cdiv(rows, block * ROW_BLOCKS))), 4)

    def variants(self, ba) -> tuple[ReverseConfig, ...]:
        rows = max(1, 2 * ROW_ELEMENTS // pl.next_power_of_2(ba.arguments['x'].shape[1]))
        return _neighbours(ba.heuristics_config, rows=(1, rows), programs=(1, 2 * MAX_PROGRAMS),
                           warps=WARPS)


_RMS_FORWARD = RmsForward()
_RMS_BACKWARD = RmsBackward()


def _flat(value):
    return None if value is None else value.reshape(math.prod(value.shape[:-1]), value.shape[-1])


def _rms_with_mean_square(x, branch, scale, gate, norm: _Norm):
    """`(stream, y)` and each row's mean square with epsilon, the stream `x + branch`, or `x` itself."""
    y, stream, mean_square = _RMS_FORWARD(_flat(x), _flat(branch), scale, _flat(gate), **norm._asdict())
    return (x if branch is None else stream.reshape(x.shape), y.reshape(x.shape)), mean_square


def _rms_backward_call(stream, scale, gate, mean_square, dy, ds, norm: _Norm):
    """The stream's cotangent (the norm's plus `ds`, None for none), the
    gate's and the weight's."""
    if dy is None:
        zeros = [None if value is None else jnp.zeros_like(value) for value in (stream, gate, scale)]
        return (zeros[0] if ds is None else ds), zeros[1], zeros[2]
    dx, dgate, partial, correction = _RMS_BACKWARD(
        _flat(stream), scale, _flat(gate), mean_square, _flat(dy), _flat(ds), offset=norm.offset,
        stored=norm.stored, after_cast=norm.after_cast, inverse_rounded=norm.inverse_rounded,
        activation=norm.activation, interpret=norm.interpret)
    dscale = None if scale is None else _scale_sum(partial, correction)
    return dx.reshape(stream.shape), None if gate is None else dgate.reshape(gate.shape), dscale


def _cotangent(value):
    return None if isinstance(value, SymbolicZero) else value


def _norm(epsilon: float, dtype, offset: bool, after_cast: bool, scale, activation: str = 'silu') -> _Norm:
    stored = jnp.dtype(jnp.float32 if scale is None else scale.dtype).name
    return _Norm(float(epsilon), jnp.dtype(dtype).name, offset, stored, after_cast, activation,
                 not generation.triton_runs())


def _widened(scale):
    """The weight in fp32 for the kernels, so its gradient is summed over a
    mesh's row shards before the cast back to its own dtype rounds it."""
    return None if scale is None else scale.astype(at_least_fp32(scale.dtype))


def _stored(portable, norm: _Norm, at: int):
    """`portable` given the weight, operand `at`, in its stored dtype: the kernels read it widened
    (`_widened`) and round it back there (`_weight`), and the tangent is the widened weight's."""
    def stated(*operands):
        weight = operands[at]
        stored = None if weight is None else weight.astype(norm.stored)
        return portable(*operands[:at], stored, *operands[at + 1:])

    return stated


def stream_rms_norm(x, branch, scale, epsilon: float, dtype, offset: bool, after_cast: bool, portable, *,
                    axes=None):
    """The pre-norm seam fused: `(stream, y)`, the stream `x + branch` and its RMSNorm, `portable(x, branch,
    scale)` its arithmetic in XLA. The forward reads both addends once and writes the sum beside the norm;
    the reverse takes the stream's own cotangent into the norm's pass, so the sum of the two gradients is
    never a separate pass over the stream; an output nothing reads still gives the reverse a cotangent
    of zeros, which `linear_call`'s transpose instantiates, and a pre-norm seam reads both. A None
    branch is `rms_norm`'s, the stream `x` itself. It
    interprets where no Triton GPU runs it (`generation.triton_runs`), as the CPU suite does, and lowers
    for the assumed GPU within `generation.assumed`, as the GDN kernels decide. Every order of
    derivative goes through `portable` and each level's value through the kernels
    (`generation.linear_tangent`)."""
    if branch is None:
        return x, rms_norm(x, scale, epsilon, dtype, offset, after_cast,
                           lambda x, scale: portable(x, None, scale)[1], axes=axes)
    norm = _norm(epsilon, dtype, offset, after_cast, scale)

    def forward(x, branch, scale):
        outputs, mean_square = _rms_with_mean_square(x, branch, scale, None, norm)
        return outputs, (outputs[0], mean_square)

    def backward(saved, operands, cotangents):
        stream, mean_square = saved
        ds, dy = map(_cotangent, cotangents)
        dstream, _, dscale = _rms_backward_call(stream, operands[2], None, mean_square, dy, ds, norm)
        # The sum's two addends share its cotangent: one buffer, read by both.
        return dstream, dstream, dscale

    local = generation.linear_tangent(forward, backward, _stored(portable, norm, 2))
    return _per_shard(local, axes, (x, branch, _widened(scale)), (False, False, True))


def rms_norm(x, scale, epsilon: float, dtype, offset: bool, after_cast: bool, portable, *, axes=None,
             inverse_rounded: bool = False):
    """The fused RMSNorm alone, `portable(x, scale)` its arithmetic in XLA, differentiated as
    `stream_rms_norm` is. With `inverse_rounded` the inverse RMS rounds to `dtype` before it
    multiplies, as `unweighted_rmsnorm` rounds it."""
    norm = _norm(epsilon, dtype, offset, after_cast, scale)._replace(inverse_rounded=inverse_rounded)

    def forward(x, scale):
        (_, y), mean_square = _rms_with_mean_square(x, None, scale, None, norm)
        return y, mean_square

    def backward(mean_square, operands, dy):
        x, scale = operands
        dx, _, dscale = _rms_backward_call(x, scale, None, mean_square, dy, None, norm)
        return dx, dscale

    local = generation.linear_tangent(forward, backward, _stored(portable, norm, 1))
    return _per_shard(local, axes, (x, _widened(scale)), (False, True))


def gated_rms_norm(x, gate, scale, epsilon: float, dtype, activation: str, portable, *, axes=None):
    """The gated RMSNorm fused, Qwen3-Next's: `x` normalized in fp32, times `scale`, times
    `activation(gate)` ('silu' or 'sigmoid'), cast once to `dtype`; `portable(x, gate, scale)` its
    arithmetic in XLA. The reverse pass reads `x`, `gate` and the cotangent once and writes both input
    gradients beside the weight's partial sums; it is differentiated as `stream_rms_norm` is."""
    norm = _norm(epsilon, dtype, offset=False, after_cast=False, scale=scale, activation=activation)

    def forward(x, gate, scale):
        (_, y), mean_square = _rms_with_mean_square(x, None, scale, gate, norm)
        return y, mean_square

    def backward(mean_square, operands, dy):
        x, gate, scale = operands
        return _rms_backward_call(x, scale, gate, mean_square, dy, None, norm)

    local = generation.linear_tangent(forward, backward, _stored(portable, norm, 2))
    return _per_shard(local, axes, (x, gate, _widened(scale)), (False, False, True))


def _product(gate, up, dtype, interpret: bool):
    """SwiGLU of fp32 halves: SiLU rounded to the operands' `dtype`, as the model rounds it, times up."""
    return _round(gate * jax.nn.sigmoid(gate), dtype, interpret) * up


def _gradients(gate, up, dy):
    """SwiGLU's fp32 gradients of its gate and up, SiLU recomputed from the gate (`activation_and_slope`)."""
    activated, slope = activation_and_slope(gate, 'silu')
    return dy * slope * up, dy * activated


def _swiglu_forward(gate_ref, up_ref, y_ref, *, block: int, interpret: bool):
    index = pl.program_id(0) * block + jnp.arange(block)
    mask = index < gate_ref.shape[0]
    gate = plgpu.load(gate_ref.at[index], mask=mask, other=0).astype(jnp.float32)
    up = plgpu.load(up_ref.at[index], mask=mask, other=0).astype(jnp.float32)
    plgpu.store(y_ref.at[index], _product(gate, up, gate_ref.dtype, interpret).astype(y_ref.dtype), mask=mask)


def _swiglu_backward(gate_ref, up_ref, dy_ref, dg_ref, du_ref, *, block: int, interpret: bool):
    index = pl.program_id(0) * block + jnp.arange(block)
    mask = index < gate_ref.shape[0]
    gate = plgpu.load(gate_ref.at[index], mask=mask, other=0).astype(jnp.float32)
    up = plgpu.load(up_ref.at[index], mask=mask, other=0).astype(jnp.float32)
    dy = plgpu.load(dy_ref.at[index], mask=mask, other=0).astype(jnp.float32)
    dg, du = _gradients(gate, up, dy)
    plgpu.store(dg_ref.at[index], dg.astype(gate_ref.dtype), mask=mask)
    plgpu.store(du_ref.at[index], du.astype(up_ref.dtype), mask=mask)


def _halves(gate_up_ref, block: int):
    """A packed program's row of `[rows, 2 * width]`, its block of columns, their mask, and the
    gate's and up's columns: contiguous runs, which the loads vectorize."""
    width = gate_up_ref.shape[1] // 2
    columns = pl.program_id(1) * block + jnp.arange(block)
    return pl.program_id(0), columns, columns < width, columns + width


def _packed_forward(gate_up_ref, y_ref, *, block: int, interpret: bool):
    row, columns, mask, up_columns = _halves(gate_up_ref, block)
    gate = plgpu.load(gate_up_ref.at[row, columns], mask=mask, other=0).astype(jnp.float32)
    up = plgpu.load(gate_up_ref.at[row, up_columns], mask=mask, other=0).astype(jnp.float32)
    y = _product(gate, up, gate_up_ref.dtype, interpret)
    plgpu.store(y_ref.at[row, columns], y.astype(y_ref.dtype), mask=mask)


def _packed_backward(gate_up_ref, dy_ref, d_ref, *, block: int, interpret: bool):
    row, columns, mask, up_columns = _halves(gate_up_ref, block)
    gate = plgpu.load(gate_up_ref.at[row, columns], mask=mask, other=0).astype(jnp.float32)
    up = plgpu.load(gate_up_ref.at[row, up_columns], mask=mask, other=0).astype(jnp.float32)
    dy = plgpu.load(dy_ref.at[row, columns], mask=mask, other=0).astype(jnp.float32)
    dg, du = _gradients(gate, up, dy)
    plgpu.store(d_ref.at[row, columns], dg.astype(d_ref.dtype), mask=mask)
    plgpu.store(d_ref.at[row, up_columns], du.astype(d_ref.dtype), mask=mask)


@dataclasses.dataclass(frozen=True)
class _Elementwise(tuning.KernelOp):
    """An elementwise launch over flat operands: `ELEMENT_BLOCK` elements and
    four warps a program."""

    config_cls: ClassVar[type[BlockConfig]] = BlockConfig

    def _get_heuristics_config(self, ba) -> BlockConfig:
        return BlockConfig(ELEMENT_BLOCK, 4)

    def variants(self, ba) -> tuple[BlockConfig, ...]:
        return _neighbours(ba.heuristics_config, block=(128, 4 * PACKED_BLOCK), warps=WARPS)


def _launch(kernel, name: str, operands, outputs, grid: tuple[int, ...], config: BlockConfig,
            interpret: bool):
    return pl.pallas_call(
        functools.partial(kernel, block=config.block, interpret=interpret), grid=grid, out_shape=outputs,
        compiler_params=plgpu.CompilerParams(num_warps=config.warps, num_stages=1),
        interpret=interpret, name=name,
    )(*operands), None


@dataclasses.dataclass(frozen=True)
class SwigluForward(_Elementwise):
    """SwiGLU's forward launch over flat `gate` and `up` (`_swiglu_forward`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('gate', 'up')

    def _fwd(self, gate: jax.Array, up: jax.Array, *, interpret: bool, config: BlockConfig,
             return_residuals: bool = False):
        return _launch(_swiglu_forward, "swiglu_forward", (gate, up),
                       jax.ShapeDtypeStruct(gate.shape, gate.dtype), (pl.cdiv(gate.size, config.block),),
                       config, interpret)


@dataclasses.dataclass(frozen=True)
class SwigluBackward(_Elementwise):
    """SwiGLU's reverse launch (`_swiglu_backward`): both inputs' gradients."""

    ROWS: ClassVar[tuple[str, ...]] = ('gate', 'up', 'dy')

    def _fwd(self, gate: jax.Array, up: jax.Array, dy: jax.Array, *, interpret: bool, config: BlockConfig,
             return_residuals: bool = False):
        outputs = (jax.ShapeDtypeStruct(gate.shape, gate.dtype), jax.ShapeDtypeStruct(up.shape, up.dtype))
        return _launch(_swiglu_backward, "swiglu_backward", (gate, up, dy), outputs,
                       (pl.cdiv(gate.size, config.block),), config, interpret)


@dataclasses.dataclass(frozen=True)
class _Packed(_Elementwise):
    """A packed projection's launch over `[rows, 2 * width]`, gate then up in each row: a program a
    row's block of columns, the whole row up to `PACKED_BLOCK` columns."""

    def _get_heuristics_config(self, ba) -> BlockConfig:
        return BlockConfig(min(PACKED_BLOCK, pl.next_power_of_2(ba.arguments['gate_up'].shape[1] // 2)), 4)


@dataclasses.dataclass(frozen=True)
class PackedSwigluForward(_Packed):
    """SwiGLU's forward launch over a packed projection's output, read in place (`_packed_forward`)."""

    ROWS: ClassVar[tuple[str, ...]] = ('gate_up',)

    def _fwd(self, gate_up: jax.Array, *, interpret: bool, config: BlockConfig,
             return_residuals: bool = False):
        rows, width = gate_up.shape[0], gate_up.shape[1] // 2
        return _launch(_packed_forward, "swiglu_forward", (gate_up,),
                       jax.ShapeDtypeStruct((rows, width), gate_up.dtype),
                       (rows, pl.cdiv(width, config.block)), config, interpret)


@dataclasses.dataclass(frozen=True)
class PackedSwigluBackward(_Packed):
    """The packed SwiGLU's reverse launch (`_packed_backward`): the gradient in its input's layout."""

    ROWS: ClassVar[tuple[str, ...]] = ('gate_up', 'dy')

    def _fwd(self, gate_up: jax.Array, dy: jax.Array, *, interpret: bool, config: BlockConfig,
             return_residuals: bool = False):
        return _launch(_packed_backward, "swiglu_backward", (gate_up, dy),
                       jax.ShapeDtypeStruct(gate_up.shape, gate_up.dtype),
                       (gate_up.shape[0], pl.cdiv(dy.shape[1], config.block)), config, interpret)


_SWIGLU_FORWARD = SwigluForward()
_SWIGLU_BACKWARD = SwigluBackward()
_PACKED_FORWARD = PackedSwigluForward()
_PACKED_BACKWARD = PackedSwigluBackward()


def _swiglu(gate, up, interpret: bool):
    return _SWIGLU_FORWARD(gate.reshape(-1), up.reshape(-1), interpret=interpret).reshape(gate.shape)


def _swiglu_bwd(interpret: bool, residual, dy):
    gate, up = residual
    dg, du = _SWIGLU_BACKWARD(gate.reshape(-1), up.reshape(-1), dy.reshape(-1), interpret=interpret)
    return dg.reshape(gate.shape), du.reshape(up.shape)


def _rows(gate_up):
    """A packed `[..., 2, H]` output as the launches' `[rows, 2 * H]`."""
    return gate_up.reshape(-1, 2 * gate_up.shape[-1])


def _packed(gate_up, interpret: bool):
    y = _PACKED_FORWARD(_rows(gate_up), interpret=interpret)
    return y.reshape(gate_up.shape[:-2] + gate_up.shape[-1:])


def _packed_bwd(interpret: bool, gate_up, dy):
    return _PACKED_BACKWARD(_rows(gate_up), dy.reshape(-1, dy.shape[-1]), interpret=interpret).reshape(
        gate_up.shape),


def swiglu(gate, up, portable, *, axes=None):
    """The fused SiLU product, `portable(gate, up)` its arithmetic in XLA, interpreted or lowered,
    placed on a mesh and differentiated as `stream_rms_norm` is."""
    interpret = not generation.triton_runs()
    local = generation.linear_tangent(lambda gate, up: (_swiglu(gate, up, interpret), ()),
                                      lambda saved, operands, dy: _swiglu_bwd(interpret, operands, dy),
                                      portable)
    return _per_shard(local, axes, (gate, up), (False, False))


def packed_swiglu(gate_up, portable, *, axes=None):
    """`swiglu` of one packed projection's `[..., 2, H]` output, gate then up, read in place, its
    gradient written in the same layout. `axes` are the product's own logical axes, `[..., H]`'s;
    the pair axis is never split; `portable(gate_up)` is its arithmetic in XLA, which differentiates it
    as `stream_rms_norm`'s does."""
    interpret = not generation.triton_runs()
    local = generation.linear_tangent(lambda gate_up: (_packed(gate_up, interpret), ()),
                                      lambda saved, operands, dy: _packed_bwd(interpret, operands[0], dy),
                                      portable)

    if not _split():
        return local(gate_up)
    if axes is None:
        raise ValueError("a mesh splits the packed SwiGLU's operands, and it places them by their logical "
                         "axes, which the call does not name")
    shape = gate_up.shape[:-2] + gate_up.shape[-1:]
    spec = logical_spec(axes, shape)
    entries = (*spec, *(None,) * (len(shape) - len(spec)))
    return manual_map(local, (jax.sharding.PartitionSpec(*entries[:-1], None, entries[-1]),), spec)(gate_up)
