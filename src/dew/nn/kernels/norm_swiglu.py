"""Row RMSNorm and elementwise SwiGLU, with hand-written reverse passes.

Qwen3.5-0.8B's decision step (A100-80, 8 x 3072 tokens, 936 ms) spent 49 ms in
norms, 44.6 ms in reduction fusions, and 31.7 ms in SwiGLU. Liger's torch
reference spent 12.5 ms in all reductions and 19.2 ms in SwiGLU. These
kernels follow linkedin/Liger-Kernel's ops/rms_norm.py and ops/swiglu.py:
save only a row's inverse RMS, accumulate weight gradients per program,
and recompute sigmoid from the two narrow SwiGLU inputs in reverse.

Dew's portable rounded_to also rounds cotangents. These fused VJPs instead
use Liger's numerical contract: fp32 intermediate gradients and one cast at
each gradient output. Forwards retain the model's cast sites, including
cast-before-scale RMSNorm and SiLU rounded before the up product. RMSNorm's
weight gradient reads that forward's rounded activation; SwiGLU's backward
recomputes SiLU in fp32, as Liger does. At c53's [257,127] bf16 shape, Liger's
norm dx was 1.26e-3 RMS from float64 against the rounded VJP's 1.78e-3;
SwiGLU's gate gradient was 1.00e-3 against 1.45e-3 (A100). No generation
selects these kernels until c53 measures their physical GPU kernel times.

The fp32 activation/input-gradient tests hold relative differences below
1e-5. Weight gradients keep Liger's rounded forward coefficient. Their
whole-function RMS error uses the project's unchanged 2x-XLA float64
budget. Accumulation is separately judged by the exact float64 sum of
the identical rounded factors, within XLA's sum error plus one final
rounding, and at 24,576 rows within 2 ulp outright. The
A100 verify found a 24,576-row fp32 dscale at 4.91e-3 RMS from float64,
against XLA's 1.22e-3. Both accumulation levels now retain separate fp32
Neumaier sums/corrections, instead of a left fold followed by plain sum.
The bf16 tests compare outputs and every gradient against float64, not
bitwise against XLA's differently rounded autodiff. Forward midpoint flips
can also differ by one ulp because reductions group differently. c53 keeps
both distances and maximum-relative differences in its measurement report.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jax.interpreters.ad import JVPTracer

from ..precision import rounded_to
from ..sharding import seen_whole
from . import generation
from .generation import first_refusal

IMPLEMENTATIONS = ('auto', 'xla', 'pallas')
DTYPES = (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16))
ROW_ELEMENTS = 4096
MAX_PROGRAMS = 256
ELEMENT_BLOCK = 1024
SCALE_BLOCK = 128


def _row_block(width: int) -> int:
    """At most sixteen rows, with the live register tile bounded at 4096
    elements except when a single row is wider. A 4096-wide norm takes one
    row rather than spilling four copies of every backward intermediate."""
    return min(16, max(1, ROW_ELEMENTS // pl.next_power_of_2(width)))


def _round(x, dtype, interpret: bool):
    """Triton's explicit narrowing casts survive widening, unlike XLA's
    excess-precision casts. Interpretation uses Dew's explicit rounding."""
    return rounded_to(x, dtype) if interpret else x.astype(dtype).astype(x.dtype)


def _compensated_add(total, correction, value, *, interpret: bool) -> tuple[jax.Array, jax.Array]:
    """Neumaier's fp32 sum: keep the lost low bits even under cancellation.

    Explicit add/sub rounding protects the residual from compiler fusion
    and cancellation rewrites. CPU interpretation uses optimizer barriers;
    Triton uses round-to-nearest instructions, not a compiler-only option.
    """
    if not interpret:
        high, low = plgpu.elementwise_inline_asm(
            """{
            .reg .f32 a, b, e, f, s, c;
            .reg .pred p;
            add.rn.f32 s, $2, $4;
            abs.f32 a, $2;
            abs.f32 b, $4;
            setp.ge.f32 p, a, b;
            sub.rn.f32 e, $2, s;
            add.rn.f32 e, e, $4;
            sub.rn.f32 f, $4, s;
            add.rn.f32 f, f, $2;
            selp.f32 e, e, f, p;
            add.rn.f32 c, $3, e;
            mov.f32 $0, s;
            mov.f32 $1, c;
            }""", constraints="=f,=f,f,f,f", args=(total, correction, value),
            result_shape_dtypes=(jax.ShapeDtypeStruct(total.shape, jnp.float32),
                                 jax.ShapeDtypeStruct(total.shape, jnp.float32)), pack=1)
        return high, low
    value = jax.lax.optimization_barrier(value)
    merged = jax.lax.optimization_barrier(total + value)
    lost = jnp.where(jnp.abs(total) >= jnp.abs(value),
                     (total - merged) + value, (value - merged) + total)
    return merged, correction + lost


def _product_parts(left, right, *, interpret: bool) -> tuple[jax.Array, jax.Array]:
    """The rounded fp32 product and its lost bits, without a float64 dtype."""
    if not interpret:
        high, low = plgpu.elementwise_inline_asm(
            """{
            .reg .f32 p, e;
            mul.rn.f32 p, $2, $3;
            neg.f32 e, p;
            fma.rn.f32 e, $2, $3, e;
            mov.f32 $0, p;
            mov.f32 $1, e;
            }""", constraints="=f,=f,f,f", args=(left, right),
            result_shape_dtypes=(jax.ShapeDtypeStruct(left.shape, jnp.float32),
                                 jax.ShapeDtypeStruct(left.shape, jnp.float32)), pack=1)
        return high, low
    barrier = jax.lax.optimization_barrier
    # Split significands in half by bits, avoiding splitter*value overflow.
    def split(value):
        bits = jax.lax.bitcast_convert_type(value, jnp.uint32) & jnp.uint32(0xfffff000)
        high = jax.lax.bitcast_convert_type(bits, jnp.float32)
        return high, barrier(value - high)
    lh, ll = split(left)
    rh, rl = split(right)
    product = barrier(left * right)
    lost = barrier(barrier(lh * rh) - product)
    lost = barrier(lost + barrier(lh * rl))
    lost = barrier(lost + barrier(ll * rh))
    return product, barrier(lost + barrier(ll * rl))


def _selected(choice: str, implementation: str, refusal: str | None) -> str:
    from ..attention import FORWARD_MODE

    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")
    chosen = generation.measured_kernel(choice, 'xla') if implementation == 'auto' else implementation
    if chosen != 'pallas':
        return chosen
    placed = generation.triton_runs() or (implementation == 'pallas' and jax.default_backend() == 'cpu')
    reason = first_refusal(
        (placed, "the Pallas kernels need a GPU of sm80 or later"),
        (not FORWARD_MODE.get(), "forward_mode_attention needs a JVP; these kernels implement VJPs"),
    ) or refusal
    return 'pallas' if reason is None else generation.ran_kernel(choice, 'xla', reason)


def rms_kernel(implementation: str, x, scale, dtype, fp32_statistics: bool, *,
                inverse_after_cast: bool = False) -> str:
    """Resolve RMSNorm without changing its arithmetic or collecting a split
    feature axis. Float64 and input-dtype statistics belong to XLA."""
    return _selected('rms_norm', implementation, first_refusal(
        (not any(isinstance(operand, JVPTracer) for operand in (x, scale)),
         "a JVP trace needs forward-mode derivatives; the RMSNorm kernel implements a VJP"),
        (not inverse_after_cast, "unweighted_rmsnorm rounds the inverse, not the normalized activation"),
        (fp32_statistics, "input-dtype statistics round each reduction step; the kernel reduces in fp32"),
        (x.dtype in DTYPES and jnp.dtype(dtype) in DTYPES,
         f"RMSNorm input {x.dtype} and output {dtype}; the kernel supports float32 and bfloat16"),
        (bool(x.shape) and x.size > 0, "RMSNorm needs nonempty rows with a feature axis"),
        (bool(x.shape) and 0 < x.shape[-1] <= 8192, "RMSNorm's register tile holds at most 8192 features"),
        (scale is None or (scale.shape == x.shape[-1:] and scale.dtype in DTYPES),
         "the kernel takes a float32 or bfloat16 vector scale, not a broadcast affine tensor"),
        (seen_whole(x) and (scale is None or seen_whole(scale)),
         "a mesh axis splits RMSNorm's operands; the kernel sees local whole rows"),
    ))


def swiglu_kernel(implementation: str, gate, up) -> str:
    """Resolve the SiLU product only. GeGLU and SiTU keep their own XLA
    arithmetic; broadcasting or mixed operand dtypes also stay there."""
    return _selected('swiglu', implementation, first_refusal(
        (not any(isinstance(operand, JVPTracer) for operand in (gate, up)),
         "a JVP trace needs forward-mode derivatives; the SwiGLU kernel implements a VJP"),
        (gate.dtype in DTYPES and gate.dtype == up.dtype,
         "SwiGLU needs matching float32 or bfloat16 operands"),
        (gate.shape == up.shape and bool(gate.shape) and gate.size > 0,
         "SwiGLU needs nonempty equal shapes, not broadcasting"),
        (seen_whole(gate) and seen_whole(up),
         "a mesh axis splits SwiGLU's operands; the kernel sees local whole arrays"),
    ))


def _weight(scale_ref, columns, width: int, offset: bool, interpret: bool):
    scale = plgpu.load(scale_ref.at[columns], mask=columns < width, other=0)
    # The offset adds in the stored weight's dtype, including a bf16 weight.
    weight = scale.astype(jnp.float32)
    return _round(weight + 1.0, scale.dtype, interpret) if offset else weight


def _rms_forward(x_ref, scale_ref, y_ref, inverse_ref, *, epsilon: float,
                 offset: bool, after_cast: bool, interpret: bool):
    rows, width = x_ref.shape
    columns = jnp.arange(pl.next_power_of_2(width))
    row_block = _row_block(width)
    row = pl.program_id(0) * row_block + jnp.arange(row_block)
    mask = (row[:, None] < rows) & (columns[None, :] < width)
    x = plgpu.load(x_ref.at[row[:, None], columns[None, :]], mask=mask, other=0).astype(jnp.float32)
    # XLA's mean multiplies by the fp32 reciprocal, not an fp32 division.
    inverse = jax.lax.rsqrt(jnp.sum(x * x, axis=1) * (1.0 / width) + epsilon)
    y = x * inverse[:, None]
    if scale_ref is not None:
        if after_cast:
            y = _round(y, y_ref.dtype, interpret)
        y = y * _weight(scale_ref, columns, width, offset, interpret)[None, :]
    plgpu.store(y_ref.at[row[:, None], columns[None, :]], y.astype(y_ref.dtype), mask=mask)
    plgpu.store(inverse_ref.at[row], inverse, mask=row < rows)


def _rms_backward(x_ref, scale_ref, inverse_ref, dy_ref, dx_ref, partial_ref, correction_ref, *,
                  offset: bool, after_cast: bool, interpret: bool):
    rows, width = x_ref.shape
    columns = jnp.arange(pl.next_power_of_2(width))
    groups = pl.num_programs(0)
    group = pl.program_id(0)
    row_block = _row_block(width)
    weight = _weight(scale_ref, columns, width, offset, interpret) if scale_ref is not None else 1.0

    def block(i, accumulated):
        row = (i * groups + group) * row_block + jnp.arange(row_block)
        mask = (row[:, None] < rows) & (columns[None, :] < width)
        x = plgpu.load(x_ref.at[row[:, None], columns[None, :]], mask=mask, other=0).astype(jnp.float32)
        dy = plgpu.load(dy_ref.at[row[:, None], columns[None, :]], mask=mask, other=0).astype(jnp.float32)
        inverse = plgpu.load(inverse_ref.at[row], mask=row < rows, other=0)
        m = dy * weight
        # The two x contributions are kept separate, as XLA's square VJP is.
        correction = (-0.5 * jnp.sum(m * x, axis=1)) * (inverse * inverse * inverse) * (1.0 / width)
        dx = m * inverse[:, None] + (2.0 * x) * correction[:, None]
        plgpu.store(dx_ref.at[row[:, None], columns[None, :]], dx.astype(dx_ref.dtype), mask=mask)
        if scale_ref is not None:
            # Reduce every term accurately, not a rounded row-block sum
            # whose error compensation at the outer level cannot recover.
            for lane in range(row_block):
                at = (i * groups + group) * row_block + lane
                valid = (at < rows) & (columns < width)
                value = plgpu.load(x_ref.at[at, columns], mask=valid, other=0).astype(jnp.float32)
                cotangent = plgpu.load(dy_ref.at[at, columns], mask=valid, other=0).astype(jnp.float32)
                rstd = plgpu.load(inverse_ref.at[at], mask=at < rows, other=0)
                normalized = value * rstd
                if after_cast:
                    normalized = _round(normalized, dy_ref.dtype, interpret)
                product, low = _product_parts(cotangent, normalized, interpret=interpret)
                accumulated = _compensated_add(*accumulated, product, interpret=interpret)
                accumulated = _compensated_add(*accumulated, low, interpret=interpret)
        return accumulated

    partial, correction = jax.lax.fori_loop(0, pl.cdiv(rows, groups * row_block), block,
                                           (jnp.zeros(columns.shape, jnp.float32),
                                            jnp.zeros(columns.shape, jnp.float32)))
    if partial_ref is not None:
        plgpu.store(partial_ref.at[group, columns], partial, mask=columns < width)
        plgpu.store(correction_ref.at[group, columns], correction, mask=columns < width)


def _rms_scale_reduce(partial_ref, correction_ref, scale_ref, *, interpret: bool):
    """Combine both fp32 parts before the only final cast to the scale dtype."""
    groups, width = partial_ref.shape
    columns = pl.program_id(0) * SCALE_BLOCK + jnp.arange(SCALE_BLOCK)
    def group(i, accumulated):
        high = plgpu.load(partial_ref.at[i, columns], mask=columns < width, other=0)
        low = plgpu.load(correction_ref.at[i, columns], mask=columns < width, other=0)
        added = _compensated_add(*accumulated, high, interpret=interpret)
        return _compensated_add(*added, low, interpret=interpret)
    total, correction = jax.lax.fori_loop(0, groups, group,
                                         (jnp.zeros(columns.shape, jnp.float32),
                                          jnp.zeros(columns.shape, jnp.float32)))
    plgpu.store(scale_ref.at[columns], (total + correction).astype(scale_ref.dtype), mask=columns < width)


def _rms_with_inverse(x, scale, epsilon: float, dtype, offset: bool, after_cast: bool,
                      interpret: bool):
    rows, width = math.prod(x.shape[:-1]), x.shape[-1]
    y, inverse = pl.pallas_call(
        functools.partial(_rms_forward, epsilon=epsilon, offset=offset, after_cast=after_cast,
                          interpret=interpret),
        grid=(pl.cdiv(rows, _row_block(width)),),
        out_shape=(jax.ShapeDtypeStruct((rows, width), dtype),
                   jax.ShapeDtypeStruct((rows,), jnp.float32)),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret, name="rms_norm_forward",
    )(x.reshape(rows, width), scale)
    return y.reshape(x.shape), inverse


def _rms(x, scale, epsilon: float, dtype, offset: bool, after_cast: bool, interpret: bool):
    return _rms_with_inverse(x, scale, epsilon, dtype, offset, after_cast, interpret)[0]


def _rms_fwd(x, scale, epsilon: float, dtype, offset: bool, after_cast: bool, interpret: bool):
    y, inverse = _rms_with_inverse(x, scale, epsilon, dtype, offset, after_cast, interpret)
    return y, (x, scale, inverse)


def _rms_bwd(epsilon: float, dtype, offset: bool, after_cast: bool, interpret: bool, residual, dy):
    del epsilon, dtype
    x, scale, inverse = residual
    rows, width = math.prod(x.shape[:-1]), x.shape[-1]
    groups = min(MAX_PROGRAMS, pl.cdiv(rows, _row_block(width)))
    partial_shape = None if scale is None else jax.ShapeDtypeStruct((groups, width), jnp.float32)
    dx, partial, correction = pl.pallas_call(
        functools.partial(_rms_backward, offset=offset, after_cast=after_cast, interpret=interpret),
        grid=(groups,),
        out_shape=(jax.ShapeDtypeStruct((rows, width), x.dtype), partial_shape, partial_shape),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret, name="rms_norm_backward",
    )(x.reshape(rows, width), scale, inverse, dy.reshape(rows, width))
    dscale = None if scale is None else pl.pallas_call(
        functools.partial(_rms_scale_reduce, interpret=interpret), grid=(pl.cdiv(width, SCALE_BLOCK),),
        out_shape=jax.ShapeDtypeStruct((width,), scale.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret, name="rms_norm_scale_backward",
    )(partial, correction)
    return dx.reshape(x.shape), dscale


rms_norm = jax.custom_vjp(_rms, nondiff_argnums=(2, 3, 4, 5, 6))
rms_norm.defvjp(_rms_fwd, _rms_bwd)


def _swiglu_forward(gate_ref, up_ref, y_ref, *, interpret: bool):
    index = pl.program_id(0) * ELEMENT_BLOCK + jnp.arange(ELEMENT_BLOCK)
    mask = index < gate_ref.shape[0]
    gate = plgpu.load(gate_ref.at[index], mask=mask, other=0).astype(jnp.float32)
    up = plgpu.load(up_ref.at[index], mask=mask, other=0).astype(jnp.float32)
    activated = _round(gate * jax.nn.sigmoid(gate), gate_ref.dtype, interpret)
    plgpu.store(y_ref.at[index], (activated * up).astype(y_ref.dtype), mask=mask)


def _swiglu_backward(gate_ref, up_ref, dy_ref, dg_ref, du_ref, *, interpret: bool):
    index = pl.program_id(0) * ELEMENT_BLOCK + jnp.arange(ELEMENT_BLOCK)
    mask = index < gate_ref.shape[0]
    gate = plgpu.load(gate_ref.at[index], mask=mask, other=0).astype(jnp.float32)
    up = plgpu.load(up_ref.at[index], mask=mask, other=0).astype(jnp.float32)
    dy = plgpu.load(dy_ref.at[index], mask=mask, other=0).astype(jnp.float32)
    sigmoid = jax.nn.sigmoid(gate)
    activated = gate * sigmoid
    dg = dy * (activated * (1.0 - sigmoid) + sigmoid) * up
    plgpu.store(dg_ref.at[index], dg.astype(gate_ref.dtype), mask=mask)
    plgpu.store(du_ref.at[index], (dy * activated).astype(up_ref.dtype), mask=mask)


def _swiglu(gate, up, interpret: bool):
    y = pl.pallas_call(
        functools.partial(_swiglu_forward, interpret=interpret), grid=(pl.cdiv(gate.size, ELEMENT_BLOCK),),
        out_shape=jax.ShapeDtypeStruct((gate.size,), gate.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret, name="swiglu_forward",
    )(gate.reshape(-1), up.reshape(-1))
    return y.reshape(gate.shape)


def _swiglu_fwd(gate, up, interpret: bool):
    return _swiglu(gate, up, interpret), (gate, up)


def _swiglu_bwd(interpret: bool, residual, dy):
    gate, up = residual
    dg, du = pl.pallas_call(
        functools.partial(_swiglu_backward, interpret=interpret), grid=(pl.cdiv(gate.size, ELEMENT_BLOCK),),
        out_shape=(jax.ShapeDtypeStruct((gate.size,), gate.dtype),
                   jax.ShapeDtypeStruct((up.size,), up.dtype)),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret, name="swiglu_backward",
    )(gate.reshape(-1), up.reshape(-1), dy.reshape(-1))
    return dg.reshape(gate.shape), du.reshape(up.shape)


swiglu = jax.custom_vjp(_swiglu, nondiff_argnums=(2,))
swiglu.defvjp(_swiglu_fwd, _swiglu_bwd)
