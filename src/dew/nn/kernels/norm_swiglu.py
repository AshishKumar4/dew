"""Row RMSNorm and elementwise SwiGLU, with hand-written reverse passes.

Qwen3.5-0.8B's decision step (A100-80, 8 x 3072 tokens, 936 ms) spent 49 ms in
norms, 44.6 ms in reduction fusions, and 31.7 ms in SwiGLU. Liger's torch
reference spent 12.5 ms in all reductions and 19.2 ms in SwiGLU. These
kernels follow linkedin/Liger-Kernel's ops/rms_norm.py and ops/swiglu.py:
save only a row's inverse RMS, accumulate weight gradients per program,
and recompute sigmoid from the two narrow SwiGLU inputs in reverse.

Dew's rounded_to also rounds cotangents, unlike Liger's straight-through
casts. Both passes reproduce those roundings; in particular SwiGLU's up
gradient reads the rounded activation, not the unrounded SiLU. No hardware
generation selects these kernels until c53 measures the complete VJPs.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from ..precision import rounded_to
from ..sharding import seen_whole
from . import generation
from .generation import first_refusal

IMPLEMENTATIONS = ('auto', 'xla', 'pallas')
DTYPES = (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16))
ROW_BLOCK = 4
MAX_PROGRAMS = 128
ELEMENT_BLOCK = 1024


def _round(x, dtype, interpret: bool):
    """Triton's explicit narrowing casts survive widening, unlike XLA's
    excess-precision casts. Interpretation uses Dew's explicit rounding."""
    return rounded_to(x, dtype) if interpret else x.astype(dtype).astype(x.dtype)


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


def rms_kernel(implementation: str, x, scale, dtype, fp32_statistics: bool) -> str:
    """Resolve RMSNorm without changing its arithmetic or collecting a split
    feature axis. Float64 and input-dtype statistics belong to XLA."""
    return _selected('rms_norm', implementation, first_refusal(
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
        (gate.dtype in DTYPES and gate.dtype == up.dtype,
         "SwiGLU needs matching float32 or bfloat16 operands"),
        (gate.shape == up.shape and bool(gate.shape) and gate.size > 0,
         "SwiGLU needs nonempty equal shapes, not broadcasting"),
        (seen_whole(gate) and seen_whole(up),
         "a mesh axis splits SwiGLU's operands; the kernel sees local whole arrays"),
    ))


def _weight(scale_ref, columns, width: int, offset: bool):
    scale = plgpu.load(scale_ref, (columns,), mask=columns < width, other=0)
    # The offset adds in the stored weight's dtype, including a bf16 weight.
    return (scale + 1.0 if offset else scale).astype(jnp.float32)


def _rms_forward(x_ref, scale_ref, y_ref, inverse_ref, *, epsilon: float,
                 offset: bool, after_cast: bool, interpret: bool):
    rows, width = x_ref.shape
    columns = jnp.arange(pl.next_power_of_2(width))
    row = pl.program_id(0) * ROW_BLOCK + jnp.arange(ROW_BLOCK)
    mask = (row[:, None] < rows) & (columns[None, :] < width)
    x = plgpu.load(x_ref, (row[:, None], columns[None, :]), mask=mask, other=0).astype(jnp.float32)
    inverse = jax.lax.rsqrt(jnp.sum(x * x, axis=1) / width + epsilon)
    y = x * inverse[:, None]
    if scale_ref is not None:
        if after_cast:
            y = _round(y, y_ref.dtype, interpret)
        y = y * _weight(scale_ref, columns, width, offset)[None, :]
    plgpu.store(y_ref, (row[:, None], columns[None, :]), y.astype(y_ref.dtype), mask=mask)
    plgpu.store(inverse_ref, (row,), inverse, mask=row < rows)


def _rms_backward(x_ref, scale_ref, inverse_ref, dy_ref, dx_ref, partial_ref, *,
                  offset: bool, after_cast: bool, interpret: bool):
    rows, width = x_ref.shape
    columns = jnp.arange(pl.next_power_of_2(width))
    groups = pl.num_programs(0)
    weight = _weight(scale_ref, columns, width, offset) if scale_ref is not None else 1.0

    def block(i, accumulated):
        row = (i * groups + pl.program_id(0)) * ROW_BLOCK + jnp.arange(ROW_BLOCK)
        mask = (row[:, None] < rows) & (columns[None, :] < width)
        x = plgpu.load(x_ref, (row[:, None], columns[None, :]), mask=mask, other=0).astype(jnp.float32)
        dy = plgpu.load(dy_ref, (row[:, None], columns[None, :]), mask=mask, other=0).astype(jnp.float32)
        inverse = plgpu.load(inverse_ref, (row,), mask=row < rows, other=0)
        m = dy * weight
        if scale_ref is not None and after_cast:
            m = _round(m, dy_ref.dtype, interpret)
        # The two x contributions are kept separate, as XLA's square VJP is.
        correction = (-0.5 * jnp.sum(m * x, axis=1)) * (inverse * inverse * inverse) / width
        dx = m * inverse[:, None] + (2.0 * x) * correction[:, None]
        plgpu.store(dx_ref, (row[:, None], columns[None, :]), dx.astype(dx_ref.dtype), mask=mask)
        if scale_ref is not None:
            normalized = x * inverse[:, None]
            if after_cast:
                normalized = _round(normalized, dy_ref.dtype, interpret)
            accumulated = accumulated + jnp.sum(dy * normalized, axis=0)
        return accumulated

    partial = jax.lax.fori_loop(0, pl.cdiv(rows, groups * ROW_BLOCK), block,
                                jnp.zeros(columns.shape, jnp.float32))
    if partial_ref is not None:
        plgpu.store(partial_ref, (pl.program_id(0), columns), partial, mask=columns < width)


def _rms_with_inverse(x, scale, epsilon: float, dtype, offset: bool, after_cast: bool,
                      interpret: bool):
    rows, width = math.prod(x.shape[:-1]), x.shape[-1]
    y, inverse = pl.pallas_call(
        functools.partial(_rms_forward, epsilon=epsilon, offset=offset, after_cast=after_cast,
                          interpret=interpret),
        grid=(pl.cdiv(rows, ROW_BLOCK),),
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
    groups = min(MAX_PROGRAMS, pl.cdiv(rows, ROW_BLOCK))
    dx, partial = pl.pallas_call(
        functools.partial(_rms_backward, offset=offset, after_cast=after_cast, interpret=interpret),
        grid=(groups,),
        out_shape=(jax.ShapeDtypeStruct((rows, width), x.dtype),
                   None if scale is None else jax.ShapeDtypeStruct((groups, width), jnp.float32)),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret, name="rms_norm_backward",
    )(x.reshape(rows, width), scale, inverse, dy.reshape(rows, width))
    # Only the small per-program partials, never all rows, meet a column reduction.
    dscale = None if scale is None else jnp.sum(partial, axis=0).astype(scale.dtype)
    return dx.reshape(x.shape), dscale


rms_norm = jax.custom_vjp(_rms, nondiff_argnums=(2, 3, 4, 5, 6))
rms_norm.defvjp(_rms_fwd, _rms_bwd)


def _swiglu_forward(gate_ref, up_ref, y_ref, *, interpret: bool):
    index = pl.program_id(0) * ELEMENT_BLOCK + jnp.arange(ELEMENT_BLOCK)
    mask = index < gate_ref.shape[0]
    gate = plgpu.load(gate_ref, (index,), mask=mask, other=0).astype(jnp.float32)
    up = plgpu.load(up_ref, (index,), mask=mask, other=0).astype(jnp.float32)
    activated = _round(gate * jax.nn.sigmoid(gate), gate_ref.dtype, interpret)
    plgpu.store(y_ref, (index,), (activated * up).astype(y_ref.dtype), mask=mask)


def _swiglu_backward(gate_ref, up_ref, dy_ref, dg_ref, du_ref, *, interpret: bool):
    index = pl.program_id(0) * ELEMENT_BLOCK + jnp.arange(ELEMENT_BLOCK)
    mask = index < gate_ref.shape[0]
    gate = plgpu.load(gate_ref, (index,), mask=mask, other=0).astype(jnp.float32)
    up = plgpu.load(up_ref, (index,), mask=mask, other=0).astype(jnp.float32)
    dy = plgpu.load(dy_ref, (index,), mask=mask, other=0).astype(jnp.float32)
    sigmoid = jax.nn.sigmoid(gate)
    activated = _round(gate * sigmoid, gate_ref.dtype, interpret)
    dactivated = _round(dy * up, gate_ref.dtype, interpret)
    dg = dactivated * sigmoid + (dactivated * gate) * (sigmoid * (1.0 - sigmoid))
    plgpu.store(dg_ref, (index,), dg.astype(gate_ref.dtype), mask=mask)
    plgpu.store(du_ref, (index,), (dy * activated).astype(up_ref.dtype), mask=mask)


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
