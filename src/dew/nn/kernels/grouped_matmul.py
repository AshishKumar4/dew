"""The MoE grouped matmul on JAX's Pallas kernels, `gmm` and `tgmm`.

`ragged_dot.py` beside this file holds the kernels, vendored from the jax
source tree. This module owns what Dew adds around them: the predicate that
says which products they compute exactly (`ragged_dot_runs`) and the custom
VJP that trains through them (`grouped_projection`), in MaxText's megablox
shape: the forward is `gmm`, the input gradient `gmm` against the transposed
kernel, the kernel gradient `tgmm`.

The kernels are Triton kernels, compiled only for a CUDA lowering. The
choice is made where the call lowers, with `jax.lax.platform_dependent`: a
computation placed on the CPU of a GPU host lowers to `jax.lax.ragged_dot`
even though the process's default backend is the GPU. An explicit request
for the kernels interprets them on the CPU, which is how the CPU suite
checks them.

jax 0.11.2 deprecates the Pallas Triton backend and warns at every
lowering. These kernels stay on compute capability 8.0 to 8.9 on purpose:
JAX's Mosaic GPU grouped matmul (`pallas/ops/gpu/ragged_dot_mgpu.py`) uses
wgmma, which sm_80 and sm_89 do not have, and tokamax's sm80 Mosaic config
exceeds an Ada card's shared memory and has no backward. Where they run is
`dew.nn.kernels.generation.triton_runs`. jax 0.11.2 warns at each lowering
that the Triton backend is deprecated; Dew leaves that warning to the
user's filters (docs/performance.md records the deprecation).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype, PrecisionLike

from ..precision import asks_default_precision


def ragged_dot_runs(compute: Dtype, operands: tuple[Dtype, ...],
                    precision: PrecisionLike) -> bool:
    """Whether the kernels compute the product a caller asked for.

    They multiply in `compute`, accumulate in fp32 and ignore `precision`.
    With 16-bit compute that is exact products summed in fp32, which any
    precision asks for; with fp32 it is TF32, which only the default
    precision asks for (explicitly or through `jax_default_matmul_precision`).
    An operand or master wider than fp32 needs its gradient summed wider than
    the kernels do, and x64 widens their int32 group offsets, so both are
    refused.
    """
    if jax.config.jax_enable_x64:
        return False
    if jnp.result_type(compute, *operands, jnp.float32) != jnp.dtype(jnp.float32):
        return False
    compute = jnp.dtype(compute)
    if compute in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        return True
    return compute == jnp.dtype(jnp.float32) and asks_default_precision(
        precision, configured=True)


def _gmm(tokens, kernel, sizes, out_dtype, *, trans_rhs: bool, interpret: bool):
    # Imported at first use: the kernels need jax's Pallas Triton backend,
    # which jax 0.11.2 deprecates, and importing a model must not load it.
    from . import ragged_dot
    compute = tokens.dtype
    return ragged_dot.gmm(tokens, kernel.astype(compute), sizes,
                          **ragged_dot._hyperparam_selection_rule(np.dtype(compute)), trans_rhs=trans_rhs,
                          interpret=interpret, compute_dtype=compute, out_dtype=out_dtype)


def _tgmm(tokens, cotangent, sizes, out_dtype, *, interpret: bool):
    from . import ragged_dot
    compute = tokens.dtype
    return ragged_dot.tgmm(tokens, cotangent.astype(compute), sizes,
                           **ragged_dot._hyperparam_selection_rule(np.dtype(compute)), interpret=interpret,
                           compute_dtype=compute, out_dtype=out_dtype)


def _on_platform(kernel, fallback, interpret_on_cpu: bool):
    """`kernel` where the call lowers for CUDA, interpreted on the CPU when
    asked for, and `fallback` everywhere else."""
    branches = {'cuda': functools.partial(kernel, interpret=False)}
    if interpret_on_cpu:
        branches['cpu'] = functools.partial(kernel, interpret=True)
    return branches, fallback


def _forward(inputs, matrix, sizes, out_dtype, interpret_on_cpu):
    def kernel(inputs, matrix, sizes, *, interpret):
        return _gmm(inputs, matrix, sizes, out_dtype, trans_rhs=False, interpret=interpret)

    def fallback(inputs, matrix, sizes):
        return jax.lax.ragged_dot(inputs, matrix, sizes, preferred_element_type=out_dtype)

    branches, default = _on_platform(kernel, fallback, interpret_on_cpu)
    return jax.lax.platform_dependent(inputs, matrix, sizes, default=default, **branches)


def _backward(inputs, matrix, sizes, cotangent, work, interpret_on_cpu):
    def kernel(inputs, matrix, sizes, cotangent, *, interpret):
        return (_gmm(cotangent, matrix, sizes, work, trans_rhs=True, interpret=interpret),
                _tgmm(inputs, cotangent, sizes, work, interpret=interpret))

    def fallback(inputs, matrix, sizes, cotangent):
        d_inputs = jax.lax.ragged_dot(cotangent, jnp.swapaxes(matrix, 1, 2), sizes,
                                      preferred_element_type=work)
        _, pullback = jax.vjp(lambda matrix: jax.lax.ragged_dot(
            inputs, matrix, sizes, preferred_element_type=work), matrix.astype(work))
        return d_inputs, pullback(cotangent.astype(work))[0]

    branches, default = _on_platform(kernel, fallback, interpret_on_cpu)
    return jax.lax.platform_dependent(inputs, matrix, sizes, cotangent, default=default,
                                      **branches)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def grouped_projection(x: jax.Array, kernel: jax.Array, group_sizes: jax.Array,
                       compute: Dtype, interpret_on_cpu: bool) -> jax.Array:
    """`dew.nn.moe.expert_projection` on the kernels, first-order reverse mode.

    The operands are cast to `compute`, the product accumulates in fp32 and
    rounds once to `compute`. The input gradient takes `x`'s dtype and the
    kernel gradient the kernel's, each summed in fp32 from the compute-dtype
    cotangent and rounded operands: with 16-bit compute, exact products.
    Only a CUDA lowering runs the kernels (`interpret_on_cpu` adds the CPU's
    interpreter); every other lowering runs `jax.lax.ragged_dot` under the
    same contract.
    """
    return _projection_fwd(x, kernel, group_sizes, compute, interpret_on_cpu)[0]


def _projection_fwd(x, kernel, group_sizes, compute, interpret_on_cpu):
    inputs, matrix = x.astype(compute), kernel.astype(compute)
    sizes = group_sizes.astype(jnp.int32)
    output = _forward(inputs, matrix, sizes, jnp.promote_types(compute, jnp.float32),
                      interpret_on_cpu)
    # The residuals are the rounded operands, the values the forward
    # multiplied; the dtypes of the originals are what the gradients take.
    residuals = (inputs, matrix, sizes, jnp.zeros((0,), x.dtype), jnp.zeros((0,), kernel.dtype))
    return output.astype(compute), residuals


def _projection_bwd(compute, interpret_on_cpu, residuals, cotangent):
    del compute
    inputs, matrix, sizes, x_like, kernel_like = residuals
    work = jnp.result_type(inputs.dtype, x_like.dtype, kernel_like.dtype, jnp.float32)
    d_inputs, d_matrix = _backward(inputs, matrix, sizes, cotangent.astype(inputs.dtype), work,
                                   interpret_on_cpu)
    return (d_inputs.astype(x_like.dtype), d_matrix.astype(kernel_like.dtype),
            np.zeros(sizes.shape, jax.dtypes.float0))


grouped_projection.defvjp(_projection_fwd, _projection_bwd)
