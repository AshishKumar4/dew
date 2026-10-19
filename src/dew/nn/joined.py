"""Projections of one input run as one GEMM, their kernels joined per call.

A layer's projections that read one input stay separate parameters, as its
checkpoints store them. Where a generation's kernel row measured the join
faster (`KERNELS`, 'joined'), a call stacks their kernels and runs one
product: one wide GEMM fills the device where narrower ones over the same
rows do not, and reads the input once. Stacking copies the kernels once a
call, small beside the activations, and their gradients unstack the same way.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import meta

from dew.nn.dense import Dense
from dew.nn.precision import biased, fp32_result_dot_general

from ..registry import wrapper_layers
from .kernels.generation import first_refusal, measured_kernel, ran_kernel


def _intercepted() -> bool:
    """Whether the running call is a wrapped model's (`dew.registry.WrapperClass`: an adapter, an
    input quantization, Dew's Qwix training) or a Qwix-quantized one's, which intercept their
    layers' own calls. It reads the outermost module whose method is running: Flax keeps them on
    a stack, which its lifted transforms (`nn.scan`) leave whole where a scanned layer's parents
    stop at the scan, and which no public call reads; Qwix reads it the same way
    (qwix/_src/model.py, `_is_in_nn_module`)."""
    from flax.linen.module import _context

    root = type(next(module for module in _context.module_stack if module is not None))
    return bool(wrapper_layers(root)[1]) or any("_unquantized_type" in vars(layer) for layer in root.__mro__)


def join_refusal(projections: Sequence[nn.Module]) -> str | None:
    """Why one GEMM cannot stand in for `projections`' own calls, or None."""
    return first_refusal(
        (all(type(projection) is Dense for projection in projections),
         "a projection is not a plain Dense, whose call the join reproduces"),
        (all(projection.dot_general is None and projection.dot_general_cls is None
             for projection in projections), "a projection takes its own dot_general"),
        (len({(projection.features, projection.use_bias, projection.dtype, projection.precision,
               projection.promote_dtype) for projection in projections}) == 1,
         "the projections differ in width, bias, dtype, precision or promotion"),
        (not _intercepted(),
         "the model computes under a wrapper (an adapter, a quantization) that intercepts each "
         "projection's own call"),
        (all(projection.has_variable('params', 'kernel') for projection in projections),
         "a projection holds no kernel of its own"),
    )


def joins(choice: str, projections: Sequence[nn.Module]) -> bool:
    """Whether `projections` run as one GEMM (`joined_dense`): where the generation's row measured
    `choice` 'joined', unless their calls take what the join does not reproduce (`join_refusal`),
    which `ran_kernel` logs. Initialization creates each projection's parameters through its own
    call."""
    if measured_kernel(choice, 'apart') != 'joined' or projections[0].is_initializing():
        return False
    reason = join_refusal(projections)
    if reason is not None:
        ran_kernel(choice, 'apart', reason)
    return reason is None


def joined_dense(x: jax.Array, projections: Sequence[nn.Module], *, axis: int) -> jax.Array:
    """`projections`' outputs of `x` from one GEMM, stacked on output axis `axis`: -2 gives
    `[..., n, features]`, -1 `[..., features, n]`. Each kernel and bias is read and cast as its
    own call casts it; `joins` says when the result is theirs."""
    if axis not in (-2, -1):
        raise ValueError(f"the projections stack on output axis -2 or -1, not {axis}")
    promoted = [projection.promote_dtype(
        x, meta.unbox(projection.get_variable('params', 'kernel')),
        meta.unbox(projection.get_variable('params', 'bias')) if projection.use_bias else None,
        dtype=projection.dtype) for projection in projections]
    inputs = promoted[0][0]
    # A kernel is [inputs, features]; the stacked axis sits beside its features as it will in the output.
    kernel = jnp.stack([kernel for _, kernel, _ in promoted], axis=axis + 3)
    biases = (jnp.stack([bias for _, _, bias in promoted], axis=axis + 2) if projections[0].use_bias
              else None)
    dimensions = (((inputs.ndim - 1,), (0,)), ((), ()))
    precision = projections[0].precision
    if biases is None:
        return jax.lax.dot_general(inputs, kernel, dimensions, precision=precision)
    # Taken and biased as each Dense takes and biases its own (`dew.nn.dense`).
    return biased(fp32_result_dot_general(precision)(inputs, kernel, dimensions), biases, inputs.dtype)
