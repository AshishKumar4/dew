"""Projections of one input run as one GEMM, their kernels joined per call.

A layer's projections that read one input stay separate parameters, as its
checkpoints store them. Where a generation's kernel row measured the join
faster (`KERNELS`, 'joined'), a call joins their kernels and runs one
product: one wide GEMM fills the device where narrower ones over the same
rows do not, and reads the input once. Joining copies the kernels once a
call, small beside the activations, and their gradients split the same way.
"""

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import meta

from ..registry import wrapper_layers
from .kernels.generation import first_refusal, measured_kernel, ran_kernel
from .sharding import LogicalAxes, constrain, logical_spec, mesh_axes


def _root(module: nn.Module) -> nn.Module:
    while isinstance(module.parent, nn.Module):
        module = module.parent
    return module


def _kernel(projection: nn.Module) -> jax.Array:
    return meta.unbox(projection.get_variable('params', 'kernel'))


def _split(projection: nn.Module, axes: LogicalAxes) -> tuple[str, ...]:
    """The mesh axes that split `projection`'s output features, its kernel's logical `axes` placed."""
    kernel = _kernel(projection)
    spec = logical_spec(axes, kernel.shape)
    return mesh_axes(spec[1]) if len(spec) > 1 else ()


def join_refusal(projections: Sequence[nn.Module], kernel_axes: Sequence[LogicalAxes] | None = None
                 ) -> str | None:
    """Why one GEMM cannot stand in for `projections`' own calls, or None.
    `kernel_axes`, each kernel's logical axes, are a concatenation's
    (`joined_outputs`), whose features a mesh must split alike."""
    first = projections[0]
    return first_refusal(
        (all(type(projection) is nn.Dense for projection in projections),
         "a projection is not a plain nn.Dense, whose call the join reproduces"),
        (all(projection.dot_general is None and projection.dot_general_cls is None
             for projection in projections), "a projection takes its own dot_general"),
        (len({(projection.use_bias, projection.dtype, projection.precision, projection.promote_dtype)
              for projection in projections}) == 1,
         "the projections differ in bias, dtype, precision or promotion"),
        (not wrapper_layers(type(_root(first)))[1],
         "the model computes under a wrapper (an adapter, a quantization) that intercepts each "
         "projection's own call"),
        (kernel_axes is None or len({_split(projection, axes) for projection, axes
                                     in zip(projections, kernel_axes, strict=True)}) == 1,
         "a mesh splits the projections' features over different axes"),
    )


def joins(choice: str, projections: Sequence[nn.Module], kernel_axes: Sequence[LogicalAxes] | None = None
          ) -> bool:
    """Whether `projections` run as one GEMM (`joined_dense`, `joined_outputs`): where the
    generation's row measured `choice` 'joined', unless their calls take what the join does not
    reproduce (`join_refusal`), which `ran_kernel` logs. Initialization creates each projection's
    parameters through its own call."""
    if measured_kernel(choice, 'apart') != 'joined' or projections[0].is_initializing():
        return False
    reason = join_refusal(projections, kernel_axes)
    if reason is not None:
        ran_kernel(choice, 'apart', reason)
    return reason is None


def _promoted(x: jax.Array, projections: Sequence[nn.Module]):
    """Each projection's input, kernel and bias, cast as its own call casts them."""
    return [projection.promote_dtype(
        x, _kernel(projection), meta.unbox(projection.get_variable('params', 'bias')) if projection.use_bias
        else None, dtype=projection.dtype) for projection in projections]


def _product(inputs: jax.Array, kernel: jax.Array, projection: nn.Module) -> jax.Array:
    return jax.lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())),
                               precision=projection.precision)


def joined_dense(x: jax.Array, projections: Sequence[nn.Module], *, axis: int) -> jax.Array:
    """`projections`' outputs of `x` from one GEMM, stacked on output axis `axis`: -2 gives
    `[..., n, features]`, -1 `[..., features, n]`, so the projections are as wide. Each kernel and
    bias is read and cast as its own call casts it; `joins` says when the result is theirs."""
    if axis not in (-2, -1):
        raise ValueError(f"the projections stack on output axis -2 or -1, not {axis}")
    if len({projection.features for projection in projections}) != 1:
        raise ValueError("stacked projections are as wide; concatenate others (joined_outputs)")
    promoted = _promoted(x, projections)
    # A kernel is [inputs, features]; the stacked axis sits beside its features as it will in the output.
    kernel = jnp.stack([kernel for _, kernel, _ in promoted], axis=axis + 3)
    y = _product(promoted[0][0], kernel, projections[0])
    if not projections[0].use_bias:
        return y
    return y + jnp.stack([bias for _, _, bias in promoted], axis=axis + 2)


def joined_outputs(x: jax.Array, projections: Sequence[nn.Module], kernel_axes: Sequence[LogicalAxes],
                   output_axes: LogicalAxes) -> list[jax.Array]:
    """`projections`' outputs of `x`, each as its own call gives it, from one
    GEMM over their kernels joined on the features. Where a mesh splits the
    features (by `kernel_axes`, each kernel's logical axes, alike for all, as
    `joins` checks), each shard's columns of every projection sit together,
    so every device's product reads its own columns alone. The joined
    output, `[..., shards, columns]`, is placed by `output_axes`."""
    promoted = _promoted(x, projections)
    mesh = jax.sharding.get_abstract_mesh()
    shards = math.prod(mesh.shape[name] for name in _split(projections[0], kernel_axes[0]))

    def by_shard(value):
        return value.reshape(*value.shape[:-1], shards, value.shape[-1] // shards)

    y = _product(promoted[0][0], jnp.concatenate([by_shard(kernel) for _, kernel, _ in promoted], axis=-1),
                 projections[0])
    if projections[0].use_bias:
        y = y + jnp.concatenate([by_shard(bias) for _, _, bias in promoted], axis=-1)
    ends = np.cumsum([projection.features // shards for projection in projections])[:-1]
    return [part.reshape(*part.shape[:-2], -1)
            for part in jnp.split(constrain(y, output_axes), [int(end) for end in ends], axis=-1)]
