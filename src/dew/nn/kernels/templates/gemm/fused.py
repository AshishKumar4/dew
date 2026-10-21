"""The grouped GEMM template under `jax.experimental.pallas.fuser`: a call site's consumers of the
product become the template's epilogue.

Inside a function `fuser.fuse` transforms, `fusible_gemm(lhs, rhs, group_sizes)` is `template.gemm`'s
fp32 product, and whatever the function computes from it elementwise or by broadcasting (a bias, a scale,
a residual, an activation, the final cast) runs in the kernel's epilogue on each accumulator tile, the
arrays it reads loaded beside the tile:

    @fuser.fuse
    def layer(x, w, group_sizes, bias):
        return (fused.fusible_gemm(x, w, group_sizes) + bias).astype(jnp.bfloat16)

The fuser hands the consumer over as one function of the product and the arrays it captured
(`fuser.get_fusion_values`); `pull_block_spec` names the launch axes each captured array follows and
evaluates the consumer on tiles, so the epilogue is a plain JAX function that tiles itself at whatever
block the kernel traces it with (`acc.shape`), and runs whole where XLA does. A consumer the kernels
cannot take tile by tile (one that reads the tile's position, a captured array that follows no launch
axis, scalar prefetch) runs in XLA after the template's product, the value the same.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
from jax.extend import core as jax_core

from dew.nn.kernels.templates.gemm import template

AXES = ('rows', 'columns')
"""The epilogue's launch axes, in a tile's order."""


def fusible_gemm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None = None, *,
                 transpose_rhs: bool = False, interpret_on_cpu: bool = False) -> jax.Array:
    """`template.gemm`'s fp32 product of each group's rows of `lhs` and its matrix of `rhs`, its consumers
    fused into the epilogue under `fuser.fuse` (the module's docstring)."""
    if group_sizes is None:
        return _fusible(transpose_rhs, interpret_on_cpu, dense=True)(lhs, rhs)
    return _fusible(transpose_rhs, interpret_on_cpu, dense=False)(lhs, rhs, group_sizes)


@functools.cache
def _fusible(transpose_rhs: bool, interpret_on_cpu: bool, *, dense: bool):
    """The fusible product at these options, one per options so its operands are arrays alone."""

    def product(*fusions):
        *operand_fusions, out_fusion = fusions
        lhs, rhs, *rest = (fusion() for fusion in operand_fusions)
        group_sizes = None if dense else rest[0]
        options = {'transpose_rhs': transpose_rhs, 'interpret_on_cpu': interpret_on_cpu}
        if out_fusion is None:
            return template.gemm(lhs, rhs, group_sizes, out_dtype=jnp.float32, **options)
        width = rhs.shape[-2] if transpose_rhs else rhs.shape[-1]
        acc = jax.ShapeDtypeStruct((lhs.shape[0], width), jnp.float32)
        consumer, values, prefetch = fuser.get_fusion_values(out_fusion, acc)
        leaves, treedef = jax.tree.flatten(values)
        axes = _axes(consumer, values, acc) if not prefetch else None
        if axes is None:
            summed = template.gemm(lhs, rhs, group_sizes, out_dtype=jnp.float32, **options)
            return consumer(values, summed)
        outs = jax.eval_shape(consumer, values, acc)
        out_leaves, out_treedef = jax.tree.flatten(outs)
        value_types = tuple(jax.typeof(leaf) for leaf in leaves)
        epilogue = functools.partial(_tiled, consumer, treedef, value_types, out_treedef, acc.shape)
        several = len(out_leaves) > 1
        dtypes = tuple(leaf.dtype for leaf in out_leaves)
        result = template.gemm(lhs, rhs, group_sizes, epilogue=epilogue,
                               epilogue_operands=tuple(zip(leaves, axes, strict=True)), out_dtype=dtypes[0],
                               out_dtypes=dtypes if several else None, **options)
        return jax.tree.unflatten(out_treedef, result if several else (result,))

    return fuser.fusible(product, output_fusion_prefix=True)


def _axes(consumer, values, acc) -> tuple | None:
    """The launch axes each captured array follows, one per dimension, from its block spec at a probe
    tile; None where one follows none, its block is not the tile's, or the consumer reads the tile's
    position."""
    probe = (_probe(acc.shape[0]), _probe(acc.shape[1]))
    tiled, (value_specs, _), _ = fuser.pull_block_spec(consumer, _specs(consumer, values, acc, probe),
                                                       grid_len=2)(values, acc)
    axes = []
    specs = jax.tree.leaves(value_specs, is_leaf=_is_spec)
    for leaf, leaf_spec in zip(jax.tree.leaves(values), specs, strict=True):
        named = []
        indices = leaf_spec.index_map(1, 2)
        for block, index, size in zip(leaf_spec.block_shape, indices, leaf.shape, strict=True):
            if index not in (1, 2) or block != probe[index - 1] or size != acc.shape[index - 1]:
                return None
            named.append(AXES[index - 1])
        axes.append(tuple(named))
    if _reads_position(tiled, value_specs, values, probe):
        return None
    return tuple(axes)


def _specs(consumer, values, acc, block: tuple):
    """One block spec of `block` at the tile's (row, column) for each of the consumer's outputs."""
    spec = pl.BlockSpec(block, lambda row, column: (row, column))
    return jax.tree.map(lambda _: spec, jax.eval_shape(consumer, values, acc))


def _probe(size: int) -> int:
    """A probe block of `size`: a power of two dividing it where one does, so the block spec is exact."""
    for block in (64, 32, 16, 8):
        if size % block == 0 and size > block:
            return block
    return size


def _is_spec(node) -> bool:
    return isinstance(node, pl.BlockSpec)


def _reads_position(tiled, value_specs, values, probe) -> bool:
    """Whether the tiled consumer's outputs depend on the program ids it is given (an iota, a position
    mask): the template's kernels place ragged tiles where no grid index says."""
    tiles = jax.tree.map(lambda leaf, leaf_spec: jax.ShapeDtypeStruct(leaf_spec.block_shape, leaf.dtype),
                         values, value_specs, is_leaf=_is_spec)
    tile = jax.ShapeDtypeStruct(probe, jnp.float32)
    closed = jax.make_jaxpr(lambda ids, tiles, tile: tiled(ids, (), tiles, tile))(
        (jnp.int32(0), jnp.int32(0)), tiles, tile)
    needed = {var for var in closed.jaxpr.outvars if isinstance(var, jax_core.Var)}
    for equation in reversed(closed.jaxpr.eqns):
        if needed.intersection(equation.outvars):
            needed.update(var for var in equation.invars if isinstance(var, jax_core.Var))
    return any(var in needed for var in closed.jaxpr.invars[:2])


def _tiled(consumer, treedef, value_types: tuple, out_treedef, whole: tuple, acc, *tiles):
    """The consumer on one accumulator tile of `acc.shape` and the captured arrays' tiles beside it: its
    block spec pulled at that block of the `whole` product it was traced at, so the same function serves
    every kernel tile and XLA's whole array."""
    values = jax.tree.unflatten(treedef, [jax.ShapeDtypeStruct(t.shape, t.dtype) for t in value_types])
    whole_acc = jax.ShapeDtypeStruct(whole, jnp.float32)
    tiled, _, _ = fuser.pull_block_spec(consumer, _specs(consumer, values, whole_acc, acc.shape),
                                        grid_len=2)(values, whole_acc)
    outs = tiled((jnp.int32(0), jnp.int32(0)), (), jax.tree.unflatten(treedef, list(tiles)), acc)
    return tuple(jax.tree.leaves(outs)) if out_treedef.num_leaves > 1 else jax.tree.leaves(outs)[0]
