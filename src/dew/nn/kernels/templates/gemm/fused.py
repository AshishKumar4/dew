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
axis, scalar prefetch) runs in XLA after the template's product, the value the same, and the call reports
it (`generation.route`'s 'gemm_epilogue', XLA's with the refusal's reason), so a model that loses its
fusion shows where it does.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser

from dew.nn.kernels import generation
from dew.nn.kernels.templates.gemm import template

AXES = ('rows', 'columns')
"""The epilogue's launch axes, in a tile's order."""


def fusible_gemm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None = None, *,
                 transpose_rhs: bool = False, interpret_on_cpu: bool = False) -> jax.Array:
    """`template.gemm`'s fp32 product of each group's rows of `lhs` and its matrix of `rhs`, its consumers
    fused into the epilogue under `fuser.fuse` (the module's docstring)."""
    fusible = fuser.fusible(output_fusion_prefix=True)(
        functools.partial(_product, transpose_rhs, interpret_on_cpu, group_sizes is None))
    return fusible(lhs, rhs) if group_sizes is None else fusible(lhs, rhs, group_sizes)


def _product(transpose_rhs: bool, interpret_on_cpu: bool, dense: bool, *fusions):
    """The fusible product: its operands from their fusions, its consumers (`fusions`' last) as the
    template's epilogue where the kernels take them tile by tile, after its product in XLA otherwise."""
    *operand_fusions, out_fusion = fusions
    lhs, rhs, *rest = (fusion() for fusion in operand_fusions)
    group_sizes = None if dense else rest[0]
    if out_fusion is None:
        return template.gemm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, out_dtype=jnp.float32,
                             interpret_on_cpu=interpret_on_cpu)
    width = rhs.shape[-2] if transpose_rhs else rhs.shape[-1]
    acc = jax.ShapeDtypeStruct((lhs.shape[0], width), jnp.float32)
    consumer, values, prefetch = fuser.get_fusion_values(out_fusion, acc)
    leaves, treedef = jax.tree.flatten(values)
    axes = (generation.Refusal('scalar_prefetch', "the consumer reads scalar prefetch") if prefetch
            else _axes(consumer, values, acc))
    if isinstance(axes, generation.Refusal):
        generation.route('gemm_epilogue', kernel='xla', preferred='fused', refusal=axes)
        summed = template.gemm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, out_dtype=jnp.float32,
                               interpret_on_cpu=interpret_on_cpu)
        return consumer(values, summed)
    generation.route('gemm_epilogue', kernel='fused')
    out_leaves, out_treedef = jax.tree.flatten(jax.eval_shape(consumer, values, acc))
    value_types = tuple(jax.typeof(leaf) for leaf in leaves)
    epilogue = functools.partial(_tiled, consumer, treedef, value_types, out_treedef, acc.shape)
    dtypes = tuple(leaf.dtype for leaf in out_leaves)
    several = len(dtypes) > 1
    consumed = template.gemm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, epilogue=epilogue,
                             epilogue_operands=tuple(zip(leaves, axes, strict=True)), out_dtype=dtypes[0],
                             out_dtypes=dtypes if several else None, interpret_on_cpu=interpret_on_cpu)
    return jax.tree.unflatten(out_treedef, consumed if several else (consumed,))


def _axes(consumer, values, acc) -> tuple | generation.Refusal:
    """The launch axes each captured array follows, one per dimension, from its block spec at a probe
    tile; the refusal where the product's own block is not the tile at its place (a transpose), a
    captured array follows no launch axis (a scalar) or its block is not the tile's, or the consumer reads
    the tile's position."""
    probe = (_probe(acc.shape[0]), _probe(acc.shape[1]))
    tiled, (value_specs, acc_spec), _ = fuser.pull_block_spec(consumer, _specs(consumer, values, acc, probe),
                                                              grid_len=2)(values, acc)
    if not _is_tile(acc_spec, probe):
        return generation.Refusal('product_untiled',
                                  "the consumer reads the product at other than its tile")
    axes = []
    specs = jax.tree.leaves(value_specs, is_leaf=_is_spec)
    for leaf, leaf_spec in zip(jax.tree.leaves(values), specs, strict=True):
        block_shape, index_map = _block_and_map(leaf_spec)
        if block_shape is None or index_map is None:
            return generation.Refusal('value_unblocked', f"a captured {leaf.shape} array has no block")
        named = []
        for block, index, size in zip(block_shape, index_map(1, 2), leaf.shape, strict=True):
            if index not in (1, 2) or block != probe[index - 1] or size != acc.shape[index - 1]:
                return generation.Refusal(
                    'value_off_axes', f"a captured {leaf.shape} array follows no launch axis tile by tile")
            named.append(AXES[index - 1])
        axes.append(tuple(named))
    if _reads_position(tiled, value_specs, values, probe):
        return generation.Refusal('reads_position', "the consumer reads its tile's position")
    return tuple(axes)


def _specs(consumer, values, acc, block: tuple):
    """One block spec of `block` at the tile's (row, column) for each of the consumer's outputs."""
    spec = pl.BlockSpec(block, lambda row, column: (row, column))
    return jax.tree.map(lambda _: spec, jax.eval_shape(consumer, values, acc))


def _is_tile(spec, probe: tuple) -> bool:
    """Whether `spec` is the probe tile at its own place: a block of `probe` at (row, column)."""
    block_shape, index_map = _block_and_map(spec)
    return block_shape is not None and index_map is not None and tuple(block_shape) == probe and tuple(
        index_map(1, 2)) == (1, 2)


def _block_and_map(spec):
    """A block spec's block and index map; none for a spec that is not a block spec (a scalar's)."""
    if not isinstance(spec, pl.BlockSpec):
        return None, None
    return spec.block_shape, spec.index_map


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
    needed = template._needed(closed.jaxpr, closed.jaxpr.outvars)
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
