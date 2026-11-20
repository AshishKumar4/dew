"""A layer's projections through the GEMM template (`dot_general`, as flax's `Dense` takes a product): its
matrix product in the run's low-precision recipe where its role takes it (`recipe_projection`: the blockwise
FP8, MXFP8 and NVFP4 recipes, each quantizing its operands and differentiated by its own products, a surrogate
of the quantized product's derivative), else through the template's dense product (`gemm.gemm`) where a
launch takes any of its products, else XLA's own program."""

import functools
import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax.typing import PrecisionLike

from dew import _cache
from dew.mesh import mesh_axes
from dew.nn.derivatives import (
    Surrogate,
    contracted,
    declared_derivative,
    portable_formed,
    with_gradients_formed,
)
from dew.nn.kernels import generation, grouped_matmul, selection
from dew.nn.kernels.formats import MxFp8, NvFp4Block
from dew.nn.kernels.gemm import (
    blockwise_fp8_kernel_gradient,
    blockwise_fp8_product,
    gemm,
    gemm_t,
    mxfp8_product,
    nvfp4_block_product,
)
from dew.nn.kernels.templates.product import MXFP8 as MX_BLOCKS, NVFP4 as NVFP4_BLOCKS, SCALE_DEPTH, declared
from dew.nn.sharding import (
    RESIDUAL,
    automatic_axes,
    explicit_spec,
    logical_spec,
    on_shards,
    placed_layout,
    seen_whole,
    shard_shapes,
)


def blockwise_fp8_projection(x: jax.Array, w: jax.Array, group_blocks: jax.Array, *, out_dtype,
                             interpret_on_cpu: bool = False) -> jax.Array:
    """Each group's rows of `x` (`[rows, depth]`) times its matrix of `w` (`[groups, depth, width]`) in
    DeepSeek-V3's blockwise FP8 recipe (row 2513): `x` quantized to E4M3 a 1 x 128 tile at a time along the
    depth and `w` a 128 x 128 block at a time (power-of-two scales, `fake_quant.fp8_scaled_blocks`), their
    product `blockwise_fp8_product`. Its gradients are the recipe's own FP8 products: the input's the
    cotangent quantized 1 x 128 along the width times the blocks transposed, the matrices' the rows and the
    cotangent each quantized 1 x 128 along the rows (`blockwise_fp8_kernel_gradient`), a surrogate of the
    quantized product's derivative (straight through each rounding). Group `g` holds `group_blocks[g]` whole
    128-row blocks, in order: `rows` is 128 times their sum (a recipe pads each group's rows)."""
    return _blockwise_projection(x, w, group_blocks, jnp.dtype(out_dtype), interpret_on_cpu)


def recipe_refusal(depth: int, width: int) -> selection.Refusal | None:
    """Why the run's low-precision recipe (`dew.nn.precision.recipe_role`) does not take a `depth` x `width`
    projection traced in its role, None where it takes it: the run trains in none, its role keeps the run's
    dtypes (a router's, a head's), or the widths hold no whole blocks of the recipe's. The one decision
    `recipe_projection` and a join of projections (`recipe_takes`, `dew.nn.joined.Projections.recipe`)
    read."""
    from dew.nn.precision import recipe_role

    if (held := recipe_role()) is None:
        return selection.Refusal("no_recipe", "the run trains in no low-precision recipe", kind="handoff")
    recipe, role = held
    block = RECIPES[recipe.recipe].block
    if role not in recipe.roles:
        return selection.Refusal("kept_role", f"a {role or 'role-less'} projection keeps the run's dtypes "
                                 f"under {recipe.recipe}", kind="handoff")
    if depth % block or width % block:
        return selection.Refusal("block_shape", f"a {depth} x {width} projection holds no whole "
                                 f"{block}-wide blocks", kind="handoff")
    return None


def recipe_takes(depth: int, width: int) -> bool:
    """Whether the run's low-precision recipe takes a `depth` x `width` projection traced in its role
    (`recipe_refusal`)."""
    return recipe_refusal(depth, width) is None


def recipe_projection(rows: jax.Array, matrix: jax.Array, group_sizes: jax.Array | None = None, *,
                      out_dtype, bias: jax.Array | None = None) -> jax.Array | None:
    """`rows @ matrix` in the run's low-precision recipe (`dew.nn.precision.recipe_role`), where the
    projection's role takes it: a dense projection's (`matrix` `[in, out]`, every row its one group's) or
    experts' (`[groups, in, out]`, `group_sizes` counting each group's leading rows, any past their sum
    outside every group), each group's rows padded with zero rows to whole 128-row blocks, multiplied by
    the recipe's projection (`RECIPES`) and the padding dropped, a row outside every group zero; `bias`, each
    group's `[groups, out]`, added to its rows' fp32 sums, the output rounded once after. None where the run
    trains in no recipe or the recipe does not take the projection: its role keeps the run's dtypes (a
    router's, a head's) or its widths hold no whole blocks of the recipe's, each such projection's route
    reported with the reason."""
    from dew.nn.precision import recipe_role

    held, refusal = recipe_role(), recipe_refusal(*matrix.shape[-2:])
    if held is None or refusal is not None:
        if held is not None:  # a run's recipe that leaves this projection: its route reported
            generation.route(RECIPES[held[0].recipe].choice, kernel="unquantized", preferred=held[0].recipe,
                             refusal=refusal)
        return None
    recipe = held[0]
    taken = RECIPES[recipe.recipe]
    edge = SCALE_DEPTH
    # Rows an Explicit axis splits, or the mesh's automatic axes alone (a data-parallel run's, row 6719):
    # each device's rows its own projection; a launch reads its operands whole.
    if (placed := explicit_spec(rows)) is not None and placed.spec[0] is not None:
        along = placed.spec[0]
    else:
        along = None if (split := rows_split(rows)) is None else split[0]
    if along is not None:
        return _projected_apart(rows, matrix, group_sizes, out_dtype=out_dtype, bias=bias, along=along)
    stacked = matrix if matrix.ndim == 3 else matrix[None]
    groups, count = stacked.shape[0], rows.shape[0]
    sizes = (jnp.full((1,), count, jnp.int32) if group_sizes is None
             else jnp.asarray(group_sizes, jnp.int32))
    # Each row's group (`groups` past their sum) and its place in its group's padded run of whole blocks.
    ends = jnp.cumsum(sizes)
    group = jnp.searchsorted(ends, jnp.arange(count), side="right")
    blocks = -(-sizes // edge)
    padded = -(-(count + groups * (edge - 1)) // edge) * edge
    held_group = jnp.minimum(group, groups - 1)
    place = jnp.where(group < groups, (jnp.cumsum(blocks) - blocks)[held_group] * edge
                      + jnp.arange(count) - (ends - sizes)[held_group], padded)
    laid = jnp.zeros((padded, rows.shape[1]), rows.dtype).at[place].set(rows, mode="drop")
    # The run's slack, zero rows, as the last group's further blocks: the recipe's rows are whole blocks.
    group_blocks = blocks.at[-1].add(padded // edge - blocks.sum())
    # A biased product's sums stay fp32 through the bias, as the template's epilogue adds it (moe's
    # `_biased`): one rounding of the whole.
    summed_dtype = out_dtype if bias is None else jnp.float32
    product = taken.projection(laid, stacked, group_blocks, out_dtype=summed_dtype)
    out = product.at[place].get(mode="fill", fill_value=0)
    if bias is not None:
        out = jnp.where((group < groups)[:, None], out + bias[held_group].astype(jnp.float32), out)
    return out.astype(out_dtype)


def _projected_apart(rows, matrix, group_sizes, *, out_dtype, bias, along) -> jax.Array:
    """`recipe_projection` of rows the mesh's axes `along` split, Explicit (rows 4854, 4859) or automatic
    alone (row 6719): each device's rows its own projection (`on_shards`, whose map is unchecked as a Pallas
    launch's must be), laid in whole 128-row blocks of its own as a data-parallel recipe quantizes them, the
    matrix, the groups' sizes and the bias whole on every device and the matrix's and bias's gradients
    summed over the devices (the map's transpose). Experts' sizes count every device's rows together, each
    group's after the last's: a device's groups are its share of each (`_local_sizes`)."""

    names = mesh_axes(along)

    def projected(rows, matrix, group_sizes, bias):
        sizes = None if group_sizes is None else _local_sizes(group_sizes, rows.shape[0], names)
        out = recipe_projection(rows, matrix, sizes, out_dtype=out_dtype, bias=bias)
        assert out is not None, "the recipe takes the projection on every device as on the whole"
        return out

    return on_shards(projected, jax.sharding.PartitionSpec(along, None), (rows, matrix, group_sizes, bias),
                     whole=(1, 2, 3))


def _local_sizes(group_sizes: jax.Array, count: int, names: tuple) -> jax.Array:
    """A device's share of each group of `group_sizes` (counting every device's rows, each group's after the
    last's), its `count` rows the device's run of them (`names`' index along the rows): each group's rows
    within the device's, so the device's rows in groups are a prefix of its own, as `recipe_projection`
    takes them."""
    first = jax.lax.axis_index(names) * count
    ends = jnp.cumsum(group_sizes.astype(jnp.int32))
    starts = ends - group_sizes.astype(jnp.int32)
    return jnp.clip(ends - first, 0, count) - jnp.clip(starts - first, 0, count)


def rows_quantized(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`x` (`[..., n]`) as E4M3 codes a 1 x 128 tile at a time along its last axis and each tile's
    power-of-two scale (`[..., n / 128]` fp32)."""
    from dew.nn import fake_quant

    blocks, scale = fake_quant.fp8_scaled_blocks(x, SCALE_DEPTH)
    return fake_quant.fp8_codes(blocks / scale).reshape(x.shape).astype(jnp.float8_e4m3fn), scale[..., 0]


def blocks_quantized(w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`w` (`[groups, depth, width]`) as E4M3 codes a 128 x 128 block at a time and each block's power-of-two
    scale (`[groups, depth / 128, width / 128]` fp32)."""
    from dew.nn import fake_quant

    groups, depth, width = w.shape
    edge = SCALE_DEPTH
    tiled = w.reshape(groups, depth // edge, edge, width // edge, edge).transpose(0, 1, 3, 2, 4)
    blocks, scale = fake_quant.fp8_scaled_blocks(tiled.reshape(groups, depth // edge, width // edge, -1),
                                                 edge * edge)
    codes = fake_quant.fp8_codes(blocks / scale).reshape(tiled.shape).transpose(0, 1, 3, 2, 4)
    return codes.reshape(w.shape).astype(jnp.float8_e4m3fn), scale[..., 0, 0]


def _projection_value(x, w, group_blocks, out_dtype, interpret):
    rows, rows_scales = rows_quantized(x)
    blocks, blocks_scales = blocks_quantized(w)
    sizes = (group_blocks * SCALE_DEPTH).astype(jnp.int32)
    return blockwise_fp8_product(rows, jnp.swapaxes(blocks, 1, 2), sizes, rows_scales.T, blocks_scales,
                                 out_dtype=out_dtype, interpret_on_cpu=interpret)


def _projection_forward(x, w, group_blocks, out_dtype, interpret):
    return _projection_value(x, w, group_blocks, out_dtype, interpret), (x, w, group_blocks)


def _projection_backward(out_dtype, interpret, residuals, cotangent):
    x, w, group_blocks = residuals
    sizes = (group_blocks * SCALE_DEPTH).astype(jnp.int32)
    dy, dy_scales = rows_quantized(cotangent)
    blocks, blocks_scales = blocks_quantized(w)
    d_x = blockwise_fp8_product(dy, blocks, sizes, dy_scales.T, jnp.swapaxes(blocks_scales, 1, 2),
                                out_dtype=x.dtype, interpret_on_cpu=interpret)
    rows_t, rows_t_scales = rows_quantized(x.T)
    dy_t, dy_t_scales = rows_quantized(cotangent.T)
    d_w = blockwise_fp8_kernel_gradient(rows_t, dy_t, group_blocks, rows_t_scales.T, dy_t_scales.T,
                                        out_dtype=w.dtype, interpret_on_cpu=interpret)
    return (*with_gradients_formed(d_x, d_w), None)

# The recipe's gradient (`blockwise_fp8_projection`): a derivative of Dew's choosing (`Surrogate`).
_blockwise_projection = contracted("test_a_blockwise_fp8_projection_holds_every_reverse_order")(
    declared_derivative(Surrogate(), _projection_value, _projection_forward, _projection_backward,
                        nondiff_argnums=(3, 4)))


def mxfp8_projection(x: jax.Array, w: jax.Array, group_blocks: jax.Array, *, out_dtype,
                     interpret_on_cpu: bool = False) -> jax.Array:
    """Each group's rows of `x` (`[rows, depth]`) times its matrix of `w` (`[groups, depth, width]`) in the
    MXFP8 recipe (row 2513; OCP MX, arXiv 2506.08027: E4M3 every tensor, a power-of-two scale every 32 of each
    product's contraction): `x` quantized along the depth and `w` along it too, their product
    `mxfp8_product`. Its gradients are the recipe's own MX products, a surrogate of the quantized product's
    derivative (straight through each rounding): the input's the cotangent quantized along the width times
    `w` quantized along it (`mxfp8_product`); the matrices' each group's rows and cotangent quantized along
    its tokens, their decoded values (exact in bfloat16) contracted over the group's rows in fp32 (`gemm_t`).
    Group `g` holds `group_blocks[g]` whole 128-row blocks, in order: `rows` is 128 times their sum."""
    return _mx_projection(x, w, group_blocks, jnp.dtype(out_dtype), interpret_on_cpu)


def _recipe_value(form, product, x, w, group_blocks, out_dtype, interpret):
    """A block-scaled recipe's value (`mxfp8_projection`'s, `nvfp4_projection`'s): `x` and `w` quantized
    along the depth to `form`, their `product`."""
    return product(form.quantized(x), form.quantized(jnp.swapaxes(w, 1, 2)), group_blocks,
                   out_dtype=out_dtype, interpret_on_cpu=interpret)


def _recipe_forward(value, x, w, group_blocks, out_dtype, interpret):
    """A block-scaled recipe's forward: its `value`, the operands its backward reads."""
    return value(x, w, group_blocks, out_dtype, interpret), (x, w, group_blocks)


def _mx_value(x, w, group_blocks, out_dtype, interpret):
    return _recipe_value(MxFp8, mxfp8_product, x, w, group_blocks, out_dtype, interpret)


def _mx_forward(x, w, group_blocks, out_dtype, interpret):
    return _recipe_forward(_mx_value, x, w, group_blocks, out_dtype, interpret)


def _mx_backward(out_dtype, interpret, residuals, cotangent):
    del out_dtype
    x, w, group_blocks = residuals
    d_x = mxfp8_product(MxFp8.quantized(cotangent), MxFp8.quantized(w), group_blocks, out_dtype=x.dtype,
                        interpret_on_cpu=interpret)
    sizes = (group_blocks * SCALE_DEPTH).astype(jnp.int32)
    # Each group's tokens quantized along the tokens (the kernel gradient's contraction), every 32 of them a
    # scale: a 128-row group's blocks hold whole 32s.
    rows = MxFp8.quantized(x.T).decoded().T.astype(jnp.bfloat16)
    cotangents = MxFp8.quantized(cotangent.T).decoded().T.astype(jnp.bfloat16)
    d_w = gemm_t(rows, cotangents, sizes, out_dtype=w.dtype, precision=jax.lax.Precision.DEFAULT,
                 interpret_on_cpu=interpret)
    return (*with_gradients_formed(d_x, d_w), None)

# The MXFP8 recipe's gradient (`mxfp8_projection`): a derivative of Dew's choosing (`Surrogate`).
_mx_projection = contracted("test_an_mxfp8_projection_holds_every_reverse_order")(
    declared_derivative(Surrogate(), _mx_value, _mx_forward, _mx_backward, nondiff_argnums=(3, 4)))


def nvfp4_projection(x: jax.Array, w: jax.Array, group_blocks: jax.Array, *, out_dtype,
                     interpret_on_cpu: bool = False) -> jax.Array:
    """Each group's rows of `x` (`[rows, depth]`) times its matrix of `w` (`[groups, depth, width]`) in the
    NVFP4 recipe (row 2513: E2M1 every tensor, an E4M3 scale every 16 of each product's contraction under a
    tensor scale): `x` and `w` quantized along the depth, their product `nvfp4_block_product`. Its gradients
    are the recipe's own NVFP4 products, a surrogate of the quantized product's derivative (straight through
    each rounding), as `mxfp8_projection`'s: the input's the cotangent and `w` quantized along the width;
    the matrices' each group's rows and cotangent quantized along its tokens, their decoded values contracted
    over the group's rows in fp32 (`gemm_t`). Group `g` holds `group_blocks[g]` whole 128-row blocks."""
    return _nvfp4_projection(x, w, group_blocks, jnp.dtype(out_dtype), interpret_on_cpu)


def _nvfp4_value(x, w, group_blocks, out_dtype, interpret):
    return _recipe_value(NvFp4Block, nvfp4_block_product, x, w, group_blocks, out_dtype, interpret)


class Recipe(NamedTuple):
    """A low-precision recipe's projection (`recipe_projection`): `projection(x, w, group_blocks, out_dtype=)`
    over whole 128-row blocks, the width `block` its operands' scales cover along each product's
    contraction (each of a projection's depth and width), and the `choice` its products' routes report."""

    projection: Callable
    block: int
    choice: generation.Choice


RECIPES = {"blockwise_fp8": Recipe(blockwise_fp8_projection, SCALE_DEPTH, generation.Choice.BLOCKWISE_FP8),
           "mxfp8": Recipe(mxfp8_projection, MX_BLOCKS.depth, generation.Choice.MXFP8),
           "nvfp4": Recipe(nvfp4_projection, NVFP4_BLOCKS.depth, generation.Choice.NVFP4_BLOCK)}
"""The low-precision recipes a run's projections train in (`dew.nn.precision.LowPrecision.recipe`), by name:
the one declaration of them, each product's sums fp32 and its gradients the recipe's own products of
requantized operands. DeepSeek-V3's blockwise FP8, its operands E4M3 by 1 x 128 activation tiles and 128 x
128 weight blocks under power-of-two scales; MXFP8 (OCP MX), E4M3 under a power-of-two scale every 32 of
each product's contraction; NVFP4, E2M1 under an E4M3 scale every 16 of it and a tensor scale."""


def _nvfp4_forward(x, w, group_blocks, out_dtype, interpret):
    return _recipe_forward(_nvfp4_value, x, w, group_blocks, out_dtype, interpret)


def _nvfp4_backward(out_dtype, interpret, residuals, cotangent):
    del out_dtype
    x, w, group_blocks = residuals
    d_x = nvfp4_block_product(NvFp4Block.quantized(cotangent), NvFp4Block.quantized(w), group_blocks,
                              out_dtype=x.dtype, interpret_on_cpu=interpret)
    sizes = (group_blocks * SCALE_DEPTH).astype(jnp.int32)
    # Each group's tokens quantized along the tokens (the kernel gradient's contraction), every 16 a scale,
    # the tensor scales applied: a 128-row group's blocks hold whole 16s.
    rows, cotangents = (NvFp4Block.quantized(operand.T) for operand in (x, cotangent))
    d_w = gemm_t((rows.decoded() * rows.scale).T.astype(jnp.float32),
                 (cotangents.decoded() * cotangents.scale).T.astype(jnp.float32), sizes, out_dtype=w.dtype,
                 precision=jax.lax.Precision.HIGHEST, interpret_on_cpu=interpret)
    return (*with_gradients_formed(d_x, d_w), None)

# The NVFP4 recipe's gradient (`nvfp4_projection`): a derivative of Dew's choosing (`Surrogate`).
_nvfp4_projection = contracted("test_an_nvfp4_projection_holds_every_reverse_order")(
    declared_derivative(Surrogate(), _nvfp4_value, _nvfp4_forward, _nvfp4_backward, nondiff_argnums=(3, 4)))


def dot_general(xla, path: tuple | None = None):
    """`xla`, a layer's product (a `jax.lax.dot_general`, as flax's `Dense` takes one), in the run's
    low-precision recipe where its role takes the matrix product (`recipe_projection`), else through the
    template's dense product (`gemm` of one group) where the call is a matrix product (`lhs`'s last axis
    against a matrix's first, no batch axes, both of one dtype), or its grouped product where it batches
    (each batch member a group, `grouped_rows`), and a launch takes any of its products: the forward, the
    input gradient or the kernel gradient (`launched`). A call no launch takes is `xla`'s own
    program, differentiated by JAX, each of its products' routes reported with the reason no launch takes
    it (a declared cell's old route, a mesh axis splitting the operands, a CPU), so a layer runs as it ran
    before wherever the template declares XLA faster; every other product is `xla`'s. On a mesh whose
    automatic axes split the rows alone, where a launch takes a device's rows' product, each device's rows
    are its own call (`mapped_shards`), its matrix, the parameter at its variables `path`, placed by the
    traced program's layout: a kernel launch per device on its own rows (bible §7, rows 1306 and 1923)."""
    def product(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None, **kwargs):
        (contracted, against), batch = dimension_numbers
        out_dtype = jnp.dtype(preferred_element_type or lhs.dtype)
        # The run's low-precision recipe takes the product where the projection's role takes it.
        if not kwargs and (recipe := recipe_general(lhs, rhs, dimension_numbers, out_dtype)) is not None:
            return recipe
        plain = (not kwargs and jnp.ndim(rhs) == 2 and lhs.dtype == rhs.dtype
                 and tuple(contracted) == (lhs.ndim - 1,) and tuple(against) == (0,)
                 and tuple(map(tuple, batch)) == ((), ()))
        if plain and path is not None and (shards := mapped_shards(lhs, (rhs,), (path,), out_dtype,
                                                                     precision)) is not None:
            split, gathered = shards
            return on_shards(lambda lhs, rhs: product(lhs, rhs, dimension_numbers, precision,
                                                      preferred_element_type), split, (lhs, rhs),
                             gathered=gathered)
        if plain:
            rows = lhs.reshape(-1, lhs.shape[-1])
            if launched(rows, rhs, out_dtype, precision):
                out = gemm(rows, rhs, out_dtype=out_dtype, precision=precision)
                return out.reshape(*lhs.shape[:-1], rhs.shape[1])
        elif not kwargs and lhs.dtype == rhs.dtype and batch[0] and seen_whole(lhs) and seen_whole(rhs):
            # A batched projection's members as uniform groups (`grouped_rows`, Appendix C row 4): one
            # grouped GEMM where a launch takes any of its products, as a dense one's (DeepSeek V4's
            # `o_a_proj`, a `DenseGeneral` over its groups; spec alignment SA20261119pm F36).
            rows, matrices, count, shape = grouped_rows(lhs, rhs, dimension_numbers)
            # One member is the dense product it is, probed and run as one: the dense product's cells, not
            # one group's (head review of 0099a4701 F3).
            matrix, sizes = ((matrices[0], None) if matrices.shape[0] == 1
                             else (matrices, jnp.full((matrices.shape[0],), count, jnp.int32)))
            if launched(rows, matrix, out_dtype, precision):
                return gemm(rows, matrix, sizes, out_dtype=out_dtype, precision=precision).reshape(shape)
        taken = _in_layer_product(xla, dimension_numbers, precision, preferred_element_type, **kwargs)
        return taken(lhs, rhs)

    return product


def _in_layer_product(xla, dimension_numbers, precision, preferred_element_type, **kwargs):
    """A projection's product no launch takes, XLA's own (`xla`, any of its forms: a batched one's, DeepSeek
    V4's grouped `o_a_proj`), through the rule as the template's is (`derivatives.portable_formed`): the
    input's cotangent returned only with the kernel's gradient formed."""
    def portable(lhs, rhs):
        return xla(lhs, rhs, dimension_numbers, precision=precision,
                   preferred_element_type=preferred_element_type, **kwargs)

    return portable_formed(portable)


def rows_split(lhs: jax.Array) -> jax.sharding.PartitionSpec | None:
    """How the context's mesh splits a projection's rows (`lhs`'s every axis but its last, an activation's
    batch and length, `RESIDUAL`), where its automatic axes (`automatic_axes`) split those rows alone, each
    axis a row axis's, the contracted axis whole; None where no automatic axis splits the call or one
    splits more than its rows (a tensor axis's split of the depth or the outputs, which the partitioner
    owns), or where an Explicit axis splits any operand (its placement the operands' types carry, which the
    product's own sharding rules read: no rule table infers one here)."""
    mesh = jax.sharding.get_abstract_mesh()
    if any(mesh.shape[name] > 1 for name in mesh.explicit_axes):
        return None
    automatic = set(automatic_axes())
    if not automatic or lhs.ndim < 2:
        return None
    lead = min(lhs.ndim - 1, 2)
    rows = logical_spec(RESIDUAL[:lead], lhs.shape[:lead])
    named = {axis for entry in rows for axis in ((entry,) if isinstance(entry, str) else entry or ())}
    return jax.sharding.PartitionSpec(*rows) if named == automatic else None


def mapped_shards(lhs: jax.Array, matrices: tuple[jax.Array, ...], paths: tuple, out_dtype,
                  precision: PrecisionLike) -> tuple[jax.sharding.PartitionSpec, dict] | None:
    """Where a projection of `lhs` by `matrices` (one, or a joined projection's kernels side by side, each
    the parameter at its variables `path`) runs per shard, as fully sharded data parallelism defines one
    (`shard_placement`): the rows as `rows_split` splits them and each matrix where the traced program's
    layout places it (`placing`, `Layout.leaf_spec`), gathered whole in the map (`on_shards`' `gathered`),
    so its gradient leaves the map reduce-scattered back to that placement; where a launch takes the product
    of a device's own rows (`launched`'s decision, reported once inside the map). None elsewhere, the
    product the partitioner's: no rows split alone, no layout traced under, or a product no launch takes per
    device (an XLA-declared cell, a CPU or a TPU), which runs as it ran before (train 816 F3)."""
    placement = shard_placement(lhs, matrices, paths)
    if placement is None:
        return None
    split, gathered, rows = placement
    matrix = jax.ShapeDtypeStruct((lhs.shape[-1], sum(each.shape[1] for each in matrices)), matrices[0].dtype)
    product = declared(rows.dtype, precision, outputs=(out_dtype,), operands=(rows.dtype, matrix.dtype))
    taken, _ = _decided(rows, matrix, jnp.dtype(out_dtype), product, whole=True,
                        target=generation.gpu_target(), choices=_cache.KERNEL_CHOICES.value,
                        pins=_cache.KERNEL_PINS.value)
    return (split, gathered) if taken else None


def shard_placement(lhs: jax.Array, matrices: tuple[jax.Array, ...], paths: tuple
                    ) -> tuple[jax.sharding.PartitionSpec, dict, jax.ShapeDtypeStruct] | None:
    """How products of `lhs` by `matrices` (each the parameter at its variables `path`) run per shard where
    they do, whichever launch takes them (`mapped_shards`' for one projection, `dew.nn.gated_mlp`'s for a
    fused MLP's three): the rows as `rows_split` splits them, each matrix where the traced program's layout
    places it (`placing`, `Layout.leaf_spec`), by its index among the map's operands after the rows, gathered
    whole in the map (`on_shards`' `gathered`), and the rows each shard holds, flattened to a matrix's;
    None where no rows split alone or no layout is traced under."""
    split, layout = rows_split(lhs), placed_layout()
    if split is None or layout is None:
        return None
    mesh = jax.sharding.get_abstract_mesh()
    held = shard_shapes((lhs,), (split,))[0]
    rows = jax.ShapeDtypeStruct((math.prod(held.shape[:-1]), held.shape[-1]), lhs.dtype)
    return split, {1 + index: layout.leaf_spec(mesh, path, each)
                   for index, (path, each) in enumerate(zip(paths, matrices, strict=True))}, rows


def kernel_path(layer) -> tuple:
    """The variables path of `layer`'s kernel (`('params', *its scope, 'kernel')`), as the layout reads a
    parameter's (`Layout.leaf_spec`)."""
    return tuple(jax.tree_util.DictKey(name) for name in ('params', *layer.scope.path, 'kernel'))


def extents(lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...], dimension_numbers) -> tuple[int, int]:
    """A layer's product's depth and width as matrices (`jax.lax.dot_general`'s `dimension_numbers` over
    operands of these shapes), the extents a recipe reads (`recipe_refusal`), `recipe_general`'s and a
    wrapped product's refusal's alike (`dew.nn.dense`): the depth the product of the left operand's
    contracting axes, the width the product of the kernel's free axes, neither contracting nor batch."""
    (contracted, against), (_, batched_rhs) = (tuple(map(tuple, pair)) for pair in dimension_numbers)
    return (math.prod(lhs_shape[axis] for axis in contracted),
            math.prod(rhs_shape[axis] for axis in range(len(rhs_shape)) if axis not in against + batched_rhs))


def recipe_general(lhs: jax.Array, rhs: jax.Array, dimension_numbers, out_dtype) -> jax.Array | None:
    """A layer's product (`jax.lax.dot_general`'s `dimension_numbers`) in the run's low-precision recipe
    (`recipe_projection`), as matrices (`grouped_rows`): a dense projection's free rows against its kernel's
    contracted axes, a batched one's (a grouped output projection's, DeepSeek V4's `o_a_proj`) each batch
    member a group of rows against its own matrix, the result in `dot_general`'s order; None where
    `recipe_projection` takes none of it, and where the run trains in no recipe before any operand is laid
    out as matrices: an operand an explicit mesh axis splits takes no reshape that does not state where its
    rows lie."""
    from dew.nn.precision import recipe_role

    if recipe_role() is None:
        return None
    rows, matrices, count, shape = grouped_rows(lhs, rhs, dimension_numbers)
    groups = matrices.shape[0]
    batched = bool(dimension_numbers[1][0])
    out = (recipe_projection(rows, matrices, jnp.full((groups,), count, jnp.int32), out_dtype=out_dtype)
           if batched else recipe_projection(rows, matrices[0], out_dtype=out_dtype))
    return None if out is None else out.reshape(shape)


def grouped_rows(lhs: jax.Array, rhs: jax.Array, dimension_numbers
                 ) -> tuple[jax.Array, jax.Array, int, tuple]:
    """A layer's product (`jax.lax.dot_general`'s `dimension_numbers`) laid out as a grouped GEMM's operands:
    its rows `[groups * count, depth]` (each batch member's free rows, `count` a member, in turn), its
    matrices `[groups, depth, width]` (a member's kernel each; one group where it batches none), the rows a
    group, and the result's shape in `dot_general`'s order (batch, `lhs`'s free axes, `rhs`'s): Appendix C
    row 4's batch axes as uniform groups, the one layout the recipe's and the template's products read."""
    (contracted, against), (batched, batched_rhs) = (tuple(map(tuple, pair)) for pair in dimension_numbers)
    free = [axis for axis in range(lhs.ndim) if axis not in contracted + batched]
    free_rhs = [axis for axis in range(rhs.ndim) if axis not in against + batched_rhs]
    groups = math.prod(lhs.shape[axis] for axis in batched)
    count = math.prod(lhs.shape[axis] for axis in free)
    depth, width = extents(lhs.shape, rhs.shape, dimension_numbers)
    rows = jnp.transpose(lhs, (*batched, *free, *contracted)).reshape(groups * count, depth)
    matrices = jnp.transpose(rhs, (*batched_rhs, *against, *free_rhs)).reshape(groups, depth, width)
    shape = (*(lhs.shape[axis] for axis in (*batched, *free)), *(rhs.shape[axis] for axis in free_rhs))
    return rows, matrices, count, shape


def launched(rows: jax.Array, matrix: jax.Array, out_dtype, precision: PrecisionLike) -> bool:
    """Whether a launch takes any of the product `rows @ matrix`'s products on the call's target
    (`grouped_matmul.GROUPED`), `matrix` one `[depth, width]` (a dense product) or `[groups, depth, width]`
    (a batched one's uniform groups of rows, `grouped_rows`): its forward, its input gradient or its kernel
    gradient, as `gemm` and its reverse mode make them, each read without its report (`selection.selected`).
    Where none does, the forward's route is reported, as XLA runs it. Decided once a key (`_decided`), its
    routes reported again at every call: a model's projections ask the same question at every layer's
    trace (row 3213: a warm start spent 10.9 of its 75.6 seconds selecting)."""
    rows_type, matrix_type = (jax.ShapeDtypeStruct(each.shape, each.dtype) for each in (rows, matrix))
    whole = seen_whole(rows) and seen_whole(matrix)
    return launches(rows_type, matrix_type, out_dtype, precision, whole=whole)


def launches(rows: jax.ShapeDtypeStruct, matrix: jax.ShapeDtypeStruct, out_dtype, precision: PrecisionLike, *,
             whole: bool) -> bool:
    """`launched` of a product's operands by their shapes and dtypes, where they are seen `whole`
    (`seen_whole`): a product formed later in the call (a fused MLP's down product, `dew.nn.gated_mlp`)."""
    product = declared(rows.dtype, precision, outputs=(out_dtype,), operands=(rows.dtype, matrix.dtype))
    taken, found = _decided(rows, matrix, jnp.dtype(out_dtype), product, whole, generation.gpu_target(),
                            _cache.KERNEL_CHOICES.value, _cache.KERNEL_PINS.value)
    for route in found:
        generation.route(route.choice, kernel=route.kernel, preferred=route.preferred, refusal=route.refusal,
                         forced=route.forced)
    return taken


@functools.lru_cache(maxsize=1024)
def _decided(rows: jax.ShapeDtypeStruct, matrix: jax.ShapeDtypeStruct, out_dtype: jnp.dtype, product,
             whole: bool, target: generation.GpuTarget, choices: tuple, pins: tuple
             ) -> tuple[bool, tuple[generation.Route, ...]]:
    """`launched`'s decision and the routes it found, once a key, the most recent keys kept: the operands'
    shapes and dtypes, the product, whether each is seen whole, and the target and the kernels a block names
    or pins, which the decision reads from its context, so they key it alone. Its routes are collected apart
    (`generation.routes`' `alone`), `launched` reporting them at every call."""
    del target, choices, pins
    with generation.routes(alone=True) as found:
        taken = _launched_once(rows, matrix, out_dtype, product, whole)
    return taken, tuple(found)


def _launched_once(rows: jax.ShapeDtypeStruct, matrix: jax.ShapeDtypeStruct, out_dtype, product,
                   whole: bool) -> bool:
    """`launched`'s decision, its routes reported."""
    width = matrix.shape[-1]
    abstract = functools.partial(jax.ShapeDtypeStruct, dtype=rows.dtype)
    # Dense as `gemm` runs it: a `[depth, width]` matrix, never a stack of one group (row 9707).
    dense = matrix.ndim == 2
    matrices = abstract(matrix.shape if matrix.ndim == 3 else (1, *matrix.shape))
    sizes = jax.ShapeDtypeStruct((matrices.shape[0],), jnp.int32)
    calls = (grouped_matmul.gmm_call(rows, matrices, sizes, out_dtype=out_dtype, trans_rhs=False,
                                     interpret=False, promoted=product.promoted, dense=dense),
             grouped_matmul.gmm_call(abstract((rows.shape[0], width)), matrices, sizes, out_dtype=rows.dtype,
                                     trans_rhs=True, interpret=False, promoted=product.promoted, dense=dense),
             grouped_matmul.tgmm_call(rows, abstract((rows.shape[0], width)), sizes, out_dtype=matrix.dtype,
                                      interpret=False, promoted=product.promoted, dense=dense))
    cases = [grouped_matmul.grouped_case(*call, product=product, whole=whole) for call in calls]
    if any(selection.selected(grouped_matmul.GROUPED, case).name != 'xla' for case in cases):
        return True
    # No launch takes any of them: the product is XLA's dot, its forward reported here; its gradients are
    # XLA's own where a reverse pass runs (a decode's trace derives no plan for one it never takes).
    selection.select(grouped_matmul.GROUPED, cases[0])
    return False
