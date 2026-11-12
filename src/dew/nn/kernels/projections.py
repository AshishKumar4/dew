"""A layer's projections through the GEMM template (`dot_general`, as flax's `Dense` takes a product): its
matrix product in the run's low-precision recipe where its role takes it (`recipe_projection`: the blockwise
FP8, MXFP8 and NVFP4 recipes, each quantizing its operands and differentiated by its own products, a surrogate
of the quantized product's derivative), else through the template's dense product (`gemm.gemm`) where a
launch takes any of its products, else XLA's own program."""

import functools
import math

import jax
import jax.numpy as jnp
from flax.typing import PrecisionLike

from dew import _cache
from dew.nn.derivatives import Surrogate, contracted, declared_derivative
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
from dew.nn.kernels.templates.product import SCALE_DEPTH, declared
from dew.nn.precision import projected
from dew.nn.sharding import explicit_spec, seen_whole


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


def recipe_projection(rows: jax.Array, matrix: jax.Array, group_sizes: jax.Array | None = None, *,
                      out_dtype, bias: jax.Array | None = None) -> jax.Array | None:
    """`rows @ matrix` in the run's low-precision recipe (`dew.nn.precision.recipe_role`), where the
    projection's role takes it: a dense projection's (`matrix` `[in, out]`, every row its one group's) or
    experts' (`[groups, in, out]`, `group_sizes` counting each group's leading rows, any past their sum
    outside every group), each group's rows padded with zero rows to whole 128-row blocks, multiplied by
    `blockwise_fp8_projection` and the padding dropped, a row outside every group zero; `bias`, each group's
    `[groups, out]`, added to its rows' fp32 sums, the output rounded once after. None where the run trains
    in no recipe or the recipe does not take the projection: its role keeps the run's dtypes (a router's, a
    head's) or its widths hold no whole 128-wide blocks, each such projection's route reported with the
    reason."""
    from dew.nn.precision import recipe_role

    held = recipe_role()
    if held is None:
        return None
    recipe, role = held
    edge = SCALE_DEPTH
    depth, width = matrix.shape[-2:]
    refusal = (None if role in recipe.roles else
               selection.Refusal("kept_role", f"a {role or 'role-less'} projection keeps the run's dtypes "
                                 f"under {recipe.recipe}", kind="handoff"))
    if refusal is None and (depth % edge or width % edge):
        refusal = selection.Refusal("block_shape", f"a {depth} x {width} projection holds no whole "
                                    f"{edge}-wide blocks", kind="handoff")
    if refusal is not None:
        generation.route(generation.Choice.BLOCKWISE_FP8, kernel="unquantized", preferred=recipe.recipe,
                         refusal=refusal)
        return None
    if (placed := explicit_spec(rows)) is not None and placed.spec[0] is not None:
        return _projected_apart(rows, matrix, group_sizes, out_dtype=out_dtype, bias=bias, placed=placed)
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
    laid = jnp.zeros((padded, depth), rows.dtype).at[place].set(rows, mode="drop")
    # The run's slack, zero rows, as the last group's further blocks: the recipe's rows are whole blocks.
    group_blocks = blocks.at[-1].add(padded // edge - blocks.sum())
    # A biased product's sums stay fp32 through the bias, as the template's epilogue adds it (moe's
    # `_biased`): one rounding of the whole.
    summed_dtype = out_dtype if bias is None else jnp.float32
    product = blockwise_fp8_projection(laid, stacked, group_blocks, out_dtype=summed_dtype)
    out = product.at[place].get(mode="fill", fill_value=0)
    if bias is not None:
        out = jnp.where((group < groups)[:, None], out + bias[held_group].astype(jnp.float32), out)
    return out.astype(out_dtype)


def _projected_apart(rows, matrix, group_sizes, *, out_dtype, bias, placed) -> jax.Array:
    """`recipe_projection` of rows an Explicit mesh axis splits (rows 4854, 4859): each device's rows its own
    projection (`jax.shard_map` over the axes splitting the rows), laid in whole 128-row blocks of its own
    as a data-parallel recipe quantizes them, the matrices and bias whole on every device and their
    gradients summed over the devices (shard_map's transpose). Experts' sizes count every device's rows
    together, each group's after the last's: a device's groups are its share of each (`_local_sizes`)."""
    from dew.mesh import mesh_axes

    names = mesh_axes(placed.spec[0])
    split = jax.sharding.PartitionSpec(placed.spec[0], None)
    whole = jax.sharding.PartitionSpec()
    extra = tuple(each for each in (group_sizes, bias) if each is not None)

    def projected(rows, matrix, *extra):
        sizes = None if group_sizes is None else _local_sizes(extra[0], rows.shape[0], names)
        held = extra[1:] if group_sizes is not None else extra
        out = recipe_projection(rows, matrix, sizes, out_dtype=out_dtype, bias=held[0] if held else None)
        assert out is not None, "the recipe takes the projection on every device as on the whole"
        return out

    # The recipe's own loops start their sums typed for no device (`grouped_matmul.step_scaled`'s carry), so
    # the program is not checked for the devices' varying values (`check_vma`); each device's rows are its
    # own by the specs, and the whole operands' gradients sum over the devices as shard_map transposes them.
    return jax.shard_map(projected, mesh=placed.mesh, in_specs=(split, whole, *(whole,) * len(extra)),
                         out_specs=split, axis_names=frozenset(names), check_vma=False)(rows, matrix, *extra)


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
    return d_x, d_w, None

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
    return d_x, d_w, None

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
    return d_x, d_w, None

# The NVFP4 recipe's gradient (`nvfp4_projection`): a derivative of Dew's choosing (`Surrogate`).
_nvfp4_projection = contracted("test_an_nvfp4_projection_holds_every_reverse_order")(
    declared_derivative(Surrogate(), _nvfp4_value, _nvfp4_forward, _nvfp4_backward, nondiff_argnums=(3, 4)))


def dot_general(xla):
    """`xla`, a layer's product (a `jax.lax.dot_general`, as flax's `Dense` takes one), in the run's
    low-precision recipe where its role takes the matrix product (`recipe_projection`), else through the
    template's dense product (`gemm` of one group) where the call is a matrix product (`lhs`'s last axis
    against a matrix's first, no batch axes, both of one dtype) and a launch takes any of its products: the
    forward, the input gradient or the kernel gradient (`_launched`). A call no launch takes is `xla`'s own
    program, differentiated by JAX, each of its products' routes reported with the reason no launch takes
    it (a declared cell's old route, a mesh axis splitting the operands, a CPU), so a layer runs as it ran
    before wherever the template declares XLA faster; every other product is `xla`'s."""
    def product(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None, **kwargs):
        (contracted, against), batch = dimension_numbers
        out_dtype = jnp.dtype(preferred_element_type or lhs.dtype)
        # The run's low-precision recipe takes the product where the projection's role takes it.
        if not kwargs and (recipe := _recipe_general(lhs, rhs, dimension_numbers, out_dtype)) is not None:
            return recipe
        plain = (not kwargs and jnp.ndim(rhs) == 2 and lhs.dtype == rhs.dtype
                 and tuple(contracted) == (lhs.ndim - 1,) and tuple(against) == (0,)
                 and tuple(map(tuple, batch)) == ((), ()))
        if plain:
            rows = lhs.reshape(-1, lhs.shape[-1])
            if _launched(rows, rhs, out_dtype, precision):
                product = gemm(rows, rhs, out_dtype=out_dtype, precision=precision)
                return product.reshape(*lhs.shape[:-1], rhs.shape[1])
        # A CPU sums a layer's float32 projection as XLA's own dot does (row 5448).
        return projected(xla)(lhs, rhs, dimension_numbers, precision=precision,
                              preferred_element_type=preferred_element_type, **kwargs)

    return product


def _recipe_general(lhs: jax.Array, rhs: jax.Array, dimension_numbers, out_dtype) -> jax.Array | None:
    """A layer's product (`jax.lax.dot_general`'s `dimension_numbers`) in the run's low-precision recipe
    (`recipe_projection`), as matrices: a dense projection's free rows against its kernel's contracted
    axes, a batched one's (a grouped output projection's, DeepSeek V4's `o_a_proj`) each batch member a
    group of rows against its own matrix, the result in `dot_general`'s order (batch, `lhs`'s free axes,
    `rhs`'s); None where `recipe_projection` takes none of it, and where the run trains in no recipe before
    any operand is laid out as matrices: an operand an explicit mesh axis splits takes no reshape that does
    not state where its rows lie."""
    from dew.nn.precision import recipe_role

    if recipe_role() is None:
        return None
    (contracted, against), (batched, batched_rhs) = (tuple(map(tuple, pair)) for pair in dimension_numbers)
    free = [axis for axis in range(lhs.ndim) if axis not in contracted + batched]
    free_rhs = [axis for axis in range(rhs.ndim) if axis not in against + batched_rhs]
    groups = math.prod(lhs.shape[axis] for axis in batched)
    count = math.prod(lhs.shape[axis] for axis in free)
    depth = math.prod(lhs.shape[axis] for axis in contracted)
    width = math.prod(rhs.shape[axis] for axis in free_rhs)
    rows = jnp.transpose(lhs, (*batched, *free, *contracted)).reshape(groups * count, depth)
    matrices = jnp.transpose(rhs, (*batched_rhs, *against, *free_rhs)).reshape(groups, depth, width)
    out = (recipe_projection(rows, matrices, jnp.full((groups,), count, jnp.int32), out_dtype=out_dtype)
           if batched else recipe_projection(rows, matrices[0], out_dtype=out_dtype))
    if out is None:
        return None
    return out.reshape(*(lhs.shape[axis] for axis in (*batched, *free)),
                       *(rhs.shape[axis] for axis in free_rhs))


def _launched(rows: jax.Array, matrix: jax.Array, out_dtype, precision: PrecisionLike) -> bool:
    """Whether a launch takes any of the dense product `rows @ matrix`'s products on the call's target
    (`grouped_matmul.GROUPED`): its forward, its input gradient or its kernel gradient, as `gemm` and
    its reverse mode make them, each read without its report (`selection.selected`). Where none does, the
    forward's route is reported, as XLA runs it. Decided
    once a key (`_DECIDED`: the operands' shapes and dtypes, the product, the target and the kernels a
    block names or pins), its routes reported again at every call: a model's projections ask the same
    question at every layer's trace (row 3213: a warm start spent 10.9 of its 75.6 seconds selecting)."""
    product = declared(rows.dtype, precision, outputs=(out_dtype,), operands=(rows.dtype, matrix.dtype))
    whole = seen_whole(rows) and seen_whole(matrix)
    key = (rows.shape, jnp.dtype(rows.dtype), matrix.shape, jnp.dtype(matrix.dtype), jnp.dtype(out_dtype),
           repr(precision), whole, product, generation.gpu_target(), _cache.KERNEL_CHOICES.value,
           _cache.KERNEL_PINS.value)
    if (decided := _DECIDED.get(key)) is None:
        with generation.routes() as found:
            taken = _launched_once(rows, matrix, out_dtype, product, whole)
        decided = _DECIDED[key] = (taken, tuple(found))
    else:
        for route in decided[1]:
            generation.route(route.choice, kernel=route.kernel, preferred=route.preferred,
                             refusal=route.refusal, forced=route.forced)
    return decided[0]

_DECIDED: dict[tuple, tuple[bool, tuple]] = {}
"""`_launched`'s decisions by their key, and the routes each reported."""


def _launched_once(rows: jax.Array, matrix: jax.Array, out_dtype, product, whole: bool) -> bool:
    """`_launched`'s decision, its routes reported."""
    width = matrix.shape[1]
    abstract = functools.partial(jax.ShapeDtypeStruct, dtype=rows.dtype)
    matrices, sizes = abstract((1, *matrix.shape)), jax.ShapeDtypeStruct((1,), jnp.int32)
    calls = (grouped_matmul.gmm_call(rows, matrices, sizes, out_dtype=out_dtype, trans_rhs=False,
                                     interpret=False, promoted=product.promoted, dense=True),
             grouped_matmul.gmm_call(abstract((rows.shape[0], width)), matrices, sizes, out_dtype=rows.dtype,
                                     trans_rhs=True, interpret=False, promoted=product.promoted, dense=True),
             grouped_matmul.tgmm_call(rows, abstract((rows.shape[0], width)), sizes, out_dtype=matrix.dtype,
                                      interpret=False, promoted=product.promoted, dense=True))
    cases = [grouped_matmul.grouped_case(*call, product=product, whole=whole) for call in calls]
    if any(selection.selected(grouped_matmul.GROUPED, case).name != 'xla' for case in cases):
        return True
    # No launch takes any of them: the product is XLA's dot, its forward reported here; its gradients are
    # XLA's own where a reverse pass runs (a decode's trace derives no plan for one it never takes).
    selection.select(grouped_matmul.GROUPED, cases[0])
    return False
