"""A layer's projections through the GEMM template (`dot_general`, as flax's `Dense` takes a product): its
matrix product in the run's low-precision recipe where its role takes it (`recipe_projection`: the blockwise
FP8, MXFP8 and NVFP4 recipes, each quantizing its operands and differentiated by its own products, a surrogate
of the quantized product's derivative), else through the template's dense product (`gemm.gemm`) where a
launch takes any of its products, else XLA's own program."""

import collections
import functools
import math
from collections.abc import Callable, Hashable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax.typing import PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from dew import _cache
from dew._facts import Held
from dew.mesh import mesh_axes, spec_axes
from dew.nn.derivatives import (
    Protected,
    Surrogate,
    contracted,
    declared_derivative,
    marked,
    portable_formed,
    with_gradients_formed,
)
from dew.nn.kernels import generation, grouped_matmul, selection
from dew.nn.kernels.formats import MxFp8, NvFp4Block
from dew.nn.kernels.gemm import (
    blockwise_fp8_finished,
    blockwise_fp8_kernel_gradient,
    blockwise_fp8_product,
    gemm,
    gemm_t,
    mxfp8_product,
    nvfp4_block_finished,
    nvfp4_block_product,
)
from dew.nn.kernels.templates.hooks import Hook, Side
from dew.nn.kernels.templates.product import MXFP8 as MX_BLOCKS, NVFP4 as NVFP4_BLOCKS, SCALE_DEPTH, declared
from dew.nn.precision import EXACT_FLOAT32, biased, biased_value
from dew.nn.sharding import (
    RESIDUAL,
    automatic_axes,
    explicit_spec,
    logical_spec,
    on_shards,
    placed_layout,
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
                      out_dtype, bias: jax.Array | None = None,
                      tensor_scale: jax.Array | None = None) -> jax.Array | None:
    """`rows @ matrix` in the run's low-precision recipe (`dew.nn.precision.recipe_role`), where the
    projection's role takes it: a dense projection's (`matrix` `[in, out]`, every row its one group's) or
    experts' (`[groups, in, out]`, `group_sizes` counting each group's leading rows, any past their sum
    outside every group), each group's rows padded with zero rows to whole 128-row blocks, multiplied by
    the recipe's projection (`RECIPES`) and the padding dropped, a row outside every group zero; `bias`, each
    group's `[groups, out]`, added to its rows' fp32 sums, the output rounded once after. None where the run
    trains in no recipe or the recipe does not take the projection: its role keeps the run's dtypes (a
    router's, a head's) or its widths hold no whole blocks of the recipe's, each such projection's route
    reported with the reason. `tensor_scale`, the rows' NVFP4 tensor scale under the run's delayed scaling
    (`LowPrecision.tensor_scaling`, `delayed_tensor_scale` of the run's amax history), which it requires
    and no other recipe takes (row 12721)."""
    from dew.nn.precision import recipe_role

    held, refusal = recipe_role(), recipe_refusal(*matrix.shape[-2:])
    if held is None or refusal is not None:
        if held is not None:  # a run's recipe that leaves this projection: its route reported
            generation.route(RECIPES[held[0].recipe].choice, kernel="unquantized", preferred=held[0].recipe,
                             refusal=refusal)
        return None
    recipe = held[0]
    if (recipe.tensor_scaling == "delayed") != (tensor_scale is not None):
        raise ValueError("a delayed-scaling NVFP4 recipe's projection takes its rows' tensor scale, and only "
                         "it does (`LowPrecision.tensor_scaling`)")
    taken = RECIPES[recipe.recipe]
    edge = SCALE_DEPTH
    # Rows an Explicit axis splits, or the mesh's automatic axes alone (a data-parallel run's, row 6719):
    # each device's rows its own projection; a launch reads its operands whole.
    if (placed := explicit_spec(rows)) is not None and placed.spec[0] is not None:
        along = placed.spec[0]
    else:
        along = None if (split := rows_split(rows)) is None else split[0]
    if along is not None:
        return _projected_apart(rows, matrix, group_sizes, out_dtype=out_dtype, bias=bias, along=along,
                                tensor_scale=tensor_scale)
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
    scaled = {} if tensor_scale is None else {"tensor_scale": jnp.asarray(tensor_scale, jnp.float32)}
    product = taken.projection(laid, stacked, group_blocks, out_dtype=summed_dtype, **scaled)
    out = product.at[place].get(mode="fill", fill_value=0)
    if bias is not None:
        out = jnp.where((group < groups)[:, None], out + bias[held_group].astype(jnp.float32), out)
    return out.astype(out_dtype)


def recipe_chain(rows: jax.Array, up: jax.Array, down: jax.Array, activation: Callable, *, out_dtype,
                 gate: jax.Array | None = None, tensor_scales: tuple | None = None,
                 names: tuple[str, str] | None = None) -> jax.Array | None:
    """`recipe_projection(activation(recipe_projection(rows, up)), down)` of an ungated MLP's rows (`[rows,
    in]`, any count; `up` `[in, hidden]`, `down` `[hidden, out]`, dense) as one chain where the run trains in
    DeepSeek-V3's blockwise FP8 recipe, or in NVFP4's under delayed tensor scaling, and its role takes both
    projections (`recipe_refusal`): the rows laid in whole 128-row blocks, zero rows after them dropped after
    (each row's quantization its own, so the real rows are the two apart's, bitwise), the down projection's
    rows quantized in the up projection's epilogue (`blockwise_chain`, row 8547; `nvfp4_chain`, row 12721);
    rows a mesh splits each device's own chain, the matrices whole and their gradients summed (`on_shards`,
    as `_projected_apart`). With `gate` (`up`'s shape), a gated MLP's: `activation(gate_rows, up_rows)`, the
    up projection's epilogue taking it (row 12689). `tensor_scales`, the rows' and the hidden activation's
    NVFP4 tensor scales under the run's delayed scaling (`delayed_tensor_scale` of the run's histories), as
    `recipe_projection`'s `tensor_scale`: the chain's there, refused under every other recipe. None
    elsewhere: the run's recipe another's (MXFP8's, current NVFP4's: an activation's own tensor scale is known
    only once the activation is) or none, a delayed recipe's chain given no scales, no rows (each projection
    apart lays one zero block), or a projection its role keeps, which the `recipe_projection` calls then take
    apart. Each product's route is its selector's (`BLOCKWISE_FP8`, `NVFP4_BLOCK_OP`). `names`, the
    checkpoint names of the gate's and the up projection's rounded products (`_chain_forward`), which a remat
    policy reads, as `GatedMLP` names them apart (row 12758)."""
    from dew.nn.precision import recipe_role

    held = recipe_role()
    if (held is None or rows.shape[0] == 0 or up.ndim != 2 or down.ndim != 2
            or recipe_refusal(*up.shape) is not None or recipe_refusal(*down.shape) is not None
            or (gate is not None and gate.shape != up.shape)):
        return None
    recipe = held[0]
    if tensor_scales is not None and recipe.tensor_scaling != "delayed":
        raise ValueError("a chain's tensor scales are a delayed-scaling NVFP4 recipe's alone "
                         "(`LowPrecision.tensor_scaling`)")
    # A current-scaled NVFP4 recipe's hidden activation's tensor scale is its own, known once it is whole;
    # a delayed one's chain without its scales a caller's probe its projections apart then take, each
    # given its own.
    chained = recipe.recipe
    if chained not in _CHAINS or (chained == "nvfp4" and tensor_scales is None):
        return None
    if (placed := explicit_spec(rows)) is not None and placed.spec[0] is not None:
        along = placed.spec[0]
    else:
        along = None if (split := rows_split(rows)) is None else split[0]
    if along is not None:
        def each(rows, up, down, gate, *scales):
            out = recipe_chain(rows, up, down, activation, out_dtype=out_dtype, gate=gate,
                               tensor_scales=None if tensor_scales is None else scales, names=names)
            assert out is not None, "the recipe takes the chain on every device as on the whole"
            return out

        return on_shards(each, jax.sharding.PartitionSpec(along, None), (rows, up, down, gate,
                                                                          *(tensor_scales or ())),
                         whole=(1, 2, 3, 4, 5))
    count = rows.shape[0]
    laid = jnp.pad(rows, ((0, -count % SCALE_DEPTH), (0, 0)))
    # Its padding zero rows the backward keeps out of every gradient (`_chain_pulled`'s live rows).
    return _chained(chained, laid, up[None], down[None], activation, out_dtype=out_dtype,
                    gate=None if gate is None else gate[None], live=count, interpret=False,
                    tensor_scales=tensor_scales, names=names)[:count]


def _projected_apart(rows, matrix, group_sizes, *, out_dtype, bias, along, tensor_scale=None) -> jax.Array:
    """`recipe_projection` of rows the mesh's axes `along` split, Explicit (rows 4854, 4859) or automatic
    alone (row 6719): each device's rows its own projection (`on_shards`, whose map is unchecked as a Pallas
    launch's must be), laid in whole 128-row blocks of its own as a data-parallel recipe quantizes them, the
    matrix, the groups' sizes and the bias whole on every device and the matrix's and bias's gradients
    summed over the devices (the map's transpose). Experts' sizes count every device's rows together, each
    group's after the last's: a device's groups are its share of each (`_local_sizes`). A delayed recipe's
    `tensor_scale` whole on every device too: the run's one scale, every device's rows quantized under it."""

    names = mesh_axes(along)

    def projected(rows, matrix, group_sizes, bias, tensor_scale):
        sizes = None if group_sizes is None else _local_sizes(group_sizes, rows.shape[0], names)
        out = recipe_projection(rows, matrix, sizes, out_dtype=out_dtype, bias=bias,
                                tensor_scale=tensor_scale)
        assert out is not None, "the recipe takes the projection on every device as on the whole"
        return out

    return on_shards(projected, jax.sharding.PartitionSpec(along, None),
                     (rows, matrix, group_sizes, bias, tensor_scale), whole=(1, 2, 3, 4))


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


def _k_major(w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`w`'s blockwise FP8 codes, K-major (`[groups, width, depth]`, as a step-scaled product streams its
    matrix), and their 128 x 128 blocks' scales (`blocks_quantized`)."""
    blocks, scales = blocks_quantized(w)
    return jnp.swapaxes(blocks, 1, 2), scales


def _quantized_rows_product(rows, rows_scales, w, sizes, out_dtype, interpret) -> jax.Array:
    """A blockwise FP8 projection of rows already quantized (E4M3 codes and their 1 x 128 scales, `[depth /
    128, rows]`) by `w`: the one statement a projection, a chain's gate and its down projection read."""
    blocks, scales = _k_major(w)
    return blockwise_fp8_product(rows, blocks, sizes, rows_scales, scales, out_dtype=out_dtype,
                                 interpret_on_cpu=interpret)


def _projection_value(x, w, group_blocks, out_dtype, interpret):
    rows, rows_scales = rows_quantized(x)
    sizes = (group_blocks * SCALE_DEPTH).astype(jnp.int32)
    return _quantized_rows_product(rows, rows_scales.T, w, sizes, out_dtype, interpret)


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


def _chain_value(x, w_ins, w_down, group_blocks, tensor_scales, activation, out_dtype, interpret):
    """`blockwise_chain`'s value and its first projections' rounded products: an ungated MLP's up, or a gated
    one's gate and up (the gate's its side tile, row 12689), the up projection's epilogue quantizing their
    activation as the down projection's rows (`quantized.blockwise_fp8_epilogue`: codes and 1 x 128 scales,
    `[width / 128, rows]`, as `blockwise_fp8_product` reads its rows' scales). No tensor scales: each
    tile's is its own."""
    from dew.nn.kernels.templates.gemm import quantized

    del tensor_scales

    sizes = (group_blocks * SCALE_DEPTH).astype(jnp.int32)
    rows, rows_scales = rows_quantized(x)
    *gate, w_up = w_ins
    up, up_scales = _k_major(w_up)
    # A gate's product stored rounded, the up product's epilogue reading it as a tile beside its own sum:
    # one accumulator a launch (two exact sums, each a running total and a step's partial, take a Hopper
    # consumer's 336 registers at FP8's least 128 columns; row 12689).
    gated = tuple(_quantized_rows_product(rows, rows_scales.T, matrix, sizes, out_dtype, interpret)
                  for matrix in gate)
    product, codes, scales = blockwise_fp8_finished(
        rows, up, sizes, rows_scales.T, up_scales,
        epilogue=Hook(quantized.blockwise_fp8_epilogue(activation, out_dtype),
                      tuple(Side(gate, ('rows', 'columns')) for gate in gated)),
        out_dtypes=(out_dtype, jnp.float8_e4m3fn, jnp.float32),
        out_forms=(None, None, (SCALE_DEPTH, True)), dense=True, interpret_on_cpu=interpret)
    products = (*gated, product)
    return _quantized_rows_product(codes, scales, w_down, sizes, out_dtype, interpret), tuple(products)


def _chain(recipe, x, w_ins, w_down, group_blocks, tensor_scales, activation, out_dtype, interpret, live,
           names):
    del live, names  # a padded row's forward is its own row's, which the caller drops; no residuals
    return _CHAINS[recipe].value(x, w_ins, w_down, group_blocks, tensor_scales, activation, out_dtype,
                                 interpret)[0]


def _chain_forward(recipe, x, w_ins, w_down, group_blocks, tensor_scales, activation, out_dtype, interpret,
                   live, names):
    del live
    out, products = _CHAINS[recipe].value(x, w_ins, w_down, group_blocks, tensor_scales, activation,
                                          out_dtype, interpret)
    if names is not None:
        # The residuals the backward reads, named as the projections apart name their outputs (`names`: the
        # gate's and the up's; an ungated chain's one product the up's), so a remat policy saving them keeps
        # the first projections and their quantizing epilogue out of the backward (row 12758).
        products = tuple(checkpoint_name(product, name)
                         for product, name in zip(products, names[-len(products):], strict=True))
    return out, (x, w_ins, w_down, group_blocks, products)


def _chain_backward(recipe, activation, out_dtype, interpret, live, names, residuals, cotangent):
    del names
    # No gradient through the group blocks or the tensor scales (constants of the step, the run's histories').
    return (*_chain_pulled(_CHAINS[recipe].backward, activation, out_dtype, interpret, live, residuals,
                           cotangent), None, None)


def _chain_pulled(backward, activation, out_dtype, interpret, live, residuals, cotangent) -> tuple:
    """A recipe chain's gradients (`blockwise_chain`'s, `nvfp4_chain`'s; the rows', the first projections'
    matrices' and the down projection's): each projection's own reverse mode (`backward`, the recipe's
    surrogate: `_projection_backward`, `_nvfp4_backward`) through the activation's derivative at the first
    projections' rounded products, as they run apart, the rows' cotangent each first projection's added.
    Rows past the first `live` (a caller's padding, `recipe_chain`) hold zeros in its cotangent, the down
    projection's rows and the first projections' cotangents, as the projections apart pad them: the down
    kernel gradient and the first ones quantize their rows along the tokens, which an activation of the
    padding (nonzero, or of a nonfinite derivative at zero) would enter."""
    x, w_ins, w_down, group_blocks, products = residuals
    kept = (jnp.arange(x.shape[0]) < live)[:, None]
    # The padding's incoming cotangent held out too, whatever the caller gives it.
    cotangent = jnp.where(kept, cotangent, jnp.zeros((), cotangent.dtype))
    activated, pulled = jax.vjp(lambda *values: activation(*values).astype(out_dtype), *products)
    activated = jnp.where(kept, activated, jnp.zeros((), activated.dtype))
    d_activated, d_w_down, *_ = backward(out_dtype, interpret, (activated, w_down, group_blocks), cotangent)
    pulled_back = [backward(out_dtype, interpret, (x, w_in, group_blocks),
                            jnp.where(kept, d_product, jnp.zeros((), d_product.dtype)))
                   for w_in, d_product in zip(w_ins, pulled(d_activated), strict=True)]
    d_x = functools.reduce(jnp.add, [d_rows for d_rows, *_ in pulled_back])
    return d_x, tuple(d_w for _, d_w, *_ in pulled_back), d_w_down


def blockwise_chain(x: jax.Array, w_up: jax.Array, w_down: jax.Array, activation, *, out_dtype,
                    gate: jax.Array | None = None, names: tuple[str, str] | None = None,
                    interpret_on_cpu: bool = False) -> jax.Array:
    """`blockwise_fp8_projection(activation(blockwise_fp8_projection(x, w_up)), w_down)` of `x`'s rows
    (`[rows, depth]`, whole 128-row blocks, one group of them; `w_up` `[1, depth, hidden]`, `w_down` `[1,
    hidden, width]`), each product rounded to `out_dtype`, the activation of the rounded up projection rounded
    again, as an ungated MLP's two projections run apart in DeepSeek-V3's blockwise FP8 recipe: the down
    projection's rows quantized in the up projection's epilogue, its codes and scales stored there
    (`quantized.blockwise_fp8_epilogue`, the scales `[hidden / 128, rows]`), no pass of their own between the
    two (row 8547). With `gate` (`w_up`'s shape), a gated MLP's: `activation(gate_product, up_product)`,
    the gate's product stored and read by the up projection's epilogue (row 12689). Its gradients bitwise
    the projections apart (`_chain_backward`: each projection's own surrogate, the activation XLA's in
    both); its value bitwise theirs where the kernel's activation computes as XLA's does (an exact one,
    ReLU's; any, interpreted), else its rows' codes those of the kernel's activation, within check 1 of
    theirs (a GH200's GELU moved 0.38% of the codes by its tanh, M-NG-21b; testing.md's kernel against the
    composition it replaces); `activation` is an elementwise function of `out_dtype` arrays, the MLP's.
    `names`, `recipe_chain`'s."""
    _one_group(x, w_up, w_down, gate)
    return _chained("blockwise_fp8", x, w_up, w_down, activation, out_dtype=out_dtype, gate=gate,
                    live=x.shape[0], interpret=interpret_on_cpu, names=names)


def nvfp4_chain(x: jax.Array, w_up: jax.Array, w_down: jax.Array, activation, *, out_dtype,
                tensor_scales: tuple, gate: jax.Array | None = None, names: tuple[str, str] | None = None,
                interpret_on_cpu: bool = False) -> jax.Array:
    """`nvfp4_projection(activation(nvfp4_projection(x, w_up, tensor_scale=s)), w_down, tensor_scale=t)` of
    `x`'s rows (`[rows, depth]`, whole 128-row blocks, one group of them; `w_up` `[1, depth, hidden]`,
    `w_down` `[1, hidden, width]`) at a delayed-scaling NVFP4 recipe's `tensor_scales` `(s, t)`, the rows'
    and the hidden activation's (`delayed_tensor_scale` of the run's histories), each product rounded to
    `out_dtype`, the activation of the rounded up projection rounded again, as an MLP's projections run
    apart: the up projection's epilogue scaling its sums by both its operands' tensor scales and quantizing
    their activation at `t` as the down projection's rows (`quantized.nvfp4_epilogue`: E2M1 codes stored
    two a byte, row 12743, and E4M3 block scales stored `[hidden / 16, rows]`), which the down projection
    reads as they are (`NvFp4Block.packed`), no quantizing or packing pass of their own between the two (row
    12721).
    With `gate` (`w_up`'s shape), a gated MLP's: `activation(gate_product, up_product)`, the gate's product
    (at `s` too) stored and read by the up projection's epilogue. Its gradients bitwise the projections
    apart at the same scales (each projection's own surrogate, `_nvfp4_backward`), its value as
    `blockwise_chain`'s is: bitwise where the kernel's activation computes as XLA's does, else within check
    1; `activation` is an elementwise function of `out_dtype` arrays, the MLP's. `names`, `recipe_chain`'s."""
    _one_group(x, w_up, w_down, gate)
    return _chained("nvfp4", x, w_up, w_down, activation, out_dtype=out_dtype, gate=gate, live=x.shape[0],
                    interpret=interpret_on_cpu, tensor_scales=tensor_scales, names=names)


def _one_group(x, w_up, w_down, gate) -> None:
    """Refuse a chain (`blockwise_chain`, `nvfp4_chain`) of more than one group, of no rows or of rows of no
    whole 128-row blocks (its next rows' scales stored transposed, each share's rows whole; an empty tile
    holds no block to quantize), or a gate not its up projection's shape."""
    if w_up.shape[0] != 1 or w_down.shape[0] != 1 or x.shape[0] == 0 or x.shape[0] % SCALE_DEPTH:
        raise ValueError("a recipe chain is one group's of one or more whole 128-row blocks: its rows' "
                         "quantized scales are stored transposed, each share's rows whole")
    if gate is not None and gate.shape != w_up.shape:
        raise ValueError(f"a gated chain's gate is its up projection's shape {w_up.shape}, not {gate.shape}")


def _chained(recipe: str, x, w_up, w_down, activation, *, out_dtype, gate, live: int, interpret: bool,
             tensor_scales: tuple | None = None, names: tuple[str, str] | None = None) -> jax.Array:
    """The `recipe` chain's declared-derivative call (`_recipe_chain`) of one group of `x`'s whole blocks,
    its first `live` rows the caller's: a public chain's, every row; `recipe_chain`'s, the rows it padded. An
    NVFP4 chain's `tensor_scales` its operands, no gradient through them; a blockwise FP8 chain's none."""
    group_blocks = jnp.full((1,), x.shape[0] // SCALE_DEPTH, jnp.int32)
    w_ins = (w_up,) if gate is None else (gate, w_up)
    from dew.algebra import bits

    # Each scale one scalar, refused before it reaches a product (`bits.nvfp4_given_scale`).
    scales = tuple(bits.nvfp4_given_scale(scale) for scale in tensor_scales or ())
    if names is not None and len(names) != 2:
        raise ValueError(f"a chain's names are its gate's and its up projection's, not {names}")
    return _recipe_chain(recipe, x, w_ins, w_down, group_blocks, scales, activation, jnp.dtype(out_dtype),
                         interpret, live, None if names is None else tuple(names))


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


def _mx_value(x, w, group_blocks, out_dtype, interpret):
    return mxfp8_product(MxFp8.quantized(x), MxFp8.quantized(jnp.swapaxes(w, 1, 2)), group_blocks,
                         out_dtype=out_dtype, interpret_on_cpu=interpret)


def _mx_forward(x, w, group_blocks, out_dtype, interpret):
    return _mx_value(x, w, group_blocks, out_dtype, interpret), (x, w, group_blocks)


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
                     tensor_scale: jax.Array | None = None, interpret_on_cpu: bool = False) -> jax.Array:
    """Each group's rows of `x` (`[rows, depth]`) times its matrix of `w` (`[groups, depth, width]`) in the
    NVFP4 recipe (row 2513: E2M1 every tensor, an E4M3 scale every 16 of each product's contraction under a
    tensor scale): `x` and `w` quantized along the depth, their product `nvfp4_block_product`. Its gradients
    are the recipe's own NVFP4 products, a surrogate of the quantized product's derivative (straight through
    each rounding), as `mxfp8_projection`'s: the input's the cotangent and `w` quantized along the width;
    the matrices' each group's rows and cotangent quantized along its tokens, their decoded values contracted
    over the group's rows in fp32 (`gemm_t`). Group `g` holds `group_blocks[g]` whole 128-row blocks.
    `tensor_scale`, `x`'s tensor scale in place of its own (a delayed-scaling recipe's, row 12721); the
    matrices' and the gradients' operands their own, each exactly as known when it is quantized."""
    return _nvfp4_projection(x, w, group_blocks, tensor_scale, jnp.dtype(out_dtype), interpret_on_cpu)


def _nvfp4_value(x, w, group_blocks, tensor_scale, out_dtype, interpret):
    return nvfp4_block_product(NvFp4Block.quantized(x, tensor_scale),
                               NvFp4Block.quantized(jnp.swapaxes(w, 1, 2)), group_blocks, out_dtype=out_dtype,
                               interpret_on_cpu=interpret)


def nvfp4_amax(x: jax.Array) -> jax.Array:
    """`x`'s largest finite magnitude in fp32 (0 for none: an empty `x`, or one all NaN or infinite), no
    gradient through it: the value a delayed-scaling NVFP4 run records in its amax history for the
    projection `x` is the rows of (row 12721), the max over devices its state's to take."""
    return jax.lax.stop_gradient(_finite_largest(jnp.abs(x.astype(jnp.float32))))


def delayed_tensor_scale(history: jax.Array) -> jax.Array:
    """The NVFP4 tensor scale of an amax `history` (its last steps' largest magnitudes): the scale of their
    largest finite one (`bits.nvfp4_tensor_scale`, 1 for none), as Transformer Engine's delayed scaling
    takes it (row 12721): an empty history, or a step's NaN or infinite record, leaves the scale the other
    steps' rather than infinite or reset to 1."""
    from dew.algebra import bits

    return bits.nvfp4_tensor_scale(_finite_largest(jnp.asarray(history, jnp.float32)))


def _finite_largest(values: jax.Array) -> jax.Array:
    """The largest of `values`' finite entries and 0: an amax's reduction, defined on none."""
    return jnp.max(jnp.where(jnp.isfinite(values), values, jnp.float32(0)), initial=jnp.float32(0))


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


def _nvfp4_forward(x, w, group_blocks, tensor_scale, out_dtype, interpret):
    return _nvfp4_value(x, w, group_blocks, tensor_scale, out_dtype, interpret), (x, w, group_blocks)


def _nvfp4_backward(out_dtype, interpret, residuals, cotangent):
    del out_dtype
    x, w, group_blocks = residuals
    d_x = nvfp4_block_product(NvFp4Block.quantized(cotangent), NvFp4Block.quantized(w), group_blocks,
                              out_dtype=x.dtype, interpret_on_cpu=interpret)
    sizes = (group_blocks * SCALE_DEPTH).astype(jnp.int32)
    # Each group's tokens quantized along the tokens (the kernel gradient's contraction), every 16 a scale,
    # the tensor scales applied: a 128-row group's blocks hold whole 16s.
    rows, cotangents = (NvFp4Block.quantized(operand.T) for operand in (x, cotangent))
    # An FP4 backward's exact fp32 products: the decoded codes times their scales, held whole.
    d_w = gemm_t((rows.decoded() * rows.scale).T.astype(jnp.float32),
                 (cotangents.decoded() * cotangents.scale).T.astype(jnp.float32), sizes, out_dtype=w.dtype,
                 precision=EXACT_FLOAT32, interpret_on_cpu=interpret)
    return (*with_gradients_formed(d_x, d_w), None, None)

# The NVFP4 recipe's gradient (`nvfp4_projection`): a derivative of Dew's choosing (`Surrogate`); its tensor
# scale a constant of the step (a delayed recipe's, its history's), no gradient through it.
_nvfp4_projection = contracted("test_an_nvfp4_projection_holds_every_reverse_order")(
    declared_derivative(Surrogate(), _nvfp4_value, _nvfp4_forward, _nvfp4_backward, nondiff_argnums=(4, 5)))


def _nvfp4_chain_value(x, w_ins, w_down, group_blocks, tensor_scales, activation, out_dtype, interpret):
    """`nvfp4_chain`'s value and its first projections' rounded products: an ungated MLP's up, or a gated
    one's gate and up (the gate's its side tile), each of the rows at their delayed tensor scale, the up
    projection's epilogue (`gemm.nvfp4_block_finished`: its sums times both tensor scales) quantizing their
    activation at the hidden activation's (`quantized.nvfp4_epilogue`: E2M1 codes stored two a byte, the
    form's `packed`, row 12743, and E4M3 block scales `[width / 16, rows]`), read by the down projection as
    they are stored (`NvFp4Block.packed`)."""
    from dew.nn.kernels.templates.gemm import quantized

    rows_scale, hidden_scale = tensor_scales
    rows = NvFp4Block.quantized(x, rows_scale)
    *gate, up = (NvFp4Block.quantized(jnp.swapaxes(matrix, 1, 2)) for matrix in w_ins)
    gated = tuple(nvfp4_block_product(rows, matrix, group_blocks, out_dtype=out_dtype,
                                      interpret_on_cpu=interpret) for matrix in gate)
    product, codes, scales = nvfp4_block_finished(
        rows, up, group_blocks,
        epilogue=Hook(quantized.nvfp4_epilogue(activation, out_dtype),
                      tuple(Side(gate, ('rows', 'columns')) for gate in gated)),
        out_dtypes=(out_dtype, jnp.uint8, jnp.float8_e4m3fn),
        epilogue_scalars=(hidden_scale,),
        out_forms=(None, quantized.Form(2, packed=True), quantized.Form(NVFP4_BLOCKS.depth, transposed=True)),
        dense=True, interpret_on_cpu=interpret)
    hidden = NvFp4Block.packed(codes, scales.T, hidden_scale)
    out = nvfp4_block_product(hidden, NvFp4Block.quantized(jnp.swapaxes(w_down, 1, 2)), group_blocks,
                              out_dtype=out_dtype, interpret_on_cpu=interpret)
    return out, (*gated, product)


class _Chain(NamedTuple):
    """A recipe chain's (`_recipe_chain`): its `value` (`(x, w_ins, w_down, group_blocks, tensor_scales,
    activation, out_dtype, interpret)` to its output and its first projections' rounded products) and its
    projections' own reverse mode (`backward`, the recipe's surrogate, `_chain_pulled` reads)."""

    value: Callable
    backward: Callable


_CHAINS = {"blockwise_fp8": _Chain(_chain_value, _projection_backward),
           "nvfp4": _Chain(_nvfp4_chain_value, _nvfp4_backward)}
"""The recipe chains (`recipe_chain`, `_chained`) by their recipe's name (`RECIPES`'): an MLP's projections
whose down projection's rows its up projection's epilogue quantizes. MXFP8's and current-scaled NVFP4's
take none: an activation's block or tensor scales its own, known only once it is formed whole."""

# A recipe chain's gradient (`blockwise_chain`, `nvfp4_chain`): the recipe's surrogate of each projection
# (`Surrogate`), its tensor scales constants of the step.
_recipe_chain = contracted("test_a_recipe_chain_holds_every_reverse_order")(
    declared_derivative(Surrogate(), _chain, _chain_forward, _chain_backward,
                        nondiff_argnums=(0, 6, 7, 8, 9, 10)))


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
    traced program's layout: a kernel launch per device on its own rows (bible §7, rows 1306 and 1923).

    A `bias`, shaped to broadcast over the product's trailing axes, is added as `precision.biased` adds
    it; XLA's own product takes it inside its rule, the input's cotangent returned only with the kernel's
    and the bias's gradients formed, one barrier for both where the bias's own rule would add two (row
    6740). A `tensor_scale`, the rows' tensor scale under the run's delayed NVFP4 scaling, goes to the
    recipe's product (`recipe_general`); a product the recipe leaves reads none."""
    def product(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None, bias=None,
                tensor_scale=None, **kwargs):
        def added(out):
            # A product another rule takes, the bias added past it by its own (`precision.biased`).
            return out if bias is None else biased(out, bias)

        (contracted, against), batch = dimension_numbers
        out_dtype = jnp.dtype(preferred_element_type or lhs.dtype)
        # An empty product is the layer's plain dot: no launch takes zero work (`recipe_general` declines it).
        nonempty = lhs.size != 0 and rhs.size != 0
        # The run's low-precision recipe takes the product where the projection's role takes it.
        if nonempty and not kwargs and (recipe := recipe_general(
                lhs, rhs, dimension_numbers, out_dtype, tensor_scale=tensor_scale)) is not None:
            return added(recipe)
        plain = (nonempty and not kwargs and jnp.ndim(rhs) == 2 and lhs.dtype == rhs.dtype
                 and tuple(contracted) == (lhs.ndim - 1,) and tuple(against) == (0,)
                 and tuple(map(tuple, batch)) == ((), ()))
        if plain and path is not None and (shards := mapped_shards(lhs, (rhs,), (path,), out_dtype,
                                                                     precision)) is not None:
            split, gathered = shards
            return added(on_shards(lambda lhs, rhs: product(lhs, rhs, dimension_numbers, precision,
                                                            preferred_element_type), split, (lhs, rhs),
                                   gathered=gathered))
        if plain:
            rows = lhs.reshape(-1, lhs.shape[-1])
            if launched(rows, rhs, out_dtype, precision):
                out = gemm(rows, rhs, out_dtype=out_dtype, precision=precision)
                return added(out.reshape(*lhs.shape[:-1], rhs.shape[1]))
        elif nonempty and not kwargs and lhs.dtype == rhs.dtype and batch[0] and not automatic_axes():
            # A batched projection's members as uniform groups (`grouped_rows`, Appendix C row 4): one
            # grouped GEMM where a launch takes any of its products, as a dense one's (DeepSeek V4's
            # `o_a_proj`, a `DenseGeneral` over its groups; spec alignment SA20261119pm F36).
            rows, matrices, count, shape = grouped_rows(lhs, rhs, dimension_numbers)
            # One member is the dense product it is, probed and run as one: the dense product's cells, not
            # one group's (head review of 0099a4701 F3).
            matrix, sizes = ((matrices[0], None) if matrices.shape[0] == 1
                             else (matrices, jnp.full((matrices.shape[0],), count, jnp.int32)))
            if launched(rows, matrix, out_dtype, precision):
                return added(gemm(rows, matrix, sizes, out_dtype=out_dtype, precision=precision).reshape(
                    shape))
        taken = _in_layer_product(xla, dimension_numbers, precision, preferred_element_type, **kwargs)
        return taken(lhs, (rhs,) if bias is None else (rhs, bias))

    return product


def _in_layer_product(xla, dimension_numbers, precision, preferred_element_type, **kwargs):
    """A projection's product no launch takes, XLA's own (`xla`, any of its forms: a batched one's, DeepSeek
    V4's grouped `o_a_proj`), of its rows and its parameters, its kernel and its bias where it has one
    (added as `precision.biased` adds it, a span the algebra pass keeps as that one is), through the rule as
    the template's is (`derivatives.portable_formed`): the input's cotangent returned only with the
    parameters' gradients formed, one barrier for both."""
    def portable(lhs, parameters):
        rhs, *bias = parameters
        out = xla(lhs, rhs, dimension_numbers, precision=precision,
                  preferred_element_type=preferred_element_type, **kwargs)
        return marked(Protected(), biased_value)(out, *bias) if bias else out

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
    named = spec_axes(rows)
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
    taken, _ = _decided(rows, matrix, jnp.dtype(out_dtype), product, whole=True)
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


def recipe_general(lhs: jax.Array, rhs: jax.Array, dimension_numbers, out_dtype, *,
                   tensor_scale: jax.Array | None = None) -> jax.Array | None:
    """A layer's product (`jax.lax.dot_general`'s `dimension_numbers`) in the run's low-precision recipe
    (`recipe_projection`), as matrices (`grouped_rows`): a dense projection's free rows against its kernel's
    contracted axes, a batched one's (a grouped output projection's, DeepSeek V4's `o_a_proj`) each batch
    member a group of rows against its own matrix, the result in `dot_general`'s order; None where
    `recipe_projection` takes none of it, and where the run trains in no recipe before any operand is laid
    out as matrices: an operand an explicit mesh axis splits takes no reshape that does not state where its
    rows lie. `tensor_scale`, the rows' tensor scale under the run's delayed scaling, an operand of the
    product as `recipe_projection` takes it (a traced value: a scale held in no trace's key would serve a
    later scale's call the first one's program)."""
    from dew.nn.precision import recipe_role

    # An empty product is its layer's plain dot, whatever the recipe: no quantizer takes zero work, and no
    # empty operand lays out as matrices (row 13328).
    if recipe_role() is None or lhs.size == 0 or rhs.size == 0:
        return None
    rows, matrices, count, shape = grouped_rows(lhs, rhs, dimension_numbers)
    groups = matrices.shape[0]
    batched = bool(dimension_numbers[1][0])
    sizes = jnp.full((groups,), count, jnp.int32) if batched else None
    out = recipe_projection(rows, matrices if batched else matrices[0], sizes, out_dtype=out_dtype,
                            tensor_scale=tensor_scale)
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
    whole = not automatic_axes()
    return launches(rows_type, matrix_type, out_dtype, precision, whole=whole)


def launches(rows: jax.ShapeDtypeStruct, matrix: jax.ShapeDtypeStruct, out_dtype, precision: PrecisionLike, *,
             whole: bool) -> bool:
    """`launched` of a product's operands by their shapes and dtypes, where a launch sees them `whole` (no
    automatic mesh axis splits them): a product formed later in the call (a fused MLP's down product,
    `dew.nn.gated_mlp`)."""
    product = declared(rows.dtype, precision, outputs=(out_dtype,), operands=(rows.dtype, matrix.dtype))
    taken, found = _decided(rows, matrix, jnp.dtype(out_dtype), product, whole)
    generation.reported(found)
    return taken


_DECISIONS: collections.OrderedDict[Hashable, Held[tuple[bool, tuple[generation.Route, ...]]]] = (
    collections.OrderedDict())


def _decided(rows: jax.ShapeDtypeStruct, matrix: jax.ShapeDtypeStruct, out_dtype: jnp.dtype, product,
             whole: bool) -> tuple[bool, tuple[generation.Route, ...]]:
    """`launched`'s decision and the routes it found, held as every held kernel decision is
    (`_cache.holding`, `fusion.DECISIONS_HELD` of them, as the algebra's): by the operands' shapes
    and dtypes, the product and whether each is seen whole, in the context a kernel is chosen in
    (`fusion.choice_context`: the target, the kernels a block names or pins, a draw's, the refused launches),
    the facts it read recorded again wherever it is taken (row 12134: a hand-kept subset of that context
    missed a draw's, Train 1209's red main). Its routes are collected apart (`generation.routes`' `alone`),
    `launched` reporting them at every call."""
    from dew.algebra.fusion import DECISIONS_HELD, choice_context

    def made() -> tuple[bool, tuple[generation.Route, ...]]:
        with generation.routes(alone=True) as found:
            taken = _launched_once(rows, matrix, out_dtype, product, whole)
        return taken, tuple(found)

    key = (rows, matrix, out_dtype, product, whole, choice_context())
    return _cache.holding(_DECISIONS, key, made, DECISIONS_HELD)


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
