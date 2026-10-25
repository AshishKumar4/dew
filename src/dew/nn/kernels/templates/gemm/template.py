"""The grouped GEMM template (docs/design/kernel-templates.md): one grouped matmul with a prologue and an
epilogue.

`gemm(lhs, rhs, group_sizes)` multiplies each group's rows of `lhs` `[rows, depth]` by its matrix of
`rhs` `[groups, depth, width]` (or `[groups, width, depth]` with `transpose_rhs`), a dense product
where `group_sizes` is None. `gemm_t(lhs, rhs, group_sizes)` is each group's rows of `lhs` transposed
times its rows of `rhs`, `[groups, depth, width]`: the kernel gradient. Both multiply in `lhs`'s dtype,
`rhs` rounded to it, sum in fp32 and round once.

The hooks are plain JAX functions the kernels trace into their bodies: `prologue(lhs_tile, *side_tiles)`
transforms each loaded tile of `lhs` before its product, its result rounded to `lhs`'s dtype;
`epilogue(acc, *side_tiles)` each fp32 accumulator tile before its one rounding, returning one output
or a tuple (`out_dtypes`). A side operand is `(array, axes)`, `axes` naming per dimension the launch
axis it follows ('rows', 'depth' for the prologue's; 'rows', 'columns' for the epilogue's) or None for
a dimension of one the hook broadcasts; the kernels load each side tile at their own origins and the
hook broadcasts a vector itself (`ms[:, None]`). The epilogue applies to every row: a row past the
routed ones sums nothing, so `gemm(..., epilogue=e)` is `e` of the plain product, and a function
`jax.experimental.pallas.fuser` hands over keeps its value (`fused`).

Each call runs the first of the implementations that takes it, in a fixed order
(`grouped_matmul.GROUPED`: its Mosaic GPU launches, the TPU's on Megablox's schedule, then Triton's;
CUTLASS's only by name) where it lowers for its target's platform, interpreted on the CPU where
`interpret_on_cpu` asks, and XLA's ragged dot with the hooks applied around it everywhere else, for a
product no launch takes (the route says why), or where `generation.chosen` names XLA; a launch it
names that does not take the call raises (`selection.KernelRefused`).
`gemm` is differentiable at every order (`generation.linear_tangent`): its reverse mode runs the
template's own launches with the arithmetic of the portable program's derivative, the products'
cotangents through `gemm` and `gemm_t` and the hooks' through their VJPs in XLA, recomputing the
accumulator only where the epilogue's VJP reads it.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import PrecisionLike
from jax.extend import core as jax_core

from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.templates.product import declared
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import seen_whole

LAUNCH_AXES = {'prologue': ('rows', 'depth'), 'epilogue': ('rows', 'columns')}
"""The launch axes each hook's side operands may follow."""


def gemm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None = None, *, transpose_rhs: bool = False,
         prologue=None, prologue_operands: tuple = (), epilogue=None, epilogue_operands: tuple = (),
         out_dtype, out_dtypes: tuple | None = None, precision: PrecisionLike = None,
         interpret_on_cpu: bool = False, prologue_jvp=None, compute=None, layer: jax.Array | None = None):
    """Each group's rows of `lhs` times its matrix of `rhs`, through the hooks (the module's docstring):
    `[rows, width]` in `out_dtype`, or a tuple in `out_dtypes`. A `precision` above the default
    (explicitly or through `jax_default_matmul_precision`) sums an fp32 output in IEEE fp32, in XLA as in
    the kernels: the product `precision` declares (`templates.product.declared`), which every launch of
    the call and of its derivative keeps. `prologue_jvp(lhs, *side_arrays)`, where given, is the prologue's
    derivative in place of the prologue's own: a function of `lhs`'s and each side array's tangent giving
    the multiplied operand's (a fused producer's whole JVP, `fused`), which forward mode takes, reverse mode
    transposes and every higher order differentiates (`_prologued`). `compute`, where given, is the
    dtype the products take, `lhs` rounded to it as `rhs` is, where `lhs`'s own dtype is what its
    cotangent takes, summed in fp32 and rounded once to it (an fp32 input's of a bf16 product); the
    rounding passes tangents through unrounded, the MoE projection's law
    (`grouped_matmul.portable_projection`), and there is no prologue. `layer`, where given (a traced
    index), picks one layer of a stacked `rhs` (`[layers, *matrices]`: a layer's groups, or its one
    matrix for a dense product), read in place: the stack's matrices are the call's groups, every other
    layer's empty, so every launch reads the layer's matrices where they lie and no slice of the stack
    is made (`_grouped`). Its gradient is the stack's, every other layer's zero."""
    if compute is not None and prologue is not None:
        raise ValueError("a prologue's operand is multiplied in its own dtype, not a named `compute`")
    straight = compute is not None
    compute = jnp.dtype(lhs.dtype if compute is None else compute)
    dense = group_sizes is None and layer is None
    prologue = _prologued(prologue, prologue_jvp)
    rhs, group_sizes = _grouped(lhs, rhs, group_sizes, layer)
    prologue_arrays, prologue_axes = _split(prologue_operands, 'prologue')
    epilogue_arrays, epilogue_axes = _split(epilogue_operands, 'epilogue')
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    several = out_dtypes is not None
    plain = {'transpose_rhs': transpose_rhs, 'interpret_on_cpu': interpret_on_cpu, 'precision': precision,
             'dense': dense}

    def value(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays):
        return _gemm(lhs.astype(compute), rhs, group_sizes, prologue, prologue_arrays, prologue_axes,
                     epilogue, epilogue_arrays, epilogue_axes, dtypes, several=several, **plain)

    def portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays):
        return _portable_gemm(lhs if straight else lhs.astype(compute), rhs, group_sizes, prologue,
                              prologue_arrays, epilogue, epilogue_arrays, dtypes, several, transpose_rhs,
                              precision, compute if straight else None)

    def forward(*operands):
        return value(*operands), ()

    def backward(saved, operands, cotangent, live):
        del saved
        lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays = operands
        # The operands a reverse pass differentiates, by their place among the operands' leaves: a frozen
        # matrix's kernel gradient, or an operand no reverse pass reaches, is never formed.
        wanted = set(live)
        prologue_places = set(range(3, 3 + len(prologue_arrays)))
        cotangents = cotangent if several else (cotangent,)
        acc_shape = (lhs.shape[0], rhs.shape[1] if transpose_rhs else rhs.shape[2])
        if epilogue is None:
            d_acc, d_epilogue = cotangents[0], ()
            narrow = jnp.promote_types(cotangents[0].dtype, compute) == compute
        else:
            if _reads_accumulator(epilogue, acc_shape, epilogue_arrays):
                acc = _gemm(lhs.astype(compute), rhs, group_sizes, prologue, prologue_arrays, prologue_axes,
                            None, (), (),
                            (jnp.dtype(jnp.float32),), several=False, **plain)
            else:
                acc = jnp.zeros(acc_shape, jnp.float32)
            outs, pull = jax.vjp(functools.partial(_tupled, epilogue), acc, *epilogue_arrays)
            d_acc, *d_epilogue = pull(tuple(c.astype(out.dtype)
                                            for c, out in zip(cotangents, outs, strict=True)))
            narrow = _passes_cotangent(epilogue, acc_shape, epilogue_arrays, dtypes, compute)
        # The accumulator's cotangent meets the forward's rounded operands as the portable program's
        # derivative has it: in `lhs`'s dtype where that holds it exactly (a cotangent no wider, passed
        # through), wide otherwise, each product rounding once.
        work = compute if narrow else at_least_fp32(d_acc.dtype)
        d_product = d_acc.astype(work)
        d_lhs = d_rhs = None
        d_prologue = ()
        if wanted & ({0} | prologue_places):
            matrix = rhs.astype(compute)
            # The cotangent of the operand the forward multiplied, `prologue(lhs)` rounded to `lhs`'s dtype.
            d_operand = jnp.asarray(_gemm(d_product, matrix, group_sizes, None, (), (), None, (), (),
                                          (jnp.dtype(lhs.dtype),), several=False,
                                          transpose_rhs=not transpose_rhs, interpret_on_cpu=interpret_on_cpu,
                                          precision=precision, dense=dense))
            d_lhs = d_operand
            if prologue is not None:
                d_lhs, d_prologue = _prologue_cotangents(prologue, lhs, prologue_arrays, prologue_axes,
                                                         group_sizes, d_operand)
            d_lhs = d_lhs.astype(lhs.dtype)
            d_prologue = tuple(_as_cotangent(d, a) for d, a in zip(d_prologue, prologue_arrays, strict=True))
        if 1 in wanted:
            if narrow:
                d_rhs = gemm_t(lhs.astype(compute), d_product, None if dense else group_sizes,
                               prologue=prologue,
                               prologue_operands=tuple(zip(prologue_arrays, prologue_axes, strict=True)),
                               out_dtype=rhs.dtype, precision=precision, interpret_on_cpu=interpret_on_cpu)
            else:
                operand = (lhs.astype(compute) if prologue is None
                           else prologue(lhs, *prologue_arrays).astype(lhs.dtype))
                d_rhs = gemm_t(operand.astype(work), d_product, None if dense else group_sizes,
                               out_dtype=rhs.dtype, precision=precision, interpret_on_cpu=interpret_on_cpu)
            d_rhs = jnp.swapaxes(d_rhs, 1, 2) if transpose_rhs else d_rhs
        leaves = [d_lhs, d_rhs, np.zeros(group_sizes.shape, jax.dtypes.float0),
                  *(d_prologue or (None,) * len(prologue_arrays)),
                  *(_as_cotangent(d, a) for d, a in zip(d_epilogue, epilogue_arrays, strict=True))]
        return tuple(leaves[index] for index in live)

    differentiable = generation.linear_tangent(forward, backward, portable, live_cotangents=True)
    return differentiable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays)


def gemm_t(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None, *, prologue=None,
           prologue_operands: tuple = (), out_dtype, precision: PrecisionLike = None,
           interpret_on_cpu: bool = False) -> jax.Array:
    """Each group's rows of `prologue(lhs)` transposed times its rows of `rhs`: `[groups, depth, width]`
    in `out_dtype`, an empty group's zero, summed as `gemm`'s `precision` asks; without `group_sizes`,
    every row one group, a dense product's (`[1, depth, width]`). A value, as `gemm`'s reverse mode
    takes it."""
    dense = group_sizes is None
    group_sizes = jnp.full((1,), lhs.shape[0], jnp.int32) if dense else group_sizes.astype(jnp.int32)
    prologue_arrays, prologue_axes = _split(prologue_operands, 'prologue')

    def kernel(lhs, rhs, group_sizes, prologue_arrays, *, interpret):
        sides, axes, broadcast = _launchable(prologue_arrays, prologue_axes)
        hooks = grouped_matmul.Hooks(prologue, axes, prologue_broadcast=broadcast)
        product = declared(lhs.dtype, precision, outputs=(out_dtype,), operands=(lhs.dtype, rhs.dtype))
        call = grouped_matmul.tgmm_call(lhs, rhs.astype(lhs.dtype), group_sizes, sides, out_dtype=out_dtype,
                                        hooks=hooks, interpret=interpret, promoted=product.promoted,
                                        dense=dense)
        launch = grouped_matmul.launched(*call, product=product, whole=seen_whole(lhs) and seen_whole(rhs))
        return portable(lhs, rhs, group_sizes, prologue_arrays) if launch is None else launch()

    def portable(lhs, rhs, group_sizes, prologue_arrays):
        routed = _routed(group_sizes, lhs.shape[0])
        if prologue is not None:
            lhs = prologue(lhs, *prologue_arrays).astype(lhs.dtype)
        lhs = jnp.where(routed[:, None], lhs, 0)
        wide = at_least_fp32(lhs.dtype)  # an fp64 product's sum in fp64
        matrix = jnp.zeros((group_sizes.shape[0], lhs.shape[1], rhs.shape[1]), wide)
        _, pullback = jax.vjp(lambda matrix: grouped_matmul.xla_ragged_dot(
            lhs, matrix, group_sizes, precision=precision, preferred_element_type=wide), matrix)
        return pullback(rhs.astype(lhs.dtype).astype(wide))[0].astype(out_dtype)

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays)


def _gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue, epilogue_arrays,
          epilogue_axes, dtypes, *, several: bool, transpose_rhs: bool, interpret_on_cpu: bool,
          precision: PrecisionLike, dense: bool = False):
    """`gemm`'s value, on grouped operands."""
    def kernel(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, *, interpret):
        before_sides, before_axes, before_broadcast = _launchable(prologue_arrays, prologue_axes)
        after_sides, after_axes, after_broadcast = _launchable(epilogue_arrays, epilogue_axes)
        hooks = grouped_matmul.Hooks(prologue, before_axes, epilogue, after_axes,
                                     tuple(dtype.name for dtype in dtypes) if several else None,
                                     prologue_broadcast=before_broadcast, epilogue_broadcast=after_broadcast)
        product = declared(lhs.dtype, precision, outputs=dtypes, operands=(lhs.dtype, rhs.dtype))
        call = grouped_matmul.gmm_call(lhs, rhs.astype(lhs.dtype), group_sizes, before_sides, after_sides,
                                       out_dtype=dtypes[0], trans_rhs=transpose_rhs, hooks=hooks,
                                       interpret=interpret, promoted=product.promoted, dense=dense)
        # A call no launch takes (a TPU's widths its blocks do not divide) runs XLA's program.
        launch = grouped_matmul.launched(*call, product=product, whole=seen_whole(lhs) and seen_whole(rhs))
        if launch is None:
            return portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays)
        return launch()

    def portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays):
        return _portable_gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, epilogue, epilogue_arrays,
                              dtypes, several, transpose_rhs, precision)

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays,
                       epilogue_arrays)


def _portable_gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, epilogue, epilogue_arrays, dtypes,
                   several: bool, transpose_rhs: bool, precision: PrecisionLike, compute=None):
    """`gemm`'s arithmetic in XLA: the kernels' products of `lhs`'s dtype summed in at least fp32 (an fp64
    product's in fp64, which no kernel takes), one rounding;
    with `compute`, the products of the operands rounded to it, its tangent the MoE projection's
    (`grouped_matmul.portable_projection`, whose sum, at least fp32, it reads)."""
    matrix = jnp.swapaxes(rhs, 1, 2) if transpose_rhs else rhs
    if compute is not None:
        # Summed and kept at least fp32 (an fp64 product's in fp64), the accumulator the epilogue reads.
        wide = at_least_fp32(compute)
        summed = grouped_matmul.portable_projection(lhs, matrix, group_sizes, compute, precision, wide)
    else:
        product = lhs if prologue is None else prologue(lhs, *prologue_arrays).astype(lhs.dtype)
        wide = at_least_fp32(lhs.dtype)
        summed = grouped_matmul.xla_ragged_dot(product, matrix.astype(lhs.dtype), group_sizes,
                                               precision=precision, preferred_element_type=wide)
    outs = _as_tuple(summed if epilogue is None else epilogue(summed, *epilogue_arrays))
    outs = tuple(out.astype(dtype) for out, dtype in zip(outs, dtypes, strict=True))
    return outs if several else outs[0]


def _prologued(prologue, prologue_jvp):
    """`prologue`, its derivative `prologue_jvp`'s where given (`gemm`): one derivative contract for
    forward mode, which takes it, reverse mode, which transposes it, and every higher order."""
    if prologue is None or prologue_jvp is None:
        return prologue

    @jax.custom_jvp
    def prologued(lhs, *sides):
        return prologue(lhs, *sides)

    @prologued.defjvp
    def _(primals, tangents):
        return prologued(*primals), prologue_jvp(*primals)(*tangents)

    return prologued


def _prologue_cotangents(prologue, lhs, arrays, axes_of, group_sizes, d_operand):
    """The cotangents of `lhs` and the prologue's side operands from the multiplied operand's: through
    the VJP of `prologue(...)` rounded to `lhs`'s dtype. A row past the routed ones multiplied nothing,
    so its cotangent is zero, and its inputs are taken at the first row's, a routed one's where any is,
    so that a derivative its own inputs leave undefined (`rsqrt` of an empty row's zero mean square)
    reaches no gradient the routed rows share."""
    routed = _routed(group_sizes, lhs.shape[0])
    nonempty = jnp.sum(group_sizes) > 0

    def kept(array, axis):
        shape = [1] * array.ndim
        shape[axis] = array.shape[axis]
        return jnp.where(routed.reshape(shape), array, jax.lax.slice_in_dim(array, 0, 1, axis=axis))

    rows_at = tuple(axes.index('rows') if 'rows' in axes else None for axes in axes_of)
    inputs = (kept(lhs, 0), *(array if axis is None else kept(array, axis)
                              for array, axis in zip(arrays, rows_at, strict=True)))
    _, pull = jax.vjp(lambda x, *sides: prologue(x, *sides).astype(lhs.dtype), *inputs)
    d_lhs, *d_sides = pull(d_operand.astype(lhs.dtype))
    d_lhs = jnp.where(routed[:, None] & nonempty, d_lhs, 0)
    d_sides = [d if d.dtype == jax.dtypes.float0 else jnp.where(nonempty, d, 0) for d in d_sides]
    return d_lhs, tuple(d_sides)


def _as_cotangent(cotangent, operand):
    """`cotangent` in `operand`'s dtype; an integer or boolean operand's stays float0, its tangent space."""
    return cotangent if cotangent.dtype == jax.dtypes.float0 else cotangent.astype(operand.dtype)


def _reads_accumulator(epilogue, acc_shape: tuple, sides: tuple) -> bool:
    """Whether the epilogue's VJP reads its accumulator: a bias's does not, a product with a side
    operand's does. Read from the VJP's own program, shapes alone."""
    closed, _ = _epilogue_vjp(epilogue, acc_shape, sides)
    return closed.jaxpr.invars[0] in _needed(closed.jaxpr, closed.jaxpr.outvars)


def _passes_cotangent(epilogue, acc_shape: tuple, sides: tuple, dtypes: tuple, narrow) -> bool:
    """Whether the epilogue's VJP hands the accumulator one output's cotangent unchanged (a bias's,
    a cast's) whose dtype `narrow` holds exactly: read from the VJP's own program, its result traced
    back through conversions alone."""
    closed, cotangents = _epilogue_vjp(epilogue, acc_shape, sides)
    var = closed.jaxpr.outvars[0]
    producers = {out: equation for equation in closed.jaxpr.eqns for out in equation.outvars}
    while var in producers and producers[var].primitive.name == 'convert_element_type':
        var = producers[var].invars[0]
    if var not in cotangents:
        return False
    return jnp.promote_types(dtypes[cotangents.index(var)], narrow) == jnp.dtype(narrow)


def _epilogue_vjp(epilogue, acc_shape: tuple, sides: tuple):
    """The epilogue's VJP as a program of the accumulator, its side operands and its outputs'
    cotangents, and the cotangents' variables in it."""
    tupled = functools.partial(_tupled, epilogue)

    def pulled(acc, sides, cotangents):
        return jax.vjp(tupled, acc, *sides)[1](cotangents)

    acc = jax.ShapeDtypeStruct(acc_shape, jnp.float32)
    outs = jax.eval_shape(tupled, acc, *sides)
    closed = jax.make_jaxpr(pulled)(acc, sides, outs)
    return closed, list(closed.jaxpr.invars[1 + len(sides):])


def _needed(jaxpr, outvars) -> set:
    """The variables of `jaxpr` that `outvars` depend on."""
    needed = {var for var in outvars if isinstance(var, jax_core.Var)}
    for equation in reversed(jaxpr.eqns):
        if needed.intersection(equation.outvars):
            needed.update(var for var in equation.invars if isinstance(var, jax_core.Var))
    return needed


def _as_tuple(outs) -> tuple:
    return outs if isinstance(outs, tuple) else (outs,)


def _tupled(hook, *args) -> tuple:
    """`hook`'s outputs as a tuple, one or several."""
    return _as_tuple(hook(*args))


def _grouped(lhs, rhs, group_sizes, layer=None):
    """`rhs` with a group axis and the group sizes: a dense product is one group of every row. A stacked
    `rhs` at `layer` (`gemm`) is its every layer's groups as one axis, the sizes `group_sizes` at the
    layer's and zero at every other's, so a launch reads the layer's matrices in place."""
    sizes = (jnp.full((1,), lhs.shape[0], jnp.int32) if group_sizes is None
             else group_sizes.astype(jnp.int32))
    if layer is None:
        return (rhs[None] if group_sizes is None else rhs), sizes
    stack = rhs[:, None] if group_sizes is None else rhs
    layers, groups = stack.shape[:2]
    stacked = jnp.zeros((layers, groups), jnp.int32).at[layer].set(sizes)
    return stack.reshape(layers * groups, *stack.shape[2:]), stacked.reshape(layers * groups)


def _split(operands: tuple, hook: str) -> tuple[tuple, tuple]:
    """The side operands' arrays and their axes, each checked: one axis per dimension, each a launch
    axis the hook's operands follow or None for a dimension of one, at least one a launch axis."""
    arrays, axes_of = [], []
    for array, axes in operands:
        axes = tuple(axes)
        named = [axis for axis in axes if axis is not None]
        if (len(axes) != array.ndim or not named or any(axis not in LAUNCH_AXES[hook] for axis in named)
                or any(extent != 1 for axis, extent in zip(axes, array.shape, strict=True) if axis is None)):
            raise ValueError(f"a {hook} side operand of shape {array.shape} does not take axes {axes}: one "
                             f"per dimension, each one of {LAUNCH_AXES[hook]} or None for a dimension of one")
        arrays.append(array)
        axes_of.append(axes)
    return tuple(arrays), tuple(axes_of)


def _launchable(arrays: tuple, axes_of: tuple) -> tuple[tuple, tuple, tuple]:
    """The side operands and axes a launch takes, each dimension of one squeezed out, and each one's
    positions of them, which the launch gives the hook back (`grouped_matmul.Hooks`)."""
    positions = tuple(tuple(i for i, axis in enumerate(axes) if axis is None) for axes in axes_of)
    squeezed = tuple(jnp.squeeze(array, where) if where else array
                     for array, where in zip(arrays, positions, strict=True))
    named = tuple(tuple(axis for axis in axes if axis is not None) for axes in axes_of)
    return squeezed, named, positions if any(positions) else ()


def _routed(group_sizes: jax.Array, rows: int) -> jax.Array:
    """Whether each of `rows` rows belongs to a group."""
    return jnp.arange(rows) < jnp.sum(group_sizes)


def _dispatched(kernel, portable, interpret_on_cpu: bool, *operands):
    """`kernel` where the call lowers for CUDA, interpreted on the CPU where asked, `portable` elsewhere
    (`grouped_matmul._on_platform`)."""
    branches, default = grouped_matmul._on_platform(kernel, portable, interpret_on_cpu)
    return jax.lax.platform_dependent(*operands, default=default, **branches)
