"""The grouped GEMM template (bible §6's third): one grouped matmul with a prologue and an epilogue.

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
`jax.experimental.pallas.fuser` hands over keeps its value (`fused`). A measured pick for a launch
with a hook needs a module-level hook (`tuning.encoded`).

Each call runs the launch the cost model picks among the implementations that take it
(`grouped_matmul`'s Triton and Mosaic GPU launches) where it lowers for CUDA, interpreted on the CPU
where `interpret_on_cpu` asks, and XLA's ragged dot with the hooks applied around it everywhere else.
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
from dew.nn.precision import asks_default_precision

LAUNCH_AXES = {'prologue': ('rows', 'depth'), 'epilogue': ('rows', 'columns')}
"""The launch axes each hook's side operands may follow."""


def gemm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None = None, *, transpose_rhs: bool = False,
         prologue=None, prologue_operands: tuple = (), epilogue=None, epilogue_operands: tuple = (),
         out_dtype, out_dtypes: tuple | None = None, precision: PrecisionLike = None,
         interpret_on_cpu: bool = False):
    """Each group's rows of `lhs` times its matrix of `rhs`, through the hooks (the module's docstring):
    `[rows, width]` in `out_dtype`, or a tuple in `out_dtypes`. A `precision` above the default
    (explicitly or through `jax_default_matmul_precision`) sums an fp32 output in IEEE fp32
    (`grouped_matmul.ragged_dot_refusal`), in XLA as in the kernels."""
    rhs, group_sizes = _grouped(lhs, rhs, group_sizes)
    prologue_arrays, prologue_axes = _split(prologue_operands, 'prologue')
    epilogue_arrays, epilogue_axes = _split(epilogue_operands, 'epilogue')
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    several = out_dtypes is not None
    plain = {'transpose_rhs': transpose_rhs, 'interpret_on_cpu': interpret_on_cpu, 'precision': precision}

    def value(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays):
        return _gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue,
                     epilogue_arrays, epilogue_axes, dtypes, several=several, **plain)

    def portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays):
        return _portable_gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, epilogue, epilogue_arrays,
                              dtypes, several, transpose_rhs, precision)

    def forward(*operands):
        return value(*operands), ()

    def backward(saved, operands, cotangent):
        del saved
        lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays = operands
        cotangents = cotangent if several else (cotangent,)
        acc_shape = (lhs.shape[0], rhs.shape[1] if transpose_rhs else rhs.shape[2])
        if epilogue is None:
            d_acc, d_epilogue = cotangents[0], ()
            narrow = jnp.promote_types(cotangents[0].dtype, lhs.dtype) == lhs.dtype
        else:
            if _reads_accumulator(epilogue, acc_shape, epilogue_arrays):
                acc = _gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, None, (), (),
                            (jnp.dtype(jnp.float32),), several=False, **plain)
            else:
                acc = jnp.zeros(acc_shape, jnp.float32)
            outs, pull = jax.vjp(functools.partial(_tupled, epilogue), acc, *epilogue_arrays)
            d_acc, *d_epilogue = pull(tuple(c.astype(out.dtype)
                                            for c, out in zip(cotangents, outs, strict=True)))
            narrow = _passes_cotangent(epilogue, acc_shape, epilogue_arrays, dtypes, lhs.dtype)
        # The accumulator's cotangent meets the forward's rounded operands as the portable program's
        # derivative has it: in `lhs`'s dtype where that holds it exactly (a cotangent no wider, passed
        # through), wide otherwise, each product rounding once.
        work = lhs.dtype if narrow else jnp.promote_types(d_acc.dtype, jnp.float32)
        d_product = d_acc.astype(work)
        matrix = rhs.astype(lhs.dtype)
        # The cotangent of the operand the forward multiplied, `prologue(lhs)` rounded to `lhs`'s dtype.
        d_operand = _gemm(d_product, matrix, group_sizes, None, (), (), None, (), (), (jnp.dtype(lhs.dtype),),
                          several=False, transpose_rhs=not transpose_rhs, interpret_on_cpu=interpret_on_cpu,
                          precision=precision)
        if narrow:
            d_rhs = gemm_t(lhs, d_product, group_sizes, prologue=prologue,
                           prologue_operands=tuple(zip(prologue_arrays, prologue_axes, strict=True)),
                           out_dtype=rhs.dtype, precision=precision, interpret_on_cpu=interpret_on_cpu)
        else:
            operand = lhs if prologue is None else prologue(lhs, *prologue_arrays).astype(lhs.dtype)
            d_rhs = gemm_t(operand.astype(work), d_product, group_sizes, out_dtype=rhs.dtype,
                           precision=precision, interpret_on_cpu=interpret_on_cpu)
        d_rhs = jnp.swapaxes(d_rhs, 1, 2) if transpose_rhs else d_rhs
        d_lhs, d_prologue = d_operand, ()
        if prologue is not None:
            d_lhs, d_prologue = _prologue_cotangents(prologue, lhs, prologue_arrays, prologue_axes,
                                                     group_sizes, d_operand)
        return (d_lhs.astype(lhs.dtype), d_rhs, np.zeros(group_sizes.shape, jax.dtypes.float0),
                tuple(d.astype(a.dtype) for d, a in zip(d_prologue, prologue_arrays, strict=True)),
                tuple(d.astype(a.dtype) for d, a in zip(d_epilogue, epilogue_arrays, strict=True)))

    differentiable = generation.linear_tangent(forward, backward, portable)
    return differentiable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays)


def gemm_t(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, prologue=None,
           prologue_operands: tuple = (), out_dtype, precision: PrecisionLike = None,
           interpret_on_cpu: bool = False) -> jax.Array:
    """Each group's rows of `prologue(lhs)` transposed times its rows of `rhs`: `[groups, depth, width]`
    in `out_dtype`, an empty group's zero, summed as `gemm`'s `precision` asks. A value, as `gemm`'s
    reverse mode takes it."""
    group_sizes = group_sizes.astype(jnp.int32)
    prologue_arrays, prologue_axes = _split(prologue_operands, 'prologue')

    def kernel(lhs, rhs, group_sizes, prologue_arrays, *, interpret):
        hook, sides, axes = _launchable(prologue, prologue_arrays, prologue_axes)
        return grouped_matmul.launched_tgmm(lhs, rhs, group_sizes, sides, out_dtype=out_dtype,
                                            hooks=grouped_matmul.Hooks(hook, axes), interpret=interpret,
                                            exact=_exact(precision))

    def portable(lhs, rhs, group_sizes, prologue_arrays):
        routed = _routed(group_sizes, lhs.shape[0])
        if prologue is not None:
            lhs = prologue(lhs, *prologue_arrays).astype(lhs.dtype)
        lhs = jnp.where(routed[:, None], lhs, 0)
        matrix = jnp.zeros((group_sizes.shape[0], lhs.shape[1], rhs.shape[1]), jnp.float32)
        _, pullback = jax.vjp(lambda matrix: grouped_matmul.xla_ragged_dot(
            lhs, matrix, group_sizes, precision=precision, preferred_element_type=jnp.float32), matrix)
        return pullback(rhs.astype(lhs.dtype).astype(jnp.float32))[0].astype(out_dtype)

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays)


def _gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue, epilogue_arrays,
          epilogue_axes, dtypes, *, several: bool, transpose_rhs: bool, interpret_on_cpu: bool,
          precision: PrecisionLike):
    """`gemm`'s value, on grouped operands."""
    def kernel(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, *, interpret):
        before, before_sides, before_axes = _launchable(prologue, prologue_arrays, prologue_axes)
        after, after_sides, after_axes = _launchable(epilogue, epilogue_arrays, epilogue_axes)
        hooks = grouped_matmul.Hooks(before, before_axes, after, after_axes,
                                     tuple(dtype.name for dtype in dtypes) if several else None)
        return grouped_matmul.launched_gmm(lhs, rhs, group_sizes, before_sides, after_sides,
                                           out_dtype=dtypes[0], trans_rhs=transpose_rhs, hooks=hooks,
                                           interpret=interpret, exact=_exact(precision))

    def portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays):
        return _portable_gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, epilogue, epilogue_arrays,
                              dtypes, several, transpose_rhs, precision)

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays,
                       epilogue_arrays)


def _portable_gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, epilogue, epilogue_arrays, dtypes,
                   several: bool, transpose_rhs: bool, precision: PrecisionLike):
    """`gemm`'s arithmetic in XLA: the kernels' products of `lhs`'s dtype summed in fp32, one rounding."""
    product = lhs if prologue is None else prologue(lhs, *prologue_arrays).astype(lhs.dtype)
    matrix = (jnp.swapaxes(rhs, 1, 2) if transpose_rhs else rhs).astype(lhs.dtype)
    summed = grouped_matmul.xla_ragged_dot(product, matrix, group_sizes, precision=precision,
                                           preferred_element_type=jnp.float32)
    outs = _as_tuple(summed if epilogue is None else epilogue(summed, *epilogue_arrays))
    outs = tuple(out.astype(dtype) for out, dtype in zip(outs, dtypes, strict=True))
    return outs if several else outs[0]


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
    d_sides = [jnp.where(nonempty, d, 0) for d in d_sides]
    return d_lhs, tuple(d_sides)


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


def _exact(precision: PrecisionLike) -> bool:
    """Whether `precision` asks for IEEE fp32 sums: a precision above the default."""
    return not asks_default_precision(precision, configured=True)


def _as_tuple(outs) -> tuple:
    return outs if isinstance(outs, tuple) else (outs,)


def _tupled(hook, *args) -> tuple:
    """`hook`'s outputs as a tuple, one or several."""
    return _as_tuple(hook(*args))


def _grouped(lhs, rhs, group_sizes):
    """`rhs` with a group axis and the group sizes: a dense product is one group of every row."""
    if group_sizes is None:
        return rhs[None], jnp.full((1,), lhs.shape[0], jnp.int32)
    return rhs, group_sizes.astype(jnp.int32)


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


def _launchable(hook, arrays: tuple, axes_of: tuple):
    """The hook, side operands and axes a launch takes: each dimension of one squeezed out, and the
    hook given its tiles back with those dimensions (`_unsqueezed`)."""
    positions = tuple(tuple(i for i, axis in enumerate(axes) if axis is None) for axes in axes_of)
    if hook is None or not any(positions):
        return hook, arrays, axes_of
    squeezed = tuple(jnp.squeeze(array, where) if where else array
                     for array, where in zip(arrays, positions, strict=True))
    named = tuple(tuple(axis for axis in axes if axis is not None) for axes in axes_of)
    return functools.partial(_unsqueezed, hook, positions), squeezed, named


def _unsqueezed(hook, positions: tuple, tile, *sides):
    """`hook` on `tile` and its side tiles with their dimensions of one back at `positions`."""
    return hook(tile, *(jnp.expand_dims(side, where) if where else side
                        for side, where in zip(sides, positions, strict=True)))


def _routed(group_sizes: jax.Array, rows: int) -> jax.Array:
    """Whether each of `rows` rows belongs to a group."""
    return jnp.arange(rows) < jnp.sum(group_sizes)


def _dispatched(kernel, portable, interpret_on_cpu: bool, *operands):
    """`kernel` where the call lowers for CUDA, interpreted on the CPU where asked, `portable` elsewhere
    (`grouped_matmul._on_platform`)."""
    branches, default = grouped_matmul._on_platform(kernel, portable, interpret_on_cpu)
    return jax.lax.platform_dependent(*operands, default=default, **branches)
