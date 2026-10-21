"""The grouped GEMM template (bible §6's third): one grouped matmul with a prologue and an epilogue.

`gemm(lhs, rhs, group_sizes)` multiplies each group's rows of `lhs` `[rows, depth]` by its matrix of
`rhs` `[groups, depth, width]` (or `[groups, width, depth]` with `transpose_rhs`), a dense product
where `group_sizes` is None. `gemm_t(lhs, rhs, group_sizes)` is each group's rows of `lhs` transposed
times its rows of `rhs`, `[groups, depth, width]`: the kernel gradient. Both multiply in `lhs`'s dtype,
`rhs` rounded to it, sum in fp32 and round once.

The hooks are plain JAX functions the kernels trace into their bodies: `prologue(lhs_tile, *side_tiles)`
transforms each loaded tile of `lhs` before its product, `epilogue(acc, *side_tiles)` each fp32
accumulator tile before its one rounding, returning one output or a tuple (`out_dtypes`). A side
operand is `(array, axes)`, `axes` naming the launch axis each of its dimensions follows ('rows',
'depth' for the prologue's; 'rows', 'columns' for the epilogue's); the kernels load each side tile at
their own origins and the hook broadcasts a vector itself (`ms[:, None]`). The rows past the routed
ones come back zero, whatever the epilogue makes of their empty sums. A measured pick for a launch
with a hook needs a module-level hook (`tuning.encoded`).

Each call runs the launch the cost model picks among the implementations that take it
(`grouped_matmul`'s Triton and Mosaic GPU launches) where it lowers for CUDA, interpreted on the CPU
where `interpret_on_cpu` asks, and XLA's ragged dot with the hooks applied around it everywhere else.
`gemm` is differentiable at every order (`generation.linear_tangent`): its reverse mode runs the
template's own launches, the products' cotangents through `gemm` and `gemm_t` and the hooks' through
their VJPs in XLA, recomputing the accumulator only where the epilogue's VJP reads it.
"""

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.kernels import generation, grouped_matmul


def gemm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None = None, *, transpose_rhs: bool = False,
         prologue=None, prologue_operands: tuple = (), epilogue=None, epilogue_operands: tuple = (),
         out_dtype, out_dtypes: tuple | None = None, interpret_on_cpu: bool = False):
    """Each group's rows of `lhs` times its matrix of `rhs`, through the hooks (the module's docstring):
    `[rows, width]` in `out_dtype`, or a tuple in `out_dtypes`."""
    rhs, group_sizes = _grouped(lhs, rhs, group_sizes)
    prologue_arrays, prologue_axes = _split(prologue_operands)
    epilogue_arrays, epilogue_axes = _split(epilogue_operands)
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    several = out_dtypes is not None
    plain = {'transpose_rhs': transpose_rhs, 'interpret_on_cpu': interpret_on_cpu}

    def value(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays):
        return _gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue,
                     epilogue_arrays, epilogue_axes, dtypes, several=several, **plain)

    def portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays):
        return _portable_gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, epilogue, epilogue_arrays,
                              dtypes, several, transpose_rhs)

    def forward(*operands):
        return value(*operands), ()

    def backward(saved, operands, cotangent):
        del saved
        lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays = operands
        routed = _routed(group_sizes, lhs.shape[0])[:, None]
        # The unrouted rows' outputs are zeros whatever the operands: their cotangents reach nothing.
        cotangents = tuple(jnp.where(routed, c, jnp.zeros_like(c))
                           for c in (cotangent if several else (cotangent,)))
        if epilogue is None:
            d_acc, d_epilogue = cotangents[0].astype(jnp.float32), ()
        else:
            acc_shape = (lhs.shape[0], rhs.shape[1] if transpose_rhs else rhs.shape[2])
            if _reads_accumulator(epilogue, acc_shape, epilogue_arrays):
                acc = _gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, None, (), (),
                            (jnp.dtype(jnp.float32),), several=False, **plain)
            else:
                acc = jnp.zeros(acc_shape, jnp.float32)
            outs, pull = jax.vjp(lambda acc, *sides: _as_tuple(epilogue(acc, *sides)), acc, *epilogue_arrays)
            d_acc, *d_epilogue = pull(tuple(c.astype(out.dtype)
                                            for c, out in zip(cotangents, outs, strict=True)))
        d_product = d_acc.astype(lhs.dtype)
        # The input's cotangent: one rounding to `lhs`'s dtype without a prologue; with one, the
        # product's cotangent rounds to the prologue's output dtype as an unfused prologue's would.
        produced = lhs.dtype if prologue is None else jnp.float32
        d_lhs = _gemm(d_product, rhs, group_sizes, None, (), (), None, (), (), (jnp.dtype(produced),),
                      several=False, transpose_rhs=not transpose_rhs, interpret_on_cpu=interpret_on_cpu)
        d_prologue = ()
        if prologue is not None:
            normed, pull = jax.vjp(prologue, lhs, *prologue_arrays)
            d_lhs, *d_prologue = pull(d_lhs.astype(normed.dtype))
        d_rhs = gemm_t(lhs, d_product, group_sizes, prologue=prologue,
                       prologue_operands=tuple(zip(prologue_arrays, prologue_axes, strict=True)),
                       out_dtype=rhs.dtype, interpret_on_cpu=interpret_on_cpu)
        d_rhs = jnp.swapaxes(d_rhs, 1, 2) if transpose_rhs else d_rhs
        return (d_lhs.astype(lhs.dtype), d_rhs, np.zeros(group_sizes.shape, jax.dtypes.float0),
                tuple(d.astype(a.dtype) for d, a in zip(d_prologue, prologue_arrays, strict=True)),
                tuple(d.astype(a.dtype) for d, a in zip(d_epilogue, epilogue_arrays, strict=True)))

    differentiable = generation.linear_tangent(forward, backward, portable)
    return differentiable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays)


def gemm_t(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, prologue=None,
           prologue_operands: tuple = (), out_dtype, interpret_on_cpu: bool = False) -> jax.Array:
    """Each group's rows of `prologue(lhs)` transposed times its rows of `rhs`: `[groups, depth, width]`
    in `out_dtype`, an empty group's zero. A value, as `gemm`'s reverse mode takes it."""
    group_sizes = group_sizes.astype(jnp.int32)
    prologue_arrays, prologue_axes = _split(prologue_operands)
    hooks = grouped_matmul.Hooks(prologue, prologue_axes)

    def kernel(lhs, rhs, group_sizes, prologue_arrays, *, interpret):
        return grouped_matmul.launched_tgmm(lhs, rhs, group_sizes, prologue_arrays, out_dtype=out_dtype,
                                            hooks=hooks, interpret=interpret)

    def portable(lhs, rhs, group_sizes, prologue_arrays):
        routed = _routed(group_sizes, lhs.shape[0])
        if prologue is not None:
            lhs = prologue(lhs, *prologue_arrays).astype(lhs.dtype)
        lhs = jnp.where(routed[:, None], lhs, 0)
        matrix = jnp.zeros((group_sizes.shape[0], lhs.shape[1], rhs.shape[1]), jnp.float32)
        _, pullback = jax.vjp(lambda matrix: grouped_matmul.xla_ragged_dot(
            lhs, matrix, group_sizes, preferred_element_type=jnp.float32), matrix)
        return pullback(rhs.astype(lhs.dtype).astype(jnp.float32))[0].astype(out_dtype)

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays)


def _gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue, epilogue_arrays,
          epilogue_axes, dtypes, *, several: bool, transpose_rhs: bool, interpret_on_cpu: bool):
    """`gemm`'s value, on grouped operands."""
    hooks = grouped_matmul.Hooks(prologue, prologue_axes, epilogue, epilogue_axes,
                                 tuple(dtype.name for dtype in dtypes) if several else None)

    def kernel(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, *, interpret):
        return grouped_matmul.launched_gmm(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays,
                                           out_dtype=dtypes[0], trans_rhs=transpose_rhs, hooks=hooks,
                                           interpret=interpret)

    def portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays):
        return _portable_gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, epilogue, epilogue_arrays,
                              dtypes, several, transpose_rhs)

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays,
                       epilogue_arrays)


def _portable_gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, epilogue, epilogue_arrays, dtypes,
                   several: bool, transpose_rhs: bool):
    """`gemm`'s arithmetic in XLA: the kernels' products of `lhs`'s dtype summed in fp32, one rounding."""
    routed = _routed(group_sizes, lhs.shape[0])[:, None]
    product = lhs if prologue is None else prologue(lhs, *prologue_arrays).astype(lhs.dtype)
    matrix = (jnp.swapaxes(rhs, 1, 2) if transpose_rhs else rhs).astype(lhs.dtype)
    summed = grouped_matmul.xla_ragged_dot(product, matrix, group_sizes, preferred_element_type=jnp.float32)
    outs = _as_tuple(summed if epilogue is None else epilogue(summed, *epilogue_arrays))
    outs = tuple(jnp.where(routed, out, 0).astype(dtype) for out, dtype in zip(outs, dtypes, strict=True))
    return outs if several else outs[0]


def _reads_accumulator(epilogue, acc_shape: tuple, sides: tuple) -> bool:
    """Whether the epilogue's VJP reads its accumulator: a bias's does not, a product with a side
    operand's does. Read from the VJP's own program, shapes alone."""
    def pulled(acc, sides, cotangents):
        return jax.vjp(lambda acc, *sides: _as_tuple(epilogue(acc, *sides)), acc, *sides)[1](cotangents)

    acc = jax.ShapeDtypeStruct(acc_shape, jnp.float32)
    outs = jax.eval_shape(lambda acc, *sides: _as_tuple(epilogue(acc, *sides)), acc, *sides)
    closed = jax.make_jaxpr(pulled)(acc, sides, outs)
    needed = {var for var in closed.jaxpr.outvars if isinstance(var, jax.core.Var)}
    for equation in reversed(closed.jaxpr.eqns):
        if needed.intersection(equation.outvars):
            needed.update(var for var in equation.invars if isinstance(var, jax.core.Var))
    return closed.jaxpr.invars[0] in needed


def _as_tuple(outs) -> tuple:
    return outs if isinstance(outs, tuple) else (outs,)


def _grouped(lhs, rhs, group_sizes):
    """`rhs` with a group axis and the group sizes: a dense product is one group of every row."""
    if group_sizes is None:
        return rhs[None], jnp.full((1,), lhs.shape[0], jnp.int32)
    return rhs, group_sizes.astype(jnp.int32)


def _split(operands: tuple) -> tuple[tuple, tuple]:
    """The side operands' arrays and their axes."""
    return tuple(array for array, _ in operands), tuple(tuple(axes) for _, axes in operands)


def _routed(group_sizes: jax.Array, rows: int) -> jax.Array:
    """Whether each of `rows` rows belongs to a group."""
    return jnp.arange(rows) < jnp.sum(group_sizes)


def _dispatched(kernel, portable, interpret_on_cpu: bool, *operands):
    """`kernel` where the call lowers for CUDA, interpreted on the CPU where asked, `portable` elsewhere
    (`grouped_matmul._on_platform`)."""
    branches, default = grouped_matmul._on_platform(kernel, portable, interpret_on_cpu)
    return jax.lax.platform_dependent(*operands, default=default, **branches)
