"""The grouped GEMM template (bible §6's third): one grouped matmul with a prologue and an epilogue.

`gemm(lhs, rhs, group_sizes)` multiplies each group's rows of `lhs` `[rows, depth]` by its matrix of
`rhs` `[groups, depth, width]` (or `[groups, width, depth]` with `transpose_rhs`), a dense product
where `group_sizes` is None. `gemm_t(lhs, rhs, group_sizes)` is each group's rows of `lhs` transposed
times its rows of `rhs`, `[groups, depth, width]`: the kernel gradient. Both sum in fp32 and round once.

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
"""

import jax
import jax.numpy as jnp

from dew.nn.kernels import grouped_matmul


def gemm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None = None, *, transpose_rhs: bool = False,
         prologue=None, prologue_operands: tuple = (), epilogue=None, epilogue_operands: tuple = (),
         out_dtype, out_dtypes: tuple | None = None, interpret_on_cpu: bool = False):
    """Each group's rows of `lhs` times its matrix of `rhs`, through the hooks (the module's docstring):
    `[rows, width]` in `out_dtype`, or a tuple in `out_dtypes`."""
    rhs, group_sizes = _grouped(lhs, rhs, group_sizes)
    prologue_arrays, prologue_axes = _split(prologue_operands)
    epilogue_arrays, epilogue_axes = _split(epilogue_operands)
    hooks = grouped_matmul.Hooks(prologue, prologue_axes, epilogue, epilogue_axes,
                                 None if out_dtypes is None else tuple(jnp.dtype(d).name for d in out_dtypes))

    def kernel(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, *, interpret):
        return grouped_matmul.launched_gmm(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays,
                                           out_dtype=out_dtype, trans_rhs=transpose_rhs, hooks=hooks,
                                           interpret=interpret)

    def portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays):
        routed = _routed(group_sizes, lhs.shape[0])
        if prologue is not None:
            lhs = prologue(lhs, *prologue_arrays)
        summed = grouped_matmul.xla_ragged_dot(lhs, jnp.swapaxes(rhs, 1, 2) if transpose_rhs else rhs,
                                               group_sizes, preferred_element_type=jnp.float32)
        outs = summed if epilogue is None else epilogue(summed, *epilogue_arrays)
        outs = outs if isinstance(outs, tuple) else (outs,)
        dtypes = out_dtypes or (out_dtype,)
        outs = tuple(jnp.where(routed[:, None], out, 0).astype(dtype)
                     for out, dtype in zip(outs, dtypes, strict=True))
        return outs if out_dtypes else outs[0]

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays,
                       epilogue_arrays)


def gemm_t(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, prologue=None,
           prologue_operands: tuple = (), out_dtype, interpret_on_cpu: bool = False) -> jax.Array:
    """Each group's rows of `prologue(lhs)` transposed times its rows of `rhs`: `[groups, depth, width]`
    in `out_dtype`, an empty group's zero."""
    group_sizes = group_sizes.astype(jnp.int32)
    prologue_arrays, prologue_axes = _split(prologue_operands)
    hooks = grouped_matmul.Hooks(prologue, prologue_axes)

    def kernel(lhs, rhs, group_sizes, prologue_arrays, *, interpret):
        return grouped_matmul.launched_tgmm(lhs, rhs, group_sizes, prologue_arrays, out_dtype=out_dtype,
                                            hooks=hooks, interpret=interpret)

    def portable(lhs, rhs, group_sizes, prologue_arrays):
        routed = _routed(group_sizes, lhs.shape[0])
        if prologue is not None:
            lhs = prologue(lhs, *prologue_arrays)
        lhs = jnp.where(routed[:, None], lhs, 0)
        matrix = jnp.zeros((group_sizes.shape[0], lhs.shape[1], rhs.shape[1]), jnp.float32)
        _, pullback = jax.vjp(lambda matrix: grouped_matmul.xla_ragged_dot(
            lhs, matrix, group_sizes, preferred_element_type=jnp.float32), matrix)
        return pullback(rhs.astype(jnp.float32))[0].astype(out_dtype)

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays)


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
