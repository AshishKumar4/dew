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

from dew.nn.kernels import generation, grouped_matmul, selection
from dew.nn.kernels.templates.product import declared
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import seen_whole

LAUNCH_AXES = {'prologue': ('rows', 'depth'), 'epilogue': ('rows', 'columns')}
"""The launch axes each hook's side operands may follow."""


def gemm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None = None, *, transpose_rhs: bool = False,
         prologue=None, prologue_operands: tuple = (), epilogue=None, epilogue_operands: tuple = (),
         out_dtype, out_dtypes: tuple | None = None, precision: PrecisionLike = None,
         interpret_on_cpu: bool = False, prologue_jvp=None, compute=None, layer: jax.Array | None = None,
         paired: jax.Array | None = None):
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
    is made (`_grouped`). Its gradient is the stack's, every other layer's zero. `paired`, a second matrix
    of `rhs`'s shape, the rows multiply into a second accumulator the epilogue takes before its side
    operands (`epilogue(acc, paired_acc, *sides)`: a gate's and an up projection's SwiGLU), else an output."""
    if paired is not None and layer is not None:
        raise ValueError("a paired product takes no stacked layer")
    if compute is not None and prologue is not None:
        raise ValueError("a prologue's operand is multiplied in its own dtype, not a named `compute`")
    straight = compute is not None
    compute = jnp.dtype(lhs.dtype if compute is None else compute)
    dense = group_sizes is None and layer is None
    prologue = _prologued(prologue, prologue_jvp)
    paired = None if paired is None else _grouped(lhs, paired, group_sizes, layer)[0]
    rhs, group_sizes = _grouped(lhs, rhs, group_sizes, layer)
    prologue_arrays, prologue_axes = _split(prologue_operands, 'prologue')
    epilogue_arrays, epilogue_axes = _split(epilogue_operands, 'epilogue')
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    several = out_dtypes is not None
    plain = {'transpose_rhs': transpose_rhs, 'interpret_on_cpu': interpret_on_cpu, 'precision': precision,
             'dense': dense}

    def value(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired):
        return _gemm(lhs.astype(compute), rhs, group_sizes, prologue, prologue_arrays, prologue_axes,
                     epilogue, epilogue_arrays, epilogue_axes, dtypes, several=several, paired=paired,
                     **plain)

    def portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired):
        return _portable_gemm(lhs if straight else lhs.astype(compute), rhs, group_sizes, prologue,
                              prologue_arrays, epilogue, epilogue_arrays, dtypes, several, transpose_rhs,
                              precision, compute if straight else None, paired=paired)

    def forward(*operands):
        return value(*operands), ()

    # A kernel the call names (`generation.chosen`, MoE's implementation='xla') runs its derivative's products
    # too, wherever the pullback runs: a reverse pass traced outside the call's block would otherwise select
    # them afresh.
    named = selection.forced(grouped_matmul.GROUPED.choice)

    def backward(saved, operands, cotangent, live):
        if named is None:
            return _pulled(operands, cotangent, live)
        with generation.chosen(**{grouped_matmul.GROUPED.choice: named}):
            return _pulled(operands, cotangent, live)

    def _pulled(operands, cotangent, live):
        lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired = operands
        # The operands a reverse pass differentiates, by their place among the operands' leaves: a frozen
        # matrix's kernel gradient, or an operand no reverse pass reaches, is never formed.
        wanted = set(live)
        matrices = (rhs,) if paired is None else (rhs, paired)
        places = (1, 3 + len(prologue_arrays) + len(epilogue_arrays))[:len(matrices)]
        cotangents = cotangent if several else (cotangent,)
        acc_shape = (lhs.shape[0], rhs.shape[1] if transpose_rhs else rhs.shape[2])
        d_accs, d_epilogue, narrow = _cotangents(lhs.astype(compute), matrices, group_sizes, prologue,
                                                 prologue_arrays, prologue_axes, epilogue, epilogue_arrays,
                                                 cotangents, dtypes, compute, acc_shape, plain)
        # Each accumulator's cotangent meets the forward's rounded operands as the portable program's
        # derivative has it: in `lhs`'s dtype where that holds it exactly (a cotangent no wider, passed
        # through), wide otherwise, each product rounding once.
        works = tuple(compute if each else at_least_fp32(d.dtype)
                      for d, each in zip(d_accs, narrow, strict=True))
        d_products = tuple(d.astype(work) for d, work in zip(d_accs, works, strict=True))
        d_lhs, d_prologue = None, ()
        if wanted & set(range(3 + len(prologue_arrays))) - {1, 2}:
            d_lhs, d_prologue = _input_cotangents(lhs, d_products, matrices, group_sizes, prologue,
                                                  prologue_arrays, prologue_axes, compute, plain)
        gradients = [None if place not in wanted
                     else _kernel_gradient(lhs, d_product, matrix, group_sizes, prologue, prologue_arrays,
                                           prologue_axes, compute, work if not each else None, plain)
                     for place, d_product, matrix, work, each in zip(places, d_products, matrices, works,
                                                                     narrow, strict=True)]
        leaves = [d_lhs, gradients[0], np.zeros(group_sizes.shape, jax.dtypes.float0),
                  *(d_prologue or (None,) * len(prologue_arrays)),
                  *(_as_cotangent(d, a) for d, a in zip(d_epilogue, epilogue_arrays, strict=True)),
                  *gradients[1:]]
        return tuple(leaves[index] for index in live)

    differentiable = generation.linear_tangent(forward, backward, portable, live_cotangents=True)
    return differentiable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired)


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
        hooks = grouped_matmul.Hooks(prologue, axes, prologue_broadcast=broadcast,
                                     prologue_dtypes=_dtypes(prologue, lhs, sides))
        product = declared(lhs.dtype, precision, outputs=(out_dtype,), operands=(lhs.dtype, rhs.dtype))
        call = grouped_matmul.tgmm_call(lhs, rhs.astype(lhs.dtype), group_sizes, sides, out_dtype=out_dtype,
                                        hooks=hooks, interpret=interpret, promoted=product.promoted,
                                        dense=dense)
        launch = grouped_matmul.launched(*call, product=product, whole=seen_whole(lhs) and seen_whole(rhs))
        # A kernel gradient is one product: no launch takes it apart.
        return launch() if callable(launch) else portable(lhs, rhs, group_sizes, prologue_arrays)

    def portable(lhs, rhs, group_sizes, prologue_arrays):
        routed = _routed(group_sizes, lhs.shape[0])
        if prologue is not None:
            lhs = prologue(lhs, *prologue_arrays).astype(lhs.dtype)
        lhs = jnp.where(routed[:, None], lhs, 0)
        wide = at_least_fp32(lhs.dtype)  # an fp64 product's sum in fp64
        # Each group's rows contracted, XLA's ragged dot's own weight gradient (its transpose's form) on the
        # operands as they are: a pullback from a matrix made here would hand a vmapped call (a pipeline
        # stage's) an unbatched operand, which ragged_dot's batching rule refuses.
        return jax.lax.ragged_dot_general(
            lhs.astype(wide), rhs.astype(lhs.dtype).astype(wide), group_sizes, _CONTRACTED_ROWS,
            precision=precision, preferred_element_type=wide).astype(out_dtype)

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays)


def dot_general(xla):
    """`xla`, a layer's product (a `jax.lax.dot_general`, as flax's `Dense` takes one), through the
    template's dense product (`gemm` of one group) where the call is a matrix product (`lhs`'s last axis
    against a matrix's first, no batch axes, both of one dtype) and a launch takes any of its products: the
    forward, the input gradient or the kernel gradient (`_launched`). A call no launch takes is `xla`'s own
    program, differentiated by JAX, each of its products' routes reported with the reason no launch takes
    it (a declared cell's old route, a mesh axis splitting the operands, a CPU), so a layer runs as it ran
    before wherever the template declares XLA faster; every other product is `xla`'s."""
    def product(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None, **kwargs):
        (contracted, against), batch = dimension_numbers
        plain = (not kwargs and jnp.ndim(rhs) == 2 and lhs.dtype == rhs.dtype
                 and tuple(contracted) == (lhs.ndim - 1,) and tuple(against) == (0,)
                 and tuple(map(tuple, batch)) == ((), ()))
        if plain:
            rows = lhs.reshape(-1, lhs.shape[-1])
            out_dtype = jnp.dtype(preferred_element_type or lhs.dtype)
            if _launched(rows, rhs, out_dtype, precision):
                product = gemm(rows, rhs, out_dtype=out_dtype, precision=precision)
                return product.reshape(*lhs.shape[:-1], rhs.shape[1])
        return xla(lhs, rhs, dimension_numbers, precision=precision,
                   preferred_element_type=preferred_element_type, **kwargs)

    return product


def _launched(rows: jax.Array, matrix: jax.Array, out_dtype, precision: PrecisionLike) -> bool:
    """Whether a launch takes any of the dense product `rows @ matrix`'s products on the call's target
    (`grouped_matmul.GROUPED`): its forward, its input gradient or its kernel gradient, as `gemm` and
    its reverse mode make them. Where none does, each one's route is reported, as XLA runs them."""
    product = declared(rows.dtype, precision, outputs=(out_dtype,), operands=(rows.dtype, matrix.dtype))
    whole = seen_whole(rows) and seen_whole(matrix)
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
    for case in cases:
        selection.select(grouped_matmul.GROUPED, case)
    return False


_CONTRACTED_ROWS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([0], [0]), ([], [])), lhs_ragged_dimensions=[0], rhs_group_dimensions=[])
"""A ragged dot over its rows, each group's rows of both operands contracted into the group's matrix: a
grouped GEMM's weight gradient, `[rows, depth]` and `[rows, width]` to `[groups, depth, width]`."""

_ACCUMULATOR = jax.ShapeDtypeStruct((), jnp.float32)
"""An epilogue's tile: the fp32 accumulator's."""


def _dtypes(hook, tile, sides: tuple) -> tuple[str, ...]:
    """The dtypes a kernel traces `hook` at: its tile's (`tile`'s), then each side operand's; none without
    a hook."""
    return () if hook is None else tuple(jnp.dtype(array.dtype).name for array in (tile, *sides))


def _gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue, epilogue_arrays,
          epilogue_axes, dtypes, *, several: bool, transpose_rhs: bool, interpret_on_cpu: bool,
          precision: PrecisionLike, dense: bool = False, paired: jax.Array | None = None,
          summed_as: tuple = ()):
    """`gemm`'s value, on grouped operands; `summed_as`, the outputs of the product whose accumulator it
    recomputes (a reverse pass's, at fp32), whose sums it repeats (`product.promoted`: an fp32 output's
    exact sums, a 16-bit one's not)."""
    def kernel(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, *, interpret):
        before_sides, before_axes, before_broadcast = _launchable(prologue_arrays, prologue_axes)
        after_sides, after_axes, after_broadcast = _launchable(epilogue_arrays, epilogue_axes)
        hooks = grouped_matmul.Hooks(prologue, before_axes, epilogue, after_axes,
                                     tuple(dtype.name for dtype in dtypes) if several else None,
                                     prologue_broadcast=before_broadcast, epilogue_broadcast=after_broadcast,
                                     prologue_dtypes=_dtypes(prologue, lhs, before_sides),
                                     epilogue_dtypes=_dtypes(epilogue, _ACCUMULATOR, after_sides))
        product = declared(lhs.dtype, precision, outputs=summed_as or dtypes, operands=(lhs.dtype, rhs.dtype))
        call = grouped_matmul.gmm_call(lhs, rhs.astype(lhs.dtype), group_sizes, before_sides, after_sides,
                                       None if paired is None else paired.astype(lhs.dtype),
                                       out_dtype=dtypes[0], trans_rhs=transpose_rhs, hooks=hooks,
                                       interpret=interpret, promoted=product.promoted, dense=dense)
        # A call no launch takes (a TPU's widths its blocks do not divide) runs XLA's program; a paired one no
        # launch takes whole its two products apart, each through the launches that take it (`_apart`).
        launch = grouped_matmul.launched(*call, product=product, whole=seen_whole(lhs) and seen_whole(rhs))
        if launch is grouped_matmul.APART:
            return _apart(lhs, (rhs, paired), group_sizes, prologue, prologue_arrays, prologue_axes, epilogue,
                          epilogue_arrays, dtypes, several=several, summed_as=summed_as or dtypes,
                          transpose_rhs=transpose_rhs, interpret_on_cpu=interpret_on_cpu, precision=precision,
                          dense=dense)
        if not callable(launch):
            return portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired)
        return launch()

    def portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired):
        return _portable_gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, epilogue, epilogue_arrays,
                              dtypes, several, transpose_rhs, precision, paired=paired)

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays,
                       epilogue_arrays, paired)


def _cotangents(lhs, matrices: tuple, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue,
                epilogue_arrays, cotangents: tuple, dtypes: tuple, compute, acc_shape: tuple,
                plain: dict) -> tuple[tuple, tuple, tuple]:
    """Each accumulator's cotangent (one a matrix of `matrices`: the product's, and its `paired`'s), the
    epilogue side operands' and whether each accumulator's passes an output's through in `compute`'s dtype
    (`_passes_cotangent`): the outputs' own without an epilogue, else through the epilogue's VJP at the
    accumulators recomputed as the outputs (`dtypes`) summed them, where the VJP reads them."""
    count = len(matrices)
    if epilogue is None:
        d_accs = tuple(cotangents[:count])
        return d_accs, (), tuple(jnp.promote_types(d.dtype, compute) == compute for d in d_accs)
    if _reads_accumulators(epilogue, count, acc_shape, epilogue_arrays):
        accs = _as_tuple(_gemm(lhs, matrices[0], group_sizes, prologue, prologue_arrays, prologue_axes, None,
                               (), (), (jnp.dtype(jnp.float32),) * count, several=count > 1,
                               paired=matrices[1] if count > 1 else None, summed_as=dtypes, **plain))
    else:
        accs = (jnp.zeros(acc_shape, jnp.float32),) * count
    outs, pull = jax.vjp(functools.partial(_tupled, epilogue), *accs, *epilogue_arrays)
    pulled = pull(tuple(c.astype(out.dtype) for c, out in zip(cotangents, outs, strict=True)))
    narrow = tuple(_passes_cotangent(epilogue, count, at, acc_shape, epilogue_arrays, dtypes, compute)
                   for at in range(count))
    return tuple(pulled[:count]), tuple(pulled[count:]), narrow


def _input_cotangents(lhs, d_products: tuple, matrices: tuple, group_sizes, prologue, prologue_arrays,
                      prologue_axes, compute, plain: dict) -> tuple:
    """`lhs`'s and the prologue side operands' cotangents from the cotangent of the operand the forward
    multiplied (`prologue(lhs)` rounded to `lhs`'s dtype): each matrix's product with its accumulator's
    cotangent, one matrix's rounded once to `lhs`'s dtype, a paired product's two summed wide and rounded
    once (the unfused program's law: each projection's cotangent rounded, their products added), each
    summed as the gradient's own dtype sums (`_gemm`'s `summed_as`)."""
    out = (jnp.dtype(lhs.dtype),) if len(matrices) == 1 else (at_least_fp32(lhs.dtype),)
    options = {'interpret_on_cpu': plain['interpret_on_cpu'], 'precision': plain['precision'],
               'transpose_rhs': not plain['transpose_rhs'], 'dense': plain['dense']}
    d_operand = functools.reduce(jnp.add, (jnp.asarray(_gemm(d, matrix.astype(compute), group_sizes, None, (),
                                                             (), None, (), (), out, several=False,
                                                             summed_as=(lhs.dtype,), **options))
                                           for d, matrix in zip(d_products, matrices, strict=True)))
    if prologue is None:
        return d_operand.astype(lhs.dtype), ()
    d_lhs, d_prologue = _prologue_cotangents(prologue, lhs, prologue_arrays, prologue_axes, group_sizes,
                                             d_operand)
    return d_lhs.astype(lhs.dtype), tuple(_as_cotangent(d, a) for d, a in zip(d_prologue, prologue_arrays,
                                                                                strict=True))


def _kernel_gradient(lhs, d_product, matrix, group_sizes, prologue, prologue_arrays, prologue_axes, compute,
                     work, plain: dict):
    """A matrix's kernel gradient from its accumulator's cotangent `d_product`: the rows multiplied as the
    forward multiplied them (through the prologue, fused, where the cotangent is in `compute`'s dtype; the
    forward's rounded operand in the cotangent's wider dtype `work` otherwise)."""
    options = {'interpret_on_cpu': plain['interpret_on_cpu'], 'precision': plain['precision']}
    sizes = None if plain['dense'] else group_sizes
    if work is None:
        gradient = gemm_t(lhs.astype(compute), d_product, sizes, prologue=prologue,
                          prologue_operands=tuple(zip(prologue_arrays, prologue_axes, strict=True)),
                          out_dtype=matrix.dtype, **options)
    else:
        operand = (lhs.astype(compute) if prologue is None
                   else prologue(lhs, *prologue_arrays).astype(lhs.dtype))
        gradient = gemm_t(operand.astype(work), d_product, sizes, out_dtype=matrix.dtype, **options)
    return jnp.swapaxes(gradient, 1, 2) if plain['transpose_rhs'] else gradient


def _apart(lhs, matrices: tuple, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue,
           epilogue_arrays, dtypes, *, several: bool, summed_as: tuple, **options):
    """A paired product no launch takes whole (`grouped_matmul.APART`): each matrix's product through the
    launches that take it (`_gemm`), summed as the call's outputs sum (`summed_as`), then the epilogue in
    XLA on the two fp32 accumulators, as the paired launch's and XLA's program give it them (so a forward,
    its recomputation and its tangent read the same values); without an epilogue, each product its own
    output. Where XLA's ragged dot would run the pair instead, it ran 70 to 200 times the launches' time
    (GMM-h100-2)."""
    wide = at_least_fp32(lhs.dtype)
    kept = (wide,) * len(matrices) if epilogue is not None else tuple(dtypes)
    products = tuple(_gemm(lhs, matrix, group_sizes, prologue, prologue_arrays, prologue_axes, None, (), (),
                           (dtype,), several=False, summed_as=summed_as, **options)
                     for matrix, dtype in zip(matrices, kept, strict=True))
    if epilogue is not None:
        products = _tupled(epilogue, *products, *epilogue_arrays)
    outs = tuple(out.astype(dtype) for out, dtype in zip(products, dtypes, strict=True))
    return outs if several else outs[0]


def _portable_gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, epilogue, epilogue_arrays, dtypes,
                   several: bool, transpose_rhs: bool, precision: PrecisionLike, compute=None,
                   paired: jax.Array | None = None):
    """`gemm`'s arithmetic in XLA: the kernels' products of `lhs`'s dtype summed in at least fp32 (an fp64
    product's in fp64, which no kernel takes), one rounding;
    with `compute`, the products of the operands rounded to it, its tangent the MoE projection's
    (`grouped_matmul.portable_projection`, whose sum, at least fp32, it reads); with `paired`, both matrices'
    products, the epilogue's two accumulators."""
    matrices = tuple(jnp.swapaxes(each, 1, 2) if transpose_rhs else each
                     for each in (rhs, *(() if paired is None else (paired,))))
    if compute is not None:
        # Summed and kept at least fp32 (an fp64 product's in fp64), the accumulator the epilogue reads.
        wide = at_least_fp32(compute)
        accs = tuple(grouped_matmul.portable_projection(lhs, matrix, group_sizes, compute, precision, wide)
                     for matrix in matrices)
    else:
        product = lhs if prologue is None else prologue(lhs, *prologue_arrays).astype(lhs.dtype)
        wide = at_least_fp32(lhs.dtype)
        accs = tuple(grouped_matmul.xla_ragged_dot(product, each.astype(lhs.dtype), group_sizes,
                                                   precision=precision, preferred_element_type=wide)
                     for each in matrices)
    outs = accs if epilogue is None else _as_tuple(epilogue(*accs, *epilogue_arrays))
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


def _reads_accumulators(epilogue, count: int, acc_shape: tuple, sides: tuple) -> bool:
    """Whether the epilogue's VJP reads any of its `count` accumulators: a bias's does not, a product with
    a side operand's does. Read from the VJP's own program, shapes alone."""
    closed, _ = _epilogue_vjp(epilogue, count, acc_shape, sides)
    return bool(set(closed.jaxpr.invars[:count]) & _needed(closed.jaxpr, closed.jaxpr.outvars))


def _passes_cotangent(epilogue, count: int, at: int, acc_shape: tuple, sides: tuple, dtypes: tuple,
                      narrow) -> bool:
    """Whether the epilogue's VJP hands its accumulator `at` (of `count`) one output's cotangent unchanged
    (a bias's, a cast's) whose dtype `narrow` holds exactly: read from the VJP's own program, its result
    traced back through conversions alone."""
    closed, cotangents = _epilogue_vjp(epilogue, count, acc_shape, sides)
    var = closed.jaxpr.outvars[at]
    producers = {out: equation for equation in closed.jaxpr.eqns for out in equation.outvars}
    while var in producers and producers[var].primitive.name == 'convert_element_type':
        var = producers[var].invars[0]
    if var not in cotangents:
        return False
    return jnp.promote_types(dtypes[cotangents.index(var)], narrow) == jnp.dtype(narrow)


def _epilogue_vjp(epilogue, count: int, acc_shape: tuple, sides: tuple):
    """The epilogue's VJP as a program of its `count` accumulators, its side operands and its outputs'
    cotangents, and the cotangents' variables in it."""
    tupled = functools.partial(_tupled, epilogue)

    def pulled(accs, sides, cotangents):
        return jax.vjp(tupled, *accs, *sides)[1](cotangents)

    accs = (jax.ShapeDtypeStruct(acc_shape, jnp.float32),) * count
    outs = jax.eval_shape(tupled, *accs, *sides)
    closed = jax.make_jaxpr(pulled)(accs, sides, outs)
    return closed, list(closed.jaxpr.invars[count + len(sides):])


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
    (`grouped_matmul._on_platform`). Evaluated eagerly, `platform_dependent` knows its platform and runs
    its branch alone, so a CPU's would skip the selection the kernel's branch makes (its routes, a named
    kernel's refusal): an eager call is first traced abstractly as a jit traces it, every branch
    selecting, then runs its platform's branch eagerly, op by op as any eager call does: run under a jit
    of its own (83d19a780), a quantized block's reload no longer computed its logits bit for bit
    (test_wrapped_runs)."""
    branches, default = grouped_matmul._on_platform(kernel, portable, interpret_on_cpu)

    def dispatched(*operands):
        return jax.lax.platform_dependent(*operands, default=default, **branches)

    from jax.core import Tracer

    if not any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(operands)):
        jax.eval_shape(dispatched, *operands)
    return dispatched(*operands)
