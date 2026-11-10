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
axis it follows ('rows', 'depth' for the prologue's; 'rows', 'columns' for the epilogue's), 'groups' for
one value a group, or None for a dimension of one the hook broadcasts; the kernels load each side tile at
their own origins, a `groups` axis at the tile's group, handing the hook each row its group's value, and
the hook broadcasts a vector itself (`ms[:, None]`). The epilogue applies to every row: a row past the
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

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import PrecisionLike
from jax.custom_derivatives import zero_from_primal
from jax.extend import core as jax_core

from dew import _cache
from dew.algebra import bits
from dew.nn.derivatives import KernelGlue, Surrogate, contracted, declared_derivative, declared_tangent
from dew.nn.kernels import generation, grouped_matmul, selection
from dew.nn.kernels.templates import axes as side_axes_of, product as formats_of, swizzling
from dew.nn.kernels.templates.axes import GROUPS
from dew.nn.kernels.templates.gemm import mosaic_gpu, rows_padded, scattered
from dew.nn.kernels.templates.product import EXACT_PARTS, SCALE_DEPTH, declared, formed, parts, terms
from dew.nn.precision import (
    accumulated,
    asks_default_precision,
    at_least_fp32,
    exact_products,
    projected,
    refuse_recipe,
)
from dew.nn.sharding import explicit_spec, seen_whole

LAUNCH_AXES = {'prologue': ('rows', 'depth'), 'epilogue': ('rows', 'columns')}
"""The launch axes each hook's side operands may follow; `groups` too beside them (`_split`), which a launch
reads at its tile's group (`axes.GROUPS`) and XLA's program at each row (`axes.at_rows`)."""


def gemm(lhs: jax.Array, rhs: "jax.Array | NvFp4", group_sizes: jax.Array | None = None, *,
         transpose_rhs: bool = False, prologue=None, prologue_operands: tuple = (), epilogue=None,
         epilogue_operands: tuple = (), out_dtype, out_dtypes: tuple | None = None,
         precision: PrecisionLike = None, interpret_on_cpu: bool = False, prologue_jvp=None, compute=None,
         layer: jax.Array | None = None, paired: jax.Array | None = None, gather: jax.Array | None = None,
         scatter: jax.Array | None = None):
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
    matrix for a dense product), read in place: the launches take the layer's groups and read their
    matrices where they lie in the stack, from the layer's first (`_grouped`), so no slice of the stack
    is made and they tile, split and sum as the call over the layer's matrices alone. Its gradient is
    the stack's, every other layer's zero. `paired`, a second matrix
    of `rhs`'s shape, the rows multiply into a second accumulator the epilogue takes before its side
    operands (`epilogue(acc, paired_acc, *sides)`: a gate's and an up projection's SwiGLU), else an output.
    `gather`, int32 `[rows]` of a grouped call with no prologue or stacked layer, makes the product's rows
    `lhs[gather]`, read where they lie: a mixture of experts' tokens in their own order, the router's sort
    fused into the rows' loads where Mosaic GPU's launch takes the call (its producer's gather, row 1376),
    gathered by XLA elsewhere; the rows' cotangent is summed back into each token's (`lhs`'s, a token every
    slot reads). `scatter`, int32 `[rows]` of a grouped call, a permutation, writes the product's row `i` at
    each output's row `scatter[i]`: the slots in token order, the un-permute fused into Mosaic GPU's stores
    (`mosaic.Scattered`), written by XLA elsewhere; the outputs' cotangents are read back through it, a
    gather. An NVFP4 `rhs` (`NvFp4`; a stack of banks at its `layer`, row 4861) is `_nvfp4_gemm`'s, frozen."""
    if isinstance(rhs, NvFp4):
        return _nvfp4_gemm(lhs, rhs, group_sizes, epilogue=epilogue, epilogue_operands=epilogue_operands,
                           out_dtype=out_dtype, layer=layer, gather=gather, scatter=scatter,
                           interpret_on_cpu=interpret_on_cpu,
                           refused=(transpose_rhs, prologue, prologue_operands, paired, prologue_jvp,
                                    out_dtypes))
    _gemm_refused(group_sizes, prologue=prologue, layer=layer, paired=paired, gather=gather, scatter=scatter,
                  compute=compute)
    straight = compute is not None
    compute = jnp.dtype(lhs.dtype if compute is None else compute)
    dense = group_sizes is None and layer is None
    prologue = _prologued(prologue, prologue_jvp)
    paired = None if paired is None else _grouped(lhs, paired, group_sizes, layer)[0]
    prologue_arrays, prologue_axes = _split(prologue_operands, 'prologue', group_sizes)
    epilogue_arrays, epilogue_axes = _split(epilogue_operands, 'epilogue', group_sizes)
    rhs, group_sizes, first = _grouped(lhs, rhs, group_sizes, layer)
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    several = out_dtypes is not None
    plain = {'transpose_rhs': transpose_rhs, 'interpret_on_cpu': interpret_on_cpu, 'precision': precision,
             'dense': dense}

    def value(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first, gather, scatter):
        return _gemm(lhs.astype(compute), rhs, group_sizes, prologue, prologue_arrays, prologue_axes,
                     epilogue, epilogue_arrays, epilogue_axes, dtypes, several=several, paired=paired,
                     first=first, gather=gather, scatter=scatter, **plain)

    def portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first, gather, scatter):
        config = XlaConfig(transpose_rhs, precision, compute if straight else None, dense)
        lhs = lhs if gather is None else lhs[gather]
        return scattered(xla_mainloop(config, lhs if straight else lhs.astype(compute), rhs, group_sizes,
                                       prologue, prologue_arrays, prologue_axes, epilogue, epilogue_arrays,
                                       epilogue_axes, dtypes, several, paired=paired, first=first), scatter)

    def forward(*operands):
        return value(*operands), ()

    def backward(saved, operands, cotangent, live):
        return _pulled(operands, cotangent, live)

    def _pulled(operands, cotangent, live):
        tokens, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first, gather, scatter = operands
        # Gathered rows are read where they lie (row 1376), their cotangent summed back into the tokens.
        cotangent, read = _scattered_cotangent(cotangent, scatter, epilogue=epilogue, first=first)
        # The operands a reverse pass differentiates, by their place among the operands' leaves: a frozen
        # matrix's kernel gradient, or an operand no reverse pass reaches, is never formed.
        wanted = set(live)
        held = {**plain, 'first': first}
        matrices = (rhs,) if paired is None else (rhs, paired)
        places = (1, 3 + len(prologue_arrays) + len(epilogue_arrays))[:len(matrices)]
        cotangents = cotangent if several else (cotangent,)
        acc_shape = ((tokens if gather is None else gather).shape[0], rhs.shape[1 if transpose_rhs else 2])
        d_accs, d_epilogue, narrow = _cotangents(tokens.astype(compute), matrices, group_sizes, prologue,
                                                 prologue_arrays, prologue_axes, epilogue, epilogue_arrays,
                                                 epilogue_axes, cotangents, dtypes, compute, acc_shape,
                                                 {**held, 'gather': gather})
        # Each accumulator's cotangent meets the forward's rounded operands as the portable program's
        # derivative has it: in `lhs`'s dtype where that holds it exactly (a cotangent no wider, passed
        # through), wide otherwise, each product rounding once.
        works = tuple(compute if each else at_least_fp32(d.dtype)
                      for d, each in zip(d_accs, narrow, strict=True))
        d_products = tuple(d.astype(work) for d, work in zip(d_accs, works, strict=True))
        d_lhs, d_prologue = None, ()
        if wanted & set(range(3 + len(prologue_arrays))) - {1, 2}:
            d_lhs, d_prologue = _input_cotangents(tokens, d_products, matrices, group_sizes, prologue,
                                                  prologue_arrays, prologue_axes, compute,
                                                  {**held, 'gather': read})
            if gather is not None:
                d_lhs = jnp.zeros(tokens.shape, d_lhs.dtype).at[gather].add(d_lhs)
        gradients = [None if place not in wanted
                     else _kernel_gradient(tokens, d_product, matrix, group_sizes, prologue, prologue_arrays,
                                           prologue_axes, compute, work if not each else None,
                                           {**held, 'tables': (gather, read)})
                     for place, d_product, matrix, work, each in zip(places, d_products, matrices, works,
                                                                     narrow, strict=True)]
        leaves = [d_lhs, gradients[0], zero_from_primal(group_sizes),
                  *(d_prologue or (None,) * len(prologue_arrays)),
                  *(_as_cotangent(d, a) for d, a in zip(d_epilogue, epilogue_arrays, strict=True)),
                  *gradients[1:], *(() if first is None else (zero_from_primal(first),)),
                  *(() if gather is None else (zero_from_primal(gather),)),
                  *(() if scatter is None else (zero_from_primal(scatter),))]
        return tuple(leaves[index] for index in live)

    differentiable = contracted("test_a_grouped_gemm_holds_every_order")(
        generation.linear_tangent(forward, grouped_matmul.GROUPED.bound(backward), portable,
                                   live_cotangents=True, kind=KernelGlue()))
    return differentiable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first, gather,
                          scatter)


def _scattered_cotangent(cotangent, scatter, *, epilogue, first) -> tuple:
    """A scattered product's cotangent (`gemm`'s `scatter`) and the table its gradients' launches read it
    through: read where its rows landed (row 1376), or gathered here for an epilogue's derivative, which
    reads it row by row, or a stack's input gradient (no gathered product reads a stack), no table then."""
    if scatter is None or (epilogue is None and first is None):
        return cotangent, scatter
    return jax.tree.map(lambda d: d[scatter], cotangent), None


def _gemm_refused(group_sizes, *, prologue, layer, paired, gather, scatter, compute) -> None:
    """`gemm`'s refusal of a call it does not take: a pair beside a stacked layer, gathered rows but a
    grouped call's with no prologue or stacked layer, scattered rows but a grouped call's, a prologue beside
    a named `compute`."""
    if paired is not None and layer is not None:
        raise ValueError("a paired product takes no stacked layer")
    if gather is not None and (group_sizes is None or prologue is not None or layer is not None):
        raise ValueError("gathered rows are a grouped call's, with no prologue or stacked layer")
    if scatter is not None and group_sizes is None:
        raise ValueError("scattered rows are a grouped call's")
    if compute is not None and prologue is not None:
        raise ValueError("a prologue's operand is multiplied in its own dtype, not a named `compute`")


def _nvfp4_gemm(lhs, matrix: "NvFp4", group_sizes, *, epilogue, epilogue_operands: tuple, out_dtype, layer,
                gather, scatter, interpret_on_cpu: bool, refused: tuple) -> jax.Array:
    """`gemm` of an NVFP4 `matrix` (a matrix, or a bank of experts' with `group_sizes`, or a stack of banks
    at `layer`: `nvfp4_product`'s fp32 sums times the deferred factor, frozen), its epilogue on them in XLA
    before the one rounding, its rows gathered and scattered as any, one output; none of `refused` (a
    transposed matrix, a prologue, its side operands, a pair, a prologue's JVP, several outputs) given."""
    if any(given not in (None, False, ()) for given in refused):
        raise ValueError("an NVFP4 matrix's product takes no prologue, pair, transposed matrix or several "
                         "outputs")
    summed = nvfp4_product(lhs, matrix, group_sizes, gather=gather, layer=layer, out_dtype=jnp.float32,
                           interpret_on_cpu=interpret_on_cpu)
    if epilogue is not None:
        arrays, axes = _split(epilogue_operands, 'epilogue', group_sizes)
        sizes = jnp.asarray([summed.shape[0]], jnp.int32) if group_sizes is None else group_sizes
        [summed] = _tupled(epilogue, summed, *_rows(arrays, axes, sizes, summed.shape[0]))
    return scattered(summed.astype(out_dtype), scatter)


def gemm_t(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None, *, prologue=None,
           prologue_operands: tuple = (), out_dtype, precision: PrecisionLike = None,
           interpret_on_cpu: bool = False, gather: tuple | None = None) -> jax.Array:
    """Each group's rows of `prologue(lhs)` transposed times its rows of `rhs`: `[groups, depth, width]`
    in `out_dtype`, an empty group's zero, summed as `gemm`'s `precision` asks; without `group_sizes`,
    every row one group, a dense product's (`[1, depth, width]`). A value, as `gemm`'s reverse mode
    takes it. `gather`, a grouped call's table for either operand (`(lhs_table, rhs_table)`, int32
    `[rows]` or None), makes the rows that operand's rows at its entries, read where they lie (`gemm`'s
    `gather`: the reverse pass of gathered tokens and of a scattered product's cotangent, row 1376), with
    no prologue on gathered rows."""
    if gather is not None and (group_sizes is None or (gather[0] is not None and prologue is not None)):
        raise ValueError("gathered rows are a grouped call's, with no prologue on the gathered rows")
    prologue_arrays, prologue_axes = _split(prologue_operands, 'prologue', group_sizes)
    return _gemm_t(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, out_dtype=out_dtype,
                   precision=precision, interpret_on_cpu=interpret_on_cpu, gather=gather)


def _gemm_t(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, *, out_dtype,
            precision: PrecisionLike, interpret_on_cpu: bool, gather: tuple | None = None) -> jax.Array:
    """`gemm_t` of side operands checked (`_split`), its groups the call's, each operand multiplied whole: in
    the wider of their dtypes (a wider cotangent never rounded to the rows')."""
    dense = group_sizes is None
    compute = jnp.promote_types(lhs.dtype, rhs.dtype)
    group_sizes = jnp.full((1,), lhs.shape[0], jnp.int32) if dense else group_sizes.astype(jnp.int32)
    tables = (None, None) if gather is None else tuple(gather)

    def gathered(lhs, rhs, tables):
        # Each operand's rows at its table's entries, by XLA.
        return tuple(operand if table is None else operand[table]
                     for operand, table in zip((lhs, rhs), tables, strict=True))

    def kernel(lhs, rhs, group_sizes, prologue_arrays, tables, *, interpret):
        tabled = any(table is not None for table in tables)
        # An fp32 product asked above the default precision runs as its operands' bf16 parts (`_in_parts`),
        # each operand's values its own dtype's.
        counts = _part_counts(lhs.dtype, rhs.dtype, precision)
        parted = declared(jnp.bfloat16, precision, outputs=(jnp.float32,))
        sides, axes, broadcast = _launchable(prologue_arrays, prologue_axes)
        # A prologue rewrites the rows as they are, so the rows run whole, one part: their own 16 bits.
        hooked = grouped_matmul.Hooks(prologue, axes, prologue_broadcast=broadcast)
        if (counts is not None and (prologue is None or counts[0] == 1) and _parts_launched(
                grouped_matmul.tgmm_call(lhs.astype(jnp.bfloat16), rhs.astype(jnp.bfloat16), group_sizes,
                                         sides, tables if tabled else None, out_dtype=jnp.float32,
                                         hooks=hooked, interpret=interpret, promoted=parted.promoted,
                                         dense=dense),
                parted, whole=seen_whole(lhs) and seen_whole(rhs))):
            def part(left, right, finite, epilogue, prior, axes, dtypes):
                # A kernel gradient takes no epilogue: the earlier parts' sum is added after. The rows' finite
                # values are read after their prologue, which rewrites them.
                del epilogue, axes, dtypes
                rewrite = (prologue if not finite[0] or prologue is None
                           else functools.partial(_finite_after, prologue))
                left = _finite(left) if finite[0] and prologue is None else left
                right = _finite(right) if finite[1] else right
                out = _gemm_t(left, right, None if dense else group_sizes, rewrite, prologue_arrays,
                              prologue_axes, out_dtype=jnp.float32, precision=precision,
                              interpret_on_cpu=interpret_on_cpu, gather=tables if tabled else None)
                return out + prior[0] if prior else out
            wide = (jnp.dtype(jnp.float32),)
            return _in_parts(part, lhs, rhs, counts, None, (), (), wide).astype(out_dtype)
        # The rows widen to the product's dtype after the prologue, which reads and rounds them in their own
        # (as the portable program's: `prologue(lhs).astype(lhs.dtype)`, then the product).
        rewrite = (prologue if prologue is None or compute == jnp.dtype(lhs.dtype)
                   else functools.partial(_in_dtype, prologue, jnp.dtype(lhs.dtype)))
        hooks = grouped_matmul.Hooks(rewrite, axes, prologue_broadcast=broadcast)
        product = declared(compute, precision, outputs=(out_dtype,), operands=(lhs.dtype, rhs.dtype))
        call = grouped_matmul.tgmm_call(lhs.astype(compute), rhs.astype(compute), group_sizes, sides,
                                        tables if tabled else None, out_dtype=out_dtype,
                                        hooks=hooks, interpret=interpret, promoted=product.promoted,
                                        dense=dense)
        launch = grouped_matmul.launched(*call, product=product, whole=seen_whole(lhs) and seen_whole(rhs))
        # A kernel gradient is one product: no launch takes it apart.
        if callable(launch):
            return launch()
        if tabled:
            # No launch reads the rows where they lie (an fp32 call's, which only Mosaic GPU's gathering
            # producer would): the rows gathered first, the grouped launches taking the product, never XLA's
            # ragged dot over every group's matrix (row 5213's, as `_gemm`'s).
            rows, columns = gathered(lhs, rhs, tables)
            return kernel(rows, columns, group_sizes, prologue_arrays, (None, None), interpret=interpret)
        return portable(lhs, rhs, group_sizes, prologue_arrays, tables)

    def portable(lhs, rhs, group_sizes, prologue_arrays, tables):
        lhs, rhs = gathered(lhs, rhs, tables)
        routed = _routed(group_sizes, lhs.shape[0])
        if prologue is not None:
            sides = _rows(prologue_arrays, prologue_axes, group_sizes, lhs.shape[0])
            lhs = prologue(lhs, *sides).astype(lhs.dtype)
        lhs = jnp.where(routed[:, None], lhs, 0)
        wide = at_least_fp32(compute)  # an fp64 product's sum in fp64
        # Each group's rows contracted, XLA's ragged dot's own weight gradient (its transpose's form) on the
        # operands as they are: a pullback from a matrix made here would hand a vmapped call (a pipeline
        # stage's) an unbatched operand, which ragged_dot's batching rule refuses. The operands keep their
        # dtype, summed in `wide`: a bf16 product is exact in fp32 either way, where operands widened first
        # made the GPU's product an fp32 one (an sm120's 4096-token gate_up weight gradient 5.91 ms against
        # 2.72 for the bf16 product).
        # A sum over rows an Explicit axis splits lands whole (`ragged_placement`), the depth and width placed
        # as the rows' and the cotangent's columns are. A dense call's is XLA's own dot over the rows, as the
        # forward's (`grouped_matmul.xla_dot`): its ragged dot of one group took an A100's gate_up weight
        # gradient 470 ms against the template's 11.4 (W121-a100), where DENSE1 timed XLA's dot.
        # Each summed as `precision.accumulated` sums a contraction (a CPU's float32 product as a vectorized
        # dot sums it, row 4863).
        if dense:
            product = functools.partial(jax.lax.dot_general, dimension_numbers=(((0,), (0,)), ((), ())),
                                        precision=precision,
                                        out_sharding=grouped_matmul.ragged_placement(((lhs, 1), (rhs, 1))))
            return accumulated(product, lhs.astype(compute), rhs.astype(compute),
                               preferred_element_type=wide)[None].astype(out_dtype)
        product = functools.partial(
            jax.lax.ragged_dot_general, ragged_dot_dimension_numbers=_CONTRACTED_ROWS, precision=precision,
            out_sharding=grouped_matmul.ragged_placement((None, (lhs, 1), (rhs, 1))))
        return accumulated(product, lhs.astype(compute), rhs.astype(compute), group_sizes,
                           preferred_element_type=wide).astype(out_dtype)

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays, tables)


LOW_RANK_ROW = 'ledger 1304'
"""The adapted product's fused form (docs/design/kernel-templates.md, Appendix C row 26)."""

LOW_RANK: selection.Op[None] = selection.Op(generation.Choice.LOW_RANK, (
    selection.Kernel('fused', (selection.Requirement(
        'measured', lambda case: False, "no receipt yet has the fused form beat the base product and the "
        "update apart", LOW_RANK_ROW, kind='measured'),)),
    selection.Kernel('xla', (), generic=True)))
"""An adapted projection's selector (`low_rank`'s caller, `dew.lora`): one template product over the depth
the rank extends (`low_rank`) where named (`generation.chosen(low_rank='fused')`), which passes the
measured requirement no receipt has met yet; else its base product and its update apart, the adapter's own
program."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NvFp4:
    """An NVFP4 matrix `[depth, width]` as the GEMM stores it (row 3621), K-major, each column's depths
    along its row (as a Dense kernel read transposed): its E2M1 codes eight a uint32 in the GEMM's staged
    depth order (`words`, `[width, depth / 8]`: `staged_rows`, each word the
    `bits.NVFP4_BFLOAT16_FORM.prepared` word of its eight codes), each sixteen consecutive depths' E4M3
    scale (`scales`, `[width, depth / 16]`, in the depths' own order), and the tensor's fp32 scale
    (`scale`, a scalar): each value its code times its block's scale times the tensor's. Its depth is whole
    stages (`NVFP4_STAGE`): the mainloop's stage decode reads each column's words and scales as rows a TMA
    copy takes, at least 16 bytes. A bank of experts' matrices (`[groups, depth, width]`, row 4516) leads
    each field with its group axis: `words` `[groups, width, depth / 8]`, `scales` `[groups, width,
    depth / 16]`, `scale` `[groups]`, each expert's matrix its own tensor scale. Its fields are NumPy
    arrays where it was packed on the host (`of`)."""

    words: jax.Array | np.ndarray
    scales: jax.Array | np.ndarray
    scale: jax.Array | np.ndarray

    @classmethod
    def quantized(cls, x: jax.Array) -> "NvFp4":
        """`x` (`[depth, width]`, or a bank's `[groups, depth, width]`) quantized to NVFP4
        (`bits.nvfp4_quantized`, each matrix: its block scales and tensor scale, each value rounded to E2M1),
        its codes packed in the staged order."""
        def one(x):
            words, scales, scale = bits.nvfp4_quantized(jnp.asarray(x, jnp.float32))
            return _codes(words), scales, scale

        return cls.of(*(jax.vmap(one) if x.ndim == 3 else one)(x))

    @classmethod
    def of(cls, codes, scales, scale) -> "NvFp4":
        """The matrix of `codes` (`[depth, width]`, or a bank's `[groups, depth, width]`, in the depths' own
        order: each an E2M1 value, or its 4-bit code as a uint8 nibble, its sign bit 3), its block scales
        `scales` (`[..., depth / 16, width]` float8_e4m3fn) and its tensor scale `scale` (one a matrix): a
        checkpoint's NVFP4 weight, its codes packed in the staged order, K-major. NumPy codes, scales and
        scale give a NumPy matrix (packed on the host, nothing copied to a device: a load places each
        expert shard's itself)."""
        depth = codes.shape[-2]
        if depth % NVFP4_STAGE:
            raise ValueError(f"an NVFP4 matrix's depth is whole {NVFP4_STAGE}s, not {depth}")
        host = isinstance(codes, np.ndarray) and isinstance(scales, np.ndarray)
        xp = np if host and not isinstance(scale, jax.Array) else jnp
        words = _staged_words(codes, xp)
        return cls(xp.swapaxes(words, -1, -2), xp.swapaxes(scales, -1, -2), xp.asarray(scale, xp.float32))

    def codes(self) -> jax.Array:
        """Its codes' values (`[..., depth, width]` float32, each code's E2M1 value) in the depths' own
        order."""
        words = jnp.swapaxes(self.words, -1, -2)
        staged = functools.partial(_codes, staged=True)
        # Each matrix's own, over every leading axis: a bank's groups, a stack of banks' layers before them.
        for _ in range(words.ndim - 2):
            staged = jax.vmap(staged)
        return staged(words)

    def decoded(self) -> jax.Array:
        """Its values over `bits.NVFP4_OWED` and the tensor scale, exactly, in bfloat16 (`[..., depth,
        width]`): each code times its block's scale over 2^8, as the mainloop's stage decode forms it."""
        owed = jnp.swapaxes(self.scales, -1, -2).astype(jnp.float32) * jnp.float32(1 / bits.NVFP4_OWED)
        return (self.codes() * jnp.repeat(owed, 16, axis=-2)).astype(jnp.bfloat16)

    @property
    def dtype(self) -> jnp.dtype:
        """The dtype its decode yields (`decoded`), which it is multiplied as: bfloat16."""
        return jnp.dtype(jnp.bfloat16)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the matrix it stands for: `[depth, width]`, a bank's `[groups, depth, width]`."""
        *groups, width, words = self.words.shape
        return (*groups, 8 * words, width)

    def layer(self, index: jax.Array) -> "NvFp4":
        """A stack of banks' bank at layer `index` (a traced index): its fields led by a layers axis before
        their group axis (a scanned load's stacked banks, row 4861), each field's slice at it."""
        return NvFp4(*(jax.lax.dynamic_index_in_dim(jnp.asarray(field), index, keepdims=False)
                       for field in (self.words, self.scales, self.scale)))

    @property
    def factor(self) -> jax.Array:
        """The factor its decode defers to the product's accumulator (`bits.NVFP4_OWED` times the tensor's
        scale, a bank's one a matrix): `bits.nvfp4_decoded` keeps each value within bfloat16's normal range
        by owing 2^8."""
        return jnp.float32(bits.NVFP4_OWED) * jax.lax.stop_gradient(jnp.asarray(self.scale, jnp.float32))


NVFP4_STAGE = mosaic_gpu.NVFP4_STEP
"""The depths a stage of the mainloop's NVFP4 decode reads: two groups, so each column's scales are a
16-byte row (a TMA copy's least)."""


def staged_rows(depth: int) -> np.ndarray:
    """The depth each code of the staged order holds (`NvFp4`): packed row `8w + j` of a group (word `w`'s
    code `j`) holds the depth `mosaic_gpu.nvfp4_staged_depth` gives it, so a stage's word rows decode, code by
    code, into its sixteen-row depth slices, the block of one scale each (the mainloop's stage decode)."""
    group = mosaic_gpu.NVFP4_GROUP
    if depth % group:
        raise ValueError(f"an NVFP4 matrix's depth is whole {group}-deep groups, not {depth}")
    groups, within = np.divmod(np.arange(depth), group)
    word, code = np.divmod(within, 32 // bits.E2M1.width)
    return mosaic_gpu.nvfp4_staged_depth(groups, word, code)


def _codes(words: jax.Array, *, staged: bool = False) -> jax.Array:
    """The codes' values of `words` (each its E2M1 value, float32 `[depth, width]`), in the depths' own order
    where `staged` (`staged_rows`), else the words' own: `bits.nvfp4_decoded` at a block scale of 2^8."""
    unit = jnp.full((words.shape[0] // 2, words.shape[1]), bits.NVFP4_OWED, jnp.float8_e4m3fn)
    codes = bits.nvfp4_decoded(words, unit).astype(jnp.float32)
    return codes[np.argsort(staged_rows(codes.shape[0]))] if staged else codes


def _staged_words(codes, xp=jnp):
    """`codes` (`[..., depth, width]` in the depths' own order: E2M1 values, or uint8 nibbles, the codes
    themselves) packed in the staged order (`NvFp4`), uint32 `[..., depth / 8, width]`, by `xp`'s arrays
    (NumPy's on the host)."""
    ordered = codes[..., staged_rows(codes.shape[-2]), :]
    if xp.dtype(ordered.dtype) == xp.dtype(xp.uint8):
        nibbles = ordered.astype(xp.uint32)
    else:
        fp32 = ordered.astype(xp.float32)
        words = fp32.view(np.uint32) if xp is np else jax.lax.bitcast_convert_type(fp32, jnp.uint32)
        nibbles = bits.encoded(words, bits.E2M1, xp)
    return xp.swapaxes(bits.packed(xp.swapaxes(nibbles, -1, -2), bits.E2M1, xp), -1, -2)


NVFP4_ROW = 'ledger 3621'
"""The NVFP4 product's in-stage decode (K51)."""


@dataclasses.dataclass(frozen=True)
class PackedCase:
    """An NVFP4 product as its selector reads it (`NVFP4`): the tile the template's packed launch plans for
    it on the target (`grouped_matmul.MosaicPackedGmm`), None where it plans none, and its `rows`."""

    tile: mosaic_gpu.Tiles | None
    rows: int

    @property
    def cell(self) -> tuple[str, str]:
        """The cell a receipt measures (`selection.Celled`): the device's generation and whether the call is
        decode-sized (at most `NVFP4_DECODE_ROWS` rows, the packed matrix's bytes its time) or a prefill."""
        return (generation.device_generation(),
                'decode' if self.rows <= NVFP4_DECODE_ROWS else 'prefill')


NVFP4_DECODE_ROWS = 64
"""The most rows a decode-sized NVFP4 call has (`PackedCase.cell`): the receipts' largest decode row count."""

PACKED_FASTER = ("W13-a100-4-M-NS-8b-a100 (base/kernels 8d23bb9872, CUPTI, arms rotated): Llama-3-8B's "
                 "q/o, up and down projections at 1, 16 and 64 rows on an A100-SXM4-40GB, the packed kernel "
                 "0.119-0.398 ms against the matrix decoded once a call's 0.309-1.418 (2.6-3.6x), its "
                 "temporary memory 0.66-3.7 MB against 71-250 MB; at 4096 rows 4.4-4.7x slower, each row "
                 "tile decoding the matrix again")
"""The receipt behind the packed NVFP4 kernel's declared A100 cell (`NVFP4_CELLS`)."""
PACKED_FASTER_SM90 = (
    "W7-gh200-8-M-NS-8-h100 (base/kernels d4bb24d4f8, CUPTI, arms rotated): Llama-3-8B's q/o, up and down "
    "projections at 1, 16 and 64 rows on a GH200, the packed kernel 0.059-0.193 ms against the matrix "
    "decoded once a call's 0.142-0.505 (2.4x q/o, 4.5x up, 2.6x down), its temporary memory 0.2-3.7 MB "
    "against 71-250 MB; at 4096 rows 5.2-5.3x slower, its stage decode unhidden under the products")
"""The receipt behind the packed NVFP4 kernel's declared H100 cell (`NVFP4_CELLS`)."""

NVFP4_CELLS: dict[tuple[str | None, str], tuple[str, str]] = {
    ('sm80', 'decode'): ('packed', PACKED_FASTER),
    ('sm90', 'decode'): ('packed', PACKED_FASTER_SM90),
}
"""The NVFP4 product's declared cells (`selection.Op.declared`), (device generation, call size) to the
packed kernel and its receipt: a literal of them, which the selector reads and thresh's declared-set seam
registers (.thresh/thresh.cue)."""


NVFP4: selection.Op[PackedCase] = selection.Op(generation.Choice.NVFP4, (
    selection.Kernel('packed', (
        selection.Requirement(
            'planned', lambda case: case.tile is not None, "the template plans no tile of the packed "
            "product here: a Mosaic GPU family's, its depth whole 256s and its width whole 64-column tiles",
            NVFP4_ROW),
        )),
    selection.Kernel('xla', (), generic=True)), lambda case: 'xla', declared=NVFP4_CELLS)
"""An NVFP4 product's selector (`nvfp4_product`): the matrix decoded once a call and its product (`gemm`),
but the template's mainloop decoding each stage of the packed matrix in place (`mosaic_gpu.gmm`'s `scales`)
where a receipt measured it faster (a decode-sized call on an A100 or an H100: `NVFP4_CELLS`) or it is
named (`generation.chosen(nvfp4='packed')`), and it plans a tile."""


def nvfp4_product(lhs: jax.Array, matrix: NvFp4, group_sizes: jax.Array | None = None,
                  gather: jax.Array | None = None, *, layer: jax.Array | None = None, out_dtype,
                  interpret_on_cpu: bool = False) -> jax.Array:
    """`lhs · W` of an NVFP4 matrix `W` (`NvFp4`), its weights frozen (a quantized checkpoint's): `lhs`
    rounded to bfloat16, each value of `W` decoded exactly to bfloat16 (`NvFp4.decoded`, its block's
    scale applied, 2^8 owed), the products summed in fp32, each step's apart (`templates.product.promoted`,
    row 4331), and the accumulator times the deferred factor
    (`NvFp4.factor`) before the one rounding to `out_dtype`: the deferred factor is the epilogue's first
    operation, ahead of any output hook (PLAN piece 10). Its selector (`NVFP4`) runs the template's mainloop
    decoding each stage in place (`_packed`) where named, else decodes the matrix once a call (`gemm`).
    With `group_sizes`, `matrix` is a bank of experts' matrices (`NvFp4`'s group axis, row 4516) and `lhs`
    their sorted rows: each group's rows times its expert's matrix, each times its expert's factor, frozen
    (no gradient reaches the bank), each tile decoding its group's expert's stages in place on the packed
    kernel; with `gather` besides, its rows `lhs[gather]`, read where they lie (`gemm`'s `gather`). With
    `layer` (a traced index) instead, `matrix` is a stack of banks (`NvFp4.layer`: a scanned load's, row
    4861) read at that layer in place, as `gemm` reads a stacked matrix's (`gemm`'s `layer`): the packed
    kernel's groups read the layer's matrices from its first where they lie in the stack, no slice of it
    made, and elsewhere the layer's bank alone is decoded."""
    if layer is not None and (group_sizes is None or gather is not None):
        raise ValueError("a stack of banks' layer is a grouped call's, its rows gathered before it")
    bank = matrix if layer is None else matrix.layer(layer)
    rows = lhs.astype(jnp.bfloat16)
    launch = grouped_matmul.MosaicPackedGmm()
    words, scales = (jnp.asarray(field if group_sizes is not None else field[None])
                     for field in (matrix.words, matrix.scales))
    count = rows.shape[0] if gather is None else gather.shape[0]
    sizes = jnp.asarray([count], jnp.int32) if group_sizes is None else group_sizes.astype(jnp.int32)
    first = None
    if layer is not None:
        # The stack's every layer's matrices as one axis, the call's groups the layer's, from its first.
        words, scales = (field.reshape(-1, *field.shape[2:]) for field in (words, scales))
        first = jnp.asarray(layer, jnp.int32) * sizes.shape[0]
    def spec(array):
        return None if array is None else jax.ShapeDtypeStruct(jnp.shape(array), jnp.result_type(array))

    # The launch's matrix is the bank as it lies (`NvFp4`, each matrix's factor deferred to the epilogue):
    # its words and block scales, its shape and dtype the bfloat16 matrices it decodes to.
    bound = launch.bind(spec(rows), jax.tree.map(spec, _undeferred(words, scales)), spec(sizes), (), (), None,
                        spec(first), spec(gather), out_dtype='float32', trans_rhs=False,
                        interpret=interpret_on_cpu, dense=group_sizes is None, exact_sums=True)
    tile = _planned(launch, bound)
    if selection.select(NVFP4, PackedCase(tile, count)).kernel == 'packed':
        assert tile is not None, "the selector names the packed kernel where it plans a tile"
        summed = _packed(rows, words, scales, sizes, gather, first, launch, interpret_on_cpu,
                         dense=group_sizes is None)
    else:
        # Each product of two bfloat16 values is exact in fp32, so the default precision is the exact
        # product's, whatever precision the program defaults to.
        summed = gemm(rows, bank.decoded(), group_sizes, out_dtype=jnp.float32,
                      precision=jax.lax.Precision.DEFAULT, interpret_on_cpu=interpret_on_cpu, gather=gather)
        assert isinstance(summed, jax.Array), "an unpaired product has one output"
    factor = bank.factor if group_sizes is None else jnp.repeat(
        bank.factor, sizes, total_repeat_length=count)[:, None]
    return (summed * factor).astype(out_dtype)


def _undeferred(words, scales) -> NvFp4:
    """A bank's words and block scales (`NvFp4`'s, led by their group axis) as an `NvFp4` of unit tensor
    scales: the matrices the packed kernel decodes, each one's factor deferred to its product's epilogue."""
    return NvFp4(words, scales, jnp.ones((words.shape[0],), jnp.float32))


def _packed(rows: jax.Array, words: jax.Array, scales: jax.Array, sizes: jax.Array, gather, first,
            launch: grouped_matmul.MosaicPackedGmm, interpret: bool, *, dense: bool) -> jax.Array:
    """The packed matrices' product in fp32 (`nvfp4_product`'s packed kernel, `words` and `scales` led by
    their group axis, `sizes` each group's rows, `gather` the rows of `rows` they are where given, `first`
    the matrix of a stack the first group reads where given, `gmm`'s): the
    template's mainloop, each stage of `words` and `scales` decoded in place (`mosaic_gpu.gmm`'s `scales`),
    differentiable at every order as `gemm` is (`generation.linear_tangent`): its tangent the decoded
    product's in XLA, its reverse mode the rows' gradient through the decoded values, summed back into the
    rows a gather read (the frozen words, scales, sizes and table taking none). It runs through `launch`
    (`grouped_matmul.MosaicPackedGmm`), whose config a pin or a stored pick may name."""

    def decoded(words, scales, first):
        # The bank's matrices, a dense call's one alone (`gemm`'s dense matrix is `[depth, width]`), a stack's
        # layer's alone (its groups' from `first`).
        if first is not None:
            words, scales = (jax.lax.dynamic_slice_in_dim(field, first, sizes.shape[0])
                             for field in (words, scales))
        matrices = _undeferred(words, scales).decoded()
        return matrices[0] if dense else matrices

    def forward(rows, words, scales, sizes, gather, first):
        out = launch(rows, _undeferred(words, scales), sizes, (), (), None, first, gather,
                     out_dtype='float32', trans_rhs=False, interpret=interpret, dense=dense, exact_sums=True)
        return out, ()

    def backward(saved, operands, cotangent, live):
        # The cotangent at its own precision times the decoded values' transpose, rounded once to the rows',
        # each gathered row's summed back into the row it read: the fp32 cotangent as its bfloat16 parts
        # (`_bf16_parts`: their sum it exactly, a finite value past bfloat16's largest saturating into the
        # parts, not rounding to infinity), each part's product with the bfloat16
        # values exact and summed in fp32, the parts' sums added in fp32 (narrowed to bfloat16 first,
        # 1 - 511/512 summed to 0, audit A2).
        del saved
        rows, words, scales, sizes, gather, first = operands
        matrices = decoded(words, scales, first)
        parts = _bf16_parts(cotangent, EXACT_PARTS)
        d_rows = functools.reduce(jnp.add, [
            gemm(part, matrices, None if dense else sizes, transpose_rhs=True, out_dtype=jnp.float32,
                 precision=jax.lax.Precision.DEFAULT, interpret_on_cpu=interpret) for part in parts])
        if gather is not None:
            d_rows = jnp.zeros(rows.shape, d_rows.dtype).at[gather].add(d_rows)
        leaves = (d_rows.astype(rows.dtype), zero_from_primal(words), zero_from_primal(scales),
                  zero_from_primal(sizes), *(() if gather is None else (zero_from_primal(gather),)),
                  *(() if first is None else (zero_from_primal(first),)))
        return tuple(leaves[index] for index in live)

    def portable(rows, words, scales, sizes, gather, first):
        matrices = decoded(words, scales, first)
        return grouped_matmul.xla_ragged_dot(rows if gather is None else rows[gather],
                                             matrices if not dense else matrices[None], sizes,
                                             precision=jax.lax.Precision.DEFAULT,
                                             preferred_element_type=jnp.float32)

    differentiable = contracted("test_an_nvfp4_products_kernels_hold_every_order")(
        generation.linear_tangent(forward, backward, portable, live_cotangents=True, kind=KernelGlue()))
    return differentiable(rows, words, scales, sizes, gather, first)


BLOCKWISE_ROW = 'ledger 4518'
"""The blockwise-scaled FP8 product (K51)."""


@dataclasses.dataclass(frozen=True)
class BlockwiseCase:
    """A blockwise-scaled FP8 product as its selector reads it (`BLOCKWISE_FP8`): the tile the template's
    launch plans for it on the target (`mosaic_gpu.gmm`'s `step_scales`), None where it plans none."""

    tile: mosaic_gpu.Tiles | None


BLOCKWISE_FP8: selection.Op[BlockwiseCase] = selection.Op(generation.Choice.BLOCKWISE_FP8, (
    selection.Kernel('mosaic', (
        selection.Requirement(
            'planned', lambda case: case.tile is not None, "the template plans no step-scaled tile here: an "
            "FP8 product on warpgroup MMA, its matrix K-major, a step 128 deep, its column scales each a "
            "column's or a multiple of its tile's columns", BLOCKWISE_ROW),
        selection.Requirement(
            'measured', lambda case: False, "no receipt yet has the step-scaled template beat the scaled "
            "blocks' product in XLA", BLOCKWISE_ROW, kind='measured'))),
    selection.Kernel('xla', (), generic=True)))
"""A blockwise-scaled FP8 product's selector (`blockwise_fp8_product`): the template's exactly summed walk,
each step's partial scaled before its add (`mosaic_gpu.gmm`'s `step_scales`), where named
(`generation.chosen(blockwise_fp8='mosaic')`) and planned, past the measured requirement no receipt has
met yet; else the scaled blocks summed in XLA (`grouped_matmul.step_scaled`)."""


def blockwise_fp8_product(lhs: jax.Array, kernel: jax.Array, group_sizes: jax.Array, row_scales: jax.Array,
                          column_scales: jax.Array, *, out_dtype, interpret_on_cpu: bool = False
                          ) -> jax.Array:
    """Each group's rows of `lhs` (FP8 codes, `[rows, depth]`) times its matrix of `kernel` (FP8 codes,
    K-major `[groups, width, depth]`, as Hopper's FP8 tensor cores read it), each 128-deep block's partial,
    summed exactly in fp32, times its rows' scales (`row_scales`, `[depth / 128, rows]`) and its group's
    column scales (`column_scales`, `[groups, depth / 128, width / c]`, a scale every `c` columns: 128 a
    weight block's, 1 a column's) before it is added, rounded once to `out_dtype`: DeepSeek-V3's
    blockwise FP8 product (1 x 128 activation tiles, 128 x 128 weight blocks; row 4518). Its forward alone:
    a recipe's gradients are their own FP8 products of requantized operands (row 2513). Its selector
    (`BLOCKWISE_FP8`) runs the template's exactly summed walk where named, else XLA's."""
    launch = grouped_matmul.MosaicStepScaledGmm()
    sizes = group_sizes.astype(jnp.int32)
    call = functools.partial(launch, out_dtype='float32', trans_rhs=True, interpret=interpret_on_cpu,
                             exact_sums=True)
    per = kernel.shape[1] // column_scales.shape[-1]
    # A column scale covers each column, or whole weight blocks of them, which a column tile falls within.
    tile = (_planned(launch, launch.bind(*_abstract(lhs, kernel, sizes), **call.keywords,
                                         scales=_abstract(row_scales, column_scales)))
            if per == 1 or per % SCALE_DEPTH == 0 else None)
    if selection.select(BLOCKWISE_FP8, BlockwiseCase(tile)).kernel == 'mosaic':
        assert tile is not None, "the selector names the template where it plans a tile"
        summed = call(lhs, kernel, sizes, scales=(row_scales, column_scales))
    else:
        # Each product of two FP8 values is exact in bfloat16's range and in fp32.
        matrices = jnp.swapaxes(kernel, 1, 2).astype(jnp.bfloat16)
        summed = grouped_matmul.step_scaled(lhs.astype(jnp.bfloat16), matrices, group_sizes, row_scales,
                                            column_scales, precision=jax.lax.Precision.DEFAULT)
    return summed.astype(out_dtype)


def blockwise_fp8_kernel_gradient(lhs_t: jax.Array, rhs_t: jax.Array, group_blocks: jax.Array,
                                  row_scales: jax.Array, column_scales: jax.Array, *, out_dtype,
                                  interpret_on_cpu: bool = False) -> jax.Array:
    """Each group's rows of `lhs_t` (FP8 codes, `[depth, rows]`, the rows contiguous) times its rows of
    `rhs_t` (FP8 codes, `[width, rows]`), contracted over the rows: `[groups, depth, width]` in `out_dtype`,
    each 128-row block's partial, summed exactly in fp32, times its depth rows' scales (`row_scales`,
    `[rows / 128, depth]`) and its columns' (`column_scales`, `[rows / 128, width]`) before it is added to
    its group's sum: a blockwise FP8 recipe's kernel gradient (DeepSeek-V3's, row 4518). Group `g` holds
    `group_blocks[g]` whole 128-row blocks, in order (a recipe pads each group's rows to them, DeepGEMM's
    contiguous layout), so no block straddles groups. Its selector (`BLOCKWISE_FP8`) runs the template's
    rows-contiguous kernel gradient where named (`mosaic_gpu.tgmm`'s `k_major`), else XLA's."""
    sizes = (group_blocks * SCALE_DEPTH).astype(jnp.int32)
    launch = grouped_matmul.MosaicStepScaledTgmm()
    # Bound as the call it is, its operands' rows first as a kernel gradient states them: the rows-contiguous
    # operands' transposes, which XLA folds into the kernel's operands (`MosaicStepScaledTgmm`).
    call = functools.partial(launch, lhs_t.T, rhs_t.T, sizes, out_dtype='float32', interpret=interpret_on_cpu,
                             exact_sums=True, step_scales=(row_scales, column_scales))
    tile = _planned(launch, launch.bind(*_abstract(*call.args), **{
        **call.keywords, 'step_scales': _abstract(row_scales, column_scales)}))
    if selection.select(BLOCKWISE_FP8, BlockwiseCase(tile)).kernel == 'mosaic':
        assert tile is not None, "the selector names the template where it plans a tile"
        summed = call()
    else:
        summed = _blocks_summed(lhs_t, rhs_t, group_blocks, row_scales, column_scales)
    return summed.astype(out_dtype)


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


def _abstract(*arrays) -> tuple:
    """Each of `arrays` as its shape and dtype, as a launch's planning binds them (`_planned`)."""
    return tuple(jax.ShapeDtypeStruct(array.shape, array.dtype) for array in arrays)


def _planned(launch, bound) -> mosaic_gpu.Tiles | None:
    """The tile `launch` plans for the call it is `bound` to on the target (its derived tile, its allocations
    its kernel's), None where it takes none."""
    target = generation.gpu_target()
    return launch.derived(bound, target) if launch.eligible(bound, target) else None


def _blocks_summed(lhs_t, rhs_t, group_blocks, row_scales, column_scales) -> jax.Array:
    """`blockwise_fp8_kernel_gradient`'s portable arm: each 128-row block's product in fp32 (FP8 values exact
    in bfloat16, their products in fp32), times its scales, added in order into its group's fp32 sum."""
    depth, width, blocks = lhs_t.shape[0], rhs_t.shape[0], row_scales.shape[0]
    groups = group_blocks.shape[0]
    owner = jnp.repeat(jnp.arange(groups), group_blocks, total_repeat_length=blocks)

    def block(index, total):
        rows = jax.lax.dynamic_slice_in_dim(lhs_t, index * SCALE_DEPTH,
                                            SCALE_DEPTH, axis=1)
        columns = jax.lax.dynamic_slice_in_dim(rhs_t, index * SCALE_DEPTH,
                                               SCALE_DEPTH, axis=1)
        partial = jax.lax.dot_general(rows.astype(jnp.bfloat16), columns.astype(jnp.bfloat16),
                                      (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        scaled = partial * row_scales[index][:, None] * column_scales[index][None, :]
        return total.at[owner[index]].add(scaled)

    return jax.lax.fori_loop(0, blocks, block, jnp.zeros((groups, depth, width), jnp.float32))




@dataclasses.dataclass(frozen=True)
class MxFp8:
    """An MXFP8 operand (OCP MX: E4M3 codes, a power-of-two E8M0 scale every 32 depths, `product.MXFP8`):
    `codes` float8_e4m3fn `[..., depth]`, the depth K-major (last), and `scales` float8_e8m0fnu
    `[..., depth / 32]`, each value its code times its block's scale. Rows `[rows, depth]` or a grouped
    product's matrices `[groups, width, depth]`, as Blackwell's block-scaled tensor cores read both."""

    codes: jax.Array
    scales: jax.Array

    @classmethod
    def quantized(cls, x: jax.Array) -> "MxFp8":
        """`x` (`[..., depth]`) as MXFP8, by OCP MX's rule: each 32 depths' scale 2 to the power of its
        largest magnitude's exponent less E4M3's largest's (8), so that magnitude over the scale lies in
        [256, 512); each value over its scale rounded to E4M3, ties to even, saturating at +-448. A block of
        zeros takes the least scale. (A scale the ceiling of the largest magnitude over 448, DeepSeek's
        rule, rounds a code up past the largest float32 at float32's top binade: 256 x 2^120.)"""
        from dew.nn import fake_quant

        x = jnp.asarray(x, jnp.float32)
        per = formats_of.MXFP8.depth
        blocks = x.reshape(*x.shape[:-1], x.shape[-1] // per, per)
        largest = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)
        exponent = jnp.where(largest > 0, jnp.floor(jnp.log2(jnp.where(largest > 0, largest, 1.0))), -119.0)
        scale = jnp.exp2(jnp.clip(exponent - 8, -127, 127))
        codes = fake_quant.fp8_codes(blocks / scale).reshape(x.shape)
        return cls(codes.astype(jnp.float8_e4m3fn), scale[..., 0].astype(formats_of.MXFP8.scale))

    def decoded(self) -> jax.Array:
        """Its values, exactly, in float32 (`[..., depth]`): each code times its block's scale."""
        return _block_values(self.codes, self.scales, formats_of.MXFP8.depth)


def _block_values(codes: jax.Array, scales: jax.Array, block: int) -> jax.Array:
    """Block-scaled codes' values, exactly, in float32: each code times its `block` depths' scale."""
    return codes.astype(jnp.float32) * jnp.repeat(scales.astype(jnp.float32), block, axis=-1)


MXFP8_ROW = 'ledger 4519'
"""The MX product on Blackwell's block-scaled tensor cores (K51)."""


@dataclasses.dataclass(frozen=True)
class MxCase:
    """An MX product as its selector reads it (`MXFP8`): the tile the template's launch plans for it on the
    target (`grouped_matmul.MosaicMxGmm`), None where it plans none."""

    tile: mosaic_gpu.Tiles | None


MXFP8: selection.Op[MxCase] = selection.Op(generation.Choice.MXFP8, (
    selection.Kernel('mosaic', (
        selection.Requirement(
            'planned', lambda case: case.tile is not None, "the template plans no block-scaled tile here: "
            "an MX product on a family whose products take block scales (tcgen05's), whole 128s of rows, "
            "columns and depth", MXFP8_ROW),
        selection.Requirement(
            'measured', lambda case: False, "no receipt yet has the block-scaled template beat the decoded "
            "operands' product", MXFP8_ROW, kind='measured'))),
    selection.Kernel('xla', (), generic=True)))
"""An MX product's selector (`mxfp8_product`): the template's products scaling each block in the instruction
(`mosaic_gpu.gmm`'s `mx_scales`) where named (`generation.chosen(mxfp8='mosaic')`) and planned, past the
measured requirement no receipt has met yet; else the decoded operands' product in XLA."""


def mxfp8_product(lhs: MxFp8, matrices: MxFp8, group_blocks: jax.Array, *, out_dtype,
                  interpret_on_cpu: bool = False, gather: jax.Array | None = None) -> jax.Array:
    """Each group's rows of `lhs` (`[rows, depth]`) times its matrix of `matrices` (`[groups, width, depth]`,
    K-major), both MXFP8 (`MxFp8`): each value its code times its 32 depths' scale, every product exact in
    fp32, summed in fp32 and rounded once to `out_dtype` (row 4519). Group `g` holds `group_blocks[g]` whole
    128-row blocks, in order (the block scales' tiles), the rows past them zero. Its selector (`MXFP8`) runs
    the template on Blackwell's block-scaled tensor cores where named (`mosaic_gpu.gmm`'s `mx_scales`), else
    the decoded operands' product in XLA. `gather`, int32 `[rows]`, makes the rows `lhs`'s rows at its entries
    (`gemm`'s `gather`, a mixture of experts' tokens): their codes read where they lie, their scales by the
    same table (`_block_scaled`)."""
    return _block_scaled(lhs, matrices, group_blocks, MXFP8, interpret_on_cpu, gather).astype(out_dtype)


def _block_scaled(lhs, matrices, group_blocks: jax.Array, op: selection.Op, interpret: bool,
                  gather: jax.Array | None = None) -> jax.Array:
    """A block-scaled grouped product summed in fp32 (`mxfp8_product`'s, `nvfp4_block_product`'s): `op`
    selects the template on Blackwell's block-scaled tensor cores (`mosaic_gpu.gmm`'s `mx_scales`, the
    operands' `codes` and their block `scales`, `product.block_format`'s) where named and planned, else
    the operands' block-scaled values (`decoded`: each exact in bfloat16 and in TF32, so DEFAULT's products
    are exact) summed as XLA sums fp32 products. Gathered rows (`gather`) are the template's gather
    (tcgen05's load warp's, row 1376) and their block scales the same table's rows of `lhs.scales`, gathered
    as every call's scales are laid out anew for the products (`_scales_tiled`), a sixteenth or a
    thirty-second of the rows' bytes (row 4851)."""
    sizes = (group_blocks * SCALE_DEPTH).astype(jnp.int32)
    launch = grouped_matmul.MosaicMxGmm()
    tables = (() if gather is None else (gather,))
    # Gathered rows' block scales are the same table's rows of the rows' scales, laid out for the products.
    scales = (_scales_tiled(lhs.scales if gather is None else lhs.scales[gather]),
              _scales_tiled(matrices.scales))
    call = functools.partial(launch, lhs.codes, matrices.codes, sizes, (), (), None, None, *tables,
                             out_dtype='float32', trans_rhs=True, interpret=interpret, scales=scales)
    tile = _planned(launch, launch.bind(*_abstract(*call.args[:3]), (), (), None, None,
                                        *_abstract(*tables), **{**call.keywords,
                                                                'scales': _abstract(*scales)}))
    if selection.select(op, MxCase(tile)).kernel != 'mosaic':
        values = lhs.decoded() if gather is None else lhs.decoded()[gather]
        return grouped_matmul.xla_ragged_dot(values, jnp.swapaxes(matrices.decoded(), 1, 2), sizes,
                                             precision=jax.lax.Precision.DEFAULT,
                                             preferred_element_type=jnp.float32)
    assert tile is not None, "the selector names the template where it plans a tile"
    return call()


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


def _scales_tiled(scales: jax.Array) -> jax.Array:
    """MX scales `[..., rows, depth / 32]` tiled as `tcgen05.scales_layout` lays them out in shared memory,
    `[..., rows / 128, depth / 128, 32, 16]`: row `i` and scale `j` at `[i // 128, j // 4, i % 32,
    (i // 32) % 4 * 4 + j % 4]` (`mosaic_gpu.gmm`'s `mx_scales`)."""
    *lead, rows, columns = scales.shape
    tiled = scales.reshape(*lead, rows // 128, 4, 32, columns // 4, 4)
    order = tuple(range(len(lead)))
    tiled = jnp.transpose(tiled, (*order, *(len(lead) + axis for axis in (0, 3, 2, 1, 4))))
    return tiled.reshape(*lead, rows // 128, columns // 4, 32, 16)


@dataclasses.dataclass(frozen=True)
class NvFp4Block:
    """An NVFP4 operand as Blackwell's block-scaled tensor cores read it (row 4519's NVFP4 arm): `codes`
    float4_e2m1fn `[..., depth]`, the depth K-major (last), each sixteen depths' E4M3 scale (`scales`,
    `[..., depth / 16]`) and the tensor's fp32 `scale`: each value its code times its block's scale times
    the tensor's (`NvFp4`'s values, laid as the products read them). Rows `[rows, depth]` or a grouped
    product's matrices `[groups, width, depth]`."""

    codes: jax.Array
    scales: jax.Array
    scale: jax.Array

    @classmethod
    def quantized(cls, x: jax.Array) -> "NvFp4Block":
        """`x` (`[..., depth]`) as NVFP4 along its depth, `bits.nvfp4_quantized`'s rule (one tensor scale,
        its largest magnitude over 6 x 448; each block's the least E4M3 at or above its largest magnitude
        over 6 times it; each value rounded to E2M1, ties to even)."""
        x = jnp.asarray(x, jnp.float32)
        words, scales, scale = bits.nvfp4_quantized(x.reshape(-1, x.shape[-1]).T)
        codes = _codes(words).T.reshape(x.shape).astype(jnp.float4_e2m1fn)
        return cls(codes, scales.T.reshape(*x.shape[:-1], -1), scale)

    @classmethod
    def of(cls, matrix: NvFp4) -> "NvFp4Block":
        """A stored NVFP4 matrix (`NvFp4`, `[depth, width]`, its codes in the packed kernel's staged order)
        as the block-scaled products read it: `[1, width, depth]`, its depths in their own order."""
        return cls(matrix.codes().T.astype(jnp.float4_e2m1fn)[None], jnp.asarray(matrix.scales)[None],
                   jnp.asarray(matrix.scale))

    def decoded(self) -> jax.Array:
        """Its block-scaled values, exactly, in float32 (`[..., depth]`): each code times its block's scale,
        the tensor's apart (`scale`)."""
        return _block_values(self.codes, self.scales, formats_of.NVFP4.depth)


NVFP4_BLOCK_OP: selection.Op[MxCase] = selection.Op(generation.Choice.NVFP4_BLOCK, (
    selection.Kernel('mosaic', (
        selection.Requirement(
            'planned', lambda case: case.tile is not None, "the template plans no block-scaled tile here: "
            "an NVFP4 product on a family whose products take block scales (tcgen05's), whole 128s of rows, "
            "columns and depth", MXFP8_ROW),
        selection.Requirement(
            'measured', lambda case: False, "no receipt yet has the block-scaled template beat the decoded "
            "operands' product", MXFP8_ROW, kind='measured'))),
    selection.Kernel('xla', (), generic=True)))
"""An NVFP4 block product's selector (`nvfp4_block_product`): the template's products scaling each block in
the instruction where named (`generation.chosen(nvfp4_block='mosaic')`) and planned, past the measured
requirement no receipt has met yet; else the decoded operands' product in XLA."""


def nvfp4_block_product(lhs: NvFp4Block, matrices: NvFp4Block, group_blocks: jax.Array, *, out_dtype,
                        interpret_on_cpu: bool = False, gather: jax.Array | None = None) -> jax.Array:
    """Each group's rows of `lhs` (`[rows, depth]`) times its matrix of `matrices` (`[groups, width,
    depth]`, K-major), both NVFP4 (`NvFp4Block`): each code times its sixteen depths' scale, every product
    exact in fp32, summed in fp32, times both tensor scales, rounded once to `out_dtype`. Group `g` holds
    `group_blocks[g]` whole 128-row blocks, in order (the block scales' tiles). Its selector
    (`NVFP4_BLOCK_OP`) runs the template on Blackwell's block-scaled tensor cores where named
    (`mosaic_gpu.gmm`'s `mx_scales`, `kind::mxf4nvf4`), else the decoded operands' product in XLA; `gather`
    as `mxfp8_product`'s."""
    summed = _block_scaled(lhs, matrices, group_blocks, NVFP4_BLOCK_OP, interpret_on_cpu, gather)
    return (summed * (lhs.scale * matrices.scale)).astype(out_dtype)


def nvfp4_quantized_product(lhs: jax.Array, matrix: NvFp4, *, out_dtype,
                            interpret_on_cpu: bool = False) -> jax.Array:
    """`lhs · W` of a stored NVFP4 matrix `W` (`NvFp4`) with `lhs` quantized to NVFP4 too, as it is used
    (`NvFp4Block.quantized`: an inference projection on Blackwell's block-scaled tensor cores, both operands
    E2M1, row 4519's NVFP4 arm): `nvfp4_block_product` of the rows, padded to whole 128s, and the matrix
    as one group. Unlike `nvfp4_product` (the rows in bfloat16), the rows round to E2M1 under their own
    block scales: a run opts in by calling it."""
    rows = lhs.reshape(-1, lhs.shape[-1])
    padded = jnp.pad(rows, ((0, -rows.shape[0] % SCALE_DEPTH), (0, 0)))
    blocks = jnp.asarray([padded.shape[0] // SCALE_DEPTH], jnp.int32)
    out = nvfp4_block_product(NvFp4Block.quantized(padded), NvFp4Block.of(matrix), blocks,
                              out_dtype=out_dtype, interpret_on_cpu=interpret_on_cpu)
    return out[:rows.shape[0]].reshape(*lhs.shape[:-1], out.shape[-1])


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


def low_rank(lhs: jax.Array, kernel: jax.Array, a: jax.Array, b: jax.Array, scale, *, out_dtype,
             precision: PrecisionLike = None, interpret_on_cpu: bool = False) -> jax.Array:
    """`lhs · kernel + scale · (lhs · a) · b`, LoRA's adapted product (Appendix C row 26, strategy (b)), as
    one template product over the depth the rank extends: `[lhs | t] · [kernel ; b]`, where `t = scale ·
    (lhs · a)` is summed in fp32 and rounded once to `lhs`'s dtype, so the update accumulates into the base
    product's accumulator and no output-sized pass adds it. `lhs` is `[rows, depth]`, `kernel` `[depth,
    width]`, `a` `[depth, rank]` and `b` `[rank, width]`; `scale` a scalar.

    Its reverse mode (first order): the input gradient is one product too, `[dy | u] · [kernel | a]ᵀ` with
    `u = scale · (dy · bᵀ)` rounded once; the factors' gradients are `lhsᵀ · (scale · (dy · bᵀ))` and `tᵀ ·
    dy`, the rank-wide kernel gradients; each operand's only where it is differentiated (a frozen kernel's,
    `lhsᵀ · dy`, never). The concatenations are copies, a materialized strategy (b); a mainloop walking the
    appended pair in place is row 1304's next form. A projection the run's low-precision recipe takes by
    its role is refused (`refuse_recipe`): the adapted product does not take the recipe."""
    refuse_recipe("an adapter's fused product (`gemm.low_rank`)")
    return _low_rank(lhs, kernel, a, b, jnp.asarray(scale, jnp.float32), out_dtype, precision,
                     interpret_on_cpu)


def _rank_product(lhs: jax.Array, factor: jax.Array, scale: jax.Array, precision: PrecisionLike) -> jax.Array:
    """`scale · (lhs · factor)`, summed and kept in at least fp32: the rank-wide operand an adapted product
    appends to its depth, rounded once to `lhs`'s dtype where it is appended (`_appended`)."""
    wide = at_least_fp32(lhs.dtype)
    summed = jnp.dot(lhs, factor.astype(lhs.dtype), precision=precision, preferred_element_type=wide)
    return scale.astype(wide) * summed


def _appended(base: jax.Array, rank_part: jax.Array, axis: int, compute) -> jax.Array:
    """`base` with `rank_part` after it along `axis`, the rank padded with zeros to whole steps of the widest
    depth the template's families take on the product's `compute` dtype (its rows', which its matrix is
    multiplied in: a 128-byte swizzle's row, `swizzling.SWIZZLE_BYTES`), so the extended depth divides into
    the base's own steps and both operands extend alike, an fp32 master matrix's as its rows'."""
    step = swizzling.swizzle_columns(compute, swizzling.SWIZZLE_BYTES)
    rank_part = rows_padded(rank_part.astype(base.dtype), axis, step)
    return jnp.concatenate([base, rank_part], axis=axis)


def _extended(lhs, kernel, a, b, scale, out_dtype, precision, interpret_on_cpu):
    """`low_rank`'s value and its appended operand `t`, in at least fp32."""
    t = _rank_product(lhs, a, scale, precision)
    out = gemm(_appended(lhs, t, 1, lhs.dtype), _appended(kernel, b, 0, lhs.dtype), out_dtype=out_dtype,
               precision=precision, interpret_on_cpu=interpret_on_cpu)
    return out, t


def _low_rank_value(lhs, kernel, a, b, scale, out_dtype, precision, interpret_on_cpu):
    return _extended(lhs, kernel, a, b, scale, out_dtype, precision, interpret_on_cpu)[0]


def _low_rank_forward(lhs, kernel, a, b, scale, out_dtype, precision, interpret_on_cpu):
    primals = (lhs, kernel, a, b, scale)
    values = (lhs.value, kernel.value, a.value, b.value, scale.value)
    out, t = _extended(*values, out_dtype, precision, interpret_on_cpu)
    # Which operands are differentiated, as the residuals' static shapes: a frozen kernel's gradient is
    # never computed.
    perturbed = jnp.zeros(tuple(int(primal.perturbed) for primal in primals), jnp.int8)
    return out, (*values, t, perturbed)


def _low_rank_backward(out_dtype, precision, interpret_on_cpu, residuals, cotangent):
    del out_dtype
    lhs, kernel, a, b, scale, t, perturbed = residuals
    wants_lhs, wants_kernel, wants_a, wants_b, wants_scale = map(bool, perturbed.shape)
    # The input gradient's product takes the cotangent in the rows' dtype, as the forward's operands; the
    # wide gradients (the kernel's, the factors' and the scale's) take it at its own precision where it is
    # wider (an fp32 output's), the rows widened to it, as the plain product's kernel gradient
    # (`_kernel_gradient`'s `work`).
    work = jnp.promote_types(cotangent.dtype, lhs.dtype)
    dy, wide_dy = cotangent.astype(lhs.dtype), cotangent.astype(work)
    wide = at_least_fp32(lhs.dtype)
    # The cotangent of `t` before its scale, dy · bᵀ (b as the forward multiplied it, in the rows' dtype),
    # summed wide, and after it: the scale taken before any product with it, as the adapter's own program
    # takes it (a product near float32's largest that the scale brings back in range never overflows first).
    through = jnp.dot(wide_dy, b.astype(lhs.dtype).astype(work).T, precision=precision,
                      preferred_element_type=wide)
    scaled = scale.astype(wide) * through
    d_lhs = d_kernel = d_a = d_b = d_scale = None
    if wants_lhs:
        d_lhs = gemm(_appended(dy, scaled.astype(lhs.dtype), 1, lhs.dtype),
                     _appended(kernel, a, 1, lhs.dtype), transpose_rhs=True, out_dtype=lhs.dtype,
                     precision=precision, interpret_on_cpu=interpret_on_cpu)
    if wants_kernel:
        d_kernel = gemm_t(lhs.astype(work), wide_dy, None, out_dtype=kernel.dtype, precision=precision,
                          interpret_on_cpu=interpret_on_cpu)[0]
    # The factors' gradients from the wide `scaled` and `t` (the rank-wide products' sums, unrounded), as
    # the adapter's own program differentiates its fp32 low-rank products.
    if wants_a:
        d_a = jnp.dot(lhs.T.astype(wide), scaled, precision=precision,
                      preferred_element_type=wide).astype(a.dtype)
    if wants_b:
        # `t` holds the scale already: tᵀ · dy is the factor's gradient whole.
        d_b = jnp.dot(t.T, wide_dy.astype(wide), precision=precision,
                      preferred_element_type=wide).astype(b.dtype)
    if wants_scale:
        unscaled = jnp.dot(lhs, a.astype(lhs.dtype), precision=precision, preferred_element_type=wide)
        d_scale = jnp.sum(through * unscaled).astype(scale.dtype)
    return d_lhs, d_kernel, d_a, d_b, d_scale


# The launches' derivative glue (`derivatives.KernelGlue`): an operand no reverse pass reaches is a symbolic
# zero, whose gradient is never formed.
_low_rank = contracted("test_an_adapted_product_holds_every_reverse_order")(
    declared_derivative(KernelGlue(), _low_rank_value, _low_rank_forward, _low_rank_backward,
                        nondiff_argnums=(5, 6, 7), symbolic_zeros=True))


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
        # A CPU sums a layer's float32 projection as a vectorized dot does (row 4863).
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


_CONTRACTED_ROWS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([0], [0]), ([], [])), lhs_ragged_dimensions=[0], rhs_group_dimensions=[])
"""A ragged dot over its rows, each group's rows of both operands contracted into the group's matrix: a
grouped GEMM's weight gradient, `[rows, depth]` and `[rows, width]` to `[groups, depth, width]`."""

def _gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue, epilogue_arrays,
          epilogue_axes, dtypes, *, several: bool, transpose_rhs: bool, interpret_on_cpu: bool,
          precision: PrecisionLike, dense: bool = False, paired: jax.Array | None = None,
          first: jax.Array | None = None, gather: jax.Array | None = None, scatter: jax.Array | None = None,
          summed_as: tuple = ()):
    """`gemm`'s value, on grouped operands; `summed_as`, the outputs of the product whose accumulator it
    recomputes (a reverse pass's, at fp32), whose sums it repeats (`product.promoted`: an fp32 output's
    exact sums, a 16-bit one's not); `first`, a stacked `rhs`'s matrix its first group reads
    (`_grouped`); `gather`, the rows of `lhs` its rows are, and `scatter`, the outputs' rows they land at
    (`gemm`'s)."""
    def kernel(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first, gather, scatter, *,
               interpret):
        # An fp32 product asked above the default precision runs as its operands' bf16 parts where a launch
        # takes them (`_in_parts`); the matrix's values are `rhs`'s where the rows' dtype holds them, else
        # theirs rounded to it, as the product multiplies them.
        holds = jnp.promote_types(rhs.dtype, lhs.dtype) == jnp.dtype(lhs.dtype)
        counts = _part_counts(lhs.dtype, rhs.dtype if holds else lhs.dtype, precision)
        whole = seen_whole(lhs) and seen_whole(rhs)
        if (counts is not None and prologue is None and paired is None and scatter is None and not several
                and _parts_launched(*_gemm_call(
                    lhs.astype(jnp.bfloat16), rhs.astype(jnp.bfloat16), group_sizes, None, (), (), epilogue,
                    epilogue_arrays, epilogue_axes, dtypes, several=False, transpose_rhs=transpose_rhs,
                    interpret=interpret, precision=precision, dense=dense, paired=None, first=first,
                    gather=gather, scatter=None, summed_as=summed_as), whole=whole)):
            def part(left, right, finite, epilogue, epilogue_arrays, epilogue_axes, dtypes):
                left, right = (_finite(x) if each else x
                               for x, each in zip((left, right), finite, strict=True))
                return _gemm(left, right, group_sizes, None, (), (), epilogue, epilogue_arrays, epilogue_axes,
                             dtypes, several=False, transpose_rhs=transpose_rhs,
                             interpret_on_cpu=interpret_on_cpu, precision=precision, dense=dense, first=first,
                             gather=gather, summed_as=summed_as or dtypes)
            return _in_parts(part, lhs, rhs, counts, epilogue, epilogue_arrays, epilogue_axes, dtypes)
        call, product = _gemm_call(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue,
                                   epilogue_arrays, epilogue_axes, dtypes, several=several,
                                   transpose_rhs=transpose_rhs, interpret=interpret, precision=precision,
                                   dense=dense, paired=paired, first=first, gather=gather, scatter=scatter,
                                   summed_as=summed_as)
        # A call no launch takes (a TPU's widths its blocks do not divide) runs XLA's program; a paired one no
        # launch takes whole its two products apart, each through the launches that take it (`_apart`).
        launch = grouped_matmul.launched(*call, product=product, whole=seen_whole(lhs) and seen_whole(rhs))
        if launch is grouped_matmul.APART:
            return scattered(_apart(lhs if gather is None else lhs[gather], (rhs, paired), group_sizes,
                                     prologue, prologue_arrays, prologue_axes, epilogue, epilogue_arrays,
                                     epilogue_axes, dtypes, several=several, summed_as=summed_as or dtypes,
                                     transpose_rhs=transpose_rhs, interpret_on_cpu=interpret_on_cpu,
                                     precision=precision, dense=dense), scatter)
        if not callable(launch) and (gather is not None or scatter is not None):
            # No launch reads the rows where they lie or writes them at their slots (mma.sync's bundle
            # gathers none, row 5213): the rows gathered first and the outputs put at their slots after, the
            # grouped launches taking the rows between, never XLA's ragged dot over every group's matrix.
            return scattered(kernel(lhs if gather is None else lhs[gather], rhs, group_sizes, prologue_arrays,
                                    epilogue_arrays, paired, first, None, None, interpret=interpret), scatter)
        if not callable(launch):
            return portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first, gather,
                            scatter)
        return launch()

    def portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first, gather, scatter):
        config = XlaConfig(transpose_rhs, precision, dense=dense)
        outs = xla_mainloop(config, lhs if gather is None else lhs[gather], rhs, group_sizes, prologue,
                            prologue_arrays, prologue_axes, epilogue, epilogue_arrays, epilogue_axes, dtypes,
                            several, paired=paired, first=first)
        return scattered(outs, scatter)

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays,
                       epilogue_arrays, paired, first, gather, scatter)


def _gemm_call(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue, epilogue_arrays,
               epilogue_axes, dtypes, *, several: bool, transpose_rhs: bool, interpret: bool,
               precision: PrecisionLike, dense: bool, paired, first, gather, scatter, summed_as: tuple):
    """`_gemm`'s grouped call of its operands (`grouped_matmul.gmm_call`'s launches, arguments and options)
    and the product it declares: one construction for the launch it runs and for a reverse pass's question
    of which launch a call would run (`_keeps_launch`)."""
    before_sides, before_axes, before_broadcast = _launchable(prologue_arrays, prologue_axes)
    after_sides, after_axes, after_broadcast = _launchable(epilogue_arrays, epilogue_axes)
    hooks = grouped_matmul.Hooks(prologue, before_axes, epilogue, after_axes,
                                 tuple(dtype.name for dtype in dtypes) if several else None,
                                 prologue_broadcast=before_broadcast, epilogue_broadcast=after_broadcast)
    product = declared(lhs.dtype, precision, outputs=summed_as or dtypes, operands=(lhs.dtype, rhs.dtype))
    call = grouped_matmul.gmm_call(lhs, rhs.astype(lhs.dtype), group_sizes, before_sides, after_sides,
                                   None if paired is None else paired.astype(lhs.dtype), first, gather,
                                   scatter, out_dtype=dtypes[0], trans_rhs=transpose_rhs, hooks=hooks,
                                   interpret=interpret, promoted=product.promoted, dense=dense)
    return call, product


def _cotangents(lhs, matrices: tuple, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue,
                epilogue_arrays, epilogue_axes, cotangents: tuple, dtypes: tuple, compute, acc_shape: tuple,
                plain: dict) -> tuple[tuple, tuple, tuple]:
    """Each accumulator's cotangent (one a matrix of `matrices`: the product's, and its `paired`'s), the
    epilogue side operands' and whether each accumulator's passes an output's through in `compute`'s dtype
    (`_passes_cotangent`): the outputs' own without an epilogue, else through the epilogue's VJP at the
    accumulators recomputed at their own width, where the VJP reads them (from `lhs`'s rows at `plain`'s
    `gather`, a gathered product's)."""
    count = len(matrices)
    if epilogue is None:
        d_accs = tuple(cotangents[:count])
        return d_accs, (), tuple(jnp.promote_types(d.dtype, compute) == compute for d in d_accs)
    rows = functools.partial(_rows, axes_of=epilogue_axes, group_sizes=group_sizes, rows=acc_shape[0])
    sides = jax.eval_shape(rows, epilogue_arrays)
    # The accumulator the products sum in: fp32, an fp64 product's fp64 (`xla_mainloop`). The epilogue's
    # derivative reads it at that width (a row scale's gradient sums it times the cotangent), so it is
    # summed as an output of its own dtype sums (`product.promoted`: exactly at a precision above the
    # default), not as the forward's narrower outputs, whose rounding hid the tensor cores' sums within a
    # step (row 5406: an A100's fp16 row-scaled product's scale gradient 1.72 times XLA's error).
    acc = jax.ShapeDtypeStruct(acc_shape, at_least_fp32(compute))
    if _reads_accumulators(epilogue, count, acc, sides):
        accs = _as_tuple(_gemm(lhs, matrices[0], group_sizes, prologue, prologue_arrays, prologue_axes, None,
                               (), (), (acc.dtype,) * count, several=count > 1,
                               paired=matrices[1] if count > 1 else None, summed_as=(acc.dtype,) * count,
                               **plain))
    else:
        accs = (jnp.zeros(acc.shape, acc.dtype),) * count

    def epilogued(*args):
        # Each row's side values, whose transpose sums a groups axis's cotangents over each group's rows.
        return _tupled(epilogue, *args[:count], *rows(args[count:]))

    outs, pull = jax.vjp(epilogued, *accs, *epilogue_arrays)
    pulled = pull(tuple(c.astype(out.dtype) for c, out in zip(cotangents, outs, strict=True)))
    narrow = tuple(_passes_cotangent(epilogue, count, at, acc, sides, dtypes, compute)
                   for at in range(count))
    return tuple(pulled[:count]), tuple(pulled[count:]), narrow


def _input_cotangents(lhs, d_products: tuple, matrices: tuple, group_sizes, prologue, prologue_arrays,
                      prologue_axes, compute, plain: dict) -> tuple:
    """`lhs`'s and the prologue side operands' cotangents from the cotangent of the operand the forward
    multiplied (`prologue(lhs)` rounded to `lhs`'s dtype): each matrix's product with its accumulator's
    cotangent, one matrix's rounded once to `lhs`'s dtype, a paired product's two summed wide and rounded
    once (the unfused program's law: each projection's cotangent rounded, their products added), each
    summed as the gradient's own dtype sums (`_gemm`'s `summed_as`); each cotangent's rows read at
    `plain`'s `gather` where given (a scattered product's cotangent, read where its rows landed).
    A gathered product's `lhs` is its tokens, whose dtype alone is read (no prologue rewrites its rows)."""
    options = {'interpret_on_cpu': plain['interpret_on_cpu'], 'precision': plain['precision'],
               'transpose_rhs': not plain['transpose_rhs'], 'dense': plain['dense'], 'first': plain['first'],
               'gather': plain['gather']}
    # A paired product's first product, wide, is added in the second's epilogue (`_added`): the same sum,
    # rounded once, with no pass of XLA's own between them (row 2164: FSPAIRED2's 0.42 ms add); where that
    # hook would move the second product off the launch its plain product runs on (a family whose register
    # model counts no hooks: mma.sync's, tcgen05's), the two are added in XLA as before.
    wide, d_operand = at_least_fp32(lhs.dtype), None
    for at, (d, matrix) in enumerate(zip(d_products, matrices, strict=True)):
        out = wide if at < len(matrices) - 1 or (at and prologue is not None) else jnp.dtype(lhs.dtype)
        matrix = matrix.astype(compute)
        if d_operand is not None and _keeps_launch(d, matrix, group_sizes, d_operand, out, (lhs.dtype,),
                                                   options):
            d_operand = jnp.asarray(_gemm(d, matrix, group_sizes, None, (), (), _added, (d_operand,),
                                          (('rows', 'columns'),), (out,), several=False,
                                          summed_as=(lhs.dtype,), **options))
            continue
        product = jnp.asarray(_gemm(d, matrix, group_sizes, None, (), (), None, (), (),
                                    (out if d_operand is None else wide,), several=False,
                                    summed_as=(lhs.dtype,), **options))
        d_operand = product if d_operand is None else (d_operand + product).astype(out)
    assert d_operand is not None, "a product has a matrix"
    if prologue is None:
        return d_operand.astype(lhs.dtype), ()
    d_lhs, d_prologue = _prologue_cotangents(prologue, lhs, prologue_arrays, prologue_axes, group_sizes,
                                             d_operand)
    return d_lhs.astype(lhs.dtype), tuple(_as_cotangent(d, a) for d, a in zip(d_prologue, prologue_arrays,
                                                                                strict=True))


def _added(acc, prior):
    """An accumulator plus a wide product before it (`_input_cotangents`' paired sum)."""
    return acc + prior


def _keeps_launch(lhs, rhs, group_sizes, prior, out, summed_as: tuple, options: dict) -> bool:
    """Whether `lhs @ rhs` with `prior` added in its epilogue (`_added`) runs on the launch its plain product
    runs on (`grouped_matmul.GROUPED`'s choice, read without reporting it: `selection.selected`), each the
    call `_gemm` makes of it (`_gemm_call`), its output `out`; options as `_gemm`'s (`_input_cotangents`').
    A hooked call a chosen kernel refuses keeps nothing (its plain products run apart, added in XLA)."""
    interpret = options['interpret_on_cpu'] and jax.default_backend() == 'cpu'
    group_sizes = jnp.full((1,), lhs.shape[0], jnp.int32) if group_sizes is None else group_sizes
    kernels = []
    for epilogue, after in ((None, ()), (_added, (prior,))):
        call, product = _gemm_call(lhs, rhs, group_sizes, None, (), (), epilogue, after,
                                   (('rows', 'columns'),) * len(after), (out,), several=False,
                                   transpose_rhs=options['transpose_rhs'], interpret=interpret,
                                   precision=options['precision'], dense=options['dense'], paired=None,
                                   first=options['first'], gather=options['gather'], scatter=None,
                                   summed_as=summed_as)
        case = grouped_matmul.grouped_case(*call, product=product, whole=seen_whole(lhs) and seen_whole(rhs))
        try:
            kernels.append(selection.selected(grouped_matmul.GROUPED, case).name)
        except selection.KernelRefused:
            if epilogue is None:
                raise
            return False
    return kernels[0] == kernels[1]


def _kernel_gradient(lhs, d_product, matrix, group_sizes, prologue, prologue_arrays, prologue_axes, compute,
                     work, plain: dict):
    """A matrix's kernel gradient from its accumulator's cotangent `d_product`: the rows multiplied as the
    forward multiplied them (through the prologue, fused, where the cotangent is in `compute`'s dtype; the
    forward's rounded operand in the cotangent's wider dtype `work` otherwise). A stack's whose groups read
    its matrices from `plain['first']` (`_grouped`) is theirs there and zero at every other. Each operand's
    rows are read at its table of `plain['tables']` where given (`gemm_t`'s `gather`: a gathered product's
    tokens, `lhs` then the tokens, and a scattered product's cotangent)."""
    tables = plain['tables']
    options = {'interpret_on_cpu': plain['interpret_on_cpu'], 'precision': plain['precision'],
               'gather': None if all(table is None for table in tables) else tables}
    sizes = None if plain['dense'] else group_sizes
    if work is None:
        gradient = _gemm_t(lhs.astype(compute), d_product, sizes, prologue, prologue_arrays, prologue_axes,
                           out_dtype=matrix.dtype, **options)
    else:
        operand = (lhs.astype(compute) if prologue is None else prologue(
            lhs, *_rows(prologue_arrays, prologue_axes, group_sizes, lhs.shape[0])).astype(lhs.dtype))
        # The operand in its own dtype beside the wider cotangent, multiplied in the wider (`_gemm_t`).
        gradient = _gemm_t(operand, d_product, sizes, None, (), (), out_dtype=matrix.dtype, **options)
    gradient = jnp.swapaxes(gradient, 1, 2) if plain['transpose_rhs'] else gradient
    if plain['first'] is None:
        return gradient
    return jax.lax.dynamic_update_slice_in_dim(jnp.zeros(matrix.shape, gradient.dtype), gradient,
                                               plain['first'], 0)


def _part_counts(lhs_dtype, rhs_dtype, precision) -> tuple[int, int] | None:
    """The bf16 parts each operand of a product is multiplied as on a family's launch, its operands' values
    `lhs_dtype`'s and `rhs_dtype`'s: where the product is fp32's and asked whole above the default precision,
    which the families' tensor cores multiply only in TF32, each operand's bf16 parts (`product.parts`: an
    fp32 operand's three, a 16-bit one's one or two), each pair's product exact and their sums fp32's
    (`_in_parts`); None where the product is the operands' own (a 16-bit one's, an fp32 one at the default
    precision TF32's, a declared algorithm's a portable route's)."""
    if (jnp.promote_types(lhs_dtype, rhs_dtype) != jnp.float32
            or asks_default_precision(precision, configured=True)
            or not exact_products(precision, jnp.float32)):
        return None
    return parts(lhs_dtype), parts(rhs_dtype)


def _in_dtype(prologue, dtype, rows, *sides):
    """`prologue` on widened `rows` read back in their own `dtype` (exact) and its result rounded there, as
    the portable program rounds a prologue's rows before a product in a wider dtype (`_gemm_t`)."""
    return prologue(rows.astype(dtype), *sides).astype(dtype)


def _parts_launched(call: tuple, product, *, whole: bool) -> bool:
    """Whether a launch of the template's or Triton's takes a call of an fp32 product's bf16 parts (`call`, a
    part's, and its `product`) as the grouped selector chooses (`selection.selected`, unreported: each part's
    own call reports its route): where XLA's program would run it instead, the fp32 product runs whole."""
    case = grouped_matmul.grouped_case(*call, product=product, whole=whole)
    return selection.selected(grouped_matmul.GROUPED, case).name not in ('xla', grouped_matmul.APART.name)


def _in_parts(product, lhs, rhs, counts: tuple[int, int], epilogue, epilogue_arrays, epilogue_axes, dtypes):
    """`product(left, right, finite, epilogue, epilogue_arrays, epilogue_axes, dtypes)` of `lhs` and `rhs` as
    their bf16 parts (`counts` of them, `product.formed`), `finite` whether each part is read with its values
    that are not finite zeroed (`_finite`): each pair of parts a product issues (`product.terms`,
    smallest first, those under fp32's significand), each exact, the earlier ones' fp32 sum added in the
    next one's epilogue (`_added`), the last one's epilogue `epilogue` on the whole sum; as exact as the fp32
    product whole, its sums fp32's (an exactly split operand's parts sum to it, row 5405)."""
    left, right = _bf16_parts(lhs, counts[0]), _bf16_parts(rhs, counts[1])

    def operands(i: int, j: int):
        # A value that is not finite meets the other operand's leading part alone, as the whole product
        # meets it once: a leading part times the other's later parts reads its finite values alone
        # (`product`'s `finite`; an infinity times a later part's zero would add a NaN the whole product has
        # not).
        return left[i], right[j], (i == 0 and j > 0, j == 0 and i > 0)

    pairs = terms(*counts)
    wide = (jnp.dtype(jnp.float32),)
    total = None
    for index, (i, j) in enumerate(pairs):
        last = index == len(pairs) - 1
        if total is None and not last:
            total = product(*operands(i, j), None, (), (), wide)
            continue
        if not last:
            total = product(*operands(i, j), _added, (total,), (('rows', 'columns'),), wide)
            continue
        if total is None:
            return product(*operands(i, j), epilogue, epilogue_arrays, epilogue_axes, dtypes)
        whole = (functools.partial(_summed_then, epilogue) if epilogue is not None else _added)
        return product(*operands(i, j), whole, (total, *epilogue_arrays),
                       (('rows', 'columns'), *epilogue_axes), dtypes)
    raise AssertionError("a product has a pair of parts")


def _bf16_parts(x, count: int) -> tuple[jax.Array, ...]:
    """`x`'s `count` bf16 parts (`product.formed`: their sum it exactly, a finite value past bf16's largest
    saturating into them), a value that is not finite whole in its leading part and the later parts zero, so
    its products are the whole value's (an infinity's infinity, times zero NaN), where its saturated parts
    summed to a finite value; one part, `x` rounded to bf16."""
    if count == 1:
        return (x.astype(jnp.bfloat16),)
    wide = x.astype(jnp.float32)
    finite = jnp.isfinite(wide)
    leading, *rest = formed(jnp.where(finite, wide, 0), count)
    return jnp.where(finite, leading, wide.astype(jnp.bfloat16)), *rest


def _finite_after(prologue, rows, *sides):
    """`prologue`'s rows with their values that are not finite zeroed (`_finite`)."""
    return _finite(prologue(rows, *sides))


def _finite(x):
    """`x` with its values that are not finite zeroed (`_in_parts`' cross terms)."""
    return jnp.where(jnp.isfinite(x), x, 0).astype(x.dtype)


def _summed_then(epilogue, acc, prior, *sides):
    """`epilogue` on an accumulator plus the earlier parts' sum before it (`_in_parts`' last part)."""
    return epilogue(acc + prior, *sides)


def _apart(lhs, matrices: tuple, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue,
           epilogue_arrays, epilogue_axes, dtypes, *, several: bool, summed_as: tuple, **options):
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
        sides = _rows(epilogue_arrays, epilogue_axes, group_sizes, lhs.shape[0])
        products = _tupled(epilogue, *products, *sides)
    outs = tuple(out.astype(dtype) for out, dtype in zip(products, dtypes, strict=True))
    return outs if several else outs[0]


@dataclasses.dataclass(frozen=True)
class XlaConfig:
    """The plain-JAX mainloop's config (`xla_mainloop`): whether the matrices lie `[groups, width, depth]`
    (`transpose_rhs`), the products' `precision`, the dtype the operands round to before them (`compute`,
    None for their own), and whether the call is `dense` (one group of every row)."""

    transpose_rhs: bool
    precision: PrecisionLike
    compute: jnp.dtype | None = None
    dense: bool = False


def xla_mainloop(config: XlaConfig, lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue,
                 epilogue_arrays, epilogue_axes, dtypes, several: bool, *, paired: jax.Array | None = None,
                 first: jax.Array | None = None):
    """The template's portable mainloop at `config` (docs/design/kernel-templates.md's `xla_mainloop`):
    `gemm`'s arithmetic in XLA, the kernels' products of `lhs`'s dtype summed in at least fp32 (an fp64
    product's in fp64, which no kernel takes), one rounding, a dense call's as XLA's own dot
    (`grouped_matmul.xla_dot`: one group of every row, nothing to mask, no ragged dot to trace);
    with `config.compute`, the products of the operands rounded to it, its tangent the MoE projection's
    (`grouped_matmul.portable_projection`, whose sum, at least fp32, it reads); with `paired`, both matrices'
    products, the epilogue's two accumulators; with `first`, the groups' matrices of a stacked `rhs` from
    that one (`_grouped`), sliced out, as XLA's ragged dot reads a matrix a group."""
    transpose_rhs, precision, compute, dense = (config.transpose_rhs, config.precision, config.compute,
                                                config.dense)
    if first is not None:
        rhs = jax.lax.dynamic_slice_in_dim(rhs, first, group_sizes.shape[0], 0)
    matrices = tuple(jnp.swapaxes(each, 1, 2) if transpose_rhs else each
                     for each in (rhs, *(() if paired is None else (paired,))))
    prologue_arrays = _rows(prologue_arrays, prologue_axes, group_sizes, lhs.shape[0])
    epilogue_arrays = _rows(epilogue_arrays, epilogue_axes, group_sizes, lhs.shape[0])
    if compute is not None:
        # Summed and kept at least fp32 (an fp64 product's in fp64), the accumulator the epilogue reads.
        wide = at_least_fp32(compute)
        accs = tuple(grouped_matmul.portable_projection(lhs, matrix, group_sizes, compute, precision, wide)
                     for matrix in matrices)
    else:
        product = lhs if prologue is None else prologue(lhs, *prologue_arrays).astype(lhs.dtype)
        wide = at_least_fp32(lhs.dtype)
        accs = tuple(grouped_matmul.xla_dot(product, each[0].astype(lhs.dtype), precision=precision,
                                            preferred_element_type=wide) if dense
                     else grouped_matmul.xla_ragged_dot(product, each.astype(lhs.dtype), group_sizes,
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

    def tangent(primals, tangents):
        return prologued(*primals), prologue_jvp(*primals)(*tangents)

    # The fused kernel's derivative glue (`derivatives.KernelGlue`): the producer's own derivative, true and
    # rewritten as any other math, written once for the prologue the kernel applies to its tiles.
    prologued = contracted("test_a_gemm_prologue_holds_every_order")(
        declared_tangent(KernelGlue(), lambda lhs, *sides: prologue(lhs, *sides), tangent))
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

    def multiplied(x, *sides):
        # A groups axis's value at each row, kept as a row operand's is; its transpose sums each group's rows.
        sides = tuple(kept(side, axes.index(GROUPS)) if GROUPS in axes else side
                      for side, axes in zip(_rows(sides, axes_of, group_sizes, lhs.shape[0]), axes_of,
                                            strict=True))
        return prologue(x, *sides).astype(lhs.dtype)

    _, pull = jax.vjp(multiplied, *inputs)
    d_lhs, *d_sides = pull(d_operand.astype(lhs.dtype))
    d_lhs = jnp.where(routed[:, None] & nonempty, d_lhs, 0)
    d_sides = [d if d.dtype == jax.dtypes.float0 else jnp.where(nonempty, d, 0) for d in d_sides]
    return d_lhs, tuple(d_sides)


def _as_cotangent(cotangent, operand):
    """`cotangent` in `operand`'s dtype; an integer or boolean operand's stays float0, its tangent space."""
    return cotangent if cotangent.dtype == jax.dtypes.float0 else cotangent.astype(operand.dtype)


def _reads_accumulators(epilogue, count: int, acc: jax.ShapeDtypeStruct, sides: tuple) -> bool:
    """Whether the epilogue's VJP reads any of its `count` accumulators: a bias's does not, a product with
    a side operand's does. Read from the VJP's own program, shapes alone."""
    closed, _ = _epilogue_vjp(epilogue, count, acc, sides)
    return bool(set(closed.jaxpr.invars[:count]) & _needed(closed.jaxpr, closed.jaxpr.outvars))


def _passes_cotangent(epilogue, count: int, at: int, acc: jax.ShapeDtypeStruct, sides: tuple, dtypes: tuple,
                      narrow) -> bool:
    """Whether the epilogue's VJP hands its accumulator `at` (of `count`) one output's cotangent unchanged
    (a bias's, a cast's) whose dtype `narrow` holds exactly: read from the VJP's own program, its result
    traced back through conversions alone."""
    closed, cotangents = _epilogue_vjp(epilogue, count, acc, sides)
    var = closed.jaxpr.outvars[at]
    producers = {out: equation for equation in closed.jaxpr.eqns for out in equation.outvars}
    while var in producers and producers[var].primitive.name == 'convert_element_type':
        var = producers[var].invars[0]
    if var not in cotangents:
        return False
    return jnp.promote_types(dtypes[cotangents.index(var)], narrow) == jnp.dtype(narrow)


def _epilogue_vjp(epilogue, count: int, acc: jax.ShapeDtypeStruct, sides: tuple):
    """The epilogue's VJP as a program of its `count` accumulators (each `acc`), its side operands and its
    outputs' cotangents, and the cotangents' variables in it."""
    tupled = functools.partial(_tupled, epilogue)

    def pulled(accs, sides, cotangents):
        return jax.vjp(tupled, *accs, *sides)[1](cotangents)

    accs = (acc,) * count
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
    """`rhs` with a group axis, the group sizes, and the matrix of it the first group reads (None: the
    first): a dense product is one group of every row. A stacked `rhs` at `layer` (`gemm`) is its every
    layer's groups as one axis, of which the call's groups, the layer's own, read the layer's from its
    first, where they lie. Where the layer's sizes were spread over every layer's groups instead, a launch
    tiled the stack's groups, most of them empty: at Qwen3-30B-A3B's 128 experts over 8 layers on an H100,
    0.69 ms a decode step of grouped products over 1024 groups, and 0.32 summing their offsets
    (OM-1342-cache-base-2474a6e4c)."""
    sizes = (jnp.full((1,), lhs.shape[0], jnp.int32) if group_sizes is None
             else group_sizes.astype(jnp.int32))
    if layer is None:
        return (rhs[None] if group_sizes is None else rhs), sizes, None
    stack = rhs[:, None] if group_sizes is None else rhs
    layers, groups = stack.shape[:2]
    first = jnp.asarray(layer, jnp.int32) * groups
    return stack.reshape(layers * groups, *stack.shape[2:]), sizes, first


def _split(operands: tuple, hook: str, group_sizes: jax.Array | None) -> tuple[tuple, tuple]:
    """The side operands' arrays and their axes, each checked: one axis per dimension, each a launch
    axis the hook's operands follow, `groups` beside no rows (one value a group of the call's, a dense
    product's one group: `axes.GROUPS`) or None for a dimension of one, at least one a launch axis."""
    groups = 1 if group_sizes is None else group_sizes.shape[0]
    arrays, axes_of = [], []
    for array, axes in operands:
        axes = tuple(axes)
        named = [axis for axis in axes if axis not in (None, GROUPS)]
        if (len(axes) != array.ndim or not named or any(axis not in LAUNCH_AXES[hook] for axis in named)
                or any(extent != 1 for axis, extent in zip(axes, array.shape, strict=True) if axis is None)):
            raise ValueError(f"a {hook} side operand of shape {array.shape} does not take axes {axes}: one "
                             f"per dimension, each one of {LAUNCH_AXES[hook]}, {GROUPS!r} beside one, or "
                             f"None for a dimension of one")
        if GROUPS in axes and ('rows' in axes or axes.count(GROUPS) != 1
                               or array.shape[axes.index(GROUPS)] != groups):
            raise ValueError(f"a {hook} side operand of shape {array.shape} on axes {axes} names groups "
                             f"once, beside no rows, at the call's {groups} groups")
        arrays.append(array)
        axes_of.append(axes)
    return tuple(arrays), tuple(axes_of)


def _rows(arrays: tuple, axes_of: tuple, group_sizes: jax.Array, rows: int) -> tuple:
    """The side operands as XLA's program reads them: a `groups` axis as `rows` rows, each its group's
    value (`axes.at_rows`), every other operand as it is."""
    return tuple(side_axes_of.at_rows(array, axes, group_sizes, rows) if GROUPS in axes else array
                 for array, axes in zip(arrays, axes_of, strict=True))


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
    of its own (23c59c5fb), a quantized block's reload no longer computed its logits bit for bit
    (test_wrapped_runs)."""
    branches, default = grouped_matmul._on_platform(kernel, portable, interpret_on_cpu)

    def dispatched(*operands):
        return jax.lax.platform_dependent(*operands, default=default, **branches)

    from jax.core import Tracer

    if not any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(operands)):
        jax.eval_shape(dispatched, *operands)
    return dispatched(*operands)
