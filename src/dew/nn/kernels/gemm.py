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

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import PrecisionLike
from jax.custom_derivatives import zero_from_primal
from jax.extend import core as jax_core

from dew import _cache
from dew.algebra import bits
from dew.nn.derivatives import KernelGlue, contracted, declared_derivative, declared_tangent
from dew.nn.kernels import generation, grouped_matmul, selection
from dew.nn.kernels.templates import axes as side_axes_of, mosaic
from dew.nn.kernels.templates.axes import GROUPS
from dew.nn.kernels.templates.gemm import mosaic_gpu, rows_padded
from dew.nn.kernels.templates.product import declared
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import seen_whole

LAUNCH_AXES = {'prologue': ('rows', 'depth'), 'epilogue': ('rows', 'columns')}
"""The launch axes each hook's side operands may follow; `groups` too beside them (`_split`), which a launch
reads at its tile's group (`axes.GROUPS`) and XLA's program at each row (`axes.at_rows`)."""


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
    matrix for a dense product), read in place: the launches take the layer's groups and read their
    matrices where they lie in the stack, from the layer's first (`_grouped`), so no slice of the stack
    is made and they tile, split and sum as the call over the layer's matrices alone. Its gradient is
    the stack's, every other layer's zero. `paired`, a second matrix
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
    prologue_arrays, prologue_axes = _split(prologue_operands, 'prologue', group_sizes)
    epilogue_arrays, epilogue_axes = _split(epilogue_operands, 'epilogue', group_sizes)
    rhs, group_sizes, first = _grouped(lhs, rhs, group_sizes, layer)
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    several = out_dtypes is not None
    plain = {'transpose_rhs': transpose_rhs, 'interpret_on_cpu': interpret_on_cpu, 'precision': precision,
             'dense': dense}

    def value(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first):
        return _gemm(lhs.astype(compute), rhs, group_sizes, prologue, prologue_arrays, prologue_axes,
                     epilogue, epilogue_arrays, epilogue_axes, dtypes, several=several, paired=paired,
                     first=first, **plain)

    def portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first):
        return _portable_gemm(lhs if straight else lhs.astype(compute), rhs, group_sizes, prologue,
                              prologue_arrays, prologue_axes, epilogue, epilogue_arrays, epilogue_axes,
                              dtypes, several, transpose_rhs, precision, compute if straight else None,
                              paired=paired, dense=dense, first=first)

    def forward(*operands):
        return value(*operands), ()

    def backward(saved, operands, cotangent, live):
        return _pulled(operands, cotangent, live)

    def _pulled(operands, cotangent, live):
        lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first = operands
        # The operands a reverse pass differentiates, by their place among the operands' leaves: a frozen
        # matrix's kernel gradient, or an operand no reverse pass reaches, is never formed.
        wanted = set(live)
        held = {**plain, 'first': first}
        matrices = (rhs,) if paired is None else (rhs, paired)
        places = (1, 3 + len(prologue_arrays) + len(epilogue_arrays))[:len(matrices)]
        cotangents = cotangent if several else (cotangent,)
        acc_shape = (lhs.shape[0], rhs.shape[1] if transpose_rhs else rhs.shape[2])
        d_accs, d_epilogue, narrow = _cotangents(lhs.astype(compute), matrices, group_sizes, prologue,
                                                 prologue_arrays, prologue_axes, epilogue, epilogue_arrays,
                                                 epilogue_axes, cotangents, dtypes, compute, acc_shape, held)
        # Each accumulator's cotangent meets the forward's rounded operands as the portable program's
        # derivative has it: in `lhs`'s dtype where that holds it exactly (a cotangent no wider, passed
        # through), wide otherwise, each product rounding once.
        works = tuple(compute if each else at_least_fp32(d.dtype)
                      for d, each in zip(d_accs, narrow, strict=True))
        d_products = tuple(d.astype(work) for d, work in zip(d_accs, works, strict=True))
        d_lhs, d_prologue = None, ()
        if wanted & set(range(3 + len(prologue_arrays))) - {1, 2}:
            d_lhs, d_prologue = _input_cotangents(lhs, d_products, matrices, group_sizes, prologue,
                                                  prologue_arrays, prologue_axes, compute, held)
        gradients = [None if place not in wanted
                     else _kernel_gradient(lhs, d_product, matrix, group_sizes, prologue, prologue_arrays,
                                           prologue_axes, compute, work if not each else None, held)
                     for place, d_product, matrix, work, each in zip(places, d_products, matrices, works,
                                                                     narrow, strict=True)]
        leaves = [d_lhs, gradients[0], zero_from_primal(group_sizes),
                  *(d_prologue or (None,) * len(prologue_arrays)),
                  *(_as_cotangent(d, a) for d, a in zip(d_epilogue, epilogue_arrays, strict=True)),
                  *gradients[1:], *(() if first is None else (zero_from_primal(first),))]
        return tuple(leaves[index] for index in live)

    differentiable = contracted("test_a_grouped_gemm_holds_every_order")(
        generation.linear_tangent(forward, grouped_matmul.GROUPED.bound(backward), portable,
                                   live_cotangents=True, kind=KernelGlue()))
    return differentiable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first)


def gemm_t(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None, *, prologue=None,
           prologue_operands: tuple = (), out_dtype, precision: PrecisionLike = None,
           interpret_on_cpu: bool = False) -> jax.Array:
    """Each group's rows of `prologue(lhs)` transposed times its rows of `rhs`: `[groups, depth, width]`
    in `out_dtype`, an empty group's zero, summed as `gemm`'s `precision` asks; without `group_sizes`,
    every row one group, a dense product's (`[1, depth, width]`). A value, as `gemm`'s reverse mode
    takes it."""
    prologue_arrays, prologue_axes = _split(prologue_operands, 'prologue', group_sizes)
    return _gemm_t(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, out_dtype=out_dtype,
                   precision=precision, interpret_on_cpu=interpret_on_cpu)


def _gemm_t(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, *, out_dtype,
            precision: PrecisionLike, interpret_on_cpu: bool) -> jax.Array:
    """`gemm_t` of side operands checked (`_split`), its groups the call's."""
    dense = group_sizes is None
    group_sizes = jnp.full((1,), lhs.shape[0], jnp.int32) if dense else group_sizes.astype(jnp.int32)

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
            sides = _rows(prologue_arrays, prologue_axes, group_sizes, lhs.shape[0])
            lhs = prologue(lhs, *sides).astype(lhs.dtype)
        lhs = jnp.where(routed[:, None], lhs, 0)
        wide = at_least_fp32(lhs.dtype)  # an fp64 product's sum in fp64
        # Each group's rows contracted, XLA's ragged dot's own weight gradient (its transpose's form) on the
        # operands as they are: a pullback from a matrix made here would hand a vmapped call (a pipeline
        # stage's) an unbatched operand, which ragged_dot's batching rule refuses. The operands keep their
        # dtype, summed in `wide`: a bf16 product is exact in fp32 either way, where operands widened first
        # made the GPU's product an fp32 one (an sm120's 4096-token gate_up weight gradient 5.91 ms against
        # 2.72 for the bf16 product).
        return jax.lax.ragged_dot_general(
            lhs, rhs.astype(lhs.dtype), group_sizes, _CONTRACTED_ROWS,
            precision=precision, preferred_element_type=wide).astype(out_dtype)

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays)


LOW_RANK_ROW = 'ledger 1304'
"""The adapted product's fused form (docs/design/kernel-templates.md, Appendix C row 26)."""

LOW_RANK: selection.Op[None] = selection.Op('low_rank', (
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
    copy takes, at least 16 bytes."""

    words: jax.Array
    scales: jax.Array
    scale: jax.Array

    @classmethod
    def quantized(cls, x: jax.Array) -> "NvFp4":
        """`x` (`[depth, width]`) quantized to NVFP4 (`bits.nvfp4_quantized`: its block scales and tensor
        scale, each value rounded to E2M1), its codes packed in the staged order."""
        words, scales, scale = bits.nvfp4_quantized(jnp.asarray(x, jnp.float32))
        return cls.of(_codes(words), scales, scale)

    @classmethod
    def of(cls, codes: jax.Array, scales: jax.Array, scale) -> "NvFp4":
        """The matrix of `codes` (`[depth, width]`, each an E2M1 value, in the depths' own order), its block
        scales `scales` (`[depth / 16, width]` float8_e4m3fn) and its tensor scale `scale`: a checkpoint's
        NVFP4 weight, its codes packed in the staged order, K-major."""
        if codes.shape[0] % NVFP4_STAGE:
            raise ValueError(f"an NVFP4 matrix's depth is whole {NVFP4_STAGE}s, not {codes.shape[0]}")
        return cls(_staged_words(codes).T, scales.T, jnp.asarray(scale, jnp.float32))

    def codes(self) -> jax.Array:
        """Its codes' values (`[depth, width]` float32, each code's E2M1 value) in the depths' own order."""
        return _codes(self.words.T, staged=True)

    def decoded(self) -> jax.Array:
        """Its values over `bits.NVFP4_OWED` and the tensor scale, exactly, in bfloat16 (`[depth, width]`):
        each code times its block's scale over 2^8, as the mainloop's stage decode forms it."""
        owed = self.scales.T.astype(jnp.float32) * jnp.float32(1 / bits.NVFP4_OWED)
        return (self.codes() * jnp.repeat(owed, 16, axis=0)).astype(jnp.bfloat16)

    @property
    def factor(self) -> jax.Array:
        """The factor its decode defers to the product's accumulator (`bits.NVFP4_OWED` times the tensor's
        scale): `bits.nvfp4_decoded` keeps each value within bfloat16's normal range by owing 2^8."""
        return jnp.float32(bits.NVFP4_OWED) * jax.lax.stop_gradient(jnp.asarray(self.scale, jnp.float32))


NVFP4_GROUP = 128
"""The depths an NVFP4 matrix's staged order groups (`staged_rows`): a group's sixteen word rows decode, code
by code, into its eight sixteen-row depth slices."""
NVFP4_STAGE = mosaic_gpu.NVFP4_STEP
"""The depths a stage of the mainloop's NVFP4 decode reads: two groups, so each column's scales are a
16-byte row (a TMA copy's least)."""


def staged_rows(depth: int) -> np.ndarray:
    """The depth each code of the staged order holds (`NvFp4`): within each `NVFP4_GROUP`-deep group, packed
    row `8w + j` (word `w`'s code `j`) holds depth `16j + w`, so a stage's word rows decode, code by code,
    into its sixteen-row depth slices, the block of one scale each (the mainloop's stage decode)."""
    if depth % NVFP4_GROUP:
        raise ValueError(f"an NVFP4 matrix's depth is whole {NVFP4_GROUP}-deep groups, not {depth}")
    word, code = np.divmod(np.arange(NVFP4_GROUP), 8)
    within = 16 * code + word
    return (np.arange(0, depth, NVFP4_GROUP)[:, None] + within[None]).reshape(-1)


def _codes(words: jax.Array, *, staged: bool = False) -> jax.Array:
    """The codes' values of `words` (each its E2M1 value, float32 `[depth, width]`), in the depths' own order
    where `staged` (`staged_rows`), else the words' own: `bits.nvfp4_decoded` at a block scale of 2^8."""
    unit = jnp.full((words.shape[0] // 2, words.shape[1]), bits.NVFP4_OWED, jnp.float8_e4m3fn)
    codes = bits.nvfp4_decoded(words, unit).astype(jnp.float32)
    return codes[np.argsort(staged_rows(codes.shape[0]))] if staged else codes


def _staged_words(codes: jax.Array) -> jax.Array:
    """`codes` (values, `[depth, width]`, in the depths' own order) packed in the staged order (`NvFp4`)."""
    ordered = codes[staged_rows(codes.shape[0])]
    bits_ = jax.lax.bitcast_convert_type(ordered.astype(jnp.float32), jnp.uint32)
    return bits.packed(bits.encoded(bits_, bits.E2M1).T, bits.E2M1).T


NVFP4_ROW = 'ledger 3621'
"""The NVFP4 product's in-stage decode (K51)."""


@dataclasses.dataclass(frozen=True)
class PackedCase:
    """An NVFP4 product as its selector reads it (`NVFP4`): the tile the template's packed launch plans for
    it on the target (`grouped_matmul.MosaicPackedGmm`), None where it plans none."""

    tile: mosaic_gpu.Tiles | None


NVFP4: selection.Op[PackedCase] = selection.Op('nvfp4', (
    selection.Kernel('packed', (
        selection.Requirement(
            'planned', lambda case: case.tile is not None, "the template plans no tile of the packed "
            "product here: a Mosaic GPU family's, its depth whole 256s and its width whole 64-column tiles",
            NVFP4_ROW),
        selection.Requirement(
            'measured', lambda case: False, "no receipt yet has the in-stage decode beat the decoded "
            "matrix's product", NVFP4_ROW, kind='measured'))),
    selection.Kernel('xla', (), generic=True)))
"""An NVFP4 product's selector (`nvfp4_product`): the template's mainloop decoding each stage of the packed
matrix in place (`mosaic_gpu.gmm`'s `scales`) where named (`generation.chosen(nvfp4='packed')`) and planned,
which passes the measured requirement no receipt has met yet; else the matrix decoded once a call and its
product (`gemm`)."""


def nvfp4_product(lhs: jax.Array, matrix: NvFp4, *, out_dtype, interpret_on_cpu: bool = False) -> jax.Array:
    """`lhs · W` of an NVFP4 matrix `W` (`NvFp4`), its weights frozen (a quantized checkpoint's): `lhs`
    rounded to bfloat16, each value of `W` decoded exactly to bfloat16 (`NvFp4.decoded`, its block's
    scale applied, 2^8 owed), the products summed in fp32, each step's apart (`templates.product.promoted`,
    row 4331), and the accumulator times the deferred factor
    (`NvFp4.factor`) before the one rounding to `out_dtype`: the deferred factor is the epilogue's first
    operation, ahead of any output hook (PLAN piece 10). Its selector (`NVFP4`) runs the template's mainloop
    decoding each stage in place (`_packed`) where named, else decodes the matrix once a call (`gemm`)."""
    rows = lhs.astype(jnp.bfloat16)
    launch = grouped_matmul.MosaicPackedGmm()
    width, depth = matrix.words.shape[0], 8 * matrix.words.shape[1]
    bound = launch.bind(jax.ShapeDtypeStruct(rows.shape, rows.dtype),
                        jax.ShapeDtypeStruct((1, width, depth), jnp.bfloat16),
                        jax.ShapeDtypeStruct((1,), jnp.int32), out_dtype='float32', trans_rhs=True,
                        interpret=interpret_on_cpu, dense=True, exact_sums=True)
    target = generation.gpu_target()
    tile = launch.derived(bound, target) if launch.eligible(bound, target) else None
    if selection.select(NVFP4, PackedCase(tile)).kernel == 'packed':
        assert tile is not None, "the selector names the packed kernel where it plans a tile"
        summed = _packed(rows, matrix.words, matrix.scales, tile, interpret_on_cpu)
    else:
        # Each product of two bfloat16 values is exact in fp32, so the default precision is the exact
        # product's, whatever precision the program defaults to.
        summed = gemm(rows, matrix.decoded(), None, out_dtype=jnp.float32,
                      precision=jax.lax.Precision.DEFAULT, interpret_on_cpu=interpret_on_cpu)
    return (summed * matrix.factor).astype(out_dtype)


def _packed(rows: jax.Array, words: jax.Array, scales: jax.Array, tile: mosaic_gpu.Tiles,
            interpret: bool) -> jax.Array:
    """The packed matrix's product in fp32 (`nvfp4_product`'s packed kernel): the template's mainloop, each
    stage of `words` and `scales` decoded in place (`mosaic_gpu.gmm`'s `scales`), differentiable at every
    order as `gemm` is (`generation.linear_tangent`): its tangent the decoded product's in XLA, its reverse
    mode the rows' gradient through the decoded values (the frozen words and scales taking none)."""
    launch = grouped_matmul.MosaicPackedGmm()

    def forward(rows, words, scales):
        out = mosaic_gpu.gmm(rows, words[None], jnp.asarray([rows.shape[0]], jnp.int32), tiles=tile,
                             transpose_rhs=True, out_dtype=jnp.float32,
                             multiprocessors=launch._launched(tile, interpret),
                             interpret=grouped_matmul.INTERPRETED if interpret else False, dense=True,
                             exact_sums=True, scales=scales[None])
        return out, ()

    def backward(saved, operands, cotangent, live):
        # The cotangent times the decoded values' transpose, summed in fp32, rounded once to the rows'.
        del saved
        rows, words, scales = operands
        leaves = (gemm(cotangent.astype(jnp.bfloat16), NvFp4(words, scales, jnp.float32(1)).decoded(), None,
                       transpose_rhs=True, out_dtype=rows.dtype, precision=jax.lax.Precision.DEFAULT,
                       interpret_on_cpu=interpret),
                  zero_from_primal(words), zero_from_primal(scales))
        return tuple(leaves[index] for index in live)

    def portable(rows, words, scales):
        decoded = NvFp4(words, scales, jnp.float32(1)).decoded()
        return jnp.dot(rows, decoded, preferred_element_type=jnp.float32)

    differentiable = generation.linear_tangent(forward, backward, portable, live_cotangents=True,
                                                kind=KernelGlue())
    return differentiable(rows, words, scales)


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
    appended pair in place is row 1304's next form."""
    return _low_rank(lhs, kernel, a, b, jnp.asarray(scale, jnp.float32), out_dtype, precision,
                     interpret_on_cpu)


def _rank_product(lhs: jax.Array, factor: jax.Array, scale: jax.Array, precision: PrecisionLike) -> jax.Array:
    """`scale · (lhs · factor)`, summed and kept in at least fp32: the rank-wide operand an adapted product
    appends to its depth, rounded once to `lhs`'s dtype where it is appended (`_appended`)."""
    wide = at_least_fp32(lhs.dtype)
    summed = jnp.dot(lhs, factor.astype(lhs.dtype), precision=precision, preferred_element_type=wide)
    return scale.astype(wide) * summed


def _appended(base: jax.Array, rank_part: jax.Array, axis: int) -> jax.Array:
    """`base` with `rank_part` after it along `axis`, the rank padded with zeros to whole steps of the widest
    depth the template's families take on its dtype (a 128-byte swizzle's row, `mosaic.SWIZZLE_BYTES`), so
    the extended depth divides into the base's own steps."""
    step = mosaic.swizzle_columns(base.dtype, mosaic.SWIZZLE_BYTES)
    rank_part = rows_padded(rank_part.astype(base.dtype), axis, step)
    return jnp.concatenate([base, rank_part], axis=axis)


def _extended(lhs, kernel, a, b, scale, out_dtype, precision, interpret_on_cpu):
    """`low_rank`'s value and its appended operand `t`, in at least fp32."""
    t = _rank_product(lhs, a, scale, precision)
    out = gemm(_appended(lhs, t, 1), _appended(kernel, b, 0), out_dtype=out_dtype, precision=precision,
               interpret_on_cpu=interpret_on_cpu)
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
    dy = cotangent.astype(lhs.dtype)
    wide = at_least_fp32(lhs.dtype)
    # The cotangent of `t` before its scale, dy · bᵀ, summed wide, and after it: the scale taken before any
    # product with it, as the adapter's own program takes it (a product near float32's largest that the
    # scale brings back in range never overflows first).
    through = jnp.dot(dy, b.astype(lhs.dtype).T, precision=precision, preferred_element_type=wide)
    scaled = scale.astype(wide) * through
    d_lhs = d_kernel = d_a = d_b = d_scale = None
    if wants_lhs:
        d_lhs = gemm(_appended(dy, scaled.astype(lhs.dtype), 1), _appended(kernel, a, 1), transpose_rhs=True,
                     out_dtype=lhs.dtype, precision=precision, interpret_on_cpu=interpret_on_cpu)
    if wants_kernel:
        d_kernel = gemm_t(lhs, dy, None, out_dtype=kernel.dtype, precision=precision,
                          interpret_on_cpu=interpret_on_cpu)[0]
    # The factors' gradients from the wide `scaled` and `t` (the rank-wide products' sums, unrounded), as
    # the adapter's own program differentiates its fp32 low-rank products.
    if wants_a:
        d_a = jnp.dot(lhs.T.astype(wide), scaled, precision=precision,
                      preferred_element_type=wide).astype(a.dtype)
    if wants_b:
        # `t` holds the scale already: tᵀ · dy is the factor's gradient whole.
        d_b = jnp.dot(t.T, dy.astype(wide), precision=precision, preferred_element_type=wide).astype(b.dtype)
    if wants_scale:
        unscaled = jnp.dot(lhs, a.astype(lhs.dtype), precision=precision, preferred_element_type=wide)
        d_scale = jnp.sum(through * unscaled).astype(scale.dtype)
    return d_lhs, d_kernel, d_a, d_b, d_scale


# The launches' derivative glue (`derivatives.KernelGlue`): an operand no reverse pass reaches is a symbolic
# zero, whose gradient is never formed.
_low_rank = declared_derivative(KernelGlue(), _low_rank_value, _low_rank_forward, _low_rank_backward,
                                nondiff_argnums=(5, 6, 7), symbolic_zeros=True)


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
    its reverse mode make them. Where none does, each one's route is reported, as XLA runs them. Decided
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
                             refusal=route.refusal, ran=route.ran, forced=route.forced)
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
          first: jax.Array | None = None, summed_as: tuple = ()):
    """`gemm`'s value, on grouped operands; `summed_as`, the outputs of the product whose accumulator it
    recomputes (a reverse pass's, at fp32), whose sums it repeats (`product.promoted`: an fp32 output's
    exact sums, a 16-bit one's not); `first`, a stacked `rhs`'s matrix its first group reads
    (`_grouped`)."""
    def kernel(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first, *, interpret):
        before_sides, before_axes, before_broadcast = _launchable(prologue_arrays, prologue_axes)
        after_sides, after_axes, after_broadcast = _launchable(epilogue_arrays, epilogue_axes)
        hooks = grouped_matmul.Hooks(prologue, before_axes, epilogue, after_axes,
                                     tuple(dtype.name for dtype in dtypes) if several else None,
                                     prologue_broadcast=before_broadcast, epilogue_broadcast=after_broadcast,
                                     prologue_dtypes=_dtypes(prologue, lhs, before_sides),
                                     epilogue_dtypes=_dtypes(epilogue, _ACCUMULATOR, after_sides))
        product = declared(lhs.dtype, precision, outputs=summed_as or dtypes, operands=(lhs.dtype, rhs.dtype))
        call = grouped_matmul.gmm_call(lhs, rhs.astype(lhs.dtype), group_sizes, before_sides, after_sides,
                                       None if paired is None else paired.astype(lhs.dtype), first,
                                       out_dtype=dtypes[0], trans_rhs=transpose_rhs, hooks=hooks,
                                       interpret=interpret, promoted=product.promoted, dense=dense)
        # A call no launch takes (a TPU's widths its blocks do not divide) runs XLA's program; a paired one no
        # launch takes whole its two products apart, each through the launches that take it (`_apart`).
        launch = grouped_matmul.launched(*call, product=product, whole=seen_whole(lhs) and seen_whole(rhs))
        if launch is grouped_matmul.APART:
            return _apart(lhs, (rhs, paired), group_sizes, prologue, prologue_arrays, prologue_axes, epilogue,
                          epilogue_arrays, epilogue_axes, dtypes, several=several,
                          summed_as=summed_as or dtypes,
                          transpose_rhs=transpose_rhs, interpret_on_cpu=interpret_on_cpu, precision=precision,
                          dense=dense)
        if not callable(launch):
            return portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first)
        return launch()

    def portable(lhs, rhs, group_sizes, prologue_arrays, epilogue_arrays, paired, first):
        return _portable_gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue,
                              epilogue_arrays, epilogue_axes, dtypes, several, transpose_rhs, precision,
                              paired=paired, dense=dense, first=first)

    return _dispatched(kernel, portable, interpret_on_cpu, lhs, rhs, group_sizes, prologue_arrays,
                       epilogue_arrays, paired, first)


def _cotangents(lhs, matrices: tuple, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue,
                epilogue_arrays, epilogue_axes, cotangents: tuple, dtypes: tuple, compute, acc_shape: tuple,
                plain: dict) -> tuple[tuple, tuple, tuple]:
    """Each accumulator's cotangent (one a matrix of `matrices`: the product's, and its `paired`'s), the
    epilogue side operands' and whether each accumulator's passes an output's through in `compute`'s dtype
    (`_passes_cotangent`): the outputs' own without an epilogue, else through the epilogue's VJP at the
    accumulators recomputed as the outputs (`dtypes`) summed them, where the VJP reads them."""
    count = len(matrices)
    if epilogue is None:
        d_accs = tuple(cotangents[:count])
        return d_accs, (), tuple(jnp.promote_types(d.dtype, compute) == compute for d in d_accs)
    rows = functools.partial(_rows, axes_of=epilogue_axes, group_sizes=group_sizes, rows=acc_shape[0])
    sides = jax.eval_shape(rows, epilogue_arrays)
    # The accumulator the products sum in: fp32, an fp64 product's fp64 (`_portable_gemm`).
    acc = jax.ShapeDtypeStruct(acc_shape, at_least_fp32(compute))
    if _reads_accumulators(epilogue, count, acc, sides):
        accs = _as_tuple(_gemm(lhs, matrices[0], group_sizes, prologue, prologue_arrays, prologue_axes, None,
                               (), (), (acc.dtype,) * count, several=count > 1,
                               paired=matrices[1] if count > 1 else None, summed_as=dtypes, **plain))
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
    summed as the gradient's own dtype sums (`_gemm`'s `summed_as`)."""
    out = (jnp.dtype(lhs.dtype),) if len(matrices) == 1 else (at_least_fp32(lhs.dtype),)
    options = {'interpret_on_cpu': plain['interpret_on_cpu'], 'precision': plain['precision'],
               'transpose_rhs': not plain['transpose_rhs'], 'dense': plain['dense'], 'first': plain['first']}
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
    forward's rounded operand in the cotangent's wider dtype `work` otherwise). A stack's whose groups read
    its matrices from `plain['first']` (`_grouped`) is theirs there and zero at every other."""
    options = {'interpret_on_cpu': plain['interpret_on_cpu'], 'precision': plain['precision']}
    sizes = None if plain['dense'] else group_sizes
    if work is None:
        gradient = _gemm_t(lhs.astype(compute), d_product, sizes, prologue, prologue_arrays, prologue_axes,
                           out_dtype=matrix.dtype, **options)
    else:
        operand = (lhs.astype(compute) if prologue is None else prologue(
            lhs, *_rows(prologue_arrays, prologue_axes, group_sizes, lhs.shape[0])).astype(lhs.dtype))
        gradient = _gemm_t(operand.astype(work), d_product, sizes, None, (), (), out_dtype=matrix.dtype,
                           **options)
    gradient = jnp.swapaxes(gradient, 1, 2) if plain['transpose_rhs'] else gradient
    if plain['first'] is None:
        return gradient
    return jax.lax.dynamic_update_slice_in_dim(jnp.zeros(matrix.shape, gradient.dtype), gradient,
                                               plain['first'], 0)


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


def _portable_gemm(lhs, rhs, group_sizes, prologue, prologue_arrays, prologue_axes, epilogue, epilogue_arrays,
                   epilogue_axes, dtypes, several: bool, transpose_rhs: bool, precision: PrecisionLike,
                   compute=None, paired: jax.Array | None = None, dense: bool = False,
                   first: jax.Array | None = None):
    """`gemm`'s arithmetic in XLA: the kernels' products of `lhs`'s dtype summed in at least fp32 (an fp64
    product's in fp64, which no kernel takes), one rounding, a `dense` call's as XLA's own dot
    (`grouped_matmul.xla_dot`: one group of every row, nothing to mask, no ragged dot to trace);
    with `compute`, the products of the operands rounded to it, its tangent the MoE projection's
    (`grouped_matmul.portable_projection`, whose sum, at least fp32, it reads); with `paired`, both matrices'
    products, the epilogue's two accumulators; with `first`, the groups' matrices of a stacked `rhs` from
    that one (`_grouped`), sliced out, as XLA's ragged dot reads a matrix a group."""
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
