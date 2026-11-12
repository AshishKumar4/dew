"""XLA's portable grouped and dense products, the arm every grouped launch's selector falls back to and the
GEMM's own portable branch (`kernels.gemm`): its products as the call's precision multiplies them
(`_as_multiplied`), dense (`xla_dot`) and ragged (`xla_ragged_dot`, its rows placed past the groups,
`ragged_placement`), a step-scaled one (`step_scaled`), a projection differentiated by its own products
(`portable_projection`), and the platform choice between a kernel and them (`_on_platform`)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax.custom_derivatives import SymbolicZero

from ..derivatives import Protected, contracted, declared_tangent
from ..precision import accumulated, at_least_fp32, rounded_operand
from . import generation
from .templates.product import SCALE_DEPTH, eight


def _on_platform(kernel, fallback, interpret_on_cpu: bool):
    """`kernel` where the call lowers for its target's platform (CUDA, or a TPU's), and `fallback`
    everywhere else: a TPU's program, which never runs on CUDA, holds no CUDA branch, and a GPU's no TPU
    branch. A test's `interpret_on_cpu` interprets the kernel in a CPU process, as the GPU or TPU it
    emulates (`generation.assumed`), which it must state: interpreted for no device's facts, a launch
    would have none to derive its tile from, and the call would quietly run `fallback`. A new family
    member runs its template after the hardware-free checks, its native qualification an open item until
    measured (bible §6)."""
    target = generation.gpu_target()
    tpu = generation.tpu(target)
    branches = {'tpu' if tpu else 'cuda': functools.partial(kernel, interpret=False)}
    if interpret_on_cpu and jax.default_backend() == 'cpu':
        if not tpu and generation.capability(target.generation) is None:
            raise ValueError("interpret_on_cpu interprets the kernel as the GPU or TPU this CPU process "
                             "emulates; state it with generation.assumed(...)")
        branches['cpu'] = functools.partial(kernel, interpret=True)
    return branches, fallback


def _as_multiplied(tokens: jax.Array, kernel: jax.Array, precision: PrecisionLike) -> tuple:
    """`tokens` and `kernel` as XLA's portable product multiplies them, and its precision (`xla_ragged_dot`,
    `xla_dot`). A precision level (`lax.Precision`) on 16-bit operands is their product's at the default,
    every level the same exact products; an explicit dot algorithm a call states (`lax.DotAlgorithm`,
    `DotAlgorithmPreset`: its accumulator among it) reaches XLA as stated, run or refused by the backend,
    never another computation (audit K2-7)."""
    if any(eight(operand.dtype) for operand in (tokens, kernel)):
        # FP8 codes are each exact in bfloat16: the product is theirs there, summed in the preferred type
        # (the FP8 contract's reference arm, testing.md, FP8 products). XLA's GPU compiler rewrites no
        # ragged dot of widened FP8 codes into its narrower-summing FP8 GEMM (sm90's lowering of this
        # product holds no F8 GEMM, FS-1625-HLO, row 1625).
        return tokens.astype(jnp.bfloat16), kernel.astype(jnp.bfloat16), jax.lax.Precision.DEFAULT
    if all(jnp.finfo(operand.dtype).bits == 16 for operand in (tokens, kernel)):
        explicit = isinstance(precision, (jax.lax.DotAlgorithm, jax.lax.DotAlgorithmPreset))
        return tokens, kernel, precision if explicit else jax.lax.Precision.DEFAULT
    if tokens.dtype != kernel.dtype:
        # A 16-bit operand beside a wider one would reach the TPU kernel
        # 16-bit at HIGHEST too; widened, it is the same values (exactly),
        # and the wider operand keeps the precision the call asked for.
        wide = jnp.promote_types(tokens.dtype, kernel.dtype)
        return tokens.astype(wide), kernel.astype(wide), precision
    return tokens, kernel, precision


def xla_dot(tokens: jax.Array, matrix: jax.Array, *, precision: PrecisionLike = None,
            preferred_element_type: Dtype | None = None) -> jax.Array:
    """A dense product, every row through one matrix (`[depth, width]`), as `xla_ragged_dot` multiplies a
    group's rows (`_as_multiplied`), as XLA's own dot: no group, so no row masked, and no ragged dot to
    trace or to expand (a dense call's portable branch, `gemm`'s), summed as `precision.accumulated` sums a
    contraction (a CPU's float32 product as a vectorized dot sums it, row 4863)."""
    tokens, matrix, precision = _as_multiplied(tokens, matrix, precision)
    return accumulated(functools.partial(jax.lax.dot_general, dimension_numbers=(((1,), (0,)), ((), ())),
                                         precision=precision),
                       tokens, matrix, preferred_element_type=preferred_element_type)


def xla_ragged_dot(tokens: jax.Array, kernel: jax.Array, group_sizes: jax.Array, *,
                   precision: PrecisionLike = None,
                   preferred_element_type: Dtype | None = None) -> jax.Array:
    """`jax.lax.ragged_dot` as every backend runs it alike.

    A 16-bit product is exact at any precision, and XLA's TPU ragged-dot
    kernel refuses 16-bit operands at HIGHEST ("Bad lhs type"), so two of
    them multiply at DEFAULT: the same products, summed in the preferred
    type. A 16-bit operand beside a wider one is widened to it.
    XLA's TPU kernel writes values into the rows past the groups, where
    other backends write zeros, and its lhs cotangent is the same kernel.
    Those rows are zeroed going in and coming out, so neither the output
    nor a dropped row's gradient carries them. Its sums are `precision.accumulated`'s (a CPU's float32
    product as a vectorized dot sums it, row 4863).
    """
    tokens, kernel, precision = _as_multiplied(tokens, kernel, precision)
    # bincount widens under x64; TPU ragged-dot only lowers int32 counts.
    # Keep the wider domain for CPU calls whose row count actually needs it.
    index_dtype = jnp.int32 if tokens.shape[0] <= np.iinfo(np.int32).max else jnp.int64
    if group_sizes.dtype == jnp.int64:
        group_sizes = group_sizes.astype(index_dtype)
    grouped = jnp.arange(tokens.shape[0], dtype=index_dtype)[:, None] < jnp.sum(
        group_sizes, dtype=index_dtype
    )
    product = functools.partial(jax.lax.ragged_dot_general, ragged_dot_dimension_numbers=_GROUPED_ROWS,
                                precision=precision,
                                out_sharding=ragged_placement(((tokens, 0), (kernel, kernel.ndim - 1))))
    out = accumulated(product, jnp.where(grouped, tokens, 0), kernel, group_sizes,
                      preferred_element_type=preferred_element_type)
    return jnp.where(grouped, out, 0)


_GROUPED_ROWS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (1,)), ((), ())), lhs_ragged_dimensions=(0,), rhs_group_dimensions=(0,))
"""`jax.lax.ragged_dot`'s own: each group's rows of `[rows, depth]` times its `[depth, width]` matrix."""


def ragged_placement(sources: tuple) -> jax.sharding.NamedSharding | None:
    """Where a ragged product lands where an operand's type places it on a mesh with Explicit axes, split or
    not (`out_sharding`: JAX infers no placement for a ragged dot, "Explicit sharding inference for
    ragged_dot_general is not currently implemented"): each output dimension placed as the operand
    dimension it comes from (`sources`, each `(operand, axis)`, or None for one no operand dimension gives: a
    group axis), a contraction's sum whole, a mesh axis placing two dimensions placing the first alone; None
    where no operand's type names such a mesh."""
    from dew.mesh import mesh_axes

    def placed(operand):
        sharding = jax.typeof(operand).sharding
        explicit = isinstance(sharding, jax.sharding.NamedSharding) and sharding.mesh.explicit_axes
        return sharding if explicit else None

    shardings = [None if source is None else placed(source[0]) for source in sources]
    mesh = next((each.mesh for each in shardings if each is not None), None)
    if mesh is None:
        return None
    spec, used = [], set()
    for source, each in zip(sources, shardings, strict=True):
        axis = 0 if source is None else source[1] % source[0].ndim
        entry = None if each is None or axis >= len(each.spec) else each.spec[axis]
        axes = set(mesh_axes(entry)) if entry is not None else set()
        spec.append(entry if axes and not axes & used else None)
        used |= axes if spec[-1] is not None else set()
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))


def step_scaled(tokens: jax.Array, kernel: jax.Array, group_sizes: jax.Array, row_scales: jax.Array,
                column_scales: jax.Array, *, precision: PrecisionLike = None) -> jax.Array:
    """Each group's rows of `tokens` `[rows, depth]` times its matrix of `kernel` `[groups, depth, width]`,
    each `SCALE_DEPTH`-deep block's fp32 partial times its rows' scales (`row_scales`, `[depth / 128,
    rows]`) and its group's column scales (`column_scales`, `[groups, depth / 128, width / c]`, each
    scale `c` columns': 128 a weight block's, 1 a column's) before it is added into the fp32 sum, in the
    blocks' order: a blockwise-scaled FP8 product's sum (DeepSeek-V3's recipe, row 4518), the GEMM
    template's portable branch. The rows past the groups are zero."""
    blocks, rows = row_scales.shape
    groups, _, scaled = column_scales.shape
    width = kernel.shape[2]
    if tokens.shape[1] != blocks * SCALE_DEPTH or width % scaled:
        raise ValueError(f"scales {row_scales.shape} and {column_scales.shape} do not cover a "
                         f"{tokens.shape} x {kernel.shape} product in {SCALE_DEPTH}-deep blocks")
    owner = jnp.minimum(jnp.repeat(jnp.arange(groups), group_sizes, total_repeat_length=rows), groups - 1)

    def block(index, total):
        lhs = jax.lax.dynamic_slice_in_dim(tokens, index * SCALE_DEPTH, SCALE_DEPTH, axis=1)
        rhs = jax.lax.dynamic_slice_in_dim(kernel, index * SCALE_DEPTH, SCALE_DEPTH, axis=1)
        partial = xla_ragged_dot(lhs, rhs, group_sizes, precision=precision,
                                 preferred_element_type=jnp.float32)
        columns = jnp.repeat(column_scales[:, index][owner], width // scaled, axis=1)
        return total + partial * row_scales[index][:, None].astype(jnp.float32) * columns.astype(jnp.float32)

    return jax.lax.fori_loop(0, blocks, block, jnp.zeros((rows, width), jnp.float32))


def _portable_projection(x: jax.Array, kernel: jax.Array, group_sizes: jax.Array,
                dtype: Dtype | None, precision: PrecisionLike, out_dtype: Dtype | None = None) -> jax.Array:
    """The grouped projection on `xla_ragged_dot`, in every differentiation mode: `x`'s rows `[rows, in]`
    through each group's matrix of `kernel` `[groups, in, out]`, both cast to `dtype` (flax's promotion
    where None), summed in at least fp32 and rounded once. Its tangent is `dx @ Q(kernel) + Q(x) @
    dkernel`, `Q` the rounded operands, summed over the widest of the compute, input and kernel dtypes,
    so an input gradient keeps its input's dtype and a kernel gradient its master's: the portable
    projection `dew.nn.moe.expert_projection`'s XLA route and the GEMM template's portable branch run.
    `out_dtype`, where given, is the dtype its sum rounds to in place of the products' (the template's
    fp32 accumulator, which its epilogue reads). A protected span (`portable_projection`,
    `dew.nn.derivatives.declared_tangent`): each rounded operand and barrier holds a rounding the algebra
    pass must not move."""
    x, kernel = promote_dtype(x, kernel, dtype=dtype)
    accumulated = xla_ragged_dot(x, kernel, group_sizes, precision=precision,
                                 preferred_element_type=at_least_fp32(x.dtype))
    return accumulated.astype(out_dtype or x.dtype)


def _portable_projection_jvp(dtype: Dtype | None, precision: PrecisionLike, out_dtype: Dtype | None,
                    primals: tuple[jax.Array, jax.Array, jax.Array],
                    tangents: tuple[jax.Array, jax.Array, jax.Array]
                    ) -> tuple[jax.Array, jax.Array | SymbolicZero]:
    x, kernel, group_sizes = primals
    dx, dkernel, _ = tangents
    output = jnp.asarray(portable_projection(x, kernel, group_sizes, dtype, precision, out_dtype))
    work = at_least_fp32(jnp.result_type(output.dtype, x.dtype, kernel.dtype))
    compute = jnp.dtype(dtype) if dtype is not None else jnp.result_type(x.dtype, kernel.dtype)
    # A symbolic zero tangent contributes no term, so none is formed.
    live = [(name, tangent) for name, tangent in (('x', dx), ('kernel', dkernel))
            if not isinstance(tangent, SymbolicZero)]
    if not live:
        return output, SymbolicZero(jax.typeof(output))
    # The tangents keep their dtypes to this point whatever produced them, so an activation's bf16 cotangent
    # is not fused into a wider expression: XLA's excess-precision simplification (`SimplifyFPConversions`,
    # on under `xla_allow_excess_precision`, its default) drops a round trip through bf16 a fusion forms.
    held = dict(zip((name for name, _ in live),
                    jax.lax.optimization_barrier(tuple(tangent for _, tangent in live)),
                    strict=True))
    terms = []
    # Each rounded operand is held in the work dtype, which holds the rounding whatever the operand's own
    # (an fp16 master's largest value rounds to bf16's 65536, which fp16 does not hold).
    if 'x' in held:
        matrix = jnp.asarray(rounded_operand(kernel.astype(work), compute))
        terms.append(xla_ragged_dot(
            held['x'].astype(work), matrix.astype(work), group_sizes, precision=precision,
            preferred_element_type=work))
    if 'kernel' in held:
        inputs = jnp.asarray(rounded_operand(x.astype(work), compute))
        terms.append(xla_ragged_dot(
            inputs.astype(work), held['kernel'].astype(work), group_sizes, precision=precision,
            preferred_element_type=work))
    total = terms[0] if len(terms) == 1 else terms[0] + terms[1]
    tangent = jax.lax.optimization_barrier(total.astype(output.dtype))
    return output, tangent


portable_projection = contracted("test_a_portable_grouped_projection_holds_every_order")(
    declared_tangent(Protected(), _portable_projection, _portable_projection_jvp,
                     nondiff_argnums=(3, 4, 5), symbolic_zeros=True))
