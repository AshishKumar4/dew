"""XLA's portable grouped and dense products, the arm every grouped launch's selector falls back to and the
GEMM's own portable branch (`kernels.gemm`): its products as the call's precision multiplies them
(`as_multiplied`), dense (`xla_dot`) and ragged (`xla_ragged_dot`, its rows placed past the groups,
`ragged_placement`), a step-scaled one (`step_scaled`), a projection differentiated by its own products
(`portable_projection`), and the platform choice between a kernel and them (`_on_platform`)."""

import contextlib
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax.custom_derivatives import SymbolicZero

from ..derivatives import Protected, contracted, declared_tangent
from ..precision import accumulated, at_least_fp32, rounded_operand, twice_scaled
from . import generation
from .templates.product import SCALE_DEPTH, eight


def interpreted(interpret_on_cpu: bool) -> bool:
    """Whether a call asking `interpret_on_cpu` interprets its kernel: where the context places it on a CPU
    (`generation.placed_platform`), not wherever this process's default backend is one. A GPU's program a
    CPU host compiles ahead (its default device a compile-only GPU's, `dew.training.ahead`) holds no
    interpreted branch: the interpreter's host callbacks' ordered effect, every branch's in the program, took
    a token parameter on its CUDA executable, which its run on the GPU did not pass (row 8099)."""
    return interpret_on_cpu and generation.placed_platform() == 'cpu'


@contextlib.contextmanager
def interpreted_context(interpret_on_cpu: bool):
    """Bind `generation.assumed(generation.interpreted_target())` while `interpreted(interpret_on_cpu)` holds
    (row 12158)."""
    with (generation.assumed(generation.interpreted_target())
          if interpreted(interpret_on_cpu) else contextlib.nullcontext()):
        yield


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
    if interpreted(interpret_on_cpu):
        if not tpu and generation.capability(target.generation) is None:
            raise ValueError("interpret_on_cpu interprets the kernel as the GPU or TPU this CPU process "
                             "emulates; state it with generation.assumed(...)")
        branches['cpu'] = functools.partial(kernel, interpret=True)
    return branches, fallback


def committed_device(operands):
    """A process-local device of concrete `operands` when every committed leaf shares one platform
    (including homogeneous multi-host sharding), or None if uncommitted, traced, empty, or mixed-platform."""
    from jax.core import Tracer

    devices = {
        device
        for leaf in jax.tree.leaves(operands)
        if isinstance(leaf, jax.Array) and not isinstance(leaf, Tracer) and leaf.committed
        for device in leaf.devices()
    }
    platforms = {device.platform for device in devices}
    if len(platforms) != 1:
        return None
    local = next((device for device in devices if device.process_index == jax.process_index()), None)
    return local if local is not None else jax.local_devices(backend=next(iter(platforms)))[0]


@contextlib.contextmanager
def committed_context(operands):
    """Bind `jax.default_device(committed_device(operands))` for concrete committed `operands` (row 12158)."""
    from jax.core import Tracer

    if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(operands)):
        yield
        return
    device = committed_device(operands)
    with jax.default_device(device) if device is not None else contextlib.nullcontext():
        yield


def dispatched(kernel, portable, interpret_on_cpu: bool, *operands):
    """`kernel` where the call lowers for CUDA, interpreted on the CPU where asked, `portable` elsewhere
    (`_on_platform`). Resolves concrete committed placement before `_on_platform`, traces abstractly so every
    branch selects, then runs eagerly on the committed device (row 11887)."""
    from jax.core import Tracer

    def run():
        branches, default = _on_platform(kernel, portable, interpret_on_cpu)

        def call(*args):
            return jax.lax.platform_dependent(*args, default=default, **branches)

        return call

    if not any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(operands)):
        with committed_context(operands):
            call = run()
            jax.eval_shape(call, *operands)
            return call(*operands)
    return run()(*operands)


def as_multiplied(tokens: jax.Array, kernel: jax.Array, precision: PrecisionLike) -> tuple:
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
    group's rows (`as_multiplied`), as XLA's own dot: no group, so no row masked, and no ragged dot to
    trace or to expand (a dense call's portable branch, `gemm`'s), summed as `precision.accumulated` sums a
    contraction (XLA's own on a CPU, row 5448)."""
    tokens, matrix, precision = as_multiplied(tokens, matrix, precision)
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
    nor a dropped row's gradient carries them. Its sums are `precision.accumulated`'s (XLA's own on a CPU,
    row 5448).
    """
    tokens, kernel, precision = as_multiplied(tokens, kernel, precision)
    # bincount widens under x64; TPU ragged-dot only lowers int32 counts.
    # Keep the wider domain for CPU calls whose row count actually needs it.
    index_dtype = jnp.int32 if tokens.shape[0] <= np.iinfo(np.int32).max else jnp.int64
    if group_sizes.dtype == jnp.int64:
        group_sizes = group_sizes.astype(index_dtype)
    grouped = jnp.arange(tokens.shape[0], dtype=index_dtype)[:, None] < jnp.sum(
        group_sizes, dtype=index_dtype
    )
    product = ragged_product(tokens, kernel, precision=precision)
    out = accumulated(product, jnp.where(grouped, tokens, 0), kernel, group_sizes,
                      preferred_element_type=preferred_element_type)
    return jnp.where(grouped, out, 0)


_GROUPED_ROWS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (1,)), ((), ())), lhs_ragged_dimensions=(0,), rhs_group_dimensions=(0,))
"""`jax.lax.ragged_dot`'s own: each group's rows of `[rows, depth]` times its `[depth, width]` matrix."""


def ragged_product(values: jax.Array, matrices: jax.Array, *,
                   precision: PrecisionLike = None) -> Callable[..., jax.Array]:
    """`jax.lax.ragged_dot_general` bound for `values` times `matrices` (`product(values, matrices,
    group_sizes, preferred_element_type=...)`): its grouped rows, `precision`, and its output placed where an
    Explicit mesh axis places its operands (`ragged_placement`): JAX infers no placement for a ragged dot
    there (row 12401). A partial of the primitive itself, which `precision.accumulated` reads as a ragged
    dot, its group sizes counted, not multiplied (row 12460)."""
    return functools.partial(jax.lax.ragged_dot_general, ragged_dot_dimension_numbers=_GROUPED_ROWS,
                             precision=precision,
                             out_sharding=ragged_placement(((values, 0), (matrices, matrices.ndim - 1))))


def placed_ragged_dot(values: jax.Array, matrices: jax.Array, group_sizes: jax.Array, *,
                      precision: PrecisionLike = None,
                      preferred_element_type: Dtype | None = None) -> jax.Array:
    """`jax.lax.ragged_dot` as written (its one primitive, which a fuser reads as the program's dot), placed
    where an Explicit mesh axis places its operands (`ragged_product`)."""
    return ragged_product(values, matrices, precision=precision)(
        values, matrices, group_sizes, preferred_element_type=preferred_element_type)


def ragged_placement(sources: tuple) -> jax.sharding.NamedSharding | None:
    """Where a ragged product lands where an operand's type places it on a mesh with Explicit axes, split or
    not (`out_sharding`: JAX infers no placement for a ragged dot, "Explicit sharding inference for
    ragged_dot_general is not currently implemented"): each output dimension placed as the operand
    dimension it comes from (`sources`, each `(operand, axis)`, or None for one no operand dimension gives: a
    group axis), a contraction's sum whole, a mesh axis placing two dimensions placing the first alone; whole
    on the context's mesh where no operand's type names one but the context's has an Explicit axis; None
    otherwise."""
    from dew.mesh import mesh_axes
    from dew.nn.sharding import explicit_placement

    shardings = [None if source is None else explicit_placement(jax.typeof(source[0])) for source in sources]
    mesh = next((each.mesh for each in shardings if each is not None), None)
    if mesh is None:
        # Operands no mesh types (a jitted program's constants) under a context mesh with an Explicit axis,
        # which still asks a ragged dot its placement: whole on it.
        context = jax.sharding.get_abstract_mesh()
        if context.empty or not context.explicit_axes:
            return None
        return jax.sharding.NamedSharding(context, jax.sharding.PartitionSpec())
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
    blocks' order (`_block_walk`): a blockwise-scaled FP8 product's sum (DeepSeek-V3's recipe, row 4518),
    the GEMM template's portable branch. The rows past the groups are zero, whatever their scales."""
    blocks = row_scales.shape[0]
    scaled = column_scales.shape[2]
    width = kernel.shape[2]
    if tokens.shape[1] != blocks * SCALE_DEPTH or width % scaled:
        raise ValueError(f"scales {row_scales.shape} and {column_scales.shape} do not cover a "
                         f"{tokens.shape} x {kernel.shape} product in {SCALE_DEPTH}-deep blocks")

    def added(index, partial, owner, total, aligned):
        columns = aligned(jnp.repeat(column_scales[:, index][owner], width // scaled, axis=1))
        # Each scale in the order that keeps the intermediate normal (`twice_scaled`, ledger 2715).
        return total + twice_scaled(partial, row_scales[index][:, None].astype(jnp.float32),
                                    columns.astype(jnp.float32))

    def start(zeros):
        return zeros

    return _block_walk(tokens, kernel, group_sizes, SCALE_DEPTH, start, added, lambda total: total,
                       precision=precision)


_NO_EXPONENT = -(1 << 20)
"""A running sum's exponent before any nonzero partial (`exponent_scaled`): below any a partial reaches."""


def exponent_scaled(lhs_codes: jax.Array, lhs_exponents: jax.Array, kernel_codes: jax.Array,
                    kernel_exponents: jax.Array, group_sizes: jax.Array, *, depth: int,
                    unscaled: jax.Array, kernel_unscaled: jax.Array) -> jax.Array:
    """Each group's rows of `lhs_codes` `[rows, k]` times its matrix of `kernel_codes` `[groups, k, width]`,
    every `depth`-deep block's codes multiplied whole (each product exact; the block's partial their sum,
    rounded as XLA's dot rounds it) and scaled by the power two of its row's and column's exponents summed
    (`lhs_exponents` `[rows, k / depth]`, `kernel_exponents` `[groups, k / depth, width]`), summed in the
    blocks' order (`_block_walk`): a block-scaled product whose values no operand holds apart (MXFP8's,
    `gemm.mxfp8_product`'s portable branch, OCP MX's dot), so a scale of 2^-127 meets one of 2^127 in the
    exponents, not as a subnormal operand a backend flushing subnormals reads as zero (row 13164). Its sum
    holds its reference's error (testing.md's check 1): where a CPU sums wide
    (`precision.WIDE_CPU_ARITHMETIC`, Dew's tests, in no scope stating `FP32_SUMS`) in float64 and rounded
    once, as the decoded values' product sums there; elsewhere fp32's, a compensated pair
    (`precision.compensated_add`) held at an exponent of its own, each partial brought to the larger of its
    own and the sum's exponent by an exact power of two (`precision.power2`) before it is added and the pair
    renormalized to its lead after, raised to its exponent once at the end: no partial or running sum a
    flushable subnormal, partials of 2^-127 summing to a normal 2^-126. A cancellation of more binades than
    the sum's significand holds (a double float's, float64's) loses what fp32's accumulation and the native
    tcgen05 kernel's lose: check 1 holds a product to its reference's error, no exact accumulator's (Main's
    ruling on row 13248). A block either of whose scales is NaN (`unscaled`, `kernel_unscaled`, its
    exponents' shape) makes its sums NaN. The rows past the groups are zero."""
    from ..precision import FP32_SUMS, compensated_add, power2, wide_cpu_arithmetic

    def scaled_of(index, owner, aligned):
        return lhs_exponents[:, index][:, None] + aligned(kernel_exponents[:, index][owner])

    def poisoned_of(index, owner, aligned):
        return unscaled[:, index][:, None] | aligned(kernel_unscaled[:, index][owner])

    def start(zeros):
        # Placed as the sum is (`_block_walk`'s zeros), each part of the carry alike.
        return (zeros, zeros, jnp.full_like(zeros, _NO_EXPONENT, dtype=jnp.int32),
                jnp.zeros_like(zeros, dtype=bool))

    def added(index, partial, owner, carry, aligned):
        high, low, exponent, poisoned = carry
        scale = scaled_of(index, owner, aligned)
        own = jnp.where(partial != 0, scale + jnp.frexp(partial)[1], _NO_EXPONENT)
        top = jnp.maximum(exponent, own)
        high, low = compensated_add((power2(high, exponent - top), power2(low, exponent - top)),
                                    (power2(partial, scale - top), jnp.zeros_like(partial)))
        # The pair at its lead's exponent: none where it cancelled to zero.
        lead = jnp.where(high != 0, jnp.frexp(high)[1], 0)
        exponent = jnp.where(high != 0, top + lead, _NO_EXPONENT)
        poisoned = poisoned | poisoned_of(index, owner, aligned)
        return power2(high, -lead), power2(low, -lead), exponent, poisoned

    def finished(carry):
        high, _, exponent, poisoned = carry
        return jnp.where(poisoned, jnp.float32(jnp.nan), power2(high, exponent))

    def narrow():
        return _block_walk(lhs_codes, kernel_codes, group_sizes, depth, start, added, finished,
                           precision=jax.lax.Precision.DEFAULT)

    if not wide_cpu_arithmetic() or jax.config.jax_default_matmul_precision == FP32_SUMS:
        return narrow()

    def wide():
        with jax.enable_x64(new_val=True):
            def widened(index, partial, owner, carry, aligned):
                total, poisoned = carry
                return (total + power2(partial, scaled_of(index, owner, aligned)),
                        poisoned | poisoned_of(index, owner, aligned))

            def rounded(carry):
                total, poisoned = carry
                return jnp.where(poisoned, jnp.float32(jnp.nan), total.astype(jnp.float32))

            return _block_walk(lhs_codes, kernel_codes, group_sizes, depth,
                               lambda zeros: (zeros, jnp.zeros_like(zeros, dtype=bool)), widened, rounded,
                               precision=jax.lax.Precision.DEFAULT, dtype=jnp.float64)

    # Eagerly, the platform the committed operands lie on, not the default device's (`committed_context`).
    with committed_context((lhs_codes, kernel_codes)):
        return jax.lax.platform_dependent(cpu=wide, default=narrow)


def _block_walk(lhs: jax.Array, kernel: jax.Array, group_sizes: jax.Array, depth: int, start: Callable,
                added: Callable, finished: Callable, *, precision: PrecisionLike,
                dtype=jnp.float32) -> jax.Array:
    """The block-scaled products' one walk (`step_scaled`'s, `exponent_scaled`'s): each `depth`-deep block of
    each group's rows of `lhs` `[rows, k]` times its matrix of `kernel` `[groups, k, width]`, its partial
    in `dtype` added in the blocks' order into a carry (`start` of the zero sum, `added(index, partial,
    owner, carry, aligned)`, `owner` each row's group, `aligned` placing a matrices'-side `[rows, width]`
    operand as the sum) that `finished` makes the sum, placed as the products'
    operands place it (`ragged_placement`); then the rows past the groups zero, selected once after every
    block, so a padded row's infinite scale leaves no NaN (row 13164). No contraction, or no group: zeros."""
    rows, groups, width = lhs.shape[0], kernel.shape[0], kernel.shape[2]
    placed = ragged_placement(((lhs, 0), (kernel, kernel.ndim - 1)))
    zeros = jnp.zeros((rows, width), dtype, out_sharding=placed)

    def aligned(value: jax.Array) -> jax.Array:
        # A `[rows, width]` operand gathered by its rows' groups from the matrices' side, placed as the sum
        # is: its columns' axis may be the rows' too, which their broadcast together could not place.
        return value if placed is None else jax.sharding.reshard(value, placed)
    blocks = lhs.shape[1] // depth
    if blocks == 0 or groups == 0:
        # No contraction, or no group a row could belong to: the empty sum.
        return zeros.astype(jnp.float32)
    owner = jnp.minimum(jnp.repeat(jnp.arange(groups), group_sizes, total_repeat_length=rows), groups - 1)

    def block(index, carry):
        lhs_block = jax.lax.dynamic_slice_in_dim(lhs, index * depth, depth, axis=1)
        kernel_block = jax.lax.dynamic_slice_in_dim(kernel, index * depth, depth, axis=1)
        partial = xla_ragged_dot(lhs_block.astype(dtype) if dtype == jnp.float64 else lhs_block,
                                 kernel_block.astype(dtype) if dtype == jnp.float64 else kernel_block,
                                 group_sizes, precision=precision, preferred_element_type=dtype)
        return added(index, partial, owner, carry, aligned)

    total = finished(jax.lax.fori_loop(0, blocks, block, start(zeros)))
    return jnp.where((jnp.arange(rows) < jnp.sum(group_sizes))[:, None], total, jnp.zeros((), total.dtype))


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
    # Both live, the same barrier is the rule's dual (`with_tangents_formed`): its transpose returns `x`'s
    # cotangent only with the kernel's gradient formed.
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
