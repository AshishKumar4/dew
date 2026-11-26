"""A product's declaration (design §3), the same for every product of every template: what its operands
multiply in, whether each depth step's partial is promoted into the fp32 accumulator apart, and why the
kernels do not compute it where they do not. A template declares a product once, from the precision its
caller asked for, and keeps the declaration through every mainloop and its VJP's products."""

import dataclasses
import functools
import itertools
from collections.abc import Callable, Iterator, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype, PrecisionLike

from dew.nn import programs
from dew.nn.kernels import generation
from dew.nn.precision import (
    accumulated,
    asks_default_precision,
    effective_precision,
    exact_products,
    float64_sum,
    precision_names,
    products_dtype,
)


def _floating_bits(dtype) -> int | None:
    """`dtype`'s bits where it is a floating dtype (`jnp.finfo`'s), None where it is not."""
    dtype = jnp.dtype(dtype)
    return jnp.finfo(dtype).bits if jnp.issubdtype(dtype, jnp.floating) else None


def half(dtype) -> bool:
    """Whether `dtype` is a 16-bit float (bfloat16, float16), whose products the kernels multiply exactly
    into fp32."""
    return _floating_bits(dtype) == 16


def eight(dtype) -> bool:
    """Whether `dtype` is an 8-bit float, an FP8 operand (testing.md, FP8 products): each code exact in
    bfloat16, so its products exact as a 16-bit operand's, its running sum promoted into fp32 every depth
    step (`Product.promoted`), as Hopper's FP8 tensor cores sum within a step in a narrower accumulator.
    Which formats a family's products read is its own fact (`mosaic.Issuer.fp8_formats`)."""
    return _floating_bits(dtype) == 8


_MODES = {"16bit": "flops", "tf32": "tf32_flops", "8bit": "fp8_flops", "fp32": "operations"}
"""The `templates.cost.Work` field a product's mode's FLOPs are told in (`generation.Architecture`'s modes;
IEEE fp32 is the CUDA cores')."""

TPU_PASSES = {"DEFAULT": 1, "HIGH": 3, "HIGHEST": 6}
"""The bfloat16 passes a TPU's matrix units multiply fp32 operands in, by precision (`jax.lax.Precision`'s own
statement: "On TPU: performs float32 computations in bfloat16", "in 3 bfloat16 passes", "in 6 bfloat16")."""

TPU_ALGORITHMS = {jax.lax.DotAlgorithmPreset.BF16_BF16_F32: "DEFAULT",
                  jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3: "HIGH",
                  jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6: "HIGHEST"}
"""The dot algorithms XLA:TPU compiles, each as the precision it lowers to (`TPU_PASSES`'): over fp32
operands on a v5e, the bfloat16 preset as the default's one pass, X3 as `operand_precision={high,high}` and
X6 as `{highest,highest}`. It refuses every other preset, F16's, BF16_BF16_BF16, X9, TF32's, F32_F32_F32 and
F64's ("UNIMPLEMENTED: Dot algorithm ... is not supported"; measurement M-JS-160, row 10845), and no other
algorithm is shown to compile there."""

_UNRUN = "unpriced_operations"
"""The `Work` field an algorithm its target does not run is told in (`TPU_ALGORITHMS`): work at no rate, so
no form is cheaper for taking it (`dew.algebra.cost.cheaper`), its price never a TPU's zero seconds."""


def product_mode(precision: jax.lax.PrecisionLike, lhs_dtype, rhs_dtype,
                 target: generation.GpuTarget) -> tuple[str, int]:
    """The `Work` field a product of operands of those dtypes at `precision` on `target` is told in
    (`_MODES`), and the passes it multiplies in: where `precision` names a dot algorithm, on a TPU its
    bfloat16 passes where XLA:TPU compiles it, else none it runs (`TPU_ALGORITHMS`, `_UNRUN`), elsewhere
    that algorithm's as JAX lowers it (`programs.dot_arithmetic`: TF32's, a 16-bit or an 8-bit precision's
    whatever the operands' storage, each over its primitive operations, an X3's three; any other on the CUDA
    cores); else the
    operands': 16-bit operands' 16-bit mode, FP8 operands' 8-bit mode (`eight`), fp32 operands' on a TPU its
    matrix units' bfloat16 passes (`TPU_PASSES`, a pair of precisions at its higher's), elsewhere TF32 at the
    default precision (the product declaration's, `declared`) and IEEE fp32 above it; any other operand's on
    the CUDA cores. The one rule an XLA product's work is priced by: a term's (`dew.algebra.cost.work`,
    through `emission.operation_work`) and a chunk form's (`dew.algebra.cost.product_work`); a TPU kernel
    body's product is its counter's own (`emission._Counting`), whose passes Mosaic TPU's lowering decides."""
    arithmetic = programs.dot_arithmetic(precision, lhs_dtype, rhs_dtype)
    if arithmetic is not None and generation.tpu(target):
        named = programs.canonical_precision(precision)
        taken = TPU_ALGORITHMS.get(named) if isinstance(named, jax.lax.DotAlgorithmPreset) else None
        return (_MODES["16bit"], TPU_PASSES[taken]) if taken is not None else (_UNRUN, 1)
    if arithmetic is not None:
        bits = set(arithmetic.operand_bits)
        mode = "tf32" if arithmetic.tf32 else "16bit" if bits == {16} else "8bit" if bits == {8} else "fp32"
        return _MODES[mode], arithmetic.operations
    dtypes = {jnp.dtype(lhs_dtype), jnp.dtype(rhs_dtype)}
    if all(half(dtype) for dtype in dtypes):
        return _MODES["16bit"], 1
    if all(eight(dtype) for dtype in dtypes):
        return _MODES["8bit"], 1
    if dtypes == {jnp.dtype(jnp.float32)} and generation.tpu(target):
        named = precision_names(effective_precision(precision)) or {"DEFAULT"}
        return _MODES["16bit"], max(TPU_PASSES[name] for name in named)
    if dtypes == {jnp.dtype(jnp.float32)} and asks_default_precision(precision, configured=True):
        return _MODES["tf32"], 1
    return _MODES["fp32"], 1


@dataclasses.dataclass(frozen=True)
class Product:
    """One declared product. `compute` is the operands' dtype at the multiply: a 16-bit one's products are
    exact, fp32's are TF32 on the tensor cores. `promoted` adds each depth step's partial into the fp32
    accumulator apart, the IEEE sums a precision above the default asks of an fp32 output, since the tensor
    cores truncate each add within a step; a 16-bit output's one rounding hides the difference.
    `refusal` says why the kernels do not compute the product, None where they do. An FP8 product
    (`eight`) is always promoted, each 128-deep step's partial added into the fp32 accumulator apart: the
    FP8 contract's budget is met so (testing.md, FP8 products)."""
    compute: jnp.dtype
    promoted: bool
    refusal: str | None


class BlockFormat(NamedTuple):
    """A block-scaled product's operand format (row 4519): its scales' dtype and the depths one scale
    covers, which its operands' quantization, their values and the native products' scale streams all read
    (`formats.MxFp8`, `formats.NvFp4Block`, `gemm.mosaic_gpu.gmm`'s `mx_scales`)."""

    scale: np.dtype
    depth: int


MXFP8 = BlockFormat(np.dtype(jnp.float8_e8m0fnu), 32)
"""OCP MX's FP8: an E8M0 scale every 32 depths (tcgen05's block-scaled MMA's `scale_vec::1X`)."""

NVFP4 = BlockFormat(np.dtype(jnp.float8_e4m3fn), 16)
"""NVFP4: an E4M3 scale every 16 depths (tcgen05's `scale_vec::4X`, `kind::mxf4nvf4`)."""


def block_format(dtype) -> BlockFormat:
    """The block format of a block-scaled product of `dtype` operands: NVFP4's for 4-bit floats (E2M1),
    MXFP8's for 8-bit ones."""
    return NVFP4 if jnp.finfo(dtype).bits == 4 else MXFP8


SCALE_DEPTH = 128
"""The depth one block scale of a blockwise-scaled FP8 product covers: DeepSeek-V3's 1 x 128 activation and
128 x 128 weight blocks (row 4518), each block's fp32 partial times its scales before its add. The template's
step-scaled launches step it whole (`gemm.mosaic_gpu.gmm`'s and `tgmm`'s `step_scales`), and the portable
product sums it a block at a time (`xla_products.step_scaled`)."""

SUMMED_DEPTH = 128
"""The depth an exactly summed partial covers on a family whose accumulator carries across steps
(`mosaic.Issuer.summed_window`): DeepSeek-V3's promotion interval for FP8 sums, 128."""


def declared(compute: Dtype, precision: PrecisionLike, *, outputs: tuple = (),
             operands: tuple = ()) -> Product:
    """The product a caller asks for with `precision` (explicitly or through
    `jax_default_matmul_precision`), multiplying in `compute` into `outputs`' dtypes from operands (or
    masters) of `operands`' dtypes. The kernels sum in fp32 and offset groups in int32, so x64 is
    refused, and an operand fp32 does not hold (a complex master), whose gradient they cannot write;
    their fp32 products are TF32, which only the default precision asks for. A declared dot algorithm
    whose products are not `compute`'s whole ones (TF32's, bf16 parts of an fp32 product, F64's:
    `precision.exact_products`) is refused too, forward and gradient alike: the kernels multiply
    `compute` whole and sum in fp32, so a portable route computes it."""
    compute = jnp.dtype(compute)
    above = not asks_default_precision(precision, configured=True)
    return Product(compute, eight(compute) or promoted(above=above, outputs=outputs),
                   _refusal(compute, operands, above, precision))


def promoted(*, above: bool, outputs: tuple) -> bool:
    """Whether a product promotes each depth step's partial apart (`exact_sums`): where an output is fp32,
    whose sums a 16-bit output's one rounding hides, and the product is asked above the default precision
    (`above`). At the default precision a 16-bit product's fp32 sums are the tensor cores' own, as cuBLAS's
    are: a dense kernel gradient's tensor-core sums were 0.98-1.00 of XLA's (cuBLAS's) RMS error from
    float64 at 8192 tokens and at depths of 128 to 2048 on an A100 and an H100 (M-EM-3), where its exact
    sums, 40 to 60 times closer to float64 than the reference, cost an H100 1.10-2.06x their time (W124) and
    an A100 the 128 x 128 tile; row 4331's premise (M-NS-1: 1.661x XLA's error at depth 128, 3.88x at 512)
    did not reproduce. An underfilled wave's split of the contraction (`gemm.pricing.split_count`)
    shortens each sum as cuBLAS's split-K does. An FP8 product promotes apart whatever its outputs
    (`declared`)."""
    return above and jnp.dtype(jnp.float32) in {jnp.dtype(dtype) for dtype in outputs}


def _refusal(compute: jnp.dtype, operands: tuple, above: bool, precision: PrecisionLike = None
             ) -> str | None:
    if jax.config.jax_enable_x64:
        return "x64 widens the kernels' int32 group offsets"
    if not eight(compute) and not exact_products(precision, compute):
        return ("a declared dot algorithm narrows or widens the products, which the kernels multiply whole "
                "with fp32 sums")
    # An operand fp32 holds is no complex one and none wider (8-bit floats promote implicitly to nothing).
    if any(jnp.issubdtype(dtype, jnp.complexfloating) or jnp.dtype(dtype).itemsize > 4
           for dtype in (compute, *operands)):
        return "an operand fp32 does not hold, and the kernels sum gradients in fp32"
    if half(compute) or eight(compute):
        return None
    if compute != jnp.dtype(jnp.float32):
        return f"the kernels do not multiply in {compute.name}"
    return "fp32 products above the default precision, and the kernels multiply in TF32" if above else None


def interpreted(product: Product, lhs: jax.Array, rhs: jax.Array, dimension_numbers) -> jax.Array:
    """The declared `product` of `lhs` and `rhs` as an interpreted or portable mainloop issues it off a GPU,
    an fp32 result. A promoted fp32 product (one asked above the default precision) accumulates in float64
    and rounds once to fp32: each fp32 product is exact in float64 and the sum rounds in float64, so the
    result is within half an fp32 ulp of the float64 sum, which is within gamma_n (float64's) times the
    sum of its terms' magnitudes of the exact one (adding 2^60, 1 and -2^60 in that order loses the 1).
    That bound holds at every tiling: XLA's CPU dot picks its fp32 accumulation by shape and instruction
    set (an fp32 [128, 128] by [128, N] product is 2.32e-6 from float64 at N a multiple of 128, 1.25e-6
    at N = 300), which a tile's shape would otherwise choose. float64 runs on this path alone, under
    `jax.enable_x64` for its own ops; the kernels' refusal of x64 (`declared`) is their int32 offsets'.
    An FP8 product is its codes' in bfloat16, each exact there, accumulated in fp32: the FP8 contract's
    reference arm, the dequantized product with fp32 sums (testing.md, FP8 products). Any other product is
    `jax.lax.dot_general`'s, accumulated in fp32: a 16-bit one's products are exact, and a float32 one's
    summed as every float32 contraction is (`precision.accumulated`: in float64 under Dew's wide test
    arithmetic, else as XLA sums it), so a mainloop's sums and the fp32 references they
    are held to sum alike."""
    if eight(product.compute):
        return jax.lax.dot_general(lhs.astype(jnp.bfloat16), rhs.astype(jnp.bfloat16), dimension_numbers,
                                   preferred_element_type=jnp.float32)
    contraction = functools.partial(jax.lax.dot_general, dimension_numbers=dimension_numbers)
    if product.compute == jnp.dtype(jnp.float32) and product.promoted:
        # In float64 at HIGHEST, whatever fp32 algorithm the caller bound (`precision.float64_sum`): under
        # check 1's `FP32_SUMS` a float64 dot stating none narrowed its operands back to fp32.
        return float64_sum(contraction, lhs, rhs, dtype=jnp.float32)
    if products_dtype(lhs, rhs) == jnp.dtype(jnp.float32):
        return accumulated(contraction, lhs, rhs, preferred_element_type=jnp.float32)
    return jax.lax.dot_general(lhs, rhs, dimension_numbers, preferred_element_type=jnp.float32)


BF16_LARGEST = float(jnp.finfo(jnp.bfloat16).max)
"""bf16's largest finite value, `(2 - 2^-7) 2^127`, an fp32 value."""

def parts(dtype) -> int:
    """The bf16 parts that hold `dtype`'s whole significand (`formed`), 8 bits each: one of bf16's, two of
    fp16's 11, three of fp32's 24 (fp16's and fp32's exponents within bf16's)."""
    return -(-(jnp.finfo(dtype).nmant + 1) // (jnp.finfo(jnp.bfloat16).nmant + 1))


_BF16_PARTS: dict[tuple[str, bool], int] = {}


def bf16_part_count(dtype=jnp.float32, *, accurate: bool = True) -> int:
    """bf16 parts for `dtype` operands via `dew.algebra.error.product_precision` (rows 5473, 11943).

    Non-fp32 operands return `parts(dtype)`. For fp32, resolves the cheapest bf16-split dot algorithm on
    the active target (`stated=programs.dot_algorithms(operand)` so GPU and TPU targets both resolve
    cleanly) and returns `DotArithmetic.parts` (3 when `accurate=True`, 1 when `accurate=False`)."""
    operand = jnp.dtype(dtype)
    if operand != jnp.float32:
        return parts(operand)
    from dew.algebra.error import product_precision

    target = generation.gpu_target()
    key = (target.generation or generation.placed_platform(), accurate)
    if key not in _BF16_PARTS:
        chosen = product_precision(
            64, 64, 64, operand,
            stated=programs.dot_algorithms(operand),
            admits=lambda p: (a := programs.dot_arithmetic(p, operand, operand)) is not None
            and a.operand_bits == (16, 16) and not a.tf32 and a.accumulation_bits == 32
            and not a.imprecise and a.operations >= (6 if accurate else 1))
        arith = programs.dot_arithmetic(chosen, operand, operand)
        assert arith is not None, chosen
        _BF16_PARTS[key] = arith.parts
    return _BF16_PARTS[key]


def bf16_parts(x: jax.Array, count: int, *, saturating: int,
               narrowed: Callable[[jax.Array], jax.Array] | None = None) -> Iterator[jax.Array]:
    """fp32 `x` as `count` bf16 parts, largest first, each the last's remainder rounded (exact) and formed
    only as it is taken, so a mainloop retiring each part's product before the next holds one part's
    registers: three sum to `x` exactly, two to 16 of its significand's bits. The first `saturating`
    saturate at bf16's largest (an fp32 value past it rounds to infinity in bf16, its remainder then NaN),
    the next parts taking the difference; on a finite `x` the later ones never reach it, so only a
    nonfinite one tells them apart. A saturating part is clipped then rounded, or `narrowed` where given:
    the same bits in one step (a kernel's one instruction a pair, `mosaic.Family.formed`)."""
    rest = x
    for index in range(count):
        if index >= saturating:
            part = rest.astype(jnp.bfloat16)
        elif narrowed is None:
            part = jnp.clip(rest, -BF16_LARGEST, BF16_LARGEST).astype(jnp.bfloat16)
        else:
            part = narrowed(rest)
        yield part
        if index + 1 < count:
            rest = rest - part.astype(jnp.float32)


def formed(x: jax.Array, count: int,
           narrowed: Callable[[jax.Array], jax.Array] | None = None) -> tuple[jax.Array, ...]:
    """`bf16_parts` of `x`, every part saturating (through `narrowed` where given), all formed at once."""
    return tuple(bf16_parts(x, count, saturating=count, narrowed=narrowed))


def whole(parts: Sequence[jax.Array]) -> jax.Array:
    """The value `parts` (a value's parts, largest first, `formed`'s) sum to, fp32, summed from the smallest
    up: each part the last's remainder, the sum is exact. Summed largest first, a finite value past bf16's
    largest by `2^120 - 2^111` or more took its leading two parts past fp32's, to infinity, before the third's
    negative remainder brought it back (rows 6842, 6866)."""
    found = parts[-1].astype(jnp.float32)
    for part in reversed(parts[:-1]):
        found = part.astype(jnp.float32) + found
    return found


def stacked(x: jax.Array, count: int) -> jax.Array:
    """fp32 `x` `[rows, ...]` as its `count` bf16 parts (`formed`) stacked along its first axis `[count *
    rows, ...]`, the largest first: an operand as a launch reads its parts."""
    return jnp.concatenate(formed(x.astype(jnp.float32), count), 0)


def joined(x: jax.Array, count: int) -> jax.Array:
    """`stacked`'s inverse: the value `[rows, ...]` its `count` stacked parts sum to (`whole`)."""
    parts = x.astype(jnp.float32).reshape(count, x.shape[0] // count, *x.shape[1:])
    return whole(list(parts))


def terms(left: int, right: int) -> list[tuple[int, int]]:
    """The pairs `(i, j)` of two operands' parts (`formed`, `left` and `right` of them) a product of them
    issues, smallest first: those whose orders sum under the larger count, each part's bits beside the
    other's leading ones (three parts each, 24 bits of each operand's significand)."""
    count = max(left, right)
    return sorted(((i, j) for i, j in itertools.product(range(left), range(right)) if i + j < count),
                  key=lambda term: -sum(term))


class Transposed:
    """A register operand a product takes transposed, `value^T`: a mainloop stages it and reads the tile
    transposed, where a transpose of registers would move every element between threads."""

    def __init__(self, value: jax.Array):
        self.value = value

    @property
    def shape(self) -> tuple:
        return (*self.value.shape[:-2], self.value.shape[-1], self.value.shape[-2])


class Exact:
    """A register operand exact in a 16-bit dtype (a mask of zeros and ones): a mainloop stages it as one
    part, where an fp32 operand takes three."""

    def __init__(self, value: jax.Array):
        self.value = value

    @property
    def shape(self) -> tuple:
        return self.value.shape


@jax.tree_util.register_pytree_node_class
class Scaled:
    """A register operand stored scaled, its product taken back by `2**exponent` (traced: an fp16 part held
    at its tile's, or each row's, largest magnitude's scale; or a static number: a scaled walk's small
    weights, `attention_algebra.small_parts`): a mainloop issues its product into an accumulator of its own,
    which no unscaled product shares, and restores it. A pytree of both, so a branch's parts leave it
    (`lax.cond`)."""

    def __init__(self, value: jax.Array, exponent: jax.Array | float):
        self.value, self.exponent = value, exponent

    def tree_flatten(self):
        return (self.value, self.exponent), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)

    @property
    def shape(self) -> tuple:
        return self.value.shape


@dataclasses.dataclass(frozen=True, eq=False)
class Support:
    """Where a product's terms may be nonzero, as its caller states them statically: `left[r, k]` where its
    left operand's entries may be, `right[k, c]` its right's, `kept[r, c]` the outputs it reads (numpy bool,
    each its operand's shape). Its other terms are zero, so a product issued of its support alone
    (`mosaic.Issuer.products`' `support`) skips each instruction none of whose terms it holds: its kept
    outputs the whole product's bit for bit where every operand an instruction reads is finite as it reads
    it (staged and converted: a skipped zero times an infinity or a NaN was a NaN in the whole, and an
    earlier product's overflow is such an operand), up to the sign of a zero (an accumulator at -0 a skipped
    +0 left -0), its other outputs anything."""

    left: np.ndarray
    right: np.ndarray
    kept: np.ndarray

    def divides(self, rows: int, columns: int, depth: int) -> bool:
        """Whether tiles of `rows` by `columns` of its outputs and `depth` of its terms tile its product."""
        (m, k), n = self.left.shape, self.right.shape[1]
        return not (m % rows or n % columns or k % depth)

    def tiles(self, rows: int, columns: int, depth: int) -> np.ndarray:
        """`[M / rows, N / columns, K / depth]` bool: whether the instruction a product issues on a tile of
        `rows` by `columns` of its outputs and `depth` of its terms holds one of its support's; refused where
        those tiles do not tile it (`divides`)."""
        if not self.divides(rows, columns, depth):
            raise ValueError(f"tiles of {rows} x {columns} by {depth} do not tile a {self.left.shape[0]} x "
                             f"{self.right.shape[1]} product of depth {self.left.shape[1]}")
        terms = self.left[:, None, :] & self.right.T[None, :, :] & self.kept[:, :, None]
        m, n, k = terms.shape
        return terms.reshape(m // rows, rows, n // columns, columns, k // depth, depth).any(axis=(1, 3, 5))


class Blocks:
    """`count` products named `name` to sum, `left(j) @ right(j)` a block `j` (tiles in memory): one a
    block, which a mainloop may issue as a loop over `j`, its operands' fragments loaded a block at a time
    (`operands.Issuer.summed`), where unrolled ptxas held every block's at once."""

    def __init__(self, name: str, left: Callable, right: Callable, count: int):
        self.name, self.left, self.right, self.count = name, left, right, count
