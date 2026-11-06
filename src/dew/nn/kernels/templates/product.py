"""A product's declaration (design §3), the same for every product of every template: what its operands
multiply in, whether each depth step's partial is promoted into the fp32 accumulator apart, and why the
kernels do not compute it where they do not. A template declares a product once, from the precision its
caller asked for, and keeps the declaration through every mainloop and its VJP's products."""

import dataclasses
import itertools
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
from flax.typing import Dtype, PrecisionLike

from dew.nn.precision import asks_default_precision

HALF = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
"""The 16-bit dtypes the kernels multiply exactly."""

EIGHT = (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2))
"""The FP8 dtypes the kernels multiply (testing.md, FP8 products): each code exact in bfloat16, so its
products exact as a 16-bit operand's, its running sum promoted into fp32 every depth step
(`Product.promoted`), as Hopper's FP8 tensor cores sum within a step in a narrower accumulator."""


@dataclasses.dataclass(frozen=True)
class Product:
    """One declared product. `compute` is the operands' dtype at the multiply: a 16-bit one's products are
    exact, fp32's are TF32 on the tensor cores. `promoted` adds each depth step's partial into the fp32
    accumulator apart, the IEEE sums a precision above the default asks of an fp32 output, since the tensor
    cores truncate each add within a step; a 16-bit output's one rounding hides the difference.
    `refusal` says why the kernels do not compute the product, None where they do. An FP8 product
    (`EIGHT`) is always promoted, each 128-deep step's partial added into the fp32 accumulator apart: the
    FP8 contract's budget is met so (testing.md, FP8 products)."""
    compute: jnp.dtype
    promoted: bool
    refusal: str | None


def declared(compute: Dtype, precision: PrecisionLike, *, outputs: tuple = (),
             operands: tuple = ()) -> Product:
    """The product a caller asks for with `precision` (explicitly or through
    `jax_default_matmul_precision`), multiplying in `compute` into `outputs`' dtypes from operands (or
    masters) of `operands`' dtypes. The kernels sum in fp32 and offset groups in int32, so x64 is
    refused, and an operand fp32 does not hold (a complex master), whose gradient they cannot write;
    their fp32 products are TF32, which only the default precision asks for."""
    compute = jnp.dtype(compute)
    above = not asks_default_precision(precision, configured=True)
    return Product(compute, compute in EIGHT or promoted(above=above, outputs=outputs, compute=compute),
                   _refusal(compute, operands, above))


def promoted(*, above: bool, outputs: tuple, compute: Dtype | None = None) -> bool:
    """Whether a product promotes each depth step's partial apart (`exact_sums`): where an output is fp32,
    whose sums a 16-bit output's one rounding hides, and the product is asked above the default precision
    (`above`) or multiplies 16-bit operands (`compute`). A tensor core adds each instruction's products
    into one fp32 accumulator in order, and its error grows with the depth: every Dew launch's fp32 sum of
    bf16 products, Triton's and the template's at four tiles alike, was 1.661x XLA's RMS error from
    float64 at a depth of 128 and 3.88x at 512 on an A100 (M-NS-1, NS4331-VALUE4-a100; row 4331)."""
    if jnp.dtype(jnp.float32) not in {jnp.dtype(dtype) for dtype in outputs}:
        return False
    return above or (compute is not None and jnp.dtype(compute) in HALF)


def _refusal(compute: jnp.dtype, operands: tuple, above: bool) -> str | None:
    if jax.config.jax_enable_x64:
        return "x64 widens the kernels' int32 group offsets"
    # An operand fp32 holds is no complex one and none wider (8-bit floats promote implicitly to nothing).
    if any(jnp.issubdtype(dtype, jnp.complexfloating) or jnp.dtype(dtype).itemsize > 4
           for dtype in (compute, *operands)):
        return "an operand fp32 does not hold, and the kernels sum gradients in fp32"
    if compute in HALF or compute in EIGHT:
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
    `jax.lax.dot_general`'s, accumulated in fp32: a 16-bit one's products are exact."""
    if product.compute in EIGHT:
        return jax.lax.dot_general(lhs.astype(jnp.bfloat16), rhs.astype(jnp.bfloat16), dimension_numbers,
                                   preferred_element_type=jnp.float32)
    if product.compute == jnp.dtype(jnp.float32) and product.promoted:
        with jax.enable_x64(new_val=True):
            wide = jax.lax.dot_general(lhs.astype(jnp.float64), rhs.astype(jnp.float64), dimension_numbers,
                                       preferred_element_type=jnp.float64)
            return wide.astype(jnp.float32)
    return jax.lax.dot_general(lhs, rhs, dimension_numbers, preferred_element_type=jnp.float32)


BF16_LARGEST = float(jnp.finfo(jnp.bfloat16).max)
"""bf16's largest finite value, `(2 - 2^-7) 2^127`, an fp32 value."""


def bf16_parts(x: jax.Array, count: int, *, saturating: int) -> Iterator[jax.Array]:
    """fp32 `x` as `count` bf16 parts, largest first, each the last's remainder rounded (exact) and formed
    only as it is taken, so a mainloop retiring each part's product before the next holds one part's
    registers: three sum to `x` exactly, two to 16 of its significand's bits. The first `saturating`
    saturate at bf16's largest (an fp32 value past it rounds to infinity in bf16, its remainder then NaN),
    the next parts taking the difference; on a finite `x` the later ones never reach it, so only a
    nonfinite one tells them apart."""
    rest = x
    for index in range(count):
        held = jnp.clip(rest, -BF16_LARGEST, BF16_LARGEST) if index < saturating else rest
        part = held.astype(jnp.bfloat16)
        yield part
        if index + 1 < count:
            rest = rest - part.astype(jnp.float32)


def formed(x: jax.Array, count: int) -> tuple[jax.Array, ...]:
    """`bf16_parts` of `x`, every part saturating, all formed at once."""
    return tuple(bf16_parts(x, count, saturating=count))


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
    at its tile's, or each row's, largest magnitude's scale): a mainloop issues its product into an
    accumulator of its own, which no unscaled product shares, and restores it. A pytree of both, so a
    branch's parts leave it (`lax.cond`)."""

    def __init__(self, value: jax.Array, exponent: jax.Array):
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


class Blocks:
    """`count` products named `name` to sum, `left(j) @ right(j)` a block `j` (tiles in memory): one a
    block, which a mainloop may issue as a loop over `j`, its operands' fragments loaded a block at a time
    (`operands.Issuer.summed`), where unrolled ptxas held every block's at once."""

    def __init__(self, name: str, left: Callable, right: Callable, count: int):
        self.name, self.left, self.right, self.count = name, left, right, count
