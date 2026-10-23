"""A product's declaration (design §3), the same for every product of every template: what its operands
multiply in, whether each depth step's partial is promoted into the fp32 accumulator apart, and why the
kernels do not compute it where they do not. A template declares a product once, from the precision its
caller asked for, and keeps the declaration through every mainloop and its VJP's products."""

import dataclasses

import jax
import jax.numpy as jnp
from flax.typing import Dtype, PrecisionLike

from dew.nn.precision import asks_default_precision

HALF = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
"""The 16-bit dtypes the kernels multiply exactly."""


@dataclasses.dataclass(frozen=True)
class Product:
    """One declared product. `compute` is the operands' dtype at the multiply: a 16-bit one's products are
    exact, fp32's are TF32 on the tensor cores. `promoted` adds each depth step's partial into the fp32
    accumulator apart, the IEEE sums a precision above the default asks of an fp32 output, since the tensor
    cores truncate each add within a step; a 16-bit output's one rounding hides the difference.
    `refusal` says why the kernels do not compute the product, None where they do."""
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
    return Product(compute, promoted(above=above, outputs=outputs), _refusal(compute, operands, above))


def promoted(*, above: bool, outputs: tuple) -> bool:
    """Whether a product asked for above the default precision (`above`) promotes each depth step's
    partial apart: where an output is fp32, whose IEEE sums a 16-bit output's one rounding hides."""
    return above and jnp.dtype(jnp.float32) in {jnp.dtype(dtype) for dtype in outputs}


def _refusal(compute: jnp.dtype, operands: tuple, above: bool) -> str | None:
    if jax.config.jax_enable_x64:
        return "x64 widens the kernels' int32 group offsets"
    if jnp.result_type(compute, *operands, jnp.float32) != jnp.dtype(jnp.float32):
        return "an operand fp32 does not hold, and the kernels sum gradients in fp32"
    if compute in HALF:
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
    Any other product is `jax.lax.dot_general`'s, accumulated in fp32: a 16-bit one's products are
    exact."""
    if product.compute == jnp.dtype(jnp.float32) and product.promoted:
        with jax.enable_x64(new_val=True):
            wide = jax.lax.dot_general(lhs.astype(jnp.float64), rhs.astype(jnp.float64), dimension_numbers,
                                       preferred_element_type=jnp.float64)
            return wide.astype(jnp.float32)
    return jax.lax.dot_general(lhs, rhs, dimension_numbers, preferred_element_type=jnp.float32)
