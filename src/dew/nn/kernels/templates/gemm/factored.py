"""A packed matrix's product times its scales, `run(rows) * factor` (`factored`), its transpose range-safe
and exact: the packed products' (`gemm.nvfp4_product`, `ternary.ternary_weight_product`), ledger rows 1464,
12392 and 12403."""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero

from dew.nn import float_bits, precision
from dew.nn.derivatives import KernelGlue, contracted, declared_tangent
from dew.nn.kernels import generation
from dew.nn.sharding import explicit_spec, varying_like


def factored(run: Callable, operands: tuple, factor: jax.Array, rows: jax.Array, *, wide: Callable,
             wide_operands: tuple, prepared: Callable | None = None,
             gather: jax.Array | None = None) -> jax.Array:
    """`run(rows as bf16, *operands) * factor`, linear in `rows`: `run` a product of bf16 rows (those
    `gather` reads where given, `gemm`'s `gather`, among `operands`), and `factor` a packed matrix's fp32
    scales broadcast against its output (`formats.scale_factor`), one a row's or one for all (a column's,
    which the transpose sums over, is the plain product's). Its transpose is the plain one, the cotangent
    times its factor through `run`'s pullback, wherever that is finite; where it is not (a product past
    fp32's largest), the exact one (`_Transpose.moved`): each product `cotangent * factor` and each
    coefficient of the decoded matrix (`prepared(*wide_operands)`, its depth its axis -2) cut into integer
    slices on grids of their own, each pair of slices multiplied by a bf16 GEMM whose fp32 sums are exact
    integers (`wide`, the transpose over the rows `run` reads, a route each where gathered), the sums kept
    in int32 limbs, added across routes into their tokens and across the devices that sum them, then
    rounded once (`_rounded`). A product of a cotangent, factor or coefficient not finite is IEEE's: no
    product overflows or flushes, so a scale of 2**126 under cotangents that cancel pulls back their
    remainder, a subnormal one among them. A checked `shard_map`'s devices sum a cotangent only they vary
    over before its pullback, and their partials after."""
    if jnp.ndim(factor) and jnp.shape(factor)[-1] != 1:
        return scaled_exactly(run(rows.astype(jnp.bfloat16), *operands), factor)
    transpose = _Transpose(run, rows.shape, rows.dtype, frozenset(jax.typeof(rows).mat.varying),
                           explicit_spec(rows), wide, prepared or (lambda *each: each))

    def apply(residuals, moving):
        scale, ran, _ = residuals
        return scaled_exactly(run(moving.astype(jnp.bfloat16), *ran), scale)

    def pullback(residuals, cotangent):
        scale, ran, routed = residuals
        return transpose.declared(cotangent.astype(jnp.float32), scale.astype(jnp.float32), ran, routed
                                  ).astype(rows.dtype)

    return generation.linear_map(apply, pullback, (factor, operands, (gather, wide_operands)), rows)


FLUSHING = ('cpu', 'tpu')
"""The platforms whose float32 arithmetic flushes subnormal operands and results to zero (XLA's CPU, row 4023,
and TPU), where a packed product's scales are applied in software (`scaled_exactly`, `raised_factor`); every
other keeps them, its own multiply IEEE's."""

FACTOR_RAISE = 62
"""The power of two `raised_factor` raises a subnormal factor by: the least (`2**-149`) into the normals
(`2**-87`), the largest still below `2**-64`, so no product of it overflows that the plain one would not."""


def flushes() -> bool:
    """Whether the platform the context places work on flushes float32 subnormals (`FLUSHING`)."""
    return generation.placed_platform() in FLUSHING


def raised_factor(factor: jax.Array) -> tuple[jax.Array, jax.Array]:
    """An fp32 `factor` as `(raised, k)`, `factor = raised * 2**-k`, exactly: a subnormal one raised by
    `2**FACTOR_RAISE` from its bits (`float_bits.raised_exactly`) where the platform flushes it (`flushes`),
    every other as it is (`k` 0), so the int8-row epilogue's products read it whole and lower by `k` with its
    rows' shift (`ternary.ternary_weight_product`, `lowered_product`)."""
    factor = jnp.asarray(factor, jnp.float32)
    if not flushes():
        return factor, jnp.zeros(factor.shape, jnp.int32)
    binade = float_bits.binade_of(factor)
    k = jnp.where((binade < jnp.finfo(jnp.float32).minexp) & (binade > -1024), jnp.int32(FACTOR_RAISE),
                  jnp.int32(0))
    return float_bits.raised_exactly(factor, k), k


def _held_apart(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """An fp32 `x` as its signed significand in `[1, 2)` (`_significand`, read from its bits) and its binary
    exponent; a zero, an infinity or a NaN as it is, exponent 0."""
    _, biased, fraction, _, exponent = float_bits.float_bits(x)
    finite = (biased != 0xFF) & ((biased != 0) | (fraction != 0))
    return jnp.where(finite, _significand(x), x), jnp.where(finite, exponent, 0).astype(jnp.int32)


def lowered_product(left: jax.Array, right: jax.Array, k: jax.Array | int) -> jax.Array:
    """`left * right * 2**-k` (fp32 operands, `k` int32 at least 0, broadcast) rounded once to nearest even in
    fp32, however far below its least normal, whatever the platform flushes: each operand's significand in
    `[1, 2)` read from its bits (`_held_apart`, a subnormal's normalized), their product formed exactly
    into a rounded product and its residual (`precision.two_product`, every part of it a normal float32),
    its exponents carried apart and the product rounded once at its own place, the residual deciding a tie
    (`float_bits.rounded_binary`'s `tail`); a zero, an infinity or a NaN multiplied as IEEE's."""
    left, right = jnp.broadcast_arrays(jnp.asarray(left, jnp.float32), jnp.asarray(right, jnp.float32))
    (left, left_exponent), (right, right_exponent) = _held_apart(left), _held_apart(right)
    high, low = precision.two_product(left, right)
    _, sign, biased, fraction = float_bits.raw_fields(high)
    biased = biased.astype(jnp.int32)
    significand = jnp.where(biased > 0, fraction | np.uint32(0x800000), fraction)
    exponent = jnp.maximum(biased, 1) - 150 + left_exponent + right_exponent - k
    # The exact product's residual decides a tie at the lowered place, so it rounds once.
    tail = jnp.where(low == 0, 0, jnp.where((low > 0) == (high > 0), 1, -1)).astype(jnp.int32)
    top = biased == 0xFF
    return float_bits.rounded_binary(sign, significand, exponent, jnp.float32, tail=tail,
                                     last=jnp.maximum(exponent, -149), infinite=top & (fraction == 0),
                                     nan=top & (fraction != 0))


def _scaled_value(product: jax.Array, factor: jax.Array) -> jax.Array:
    product, factor = product.astype(jnp.float32), jnp.asarray(factor, jnp.float32)
    return lowered_product(product, factor, 0) if flushes() else product * factor


def _scaled_tangent(primals, tangents):
    (product, factor), (d_product, d_factor) = primals, tangents
    value = scaled_exactly(product, factor)
    # Each perturbed operand's term alone (`_Transpose.tangent`'s rule): a factor held still meets no
    # cotangent of its own, so an infinite one gives no NaN, nor an absent term a lost negative zero.
    terms = [*(() if isinstance(d_product, SymbolicZero) else (d_product.astype(jnp.float32) * factor,)),
             *(() if isinstance(d_factor, SymbolicZero) else (product.astype(jnp.float32) * d_factor,))]
    if not terms:
        return value, jnp.zeros_like(value)
    return value, jnp.broadcast_to(functools.reduce(jnp.add, terms), value.shape)


scaled_exactly = contracted("test_a_ternary_weight_product_holds_every_order")(
    declared_tangent(KernelGlue(), _scaled_value, _scaled_tangent, symbolic_zeros=True))
"""`product * factor` in fp32 as IEEE's multiply rounds it, however small the factor or the result: where the
platform flushes subnormals (`flushes`), `lowered_product`'s one rounding, so a subnormal scale (a packed
matrix's `2**-127`) or a subnormal result survives (row 13340); elsewhere the plain multiply. Its derivative
is the plain product's, each perturbed operand's term alone."""


@dataclasses.dataclass(frozen=True)
class _Transpose:
    """`factored`'s transpose of `run` at rows of `shape`, typed as the rows are (`held`, their manual axes;
    `placed`, their explicit placement)."""
    run: Callable
    shape: tuple
    dtype: jnp.dtype
    held: frozenset
    placed: jax.sharding.NamedSharding | jax.sharding.PartitionSpec | None
    wide: Callable
    prepared: Callable

    @functools.cached_property
    def declared(self) -> Callable:
        """`value`, its derivative the plain transpose's (`tangent`)."""
        return contracted("test_a_ternary_weight_product_holds_every_order")(
            declared_tangent(KernelGlue(), self.value, self.tangent, nondiff_argnums=(2, 3),
                             symbolic_zeros=True))

    def operand_axes(self, ran) -> frozenset:
        """The manual axes the product's own operands, its rows and its matrix's, vary over."""
        return self.held.union(*(jax.typeof(leaf).mat.varying for leaf in jax.tree.leaves(ran)))

    def alone(self, ran, *values) -> tuple:
        """The axes `values` (a cotangent, a scale) vary over and the product's operands do not: there each
        device holds the same product, and its cotangent is summed before the pullback, as the plain
        transpose sums the cotangent of a product every device holds alike."""
        return tuple(sorted(frozenset().union(*(jax.typeof(value).mat.varying for value in values))
                            - self.operand_axes(ran)))

    def pulled(self, near, ran) -> tuple:
        """`near` pulled back through `run`, summed first over the axes it alone varies over, and the axes the
        partial varies over and the rows do not (each device's to sum after)."""
        before = self.alone(ran, near)
        near = jax.lax.psum(near, before) if before else near
        # The pullback at zeros typed as the rows are: placed, and varying as the cotangent's partials do.
        zeros = jnp.zeros(self.shape, jnp.bfloat16, out_sharding=self.placed)
        at = varying_like(zeros, near, *jax.tree.leaves(ran))
        (partial,) = jax.vjp(lambda each: self.run(each, *ran), at)[1](near)
        partial = partial.astype(jnp.float32)
        return partial, tuple(sorted(jax.typeof(partial).mat.varying - self.held))

    @staticmethod
    def summed(partial, excess):
        return jax.lax.psum(partial, excess) if excess else partial

    def value(self, cotangent, scale, ran, routed):
        """The plain transpose, the cotangent times its factor pulled back; where an entry of it is not
        finite, the moved one's (`moved`)."""
        plain = self.summed(*self.pulled(cotangent * scale, ran))
        if 0 in self.shape or cotangent.shape[-1] == 0:
            # No row, no depth or no column: the plain transpose's empty or zero gradient is the whole of it.
            return plain

        def moved():
            # Summed before its pullback where the operands are every device's; the axes its partials will
            # vary over that the rows do not, the operands', from types alone.
            before = self.alone(ran, cotangent, scale)
            excess = tuple(sorted(self.operand_axes(ran) - self.held))
            rescued = self.moved(cotangent, scale, routed, before, excess, plain)
            return jnp.where(jnp.isfinite(plain), plain, rescued)

        # The moved pullback where the plain one is not finite alone: each device of a sum decides alike,
        # the plain transpose summed over its axes.
        return jax.lax.cond(jnp.all(jnp.isfinite(plain)), lambda: plain, moved)

    def moved(self, cotangent, scale, routed, before, excess, plain):
        """The exact rescue (`factored`): every product `cotangent * scale` as fp32 forms it, from its
        operands' significands and binades, on a grid of `FACTORED_SLICE` bits a row (its token's, gathered,
        shared by every device that sums it), and the decoded matrix on its own; each slice pair's
        product summed by a bf16 GEMM whose fp32 sums are integers below 2**24, so exact
        (`FACTORED_WIDTH` columns a chunk), into int32 limbs, its routes' limbs added into their tokens and
        the devices' over the axes they sum, carried, then rounded once (`_rounded`). No product
        overflows, flushes or rounds before the one rounding, within a route, across its routes, or across
        its devices: what cancels anywhere leaves its exact remainder (rows 1464, 12392, 12403). A product
        not finite (a cotangent, factor or coefficient not finite) is left out of the sums and counted by
        its class (`_nonfinite_terms`): a coordinate meeting a NaN or both infinities is NaN, else one
        infinity's sign."""
        gather, wide_operands = routed
        f = jnp.broadcast_to(scale, cotangent.shape)
        bits, binade, negative = _products(cotangent, f)
        grid, token_grid = self.grids(binade, gather, tuple(sorted({*before, *excess})))
        matrix, *rest = self.prepared(*wide_operands)
        # The finite coefficients' grid: one not finite is IEEE's (`value`), outside the exact sums.
        held = jnp.abs(matrix.astype(jnp.float32))
        held = jnp.where(jnp.isfinite(held), held, 0.0)
        matrix_top = jnp.max(jnp.where(held > 0, float_bits.binade_of(held), -1024), initial=-1024)
        matrix_grid = _edge(_shared_max(matrix_top, tuple(sorted(jax.typeof(matrix_top).mat.varying
                                                                   - self.held))))
        depth, width = self.shape[-1], cotangent.shape[-1]
        self.bounded(cotangent, f, gather, matrix, rest)
        chunk = math.gcd(depth, FACTORED_DEPTH)
        columns_a_chunk = max(d for d in range(1, min(width, FACTORED_WIDTH) + 1) if width % d == 0)
        # The products' limbs below `FACTORED_HEADROOM` limbs over them that their sums carry into.
        limbs_count = FACTORED_HEADROOM + FACTORED_ROW_SLICES + FACTORED_MATRIX_SLICES - 1

        def pulled_chunk(at, out):
            columns = jax.lax.dynamic_slice_in_dim(matrix, at * chunk, chunk, axis=-2)
            limbs = varying_like(_limbs(limbs_count, cotangent, chunk), cotangent, columns,
                                 *jax.tree.leaves(rest), grid)
            # Whether a coordinate sums a product of each class not finite (`_nonfinite_terms`).
            classes = varying_like(_limbs(3, cotangent, chunk), cotangent, f, columns, *jax.tree.leaves(rest))

            def width_chunk(index, carry):
                limbs, classes = carry
                start = index * columns_a_chunk
                row_bits, row_binade, row_negative = (
                    jax.lax.dynamic_slice_in_dim(v, start, columns_a_chunk, axis=-1)
                    for v in (bits, binade, negative))
                block = jax.lax.dynamic_slice_in_dim(columns, start, columns_a_chunk, axis=-1)
                row_values = jax.lax.dynamic_slice_in_dim(cotangent, start, columns_a_chunk, axis=-1)
                row_factors = jax.lax.dynamic_slice_in_dim(f, start, columns_a_chunk, axis=-1)
                classes = self.classified(classes, row_values, row_factors, block, rest)
                block = jnp.where(jnp.isfinite(block), block, jnp.zeros_like(block))
                block_abs = jnp.abs(block.astype(jnp.float32))
                block_binade = jnp.where(block_abs > 0, float_bits.binade_of(block_abs), -1024)
                block_bits = (block_abs * float_bits.exact_power2(jnp.clip(7 - block_binade, -126, 127))
                              ).astype(jnp.int32)
                block_negative = block < 0

                def row_slice(i, limbs):
                    a = _slice(row_bits, row_binade - 23, grid, i, row_negative)

                    def taken(limbs):
                        def matrix_slice(j, limbs):
                            b = _slice(block_bits, block_binade - 7, matrix_grid, j, block_negative)

                            def multiplied(limbs):
                                summed = self.wide(a, b, *rest).astype(jnp.int32)
                                t = FACTORED_HEADROOM + i + j
                                at_t = jax.lax.dynamic_index_in_dim(limbs, t, 0, keepdims=False)
                                return jax.lax.dynamic_update_index_in_dim(limbs, at_t + summed, t, 0)

                            return jax.lax.cond(jnp.any(b != 0), multiplied, lambda limbs: limbs, limbs)

                        return jax.lax.fori_loop(0, FACTORED_MATRIX_SLICES, matrix_slice, limbs)

                    return jax.lax.cond(jnp.any(a != 0), taken, lambda limbs: limbs, limbs)

                return _carried(jax.lax.fori_loop(0, FACTORED_ROW_SLICES, row_slice, limbs)), classes

            limbs, classes = jax.lax.fori_loop(0, width // columns_a_chunk, width_chunk, (limbs, classes))
            # In fp32 from here, each 0 or 1: a sum over routes and devices is positive where any is.
            classes = classes.astype(jnp.float32)
            if gather is not None:
                routed_limbs = varying_like(_limbs(limbs_count, plain, chunk), limbs, gather)
                limbs = _routed(routed_limbs, gather, limbs)
                tokens = varying_like(_limbs(3, plain, chunk).astype(jnp.float32), classes, gather)
                classes = tokens.at[:, gather].add(classes)
            over = tuple(sorted(jax.typeof(limbs).mat.varying - self.held))
            if over:
                limbs = _carried(jax.lax.psum(limbs, over))
            if counted := tuple(sorted(jax.typeof(classes).mat.varying - self.held)):
                classes = jax.lax.psum(classes, counted)
            # The first limb's unit: 2**(grid + matrix grid - 2 slices + the headroom's slices).
            summed = _rounded(limbs, token_grid + matrix_grid + FACTORED_SLICE * (FACTORED_HEADROOM - 2))
            return jax.lax.dynamic_update_slice_in_dim(out, varying_like(_ieee(classes, summed), out),
                                                       at * chunk, axis=-1)

        # The rows' gradient, typed and placed as the plain transpose's is (its devices' sums taken, as each
        # chunk's are), so the loop's carry and the branch it returns from keep its type.
        out = varying_like(jnp.zeros(self.shape, jnp.float32, out_sharding=explicit_spec(plain)), plain)
        return jax.lax.fori_loop(0, depth // chunk, pulled_chunk, out)

    def grids(self, binade, gather, shared) -> tuple[jax.Array, jax.Array]:
        """A row's grid (its products' largest binade rounded up to a slice's edge, shared over `shared`
        axes) and its token's, the same where nothing is gathered: a gathered row's is its token's, every
        route's and every device's that sums it."""
        top = jnp.max(binade, axis=-1, keepdims=True, initial=-1024)
        if gather is None:
            grid = _edge(_shared_max(top, shared))
            return grid, grid
        token = varying_like(jnp.full((self.shape[0], 1), -1024, top.dtype), top, gather)
        token = token.at[gather].max(top)
        token_grid = _edge(_shared_max(token, tuple(sorted(jax.typeof(token).mat.varying - self.held))))
        return token_grid[gather], token_grid

    def bounded(self, cotangent, factor, gather, matrix, rest) -> None:
        """Refuse a rescue whose sums its int32 limbs cannot hold: more devices summing its digits than
        `FACTORED_FAN_IN`, or more products (columns, routes, devices) than `FACTORED_REACH`. Its devices
        are every axis a contributing operand varies over the rows do not, of which each psum sums those
        its own sums vary over."""
        leaves = jax.tree.leaves((cotangent, factor, gather, matrix, rest))
        varying = frozenset().union(*(jax.typeof(leaf).mat.varying for leaf in leaves))
        devices = math.prod(jax.lax.axis_size(axis) for axis in sorted(varying - self.held))
        if devices > FACTORED_FAN_IN:
            raise ValueError(f"the exact rescue sums {devices} devices' digits, past the {FACTORED_FAN_IN} "
                             f"whose sum int32 holds")
        width = cotangent.shape[-1]
        if width * (1 if gather is None else gather.shape[0]) * devices >= FACTORED_REACH:
            raise ValueError(f"the exact rescue's sums of {width} columns, {cotangent.shape[0]} routes and "
                             f"{devices} devices could pass its leading limb's int32 ({FACTORED_REACH})")

    def classified(self, classes, cotangent, factor, coefficients, rest):
        """`classes` (int32 0 or 1: whether each coordinate's products include one of a class not finite,
        +inf, -inf, NaN) with a chunk's (`_nonfinite_terms`, one 0/1 bf16 GEMM a class, its count below
        2**24) present or not (`_present`, never a sum that could wrap), read only where the chunk holds a
        product not finite: a finite cancellation runs none of their GEMMs where its predicate is one value
        (unmapped under `vmap`); under `vmap` a batched predicate's `cond` runs both branches and
        selects, the same classes. Each device decides alone; their classes meet after the chunk's
        columns."""
        def counted(classes):
            for index, (rows, matrix) in enumerate(_nonfinite_terms(cotangent, factor, coefficients)):
                counts = self.wide(rows.astype(jnp.bfloat16), matrix.astype(jnp.bfloat16), *rest)
                held = jax.lax.dynamic_index_in_dim(classes, index, 0, keepdims=False)
                classes = jax.lax.dynamic_update_index_in_dim(classes, _present(held, counts), index, 0)
            return classes

        special = functools.reduce(jnp.logical_or, [jnp.any(~jnp.isfinite(v))
                                                    for v in (cotangent, factor, coefficients)])
        return jax.lax.cond(special, counted, lambda classes: classes, classes)

    def tangent(self, ran, routed, primals, tangents):
        """The plain transpose's derivative, `cotangent * scale` pulled back, every term of a perturbed
        operand alone (a scale held still meets no cotangent of its own: an infinite one is no NaN)."""
        (cotangent, scale), (d_cotangent, d_scale) = primals, tangents
        terms = [*(() if isinstance(d_cotangent, SymbolicZero) else (d_cotangent * scale,)),
                 *(() if isinstance(d_scale, SymbolicZero) else (cotangent * d_scale,))]
        value = self.declared(cotangent, scale, ran, routed)
        if not terms:
            return value, jnp.zeros_like(value)
        return value, self.summed(*self.pulled(functools.reduce(jnp.add, terms), ran))


FACTORED_SLICE = 6
"""The bits a slice of the exact rescue's grids holds (`_Transpose.moved`): two slices' product at most
`2**12`, so `FACTORED_WIDTH` of them sum below `2**24`, exactly in a bf16 GEMM's fp32 sums."""

FACTORED_WIDTH = 4096
"""The cotangent's columns a chunk the exact rescue's GEMMs sum at once (a divisor of its width at most
this): their slices' products sum below `2**24`, exact."""

FACTORED_DEPTH = 64
"""The depth a chunk the exact rescue pulls back at once (a divisor of the rows' depth): its limbs are
`FACTORED_ROW_SLICES + FACTORED_MATRIX_SLICES - 1` int32 of rows x chunk."""

FACTORED_HEADROOM = 6
"""Limbs above the products' first its sums carry into (36 bits): a sum over 2**19 columns, every route of a
token and every device of a mesh, each product below the two grids' tops."""

FACTORED_ROW_SLICES = 97
"""A row's slices: its products' binades span at most a cotangent's and a factor's ranges multiplied (-298
to 254, the largest product's grid rounded up a slice) and 24 bits of significand below them."""

FACTORED_FAN_IN = 1 << 24
"""The carried limbs (each digit below `2**FACTORED_SLICE`) the exact rescue adds into one before carrying:
routes into a token, devices into their sum; `2**24` of them sum below int32's largest."""

FACTORED_REACH = 1 << 54
"""The products, columns times routes times devices, a sum of which the exact rescue's leading limb holds:
each below its grids' tops, so their sum is below `2**54` times its unit's `2**-24`, short of int32's
`2**31`."""

FACTORED_MATRIX_SLICES = 6
"""A decoded packed matrix's slices on its grid: 36 bits, past any format's span (NVFP4's E2M1 codes times
E4M3 block scales, 21 binades and 5 bits; ternary's one)."""


def _limbs(count: int, like: jax.Array, columns: int) -> jax.Array:
    """`count` int32 limbs of zeros, each `like`'s rows by `columns`, its rows placed as `like`'s are where an
    Explicit axis splits them (a chunk's sums, which the GEMMs' row-placed products add into)."""
    placed = explicit_spec(like)
    rows = None if placed is None else placed.spec[0]
    return jnp.zeros((count, like.shape[0], columns), jnp.int32, out_sharding=None if placed is None else
                     jax.sharding.NamedSharding(placed.mesh, jax.sharding.PartitionSpec(None, rows, None)))


def _routed(tokens: jax.Array, gather: jax.Array, limbs: jax.Array,
            fan_in: int = FACTORED_FAN_IN) -> jax.Array:
    """Each route's carried `limbs` added into its token's of `tokens` (`gather` its token), `fan_in` routes
    at a time, carried after each, so no digit's sum passes int32."""
    for first in range(0, gather.shape[0], fan_in):
        routes = slice(first, first + fan_in)
        tokens = _carried(tokens.at[:, gather[routes]].add(limbs[:, routes]))
    return tokens


def _nonfinite_terms(cotangent: jax.Array, factor: jax.Array, matrix: jax.Array) -> tuple:
    """The classes of products a coordinate's IEEE sum reads where any is not finite, each as a cotangent
    indicator beside a matrix indicator over the columns, side by side, so one GEMM counts the class: a
    product `cotangent * factor` (exact, finite past fp32's largest or not) times a coefficient that is
    +inf, -inf, or NaN (a NaN operand, or zero times an infinity). Each operand's class from its bits
    (`_classes`), so a subnormal is no zero where a process flushes them."""
    c_negative, c_zero, c_inf, c_nan = _classes(cotangent)
    f_negative, f_zero, f_inf, f_nan = _classes(factor)
    finite = ~(c_inf | c_nan | f_inf | f_nan)
    nan = c_nan | f_nan | (c_inf & f_zero) | (f_inf & c_zero)
    zero = finite & (c_zero | f_zero)
    negative = c_negative ^ f_negative
    signed = ~nan & ~zero
    c_pos, c_neg = finite & signed & ~negative, finite & signed & negative
    c_pinf, c_ninf = ~finite & signed & ~negative, ~finite & signed & negative
    m_negative, m_zero, m_inf, m_nan = _classes(matrix)
    m_finite = ~(m_inf | m_nan)
    m_pinf, m_ninf = m_inf & ~m_negative, m_inf & m_negative
    m_pos, m_neg = m_finite & ~m_zero & ~m_negative, m_finite & ~m_zero & m_negative
    c_all, m_all = jnp.ones_like(c_pos), jnp.ones_like(m_pos)

    def side_by_side(cotangents, coefficients):
        return jnp.concatenate(cotangents, axis=-1), jnp.concatenate(coefficients, axis=-1)

    positive = side_by_side((c_pos, c_neg, c_pinf, c_ninf), (m_pinf, m_ninf, m_pos | m_pinf, m_neg | m_ninf))
    negative = side_by_side((c_pos, c_neg, c_pinf, c_ninf), (m_ninf, m_pinf, m_neg | m_ninf, m_pos | m_pinf))
    invalid = side_by_side((nan, c_all, c_pinf | c_ninf, zero), (m_all, m_nan, m_zero, m_pinf | m_ninf))
    return positive, negative, invalid


def _products(cotangent: jax.Array, factor: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each product `cotangent * factor` as fp32 forms it, from its operands' bits (a subnormal's kept where
    a process flushes subnormal operands): its significand normalized into [1, 2) as a 24-bit integer at
    2**23, its binade (-1024 where it is zero or not finite) and its sign."""
    e_c, e_f = (float_bits.binade_of(v) for v in (cotangent, factor))
    counted = (e_c > -1024) & (e_c < 128) & (e_f > -1024) & (e_f < 128)
    product = jnp.where(counted, _significand(cotangent) * _significand(factor), 0.0)
    doubled = jnp.abs(product) >= 2
    product = jnp.where(doubled, product * 0.5, product)
    binade = jnp.where(counted, e_c + e_f + doubled.astype(e_c.dtype), -1024)
    return (jnp.abs(product) * 2.0 ** 23).astype(jnp.int32), binade, product < 0


def _present(held: jax.Array, counts: jax.Array) -> jax.Array:
    """`held` (int32 0 or 1) or whether `counts` (a chunk's count of a class, fp32) holds any: a class's
    presence, bounded whatever the columns, routes or devices that meet it."""
    return jnp.maximum(held, (counts > 0).astype(held.dtype))


def _ieee(classes: jax.Array, summed: jax.Array) -> jax.Array:
    """`summed`, the exact finite products' rounded sum, where a coordinate met no product not finite; else
    IEEE's: NaN beside a NaN or both infinities (`classes`' counts: +inf, -inf, NaN), else its infinity."""
    plus, minus, invalid = classes[0] > 0, classes[1] > 0, classes[2] > 0
    return jnp.where(invalid | (plus & minus), jnp.nan,
                     jnp.where(plus, jnp.inf, jnp.where(minus, -jnp.inf, summed)))


def _classes(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Whether each of `x` is negative, zero, infinite or NaN, read from its bits (`float_bits.raw_fields`):
    a subnormal is neither zero nor of no sign, whatever a process's float arithmetic flushes."""
    _, sign, biased, fraction = float_bits.raw_fields(x)
    top = biased == (1 << jnp.finfo(x.dtype).nexp) - 1
    return sign != 0, (biased == 0) & (fraction == 0), top & (fraction == 0), top & (fraction != 0)


def _edge(top: jax.Array) -> jax.Array:
    """A grid's top: a slice's edge above `top`'s binade (`FACTORED_SLICE`); 0 where nothing is counted."""
    return jnp.where(top > -1024, FACTORED_SLICE * (jnp.floor_divide(top, FACTORED_SLICE) + 1), 0)


def _shared_max(x: jax.Array, axes: tuple) -> jax.Array:
    """`x`'s largest over the manual `axes` it varies over (a grid every device that sums it shares)."""
    over = tuple(axis for axis in axes if axis in jax.typeof(x).mat.varying)
    return jax.lax.pmax(x, over) if over else x


def _slice(bits: jax.Array, lowest: jax.Array, grid: jax.Array, index, negative: jax.Array) -> jax.Array:
    """Slice `index` of values `bits * 2**lowest` on a grid topped at `grid`: their bits in
    `[grid - FACTORED_SLICE * (index + 1), grid - FACTORED_SLICE * index)` as an integer, signed, in bf16
    (exact: at most `2**FACTORED_SLICE - 1`)."""
    shift = (grid - FACTORED_SLICE * (index + 1)) - lowest
    right = jnp.right_shift(bits, jnp.clip(shift, 0, 31))
    left = jnp.left_shift(bits, jnp.clip(-shift, 0, 31))
    piece = jnp.bitwise_and(jnp.where(shift >= 0, right, left), 2 ** FACTORED_SLICE - 1)
    return jnp.where(negative, -piece, piece).astype(jnp.bfloat16)


def _carried(limbs: jax.Array) -> jax.Array:
    """int32 `limbs` (the highest first, each `2**FACTORED_SLICE` times the next) carried so every limb but
    the first is in `[0, 2**FACTORED_SLICE)`: the same sum, no limb near int32's range."""
    def carry(step, limbs):
        t = limbs.shape[0] - 1 - step
        low = jax.lax.dynamic_index_in_dim(limbs, t, 0, keepdims=False)
        up = jnp.right_shift(low, FACTORED_SLICE)
        limbs = jax.lax.dynamic_update_index_in_dim(limbs, low - jnp.left_shift(up, FACTORED_SLICE), t, 0)
        above = jax.lax.dynamic_index_in_dim(limbs, t - 1, 0, keepdims=False)
        return jax.lax.dynamic_update_index_in_dim(limbs, above + up, t - 1, 0)

    return jax.lax.fori_loop(0, limbs.shape[0] - 1, carry, limbs)


def _rounded(limbs: jax.Array, unit: jax.Array) -> jax.Array:
    """The fp32 value nearest (ties to even) `sum(limbs[t] * 2**(unit - FACTORED_SLICE * t))`, carried int32
    `limbs` (the first signed and of any size, `FACTORED_HEADROOM` deep; each other in
    `[0, 2**FACTORED_SLICE)`): the first limb read as `FACTORED_LEAD` digits of its own above the others,
    the sum's leading 30 bits and a sticky bit below them where any lower one is set, rounded once in
    integers (`float_bits.rounded_float32`): a subnormal holds where a process flushes its float32
    arithmetic, and a sum past fp32's largest is an infinity."""
    width = FACTORED_SLICE
    negative = limbs[0] < 0
    limbs = _carried(jnp.where(negative[None], -limbs, limbs))
    # The digits: the first limb's `FACTORED_LEAD` (the highest first), then every other limb's.
    head = limbs[0].astype(jnp.uint32)
    lead_digits = FACTORED_LEAD
    rest = limbs[1:]
    positions = jnp.arange(rest.shape[0], dtype=jnp.int32).reshape(-1, *(1,) * (rest.ndim - 1))
    later = jnp.cumsum((rest != 0)[::-1].astype(jnp.int32), axis=0)[::-1]

    def at(values, index):
        return jnp.sum(jnp.where(positions == index[None], values, 0), axis=0, dtype=values.dtype)

    def digit(index):
        shift = (width * (lead_digits - 1 - jnp.clip(index, 0, lead_digits - 1))).astype(jnp.uint32)
        own = (jnp.right_shift(head, shift) & jnp.uint32(2 ** width - 1)).astype(jnp.int32)
        return jnp.where(index < lead_digits, own, at(rest, index - lead_digits))

    length = 32 - jax.lax.clz(head).astype(jnp.int32)
    lead = jnp.where(head > 0, lead_digits - 1 - (length - 1) // width,
                     lead_digits + jnp.argmax(rest != 0, axis=0).astype(jnp.int32))
    mantissa = functools.reduce(jnp.add, [jnp.left_shift(digit(lead + k), width * (4 - k)) for k in range(5)])
    below = lead + 5
    head_below = (head & (jnp.left_shift(jnp.uint32(1), (width * jnp.clip(lead_digits - below, 0, 5)
                                                         ).astype(jnp.uint32)) - jnp.uint32(1))) != 0
    sticky = jnp.where(below < lead_digits, head_below | (later[0] > 0), at(later, below - lead_digits) > 0)
    significand = jnp.left_shift(mantissa.astype(jnp.uint32), jnp.uint32(1)) | sticky.astype(jnp.uint32)
    sign = jnp.where(negative, jnp.uint32(1 << 31), jnp.uint32(0))
    exponent = unit + width * (lead_digits - 1) - width * (lead + 4) - 1
    return float_bits.rounded_float32(sign, significand, exponent)


FACTORED_LEAD = 6
"""The digits of `FACTORED_SLICE` bits the first limb's int32 magnitude is read as (`_rounded`): 36 bits."""


def _significand(x: jax.Array) -> jax.Array:
    """A finite fp32 `x`'s significand with its sign, in [1, 2), from its bits (`float_bits`): a subnormal's
    normalized, read where a process flushes subnormal operands; 0 for zero."""
    sign, biased, fraction, normalized, _ = float_bits.float_bits(x.astype(jnp.float32))
    one = sign | (127 << 23) | normalized
    return jnp.where((biased == 0) & (fraction == 0), 0.0, jax.lax.bitcast_convert_type(one, jnp.float32))
