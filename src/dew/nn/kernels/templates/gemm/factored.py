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
from jax.core import Tracer
from jax.custom_derivatives import SymbolicZero

from dew.nn import float_bits, precision
from dew.nn.derivatives import KernelGlue, contracted, declared_tangent
from dew.nn.kernels import generation
from dew.nn.kernels.templates import axes as side_axes
from dew.nn.sharding import (
    concrete_mesh,
    explicit_placement,
    explicit_spec,
    placed_zeros,
    summed_to,
    unbroadcast,
    varying_like,
)


def factored(run: Callable, operands: tuple, factor: jax.Array, rows: jax.Array, *, wide: Callable,
             wide_operands: tuple, prepared: Callable | None = None, gather: jax.Array | None = None,
             groups: jax.Array | None = None, owed: float = 1.0, least: float = 1.0,
             finished: Callable | None = None) -> jax.Array:
    """`run(rows as bf16, *operands) * factor * owed`, linear in `rows`: `run` a product of bf16 rows (those
    `gather` reads where given, `gemm`'s `gather`, among `operands`), `factor` a packed matrix's fp32 scales
    broadcast against its output (`formats.scale_factor`, an NVFP4 matrix's tensor scale): one a row's, one a
    column's, one an entry's or one for all; with `groups` (the group sizes of the output's rows), one a
    group's (`[groups]`) or one a group's column (`[groups, width]`), which each row takes from its group
    where the product and its pullback read it (`_per_row`), no row's saved. `owed` is a power of two the
    product owes besides (an NVFP4 decode's, `bits.NVFP4_OWED`), applied with the factor in its one rounding
    (`owed_exactly`), never folded into a factor it would carry past fp32's largest (a tensor scale of 2**122
    under the decode's 2**8) or below its normals. `least`, the least magnitude of a nonzero coefficient of
    the decoded matrix (`prepared`'s). `finished`, where given, `run` with the factor and the owed power of
    two applied in its own finish (`finished(rows, *operands, factor, groups)`: a launch's epilogue,
    `gemm.nvfp4_product`'s), which the product runs in their place. Its transpose is the plain one, the
    cotangent times its factor through `run`'s pullback, wherever that holds it: finite, and every nonzero
    product of a row's route at least the plain transpose's least (`plain_least`); where it does not (a
    product past fp32's largest, or one its bf16 parts or fp32's normals lose), the exact one
    (`_Transpose.moved`): each product `cotangent * factor` and each coefficient of the decoded matrix
    (`prepared(*wide_operands)`, its depth its axis -2) cut into integer slices on grids of their own, each
    pair of slices multiplied by a bf16 GEMM whose fp32 sums are exact integers (`wide`, the transpose over
    the rows `run` reads, a route each where gathered), the sums kept in int32 limbs, added across routes into
    their tokens and across the devices that sum them, then rounded once (`_rounded`). A product of a
    cotangent, factor or coefficient not finite is IEEE's: no product overflows or flushes, so a scale of
    2**126 under cotangents that cancel pulls back their remainder, a subnormal one among them. A checked
    `shard_map`'s devices sum a cotangent only they vary over before its pullback, and their partials
    after."""
    raised = round(math.log2(owed))
    if owed != 2.0 ** raised or raised < 0:
        raise ValueError(f"a product's owed factor is a power of two of at least 1, not {owed}")
    transpose = _Transpose(run, rows.shape, rows.dtype, frozenset(jax.typeof(rows).mat.varying),
                           explicit_spec(rows), wide, prepared or (lambda *each: each), raised,
                           plain_least(least))

    def apply(residuals, moving):
        scale, sizes, ran, _ = residuals
        if finished is not None:
            return finished(moving.astype(jnp.bfloat16), *ran, scale, sizes)
        product = run(moving.astype(jnp.bfloat16), *ran)
        return owed_exactly(product, per_row(scale, sizes, product.shape[0]), raised)

    def pullback(residuals, cotangent):
        scale, sizes, ran, routed = residuals
        scale = per_row(scale, sizes, cotangent.shape[0]).astype(jnp.float32)
        return transpose.paired(ran, routed, (cotangent.astype(jnp.float32),), (scale,)).astype(rows.dtype)

    # An Explicit placement of the rows kept as a residual of no rows (`rows[:0]`, no memory): an eager
    # pullback's allocations and transfers take its concrete devices (`sharding.placed_zeros`, `_placed`).
    anchor = None if transpose.placed is None else jax.lax.stop_gradient(rows[:0])
    routed = (gather, wide_operands, anchor)
    return generation.linear_map(apply, pullback, (factor, groups, operands, routed), rows)


def plain_least(least: float) -> int:
    """The least binade of a nonzero scaled cotangent the plain transpose holds, its matrix's least nonzero
    coefficient `least`: every bit of it, its least a bf16 part may carry (`gemm._bf16_parts`, 23 binades
    below its own), times that coefficient a normal float32, which a flushing platform keeps, so neither its
    parts nor its products lose any of it (train 1409's review: a column scale of 2**-142 made every route's
    2**-134, which its bf16 parts dropped, and the depth's normal 2**-125 read 0; a cancelling pair's
    residual's product, 2**-141, flushed): -85 for an NVFP4 matrix's 2**-18, -103 for a ternary one's 1."""
    return (jnp.finfo(jnp.float32).minexp + jnp.finfo(jnp.float32).nmant
            - math.floor(math.log2(least)))


def per_row(factor: jax.Array, groups: jax.Array | None, rows: int) -> jax.Array:
    """`factor` as `rows` output rows take it: as it is without `groups`; with them, each row its group's
    (`side_axes.at_rows`, the one group-to-row mapping): `[groups]` a row's, `[rows, 1]`; `[groups, width]`
    a row's columns', `[rows, width]`."""
    if groups is None:
        return factor
    each = factor if jnp.ndim(factor) == 2 else factor[:, None]
    return side_axes.at_rows(each, (side_axes.GROUPS, 'columns'), groups, rows)


FACTOR_RAISE = 62
"""The power of two `int8_epilogue` raises a subnormal scale by: the least
(`2**-149`) into the normals (`2**-87`), the largest still below `2**-64`, so no product of it overflows that
the plain one would not."""


def int8_epilogue(product: jax.Array, peak: jax.Array, factor: jax.Array, shift: jax.Array) -> jax.Array:
    """The int8-row ternary product's epilogue (`ternary.ternary_weight_product`, an XLA program after its
    launch): `product * peak * factor * 2**-shift` in `precision.twice_scaled`'s order, a subnormal `factor`
    raised by `2**FACTOR_RAISE` from its bits and the last multiply exact and rounded once with the lowering
    (`lowered_product`), so a subnormal scale survives whatever the platform flushes: XLA's CPU and TPU
    flush subnormals (row 4023), and so did a GH200's compiled program (row 14172: a column scale of 2^-127
    read zero there, where each multiply alone kept it, M-NG-73)."""
    product, peak, factor = (jnp.asarray(each, jnp.float32) for each in (product, peak, factor))
    binade = float_bits.binade_of(factor)
    raised = jnp.where((binade < jnp.finfo(jnp.float32).minexp) & (binade > -1024),
                       jnp.int32(FACTOR_RAISE), jnp.int32(0))
    whole = float_bits.raised_exactly(factor, raised)
    left, right = precision.twice_scaled_operands(product, peak, whole)
    return lowered_product(left, right, jnp.asarray(shift, jnp.int32) + raised)


def lowered_product(left: jax.Array, right: jax.Array, k: jax.Array | int) -> jax.Array:
    """`left * right * 2**-k` (fp32 operands, `k` int32 either sign) rounded once to nearest even in fp32,
    however far below its least normal, whatever the platform flushes (`float_bits.owed_product`)."""
    k = jnp.clip(k, -float_bits.EXPONENT_REACH, float_bits.EXPONENT_REACH)
    return float_bits.owed_product(jnp.asarray(left, jnp.float32), right, -k)


def _linear_scaled(tangent: jax.Array, factor: jax.Array, raised: int = 0) -> jax.Array:
    """`tangent * factor * 2**raised` rounded once as `owed_exactly` rounds it (a subnormal factor's product
    formed among the normals on a flushing platform, `lowered_product`: `2**126 * 2**-127` is 0.5, row 13703),
    linear in `tangent`, its transpose the cotangent times the same factor alike (`generation.linear_map`),
    its derivative in `factor` `scaled_exactly`'s own (the mixed term a higher order meets). The moving
    operand is widened to fp32 before it is broadcast to the factor's shape and made to vary where it does,
    so the transposes of that broadcast and that variation sum fp32 cotangents, rounded once to the
    tangent's own dtype after (train 1362's review: a bf16 product over fp32 factors `[256, -255.5]` sums
    0.5, not the 0 of each term rounded first)."""

    def apply(factor, moving):
        return owed_exactly(moving, factor, raised)

    def pullback(factor, cotangent):
        # Placed as the moving operand is on an Explicit mesh (a replicated tangent's cotangent gathered from
        # a split scale's product), as the plain multiply's transpose places it: by its type's sharding
        # itself, which no mesh context need hold (train 1426's review).
        return _placed(owed_exactly(cotangent.astype(jnp.float32), factor, raised), placed)

    factor = jnp.asarray(factor, jnp.float32)
    wide = jnp.asarray(tangent).astype(jnp.float32)
    moving = varying_like(jnp.broadcast_to(wide, jnp.broadcast_shapes(jnp.shape(wide), jnp.shape(factor))),
                          factor)
    placed = jax.typeof(moving).sharding
    return generation.linear_map(apply, pullback, factor, moving)


def _scaled_value(raised: int, product: jax.Array, factor: jax.Array) -> jax.Array:
    """`owed_exactly`'s value, an XLA program's: `lowered_product`'s one rounding, the owed power of two its
    exponent's, so a subnormal factor or result survives whatever the platform flushes (no multiply of a
    subnormal: a packed matrix's scale of `2**-127`, row 13340; a tensor scale of 2**-126 whose product is
    2**-129, train 1403's review; on a GH200 the native multiply's program read a column scale of 2^-127's
    products as zero, where each multiply alone kept them, row 14172, M-NG-73)."""
    product, factor = jnp.broadcast_arrays(product.astype(jnp.float32), jnp.asarray(factor, jnp.float32))
    return lowered_product(product, factor, -raised)


def owed_native(product: jax.Array, factor: jax.Array, *, raised: int) -> jax.Array:
    """`owed_exactly`'s value in a packed launch's epilogue alone (`gemm._packed_finish`, its kernel's float32
    keeping subnormals; tests/test_gemm_nvfp4.py's GPU cells hold it to the exact finish), rounded once: the
    plain multiply, the factor and the owed power of two one multiplier where that is finite (a subnormal
    factor raised exactly), else the product times the power of two, exact (a subnormal product raised; one it
    overflows is one whose owed product past a factor of 2**(128 - raised) overflows too), then times the
    factor, rounded once (train 1450's review: the product times the factor first rounded a subnormal one on
    the subnormal grid before the power raised it). A power past fp32's (`raised` 128 or more, which no
    float32 multiplier holds) is `lowered_product`'s one rounding in integer arithmetic."""
    # The side's columns broadcast to the accumulator's tile explicitly (an epilogue's `[columns]` factor).
    product, factor = jnp.broadcast_arrays(product, jnp.asarray(factor, product.dtype))
    if not raised:
        return product * factor
    if raised > jnp.finfo(jnp.float32).maxexp - 1:
        return lowered_product(product, factor, -raised)
    power = np.float32(2.0 ** raised)
    whole = factor * power
    together = jnp.isfinite(whole) | ~jnp.isfinite(factor)
    return jnp.where(together, product * whole, product * power * factor)


def _scaled_tangent(raised: int, primals, tangents):
    (product, factor), (d_product, d_factor) = primals, tangents
    value = _scaled(raised, product, factor)
    # Each perturbed operand's term alone (`_Transpose.tangent`'s rule): a factor held still meets no
    # cotangent of its own, so an infinite one gives no NaN, nor an absent term a lost negative zero; each
    # term linear in its direction (`_linear_scaled`, its transpose the other operand times the cotangent:
    # train 1426's review, the factor's term through the value's bit reading had none), the product's range-
    # safe in a subnormal factor, the factor's owed its power of two exactly where it is.
    wide = product.astype(jnp.float32)
    terms = [*(() if isinstance(d_product, SymbolicZero) else (_linear_scaled(d_product, factor, raised),)),
             *(() if isinstance(d_factor, SymbolicZero) else
               (_linear_scaled(d_factor, wide, raised) if raised else wide * d_factor,))]
    if not terms:
        return value, jnp.zeros_like(value)
    return value, jnp.broadcast_to(functools.reduce(jnp.add, terms), value.shape)


_scaled = contracted("test_a_ternary_weight_product_holds_every_order")(
    declared_tangent(KernelGlue(), _scaled_value, _scaled_tangent, nondiff_argnums=(0,), symbolic_zeros=True))


def scaled_exactly(product: jax.Array, factor: jax.Array) -> jax.Array:
    """`product * factor` in fp32 as IEEE's multiply rounds it, however small the factor or the result
    (`owed_exactly` owing nothing): a subnormal scale (a packed matrix's `2**-127`) or a subnormal result
    survives a platform that flushes them (row 13340). Its derivative is the plain product's, each perturbed
    operand's term alone."""
    return _scaled(0, product, factor)


def owed_exactly(product: jax.Array, factor: jax.Array, raised: int) -> jax.Array:
    """`product * factor * 2**raised` (`raised` a whole exponent of at least 0, an owed power of two's: an
    NVFP4 decode's, `bits.NVFP4_OWED`) in fp32 rounded once, however large or small the factor, the result
    or the power, in integer arithmetic on every platform (`lowered_product`, `_scaled_value`: an XLA program
    whatever it flushes, row 14172; train 1403's review: an NVFP4 column's tensor scale of 2**122 under its
    decode's 2**8 overflowed a folded multiplier; train 1450's: 2**128 overflowed the power itself); only a
    packed launch's epilogue folds it natively (`owed_native`). Its derivative is the plain product's, each
    perturbed operand's term alone."""
    return _scaled(raised, product, factor)


@dataclasses.dataclass(frozen=True)
class _Transpose:
    """`factored`'s transpose of `run` at rows of `shape`, typed as the rows are (`held`, their manual axes;
    `placed`, their explicit placement)."""
    run: Callable
    shape: tuple
    dtype: jnp.dtype
    held: frozenset
    placed: jax.sharding.NamedSharding | None
    wide: Callable
    prepared: Callable
    raised: int
    least: int

    @functools.cached_property
    def paired(self) -> Callable:
        """`paired(ran, routed, cotangents, scales)`: the transpose of the sum of the pairs' products
        (`combined`), its derivative each perturbed operand's term a pair of the same exact sum
        (`_paired_tangent`)."""
        return contracted("test_a_ternary_weight_product_holds_every_order")(
            declared_tangent(KernelGlue(), self._paired_value, self._paired_tangent, nondiff_argnums=(0, 1),
                             symbolic_zeros=True))

    def _paired_value(self, ran, routed, cotangents, scales):
        return self.combined(tuple(zip(cotangents, scales, strict=True)), ran, routed)

    def operand_axes(self, ran) -> frozenset:
        """The manual axes the product's own operands, its rows and its matrix's, vary over."""
        return self.held.union(*(jax.typeof(leaf).mat.varying for leaf in jax.tree.leaves(ran)))

    def alone(self, ran, *values) -> tuple:
        """The axes `values` (a cotangent, a scale) vary over and the product's operands do not: there each
        device holds the same product, and its cotangent is summed before the pullback, as the plain
        transpose sums the cotangent of a product every device holds alike."""
        return tuple(sorted(frozenset().union(*(jax.typeof(value).mat.varying for value in values))
                            - self.operand_axes(ran)))

    def pulled(self, near, ran, anchor) -> tuple:
        """`near` pulled back through `run`, summed first over the axes it alone varies over, and the axes the
        partial varies over and the rows do not (each device's to sum after); its seed made where the rows
        are (`anchor`, theirs of no rows)."""
        before = self.alone(ran, near)
        near = jax.lax.psum(near, before) if before else near
        # The pullback at zeros typed as the rows are: placed, and varying as the cotangent's partials do.
        zeros = placed_zeros(self.shape, jnp.bfloat16, self.placed, anchor)
        at = varying_like(zeros, near, *jax.tree.leaves(ran))
        out, pull = jax.vjp(lambda each: self.run(each, *ran), at)
        # The cotangent typed as the product is (a split column scale's product split as it is not: train
        # 1426's review's column-split witness).
        (partial,) = pull(_placed(near, jax.typeof(out).sharding, out))
        partial = partial.astype(jnp.float32)
        return partial, tuple(sorted(jax.typeof(partial).mat.varying - self.held))

    @staticmethod
    def summed(partial, excess):
        return jax.lax.psum(partial, excess) if excess else partial

    def combined(self, pairs, ran, routed):
        """The transpose of the sum of `pairs`' products, each a cotangent times its factor and the owed power
        of two: the plain one, their products summed and pulled back; where an entry of it is not finite,
        or a row's product is smaller than it holds (`lost`), the moved one's (`moved`), every pair's
        products in its one exact sum. Pairs whose products lie within the plain range and cancel past
        fp32's resolution of them (`2**-102` and `-(2**-102)(1 + 2**-46)`) round as the plain products'
        sum, as fp32's own would (train 1437's review): a check of every pair's cancellation would cost the
        plain path its zero overhead."""
        near = functools.reduce(jnp.add, [owed_exactly(cotangent, scale, self.raised)
                                          for cotangent, scale in pairs])
        plain = self.summed(*self.pulled(near, ran, routed[2]))
        if 0 in self.shape or near.shape[-1] == 0:
            # No row, no depth or no column: the plain transpose's empty or zero gradient is the whole of it.
            return plain
        # Summed before its pullback where the operands are every device's; the axes its partials will vary
        # over that the rows do not, the operands', from types alone.
        before = self.alone(ran, *(value for pair in pairs for value in pair))
        excess = tuple(sorted(self.operand_axes(ran) - self.held))
        kept = jnp.isfinite(plain) & ~self.lost(pairs, routed[0], tuple(sorted({*before, *excess})))

        def moved():
            rescued = self.moved(pairs, routed, before, excess, plain)
            return jnp.where(kept, plain, rescued)

        # The moved pullback where the plain one does not hold every row: each device of a sum decides alike,
        # the plain transpose summed over its axes and the range shared over them.
        return jax.lax.cond(jnp.all(kept), lambda: plain, moved)

    def lost(self, pairs, gather, shared) -> jax.Array:
        """Whether the plain transpose loses a row's range below (`[rows, 1]`): a route of it a pair's product
        of which, finite and nonzero, falls below the least binade it holds (`least`), read from its
        operands' binades and the owed power of two (a subnormal's from its bits, where a process flushes
        them; a product that rounds to zero by its operands'), every device that sums it (`shared`)
        deciding alike."""
        small = functools.reduce(jnp.logical_or,
                                 [self._small(cotangent, scale) for cotangent, scale in pairs])
        small = jnp.any(small, axis=-1, keepdims=True).astype(jnp.int32)
        if gather is not None:
            token = varying_like(jnp.zeros((self.shape[0], 1), jnp.int32), small, gather)
            small = token.at[gather].max(small)
        return _shared_max(small, shared) > 0

    def _small(self, cotangent, scale) -> jax.Array:
        # A product of finite nonzero operands whose binades and the owed power of two sum below `least`.
        e_c = float_bits.binade_of(cotangent)
        e_f = float_bits.binade_of(jnp.broadcast_to(jnp.asarray(scale, jnp.float32), cotangent.shape))
        counted = (e_c > -1024) & (e_c < 128) & (e_f > -1024) & (e_f < 128)
        return counted & (e_c + e_f + self.raised < self.least)

    def moved(self, pairs, routed, before, excess, plain):
        """The exact rescue (`factored`): every pair's product `cotangent * scale` as fp32 forms it, from its
        operands' significands and binades, on a grid of `FACTORED_SLICE` bits a row (its token's, gathered,
        shared by every device that sums it), and the decoded matrix on its own; each slice pair's
        product summed by a bf16 GEMM whose fp32 sums are integers below 2**24, so exact
        (`FACTORED_WIDTH` columns a chunk), into int32 limbs, its routes' limbs added into their tokens and
        the devices' over the axes they sum, carried, then rounded once (`_rounded`). No product
        overflows, flushes or rounds before the one rounding, within a route, across its routes, or across
        its devices: what cancels in it leaves its exact remainder (rows 1464, 12392, 12403); it runs where
        the plain transpose cannot hold a row (`combined`'s reach). A product not finite (a cotangent,
        factor or coefficient not finite) is left out of the sums and counted by its class
        (`_nonfinite_terms`): a coordinate meeting a NaN or both infinities is NaN, else one infinity's
        sign."""
        gather, wide_operands, _ = routed
        cotangents = tuple(cotangent for cotangent, _ in pairs)
        factors = tuple(jnp.broadcast_to(scale, cotangent.shape) for cotangent, scale in pairs)
        products = [_products(cotangent, f) for cotangent, f in zip(cotangents, factors, strict=True)]
        # One grid a row for every pair's products, which meet in its one sum.
        grid, token_grid = self.grids(functools.reduce(jnp.maximum, [binade for _, _, binade, _ in products]),
                                      gather, tuple(sorted({*before, *excess})))
        matrix, *rest = self.prepared(*wide_operands)
        # The finite coefficients' grid: one not finite is IEEE's (`combined`), outside the exact sums.
        held = jnp.abs(matrix.astype(jnp.float32))
        held = jnp.where(jnp.isfinite(held), held, 0.0)
        matrix_top = jnp.max(jnp.where(held > 0, float_bits.binade_of(held), -1024), initial=-1024)
        matrix_grid = _edge(_shared_max(matrix_top, tuple(sorted(jax.typeof(matrix_top).mat.varying
                                                                   - self.held))))
        depth = self.shape[-1]
        columns_a_chunk = self.column_chunk(cotangents, factors, gather, matrix, rest)
        chunk = self.depth_chunk(matrix)

        def pulled_chunk(at, out):
            columns = jax.lax.dynamic_slice_in_dim(matrix, at * chunk, chunk, axis=-2)
            # Placed as the GEMMs' sums are: rows the cotangents' and factors', the mesh any operand's.
            rows, read = (*cotangents, *factors), (*cotangents, *factors, matrix)
            limbs = varying_like(_limbs(FACTORED_LIMBS, rows, chunk, read), *cotangents, columns,
                                 *jax.tree.leaves(rest), grid)
            # Whether a coordinate sums a product of each class not finite (`_nonfinite_terms`).
            classes = varying_like(_limbs(3, rows, chunk, read), *cotangents, *factors, columns,
                                   *jax.tree.leaves(rest))

            def width_chunk(index, carry):
                limbs, classes = carry
                start = index * columns_a_chunk

                def cut(value):
                    return jax.lax.dynamic_slice_in_dim(value, start, columns_a_chunk, axis=-1)

                block = cut(columns)
                for cotangent, f in zip(cotangents, factors, strict=True):
                    classes = self.classified(classes, cut(cotangent), cut(f), block, rest)
                block = jnp.where(jnp.isfinite(block), block, jnp.zeros_like(block))
                block_abs = jnp.abs(block.astype(jnp.float32))
                block_binade = jnp.where(block_abs > 0, float_bits.binade_of(block_abs), -1024)
                block_bits = (block_abs * float_bits.exact_power2(jnp.clip(7 - block_binade, -126, 127))
                              ).astype(jnp.int32)
                block_negative = block < 0
                # Each pair's products sliced against the one block's coefficients, into the same limbs.
                for high, low, binade, negative in products:
                    limbs = pair_slices(limbs, cut(high), cut(low), cut(binade), cut(negative), block_bits,
                                        block_binade, block_negative)
                return limbs, classes

            def pair_slices(limbs, row_high, row_low, row_binade, row_negative, block_bits, block_binade,
                            block_negative):
                def row_slice(i, limbs):
                    # The exact product's digit: its halves' bits are disjoint, so their pieces add to it.
                    a = (_slice(row_high, row_binade - 23, grid, i, row_negative)
                         + _slice(row_low, row_binade - 47, grid, i, row_negative))

                    def taken(limbs):
                        def matrix_slice(j, limbs):
                            b = _slice(block_bits, block_binade - 7, matrix_grid, j, block_negative)

                            def multiplied(limbs):
                                t = FACTORED_HEADROOM + i + j
                                at_t = jax.lax.dynamic_index_in_dim(limbs, t, 0, keepdims=False)
                                # Placed as the limb it adds into (the GEMM's own placement its operands').
                                summed = _placed_like(self.wide(a, b, *rest).astype(jnp.int32), at_t)
                                return jax.lax.dynamic_update_index_in_dim(limbs, at_t + summed, t, 0)

                            return jax.lax.cond(jnp.any(b != 0), multiplied, lambda limbs: limbs, limbs)

                        return jax.lax.fori_loop(0, FACTORED_MATRIX_SLICES, matrix_slice, limbs)

                    return jax.lax.cond(jnp.any(a != 0), taken, lambda limbs: limbs, limbs)

                return _carried(jax.lax.fori_loop(0, FACTORED_ROW_SLICES, row_slice, limbs))

            limbs, classes = jax.lax.fori_loop(0, cotangents[0].shape[-1] // columns_a_chunk, width_chunk,
                                               (limbs, classes))
            # In fp32 from here, each 0 or 1: a sum over routes and devices is positive where any is.
            classes = classes.astype(jnp.float32)
            if gather is not None:
                routed_limbs = varying_like(_limbs(FACTORED_LIMBS, (plain,), chunk, (plain, limbs)), limbs,
                                            gather)
                limbs = _routed(routed_limbs, gather, limbs)
                tokens = varying_like(_limbs(3, (plain,), chunk, (plain, classes)).astype(jnp.float32),
                                      classes, gather)
                classes = tokens.at[:, gather].add(classes)
            over = tuple(sorted(jax.typeof(limbs).mat.varying - self.held))
            if over:
                limbs = _carried(jax.lax.psum(limbs, over))
            if counted := tuple(sorted(jax.typeof(classes).mat.varying - self.held)):
                classes = jax.lax.psum(classes, counted)
            # The first limb's unit: 2**(grid + matrix grid - 2 slices + the headroom's slices), times the
            # owed power of two, which moves the unit alone.
            summed = _rounded(limbs, token_grid + matrix_grid + FACTORED_SLICE * (FACTORED_HEADROOM - 2)
                              + self.raised)
            # Its chunk placed as the gradient it is written into (a replicated one's mesh too).
            written = _placed_like(varying_like(_ieee(classes, summed), out), out)
            return jax.lax.dynamic_update_slice_in_dim(out, written, at * chunk, axis=-1)

        # The rows' gradient, typed and placed as the plain transpose's is (its devices' sums taken, as each
        # chunk's are; on its Explicit mesh where it has one, replicated too), so the loop's carry and the
        # branch it returns from keep its type.
        zeros = placed_zeros(self.shape, jnp.float32, explicit_placement(jax.typeof(plain)), plain)
        out = varying_like(zeros, plain)
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

    def depth_chunk(self, matrix) -> int:
        """The depth a chunk the rescue pulls back at once, whole on every device an Explicit axis splits the
        matrix's depth (its axis -2) or the rows' (their last, the gradient's) over: a multiple of their
        splits' count dividing the depth, at most `FACTORED_DEPTH` where that holds one (train 1455's review:
        64 of a depth of 768 split over three devices was unplaceable)."""
        split = math.lcm(_split(explicit_spec(matrix), -2), _split(self.placed, -1))
        depth = self.shape[-1]
        most = max(FACTORED_DEPTH // split, 1)
        return split * max(d for d in range(1, most + 1) if (depth // split) % d == 0)

    def column_chunk(self, cotangents, factors, gather, matrix, rest) -> int:
        """The cotangents' columns a chunk the rescue's GEMMs sum at once, whole on every device an Explicit
        axis splits any operand it cuts by its columns over (cotangents, factors, the matrix): a multiple of
        their splits' count, at most `FACTORED_WIDTH`, so its GEMMs' sums stay exact (`bounded` refusing a
        split past it; train 1426's review: 3073 of 6146 columns split over two devices was unplaceable)."""
        split = math.lcm(*(_split(explicit_spec(each), -1) for each in (*cotangents, *factors, matrix)))
        self.bounded(cotangents, factors, gather, matrix, rest, split)
        width = cotangents[0].shape[-1]
        return split * max(d for d in range(1, FACTORED_WIDTH // split + 1) if (width // split) % d == 0)

    def bounded(self, cotangents, factors, gather, matrix, rest, split: int) -> None:
        """Refuse a rescue whose sums its int32 limbs cannot hold: more devices summing its digits than
        `FACTORED_FAN_IN`, or more products (columns, routes, devices) than `FACTORED_REACH`. Its devices
        are every axis a contributing operand varies over the rows do not, of which each psum sums those
        its own sums vary over."""
        leaves = jax.tree.leaves((cotangents, factors, gather, matrix, rest))
        varying = frozenset().union(*(jax.typeof(leaf).mat.varying for leaf in leaves))
        devices = math.prod(jax.lax.axis_size(axis) for axis in sorted(varying - self.held))
        if devices > FACTORED_FAN_IN:
            raise ValueError(f"the exact rescue sums {devices} devices' digits, past the {FACTORED_FAN_IN} "
                             f"whose sum int32 holds")
        if split > FACTORED_WIDTH:
            raise ValueError(f"the exact rescue's column chunks, whole on each of the {split} devices that "
                             f"split its columns, pass the {FACTORED_WIDTH} whose GEMM sums fp32 holds "
                             "exactly")
        # Each pair's products a column of the sum (`combined`).
        width = cotangents[0].shape[-1] * len(cotangents)
        if width * (1 if gather is None else gather.shape[0]) * devices >= FACTORED_REACH:
            raise ValueError(f"the exact rescue's sums of {width} columns, {cotangents[0].shape[0]} routes "
                             f"and {devices} devices could pass its leading limb's int32 ({FACTORED_REACH})")

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
                held = jax.lax.dynamic_index_in_dim(classes, index, 0, keepdims=False)
                # Placed as the class it updates, as a limb's sum is (train 1455's review: a depth split
                # left the counts split where the classes were whole).
                counts = _placed_like(self.wide(rows.astype(jnp.bfloat16), matrix.astype(jnp.bfloat16),
                                                *rest), held)
                classes = jax.lax.dynamic_update_index_in_dim(classes, _present(held, counts), index, 0)
            return classes

        special = functools.reduce(jnp.logical_or, [jnp.any(~jnp.isfinite(v))
                                                    for v in (cotangent, factor, coefficients)])
        return jax.lax.cond(special, counted, lambda classes: classes, classes)

    def _paired_tangent(self, ran, routed, primals, tangents):
        """The pair sum's derivative: bilinear in each pair's cotangent and scale, each perturbed operand's
        term (a held one's none: an infinite scale meets no cotangent of its own) a pair of one exact sum
        (`paired` again, so terms that cancel keep their remainder and every order's derivative is
        declared: train 1409's review, 2**126 and 2**125 at scales 4 and 8 and 1 at scale 1 read NaN; a
        cotangent's and a scale's terms each infinite read NaN), as a linear map of the moving tangents
        whose residuals carry every array it reads and whose transpose is the product's own (`adjoint`),
        so reverse mode over it takes no branch of the moving tangents'."""
        cotangents, scales = primals
        value = self.paired(ran, routed, cotangents, scales)
        # Each live tangent's slot: a pair whose cotangent moves (its scale held), or whose scale does.
        slots = [(moves_cotangent, index)
                 for index, pair in enumerate(zip(*tangents, strict=True))
                 for moves_cotangent, tangent in zip((True, False), pair, strict=True)
                 if not isinstance(tangent, SymbolicZero)]
        if not slots:
            return value, jnp.zeros_like(value)
        moving = tuple(tangents[0 if cotangent else 1][index] for cotangent, index in slots)

        def apply(residuals, moving):
            ran, routed, cotangents, scales = residuals
            pairs = [(each, scales[index]) if cotangent else (cotangents[index], each)
                     for (cotangent, index), each in zip(slots, moving, strict=True)]
            return self.paired(ran, routed, *map(tuple, zip(*pairs, strict=True)))

        def adjoint(residuals, rows):
            # The rows' map's transpose: their product (`factored`'s apply) times the held operand of each
            # slot's pair, summed to the moving one's shape and typed and placed as it is.
            ran, _, cotangents, scales = residuals
            product = self.run(rows.astype(jnp.bfloat16), *ran).astype(jnp.float32)
            terms = []
            for cotangent, index in slots:
                held, like = (scales[index], cotangents[index]) if cotangent else (cotangents[index],
                                                                                    scales[index])
                terms.append(_placed_like(unbroadcast(owed_exactly(product, held, self.raised),
                                                      jnp.shape(like)), like))
            return tuple(terms)

        # Its joint derivative in the moving tangents and the residuals the declared pair sum's own
        # (`traced_tangent`: terms that cancel across them meet in one sum), its type the primal's.
        return value, generation.linear_map(apply, adjoint, (ran, routed, cotangents, scales), moving,
                                            out_aval=jax.typeof(value).to_tangent_aval(), traced_tangent=True)


def _placed_like(value: jax.Array, like: jax.Array) -> jax.Array:
    """`value` typed as `like` is (a cotangent of a broadcast product of it, a chunk of the gradient it is):
    summed over each manual axis it varies over that `like` does not (`sharding.summed_to`), and placed as
    `like`'s type places it on an Explicit mesh, a replicated one's too (`sharding.explicit_placement`),
    by its sharding itself, so no mesh context need hold it."""
    return _placed(summed_to(value, like), jax.typeof(like).sharding)


def _placed(value: jax.Array, sharding: jax.sharding.NamedSharding, anchor=None) -> jax.Array:
    """`value` placed at `sharding` (a type's), resharded under that sharding's own abstract mesh, so its
    placement holds outside any mesh context (train 1426's review: one made at a type's sharding outside the
    context lost it), or run eagerly (`value` and `anchor` concrete arrays), put on `anchor`'s own devices;
    where `sharding` has no Explicit mesh and `value` does, whole on its own (a meshless cotangent's
    operand: as JAX's type check reads either)."""
    held = jax.typeof(value).sharding
    if sharding.mesh.explicit_axes:
        if held == sharding:
            return value
        # A concrete value (an uncommitted one's single device too) beside a concrete anchor: run eagerly.
        mesh = concrete_mesh(anchor)
        if not isinstance(value, Tracer) and mesh is not None:
            return jax.device_put(value, jax.sharding.NamedSharding(mesh, sharding.spec))
        with jax.sharding.use_abstract_mesh(sharding.mesh.abstract_mesh):
            return jax.sharding.reshard(value, sharding.spec)
    if held.mesh.explicit_axes and any(entry is not None for entry in held.spec):
        with jax.sharding.use_abstract_mesh(held.mesh.abstract_mesh):
            return jax.sharding.reshard(value, jax.sharding.PartitionSpec())
    return value



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

FACTORED_ROW_SLICES = 101
"""A row's slices: its products' binades span at most a cotangent's and a factor's ranges multiplied (-297
to 255, the largest product's grid rounded up a slice) and the 48 bits of their significands' exact
product below them (`_products`)."""

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

FACTORED_LIMBS = FACTORED_HEADROOM + FACTORED_ROW_SLICES + FACTORED_MATRIX_SLICES - 1
"""The exact rescue's int32 limbs a sum: its products' (a row slice's by a matrix slice's), below
`FACTORED_HEADROOM` limbs over them that their sums carry into."""


def _split(placed: jax.sharding.NamedSharding | None, axis: int) -> int:
    """The devices an Explicit axis splits an array placed at `placed` (`explicit_spec`'s) over along `axis`:
    1 where none does."""
    if placed is None or len(placed.spec) < -axis or placed.spec[axis] is None:
        return 1
    entry = placed.spec[axis]
    return math.prod(placed.mesh.shape[name] for name in ((entry,) if isinstance(entry, str) else entry))


def _limbs(count: int, operands: tuple, columns: int, read: tuple) -> jax.Array:
    """`count` int32 limbs of zeros, each the `operands`' rows by `columns`, placed as the products the sums
    `read` make are (a chunk's sums, which they add into): on the Explicit mesh of any array they read (a
    replicated one's, or a matrix split by the columns its GEMMs contract, too), their rows as the first of
    `operands` an Explicit axis splits by its rows places them, as JAX places an elementwise product of a
    split and an unsplit operand, else whole (train 1409's review: placed by one operand alone, an
    unplaced cotangent's limbs met a split factor's products, row 13986; train 1426's: meshless beside a
    column split's sums)."""
    placements = [explicit_placement(jax.typeof(each)) for each in read]
    meshes = [placed.mesh for placed in placements if placed is not None]
    specs = [explicit_spec(each) for each in operands]
    rows = next((placed.spec[0] for placed in specs if placed is not None and placed.spec[0] is not None),
                None)
    spec = None if not meshes else jax.sharding.NamedSharding(
        meshes[0], jax.sharding.PartitionSpec(None, rows, None))
    return placed_zeros((count, operands[0].shape[0], columns), jnp.int32, spec, *operands, *read)


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


def _products(cotangent: jax.Array, factor: jax.Array) -> tuple[jax.Array, ...]:
    """Each product `cotangent * factor` exactly, from its operands' bits (a subnormal's kept where a process
    flushes subnormal operands): its two significands' 48-bit product as two 24-bit integers, the high one
    at `2**23` of the product's binade over [1, 4) and the low one 24 binades below it, that binade (-1024
    where the product is zero or not finite) and its sign. No bit of it rounds (train 1426's review: a
    significand product rounded in fp32 lost the residual two cancelling products leave, 2**-46 of
    them)."""
    def parts(x):
        sign, biased, fraction, normalized, exponent = float_bits.float_bits(x.astype(jnp.float32))
        held = ((biased != 0) | (fraction != 0)) & (biased != 0xFF)
        return (1 << 23) | normalized, exponent, sign != 0, held

    (m_c, e_c, s_c, c_held), (m_f, e_f, s_f, f_held) = parts(cotangent), parts(factor)
    counted = c_held & f_held
    high, low = float_bits.significand_product(m_c, m_f)
    return (jnp.where(counted, high, 0), jnp.where(counted, low, 0),
            jnp.where(counted, e_c + e_f + 1, -1024), s_c != s_f)


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
