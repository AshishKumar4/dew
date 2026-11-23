"""A packed matrix's product times its scales, `run(rows) * factor` (`factored`), its transpose range-safe:
the fp8 and ternary products' (`gemm.fp8_weight_product`, `ternary.ternary_weight_product`), ledger 1464."""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from dew.nn import float_bits
from dew.nn.aligned import aligned_sum
from dew.nn.derivatives import KernelGlue, contracted, declared_tangent
from dew.nn.kernels import generation
from dew.nn.sharding import explicit_spec, varying_like


def factored(run: Callable, operands: tuple, factor: jax.Array, rows: jax.Array, *, wide: Callable,
             wide_operands: tuple, prepared: Callable | None = None,
             gather: jax.Array | None = None) -> jax.Array:
    """`run(rows as bf16, *operands) * factor`, linear in `rows`: `run` a product of bf16 rows (those
    `gather` reads where given, `gemm`'s `gather`, among `operands`), and `factor` a packed matrix's fp32
    scales broadcast against its output (`formats.fp8_factor`), one a row's or one for all (a column's,
    which the transpose sums over, is the plain product's). Its transpose is the plain one, the cotangent
    times its factor through `run`'s pullback, wherever that is finite; where it is not (a product past
    fp32's largest), the moved one (`_Transpose.moved`): each product formed from its cotangent's and
    factor's significands and binades, split into `FACTORED_BANDS` bands of `FACTORED_BAND` binades down
    from its row's (its token's, gathered) largest, each band moved to fp32's middle and pulled back wide
    (`wide(cotangent, *prepared(*wide_operands))`, the transpose in fp32 over the rows `run` reads, a route
    each where gathered, its operands `prepared` once a rescue: a decoded matrix whose depth is its axis
    -2, a chunk of it at a time), and every band's rows summed into their tokens (gathered) at their
    powers exactly, rounded once (`aligned.aligned_segment_sum`), a product of a cotangent or factor not
    finite kept as IEEE's: no product overflows or flushes, so a scale of 2**126 under cotangents that
    cancel pulls back their remainder, as fp32's sums would in range (ledger 1464). A checked `shard_map`'s
    devices sum a cotangent only they vary over before its pullback, and their partials after."""
    if jnp.ndim(factor) and jnp.shape(factor)[-1] != 1:
        return run(rows.astype(jnp.bfloat16), *operands) * factor
    transpose = _Transpose(run, rows.shape, rows.dtype, frozenset(jax.typeof(rows).mat.varying),
                           explicit_spec(rows), wide, prepared or (lambda *each: each))

    def apply(residuals, moving):
        scale, ran, _ = residuals
        return run(moving.astype(jnp.bfloat16), *ran) * scale

    def pullback(residuals, cotangent):
        scale, ran, routed = residuals
        return transpose.declared(cotangent.astype(jnp.float32), scale.astype(jnp.float32), ran, routed
                                  ).astype(rows.dtype)

    return generation.linear_map(apply, pullback, (factor, operands, (gather, wide_operands)), rows)


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
        if 0 in self.shape:
            # No row or no depth: the plain transpose's empty or zero gradient is the whole of it.
            return plain

        def moved():
            # Summed before its pullback where the operands are every device's; the axes its partials will
            # vary over that the rows do not, the operands', from types alone.
            before = self.alone(ran, cotangent, scale)
            excess = tuple(sorted(self.operand_axes(ran) - self.held))
            rescued = self.moved(cotangent, scale, ran, routed, before, excess, plain)
            # A product of a cotangent or factor not finite pulled back as IEEE's, as the plain transpose's.
            special = ~jnp.isfinite(cotangent) | ~jnp.isfinite(jnp.broadcast_to(scale, cotangent.shape))
            ieee = self.summed(*self.pulled(jnp.where(special, cotangent * scale, 0.0), ran))
            rescued = jnp.where(jnp.isfinite(ieee), rescued, ieee)
            return jnp.where(jnp.isfinite(plain), plain, rescued)

        # The moved pullback where the plain one is not finite alone: each device of a sum decides alike,
        # the plain transpose summed over its axes.
        return jax.lax.cond(jnp.all(jnp.isfinite(plain)), lambda: plain, moved)

    def moved(self, cotangent, scale, ran, routed, before, excess, plain):
        """The exact rescue (`factored`): every product `cotangent * scale` as fp32 forms it, from its
        operands' significands and binades, on a grid of `FACTORED_SLICE` bits a row (its token's, gathered,
        shared by every device that sums it), and the decoded matrix on its own; each slice pair's
        product summed by a bf16 GEMM whose fp32 sums are integers below 2**24, so exact
        (`FACTORED_WIDTH` columns a chunk), into int32 limbs, its routes' limbs added into their tokens and
        the devices' over the axes they sum, carried, then rounded once (`aligned.aligned_sum`). No product
        overflows, flushes or rounds before the one rounding, within a route, across its routes, or across
        its devices: what cancels anywhere leaves its exact remainder (rows 1464, 12392, 12403)."""
        gather, wide_operands = routed
        f = jnp.broadcast_to(scale, cotangent.shape)
        e_c, e_f = (float_bits.binade_of(v) for v in (cotangent, f))
        counted = (e_c > -1024) & (e_c < 128) & (e_f > -1024) & (e_f < 128)
        # Each product's significand as fp32 forms it, from its operands' bits (a subnormal's kept where a
        # process flushes subnormal operands), normalized into [1, 2): 24 bits, an integer at 2**23.
        product = jnp.where(counted, _significand(cotangent) * _significand(f), 0.0)
        doubled = jnp.abs(product) >= 2
        product = jnp.where(doubled, product * 0.5, product)
        binade = jnp.where(counted, e_c + e_f + doubled.astype(e_c.dtype), -1024)
        bits = (jnp.abs(product) * 2.0 ** 23).astype(jnp.int32)
        negative = product < 0
        # A row's grid: its largest product's binade, a token's for its routes and every device's that sums
        # it, rounded up to a slice's edge.
        top = jnp.max(binade, axis=-1, keepdims=True, initial=-1024)
        if gather is None:
            grid = _shared_max(top, tuple(sorted({*before, *excess})))
        else:
            token = varying_like(jnp.full((self.shape[0], 1), -1024, top.dtype), top, gather)
            token = token.at[gather].max(top)
            token = _shared_max(token, tuple(sorted(jax.typeof(token).mat.varying - self.held)))
            grid = token[gather]
        grid = _edge(grid)
        matrix, *rest = self.prepared(*wide_operands)
        held = jnp.abs(matrix.astype(jnp.float32))
        matrix_top = jnp.max(jnp.where(held > 0, float_bits.binade_of(held), -1024), initial=-1024)
        matrix_grid = _edge(_shared_max(matrix_top, tuple(sorted(jax.typeof(matrix_top).mat.varying
                                                                   - self.held))))
        depth, width = self.shape[-1], cotangent.shape[-1]
        chunk = math.gcd(depth, FACTORED_DEPTH)
        columns_a_chunk = max(d for d in range(1, min(width, FACTORED_WIDTH) + 1) if width % d == 0)
        limbs_count = FACTORED_ROW_SLICES + FACTORED_MATRIX_SLICES - 1
        rows = cotangent.shape[0]

        def pulled_chunk(at, out):
            columns = jax.lax.dynamic_slice_in_dim(matrix, at * chunk, chunk, axis=-2)
            limbs = varying_like(jnp.zeros((limbs_count, rows, chunk), jnp.int32), cotangent, columns,
                                 *jax.tree.leaves(rest), grid)

            def width_chunk(index, limbs):
                start = index * columns_a_chunk
                row_bits, row_binade, row_negative = (
                    jax.lax.dynamic_slice_in_dim(v, start, columns_a_chunk, axis=-1)
                    for v in (bits, binade, negative))
                block = jax.lax.dynamic_slice_in_dim(columns, start, columns_a_chunk, axis=-1)
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
                                at_t = jax.lax.dynamic_index_in_dim(limbs, i + j, 0, keepdims=False)
                                return jax.lax.dynamic_update_index_in_dim(limbs, at_t + summed, i + j, 0)

                            return jax.lax.cond(jnp.any(b != 0), multiplied, lambda limbs: limbs, limbs)

                        return jax.lax.fori_loop(0, FACTORED_MATRIX_SLICES, matrix_slice, limbs)

                    return jax.lax.cond(jnp.any(a != 0), taken, lambda limbs: limbs, limbs)

                return _carried(jax.lax.fori_loop(0, FACTORED_ROW_SLICES, row_slice, limbs))

            limbs = jax.lax.fori_loop(0, width // columns_a_chunk, width_chunk, limbs)
            if gather is not None:
                limbs = varying_like(jnp.zeros((limbs_count, self.shape[0], chunk), jnp.int32), limbs,
                                     gather).at[:, gather].add(limbs)
            axes = tuple(sorted(jax.typeof(limbs).mat.varying - self.held))
            limbs = _carried(jax.lax.psum(limbs, axes) if axes else limbs)
            sums_grid = grid if gather is None else _edge(token)
            # Limb t's weight: 2**(grid + matrix grid - 2 slices - t slices), as factors each normal.
            terms = [(limbs[t].astype(jnp.float32),
                      *_powers(sums_grid + matrix_grid - FACTORED_SLICE * (t + 2)))
                     for t in range(limbs_count)]
            summed = aligned_sum(terms)
            return jax.lax.dynamic_update_slice_in_dim(out, varying_like(summed, out), at * chunk, axis=-1)

        # The rows' gradient, typed and placed as the plain transpose's is (its devices' sums taken, as each
        # chunk's are), so the loop's carry and the branch it returns from keep its type.
        out = varying_like(jnp.zeros(self.shape, jnp.float32, out_sharding=explicit_spec(plain)), plain)
        return jax.lax.fori_loop(0, depth // chunk, pulled_chunk, out)

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

FACTORED_ROW_SLICES = 97
"""A row's slices: its products' binades span at most a cotangent's and a factor's ranges multiplied (-298
to 254, the largest product's grid rounded up a slice) and 24 bits of significand below them."""

FACTORED_MATRIX_SLICES = 6
"""A decoded packed matrix's slices on its grid: 36 bits, past any format's span (E5M2's 31 binades and 3
bits; NVFP4's codes times E4M3 block scales; ternary's one)."""


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


def _powers(power: jax.Array) -> list[jax.Array]:
    """`2**power` as factors each a normal float's (`float_bits.exact_power2`), their product exact."""
    factors, left = [], power
    for _ in range(9):
        part = jnp.clip(left, -126, 127)
        factors.append(float_bits.exact_power2(part))
        left = left - part
    return factors


def _significand(x: jax.Array) -> jax.Array:
    """A finite fp32 `x`'s significand with its sign, in [1, 2), from its bits (`float_bits`): a subnormal's
    normalized, read where a process flushes subnormal operands; 0 for zero."""
    sign, biased, fraction, normalized, _ = float_bits.float_bits(x.astype(jnp.float32))
    one = sign | (127 << 23) | normalized
    return jnp.where((biased == 0) & (fraction == 0), 0.0, jax.lax.bitcast_convert_type(one, jnp.float32))
