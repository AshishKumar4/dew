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
from dew.nn.aligned import aligned_segment_sum
from dew.nn.derivatives import KernelGlue, contracted, declared_tangent
from dew.nn.kernels import generation
from dew.nn.precision import power2
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
        """The banded rescue (`factored`): every product `cotangent * scale` from its significands and
        binades, each band of it moved by its row's (token's) power to fp32's middle and pulled back wide, a
        chunk of the depth at a time, and every band's rows summed into their segments (tokens, gathered;
        each row its own, dense) at their powers, over every device, exactly and rounded once: products
        that cancel across a band's seam or a route's leave their remainder."""
        gather, wide_operands = routed
        f = jnp.broadcast_to(scale, cotangent.shape)
        e_c, e_f = (float_bits.binade_of(v) for v in (cotangent, f))
        counted = (e_c > -1024) & (e_c < 128) & (e_f > -1024) & (e_f < 128)
        binade = jnp.where(counted, e_c + e_f, -1024)
        # Each product's significand from its operands' bits, in [1, 4): a subnormal's kept where a process
        # flushes subnormal operands.
        significands = jnp.where(counted, _significand(cotangent) * _significand(f), 0.0)
        top = jnp.max(binade, axis=-1, keepdims=True, initial=-1024)
        top = jax.lax.pmax(top, before) if before else top
        if gather is None:
            top = jax.lax.pmax(top, excess) if excess else top
            rows_top = sums_top = top
        else:
            # A token's largest, every route of it on every device: its bands' edges, shared by its routes.
            token_top = varying_like(jnp.full((self.shape[0], 1), -1024, top.dtype), top, gather)
            token_top = token_top.at[gather].max(top)
            axes = tuple(sorted(jax.typeof(token_top).mat.varying - self.held))
            sums_top = jax.lax.pmax(token_top, axes) if axes else token_top
            rows_top = sums_top[gather]
        matrix, *rest = self.prepared(*wide_operands)
        depth = self.shape[-1]
        chunk = math.gcd(depth, FACTORED_CHUNK)
        # Each band's highest binade, moved to 2**0, and its products in (2**-FACTORED_BAND, 4), formed once a
        # chunk of the depth (no band of the whole cotangent held at once).
        highs = [rows_top - FACTORED_BAND * index for index in range(FACTORED_BANDS)]
        # Each band's power as three factors each a normal float's, a row's.
        factors = []
        for high in highs:
            parts, left = [], high
            for _ in range(3):
                parts.append(jnp.clip(left, -126, 127))
                left = left - parts[-1]
            factors.append([float_bits.exact_power2(part) for part in parts])
        stacked = jnp.stack(highs)

        def pulled_chunk(at, out):
            # Every band's pullback over this chunk of the depth (`lax.map`, placed as the partials are), its
            # bands' rows summed into their segments (tokens, gathered; each row its own, dense) over every
            # device exactly and rounded once (`aligned_segment_sum`).
            columns = jax.lax.dynamic_slice_in_dim(matrix, at * chunk, chunk, axis=-2)

            def band(high):
                held = (binade <= high) & (binade > high - FACTORED_BAND)
                near = jnp.where(held, power2(significands, binade - high), 0.0)
                near = jax.lax.psum(near, before) if before else near
                return self.wide(near, columns, *rest).astype(jnp.float32)

            partials = jax.lax.map(band, stacked)
            terms = [(partials[index], *factors[index]) for index in range(FACTORED_BANDS)]
            axes = tuple(sorted(frozenset().union(
                *(jax.typeof(v).mat.varying for v in (partials, *(() if gather is None else (gather,)))))
                - self.held))
            summed = aligned_segment_sum(terms, gather, self.shape[0], axes)
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


FACTORED_CHUNK = 64
"""The depth `factored`'s moved transpose pulls back a chunk at a time (a divisor of the rows' depth): its
every band's partials of a chunk held at once, summed exactly, so its scratch is bands x rows x chunk."""

FACTORED_BANDS = 8
"""The bands `factored`'s moved transpose splits a row's products into, `FACTORED_BAND` binades each down from
its largest: together past a cotangent's and a factor's ranges multiplied (binades -298 to 254), so every
product sits in one."""

FACTORED_BAND = 70
"""A band's binades (`FACTORED_BANDS`): its products moved into (2**-70, 4), so times a decoded matrix's
least value (an NVFP4 entry's 2**-18) none falls below fp32's least normal, and summed over any width a
matrix holds none passes its largest."""



def _significand(x: jax.Array) -> jax.Array:
    """A finite fp32 `x`'s significand with its sign, in [1, 2), from its bits (`float_bits`): a subnormal's
    normalized, read where a process flushes subnormal operands; 0 for zero."""
    sign, biased, fraction, normalized, _ = float_bits.float_bits(x.astype(jnp.float32))
    one = sign | (127 << 23) | normalized
    return jnp.where((biased == 0) & (fraction == 0), 0.0, jax.lax.bitcast_convert_type(one, jnp.float32))
