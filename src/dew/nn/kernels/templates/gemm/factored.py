"""A packed matrix's product times its scales, `run(rows) * factor` (`factored`), its transpose range-safe:
the fp8 and ternary products' (`gemm.fp8_weight_product`, `ternary.ternary_weight_product`), ledger 1464."""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from dew.nn import float_bits
from dew.nn.derivatives import KernelGlue, contracted, declared_tangent
from dew.nn.kernels import generation
from dew.nn.precision import power2
from dew.nn.sharding import explicit_spec, varying_like


def factored(run: Callable, operands: tuple, factor: jax.Array, rows: jax.Array, *, gathered: bool = False
             ) -> jax.Array:
    """`run(rows as bf16, *operands) * factor`, linear in `rows`: `run` a product of bf16 rows, each output
    row its own input row's (no gather), and `factor` a packed matrix's fp32 scales broadcast against its
    output (`formats.fp8_factor`). Its transpose is the plain one, the cotangent times its factor through
    `run`'s pullback, but for a factor past 1 that every column of a row shares (a tensor's scale, a group's
    under a grouped call): its power of two leaves the cotangent's product (`precision.power2`) and joins
    after the pullback's sum, so a scale of 2**126 under cotangents that cancel pulls back 0, not inf - inf
    (ledger 1464), each value otherwise the plain transpose's, rounded where it rounds; a replicated factor
    joins after a checked `shard_map`'s devices sum their partials. A factor of each column, which the
    transpose sums over, stays with the cotangent, as the plain transpose has it. A `gathered` product's
    output rows are rows of `rows` read by a table (`gemm`'s `gather`), which the pullback sums back into
    them: its rescue moves the whole cotangent by one power, not each row by its own."""
    if jnp.ndim(factor) and jnp.shape(factor)[-1] != 1:
        return run(rows.astype(jnp.bfloat16), *operands) * factor
    shape, dtype = rows.shape, rows.dtype
    held = frozenset(jax.typeof(rows).mat.varying)
    placed = explicit_spec(rows)

    def apply(residuals, moving):
        scale, ran = residuals
        return run(moving.astype(jnp.bfloat16), *ran) * scale

    def operand_axes(ran) -> frozenset:
        # The manual axes the product's own operands, its rows and its matrix's, vary over.
        return held.union(*(jax.typeof(leaf).mat.varying for leaf in jax.tree.leaves(ran)))

    def alone(ran, *values) -> tuple:
        # The axes `values` (a cotangent, a scale) vary over and the product's operands do not: there each
        # device holds the same product, and its cotangent is summed before the pullback, as the plain
        # transpose sums the cotangent of a product every device holds alike.
        return tuple(sorted(frozenset().union(*(jax.typeof(value).mat.varying for value in values))
                            - operand_axes(ran)))

    def pulled(near, ran):
        before = alone(ran, near)
        near = jax.lax.psum(near, before) if before else near
        # The pullback at zeros typed as the rows are: placed, and varying as the cotangent's partials do.
        at = varying_like(jnp.zeros(shape, jnp.bfloat16, out_sharding=placed), near, *jax.tree.leaves(ran))
        (partial,) = jax.vjp(lambda each: run(each, *ran), at)[1](near)
        partial = partial.astype(jnp.float32)
        return partial, tuple(sorted(jax.typeof(partial).mat.varying - held))

    def summed(partial, excess):
        return jax.lax.psum(partial, excess) if excess else partial

    def gathered_rescue(cotangent, scale, ran, before, excess):
        # A gathered product's rows sum back into the rows its table read, so one power moves its whole
        # cotangent: each entry's product from its cotangent's and factor's significands (the cotangent
        # halved where its own binade would carry their product past fp32's largest) and powers, moved so the
        # largest product's binade is `FACTORED_RESCUE`, then back after the sum.
        f = jnp.broadcast_to(scale, cotangent.shape)
        e_c, e_f = float_bits.binade_of(cotangent), float_bits.binade_of(f)
        counted = (e_c > -1024) & (e_c < 128) & (e_f > -1024) & (e_f < 128)
        top = jnp.max(jnp.where(counted, e_c + e_f, -1024), initial=-1024)
        for axes in (before, excess):
            top = jax.lax.pmax(top, axes) if axes else top
        lift = jnp.where(top > -1024, FACTORED_RESCUE - top, 0)
        e = jnp.clip(e_f, -126, 127)
        halved = e_c >= 126
        near = power2((cotangent * jnp.where(halved, 0.5, 1.0)) * power2(f, -e), e + halved + lift)
        partial, excess = pulled(near, ran)
        return power2(summed(partial, excess), -lift)

    def transposed_value(cotangent, scale, ran):
        # The plain transpose, the cotangent times its factor pulled back; where an entry of it is past
        # fp32's largest, the factor past 1 as its significand in the cotangent and its power of two
        # (`precision.power2`) after the sum (a scale of a row, `[rows, 1]`, or of all, `[1]` or `[]`,
        # broadcasting against the rows' gradient alike), after every device's sum where it is replicated.
        plain = summed(*pulled(cotangent * scale, ran))

        def moved():
            # Summed before its pullback where the operands are every device's, at one shared power.
            before = alone(ran, cotangent, scale)
            # The axes its partials will vary over that the rows do not, the operands', from types alone.
            excess = tuple(sorted(operand_axes(ran) - held))
            if gathered:
                return jnp.where(jnp.isfinite(plain), plain, gathered_rescue(cotangent, scale, ran, before,
                                                                             excess))
            binade = float_bits.binade_of(scale)
            power = jnp.where((jnp.abs(scale) > 1) & (binade < 128), binade, 0)
            power = jax.lax.pmax(power, before) if before else power
            near = cotangent * power2(scale, -power)
            near = jax.lax.psum(near, before) if before else near
            # Each row moved so its largest term's binade is `FACTORED_RESCUE`, shared by every device of a
            # sum: what cancels in the pullback's sum leaves a remainder its bf16 rows still hold.
            e = float_bits.binade_of(near)
            top = jnp.max(jnp.where((e > -1024) & (e < 128), e, -1024), axis=-1, keepdims=True,
                          initial=-1024)
            top = jax.lax.pmax(top, excess) if excess else top
            lift = jnp.where(top > -1024, FACTORED_RESCUE - top, 0)
            partial, excess = pulled(power2(near, lift), ran)
            if excess and not jax.typeof(power).mat.varying & set(excess):
                rescued = power2(summed(partial, excess), power - lift)
            else:
                rescued = summed(power2(partial, power - lift), excess)
            return jnp.where(jnp.isfinite(plain), plain, rescued)

        # The moved pullback where the plain one is not finite alone: each device of a sum decides alike,
        # the plain transpose summed over its axes.
        return jax.lax.cond(jnp.all(jnp.isfinite(plain)), lambda: plain, moved)

    def transposed_tangent(ran, primals, tangents):
        # Its derivatives the plain transpose's, `cotangent * scale` pulled back, every term of a perturbed
        # operand alone (a scale held still meets no cotangent of its own: an infinite one is no NaN).
        (cotangent, scale), (d_cotangent, d_scale) = primals, tangents
        terms = [*(() if isinstance(d_cotangent, SymbolicZero) else (d_cotangent * scale,)),
                 *(() if isinstance(d_scale, SymbolicZero) else (cotangent * d_scale,))]
        value = transposed(cotangent, scale, ran)
        if not terms:
            return value, jnp.zeros_like(value)
        return value, summed(*pulled(functools.reduce(jnp.add, terms), ran))

    transposed = contracted("test_a_ternary_weight_product_holds_every_order")(
        declared_tangent(KernelGlue(), transposed_value, transposed_tangent, nondiff_argnums=(2,),
                         symbolic_zeros=True))

    def pullback(residuals, cotangent):
        scale, ran = residuals
        return transposed(cotangent.astype(jnp.float32), scale.astype(jnp.float32), ran).astype(dtype)

    return generation.linear_map(apply, pullback, (factor, operands), rows)


FACTORED_RESCUE = 100
"""The binade `factored`'s transpose moves a row's largest term to where its plain transpose overflowed: its
sum over any width a decoded matrix holds (an FP8 entry at most 448) within fp32, and a remainder 2**-233 of
it still a bf16 value."""
