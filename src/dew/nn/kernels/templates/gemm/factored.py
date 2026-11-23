"""A packed matrix's product times its scales, `run(rows) * factor` (`factored`), its transpose range-safe:
the fp8 and ternary products' (`gemm.fp8_weight_product`, `ternary.ternary_weight_product`), ledger 1464."""

from __future__ import annotations

import dataclasses
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


def factored(run: Callable, operands: tuple, factor: jax.Array, rows: jax.Array) -> jax.Array:
    """`run(rows as bf16, *operands) * factor`, linear in `rows`: `run` a product of bf16 rows, each output
    row its own input row's (no gather), and `factor` a packed matrix's fp32 scales broadcast against its
    output (`formats.fp8_factor`). Its transpose is the plain one, the cotangent times its factor through
    `run`'s pullback, but for a factor past 1 that every column of a row shares (a tensor's scale, a group's
    under a grouped call): its power of two leaves the cotangent's product (`precision.power2`) and joins
    after the pullback's sum, so a scale of 2**126 under cotangents that cancel pulls back 0, not inf - inf
    (ledger 1464), each value otherwise the plain transpose's, rounded where it rounds; a replicated factor
    joins after a checked `shard_map`'s devices sum their partials. A factor of each column, which the
    transpose sums over, stays with the cotangent, as the plain transpose has it."""
    if jnp.ndim(factor) and jnp.shape(factor)[-1] != 1:
        return run(rows.astype(jnp.bfloat16), *operands) * factor
    transpose = _Transpose(run, rows.shape, rows.dtype, frozenset(jax.typeof(rows).mat.varying),
                           explicit_spec(rows))

    def apply(residuals, moving):
        scale, ran = residuals
        return run(moving.astype(jnp.bfloat16), *ran) * scale

    def pullback(residuals, cotangent):
        scale, ran = residuals
        return transpose.declared(cotangent.astype(jnp.float32), scale.astype(jnp.float32), ran
                                  ).astype(rows.dtype)

    return generation.linear_map(apply, pullback, (factor, operands), rows)


@dataclasses.dataclass(frozen=True)
class _Transpose:
    """`factored`'s transpose of `run` at rows of `shape`, typed as the rows are (`held`, their manual axes;
    `placed`, their explicit placement)."""
    run: Callable
    shape: tuple
    dtype: jnp.dtype
    held: frozenset
    placed: object

    @functools.cached_property
    def declared(self) -> Callable:
        """`value`, its derivative the plain transpose's (`tangent`)."""
        return contracted("test_a_ternary_weight_product_holds_every_order")(
            declared_tangent(KernelGlue(), self.value, self.tangent, nondiff_argnums=(2,),
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
        """`near` pulled back through `run`, summed first over the axes it alone varies over, and the axes
        the partial varies over and the rows do not (each device's to sum after)."""
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

    def value(self, cotangent, scale, ran):
        """The plain transpose, the cotangent times its factor pulled back; where an entry of it is not
        finite, the moved one's (`rows_moved`)."""
        plain = self.summed(*self.pulled(cotangent * scale, ran))

        def moved():
            # Summed before its pullback where the operands are every device's; the axes its partials will
            # vary over that the rows do not, the operands', from types alone.
            before = self.alone(ran, cotangent, scale)
            excess = tuple(sorted(self.operand_axes(ran) - self.held))
            rescued = self.rows_moved(cotangent, scale, ran, before, excess)
            return jnp.where(jnp.isfinite(plain), plain, rescued)

        # The moved pullback where the plain one is not finite alone: each device of a sum decides alike,
        # the plain transpose summed over its axes.
        return jax.lax.cond(jnp.all(jnp.isfinite(plain)), lambda: plain, moved)

    def rows_moved(self, cotangent, scale, ran, before, excess):
        """The transpose with a factor past 1 as its significand in the cotangent and its power of two
        (`precision.power2`) after the sum (a scale of a row, `[rows, 1]`, or of all, `[1]` or `[]`), after
        every device's sum where it is replicated; each row moved so its largest term's binade is
        `FACTORED_RESCUE`, shared by a sum's devices, so what cancels leaves a remainder its bf16 rows
        hold."""
        binade = float_bits.binade_of(scale)
        power = jnp.where((jnp.abs(scale) > 1) & (binade < 128), binade, 0)
        power = jax.lax.pmax(power, before) if before else power
        near = cotangent * power2(scale, -power)
        near = jax.lax.psum(near, before) if before else near
        e = float_bits.binade_of(near)
        top = jnp.max(jnp.where((e > -1024) & (e < 128), e, -1024), axis=-1, keepdims=True, initial=-1024)
        top = jax.lax.pmax(top, excess) if excess else top
        lift = jnp.where(top > -1024, FACTORED_RESCUE - top, 0)
        partial, excess = self.pulled(power2(near, lift), ran)
        if excess and not jax.typeof(power).mat.varying & set(excess):
            return power2(self.summed(partial, excess), power - lift)
        return self.summed(power2(partial, power - lift), excess)

    def tangent(self, ran, primals, tangents):
        """The plain transpose's derivative, `cotangent * scale` pulled back, every term of a perturbed
        operand alone (a scale held still meets no cotangent of its own: an infinite one is no NaN)."""
        (cotangent, scale), (d_cotangent, d_scale) = primals, tangents
        terms = [*(() if isinstance(d_cotangent, SymbolicZero) else (d_cotangent * scale,)),
                 *(() if isinstance(d_scale, SymbolicZero) else (cotangent * d_scale,))]
        value = self.declared(cotangent, scale, ran)
        if not terms:
            return value, jnp.zeros_like(value)
        return value, self.summed(*self.pulled(functools.reduce(jnp.add, terms), ran))


FACTORED_RESCUE = 100
"""The binade `factored`'s transpose moves a row's largest term to where its plain transpose overflowed: its
sum over any width a decoded matrix holds (an FP8 entry at most 448) within fp32, and a remainder 2**-233 of
it still a bf16 value."""
