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
from dew.nn.aligned import aligned_segment_sum
from dew.nn.derivatives import KernelGlue, contracted, declared_tangent
from dew.nn.kernels import generation
from dew.nn.precision import power2
from dew.nn.sharding import explicit_spec, varying_like


def factored(run: Callable, operands: tuple, factor: jax.Array, rows: jax.Array, *,
             routes: tuple = (), wide: Callable | None = None) -> jax.Array:
    """`run(rows as bf16, *operands) * factor`, linear in `rows`: `run` a product of bf16 rows, each output
    row its own input row's (no gather), and `factor` a packed matrix's fp32 scales broadcast against its
    output (`formats.fp8_factor`). Its transpose is the plain one, the cotangent times its factor through
    `run`'s pullback, but for a factor past 1 that every column of a row shares (a tensor's scale, a group's
    under a grouped call): its power of two leaves the cotangent's product (`precision.power2`) and joins
    after the pullback's sum, so a scale of 2**126 under cotangents that cancel pulls back 0, not inf - inf
    (ledger 1464), each value otherwise the plain transpose's, rounded where it rounds; a replicated factor
    joins after a checked `shard_map`'s devices sum their partials. A factor of each column, which the
    transpose sums over, stays with the cotangent, as the plain transpose has it. `routes`, where `run`
    reads `rows` through a gather table (`gemm`'s `gather`, among `operands`), is that table and the
    operands of `wide` (`(gather, wide_operands)`), `wide(cotangent, *wide_operands)` the routed rows'
    fp32 transpose of `run`'s product (no rounding to the rows' dtype): the plain transpose is `run`'s own,
    and the moved one pulls each route back wide at its token's power (its routes' largest, on every
    device), summed into its token in fp32 and over the devices whose routes differ, then moved back once
    and rounded once (a route whose products cancel past fp32's largest keeps other routes' small sum)."""
    if jnp.ndim(factor) and jnp.shape(factor)[-1] != 1:
        return run(rows.astype(jnp.bfloat16), *operands) * factor
    transpose = _Transpose(run, rows.shape, rows.dtype, frozenset(jax.typeof(rows).mat.varying),
                           explicit_spec(rows), wide)

    def apply(residuals, moving):
        scale, ran, _ = residuals
        return run(moving.astype(jnp.bfloat16), *ran) * scale

    def pullback(residuals, cotangent):
        scale, ran, routed = residuals
        return transpose.declared(cotangent.astype(jnp.float32), scale.astype(jnp.float32), ran, routed
                                  ).astype(rows.dtype)

    return generation.linear_map(apply, pullback, (factor, operands, routes), rows)


@dataclasses.dataclass(frozen=True)
class _Transpose:
    """`factored`'s transpose of `run` at rows of `shape`, typed as the rows are (`held`, their manual axes;
    `placed`, their explicit placement)."""
    run: Callable
    shape: tuple
    dtype: jnp.dtype
    held: frozenset
    placed: jax.sharding.NamedSharding | jax.sharding.PartitionSpec | None
    wide: Callable | None = None

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

    def pulled(self, near, ran, shape=None) -> tuple:
        """`near` pulled back through `run` (at rows of `shape`, the rows' where None), summed first over the
        axes it alone varies over, and the axes the partial varies over and the rows do not (each device's
        to sum after)."""
        before = self.alone(ran, near)
        near = jax.lax.psum(near, before) if before else near
        # The pullback at zeros typed as the rows are: placed, and varying as the cotangent's partials do.
        zeros = jnp.zeros(self.shape if shape is None else shape, jnp.bfloat16,
                          out_sharding=self.placed if shape is None else None)
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

        def moved():
            # Summed before its pullback where the operands are every device's; the axes its partials will
            # vary over that the rows do not, the operands', from types alone.
            before = self.alone(ran, cotangent, scale)
            excess = tuple(sorted(self.operand_axes(ran) - self.held))
            rescued = self.moved(cotangent, scale, ran, routed, before, excess)
            return jnp.where(jnp.isfinite(plain), plain, rescued)

        # The moved pullback where the plain one is not finite alone: each device of a sum decides alike,
        # the plain transpose summed over its axes.
        return jax.lax.cond(jnp.all(jnp.isfinite(plain)), lambda: plain, moved)

    def moved(self, cotangent, scale, ran, routed, before, excess):
        """The transpose with each row moved by a power of two (`precision.power2`) so its largest
        cotangent-times-factor product's binade is `FACTORED_RESCUE` (a row's power shared by a sum's
        devices), and moved back after the sum, so what cancels leaves a remainder its bf16 rows hold. Each
        product is formed from its cotangent's and factor's significands, exact powers of two apart from
        them, and their binades, so no product overflows or flushes before it is moved (a cotangent of
        3 * 2**126 times a factor of 3 is 9 * 2**126 moved, not inf)."""
        f = jnp.broadcast_to(scale, cotangent.shape)
        e_c, e_f = (float_bits.binade_of(v) for v in (cotangent, f))
        counted = (e_c > -1024) & (e_c < 128) & (e_f > -1024) & (e_f < 128)
        e_c, e_f = jnp.where(counted, e_c, 0), jnp.where(counted, e_f, 0)
        top = jnp.max(jnp.where(counted, e_c + e_f, -1024), axis=-1, keepdims=True, initial=-1024)
        top = jax.lax.pmax(top, before) if before else top
        if not routed:
            top = jax.lax.pmax(top, excess) if excess else top
        lift = jnp.where(top > -1024, FACTORED_RESCUE - top, 0)
        near = power2(power2(cotangent, -e_c) * power2(f, -e_f), e_c + e_f + lift)
        near = jax.lax.psum(near, before) if before else near
        if not routed:
            partial, excess = self.pulled(near, ran)
            return power2(self.summed(partial, excess), -lift)
        # Products the move takes below fp32's least normal, which it flushes or rounds.
        tiny = counted & (e_c + e_f + lift < jnp.finfo(jnp.float32).minexp)
        return self.routed_moved(near, lift, routed, tiny)

    def routed_moved(self, near, lift, routed, tiny):
        """A gathered product's rescue: each route pulled back wide at its own power (`lift`, so what cancels
        within it cancels there), then each moved back and summed into its token, and over the devices that
        route it, exactly, rounded once (`aligned.aligned_segment_sum`): routes of +-2**140 from two devices
        sum to 0 beside a 2**-38 one, and 2**128 - 2**105 to its finite value. Within a route the pullback
        sums in fp32: where its products past `FACTORED_WINDOW` binades below the route's largest reach an
        entry whose larger products cancel to below `2**-FACTORED_WINDOW` of their magnitude, the entry is
        NaN, as the plain transpose's is, not a sum that dropped them (ledger row of the exact rescue)."""
        assert self.wide is not None, "a routed product's rescue pulls its routes back wide"
        gather, wide_operands = routed
        partial = self.wide(near, *wide_operands).astype(jnp.float32)
        # Each route's entries past the window below its largest, moved to fp32's least normal where their
        # move flushed them: what they reach, and what the rest reach, bounded through the matrices'
        # magnitudes.
        scale = float_bits.binade_of(near)
        small = ((near != 0) & (scale < FACTORED_RESCUE - FACTORED_WINDOW)) | tiny
        held = jnp.where(small, jnp.maximum(jnp.abs(near), jnp.finfo(jnp.float32).tiny), 0.0)
        low = self.wide(held, *wide_operands, magnitude=True).astype(jnp.float32)
        high = self.wide(jnp.where(small, 0.0, jnp.abs(near)), *wide_operands, magnitude=True)
        unsure = (low > 0) & (jnp.abs(partial) < high.astype(jnp.float32) * 2.0 ** -FACTORED_WINDOW)
        partial = jnp.where(unsure, jnp.nan, partial)
        # The power each route moves back by, as three factors each a normal float's.
        powers, rest = [], -lift
        for _ in range(3):
            part = jnp.clip(rest, -126, 127)
            powers.append(float_bits.exact_power2(part))
            rest = rest - part
        axes = tuple(sorted(frozenset().union(*(jax.typeof(v).mat.varying for v in (partial, gather)))
                            - self.held))
        return aligned_segment_sum([(partial, *powers)], gather, self.shape[0], axes)

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


FACTORED_WINDOW = 20
"""The binades below a route's largest cotangent-times-factor product past which its rescue's fp32 pullback
may drop a product (`_Transpose.routed_moved`), and the cancellation below which such a drop is refused."""

FACTORED_RESCUE = 100
"""The binade `factored`'s transpose moves a row's largest term to where its plain transpose overflowed: its
sum over any width a decoded matrix holds (an FP8 entry at most 448) within fp32, and a remainder 2**-233 of
it still a bf16 value."""
