"""Terms an attention layer computes from its own input beside its queries, keys and values: a head's
max-L2 clip (`QKClip`), a query's log-ratio temperature (`log_ratio_scale`), and an input-conditioned
relative-position bias (`RelativeBias`), each general and each its own field of the attention mixer
(`dew.nn.mixers.attention.Attention`; theme M71).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from .derivatives import Protected, contracted, declared_tangent
from .precision import at_least_fp32, computed_einsum


@dataclasses.dataclass(frozen=True)
class QKClip:
    """Each query and key head's length clipped to at most `threshold`: `x * min(threshold / (||x|| +
    eps), 1)` in at least fp32, rounded back, queries first multiplied by a learned scale a head channel
    where `query_scale` (`q_clip_scale`, at ones), keys not (Inkling 2's attention,
    modeling_inkling2_loop.py:185-197). It sits where a head norm (`MixerContext.qk_norm`) would, after the
    projections and before any rotation, and stands in for one: a layer takes one or the other."""

    threshold: float
    query_scale: bool = True
    eps: float = 1e-6

    def __post_init__(self):
        if not self.threshold > 0 or not self.eps > 0:
            raise ValueError(f"a clip's threshold and epsilon are positive, got {self.threshold}, {self.eps}")


def clipped(x: jax.Array, threshold: float, eps: float, scale: jax.Array | None = None) -> jax.Array:
    """`x` `[..., D]` times `scale` `[..., D]` where given, its last axis's L2 length clipped to at most
    `threshold` (`QKClip`): `x * min(threshold / (|x| + eps), 1)`, in at least fp32 and returned in `x`'s
    dtype. A head whose D squares' sum cannot overflow, past which its largest component lies beyond 2^cut
    (`_bound`), computes the formula as written, its derivatives JAX's of it at every order; a larger one
    (row 13137: 1e20's square is fp32's infinity) takes its length over a scaled copy (`_giant`). Each
    branch reads the other's head as zero, so neither's derivative reaches the other's."""
    wide = at_least_fp32(x.dtype)
    x_wide = x.astype(wide) if scale is None else x.astype(wide) * scale.astype(wide)
    cut = _bound(x_wide)
    giant = jax.lax.stop_gradient(jnp.max(jnp.abs(x_wide), axis=-1, keepdims=True) > 2.0 ** cut)
    ordinary = jnp.where(giant, 0, x_wide)
    squared = jnp.sum(jnp.square(ordinary), axis=-1, keepdims=True)
    # A zero head's length is zero, its derivative there taken as the clip's (none: it lies in the clip's
    # constant region), where the square root's would be infinite times zero.
    held = squared > 0
    length = jnp.where(held, jnp.sqrt(jnp.where(held, squared, 1)), 0)
    plain = ordinary * jnp.minimum(threshold / (length + eps), 1)
    # A giant head's scale `s` = 2^(cut - maxexp), a power of two whose reciprocal is normal, so its scaled
    # copy's largest is at most 2^cut; 1 for the other branch's zero head.
    reduced = jnp.asarray(2.0 ** (cut - jnp.finfo(wide).maxexp), wide)
    factor = jnp.where(giant, reduced, jnp.asarray(1, wide))
    clipped_giant = _giant(jnp.where(giant, x_wide, 0), factor, threshold, eps)
    return jnp.where(giant, clipped_giant, plain).astype(x.dtype)


def _bound(x: jax.Array) -> int:
    """`cut` = (maxexp - bits(D)) // 2 - 1 of `x`'s dtype and width D: the exponent of the largest component
    whose head's D squares sum within range."""
    return (jnp.finfo(x.dtype).maxexp - x.shape[-1].bit_length()) // 2 - 1


def _scaled(x: jax.Array, factor: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A giant head's scaled copy `y = x s` (`clipped`'s `factor`) and `|y|`."""
    scaled = x * factor
    squared = jnp.sum(jnp.square(scaled), axis=-1, keepdims=True)
    held = squared > 0
    return scaled, jnp.where(held, jnp.sqrt(jnp.where(held, squared, 1)), 0)


def _giant_primal(x: jax.Array, factor: jax.Array, threshold: float, eps: float) -> jax.Array:
    """A giant head's clip (`clipped`): its gain `g = min(threshold s / (|y| + eps s), 1)` over its scaled
    length (`_scaled`) times `x` itself where `g` is normal, so a small component beside a large one is
    kept; where it is subnormal (a head near the largest under a small threshold, which XLA's products
    flush to zero), `y * min(threshold / (|y| + eps s), 1 / s)`, the same clip with no factor below normal.
    Its derivative is the formula's (`_giant_jvp`), no product of the head and a tangent forming at the
    head's scale."""
    scaled, length = _scaled(x, factor)
    denominator = length + eps * factor
    gain = jnp.minimum(threshold * factor / denominator, 1)
    # Clipped where the gain is subnormal: the scaled head times `threshold / (|y| + eps s)` where that is
    # normal, else over its scaled length first (at most 1) and then the threshold, so no factor below
    # normal forms first.
    tiny = jnp.finfo(x.dtype).tiny
    quotient = threshold / denominator
    small = jnp.where(quotient >= tiny, scaled * quotient, (scaled / denominator) * threshold)
    return jnp.where(gain >= tiny, x * gain, small)


def _giant_jvp(threshold: float, eps: float, primals, tangents):
    """The clip's derivative, a symmetric map of the tangent (`_moved`, its own transpose, so reverse mode
    evaluates the same expression: `generation.linear_map`): unclipped, the tangent itself; else `g ((dx -
    u (u . dx)) + u (u . dx) ((1 - c) + c eps s / (|y| + eps s)))` of the head's direction `u = y / |y|`, the
    share `c` of the ratio's derivative `min` takes (JAX's: all where it clips, half at a tie), the radial
    part kept apart so epsilon's share survives where `|y| / (|y| + eps s)` rounds to 1, and `g` applied
    whole where it is normal, else as `((v threshold) / (|y| + eps s)) s`. The scale is constant (no
    tangent). Exact to fp32's normal range: a component or a product of terms below it relative to the head
    (`2^-126` of its largest) flushes, as any fp32 norm of a scaled head does."""
    from .kernels.generation import linear_map

    (x, factor), (tangent, _) = primals, tangents
    scaled, length = _scaled(x, factor)
    denominator = length + eps * factor
    ratio = threshold * factor / denominator
    share = jnp.where(ratio < 1, 1.0, jnp.where(ratio == 1, 0.5, 0.0)).astype(x.dtype)
    radial_share = (1 - share) + share * (eps * factor / denominator)
    residuals = (scaled / jnp.where(length > 0, length, 1), radial_share, ratio, factor, denominator)

    def moved(residuals, tangent):
        direction, radial_share, ratio, factor, denominator = residuals
        radial = direction * jnp.sum(direction * tangent, axis=-1, keepdims=True)
        kept = (tangent - radial) + radial * radial_share
        gain = jnp.minimum(ratio, 1)
        scaled_down = jnp.where(gain >= jnp.finfo(tangent.dtype).tiny, kept * gain,
                                ((kept * threshold) / denominator) * factor)
        return jnp.where(ratio > 1, tangent, scaled_down)

    return _giant(x, factor, threshold, eps), linear_map(moved, moved, residuals, tangent)


_giant = contracted("test_a_giant_heads_clip_holds_every_order")(
    declared_tangent(Protected(), _giant_primal, _giant_jvp, nondiff_argnums=(2, 3)))
"""A giant head's clip (`_giant_primal`) and its derivative's rule (`_giant_jvp`), as their numerics need
them written."""


def log_ratio_scale(positions, floor: float, scale: float, *, dtype: DTypeLike) -> jax.Array:
    """A query's multiplier at each absolute position `p`, `1 + scale * log(max((p + 1) / floor, 1))` in
    `dtype`: 1 through position `floor - 1`, then growing with the log of the length past it (Inkling 2's
    global layers, modeling_inkling2_loop.py:199-229), where Llama 4's floors the ratio and adds one
    (`temperature_scale`)."""
    positions = jnp.asarray(positions, dtype)
    return 1.0 + scale * jnp.log(jnp.maximum((positions + 1.0) / floor, 1.0))


@dataclasses.dataclass(frozen=True)
class RelativeBias:
    """A relative-position bias each query computes from its own input: `rank` states a head, projected
    from the query's input (`relative`), times a learned scale a head and state (`relative_scale`, at
    ones), clipped to `clip` where given (`clipped`) and scaled by the query's temperature where
    `temperature`, against a learned `[rank, extent]` bank (`relative_bank`, at zeros), read at the
    query's distance from each key (its place less the key's) and zero outside `[0, extent)` (Inkling 2,
    modeling_inkling2_loop.py:231-262). It enters the logits as a `[B, H, Q, K]` dense bias in at least fp32
    beside the call's own (`AttentionMetadata.position_bias`), O(Q K) memory at a prefill: a bias the fused
    kernels take where they take a dense one, the reference's elsewhere."""

    rank: int
    extent: int
    clip: float | None = None
    temperature: bool = False

    def __post_init__(self):
        if self.rank < 1 or self.extent < 1 or (self.clip is not None and not self.clip > 0):
            raise ValueError(f"a relative bias holds a positive rank, extent and clip, got {self}")


def relative_logits(states: jax.Array, bank: jax.Array, query_places: jax.Array, key_places: jax.Array,
                    ) -> jax.Array:
    """The bias `[B, H, Q, K]` in at least fp32 of each query's relative states `states` `[B, Q, H, rank]`
    against `bank` `[rank, extent]`, read at the query's place less each key's (`query_places` `[B, Q]` or
    `[Q]`, `key_places` `[B, K]` or `[K]`) and zero outside `[0, extent)` (`RelativeBias`)."""
    extent = bank.shape[-1]
    wide = at_least_fp32(jnp.result_type(states, bank))
    profile = computed_einsum("bqhr,re->bhqe", states, bank, dtype=wide)
    queries, keys = jnp.atleast_2d(query_places), jnp.atleast_2d(key_places)
    distance = (queries[:, :, None] - keys[:, None, :]).astype(jnp.int32)
    index = jnp.broadcast_to(jnp.clip(distance, 0, extent - 1)[:, None],
                             (profile.shape[0], profile.shape[1], *distance.shape[1:]))
    read = jnp.take_along_axis(profile, index, axis=-1)
    return jnp.where(((distance >= 0) & (distance < extent))[:, None], read, 0.0)


def relative_band(states: jax.Array, bank: jax.Array, offset: jax.Array):
    """`relative_logits`' bias as a band (`attention.BandBias`): each query's relative states `states` `[B, Q,
    H, rank]` against `bank` `[rank, extent]`, read at the query's place less each key's within its band and
    zero outside `[0, extent)`, the keys at their indices and each query at its index plus `offset` (`[]` or
    `[B]` int32: 0 for a prefill from the first key, a chunk's first place for a chunked one). Formed a
    `template.BAND_GROUP` of queries at a time, each group's `[B, H, BAND_GROUP, extent]` products gathered
    into its band and dropped, so no call holds every query's products beside the band they are read into
    (an Inkling 2 prefill's 32768 queries' would hold 3.2 GiB a layer beside it). A caller whose places are
    not their indices so (counted places, `attention_key_positions`) takes `relative_logits`."""
    from .attention import BandBias
    from .kernels.templates import attention as template

    batch, queries, heads, _ = states.shape
    extent = bank.shape[-1]
    wide = at_least_fp32(jnp.result_type(states, bank))
    width = template.band_width(extent)
    group = max(1, min(queries, template.BAND_GROUP))
    groups = -(-queries // group)
    padded = jnp.pad(states, ((0, 0), (0, groups * group - queries), (0, 0), (0, 0)))
    offset = jnp.asarray(offset, jnp.int32)
    origins = template.band_origins(offset, groups * group, extent)
    places = jnp.reshape(offset, (-1, 1))

    def banded(index):
        first = index * group
        rows = jax.lax.dynamic_slice_in_dim(padded, first, group, axis=1)
        products = computed_einsum("bqhr,re->bhqe", rows, bank, dtype=wide)
        # Column c's distance: the query's place less its key's (`band_origins`' origin plus c).
        queries_at = first + jnp.arange(group, dtype=jnp.int32)
        lead = places + queries_at[None, :] - jax.lax.dynamic_slice_in_dim(origins, first, group, axis=1)
        distance = lead[:, :, None] - jnp.arange(width, dtype=jnp.int32)[None, None, :]
        read_at = jnp.broadcast_to(jnp.clip(distance, 0, extent - 1)[:, None], (batch, heads, group, width))
        read = jnp.take_along_axis(products, read_at, axis=-1)
        return jnp.where(((distance >= 0) & (distance < extent))[:, None], read, jnp.zeros((), wide))

    if groups == 0:
        return BandBias(jnp.zeros((batch, heads, 0, width), wide), offset, extent)
    band = jax.lax.map(banded, jnp.arange(groups))
    values = jnp.moveaxis(band, 0, 2).reshape(batch, heads, groups * group, width)[:, :, :queries]
    return BandBias(values, offset, extent)
