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
    `threshold` (`QKClip`), in at least fp32 and returned in `x`'s dtype."""
    wide = at_least_fp32(x.dtype)
    x_wide = x.astype(wide) if scale is None else x.astype(wide) * scale.astype(wide)
    squared = jnp.sum(jnp.square(x_wide), axis=-1, keepdims=True)
    # A zero head's length is zero, its derivative there taken as the clip's (none: it lies in the clip's
    # constant region), where the square root's would be infinite times zero.
    held = squared > 0
    length = jnp.where(held, jnp.sqrt(jnp.where(held, squared, 1)), 0)
    return (x_wide * jnp.minimum(threshold / (length + eps), 1)).astype(x.dtype)


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
    group = min(queries, template.BAND_GROUP)
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

    band = jax.lax.map(banded, jnp.arange(groups))
    values = jnp.moveaxis(band, 0, 2).reshape(batch, heads, groups * group, width)[:, :, :queries]
    return BandBias(values, offset, extent)
