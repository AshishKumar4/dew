"""Terms an attention layer computes from its own input beside its queries, keys and values: a head's
max-L2 clip (`QKClip`), a query's log-ratio temperature (`log_ratio_scale`), and an input-conditioned
relative-position bias (`RelativeBias`), each general and each its own field of the attention mixer
(`dew.nn.mixers.attention.Attention`; theme M71).
"""

from __future__ import annotations

import dataclasses
import functools

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
    return _clip(x_wide, threshold, eps).astype(x.dtype)


def _scaled_length(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`x`'s length taken over `y = x s` (row 13137): `s` = 2^(cut - maxexp) for a head past 2^cut, `cut` =
    (maxexp - bits(D)) // 2 - 1, whose D squares' sum can overflow (1e20's square is fp32's infinity), so
    its largest is at most 2^cut, a power of two whose reciprocal is normal; else 1. `s` is held constant
    (`stop_gradient`): the clip is the same at either, and so its derivative. Returns `s`, `y` and `|y|`;
    a zero head's length is zero, its derivative there taken as the clip's (none: it lies in the clip's
    constant region), where the square root's would be infinite times zero."""
    wide = x.dtype
    maxexp = jnp.finfo(wide).maxexp
    cut = (maxexp - x.shape[-1].bit_length()) // 2 - 1
    largest = jax.lax.stop_gradient(jnp.max(jnp.abs(x), axis=-1, keepdims=True))
    factor = jnp.where(largest > 2.0 ** cut, jnp.asarray(2.0 ** (cut - maxexp), wide), jnp.asarray(1, wide))
    scaled = x * factor
    squared = jnp.sum(jnp.square(scaled), axis=-1, keepdims=True)
    held = squared > 0
    return factor, scaled, jnp.where(held, jnp.sqrt(jnp.where(held, squared, 1)), 0)


def _rescaled(x: jax.Array, threshold: float, eps: float) -> jax.Array:
    """The clip of `x` in its scaled space (`_scaled_length`): `y * min(threshold / (|y| + eps s), 1 / s)`,
    `x * min(threshold / (|x| + eps), 1)` with no value below normal nor past the largest, and so its
    derivative too."""
    factor, scaled, length = _scaled_length(x)
    return scaled * jnp.minimum(threshold / (length + eps * factor), 1 / factor)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _clip(x: jax.Array, threshold: float, eps: float) -> jax.Array:
    """`x`'s clip (`clipped`): its gain `min(threshold / (|x| + eps), 1)`, taken over its scaled length as
    `min(threshold s / (|y| + eps s), 1)`, multiplying `x` itself where it is normal, so a small component
    beside a large one is kept; where it is subnormal (a head near the largest under a small threshold,
    which XLA's products flush to zero), the clip in its scaled space (`_rescaled`). Its derivative is
    always the scaled space's (`_clip_jvp`): `x` times a cotangent can overflow where the clip cannot. A
    head within 2^cut takes `s` = 1, its arithmetic and derivative unchanged."""
    factor, _, length = _scaled_length(x)
    gain = jnp.minimum(threshold * factor / (length + eps * factor), 1)
    return jnp.where(gain >= jnp.finfo(x.dtype).tiny, x * gain, _rescaled(x, threshold, eps))


@_clip.defjvp
def _clip_jvp(threshold: float, eps: float, primals, tangents):
    (x,), (tangent,) = primals, tangents
    _, moved = jax.jvp(lambda x: _rescaled(x, threshold, eps), (x,), (tangent,))
    return _clip(x, threshold, eps), moved


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
