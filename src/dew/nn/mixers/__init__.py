"""Token mixers: what a decoder layer mixes across the sequence, by kind.

A mixer is the per-layer token interaction a `DecoderBlock` holds as
`self_attn`: any module with the `(x, decode=..., positions=...,
segment_ids=...) -> x` signature. Today's grouped-query causal attention is
the `attention` kind; MLA, gated delta rule and the other frontier mixers
register beside it, each as a frozen dataclass value carrying the reference's
field names.

The backbone names one value on its `mixer` field, None for today's
attention, and a kind builds its own `DecoderBlock` factory from a
`MixerContext`: the layer geometry the backbone owns (heads, head dims, the
kind-resolved rotary base, the window, the KV-sharing slot) plus the run's
dtype and kernel choices. Geometry is stated once, here, so a new kind reads
what it needs without the backbone growing a branch per kind: the one
dispatch is `mixer.build(ctx)`, and a second switch on the kind anywhere else
is a bug.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import Optional

from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.registry import Registry

mixers: Registry[type] = Registry("mixer")
"""Each token-mixer kind by name: the value class, not the module."""


@dataclasses.dataclass(frozen=True)
class MixerContext:
    """Everything a kind needs from the model and the layer to build its mixer.

    The values are the layer's resolved geometry: `head_dim` and `rope_theta`
    already carry the layer kind's overrides, `partial_rotary_factor` is None
    on a windowed kind (partial rotary belongs to the full-sequence kinds),
    and `kv_shared` with `kv_store_key` mark a layer that reads another
    layer's keys and values. A kind's own record (LoRA ranks, head splits, a
    yarn scaling) lives on the kind's value, not here: this is what the
    backbone configures, that is what the reference names.
    """

    emb_features: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    causal: bool = True
    rope_theta: float = 10000.0
    """The kind-resolved rotary base: a kind's yarn record transforms this, never replaces it."""
    qk_norm: bool = True
    v_norm: bool = False
    norm_eps: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    kv_shared: bool = False
    kv_store_key: Optional[str] = None
    sliding_window: Optional[int] = None
    attention_bias: bool = False
    attention_scale: Optional[float] = None
    partial_rotary_factor: Optional[float] = None
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    attention_impl: Optional[str] = None
    force_fp32_for_softmax: bool = True


class MixerBase:
    """One mixer kind's value: its fields, and how it builds its mixer.

    Each kind is a frozen dataclass of the reference's field names, registered
    under its name (`@mixers("mla")`), dict-constructible through
    `mixers.build`: an unknown kind or field raises there. `build` turns the
    value and the layer's context into the `DecoderBlock` factory, the
    `Callable[..., nn.Module]` the block calls with `name='self_attn'`.
    """

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        """The block's mixer factory for this value at this layer's geometry."""
        raise NotImplementedError(
            f"{type(self).__name__} names a mixer kind but builds no mixer")


@mixers("attention")
@dataclasses.dataclass(frozen=True)
class AttentionMixer(MixerBase):
    """Today's grouped-query causal attention: no fields of its own.

    Every dial is the model's (heads, norms, bias, scale, kernel), read from
    the context, so this value only selects the kind. A record names it with
    `{"kind": "attention"}`.
    """

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        # Imported here, not at module scope: the backbone imports this
        # package for the registry, so importing the backbone back at scope
        # would be a cycle. By the time a layer builds, both are loaded.
        from dew.nn.backbones.causal_transformer import CausalSelfAttention

        return functools.partial(
            CausalSelfAttention,
            emb_features=ctx.emb_features,
            num_heads=ctx.num_heads,
            num_kv_heads=ctx.num_kv_heads,
            head_dim=ctx.head_dim,
            max_seq_len=ctx.max_seq_len,
            causal=ctx.causal,
            rope_theta=ctx.rope_theta,
            qk_norm=ctx.qk_norm,
            v_norm=ctx.v_norm,
            norm_eps=ctx.norm_eps,
            scale_offset=ctx.scale_offset,
            scale_after_cast=ctx.scale_after_cast,
            kv_shared=ctx.kv_shared,
            kv_store_key=ctx.kv_store_key,
            sliding_window=ctx.sliding_window,
            attention_bias=ctx.attention_bias,
            attention_scale=ctx.attention_scale,
            dtype=ctx.dtype,
            precision=ctx.precision,
            attention_impl=ctx.attention_impl,
            force_fp32_for_softmax=ctx.force_fp32_for_softmax,
            partial_rotary_factor=ctx.partial_rotary_factor)


def mixer_from_record(record: Mapping[str, object]) -> MixerBase:
    """A `{"kind": ..., ...fields}` record as the kind value it names.

    The one place a record becomes a mixer: the backbone's `__post_init__`
    and anything else that takes a mixer from a config call this, so
    `mixer={"kind": "mla", ...}` from a CLI and the dataclass from code meet
    in the same `mixers.build`. A record without a kind, or one naming
    nothing registered, raises naming what it wanted.
    """
    fields = dict(record)
    try:
        kind = fields.pop("kind")
    except KeyError:
        raise ValueError(
            f"a mixer record names its kind, got {sorted(fields)}; known: "
            f"{', '.join(sorted(mixers))}") from None
    if not isinstance(kind, str):
        raise ValueError(
            f"a mixer kind is a registered name, not {kind!r}; known: "
            f"{', '.join(sorted(mixers))}")
    built: MixerBase = mixers.build(kind, **fields)
    if not isinstance(built, MixerBase):
        raise ValueError(
            f"mixer {kind!r} built {type(built).__name__}, which is not a "
            "mixer value")
    return built


# The kind modules register where they are defined; this hub imports them,
# one line per kind module, alphabetical.
from .. import mla
