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

from dew.nn.attention import RopeScaling
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
    rope_scaling: Optional[RopeScaling] = None
    """The kind-resolved llama3 ramp over the base frequencies, or None for plain rope."""
    qk_norm: bool = True
    qk_norm_scope: str = 'head'
    """Where the q/k RMSNorm applies: 'head' norms each head after the split
    (Qwen3, the Gemmas), 'projection' the whole projection before it (OLMo 3)."""
    v_norm: bool = False
    k_eq_v: bool = False
    """Gemma 4's attention_k_eq_v on this layer: no value projection, the
    values are the raw keys under the values norm."""
    norm_eps: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    kv_shared: bool = False
    kv_store_key: Optional[str] = None
    sliding_window: Optional[int] = None
    attention_bias: bool = False
    o_proj_bias: Optional[bool] = None
    attention_scale: Optional[float] = None
    attention_sinks: bool = False
    yarn: Optional[mla.YarnScaling] = None
    attn_logit_softcap: Optional[float] = None
    partial_rotary_factor: Optional[float] = None
    partial_rotary_type: str = 'proportional'
    """Which convention the partial rotary follows, 'proportional' (Gemma 4)
    or 'default' (Qwen3.5); `dew.nn.attention.rotary_freqs` cites both."""
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    attention_impl: Optional[str] = None
    force_fp32_for_softmax: bool = True
    output_gate: bool = False
    """The attention's output gate (Qwen3.5's attn_output_gate), where the
    kind's projection doubles its query and a sigmoid of the second half
    gates the branch. The delta net's own gate activation is a field of its
    kind (qwen4_exp's output_gate_type), which is the only gate a reference
    varies."""


class MixerBase:
    """One mixer kind's value: its fields, and how it builds its mixer.

    Each kind is a frozen dataclass of the reference's field names, registered
    under its name (`@mixers("mla")`), dict-constructible through
    `mixers.build`: an unknown kind or field raises there. `build` turns the
    value and the layer's context into the `DecoderBlock` factory, the
    `Callable[..., nn.Module]` the block calls with `name='self_attn'`.

    The backbone types its `mixer` field as this base, not as the registry
    union: a union of members that register over time cannot be spelled
    before they exist, and the checker rejects a late-bound name in type
    position. Records still dispatch on their kind through `mixers.build`,
    and `mixers.union` stays the live union for config introspection and a
    tyro subcommand per kind.
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
            rope_scaling=ctx.rope_scaling,
            qk_norm=ctx.qk_norm,
            qk_norm_scope=ctx.qk_norm_scope,
            v_norm=ctx.v_norm,
            k_eq_v=ctx.k_eq_v,
            norm_eps=ctx.norm_eps,
            scale_offset=ctx.scale_offset,
            scale_after_cast=ctx.scale_after_cast,
            kv_shared=ctx.kv_shared,
            kv_store_key=ctx.kv_store_key,
            sliding_window=ctx.sliding_window,
            attention_bias=ctx.attention_bias,
            o_proj_bias=ctx.o_proj_bias,
            attention_scale=ctx.attention_scale,
            attention_sinks=ctx.attention_sinks,
            yarn=ctx.yarn,
            attn_logit_softcap=ctx.attn_logit_softcap,
            output_gate=ctx.output_gate,
            dtype=ctx.dtype,
            precision=ctx.precision,
            attention_impl=ctx.attention_impl,
            force_fp32_for_softmax=ctx.force_fp32_for_softmax,
            partial_rotary_factor=ctx.partial_rotary_factor,
            partial_rotary_type=ctx.partial_rotary_type)


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
from . import gated_delta_net  # noqa: E402,F401  (registers the kind)
from .. import llama4  # noqa: E402,F401  (registers the kind)
from .. import mla  # noqa: E402,F401  (registers the kind)
