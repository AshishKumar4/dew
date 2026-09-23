"""Token mixers: what a decoder layer mixes across the sequence, by kind.

A mixer is the per-layer token interaction a `DecoderBlock` holds as
`self_attn`: any module with the `(x, decode=..., positions=...,
segment_ids=...) -> x` signature. Grouped-query causal attention is the
`attention` kind; MLA, the gated delta rule and the other mixers register
beside it, each as a frozen dataclass value carrying the reference's field
names.

The backbone names one value on its `mixer` field, None for attention, and
a kind builds its own `DecoderBlock` factory from a `MixerContext`: the
layer geometry the backbone owns (heads, head dims, the kind-resolved rotary
base, the window, the KV-sharing slot) plus the run's dtype and kernel
choices. Geometry is stated once, here, so a new kind reads what it needs
without the backbone growing a branch per kind. The backbone builds every
mixer through `mixer.build(ctx)`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.kv_cache import KVCache
from dew.nn.rope import RopeScaling, YarnScaling
from dew.registry import mixers


@dataclasses.dataclass(frozen=True)
class MixerContext:
    """Everything a kind needs from the model and the layer to build its mixer.

    The values are the layer's resolved geometry: `head_dim` and `rope_theta`
    already carry the layer kind's overrides, `partial_rotary_factor` is None
    on a windowed kind (partial rotary belongs to the full-sequence kinds),
    and `kv_shared` with `kv_store_key` mark a layer that reads another
    layer's keys and values. A kind's own record (LoRA ranks, head splits, a
    yarn scaling) lives on the kind's value; this holds what the backbone
    configures.
    """

    emb_features: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    causal: bool = True
    rope_theta: float = 10000.0
    """The kind-resolved rotary base; a kind's yarn record transforms it."""
    rope_scaling: RopeScaling | None = None
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
    kv_store_key: str | None = None
    sliding_window: int | None = None
    attention_chunk: int | None = None
    """The kind's chunk: a query reads only the keys whose position shares
    its `position // attention_chunk` (`LayerKind.chunk`)."""
    attention_bias: bool = False
    o_proj_bias: bool | None = None
    attention_scale: float | None = None
    attention_sinks: bool = False
    yarn: YarnScaling | None = None
    attn_logit_softcap: float | None = None
    partial_rotary_factor: float | None = None
    partial_rotary_type: str = 'proportional'
    """Which convention the partial rotary follows, 'proportional' (Gemma 4)
    or 'default' (Qwen3.5); `dew.nn.rope.rotary_freqs` cites both."""
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl
    force_fp32_for_softmax: bool = True
    kv_cache: KVCache = KVCache()
    """The decode cache's storage layout (`dew.nn.kv_cache`)."""
    output_gate: bool = False
    """The attention's output gate (Qwen3.5's attn_output_gate), where the
    kind's projection doubles its query and a sigmoid of the second half
    gates the branch. The delta net's own gate activation is a field of its
    kind (qwen4_exp's output_gate_type), which is the only gate a reference
    varies."""
    init_std: float | None = None
    """The normal std the mixer's projections draw from; None keeps each
    module's own initializer (`CausalTransformer.initializer_range`)."""
    output_init_std: float | None = None
    """The std of the projection back to the residual stream, which a
    depth-scaled init shrinks; None follows `init_std`."""


class MixerBase:
    """One mixer kind's value: its fields, and how it builds its mixer.

    Each kind is a frozen dataclass of the reference's field names, registered
    under its name (`@mixers("mla")`), dict-constructible through
    `mixers.build`: an unknown kind or field raises there. `build` turns the
    value and the layer's context into the `DecoderBlock` factory, the
    `Callable[..., nn.Module]` the block calls with `name='self_attn'`.

    The backbone types its `mixer` field as this base, not the registry
    union, because a union of members that register over time cannot be
    spelled before they exist. Records still dispatch on their kind through
    `mixers.build`, and `mixers.union` is the live union for config
    introspection and a tyro subcommand per kind.
    """

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        """The block's mixer factory for this value at this layer's geometry."""
        raise NotImplementedError(
            f"{type(self).__name__} names a mixer kind but builds no mixer")


def mixer_from_record(record: Mapping[str, object]) -> MixerBase:
    """A `{"kind": ..., ...fields}` record as the kind value it names.

    The backbone's `__post_init__` and anything else that takes a mixer from
    a config call it, so `mixer={"kind": "mla", ...}` from a CLI and the
    dataclass from code meet in the same `mixers.build`. A record without a
    kind, or one naming nothing registered, raises ValueError.
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
    built: MixerBase = mixers.build(kind, fields)
    if not isinstance(built, MixerBase):
        raise ValueError(
            f"mixer {kind!r} built {type(built).__name__}, which is not a "
            "mixer value")
    return built


# The kind modules register where they are defined; this hub imports them,
# one name per kind module, alphabetical.
from .. import (
    deepseek_v4,  # noqa: F401  (registers the kind)
    dsa_kpool,  # noqa: F401  (registers the kind)
    kda,  # noqa: F401  (registers the kind)
    llama4,  # noqa: F401  (registers the kind)
    mla,  # noqa: F401  (registers the kind)
)
from . import (
    gated_delta_net,  # noqa: F401  (registers the kind)
    mamba2,  # noqa: F401  (registers the kind)
)
from .attention import AttentionMixer  # noqa: F401  (registers the kind)
