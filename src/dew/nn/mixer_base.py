"""Token mixers: what a decoder layer mixes across the sequence, by kind.

A mixer is the module a `DecoderBlock` holds as `self_attn`, with the `(x,
decode=..., positions=..., segment_ids=...) -> x` signature. Grouped-query
causal attention is the `attention` kind; MLA, the gated delta rule and the
rest sit beside it as frozen dataclass values under the reference's
field names. A kind builds its `DecoderBlock` factory with `mixer.build(ctx)`
from a `MixerContext`, the layer geometry the backbone owns (heads, head
dims, the kind-resolved rotary base, the window, the KV-sharing slot) plus
the run's dtype and kernel choices, so a new kind needs no backbone branch.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.kv_cache import KVCache
from dew.nn.rope import LongRopeScaling, RopeScaling, YarnScaling


@dataclasses.dataclass(frozen=True)
class MixerContext:
    """Everything a kind needs from the model and the layer to build its mixer.

    The values are the layer's resolved geometry: `head_dim` and `rope_theta`
    already carry the layer kind's overrides, `partial_rotary_factor` is None
    on a windowed kind (partial rotary belongs to the full-sequence kinds),
    and `kv_shared` with `kv_store_key` mark a layer that reads another
    layer's keys and values. A kind's own record (LoRA ranks, head splits, a
    yarn scaling) lives on the kind's value; this holds what the backbone
    configures. `AttentionMixer` passes every field to `CausalSelfAttention`
    by name, so a new field needs a namesake there or the build raises.
    """

    emb_features: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    causal: bool = True
    rope_theta: float = 10000.0
    """The kind-resolved rotary base; a kind's yarn record transforms it."""
    rope_scaling: RopeScaling | LongRopeScaling | None = None
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
    bidirectional_window: bool = False
    """Whether a non-causal layer keeps `sliding_window` on both sides of a
    query (`LayerKind.bidirectional_window`); without it the window bounds
    only a cached prefix and the layer otherwise reads its whole row."""
    attention_chunk: int | None = None
    """The kind's chunk: a query reads only the keys whose position shares
    its `position // attention_chunk` (`LayerKind.chunk`)."""
    attention_bias: bool = False
    o_proj_bias: bool | None = None
    attention_scale: float | None = None
    attention_dropout_rate: float = 0.0
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
    kv_cache: KVCache = dataclasses.field(default_factory=KVCache)
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

    A `{"class": ..., "fields": ...}` record, by alias (`"mla"`) or import
    path, builds it through `mixers.from_record`, which refuses unknown kinds
    and fields. `build` returns the `DecoderBlock` factory called with
    `name='self_attn'`. The backbone types its field as this base, so a kind
    defined outside Dew is taken too; `mixers.union` is the union of the
    aliased kinds, for introspection and tyro.
    """

    keeps_triton_gemm = False
    """Whether a step with this mixer keeps XLA's Triton GEMM fusions where
    `dew.telemetry.devices.TRITON_GEMM_OFF_GENERATIONS` turns them off."""

    mixed_step = False
    """Whether the mixer this kind builds runs a server's mixed call
    (`dew.nn.inputs.Admitted`) as the separate decode and prefill calls
    would, or keeps no cache for it to read: one that does not read the
    call's layout would take its one row of tokens as one sequence."""

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        """The block's mixer factory for this value at this layer's geometry."""
        raise NotImplementedError(
            f"{type(self).__name__} names a mixer kind but builds no mixer")
