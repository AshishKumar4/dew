"""Token mixers: what a decoder layer mixes across the sequence, by kind.

A mixer is the module a `DecoderBlock` holds as `self_attn`, called `(x, call)
-> x` with one `dew.nn.inputs.MixerCall`, its tokens' places and documents the
call metadata's `Coordinates` (`dew.nn.inputs.coordinates_of`). It declares the
call's `MIXING` fields it reads (`reads`), and the block calls it with no other
one set. Grouped-query
causal attention is the `attention` kind; MLA, the gated delta rule and the
rest sit beside it as frozen dataclass values under the reference's
field names. A kind builds its `DecoderBlock` factory with `mixer.build(ctx)`
from a `MixerContext`, the layer geometry the backbone owns (heads, head
dims, the kind-resolved rotary base, the window, the KV-sharing slot) plus
the run's dtype and kernel choices, so a new kind needs no backbone branch.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import ClassVar

from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.kv_cache import KVCache
from dew.nn.ngpt import NGPT
from dew.nn.rope import LongRopeScaling, RopeScaling, YarnScaling

QKV_RESIDUALS = ('q_proj', 'k_proj', 'v_proj', 'kv_proj')
ATTENTION_RESIDUALS = (*QKV_RESIDUALS, 'o_proj')
MLP_RESIDUALS = ('gate_proj', 'up_proj', 'down_proj')
DELTA_RESIDUALS = ('delta_decay', 'delta_inverse', 'delta_memory')
MEMORY_RESIDUALS = ('memory_carry',)
RESIDUALS = (*ATTENTION_RESIDUALS, 'context', 'scan', 'conv', *DELTA_RESIDUALS, *MEMORY_RESIDUALS,
             *MLP_RESIDUALS)
"""The values a block names as it runs, each after the projection that
produced it: `kv_proj` is latent attention's fused `kv_b_proj`, `context`
a mixer's output before `o_proj`, the attention kernel's or a recurrence's,
`scan` a recurrence's (an SSM's or a delta rule's) alone, which a backward
that drops it recomputes by running the scan again, `conv` a
linear-attention or SSM mixer's short conv output (`causal_conv1d`,
its queries, keys and values for a delta rule), `delta_decay`,
`delta_inverse` and `delta_memory` the rule's cumulated decay, chunk inverse
and memory (with `w` and the corrected values) the port saves for its
backward (`dew.nn.kernels.fla_gdn`), `memory_carry` an in-context memory's
network and momentum as each chunk of its scan hands them on
(`dew.nn.memory.CARRY`), and the MLP names cover the dense MLP
and the routed experts alike. A remat policy picks from these; a
name off the list is a typo that would otherwise recompute silently."""
ATTENTION_CONTEXT_RESIDUALS = (*ATTENTION_RESIDUALS, 'context')
"""An attention mixer's names, its projections and its context."""
DELTA_RULE_RESIDUALS = ('context', 'scan', 'conv', *DELTA_RESIDUALS)
"""A delta-rule mixer's names: its scan's output as its context and as a scan, its convolution's output,
the delta rule's three."""


@dataclasses.dataclass(frozen=True)
class MixerContext:
    """Everything a kind needs from the model and the layer to build its mixer.

    The values are the layer's resolved geometry: `head_dim` and `rope_theta`
    already carry the layer kind's overrides, `partial_rotary_factor` is None
    on a windowed kind (partial rotary belongs to the full-sequence kinds),
    and `kv_shared` with `kv_store_key` mark a layer that reads another
    layer's keys and values. A kind's own record (LoRA ranks, head splits, a
    yarn scaling) lives on the kind's value; this holds what the backbone
    configures. `CausalSelfAttention` holds it whole (`ctx`) and reads its
    geometry and policies off it.
    """

    emb_features: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    value_head_dim: int | None = None
    """The value heads' width; None is `head_dim`. A narrower value is cached
    at its own width and widened only inside the kernel."""
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
    """The q/k/v biases, as config.attention_bias in HF."""
    o_proj_bias: bool | None = None
    """None follows attention_bias; Qwen2 biases q/k/v only."""
    attention_scale: float | None = None
    """None is the kernel's own 1/sqrt(head_dim)."""
    value_scale: float | None = None
    """Multiplies the values as projected, before the cache and the kernel
    read them (MiMo-V2-Flash's `attention_value_scale`)."""
    attention_dropout_rate: float = 0.0
    attention_sinks: bool = False
    yarn: YarnScaling | None = None
    attn_logit_softcap: float | None = None
    """Gemma 2's tanh on the logits (attn_logit_softcapping)."""
    partial_rotary_factor: float | None = None
    """The share of each head that rotates; None rotates all of it."""
    partial_rotary_type: str = 'proportional'
    """Which convention the partial rotary follows, 'proportional' (Gemma 4)
    or 'default' (Qwen3.5); `dew.nn.rope.rotary_freqs` cites both."""
    narrow_rotary_tables: bool = False
    """Whether the rotary's cos and sin round to the activations' dtype before the rotation, which
    stays fp32: vLLM's rotary casts its cos/sin cache to the query's dtype
    (rotary_embedding/base.py, `_match_cos_sin_cache_dtype`) and its kernel rotates in fp32
    (csrc/pos_encoding_kernels.cu, `apply_token_rotary_embedding`)."""
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    attention_impl: str = "auto"  # an AttentionImpl
    force_fp32_for_softmax: bool = True
    kv_cache: KVCache = dataclasses.field(default_factory=KVCache)
    """The decode cache's storage layout (`dew.nn.kv_cache`)."""
    ngpt: NGPT | None = None
    """The normalized transformer's gains (`dew.nn.ngpt.NGPT`): the attention unit-norms each
    rotated query and key head, scales the query by its learned gain, scales the output gate's
    logits by theirs and unit-norms its gated output; None for none."""
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

    def policies(self) -> dict[str, bool]:
        """Whether this layer asks each of the attention policies that change
        what a mixer computes, beyond its geometry: a mixer implements each it
        is asked or refuses it by name (`refuse`), never computes without it."""
        return {
            "force_fp32_for_softmax": not self.force_fp32_for_softmax,
            "attn_logit_softcap": self.attn_logit_softcap is not None,
            "attention_sinks": self.attention_sinks,
            "value_scale": self.value_scale is not None,
            "o_proj_bias": bool(self.o_proj_bias),
            "attention_dropout_rate": bool(self.attention_dropout_rate),
            "narrow_rotary_tables": self.narrow_rotary_tables,
            "ngpt": self.ngpt is not None,
        }

    def refuse(self, kind: str, asked: Mapping[str, object], why: str) -> None:
        """Raise `MixerRefused` naming every entry of `asked` this layer sets (a
        truthy value: a flag on, a field given), none of which mixer `kind`
        implements: `why`."""
        names = sorted(name for name, value in asked.items() if value)
        if names:
            raise MixerRefused(kind, tuple(names), why)


class MixerRefused(ValueError):
    """A layer asking mixer `kind` for the context entries `names`, which it does not implement, and
    `why` (`MixerContext.refuse`)."""

    def __init__(self, kind: str, names: tuple[str, ...], why: str):
        super().__init__(f"the {kind} mixer has no {', '.join(names)}: {why}")
        self.kind, self.names, self.why = kind, names, why

    def __reduce__(self):
        # Rebuilt from its own fields, as a pickle or a process boundary carries it.
        return type(self), (self.kind, self.names, self.why)


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
    `gpu_target().kernels['xla_triton_gemm']` turns them off."""

    residuals: ClassVar[tuple[str, ...] | None] = None
    """The names (`dew.nn.mixer_base.RESIDUALS`) the mixer this kind
    builds marks for a remat policy to keep, or every name where None: a
    decoder skips a remat rung that keeps nothing more of its layers' names
    than the next (`CausalTransformer.recompute_more`)."""

    keeps_cache = True
    """Whether the mixer this kind builds keeps a decode cache, which a loop's
    shared cache has every pass rewrite (`CausalTransformer.split_prefill_refusal`)."""

    mixed_step = False
    """Whether the mixer this kind builds runs a server's mixed call
    (`dew.nn.inputs.Admitted`) as the separate decode and prefill calls
    would, or keeps no cache for it to read: one that does not read the
    call's layout would take its one row of tokens as one sequence."""

    def mixed_refusal(self, ctx: MixerContext) -> str | None:
        """Why the mixer this kind builds over `ctx` would not run a server's
        mixed call as the separate calls would, or None: by default, unless
        it declares `mixed_step`."""
        return None if self.mixed_step else f"{type(self).__name__}'s cache, which a mixed step does not run"

    def generation_refusal(self) -> str | None:
        """Why a generation loop cannot run the mixer this kind builds, or None
        (`dew.nn.protocols.Generating`): by default nothing refuses it."""
        return None

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        """The block's mixer factory for this value at this layer's geometry."""
        raise NotImplementedError(
            f"{type(self).__name__} names a mixer kind but builds no mixer")
