"""The autoregressive transformer decoder that every language model in Dew trains.

By default the model has a token embedding, rotary positions, pre-norm blocks of
grouped-query causal attention and a gated MLP, a final RMSNorm, and an fp32 head.
Attention goes through the shared kernel path in dew.nn.attention, so a run picks
reference/xla/cudnn/tpu the same way a diffusion run does, and decoding uses the
same fixed-size KV cache helpers.

Parameter names follow the HF decoder layout: embed_tokens,
layers_N.{input_layernorm, self_attn.{q,k,v,o}_proj, post_attention_layernorm,
mlp.{gate,up,down}_proj}, norm, lm_head. Gemma's two extra norms are the
exception. HF calls them post_attention_layernorm and post_feedforward_layernorm
even though they normalize sublayer outputs, so here they are
attention_output_norm and mlp_output_norm, and the pre-norms keep their names.
dew.interop.hf_decoders does that rename. A model family is supported only after
its translator and a same-weight parity test against the reference have landed.

The block holds its token mixer in a slot. Any module with the
(x, decode=..., positions=..., segment_ids=...) -> x signature of
CausalSelfAttention can be self_attn without changes to the block, and that is
where a linear-attention mixer goes.
"""

import dataclasses
import functools
import math
from collections.abc import Mapping, Sequence
from typing import Literal, Self

import flax.core
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.records import JSON
from dew.registry import from_record, mixers, models

from ..activations import ungated_activation
from ..attention import RMSNorm
from ..attention_residuals import AttentionResiduals, DepthAttention
from ..blocks import TokenEmbedding, normal_kernel
from ..deepseek_v4 import DeepseekV4Mixer
from ..dsa_kpool import KPoolSparseAttentionMixer
from ..dspark import DSpark, DSparkStage, draft as dspark_draft
from ..engram import Engram, EngramHashes, EngramLayer
from ..gemma3n import AltUp, rescale_to
from ..gemma4_moe import Gemma4Experts
from ..gpt_oss import GptOssMLP
from ..hyper_connections import (
    Carried,
    HyperConnections,
    HyperHead,
    collapse_by,
    collapse_streams,
    expand_streams,
    first_stream,
)
from ..inputs import AttentionMetadata, LayerInputs, PredictionPhase
from ..kv_cache import KVCache
from ..mixers import AttentionMixer, MixerBase, MixerContext
from ..mixers.mamba2 import Mamba2Mixer
from ..mla import MLAMixer
from ..moe import GatedActivation, Situ
from ..precision import at_least_fp32, head_dot_general, head_product, scaled
from ..protocols import OutputTable, ProjectionGroup, declared_groups
from ..rope import LongRopeScaling, RopeScaling, YarnScaling, rope_scaling_from_record
from ..sharding import (
    RESIDUAL,
    LayoutRefused,
    constrain,
    logical_axes,
    microbatches,
    pipeline_stages,
    row_axes,
)
from .decoder_block import (
    DECODER_REMAT,
    BlockWiring,
    DecoderBlock,
    GatedMLP,
    Mixture,
    MTPBlock,
    RematPolicy,
    decoder_norm,
    remat_policy,
    remat_record,
)
from .decoder_stack import DecoderBank, PipelineStage, StackView, _merged, run_pipeline, run_stack
from .layer_plan import LayerKind, LayerSpec, ResolvedKind, group_name, scan_groups

INTERMEDIATES = "intermediates"
"""The collection flax's `capture_intermediates` fills."""


def layer_outputs(module: nn.Module, method: str) -> bool:
    """The `capture_intermediates` filter that keeps every layer's output.

    It matches the `__call__` of the stack's `layers_N` modules, a scanned
    run's `layers_3_7` and a pipeline stage's `layers_j` included, and
    `StackView.unstack` lays what those sow out per layer, as the plain loop
    does; `layer_output` reads one layer back.
    """
    return method == "__call__" and (module.name or "").startswith("layers_")


def layer_output(intermediates: Mapping[str, Mapping[str, Sequence[jax.Array]]], index: int) -> jax.Array:
    """Layer `index`'s output out of the collection `layer_outputs` filled."""
    kept = intermediates.get(f"layers_{index}")
    if kept is None:
        held = sorted(int(name[len("layers_"):]) for name in intermediates
                      if name.startswith("layers_") and name[len("layers_"):].isdigit())
        raise ValueError(f"the model kept no layer {index}; it has layers {held}")
    return kept["__call__"][0]


@models("causal_transformer")
@logical_axes({
    ("embed_tokens",): ("vocab", "embed"),
    ("embed_positions",): (None, "embed"),
    ("lm_head",): ("embed", "vocab"),
    ("head_bias",): ("vocab",),
    ("embed_tokens_per_layer",): ("vocab", None),
    ("per_layer_model_projection",): ("embed", "mlp"),
    # AltUp's copies enter and leave through embed-by-embed projections, one
    # per copy past the first, named by their index like the layers; a
    # square kernel takes the shape heuristic the way the other indexed
    # projections do.
})
class CausalTransformer(nn.Module):
    """Decoder-only transformer over token ids: [B, S] int32 -> [B, S, vocab] fp32.

    The defaults train a model from scratch: multi-head attention, swiglu,
    tied embeddings, no softcap. Every setting that differs between open
    decoders is a field here, so loading Qwen3 or Gemma3 is a mapping of config
    fields and needs no subclass. The field comments name the family that sets
    each one. A classic GPT block is `norm_type='layer'`, `norm_bias`, an
    ungated `mlp`, `mlp_bias` and learned positions; the learned positions
    advance from each decode row's cache cursor.

    `layer_types` is the pattern, one kind per layer, and `kinds` says what
    each kind does. `kv_shared_layers` lists the layers that share by index: a
    trailing run of layers for Gemma 3n/4, or the layer after every indexer
    layer for GLM's IndexShare. A sharing layer reads what the last earlier
    non-sharing layer of its kind stored: keys and values for attention, or the
    indexer's selection for MLA.

    When attention dropout is active the model runs the reference kernel, and
    it refuses an explicitly chosen fused kernel that cannot drop
    probabilities.

    Interleaved mRoPE (Qwen3.5's mrope_section) is `partial_rotary_factor` for
    text. With one position per token the three grids' angles are equal, so
    text-only input reduces exactly to the partial rope; image-grid positions
    are not modelled. A `mixer` kind other than attention reads its own record,
    but the GQA geometry fields are still validated, so a translation has to
    fill them consistently.

    MTP depth d pairs the previous depth's state at position p with the
    embedding of the token at p + d, and scores the token that follows p + d
    (arXiv 2412.19437, section 2.2). Each depth is therefore one position
    shorter than the one before it.

    With `scan_layers`, each run of like layers (layers whose resolved specs
    give the same parameter shapes and computation) runs as one scanned body,
    so compile time stops growing with depth. The variables tree stays the
    unscanned one (`StackView`). `init` always draws a run of like layers under
    the scan, once per run. When the mesh has a stage axis larger than one, the
    stack runs as a pipeline.
    """
    vocab_size: int
    emb_features: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int | None = None       # None: as many as the query heads
    head_dim: int | None = None  # None: emb_features // num_heads
    mlp: GatedActivation = "swiglu"  # 'swiglu' | 'geglu' | 'geglu_exact' | 'swigluoai', or Kimi K3's Situ
    mlp_bias: bool = False
    """Whether both feed-forward projections have a bias. The `mlp` activations gelu,
    gelu_exact and relu give an ungated feed-forward."""
    mlp_features: int | tuple[int, ...] | None = None
    """The dense feed-forward width. None means four times `emb_features`, a
    tuple gives one width per layer (Gemma 3n), and 0 means no feed-forward
    (Mamba-2)."""
    max_seq_len: int = 2048
    position_embedding: Literal['rotary', 'learned', 'alibi'] = 'rotary'
    position_embedding_size: int | None = None
    """The number of rows in the learned position table; None means `max_seq_len`.
    It does not depend on the decode cache's capacity."""
    position_embedding_offset: int = 0
    """The number of reserved rows before learned position zero; OPT checkpoints have two."""
    rope_theta: float = 10000.0              # the base a kind does not override
    rope_scaling: RopeScaling | LongRopeScaling | None = None
    partial_rotary_factor: float | None = None  # None: every dim rotates
    partial_rotary_type: str = 'proportional'  # 'proportional' (Gemma 4) | 'default' (Qwen3.5)
    layer_types: tuple[str, ...] | None = None  # the pattern, one kind per layer
    kinds: Mapping[str, LayerKind] | None = None  # what each named kind does
    norm_eps: float = 1e-5
    norm_type: Literal['rms', 'layer'] = 'rms'
    norm_bias: bool = False
    embedding_norm: bool = False
    """Normalize token embeddings before the first block, as BLOOM and ModernBERT do."""
    first_attention_norm: bool = True
    """Whether the first block norms its attention input. ModernBERT's first
    block reads the embedding norm's output as it is (modeling_modernbert.py,
    ModernBertEncoderLayer: attn_norm is Identity at layer 0)."""
    scale_offset: bool = False       # RMSNorm weight is (1 + w), as Gemma stores it
    scale_after_cast: bool = False   # apply the weight after casting, as Llama and Qwen3 do
    sandwich_norms: bool = False     # add a norm after each sublayer, as Gemma does
    pre_norms: bool = True           # norm each sublayer's input; False + sandwich is OLMo 3
    parallel_residual: bool = False
    """Whether attention and the feed-forward both read the same residual, as in GPT-NeoX."""
    shared_parallel_norm: bool = False
    """Both parallel branches read one LayerNorm, as in Phi, Falcon-7B and GPT-J."""
    qk_norm: bool = True
    qk_norm_scope: str = 'head'              # 'head' per head (Qwen3); 'projection' whole (OLMo 3)
    v_norm: bool = False                     # Gemma 4's scale-free values norm
    attention_k_eq_v: bool = False           # Gemma 4's global layers read their values off the keys
    layer_scalar: Literal["frozen", "trainable"] | None = None
    attention_bias: bool = False             # q/k/v biases, and o_proj unless o_proj_bias says
    o_proj_bias: bool | None = None       # Qwen2 biases q/k/v while o_proj stays bias-free
    attention_scale: float | None = None  # None: head_dim ** -0.5
    attention_sinks: bool = False
    yarn: YarnScaling | None = None
    attn_logit_softcap: float | None = None  # Gemma 2's attn_logit_softcapping
    output_gate: bool = False                 # Qwen3.5 gates the attention branch
    embedding_scale: bool = False            # Gemma scales embeddings by sqrt(d)
    embedding_multiplier: float = 1.0
    """The factor that multiplies the token embeddings before the first layer:
    muP's m_emb in lm-engine, and GraniteMoeHybrid's `embedding_multiplier`."""
    residual_multiplier: float = 1.0
    """The factor that multiplies every sublayer output before it is added to the
    residual stream: lm-engine's m_residual, and GraniteMoeHybrid's
    `residual_multiplier`."""
    logits_scaling: float = 1.0
    """The divisor of the logits: lm-engine's m_width (`lm_logits *
    (1 / m_width)`), and GraniteMoeHybrid's `logits_scaling`. The division is
    applied to the final states, in fp32, so every head that contracts them
    with `head_weight` gets the same logits that `__call__` returns."""
    # lm-engine's init_utils.py at 45b6b57b.
    initializer_range: float | None = None
    """The std of lm-engine's initialisation; None keeps each module's own
    initializer.

    When it is set, the embedding table (and an untied head) is drawn from
    N(0, initializer_range^2), and every hidden matrix (attention and Mamba-2
    projections, conv taps, router, experts, dense MLPs) from
    N(0, (initializer_range / sqrt(logits_scaling))^2). Biases start at zero
    and norms at one. With `logits_scaling` set to m_width this is lm-engine's
    `init_method="mup"`, and with `logits_scaling` at 1 it is lm-engine's
    `"normal"`."""
    depth_scaled_init: bool = False
    """Whether, with `initializer_range` set, the projections back into the
    residual stream (o_proj, out_proj, down_proj) divide their std by
    sqrt(2 * num_layers). This is lm-engine's `use_depth_scaled_init`."""
    final_logit_softcap: float | None = None
    head_transform: Literal['gelu', 'gelu_exact'] | None = None
    """The activation of a BERT-style prediction head between the final norm
    and the vocabulary projection: a bias-free dense layer, this activation and
    a norm of the model's kind (ModernBertPredictionHead,
    modeling_modernbert.py). None projects the final states directly. As in the
    feed-forward, 'gelu' is the tanh form and 'gelu_exact' the erf form.
    `hidden_states` stays the final norm's output, the encoder's."""
    head_bias: bool = False
    """A vocabulary bias added in fp32 after a tied or untied head's product, as
    Phi's, GPT-J's and ModernBERT's decoders add one (`vocabulary_bias`)."""
    tie_embeddings: bool = True
    # Kimi K2.5's text-only wrapper path, modeling_kimi_k25.py:686-690.
    embedding_zero_ids: tuple[int, ...] = ()
    """Placeholder ids that the embedding lookup reads as token zero. The labels
    keep the original ids."""
    dropout_rate: float = 0.0
    embedding_dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    attention_impl: str = "auto"  # an AttentionImpl
    kv_cache: KVCache = KVCache()
    """How attention layers store the decode cache: dense or paged, full or
    quantized (`dew.nn.kv_cache`). The parameters do not depend on it."""
    mixture: Mixture | None = None        # some layers' feed-forward as `moe.SparseMLP`; None: dense
    use_double_wide_mlp: bool = False        # Gemma 4 doubles sharing layers' MLP width; needs sharing
    causal: bool = True
    """Whether attention is causal. False gives full attention, the encoder that
    a masked diffusion language model denoises with. The parameter tree is the
    same either way."""
    per_layer_input_dim: int | None = None
    """The width of Gemma 3n/4 per-layer inputs, or None for a model without
    them. Each layer reads its own slice of an extra table and adds it to its
    input through its own gate."""
    per_layer_input_vocab: int | None = None  # None: vocab_size
    kv_shared_layers: tuple[int, ...] | None = None  # layers reusing a provider's K/V; None disables
    mixer: MixerBase | None = None         # None: today's attention; a kind value or its record
    num_nextn_predict_layers: int = 0         # MTP depths; their input/residual policy is independent below
    index_share_for_mtp_iteration: bool = False
    mtp_layer_type: str | None = None
    """The layer kind the prediction depths build their mixer from. It need not
    occur in the trunk. None uses the full-attention kind when the pattern has
    one, and the first layer's kind otherwise."""
    mtp_hyper_connections: HyperConnections | None = None
    """The residual streams of the prediction depths. None gives the depths
    plain residuals and normalized trunk inputs. Setting it opts the depth into
    streams, separately from the trunk's `hyper_connections`."""
    altup: AltUp | None = None             # Gemma 3n's residual copies (`dew.nn.gemma3n`); None disables
    laurel_rank: int | None = None         # Gemma 3n's learned augmented residual; None disables
    hyper_connections: HyperConnections | None = None  # mHC's stack of residual streams; None disables
    attention_residuals: AttentionResiduals | None = None
    """Kimi K3's attention residuals (`dew.nn.attention_residuals`): a softmax
    over finished blocks of layers that replaces the running residual. None
    disables them."""
    engram: Engram | None = None
    """DeepSeek-V4.1's n-gram lookups (`dew.nn.engram`). Each layer it names
    gates its table rows into the residual streams before its attention. The
    tokenizer's compressed vocabulary is stored as `engram_hashes/token_map` in
    the `constants` collection. A loaded checkpoint derives it from its
    tokenizer, and a fresh model fills it with each id modulo the compressed
    vocabulary's size."""
    dspark: DSpark | None = None
    """DeepSeek's block drafter (`dew.nn.dspark`, V4.1 and V4-Flash-0731), or
    None for a model without one. Its stages are decoder blocks that attend
    through a `DSparkAttention` built from the V4 attention of its
    `layer_type` kind. Its target layers sow the stream means that the
    drafter reads (`draft_context`)."""
    swiglu_limit: float | None = None  # GLM-5.3-Flash's clamp before every gated MLP's activation
    activation_sparsity_pattern: tuple[float, ...] | None = None
    """Gemma 3n's gaussian top-k activation sparsity, one fraction per layer."""
    mask_token_id: int | None = None
    """The vocabulary id that a masked-diffusion objective corrupts tokens to.
    None means plain training."""
    scan_layers: bool = False                 # runs of like layers under flax's scan
    bank_layers: int | None = None
    """The most layers one scanned run holds, which is also how many layers its
    parameter bank stacks. A longer run of like layers is split into
    consecutive runs of at most this many, each with its own bank under its own
    name. None keeps a whole run in one bank. `scan_layers` and `init` read it,
    and because init draws each run under one scan, it also changes what init
    draws for a given seed. The split bounds the memory needed to build a
    host-resident bank and read it back, so set it for a deep stack that is
    offloaded to the host."""
    remat: RematPolicy | None = None
    """The rematerialization policy. Each block is recomputed in the backward
    pass, keeping its inputs, any K/V passed to later layers, and the residuals
    the policy names. A config gives a name from `REMAT_POLICIES` or a record of
    residual names to save or offload. None recomputes nothing. Init and cached
    decode call the blocks directly, and the stored parameters have the same
    layout either way."""

    def __post_init__(self):
        if self.layer_scalar not in (None, "frozen", "trainable"):
            raise ValueError("layer_scalar must be None, frozen or trainable")
        if self.layer_types is not None:
            object.__setattr__(self, "layer_types", tuple(self.layer_types))
        object.__setattr__(self, "embedding_zero_ids", tuple(self.embedding_zero_ids))
        if self.kv_shared_layers is not None:
            object.__setattr__(self, "kv_shared_layers",
                               tuple(int(index) for index in self.kv_shared_layers))
        if isinstance(self.mlp_features, (tuple, list)):
            object.__setattr__(self, "mlp_features", tuple(int(width) for width in self.mlp_features))
        if self.activation_sparsity_pattern is not None:
            object.__setattr__(self, "activation_sparsity_pattern",
                               tuple(float(fraction) for fraction in self.activation_sparsity_pattern))
        # A value arrives as a record from a config and as itself from code,
        # and `models.build` already reads one; doing it here too means the
        # plain constructor takes the same records, as a test or a notebook
        # writes them.
        for name, record in (("altup", AltUp), ("hyper_connections", HyperConnections),
                             ("mtp_hyper_connections", HyperConnections),
                             ("attention_residuals", AttentionResiduals), ("engram", Engram),
                             ("dspark", DSpark), ("yarn", YarnScaling), ("mixture", Mixture)):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, record(**value))
        if isinstance(self.rope_scaling, Mapping):
            object.__setattr__(self, 'rope_scaling', rope_scaling_from_record(self.rope_scaling))
        if isinstance(self.mlp, Mapping):
            # A config states SiTU's betas as a record in the activation's place.
            object.__setattr__(self, "mlp", from_record(Situ, self.mlp))
        if self.kinds is not None:
            # Frozen, because a module's fields are static to jit and a plain
            # dict cannot be hashed.
            object.__setattr__(self, "kinds", flax.core.freeze({
                name: kind if isinstance(kind, LayerKind) else LayerKind(**kind)
                for name, kind in self.kinds.items()}))
        # A mixer arrives as a kind value from code and as a {"name": ...,
        # "fields": ...} record from a config; the record dispatches on its
        # name through the same `mixers.build` a value is constructed with,
        # so an unknown kind or field raises either way. Anything else is neither.
        if isinstance(self.mixer, Mapping):
            object.__setattr__(self, "mixer", mixers.from_record(self.mixer))
        elif self.mixer is not None and not isinstance(self.mixer, MixerBase):
            raise ValueError(
                f"mixer is a mixer value, its record, or None, not {self.mixer!r}")
        object.__setattr__(self, "remat", remat_policy(self.remat))
        super().__post_init__()

    @property
    def init_stds(self) -> tuple[float | None, float | None]:
        """The normal std of the hidden matrices and of the projections back
        into the residual stream. Both are None when `initializer_range` is
        None, and the modules then use their own initializers."""
        if self.initializer_range is None:
            return None, None
        hidden = self.initializer_range / math.sqrt(self.logits_scaling)
        output = hidden / math.sqrt(2 * self.num_layers) if self.depth_scaled_init else hidden
        return hidden, output

    @property
    def kv_heads(self) -> int:
        return self.num_heads if self.num_kv_heads is None else self.num_kv_heads

    @property
    def features_per_head(self) -> int:
        return (self.emb_features // self.num_heads
                if self.head_dim is None else self.head_dim)

    @property
    def mlp_widths(self) -> tuple[int, ...]:
        """Each layer's dense feed-forward width."""
        if self.mlp_features is None:
            return (4 * self.emb_features,) * self.num_layers
        if isinstance(self.mlp_features, tuple):
            return self.mlp_features
        return (self.mlp_features,) * self.num_layers

    @property
    def hidden_features(self) -> int:
        """The model's single feed-forward width. The routed experts fall back
        to it and the prediction depths use it. A model whose layers have
        different widths has none and raises `ValueError`."""
        widths = set(self.mlp_widths)
        if len(widths) != 1:
            raise ValueError(
                f"mlp_features names a width per layer ({self.mlp_features}), so "
                f"the model has no single feed-forward width; set the mixture's "
                f"expert_features and leave the prediction depths off")
        return widths.pop()

    @property
    def per_layer_types(self) -> tuple[str, ...]:
        if self.layer_types is None:
            return ('full_attention',) * self.num_layers
        return tuple(self.layer_types)

    def kind_of(self, layer_type: str) -> "ResolvedKind":
        """Return the `ResolvedKind` of `layer_type`, with the model's defaults where the kind is unset."""
        kind = (self.kinds or {}).get(layer_type, LayerKind())
        return ResolvedKind(
            window=kind.window,
            bidirectional_window=kind.bidirectional_window,
            chunk=kind.chunk,
            num_kv_heads=self.kv_heads if kind.num_kv_heads is None else kind.num_kv_heads,
            rope_theta=self.rope_theta if kind.rope_theta is None else kind.rope_theta,
            rope_scaling=self.rope_scaling if kind.rope_scaling is None else kind.rope_scaling,
            yarn=self.yarn if kind.yarn is None else kind.yarn,
            head_dim=(self.features_per_head if kind.head_dim is None else kind.head_dim),
            mixer=kind.mixer)

    @property
    def hash_layers(self) -> set:
        """The sparse layers that route by the mixture's token table."""
        mixture = self.mixture
        return set() if mixture is None or mixture.hash_layers is None else set(mixture.hash_layers)

    @property
    def sparse_layers(self) -> tuple[int, ...]:
        """The layers whose feed-forward routes to experts."""
        mixture = self.mixture
        if mixture is None:
            return ()
        return tuple(range(self.num_layers)) if mixture.layers is None else mixture.layers

    @property
    def sharing_layers(self) -> tuple[int, ...]:
        """The sorted indices of the layers that read what another layer stored."""
        if self.kv_shared_layers is None:
            return ()
        outside = sorted(index for index in self.kv_shared_layers
                         if not 0 <= index < self.num_layers)
        if outside:
            raise ValueError(
                f"kv_shared_layers {outside} name no layer of a "
                f"{self.num_layers}-layer model")
        return tuple(sorted(set(self.kv_shared_layers)))

    @property
    def kv_sharing(self) -> dict:
        """A map from each sharing layer's index to the provider layer it reads.

        A sharing layer has no K/V of its own (and, for MLA, no indexer). It
        reads the last earlier non-sharing layer of its own type. The map is
        empty unless `kv_shared_layers` is set, and a sharing layer with no
        earlier provider raises `ValueError`.
        """
        # As Gemma4TextAttention does in modeling_gemma4.py. GLM's IndexShare
        # carries the last full layer's top-k forward the same way
        # (modeling_glm_moe_dsa.py:739-748).
        sharing = set(self.sharing_layers)
        types = self.per_layer_types
        providers = {}
        for index in sorted(sharing):
            earlier = [j for j in range(index)
                       if j not in sharing and types[j] == types[index]]
            if not earlier:
                raise ValueError(
                    f"layer {index} shares K/V but no earlier {types[index]} layer "
                    "exists to provide them")
            providers[index] = earlier[-1]
        return providers

    @property
    def per_layer_vocab(self) -> int:
        return self.vocab_size if self.per_layer_input_vocab is None else self.per_layer_input_vocab

    def mixer_context(self, kind: "ResolvedKind", layer_type: str,
                      kv_shared: bool) -> MixerContext:
        """Return one layer's mixer geometry as a `MixerContext` built from the resolved kind.

        `head_dim`, `rope_theta`, `window` and the two rotary ramps take the
        layer kind's overrides. A windowed kind rotates every dimension, so the
        partial rotary applies only to the kinds that attend the whole
        sequence, which is where Gemma 4 puts it. Every other field is the
        model's field of the same name. A kind builds its `DecoderBlock` factory
        from this context and its own record, and `setup` chooses the mixer
        there and nowhere else.
        """
        resolved = {
            "num_kv_heads": kind.num_kv_heads, "head_dim": kind.head_dim,
            "rope_theta": kind.rope_theta, "rope_scaling": kind.rope_scaling, "yarn": kind.yarn,
            "sliding_window": kind.window, "bidirectional_window": kind.bidirectional_window,
            "attention_chunk": kind.chunk,
            "k_eq_v": self.attention_k_eq_v and kind.window is None,
            "kv_shared": kv_shared, "kv_store_key": layer_type,
            # LongRoPE's factors fix the rotated width on every layer;
            # Gemma's windowed layers otherwise rotate whole heads.
            "partial_rotary_factor": (self.partial_rotary_factor if kind.window is None
                                      or isinstance(kind.rope_scaling, LongRopeScaling) else None),
            "init_std": self.init_stds[0], "output_init_std": self.init_stds[1]}
        return MixerContext(**resolved, **{field.name: getattr(self, field.name)
                                           for field in dataclasses.fields(MixerContext)
                                           if field.name not in resolved})

    @property
    def bank_sites(self) -> tuple[DecoderBank, ...]:
        """Where this decoder's layer stack is stored: scanned runs, or one layer per bank.

        `groups` already records which of the two it is. A scanned stack
        declares its runs and a plain loop declares one-layer runs, and
        `run_stack` fetches a one-layer run the same way it fetches a longer
        one. `scanned` records whether those fetches are sequenced, which a host
        layout requires.
        """
        bound = self if self.scope is not None else self.bind({})
        return (DecoderBank((), StackView(bound.groups), scanned=self.scan_layers),)

    def layer_kinds(self, types: Sequence[str]) -> dict[str, "ResolvedKind"]:
        """Resolve every kind the pattern names, plus the kinds of the prediction depths and the drafter.

        Raises `ValueError` when the pattern does not have one kind per layer,
        or when `kinds` describes a kind that no layer has.
        """
        if len(types) != self.num_layers:
            raise ValueError(
                f"layer_types has {len(types)} entries for {self.num_layers} layers")
        prediction_kinds = (
            {self.mtp_layer_type} if self.num_nextn_predict_layers and self.mtp_layer_type else set()
        )
        if self.dspark is not None:
            prediction_kinds.add(self.dspark.layer_type)
        unnamed = sorted(set(self.kinds or {}) - set(types) - prediction_kinds)
        if unnamed:
            raise ValueError(
                f"kinds {unnamed} name no layer of this model, whose pattern is "
                f"{sorted(set(types))}")
        return {layer_type: self.kind_of(layer_type) for layer_type in set(types) | prediction_kinds}

    def refuse_unbuildable_mup(self, kinds: Mapping[str, "ResolvedKind"]):
        """Raise `ValueError` if lm-engine's multipliers or initialisation are set on a model that
        cannot apply them."""
        mup = (self.embedding_multiplier != 1.0 or self.residual_multiplier != 1.0
               or self.logits_scaling != 1.0 or self.initializer_range is not None)
        if mup and (self.num_nextn_predict_layers or self.hyper_connections is not None
                    or (self.mixture is not None and self.mixture.parallel)
                    or self.mlp == 'swigluoai' or self.laurel_rank is not None
                    or self.altup is not None or self.per_layer_input_dim):
            raise ValueError(
                "embedding_multiplier, residual_multiplier, logits_scaling and "
                "initializer_range are lm-engine's dense and routed blocks; prediction "
                "depths, hyper-connection streams, Gemma 4's parallel experts, gpt-oss "
                "experts, LAuReL, AltUp and per-layer inputs do not carry them")
        if self.logits_scaling <= 0 or (self.initializer_range is not None
                                        and self.initializer_range <= 0):
            raise ValueError("logits_scaling and initializer_range are positive")
        if self.initializer_range is not None:
            for layer_type, kind in sorted(kinds.items()):
                mixer = kind.mixer or self.mixer
                if mixer is not None and not isinstance(mixer, (AttentionMixer, Mamba2Mixer)):
                    raise ValueError(
                        f"initializer_range draws the attention and mamba2 mixers' "
                        f"projections; {layer_type!r} runs {type(mixer).__name__}, "
                        f"which keeps its own initializers")
        elif self.depth_scaled_init:
            raise ValueError("depth_scaled_init scales initializer_range's std; set it")

    def refuse_unbuildable_dropout(self, kinds: Mapping[str, "ResolvedKind"]) -> None:
        """Raise `ValueError` for a dropout rate outside [0, 1), or for attention dropout with a mixer that
        cannot drop probabilities."""
        for field in ("embedding_dropout_rate", "attention_dropout_rate"):
            rate = getattr(self, field)
            if not 0 <= rate < 1:
                raise ValueError(f"{field} must be within [0, 1), got {rate}")
        if self.attention_dropout_rate:
            for kind in kinds.values():
                if not isinstance(kind.mixer or self.mixer or AttentionMixer(), AttentionMixer):
                    raise ValueError("attention_dropout_rate requires ordinary attention mixers")

    def refuse_unbuildable_classic_fields(self):
        """Raise `ValueError` for norm, position and feed-forward combinations that no supported
        model uses."""
        if self.norm_type not in ('rms', 'layer'):
            raise ValueError(f'norm_type must be rms or layer, got {self.norm_type!r}')
        if self.norm_type == 'rms' and self.norm_bias:
            raise ValueError('norm_bias requires LayerNorm')
        if not self.first_attention_norm and not self.pre_norms:
            raise ValueError('first_attention_norm=False drops a pre-norm, so it requires pre_norms')
        if self.head_transform not in (None, 'gelu', 'gelu_exact'):
            raise ValueError("head_transform must be None, 'gelu' or 'gelu_exact'")
        if self.norm_type == 'layer' and self.scale_offset:
            raise ValueError('scale_offset describes RMSNorm weights')
        if self.position_embedding not in ('rotary', 'learned', 'alibi'):
            raise ValueError('position_embedding must be rotary, learned or alibi')
        if self.position_embedding != "rotary" and (
            self.mixer is not None or any(kind.mixer is not None for kind in (self.kinds or {}).values())
        ):
            raise ValueError('learned positions and ALiBi require the default unrotated attention mixer')
        if self.position_embedding_size is not None and (
                self.position_embedding != 'learned' or self.position_embedding_size < self.max_seq_len):
            raise ValueError('position_embedding_size requires learned positions and covers max_seq_len')
        if self.position_embedding_offset < 0 or (self.position_embedding_offset and (
                self.position_embedding != 'learned' or self.position_embedding_size is None
                or self.position_embedding_size < self.max_seq_len + self.position_embedding_offset)):
            raise ValueError('position_embedding_offset requires learned table rows past max_seq_len')
        if self.position_embedding != "rotary" and (
            self.partial_rotary_factor is not None or self.rope_scaling is not None or self.yarn is not None
        ):
            raise ValueError('rotary scaling requires rotary positions')
        if self.mixture is not None and (self.mlp_bias or self.mlp in ('gelu', 'gelu_exact', 'relu')):
            raise ValueError('the routed experts require a bias-free gated MLP')
        if self.num_nextn_predict_layers and (self.norm_type != 'rms' or self.position_embedding != 'rotary'):
            raise ValueError('prediction depths require RMSNorm and rotary positions')

    def refuse_unbuildable_kinds(self, kinds: Mapping[str, "ResolvedKind"]):
        """Raise `ValueError` for a kind with invalid rotary, local-mask or grouped-head geometry."""
        for layer_type, kind in sorted(kinds.items()):
            if self.position_embedding == 'rotary' and kind.head_dim % 2:
                raise ValueError(
                    "rotary positions rotate pairs, so the head dim of "
                    f"{layer_type!r} must be even, got {kind.head_dim}")
            if kind.window is not None and kind.window < 1:
                raise ValueError(
                    f"the window of {layer_type!r} must be positive, got {kind.window}")
            if kind.chunk is not None and kind.chunk < 1:
                raise ValueError(
                    f"the chunk of {layer_type!r} must be positive, got {kind.chunk}")
            if kind.chunk is not None and kind.window is not None:
                raise ValueError(
                    f"{layer_type!r} sets a window and a chunk; a local layer "
                    f"reads one or the other")
            if kind.chunk is not None and not self.causal:
                raise ValueError(
                    f"{layer_type!r} chunks a causal layer's keys, and this model "
                    f"is not causal")
            if kind.bidirectional_window and (self.causal or kind.window is None):
                raise ValueError(
                    f"{layer_type!r} keeps a window on both sides of a query, which "
                    f"needs a window and a model that is not causal")
            if kind.num_kv_heads < 1 or self.num_heads % kind.num_kv_heads:
                raise ValueError(
                    f"num_heads ({self.num_heads}) must be a multiple of the key/value "
                    f"heads of {layer_type!r} ({kind.num_kv_heads})")

    def refuse_unbuildable_fields(self, kinds: Mapping[str, "ResolvedKind"]):
        """Raise `ValueError` for a field, or a pair of fields, that this model cannot build.

        Each error names the field the caller set and what a model without it
        looks like, so whoever translated the config can see which entry to fix.
        """
        self.refuse_unbuildable_mup(kinds)
        self.refuse_unbuildable_dropout(kinds)
        self.refuse_unbuildable_classic_fields()
        mtp_hc = self.mtp_hyper_connections
        if mtp_hc is not None:
            if self.num_nextn_predict_layers != 1:
                raise ValueError("mtp_hyper_connections requires a single prediction depth")
            if mtp_hc.head != 'weighted':
                raise ValueError("mtp_hyper_connections requires the V4 weighted stream-collapse head")
            if self.hyper_connections is None or mtp_hc.hc_mult != self.hyper_connections.hc_mult:
                raise ValueError("mtp_hyper_connections must match the trunk's residual stream count")
        self.refuse_unbuildable_kinds(kinds)
        if self.num_heads % self.kv_heads:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be a multiple of num_kv_heads "
                f"({self.kv_heads})")
        if self.partial_rotary_factor is not None and not 0 < self.partial_rotary_factor <= 1:
            raise ValueError(
                "partial_rotary_factor must be within (0, 1], "
                f"got {self.partial_rotary_factor}")
        if self.partial_rotary_type not in ('proportional', 'default'):
            raise ValueError(
                "partial_rotary_type names the convention of a partial rotary, "
                f"'proportional' or 'default', got {self.partial_rotary_type!r}")
        sparse = self.sparse_layers
        outside = sorted(index for index in sparse
                         if not 0 <= index < self.num_layers)
        if outside:
            raise ValueError(
                f"the mixture's layers {outside} are outside the "
                f"{self.num_layers} layers of this model")
        hashed = self.hash_layers
        if hashed - set(sparse):
            raise ValueError(
                f"the mixture's hash_layers {sorted(hashed - set(sparse))} are not "
                "among its sparse layers, which are the ones with experts to route")
        sharing = self.kv_sharing
        if self.use_double_wide_mlp and not sharing:
            raise ValueError(
                "use_double_wide_mlp widens the MLP of the layers that share "
                "their keys and values, so it needs kv_shared_layers set")
        ple = self.per_layer_input_dim
        if ple is not None and ple < 1:
            raise ValueError(
                f"per_layer_input_dim is the width of a layer's own input, got "
                f"{ple}; None is a model without them")
        if self.num_nextn_predict_layers < 0:
            raise ValueError(
                f"num_nextn_predict_layers counts prediction depths, got "
                f"{self.num_nextn_predict_layers}; 0 is a model without them")
        widths = self.mlp_widths
        if len(widths) != self.num_layers or min(widths) < 0:
            raise ValueError(
                f"mlp_features names one width per layer of {self.num_layers}, "
                f"0 for a layer without a feed-forward, got {self.mlp_features}")
        sparsity = self.activation_sparsity_pattern
        if sparsity is not None and (len(sparsity) != self.num_layers
                                     or not all(0 <= fraction < 1 for fraction in sparsity)):
            raise ValueError(
                f"activation_sparsity_pattern names one fraction within [0, 1) per "
                f"layer of {self.num_layers}, got {sparsity}")
        if self.altup is not None and self.num_nextn_predict_layers:
            raise ValueError(
                "altup carries a stack of residual copies through the layers and "
                "the prediction depths read one, so a model has one or the other")
        if self.altup is not None and self.hyper_connections is not None:
            raise ValueError(
                "altup and hyper_connections each carry their own stack of residual "
                "copies through the layers, so a model has one or the other")
        self.refuse_unbuildable_stream_fields()
        if self.swiglu_limit is not None and self.swiglu_limit <= 0:
            raise ValueError(
                f"swiglu_limit caps the gate and up projections, so it is positive, "
                f"got {self.swiglu_limit}; None leaves them unclamped")
        mask = self.mask_token_id
        if mask is not None and (isinstance(mask, bool) or not isinstance(mask, int) or mask < 0):
            raise ValueError(
                f"mask_token_id names a vocabulary id, got {mask!r}; None is a "
                "model trained on plain next-token prediction")

    def refuse_unbuildable_stream_fields(self):
        """Raise `ValueError` for settings on mHC's residual streams that cannot be
        built: engram's layers, DSpark's stages and the Single-Pass schedule."""
        engram, hc = self.engram, self.hyper_connections
        if engram is not None:
            if hc is None:
                raise ValueError("engram gates its lookup into each of mHC's residual streams, "
                                 "so it needs hyper_connections")
            outside = sorted(set(engram.layer_ids) - set(range(self.num_layers)))
            if outside:
                raise ValueError(f"engram layers {outside} are outside the {self.num_layers} layers")
        if self.dspark is not None:
            # The stages are the trunk's mHC blocks, V4.1's Single-Pass ones
            # (v41:1100-1156) or V4's with a learned head on the last stage
            # (V4-Flash-0731 model.py:818-874); a drafter for another trunk's
            # residual is not built here.
            if hc is None or not (hc.single_pass or hc.head == 'weighted') or self.mixture is None:
                raise ValueError("the DSpark drafter chains mHC blocks over routed experts and "
                                 "collapses them by Single-Pass's carried pre (V4.1) or by a "
                                 "learned head (V4), as the trunk it drafts for does")
            outside = sorted(set(self.dspark.target_layers) - set(range(self.num_layers)))
            if outside:
                raise ValueError(f"DSpark target layers {outside} are outside the {self.num_layers} layers")
            if self.num_nextn_predict_layers:
                raise ValueError("a model drafts with DSpark or with MTP depths, not both")
        if hc is not None and hc.single_pass and self.mtp_hyper_connections:
            raise ValueError("Single-Pass mHC carries the next sublayer's collapse beside the "
                             "streams, which no stream prediction depth reads")

    def feedforward_factories(self):
        """Build the three feed-forward factories a layer can use.

        Returns the dense gated MLP, the routed experts and Gemma 4's parallel
        branch, each a partial that the block calls with a name. A model with
        no mixture gets only the first, and None for the other two.
        """
        init_std, output_init_std = self.init_stds
        # Every gated MLP in the model shares the activation and the clamp:
        # the dense feed-forwards, the shared branch and the routed experts.
        gated_mlp = functools.partial(GatedMLP, out_features=self.emb_features,
                                      activation=self.mlp, use_bias=self.mlp_bias,
                                      swiglu_limit=self.swiglu_limit,
                                      init_std=init_std, output_init_std=output_init_std,
                                      dtype=self.dtype, precision=self.precision)
        mixture = self.mixture
        if self.mlp == 'swigluoai' and (mixture is None or mixture.shared_features
                                        or len(self.sparse_layers) != self.num_layers):
            raise ValueError('swigluoai requires routed experts on every layer and no shared experts')
        if mixture is None:
            return gated_mlp, None, None
        expert_features = (self.hidden_features if mixture.expert_features is None
                           else mixture.expert_features)
        parallel = None if not mixture.parallel else functools.partial(
            Gemma4Experts,
            num_experts=mixture.experts,
            top_k=mixture.top_k,
            hidden_features=expert_features,
            out_features=self.emb_features,
            activation=self.mlp,
            implementation=mixture.implementation,
            dispatch=mixture.dispatch,
            capacity_factor=mixture.capacity_factor,
            norm_eps=self.norm_eps,
            scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast,
            dtype=self.dtype,
            precision=self.precision)
        if self.mlp == 'swigluoai':
            routed = functools.partial(
                GptOssMLP, hidden_size=self.emb_features,
                intermediate_size=self.hidden_features,
                num_local_experts=mixture.experts, num_experts_per_tok=mixture.top_k,
                implementation=mixture.implementation,
                dispatch=mixture.dispatch,
                capacity_factor=mixture.capacity_factor,
                dtype=self.dtype, precision=self.precision)
        elif mixture.parallel:
            # The branch rides beside every sparse layer's dense feed-forward.
            routed = None
        else:
            routed = mixture.build(
                out_features=self.emb_features, hidden_features=self.hidden_features, activation=self.mlp,
                dense=gated_mlp, swiglu_limit=self.swiglu_limit,
                init_std=init_std, output_init_std=output_init_std,
                norm=functools.partial(
                    RMSNorm, epsilon=self.norm_eps, scale_offset=self.scale_offset,
                    scale_after_cast=self.scale_after_cast, dtype=self.dtype),
                dtype=self.dtype,
                precision=self.precision)
        return gated_mlp, routed, parallel

    def setup(self):
        """Build the embeddings, the layers, the prediction depths `mtp`, the final
        `norm`, and `lm_head` when the embeddings are not tied.

        Each layer has one `LayerSpec`, is built by `block`, and is scanned in
        `groups`.
        """
        types = self.per_layer_types
        kinds = self.layer_kinds(types)
        self.refuse_unbuildable_fields(kinds)
        ple = self.per_layer_input_dim

        self.embed_tokens = TokenEmbedding(
            num_embeddings=self.vocab_size, features=self.emb_features,
            dtype=self.dtype, name='embed_tokens',
            embedding_init=(TokenEmbedding.embedding_init if self.initializer_range is None
                            else nn.initializers.normal(self.initializer_range)))
        if self.embedding_dropout_rate:
            self.embedding_dropout = nn.Dropout(self.embedding_dropout_rate, name="embedding_dropout")
        if self.embedding_norm:
            self.embedding_layernorm = decoder_norm(
                self.norm_type, epsilon=self.norm_eps, bias=self.norm_bias,
                scale_offset=self.scale_offset, scale_after_cast=self.scale_after_cast,
                dtype=self.dtype)(name='embedding_layernorm')
        if self.position_embedding == 'learned':
            self.embed_positions = TokenEmbedding(
                num_embeddings=self.position_embedding_size or self.max_seq_len,
                features=self.emb_features, dtype=self.dtype, name='embed_positions',
                embedding_init=(TokenEmbedding.embedding_init if self.initializer_range is None
                                else nn.initializers.normal(self.initializer_range)))
        if ple:
            # The packed table every layer reads its own slice of
            # (modeling_gemma4.py, Gemma4TextModel): one row per token, a
            # hidden_size_per_layer_input slice per layer.
            self.embed_tokens_per_layer = TokenEmbedding(
                num_embeddings=self.per_layer_vocab, features=self.num_layers * ple,
                dtype=self.dtype, name='embed_tokens_per_layer')
            self.per_layer_model_projection = nn.Dense(
                self.num_layers * ple, use_bias=False,
                dtype=self.dtype, precision=self.precision,
                name='per_layer_model_projection')
            self.per_layer_projection_norm = RMSNorm(
                epsilon=self.norm_eps, scale_offset=self.scale_offset,
                scale_after_cast=self.scale_after_cast, dtype=self.dtype,
                name='per_layer_projection_norm')
        gated_mlp, routed, parallel = self.feedforward_factories()
        # None is today's attention; a kind names its own mixer on LayerKind
        # and otherwise rides the model's. Both build over the layer's
        # context.
        mixer_spec = self.mixer if self.mixer is not None else AttentionMixer(
            nope=self.position_embedding != 'rotary', alibi=self.position_embedding == 'alibi')
        specs = self._layer_specs(types, kinds, mixer_spec)
        wiring = BlockWiring(pre_norms=self.pre_norms, output_norms=self.sandwich_norms,
                             layer_scalar=self.layer_scalar, parallel_residual=self.parallel_residual,
                             shared_parallel_norm=self.shared_parallel_norm)

        block = functools.partial(self._block, specs, mixer_spec, (gated_mlp, routed, parallel), wiring)

        self.specs = specs
        self.block = block
        self.layers = [block(index, f'layers_{index}') for index in range(self.num_layers)]
        self.groups = scan_groups(specs, self.bank_layers) if self.scan_layers else tuple(
            (index, 1) for index in range(self.num_layers))
        self.mtp = self._prediction_depths(types, kinds, mixer_spec, gated_mlp, routed, wiring)
        if self.dspark is not None:
            assert routed is not None
            self.dspark_stages = self._dspark_stages(kinds, mixer_spec, routed, wiring)
        if self.altup is not None:
            # The copies past the first enter through their own projections
            # and leave through their own (modeling_gemma3n.py,
            # Gemma3nTextModel.altup_projections and altup_unembed_projections).
            projection = functools.partial(
                nn.Dense, self.emb_features, use_bias=False, dtype=self.dtype,
                precision=self.precision)
            self.altup_projections = [
                projection(name=f'altup_projections_{index}')
                for index in range(self.altup.num_inputs - 1)]
            self.altup_unembed_projections = [
                projection(name=f'altup_unembed_projections_{index}')
                for index in range(self.altup.num_inputs - 1)]
        if self.hyper_connections is not None and self.hyper_connections.head == 'weighted':
            self.hc_head = HyperHead(spec=self.hyper_connections, emb_features=self.emb_features,
                                     norm_eps=self.norm_eps, name='hc_head')
        if self.attention_residuals is not None:
            self.output_res = DepthAttention(emb_features=self.emb_features, norm_eps=self.norm_eps,
                                             name='output_res')
        if self.engram is not None:
            self.engram_hashes = EngramHashes(spec=self.engram, vocab_size=self.vocab_size,
                                              name='engram_hashes')
        self.norm = decoder_norm(
            self.norm_type, epsilon=self.norm_eps, bias=self.norm_bias,
            scale_offset=self.scale_offset, scale_after_cast=self.scale_after_cast,
            dtype=self.dtype)(name='norm')
        if not self.tie_embeddings:
            self.lm_head = nn.Dense(
                features=self.vocab_size, use_bias=False, dtype=at_least_fp32(self.dtype),
                precision=self.precision,
                dot_general=head_dot_general(self.dtype, self.precision),
                name='lm_head', **normal_kernel(self.initializer_range))
        if self.head_transform is not None:
            self.head_dense = nn.Dense(
                self.emb_features, use_bias=False, dtype=self.dtype, precision=self.precision,
                name='head_dense', **normal_kernel(self.initializer_range))
            self.head_norm = decoder_norm(
                self.norm_type, epsilon=self.norm_eps, bias=self.norm_bias,
                scale_offset=self.scale_offset, scale_after_cast=self.scale_after_cast,
                dtype=self.dtype)(name='head_norm')
        if self.head_bias:
            self.head_bias_value = self.param(
                'head_bias', nn.initializers.zeros_init(), (self.vocab_size,), jnp.float32)

    @nn.nowrap
    def _layer_specs(self, types: Sequence[str], kinds: dict[str, ResolvedKind], mixer_spec
                     ) -> tuple[LayerSpec, ...]:
        """One `LayerSpec` per layer: its kind, whether it routes and how,
        its feed-forward width and sparsity, the KV it shares or provides,
        and its attention-residual, engram and DSpark slots."""
        sparse, hashed, sharing = self.sparse_layers, self.hash_layers, self.kv_sharing
        widths, sparsity = self.mlp_widths, self.activation_sparsity_pattern
        providers = set(sharing.values())

        def provides(index: int, layer_type: str) -> bool:
            mixer = kinds[layer_type].mixer or mixer_spec
            return index in providers or (isinstance(mixer, DeepseekV4Mixer)
                                          and mixer.publishes(index in sharing))

        return tuple(
            LayerSpec(
                layer_type=layer_type,
                kind=kinds[layer_type],
                routed=index in sparse,
                hash_routed=index in hashed,
                width=(2 * widths[index] if self.use_double_wide_mlp and index in sharing
                       else widths[index]),
                sparsity=0.0 if sparsity is None else sparsity[index],
                kv_shared=index in sharing,
                provider=index if provides(index, layer_type) else None,
                residual_site=(None if self.attention_residuals is None
                               else self.attention_residuals.site(index)),
                engram=(None if self.engram is None or index not in self.engram.layer_ids
                        else self.engram.layer_ids.index(index)),
                prediction_slot=(None if self.dspark is None or index not in self.dspark.target_layers
                                 else self.dspark.target_layers.index(index)),
                attention_norm=index > 0 or self.first_attention_norm)
            for index, layer_type in enumerate(types))

    @nn.nowrap
    def _block(self, specs: tuple[LayerSpec, ...], mixer_spec, factories, wiring: BlockWiring,
               index: int, name: str) -> DecoderBlock:
        """The decoder layer at `index`, named `name`: its mixer and its
        feed-forward (hash-routed, routed, none, or the gated MLP at its
        width) from its spec, the rest from the model's fields."""
        gated_mlp, routed, parallel = factories
        ple = self.per_layer_input_dim
        spec = specs[index]
        return DecoderBlock(
            mixer=(spec.kind.mixer or mixer_spec).build(
                self.mixer_context(spec.kind, spec.layer_type, spec.kv_shared)),
            feedforward=(
                functools.partial(routed, expert_bias=False, media_bias=False, hash_vocab=self.vocab_size)
                if spec.hash_routed and routed is not None else
                routed
                if spec.routed and routed is not None else
                None
                if spec.width == 0 else
                functools.partial(gated_mlp, hidden_features=spec.width,
                                  activation_sparsity=spec.sparsity)),
            hash_routed=spec.hash_routed,
            residual_multiplier=self.residual_multiplier,
            routed=spec.routed,
            media_routed=(spec.routed and not spec.hash_routed
                          and self.mixture is not None and self.mixture.media_bias),
            engram=None if spec.engram is None or self.engram is None else functools.partial(
                EngramLayer, rows=self.engram.num_embeddings[spec.engram], head_dim=self.engram.head_dim,
                hc_mult=self.hyper_connections.hc_mult if self.hyper_connections else 1,
                emb_features=self.emb_features, norm_eps=self.norm_eps,
                dtype=self.dtype, precision=self.precision),
            engram_index=spec.engram,
            prediction_slot=spec.prediction_slot,
            prediction_site='input' if self.dspark is None else self.dspark.reads,
            attention_norm=spec.attention_norm,
            emb_features=self.emb_features,
            norm_eps=self.norm_eps,
            norm_type=self.norm_type,
            norm_bias=self.norm_bias,
            scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast,
            wiring=wiring,
            per_layer_input_dim=ple or 0,
            gate_activation=self.mlp,
            parallel=parallel if spec.routed else None,
            altup=self.altup,
            laurel_rank=self.laurel_rank,
            hyper_connections=self.hyper_connections,
            residual_site=spec.residual_site,
            dropout_rate=self.dropout_rate,
            remat=self.remat,
            dtype=self.dtype,
            precision=self.precision,
            name=name)

    @nn.nowrap
    def _prediction_depths(self, types: Sequence[str], kinds: dict[str, ResolvedKind], mixer_spec,
                           gated_mlp, routed, wiring: BlockWiring) -> list[MTPBlock]:
        """The multi-token prediction depths."""
        if not self.num_nextn_predict_layers:
            return []
        # Prediction depths mirror whole-sequence hidden states, so their
        # mixer builds from the full-attention kind where the pattern has
        # one, else from the first layer's kind; the feed-forward routes
        # like the last layer's (GLM 4.5 ships its depth with the trunk's
        # experts) and is dense otherwise.
        mtp_type = self.mtp_layer_type or ('full_attention' if 'full_attention' in types else types[0])
        prediction_mixer = kinds[mtp_type].mixer or mixer_spec
        if (self.index_share_for_mtp_iteration
                and not isinstance(prediction_mixer, KPoolSparseAttentionMixer)):
            raise ValueError("index_share_for_mtp_iteration requires a k-pool prediction mixer")
        mtp_mixer = prediction_mixer.build(self.mixer_context(
            kinds[mtp_type], mtp_type, kv_shared=False))
        widths = self.mlp_widths
        mtp_feedforward = (
            routed if routed is not None and self.num_layers - 1 in self.sparse_layers else
            None if widths[-1] == 0 else
            # The last layer's width: the one width of every model with
            # depths, since the widths that vary are Gemma 3n's alone.
            functools.partial(gated_mlp, hidden_features=widths[-1]))
        return [
            MTPBlock(
                mixer=mtp_mixer, feedforward=mtp_feedforward,
                emb_features=self.emb_features,
                hyper_connections=self.mtp_hyper_connections,
                norm_eps=self.norm_eps,
                scale_offset=self.scale_offset,
                scale_after_cast=self.scale_after_cast,
                wiring=wiring,
                dropout_rate=self.dropout_rate,
                remat=self.remat,
                dtype=self.dtype, precision=self.precision, name=f'mtp_{depth}')
            for depth in range(self.num_nextn_predict_layers)]

    @nn.nowrap
    def _dspark_stages(self, kinds: dict[str, ResolvedKind], mixer_spec, routed, wiring: BlockWiring
                       ) -> list[DSparkStage]:
        """DSpark's drafting stages, attending with the V4 attention of their
        layer kind and routing through the trunk's experts."""
        assert self.dspark is not None
        layer_type = self.dspark.layer_type
        drafting = kinds[layer_type].mixer or mixer_spec
        if not isinstance(drafting, DeepseekV4Mixer):
            raise ValueError(f"DSpark's stages attend with V4 attention, and kind {layer_type!r} "
                             f"builds {type(drafting).__name__}")
        drafter = drafting.drafter(self.mixer_context(kinds[layer_type], layer_type, kv_shared=False))
        stages, hc = self.dspark.stages, self.hyper_connections
        assert hc is not None
        return [
            DSparkStage(
                block=functools.partial(
                    DecoderBlock, mixer=drafter,
                    feedforward=functools.partial(routed, num_experts=self.dspark.experts,
                                                  top_k=self.dspark.top_k),
                    emb_features=self.emb_features, norm_eps=self.norm_eps,
                    scale_offset=self.scale_offset, scale_after_cast=self.scale_after_cast,
                    wiring=wiring, hyper_connections=self.hyper_connections,
                    dtype=self.dtype, precision=self.precision),
                emb_features=self.emb_features, vocab_size=self.vocab_size,
                markov_rank=self.dspark.markov_rank,
                first=stage == 0, last=stage == stages - 1, norm_eps=self.norm_eps,
                weighted_head=None if hc.single_pass else hc,
                dtype=self.dtype, precision=self.precision, name=f'dspark_{stage}')
            for stage in range(stages)]

    def _expand(self, x):
        """The embeddings `[B, S, D]` as the residual form the blocks take and
        return (`DecoderBlock._enter`): Gemma 3n's AltUp copies, mHC's
        streams (with Single-Pass's first collapse), Kimi K3's depth state,
        or the embeddings themselves."""
        if self.altup is not None:
            # The embeddings and, rescaled to their magnitude, each projected
            # copy: [num_inputs, B, S, D].
            return jnp.stack([x] + [rescale_to(project(x), x) for project in self.altup_projections])
        hc, depth = self.hyper_connections, self.attention_residuals
        if hc is not None:
            # The embeddings copied into every residual stream: [B, S, hc_mult, D].
            streams = expand_streams(x, hc.hc_mult)
            return Carried(streams, first_stream(streams)) if hc.single_pass else streams
        if depth is not None:
            # The embeddings as the first partial sum after empty blocks: [B, S, blocks + 1, D].
            return jnp.pad(x[:, :, None], ((0, 0), (0, 0), (depth.blocks(self.num_layers), 0), (0, 0)))
        return x

    def _collapse(self, residual):
        """The last block's residual back to `[B, S, D]` for the final norm,
        as `(uncollapsed, collapsed)`: V4's MTP reads the uncollapsed streams
        (after AltUp's copies are already merged)."""
        x = residual
        if self.altup is not None:
            # The copies past the first come back through their own
            # projections, rescaled to the first's magnitude, and the mean of
            # all of them is what the final norm reads.
            copies = [x[0]] + [rescale_to(project(copy), x[0])
                               for project, copy in zip(self.altup_unembed_projections, x[1:], strict=True)]
            x = jnp.mean(jnp.stack(copies), axis=0)
        streams = x
        hc = self.hyper_connections
        if hc is not None and hc.head == 'carried':
            assert isinstance(x, Carried)
            x = collapse_by(x.pre, x.streams)
        elif hc is not None:
            x = collapse_streams(x.streams if isinstance(x, Carried) else x,
                                 self.hc_head if hc.head == 'weighted' else None)
        if self.attention_residuals is not None:
            # Every block is finished after the last layer, so the whole state,
            # the blocks and the last partial, is what the output site mixes
            # (modeling_kimi_linear.py:1215-1233).
            x = self.output_res(x)
        return streams, x

    def __call__(self, tokens, train: bool = False, decode: bool = False,
                 positions=None, segment_ids=None,
                 input_embeddings=None,
                 attention_mask=None, image_groups=None, rotary_positions=None,
                 attention_pairwise_mask=None, attention_key_positions=None):
        x, prediction = self.hidden_and_mtp_inputs(
            tokens, train=train, decode=decode, positions=positions, segment_ids=segment_ids,
            input_embeddings=input_embeddings, attention_mask=attention_mask, image_groups=image_groups,
            rotary_positions=rotary_positions, attention_pairwise_mask=attention_pairwise_mask,
            attention_key_positions=attention_key_positions)
        if self.is_initializing() and self.dspark is not None:
            self.reach_drafter(tokens, x.dtype)
        if self.is_initializing() and self.mtp:
            # Flax creates a parameter where a call first reaches it, and the
            # main forward never enters the prediction depths. Reaching them
            # here, during init only, makes the model's tree the model's
            # business: a plain init holds every depth.
            self.mtp_hidden_states(prediction, tokens, train=train, positions=positions,
                                   segment_ids=segment_ids, input_embeddings=input_embeddings,
                                   attention_mask=attention_mask,
                                   image_groups=image_groups, rotary_positions=rotary_positions)
        return self._logits(x)

    def states_and_logits(self, tokens, **kwargs):
        """Return the prediction-input states and the logits from one forward pass.

        A speculative decoder verifies with both. The logits give the target
        distribution, and the states seed the next block's prediction depths.
        The states are the normalized states, except for V4, where they are
        the uncollapsed residual streams.
        """
        x, prediction = self.hidden_and_mtp_inputs(tokens, **kwargs)
        return prediction, self._logits(x)

    def states_and_logits_at(self, tokens, slots, **kwargs):
        """Return the prediction-input states and the logits at one slot per row, or at `[rows, K]` slots.

        A prefill needs logits only at `slots`, the positions the first draw
        reads. The head over every prompt position would be the largest array
        the forward allocates, [rows, width, vocab], and a decoder keeps only
        one row of it. Gathering the states before the head cuts the head's
        work to [rows, features].
        """
        x, prediction = self.hidden_and_mtp_inputs(tokens, **kwargs)
        rows = jnp.arange(x.shape[0]).reshape(-1, *(1,) * (jnp.ndim(slots) - 1))
        return prediction, self._logits(x[rows, slots])

    def _logits(self, x):
        """The shared fp32 head over `x`: what `__call__` and every MTP depth score with."""
        if self.head_transform is not None:
            x = self.head_norm(ungated_activation(self.head_transform, self.head_dense(x)))
        # fp32 logits whose product follows the compute dtype
        # (`dew.nn.precision.head_product`, the chunked loss's arithmetic).
        if self.tie_embeddings:
            logits = head_product('...d,vd->...v', x, self.embed_tokens.embedding,
                                  self.precision)
        else:
            logits = self.lm_head(x)
        logits = logits.astype(at_least_fp32(logits.dtype))
        if self.head_bias:
            logits = logits + self.head_bias_value.astype(logits.dtype)
        if self.final_logit_softcap is not None:
            cap = jnp.asarray(self.final_logit_softcap, logits.dtype)
            logits = cap * jnp.tanh(logits / cap)
        return logits

    def mtp_hidden_states(self, hidden, tokens, train: bool = False,
                          positions=None, segment_ids=None, input_embeddings=None,
                          attention_mask=None,
                          image_groups=None, rotary_positions=None):
        """Return one final-normed state array per shifted prediction depth.

        Each depth combines the preceding hidden state with the next token's
        embedding, including any media replacement, and uses the next token's
        positions. A pair counts only when both of its positions are valid and
        belong to the same packed document, so padded positions in between
        never become keys. A sequence no longer than the number of depths
        raises `ValueError`.
        """
        if self.mtp and tokens.shape[1] <= len(self.mtp):
            raise ValueError("prediction depths need a sequence longer than their depth count")
        embeds = self._prepared(self.token_embeddings(tokens), input_embeddings)
        # A depth restricts its keys only where the caller's validity or a
        # document boundary does. With neither, every shifted pair is real,
        # and no validity says that: an all-true array would make the depth
        # build a mask and drop off the fused kernel.
        valid = None
        if segment_ids is not None or attention_mask is not None:
            valid = jnp.ones(tokens.shape, bool) if attention_mask is None else attention_mask
        states = []
        for depth, block in enumerate(self.mtp, start=1):
            if valid is not None:
                valid = valid[:, :-1]
                if attention_mask is not None:
                    valid = valid & attention_mask[:, depth:]
                if segment_ids is not None:
                    valid = valid & (segment_ids[:, :-depth] == segment_ids[:, depth:])
            metadata = AttentionMetadata(
                valid=valid,
                image_groups=None if image_groups is None else image_groups[:, depth:],
                rotary_positions=None if rotary_positions is None else rotary_positions[:, depth:])
            hidden = block(hidden[:, :-1], embeds[:, depth:], train=train,
                           positions=None if positions is None else positions[:, depth:],
                           segment_ids=None if segment_ids is None else segment_ids[:, depth:],
                           attention_metadata=metadata)
            states.append(hidden)
        return states

    def mtp_logits(self, hidden, tokens, train: bool = False, positions=None,
                   segment_ids=None, input_embeddings=None,
                   attention_mask=None, image_groups=None, rotary_positions=None):
        """Return the shared language head's logits for each prediction depth's hidden states."""
        return [self._logits(state) for state in self.mtp_hidden_states(
            hidden, tokens, train=train, positions=positions, segment_ids=segment_ids,
            input_embeddings=input_embeddings, attention_mask=attention_mask, image_groups=image_groups,
            rotary_positions=rotary_positions)]

    def mtp_step(self, hidden, tokens, *, depth: int = 0, positions=None,
                 input_embeddings=None, attention_mask=None, rotary_positions=None,
                 decode: bool = False, prediction_phase: PredictionPhase = "ordinary"):
        """Run one unshifted prediction step, optionally appending to the depth's own KV cache.

        Call `init_mtp_cache` before cached steps. `hidden` is the target
        model's preceding state, and `tokens` or `input_embeddings` give the
        candidate next token, as in vLLM's Qwen3_5MultiTokenPredictor. Returns
        the step's logits and hidden state; a chained draft's next step reads
        that state in place of the target's. With
        `index_share_for_mtp_iteration`, a cached "extend" step publishes its
        index selections and a "draft" step reuses them, while "ordinary"
        always recomputes them.
        """
        if prediction_phase not in ("ordinary", "extend", "draft"):
            raise ValueError("prediction_phase must be ordinary, extend or draft")
        if depth < 0 or depth >= len(self.mtp):
            raise ValueError("prediction depth is outside the model's configured depths")
        embeds = self.token_embeddings(tokens) if input_embeddings is None else input_embeddings
        state, prediction = self.mtp[depth].states(
            hidden, embeds, positions=positions, decode=decode,
            prediction_phase=prediction_phase if self.index_share_for_mtp_iteration else "ordinary",
            attention_metadata=AttentionMetadata(valid=attention_mask,
                                                 rotary_positions=rotary_positions))
        return self._logits(state), prediction

    def token_embeddings(self, tokens):
        """Return the embeddings a prediction depth pairs with `tokens`.

        `mtp_hidden_states` reads the unscaled table, so this method does that
        lookup, with `embedding_zero_ids` read as token zero, and nothing else.
        A drawn token is always text, so no media embedding replaces it.
        """
        lookup = tokens
        for token_id in self.embedding_zero_ids:
            lookup = jnp.where(tokens == token_id, 0, lookup)
        return self.embed_tokens(lookup)

    def scaled_embeddings(self, x):
        """Return the token embeddings `x` scaled the way the first layer reads them.

        `x` is multiplied by sqrt(emb_features) when `embedding_scale` is set
        (Gemma), then by `embedding_multiplier` (lm-engine, GraniteMoeHybrid).
        The decoder, the multimodal wrapper and the Qwen-Image conditioner all
        scale here.

        Gemma casts embed_scale to the embedding weight dtype. The lookup holds
        that table in fp32, so the factor stays fp32 and only the product is
        rounded (a bf16 factor would be 34.0 at hidden size 1152, not
        33.941...). lm-engine multiplies in fp32, and `scaled` does the same.
        """
        # Gemma's cast: modeling_gemma3.py:117. lm-engine's fp32 opmath:
        # mixins/dense/base.py at 45b6b57b.
        if self.embedding_scale:
            x = (x * jnp.asarray(math.sqrt(self.emb_features),
                                 self.embed_tokens.embedding.dtype)).astype(x.dtype)
        return scaled(x, self.embedding_multiplier)

    def draft(self, context, tokens, *, decode: bool = True, valid=None, choose=None):
        """Draft a DSpark block after each row's last context position.

        `context` `[B, M, targets * D]` comes from `draft_context` (the stream
        means the target layers sow), `valid` `[B, M]` marks its real
        positions, and `tokens` `[B]` are the tokens drawn after it. Returns
        `(ids [B, block + 1], logits [B, block, vocab], confidence [B, block])`.
        Tokens are chosen by `choose(index, logits)`, or greedily when `choose`
        is None (`dew.nn.dspark.draft`). For a cached draft, call
        `init_draft_cache` first. With `tokens` None the call only appends
        context, and with `context` None it drafts after what the windows
        already hold. A model without a drafter raises `ValueError`.
        """
        if self.dspark is None:
            raise ValueError("this model has no DSpark drafter")
        hc = self.hyper_connections
        assert hc is not None
        return dspark_draft(self.dspark_stages, self.dspark, self.token_embeddings, self._logits,
                            hc, context, tokens, decode=decode, valid=valid, choose=choose)

    def reach_drafter(self, tokens, dtype):
        """Create the DSpark drafter's parameters, which no forward pass reaches.

        It drafts after `tokens` over a zero context as wide as the drafter's
        first stage projects.
        """
        assert self.dspark is not None
        width = len(self.dspark.target_layers) * self.emb_features
        self.draft(jnp.zeros((tokens.shape[0], 1, width), dtype), tokens[:, -1], decode=False)

    def draft_context(self, prediction_inputs: Mapping) -> jax.Array:
        """Return DSpark's context `[B, S, targets * D]` from `prediction_inputs`.

        A forward pass sows `prediction_inputs` when the caller makes that
        collection mutable. The context is each target layer's stream mean,
        concatenated in `target_layers` order.
        """
        assert self.dspark is not None
        return jnp.concatenate([prediction_inputs[f'layers_{layer}']['draft_context']
                                for layer in self.dspark.target_layers], axis=-1)

    def init_draft_cache(self, batch_size: int):
        """Allocate the drafter's window caches, separately from the trunk's cache."""
        assert self.dspark is not None
        width = len(self.dspark.target_layers) * self.emb_features
        self.draft(jnp.zeros((batch_size, 1, width), self.dtype or jnp.float32),
                   jnp.zeros((batch_size,), jnp.int32), decode=True)

    def init_mtp_cache(self, batch_size: int):
        """Allocate the prediction depths' caches, separately from the trunk's cache."""
        shape = ((batch_size, 1, self.emb_features) if self.mtp_hyper_connections is None else
                 (batch_size, 1, self.mtp_hyper_connections.hc_mult, self.emb_features))
        for block in self.mtp:
            block(jnp.zeros(shape, self.dtype),
                  jnp.zeros((batch_size, 1, self.emb_features), self.dtype), decode=True,
                  prediction_phase="extend" if self.index_share_for_mtp_iteration else "ordinary")

    def hidden_states(self, tokens, **kwargs):
        """Return the final normalized states, without the vocabulary projection."""
        return self.hidden_and_mtp_inputs(tokens, **kwargs)[0]

    def logits(self, tokens, *, train: bool = False, **fields):
        """Return `__call__`'s fp32 logits over `hidden_states(tokens, **fields)`, which
        write no cache unless `fields` ask to `decode`."""
        return self._logits(self.hidden_states(tokens, train=train, **fields))

    def logits_from_hidden(self, hidden):
        """Return the logits of final states `hidden`: the head `__call__` and every MTP depth score with."""
        return self._logits(hidden)

    def hidden_and_mtp_inputs(self, tokens, train: bool = False, decode: bool = False,
                              positions=None, segment_ids=None,
                              input_embeddings=None,
                              attention_mask=None, image_groups=None, rotary_positions=None,
                              attention_pairwise_mask=None, attention_key_positions=None,
                              routed_experts=None, routed=None, media_mask=None, admitted=None):
        """Return the final normalized states and the prediction depths' input.

        V4's depth reads the raw residual streams from before the collapse head
        and the final norm; other depths read the final normalized states.
        Packed `positions` and `segment_ids` are passed on to the layers.

        - `input_embeddings` `[B, S, D]` replace the scaled token embeddings,
          for a caller that fuses in another encoder's output. The token ids
          still feed the per-layer inputs and routing.
        - `attention_pairwise_mask` is an explicit [B, queries, keys]
          visibility mask for ordinary attention, and the optional
          `attention_key_positions` [B, keys] give key positions for local
          windows. Both describe the cached reads of this call only.
        - `routed_experts` `[B, S, layers, top_k]` replays a rollout engine's
          routing (vLLM's `routed_experts`, SGLang's
          `meta_info.routed_experts`), and `routed` `[B, S]` marks the tokens
          it covers (`dew.nn.moe.Routes`).
        - `media_mask` [B, S] marks the positions a media encoder fills.
          Engram keeps them out of its n-grams, and a media-biased router
          selects for them.
        - `admitted` makes a cached call a serving step's mixed call
          (`dew.nn.inputs.Admitted`).
        """
        attention_metadata = self._attention_metadata(
            tokens, decode, positions, attention_mask, image_groups, rotary_positions,
            attention_pairwise_mask, attention_key_positions, media_mask, admitted)
        # The stack's entry and exit sit where the batch does, so neither the
        # lookup nor the head is computed whole on the shards of an axis that
        # splits the rows or the positions.
        x = constrain(self._prepared(self.scaled_embeddings(self.token_embeddings(tokens)),
                                     input_embeddings), RESIDUAL)
        if self.position_embedding == 'learned':
            places = positions
            if places is None:
                start = (self.layers[0].self_attn.get_variable('cache', 'cache_index')
                         if decode else None)
                valid = (jnp.ones(tokens.shape, bool) if attention_mask is None else attention_mask)
                places = jnp.cumsum(valid, axis=-1, dtype=jnp.int32) - 1
                if start is not None:
                    places = places + start[:, None]
            x = x + self.embed_positions(jnp.maximum(places + self.position_embedding_offset, 0))
        if self.embedding_norm:
            x = self.embedding_layernorm(x)
        if self.embedding_dropout_rate:
            x = self.embedding_dropout(x, deterministic=not train)
        # A prediction depth reads these unscaled embeddings, media
        # replacements in place, which token ids cannot rebuild. Sowing costs
        # nothing unless a caller opens the collection; init leaves it out.
        if not self.is_initializing():
            prepared = self._prepared(self.token_embeddings(tokens), input_embeddings)
            self.sow("embeddings", "prepared", prepared,
                     reduce_fn=lambda _, value: value, init_fn=lambda: prepared)
        ple = self._layer_inputs(tokens, x, routed_experts, routed)
        residual = self.stack(
            self._expand(x),
            train=train,
            decode=decode,
            positions=positions,
            segment_ids=segment_ids,
            per_layer_input=ple,
            attention_metadata=attention_metadata,
        )
        streams, x = self._collapse(residual)
        hidden = constrain(self.norm(x), RESIDUAL)
        if self.logits_scaling != 1.0:
            # (h W) / s is (h / s) W: dividing the states in fp32 scores every
            # head that contracts them, the chunked losses' included, as
            # lm-engine's logits * (1 / m_width) does.
            hidden = hidden.astype(jnp.float32) / jnp.float32(self.logits_scaling)
        # V4's depth input: inference/model.py MTPBlock.forward at b5968e9.
        prediction = hidden if self.mtp_hyper_connections is None else streams
        if (self.mtp_hyper_connections is not None and not self.is_initializing()
                and self.is_mutable_collection('prediction_inputs')):
            self.sow('prediction_inputs', 'states', prediction,
                     reduce_fn=lambda _, value: value, init_fn=lambda: None)
        return hidden, prediction

    def _attention_metadata(self, tokens, decode: bool, positions, attention_mask, image_groups,
                            rotary_positions, pairwise_mask, key_positions,
                            media_mask, admitted=None) -> AttentionMetadata | None:
        """What the layers read beside the residual, None for a call that
        carries none of it: the masks and positions `hidden_and_mtp_inputs`
        takes, the token ids a hash router selects by, the engram layers'
        bucket ids, and the media mask, which only a model with engram or a
        router with a media bias reads."""
        if key_positions is not None and pairwise_mask is None:
            raise ValueError("attention_key_positions requires attention_pairwise_mask")
        if pairwise_mask is not None:
            kinds = [self.mixer] + [kind.mixer for kind in (self.kinds or {}).values()]
            if any(kind is not None and not isinstance(kind, AttentionMixer) for kind in kinds):
                raise ValueError("explicit pairwise masks require ordinary attention mixers")
        if self.engram is None and (self.mixture is None or not self.mixture.media_bias):
            media_mask = None
        engram_ids = (None if self.engram is None
                      else self.engram_hashes(tokens, attention_mask, positions, decode, media_mask))
        if (attention_mask is None and image_groups is None and rotary_positions is None
                and pairwise_mask is None and key_positions is None and not self.hash_layers
                and engram_ids is None and media_mask is None and admitted is None):
            return None
        return AttentionMetadata(
            valid=attention_mask, image_groups=image_groups, rotary_positions=rotary_positions,
            pairwise_mask=pairwise_mask, key_positions=key_positions,
            token_ids=tokens if self.hash_layers else None, engram_ids=engram_ids, media=media_mask,
            admitted=admitted)

    def stack(self, x, *, train: bool, decode: bool, positions, segment_ids,
              per_layer_input, attention_metadata=None):
        """Run the layer stack over `x` as the plain loop, as scanned runs, or as a pipeline over the stages.

        `init` draws like runs under the scan and then unstacks them, so the
        tree is the plain loop's; under a stage mesh or with `decode`, init
        runs the plain loop. With `scan_layers` or a stage axis, the stack runs
        under a `StackView`, which stacks the leaves while the loops run. The
        loops carry one residual dtype, so the stream enters in the dtype it
        settles in (`residual_dtype`). A store built bank by bank
        (`dew.inference.banks`) already holds one array per run, and the view
        leaves it alone. A pipeline raises `LayoutRefused` for `decode` and
        for a banked store.
        """
        stages = pipeline_stages()
        # A seeded initialization (PTQ's abstract annotation pass) reads the
        # supplied layout; the scan's init=True pass would draw over it first.
        if (self.is_initializing() and stages == 1 and not decode
                and not self.has_variable('params', 'layers_0')):
            groups = scan_groups(self.specs, self.bank_layers)
            if any(count > 1 for _, count in groups):
                view = StackView(groups)
                run = nn.map_variables(type(self)._stacked, mapped_collections=True, trans_in_fn=view.stack,
                                       trans_out_fn=view.unstack, init=True, mutable=True)
                return run(self, view, x, train, decode, positions, segment_ids,
                           per_layer_input, attention_metadata)
        if self.is_initializing() or (stages == 1 and not self.scan_layers):
            return run_stack(
                self.layers, self.block, self.specs,
                tuple((index, 1) for index in range(self.num_layers)), x,
                train=train, decode=decode, positions=positions, segment_ids=segment_ids,
                kv_store={} if self.sharing_layers else None,
                per_layer_input=per_layer_input, attention_metadata=attention_metadata)
        banked = self.banked_collections()
        if stages == 1:
            view = StackView(self.groups, banked=banked)
        else:
            if decode:
                raise LayoutRefused(
                    "decoding appends one token at a time to the cache, which no "
                    "pipeline over the stage axis runs; decode outside jax.set_mesh")
            if banked:
                raise LayoutRefused(
                    f"a pipeline over the stage axis stacks every stage's copy of a "
                    f"layer, which a store already banked by run cannot be reshaped "
                    f"into; {list(banked)} arrived banked. Place the weights per layer "
                    f"for a pipeline, or run the stack whole")
            if isinstance(x, Carried):
                raise ValueError(
                    "Single-Pass mHC carries each sublayer's collapse beside the streams, "
                    "which the pipeline's microbatch buffers do not hold; run the stack whole")
            count = self.stage_layers(stages)
            batch_axis = 1 if self.altup is not None else 0
            rows, count_microbatches = x.shape[batch_axis], microbatches()
            if count_microbatches % stages or rows % count_microbatches:
                raise LayoutRefused(
                    f"a batch of {rows} rows over {stages} stages needs a microbatch "
                    f"count that divides the rows and is a multiple of the stages, "
                    f"got {count_microbatches}")
            # Microbatch m takes rows m, m + count, ... (`_microbatched`), a
            # share of every device's rows only where the count divides a
            # block; otherwise devices holding none of a microbatch recompute
            # another's (1.43 times one device's FLOPs for 8 rows in 4
            # microbatches over 4 row shards, against the bubble's 1.25).
            splitting = row_axes(rows)
            shards = math.prod(jax.sharding.get_abstract_mesh().shape[axis] for axis in splitting)
            if (rows // shards) % count_microbatches:
                raise LayoutRefused(
                    f"a batch of {rows} rows splits {shards} ways over {' x '.join(splitting)}, "
                    f"{rows // shards} rows a device, which {count_microbatches} microbatches "
                    f"do not divide, so the devices that hold none of a microbatch's rows would "
                    f"compute another's again; use a microbatch count that divides "
                    f"{rows // shards}, or a batch that is a multiple of "
                    f"{count_microbatches * shards} rows")
            view = StackView(
                scan_groups(self.specs[:count], self.bank_layers) if self.scan_layers
                else tuple((index, 1) for index in range(count)),
                stages=stages, microbatches=count_microbatches,
                # What enters the loop is read on every iteration; what the
                # loop creates (the routers' sowing) comes out per iteration.
                broadcast=tuple(name for name, tree in self.variables.items() if tree))
        dtype = self.residual_dtype(
            x, train=train, decode=decode, positions=positions, segment_ids=segment_ids,
            per_layer_input=per_layer_input, attention_metadata=attention_metadata)
        # A carried collapse stays fp32, as the reference keeps it.
        x = x._replace(streams=x.streams.astype(dtype)) if isinstance(x, Carried) else x.astype(dtype)
        run = nn.map_variables(type(self)._stacked, mapped_collections=True, trans_in_fn=view.stack,
                               trans_out_fn=view.unstack, init=False, mutable=True)
        return run(self, view, x, train, decode, positions, segment_ids, per_layer_input, attention_metadata)

    def banked_collections(self) -> tuple[str, ...]:
        """Return the collections whose layer subtrees the store holds as banks.

        A store built per layer holds `layers_0`, and a store built bank by bank
        holds each multi-layer run under the run's name, such as `layers_0_15`.
        A run of one layer looks the same either way. A collection that holds a
        run's bank beside its layers, with different leaves in each (a host
        layout's frozen leaves), is not banked, because its rows still stack
        (`StackView.stack`). A bank and a row that hold the same leaf raise
        `ValueError`, and so do some banks without the others when there are
        no rows.
        """
        runs = {group_name(first, count): [f'layers_{index}' for index in range(first, first + count)]
                for first, count in self.groups if count > 1}
        banked = []
        for collection, tree in self.variables.items():
            banks = set(tree) & set(runs)
            if not banks:
                continue
            mixed = False
            for bank, layers in runs.items():
                rows = [layer for layer in layers if layer in tree]
                if not rows:
                    continue
                mixed = True
                if bank not in tree:
                    continue
                shared = {tuple(entry.key for entry in path)
                          for path, _ in jax.tree_util.tree_leaves_with_path(tree[bank])}
                for layer in rows:
                    overlap = shared & {tuple(entry.key for entry in path)
                                        for path, _ in jax.tree_util.tree_leaves_with_path(tree[layer])}
                    if overlap:
                        raise ValueError(
                            f"collection {collection!r} holds {'/'.join(overlap.pop())} both in "
                            f"the bank {bank} and in its layer {layer}; a leaf is read from one")
            if mixed:
                continue
            if banks != set(runs):
                raise ValueError(
                    f"collection {collection!r} holds the banks {sorted(banks)} "
                    f"of the runs {sorted(runs)} and not the others; a store holds every "
                    f"run's bank or every layer's own subtree")
            banked.append(collection)
        return tuple(banked)

    def residual_dtype(self, x, *, train: bool, decode: bool, positions, segment_ids,
                       per_layer_input, attention_metadata=None) -> jnp.dtype:
        """Return the dtype the residual stream settles in: `x`'s promoted with the first layer's output.

        A scan carries one dtype through, and each pipeline stage takes the
        dtype the stage before returns. Under a bf16 policy the dense
        feed-forward returns fp32, so the loops use fp32 from the start. The
        first layer runs abstractly in its own scope (no cache, nothing sown,
        placeholder RNG keys), and a banked store answers from its first
        bank's first row without fetching it.
        """
        layer, scope = self.layers[0], self.layers[0].scope
        assert scope is not None
        rngs = {name: jax.random.key(0) for name in scope.rngs}
        streams = x.streams if isinstance(x, Carried) else x
        output = jax.eval_shape(lambda held: layer.apply(
            held, x, mutable=True, rngs=rngs,
            train=train, decode=decode, positions=positions, segment_ids=segment_ids,
            kv_store={} if self.sharing_layers else None,
            per_layer_input=None if per_layer_input is None else per_layer_input.layer(0),
            attention_metadata=attention_metadata)[0], self.first_layer_shapes())
        output = output.streams if isinstance(output, Carried) else output
        return jnp.result_type(streams.dtype, output.dtype)

    def first_layer_shapes(self) -> dict:
        """Return the stack's first layer's variables as shape/dtype structs.

        These are layer 0's variables, a bank's first row with the layer axis
        dropped, or layer 0's variables completed by the bank's
        (`banked_collections`). Nothing is read, fetched or sliced.
        """
        banked = self.banked_collections()
        first = StackView(self.groups).bank_names()[0]
        reader = self.variables.get('streaming', {}).get(first, {}).get('bank')
        if reader is not None:
            return reader.shapes()
        stacked = self.groups[0][1] > 1

        def shapes(leaf, drop: bool):
            return jax.ShapeDtypeStruct(leaf.shape[1:] if drop else leaf.shape, leaf.dtype)

        variables = {}
        for collection, tree in self.variables.items():
            if collection in banked:
                variables[collection] = jax.tree.map(
                    functools.partial(shapes, drop=stacked), tree[first])
                continue
            held = {}
            if stacked and first in tree:
                held = jax.tree.map(functools.partial(shapes, drop=True), tree[first])
            if 'layers_0' in tree:
                held = _merged(held, jax.tree.map(functools.partial(shapes, drop=False), tree['layers_0']))
            if held:
                variables[collection] = held
        return variables

    def stage_layers(self, stages: int) -> int:
        """Return the number of layers per stage when the stack splits into `stages`.

        Every stage runs one program, so layer `j` of every stage must match
        layer `j` of the first stage in kind, feed-forward, width and KV role.
        A pattern that does not repeat every `num_layers / stages` layers raises
        `LayoutRefused`, and the error names the first pair that differs.
        """
        if self.num_layers % stages:
            raise LayoutRefused(
                f"{self.num_layers} layers do not split into {stages} stages of "
                f"equal length; set stage to a divisor of num_layers")
        count = self.num_layers // stages
        for index, spec in enumerate(self.specs):
            first = self.specs[index % count]
            if spec == first:
                continue
            differing = [field.name for field in dataclasses.fields(LayerSpec)
                         if getattr(spec, field.name) != getattr(first, field.name)]
            raise LayoutRefused(
                f"layer {index} differs from layer {index % count} in "
                f"{', '.join(differing)}, so stage {index // count} cannot run the "
                f"first stage's program; a pipeline of {stages} stages needs the "
                f"layer pattern to repeat every {count} layers, which means a "
                f"mixture, a layer_types pattern and a K/V sharing that do")
        return count

    @nn.compact
    def _stacked(self, view: StackView, x, train: bool, decode: bool, positions,
                 segment_ids, per_layer_input, attention_metadata=None):
        """The stack inside `view`: scanned runs, or the pipeline over stages."""
        if view.stages == 1:
            return run_stack(
                self.layers, self.block, self.specs, view.groups, x,
                train=train, decode=decode, positions=positions, segment_ids=segment_ids,
                kv_store={} if self.sharing_layers else None,
                per_layer_input=per_layer_input, attention_metadata=attention_metadata,
                banked='params' in view.banked)
        return run_pipeline(
            self,
            view,
            x,
            train=train,
            positions=positions,
            segment_ids=segment_ids,
            per_layer_input=per_layer_input,
            attention_metadata=attention_metadata,
        )

    def _layer_inputs(self, tokens, x, routed_experts, routed) -> LayerInputs | None:
        """The per-layer signal and routing replay the stack slices by layer."""
        embeddings = self.per_layer_inputs(tokens, x) if self.per_layer_input_dim else None
        if routed_experts is None:
            if routed is not None:
                raise ValueError("routed marks the tokens a routing record covers; pass routed_experts")
            return None if embeddings is None else LayerInputs(embeddings=embeddings)
        routed_experts = jnp.asarray(routed_experts)
        if routed_experts.ndim != 4 or routed_experts.shape[:3] != (*tokens.shape, self.num_layers):
            raise ValueError(
                f"routed_experts is [batch, tokens, layers, top_k], {(*tokens.shape, self.num_layers)} "
                f"leading for this model; got {routed_experts.shape}")
        coverage = None if routed is None else jnp.broadcast_to(
            jnp.asarray(routed, bool)[:, :, None], routed_experts.shape[:3])
        return LayerInputs(embeddings=embeddings, experts=routed_experts, routed=coverage)

    def per_layer_inputs(self, tokens, inputs_embeds):
        """Return every layer's input signal `[B, S, L, P]` (Gemma 3n/4 PLE).

        The token-identity part is the packed table's row for each token,
        scaled like the main embedding. The context part is the input
        embeddings projected down, scaled and normed. Each layer's gate
        multiplies in their sum divided by sqrt(2). A model without
        `per_layer_input_dim` raises `ValueError`.
        """
        # modeling_gemma4.py, get_per_layer_inputs and project_per_layer_inputs.
        ple = self.per_layer_input_dim
        if ple is None:
            raise ValueError(
                "per_layer_inputs is the signal of a model with per-layer input "
                "embeddings, and this one has none: per_layer_input_dim is unset")
        table = self.embed_tokens_per_layer(tokens).reshape(
            *tokens.shape, self.num_layers, ple)
        # The reference scales by sqrt(P) cast to the table's weight dtype
        # (modeling_gemma4.py, Gemma4TextScaledWordEmbedding), not to the
        # activation dtype.
        table = table * jnp.asarray(
            math.sqrt(ple), self.embed_tokens_per_layer.embedding.dtype)
        context = self.per_layer_model_projection(inputs_embeds)
        context = context * jnp.asarray(self.emb_features ** -0.5, context.dtype)
        context = self.per_layer_projection_norm(
            context.reshape(*inputs_embeds.shape[:-1], self.num_layers, ple))
        return (context + table) * jnp.asarray(2.0 ** -0.5, context.dtype)

    @staticmethod
    def _prepared(x, input_embeddings):
        """`x`, or in its place a caller's prepared `[B, S, D]` embeddings
        (another encoder's outputs fused in), cast to the stream dtype."""
        if input_embeddings is None:
            return x
        prepared = jnp.asarray(input_embeddings)
        if prepared.shape != x.shape:
            raise ValueError(f"input_embeddings are the whole {x.shape} sequence, got {prepared.shape}")
        return prepared.astype(x.dtype)

    def head_weight(self, params):
        """Return the `[D, vocab]` head matrix in its stored dtype, as the forward pass contracts it.

        For a tied head this is the embedding table transposed, and otherwise
        `lm_head`'s kernel. It is read from `params` without an fp32 copy. The
        Gemma embedding scale applies only to input embeddings, so it is not
        part of this matrix.
        """
        # The same matrix `_logits` contracts.
        table, vocab_major = self.head_table(params)
        return table.T if vocab_major else table

    def vocabulary_bias(self, params):
        """The vocabulary bias, or None for a bias-free head."""
        return params['head_bias'] if self.head_bias else None

    def head_table(self, params):
        """Return the head matrix as the tree stores it, and whether its rows are the vocabulary.

        For a tied head this is the `[vocab, D]` embedding table and True, and
        otherwise `lm_head`'s `[D, vocab]` kernel and False. No operation sits
        between the parameter and the result, so a loss that keeps the head for
        its backward pass (`chunked_cross_entropy` with `vocab_major`) keeps the
        parameter itself and no transposed copy.
        """
        if self.head_transform is not None:
            raise ValueError(
                "a prediction head sits between the final states and the "
                "vocabulary, so no matrix alone gives this model's logits, and the "
                "objectives that score through the chunked head cannot train it yet; "
                "the model's own call scores it")
        if self.tie_embeddings:
            return params['embed_tokens']['embedding'], True
        return params['lm_head']['kernel'], False

    def output_table(self) -> OutputTable | None:
        """Return the head as the matrix `logits_from_hidden` contracts, read from the bound variables:
        the parameter itself, as `head_table` gives it, with `_logits`'s bias, softcap and precision.
        None where no matrix alone is the head: past a `head_transform`, or when `lm_head`'s scope
        holds more than its kernel, as `dew.lora`'s factors, which its interceptor adds."""
        head = {} if self.tie_embeddings else self.lm_head.variables["params"]
        if self.head_transform is not None or set(head) - {"kernel"}:
            return None
        matrix = self.embed_tokens.embedding if self.tie_embeddings else head["kernel"]
        return OutputTable(matrix, self.tie_embeddings, self.head_bias_value if self.head_bias else None,
                           self.final_logit_softcap, self.precision)

    def init_cache(self, batch_size: int):
        """Allocate a zeroed decode cache for `batch_size` sequences.

            cache = model.apply(params, batch_size, method=CausalTransformer.init_cache,
                                mutable=['cache'])[1]['cache']

        It runs the forward pass in decode mode on a single dummy token, and
        that token's keys are never written: the first decode-mode call
        allocates the cache, and the calls after it write to it.
        """
        self(jnp.zeros((batch_size, 1), jnp.int32), decode=True)

    # The serving and training hooks (`dew.nn.protocols`).

    @nn.nowrap
    def with_cache_capacity(self, capacity: int) -> Self:
        """`max_seq_len` is the cache size its layers read (`open_kv_cache`);
        a learned position table it sized keeps its rows."""
        if self.position_embedding == 'learned' and self.position_embedding_size is None:
            return self.clone(max_seq_len=capacity, position_embedding_size=self.max_seq_len)
        return self.clone(max_seq_len=capacity)

    @nn.nowrap
    def recompute_record(self) -> JSON:
        return remat_record(self.remat)

    @nn.nowrap
    def recompute_more(self) -> Self | None:
        """One rung up `DECODER_REMAT`; None at its top or off it."""
        if self.remat not in DECODER_REMAT[:-1]:
            return None
        return self.clone(remat=DECODER_REMAT[DECODER_REMAT.index(self.remat) + 1])

    @nn.nowrap
    def restore_recompute(self, record: JSON) -> Self:
        records = [remat_record(remat) for remat in DECODER_REMAT]
        here = remat_record(self.remat)
        if here in records and record in records and records.index(record) > records.index(here):
            return self.clone(remat=DECODER_REMAT[records.index(record)])
        return self

    @nn.nowrap
    def inference_projection_groups(self, variables: Mapping[str, Mapping]) -> tuple[ProjectionGroup, ...]:
        """Its layers', depths' and drafter's groups, bound without tracing a
        forward; an encoder serves no decode step and names none."""
        if not self.causal:
            return ()
        bound = self.bind(variables)
        return declared_groups(*bound.layers, *bound.mtp,
                               *(bound.dspark_stages if self.dspark is not None else ()))

    @property
    def cache_rebuild_position(self) -> int | None:
        scaling = self.rope_scaling
        return scaling.original_max_position_embeddings if isinstance(scaling, LongRopeScaling) else None

    @nn.nowrap
    def mixed_admission_refusal(self) -> str | None:
        """A cache rebuilt per request, a reader beyond the token, or a layer
        whose mixer kind declares no `mixed_step`."""
        position = self.cache_rebuild_position
        if position is not None:
            return (f'LongRoPE crossing position {position} requires separate admission so each '
                    'request keeps its own table and rebuild history')
        if self.num_nextn_predict_layers:
            return "it runs prediction depths"
        if self.dspark is not None:
            return "its block drafter keeps a cache of its own, which a mixed step does not run"
        if self.position_embedding == "learned" or self.engram is not None or self.hash_layers:
            return "a learned position embedding, n-gram or hash routing reads beyond the token"
        default = self.mixer if self.mixer is not None else AttentionMixer()
        for index, layer_type in enumerate(self.per_layer_types):
            mixer = self.kind_of(layer_type).mixer or default
            if not mixer.mixed_step:
                return f"layer {index}'s cache is {type(mixer).__name__}'s, which a mixed step does not run"
        return None

    @property
    def declared_mixers(self) -> tuple[MixerBase, ...]:
        """The mixer values it names: its own, then each layer kind's."""
        return tuple(mixer for mixer in (self.mixer, *(kind.mixer for kind in (self.kinds or {}).values()))
                     if mixer is not None)

    @property
    def indexed_mixers(self) -> tuple[MLAMixer, ...]:
        return tuple(mixer for mixer in self.declared_mixers if isinstance(mixer, MLAMixer) and mixer.indexed)

    @property
    def keeps_triton_gemm(self) -> bool:
        return any(mixer.keeps_triton_gemm for mixer in self.declared_mixers)


__all__ = ["CausalTransformer", "DecoderBank", "PipelineStage", "StackView"]
