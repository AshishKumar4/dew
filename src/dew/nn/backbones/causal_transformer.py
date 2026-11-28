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

The block holds its token mixer in a slot. Any module called (x, call) -> x
with one `dew.nn.inputs.MixerCall`, as CausalSelfAttention is, reading its
tokens' places and documents off the call metadata's `Coordinates` and
declaring the masks it reads (`reads`), can be self_attn without changes to the
block, and that is where a linear-attention mixer goes.
"""

import dataclasses
import functools
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import ClassVar, Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew._records import JSON, NUMBERS, ROUNDS, RUNS, TRAINS
from dew.batch import (
    COORDINATES_FIELD,
    Coordinates,
    Variables,
    merge,
    prefix_sum,
    real_slots,
    token_coordinates,
    with_validity,
)
from dew.nn.continuum import (
    HeadTerms,
    Levels,
    chain_states,
    continuum_chain,
    level_of,
    level_parameters,
    with_level_parameters,
)
from dew.nn.dense import Dense, pre_call_capture
from dew.quantization import LinearProducts

from ..activations import GatedActivation
from ..attention_residuals import AttentionResiduals, DepthAttention
from ..block_draft import BlockDrafter, MarkovBlockDraft
from ..cross_attention import CrossBlocks, crossed_flops
from ..cross_pass import CrossPass
from ..dex import DexStart, DifferentialExtension, HeadOutputs, LayerDex, folded
from ..dspark import DSpark
from ..engram import Engram, EngramHashes, EngramLayer, PLELayer
from ..flops import counted_flops, parameter_flops
from ..gemma3n import AltUp, clamped_predictions, rescale_to
from ..hyper_connections import Carried, HyperConnections, enter_streams, leave_streams, stream_head
from ..inputs import LayerInputs, PredictionPhase, coordinates_of
from ..kernels import generation
from ..kv_cache import CacheSpec, KVCache, put_opened
from ..mixers import AttentionMixer, MixerBase, MixerContext
from ..ngpt import NGPT
from ..norms import NormSite
from ..precision import at_least_fp32, float32_dot_general, l2norm
from ..progress import Progress
from ..protocols import (
    EXITS,
    ExitPasses,
    IndexedMixer,
    OutputTable,
    ProjectionGroup,
    RecomputeStatement,
    declared_groups,
)
from ..rope import Scaling
from ..sharding import RESIDUAL, constrain
from ..value_experts import ValueExperts
from . import (
    decoder_cache,
    decoder_counts,
    decoder_drafts,
    decoder_head,
    decoder_predictions,
    decoder_recompute,
    decoder_refusals,
    decoder_stages,
    embeddings,
)
from .call_metadata import call_metadata
from .decoder_block import (
    BlockWiring,
    DecoderBlock,
    Mixture,
    MTPBlock,
    NormSpec,
    RematPolicy,
    decoder_norm,
    feedforward_factories,
    remat_record,
    residual_normed,
)
from .decoder_cache import named_caches, trunk_cache
from .decoder_fields import refuse_unbuildable, settle_fields, softmax_attention_layers
from .decoder_stack import DecoderBank, Loop, StackView, StageSplit, banked_collections, split_stages
from .embeddings import prepared_embeddings
from .layer_names import group_layers, layer_index
from .layer_plan import LayerKind, LayerSpec, ResolvedKind, layer_streams, run_period, scan_groups
from .looping import run_loop

_INTERMEDIATES = "intermediates"
"""The collection flax's `capture_intermediates` fills."""
_FINAL = "final_residual"
"""The collection the forward sows its stack's output into (`layer_state`)."""




class CausalTransformer(LinearProducts):
    """Decoder-only transformer over token ids: [B, S] int32 -> [B, S, vocab] fp32.

    The defaults train a model from scratch: multi-head attention, swiglu,
    tied embeddings, no softcap. Every setting that differs between open
    decoders is a field here, so loading Qwen3 or Gemma3 is a mapping of config
    fields and needs no subclass. The field comments name the family that sets
    each one. A classic GPT block is `norm_type='layer'`, `norm_bias`, an
    ungated `mlp`, `mlp_bias` and learned positions; the learned positions
    advance from each decode row's count of its real tokens (`DecodeCursor`).

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

    Each run of like layers (layers whose resolved specs give the same
    parameter shapes and computation), and a step's periodic repeats, run as
    one scanned body (`decoder_stages.scanned`), so compile time grows with
    the period, not the depth. The variables tree stays the unscanned one
    (`StackView`). `init` always draws a run of like layers under
    the scan, once per run. When the mesh has a stage axis larger than one, the
    stack runs as a pipeline.
    """
    vocab_size: int
    emb_features: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int | None = None       # None: as many as the query heads
    head_dim: int | None = None  # None: emb_features // num_heads
    value_head_dim: int | None = None
    """The width of a value head; None is the query and key heads' width.
    MiMo-V2-Flash attends 192-wide queries and keys over 128-wide values."""
    mlp: GatedActivation = "swiglu"  # 'swiglu' | 'geglu' | 'geglu_exact' | 'swigluoai', or Kimi K3's Situ
    mlp_bias: bool = False
    """Whether both feed-forward projections have a bias: none by default, as Llama's published
    `mlp_bias: false`, where Flax's Dense adds one. The `mlp` activations gelu, gelu_exact and relu
    give an ungated feed-forward."""
    mlp_features: int | tuple[int, ...] | None = None
    """The dense feed-forward width. None means four times `emb_features`, a
    tuple gives one width per layer (Gemma 3n), and 0 means no feed-forward
    (Mamba-2)."""
    mlp_multipliers: tuple[float, float] | None = None
    """The dense feed-forward's gate and output multipliers (Falcon-H1's `mlp_multipliers`,
    `GatedMLP.gate_multiplier` and `down_multiplier`); None for neither. The shared and routed experts
    take none."""
    max_seq_len: int = dataclasses.field(default=2048, metadata={RUNS: True})
    context_length: int | None = None
    """The context a checkpoint states (max_position_embeddings), which an export writes back;
    None for `max_seq_len`. The model computes nothing from it: `max_seq_len` is its decode cache's
    capacity, which a loader caps below a long context (`decoder_parts.DEFAULT_MAX_SEQ_LEN`)."""
    position_embedding: Literal['rotary', 'learned', 'alibi'] = 'rotary'
    """How a layer whose kind names no mixer places its tokens (`layer_mixer`): rotary attention,
    or unrotated attention after a learned table the embedding adds ('learned') or with ALiBi's
    biases ('alibi'). A mixer a model or a kind names places its own tokens; the learned table is
    added whatever the mixers."""
    position_embedding_size: int | None = None
    """The number of rows in the learned position table; None means `max_seq_len`.
    It does not depend on the decode cache's capacity."""
    position_embedding_offset: int = 0
    """The number of reserved rows before learned position zero; OPT checkpoints have two."""
    rope_theta: float = 10000.0              # the base a kind does not override
    rope_scaling: Scaling | None = None
    """The model's rope scaling: Llama 3.1's ramp, LongRoPE's factors or YaRN's record (`rotary_freqs`)."""
    partial_rotary_factor: float | None = None  # None: every dim rotates
    partial_rotary_type: str = 'proportional'  # 'proportional' (Gemma 4) | 'default' (Qwen3.5)
    narrow_rotary_tables: bool = False
    """The rotary's cos and sin round to the activations' dtype before the fp32 rotation, as vLLM
    casts its cos/sin cache to the query's dtype (`MixerContext.narrow_rotary_tables`); False keeps
    them at the rotation's precision."""
    layer_types: tuple[str, ...] | None = None  # the pattern, one kind per layer
    kinds: Mapping[str, LayerKind] | None = None  # what each named kind does
    norm_eps: float = 1e-5
    """Llama 2's and 3's and Mistral's published `rms_norm_eps`, where Flax's RMSNorm takes 1e-6."""
    norm_type: Literal['rms', 'layer'] = 'rms'
    norm_bias: bool = False
    norms: Mapping[str, NormSpec] | None = None
    """The norms that differ from the model's others, by module name
    (`decoder_block.NormSite`): K2 Horizon's grouped statistics, an
    epsilon or a weight of a norm's own."""
    embedding_norm: bool = False
    """Normalize token embeddings before the first block, as BLOOM and ModernBERT do."""
    embedding_norm_reads: Literal['inputs', 'lookup'] = 'inputs'
    """What `embedding_norm` normalizes: every input embedding, a caller's own and a media encoder's
    among them, as BLOOM's and ModernBERT's embedding norms read their `inputs_embeds` ('inputs'); or
    the table's lookup alone, before media take their slots, as Muse Glimmer's normed embedding module
    does (`scaled_embeddings`, 'lookup')."""
    first_attention_norm: bool = True
    """Whether the first block norms its attention input. ModernBERT's first
    block reads the embedding norm's output as it is (modeling_modernbert.py,
    ModernBertEncoderLayer: attn_norm is Identity at layer 0)."""
    scale_offset: bool = False       # a norm's weight is (1 + w): Gemma's RMSNorm, GPT-3's LayerNorm
    # Apply the weight after casting, as Llama and Qwen3 do.
    scale_after_cast: bool = dataclasses.field(default=False, metadata={RUNS: True, ROUNDS: True})
    sandwich_norms: bool = False     # add a norm after each sublayer, as Gemma does
    pre_norms: bool = True           # norm each sublayer's input; False + sandwich is OLMo 3
    post_residual_norms: bool = False  # norm the stream after each sublayer joins it: post-LN, OPT-350m's
    final_norm: bool = True            # end the trunk in a norm; OPT-350m's post-LN decoder has none
    embedding_features: int | None = None
    """The token embedding's width where it differs from the stream's (OPT-350m's word_embed_proj_dim),
    bias-free `project_in` taking the lookup in and `project_out` the final states out to the head."""
    parallel_residual: bool = False
    """Whether attention and the feed-forward both read the same residual, as in GPT-NeoX."""
    shared_parallel_norm: bool = False
    """Both parallel branches read one LayerNorm, as in Phi, Falcon-7B and GPT-J."""
    qk_norm: bool = True
    qk_norm_scope: str = 'head'              # 'head' per head (Qwen3); 'projection' whole (OLMo 3)
    v_norm: bool = False                     # Gemma 4's scale-free values norm
    attention_k_eq_v: bool = False           # Gemma 4's global layers read their values off the keys
    layer_scalar: Literal["frozen", "trainable"] | None = dataclasses.field(default=None,
                                                                            metadata={RUNS: True})
    attention_bias: bool = False
    """q/k/v biases, and o_proj's unless `o_proj_bias` says: none by default, as Llama's published
    `attention_bias: false`, where Flax's Dense adds one."""
    o_proj_bias: bool | None = None       # Qwen2 biases q/k/v while o_proj stays bias-free
    attention_scale: float | None = None  # None: head_dim ** -0.5
    value_scale: float | None = None
    """The factor that multiplies the values as they are projected, before
    they are cached: MiMo-V2-Flash's `attention_value_scale`."""
    key_multiplier: float | None = None
    """The factor that multiplies the keys as they are projected, before
    they are rotated and cached: Falcon-H1's `key_multiplier`."""
    attention_sinks: bool = False
    attn_logit_softcap: float | None = None  # Gemma 2's attn_logit_softcapping
    output_gate: bool = False                 # Qwen3.5 gates the attention branch
    output_gate_activation: str = 'sigmoid'   # the gate's activation (MixerContext.output_gate_activation)
    value_experts: ValueExperts | None = None  # MoVA: the routed layers' values from routed experts
    embedding_scale: bool = False            # Gemma scales embeddings by sqrt(d)
    embedding_multiplier: float = 1.0
    """The factor that multiplies the token embeddings before the first layer:
    muP's m_emb in lm-engine, and GraniteMoeHybrid's `embedding_multiplier`."""
    residual_multiplier: float = 1.0
    """The factor that multiplies every sublayer output before it is added to the
    residual stream: lm-engine's m_residual, and GraniteMoeHybrid's
    `residual_multiplier`."""
    normed_residual: bool = False      # MiniMax's postnorm block (`BlockWiring.normed_residual`)
    stream_multiplier: float = 1.0     # MiniMax's alpha factors (`BlockWiring.stream_multiplier`)
    residual_in_fp32: bool = False
    """Whether the residual stream between layers sums in at least fp32 whatever
    the compute dtype, as mamba_ssm's `residual_in_fp32` keeps it (transformers'
    Mamba, Mamba2 and FalconMamba blocks): each sublayer reads the stream in the
    compute dtype and its output adds to the stream in fp32 (`stack`)."""
    logits_scaling: float = 1.0
    """The divisor of the logits: lm-engine's m_width (`lm_logits *
    (1 / m_width)`), and GraniteMoeHybrid's `logits_scaling`. The division is
    applied to the final states, in fp32, so every head that contracts them
    (`output_table`) gets the same logits that `__call__` returns."""
    # lm-engine's init_utils.py at 45b6b57b.
    initializer_range: float | None = dataclasses.field(default=None, metadata={RUNS: True})
    """The std of lm-engine's initialisation; None keeps each module's own
    initializer.

    When it is set, the embedding table (and an untied head) is drawn from
    N(0, initializer_range^2), and every hidden matrix (attention and Mamba-2
    projections, conv taps, router, experts, dense MLPs) from
    N(0, (initializer_range / sqrt(logits_scaling))^2). Biases start at zero
    and norms at one. With `logits_scaling` set to m_width this is lm-engine's
    `init_method="mup"`, and with `logits_scaling` at 1 it is lm-engine's
    `"normal"`."""
    depth_scaled_init: bool = dataclasses.field(default=False, metadata={RUNS: True})
    """Whether, with `initializer_range` set, the projections back into the
    residual stream (o_proj, out_proj, down_proj) divide their std by
    sqrt(2 * num_layers). This is lm-engine's `use_depth_scaled_init`."""
    final_logit_softcap: float | None = None
    ngpt: NGPT | None = None
    """The normalized transformer (`dew.nn.ngpt.NGPT`), NVIDIA's Dory: every hidden state on the
    unit sphere, each block stepping its residual along it (`BlockWiring.ngpt`), the attention and
    feed-forward unit-norming their products and scaling them by learned gains, no norm with a
    weight anywhere, a unit norm between a loop's passes, no norm before the head, and the logits
    times a learned per-token gain (`logit_scale`). None for none."""
    head_transform: Literal['gelu', 'gelu_exact'] | None = None
    """The activation of a BERT-style prediction head between the final norm
    and the vocabulary projection: a bias-free dense layer, this activation and
    a norm of the model's kind (ModernBertPredictionHead,
    modeling_modernbert.py). None projects the final states directly. As in the
    feed-forward, 'gelu' is the tanh form and 'gelu_exact' the erf form.
    `hidden_states` stays the final norm's output, the encoder's."""
    head_bias: bool = False
    """A vocabulary bias added in fp32 after a tied or untied head's product, as
    Phi's, GPT-J's and ModernBERT's decoders add one (`output_table`'s `bias`)."""
    tie_embeddings: bool = True
    output_vocab: int | None = None
    """The head's logits' width where it is narrower than the table's rows (`vocab_size`): the head
    keeps every row as stored and scores the first `output_vocab` alone (`logits_width`). Inkling 2's
    head scores 200058 of its 201024 released rows, the rest padding no tokenizer id reaches."""
    # Kimi K2.5's text-only wrapper path, modeling_kimi_k25.py:686-690.
    embedding_zero_ids: tuple[int, ...] = ()
    """Placeholder ids that the embedding lookup reads as token zero. The labels
    keep the original ids."""
    padding_idx: int | None = None
    """The id whose token embedding rows (`embed_tokens`, `embed_tokens_per_layer`) the lookup reads
    but never trains, as a source's `nn.Embedding(padding_idx=...)` leaves them (Gemma 3n's and Phi's
    pad_token_id, `TokenEmbedding.padding_idx`); a tied head still trains the row. None trains every
    row."""
    dropout_rate: float = dataclasses.field(default=0.0, metadata={RUNS: True})
    embedding_dropout_rate: float = dataclasses.field(default=0.0, metadata={RUNS: True})
    attention_dropout_rate: float = dataclasses.field(default=0.0, metadata={RUNS: True})
    dtype: Dtype | None = dataclasses.field(default=None, metadata={RUNS: True, NUMBERS: True})
    precision: PrecisionLike = dataclasses.field(default=None, metadata={RUNS: True, NUMBERS: True})
    force_fp32_for_softmax: bool = dataclasses.field(default=True, metadata={RUNS: True, ROUNDS: True})
    kv_cache: KVCache = dataclasses.field(default=KVCache(), metadata={RUNS: True})
    """How attention layers store the decode cache: dense or paged, full or
    quantized (`dew.nn.kv_cache`). The parameters do not depend on it."""
    mixture: Mixture | None = None        # some layers' feed-forward as `moe.SparseMLP`; None: dense
    continuum: Levels | None = None
    """A continuum level on the final states before the head, or a chain of them (`dew.nn.continuum`):
    parameters an objective steps in context by its own loss (`LMObjective(adaptation=...)`), every head
    that reads the states, `__call__`'s and each prediction depth's, reading the levels first. It
    starts as the identity. None gives the plain head."""
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
    mtp_feedforward_type: str | None = None
    """The layer kind whose MLP mixer (`MLPMixer`) builds the prediction
    depths' feed-forward, as Nemotron-H's step pairs an attention block with
    a routed MLP block. None routes like the last layer, and is dense
    otherwise."""
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
    """DeepSeek-V4.1's n-gram lookups or Qwen4-Exp's per-layer embedding (`dew.nn.engram`) before each named
    layer's attention, over the compressed vocabulary `constants` holds (`engram_hashes/token_map`)."""
    dex: DifferentialExtension | None = None
    """The chosen heads whose attention outputs take a learned differential (`dew.nn.dex`), or None."""
    cross_attention: CrossBlocks | None = None
    """The layers that read a conditioning sequence through Flamingo's gated
    cross-attention (`dew.nn.cross_attention`), before their own sublayers; None
    for a model without them. A call passes the sequence as its `conditioning`."""
    dspark: DSpark | None = None
    """DeepSeek's block drafter (`dew.nn.dspark`, V4.1 and V4-Flash-0731), or
    None for a model without one. Its stages are decoder blocks that attend
    through a `DSparkAttention` built from the V4 attention of its
    `layer_type` kind. Its target layers sow the stream means that the
    drafter reads (`draft_context`)."""
    block_draft: MarkovBlockDraft | None = None
    """A Markov block drafter (`dew.nn.block_draft`, Inkling 2's multi-token prediction draft), or None: its
    layers read the residual the tapped visits of the trunk's layers are handed and its final norm's output
    (`draft_context`), and draft with the trunk's embedding and head. A model has one drafter at most."""
    swiglu_limit: float | None = None  # GLM-5.3-Flash's clamp before every gated MLP's activation
    activation_sparsity_pattern: tuple[float, ...] | None = None
    """Gemma 3n's gaussian top-k activation sparsity, one fraction per layer."""
    mask_token_id: int | None = dataclasses.field(default=None, metadata={RUNS: True, TRAINS: True})
    """The vocabulary id that a masked-diffusion objective corrupts tokens to.
    None means plain training."""
    shifted_logits: bool = False
    """Whether masked diffusion reads position i's prediction from position i-1's logits, as a
    checkpoint continued from an autoregressive one predicts the token after each position
    (Dream-v0); False reads each position's own (LLaDA). The forward ignores it: it says how
    the checkpoint's logits line up, which `dew.diffusion.discrete.aligned` applies for the
    masked loss and the sampler alike."""
    bank_layers: int | None = dataclasses.field(default=None, metadata={RUNS: True})
    """The most layers one scanned run holds, which is also how many layers its
    parameter bank stacks. A longer run of like layers is split into
    consecutive runs of at most this many, each with its own bank under its own
    name. None keeps a whole run in one bank. The scan and `init` read it,
    and because init draws each run under one scan, it also changes what init
    draws for a given seed. The split bounds the memory needed to build a
    host-resident bank and read it back, so set it for a deep stack that is
    offloaded to the host."""
    loop: Loop | None = None
    """The layer stack run several times over its own output, its parameters
    shared (`Loop`), as Ouro's looped decoder runs it. None runs it once."""
    remat: RematPolicy | None = dataclasses.field(default=None, metadata={RUNS: True})
    """The rematerialization policy. Each block is recomputed in the backward
    pass, keeping its inputs, any K/V passed to later layers, and the residuals
    the policy names. A config gives a name from `REMAT_POLICIES` or a record of
    residual names to save or offload. None recomputes nothing. Init and cached
    decode call the blocks directly, and the stored parameters have the same
    layout either way."""

    def __post_init__(self):
        settle_fields(self)
        if self.output_vocab is not None and not 0 < self.output_vocab <= self.vocab_size:
            raise ValueError(f"a head scores at most its table's {self.vocab_size} rows, got "
                             f"output_vocab={self.output_vocab}")
        # The wiring checks its own combinations (`BlockWiring.__post_init__`) as the model is made.
        _ = self.wiring
        super().__post_init__()

    @property
    def logits_width(self) -> int:
        """`LogitsWidth`: the head's logits' width, its first `output_vocab` rows or every row."""
        return self.vocab_size if self.output_vocab is None else self.output_vocab

    @property
    def wiring(self) -> BlockWiring:
        """How every block wires its two sublayers (`BlockWiring`), from the model's own dials, each a
        field of its own (docs/design/api.md): the one value the stack's blocks, the prediction depths,
        DSpark's stages and the FLOP count build with."""
        return BlockWiring(pre_norms=self.pre_norms, output_norms=self.sandwich_norms,
                           residual_norms=self.post_residual_norms,
                           layer_scalar=self.layer_scalar, parallel_residual=self.parallel_residual,
                           shared_parallel_norm=self.shared_parallel_norm, ngpt=self.ngpt,
                           normed_residual=self.normed_residual, stream_multiplier=self.stream_multiplier,
                           residual_in_fp32=self.residual_in_fp32)

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
    def position_rows(self) -> int:
        """The rows of the learned position table (`position_embedding_size`, else `max_seq_len`)."""
        return self.position_embedding_size or self.max_seq_len

    @property
    def position_capacity(self) -> int | None:
        """The places a row's learned positions cover, the table's rows past its reserved offset; None for
        rotary or ALiBi positions, which cover any (`PositionCapacity`)."""
        learned = self.position_embedding == 'learned'
        return self.position_rows - self.position_embedding_offset if learned else None

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
        """Return the `ResolvedKind` of `layer_type`, with the model's defaults where the kind is unset.
        A kind's rope scaling takes the model's place."""
        kind = (self.kinds or {}).get(layer_type, LayerKind())
        return ResolvedKind(
            window=kind.window,
            bidirectional_window=kind.bidirectional_window,
            chunk=kind.chunk,
            num_heads=self.num_heads if kind.num_heads is None else kind.num_heads,
            num_kv_heads=self.kv_heads if kind.num_kv_heads is None else kind.num_kv_heads,
            rope_theta=self.rope_theta if kind.rope_theta is None else kind.rope_theta,
            rope_scaling=self.rope_scaling if kind.rope_scaling is None else kind.rope_scaling,
            head_dim=(self.features_per_head if kind.head_dim is None else kind.head_dim),
            partial_rotary_factor=self._kind_rotary(kind),
            sinks=self.attention_sinks if kind.sinks is None else kind.sinks,
            qk_norm=self.qk_norm if kind.qk_norm is None else kind.qk_norm,
            mixer=kind.mixer,
            mixer_conv=kind.mixer_conv,
            feedforward_conv=kind.feedforward_conv,
            depth_sites=kind.depth_sites)

    def _kind_rotary(self, kind: LayerKind) -> float | None:
        """The fraction of a kind's head its rotary turns, None for all of
        it: the kind's own, else the model's. A window changes what a layer
        reads, not how much of its head turns; a family whose windowed layers
        turn another fraction (Gemma 4's whole heads) states it on the kind."""
        factor = (kind.partial_rotary_factor if kind.partial_rotary_factor is not None
                  else self.partial_rotary_factor)
        return None if factor == 1.0 else factor

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

    def mixer_context(self, kind: "ResolvedKind", layer_type: str, kv_shared: bool,
                      dex: LayerDex | None = None, dex_start: DexStart | None = None,
                      routed: bool = False) -> MixerContext:
        """Return one layer's mixer geometry as a `MixerContext` built from the resolved kind.

        `head_dim`, `rope_theta`, `window`, the rope scaling, the partial
        rotary, the sinks and the query-key norm take the layer kind's resolved values
        (`kind_of`), and the precision the model's layers take (`products`). Every other field
        is the model's field of the same name. A kind builds its `DecoderBlock` factory
        from this context and its own record, and `setup` chooses the mixer
        there and nowhere else.
        """
        resolved = {
            "num_heads": kind.num_heads, "num_kv_heads": kind.num_kv_heads, "head_dim": kind.head_dim,
            "rope_theta": kind.rope_theta, "rope_scaling": kind.rope_scaling,
            "sliding_window": kind.window, "bidirectional_window": kind.bidirectional_window,
            "attention_chunk": kind.chunk,
            "k_eq_v": self.attention_k_eq_v and kind.window is None,
            "kv_shared": kv_shared, "kv_store_key": layer_type,
            "partial_rotary_factor": kind.partial_rotary_factor, "attention_sinks": kind.sinks,
            "qk_norm": kind.qk_norm, "init_std": self.init_stds[0], "output_init_std": self.init_stds[1],
            "precision": self.precision, "dex": dex, "dex_start": dex_start,
            "value_experts": self.value_experts if routed else None,
            "mixture": self.mixture if routed else None,
            # A decoder forms its logits as its kernels do; a block needing them wide asks so itself
            # (`MixerContext.wide_logits`, the world model's).
            "wide_logits": False}
        return MixerContext(**resolved, **{field.name: getattr(self, field.name)
                                           for field in dataclasses.fields(MixerContext)
                                           if field.name not in resolved})

    def bank_sites(self, store: Variables) -> tuple[DecoderBank, ...]:
        """Where this decoder's layer stack is stored in `store`, the variables its banks are drawn
        from: scanned runs, or one layer per bank.

        `groups` already records which of the two it is. A scanned stack
        declares its runs and a plain loop declares one-layer runs, and
        `run_stack` fetches a one-layer run the same way it fetches a longer
        one. A run ends where the store's rows differ (`StackView.layer_structures`:
        layers a checkpoint quantizes differently, or adapts in part), so an
        unbound decoder plans its runs from the store's shapes as a decoder
        bound to them reads them. `scanned` records whether those fetches are
        sequenced, which a host layout requires.
        """
        return (DecoderBank((), self.stored_view(store), scanned=decoder_stages.scanned()),)

    def stored_view(self, store: Variables) -> StackView:
        """How `store` holds this decoder's layer stack (`bank_sites`), as a decoder bound to its shapes
        plans its runs (`SteppedFeedForward`)."""
        return self._view_of(store, lambda module: StackView(module.groups))

    @nn.nowrap
    def _view_of(self, store: Variables, read: Callable[[Self], StackView]) -> StackView:
        """`read` of this decoder bound to `store`'s shapes: through the model's own `apply`, which an
        adapted model runs under its interceptor, so its runs split where its adapter's targets do
        (`scanned_like`), as its forward splits them. The view comes back as `read` returns it: Flax types
        `apply`'s result as anything or a pair."""
        if self.scope is not None:
            return read(self)
        views: list[StackView] = []
        shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x)), store)
        self.apply(shapes, method=lambda module: views.append(read(module)))
        (view,) = views
        return view

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
        # The depths' feed-forward kind fills no mixer slot, so only its name is checked here.
        feeding = ({self.mtp_feedforward_type} if self.num_nextn_predict_layers and self.mtp_feedforward_type
                   else set())
        unnamed = sorted(set(self.kinds or {}) - set(types) - prediction_kinds - feeding)
        if unnamed:
            raise ValueError(
                f"kinds {unnamed} name no layer of this model, whose pattern is "
                f"{sorted(set(types))}")
        return {layer_type: self.kind_of(layer_type) for layer_type in set(types) | prediction_kinds}

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Real-row hidden-state work, summing the same block factories as setup (`decoder_counts`). A
        context's cross-attention work is `cross_flops`, which needs the context's lengths."""
        return decoder_counts.model_flops(self, lengths)

    @nn.nowrap
    def parameter_flops(self, lengths) -> float:
        """Learned matrices, separately from nonparameter attention/state work."""
        return decoder_counts.trained_flops(self, lengths)

    @nn.nowrap
    def projections(self) -> dict[str, tuple[int, int]]:
        """The trunk's own maps as `{name: (inputs, outputs)}`, the head
        included: what setup builds and the counts read."""
        sites = {}
        if self.per_layer_input_dim:
            sites['per_layer_model_projection'] = (self.emb_features,
                                                   self.num_layers * self.per_layer_input_dim)
        if self.altup is not None:
            sites.update({f'{prefix}_{index}': (self.emb_features, self.emb_features)
                          for prefix in ('altup_projections', 'altup_unembed_projections')
                          for index in range(self.altup.num_inputs - 1)})
        if self.loop is not None and self.loop.exit_gate:
            sites['early_exit_gate'] = (self.emb_features, 1)
        if self.embedding_features is not None:
            sites['project_in'] = (self.embedding_features, self.emb_features)
            sites['project_out'] = (self.emb_features, self.embedding_features)
        if not self.tie_embeddings:
            sites['lm_head'] = (self.embedding_features or self.emb_features, self.vocab_size)
        if self.head_transform is not None:
            sites['head_dense'] = (self.emb_features, self.emb_features)
        for index, level in enumerate(() if self.continuum is None else continuum_chain(self.continuum)):
            sites[f'continuum_{index}_down'] = (self.emb_features, level.hidden)
            sites[f'continuum_{index}_up'] = (level.hidden, self.emb_features)
        return sites

    @nn.nowrap
    def counted_blocks(self) -> list[DecoderBlock]:
        """Each layer's block as setup builds it, unbound, which its counts read (`decoder_counts`)."""
        types, mixer = self.per_layer_types, self.layer_mixer
        specs = self.layer_specs(types, self.layer_kinds(types), mixer)
        factories = feedforward_factories(self)
        return [self._block(specs, mixer, factories, self.wiring, index, f'layers_{index}')
                for index in range(self.num_layers)]

    @nn.nowrap
    def prediction_blocks(self) -> list[MTPBlock]:
        """Each prediction depth's block as setup builds it, unbound, which its counts read."""
        types, factories = self.per_layer_types, feedforward_factories(self)
        return decoder_predictions.depths(self, types, self.layer_kinds(types), self.layer_mixer,
                                          factories[0], factories[1], self.wiring)

    @nn.nowrap
    def replicated_flops(self, stages: int, lengths) -> float:
        """The work of the layers a pipeline of `stages` runs on every stage rather than once
        (`stage_split`): those before and after the run its stages repeat."""
        return decoder_counts.replicated_flops(self, stages, lengths)

    @nn.nowrap
    def layer_passes(self, index: int) -> int:
        """How many times a call runs layer `index`: a loop's steps for a layer it repeats, else once."""
        return decoder_counts.layer_passes(self, index)

    @nn.nowrap
    def logits_flops(self, lengths) -> float:
        """The vocabulary matrix and optional prediction transform, `project_out`'s included. A tied
        head reads the embedding table, `vocab_size` rows of the embedding's width."""
        return decoder_counts.logits_flops(self, lengths)

    @nn.nowrap
    def prediction_flops(self, lengths, *, prediction_lengths=None) -> float:
        """Each trained prediction depth shortens the real sequence by one token."""
        return decoder_counts.prediction_flops(self, lengths, counted_flops, prediction_lengths)

    @nn.nowrap
    def prediction_parameter_flops(self, lengths, *, prediction_lengths=None) -> float:
        """The same shortened depth forwards' learned matrices, head included."""
        return decoder_counts.prediction_flops(self, lengths, parameter_flops, prediction_lengths)

    @nn.nowrap
    def prediction_matrix_usage(self, path: tuple[str, ...], lengths, *, prediction_lengths=None):
        """Each depth's maps run once; the shared head runs on every shorter state."""
        return decoder_counts.prediction_matrix_usage(self, path, lengths, prediction_lengths)

    @nn.nowrap
    def matrix_usage(self, path: tuple[str, ...], lengths, *, logits: bool = False):
        """Native matrix scopes declare head versus trunk work and repeated loop passes."""
        return decoder_counts.matrix_usage(self, path, lengths, logits=logits)

    def setup(self):
        """Build the embeddings, the layers, the prediction depths `mtp`, the final
        `norm`, and `lm_head` when the embeddings are not tied.

        Each layer has one `LayerSpec`, is built by `block`, and is scanned in
        `groups`.
        """
        types = self.per_layer_types
        kinds = self.layer_kinds(types)
        refuse_unbuildable(self, kinds)
        projections = self.projections()
        embeddings.build(self, projections)
        gated_mlp, routed, parallel = feedforward_factories(self)
        # None is today's attention; a kind names its own mixer on LayerKind
        # and otherwise rides the model's. Both build over the layer's
        # context.
        mixer_spec = self.layer_mixer
        specs = self.layer_specs(types, kinds, mixer_spec)
        wiring = self.wiring

        block = functools.partial(self._block, specs, mixer_spec, (gated_mlp, routed, parallel), wiring)

        self.specs = specs
        # What one scanned body shares: the layers' spec and how their variables are stored. A capture's
        # outputs (`capture_intermediates`, whatever its filter keeps: a submodule a method returns among
        # them) are no stored variable (row 11882).
        stored = {name: tree for name, tree in self.variables.items() if name != _INTERMEDIATES}
        structures = StackView.layer_structures(stored, self.num_layers)
        self.scan_keys = tuple(self.scanned_like(index, key)
                               for index, key in enumerate(zip(specs, structures, strict=True)))
        self.block = block
        self.layers = [block(index, f'layers_{index}') for index in range(self.num_layers)]
        self.groups = (tuple((first + start, count) for first, end in self._loop_spans()
                             for start, count in scan_groups(self.scan_keys[first:end], self.bank_layers))
                       if decoder_stages.scanned() else tuple((index, 1) for index in range(self.num_layers)))
        self.mtp = decoder_predictions.depths(self, types, kinds, mixer_spec, gated_mlp, routed, wiring)
        if self.dspark is not None:
            assert routed is not None
            self.dspark_stages = decoder_drafts.drafting_stages(self, kinds, mixer_spec, routed, wiring)
        if self.block_draft is not None:
            self.block_drafter = BlockDrafter(self.block_draft, self.emb_features, self.vocab_size,
                                              self.kv_cache, self.max_seq_len, dtype=self.dtype,
                                              precision=self.precision)
        if self.altup is not None:
            # The copies past the first enter through their own projections
            # and leave through their own (modeling_gemma3n.py,
            # Gemma3nTextModel.altup_projections and altup_unembed_projections).
            projection = functools.partial(
                Dense, use_bias=False, dtype=self.dtype,
                precision=self.precision, dot_general=float32_dot_general)
            self.altup_projections = [
                projection(projections[f'altup_projections_{index}'][1], name=f'altup_projections_{index}')
                for index in range(self.altup.num_inputs - 1)]
            self.altup_unembed_projections = [
                projection(projections[f'altup_unembed_projections_{index}'][1],
                           name=f'altup_unembed_projections_{index}')
                for index in range(self.altup.num_inputs - 1)]
        hc = self.hyper_connections
        if hc is not None and self.streams_end_stack and hc.head in ('weighted', 'gated'):
            self.hc_head = stream_head(hc, self.emb_features, self.norm_eps, name='hc_head')
        if self.attention_residuals is not None:
            self.output_res = DepthAttention(emb_features=self.emb_features, norm_eps=self.norm_eps,
                                             normalize_values=self.attention_residuals.normalize_values,
                                             scaled=self.attention_residuals.scaled,
                                             gained=self.attention_residuals.gained,
                                             name='output_res')
        if self.engram is not None:
            self.engram_hashes = EngramHashes(spec=self.engram, vocab_size=self.vocab_size,
                                              name='engram_hashes')
        self._final_norm()
        if self.loop is not None and self.loop.implicit:
            # Per channel, as a norm's scale (a row of it per later pass under `theta_passes`): no logical
            # axes, placed whole (`Layout.min_shard`).
            rows = () if self.loop.theta_passes is None else (self.loop.theta_passes - 1,)
            self.loop_theta = self.param('loop_theta', nn.initializers.zeros, (*rows, self.emb_features),
                                         jnp.float32)
        if self.loop is not None and self.loop.exit_gate:
            # Drawn at zero, every pass's hazard a half, where a loaded decoder gains the gate
            # (`Pretrained.with_model`); Ouro's release stores its own.
            self.early_exit_gate = Dense(
                projections['early_exit_gate'][1], dtype=self.dtype, precision=self.precision,
                name='early_exit_gate', kernel_init=nn.initializers.zeros, dot_general=float32_dot_general)
        decoder_head.build(self, projections)

    @nn.nowrap
    def layer_specs(self, types: Sequence[str], kinds: dict[str, ResolvedKind], mixer_spec
                     ) -> tuple[LayerSpec, ...]:
        """One `LayerSpec` per layer: its kind, whether it routes and how,
        its feed-forward width and sparsity, the KV it shares or provides,
        and its attention-residual, engram and DSpark slots."""
        sparse, hashed, sharing = self.sparse_layers, self.hash_layers, self.kv_sharing
        widths, sparsity = self.mlp_widths, self.activation_sparsity_pattern
        providers = set(sharing.values())

        def provides(index: int, layer_type: str) -> bool:
            mixer = kinds[layer_type].mixer or mixer_spec
            return index in providers or (mixer is not None and mixer.publishes(index in sharing))

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
                cross=((self.cross_attention.layer_key(self, index),)
                       if self.cross_attention is not None and index in self.cross_attention.layer_ids
                       else None),
                prediction_target=(index if self.drafts_blocks and index in decoder_drafts.tap_layers(self)
                                   else None),
                attention_norm=index > 0 or self.first_attention_norm,
                dex=None if self.dex is None else self.dex.layer(index),
                streams=layer_streams(self.hyper_connections, index, self.num_layers, self.loop is not None),
                messaged=self._messaged(index))
            for index, layer_type in enumerate(types))

    @nn.nowrap
    def _messaged(self, index: int) -> bool:
        """Whether layer `index` is in a looped block that passes messages (`Loop.messages`)."""
        if self.loop is None or self.loop.messages is None:
            return False
        first, end = self.loop.layers or (0, self.num_layers)
        return first <= index < end

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
            mixer=(spec.kind.mixer or mixer_spec).build(self.mixer_context(
                spec.kind, spec.layer_type, spec.kv_shared, spec.dex, self._dex_start(index), spec.routed)),
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
                EngramLayer if self.engram.ple_conv_kernel_size is None else functools.partial(
                    PLELayer, kernel_size=self.engram.ple_conv_kernel_size,
                    dilation=self.engram.max_ngram_size, parts=self.engram.table_parts,
                    use_bias=self.engram.ple_bias),
                rows=self.engram.num_embeddings[spec.engram], head_dim=self.engram.head_dim,
                hc_mult=self.hyper_connections.hc_mult if self.hyper_connections else 1,
                emb_features=self.emb_features, norm_eps=self.norm_eps,
                dtype=self.dtype, precision=self.precision),
            engram_index=spec.engram,
            cross=(self.cross_attention.block(self, index)
                   if spec.cross is not None and self.cross_attention is not None else None),
            prediction_target=spec.prediction_target is not None,
            prediction_site=('visits' if self.block_draft is not None
                             else 'input' if self.dspark is None else self.dspark.reads),
            attention_norm=spec.attention_norm,
            emb_features=self.emb_features,
            norm_eps=self.norm_eps,
            norm_type=self.norm_type,
            norm_bias=self.norm_bias,
            norms=self.norms,
            scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast,
            wiring=wiring,
            per_layer_input_dim=ple or 0,
            gate_activation=self.mlp,
            parallel=parallel if spec.routed else None,
            altup=self.altup,
            laurel_rank=self.laurel_rank,
            hyper_connections=None if spec.streams is None else self.hyper_connections,
            streams_at_ends=(False, False) if spec.streams is None else spec.streams,
            residual_site=spec.residual_site,
            depth_sites=spec.kind.depth_sites,
            mixer_conv=spec.kind.mixer_conv,
            feedforward_conv=spec.kind.feedforward_conv,
            cross_pass=self.loop.messages if spec.messaged and self.loop is not None else None,
            dropout_rate=self.dropout_rate,
            remat=self.remat,
            dtype=self.dtype,
            precision=self.precision,
            name=name)

    @nn.nowrap
    def mixer_of(self, index: int) -> nn.Module:
        """Layer `index`'s mixer, unbound, as its kind builds it over the layer's context."""
        types = self.per_layer_types
        spec = self.layer_specs(types, self.layer_kinds(types), self.layer_mixer)[index]
        context = self.mixer_context(spec.kind, spec.layer_type, spec.kv_shared, spec.dex,
                                     self._dex_start(index), spec.routed)
        return (spec.kind.mixer or self.layer_mixer).build(context)(parent=None)

    def scanned_like(self, index: int, key: Hashable) -> Hashable:
        """What layer `index`, of spec and stored structure `key`, runs and draws as under a scan (its entry
        of `scan_keys`), so like layers run as one scanned run (`groups`, `run_period`). A method interceptor
        that gives a layer parameters of its own (an adapter's factors and scales, `dew.lora`) keys it apart
        where they differ, so a run of layers adapted alike stacks one factor pair, and a layer adapted
        otherwise runs a run of its own."""
        del index
        return key

    def drawn_like(self, index: int) -> Hashable:
        """What layer `index` draws as: its scan key (`scanned_like`) and what its extension starts from
        (`_dex_start`), so like layers draw under one scan."""
        return (self.scan_keys[index], self._dex_start(index))

    @nn.nowrap
    def _dex_start(self, index: int) -> DexStart | None:
        """What layer `index`'s extension starts from, which its variables keep (`dew.nn.dex`)."""
        return None if self.dex is None else self.dex.start(index)

    @nn.nowrap
    def cross_flops(self, lengths, context_lengths) -> float:
        """The cross-attention blocks' work, which `model_flops` leaves out (`crossed_flops`)."""
        return crossed_flops(self, lengths, context_lengths)

    @nn.nowrap
    def held_cross_flops(self, lengths, context_lengths) -> float:
        """The part of `cross_flops` its own matrices do, which train as its trunk does (`crossed_flops`)."""
        return crossed_flops(self, lengths, context_lengths, held=True)

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
        if hc is not None and hc.hc_layers is None:
            # The embeddings copied into every residual stream: [B, S, hc_mult, D].
            return enter_streams(x, hc)
        if depth is not None:
            # The embeddings as the first partial sum after empty blocks: [B, S, blocks + 1, D].
            return jnp.pad(x[:, :, None], ((0, 0), (0, 0), (depth.blocks(self.num_layers), 0), (0, 0)))
        return x

    @property
    def streams_end_stack(self) -> bool:
        """Whether the stack hands the final norm mHC's streams, which its own head collapses
        (`HyperConnections.ends_stack`)."""
        hc = self.hyper_connections
        return hc is not None and hc.ends_stack(self.num_layers, self.loop is not None)

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
        # Streams over a range of layers the stack does not end in left it plain (`leave_streams`).
        hc = self.hyper_connections
        if hc is not None and self.streams_end_stack:
            x = leave_streams(x, hc, self.hc_head if hc.head in ('weighted', 'gated') else None)
        if self.attention_residuals is not None:
            # Every block is finished after the last layer, so the whole state,
            # the blocks and the last partial, is what the output site mixes
            # (modeling_kimi_linear.py:1215-1233).
            x = self.output_res(x)
        return streams, x

    def __call__(self, tokens, train: bool = False, decode: bool = False,
                 positions=None, segment_ids=None,
                 input_embeddings=None,
                 attention_mask=None, image_groups=None, block_groups=None, rotary_positions=None,
                 attention_pairwise_mask=None, attention_key_positions=None, attention_key_runs=None,
                 attention_key_valid=None,
                 right_padded: bool = False, coordinates: Coordinates | None = None,
                 conditioning: Mapping[str, jax.Array] | None = None, progress: Progress | None = None,
                 routed_experts=None, routed=None, loop_passes=None):
        # The call's token layout, converted once (`token_coordinates`): the trunk and the
        # prediction depths read its coordinates alone. A routing replay (`routed_experts`, `routed`) as
        # `hidden_and_mtp_inputs` reads it.
        coordinates = token_coordinates(tokens.shape[:2], coordinates, segments=segment_ids,
                                        valid=attention_mask, positions=positions, right_padded=right_padded)
        x, prediction = self.hidden_and_mtp_inputs(
            tokens, train=train, decode=decode, input_embeddings=input_embeddings, image_groups=image_groups,
            block_groups=block_groups, rotary_positions=rotary_positions,
            attention_pairwise_mask=attention_pairwise_mask, attention_key_positions=attention_key_positions,
            attention_key_runs=attention_key_runs, attention_key_valid=attention_key_valid,
            coordinates=coordinates, conditioning=conditioning,
            progress=progress, routed_experts=routed_experts, routed=routed, loop_passes=loop_passes)
        if self.is_initializing() and self.drafts_blocks:
            decoder_drafts.reach(self, tokens, x.dtype)
        if self.is_initializing() and self.mtp:
            # Flax creates a parameter where a call first reaches it, and the
            # main forward never enters the prediction depths. Reaching them
            # here, during init only, makes the model's tree the model's
            # business: a plain init holds every depth.
            decoder_predictions.depth_states(
                self, prediction, tokens, train=train, input_embeddings=input_embeddings,
                image_groups=image_groups, block_groups=block_groups, rotary_positions=rotary_positions,
                coordinates=coordinates)
        return self.logits_from_hidden(x)

    def states_and_logits(self, tokens, **kwargs):
        """Return the prediction-input states and the logits from one forward pass.

        A speculative decoder verifies with both. The logits give the target
        distribution, and the states seed the next block's prediction depths.
        The states are the normalized states, except for V4, where they are
        the uncollapsed residual streams.
        """
        x, prediction = self.hidden_and_mtp_inputs(tokens, **kwargs)
        return prediction, self.logits_from_hidden(x)

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
        return prediction, self.logits_from_hidden(x[rows, slots])

    @property
    def in_context(self) -> bool:
        """Whether the model holds a continuum level (`SteppedInContext`)."""
        return self.continuum is not None

    @property
    def in_context_levels(self) -> int:
        """The continuum levels the model holds, a chain's count (`SteppedInContext`)."""
        return 0 if self.continuum is None else len(continuum_chain(self.continuum))

    def head_bounds(self, states) -> HeadTerms:
        """What bounds a slot loss's curvature and slope in the `states` `head_logits` reads
        (`decoder_head.bounds`)."""
        return decoder_head.bounds(self, states)

    @nn.nowrap
    def in_context_level(self, variables):
        """The continuum level's parameters in `variables`, None without a level (`SteppedInContext`)."""
        return None if self.continuum is None else level_parameters(self.continuum, variables["params"])

    @nn.nowrap
    def with_in_context_level(self, variables, level):
        """`variables` holding `level` as the continuum level's parameters (`SteppedInContext`)."""
        return {**variables, "params": with_level_parameters(self.continuum, variables["params"], level)}

    def continuum_states(self, hidden, index: int | None = None):
        """Final states `hidden` past the continuum levels, the head's input, or past level `index` alone
        (`chain_states`), read at the levels' own parameters (`SteppedInContext`)."""
        if self.continuum is None:
            raise ValueError("this model has no continuum level")
        return chain_states(self.continuum_level, hidden, index)

    def level_terms(self, hidden, parts: tuple[str, ...], index: int = 0, passed: bool = False):
        """What bounds a damped step of continuum level `index`'s `parts` at its states `hidden`
        (`Continuum.step_terms`), its maps' inputs as a checkpoint's quantizers round them."""
        if self.continuum is None:
            raise ValueError("this model has no continuum level")
        return level_of(self.continuum_level, index).step_terms(
            hidden, parts, functools.partial(decoder_head.quantizer, self), passed)

    def head_scores(self, x):
        """The head's scores of `x` before the final logit cap (`decoder_head.head_scores`)."""
        return decoder_head.head_scores(self, x)

    def head_logits(self, x):
        """The fp32 head over states past the continuum levels (`continuum_states`), or final states
        (`decoder_head.head_logits`)."""
        return decoder_head.head_logits(self, x)

    def mtp_hidden_states(self, hidden, tokens, train: bool = False, input_embeddings=None, image_groups=None,
                          block_groups=None, rotary_positions=None, coordinates: Coordinates | None = None,
                          conditioning: Mapping[str, jax.Array] | None = None):
        """One final-normed state array per shifted prediction depth (`decoder_predictions.depth_states`)."""
        return decoder_predictions.depth_states(
            self, hidden, tokens, train=train, input_embeddings=input_embeddings, image_groups=image_groups,
            block_groups=block_groups, rotary_positions=rotary_positions, coordinates=coordinates,
            conditioning=conditioning)

    def mtp_logits(self, hidden, tokens, train: bool = False, input_embeddings=None,
                   image_groups=None, block_groups=None, rotary_positions=None,
                   coordinates: Coordinates | None = None,
                   conditioning: Mapping[str, jax.Array] | None = None):
        """Return the shared language head's logits for each prediction depth's hidden states; a
        call's `conditioning` is the trunk's (`mtp_hidden_states`)."""
        return [self.logits_from_hidden(state) for state in decoder_predictions.depth_states(
            self, hidden, tokens, train=train, input_embeddings=input_embeddings, image_groups=image_groups,
            block_groups=block_groups, rotary_positions=rotary_positions, coordinates=coordinates,
            conditioning=conditioning)]

    def mtp_step(self, hidden, tokens, *, depth: int = 0, input_embeddings=None, rotary_positions=None,
                 decode: bool = False, prediction_phase: PredictionPhase = "ordinary",
                 coordinates: Coordinates | None = None):
        """One unshifted prediction step, optionally appending to the depth's own KV cache
        (`decoder_predictions.depth_step`)."""
        return decoder_predictions.depth_step(
            self, hidden, tokens, depth=depth, input_embeddings=input_embeddings,
            rotary_positions=rotary_positions, decode=decode, prediction_phase=prediction_phase,
            coordinates=coordinates)

    def token_embeddings(self, tokens):
        """The embeddings a prediction depth pairs with `tokens` (`embeddings.token_embeddings`)."""
        return embeddings.token_embeddings(self, tokens)

    def stream_embeddings(self, tokens):
        """`token_embeddings` at the stream's width (`embeddings.stream_embeddings`)."""
        return embeddings.stream_embeddings(self, tokens)

    def scaled_embeddings(self, x):
        """The token embeddings `x` scaled as the first layer reads them (`embeddings.scaled_embeddings`)."""
        return embeddings.scaled_embeddings(self, x)

    @property
    def embedding_factor(self) -> float:
        """The factor `scaled_embeddings` scales an embedding by (`embeddings.embedding_factor`)."""
        return embeddings.embedding_factor(self)

    def draft(self, context, tokens, *, decode: bool = True, coordinates: Coordinates | None = None,
              choose=None):
        """Draft a block after each row's last context position (`decoder_drafts.draft`)."""
        return decoder_drafts.draft(self, context, tokens, decode=decode, coordinates=coordinates,
                                    choose=choose)

    def reach_drafter(self, tokens, dtype):
        """Create the block drafter's parameters, which no forward pass reaches (`decoder_drafts.reach`)."""
        return decoder_drafts.reach(self, tokens, dtype)

    def draft_context(self, prediction_inputs: Mapping) -> jax.Array:
        """The block drafter's context `[B, S, targets * D]` from `prediction_inputs`
        (`decoder_drafts.draft_context`)."""
        return decoder_drafts.draft_context(self, prediction_inputs)

    def init_draft_cache(self, batch_size: int):
        """Open the block drafter's caches for `batch_size` rows, apart from the trunk's, as its stages (or
        its Markov drafter) state them (`decoder_cache.named_caches`), tracing no forward."""
        if self.block_draft is not None:
            put_opened(self, named_caches([self.block_drafter], batch_size))
            return
        assert self.dspark is not None
        put_opened(self, named_caches(self.dspark_stages, batch_size))

    @property
    def drafts_blocks(self) -> bool:
        """Whether the model carries a block drafter (`dspark` or `block_draft`; `BlockDrafting`)."""
        return self.dspark is not None or self.block_draft is not None

    @nn.nowrap
    def visits(self) -> tuple[int, ...]:
        """The layer each run of a forward runs, in order: a looped block's once a pass (`Loop.steps`), the
        layers before and after it once; what a Markov block drafter's taps index (`MarkovBlockDraft`)."""
        if self.loop is None:
            return tuple(range(self.num_layers))
        first, end = self.loop.layers or (0, self.num_layers)
        return (*range(first), *tuple(range(first, end)) * self.loop.steps, *range(end, self.num_layers))

    def anchor_embeddings(self, tokens):
        """`tokens` `[B]` embedded as the first layer reads them (`token_embeddings`, `scaled_embeddings`):
        a Markov block drafter's anchors (vLLM's `Inkling2DSparkDecoder.embed_block`)."""
        return self.scaled_embeddings(self.token_embeddings(tokens))

    def init_mtp_cache(self, batch_size: int):
        """Open the prediction depths' caches for `batch_size` rows, apart from the trunk's, as the depths
        state them (`decoder_cache.named_caches`), tracing no forward."""
        put_opened(self, named_caches(self.mtp, batch_size))

    def hidden_states(self, tokens, *, loop_passes=None, **kwargs):
        """Return the final normalized states, without the vocabulary projection, its loop at `loop_passes`
        passes where a call gives a count (`hidden_and_mtp_inputs`)."""
        return self.hidden_and_mtp_inputs(tokens, loop_passes=loop_passes, **kwargs)[0]

    @nn.nowrap
    @pre_call_capture
    def capture_layers(self, module: nn.Module, method: str) -> bool:
        """The `capture_intermediates` filter that keeps every layer's output (`LayerStates`).

        It matches the `__call__` of the stack's `layers_N` modules, a scanned
        run's `layers_3_7` and a pipeline stage's `layers_j` included, and
        `StackView.unstack` lays what those sow out per layer, as the plain loop
        does; `layer_state` reads one layer back. Neither is a Flax method, so
        the filter the capture calls on every module runs no module call.
        """
        return method == "__call__" and group_layers(module.name or "") is not None

    layer_collections: ClassVar[tuple[str, ...]] = (_INTERMEDIATES, _FINAL)
    """The collections a reader of layer states opens (`LayerStates`)."""

    @nn.nowrap
    def layer_state(self, collections: Mapping[str, Mapping], index: int) -> jax.Array:
        """Layer `index`'s output out of what the forward kept (`LayerStates`): the last layer's as
        the stack returns it, and another's as `capture_layers` kept it, the last call's, a looped
        layer's last pass; a model that carries several residual copies (AltUp's) returns them all
        at every layer. `index` counts as a Python index does, -1 the last layer."""
        if not -self.num_layers <= index < self.num_layers:
            raise ValueError(f"a {self.num_layers}-layer model has no layer {index}")
        layer = index % self.num_layers
        final = collections.get(_FINAL, {}).get('states')
        if layer == self.num_layers - 1 and final is not None:
            return final
        kept = collections.get(_INTERMEDIATES, {}).get(f"layers_{layer}")
        if kept is None:
            held = sorted(number for number in map(layer_index, collections.get(_INTERMEDIATES, {}))
                          if number is not None)
            raise ValueError(f"the model kept no layer {layer}; it kept layers {held}")
        return kept["__call__"][-1]

    def logits(self, tokens, *, train: bool = False, loop_passes=None, **fields):
        """Return `__call__`'s fp32 logits over `hidden_states(tokens, **fields)`, which
        write no cache unless `fields` ask to `decode`, its loop at `loop_passes` passes where given."""
        return self.logits_from_hidden(self.hidden_states(tokens, train=train, loop_passes=loop_passes,
                                                          **fields))

    def logits_from_hidden(self, hidden):
        """Return the logits of final states `hidden`: the head `__call__` and every MTP depth score with,
        the continuum level first where the model has one."""
        return decoder_head.head_logits(
            self, hidden if self.continuum is None else chain_states(self.continuum_level, hidden))

    def hidden_and_mtp_inputs(self, tokens, train: bool = False, decode: bool = False,
                              input_embeddings=None, image_groups=None, block_groups=None,
                              rotary_positions=None, attention_pairwise_mask=None,
                              attention_key_positions=None, attention_key_runs=None, attention_key_valid=None,
                              routed_experts=None, routed=None, media_mask=None, admitted=None,
                              coordinates: Coordinates | None = None,
                              conditioning: Mapping[str, jax.Array] | None = None,
                              whole_blocks: bool = False, progress: Progress | None = None,
                              loop_passes=None):
        """Return the final normalized states and the prediction depths' input.

        V4's depth reads the raw residual streams from before the collapse head
        and the final norm; other depths read the final normalized states.
        The tokens' layout is their `coordinates`, the one description of them
        every layer reads; only `__call__` takes flat layout fields and converts
        them (`token_coordinates`, ledger 1122).

        - `input_embeddings` `[B, S, D]` replace the scaled token embeddings,
          for a caller that fuses in another encoder's output. The token ids
          still feed the per-layer inputs and routing.
        - `image_groups` and `block_groups` `[B, S]` number the tokens' images
          and blocks, -1 outside any, which the attention layers that read them
          see whole (`dew.nn.inputs.AttentionMetadata`).
        - `attention_pairwise_mask` is an explicit [B, queries, keys]
          visibility mask for ordinary attention, and the optional
          `attention_key_positions` [B, keys] give key positions for local
          windows. Both describe the cached reads of this call only.
          `attention_key_runs` [B, queries, R, 2] replaces the mask with each
          query's key intervals, whose union it reads, R of them, static, its
          key positions windowed within each run alone (the template's spans,
          `dew.nn.attention.scaled_dot_product_attention`'s `key_runs`).
        - `routed_experts` `[B, S, layers, top_k]` (or one such a routed site, keyed by
          it, `LayerInputs.experts`, where a layer routes at several) replays a rollout engine's
          routing (vLLM's `routed_experts`, SGLang's
          `meta_info.routed_experts`), and `routed` `[B, S]` marks the tokens
          it covers (`dew.nn.protocols.Routes`).
        - `media_mask` [B, S] marks the positions a media encoder fills.
          Engram keeps them out of its n-grams, and a media-biased router
          selects for them.
        - `admitted` makes a cached call a serving step's mixed call
          (`dew.nn.inputs.Admitted`).
        - `coordinates` are the tokens' documents, validity and positions as
          the batch's builder made them (`dew.batch.coordinates`, with the
          packer's `max_segments`, and `right_padded` where each row holds its
          real tokens first); None for one document a row, every token real.
          Their `forks` make documents that continue another (`dew.batch.Forks`):
          each reads its ancestors as its own beginning, attention their keys and a
          recurrent layer the state they leave, a shared prefix computed once. A
          cached call over forks (a speculative tree's verify) follows the row's
          cache and leaves its recurrent state as it was.
        - `conditioning` carries the sequence the `cross_attention` layers read
          (`dew.nn.inputs.CONTEXT`) and its rows' lengths (`CONTEXT_LENGTHS`).
          A cached call that carries it keeps its keys and values for the
          calls after it. A model without cross-attention takes none.
        - `whole_blocks` states that each of `block_groups`' blocks is its
          document's whole run of real tokens (`AttentionMetadata.whole_blocks`).
        - `progress` is a training step's (`dew.nn.progress`), which its DEX
          layers anneal over; None is the run's end, outside a training step.
        - `loop_passes`, a traced int32 scalar the same on every device, is
          how many of its `loop`'s passes a training call runs, from one
          program of `loop.steps` (`looping.run_loop`): a count drawn per
          microbatch. None runs every pass.
        """
        if decode and coordinates is not None and coordinates.max_segments > 1:
            # A cache keeps a packed row's real tokens alone (`real_slots`): its padding writes no key and
            # moves no recurrent state, its documents' starts counted over its real tokens.
            laid = {COORDINATES_FIELD: coordinates}
            coordinates = with_validity(laid, real_slots(laid, tokens.shape[:2]))[COORDINATES_FIELD]
        attention_metadata = call_metadata(
            self, tokens, decode, image_groups, block_groups, rotary_positions, attention_pairwise_mask,
            attention_key_positions, attention_key_runs, media_mask, admitted, coordinates, conditioning,
            whole_blocks=whole_blocks, progress=progress, key_valid=attention_key_valid, train=train)
        # The stack's entry and exit sit where the batch does, so neither the
        # lookup nor the head is computed whole on the shards of an axis that
        # splits the rows or the positions.
        x = self._entered(tokens, train, decode, input_embeddings, attention_metadata)
        ple = self._layer_inputs(tokens, x, routed_experts, routed)
        residual = self.looped(
            self._expand(x),
            train=train,
            decode=decode,
            per_layer_input=ple,
            attention_metadata=attention_metadata,
            passes=loop_passes,
        )
        return self._exited(residual)

    def _entered(self, tokens, train: bool, decode: bool, input_embeddings,
                 attention_metadata) -> jax.Array:
        """The stack's input: the tokens' scaled embeddings (or the caller's `input_embeddings`), their
        learned positions, the embeddings' norm and dropout; the unscaled embeddings a prediction depth
        reads sown where a caller opens their collection."""
        x = constrain(prepared_embeddings(self.scaled_embeddings(self.token_embeddings(tokens)),
                                     input_embeddings), RESIDUAL)
        if self.position_embedding == 'learned':
            # The tokens' places as every layer reads them (`Coordinates`), whichever fields the
            # caller gave them in; a cached call's continue from the model's own count of each row's
            # real tokens (`DecodeCursor`), whatever mixer its first layer runs.
            tokens_at = coordinates_of(attention_metadata, *tokens.shape)
            valid = jnp.ones(tokens.shape, bool) if tokens_at.valid is None else tokens_at.valid
            # Every cached call counts its real tokens, those at given places too, so a later call
            # without places continues after them; a serving step's mixed call counts each row's own.
            admitted = None if attention_metadata is None else attention_metadata.admitted
            counted = (self.decode_cursor(valid, admitted) if decode
                       else prefix_sum(valid, dtype=jnp.int32) - 1)
            places = counted if tokens_at.positions is None else tokens_at.positions
            x = x + self.embed_positions(jnp.maximum(places + self.position_embedding_offset, 0))
        if self.embedding_norm and self.embedding_norm_reads == 'inputs':
            x = self.embedding_layernorm(x)
        if self.embedding_dropout_rate:
            x = self.embedding_dropout(x, deterministic=not train)
        # A prediction depth reads these unscaled embeddings, media
        # replacements in place, which token ids cannot rebuild. They are
        # computed only where a caller opens the collection, as a forward that
        # does not would trace a second input projection; init leaves it out.
        if not self.is_initializing() and self.is_mutable_collection("embeddings"):
            prepared = prepared_embeddings(self.stream_embeddings(tokens), input_embeddings)
            self.sow("embeddings", "prepared", prepared,
                     reduce_fn=lambda _, value: value, init_fn=lambda: prepared)
        return x

    def _exited(self, residual) -> tuple[jax.Array, jax.Array]:
        """The final normalized states and the prediction depths' input from the stack's output `residual`:
        its streams collapsed, normed and scaled as the head reads them, sown where a caller opens the last
        layer's or the depths' collection."""
        if not self.is_initializing() and self.is_mutable_collection(_FINAL):
            # The last layer's output as the stack returns it whatever runs it (`layer_state`),
            # before a model that carries several residual copies collapses them; it costs
            # nothing unless a caller opens the collection.
            self.sow(_FINAL, 'states', residual, reduce_fn=lambda _, value: value, init_fn=lambda: None)
        streams, x = self._collapse(residual)
        normed = self.exit_states(x)
        if (self.block_draft is not None and not self.is_initializing()
                and self.is_mutable_collection('prediction_inputs')):
            # A Markov block drafter's last tap (`decoder_drafts.draft_context`).
            self.sow('prediction_inputs', decoder_drafts.FINAL_STATES, normed,
                     reduce_fn=lambda _, value: value, init_fn=lambda: normed)
        hidden = self.head_states(normed)
        # V4's depth input: inference/model.py MTPBlock.forward at b5968e9.
        prediction = hidden if self.mtp_hyper_connections is None else streams
        if (self.mtp_hyper_connections is not None and not self.is_initializing()
                and self.is_mutable_collection('prediction_inputs')):
            self.sow('prediction_inputs', 'states', prediction,
                     reduce_fn=lambda _, value: value, init_fn=lambda: None)
        return hidden, prediction

    def looped(self, x, **stack):
        """Run the layer stack over `x` once, or as `loop` says (`looping.run_loop`): its block's later
        passes under one scan, so a step's program holds two passes whatever the loop's count."""
        return run_loop(self, x, **stack)

    def exit_states(self, x):
        """The stack's one residual `x` as it leaves for the head and the exit gate: through the final norm
        (`normed_output`), placed as the stream is. A loop's every exit leaves so (`looping.run_loop`), its
        residual one under a loop (`_collapse`), the last pass's through `_exited`."""
        return constrain(residual_normed(self.norm, x) if self.normed_output else x, RESIDUAL)

    @property
    def normed_output(self) -> bool:
        """Whether the head reads the stack's output through the final norm: not nGPT's normalized stream."""
        return self.ngpt is None and self.final_norm

    @nn.nowrap
    def exit_passes(self, collections: Mapping[str, Mapping]) -> ExitPasses:
        """What a forward with `exits` open gathered (`Looped`): a loop of one pass has no earlier pass."""
        exits = collections[EXITS]
        states = tuple(exits.get('states', ()))
        # A forward that ran every pass: each `True`, known as it traces.
        ran = tuple(exits.get('ran', ())) or (True,) * len(states)
        return ExitPasses(states, tuple(exits.get('logits', ())), ran)

    def head_states(self, normed):
        """The states the head reads from the final norm's `normed`: divided by `logits_scaling`, in at
        least fp32, where it is not 1."""
        if self.logits_scaling == 1.0:
            return normed
        # (h W) / s is (h / s) W: dividing the states in at least fp32 scores every head that contracts
        # them, the chunked losses' included, as lm-engine's logits * (1 / m_width) does.
        wide = at_least_fp32(normed.dtype)
        return normed.astype(wide) / jnp.asarray(self.logits_scaling, wide)

    def stack(self, x, *, train: bool, decode: bool, per_layer_input, attention_metadata=None,
              span: tuple[int, int] | None = None, prior_bank=None, kv_store=None):
        """Run the layer stack over `x` as the plain loop, as scanned runs, or as a pipeline over the stages
        (`decoder_stages.stack`), each layer of a looped block that passes messages reading `prior_bank`
        (`Loop.messages`), its layers' providers' publications read from and left in `kv_store` (a loop's
        part's, `dew.nn.backbones.looping`; a fresh one where None)."""
        return decoder_stages.stack(self, x, train=train, decode=decode, per_layer_input=per_layer_input,
                                    attention_metadata=attention_metadata, span=span, prior_bank=prior_bank,
                                    kv_store=kv_store)

    def residual_summary(self, x) -> jax.Array:
        """The `[B, S, D]` summary of the residual `x` a block of a looped pass leaves its pass after
        (`Loop.messages`): the stream itself, or under attention residuals the block's partial sum, the
        last slot of the depth state."""
        return x if self.attention_residuals is None else x[:, :, -1]

    def stage_sites(self, store: Variables, stages: int) -> tuple[DecoderBank, ...]:
        """Where the run a pipeline over `stages` (or one stage's scan) stacks is stored in `store`, the
        variables a stage-local state is made from (`decoder_stages.stage_view`): none for a looped stack, nor
        on one stage where no scan stacks a run (a stack that does not scan, or whose runs are one layer
        each)."""
        if self.loop is not None:
            return ()
        # A one-stage state another run stacked, which this stack does not scan, holds the runs this stack
        # scanning would (its periods among them): read as that stack's, it goes back onto its layers.
        held = 'stages' in dict(dict(store).get('params', {}))
        scans = decoder_stages.scanned() or (stages == 1 and held)
        with generation.chosen(layer_scan="scanned" if scans else "plain"):
            view = self._view_of(store, lambda module: decoder_stages.stage_view(module, stages))
        if stages == 1 and not (scans and any(count > 1 for _, count in view.groups)):
            return ()
        frozen = (stages == 1 and not view.holds(dict(store).get('params', {}))
                  and any(isinstance(tree, Mapping) and view.holds(tree)
                          for name, tree in store.items() if name != 'params'))
        return (DecoderBank((), view, frozen=frozen),)

    def _loop_spans(self) -> tuple[tuple[int, int], ...]:
        """The layer ranges this decoder runs separately (`run_loop`), in order: the whole stack without a
        loop, or its prelude, its block (each message block apart under `Loop.messages`) and its coda."""
        if self.loop is None:
            return ((0, self.num_layers),)
        first, end = self.loop.layers or (0, self.num_layers)
        block = (self.loop.messages.blocks(first, end) if self.loop.messages is not None
                 else ((first, end),) if end > first else ())
        return (*(((0, first),) if first > 0 else ()), *block,
                *(((end, self.num_layers),) if end < self.num_layers else ()))

    def banked_collections(self, span: tuple[int, int] | None = None) -> tuple[str, ...]:
        """Return the collections whose layer subtrees the store holds as banks.

        A store built per layer holds `layers_0`, and a store built bank by bank
        holds each multi-layer run under the run's name, such as `layers_0_15`.
        A run of one layer looks the same either way. A collection that holds a
        run's bank beside its layers, with different leaves in each (a host
        layout's frozen leaves), is not banked, because its rows still stack
        (`StackView.stack`). A bank and a row that hold the same leaf raise
        `ValueError`, and so do some banks without the others when there are
        no rows. Where `span` aligns with `groups`, checks only the runs covering
        that span (a looped pass's view hides the other spans' caches).
        """
        groups = self.groups
        if span is not None:
            first, end = span
            within = tuple(run for run in groups if first <= run[0] and run[0] + run[1] <= end)
            if sum(count for _, count in within) == end - first:
                groups = within
        return banked_collections(groups, self.variables)

    def residual_dtype(self, x, *, train: bool, decode: bool, per_layer_input,
                       attention_metadata=None, first: int = 0, kv_store=None) -> jnp.dtype:
        """Return the dtype the residual stream settles in: `x`'s promoted with layer `first`'s output.

        A scan carries one dtype through, and each pipeline stage takes the
        dtype the stage before returns. Under a bf16 policy the dense
        feed-forward returns fp32, so the loops use fp32 from the start. Layer
        `first` (the span's first layer, whose cache a loop's `PassView` exposes
        where earlier layers' are hidden) runs abstractly in its own scope (a
        placeholder 'params' key, which flax draws any other stream from), with
        the collections the call opened open and no other, so it takes the
        call's own path: opened beyond them, an attention sowing its QK-Clip
        maxima decided a kernel for a mask the call never builds (a cache made
        under a chosen template refused, row 4101). Its routes are no route the
        call takes (`generation.routes`, alone). It reads a copy of the
        publications the span enters with (`kv_store`), so a span opening on
        a layer that reads its provider's reads them. A banked store answers
        from its bank's first row without fetching it.
        """
        layer = self.layers[first]
        streams = x.streams if isinstance(x, Carried) else x
        assert self.scope is not None, "a bound decoder's stack"
        opened = self.scope.mutable
        with generation.routes(alone=True):
            output = jax.eval_shape(lambda held: layer.apply(
                held, x, mutable=opened, rngs={'params': jax.random.key(0)},
                train=train, decode=decode,
                kv_store=dict(kv_store) if kv_store is not None else ({} if self.provides else None),
                per_layer_input=None if per_layer_input is None else per_layer_input.layer(first),
                attention_metadata=attention_metadata), self._layer_shapes_at(first))
        output = output if opened is False else output[0]
        output = output.streams if isinstance(output, Carried) else output
        return jnp.result_type(streams.dtype, output.dtype)

    def first_layer_shapes(self) -> dict:
        """Return the stack's first layer's variables as shape/dtype structs (`layer_shapes`)."""
        return self.layer_shapes(0)

    def layer_shapes(self, run: int) -> dict:
        """Return the first layer of the stack's run `run` (`groups`) as shape/dtype structs.

        These are the layer's variables, a bank's first row with the layer axis
        dropped, or the layer's variables completed by the bank's
        (`banked_collections`), or a stage-local store's first stage's first
        row of the run that holds the layer (`StackView.stage_form`), on one
        stage too. Nothing is read, fetched or sliced.
        """
        return self._layer_shapes_at(self.groups[run][0], run=run)

    @nn.nowrap
    def _layer_shapes_at(self, index: int, *, run: int | None = None) -> dict:
        """Layer `index`'s stored variables as shape/dtype structs (`layer_shapes`), skipping captured
        intermediates, which are outputs a layer never reads (and may hold non-array returns, row 11882)."""
        if run is None:
            run = next(position for position, (start, count) in enumerate(self.groups)
                       if start <= index < start + count)
        start, count = self.groups[run]
        # The banks the run holding the layer keeps, its own alone: a looped pass's view hides the other
        # spans' (`PassView`), which a whole-stack classification would refuse.
        banked = self.banked_collections((start, start + count))
        bank = StackView(self.groups).bank_names()[run]
        layer = f'layers_{index}'
        reader = self.variables.get('streaming', {}).get(bank, {}).get('bank')
        if reader is not None:
            return reader.shapes()
        stacked = count > 1

        def shapes(leaf, drop: bool):
            return jax.ShapeDtypeStruct(leaf.shape[1:] if drop else leaf.shape, leaf.dtype)

        variables = {}
        for collection, tree in self.variables.items():
            if collection == _INTERMEDIATES:
                continue
            if collection in banked:
                # A banked cache holds no entry for a layer that caches nothing.
                if bank in tree:
                    variables[collection] = jax.tree.map(
                        functools.partial(shapes, drop=stacked), tree[bank])
                continue
            held = {}
            if stacked and bank in tree:
                held = jax.tree.map(functools.partial(shapes, drop=True), tree[bank])
            if layer in tree:
                held = merge(held, jax.tree.map(functools.partial(shapes, drop=False), tree[layer]))
            for name, stage in tree.get('stages', {}).items():
                layers = group_layers(name)
                # A run that holds the layer: a one-stage run its group stacks, beside the leaves its
                # stage-local state keeps on the layer itself (a frozen base joined in), or a pipeline's run
                # where no such layer runs before it.
                if layers is not None and layers.start == index and (stacked or layer not in tree):
                    # The stage axis, a run's own layer axis; a periodic run's first layer at its first
                    # position.
                    rows = 1 + (len(layers) > 1)
                    if run_period(self.scan_keys, index, len(layers)) > 1:
                        stage, rows = stage['layers_0'], 2
                    held = merge(held, jax.tree.map(
                        lambda leaf, rows=rows: jax.ShapeDtypeStruct(leaf.shape[rows:], leaf.dtype), stage))
            if held:
                variables[collection] = held
        return variables

    @nn.nowrap
    def stage_split(self, stages: int) -> StageSplit:
        """How the stack runs over `stages` pipeline stages (`split_stages`): the layers before and
        after the run its stages repeat run on every stage."""
        types = self.per_layer_types
        return split_stages(self.layer_specs(types, self.layer_kinds(types), self.layer_mixer), stages)

    @property
    def provides(self) -> bool:
        """Whether a layer leaves what a later one reads (`LayerSpec.provider`): keys and
        values a sharing layer takes, or a CSA2 layer's publications, which its Reindex
        and Reuse layers read whether or not any layer shares keys. The stack then
        threads a kv_store through its layers."""
        return any(spec.provider is not None for spec in self.specs)

    def _layer_inputs(self, tokens, x, routed_experts, routed) -> LayerInputs | None:
        """The per-layer signal and routing replay the stack slices by layer."""
        embeddings = self.per_layer_inputs(tokens, x) if self.per_layer_input_dim else None
        # A mixer computing by its layer's place reads it as data (`MixerBase.reads_depth`).
        mixers = {self.kind_of(kind).mixer or self.layer_mixer for kind in set(self.per_layer_types)}
        reads = any(mixer.reads_depth for mixer in mixers)
        depth = None if not reads else jnp.broadcast_to(jnp.arange(self.num_layers, dtype=jnp.int32),
                                                        (*tokens.shape, self.num_layers))
        if routed_experts is None:
            if routed is not None:
                raise ValueError("routed marks the tokens a routing record covers; pass routed_experts")
            return None if embeddings is None and depth is None else LayerInputs(embeddings=embeddings,
                                                                                 depth=depth)
        # One record for each layer's one routed site, or one a site, keyed by it (`LayerInputs.experts`).
        keyed = isinstance(routed_experts, Mapping)
        records = {name: jnp.asarray(record) for name, record in
                   (routed_experts.items() if keyed else (('routed_experts', routed_experts),))}
        for name, record in records.items():
            if record.ndim != 4 or record.shape[:3] != (*tokens.shape, self.num_layers):
                raise ValueError(
                    f"{f'routed_experts[{name!r}]' if keyed else name} is [batch, tokens, layers, top_k], "
                    f"{(*tokens.shape, self.num_layers)} leading for this model; got {record.shape}")
        coverage = None if routed is None else jnp.broadcast_to(
            jnp.asarray(routed, bool)[:, :, None], (*tokens.shape, self.num_layers))
        return LayerInputs(embeddings=embeddings, experts=records if keyed else records['routed_experts'],
                           routed=coverage, depth=depth)

    def per_layer_inputs(self, tokens, inputs_embeds):
        """Every layer's input signal `[B, S, L, P]` (Gemma 3n/4 PLE; `embeddings.per_layer_inputs`)."""
        return embeddings.per_layer_inputs(self, tokens, inputs_embeds)

    def output_table(self) -> OutputTable | None:
        """The head as the matrix `logits_from_hidden` contracts, None where no matrix alone is the head
        (`decoder_head.output_table`)."""
        return decoder_head.output_table(self)

    def vocabulary_rows(self) -> jax.Array:
        """Each vocabulary token's `[features]` row of the head where no exact table is
        (`decoder_head.vocabulary_rows`)."""
        return decoder_head.vocabulary_rows(self)

    def init_cache(self, batch_size: int):
        """Open the decode cache for `batch_size` sequences, every row empty.

            cache = model.apply(params, batch_size, method=CausalTransformer.init_cache,
                                mutable=['cache'])[1]['cache']

        Each holder states its leaves (`cache_spec`), which its decode opens
        and writes too, so no forward is traced; a decode over no cache opens
        it so and writes nothing to it (`dew.nn.kv_cache.Opening`).
        """
        put_opened(self, self.cache_spec(batch_size))

    @nn.nowrap
    def cache_spec(self, rows: int) -> CacheSpec:
        """The trunk's decode cache for `rows` rows as its holders state it (`decoder_cache.trunk_cache`);
        a bidirectional decoder, whose every token reads the whole sequence, keeps none."""
        if not self.causal:
            raise ValueError("a bidirectional decoder keeps no decode cache: its every token reads the whole "
                             "sequence; prefill a canvas's prompt with its causal model")
        return trunk_cache(self, rows)

    # The serving and training hooks (`dew.nn.protocols`).

    def _final_norm(self):
        """The norm before the head and between a loop's passes: under nGPT the unit norm itself,
        its states being unit vectors, with the logits' per-token gain beside it."""
        if self.ngpt is not None:
            self.norm = l2norm
            self.logit_scale = self.param('logit_scale', nn.initializers.constant(self.ngpt.start),
                                          (self.vocab_size,), jnp.float32)
        else:
            self.norm = decoder_norm(
                self.norm_type, epsilon=self.norm_eps, bias=self.norm_bias,
                scale_offset=self.scale_offset, scale_after_cast=self.scale_after_cast,
                dtype=self.dtype, precision=self.precision, specs=self.norms)(name=NormSite.FINAL)

    @nn.nowrap
    def with_kv_cache(self, layout: KVCache) -> Self:
        """The same decoder over the cache layout `layout` (`Layered`)."""
        return self.clone(kv_cache=layout)

    @nn.nowrap
    def with_cache_capacity(self, capacity: int) -> Self:
        """`max_seq_len` is the cache size its layers read (`open_kv_cache`);
        a learned position table it sized keeps its rows."""
        if self.position_embedding == 'learned' and self.position_embedding_size is None:
            return self.clone(max_seq_len=capacity, position_embedding_size=self.max_seq_len)
        return self.clone(max_seq_len=capacity)

    @nn.nowrap
    def with_passes(self, steps: int | None = None, layers: tuple[int, int] | None = None, *,
                    backprop_steps: int | None = None, remat: bool | None = None,
                    implicit: bool | None = None, exit_gate: bool | None = None,
                    exit_threshold: float | None = None, exits: bool | None = None,
                    theta_passes: int | None = None, messages: CrossPass | None = None) -> Self | None:
        """This model with its loop run `steps` times (None keeps its count), over the block `layers`
        where given (None keeps its block). `backprop_steps` sets how many final passes train through,
        `remat` checkpoints each pass, `implicit` takes each later pass by the θ-method, and `exit_gate`,
        `exit_threshold` and `exits` weigh, take and give each pass's exit, `theta_passes` gives an
        implicit loop a θ row for each later pass of that many, and `messages` couples its passes through
        their block summaries (`Loop`); None keeps its own.

        A model with no loop gains one over `layers`, with no norm between its
        passes, so one pass computes the model as it was, and every count does
        where it is implicit or passes messages; None when it has no loop and the
        two do not both say what to add. Messages join only a loop of one pass
        (refused otherwise), whose later passes they then define."""
        stated = {'backprop_steps': backprop_steps, 'remat': remat, 'implicit': implicit,
                  'exit_gate': exit_gate, 'exit_threshold': exit_threshold, 'exits': exits,
                  'theta_passes': theta_passes, 'messages': messages}
        options = {name: value for name, value in stated.items() if value is not None}
        if self.loop is None:
            if steps is None or layers is None:
                return None
            return self.clone(loop=Loop(steps, step_norm=False, layers=layers, **options))
        if messages is not None and self.loop.messages is None:
            if self.loop.steps > 1:
                # Its later passes read the pass before's output; a loop passing messages reads the
                # block's input, which would change what it computes at the merges' draw.
                raise ValueError(f"this decoder already loops its block {self.loop.steps} times over the "
                                 "pass before's output, and a loop passing messages enters every pass at the "
                                 "block's input: add messages to a decoder that runs its block once")
            # A loop of one pass never ran its step norm; its passes now enter at the block's input.
            options['step_norm'] = False
        return self.clone(loop=dataclasses.replace(self.loop, steps=steps or self.loop.steps,
                                                   layers=layers or self.loop.layers, **options))

    @nn.nowrap
    def with_loop(self, loop: Loop) -> Self:
        """This model running `loop` in place of its own, whole (`Looping.with_loop`): its block, passes,
        recompute, exits, θ and messages as `loop` states them."""
        return self.clone(loop=loop)

    @nn.nowrap
    def projected(self, params, *, updated: bool):
        """`ProjectsParameters`: before a training step's update, every AltUp layer's prediction
        coefficients clamped to `altup.coef_clip`, as the reference clamps its stored weight before
        a training forward (`clamped_predictions`); the updated parameters as they are."""
        if updated or self.altup is None or self.altup.coef_clip is None:
            return params
        return clamped_predictions(params, self.altup.coef_clip)

    @property
    def context_width(self) -> int:
        """The width of a context's tokens (`CrossAttending`): the spec's, or the decoder's own."""
        cross = self.cross_attention
        if cross is None or cross.context_features is None:
            return self.emb_features
        return cross.context_features

    @property
    def context_dtype(self) -> jnp.dtype:
        """The dtype a context's tokens enter in (`CrossAttending`): the decoder's compute dtype, at
        least float32, as its projections widen what they read."""
        return at_least_fp32(self.dtype)

    @nn.nowrap
    def with_cross_attention(self, cross: CrossBlocks | None) -> Self:
        """This decoder with `cross` as its cross-attention layers (`CrossAttending`), None for none."""
        return self.clone(cross_attention=cross)

    @nn.nowrap
    def with_dex(self, extension: DifferentialExtension | None) -> Self:
        """This decoder with `extension`'s differential on its chosen heads (`dew.nn.dex`), None for none."""
        return self.clone(dex=extension)

    @nn.nowrap
    def reads_progress(self) -> bool:
        """Whether its computation changes over a run (`dew.nn.progress.ReadsProgress`): its DEX layers'
        lambda anneals over the run's updates."""
        return self.dex is not None

    @nn.nowrap
    def dex_exported(self, variables: Mapping[str, Mapping]) -> dict[str, Mapping]:
        """`variables` as an export writes them, each DEX layer's differential folded into its output
        projection at the operands its layers multiply in (`dew.nn.dex.folded`); `ValueError` names a
        layer whose heads' outputs pass through more before it, which does not fold."""
        for index, _ in () if self.dex is None else self.dex.layers:
            mixer = self.mixer_of(index)
            refusal = mixer.dex_fold_refusal() if isinstance(mixer, HeadOutputs) else None
            if refusal is not None:
                raise ValueError(f"the source's format holds no differential extension (DEX), and layer "
                                 f"{index}'s does not fold into its output projection: {refusal}; a run's "
                                 f"checkpoint keeps this model")
        return folded(variables, self.dtype)

    @nn.nowrap
    def dex_sites(self) -> tuple[tuple[str, ...], frozenset[int]]:
        """Its parameters at the root of `params`, and its softmax attention layers (`Differential`)."""
        return (), softmax_attention_layers(self)

    def with_remat(self, policy: RematPolicy | None) -> Self:
        """Start at `policy`, including policies outside `DECODER_REMAT`.

        `recompute_more` stays at a policy off that ladder, so the trainer
        cannot climb from it; None keeps every layer's residuals.
        """
        return self.clone(remat=policy)

    @nn.nowrap
    def recompute_record(self) -> JSON:
        return remat_record(self.remat)

    @nn.nowrap
    def recompute_more(self) -> Self | None:
        """One rung up `DECODER_REMAT` that keeps less of what its layers mark, None where none does
        (`decoder_recompute.more`)."""
        return decoder_recompute.more(self)

    @nn.nowrap
    def recompute_offloaded(self) -> Self | None:
        """At the ladder's top, the decoder that keeps what 'minimal' saves in pinned host memory, else None
        (`decoder_recompute.offloaded`)."""
        return decoder_recompute.offloaded(self)

    @nn.nowrap
    def recompute_statement(self) -> RecomputeStatement:
        """What this decoder's rung recomputes (`decoder_recompute.statement`)."""
        return decoder_recompute.statement(self)

    @nn.nowrap
    def marked_residuals(self) -> frozenset[str]:
        """The `RESIDUALS` this decoder's layers can mark (`decoder_recompute.marked_residuals`)."""
        return decoder_recompute.marked_residuals(self)

    @nn.nowrap
    def restore_recompute(self, record: JSON) -> Self:
        """At the recorded rung where it lies above its own (`decoder_recompute.restored`)."""
        return decoder_recompute.restored(self, record)

    @nn.nowrap
    def inference_projection_groups(self, variables: Mapping[str, Mapping]) -> tuple[ProjectionGroup, ...]:
        """Its layers', depths' and drafter's groups, bound without tracing a
        forward, causal or not: a bidirectional model's masked denoising reads
        its projections packed as a decoder's steps do. The groups read which
        parameters the layers hold, never their values, so the layers are
        bound over the variables' shapes: flax's bound tree is cyclic, and
        bound over the weights themselves it would keep them alive after the
        caller drops them until a full collection."""
        bound = self.bind(jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct(np.shape(leaf), jnp.result_type(leaf)), variables))
        return declared_groups(*bound.layers, *bound.mtp,
                               *(bound.dspark_stages if self.dspark is not None else ()))

    @property
    def cache_rebuild_position(self) -> int | None:
        """Where LongRoPE's factors switch from short to long, past which cached keys go stale
        (`decoder_cache.rebuild_position`)."""
        return decoder_cache.rebuild_position(self)

    @nn.nowrap
    def block_chunks(self) -> tuple[int | None, ...]:
        """The attention chunk of each trunk layer that reads a block whole
        (`MixerBase.reads_blocks`), None where a layer has none (`BlockReading`). Each layer's
        mixer is `layer_specs`' own, the kind's or the model's layer mixer, so a prediction depth's or the
        drafter's kind, which the trunk never runs, reads none here."""
        types = self.per_layer_types
        kinds = self.layer_kinds(types)
        default = self.layer_mixer
        built = [(spec.kind.mixer or default, spec.kind.chunk)
                 for spec in self.layer_specs(types, kinds, default)]
        return tuple(chunk for mixer, chunk in built if mixer.reads_blocks)

    @nn.nowrap
    def layer_mixers(self) -> tuple[tuple[MixerBase, MixerContext], ...]:
        """Each layer's mixer and the context it is built with, in order (`decoder_refusals`)."""
        types, default = self.per_layer_types, self.layer_mixer
        return tuple((spec.kind.mixer or default, self.mixer_context(
            spec.kind, spec.layer_type, spec.kv_shared, spec.dex, self._dex_start(index), spec.routed))
            for index, spec in enumerate(self.layer_specs(types, self.layer_kinds(types), default)))

    @nn.nowrap
    def mixed_admission_refusal(self) -> str | None:
        """Why a serving step's mixed call cannot admit this decoder (`decoder_refusals.mixed_admission`)."""
        return decoder_refusals.mixed_admission(self)

    @nn.nowrap
    def generation_refusal(self) -> str | None:
        """Why a generation loop cannot continue this decoder's rows (`decoder_refusals.generation`)."""
        return decoder_refusals.generation(self)

    @nn.nowrap
    def fork_refusal(self, train: bool = False) -> str | None:
        """Why a call over forks would not compute each document as its own row does, in training where
        `train` (`decoder_refusals.forked`)."""
        return decoder_refusals.forked(self, train)

    @nn.nowrap
    def prefix_sharing_refusal(self, train: bool = False) -> str | None:
        """Why rows that begin alike should not be packed over their shared prefixes, in training where
        `train` (`decoder_refusals.prefix_sharing`)."""
        return decoder_refusals.prefix_sharing(self, train)

    @nn.nowrap
    def split_prefill_refusal(self) -> str | None:
        """Why a prompt's later piece would not read what its whole prefill reads
        (`decoder_refusals.split_prefill`)."""
        return decoder_refusals.split_prefill(self)

    @property
    def layer_mixer(self) -> MixerBase:
        """The mixer a layer whose kind names none builds: the model's own, else
        attention rotated or not by its position embedding."""
        return self.mixer if self.mixer is not None else AttentionMixer(
            nope=self.position_embedding != 'rotary', alibi=self.position_embedding == 'alibi')

    @property
    def declared_mixers(self) -> tuple[MixerBase, ...]:
        """The mixer values it names: its own, then each layer kind's."""
        return tuple(mixer for mixer in (self.mixer, *(kind.mixer for kind in (self.kinds or {}).values()))
                     if mixer is not None)

    @property
    def indexed_mixers(self) -> tuple[IndexedMixer, ...]:
        return tuple(indexed for mixer in self.declared_mixers for indexed in mixer.indexed_mixers)

    @property
    def keeps_triton_gemm(self) -> bool:
        return any(mixer.keeps_triton_gemm for mixer in self.declared_mixers)


__all__ = ["CausalTransformer"]
