"""The autoregressive transformer decoder every language model here trains.

Token embedding, rotary positions, pre-norm blocks of grouped-query causal
attention and a gated MLP, a final RMSNorm, and an fp32 head. Attention goes
through the one shared kernel path in dew.nn.attention, so a run picks
reference/xla/cudnn/tpu the same way a diffusion run does, and decoding reuses
the same fixed-size KV cache helpers.

Parameter names mirror the HF decoder layout - embed_tokens,
layers_N.{input_layernorm, self_attn.{q,k,v,o}_proj, post_attention_layernorm,
mlp.{gate,up,down}_proj}, norm, lm_head. A model family is supported only
after its translator and same-weight reference parity test land. Gemma's two
extra norms are the exception: HF calls them post_attention_layernorm and
post_feedforward_layernorm even though they normalize sublayer outputs, so
here they are attention_output_norm and mlp_output_norm and the pre-norms keep
their names. dew.interop.hf_decoders does that rename.

The block holds its token mixer in a slot: any module with the
(x, decode=..., positions=..., segment_ids=...) -> x signature of
CausalSelfAttention becomes self_attn without the block changing, which is
where a linear-attention mixer goes.
"""

import dataclasses
import functools
import math
from typing import Callable, Mapping, Optional, Tuple, Union

import flax.core
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from ..attention import (
    RMSNorm, RopeScaling, apply_rotary, causal_attention_mask, open_kv_cache,
    rotary_freqs, scaled_dot_product_attention,
)
from ..mixers import AttentionMixer, MixerBase, MixerContext, mixer_from_record
from ..moe import SparseMLP
from ..gemma3n import AltUp, AltUpLayer, LaurelBlock, gaussian_topk, rescale_to
from ..gemma4_moe import Gemma4Experts
from ..gpt_oss import GptOssMLP
from ..mla import YarnScaling, mla_rope_freqs
from ..sharding import logical_axes
from dew.registry import models


@dataclasses.dataclass(frozen=True)
class LayerKind:
    """What the layers of one kind in the pattern do differently.

    The pattern already names each layer's kind, so what the kind means
    belongs here rather than in a field name: a windowed kind is what
    "sliding attention" used to say, and `rope_theta` and `head_dim` are the
    model's unless this kind states its own. Rotary positions rotate every
    dimension of a windowed kind, which is where Gemma 4 puts its partial
    rotary (the global layers') and where the sliding layers rotate whole.

    `mixer` is this kind's token mixer, a value from the `mixers` registry;
    None rides the model's mixer. A hybrid stack names its per-layer mixers
    here, keyed by the names already in the pattern, instead of growing a
    second switch on layer names.
    """

    window: Optional[int] = None
    """Keys a layer of this kind attends, its own included; None attends all."""
    num_kv_heads: Optional[int] = None
    """This kind's key/value head count; None takes the model's. Gemma 4's
    global layers keep fewer than its sliding ones (num_global_key_value_heads)."""
    rope_theta: Optional[float] = None  # set: this kind takes this base over the model's
    rope_scaling: Optional[RopeScaling] = None
    """This kind's llama3 ramp or its record; None rides the model's."""
    head_dim: Optional[int] = None
    mixer: Optional[MixerBase] = None
    """This kind's mixer value or its record; None is the model's mixer."""

    def __post_init__(self):
        # A kind's mixer and ramp arrive as values from code and as records
        # from a config, like the model's own; anything else is neither.
        if isinstance(self.mixer, Mapping):
            object.__setattr__(self, "mixer", mixer_from_record(self.mixer))
        elif self.mixer is not None and not isinstance(self.mixer, MixerBase):
            raise ValueError(
                f"a kind's mixer is a mixer value, its record, or None, "
                f"not {self.mixer!r}")
        if isinstance(self.rope_scaling, Mapping):
            object.__setattr__(self, "rope_scaling", RopeScaling(**self.rope_scaling))


@dataclasses.dataclass(frozen=True)
class ResolvedKind:
    """One kind of layer with the model's defaults filled in.

    `LayerKind` is what a config states, so a field it leaves to the model is
    None there. This is what the model resolved it to, so `rope_theta` and
    `head_dim` are numbers; only the window stays optional, because attending
    the whole sequence is what a kind without one does. `mixer` passes
    through: it needs no resolution, only the model's default when unset.
    """

    window: Optional[int]
    num_kv_heads: int
    rope_theta: float
    rope_scaling: Optional[RopeScaling]
    head_dim: int
    mixer: Optional[MixerBase]

@dataclasses.dataclass(frozen=True)
class Mixture:
    """The experts some layers route to, and how the router chooses.

    `experts` is what the rest depends on, which is why they live together:
    a top_k, a cadence or a balancing bias says nothing about a model with
    no experts. `layers` names the sparse layers by index, or `every` makes
    every nth layer sparse counting from the end of the first group, which
    is what Qwen3-MoE's decoder_sparse_step means; neither makes every layer
    sparse, which is Mixtral.

    The routing options are `Router`'s: `score_function` softmax, sigmoid or
    sqrtsoftplus, `norm_topk_prob` (the reference's name) for dividing a
    token's selected weights by their sum, `scaling` on the routed output,
    `groups` with `groups_per_token` for DeepSeek's node limit, `group_score`
    for how a group is scored ('top2' is V3's, 'max' is V2's), `bias`
    for V3's aux-loss-free balancing bias, and `scale_inputs` for Llama 4's
    weight on the expert input rather than its output.

    `parallel` is Gemma 4's placement (`enable_moe_block`): the experts run
    beside the dense feed-forward on the same residual and the two are
    summed after a norm each, under `Gemma4TextRouter`, which softmaxes,
    keeps the renormalised top k and scales each choice per expert; the
    routing dials above belong to the replacing routers and are refused
    with it.

    `expert_features` is the routed experts' width, None for the model's
    `mlp_features`; DeepSeek sizes its experts apart from its dense layers
    (`moe_intermediate_size` beside `intermediate_size`). `shared_features`
    is the width of the one dense gated MLP every token takes beside the
    routed experts, 0 for none: `DeepseekV3MoE` builds its `n_shared_experts`
    as a single MLP of `n_shared_experts * moe_intermediate_size`, so the
    product is the whole record of them.
    """

    experts: int
    top_k: int = 2
    layers: Optional[Tuple[int, ...]] = None
    every: Optional[int] = None
    score_function: str = 'softmax'
    norm_topk_prob: bool = True
    scaling: float = 1.0
    groups: int = 1
    groups_per_token: int = 1
    group_score: str = 'top2'
    bias: bool = False
    scale_inputs: bool = False
    parallel: bool = False
    expert_features: Optional[int] = None
    shared_features: int = 0

    def __post_init__(self):
        if self.layers is not None:
            object.__setattr__(self, "layers", tuple(self.layers))
        if self.experts < 1:
            raise ValueError(
                f"a mixture needs experts to route to, got {self.experts}; a "
                "dense model has no mixture at all")
        if self.layers is not None and self.every is not None:
            raise ValueError(
                f"layers ({self.layers}) and every ({self.every}) both choose the "
                "sparse layers, so only one of them can be set")
        if self.every is not None and self.every < 1:
            raise ValueError(f"every must be positive, got {self.every}")
        if self.expert_features is not None and self.expert_features < 1:
            raise ValueError(
                f"expert_features is the routed experts' width, got "
                f"{self.expert_features}; None takes the model's mlp_features")
        if self.shared_features < 0:
            raise ValueError(
                f"shared_features is the shared branch's width, got "
                f"{self.shared_features}; 0 is a layer without one")
        if self.parallel and (
                self.score_function != 'softmax' or not self.norm_topk_prob
                or self.scaling != 1.0 or self.groups != 1 or self.bias
                or self.scale_inputs or self.shared_features):
            raise ValueError(
                "a parallel mixture routes with Gemma 4's router, which has no "
                "score function, scaling, groups, balancing bias, input scaling "
                "or shared branch to set")


@logical_axes({
    ("q_proj",): ("embed", "heads"),
    ("k_proj",): ("embed", "kv"),
    ("v_proj",): ("embed", "kv"),
    ("o_proj",): ("attention", "embed"),
})
class CausalSelfAttention(nn.Module):
    """Causal self-attention with grouped-query heads, rotary positions, qk
    RMSNorm and a fixed-size KV cache.

    decode=True runs the call against the cache: the first call writes the
    whole prompt and each later call appends one token, so prefill and decode
    are one code path. Keys are rotated before they enter the cache, which is
    why the rotary positions come from the cache index rather than from the row
    index of the token.

    causal=False is full attention over the sequence, which a masked
    diffusion model reads the whole corrupted sequence with; there is no
    cache to decode against then, so decode=True raises.

    kv_shared marks a layer that owns no K/V projections (Gemma 3n/4 style
    cross-layer KV sharing): it reads the keys, values and their positions
    that the designated earlier layer of the same layer type stashed in
    `kv_store`, post rope and post norm, and keeps no cache of its own.
    """
    emb_features: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    causal: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[RopeScaling] = None  # Llama 3.1's ramp over the base frequencies
    qk_norm: bool = True
    qk_norm_scope: str = 'head'  # 'head': one RMSNorm per head; 'projection': over the whole q/k
    v_norm: bool = False
    norm_eps: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    kv_shared: bool = False
    kv_store_key: Optional[str] = None
    sliding_window: Optional[int] = None
    attention_bias: bool = False  # q/k/v biases, as config.attention_bias in HF
    o_proj_bias: Optional[bool] = None  # None follows attention_bias; Qwen2 biases q/k/v only
    attention_scale: Optional[float] = None  # None: the kernel's own 1/sqrt(head_dim)
    attention_sinks: bool = False
    yarn: Optional[YarnScaling] = None
    attn_logit_softcap: Optional[float] = None  # Gemma 2's tanh on the logits, attn_logit_softcapping
    k_eq_v: bool = False  # Gemma 4's global layers project no values: the raw keys, values-normed
    output_gate: bool = False  # Qwen3.5 doubles q_proj and gates the branch with a sigmoid
    partial_rotary_factor: Optional[float] = None  # None: every head dim rotates
    partial_rotary_type: str = 'proportional'  # 'proportional' (Gemma 4) | 'default' (Qwen3.5)
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    attention_impl: Optional[str] = None
    force_fp32_for_softmax: bool = True

    def setup(self):
        dense = functools.partial(
            nn.Dense, use_bias=self.attention_bias, dtype=self.dtype, precision=self.precision)
        # The gate doubles the query projection: the reference chunks its
        # output in half, one half the query and the other the gate the
        # branch multiplies by (modeling_qwen3_5.py:670-673, 701).
        self.q_proj = dense(
            self.num_heads * self.head_dim * (2 if self.output_gate else 1), name='q_proj')
        # A sharing layer reads another layer's keys and values, so it owns
        # no projections or key norm of its own, exactly as the reference
        # skips them (modeling_gemma4.py, Gemma4TextAttention.__init__).
        if not self.kv_shared:
            self.k_proj = dense(self.num_kv_heads * self.head_dim, name='k_proj')
            if not self.k_eq_v:
                self.v_proj = dense(self.num_kv_heads * self.head_dim, name='v_proj')
        self.o_proj = dense(self.emb_features, name='o_proj', use_bias=(
            self.attention_bias if self.o_proj_bias is None else self.o_proj_bias))
        if self.qk_norm:
            if self.qk_norm_scope not in ('head', 'projection'):
                raise ValueError(
                    f"qk_norm_scope is 'head' or 'projection', got {self.qk_norm_scope!r}")
            norm = functools.partial(
                RMSNorm, epsilon=self.norm_eps, scale_offset=self.scale_offset,
                scale_after_cast=self.scale_after_cast, dtype=self.dtype)
            self.q_norm = norm(name='q_norm')
            if not self.kv_shared:
                self.k_norm = norm(name='k_norm')
        if self.v_norm and not self.kv_shared:
            # Gemma 4 norms the values with a scale-free RMSNorm before they
            # are cached or shared (modeling_gemma4.py, Gemma4TextAttention).
            # The attribute cannot share the field's name, and it holds no
            # parameters either way.
            self.values_norm = RMSNorm(epsilon=self.norm_eps, with_scale=False,
                                       dtype=self.dtype, name='v_norm')

    def _rot_dim(self) -> int | None:
        """Head dims the rotary rotates, or None for all of them.

        `partial_rotary_type` says what the fraction means: 'proportional'
        rotates the first rot_dim dims of a head_dim-wide rope and passes the
        rest at frequency zero (Gemma 4), 'default' builds a rot_dim-wide rope
        and leaves the rest unrotated (Qwen3.5); `rotary_freqs` names the
        reference lines. Both rotate `int(head_dim * factor)` dims.
        """
        factor = self.partial_rotary_factor
        if factor is None:
            return None
        rot_dim = int(self.head_dim * factor)
        if not 0 < factor <= 1 or rot_dim % 2:
            raise ValueError(
                f"partial_rotary_factor must rotate an even positive number of "
                f"head dims, got {factor} of head_dim {self.head_dim}")
        return rot_dim

    @nn.compact
    def __call__(self, x, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None):
        B, S, _ = x.shape
        projected = self.q_proj(x)
        # OLMo 3 norms the whole projection, one scale of heads * head_dim,
        # before the head split (modeling_olmo3.py:162-163, :178-179); Qwen3
        # and the Gemmas norm each head after it, which the reference marks
        # "unlike olmo, only on the head dim" (modeling_qwen3_moe.py:147).
        whole = self.qk_norm and self.qk_norm_scope == 'projection'
        if whole:
            projected = self.q_norm(projected)
        gate = None
        if self.output_gate:
            # The reference views the doubled output as [.., heads, 2*head_dim]
            # and chunks it: query, then gate (modeling_qwen3_5.py:670-673).
            query, gate = jnp.split(
                projected.reshape(B, S, self.num_heads, 2 * self.head_dim),
                2, axis=-1)
        else:
            query = projected.reshape(B, S, self.num_heads, self.head_dim)
        if self.kv_shared:
            # The provider ran earlier in the same forward pass and stashed
            # its post-norm, post-rope keys and values with their positions,
            # so there is nothing to project, norm, rotate or cache here.
            if kv_store is None or self.kv_store_key not in kv_store:
                raise ValueError(
                    f"layer shares K/V under {self.kv_store_key!r} but no provider "
                    "stashed them; the model has to pass one kv_store dict down "
                    "its layer stack")
            key, value, positions = kv_store[self.kv_store_key]
        else:
            key = self.k_proj(x)
            # attention_k_eq_v reads the values off the key projection before
            # its norm (modeling_gemma4.py, Gemma4TextAttention.forward).
            value = (key if self.k_eq_v else self.v_proj(x)).reshape(
                B, S, self.num_kv_heads, self.head_dim)
            if whole:
                key = self.k_norm(key)
            key = key.reshape(B, S, self.num_kv_heads, self.head_dim)
            if self.qk_norm and not whole:
                key = self.k_norm(key)
            if self.v_norm:
                value = self.values_norm(value)
        if self.qk_norm and not whole:
            query = self.q_norm(query)

        # The cache slot carries position while decoding, so the rotation and
        # the mask both read it instead of the row index of the token. A
        # packed batch supplies the position inside its document instead of
        # the row index, which is what restarts RoPE at every boundary.
        append = None
        kv_len = key.shape[-3]
        if decode:
            if not self.causal:
                raise ValueError("full attention has no KV cache to decode against")
            if not self.kv_shared:
                positions, append = open_kv_cache(self, key, self.max_seq_len)
        elif positions is None and not self.kv_shared:
            positions = jnp.arange(S)
        elif not self.kv_shared:
            positions = jnp.asarray(positions)
        if self.yarn is None:
            freqs_cos, freqs_sin = rotary_freqs(
                positions, self.head_dim, self.rope_theta, rot_dim=self._rot_dim(),
                partial_rotary_type=self.partial_rotary_type, rope_scaling=self.rope_scaling)
        else:
            if self.partial_rotary_factor is not None or self.rope_scaling is not None:
                raise ValueError(
                    "yarn rotates whole heads at its own frequencies, so it takes "
                    "neither partial_rotary_factor nor rope_scaling")
            freqs_cos, freqs_sin = mla_rope_freqs(
                positions, self.head_dim, self.rope_theta, self.yarn)
        # Every kernel path scales the logits by 1/sqrt(head_dim) itself, so the
        # query carries the ratio to the scale the checkpoint asks for.
        query = apply_rotary(
            query, freqs_cos, freqs_sin,
            scale=(None if self.attention_scale is None
                   else self.attention_scale * math.sqrt(self.head_dim)))
        if not self.kv_shared:
            key = apply_rotary(key, freqs_cos, freqs_sin)
            if kv_store is not None and self.kv_store_key is not None:
                # Post-norm, post-rope, the same tensors the reference hands
                # its sharing layers (modeling_gemma4.py, Gemma4TextAttention).
                kv_store[self.kv_store_key] = (key, value, positions)
        causal, mask = self.causal, None
        implementation = self.attention_impl
        window = None if decode else self.sliding_window
        if self.kv_shared and decode:
            # No cache of its own: the provider's stashed keys carry the full
            # history, so the mask reads them the way the provider's own
            # decode mask does.
            mask = causal_attention_mask(positions, kv_len, self.sliding_window)
            causal = False
        elif append is not None:
            key, value = append(key, value)
            mask = causal_attention_mask(positions, key.shape[-3], self.sliding_window)
            causal = False
            if kv_store is not None and self.kv_store_key is not None:
                kv_store[self.kv_store_key] = (key, value, positions)
        elif segment_ids is not None:
            # Attention stays inside each packed document: the segment ids
            # make the mask block-diagonal, padding (segment 0) sees nothing,
            # and causality (with the layer's window) travels in the same mask
            # rather than as the kernels' flag.
            segment_ids = jnp.asarray(segment_ids)
            inside = ((segment_ids[:, :, None] == segment_ids[:, None, :])
                      & (segment_ids[:, :, None] != 0))[:, None]
            mask = inside
            if causal:
                mask = jnp.logical_and(
                    inside, causal_attention_mask(jnp.arange(S), S, self.sliding_window))
            causal, window = False, None
            if implementation in ('auto', 'cudnn'):
                # cuDNN has no mask argument: causality and the window are
                # flags, and jax hands the kernel a bool mask as an additive
                # bias of -2**41 in the compute dtype instead
                # (combine_bias_and_mask in
                # jax/_src/cudnn/fused_attention_stablehlo.py), which also
                # makes check_is_flash_attention refuse an odd length while
                # training. The xla kernel masks by exclusion, on every
                # backend and with the same fp32 softmax. It costs 83.6 ms
                # and 5.80 GiB a step where the fixed window on cuDNN costs
                # 75.8 ms and 4.99 GiB, measured in
                # docs/concepts/language_models.md.
                implementation = 'xla'

        attention = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            implementation=implementation, causal=causal,
            sliding_window=window, mask=mask,
            sinks=(self.param('sinks', nn.initializers.zeros, (self.num_heads,))
                   if self.attention_sinks else None),
            softcap=self.attn_logit_softcap)
        if gate is not None:
            # The branch multiplies by the sigmoid of its gate, then projects
            # (modeling_qwen3_5.py:701, and modeling_qwen4_exp.py:836 the same).
            attention = attention * jax.nn.sigmoid(gate).astype(attention.dtype)
        return self.o_proj(attention.reshape(B, S, self.num_heads * self.head_dim))


def _gated_activation(activation: str, gate):
    """The gate's nonlinearity by the mlp's name: silu, tanh-approximate gelu
    or the erf gelu (torch's default, ACT2FN['gelu'])."""
    if activation == 'swiglu':
        return nn.silu(gate)
    return nn.gelu(gate, approximate=activation == 'geglu')

@logical_axes({
    ("gate_proj",): ("embed", "mlp"),
    ("up_proj",): ("embed", "mlp"),
    ("down_proj",): ("mlp", "embed"),
})
class GatedMLP(nn.Module):
    """down_proj(act(gate_proj(x)) * up_proj(x)): swiglu is silu, geglu is
    the tanh approximation of gelu (HF's gelu_pytorch_tanh) and geglu_exact
    the erf form (HF's gelu, which Gemma's released config names).

    Bias-free, like the gated MLP of every open decoder this loads.

    activation_sparsity is Gemma 3n's gaussian top-k on the gate before its
    nonlinearity (`dew.nn.gemma3n.gaussian_topk`); 0 leaves the gate alone.
    """
    hidden_features: int
    out_features: int
    activation: str = 'swiglu'
    activation_sparsity: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        if self.activation not in ('swiglu', 'geglu', 'geglu_exact'):
            raise ValueError(
                f"mlp must be 'swiglu', 'geglu' or 'geglu_exact', got {self.activation!r}")
        if not 0 <= self.activation_sparsity < 1:
            raise ValueError(
                f"activation_sparsity is the fraction of gate activations dropped, "
                f"within [0, 1), got {self.activation_sparsity}")
        dense = functools.partial(
            nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
        self.gate_proj = dense(self.hidden_features, name='gate_proj')
        self.up_proj = dense(self.hidden_features, name='up_proj')
        self.down_proj = dense(self.out_features, name='down_proj')

    def __call__(self, x):
        gate = self.gate_proj(x)
        if self.activation_sparsity:
            gate = gaussian_topk(gate, self.activation_sparsity)
        gate = _gated_activation(self.activation, gate)
        return self.down_proj(gate * self.up_proj(x))


@logical_axes({
    ("per_layer_input_gate",): ("embed", "mlp"),
    ("per_layer_projection",): ("mlp", "embed"),
})
class DecoderBlock(nn.Module):
    """Pre-norm decoder block: token mixer, then feed-forward, both residual.

    `mixer` and `feedforward` are factories taking only a name. What `mixer`
    builds lands in the tree as self_attn and has to accept (x, decode=...,
    positions=..., segment_ids=...), the last two None outside a packed batch.
    What `feedforward` builds lands there as mlp and takes the normalized
    states alone, which is the one call `GatedMLP` and `moe.SparseMLP` share.

    sandwich_norms adds Gemma's second pair of norms, on the output of each
    sublayer rather than on its input; the pre-norms keep their names and their
    places, so a checkpoint without them loads into the same tree minus two
    leaves per layer. pre_norms=False drops the input pair, which with the
    output pair on is OLMo 3's post-norm block (modeling_olmo3.py:249-266):
    each sublayer reads the residual stream as it is and its output is
    normed before it is added.

    kv_store threads one dict down the layer stack so a KV-sharing mixer
    reads its provider's keys and values; a mixer without a kv_store keyword
    fails loudly when a run shares. per_layer_input is the layer's input
    signal for the per-layer residual, None when the model has none.

    altup makes the block take and return Gemma 3n's stack of residual
    copies, `[num_inputs, B, S, D]`: it predicts the copies, runs on the
    active prediction, corrects every copy by what it computed, and adds the
    per-layer residual to the copies past the first rather than to its own
    output (modeling_gemma3n.py, Gemma3nTextDecoderLayer.forward). laurel_rank
    adds the LAuReL block over the attention's normed input, averaged with
    the attention residual over sqrt(2).
    """
    mixer: Callable[..., nn.Module]
    feedforward: Callable[..., nn.Module]
    emb_features: int
    norm_eps: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    sandwich_norms: bool = False
    pre_norms: bool = True
    per_layer_input_dim: int = 0
    parallel: Optional[Callable[..., nn.Module]] = None
    """A branch summed with the feed-forward's output before its output norm,
    called with the residual and that output (Gemma 4's routed experts)."""
    layer_scalar: bool = False  # Gemma 4 multiplies each layer's output by a learned scalar
    altup: Optional[AltUp] = None  # Gemma 3n's stack of residual copies
    laurel_rank: Optional[int] = None  # Gemma 3n's learned augmented residual
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(
            RMSNorm, epsilon=self.norm_eps, scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast, dtype=self.dtype)
        if self.pre_norms:
            self.input_layernorm = norm(name='input_layernorm')
        self.self_attn = self.mixer(name='self_attn')
        if self.pre_norms:
            self.post_attention_layernorm = norm(name='post_attention_layernorm')
        if self.sandwich_norms:
            self.attention_output_norm = norm(name='attention_output_norm')
            self.mlp_output_norm = norm(name='mlp_output_norm')
        self.mlp = self.feedforward(name='mlp')
        if self.parallel is not None:
            self.moe = self.parallel(name='moe')
        if self.layer_scalar:
            # The reference's layer_scalar buffer, a checkpoint leaf of one
            # value, which the released Gemma 4 checkpoints carry.
            self.output_scalar = self.param('layer_scalar', nn.initializers.ones, (1,), jnp.float32)
        if self.per_layer_input_dim:
            dense = functools.partial(
                nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
            self.per_layer_input_gate = dense(self.per_layer_input_dim,
                                              name='per_layer_input_gate')
            self.per_layer_projection = dense(self.emb_features,
                                              name='per_layer_projection')
            self.post_per_layer_input_norm = norm(name='post_per_layer_input_norm')
        if self.altup is not None:
            if not self.pre_norms or self.parallel is not None or self.layer_scalar:
                raise ValueError(
                    "altup runs Gemma 3n's block, which has its pre-norms and "
                    "neither a parallel branch nor a layer scalar")
            self.altup_layer = AltUpLayer(
                spec=self.altup, emb_features=self.emb_features, norm_eps=self.norm_eps,
                dtype=self.dtype, precision=self.precision, name='altup')
        if self.laurel_rank is not None:
            if self.laurel_rank < 1 or not self.pre_norms:
                raise ValueError(
                    f"laurel_rank is the width of the learned augmented residual "
                    f"over the attention's normed input, got {self.laurel_rank} "
                    f"with pre_norms={self.pre_norms}")
            self.laurel = LaurelBlock(
                rank=self.laurel_rank, emb_features=self.emb_features,
                norm_eps=self.norm_eps, dtype=self.dtype, precision=self.precision,
                name='laurel')
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, train: bool = False, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None,
                 per_layer_input=None):
        altup = self.altup
        predictions = None if altup is None else self.altup_layer.predict(x, train=train)
        if altup is not None and predictions is not None:
            x = predictions[altup.active_idx]
        normed = self.input_layernorm(x) if self.pre_norms else x
        mixed = self.self_attn(normed,
                               decode=decode, positions=positions, segment_ids=segment_ids,
                               **({} if kv_store is None else {"kv_store": kv_store}))
        if self.sandwich_norms:
            mixed = self.attention_output_norm(mixed)
        x = x + self.dropout(mixed, deterministic=not train)
        if self.laurel_rank is not None:
            x = (x + self.laurel(normed)) * jnp.asarray(1 / math.sqrt(2), x.dtype)
        hidden = self.mlp(self.post_attention_layernorm(x) if self.pre_norms else x)
        if self.parallel is not None:
            hidden = self.moe(x, hidden)
        if self.sandwich_norms:
            hidden = self.mlp_output_norm(hidden)
        x = x + self.dropout(hidden, deterministic=not train)
        if altup is not None and predictions is not None:
            corrected = self.altup_layer.correct(predictions, x, train=train)
            if self.per_layer_input_dim and per_layer_input is not None:
                first = corrected[altup.active_idx]
                if altup.correct_scale:
                    first = self.altup_layer.scale_corrected_output(first)
                # The per-layer residual lands on the copies past the first,
                # the active one left as corrected.
                corrected = corrected.at[1:].add(self._per_layer_residual(first, per_layer_input))
            return corrected
        if self.per_layer_input_dim and per_layer_input is not None:
            x = x + self._per_layer_residual(x, per_layer_input)
        if self.layer_scalar:
            x = x * self.output_scalar.astype(x.dtype)
        return x

    def _per_layer_residual(self, x, per_layer_input):
        """Gemma 3n/4's per-layer residual (modeling_gemma4.py,
        Gemma4TextDecoderLayer): the layer's own gate over x, activated like
        its feed-forward, multiplied by the layer's input signal, projected
        back and normed."""
        gated = self.per_layer_input_gate(x)
        gated = _gated_activation(self._gate_activation, gated)
        projected = self.per_layer_projection(gated * per_layer_input)
        return self.post_per_layer_input_norm(projected)

    @property
    def _gate_activation(self) -> str:
        return getattr(self.mlp, 'activation', 'swiglu')

@logical_axes({
    # The input is two embed-width vectors concatenated, which no single name
    # describes and the rules must not split twice; the output side shards.
    ("eh_proj",): (None, "embed"),
})
class MTPBlock(nn.Module):
    """One multi-token-prediction depth: the next depth's hidden states.

    The depth norms the token embeddings with `enorm` and the previous
    hidden states with `hnorm`, projects the pair concatenated in that
    order back to the model width, and runs one decoder block over it. That
    composition is what the released MTP weights were trained for, which
    the engines that run them state (vLLM deepseek_mtp.py and
    glm4_moe_mtp.py, SGLang's DeepseekV3ForCausalLMNextN); transformers
    builds no depth. `block` is a plain forward block: multi-token
    prediction is a training-time auxiliary, so there is no decode path and
    no cache.
    """
    mixer: Callable[..., nn.Module]
    feedforward: Callable[..., nn.Module]
    emb_features: int
    norm_eps: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    sandwich_norms: bool = False
    pre_norms: bool = True
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(
            RMSNorm, epsilon=self.norm_eps, scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast, dtype=self.dtype)
        self.enorm = norm(name='enorm')
        self.hnorm = norm(name='hnorm')
        self.eh_proj = nn.Dense(
            self.emb_features, use_bias=False,
            dtype=self.dtype, precision=self.precision, name='eh_proj')
        self.block = DecoderBlock(
            mixer=self.mixer, feedforward=self.feedforward,
            emb_features=self.emb_features,
            norm_eps=self.norm_eps,
            scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast,
            sandwich_norms=self.sandwich_norms,
            pre_norms=self.pre_norms,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype, precision=self.precision, name='block')
        self.final_norm = norm(name='final_norm')

    def __call__(self, hidden, embeds, train: bool = False,
                 positions=None, segment_ids=None):
        fused = self.eh_proj(jnp.concatenate(
            [self.enorm(embeds), self.hnorm(hidden)], axis=-1))
        return self.final_norm(self.block(
            fused, train=train, positions=positions, segment_ids=segment_ids))


@models("causal_transformer")
@logical_axes({
    ("embed_tokens",): ("vocab", "embed"),
    ("lm_head",): ("embed", "vocab"),
    ("embed_tokens_per_layer",): ("vocab", None),
    ("per_layer_model_projection",): ("embed", "mlp"),
    # AltUp's copies enter and leave through embed-by-embed projections, one
    # per copy past the first, named by their index like the layers; a
    # square kernel takes the shape heuristic the way the other indexed
    # projections do.
}, heuristic=(("altup_projections_*",), ("altup_unembed_projections_*",)))
class CausalTransformer(nn.Module):
    """Decoder-only transformer over token ids: [B, S] int32 -> [B, S, vocab] fp32.

    The defaults are a from-scratch training recipe (multi-head attention,
    swiglu, tied embeddings, no softcap); the fields that differ between the
    open decoders are all here, so a Qwen3 or Gemma3 config is a field
    mapping: num_kv_heads/head_dim for grouped-query attention,
    attention_bias, norm_eps with scale_offset for Gemma's (1 + w) norms and
    sandwich_norms for its second pair of them, attention_scale for its
    query_pre_attn_scalar, embedding_scale and final_logit_softcap for the
    rest of Gemma.

    `layer_types` is the pattern, one kind per layer, and `kinds` says what a
    kind does: its window, and its own rope base or head dim where it has
    one. How a checkpoint's config derives the pattern (Qwen3 makes every
    layer past max_window_layers sliding) belongs to that translation, not
    here: this takes the tuple.

    `mixture` turns the feed-forward of some layers into `moe.SparseMLP`,
    routing each token to a few of its experts; a dense layer keeps the
    leaves it always had, and None is a dense model. The LM objective's
    balance_rate is what moves a mixture's balancing bias.

    causal=False turns every layer into full attention with no cache, the
    encoder a masked diffusion language model denoises with; the parameter
    tree is the same either way.

    per_layer_input_dim turns on Gemma 3n/4 style per-layer input embeddings:
    an extra table of per_layer_input_vocab by layers times dim rows, read
    per layer and added to that layer's input through its own gate. None is a
    plain decoder and leaves the tree unchanged.

    num_kv_shared_layers makes the trailing layers of that count reuse the
    keys and values of the last earlier layer of their own kind instead of
    projecting their own (Gemma 3n/4 cross-layer KV sharing). 0 is a plain
    decoder and leaves the tree unchanged, and use_double_wide_mlp, which
    widens the sharing layers' MLP, needs it.

    altup carries Gemma 3n's `altup_num_inputs` copies of the residual stream
    (`dew.nn.gemma3n`): the embeddings and their projections enter the
    layers as a stack, each block predicts the copies, runs on the active
    one and corrects them all, and the copies come back through their own
    projections to a mean the final norm reads. laurel_rank adds the LAuReL
    block to every layer, activation_sparsity_pattern the gaussian top-k on
    each layer's gate, and a tuple mlp_features gives each layer its own
    feed-forward width. None and an int are a plain decoder.

    partial_rotary_factor rotates that fraction of an unwindowed kind's head
    dims and passes the rest through; a windowed kind rotates whole.
    partial_rotary_type names which published convention the fraction
    follows, because the two rotate different angles: 'proportional' is
    Gemma 4's global layers (a head_dim-wide rope cut short), 'default' is
    Qwen3.5's (a rope of the rotated width alone). The lines of each are
    cited on `dew.nn.attention.rotary_freqs`. Interleaved mRoPE
    (Qwen3.5's mrope_section) is the same rotation for text: with one
    position per token the three grids' angles are equal and the interleave
    reads the same value from each, so text-only input reduces exactly to
    this partial rope (difference 0.0 against the reference's
    apply_interleaved_mrope) and the image-grid positions are not modelled.

    `mixer` names the per-layer token mixer as a value from the `mixers`
    registry, one frozen dataclass per kind carrying the reference's field
    names (`mixer={"kind": "mla", ...}` from a config and the dataclass from
    code agree; an unknown kind or field raises). None is today's
    grouped-query causal attention with no config change. A non-standard kind
    reads its own record and ignores the GQA projection geometry
    (num_kv_heads, head_dim) the context still carries; those fields stay
    validated, so a translation fills them with consistent values.

    `num_nextn_predict_layers` stacks that many multi-token-prediction
    depths after the final norm, each an `MTPBlock` with the model-level
    mixer and a dense feed-forward; 0 is a plain decoder and leaves the
    tree unchanged. Depth d pairs the previous depth's state at position p
    with the embedding of the token at p + d and scores what follows p + d
    (arXiv 2412.19437, section 2.2), so each depth is one position shorter
    than the last.
    """
    vocab_size: int
    emb_features: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: Optional[int] = None       # None: as many as the query heads
    head_dim: Optional[int] = None           # None: emb_features // num_heads
    mlp: str = 'swiglu'                      # 'swiglu' | 'geglu' | 'geglu_exact'
    mlp_features: Union[int, Tuple[int, ...], None] = None  # None: four times emb_features; a tuple: one width per layer (Gemma 3n)
    max_seq_len: int = 2048
    rope_theta: float = 10000.0              # the base a kind does not override
    rope_scaling: Optional[RopeScaling] = None  # Llama 3.1's ramp, unless a kind states its own
    partial_rotary_factor: Optional[float] = None  # None: every dim rotates
    partial_rotary_type: str = 'proportional'  # 'proportional' (Gemma 4) | 'default' (Qwen3.5)
    layer_types: Optional[Tuple[str, ...]] = None  # the pattern, one kind per layer
    kinds: Optional[Mapping[str, LayerKind]] = None  # what each named kind does
    norm_eps: float = 1e-5
    scale_offset: bool = False               # Gemma's (1 + w) RMSNorm scale
    scale_after_cast: bool = False           # Llama and Qwen3 scale the cast activations
    sandwich_norms: bool = False             # Gemma's norms on the sublayer outputs
    pre_norms: bool = True                   # False with sandwich_norms: OLMo 3's post-norm block
    qk_norm: bool = True
    qk_norm_scope: str = 'head'              # 'head' per head (Qwen3); 'projection' whole (OLMo 3)
    v_norm: bool = False                     # Gemma 4's scale-free values norm
    attention_k_eq_v: bool = False           # Gemma 4's global layers read their values off the keys
    layer_scalar: bool = False               # Gemma 4 scales each layer's output by a learned scalar
    attention_bias: bool = False             # q/k/v biases, and o_proj unless o_proj_bias says
    o_proj_bias: Optional[bool] = None       # Qwen2 biases q/k/v while o_proj stays bias-free
    attention_scale: Optional[float] = None  # None: head_dim ** -0.5
    attention_sinks: bool = False
    yarn: Optional[YarnScaling] = None
    attn_logit_softcap: Optional[float] = None  # Gemma 2's attn_logit_softcapping
    output_gate: bool = False                 # Qwen3.5 gates the attention branch
    embedding_scale: bool = False            # Gemma scales embeddings by sqrt(d)
    final_logit_softcap: Optional[float] = None
    tie_embeddings: bool = True
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    attention_impl: Optional[str] = None
    mixture: Optional[Mixture] = None        # None: every layer is dense
    use_double_wide_mlp: bool = False        # Gemma 4 doubles sharing layers' MLP width
    causal: bool = True                      # False: full attention, no cache
    per_layer_input_dim: Optional[int] = None  # Gemma 3n/4 per-layer inputs
    per_layer_input_vocab: Optional[int] = None  # None: vocab_size
    num_kv_shared_layers: int = 0            # trailing layers reusing a provider's K/V; 0 disables
    mixer: Optional[MixerBase] = None         # None: today's attention; a kind value or its record
    num_nextn_predict_layers: int = 0         # MTP depths after the final norm; 0 disables
    altup: Optional[AltUp] = None             # Gemma 3n's stack of residual copies; None disables
    laurel_rank: Optional[int] = None         # Gemma 3n's learned augmented residual; None disables
    activation_sparsity_pattern: Optional[Tuple[float, ...]] = None  # Gemma 3n's gaussian top-k, one fraction per layer

    def __post_init__(self):
        if self.layer_types is not None:
            object.__setattr__(self, "layer_types", tuple(self.layer_types))
        if isinstance(self.mlp_features, (tuple, list)):
            object.__setattr__(self, "mlp_features", tuple(int(width) for width in self.mlp_features))
        if self.activation_sparsity_pattern is not None:
            object.__setattr__(self, "activation_sparsity_pattern",
                               tuple(float(fraction) for fraction in self.activation_sparsity_pattern))
        if isinstance(self.altup, Mapping):
            object.__setattr__(self, "altup", AltUp(**self.altup))
        # A value arrives as a record from a config and as itself from code,
        # and `models.build` already reads one; doing it here too means the
        # plain constructor takes the same records, which is what a test or
        # a notebook writes.
        if isinstance(self.yarn, Mapping):
            object.__setattr__(self, 'yarn', YarnScaling(**self.yarn))
        if isinstance(self.mixture, Mapping):
            object.__setattr__(self, "mixture", Mixture(**self.mixture))
        if isinstance(self.rope_scaling, Mapping):
            object.__setattr__(self, "rope_scaling", RopeScaling(**self.rope_scaling))
        if self.kinds is not None:
            # Frozen, because a module's fields are static to jit and a plain
            # dict cannot be hashed.
            object.__setattr__(self, "kinds", flax.core.freeze({
                name: kind if isinstance(kind, LayerKind) else LayerKind(**kind)
                for name, kind in self.kinds.items()}))
        # A mixer arrives as a kind value from code and as a {"kind": ...}
        # record from a config; the record dispatches on its kind through the
        # same `mixers.build` a value is constructed with, so an unknown kind
        # or field raises either way. Anything else is neither.
        if isinstance(self.mixer, Mapping):
            object.__setattr__(self, "mixer", mixer_from_record(self.mixer))
        elif self.mixer is not None and not isinstance(self.mixer, MixerBase):
            raise ValueError(
                f"mixer is a mixer value, its record, or None, not {self.mixer!r}")
        super().__post_init__()


    @property
    def kv_heads(self) -> int:
        return self.num_heads if self.num_kv_heads is None else self.num_kv_heads

    @property
    def features_per_head(self) -> int:
        return (self.emb_features // self.num_heads
                if self.head_dim is None else self.head_dim)

    @property
    def mlp_widths(self) -> Tuple[int, ...]:
        """Each layer's dense feed-forward width."""
        if self.mlp_features is None:
            return (4 * self.emb_features,) * self.num_layers
        if isinstance(self.mlp_features, tuple):
            return self.mlp_features
        return (self.mlp_features,) * self.num_layers

    @property
    def hidden_features(self) -> int:
        """The model's one feed-forward width, which the routed experts fall
        back to and the prediction depths take; a model whose layers differ
        has none."""
        widths = set(self.mlp_widths)
        if len(widths) != 1:
            raise ValueError(
                f"mlp_features names a width per layer ({self.mlp_features}), so "
                f"the model has no single feed-forward width; set the mixture's "
                f"expert_features and leave the prediction depths off")
        return widths.pop()

    @property
    def per_layer_types(self) -> Tuple[str, ...]:
        if self.layer_types is None:
            return ('full_attention',) * self.num_layers
        return tuple(self.layer_types)

    def kind_of(self, layer_type: str) -> "ResolvedKind":
        """What the layers of `layer_type` do, the model's defaults included."""
        kind = (self.kinds or {}).get(layer_type, LayerKind())
        return ResolvedKind(
            window=kind.window,
            num_kv_heads=self.kv_heads if kind.num_kv_heads is None else kind.num_kv_heads,
            rope_theta=self.rope_theta if kind.rope_theta is None else kind.rope_theta,
            rope_scaling=self.rope_scaling if kind.rope_scaling is None else kind.rope_scaling,
            head_dim=(self.features_per_head if kind.head_dim is None else kind.head_dim),
            mixer=kind.mixer)

    @property
    def sparse_layers(self) -> Tuple[int, ...]:
        """The layers whose feed-forward routes to experts."""
        mixture = self.mixture
        if mixture is None:
            return ()
        if mixture.layers is not None:
            return tuple(mixture.layers)
        if mixture.every is not None:
            return tuple(index for index in range(self.num_layers)
                         if (index + 1) % mixture.every == 0)
        return tuple(range(self.num_layers))

    @property
    def kv_sharing(self) -> dict:
        """Sharing layer index to the provider it reads, both of one layer type.

        The trailing num_kv_shared_layers layers own no K/V and read the last
        non-sharing layer of their own type (modeling_gemma4.py,
        Gemma4TextAttention). Empty unless sharing is on.
        """
        if not self.num_kv_shared_layers:
            return {}
        first = self.num_layers - self.num_kv_shared_layers
        if first <= 0:
            raise ValueError(
                f"num_kv_shared_layers ({self.num_kv_shared_layers}) has to leave "
                f"a provider: it must be between 1 and num_layers - 1 "
                f"({self.num_layers - 1})")
        types = self.per_layer_types
        providers = {}
        for index in range(first, self.num_layers):
            earlier = [j for j in range(first) if types[j] == types[index]]
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
        """One layer's mixer geometry: the kind's resolved values as a context.

        `head_dim`, `rope_theta` and `window` already carry the layer kind's
        overrides; a windowed kind rotates every dimension, so the partial
        rotary belongs to the kinds that attend the whole sequence, which is
        where Gemma 4 puts it. A kind builds its `DecoderBlock` factory from
        this and its own record, which is the one place the mixer is chosen.
        """
        return MixerContext(
            emb_features=self.emb_features,
            num_heads=self.num_heads,
            num_kv_heads=kind.num_kv_heads,
            head_dim=kind.head_dim,
            max_seq_len=self.max_seq_len,
            causal=self.causal,
            rope_theta=kind.rope_theta,
            rope_scaling=kind.rope_scaling,
            qk_norm=self.qk_norm,
            qk_norm_scope=self.qk_norm_scope,
            v_norm=self.v_norm,
            k_eq_v=self.attention_k_eq_v and kind.window is None,
            norm_eps=self.norm_eps,
            scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast,
            kv_shared=kv_shared,
            kv_store_key=layer_type,
            sliding_window=kind.window,
            attention_bias=self.attention_bias,
            o_proj_bias=self.o_proj_bias,
            attention_scale=self.attention_scale,
            attention_sinks=self.attention_sinks,
            yarn=self.yarn,
            attn_logit_softcap=self.attn_logit_softcap,
            output_gate=self.output_gate,
            dtype=self.dtype,
            precision=self.precision,
            attention_impl=self.attention_impl,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            partial_rotary_factor=(None if kind.window is not None
                                   else self.partial_rotary_factor),
            partial_rotary_type=self.partial_rotary_type)

    def setup(self):
        types = self.per_layer_types
        if len(types) != self.num_layers:
            raise ValueError(
                f"layer_types has {len(types)} entries for {self.num_layers} layers")
        unnamed = sorted(set(self.kinds or {}) - set(types))
        if unnamed:
            raise ValueError(
                f"kinds {unnamed} name no layer of this model, whose pattern is "
                f"{sorted(set(types))}")
        kinds = {layer_type: self.kind_of(layer_type) for layer_type in set(types)}
        for layer_type, kind in sorted(kinds.items()):
            if kind.head_dim % 2:
                raise ValueError(
                    "rotary positions rotate pairs, so the head dim of "
                    f"{layer_type!r} must be even, got {kind.head_dim}")
            if kind.window is not None and kind.window < 1:
                raise ValueError(
                    f"the window of {layer_type!r} must be positive, got {kind.window}")
            if kind.num_kv_heads < 1 or self.num_heads % kind.num_kv_heads:
                raise ValueError(
                    f"num_heads ({self.num_heads}) must be a multiple of the key/value "
                    f"heads of {layer_type!r} ({kind.num_kv_heads})")
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
        sharing = self.kv_sharing
        if self.use_double_wide_mlp and not sharing:
            raise ValueError(
                "use_double_wide_mlp widens the MLP of the layers that share "
                "their keys and values, so it needs num_kv_shared_layers set")
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
        if len(widths) != self.num_layers or min(widths) < 1:
            raise ValueError(
                f"mlp_features names one positive width per layer of "
                f"{self.num_layers}, got {self.mlp_features}")
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

        self.embed_tokens = nn.Embed(
            num_embeddings=self.vocab_size, features=self.emb_features,
            dtype=self.dtype, name='embed_tokens')
        if ple:
            # The packed table every layer reads its own slice of
            # (modeling_gemma4.py, Gemma4TextModel): one row per token, a
            # hidden_size_per_layer_input slice per layer.
            self.embed_tokens_per_layer = nn.Embed(
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
        mixture = self.mixture
        # The shared branch is the dense feed-forward at the mixture's shared
        # width, handed to the sparse layer as a factory the way the block
        # takes its own slots.
        shared = None if mixture is None or not mixture.shared_features else (
            functools.partial(
                GatedMLP,
                hidden_features=mixture.shared_features,
                out_features=self.emb_features,
                activation=self.mlp,
                dtype=self.dtype,
                precision=self.precision))
        routed = None if mixture is None else functools.partial(
            SparseMLP,
            num_experts=mixture.experts,
            top_k=mixture.top_k,
            hidden_features=(self.hidden_features
                             if mixture.expert_features is None
                             else mixture.expert_features),
            out_features=self.emb_features,
            activation=self.mlp,
            score_function=mixture.score_function,
            normalize_weights=mixture.norm_topk_prob,
            routed_scaling_factor=mixture.scaling,
            expert_groups=mixture.groups,
            groups_per_token=mixture.groups_per_token,
            group_score=mixture.group_score,
            expert_bias=mixture.bias,
            scale_inputs=mixture.scale_inputs,
            shared=shared,
            dtype=self.dtype,
            precision=self.precision)
        parallel = None if mixture is None or not mixture.parallel else functools.partial(
            Gemma4Experts,
            num_experts=mixture.experts,
            top_k=mixture.top_k,
            hidden_features=(self.hidden_features
                             if mixture.expert_features is None
                             else mixture.expert_features),
            out_features=self.emb_features,
            activation=self.mlp,
            norm_eps=self.norm_eps,
            scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast,
            dtype=self.dtype,
            precision=self.precision)
        if parallel is not None:
            # The branch rides beside every sparse layer's dense feed-forward.
            routed = None
        if self.mlp == 'swigluoai':
            if mixture is None or mixture.shared_features or len(sparse) != self.num_layers:
                raise ValueError('swigluoai requires routed experts on every layer and no shared experts')
            routed = functools.partial(
                GptOssMLP, hidden_size=self.emb_features,
                intermediate_size=self.hidden_features,
                num_local_experts=mixture.experts, num_experts_per_tok=mixture.top_k,
                dtype=self.dtype, precision=self.precision)
        # None is today's attention; a kind names its own mixer on LayerKind
        # and otherwise rides the model's. The one dispatch stays build over
        # the layer's context.
        mixer_spec = self.mixer if self.mixer is not None else AttentionMixer()
        self.layers = [
            DecoderBlock(
                mixer=(kinds[layer_type].mixer or mixer_spec).build(
                    self.mixer_context(kinds[layer_type], layer_type, index in sharing)),
                feedforward=(
                    routed
                    if index in sparse and routed is not None else
                    functools.partial(
                        GatedMLP,
                        hidden_features=(2 * widths[index]
                                         if self.use_double_wide_mlp and index in sharing
                                         else widths[index]),
                        out_features=self.emb_features,
                        activation=self.mlp,
                        activation_sparsity=0.0 if sparsity is None else sparsity[index],
                        precision=self.precision)),
                emb_features=self.emb_features,
                norm_eps=self.norm_eps,
                scale_offset=self.scale_offset,
                scale_after_cast=self.scale_after_cast,
                sandwich_norms=self.sandwich_norms,
                pre_norms=self.pre_norms,
                per_layer_input_dim=ple or 0,
                parallel=parallel if index in sparse else None,
                layer_scalar=self.layer_scalar,
                altup=self.altup,
                laurel_rank=self.laurel_rank,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                precision=self.precision,
                name=f'layers_{index}')
            for index, layer_type in enumerate(types)]
        # Prediction depths mirror whole-sequence hidden states, so their
        # mixer builds from the full-attention kind where the pattern has
        # one, else from the first layer's kind; the feed-forward routes
        # like the last layer's (GLM 4.5 ships its depth with the trunk's
        # experts) and is dense otherwise.
        mtp_type = 'full_attention' if 'full_attention' in types else types[0]
        mtp_mixer = mixer_spec.build(self.mixer_context(
            kinds[mtp_type], mtp_type, False))
        mtp_feedforward = (
            routed if routed is not None and self.num_layers - 1 in sparse else
            functools.partial(
                GatedMLP,
                # The last layer's width: the one width of every model with
                # depths, since the widths that vary are Gemma 3n's alone.
                hidden_features=widths[-1],
                out_features=self.emb_features,
                activation=self.mlp,
                precision=self.precision))
        self.mtp = [
            MTPBlock(
                mixer=mtp_mixer, feedforward=mtp_feedforward,
                emb_features=self.emb_features,
                norm_eps=self.norm_eps,
                scale_offset=self.scale_offset,
                scale_after_cast=self.scale_after_cast,
                sandwich_norms=self.sandwich_norms,
                pre_norms=self.pre_norms,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype, precision=self.precision, name=f'mtp_{depth}')
            for depth in range(self.num_nextn_predict_layers)]
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
        self.norm = RMSNorm(
            epsilon=self.norm_eps, scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast, dtype=self.dtype, name='norm')
        if not self.tie_embeddings:
            self.lm_head = nn.Dense(
                features=self.vocab_size, use_bias=False, dtype=jnp.float32,
                precision=self.precision, name='lm_head')

    def __call__(self, tokens, train: bool = False, decode: bool = False,
                 positions=None, segment_ids=None):
        x = self.hidden_states(tokens, train=train, decode=decode,
                               positions=positions, segment_ids=segment_ids)
        if self.is_initializing() and self.mtp:
            # Flax creates a parameter where a call first reaches it, and the
            # main forward never enters the prediction depths. Reaching them
            # here, during init only, makes the model's tree the model's
            # business: a plain init holds every depth.
            self.mtp_hidden_states(x, tokens, train=train, positions=positions,
                                   segment_ids=segment_ids)
        return self._logits(x)

    def _logits(self, x):
        """The shared fp32 head over `x`: what `__call__` and every MTP depth score with."""
        # fp32 head, as in the DiT output projection: the loss is computed in fp32
        if self.tie_embeddings:
            logits = jnp.einsum(
                '...d,vd->...v', x.astype(jnp.float32),
                self.embed_tokens.embedding.astype(jnp.float32),
                precision=self.precision)
        else:
            logits = self.lm_head(x)
        logits = logits.astype(jnp.float32)
        if self.final_logit_softcap is not None:
            cap = jnp.asarray(self.final_logit_softcap, jnp.float32)
            logits = cap * jnp.tanh(logits / cap)
        return logits

    def mtp_hidden_states(self, hidden, tokens, train: bool = False,
                          positions=None, segment_ids=None):
        """One `[B, S - d, D]` final-normed state array per prediction depth d.

        Depth d reads the previous depth's states at positions p and the
        embeddings of the tokens at p + d, the sequence one shorter per
        depth, so its state at p is what scores the token after p + d
        through the shared head (`mtp_logits`, or a chunked loss over
        `head_weight`). `positions` and `segment_ids` are the main forward's,
        sliced with the states, so a packed batch keeps its documents apart
        in the depths too. Empty without prediction depths; a sequence with
        no position d tokens out raises.

        A plain init of the main forward holds these depths too: `__call__`
        reaches them while initializing, so the tree does not depend on which
        method built it.
        """
        if self.mtp and tokens.shape[1] <= len(self.mtp):
            raise ValueError(
                f"{len(self.mtp)} prediction depths need more than "
                f"{len(self.mtp)} tokens, got {tokens.shape[1]}")
        embeds = self.embed_tokens(tokens)
        states = []
        for depth, block in enumerate(self.mtp, start=1):
            hidden = block(
                hidden[:, :-1], embeds[:, depth:], train=train,
                positions=None if positions is None else positions[:, :-depth],
                segment_ids=None if segment_ids is None else segment_ids[:, :-depth])
            states.append(hidden)
        return states

    def mtp_logits(self, hidden, tokens, train: bool = False,
                   positions=None, segment_ids=None):
        """One `[B, S - d, vocab]` fp32 logits array per prediction depth d:
        the depth states of `mtp_hidden_states` through the shared head."""
        return [self._logits(state) for state in self.mtp_hidden_states(
            hidden, tokens, train=train, positions=positions, segment_ids=segment_ids)]

    def hidden_states(self, tokens, train: bool = False, decode: bool = False,
                      positions=None, segment_ids=None):
        """The final normalised states, `[B, S, D]`: everything the forward
        pass does before the head projection.

        A loss that pairs this with `head_weight` scores tokens without ever
        holding the full `[B, S, vocab]` logits tensor. A packed batch passes
        its per-document `positions` and `segment_ids` through to the layers,
        which is where RoPE and the mask read them.
        """
        x = self.embed_tokens(tokens)
        if self.embedding_scale:
            # Gemma casts embed_scale to the embedding weight dtype
            # (modeling_gemma3.py:117). Dew's nn.Embed holds that table in
            # fp32 and returns the compute dtype, so the factor keeps its
            # fp32 value and only the product rounds with the activations.
            # A factor rounded to bf16 would be 34.0 at hidden 1152, where
            # sqrt(1152) is 33.94112549695428.
            scaled = x * jnp.asarray(math.sqrt(self.emb_features),
                                     self.embed_tokens.embedding.dtype)
            x = scaled.astype(x.dtype)
        ple = self.per_layer_inputs(tokens, x) if self.per_layer_input_dim else None
        if self.altup is not None:
            # The embeddings and, rescaled to their magnitude, each projected
            # copy: [num_inputs, B, S, D].
            x = jnp.stack([x] + [rescale_to(project(x), x) for project in self.altup_projections])
        kv_store = {} if self.num_kv_shared_layers else None
        for index, layer in enumerate(self.layers):
            x = layer(x, train=train, decode=decode,
                      positions=positions, segment_ids=segment_ids,
                      kv_store=kv_store,
                      per_layer_input=None if ple is None else ple[:, :, index, :])
        if self.altup is not None:
            # The copies past the first come back through their own
            # projections, rescaled to the first's magnitude, and the mean of
            # all of them is what the final norm reads.
            copies = [x[0]] + [rescale_to(project(copy), x[0])
                               for project, copy in zip(self.altup_unembed_projections, x[1:])]
            x = jnp.mean(jnp.stack(copies), axis=0)
        return self.norm(x)

    def per_layer_inputs(self, tokens, inputs_embeds):
        """Every layer's input signal `[B, S, L, P]` (Gemma 3n/4 PLE).

        The token-identity component is the packed table's row for each
        token, scaled like the main embedding; the context component is the
        input embeddings projected down, scaled and normed. Their sum over
        sqrt(2) is what each layer's gate multiplies in
        (modeling_gemma4.py, get_per_layer_inputs/project_per_layer_inputs).
        """
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

    def head_weight(self, params):
        """The `[D, vocab]` head matrix in fp32, as the forward multiplies it.

        `params` is the parameter tree the forward runs under, so this is a
        plain read: a tied head is the embedding table transposed, an untied
        one is `lm_head`'s kernel, which is `[D, vocab]` already. The Gemma
        embedding scale multiplies the input embeddings only, so it has no
        place here.
        """
        if self.tie_embeddings:
            return params['embed_tokens']['embedding'].astype(jnp.float32).T
        return params['lm_head']['kernel'].astype(jnp.float32)

    def init_cache(self, batch_size: int):
        """Allocate a zeroed decode cache for `batch_size` sequences.

        cache = model.apply(params, batch_size, method=CausalTransformer.init_cache,
                            mutable=['cache'])[1]['cache']

        The forward pass this runs is a single dummy token whose keys are never
        written: allocation happens on the first decode-mode call, the write on
        the ones after it.
        """
        self(jnp.zeros((batch_size, 1), jnp.int32), decode=True)
