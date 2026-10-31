"""Tier 2 from parts: an unregistered decoder assembled from the parts Dew's registered families are
built from, each read from the source's own modeling code (ledger 1607).

Every module of the source's causal LM is matched to a known part, a class of a registered family's
transformers 5.16.1 modeling (`KNOWN`), by its code (`dew.interop.code_identity.same_code`): its
forward is the known class's own, inherited, or the same syntax tree under the source's names, which may
differ from the known class's in the model's class-name prefix and in a one-to-one renaming of the
attributes the code reads off `self`.
The known part then reads, off the instance, the values that code reads (a head width, a norm's epsilon,
an activation, a router's top-k, whether the head is the embedding's own parameter), so a part's
semantics are the source's own, never a guess from its names or shapes. A module no known part matches,
or two parts that state one field differently, leave the decoder to tier 3 (`Unassembled`, naming it).

The renamings map the source's module paths onto the canonical ones `decoder_parts.dew_path` reads
(`Assembly.weight_path`), so a renamed checkpoint loads by its structure. `dew.interop.verify` confirms
an assembly on its shrunken probe before any weight downloads.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from types import FunctionType, MethodType
from typing import TYPE_CHECKING

from dew import _records
from dew.interop import decoder_parts, sources
from dew.interop.code_identity import Renaming, matched_class, same_code
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import ACTIVATIONS, PLAIN_SPELLING, DecoderFields
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mla import MultiHeadLatentAttention

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


# --- the instances, read through the known names --------------------------------------------------

class PartView:
    """A source module read by the known class's attribute names (`Renaming.attributes`)."""

    def __init__(self, module: torch.nn.Module, renaming: Renaming):
        self.module, self.renaming = module, renaming

    def name(self, known: str) -> str:
        return self.renaming.attributes.get(known, known)

    def __getattr__(self, known: str):
        return getattr(self.module, self.name(known))


class Ambiguous(ValueError):
    """A part the source's code does not let tier 2 read, which leaves the decoder to tier 3."""


def gated_activation(module: torch.nn.Module) -> str:
    """The gated MLP Dew computes for the activation module a source part holds: the `hidden_act` name
    transformers' ACT2FN builds that same module for, read back by its class and state."""
    from transformers.activations import ACT2FN

    def state(activation: torch.nn.Module) -> dict[str, tuple]:
        """Each public attribute as it computes: a partial by its function and arguments, a bound
        method by its function, any other value by its class and value."""
        held = {}
        for key, value in vars(activation).items():
            if key.startswith('_') or key == 'training':
                continue
            if isinstance(value, functools.partial):
                held[key] = (value.func, value.args, tuple(sorted(value.keywords.items())))
            elif isinstance(value, MethodType):
                held[key] = (value.__func__,)
            else:
                held[key] = (type(value), value)
        return held

    for name, gated in ACTIVATIONS.items():
        built = ACT2FN[name]
        if type(built) is type(module) and state(built) == state(module):
            return gated
    raise Ambiguous(f"the activation {type(module).__name__} is none of {sorted(ACTIVATIONS)}, the gated "
                    "MLP's")


def _biased(*linears: torch.nn.Module) -> bool:
    biased = {linear.bias is not None for linear in linears}
    if len(biased) != 1:
        raise Ambiguous("projections that Dew biases together are biased apart")
    return biased.pop()


def _rope(config, *, partial: bool = False) -> Mapping[str, object]:
    """The rotary a `LlamaRotaryEmbedding`-shaped class builds from its config's standardized record
    (its rope_type through ROPE_INIT_FUNCTIONS, `decoder_parts.read_rope`), over the whole head; where
    `partial`, a class whose code turns the record's `partial_rotary_factor` of each head (GLM-4's and
    StableLM's), over that fraction, its frequencies over the dimensions it turns (`partial_rotary_type`
    'default', GPT-NeoX's)."""
    parameters = dict(config.rope_parameters or {})
    factor = parameters.get('partial_rotary_factor', 1.0)
    if factor != 1.0 and not partial:
        raise Ambiguous("the rotary turns part of each head, which this rotation applies whole")
    entry = {key: value for key, value in parameters.items() if key != 'partial_rotary_factor'}
    try:
        ropes = decoder_parts.read_rope({'rope_parameters': entry,
                                         'max_position_embeddings': config.max_position_embeddings},
                                        set(), PLAIN_SPELLING, per_kind=False)
    except ValueError as error:
        raise Ambiguous(f"the rotary's record is not one this rotation reads ({error})") from error
    fields: dict[str, object] = {'rope_theta': ropes.theta}
    if ropes.scaling is not None:
        fields['rope_scaling'] = ropes.scaling
    if factor != 1.0:
        fields.update(partial_rotary_factor=float(factor), partial_rotary_type='default')
    return fields


# Readers: each takes the part's view and returns the CausalTransformer fields its code states. A
# field several parts state must agree (`assemble`); per-layer facts go under `_LAYER`.
_LAYER = '_layer'


def _read_causal_lm(view: PartView) -> Mapping[str, object]:
    if view.lm_head.bias is not None:
        raise Ambiguous("the head is biased, which this head's code does not add")
    return {}


def _read_scaled_causal_lm(view: PartView) -> Mapping[str, object]:
    """Granite's head: Llama's, its logits divided by the config's `logits_scaling`."""
    return {**_read_causal_lm(view), 'logits_scaling': float(view.config.logits_scaling)}


def _read_scaled_model(view: PartView) -> Mapping[str, object]:
    """Granite's model: Llama's, its embeddings times its `embedding_multiplier`."""
    return {**_read_llama_model(view), 'embedding_multiplier': float(view.embedding_multiplier)}


def _read_scaled_layer(view: PartView) -> Mapping[str, object]:
    """Granite's layer: Llama's, each sublayer's output times its `residual_multiplier` into the residual."""
    return {'residual_multiplier': float(view.residual_multiplier)}


def _read_llama_model(view: PartView) -> Mapping[str, object]:
    layers = len(view.layers)
    if view.config.num_hidden_layers != layers:
        raise Ambiguous(f"the forward runs {view.config.num_hidden_layers} of its {layers} layers")
    return {'num_layers': layers}


def _read_windowed_model(view: PartView) -> Mapping[str, object]:
    """Mixtral's and Mistral's model: every layer attends a window where the config states one
    (`create_sliding_window_causal_mask`)."""
    window = view.config.sliding_window
    fields = dict(_read_llama_model(view))
    if window is not None:
        layers = _records.integer(fields['num_layers'], 'num_layers')
        fields.update(layer_types=('sliding_attention',) * layers,
                      kinds={'sliding_attention': native_fields(LayerKind)(
                          window=_records.integer(window, 'sliding_window'))})
    return fields


def _read_layer_typed_model(view: PartView) -> Mapping[str, object]:
    """Qwen2's model: each layer attends under its config layer type's mask, a window where it is
    `sliding_attention` (`create_sliding_window_causal_mask` at the config's sliding_window)."""
    fields = dict(_read_llama_model(view))
    types = tuple(_records.strings(view.config.layer_types, 'layer_types'))[:fields['num_layers']]
    if set(types) - {'full_attention', 'sliding_attention'}:
        raise Ambiguous(f"the layer types {sorted(set(types))} are not attention's full or sliding masks")
    fields['layer_types'] = types
    if 'sliding_attention' in types:
        fields['kinds'] = {'sliding_attention': native_fields(LayerKind)(
            window=_records.integer(view.config.sliding_window, 'sliding_window'))}
    return fields


def _read_attention(view: PartView) -> Mapping[str, object]:
    width = view.head_dim
    if not view.is_causal:
        raise Ambiguous("the attention is not causal")
    heads, kv_heads = view.q_proj.out_features // width, view.k_proj.out_features // width
    if heads * width != view.q_proj.out_features or kv_heads * width != view.k_proj.out_features:
        raise Ambiguous("the projections hold no whole number of heads")
    biased = _biased(view.q_proj, view.k_proj, view.v_proj)
    fields: dict[str, object] = {
        'num_heads': heads, 'num_kv_heads': kv_heads, 'head_dim': width, 'qk_norm': False,
        'attention_bias': biased, 'attention_dropout_rate': float(view.attention_dropout)}
    if (view.o_proj.bias is not None) != biased:
        # Qwen2 biases its queries, keys and values and not its output.
        fields['o_proj_bias'] = view.o_proj.bias is not None
    if view.scaling != width ** -0.5:
        fields['attention_scale'] = float(view.scaling)
    return fields


def _read_qk_norm_attention(view: PartView) -> Mapping[str, object]:
    """Qwen3's attention: Llama's with an RMSNorm over each head's queries and keys (`q_norm`, `k_norm`,
    read as norms like the block's)."""
    return {**_read_attention(view), 'qk_norm': True}


def _read_nope_attention(view: PartView) -> Mapping[str, object]:
    """SmolLM3's attention: Llama's, rotating its queries and keys only where `use_rope` (each fourth layer
    attends without positions, `no_rope_layers`), a per-layer fact `assemble` makes a kind of."""
    return {**_read_attention(view), _LAYER: {'nope': not bool(view.use_rope)}}


def _read_projection_qk_norm_attention(view: PartView) -> Mapping[str, object]:
    """OLMo 2's attention: Llama's with an RMSNorm over the whole query and key projections, before the
    heads split (`qk_norm_scope='projection'`, OLMo 3's)."""
    return {**_read_attention(view), 'qk_norm': True, 'qk_norm_scope': 'projection'}


def _read_olmoe_attention(view: PartView) -> Mapping[str, object]:
    """OLMoE's attention: OLMo 2's, which clips its queries, keys and values to the config's `clip_qkv`
    where it states one, and hands the attention backend the config's `sliding_window`, which its eager
    attention does not apply (its mask is the causal one); Dew's attention does neither."""
    if view.config.clip_qkv is not None:
        raise Ambiguous(f"the attention clips its queries, keys and values to {view.config.clip_qkv}")
    if view.config.to_dict().get('sliding_window') is not None:
        raise Ambiguous("the attention hands its backend a window its causal mask does not apply")
    return _read_projection_qk_norm_attention(view)


def _read_stablelm_attention(view: PartView) -> Mapping[str, object]:
    """StableLM's attention: Llama's projections, its queries and keys turned over their first
    `rotary_ndims` dimensions alone (the config's `partial_rotary_factor` of each head, GPT-NeoX's rotary),
    each head's queries and keys normed by a layer norm of its own where `qk_layernorm`, which Dew's
    shared query and key norm is not."""
    if view.qk_layernorm:
        raise Ambiguous("each head's queries and keys are normed by a layer norm of its own")
    factor = float(view.config.rope_parameters.get('partial_rotary_factor', 1.0))
    if view.rotary_ndims != int(view.head_dim * factor):
        raise Ambiguous(f"the attention turns {view.rotary_ndims} of each head's {view.head_dim} dimensions")
    fields = dict(_read_attention(view))
    if factor != 1.0:
        fields.update(partial_rotary_factor=factor, partial_rotary_type='default')
    return fields


def _read_stablelm_layer(view: PartView) -> Mapping[str, object]:
    """StableLM's layer: pre-norm, its attention and feed-forward beside each other off one norm where
    `use_parallel_residual` (Falcon's shared parallel norm), and its feed-forward's output dropped in
    training at `hidden_dropout`, which Dew drops with no other output."""
    if view.dropout.p:
        raise Ambiguous("the layer drops its feed-forward's output alone in training")
    parallel = bool(view.use_parallel_residual)
    return {'parallel_residual': parallel, 'shared_parallel_norm': parallel}


def _read_output_normed_layer(view: PartView) -> Mapping[str, object]:
    """OLMo 2's layer: each sublayer's output normed before it joins the residual, its input not
    (`sandwich_norms` without `pre_norms`, OLMo 3's)."""
    return {'sandwich_norms': True, 'pre_norms': False}


def _read_scale_first_norm(view: PartView) -> Mapping[str, object]:
    """OLMo 2's RMSNorm: its weight times the normalized input in float32, cast after."""
    return {'norm_eps': float(view.variance_epsilon), 'scale_after_cast': False, 'scale_offset': False}


def _read_mlp(view: PartView) -> Mapping[str, object]:
    return {'mlp': gated_activation(view.act_fn), 'mlp_features': view.gate_proj.out_features,
            'mlp_bias': _biased(view.gate_proj, view.up_proj, view.down_proj)}


def _read_mixtral_moe(view: PartView) -> Mapping[str, object]:
    if view.jitter_noise:
        raise Ambiguous("the router jitters its inputs while training")
    return {}


def _read_router(view: PartView, *, normalized: bool | None) -> Mapping[str, object]:
    norm = normalized if normalized is not None else bool(view.norm_topk_prob)
    return {_LAYER: {'top_k': int(view.top_k), 'experts': int(view.num_experts), 'norm_topk_prob': norm}}


def _read_sigmoid_router(view: PartView) -> Mapping[str, object]:
    """HY-3's router: sigmoid scores in float32, chosen by score plus the block's correction bias,
    weighted by score renormalized over the chosen and times `router_scaling_factor` (DeepSeek V3's)."""
    return {_LAYER: {'top_k': int(view.top_k), 'experts': int(view.num_experts), 'norm_topk_prob': True,
                     'score_function': 'sigmoid', 'scaling': float(view.router_scaling_factor), 'bias': True}}


def _read_grouped_router(view: PartView) -> Mapping[str, object]:
    """DeepSeek V3's router (GLM-4.7-Flash's): sigmoid scores in float32 plus its correction bias,
    experts chosen within the `topk_group` best of `num_group` groups (each scored by its two best),
    weighted by score, renormalized where `norm_topk_prob`, times `routed_scaling_factor`."""
    return {_LAYER: {'top_k': int(view.top_k), 'experts': int(view.num_experts),
                     'norm_topk_prob': bool(view.norm_topk_prob), 'score_function': 'sigmoid',
                     'scaling': float(view.routed_scaling_factor), 'bias': True,
                     'groups': int(view.num_group), 'groups_per_token': int(view.topk_group)}}


def _read_latent_attention(view: PartView) -> Mapping[str, object]:
    """DeepSeek V3's multi-head latent attention (GLM-4.7-Flash's): queries through a low-rank latent
    where `q_lora_rank` (whole otherwise), keys and values from one normed latent beside a shared rotated
    key, each head's query and key a rotated and an unrotated part (the `mla` mixer). Its latents are
    normed at their own epsilon, which the mixer holds at `MultiHeadLatentAttention.norm_eps`, and its
    scale takes the yarn ramp's mscale off its config's rope record, the rotary's (`_rope`), as the
    mixer's does off the layer's."""
    if not view.is_causal:
        raise Ambiguous("the attention is not causal")
    nope, rope = int(view.qk_nope_head_dim), int(view.qk_rope_head_dim)
    rank = view.q_lora_rank
    norms = [view.kv_a_layernorm] + ([] if rank is None else [view.q_a_layernorm])
    epsilon = MultiHeadLatentAttention.norm_eps
    if any(float(norm.variance_epsilon) != epsilon for norm in norms):
        raise Ambiguous(f"the latents are normed at an epsilon other than {epsilon}")
    heads = int(view.num_heads)
    return {**_rope(view.config), 'num_heads': heads, 'num_kv_heads': heads // int(view.num_key_value_groups),
            'head_dim': nope + rope, 'qk_norm': False,
            'attention_bias': view.kv_a_proj_with_mqa.bias is not None,
            'attention_dropout_rate': float(view.attention_dropout),
            'mixer': {'class': 'mla', 'fields': {
                'q_lora_rank': None if rank is None else int(rank), 'kv_lora_rank': int(view.kv_lora_rank),
                'qk_nope_head_dim': nope, 'qk_rope_head_dim': rope, 'v_head_dim': int(view.v_head_dim),
                'rope_interleave': bool(view.config.rope_interleave)}}}


def _read_shared_moe(view: PartView) -> Mapping[str, object]:
    """A routed block beside always-on shared experts (`shared_experts`, an MLP read as `shared`), the two
    summed: DeepSeek V3's, and HY-3's, whose float32 sum of two values of the compute dtype rounds as
    their sum there does (`enable_moe_fp32_combine`), so it reads no field."""
    return {}


def _read_shared_mlp(view: PartView) -> Mapping[str, object]:
    """A routed block's shared experts: Llama's MLP, its width the block's (`Mixture.shared_features`)."""
    width = int(view.gate_proj.out_features)
    return {'mlp': gated_activation(view.act_fn), _LAYER: {'shared_features': width}}


def _read_experts(view: PartView) -> Mapping[str, object]:
    return {'mlp': gated_activation(view.act_fn), _LAYER: {'expert_features': int(view.intermediate_dim)}}


def _read_rms_norm(view: PartView) -> Mapping[str, object]:
    return {'norm_eps': float(view.variance_epsilon), 'scale_after_cast': True, 'scale_offset': False}


def _read_latent_norm(view: PartView) -> Mapping[str, object]:
    """A latent attention's norm (Llama's RMSNorm): the latent mixer norms at the block norms' rounding
    (`MLAMixer.build` passes the layer's), so it states that rounding, which the block norms' must agree
    with; its epsilon the attention's reader holds to the mixer's own."""
    return {'scale_after_cast': True, 'scale_offset': False}


def _read_offset_norm(view: PartView) -> Mapping[str, object]:
    """Gemma's norm: (1 + w) times the normalized input, in float32, cast after."""
    return {'norm_eps': float(view.eps), 'scale_after_cast': False, 'scale_offset': True}


def _read_embedding(view: PartView) -> Mapping[str, object]:
    """A lookup's table sizes and pad row. `nn.Embedding.forward` also renormalizes rows past
    `max_norm` and scales a token's gradient by its frequency (`scale_grad_by_freq`), which Dew's
    lookup does not."""
    if view.max_norm is not None or view.scale_grad_by_freq:
        raise Ambiguous("the embedding renormalizes its rows or scales its gradient by token frequency")
    return {'vocab_size': view.num_embeddings, 'emb_features': view.embedding_dim,
            'padding_idx': view.padding_idx}


def _read_scaled_embedding(view: PartView) -> Mapping[str, object]:
    """Gemma's scaled embedding: the lookup times sqrt(width), the factor cast to the table's dtype,
    which Dew's `embedding_scale` computes; any other factor is not Dew's."""
    if float(view.scalar_embed_scale) != view.embedding_dim ** 0.5:
        raise Ambiguous(f"the embedding scales by {view.scalar_embed_scale}, not by sqrt(width)")
    return {**_read_embedding(view), 'embedding_scale': True}


def _read_rotary(view: PartView) -> Mapping[str, object]:
    return _rope(view.config)


def _read_partial_rotary(view: PartView) -> Mapping[str, object]:
    """A rotary whose code turns the config's `partial_rotary_factor` of each head (`_rope`)."""
    return _rope(view.config, partial=True)


def _read_layer_norm(view: PartView) -> Mapping[str, object]:
    """torch's LayerNorm over the width, its weight and bias (where it holds one) applied in the input's
    dtype, as `norm_type='layer'` with `scale_after_cast=False` (GPT-NeoX's)."""
    if not view.elementwise_affine or len(view.normalized_shape) != 1:
        raise Ambiguous("the layer norm scales by no weight or normalizes more than the width")
    return {'norm_type': 'layer', 'norm_bias': view.bias is not None, 'norm_eps': float(view.eps),
            'scale_after_cast': False, 'scale_offset': False}


def _read_nothing(view: PartView) -> Mapping[str, object]:
    return {}


@dataclass(frozen=True)
class KnownPart:
    """A known class's place in a decoder: its role, the roles of the children its code calls (by the
    known class's attribute names, which are also their canonical module names under it), and the
    reader of what its code states."""
    role: str
    location: str
    """`module:qualname` of the known class."""
    children: Mapping[str, tuple[str, ...]]
    read: Callable[[PartView], Mapping[str, object]]
    parameters: tuple[str, ...] = ()
    """The parameters it holds itself, by the known class's names (a norm's weight)."""
    whole: bool = False
    """Whether every method of the known class is compared, its constructor's included: a part whose
    buffers its constructor computes (a rotary's frequencies), which the forward reads and the reader
    takes from the config."""
    state: Mapping[str, str] = field(default_factory=dict)
    """The state a checkpoint stores beside its parameters (a router's correction bias, a buffer), by the
    known class's names, onto its canonical name under the part (HY-3's MoE holds the bias its router
    reads, which Dew keeps in the router, `gate.e_score_correction_bias`)."""

    def known(self) -> type:
        module, _, qualname = self.location.partition(':')
        return getattr(__import__(module, fromlist=[qualname]), qualname)


_LLAMA = 'transformers.models.llama.modeling_llama'
_MIXTRAL = 'transformers.models.mixtral.modeling_mixtral'
_QWEN3 = 'transformers.models.qwen3.modeling_qwen3'
_QWEN3_MOE = 'transformers.models.qwen3_moe.modeling_qwen3_moe'
_GEMMA = 'transformers.models.gemma.modeling_gemma'
_MISTRAL = 'transformers.models.mistral.modeling_mistral'
_QWEN2 = 'transformers.models.qwen2.modeling_qwen2'
_GRANITE = 'transformers.models.granite.modeling_granite'
_SMOLLM3 = 'transformers.models.smollm3.modeling_smollm3'
_OLMO2 = 'transformers.models.olmo2.modeling_olmo2'
_HY_V3 = 'transformers.models.hy_v3.modeling_hy_v3'
_OLMOE = 'transformers.models.olmoe.modeling_olmoe'
_STABLELM = 'transformers.models.stablelm.modeling_stablelm'
_DEEPSEEK_V3 = 'transformers.models.deepseek_v3.modeling_deepseek_v3'
_GLM4_MOE = 'transformers.models.glm4_moe.modeling_glm4_moe'
Children = Mapping[str, tuple[str, ...]]
_MODEL_CHILDREN: Children = {'embed_tokens': ('embedding',), 'layers': ('layers',), 'norm': ('norm',),
                             'rotary_emb': ('rotary',)}
_LAYER_CHILDREN: Children = {'self_attn': ('attention',), 'mlp': ('mlp', 'moe'), 'input_layernorm': ('norm',),
                             'post_attention_layernorm': ('norm',)}
_ATTENTION_CHILDREN: Children = dict.fromkeys(('q_proj', 'k_proj', 'v_proj', 'o_proj'), ('linear',))
_SHARED_MOE_CHILDREN: Children = {'gate': ('router',), 'experts': ('experts',), 'shared_experts': ('shared',)}
_LATENT_CHILDREN: Children = {
    **dict.fromkeys(('q_proj', 'q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj'),
                    ('linear',)),
    **dict.fromkeys(('q_a_layernorm', 'kv_a_layernorm'), ('latent_norm',))}

KNOWN: tuple[KnownPart, ...] = (
    KnownPart('causal_lm', f'{_LLAMA}:LlamaForCausalLM', {'model': ('model',), 'lm_head': ('linear',)},
         _read_causal_lm),
    KnownPart('causal_lm', f'{_MIXTRAL}:MixtralForCausalLM', {'model': ('model',), 'lm_head': ('linear',)},
         _read_causal_lm),
    KnownPart('causal_lm', f'{_GRANITE}:GraniteForCausalLM', {'model': ('model',), 'lm_head': ('linear',)},
              _read_scaled_causal_lm),
    KnownPart('causal_lm', f'{_HY_V3}:HYV3ForCausalLM', {'model': ('model',), 'lm_head': ('linear',)},
              _read_causal_lm),
    KnownPart('model', f'{_HY_V3}:HYV3Model', _MODEL_CHILDREN, _read_llama_model),
    KnownPart('model', f'{_OLMOE}:OlmoeModel', _MODEL_CHILDREN, _read_llama_model),
    KnownPart('causal_lm', f'{_STABLELM}:StableLmForCausalLM', {'model': ('model',), 'lm_head': ('linear',)},
              _read_causal_lm),
    KnownPart('model', f'{_STABLELM}:StableLmModel', _MODEL_CHILDREN, _read_llama_model),
    KnownPart('layer', f'{_STABLELM}:StableLmDecoderLayer', _LAYER_CHILDREN, _read_stablelm_layer),
    KnownPart('attention', f'{_STABLELM}:StableLmAttention', _ATTENTION_CHILDREN, _read_stablelm_attention),
    KnownPart('rotary', f'{_STABLELM}:StableLmRotaryEmbedding', {}, _read_partial_rotary, whole=True),
    KnownPart('norm', 'torch.nn:LayerNorm', {}, _read_layer_norm, ('weight', 'bias')),
    KnownPart('model', f'{_LLAMA}:LlamaModel', _MODEL_CHILDREN, _read_llama_model),
    KnownPart('model', f'{_GRANITE}:GraniteModel', _MODEL_CHILDREN, _read_scaled_model),
    KnownPart('model', f'{_MIXTRAL}:MixtralModel', _MODEL_CHILDREN, _read_windowed_model),
    KnownPart('model', f'{_GEMMA}:GemmaModel', _MODEL_CHILDREN, _read_llama_model),
    KnownPart('model', f'{_MISTRAL}:MistralModel', _MODEL_CHILDREN, _read_windowed_model),
    KnownPart('model', f'{_QWEN2}:Qwen2Model', _MODEL_CHILDREN, _read_layer_typed_model),
    KnownPart('layer', f'{_LLAMA}:LlamaDecoderLayer', _LAYER_CHILDREN, _read_nothing),
    KnownPart('layer', f'{_MIXTRAL}:MixtralDecoderLayer', _LAYER_CHILDREN, _read_nothing),
    KnownPart('layer', f'{_GRANITE}:GraniteDecoderLayer', _LAYER_CHILDREN, _read_scaled_layer),
    KnownPart('layer', f'{_OLMO2}:Olmo2DecoderLayer',
              {'self_attn': ('attention',), 'mlp': ('mlp', 'moe'), 'post_attention_layernorm': ('norm',),
               'post_feedforward_layernorm': ('norm',)}, _read_output_normed_layer),
    KnownPart('attention', f'{_LLAMA}:LlamaAttention', _ATTENTION_CHILDREN, _read_attention),
    KnownPart('attention', f'{_MIXTRAL}:MixtralAttention', _ATTENTION_CHILDREN, _read_attention),
    KnownPart('attention', f'{_QWEN2}:Qwen2Attention', _ATTENTION_CHILDREN, _read_attention),
    KnownPart('attention', f'{_SMOLLM3}:SmolLM3Attention', _ATTENTION_CHILDREN, _read_nope_attention),
    KnownPart('attention', f'{_DEEPSEEK_V3}:DeepseekV3Attention', _LATENT_CHILDREN, _read_latent_attention),
    KnownPart('attention', f'{_HY_V3}:HYV3Attention',
              {**_ATTENTION_CHILDREN, 'q_norm': ('norm',), 'k_norm': ('norm',)}, _read_qk_norm_attention),
    KnownPart('attention', f'{_OLMOE}:OlmoeAttention',
              {**_ATTENTION_CHILDREN, 'q_norm': ('norm',), 'k_norm': ('norm',)}, _read_olmoe_attention),
    KnownPart('attention', f'{_OLMO2}:Olmo2Attention',
              {**_ATTENTION_CHILDREN, 'q_norm': ('norm',), 'k_norm': ('norm',)},
              _read_projection_qk_norm_attention),
    KnownPart('attention', f'{_QWEN3}:Qwen3Attention',
         {**_ATTENTION_CHILDREN, 'q_norm': ('norm',), 'k_norm': ('norm',)}, _read_qk_norm_attention),
    KnownPart('mlp', f'{_LLAMA}:LlamaMLP', dict.fromkeys(('gate_proj', 'up_proj', 'down_proj'), ('linear',)),
         _read_mlp),
    KnownPart('moe', f'{_MIXTRAL}:MixtralSparseMoeBlock', {'gate': ('router',), 'experts': ('experts',)},
         _read_mixtral_moe),
    KnownPart('moe', f'{_QWEN3_MOE}:Qwen3MoeSparseMoeBlock', {'gate': ('router',), 'experts': ('experts',)},
         _read_nothing),
    KnownPart('moe', f'{_OLMOE}:OlmoeSparseMoeBlock', {'gate': ('router',), 'experts': ('experts',)},
              _read_nothing),
    KnownPart('moe', f'{_HY_V3}:HYV3MoE', _SHARED_MOE_CHILDREN, _read_shared_moe,
              state={'e_score_correction_bias': 'gate.e_score_correction_bias'}),
    KnownPart('shared', f'{_LLAMA}:LlamaMLP',
              dict.fromkeys(('gate_proj', 'up_proj', 'down_proj'), ('linear',)), _read_shared_mlp),
    KnownPart('router', f'{_HY_V3}:HYV3TopKRouter', {}, _read_sigmoid_router, ('weight',)),
    KnownPart('moe', f'{_DEEPSEEK_V3}:DeepseekV3MoE', _SHARED_MOE_CHILDREN, _read_shared_moe),
    KnownPart('router', f'{_DEEPSEEK_V3}:DeepseekV3TopkRouter', {}, _read_grouped_router,
              ('weight',), state={'e_score_correction_bias': 'e_score_correction_bias'}),
    KnownPart('latent_norm', f'{_LLAMA}:LlamaRMSNorm', {}, _read_latent_norm, ('weight',)),
    KnownPart('router', f'{_MIXTRAL}:MixtralTopKRouter', {}, lambda view: _read_router(view, normalized=True),
         ('weight',)),
    KnownPart('router', f'{_QWEN3_MOE}:Qwen3MoeTopKRouter', {},
              lambda view: _read_router(view, normalized=None), ('weight',)),
    KnownPart('experts', f'{_MIXTRAL}:MixtralExperts', {}, _read_experts, ('gate_up_proj', 'down_proj')),
    KnownPart('norm', f'{_LLAMA}:LlamaRMSNorm', {}, _read_rms_norm, ('weight',)),
    KnownPart('norm', f'{_GEMMA}:GemmaRMSNorm', {}, _read_offset_norm, ('weight',)),
    KnownPart('norm', f'{_OLMO2}:Olmo2RMSNorm', {}, _read_scale_first_norm, ('weight',)),
    KnownPart('rotary', f'{_LLAMA}:LlamaRotaryEmbedding', {}, _read_rotary, whole=True),
    KnownPart('rotary', f'{_QWEN2}:Qwen2RotaryEmbedding', {}, _read_rotary, whole=True),
    KnownPart('rotary', f'{_OLMO2}:Olmo2RotaryEmbedding', {}, _read_rotary, whole=True),
    KnownPart('rotary', f'{_GLM4_MOE}:Glm4MoeRotaryEmbedding', {}, _read_partial_rotary, whole=True),
    KnownPart('embedding', 'torch.nn:Embedding', {}, _read_embedding, ('weight',)),
    KnownPart('embedding', f'{_GEMMA}:GemmaTextScaledWordEmbedding', {}, _read_scaled_embedding, ('weight',)),
    KnownPart('linear', 'torch.nn:Linear', {}, _read_nothing, ('weight', 'bias')),
)
"""The known parts: transformers' classes whose computation Dew's decoder states by the fields their readers
return, a registered family's or a type's own (Granite's, OLMo 2's), each assembly of them still verified
against its source's code (`dew.interop.verify`). A class that computes one of these by its code (Qwen2's
MLP is Llama's, Qwen3-MoE's experts Mixtral's, GLM-4.7-Flash's attention DeepSeek V3's) matches it through
`same_code`, so each entry stands for every copy of its code."""


@cache
def _known_by_role() -> dict[str, tuple[tuple[KnownPart, type], ...]]:
    table: dict[str, list[tuple[KnownPart, type]]] = {}
    for part in KNOWN:
        table.setdefault(part.role, []).append((part, part.known()))
    return {role: tuple(entries) for role, entries in table.items()}


def identify(module: torch.nn.Module, roles: tuple[str, ...]) -> tuple[KnownPart, Renaming]:
    """The known part `module` is in one of `roles`, and the renaming its code differs by, or
    `Ambiguous` naming the module's class. Its invocation is part of its code: an instance's own
    `forward` is none, and its class's `__call__` (transformers' `GradientCheckpointingLayer`'s, or
    torch's own) is the known class's."""
    cls = type(module)
    if 'forward' in vars(module):
        raise Ambiguous(f"{cls.__qualname__} is called through its instance's own forward")
    for role in roles:
        for part, known in _known_by_role().get(role, ()):
            renaming = matched_class(cls, known, whole=part.whole)
            if renaming is not None and _same_invocation(cls, known, renaming):
                return part, renaming
    raise Ambiguous(f"{cls.__module__}.{cls.__qualname__} computes no known {' or '.join(roles)} part by its "
                    "code")


def _same_invocation(candidate: type, known: type, renaming: Renaming) -> bool:
    """Whether calling a `candidate` runs what calling a `known` does around their forwards: the same
    `__call__`, or one of the same code."""
    ours, theirs = inspect.getattr_static(candidate, '__call__'), inspect.getattr_static(known, '__call__')
    return ours is theirs or (isinstance(ours, FunctionType) and isinstance(theirs, FunctionType)
                              and same_code(ours, theirs, renaming))


# --- the assembly ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Assembly:
    """A decoder assembled from known parts: its CausalTransformer fields and the canonical module path
    of each of the source's (`weight_path`)."""
    fields: DecoderFields
    names: Mapping[str, tuple[str, Mapping[str, str]]]
    """Each source module path that holds parameters: its canonical path, and its parameters' known
    names by their own."""
    packed: tuple[decoder_parts.Packed, ...] = ()
    """The stored tensors that hold several the path map reads, by their stored names: each experts
    module's fused gate and up projection and its down projection (`decoder_parts.FUSED_EXPERTS`'s)."""
    tied: tuple[str, str] = ('lm_head.weight', 'model.embed_tokens.weight')
    """The head's and the table's stored names, which a tied checkpoint stores two copies of."""
    dropouts: tuple[tuple[str, str], ...] = ()
    """Each attention module's path and the attribute its code reads its dropout rate from, which a
    deterministic training forward sets to zero (`dew.interop.verify`'s training qualification)."""
    loaded: Callable[[str], tuple[str, tuple[str, str] | None]] = lambda name: (name, None)
    """A stored tensor's name as the source's load names it, and for one expert's own tensor, which the
    load stacks into its experts module's fused one, the expert's index and canonical projection
    (`_load_names`). It reads stored names alone: a name the model holds (a gradient's) is stored first
    as `save_pretrained` stores it (`dew.interop.verify.gradient_parity`)."""
    depths: int = 0
    """The prediction depths the source's release stores past the trunk (its config's
    `num_nextn_predict_layers`: GLM-4.7-Flash's `model.layers.47.*`, HY-3's `.80.*`), which its own code
    builds none of and its load leaves unread (`family`)."""

    def canonical(self, name: str) -> str | None:
        """A stored tensor's canonical name: its module's canonical path and its parameter's known
        name, a part a fused tensor splits into (`_SPLIT`) by its own, one expert's tensor of a
        checkpoint stored per expert by its index and projection; None where no part holds it."""
        loaded, expert = self.loaded(name)
        module, _, parameter = loaded.rpartition('.')
        held = self.names.get(module)
        if held is None:
            return None
        path, parameters = held
        if parameter.startswith(_SPLIT):
            return f'{path}.{parameter.removeprefix(_SPLIT)}'
        if parameter not in parameters:
            return None
        if expert is not None:
            index, projection = expert
            return f'{path}.{index}.{projection}.weight'
        return f'{path}.{parameters[parameter]}'


_PER_EXPERT = {'gate_up_proj': ('gate_proj', 'up_proj'), 'down_proj': ('down_proj',)}
"""Each fused experts tensor (`_fused_experts`) by the canonical projections a checkpoint stored per expert
holds it as, in the order the fused one concatenates them."""


def _load_names(model: PreTrainedModel) -> Callable[[str], tuple[str, tuple[str, str] | None]]:
    """The name `model`'s load gives a stored tensor (`sources.load_names`: HY-3's `mlp.expert_bias` is its
    `mlp.e_score_correction_bias`), an expert's own tensor, `{experts}.{index}.{stored}.weight`, named as
    the fused tensor the load stacks it into, with its index and its canonical projection, the one at its
    place among the merge's sources (Mixtral's `w1`, `w3` and `w2` are the gate, up and down projections).
    A merge is read only where it stacks experts on the first axis and concatenates its sources in
    `_PER_EXPERT`'s order on the second, as the fused tensor `_fused_experts` splits holds them; any other
    conversion's tensor keeps its stored name, which no part holds. A mapping that renames a key by
    whether another is stored is `Ambiguous`: no read of one name tells it."""
    from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter

    names = sources.load_names(model)
    if names is None:
        raise Ambiguous("its checkpoint conversion renames a key by whether another is stored")
    prefix = model.base_model_prefix

    def stacked(converter: WeightConverter, fused: str) -> tuple[str, ...]:
        """The canonical projections of `converter`'s sources, in their order, where it stacks them into
        `fused` as `_fused_experts` splits it; none otherwise."""
        projections = _PER_EXPERT.get(fused, ())
        operations = [(type(operation), operation.dim) for operation in converter.operations
                      if isinstance(operation, (MergeModulelist, Concatenate))]
        merges = [(MergeModulelist, 0)] + [(Concatenate, 1)] * (len(projections) > 1)
        if (len(converter.source_patterns) != len(projections) or len(operations) != len(converter.operations)
                or operations != merges):
            return ()
        return projections

    def loaded(name: str) -> tuple[str, tuple[str, str] | None]:
        read = names(name)
        if read.converter is None or read.pattern is None:
            return read.name, None
        module, _, fused = read.name.rpartition('.')
        spellings = (module, module.removeprefix(f'{prefix}.'), f'{prefix}.{module}')
        tails = [read.renamed.removeprefix(f'{spelled}.') for spelled in spellings
                 if read.renamed.startswith(f'{spelled}.')]
        index, _, own = tails[0].removesuffix('.weight').partition('.') if tails else ('', '', '')
        projections = stacked(read.converter, fused)
        if not (index.isdigit() and own and '.' not in own and tails[0] == f'{index}.{own}.weight'
                and projections):
            return name, None
        return read.name, (index, projections[read.converter.source_patterns.index(read.pattern)])

    return loaded



class _Assembler:
    def __init__(self):
        self.fields: dict[str, object] = {}
        self.layers: dict[int, dict[str, object]] = {}
        self.names: dict[str, tuple[str, Mapping[str, str]]] = {}
        self.tables: dict[str, tuple[str, object]] = {}
        self.packed: list[decoder_parts.Packed] = []
        self.dropouts: list[tuple[str, str]] = []

    def state(self, fields: Mapping[str, object], where: str, layer: int | None) -> None:
        for key, value in fields.items():
            if key == _LAYER:
                if layer is None:
                    raise Ambiguous(f"{where} routes outside a layer")
                self.layers.setdefault(layer, {}).update(_records.record(value, _LAYER))
            elif key in self.fields and self.fields[key] != value:
                raise Ambiguous(f"{where} states {key}={value!r} where another part states "
                                f"{self.fields[key]!r}")
            else:
                self.fields[key] = value

    def walk(self, module: torch.nn.Module, roles: tuple[str, ...], path: str, canonical: str,
             layer: int | None) -> str:
        """Identify `module` (at `path`, canonically `canonical`), read it and its children; return its
        role."""
        import torch

        if roles == ('layers',):
            if not isinstance(module, torch.nn.ModuleList):
                raise Ambiguous(f"{path} holds its layers in a {type(module).__name__}")
            for index, child in enumerate(module):
                self.walk(child, ('layer',), f'{path}.{index}', f'{canonical}.{index}', index)
            return 'layers'
        part, renaming = identify(module, roles)
        view = PartView(module, renaming)
        self.state(part.read(view), path or type(module).__name__, layer)
        # The head is tied where it is the table's own parameter, as the source holds it.
        if part.role == 'embedding' or (part.role == 'linear' and canonical == 'lm_head'):
            table = 'embedding' if part.role == 'embedding' else 'head'
            self.tables[table] = (f'{path}.{view.name("weight")}', view.weight)
        if part.role == 'experts':
            self.packed.extend(_fused_experts(path.rpartition('.')[2], view))
        if part.role == 'attention':
            self.dropouts.append((path, view.name('attention_dropout')))
        held = {name for name, _ in module.named_parameters(recurse=False)}
        expected = {view.name(name): name for name in part.parameters}
        if held - set(expected):
            raise Ambiguous(f"{path} holds parameters {sorted(held - set(expected))} its code does not read")
        stored = {view.name(name): known for name, known in part.state.items()
                  if view.name(name) in dict(module.named_buffers(recurse=False))}
        if held or stored:
            self.names[path] = (canonical, {**{name: expected[name] for name in held}, **stored})
        children = dict(module.named_children())
        claimed = set()
        for known, child_roles in part.children.items():
            name = view.name(known)
            claimed.add(name)
            child = children.get(name)
            if child is None:
                continue
            role = self.walk(child, child_roles, f'{path}.{name}' if path else name,
                             f'{canonical}.{known}' if canonical else known, layer)
            if part.role == 'layer' and known == 'mlp' and layer is not None:
                self.layers.setdefault(layer, {})['routed'] = role == 'moe'
        unclaimed = [name for name, child in children.items() if name not in claimed
                     and any(True for _ in child.parameters())]
        if unclaimed:
            raise Ambiguous(f"{path or type(module).__name__} holds {unclaimed}, which its code does not "
                            "call")
        return part.role


def assemble(model: PreTrainedModel) -> Assembly:
    """`model`, a source's causal LM, assembled from known parts, or `Ambiguous` naming the module that
    is not one."""
    assembler = _Assembler()
    assembler.walk(model, ('causal_lm',), '', '', None)
    (head, head_weight), (table, table_weight) = assembler.tables['head'], assembler.tables['embedding']
    fields = {**assembler.fields, 'tie_embeddings': head_weight is table_weight}
    routed = sorted(layer for layer, facts in assembler.layers.items() if facts.get('routed'))
    if routed:
        routes = {tuple(sorted((key, value) for key, value in assembler.layers[layer].items()
                               if key not in ('routed', 'nope'))) for layer in routed}
        if len(routes) != 1:
            raise Ambiguous("the routed layers route differently")
        mixture = dict(routes.pop())
        width = _records.integer(mixture['expert_features'], 'expert_features')
        # A router's scores, scaling and correction bias and a block's shared experts, where its parts
        # state them (DeepSeek's routing, HY-3's).
        stated = {key: mixture[key] for key in ('score_function', 'scaling', 'bias', 'shared_features',
                                                'groups', 'groups_per_token') if key in mixture}
        fields['mixture'] = {**native_fields(Mixture)(
            experts=_records.integer(mixture['experts'], 'experts'),
            top_k=_records.integer(mixture['top_k'], 'top_k'),
            norm_topk_prob=bool(mixture['norm_topk_prob']), expert_features=width,
            layers=None if len(routed) == fields['num_layers'] else tuple(routed)), **stated}
        fields.setdefault('mlp_features', width)
    for required in ('vocab_size', 'num_layers', 'num_heads', 'mlp', 'mlp_features', 'norm_eps',
                     'rope_theta'):
        if required not in fields:
            raise Ambiguous(f"no part states {required}")
    unrotated = sorted(layer for layer, facts in assembler.layers.items() if facts.get('nope'))
    fields.update(_without_positions(fields, unrotated))
    capacity, context = decoder_parts.context_lengths(
        {'max_position_embeddings': model.config.to_dict().get('max_position_embeddings')})
    fields.update(max_seq_len=capacity, context_length=context)
    kinds = fields.pop('kinds', {})
    record = native_fields(CausalTransformer)(**fields)
    record['kinds'] = kinds
    depths = _records.integer(model.config.to_dict().get('num_nextn_predict_layers') or 0,
                              'num_nextn_predict_layers')
    return Assembly(record, assembler.names, tuple(dict.fromkeys(assembler.packed)), (head, table),
                    tuple(assembler.dropouts), _load_names(model), depths)


def _without_positions(fields: Mapping[str, object], layers: list[int]) -> Mapping[str, object]:
    """The layer types and kinds that make each of `layers` (the ones whose attention rotates nothing,
    SmolLM3's NoPE layers) a kind of its own beside its layer type's, its attention taking no positions,
    as Llama 4's NoPE layers do."""
    if not layers:
        return {}
    count = _records.integer(fields['num_layers'], 'num_layers')
    stated_types = fields.get('layer_types')
    types = (list(_records.strings(stated_types, 'layer_types')) if stated_types
             else ['full_attention'] * count)
    kinds = dict(_records.record(fields.get('kinds', {}), 'kinds'))
    for layer in layers:
        stated = types[layer]
        unrotated = f'{stated}_nope'
        if unrotated not in kinds:
            kind = dict(_records.record(kinds.get(stated) or native_fields(LayerKind)(), stated))
            kind['mixer'] = {'class': 'attention', 'fields': {'nope': True}}
            kinds[unrotated] = kind
        types[layer] = unrotated
    return {'layer_types': tuple(types), 'kinds': kinds}


def _fused_experts(module: str, view: PartView) -> tuple[decoder_parts.Packed, ...]:
    """An experts module's fused tensors by their stored names (`module`, its attribute; the view, its
    parameters'), split into the parts the path map reads as `decoder_parts.FUSED_EXPERTS` splits
    Mixtral's."""
    fused, down = view.name('gate_up_proj'), view.name('down_proj')
    halves = (f'.{module}.{_SPLIT}gate_proj', f'.{module}.{_SPLIT}up_proj')
    return (decoder_parts.Packed(f'.{module}.{fused}', halves, -1, (0, 2, 1)),
            decoder_parts.Packed(f'.{module}.{down}', (f'.{module}.{_SPLIT}down_proj',), -1, (0, 2, 1)))


_SPLIT = '~'
"""The mark of a part a fused tensor splits into (`_fused_experts`), already canonical: no Python
attribute holds it, so it names no stored parameter, whatever the source renames its own to."""


_STACKED_EXPERTS: decoder_parts.Leaves = {
    f'mlp.experts.{name}': ('mlp', 'experts', name, 'kernel') for name in decoder_parts.MOE_SHARED}
"""The canonical experts, each projection one tensor over every expert (`decoder_parts.FUSED_EXPERTS`)."""


_PREDICTION_OWN = frozenset({'eh_proj.weight', 'enorm.weight', 'hnorm.weight', 'shared_head.norm.weight',
                             'final_layernorm.weight'})
"""A release's prediction depth's own tensors, by their names past `model.layers.{num_layers + d}.`
(DeepSeek V3's layout, GLM-4.7-Flash's; HY-3 norms its output as `final_layernorm`), which no part of the
trunk holds."""


def family(model_type: str, assembly: Assembly) -> decoder_parts.DecoderFamily:
    """The decoder family a type verified from parts loads through (`dew.interop.verify`): `assembly`'s
    fields for its config, every field of which the source's own code read, and each stored tensor at
    the path of its canonical name (`Assembly.canonical`), a fused expert tensor split as Mixtral's
    first (`DecoderFamily.packed`), so an export writes each back under its stored name. A tensor of one
    of the release's prediction depths past the trunk (`Assembly.depths`, by `decoder_parts.layer_depth`)
    fills no leaf: the source's own code builds no depth and its load leaves them unread, so the decoder
    assembled from that code has none (`prediction_depths` trims them before any is read). Only a depth's
    own tensors (`_PREDICTION_OWN`) and its block's, a name a trunk layer's part holds, are so; its
    copies of the trunk's table and head are the trunk's (`hf_decoders` checks each against it, and an
    export writes it from the trunk's leaf)."""
    def translated(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
        used.update(hf_config)
        return assembly.fields.copy()

    def weight_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
        depth = decoder_parts.layer_depth(name, config)
        if depth is not None and depth < assembly.depths:
            tail = name.split('.', 3)[3]
            head, table = assembly.tied
            if tail == 'shared_head.head.weight':
                return weight_path(table if config['tie_embeddings'] else head, config)
            if tail == 'embed_tokens.weight':
                return weight_path(table, config)
            last = _records.integer(config['num_layers'], 'num_layers') - 1
            if tail in _PREDICTION_OWN or assembly.canonical(f'model.layers.{last}.{tail}') is not None:
                return None
        canonical = assembly.canonical(name)
        if canonical is None:
            raise ValueError(f"the checkpoint's {name!r} is held by no part of the assembled decoder")
        return decoder_parts.renamed_path((), canonical, config, layer_leaves=_STACKED_EXPERTS)

    return decoder_parts.DecoderFamily(
        (model_type,), translated, lambda fields: False, model_type, 'ForCausalLM', lambda model: {},
        preserve_source_layout=True, weight_path=weight_path, packed=assembly.packed,
        tied_head_names=assembly.tied, prediction_depth=decoder_parts.layer_depth)
