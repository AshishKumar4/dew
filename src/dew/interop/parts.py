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
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from functools import cache
from types import FunctionType, MethodType
from typing import TYPE_CHECKING

from dew import _records
from dew.interop import decoder_parts, sources
from dew.interop.code_identity import Renaming, matched_class, same_code
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import ACTIVATIONS, PLAIN_SPELLING, DecoderFields
from dew.interop.unread import Stored, held
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture, NormSpec
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mla import MultiHeadLatentAttention

if TYPE_CHECKING:
    import torch
    from transformers import PretrainedConfig, PreTrainedModel


# --- the instances, read through the known names --------------------------------------------------

class PartView:
    """A source module read by the known class's attribute names (`Renaming.attributes`), beside the
    config of the causal LM it is part of (`source_config`), which a part whose code reads its values
    through its model rather than holding them reads (Falcon-H1's mixer, whose model builds its in
    projection's multipliers from `ssm_multipliers`)."""

    def __init__(self, module: torch.nn.Module, renaming: Renaming, source_config: PretrainedConfig):
        self.module, self.renaming, self.source_config = module, renaming, source_config

    def name(self, known: str) -> str:
        return self.renaming.attributes.get(known, known)

    def __getattr__(self, known: str):
        return getattr(self.module, self.name(known))


class Ambiguous(ValueError):
    """A part the source's code does not let tier 2 read, which leaves the decoder to tier 3."""


def gated_activation(module: torch.nn.Module | Callable[..., object]) -> str:
    """The gated MLP Dew computes for the activation module a source part holds: the `hidden_act` name
    transformers' ACT2FN builds that same module for, read back by its class and state, or for torch's
    own function that computes it (LFM2-MoE's experts hold `F.silu`)."""
    import torch
    from transformers.activations import ACT2FN

    functions = {torch.nn.functional.silu: 'silu', torch.nn.functional.gelu: 'gelu'}
    if not isinstance(module, torch.nn.Module):
        if module not in functions:
            raise Ambiguous(f"the activation {module!r} is none of torch's {sorted(functions.values())}")
        return ACTIVATIONS[functions[module]]

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
_SITE = '_site'
"""A per-layer fact of the module a part sits under in its layer (`model.layers.N.<site>`), filed under that
site's canonical name (`_Assembler.sites`): A.X-K2's gated norms, each its own site's."""


def _read_causal_lm(view: PartView) -> Mapping[str, object]:
    """Llama's causal LM: its head a Linear, which adds its bias where it holds one (ERNIE 4.5's
    `use_bias`), as Dew's head adds `head_bias` to the logits."""
    return {'head_bias': view.lm_head.bias is not None}


def _read_scaled_causal_lm(view: PartView) -> Mapping[str, object]:
    """Granite's head: Llama's, its logits divided by the config's `logits_scaling`."""
    return {**_read_causal_lm(view), 'logits_scaling': float(view.config.logits_scaling)}


def _read_scaled_model(view: PartView) -> Mapping[str, object]:
    """Granite's model: Llama's, its embeddings times its `embedding_multiplier`."""
    return {**_read_llama_model(view), 'embedding_multiplier': float(view.embedding_multiplier)}


def _read_multiplied_causal_lm(view: PartView) -> Mapping[str, object]:
    """Falcon-H1's causal LM: Llama's, its logits times its model's `lm_head_multiplier`, which Dew's head
    reads as dividing its input by the reciprocal (`logits_scaling`, in at least float32)."""
    return {**_read_causal_lm(view), 'logits_scaling': 1.0 / float(view.model.lm_head_multiplier)}


def _read_parallel_hybrid_layer(view: PartView) -> Mapping[str, object]:
    """Falcon-H1's layer: one norm's output read side by side by its Mamba-2 mixer (`mamba`, whose code
    first scales its input by its `ssm_in_multiplier`) and its attention (`self_attn`, its input times
    `attention_in_multiplier`), their outputs times `ssm_out_multiplier` and `attention_out_multiplier`
    summed in that order into one residual add, then Llama's feed-forward: a per-layer fact
    (`_Assembler.parallel`) of each branch's name and its input's and its output's multipliers
    (`dew.nn.mixers.parallel.ParallelBranch`), by the child that runs it."""
    return {_LAYER: {'parallel': {
        'mamba': {'name': 'mamba', 'input_multiplier': float(view.mamba.ssm_in_multiplier),
                  'output_multiplier': float(view.ssm_out_multiplier)},
        'self_attn': {'name': 'attention', 'input_multiplier': float(view.attention_in_multiplier),
                      'output_multiplier': float(view.attn_out_multiplier)}}}}


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


def _read_typed_rotary_model(view: PartView) -> Mapping[str, object]:
    """Laguna's model: each layer attends under its config layer type's mask (Qwen2's,
    `_read_layer_typed_model`) and rotates by its type's own rotary, which its rotary part (compared
    whole, `_read_nothing`) builds from the type's record of `rope_parameters`: its base, its rope_type's
    ramp and the fraction of each head it turns, its frequencies over that fraction
    (`decoder_parts.partial_rope_entry`, the 'default' partial convention). Each type's kind takes its
    own rotary; the model's base is its first layer's."""
    fields = dict(_read_layer_typed_model(view))
    types = _records.strings(fields['layer_types'], 'layer_types')
    kinds = dict(_records.record(fields.get('kinds', {}), 'kinds'))
    records = _records.record(view.config.rope_parameters or {}, 'rope_parameters')
    stated = view.config.max_position_embeddings
    bases: dict[str, float] = {}
    for name in dict.fromkeys(types):
        try:
            theta, factor, ramp = decoder_parts.partial_rope_entry(
                _records.record(records.get(name), f'rope_parameters {name}'), f'rope_parameters.{name}',
                default_factor=1.0,
                max_pos=None if stated is None else _records.integer(stated, 'max_position_embeddings'))
        except ValueError as error:
            raise Ambiguous(f"the {name} layers' rotary record is not one this rotation reads ({error})") \
                from error
        kind = dict(_records.record(kinds.get(name) or native_fields(LayerKind)(), name))
        kind.update(rope_theta=theta, rope_scaling=decoder_parts.nested_ramp(ramp),
                    partial_rotary_factor=float(factor))
        kinds[name], bases[name] = kind, theta
    return {**fields, 'kinds': kinds, 'rope_theta': bases[types[0]], 'partial_rotary_type': 'default'}


def _head_geometry(view: PartView) -> dict[str, int]:
    """A causal attention's heads, key and value heads and head width, read off its query and key
    projections and its `head_dim`, each a whole number of heads."""
    width = view.head_dim
    if not view.is_causal:
        raise Ambiguous("the attention is not causal")
    heads, kv_heads = view.q_proj.out_features // width, view.k_proj.out_features // width
    if heads * width != view.q_proj.out_features or kv_heads * width != view.k_proj.out_features:
        raise Ambiguous("the projections hold no whole number of heads")
    return {'num_heads': heads, 'num_kv_heads': kv_heads, 'head_dim': width}


def _read_attention(view: PartView) -> Mapping[str, object]:
    geometry = _head_geometry(view)
    width = geometry['head_dim']
    biased = _biased(view.q_proj, view.k_proj, view.v_proj)
    fields: dict[str, object] = {
        **geometry, 'qk_norm': False, 'attention_bias': biased,
        'attention_dropout_rate': float(view.attention_dropout)}
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


def _read_laguna_attention(view: PartView) -> Mapping[str, object]:
    """Laguna's attention: Qwen3's, each head's queries and keys normed, its heads its layer's own
    (`num_attention_heads_per_layer`, a per-layer fact `_with_head_counts` gives its type's kind), and
    its output gated by the softplus, in at least float32, of a projection of its own of the layer's
    input (`g_proj`, Dew's `gate_proj`): one gate a head where `gating` is per head
    (`gate_projection='head'`), else one a channel ('own'); a per-layer fact (`_with_attention_fields`)."""
    fields = dict(_read_qk_norm_attention(view))
    heads = _records.integer(fields.pop('num_heads'), 'num_heads')
    per_head = bool(view.gate_per_head)
    gates = heads if per_head else heads * _records.integer(fields['head_dim'], 'head_dim')
    if view.g_proj.out_features != gates or view.g_proj.bias is not None:
        raise Ambiguous(f"the output gate's projection holds {view.g_proj.out_features} unbiased gates where "
                        f"its code reads {gates}")
    return {**fields, 'output_gate': True, 'output_gate_activation': 'softplus',
            _LAYER: {'heads': heads, 'attention': {'gate_projection': 'head' if per_head else 'own'}}}


def _read_keyed_attention(view: PartView) -> Mapping[str, object]:
    """Falcon-H1's attention: Llama's, its keys times `key_multiplier` as projected, before their rotation
    and cache (`key_multiplier`)."""
    return {**_read_attention(view), 'key_multiplier': float(view.key_multiplier)}


def _read_nope_attention(view: PartView) -> Mapping[str, object]:
    """SmolLM3's attention: Llama's, rotating its queries and keys only where `use_rope` (each fourth layer
    attends without positions, `no_rope_layers`), a per-layer fact `assemble` makes a kind of."""
    return {**_read_attention(view), _LAYER: {'nope': not bool(view.use_rope)}}


def _read_unrotated_attention(view: PartView) -> Mapping[str, object]:
    """Jamba's attention: Llama's with no rotary at all, its queries and keys placed by nothing (each of
    its layers a kind without positions, `_without_positions`)."""
    return {**_read_attention(view), _LAYER: {'nope': True}}


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


def _read_sandwich_layer(view: PartView) -> Mapping[str, object]:
    """GLM-4's layer: Gemma 2's sandwich, each sublayer's input and output normed (`sandwich_norms`),
    its norms renamed onto Gemma 2's (`post_self_attn_layernorm` the attention's output norm,
    `post_attention_layernorm` the feed-forward's input norm, `post_mlp_layernorm` its output norm)."""
    return {'sandwich_norms': True}


def _read_output_normed_layer(view: PartView) -> Mapping[str, object]:
    """OLMo 2's layer: each sublayer's output normed before it joins the residual, its input not
    (`sandwich_norms` without `pre_norms`, OLMo 3's)."""
    return {'sandwich_norms': True, 'pre_norms': False}


def _read_scale_first_norm(view: PartView) -> Mapping[str, object]:
    """OLMo 2's RMSNorm: its weight times the normalized input in float32, cast after."""
    return {'norm_eps': float(view.variance_epsilon), 'scale_after_cast': False, 'scale_offset': False}


def _read_lfm2_attention(view: PartView) -> Mapping[str, object]:
    """LFM2's attention: Qwen3's, its queries and keys normed per head (`q_layernorm`, `k_layernorm`), its
    output projection `out_proj`, and no dropout, in training as in evaluation."""
    geometry = _head_geometry(view)
    unscaled = view.scaling != geometry['head_dim'] ** -0.5
    if _biased(view.q_proj, view.k_proj, view.v_proj, view.out_proj) or unscaled:
        raise Ambiguous("the attention biases its projections or scales by another than its head width")
    return {**geometry, 'qk_norm': True, 'attention_bias': False, 'attention_dropout_rate': 0.0}


def _read_lfm2_model(view: PartView) -> Mapping[str, object]:
    """LFM2's model: Llama's over its layers, the head geometry its config states for its attention
    layers and its rotary (`num_attention_heads`, `num_key_value_heads`, `head_dim` or the width over the
    heads), which a model of convolution layers alone states through no attention."""
    config = view.config
    heads = _records.integer(config.num_attention_heads, 'num_attention_heads')
    width = config.to_dict().get('head_dim') or _records.integer(config.hidden_size, 'hidden_size') // heads
    return {**_read_llama_model(view), 'num_heads': heads, 'head_dim': _records.integer(width, 'head_dim'),
            'num_kv_heads': _records.integer(config.num_key_value_heads, 'num_key_value_heads')}


def _read_lfm2_mlp(view: PartView) -> Mapping[str, object]:
    """LFM2's feed-forward: `w2(silu(w1(x)) * w3(x))`, Llama's SwiGLU under its own names."""
    return {'mlp': 'swiglu', 'mlp_features': view.w1.out_features,
            'mlp_bias': _biased(view.w1, view.w3, view.w2)}


def _read_short_conv(view: PartView) -> Mapping[str, object]:
    """LFM2's gated short convolution (`dew.nn.mixers.short_conv.GatedShortConv`): `in_proj` to B, C and x,
    the depthwise causal convolution of B * x over `conv_kernel_size` taps, gated by C, and `out_proj`
    back; a per-layer fact (`_with_mixers`). Its `bias` biases all three together, a left-padded
    prompt's slots holding their bias product as its zeroed inputs do, its code masking no product
    (`padding_holds_bias`); one biasing some of them alone is `Ambiguous`."""
    conv, width = view.conv, view.in_proj.in_features
    taps = int(view.conv_kernel_size)
    if (conv.groups != width or conv.kernel_size != (taps,) or conv.padding != (taps - 1,)
            or view.in_proj.out_features != 3 * width):
        raise Ambiguous("the convolution is not one causal depthwise filter per channel over its taps")
    biased = _biased(view.in_proj, view.out_proj, conv)
    return {_LAYER: {'mixer': {'kind': 'conv', 'class': 'short_conv', 'fields': {
        'conv_L_cache': taps, 'conv_bias': biased, 'padding_holds_bias': biased}}}}


def _read_step_normed_mamba(view: PartView) -> Mapping[str, object]:
    """Jamba's Mamba mixer (`dew.nn.mixers.mamba.Mamba`): `in_proj` to the inner channels and their gate,
    a causal depthwise convolution activated by silu, `x_proj` to the raw step, B and C, each RMS-normed
    (`dt_layernorm`, `b_layernorm`, `c_layernorm`, read as the block's norms are, `step_norms`), the step
    through `dt_proj` and its bias, the selective scan, its silu gate and `out_proj`; a per-layer fact
    (`_with_mixers`)."""
    conv, inner = view.conv1d, int(view.intermediate_size)
    taps = int(view.conv_kernel_size)
    if (view.activation != 'silu' or conv.groups != inner or conv.in_channels != inner
            or conv.kernel_size != (taps,) or conv.padding != (taps - 1,)):
        raise Ambiguous("the convolution is not one causal depthwise filter per channel activated by silu")
    if view.x_proj.bias is not None or view.dt_proj.bias is None:
        raise Ambiguous("the selective projection is biased or the step's projection is not")
    return {_LAYER: {'mixer': {'kind': 'mamba', 'class': 'mamba', 'fields': {
        'intermediate_size': inner, 'state_size': int(view.ssm_state_size),
        'time_step_rank': int(view.time_step_rank), 'conv_kernel': taps,
        'use_bias': _biased(view.in_proj, view.out_proj), 'use_conv_bias': conv.bias is not None,
        'step_norms': True}}}}


def _read_mlp(view: PartView) -> Mapping[str, object]:
    return {'mlp': gated_activation(view.act_fn), 'mlp_features': view.gate_proj.out_features,
            'mlp_bias': _biased(view.gate_proj, view.up_proj, view.down_proj)}


def _read_multiplied_mlp(view: PartView) -> Mapping[str, object]:
    """Falcon-H1's MLP: Llama's, its gate projection's output times the first of its `mlp_multipliers`
    before the activation and its down projection's output times the second (`mlp_multipliers`)."""
    multipliers = (float(view.gate_multiplier), float(view.down_multiplier))
    return {**_read_mlp(view), **({'mlp_multipliers': multipliers} if multipliers != (1.0, 1.0) else {})}


def _read_sectioned_mamba2(view: PartView) -> Mapping[str, object]:
    """Falcon-H1's Mamba-2 mixer (`dew.nn.mixers.mamba2.Mamba2`): its in projection's output times its
    `mup_vector`, one factor a section of z, x, B, C and dt (`section_multipliers`), Mamba-2's causal
    depthwise convolution activated by silu and chunked scan, the scan's output gated by z under its
    gated norm, in the order its `norm_before_gate` states and its weight applied in float32
    (`norm_order`, `norm_scale_wide`), or with no norm where `mamba_rms_norm` is off, and its out
    projection biased apart (`projectors_bias`, `out_bias`); a per-layer fact (`_with_mixers`). The
    multiplier its input takes first is its layer's branch's (`_read_parallel_hybrid_layer`)."""
    heads, head_dim, inner = int(view.num_heads), int(view.head_dim), int(view.intermediate_size)
    groups, state, taps = int(view.n_groups), int(view.ssm_state_size), int(view.conv_kernel_size)
    conv, width = view.conv1d, inner + 2 * groups * state
    if heads * head_dim != inner:
        raise Ambiguous(f"its {heads} heads of {head_dim} do not tile its {inner} inner channels")
    if (view.activation != 'silu' or conv.groups != width or conv.in_channels != width
            or conv.kernel_size != (taps,) or conv.padding != (taps - 1,)):
        raise Ambiguous("the convolution is not one causal depthwise filter per channel activated by silu")
    # Its `mup_vector`, a buffer its model builds from the config's five `ssm_multipliers`, one a
    # section (`compute_mup_vector`), read off that config: an unloaded model's buffers hold no values.
    sections = _records.numbers(list(view.source_config.ssm_multipliers), 'ssm_multipliers')
    if len(sections) != 5:
        raise Ambiguous(f"its ssm_multipliers {sections} are not one factor a section of z, x, B, C "
                        "and dt")
    use_bias = view.in_proj.bias is not None
    fields: dict[str, object] = {
        'num_heads': heads, 'head_dim': head_dim, 'state_size': state, 'n_groups': groups,
        'conv_kernel': taps, 'chunk_size': int(view.chunk_size), 'use_bias': use_bias,
        'use_conv_bias': conv.bias is not None,
        'time_step_limit': tuple(float(bound) for bound in view.time_step_limit),
        'section_multipliers': tuple(sections),
        'out_bias': None if (view.out_proj.bias is not None) == use_bias else view.out_proj.bias is not None}
    stated: dict[str, object] = {}
    if view.mamba_rms_norm:
        norm = view.norm
        fields.update(norm_order='norm_first' if norm.norm_before_gate else 'gate_first',
                      norm_scale_wide=True, norm_groups=int(norm.n_groups))
        # The gated norm's epsilon is the mixer's context's, the block norms' (`MixerContext.norm_eps`).
        stated['norm_eps'] = float(norm.variance_epsilon)
    else:
        fields['norm_order'] = None
    return {**stated, _LAYER: {'mixer': {'kind': 'mamba2', 'class': 'mamba2', 'fields': fields}}}


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


def _read_laguna_router(view: PartView) -> Mapping[str, object]:
    """Laguna's router: float32 logits, capped by `tanh(logit / c) * c` where its
    `router_logit_softcapping` is positive, their sigmoid scores chosen by score plus its correction
    bias (a parameter of its own), weighted by score renormalized over the chosen, nothing added to
    their sum."""
    cap = float(view.router_logit_softcapping)
    return {_LAYER: {'top_k': int(view.top_k), 'experts': int(view.num_experts), 'norm_topk_prob': True,
                     'score_function': 'sigmoid', 'bias': True, 'weight_sum_epsilon': 0.0,
                     **({'router_softcap': cap} if cap > 0 else {})}}


def _read_unnormalized_moe(view: PartView) -> Mapping[str, object]:
    """Jamba's MoE: an unbiased Linear router (`router`, Dew's `gate`), the softmax of its logits over every
    expert in float32, the `num_experts_per_tok` best weighted by their scores as they are, not
    renormalized."""
    if view.router.bias is not None:
        raise Ambiguous("the router's Linear is biased, which Dew's router is not")
    return {_LAYER: {'top_k': int(view.top_k), 'experts': int(view.num_experts), 'norm_topk_prob': False,
                     'score_function': 'softmax'}}


def _read_scaled_shared_moe(view: PartView) -> Mapping[str, object]:
    """Laguna's MoE: its routed experts' sum times `routed_scaling_factor` beside its shared experts
    (`_read_shared_moe`), which Dew's router folds into the chosen weights (`Mixture.scaling`)."""
    return {_LAYER: {'scaling': float(view.routed_scaling_factor)}}


def _read_glm4_attention(view: PartView) -> Mapping[str, object]:
    """GLM-4's attention: Llama's, its rotary turning adjacent pairs of the heads' rotated channels (its
    `apply_rotary_pos_emb` repeats each angle over a pair and rotates `(x[2i], x[2i+1])` over the first
    `rotary_dim` channels, the rest passed through) in the queries' dtype, which Dew's rotation, in at
    least float32 rounded once, holds within that dtype's rounding; a per-layer fact
    (`_with_attention_fields`)."""
    return {**_read_attention(view), _LAYER: {'attention': {'rotary_pairs': 'adjacent'}}}


def _read_fused_mlp(view: PartView) -> Mapping[str, object]:
    """GLM-4's MLP: Llama's with the gate and up projections one Linear (`gate_up_proj`, the gate's rows
    first, `chunk(2)`), read as the two (`KnownPart.split`)."""
    fused = view.gate_up_proj
    if fused.out_features % 2:
        raise Ambiguous(f"the fused gate and up projection holds {fused.out_features} rows, which split into "
                        "no equal gate and up halves, so no gated MLP of Dew's reads it")
    return {'mlp': gated_activation(view.activation_fn), 'mlp_features': fused.out_features // 2,
            'mlp_bias': _biased(fused, view.down_proj)}


def _read_xielu_mlp(view: PartView) -> Mapping[str, object]:
    """Apertus's MLP: ungated, `down_proj(act_fn(up_proj(x)))`, its activation xIELU with its learned
    parameters (`dew.nn.activations.Xielu`, Dew's `mlp='xielu'`)."""
    return {'mlp': 'xielu', 'mlp_features': view.up_proj.out_features,
            'mlp_bias': _biased(view.up_proj, view.down_proj)}


def _read_ernie_attention(view: PartView) -> Mapping[str, object]:
    """ERNIE 4.5's attention: Llama's, its rotary turning adjacent pairs of each head (its
    `apply_rotary_pos_emb` repeats each angle over a pair and rotates `(x[2i], x[2i+1])`, in float32,
    rounded once to the queries' dtype); a per-layer fact (`_with_attention_fields`)."""
    return {**_read_attention(view), _LAYER: {'attention': {'rotary_pairs': 'adjacent'}}}


def _read_hunyuan_attention(view: PartView) -> Mapping[str, object]:
    """Hunyuan's attention: Llama's, each head's queries and keys normed after their rotation
    (`query_layernorm`, `key_layernorm`); a per-layer fact (`_with_attention_fields`)."""
    return {**_read_attention(view), 'qk_norm': True, _LAYER: {'attention': {'qk_norm_after_rotary': True}}}


def _read_hunyuan_router(view: PartView) -> Mapping[str, object]:
    """Hunyuan's router: float32 logits (its `wg` held in float32) and their softmax, the `moe_topk` best
    weighted by score renormalized over the chosen."""
    return {_LAYER: {'top_k': int(view.top_k), 'experts': int(view.num_experts), 'norm_topk_prob': True,
                     'score_function': 'softmax'}}


def _read_alpha_rotary(view: PartView) -> Mapping[str, object]:
    """Hunyuan's rotary: where its rope record is `dynamic` with an `alpha`, plain rotary positions at
    `rope_theta * alpha ** (head_dim / (head_dim - 2))` (its DynamicNTKAlphaRotary), at every length.
    transformers recomputes dynamic-NTK frequencies from a batch's longest position past
    `max_position_embeddings`, which no serving stack reproduces: vLLM serves the static alpha base at
    every length, as Dew does. Any other record is the rotary `_rope` reads."""
    record = dict(view.config.rope_parameters or {})
    if record.get('rope_type') != 'dynamic' or not record.get('alpha'):
        return _rope(view.config)
    width = _records.integer(view.config.head_dim, 'head_dim')
    theta = _records.number(record['rope_theta'], 'rope_theta')
    return {'rope_theta': theta * _records.number(record['alpha'], 'alpha') ** (width / (width - 2))}


def _read_ernie_router(view: PartView) -> Mapping[str, object]:
    """ERNIE 4.5's router: float32 logits and their softmax over every expert, chosen by score plus the
    correction bias its `moe_statics` holds, weighted by score over the chosen scores' sum clamped below
    at `moe_norm_min` (`Router.weight_sum_min`), which binds where the bias picks experts whose scores
    sum to less."""
    return {_LAYER: {'top_k': int(view.top_k), 'experts': int(view.num_experts), 'norm_topk_prob': True,
                     'score_function': 'softmax', 'bias': True, 'weight_sum_epsilon': 0.0,
                     'weight_sum_min': float(view.norm_min)}}


def _read_ernie_moe(view: PartView) -> Mapping[str, object]:
    """ERNIE 4.5's MoE: its routed experts beside shared ones where `moe_num_shared_experts` (an MLP read
    as `shared`), summed, rounded to the input's dtype."""
    return {}


def _read_lfm2_moe_router(view: PartView) -> Mapping[str, object]:
    """LFM2-MoE's router: sigmoid scores, chosen by score plus the block's `expert_bias` where
    `use_expert_bias`, weighted by score, renormalized over the chosen where `norm_topk_prob` with 1e-6
    added to their sum (its code's constant), and times `routed_scaling_factor`."""
    return {_LAYER: {'top_k': int(view.top_k), 'experts': int(view.num_experts),
                     'norm_topk_prob': bool(view.norm_topk_prob), 'score_function': 'sigmoid',
                     'scaling': float(view.routed_scaling_factor), 'bias': bool(view.use_expert_bias),
                     'weight_sum_epsilon': 1e-6}}


def _read_grouped_router(view: PartView) -> Mapping[str, object]:
    """DeepSeek V3's router (GLM-4.7-Flash's): sigmoid scores in float32 plus its correction bias,
    experts chosen within the `topk_group` best of `num_group` groups (each scored by its two best, an
    expert outside them never chosen, its score masked to -inf), weighted by score, renormalized where
    `norm_topk_prob`, times `routed_scaling_factor`."""
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
    return _latent_attention(view, interleave=bool(view.config.rope_interleave))


def _read_gated_latent_attention(view: PartView) -> Mapping[str, object]:
    """A.X-K2's latent attention: DeepSeek V3's (`_latent_attention`) through its query latent, its rope
    heads always rotated in interleaved pairs, the latent's normed and raw values projected together to
    each head's query and a gate (`q_gate_proj`), the heads' output times that gate's sigmoid in float32
    before `o_proj` (`attn_gate_fused`), beside DeepSeek V3.2's lightning indexer (its own class, compared
    by its code), which selects `index_topk` keys a query, its rotary turning halves."""
    if view.q_lora_rank is None:
        raise Ambiguous("the fused query and gate project a query latent, which this attention lacks")
    fields = dict(_latent_attention(view, interleave=True))
    mixer = dict(_records.record(fields['mixer'], 'mixer'))
    indexer = view.indexer
    mixer['fields'] = {**_records.record(mixer['fields'], 'fields'), 'attn_gate_fused': True,
                       'index_topk': int(indexer.index_topk), 'index_n_heads': int(indexer.n_heads),
                       'index_head_dim': int(indexer.head_dim), 'index_rope_interleave': False}
    return {**fields, 'mixer': mixer}


def _read_indexer_norm(view: PartView) -> Mapping[str, object]:
    """The lightning indexer's key norm: torch's LayerNorm with its weight and bias at 1e-6, which Dew's
    indexer holds (`dew.nn.mla.SparseIndexer`)."""
    if not view.elementwise_affine or view.bias is None or float(view.eps) != 1e-6:
        raise Ambiguous("the indexer's key norm is not an affine LayerNorm at 1e-6")
    return {}


def _read_norm_gate(view: PartView) -> Mapping[str, object]:
    """A gated norm's gate (AXK2GateMLP, under AXK2GatedRMSNorm): `fc2(silu(fc1(y)))`, unbiased
    (`dew.nn.norms.NormGate`); a fact of its norm's site, its rank (`_with_norm_gates`)."""
    import torch

    if not isinstance(view.activation_fn, torch.nn.SiLU) or _biased(view.fc1, view.fc2):
        raise Ambiguous("the norm's gate is not an unbiased silu map")
    return {_SITE: {'gate_rank': int(view.fc1.out_features)}}


def _read_optionally_grouped_router(view: PartView) -> Mapping[str, object]:
    """A.X-K2's router: DeepSeek V3's (`_read_grouped_router`), which chooses within groups only where its
    config states `n_group`, an expert outside a token's best groups filled in at 0.0, not -inf
    (`excluded_group_score`, modeling_axk2.py:340), so a best group's experts biased below zero rank under
    it."""
    if view.num_group is not None:
        grouped = _records.record(_read_grouped_router(view)[_LAYER], _LAYER)
        return {_LAYER: {**grouped, 'excluded_group_score': 0.0}}
    return {_LAYER: {'top_k': int(view.top_k), 'experts': int(view.num_experts),
                     'norm_topk_prob': bool(view.norm_topk_prob), 'score_function': 'sigmoid',
                     'scaling': float(view.routed_scaling_factor), 'bias': True}}


def _latent_attention(view: PartView, *, interleave: bool) -> Mapping[str, object]:
    """A DeepSeek V3 latent attention's fields (`_read_latent_attention`), its rope heads rotated in
    interleaved pairs or halves."""
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
                'rope_interleave': interleave,
                # Llama's RMSNorm's rounding, the latent norms' (`_read_latent_norm`).
                'latent_scale_after_cast': True, 'latent_scale_offset': False}}}


def _read_shared_moe(view: PartView) -> Mapping[str, object]:
    """A routed block beside always-on shared experts (`shared_experts`, an MLP read as `shared`), the two
    summed: DeepSeek V3's, and HY-3's, whose float32 sum of two values of the compute dtype rounds as
    their sum there does (`enable_moe_fp32_combine`), so it reads no field."""
    return {}


def _read_shared_mlp(view: PartView) -> Mapping[str, object]:
    """A routed block's shared experts: Llama's MLP, its width the block's (`Mixture.shared_features`)."""
    width = int(view.gate_proj.out_features)
    if _biased(view.gate_proj, view.up_proj, view.down_proj):
        # ERNIE 4.5's `use_bias`: Dew's routed layers hold bias-free gated MLPs alone.
        raise Ambiguous("the shared experts are biased, which a routed layer's MLPs are not in Dew")
    return {'mlp': gated_activation(view.act_fn), _LAYER: {'shared_features': width}}


def _read_experts(view: PartView) -> Mapping[str, object]:
    return {'mlp': gated_activation(view.act_fn), _LAYER: {'expert_features': int(view.intermediate_dim)}}


def _read_rms_norm(view: PartView) -> Mapping[str, object]:
    return {'norm_eps': float(view.variance_epsilon), 'scale_after_cast': True, 'scale_offset': False}


def _read_latent_norm(view: PartView) -> Mapping[str, object]:
    """A latent attention's norm (Llama's RMSNorm, the one class the `latent_norm` role admits): its
    rounding the latent mixer's own (`_read_latent_attention` states `latent_scale_after_cast` and
    `latent_scale_offset` as this class computes), whatever the block norms' is (ledger 3114); its
    epsilon the attention's reader holds to the mixer's own."""
    return {}


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
    canonical: Mapping[str, str] = field(default_factory=dict)
    """Each child's canonical module name, by the known class's attribute name, where the two differ:
    LFM2's `operator_norm` is the shared `input_layernorm`, its `feed_forward.w1` the `mlp.gate_proj`;
    the empty name reads the child's parameters as the part's own (ERNIE 4.5's `moe_statics` holds its
    router's correction bias)."""
    flat: tuple[str, ...] = ()
    """The parameters the source stores with leading unit axes that its leaf holds flat (ERNIE 4.5's
    `[1, experts]` correction bias), by the known class's names (`Assembly.flat`)."""
    constants: tuple[str, ...] = ()
    """The buffers its checkpoint stores that its code reads and no step trains (xIELU's `beta` and
    `eps`), by the known class's names: each a constant of Dew's (`constants` collection,
    `Assembly.constants`)."""
    split: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """The children whose weight holds several canonical projections concatenated on its output rows,
    by the known class's attribute name, onto the projections in their order (GLM-4's `gate_up_proj`,
    the gate's rows first): each reads as its parts (`_fused_linear`)."""

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
_LFM2 = 'transformers.models.lfm2.modeling_lfm2'
_ERNIE = 'transformers.models.ernie4_5_moe.modeling_ernie4_5_moe'
_HUNYUAN = 'transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe'
_LFM2_MOE = 'transformers.models.lfm2_moe.modeling_lfm2_moe'
_DEEPSEEK_V3 = 'transformers.models.deepseek_v3.modeling_deepseek_v3'
_GLM4_MOE = 'transformers.models.glm4_moe.modeling_glm4_moe'
_GLM4 = 'transformers.models.glm4.modeling_glm4'
_APERTUS = 'transformers.models.apertus.modeling_apertus'
_LAGUNA = 'transformers.models.laguna.modeling_laguna'
_JAMBA = 'transformers.models.jamba.modeling_jamba'
_FALCON_H1 = 'transformers.models.falcon_h1.modeling_falcon_h1'
_AXK2 = 'transformers.models.axk2.modeling_axk2'
Children = Mapping[str, tuple[str, ...]]
_MODEL_CHILDREN: Children = {'embed_tokens': ('embedding',), 'layers': ('layers',), 'norm': ('norm',),
                             'rotary_emb': ('rotary',)}
_LAYER_CHILDREN: Children = {'self_attn': ('attention',), 'mlp': ('mlp', 'moe'),
                             **dict.fromkeys(('input_layernorm', 'post_attention_layernorm'),
                                             ('norm', 'gated_block_norm'))}
"""A Llama-shaped layer's children: its norms plain, or gated by a low-rank map of their own output
(A.X-K2's, whose layer computes Llama's)."""
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
    KnownPart('attention', f'{_HUNYUAN}:HunYuanMoEV1Attention',
              {**_ATTENTION_CHILDREN, 'query_layernorm': ('norm',), 'key_layernorm': ('norm',)},
              _read_hunyuan_attention, canonical={'query_layernorm': 'q_norm', 'key_layernorm': 'k_norm'}),
    KnownPart('moe', f'{_HUNYUAN}:HunYuanMoEV1Moe',
              {'gate': ('router',), 'experts': ('experts',), 'shared_mlp': ('shared',)}, _read_shared_moe,
              canonical={'shared_mlp': 'shared_experts'}),
    KnownPart('router', f'{_HUNYUAN}:HunYuanMoEV1Gate', {'wg': ('linear',)}, _read_hunyuan_router,
              canonical={'wg': ''}),
    KnownPart('rotary', f'{_HUNYUAN}:HunYuanMoEV1RotaryEmbedding', {}, _read_alpha_rotary, whole=True),
    KnownPart('model', f'{_ERNIE}:Ernie4_5_MoeModel', _MODEL_CHILDREN, _read_llama_model),
    KnownPart('rotary', f'{_ERNIE}:Ernie4_5_MoeRotaryEmbedding', {}, _read_rotary, whole=True),
    KnownPart('attention', f'{_ERNIE}:Ernie4_5_MoeAttention', _ATTENTION_CHILDREN, _read_ernie_attention),
    KnownPart('moe', f'{_ERNIE}:Ernie4_5_MoeSparseMoeBlock', _SHARED_MOE_CHILDREN, _read_ernie_moe),
    KnownPart('router', f'{_ERNIE}:Ernie4_5_MoeTopKRouter', {'moe_statics': ('statics',)}, _read_ernie_router,
              ('weight',), canonical={'moe_statics': ''}),
    KnownPart('statics', f'{_ERNIE}:Ernie4_5_MoeStatics', {}, _read_nothing, ('e_score_correction_bias',),
              flat=('e_score_correction_bias',)),
    KnownPart('model', f'{_LFM2}:Lfm2Model', {**_MODEL_CHILDREN, 'embedding_norm': ('norm',)},
              _read_lfm2_model, canonical={'embedding_norm': 'norm'}),
    KnownPart('model', f'{_LFM2_MOE}:Lfm2MoeModel',
              {'embed_tokens': ('embedding',), 'layers': ('layers',), 'embedding_norm': ('norm',),
               'pos_emb': ('rotary',)}, _read_lfm2_model,
              canonical={'embedding_norm': 'norm', 'pos_emb': 'rotary_emb'}),
    KnownPart('moe', f'{_LFM2_MOE}:Lfm2MoeSparseMoeBlock', {'gate': ('router',), 'experts': ('experts',)},
              _read_nothing, state={'expert_bias': 'gate.e_score_correction_bias'}),
    KnownPart('router', f'{_LFM2_MOE}:Lfm2MoeTopKRouter', {}, _read_lfm2_moe_router, ('weight',)),
    KnownPart('layer', f'{_LFM2}:Lfm2DecoderLayer',
              {'self_attn': ('attention',), 'conv': ('short_conv',), 'feed_forward': ('mlp', 'moe'),
               'operator_norm': ('norm',), 'ffn_norm': ('norm',)}, _read_nothing,
              canonical={'feed_forward': 'mlp', 'operator_norm': 'input_layernorm',
                         'ffn_norm': 'post_attention_layernorm'}),
    KnownPart('attention', f'{_LFM2}:Lfm2Attention',
              {**dict.fromkeys(('q_proj', 'k_proj', 'v_proj', 'out_proj'), ('linear',)),
               'q_layernorm': ('norm',), 'k_layernorm': ('norm',)}, _read_lfm2_attention,
              canonical={'out_proj': 'o_proj', 'q_layernorm': 'q_norm', 'k_layernorm': 'k_norm'}),
    KnownPart('mlp', f'{_LFM2}:Lfm2MLP', dict.fromkeys(('w1', 'w3', 'w2'), ('linear',)), _read_lfm2_mlp,
              canonical={'w1': 'gate_proj', 'w3': 'up_proj', 'w2': 'down_proj'}),
    KnownPart('short_conv', f'{_LFM2}:Lfm2ShortConv',
              {'in_proj': ('linear',), 'conv': ('conv1d',), 'out_proj': ('linear',)}, _read_short_conv),
    KnownPart('conv1d', 'torch.nn:Conv1d', {}, _read_nothing, ('weight', 'bias')),
    KnownPart('rotary', f'{_LFM2}:Lfm2RotaryEmbedding', {}, _read_rotary, whole=True),
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
    KnownPart('layer', f'{_GLM4}:Glm4DecoderLayer',
              {**_LAYER_CHILDREN, 'post_self_attn_layernorm': ('norm',), 'post_mlp_layernorm': ('norm',)},
              _read_sandwich_layer,
              canonical={'post_self_attn_layernorm': 'post_attention_layernorm',
                         'post_attention_layernorm': 'pre_feedforward_layernorm',
                         'post_mlp_layernorm': 'post_feedforward_layernorm'}),
    KnownPart('attention', f'{_GLM4}:Glm4Attention', _ATTENTION_CHILDREN, _read_glm4_attention),
    KnownPart('layer', f'{_APERTUS}:ApertusDecoderLayer',
              {'self_attn': ('attention',), 'mlp': ('mlp',), 'attention_layernorm': ('norm',),
               'feedforward_layernorm': ('norm',)}, _read_nothing,
              canonical={'attention_layernorm': 'input_layernorm',
                         'feedforward_layernorm': 'post_attention_layernorm'}),
    KnownPart('attention', f'{_APERTUS}:ApertusAttention',
              {**_ATTENTION_CHILDREN, 'q_norm': ('norm',), 'k_norm': ('norm',)}, _read_qk_norm_attention),
    KnownPart('mlp', f'{_APERTUS}:ApertusMLP',
              {'up_proj': ('linear',), 'down_proj': ('linear',), 'act_fn': ('xielu',)}, _read_xielu_mlp),
    KnownPart('xielu', 'transformers.activations:XIELUActivation', {}, _read_nothing, ('alpha_p', 'alpha_n'),
              constants=('beta', 'eps')),
    KnownPart('mlp', f'{_GLM4}:Glm4MLP', {'gate_up_proj': ('linear',), 'down_proj': ('linear',)},
              _read_fused_mlp, canonical={'gate_up_proj': ''},
              split={'gate_up_proj': ('gate_proj', 'up_proj')}),
    KnownPart('rotary', f'{_GLM4}:Glm4RotaryEmbedding', {}, _read_partial_rotary, whole=True),
    KnownPart('causal_lm', f'{_LAGUNA}:LagunaForCausalLM', {'model': ('model',), 'lm_head': ('linear',)},
              _read_causal_lm),
    KnownPart('model', f'{_LAGUNA}:LagunaModel', _MODEL_CHILDREN, _read_typed_rotary_model),
    KnownPart('rotary', f'{_LAGUNA}:LagunaRotaryEmbedding', {}, _read_nothing, whole=True),
    KnownPart('attention', f'{_LAGUNA}:LagunaAttention',
              {**_ATTENTION_CHILDREN, 'q_norm': ('norm',), 'k_norm': ('norm',), 'g_proj': ('linear',)},
              _read_laguna_attention, canonical={'g_proj': 'gate_proj'}),
    KnownPart('moe', f'{_LAGUNA}:LagunaSparseMoeBlock', _SHARED_MOE_CHILDREN, _read_scaled_shared_moe),
    KnownPart('router', f'{_LAGUNA}:LagunaTopKRouter', {}, _read_laguna_router,
              ('weight', 'e_score_correction_bias')),
    KnownPart('model', f'{_AXK2}:AXK2Model', _MODEL_CHILDREN, _read_llama_model),
    KnownPart('attention', f'{_AXK2}:AXK2Attention',
              {**dict.fromkeys(('q_a_proj', 'q_gate_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj'),
                               ('linear',)),
               **dict.fromkeys(('q_a_layernorm', 'kv_a_layernorm'), ('latent_norm',)),
               'indexer': ('indexer',)},
              _read_gated_latent_attention),
    KnownPart('indexer', f'{_AXK2}:AXK2Indexer',
              {**dict.fromkeys(('wq_b', 'wk', 'weights_proj'), ('linear',)), 'k_norm': ('indexer_norm',)},
              _read_nothing),
    KnownPart('indexer_norm', 'torch.nn:LayerNorm', {}, _read_indexer_norm, ('weight', 'bias')),
    KnownPart('gated_block_norm', f'{_AXK2}:AXK2GatedRMSNorm', {'norm': ('norm',), 'mlp': ('norm_gate',)},
              _read_nothing),
    KnownPart('norm_gate', f'{_AXK2}:AXK2GateMLP', {'fc1': ('linear',), 'fc2': ('linear',)}, _read_norm_gate),
    KnownPart('router', f'{_AXK2}:AXK2TopkRouter', {}, _read_optionally_grouped_router,
              ('weight',), state={'e_score_correction_bias': 'e_score_correction_bias'}),
    KnownPart('causal_lm', f'{_FALCON_H1}:FalconH1ForCausalLM', {'model': ('model',), 'lm_head': ('linear',)},
              _read_multiplied_causal_lm),
    KnownPart('model', f'{_FALCON_H1}:FalconH1Model',
              {'embed_tokens': ('embedding',), 'layers': ('layers',), 'final_layernorm': ('norm',),
               'rotary_emb': ('rotary',)}, _read_scaled_model, canonical={'final_layernorm': 'norm'}),
    KnownPart('layer', f'{_FALCON_H1}:FalconH1DecoderLayer',
              {'mamba': ('mamba2',), 'self_attn': ('attention',), 'feed_forward': ('mlp',),
               'input_layernorm': ('norm',), 'pre_ff_layernorm': ('norm',)}, _read_parallel_hybrid_layer,
              canonical={'feed_forward': 'mlp', 'pre_ff_layernorm': 'post_attention_layernorm'}),
    KnownPart('attention', f'{_FALCON_H1}:FalconH1Attention', _ATTENTION_CHILDREN, _read_keyed_attention),
    KnownPart('mamba2', f'{_FALCON_H1}:FalconH1Mixer',
              {'in_proj': ('linear',), 'conv1d': ('conv1d',), 'norm': ('gated_norm',),
               'out_proj': ('linear',)},
              _read_sectioned_mamba2, ('dt_bias', 'A_log', 'D')),
    KnownPart('gated_norm', f'{_FALCON_H1}:FalconH1RMSNormGated', {}, _read_nothing, ('weight',)),
    KnownPart('gated_norm', 'torch.nn:Identity', {}, _read_nothing),
    KnownPart('mlp', f'{_FALCON_H1}:FalconH1MLP',
              dict.fromkeys(('gate_proj', 'up_proj', 'down_proj'), ('linear',)), _read_multiplied_mlp),
    KnownPart('causal_lm', f'{_JAMBA}:JambaForCausalLM', {'model': ('model',), 'lm_head': ('linear',)},
              _read_causal_lm),
    KnownPart('model', f'{_JAMBA}:JambaModel',
              {'embed_tokens': ('embedding',), 'layers': ('layers',), 'final_layernorm': ('norm',)},
              _read_llama_model, canonical={'final_layernorm': 'norm'}),
    KnownPart('layer', f'{_JAMBA}:JambaAttentionDecoderLayer',
              {'self_attn': ('attention',), 'feed_forward': ('mlp', 'moe'), 'input_layernorm': ('norm',),
               'pre_ff_layernorm': ('norm',)}, _read_nothing,
              canonical={'feed_forward': 'mlp', 'pre_ff_layernorm': 'post_attention_layernorm'}),
    KnownPart('layer', f'{_JAMBA}:JambaMambaDecoderLayer',
              {'mamba': ('mamba',), 'feed_forward': ('mlp', 'moe'), 'input_layernorm': ('norm',),
               'pre_ff_layernorm': ('norm',)}, _read_nothing,
              canonical={'feed_forward': 'mlp', 'pre_ff_layernorm': 'post_attention_layernorm'}),
    KnownPart('attention', f'{_JAMBA}:JambaAttention', _ATTENTION_CHILDREN, _read_unrotated_attention),
    KnownPart('mamba', f'{_JAMBA}:JambaMambaMixer',
              {**dict.fromkeys(('in_proj', 'x_proj', 'dt_proj', 'out_proj'), ('linear',)),
               'conv1d': ('conv1d',),
               **dict.fromkeys(('dt_layernorm', 'b_layernorm', 'c_layernorm'), ('norm',))},
              _read_step_normed_mamba, ('A_log', 'D')),
    KnownPart('moe', f'{_JAMBA}:JambaSparseMoeBlock', {'router': ('linear',), 'experts': ('experts',)},
              _read_unnormalized_moe, canonical={'router': 'gate'}),
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
    flat: frozenset[str] = frozenset()
    """The canonical names of the parameters the source stores with leading unit axes that their leaves
    hold flat (`KnownPart.flat`), which the family's `prepare` reads flat."""
    constants: frozenset[str] = frozenset()
    """The canonical names of the stored buffers Dew holds as constants (`KnownPart.constants`)."""
    branches: Mapping[int, Mapping[str, str]] = field(default_factory=dict)
    """Each layer that runs mixers side by side (`dew.nn.mixers.parallel`, Falcon-H1's): its children's
    canonical names onto their branches' names, under which the layer's mixer holds each branch's
    parameters (`family`)."""
    depths: int = 0
    """The prediction depths the source's release stores past the trunk (its config's
    `num_nextn_predict_layers`: GLM-4.7-Flash's `model.layers.47.*`, HY-3's `.80.*`), which its own code
    builds none of and its load leaves unread (`family`)."""

    def canonical(self, name: str) -> str | None:
        """A stored tensor's canonical name: its module's canonical path and its parameter's known
        name, a part a fused tensor splits into (`_SPLIT`) by its own, one expert's tensor of a
        checkpoint stored per expert by its index and projection; None where no part holds it."""
        loaded, expert = self.loaded(name)
        stem, marked, split = loaded.partition(f'.{_SPLIT}')
        if marked and '.' in split:
            # A part of a fused Linear's weight (`_fused_linear`): the canonical projection it is, under
            # the canonical path its Linear's parameters read as.
            held = self.names.get(stem)
            projection, _, parameter = split.partition('.')
            if held is None or parameter not in held[1]:
                return None
            return f'{held[0]}.{projection}.{held[1][parameter]}'
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


def _load_names(model: PreTrainedModel, inventory: Collection[str] | None
                ) -> Callable[[str], tuple[str, tuple[str, str] | None]]:
    """The name `model`'s load gives a stored tensor (`sources.load_names`: HY-3's `mlp.expert_bias` is its
    `mlp.e_score_correction_bias`), an expert's own tensor, `{experts}.{index}.{stored}.weight`, named as
    the fused tensor the load stacks it into, with its index and its canonical projection, the one at its
    place among the merge's sources (Mixtral's `w1`, `w3` and `w2` are the gate, up and down projections).
    A merge is read only where it stacks experts on the first axis and concatenates its sources in
    `_PER_EXPERT`'s order on the second, as the fused tensor `_fused_experts` splits holds them; any other
    conversion's tensor keeps its stored name, which no part holds. A mapping that renames a key by
    whether another is stored reads the checkpoint's `inventory` (`sources.load_names`), and without one
    is `Ambiguous`: no read of one name tells it."""
    from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter

    names = sources.load_names(model, inventory)
    if names is None:
        raise Ambiguous("its checkpoint conversion renames a key by whether another is stored, and no "
                        "inventory of the checkpoint is given: pass the checkpoint's stored tensor names "
                        "(`inventory`)")
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
    def __init__(self, config: PretrainedConfig):
        self.config = config
        self.layer_paths: dict[int, str] = {}
        self.sites: dict[int, dict[str, dict[str, object]]] = {}
        self.fields: dict[str, object] = {}
        self.layers: dict[int, dict[str, object]] = {}
        self.names: dict[str, tuple[str, Mapping[str, str]]] = {}
        self.tables: dict[str, tuple[str, object]] = {}
        self.packed: list[decoder_parts.Packed] = []
        self.dropouts: list[tuple[str, str]] = []
        self.flat: set[str] = set()
        self.constants: set[str] = set()

    def state(self, fields: Mapping[str, object], where: str, layer: int | None, canonical: str) -> None:
        for key, value in fields.items():
            if key in (_LAYER, _SITE):
                if layer is None:
                    raise Ambiguous(f"{where} states a layer's fact outside a layer")
                if key == _LAYER:
                    self.layers.setdefault(layer, {}).update(_records.record(value, _LAYER))
                else:
                    site = canonical.removeprefix(f'{self.layer_paths[layer]}.').partition('.')[0]
                    held = self.sites.setdefault(layer, {}).setdefault(site, {})
                    held.update(_records.record(value, _SITE))
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
                self.layer_paths[index] = f'{canonical}.{index}'
                self.walk(child, ('layer',), f'{path}.{index}', f'{canonical}.{index}', index)
            return 'layers'
        part, renaming = identify(module, roles)
        view = PartView(module, renaming, self.config)
        self.state(part.read(view), path or type(module).__name__, layer, canonical)
        # The head is tied where it is the table's own parameter, as the source holds it.
        if part.role == 'embedding' or (part.role == 'linear' and canonical == 'lm_head'):
            table = 'embedding' if part.role == 'embedding' else 'head'
            self.tables[table] = (f'{path}.{view.name("weight")}', view.weight)
        if part.role == 'experts':
            self.packed.extend(_fused_experts(path.rpartition('.')[2], view))
        self.packed.extend(_fused_linear(path.rpartition('.')[2], view.name(known), halves)
                           for known, halves in part.split.items())
        if part.role == 'attention':
            self.dropouts.append((path, view.name('attention_dropout')))
        held = {name for name, _ in module.named_parameters(recurse=False)}
        expected = {view.name(name): name for name in part.parameters}
        if held - set(expected):
            raise Ambiguous(f"{path} holds parameters {sorted(held - set(expected))} its code does not read")
        buffers = dict(module.named_buffers(recurse=False))
        held_state = {**part.state, **{name: name for name in part.constants}}
        stored = {view.name(name): known for name, known in held_state.items() if view.name(name) in buffers}
        self.constants.update(f'{canonical}.{name}' for name in part.constants if view.name(name) in buffers)
        if held or stored:
            self.names[path] = (canonical, {**{name: expected[name] for name in held}, **stored})
        self.flat.update(f'{canonical}.{name}' for name in part.flat)
        children = dict(module.named_children())
        claimed = set()
        for known, child_roles in part.children.items():
            name = view.name(known)
            claimed.add(name)
            child = children.get(name)
            if child is None:
                continue
            spelled = part.canonical.get(known, known)
            under = canonical if not spelled else f'{canonical}.{spelled}' if canonical else spelled
            role = self.walk(child, child_roles, f'{path}.{name}' if path else name, under, layer)
            if part.role == 'layer' and spelled == 'mlp' and layer is not None:
                self.layers.setdefault(layer, {})['routed'] = role == 'moe'
        unclaimed = [name for name, child in children.items() if name not in claimed
                     and any(True for _ in child.parameters())]
        if unclaimed:
            raise Ambiguous(f"{path or type(module).__name__} holds {unclaimed}, which its code does not "
                            "call")
        return part.role

    def parallel(self) -> dict[int, dict[str, str]]:
        """Each layer whose part runs mixers side by side (`parallel`, Falcon-H1's) made one `parallel` mixer
        fact in place (`dew.nn.mixers.parallel.ParallelMixer`, a kind `_with_mixers` makes): its branches in
        its forward's order, each with its input's and its output's multipliers, the one named `attention`
        the layer's attention (its per-layer fields, `attention`, and its placing no positions, `nope`) and
        the other its mixer part's (`mixer`). Returns each such layer's children's canonical names onto
        their branches' (`Assembly.branches`)."""
        branched: dict[int, dict[str, str]] = {}
        for layer, facts in self.layers.items():
            parallel = facts.pop('parallel', None)
            if parallel is None:
                continue
            own = facts.pop('mixer', None)
            if own is None:
                raise Ambiguous(f"layer {layer} runs a mixer beside its attention that no part states")
            own = dict(_records.record(own, 'mixer'))
            attention = dict(_records.record(facts.pop('attention', {}), 'attention'))
            if facts.pop('nope', False):
                attention['nope'] = True
            runs, names = [], {}
            for child, stated in _records.record(parallel, 'parallel').items():
                run = dict(_records.record(stated, child))
                name = _records.text(run['name'], 'name')
                run['mixer'] = ({'class': 'attention', 'fields': attention} if name == 'attention'
                                else {'class': own['class'], 'fields': own['fields']})
                runs.append(run)
                names[child] = name
            facts['mixer'] = {'kind': f"{own['kind']}_attention", 'class': 'parallel',
                              'fields': {'branches': runs}}
            branched[layer] = names
        return branched


def assemble(model: PreTrainedModel, inventory: Collection[str] | None = None) -> Assembly:
    """`model`, a source's causal LM, assembled from known parts, or `Ambiguous` naming the module that
    is not one; its stored names read as its load names them, over the checkpoint's `inventory` where
    its conversion needs the whole of it (`_load_names`)."""
    assembler = _Assembler(model.config)
    assembler.walk(model, ('causal_lm',), '', '', None)
    (head, head_weight), (table, table_weight) = assembler.tables['head'], assembler.tables['embedding']
    fields = {**assembler.fields, 'tie_embeddings': head_weight is table_weight}
    routed = sorted(layer for layer, facts in assembler.layers.items() if facts.get('routed'))
    if routed:
        routes = {tuple(sorted((key, value) for key, value in assembler.layers[layer].items()
                               if key not in ('routed', 'nope', 'mixer', 'attention', 'heads', 'parallel')))
                  for layer in routed}
        if len(routes) != 1:
            raise Ambiguous("the routed layers route differently")
        mixture = dict(routes.pop())
        width = _records.integer(mixture['expert_features'], 'expert_features')
        # A router's scores, scaling and correction bias and a block's shared experts, where its parts
        # state them (DeepSeek's routing, HY-3's).
        stated = {key: mixture[key] for key in ('score_function', 'scaling', 'bias', 'shared_features',
                                                'groups', 'groups_per_token', 'weight_sum_epsilon',
                                                'weight_sum_min', 'router_softcap', 'excluded_group_score')
                  if key in mixture}
        fields['mixture'] = {**native_fields(Mixture)(
            experts=_records.integer(mixture['experts'], 'experts'),
            top_k=_records.integer(mixture['top_k'], 'top_k'),
            norm_topk_prob=bool(mixture['norm_topk_prob']), expert_features=width,
            layers=None if len(routed) == fields['num_layers'] else tuple(routed)), **stated}
        fields.setdefault('mlp_features', width)
    heads = {layer: facts['heads'] for layer, facts in assembler.layers.items() if 'heads' in facts}
    fields.update(_with_head_counts(fields, heads))
    gates = {layer: {site: stated['gate_rank'] for site, stated in sites.items() if 'gate_rank' in stated}
             for layer, sites in assembler.sites.items()}
    fields.update(_with_norm_gates(fields, {layer: held for layer, held in gates.items() if held}, routed))
    for required in ('vocab_size', 'num_layers', 'num_heads', 'mlp', 'mlp_features', 'norm_eps'):
        if required not in fields:
            raise Ambiguous(f"no part states {required}")
    # A base for the rotary wherever a layer's attention rotates: one placed by positions, its layer's
    # mixer an attention or beside one.
    count = _records.integer(fields['num_layers'], 'num_layers')
    rotating = [held for held in (assembler.layers.get(layer, {}) for layer in range(count))
                if not held.get('nope') and ('parallel' in held or 'mixer' not in held)]
    if rotating and 'rope_theta' not in fields:
        raise Ambiguous("no part states rope_theta")
    branches = assembler.parallel()
    unrotated = sorted(layer for layer, facts in assembler.layers.items() if facts.get('nope'))
    fields.update(_without_positions(fields, unrotated))
    fields.update(_with_mixers(fields, {layer: facts['mixer'] for layer, facts in assembler.layers.items()
                                        if 'mixer' in facts}))
    fields.update(_with_attention_fields(fields, {layer: facts['attention'] for layer, facts
                                                  in assembler.layers.items() if 'attention' in facts}))
    capacity, context = decoder_parts.context_lengths(
        {'max_position_embeddings': model.config.to_dict().get('max_position_embeddings')})
    fields.update(max_seq_len=capacity, context_length=context)
    kinds = fields.pop('kinds', {})
    record = native_fields(CausalTransformer)(**fields)
    record['kinds'] = kinds
    depths = _records.integer(model.config.to_dict().get('num_nextn_predict_layers') or 0,
                              'num_nextn_predict_layers')
    return Assembly(record, assembler.names, tuple(dict.fromkeys(assembler.packed)), (head, table),
                    tuple(assembler.dropouts), _load_names(model, inventory), frozenset(assembler.flat),
                    frozenset(assembler.constants), branches, depths)


def _with_attention_fields(fields: Mapping[str, object], stated: Mapping[int, object]
                           ) -> Mapping[str, object]:
    """The kinds that give each layer's attention the mixer fields its part states (`stated`, by layer):
    ERNIE 4.5's adjacent rotary pairs, Hunyuan's norm after the rotation. Each layer type whose layers
    state them takes them into its kind's attention, beside any it holds (a NoPE kind's); every layer of
    that type must state the same, and a type whose kind mixes otherwise (a convolution) states none."""
    if not stated:
        return {}
    count = _records.integer(fields['num_layers'], 'num_layers')
    stated_types = fields.get('layer_types')
    types = (list(_records.strings(stated_types, 'layer_types')) if stated_types
             else ['full_attention'] * count)
    kinds = dict(_records.record(fields.get('kinds', {}), 'kinds'))
    for name in dict.fromkeys(types[layer] for layer in stated):
        held = [dict(_records.record(stated[layer], 'attention')) if layer in stated else None
                for layer in range(count) if types[layer] == name]
        if any(mixer != held[0] for mixer in held):
            raise Ambiguous(f"the {name} layers' attentions compute differently")
        kind = dict(_records.record(kinds[name], name)) if name in kinds else dict(native_fields(LayerKind)())
        mixer = dict(_records.record(kind.get('mixer') or {'class': 'attention', 'fields': {}}, 'mixer'))
        if mixer.get('class') != 'attention':
            raise Ambiguous(f"the {name} layers' mixer is a {mixer.get('class')}, not an attention")
        kind['mixer'] = {'class': 'attention',
                         'fields': {**_records.record(mixer.get('fields', {}), 'fields'), **(held[0] or {})}}
        kinds[name] = kind
    return {'layer_types': tuple(types), 'kinds': kinds}


def _with_norm_gates(fields: Mapping[str, object], gates: Mapping[int, object], routed: list[int]
                     ) -> Mapping[str, object]:
    """The norms whose outputs a low-rank gate scales (`gates`, by layer, each gated norm's rank by its site,
    A.X-K2's): each site's `NormSpec` gated at its one rank on every layer, or on the `routed` layers
    alone; a site gated on other layers, or at several ranks, is `Ambiguous`."""
    if not gates:
        return {}
    count = _records.integer(fields['num_layers'], 'num_layers')
    held = {layer: _records.record(sites, 'norm_gates') for layer, sites in gates.items()}
    specs = dict(_records.record(fields.get('norms') or {}, 'norms'))
    for site in sorted({site for sites in held.values() for site in sites}):
        ranks = {_records.integer(sites[site], site) for sites in held.values() if site in sites}
        layers = {layer for layer, sites in held.items() if site in sites}
        if len(ranks) != 1:
            raise Ambiguous(f"the {site} norms' gates are {sorted(ranks)} wide")
        if layers == set(range(count)):
            which = 'every'
        elif layers == set(routed):
            which = 'routed'
        else:
            raise Ambiguous(f"the {site} norms are gated on layers {sorted(layers)}: neither every layer nor "
                            "the routed ones")
        specs[site] = {**native_fields(NormSpec)(), 'gate_rank': ranks.pop(), 'gated_layers': which}
    return {'norms': specs}


def _with_head_counts(fields: Mapping[str, object], heads: Mapping[int, object]) -> Mapping[str, object]:
    """The model's query heads and the kinds whose layers' attentions hold others (`heads`, by layer,
    Laguna's `num_attention_heads_per_layer`): the model's its first layer's, and each layer type whose
    layers hold another count takes it into its kind; every layer of a type must hold the same."""
    if not heads:
        return {}
    count = _records.integer(fields['num_layers'], 'num_layers')
    if set(heads) != set(range(count)):
        raise Ambiguous("some layers' attentions state their heads and others do not")
    stated_types = fields.get('layer_types')
    types = (list(_records.strings(stated_types, 'layer_types')) if stated_types
             else ['full_attention'] * count)
    model = _records.integer(heads[0], 'heads')
    kinds = dict(_records.record(fields.get('kinds', {}), 'kinds'))
    for name in dict.fromkeys(types):
        held = {_records.integer(heads[layer], 'heads') for layer in range(count) if types[layer] == name}
        if len(held) != 1:
            raise Ambiguous(f"the {name} layers' attentions hold {sorted(held)} heads")
        if (own := held.pop()) != model:
            kind = dict(_records.record(kinds.get(name) or native_fields(LayerKind)(), name))
            kind['num_heads'] = own
            kinds[name] = kind
    return {'num_heads': model, 'kinds': kinds, 'layer_types': tuple(types)}


def _with_mixers(fields: Mapping[str, object], mixers: Mapping[int, object]) -> Mapping[str, object]:
    """The layer types and kinds that make each layer of `mixers` (by layer, the kind its mixer's part names,
    the mixer's class and its fields: LFM2's gated short convolutions, `conv`; Jamba's Mamba mixers,
    `mamba`) a layer of that kind, its mixer that class, beside the attention layers, which keep their
    layer type; every layer of a kind must mix alike."""
    if not mixers:
        return {}
    count = _records.integer(fields['num_layers'], 'num_layers')
    stated_types = fields.get('layer_types')
    types = (list(_records.strings(stated_types, 'layer_types')) if stated_types
             else ['full_attention'] * count)
    kinds = dict(_records.record(fields.get('kinds', {}), 'kinds'))
    stated: dict[str, dict[str, object]] = {}
    for layer, fact in mixers.items():
        record = dict(_records.record(fact, 'mixer'))
        name = _records.text(record.pop('kind'), 'kind')
        if stated.setdefault(name, record) != record:
            raise Ambiguous(f"the {name} layers mix differently")
        types[layer] = name
    for name, record in stated.items():
        kind = dict(native_fields(LayerKind)())
        kind['mixer'] = record
        kinds[name] = kind
    return {'layer_types': tuple(types), 'kinds': kinds}


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
"""The mark of a part a fused tensor splits into (`_fused_experts`, `_fused_linear`), already canonical:
no Python attribute holds it, so it names no stored parameter, whatever the source renames its own to."""


def _fused_linear(module: str, linear: str, projections: tuple[str, ...]) -> decoder_parts.Packed:
    """The weight of `module`'s fused Linear `linear` (`KnownPart.split`) as its canonical `projections`,
    concatenated on its output rows in their order: each part names the projection after the split mark,
    its parameter after it (`Assembly.canonical`)."""
    return decoder_parts.Packed(f'.{module}.{linear}.weight',
                                tuple(f'.{module}.{linear}.{_SPLIT}{projection}.weight'
                                      for projection in projections), 0)


_STACKED_EXPERTS: decoder_parts.Leaves = {
    f'mlp.experts.{name}': ('mlp', 'experts', name, 'kernel') for name in decoder_parts.MOE_SHARED}
"""The canonical experts, each projection one tensor over every expert (`decoder_parts.FUSED_EXPERTS`)."""


def family(model_type: str, assembly: Assembly) -> decoder_parts.DecoderFamily:
    """The decoder family a type verified from parts loads through (`dew.interop.verify`): `assembly`'s
    fields for its config, every field of which the source's own code read, and each stored tensor at
    the path of its canonical name (`Assembly.canonical`), a fused expert tensor split as Mixtral's
    first (`DecoderFamily.packed`), so an export writes each back under its stored name. A tensor of one
    of the release's prediction depths past the trunk (`Assembly.depths`, by `decoder_parts.layer_depth`)
    fills no leaf: the source's own code builds no depth and its load leaves them unread, so the decoder
    assembled from that code has none (`prediction_depths` trims them before any is read). Only a depth's
    own tensors (`decoder_parts.PREDICTION_LEAVES`, `PREDICTION_STEP_END`) and its block's, a name a trunk
    layer's part holds, are so; its copies of the trunk's table and head are the trunk's
    (`decoder_parts.prediction_copy_path`: `hf_decoders` checks each against it, and an export writes it
    from the trunk's leaf)."""
    def translated(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
        used.update(hf_config)
        return assembly.fields.copy()

    def weight_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
        depth = decoder_parts.layer_depth(name, config)
        if depth is not None and depth < assembly.depths:
            tail = name.split('.', 3)[3]
            copy = decoder_parts.prediction_copy_path(tail, config)
            if copy is not None:
                return copy
            own = {**decoder_parts.PREDICTION_LEAVES, **decoder_parts.PREDICTION_STEP_END}
            last = _records.integer(config['num_layers'], 'num_layers') - 1
            if tail in own or assembly.canonical(f'model.layers.{last}.{tail}') is not None:
                return None
        canonical = assembly.canonical(name)
        if canonical is None:
            raise ValueError(f"the checkpoint's {name!r} is held by no part of the assembled decoder")
        leaves = {**_STACKED_EXPERTS, **decoder_parts.SHORT_CONV_LEAVES, **decoder_parts.XIELU_LEAVES,
                  **decoder_parts.MAMBA_LEAVES, **decoder_parts.GATED_NORM_LEAVES}
        path = decoder_parts.renamed_path((), canonical, config, layer_leaves=leaves)
        parts = canonical.split('.', 4)
        if path is not None and len(parts) == 5 and parts[:2] == ['model', 'layers'] and parts[2].isdigit():
            branch = assembly.branches.get(int(parts[2]), {}).get(parts[3])
            if branch is not None:
                # A branch of a parallel layer's mixer holds what the layer's own mixer would.
                path = (*path[:3], branch, *path[3:])
        if canonical in assembly.constants and path is not None:
            # A buffer no step trains, held where Dew keeps constants (`KnownPart.constants`).
            return ('constants', *path[1:])
        return path

    def prepare(tensors: Mapping[str, Stored], _config: Mapping[str, object] | None = None
                ) -> Mapping[str, Stored]:
        prepared = dict(tensors)
        for name, value in tensors.items():
            if assembly.canonical(name) in assembly.flat:
                stored = held(value)
                if any(size != 1 for size in stored.shape[:-1]):
                    raise ValueError(f"the checkpoint's {name!r} holds {stored.shape}, not one row")
                prepared[name] = stored.reshape(-1)
        return prepared

    return decoder_parts.DecoderFamily(
        (model_type,), translated, lambda fields: False, model_type, 'ForCausalLM', lambda model: {},
        preserve_source_layout=True, weight_path=weight_path, packed=assembly.packed,
        tied_head_names=assembly.tied, prediction_depth=decoder_parts.layer_depth, prepare=prepare)
