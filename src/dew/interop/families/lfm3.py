"""Liquid's LFM3 decoder (LiquidAI/LFM3-1.7B, remote code at 40ebe0aa; the text backbone of LFM3-VL and d1).

Its layers are gated short convolutions (`dew.nn.mixers.short_conv`) between attention layers, each
`layer_types` entry one of `conv`, `full_attention` and `sliding_attention`. The attention norms its
queries and keys per head, gates each head's output by its own sigmoid (`gate_projection='head'`), and
turns rotary positions per layer type: the full layers at their base over the first
`partial_rotary_factor` of each head, the tail unrotated (modeling_lfm3.py, Lfm3RotaryEmbedding),
the sliding layers whole at theirs. Sandwich norms wrap each mixer and feed-forward
(`use_sandwich_norm`), the feed-forward is a SwiGLU of `w1`, `w3` and `w2`, the final norm is
`embedding_norm`, and the head is tied.
"""

from __future__ import annotations

from collections.abc import Mapping

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    Leaves,
    Renames,
    Ropes,
    base_config,
    decoder_tensors,
    kind_mixers,
    read_rope_theta,
    refuse,
    renamed_name,
    renamed_path,
    specified_layer_types,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mixers.short_conv import ShortConvMixer

_KINDS = ('conv', 'full_attention', 'sliding_attention')

_SANDWICH: Renames = (
    ('model.embedding_norm', 'model.norm'), ('feed_forward', 'mlp'), ('w1', 'gate_proj'),
    ('w3', 'up_proj'), ('w2', 'down_proj'), ('operator_norm', 'input_layernorm'),
    ('post_operator_norm', 'post_attention_layernorm'), ('ffn_norm', 'pre_feedforward_layernorm'),
    ('post_ffn_norm', 'post_feedforward_layernorm'))
"""LFM3's names onto the shared ones under sandwich norms (`use_sandwich_norm`): the norm before
each mixer and after it, before the feed-forward and after it, Gemma's four."""

_PLAIN: Renames = (
    ('model.embedding_norm', 'model.norm'), ('feed_forward', 'mlp'), ('w1', 'gate_proj'),
    ('w3', 'up_proj'), ('w2', 'down_proj'), ('operator_norm', 'input_layernorm'),
    ('ffn_norm', 'post_attention_layernorm'))
"""LFM3's names onto the shared ones without sandwich norms: one norm before each mixer and one
before each feed-forward."""

_CONV_LEAVES: Leaves = {
    **{f'conv.{projection}.{leaf}': ('self_attn', projection, 'kernel' if leaf == 'weight' else 'bias')
       for projection in ('in_proj', 'out_proj') for leaf in ('weight', 'bias')},
    'conv.conv.weight': ('self_attn', 'conv', 'weight'),
    'conv.conv.bias': ('self_attn', 'conv', 'bias'),
}
"""A convolution layer's tensors, under the mixer's slot (`GatedShortConv`): its projections and its
`[D, 1, K]` taps."""


def _renames(config: Mapping[str, object]) -> Renames:
    """The renames of a model's fields record, with sandwich norms (`use_sandwich_norm`) or none."""
    return _SANDWICH if config.get('sandwich_norms') else _PLAIN


def lfm3_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    return renamed_path(_renames(config), name, config, layer_leaves=_CONV_LEAVES)


def lfm3_export_path(name: str, config: Mapping[str, object]) -> str | None:
    return renamed_name(_renames(config), name, config, layer_leaves=_CONV_LEAVES,
                        sandwich_norms=bool(config.get('sandwich_norms')))


def _lfm3_rope(hf_config: Mapping[str, object]) -> tuple[Ropes, float]:
    """The rotary of each layer type and the full layers' partial factor: LFM3's
    `rope_parameters` holds one plain record a type (`compute_default_rope_parameters`), whose
    full record turns its first `partial_rotary_factor` of each head; a sliding layer turns all of
    its own."""
    stated = _records.record(hf_config.get('rope_parameters') or {}, 'rope_parameters')
    unknown = sorted(set(stated) - {'full_attention', 'sliding_attention'})
    if unknown:
        refuse(f"rope_parameters {unknown}", "LFM3 keeps one rotary record a layer type")
    full = dict(_records.record(stated.get('full_attention') or {}, 'rope_parameters.full_attention'))
    sliding = dict(_records.record(stated.get('sliding_attention') or {},
                                   'rope_parameters.sliding_attention'))
    if sliding.get('partial_rotary_factor', 1.0) not in (1, 1.0):
        refuse("rope_parameters.sliding_attention partial_rotary_factor",
               "LFM3's sliding layers rotate whole heads in Dew's port")
    sliding.pop('partial_rotary_factor', None)
    partial = _records.number(full.pop('partial_rotary_factor', 1.0),
                              'rope_parameters.full_attention partial_rotary_factor')
    theta = read_rope_theta(full, 'rope_parameters.full_attention') or 10000.0
    local = read_rope_theta(sliding, 'rope_parameters.sliding_attention') or 10000.0
    return Ropes(theta, None, None if local == theta else local), partial


def lfm3_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    layer_types = specified_layer_types(hf_config, used)
    unknown = sorted(set(layer_types) - set(_KINDS))
    if unknown:
        refuse(f"layer_types {unknown}", f"an LFM3 layer is one of {list(_KINDS)}")
    ropes, partial = _lfm3_rope(hf_config)
    config = base_config(hf_config, used, rope=ropes, layer_types=layer_types, qk_norm=True,
                         tie_embeddings=True)
    used.update(('rope_parameters', 'conv_L_cache', 'conv_bias', 'use_sandwich_norm'))
    kinds = dict(config['kinds'])
    attention = {'class': 'attention', 'fields': {'gate_projection': 'head'}}
    for kind in ('full_attention', 'sliding_attention'):
        if kind in layer_types:
            kinds.setdefault(kind, native_fields(LayerKind)())['mixer'] = attention
    if 'full_attention' in layer_types and partial != 1.0:
        kinds['full_attention']['partial_rotary_factor'] = partial
    if 'conv' in layer_types:
        kinds['conv'] = native_fields(LayerKind)(mixer=None)
        kinds['conv']['mixer'] = {'class': 'short_conv', 'fields': {
            'conv_L_cache': _records.integer(hf_config['conv_L_cache'], 'conv_L_cache'),
            'conv_bias': bool(hf_config['conv_bias'])}}
    config.update(
        sandwich_norms=bool(hf_config['use_sandwich_norm']),
        output_gate=True,
        # Lfm3RotaryEmbedding zeroes the tail's frequencies: the first pairs at the whole head's
        # progression, the rest unrotated (`partial_rotary_type='proportional'`).
        partial_rotary_type='proportional',
        kinds=kinds,
    )
    return config


def _export(model: CausalTransformer) -> Mapping[str, object]:
    """LFM3's own config fields beside the shared ones the export writes."""
    kinds = model.kinds or {}
    conv = next((mixer for mixer in kind_mixers(model) if isinstance(mixer, ShortConvMixer)), None)
    full = kinds.get('full_attention')
    sliding = kinds.get('sliding_attention')
    rope = {'full_attention': {'rope_type': 'default', 'rope_theta': model.rope_theta,
                               'partial_rotary_factor': (full.partial_rotary_factor if full is not None
                                                         and full.partial_rotary_factor is not None
                                                         else 1.0)}}
    if sliding is not None:
        rope['sliding_attention'] = {'rope_type': 'default',
                                     'rope_theta': sliding.rope_theta or model.rope_theta}
    return {'model_type': 'lfm3', 'norm_eps': model.norm_eps, 'rms_norm_eps': None,
            'rope_parameters': rope, 'rope_theta': None,
            'use_sandwich_norm': bool(model.sandwich_norms),
            'conv_L_cache': None if conv is None else conv.conv_L_cache,
            'conv_bias': False if conv is None else conv.conv_bias,
            'sliding_window': None if sliding is None else sliding.window}


LFM3 = DecoderFamily(
    ('lfm3',),
    lfm3_config,
    lambda fields: any(isinstance(mixer, ShortConvMixer) for mixer in kind_mixers(fields)),
    'lfm3',
    'Lfm3ForCausalLM',
    _export,
    weight_path=lfm3_path,
    export_path=lfm3_export_path,
    # The path map names every leaf of the convolution layers, so the shared writer writes it whole.
    export_weights=decoder_tensors,
    preserve_source_layout=True,
    padding_idx=True,
    aliases={'norm_eps': 'rms_norm_eps'},
    # Lfm3Config's defaults for the fields its decoder reads (configuration_lfm3.py).
    defaults={'conv_L_cache': 7, 'conv_bias': False, 'use_sandwich_norm': True, 'sliding_window': 256,
              'tie_word_embeddings': True, 'max_position_embeddings': 128000, 'norm_eps': 1e-5,
              'pad_token_id': 124893},
)
