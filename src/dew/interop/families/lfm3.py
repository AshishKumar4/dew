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
from dew.interop import release_pins
from dew.interop.config_records import native_fields
from dew.interop.decoder_config import (
    DecoderSpelling,
    decoder_layout,
    expect,
    flag,
    integer,
    kind_mixer,
    place_ropes,
)
from dew.interop.decoder_parts import (
    SHORT_CONV_LEAVES,
    DecoderFamily,
    DecoderFields,
    Renames,
    Ropes,
    SourceCode,
    kind_mixers,
    kinds_of,
    read_rope_theta,
    refuse,
    specified_layer_types,
    spelled,
)
from dew.interop.layouts import ConfigLayout, ConfigPart, Table, When, constant
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mixers.short_conv import ShortConvMixer

_KINDS = ('conv', 'full_attention', 'sliding_attention')


_FEED_FORWARD: Renames = (
    ('feed_forward.w1', 'mlp.gate_proj'), ('feed_forward.w3', 'mlp.up_proj'),
    ('feed_forward.w2', 'mlp.down_proj'))
"""LFM3's feed-forward names onto the shared ones, whole: its attention has a `gate_proj` of its own,
which a bare `w1` for `gate_proj` would respell on export."""

_SANDWICH: Renames = (
    ('model.embedding_norm', 'model.norm'), *_FEED_FORWARD, ('operator_norm', 'input_layernorm'),
    ('post_operator_norm', 'post_attention_layernorm'), ('ffn_norm', 'pre_feedforward_layernorm'),
    ('post_ffn_norm', 'post_feedforward_layernorm'))
"""LFM3's names onto the shared ones under sandwich norms (`use_sandwich_norm`): the norm before
each mixer and after it, before the feed-forward and after it, Gemma's four."""

_PLAIN: Renames = (
    ('model.embedding_norm', 'model.norm'), *_FEED_FORWARD, ('operator_norm', 'input_layernorm'),
    ('ffn_norm', 'post_attention_layernorm'))
"""LFM3's names onto the shared ones without sandwich norms: one norm before each mixer and one
before each feed-forward."""



LFM3_ROUTES = Table(
    When(lambda config: bool(config.get('sandwich_norms')),
         spelled(_SANDWICH, layer_leaves=SHORT_CONV_LEAVES)),
    When(lambda config: not config.get('sandwich_norms'), spelled(_PLAIN, layer_leaves=SHORT_CONV_LEAVES)))
"""LFM3's names, by the record's `sandwich_norms`: its convolution layers' (`SHORT_CONV_LEAVES`) beside the
shared decoder's."""


_ROPE = {'full_attention': {'rope_type': 'default', 'rope_theta': 1_000_000.0, 'partial_rotary_factor': 0.5},
         'sliding_attention': {'rope_type': 'default', 'rope_theta': 10_000.0}}
"""Lfm3Config's rotary where a config states none (configuration_lfm3.py:88-93)."""


def _layer_types(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The config's `layer_types`, else Lfm3Config's own: full attention at every fourth layer from the
    third, sliding at every fourth from the fifth, convolutions between (configuration_lfm3.py:94-98)."""
    if hf_config.get('layer_types') is not None:
        layer_types = specified_layer_types(hf_config, used)
    else:
        count = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
        layer_types = tuple('full_attention' if index % 4 == 2 else 'sliding_attention' if index % 4 == 0
                            and index else 'conv' for index in range(count))
    unknown = sorted(set(layer_types) - set(_KINDS))
    if unknown:
        refuse(f"layer_types {unknown}", f"an LFM3 layer is one of {list(_KINDS)}")
    record['layer_types'] = layer_types
    return record


def _rope(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The rotary of each layer type and the full layers' partial factor: LFM3's `rope_parameters` holds
    one plain record a type (`compute_default_rope_parameters`), whose full record turns its first
    `partial_rotary_factor` of each head; a sliding layer turns all of its own. A config that states none
    takes Lfm3Config's (`_ROPE`); one that does names a record for each attention type its layers hold."""
    used.add('rope_parameters')
    layer_types = _records.strings(record['layer_types'], 'layer_types')
    given = hf_config.get('rope_parameters')
    stated = _records.record(_ROPE if given is None else given, 'rope_parameters')
    unknown = sorted(set(stated) - {'full_attention', 'sliding_attention'})
    if unknown:
        refuse(f"rope_parameters {unknown}", "LFM3 keeps one rotary record a layer type")
    missing = sorted(kind for kind in ('full_attention', 'sliding_attention')
                     if kind in layer_types and not stated.get(kind))
    if missing:
        refuse(f"rope_parameters without {missing}", "each attention layer type reads its own record")
    # A type no layer holds reads no record (Lfm3RotaryEmbedding builds its layers' types alone).
    full = dict(_records.record(stated.get('full_attention') or _ROPE['full_attention'],
                                'rope_parameters.full_attention'))
    sliding = dict(_records.record(stated.get('sliding_attention') or _ROPE['sliding_attention'],
                                   'rope_parameters.sliding_attention'))
    if sliding.get('partial_rotary_factor', 1.0) not in (1, 1.0):
        refuse("rope_parameters.sliding_attention partial_rotary_factor",
               "LFM3's sliding layers rotate whole heads in Dew's port")
    sliding.pop('partial_rotary_factor', None)
    partial = _records.number(full.pop('partial_rotary_factor', 1.0),
                              'rope_parameters.full_attention partial_rotary_factor')
    theta = read_rope_theta(full, 'rope_parameters.full_attention') or 1_000_000.0
    local = read_rope_theta(sliding, 'rope_parameters.sliding_attention') or 10_000.0
    place_ropes(record, Ropes(theta, None, None if local == theta else local))
    if 'full_attention' in layer_types and partial != 1.0:
        kinds = kinds_of(record)
        kinds.setdefault('full_attention', native_fields(LayerKind)())['partial_rotary_factor'] = partial
        record['kinds'] = kinds
    return record


def _rope_written(model: CausalTransformer) -> Mapping[str, object]:
    kinds = model.kinds or {}
    full = kinds.get('full_attention')
    sliding = kinds.get('sliding_attention')
    rope = {'full_attention': {'rope_type': 'default', 'rope_theta': model.rope_theta,
                               'partial_rotary_factor': (full.partial_rotary_factor if full is not None
                                                         and full.partial_rotary_factor is not None
                                                         else 1.0)}}
    if sliding is not None:
        rope['sliding_attention'] = {'rope_type': 'default',
                                     'rope_theta': sliding.rope_theta or model.rope_theta}
    return {'rope_parameters': rope}


def _gated_heads(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """Each attention layer gates each head's output by its own sigmoid."""
    kinds = kinds_of(record)
    layer_types = _records.strings(record['layer_types'], 'layer_types')
    for kind in ('full_attention', 'sliding_attention'):
        if kind in layer_types:
            kinds.setdefault(kind, native_fields(LayerKind)())['mixer'] = {
                'class': 'attention', 'fields': {'gate_projection': 'head'}}
    record['kinds'] = kinds
    return record


LFM3_CONFIG = ConfigLayout(
    ConfigPart(_layer_types, lambda model: {'layer_types': list(model.per_layer_types)}),
    # Every projection of its attention is unbiased (modeling_lfm3.py:224-229): attention_bias is unread.
    decoder_layout(DecoderSpelling(reads=frozenset({'sliding_window'}), qk_norm=True, tie_embeddings=True,
                                   keys={'rms_norm_eps': 'norm_eps'}),
                   rope=ConfigPart(_rope, _rope_written)),
    # Lfm3MLP is SwiGLU whatever a config names (modeling_lfm3.py:151-160).
    expect('hidden_act', "LFM3's feed-forward gates by silu, which it reads from no field", value='silu'),
    ConfigPart(_gated_heads, lambda model: {}),
    kind_mixer('conv', 'short_conv', ShortConvMixer,
               ConfigLayout(integer('conv_L_cache', 'conv_L_cache'),
                            flag('conv_bias', 'conv_bias', default=False))),
    flag('use_sandwich_norm', 'sandwich_norms', default=True),
    constant('output_gate', value=True),
    # Lfm3RotaryEmbedding zeroes the tail's frequencies: the first pairs at the whole head's
    # progression, the rest unrotated (`partial_rotary_type='proportional'`). Its cos and sin are
    # the activations' dtype (modeling_lfm3.py:119-125).
    constant('partial_rotary_type', value='proportional'), constant('narrow_rotary_tables', value=True))
"""LFM3's config, both ways."""


LFM3 = DecoderFamily(
    ('lfm3',),
    LFM3_CONFIG,
    lambda fields: any(isinstance(mixer, ShortConvMixer) for mixer in kind_mixers(fields)),
    'lfm3',
    'Lfm3ForCausalLM',
    routes=LFM3_ROUTES,
    # The path map names every leaf of the convolution layers, so the shared writer writes it whole.
    preserve_source_layout=True,
    padding_idx=True,
    release_code=SourceCode(*release_pins.LFM3),
    # Lfm3Config's defaults for the fields its decoder reads (configuration_lfm3.py).
    defaults={'conv_L_cache': 7, 'conv_bias': False, 'use_sandwich_norm': True, 'sliding_window': 256,
              'tie_word_embeddings': True, 'max_position_embeddings': 128000, 'norm_eps': 1e-5,
              'pad_token_id': 124893},
)
