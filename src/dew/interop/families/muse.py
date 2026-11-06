"""Translate Muse Glimmer's text decoder (`muse_glimmer_text`, transformers 5.16.1).

A Gemma-3-like sandwich block whose norms differ by site: the block's norms
store (1 + w) and the sublayer outputs' take `post_norm_eps`, the embeddings
pass a weightless RMS norm and the final norm stores its weight as it is
(modeling_muse_glimmer.py:101-140, 363-405, 428-457). Queries and keys take a
weightless RMS norm, the query times `qk_scale_factor`, which the logit scale
absorbs (:312-331); a `gate_proj` of the layer's input gates the branch with a
sigmoid (:305, 351); three sliding layers precede each full one, which attends
without rotary positions (`layer_rope_theta` 0, :511). The logits are
`final_logit_softcapping * tanh(logits * output_multiplier / softcap)` (:1176-1181).
"""

from __future__ import annotations

from collections.abc import Mapping

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_config import DecoderSpelling, decoder_layout
from dew.interop.decoder_parts import DecoderFamily, DecoderFields, kinds_of, refuse
from dew.interop.layouts import ConfigField, ConfigLayout, ConfigPart, constant, unread
from dew.nn.backbones.decoder_block import NormSpec
from dew.nn.backbones.layer_plan import LayerKind


def _layer_bases(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """Each layer's rotary base, both ways: the reference reads one a layer (`layer_rope_theta`), 0 for NoPE;
    Dew rotates every sliding layer at rope_theta and none of the full ones, each kind's mixer saying so."""
    used.add('layer_rope_theta')
    layer_types = _records.strings(record['layer_types'], 'layer_types')
    rope_theta = _records.number(record['rope_theta'], 'rope_theta')
    thetas = hf_config.get('layer_rope_theta')
    if thetas is None:
        refuse('layer_rope_theta missing', 'the reference reads one rotary base a layer, 0 for NoPE')
    stated = _records.numbers(thetas, 'layer_rope_theta')
    expected = tuple(0.0 if kind == 'full_attention' else rope_theta for kind in layer_types)
    if stated != expected:
        refuse(f'layer_rope_theta {stated!r}',
               'Dew rotates every sliding layer at rope_theta and none of the full ones')
    kinds = kinds_of(record)
    for kind, nope in (('sliding_attention', False), ('full_attention', True)):
        kinds.setdefault(kind, native_fields(LayerKind)())['mixer'] = {
            'class': 'attention', 'fields': {'gate_projection': 'own', 'qk_norm_weight': False, 'nope': nope}}
    record['kinds'] = kinds
    return record


def _qk_scale(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    used.add('qk_scale_factor')
    head_dim = _records.integer(record['head_dim'], 'head_dim')
    record['attention_scale'] = (_records.number(hf_config.get('qk_scale_factor', 3.87), 'qk_scale_factor')
                                 * head_dim ** -0.5)
    return record


def _post_norms(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    used.add('post_norm_eps')
    post_eps = _records.number(hf_config.get('post_norm_eps', 1e-8), 'post_norm_eps')
    record['norms'] = {'attention_output_norm': {'eps': post_eps}, 'mlp_output_norm': {'eps': post_eps},
                       'embedding_layernorm': {'scale': False}, 'norm': {'offset': False}}
    return record


MUSE_TEXT_CONFIG = ConfigLayout(
    # Each layer rotates at its own base (`_layer_bases`); a ramp over them stays refused until a check
    # holds one against MuseGlimmerTextRotaryEmbedding.
    decoder_layout(DecoderSpelling(qk_norm=True, scale_after_cast=False, tie_embeddings=True,
                                   ramps=('llama3',), activation='hidden_activation')),
    ConfigPart(_layer_bases, lambda model: {'layer_rope_theta': [
        0 if kind == 'full_attention' else model.rope_theta for kind in model.per_layer_types]}),
    ConfigPart(_qk_scale, lambda model: {
        'qk_scale_factor': model.attention_scale * model.features_per_head ** 0.5}),
    ConfigPart(_post_norms, lambda model: {'post_norm_eps': (model.norms or {}).get(
        'attention_output_norm', NormSpec()).eps or model.norm_eps}),
    ConfigField('final_logit_softcapping', 'final_logit_softcap', 20.0, parse=_records.number),
    ConfigField('output_multiplier', 'logits_scaling', 0.19611613513818404,
                parse=lambda value, key: 1 / _records.number(value, key),
                spell=lambda scaling: 1 / _records.number(scaling, 'logits_scaling')),
    constant('sandwich_norms', value=True), constant('scale_offset', value=True),
    constant('output_gate', value=True),
    # MuseGlimmerTextNormedEmbedding norms the lookup; media take their slots after it.
    constant('embedding_norm', value=True), constant('embedding_norm_reads', value='lookup'),
    unread('initializer_range', 'attention_dropout', 'use_cache', 'pad_token_id', 'bos_token_id',
           'eos_token_id'))
"""Muse Glimmer's text config, both ways."""


MUSE_GLIMMER_TEXT = DecoderFamily(
    ('muse_glimmer_text',),
    MUSE_TEXT_CONFIG,
    lambda fields: bool(fields.output_gate and fields.sandwich_norms and fields.embedding_norm
                        and fields.embedding_norm_reads == 'lookup'),
    'muse_glimmer_text',
    'MuseGlimmerTextModel',
    preserve_source_layout=True,
    # MuseGlimmerTextConfig's defaults for the fields its decoder reads.
    defaults={'rms_norm_eps': 1e-5, 'tie_word_embeddings': False, 'num_key_value_heads': 2, 'head_dim': 128,
              'max_position_embeddings': 131072},
    padding_idx=True,
)
