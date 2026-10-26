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
from dew.interop.decoder_parts import (
    PLAIN_SPELLING,
    DecoderFamily,
    DecoderFields,
    base_config,
    hf_activation,
    kinds_of,
    read_rope,
    refuse,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import NormSpec
from dew.nn.backbones.layer_plan import LayerKind


def _muse_text_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Muse Glimmer text config into `CausalTransformer` fields."""
    # Each layer rotates at its own base (layer_rope_theta, below); a ramp over them stays refused
    # until a check holds one against MuseGlimmerTextRotaryEmbedding.
    config = base_config(hf_config, used, rope=read_rope(hf_config, used, PLAIN_SPELLING, ramps=('llama3',)),
                         qk_norm=True, scale_after_cast=False, tie_embeddings=True)
    layer_types = _records.strings(config['layer_types'], 'layer_types')
    rope_theta = _records.number(config['rope_theta'], 'rope_theta')
    thetas = hf_config.get('layer_rope_theta')
    if thetas is None:
        refuse('layer_rope_theta missing', 'the reference reads one rotary base a layer, 0 for NoPE')
    stated = _records.numbers(thetas, 'layer_rope_theta')
    expected = tuple(0.0 if kind == 'full_attention' else rope_theta for kind in layer_types)
    if stated != expected:
        refuse(f'layer_rope_theta {stated!r}',
               'Dew rotates every sliding layer at rope_theta and none of the full ones')
    head_dim = _records.integer(config['head_dim'], 'head_dim')
    post_eps = _records.number(hf_config.get('post_norm_eps', 1e-8), 'post_norm_eps')
    kinds = kinds_of(config)
    for kind, nope in (('sliding_attention', False), ('full_attention', True)):
        kinds.setdefault(kind, native_fields(LayerKind)())['mixer'] = {
            'class': 'attention', 'fields': {'gate_projection': 'own', 'qk_norm_weight': False, 'nope': nope}}
    config.update(
        sandwich_norms=True, scale_offset=True, output_gate=True,
        attention_scale=_records.number(hf_config.get('qk_scale_factor', 3.87), 'qk_scale_factor')
        * head_dim ** -0.5,
        embedding_norm=True,
        norms={'attention_output_norm': {'eps': post_eps}, 'mlp_output_norm': {'eps': post_eps},
               'embedding_layernorm': {'scale': False}, 'norm': {'offset': False}},
        final_logit_softcap=_records.number(hf_config.get('final_logit_softcapping', 20.0),
                                           'final_logit_softcapping'),
        logits_scaling=1 / _records.number(hf_config.get('output_multiplier', 0.19611613513818404),
                                          'output_multiplier'))
    used.update(('layer_rope_theta', 'post_norm_eps', 'qk_scale_factor', 'final_logit_softcapping',
                 'output_multiplier', 'initializer_range', 'attention_dropout', 'use_cache',
                 'pad_token_id', 'bos_token_id', 'eos_token_id'))
    return config


def _muse_text_export(model: CausalTransformer) -> Mapping[str, object]:
    """The config fields `_muse_text_config` reads back, beside the shared ones."""
    return {
        'layer_rope_theta': [0 if kind == 'full_attention' else model.rope_theta
                             for kind in model.per_layer_types],
        'post_norm_eps': (model.norms or {}).get('attention_output_norm', NormSpec()).eps or model.norm_eps,
        'qk_scale_factor': model.attention_scale * model.features_per_head ** 0.5,
        'final_logit_softcapping': model.final_logit_softcap,
        'output_multiplier': 1 / model.logits_scaling,
        'hidden_activation': hf_activation(model.mlp),
    }


MUSE_GLIMMER_TEXT = DecoderFamily(
    ('muse_glimmer_text',),
    _muse_text_config,
    lambda fields: bool(fields.output_gate and fields.sandwich_norms and fields.embedding_norm),
    'muse_glimmer_text',
    'MuseGlimmerTextModel',
    _muse_text_export,
    sandwich_norms=True,
    preserve_source_layout=True,
    # MuseGlimmerTextConfig's defaults for the fields its decoder reads.
    defaults={'rms_norm_eps': 1e-5, 'tie_word_embeddings': False, 'num_key_value_heads': 2, 'head_dim': 128,
              'max_position_embeddings': 131072},
    padding_idx=True,
)
