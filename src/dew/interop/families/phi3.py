"""Phi-3's fused qkv/gate-up projections and short/long rotary tables."""

import dataclasses
from collections.abc import Mapping

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_config import DecoderSpelling, decoder_layout, expect, inert
from dew.interop.decoder_parts import (
    PLAIN_SPELLING,
    DecoderFamily,
    DecoderFields,
    Packed,
    qkv_widths,
    refuse,
    rope_records,
)
from dew.interop.layouts import ConfigField, ConfigLayout, ConfigPart
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.rope import LongRopeScaling


def _phi3_rope(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    # Phi3Config resolves its record as the mixin does, a legacy rope_scaling
    # replacing rope_parameters whole (configuration_phi3.py:89-101).
    (field, parameters), sliding = rope_records(hf_config, PLAIN_SPELLING)
    if sliding is not None:
        # Phi3Config's model applies one rotary to every layer, so a second record has no layer to take it.
        refuse(sliding[0], 'Phi-3 applies one rotary record to every layer')
    used.update(('rope_theta', 'rope_parameters', 'rope_scaling', 'partial_rotary_factor',
                 'original_max_position_embeddings'))
    theta = _records.number(parameters['rope_theta'], f'{field} rope_theta')
    partial = _records.number(parameters.get('partial_rotary_factor',
                                             hf_config.get('partial_rotary_factor', 1.)),
                              'partial_rotary_factor')
    rotated = int(_records.integer(record['head_dim'], 'head_dim') * partial)
    if not 0 < partial <= 1 or rotated < 2 or rotated % 2:
        refuse('partial_rotary_factor', 'the rotated head width must be positive and even')
    rope_type = parameters.get('rope_type', parameters.get('type', 'default'))
    if rope_type in ('su', 'yarn', 'longrope'):
        # The class's own field, 4096 when omitted, replaces whatever the record
        # names (modeling_rope_utils.py:790-797, standardize_rope_params).
        original = _records.integer(hf_config.get('original_max_position_embeddings', 4096),
                                    'original_max_position_embeddings')
        rope = native_fields(LongRopeScaling)(
            rope_type='longrope',
            short_factor=_records.numbers(parameters['short_factor'], 'short_factor'),
            long_factor=_records.numbers(parameters['long_factor'], 'long_factor'),
            original_max_position_embeddings=original,
            factor=_records.number(parameters.get('factor',
                                   _records.integer(hf_config.get('max_position_embeddings', 4096),
                                                    'max_position_embeddings') / original), 'factor'),
            attention_factor=(None if parameters.get('attention_factor') is None else
                              _records.number(parameters['attention_factor'], 'attention_factor')),
        )
        if len(rope.value.short_factor) != rotated // 2:
            refuse('short_factor/long_factor', 'LongRoPE has one factor per rotated head-dimension pair')
        record['rope_scaling'] = rope
        extra = set(parameters) - {'rope_type', 'type', 'rope_theta', 'partial_rotary_factor',
                                   'short_factor', 'long_factor', 'original_max_position_embeddings',
                                   'factor', 'attention_factor'}
        if extra:
            refuse(f'rope_parameters fields {sorted(extra)}',
                   'Phi-3 reads the LongRoPE factors and amplitude')
    elif (rope_type != 'default'
          or set(parameters) - {'rope_type', 'type', 'rope_theta', 'partial_rotary_factor'}):
        refuse('rope_parameters', 'Phi-3 uses default or longrope (the older su/yarn spelling)')
    elif partial != 1 and 'sliding_attention' in _records.strings(record['layer_types'], 'layer_types'):
        refuse('partial_rotary_factor', 'partial rotary on sliding layers requires its LongRoPE factors')
    record['rope_theta'] = theta
    record['partial_rotary_factor'] = None if partial == 1 else partial
    record['partial_rotary_type'] = 'default'
    return record


def _phi3_rope_written(model: CausalTransformer) -> Mapping[str, object]:
    rope = model.rope_scaling
    return {
        'original_max_position_embeddings': (rope.original_max_position_embeddings
                                             if isinstance(rope, LongRopeScaling) else model.max_seq_len),
        'rope_parameters': (dict(dataclasses.asdict(rope), rope_theta=model.rope_theta,
                                 partial_rotary_factor=model.partial_rotary_factor or 1.)
                            if isinstance(rope, LongRopeScaling) else
                            {'rope_type': 'default', 'rope_theta': model.rope_theta,
                             'partial_rotary_factor': model.partial_rotary_factor or 1.}),
    }


PHI3_CONFIG = ConfigLayout(
    # Phi3Model masks every layer by the window when one is set.
    decoder_layout(DecoderSpelling(reads=frozenset({'sliding_window'}), windowed=True),
                   rope=ConfigPart(_phi3_rope, _phi3_rope_written)),
    expect('full_attn_mod', 'Phi-3 uses one window', value=1),
    expect('interpolate_factor', 'Phi-3 uses one window', value=1),
    inert('lm_head_bias', 'Phi-3 uses no head bias', neutral=False),
    ConfigField('resid_pdrop', 'dropout_rate', 0.0, parse=_records.number),
    ConfigField('embd_pdrop', 'embedding_dropout_rate', 0.0, parse=_records.number),
    ConfigField('attention_dropout', 'attention_dropout_rate', 0.0, parse=_records.number))
"""Phi-3's config, both ways: its one rotary record, plain or LongRoPE over a fraction of each head."""


_PHI3_PACKED = (
    Packed('.self_attn.qkv_proj.weight', tuple(f'.self_attn.{part}.weight'
           for part in ('q_proj', 'k_proj', 'v_proj')), 0, widths=qkv_widths),
    Packed('.mlp.gate_up_proj.weight', ('.mlp.gate_proj.weight', '.mlp.up_proj.weight'), 0))


PHI3 = DecoderFamily(
    ('phi3',), PHI3_CONFIG,
    lambda fields: isinstance(fields.rope_scaling, LongRopeScaling),
    'phi3', 'Phi3ForCausalLM',
    packed=_PHI3_PACKED, preserve_source_layout=True,
    # Phi3Config's norm epsilon and context.
    defaults={'rms_norm_eps': 1e-5, 'max_position_embeddings': 4096, 'pad_token_id': 32000},
    padding_idx=True,
)
