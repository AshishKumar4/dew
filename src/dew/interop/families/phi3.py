"""Phi-3's fused qkv/gate-up projections and short/long rotary tables."""

import dataclasses
from collections.abc import Mapping

from dew import records
from dew.interop.config_records import native_fields
from dew.interop.decoder_config import DecoderFields, _base_config, _refuse
from dew.interop.decoder_paths import Packed, _dew_path
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.rope import LongRopeScaling


def _phi3_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    parameters = records.record(hf.get('rope_parameters') or hf.get('rope_scaling') or {}, 'rope_parameters')
    theta = records.number(parameters.get('rope_theta', hf.get('rope_theta', 10000.)), 'rope_theta')
    partial = records.number(parameters.get('partial_rotary_factor', hf.get('partial_rotary_factor', 1.)),
                              'partial_rotary_factor')
    rotated = int(records.integer(hf.get('head_dim') or records.integer(hf['hidden_size'], 'hidden_size')
                                   // records.integer(hf['num_attention_heads'], 'num_attention_heads'),
                                   'head_dim') * partial)
    if not 0 < partial <= 1 or rotated < 2 or rotated % 2:
        _refuse('partial_rotary_factor', 'the rotated head width must be positive and even')
    used.update(('original_max_position_embeddings', 'resid_pdrop', 'embd_pdrop', 'attention_dropout',
                 'partial_rotary_factor'))
    used.update(('full_attn_mod', 'interpolate_factor', 'lm_head_bias'))
    if (hf.get('full_attn_mod', 1) != 1 or hf.get('interpolate_factor', 1) != 1
            or hf.get('lm_head_bias', False)):
        _refuse('full_attn_mod/interpolate_factor/lm_head_bias', 'Phi-3 uses one window and no head bias')
    layers = records.integer(hf['num_hidden_layers'], 'num_hidden_layers')
    config = _base_config({**hf, 'rope_scaling': None,
                           'rope_parameters': {'rope_type': 'default', 'rope_theta': theta}}, used,
                          layer_types=('sliding_attention' if hf.get('sliding_window') is not None
                                       else 'full_attention',) * layers,
                          reads=frozenset({'sliding_window'}))
    rope_type = parameters.get('rope_type', parameters.get('type', 'default'))
    if rope_type in ('su', 'yarn', 'longrope'):
        original = records.integer(parameters.get('original_max_position_embeddings',
                                   hf.get('original_max_position_embeddings',
                                          hf.get('max_position_embeddings', 4096))),
                                   'original_max_position_embeddings')
        rope = native_fields(LongRopeScaling)(
            rope_type='longrope',
            short_factor=records.numbers(parameters['short_factor'], 'short_factor'),
            long_factor=records.numbers(parameters['long_factor'], 'long_factor'),
            original_max_position_embeddings=original,
            factor=records.number(parameters.get('factor',
                                   records.integer(hf.get('max_position_embeddings', 4096),
                                                   'max_position_embeddings') / original), 'factor'),
            attention_factor=(None if parameters.get('attention_factor') is None else
                              records.number(parameters['attention_factor'], 'attention_factor')),
        )
        if len(rope.value.short_factor) != rotated // 2:
            _refuse('short_factor/long_factor', 'LongRoPE has one factor per rotated head-dimension pair')
        config['rope_scaling'] = rope
        extra = set(parameters) - {'rope_type', 'type', 'rope_theta', 'partial_rotary_factor',
                                   'short_factor', 'long_factor', 'original_max_position_embeddings',
                                   'factor', 'attention_factor'}
        if extra:
            _refuse(f'rope_parameters fields {sorted(extra)}',
                    'Phi-3 reads the LongRoPE factors and amplitude')
    elif (rope_type != 'default'
          or set(parameters) - {'rope_type', 'type', 'rope_theta', 'partial_rotary_factor'}):
        _refuse('rope_parameters', 'Phi-3 uses default or longrope (the older su/yarn spelling)')
    elif partial != 1 and hf.get('sliding_window') is not None:
        _refuse('partial_rotary_factor', 'partial rotary on sliding layers requires its LongRoPE factors')
    config.update({
        'dropout_rate': records.number(hf.get('resid_pdrop', 0.), 'resid_pdrop'),
        'embedding_dropout_rate': records.number(hf.get('embd_pdrop', 0.), 'embd_pdrop'),
        'attention_dropout_rate': records.number(hf.get('attention_dropout', 0.), 'attention_dropout'),
        'partial_rotary_factor': None if partial == 1 else partial,
        'partial_rotary_type': 'default',
    })
    return config


def _qkv_widths(config: Mapping[str, object]) -> tuple[int, ...]:
    head_dim = records.integer(config['head_dim'], 'head_dim')
    query = records.integer(config['num_heads'], 'num_heads') * head_dim
    kv = records.integer(config['num_kv_heads'], 'num_kv_heads') * head_dim
    return query, kv, kv


_PHI3_PACKED = (
    Packed('.self_attn.qkv_proj.weight', tuple(f'.self_attn.{part}.weight'
           for part in ('q_proj', 'k_proj', 'v_proj')), 0, widths=_qkv_widths),
    Packed('.mlp.gate_up_proj.weight', ('.mlp.gate_proj.weight', '.mlp.up_proj.weight'), 0))


def _phi3_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    if name.endswith(tuple(packing.name for packing in _PHI3_PACKED)):
        return None
    return _dew_path(name, config)


def _phi3_export(model: CausalTransformer) -> Mapping[str, object]:
    rope = model.rope_scaling
    return {
        'resid_pdrop': model.dropout_rate, 'embd_pdrop': model.embedding_dropout_rate,
        'attention_dropout': model.attention_dropout_rate,
        'original_max_position_embeddings': (rope.original_max_position_embeddings
                                             if isinstance(rope, LongRopeScaling) else model.max_seq_len),
        'rope_parameters': (dict(dataclasses.asdict(rope), rope_theta=model.rope_theta,
                                 partial_rotary_factor=model.partial_rotary_factor or 1.)
                            if isinstance(rope, LongRopeScaling) else
                            {'rope_type': 'default', 'rope_theta': model.rope_theta,
                             'partial_rotary_factor': model.partial_rotary_factor or 1.}),
        'rope_scaling': None,
        'layer_types': None,
        'pad_token_id': None,
        'eos_token_id': None,
    }
