"""Falcon-7B's one-norm parallel block and fused multi-query projection."""

from collections.abc import Mapping

from dew import _records
from dew.interop.decoder_config import DecoderSpelling, decoder_layout, expect, inert, ungated
from dew.interop.decoder_parts import MEGATRON_NAMES, DecoderFamily, DecoderFields, grouped_qkv, spelled
from dew.interop.layouts import ConfigField, ConfigLayout, ConfigPart, constant, unread


def _multi_query(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """One key and value head under `multi_query` (true where unstated), else one a query head."""
    used.update(('multi_query', 'num_kv_heads'))
    if hf_config.get('multi_query', True):
        record['num_kv_heads'] = 1
    return record


def _falcon_width(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    used.add('ffn_hidden_size')
    record['mlp_features'] = _records.integer(
        hf_config.get('ffn_hidden_size') or 4 * _records.integer(record['emb_features'], 'hidden_size'),
        'ffn_hidden_size')
    return record


def _falcon_bias(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    used.add('bias')
    biased = bool(hf_config.get('bias', False))
    record.update(mlp_bias=biased, attention_bias=biased)
    return record


FALCON_CONFIG = ConfigLayout(
    decoder_layout(DecoderSpelling(
        reads=frozenset(), tie_embeddings=True, scale_after_cast=False, activation=None,
        defaults={'hidden_size': 4544, 'num_attention_heads': 71},
        undeclared=frozenset({'num_key_value_heads', 'head_dim', 'intermediate_size', 'rms_norm_eps'}))),
    ConfigPart(_multi_query, lambda model: {'multi_query': model.kv_heads == 1}),
    ConfigPart(_falcon_width, lambda model: {'ffn_hidden_size': model.hidden_features}),
    ungated('activation', ('gelu', 'gelu_new', 'relu'), 'Falcon', default='gelu'),
    ConfigPart(_falcon_bias, lambda model: {'bias': model.attention_bias}),
    ConfigField('layer_norm_epsilon', 'norm_eps', 1e-5,
                parse=lambda value, key: _records.number(value or 1e-5, key)),
    ConfigField('attention_dropout', 'attention_dropout_rate', 0.0,
                parse=lambda value, key: _records.number(value or 0., key)),
    constant('norm_type', value='layer'), constant('norm_bias', value=True),
    constant('parallel_residual', value=True),
    constant('shared_parallel_norm', value=True),
    inert('alibi', 'this Falcon port rotates its heads, as Falcon-7B does', neutral=False),
    inert('new_decoder_architecture', 'Falcon-7B uses its one-norm decoder', neutral=False),
    inert('num_ln_in_parallel_attn', 'Falcon-7B uses its one-norm decoder', neutral=1),
    expect('parallel_attn', 'Falcon-7B sums attention and MLP before residual dropout', value=True),
    inert('hidden_dropout', 'Falcon drops the summed parallel branch; Falcon-7B uses zero dropout',
          neutral=0.0),
    ConfigPart(lambda hf_config, used, record: record, lambda model: {
        'alibi': False, 'new_decoder_architecture': False, 'parallel_attn': True,
        'num_ln_in_parallel_attn': 1, 'hidden_dropout': model.dropout_rate}),
    unread('apply_residual_connection_post_layernorm'))
"""Falcon-7B's config, both ways: a rotary, parallel, one-norm block."""


FALCON = DecoderFamily(
    ('falcon',), FALCON_CONFIG,
    lambda fields: fields.shared_parallel_norm and fields.mlp == 'gelu_exact'
                   and fields.partial_rotary_factor is None,
    'falcon', 'FalconForCausalLM',
    routes=spelled(MEGATRON_NAMES),
    packed=grouped_qkv('.self_attention.query_key_value'),
    preserve_source_layout=False,
    tied_head_names=('lm_head.weight', 'transformer.word_embeddings.weight'),
    defaults={'max_position_embeddings': 2048},
)
