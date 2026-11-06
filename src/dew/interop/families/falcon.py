"""Falcon-7B's one-norm parallel block and fused multi-query projection."""

from collections.abc import Mapping
from functools import partial

from dew import _records
from dew.interop.decoder_parts import (
    MEGATRON_NAMES,
    PLAIN_SPELLING,
    DecoderFamily,
    DecoderFields,
    base_config,
    decoder_tensors,
    grouped_qkv,
    hf_ungated,
    refuse,
    renamed_name,
    renamed_path,
    ungated_mlp,
)
from dew.nn.backbones.causal_transformer import CausalTransformer


def _falcon_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    used.update(('alibi', 'new_decoder_architecture', 'multi_query', 'parallel_attn', 'bias',
                 'num_kv_heads', 'num_ln_in_parallel_attn', 'hidden_dropout', 'attention_dropout',
                 'ffn_hidden_size', 'activation', 'layer_norm_epsilon',
                 'apply_residual_connection_post_layernorm'))
    if hf.get('alibi', False):
        refuse('alibi', 'this Falcon port rotates its heads, as Falcon-7B does')
    if hf.get('new_decoder_architecture', False) or hf.get('num_ln_in_parallel_attn') not in (None, 1):
        refuse('new_decoder_architecture/num_ln_in_parallel_attn', 'Falcon-7B uses its one-norm decoder')
    if not hf.get('parallel_attn', True):
        refuse('parallel_attn', 'Falcon-7B sums attention and MLP before residual dropout')
    if hf.get('hidden_dropout', 0.):
        refuse('hidden_dropout', 'Falcon drops the summed parallel branch; Falcon-7B uses zero dropout')
    hidden = _records.integer(hf.get('hidden_size', 4544), 'hidden_size')
    heads = _records.integer(hf.get('num_attention_heads', 71), 'num_attention_heads')
    activation = _records.text(hf.get('activation', 'gelu') or 'gelu', 'activation')
    mlp = ungated_mlp(activation, ('gelu', 'gelu_new', 'relu'), 'activation', 'Falcon')
    config = base_config({**hf, 'hidden_size': hidden, 'num_attention_heads': heads,
                           'num_key_value_heads': 1 if hf.get('multi_query', True) else heads,
                           'intermediate_size': hf.get('ffn_hidden_size') or 4 * hidden,
                           'hidden_act': 'silu'}, used, rope=PLAIN_SPELLING, reads=frozenset(),
                         tie_embeddings=True)
    config.update({
        'mlp': mlp, 'mlp_bias': bool(hf.get('bias', False)),
        'attention_bias': bool(hf.get('bias', False)),
        'norm_type': 'layer', 'norm_bias': True, 'scale_after_cast': False,
        'norm_eps': _records.number(hf.get('layer_norm_epsilon', 1e-5) or 1e-5, 'layer_norm_epsilon'),
        'parallel_residual': True, 'shared_parallel_norm': True,
        'dropout_rate': _records.number(hf.get('hidden_dropout', 0.) or 0., 'hidden_dropout'),
        'attention_dropout_rate': _records.number(hf.get('attention_dropout', 0.) or 0., 'attention_dropout'),
    })
    return config


def _falcon_export(model: CausalTransformer) -> Mapping[str, object]:
    return {
        'alibi': False, 'new_decoder_architecture': False, 'multi_query': model.kv_heads == 1,
        'parallel_attn': True, 'num_ln_in_parallel_attn': 1, 'bias': model.attention_bias,
        'ffn_hidden_size': model.hidden_features,
        'activation': hf_ungated(model.mlp, 'Falcon'),
        'layer_norm_epsilon': model.norm_eps, 'hidden_dropout': model.dropout_rate,
        'attention_dropout': model.attention_dropout_rate,
        'num_key_value_heads': None, 'head_dim': None, 'intermediate_size': None,
        'hidden_act': None, 'rms_norm_eps': None, 'attention_bias': None,
    }


FALCON = DecoderFamily(
    ('falcon',), _falcon_config,
    lambda fields: fields.shared_parallel_norm and fields.mlp == 'gelu_exact'
                   and fields.partial_rotary_factor is None,
    'falcon', 'FalconForCausalLM', _falcon_export,
    weight_path=partial(renamed_path, MEGATRON_NAMES), export_path=partial(renamed_name, MEGATRON_NAMES),
    packed=grouped_qkv('.self_attention.query_key_value'), export_weights=decoder_tensors,
    preserve_source_layout=False,
    tied_head_names=('lm_head.weight', 'transformer.word_embeddings.weight'),
    defaults={'max_position_embeddings': 2048},
)
