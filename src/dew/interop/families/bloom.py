"""BLOOM's embedding LayerNorm, head-interleaved qkv and ALiBi decoder."""

from collections.abc import Mapping

import numpy as np

from dew import records
from dew.interop.hf_decoders import DecoderFields, _dew_path, _refuse
from dew.nn.backbones.causal_transformer import CausalTransformer


def _bloom_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    hidden = records.integer(hf.get('hidden_size', hf.get('n_embed', 64)), 'hidden_size')
    heads = records.integer(hf.get('n_head', hf.get('num_attention_heads', 8)), 'n_head')
    used.update(('vocab_size', 'hidden_size', 'n_embed', 'n_layer', 'n_head',
                 'layer_norm_epsilon', 'hidden_dropout', 'attention_dropout',
                 'apply_residual_connection_post_layernorm', 'slow_but_exact', 'tie_word_embeddings',
                 'is_decoder', 'num_attention_heads'))
    if hf.get('apply_residual_connection_post_layernorm', False):
        _refuse('apply_residual_connection_post_layernorm', 'BLOOM normalized residuals are not represented')
    if hf.get('slow_but_exact', False):
        _refuse('slow_but_exact', 'the tensor-parallel sliced reduction is not represented')
    return {
        'vocab_size': records.integer(hf.get('vocab_size', 250880), 'vocab_size'),
        'emb_features': hidden, 'num_heads': heads, 'num_kv_heads': heads, 'head_dim': hidden // heads,
        'num_layers': records.integer(hf.get('n_layer', 2), 'n_layer'),
        'mlp_features': 4 * hidden, 'mlp': 'gelu', 'mlp_bias': True,
        'norm_type': 'layer', 'norm_bias': True,
        'norm_eps': records.number(hf.get('layer_norm_epsilon', 1e-5), 'layer_norm_epsilon'),
        'embedding_norm': True, 'qk_norm': False, 'attention_bias': True,
        'mixer': {'kind': 'attention', 'nope': True, 'alibi': True},
        'tie_embeddings': bool(hf.get('tie_word_embeddings', True)),
        'dropout_rate': records.number(hf.get('hidden_dropout', 0.), 'hidden_dropout'),
        'attention_dropout_rate': records.number(hf.get('attention_dropout', 0.), 'attention_dropout'),
    }


def _bloom_prepare(tensors: Mapping[str, np.ndarray],
                     config: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
    if config is None:
        raise ValueError('BLOOM fused qkv requires translated num_heads and head_dim')
    heads = records.integer(config['num_heads'], 'num_heads')
    dimension = records.integer(config['head_dim'], 'head_dim')
    prepared = {}
    for name, tensor in tensors.items():
        if '.self_attention.query_key_value.' not in name:
            prepared[name] = tensor
            continue
        grouped = tensor.reshape(heads, 3, dimension, *tensor.shape[1:])
        for index, part in enumerate(('q_proj', 'k_proj', 'v_proj')):
            prepared[name.replace('self_attention.query_key_value', f'self_attn.{part}')] = (
                grouped[:, index].reshape(heads * dimension, *tensor.shape[1:]))
    return prepared


def _bloom_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    if '.self_attention.query_key_value.' in name:
        return None
    if name.startswith('transformer.word_embeddings_layernorm.'):
        leaf = 'scale' if name.endswith('.weight') else 'bias'
        return ('params', 'embedding_layernorm', leaf)
    name = name.replace('transformer.word_embeddings.', 'model.embed_tokens.')
    name = name.replace('transformer.ln_f.', 'model.norm.').replace('transformer.h.', 'model.layers.')
    name = name.replace('.self_attention.dense.', '.self_attn.o_proj.')
    name = name.replace('.mlp.dense_h_to_4h.', '.mlp.up_proj.')
    name = name.replace('.mlp.dense_4h_to_h.', '.mlp.down_proj.')
    return _dew_path(name, config)


def _bloom_export(model: CausalTransformer) -> Mapping[str, object]:
    fields: dict[str, object] = {
        'n_layer': model.num_layers, 'n_head': model.num_heads,
        'layer_norm_epsilon': model.norm_eps, 'hidden_dropout': model.dropout_rate,
        'attention_dropout': model.attention_dropout_rate,
    }
    fields.update(dict.fromkeys(('num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
                                 'head_dim', 'intermediate_size', 'rms_norm_eps', 'hidden_act',
                                 'attention_bias', 'max_position_embeddings', 'rope_theta')))
    return fields


def _bloom_export_weights(model: CausalTransformer, variables: Mapping[str, object],
                         config: Mapping[str, object]) -> Mapping[str, np.ndarray]:
    from dew.interop.families.gpt_neox import _gpt_neox_export_weights

    params = variables.get('params')
    if not isinstance(params, Mapping) or not isinstance(params.get('embedding_layernorm'), Mapping):
        raise ValueError('BLOOM requires its embedding LayerNorm parameters')
    normalized = params['embedding_layernorm']
    native = _gpt_neox_export_weights(model, {'params': {
        name: value for name, value in params.items() if name != 'embedding_layernorm'}}, config)
    tensors = {}
    for name, value in native.items():
        name = name.replace('gpt_neox.embed_in.', 'transformer.word_embeddings.')
        name = name.replace('gpt_neox.final_layer_norm.', 'transformer.ln_f.')
        name = name.replace('gpt_neox.layers.', 'transformer.h.').replace('embed_out.', 'lm_head.')
        name = name.replace('.attention.', '.self_attention.')
        tensors[name] = value
    tensors['transformer.word_embeddings_layernorm.weight'] = np.asarray(normalized['scale'])
    tensors['transformer.word_embeddings_layernorm.bias'] = np.asarray(normalized['bias'])
    return tensors
