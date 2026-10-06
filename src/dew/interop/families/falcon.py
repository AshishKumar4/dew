"""Falcon-7B's one-norm parallel block and fused multi-query projection."""

from collections.abc import Mapping

import numpy as np

from dew import records
from dew.interop.decoder_config import DecoderFields, _base_config, _refuse
from dew.interop.decoder_paths import Renames, _renamed_path
from dew.interop.families.gpt_neox import _gpt_neox_export_weights, _gpt_neox_prepare
from dew.interop.safetensors_io import LazyTensors
from dew.nn.backbones.causal_transformer import CausalTransformer


def _falcon_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    used.update(('alibi', 'new_decoder_architecture', 'multi_query', 'parallel_attn', 'bias',
                 'num_kv_heads', 'num_ln_in_parallel_attn', 'hidden_dropout', 'attention_dropout',
                 'ffn_hidden_size', 'activation', 'layer_norm_epsilon',
                 'apply_residual_connection_post_layernorm'))
    if hf.get('alibi', False):
        _refuse('alibi', 'this Falcon port rotates its heads, as Falcon-7B does')
    if hf.get('new_decoder_architecture', False) or hf.get('num_ln_in_parallel_attn') not in (None, 1):
        _refuse('new_decoder_architecture/num_ln_in_parallel_attn', 'Falcon-7B uses its one-norm decoder')
    if not hf.get('parallel_attn', True):
        _refuse('parallel_attn', 'Falcon-7B sums attention and MLP before residual dropout')
    if hf.get('hidden_dropout', 0.):
        _refuse('hidden_dropout', 'Falcon drops the summed parallel branch; Falcon-7B uses zero dropout')
    hidden = records.integer(hf.get('hidden_size', 4544), 'hidden_size')
    heads = records.integer(hf.get('num_attention_heads', 71), 'num_attention_heads')
    activation = records.text(hf.get('activation', 'gelu') or 'gelu', 'activation')
    activations = {'gelu': 'gelu_exact', 'gelu_new': 'gelu', 'relu': 'relu'}
    if activation not in activations:
        _refuse('activation', 'the ungated Falcon MLP supports GELU and ReLU')
    config = _base_config({**hf, 'hidden_size': hidden, 'num_attention_heads': heads,
                           'num_key_value_heads': 1 if hf.get('multi_query', True) else heads,
                           'intermediate_size': hf.get('ffn_hidden_size') or 4 * hidden,
                           'hidden_act': 'silu'}, used, reads=frozenset(), tie_embeddings=True)
    config.update({
        'mlp': activations[activation], 'mlp_bias': bool(hf.get('bias', False)),
        'attention_bias': bool(hf.get('bias', False)),
        'norm_type': 'layer', 'norm_bias': True, 'scale_after_cast': False,
        'norm_eps': records.number(hf.get('layer_norm_epsilon', 1e-5) or 1e-5, 'layer_norm_epsilon'),
        'parallel_residual': True, 'shared_parallel_norm': True,
        'dropout_rate': records.number(hf.get('hidden_dropout', 0.) or 0., 'hidden_dropout'),
        'attention_dropout_rate': records.number(hf.get('attention_dropout', 0.) or 0., 'attention_dropout'),
    })
    return config


_FALCON_NAMES: Renames = (
    ('transformer.word_embeddings', 'model.embed_tokens'), ('transformer.ln_f', 'model.norm'),
    ('transformer.h', 'model.layers'), ('self_attention.dense', 'self_attn.o_proj'),
    ('mlp.dense_h_to_4h', 'mlp.up_proj'), ('mlp.dense_4h_to_h', 'mlp.down_proj'))


def _falcon_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    if '.self_attention.query_key_value.' in name:
        return None
    return _renamed_path(_FALCON_NAMES, name, config)


def _falcon_prepare(tensors: Mapping[str, np.ndarray],
                    config: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
    if config is None:
        raise ValueError('Falcon fused qkv requires translated head geometry')
    return _gpt_neox_prepare(tensors, config, attention_name='self_attention',
                              interleaved=config['num_kv_heads'] != 1)


def _falcon_export_weights(model: CausalTransformer, variables: Mapping[str, object],
                           config: Mapping[str, object]) -> LazyTensors:
    return _gpt_neox_export_weights(model, variables, config, attention_name='self_attention',
                                    interleaved=model.kv_heads != 1)


def _falcon_export(model: CausalTransformer) -> Mapping[str, object]:
    return {
        'alibi': False, 'new_decoder_architecture': False, 'multi_query': model.kv_heads == 1,
        'parallel_attn': True, 'num_ln_in_parallel_attn': 1, 'bias': model.attention_bias,
        'ffn_hidden_size': model.hidden_features,
        'activation': {'gelu_exact': 'gelu', 'gelu': 'gelu_new', 'relu': 'relu'}[str(model.mlp)],
        'layer_norm_epsilon': model.norm_eps, 'hidden_dropout': model.dropout_rate,
        'attention_dropout': model.attention_dropout_rate,
        'num_key_value_heads': None, 'head_dim': None, 'intermediate_size': None,
        'hidden_act': None, 'rms_norm_eps': None, 'attention_bias': None,
    }
