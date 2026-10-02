"""GPT-NeoX's parallel biased LayerNorm block and head-interleaved qkv storage."""

from collections.abc import Mapping

import numpy as np

from dew import records
from dew.interop.hf_decoders import DecoderFields, _base_config, _dew_path, _refuse
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.rope import inverse_frequencies


def _gpt_neox_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    parameters = records.record(hf.get('rope_parameters') or {}, 'rope_parameters')
    partial = records.number(parameters.get('partial_rotary_factor', hf.get('rotary_pct', .25)), 'rotary_pct')
    theta = records.number(parameters.get('rope_theta', hf.get('rotary_emb_base', 10000.)), 'rotary_emb_base')
    activation = records.text(hf.get('hidden_act', 'gelu'), 'hidden_act')
    activations = {'gelu': 'gelu_exact', 'gelu_new': 'gelu', 'gelu_pytorch_tanh': 'gelu', 'relu': 'relu'}
    if activation not in activations:
        _refuse(f'hidden_act={activation!r}', 'GPT-NeoX requires an ungated GELU or ReLU MLP')
    rope = {key: value for key, value in parameters.items() if key != 'partial_rotary_factor'}
    config = _base_config({**hf, 'hidden_act': 'gelu', 'rope_theta': theta,
                           'rope_parameters': rope or None}, used, reads=frozenset({'attention_bias'}))
    used.update(('rotary_pct', 'rotary_emb_base', 'use_parallel_residual', 'layer_norm_eps',
                 'hidden_dropout', 'classifier_dropout', 'is_decoder'))
    config.update({
        'mlp': activations[activation], 'mlp_bias': True,
        'norm_type': 'layer', 'norm_bias': True,
        'norm_eps': records.number(hf.get('layer_norm_eps', 1e-5), 'layer_norm_eps'),
        'scale_after_cast': False, 'attention_bias': bool(hf.get('attention_bias', True)),
        'partial_rotary_factor': partial, 'partial_rotary_type': 'default',
        'parallel_residual': bool(hf.get('use_parallel_residual', True)),
        'dropout_rate': records.number(hf.get('hidden_dropout', 0.), 'hidden_dropout'),
        'embedding_dropout_rate': records.number(hf.get('hidden_dropout', 0.), 'hidden_dropout'),
        'attention_dropout_rate': records.number(hf.get('attention_dropout', 0.), 'attention_dropout'),
    })
    return config


def _gpt_neox_prepare(tensors: Mapping[str, np.ndarray],
                       config: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
    if config is None:
        raise ValueError('GPT-NeoX fused qkv preparation requires translated num_heads and head_dim')
    heads = records.integer(config['num_heads'], 'num_heads')
    dimension = records.integer(config['head_dim'], 'head_dim')
    prepared = {}
    for name, tensor in tensors.items():
        if name.endswith('.attention.bias'):
            if (tensor.ndim != 4 or tensor.shape[:2] != (1, 1)
                    or tensor.shape[-2] != tensor.shape[-1]
                    or not np.array_equal(tensor, np.tril(np.ones_like(tensor)))):
                raise ValueError(f'{name} must hold the fixed triangular causal mask')
            continue
        if name.endswith('.attention.masked_bias'):
            # FP16 releases cast the historical -1e9 sentinel to -inf.
            if tensor.shape != () or float(tensor) not in (-1e9, -np.inf):
                raise ValueError(f'{name} must hold the historical negative mask sentinel')
            continue
        if name.endswith('.attention.rotary_emb.inv_freq'):
            rotated = int(dimension * records.number(config.get('partial_rotary_factor', 1.), 'partial_rotary_factor'))
            expected = inverse_frequencies(records.number(config.get('rope_theta', 10000.), 'rope_theta'),
                                           rotated, dtype=np.float32).astype(tensor.dtype)
            if not np.array_equal(tensor, expected):
                raise ValueError(f'{name} disagrees with the configured rotary frequencies in its stored dtype')
            continue
        if '.attention.query_key_value.' not in name:
            prepared[name] = tensor
            continue
        shape = (heads, 3, dimension, *tensor.shape[1:])
        grouped = tensor.reshape(shape)
        for index, part in enumerate(('q_proj', 'k_proj', 'v_proj')):
            prepared[name.replace('attention.query_key_value', f'self_attn.{part}')] = (
                grouped[:, index].reshape(heads * dimension, *tensor.shape[1:]))
    return prepared


def _gpt_neox_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    if '.attention.query_key_value.' in name or name.endswith((
            '.attention.bias', '.attention.masked_bias', '.attention.rotary_emb.inv_freq')):
        return None
    if name.startswith('embed_out.'):
        name = name.replace('embed_out.', 'lm_head.')
    name = name.replace('gpt_neox.embed_in.', 'model.embed_tokens.')
    name = name.replace('gpt_neox.final_layer_norm.', 'model.norm.')
    name = name.replace('gpt_neox.layers.', 'model.layers.')
    name = name.replace('.attention.dense.', '.self_attn.o_proj.')
    name = name.replace('.mlp.dense_h_to_4h.', '.mlp.up_proj.')
    name = name.replace('.mlp.dense_4h_to_h.', '.mlp.down_proj.')
    return _dew_path(name, config)


def _gpt_neox_export(model: CausalTransformer) -> Mapping[str, object]:
    activation = model.mlp
    if not isinstance(activation, str) or activation not in ('gelu', 'gelu_exact', 'relu'):
        _refuse('mlp', 'GPT-NeoX requires an ungated MLP')
    fields: dict[str, object] = {
        'layer_norm_eps': model.norm_eps, 'use_parallel_residual': model.parallel_residual,
        'hidden_act': {'gelu': 'gelu_new', 'gelu_exact': 'gelu', 'relu': 'relu'}[activation],
        'hidden_dropout': model.dropout_rate, 'attention_dropout': model.attention_dropout_rate,
        'rope_parameters': {'rope_type': 'default', 'rope_theta': model.rope_theta,
                            'partial_rotary_factor': model.partial_rotary_factor or 1.},
    }
    fields.update(dict.fromkeys(('rms_norm_eps', 'rope_theta', 'num_key_value_heads', 'head_dim')))
    return fields


def _gpt_neox_export_weights(model: CausalTransformer, variables: Mapping[str, object],
                            config: Mapping[str, object]) -> Mapping[str, np.ndarray]:
    from flax.traverse_util import flatten_dict

    from dew.interop.hf_decoders import _hf_name

    params = variables.get('params')
    if not isinstance(params, Mapping):
        raise ValueError('params must contain the decoder parameter tree')
    flat = flatten_dict(dict(params), sep='.')
    tensors = {}
    for name, value in flat.items():
        if '.self_attn.' in name and any(f'.{part}.' in name for part in ('q_proj', 'k_proj', 'v_proj')):
            continue
        target = _hf_name(name, config)
        if target is None:
            continue
        target = target.replace('model.embed_tokens.', 'gpt_neox.embed_in.')
        target = target.replace('model.norm.', 'gpt_neox.final_layer_norm.')
        target = target.replace('model.layers.', 'gpt_neox.layers.')
        target = target.replace('lm_head.', 'embed_out.')
        target = target.replace('.self_attn.o_proj.', '.attention.dense.')
        target = target.replace('.mlp.up_proj.', '.mlp.dense_h_to_4h.')
        target = target.replace('.mlp.down_proj.', '.mlp.dense_4h_to_h.')
        tensors[target] = np.asarray(value).T if name.endswith('.kernel') else np.asarray(value)
    for index in range(model.num_layers):
        for leaf, suffix in (('kernel', 'weight'), ('bias', 'bias')):
            if f'layers_{index}.self_attn.q_proj.{leaf}' not in flat:
                continue
            values = [np.asarray(flat[f'layers_{index}.self_attn.{part}.{leaf}'])
                      for part in ('q_proj', 'k_proj', 'v_proj')]
            values = [value.T if leaf == 'kernel' else value for value in values]
            grouped = [value.reshape(model.num_heads, model.features_per_head, *value.shape[1:])
                       for value in values]
            stored = np.stack(grouped, axis=1)
            tensors[f'gpt_neox.layers.{index}.attention.query_key_value.{suffix}'] = stored.reshape(
                3 * model.emb_features, *values[0].shape[1:])
    return tensors
