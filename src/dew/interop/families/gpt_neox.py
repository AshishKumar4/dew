"""GPT-NeoX's parallel biased LayerNorm block and head-interleaved qkv storage."""

from collections.abc import Mapping
from functools import partial

import jax
import numpy as np

from dew import records
from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    Renames,
    base_config,
    decoder_tensors,
    refuse,
    renamed_name,
    renamed_path,
)
from dew.interop.safetensors_io import LazyTensors
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.rope import inverse_frequencies


def _gpt_neox_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    parameters = records.record(hf.get('rope_parameters') or {}, 'rope_parameters')
    partial = records.number(parameters.get('partial_rotary_factor', hf.get('rotary_pct', .25)), 'rotary_pct')
    theta = records.number(parameters.get('rope_theta', hf.get('rotary_emb_base', 10000.)), 'rotary_emb_base')
    activation = records.text(hf.get('hidden_act', 'gelu'), 'hidden_act')
    activations = {'gelu': 'gelu_exact', 'gelu_new': 'gelu', 'gelu_pytorch_tanh': 'gelu', 'relu': 'relu'}
    if activation not in activations:
        refuse(f'hidden_act={activation!r}', 'GPT-NeoX requires an ungated GELU or ReLU MLP')
    rope = {key: value for key, value in parameters.items() if key != 'partial_rotary_factor'}
    config = base_config({**hf, 'hidden_act': 'gelu', 'rope_theta': theta,
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


def gpt_neox_prepare(tensors: Mapping[str, np.ndarray],
                       config: Mapping[str, object] | None = None, *,
                       attention_name: str = 'attention', interleaved: bool = True
                       ) -> Mapping[str, np.ndarray]:
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
            rotated = int(
                dimension * records.number(config.get("partial_rotary_factor", 1.0), "partial_rotary_factor")
            )
            expected = inverse_frequencies(records.number(config.get('rope_theta', 10000.), 'rope_theta'),
                                           rotated, dtype=np.float32).astype(tensor.dtype)
            if not np.array_equal(tensor, expected):
                raise ValueError(
                    f"{name} disagrees with the configured rotary frequencies in its stored dtype"
                )
            continue
        fused = f'{attention_name}.query_key_value'
        if f'.{fused}.' not in name:
            prepared[name] = tensor
            continue
        if interleaved:
            grouped = tensor.reshape(heads, 3, dimension, *tensor.shape[1:])
            pieces = [grouped[:, index].reshape(heads * dimension, *tensor.shape[1:])
                      for index in range(3)]
        else:
            kv = records.integer(config['num_kv_heads'], 'num_kv_heads') * dimension
            pieces = np.split(tensor, (heads * dimension, heads * dimension + kv), axis=0)
        for index, part in enumerate(('q_proj', 'k_proj', 'v_proj')):
            prepared[name.replace(fused, f'self_attn.{part}')] = pieces[index]
    return prepared


_GPT_NEOX_NAMES: Renames = (
    ('embed_out', 'lm_head'), ('gpt_neox.embed_in', 'model.embed_tokens'),
    ('gpt_neox.final_layer_norm', 'model.norm'), ('gpt_neox.layers', 'model.layers'),
    ('attention.dense', 'self_attn.o_proj'), ('mlp.dense_h_to_4h', 'mlp.up_proj'),
    ('mlp.dense_4h_to_h', 'mlp.down_proj'))


def _gpt_neox_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    if '.attention.query_key_value.' in name or name.endswith((
            '.attention.bias', '.attention.masked_bias', '.attention.rotary_emb.inv_freq')):
        return None
    return renamed_path(_GPT_NEOX_NAMES, name, config)


def _gpt_neox_export(model: CausalTransformer) -> Mapping[str, object]:
    activation = model.mlp
    if not isinstance(activation, str) or activation not in ('gelu', 'gelu_exact', 'relu'):
        refuse('mlp', 'GPT-NeoX requires an ungated MLP')
    fields: dict[str, object] = {
        'layer_norm_eps': model.norm_eps, 'use_parallel_residual': model.parallel_residual,
        'hidden_act': {'gelu': 'gelu_new', 'gelu_exact': 'gelu', 'relu': 'relu'}[activation],
        'hidden_dropout': model.dropout_rate, 'attention_dropout': model.attention_dropout_rate,
        'rope_parameters': {'rope_type': 'default', 'rope_theta': model.rope_theta,
                            'partial_rotary_factor': model.partial_rotary_factor or 1.},
    }
    fields.update(dict.fromkeys(('rms_norm_eps', 'rope_theta', 'num_key_value_heads', 'head_dim')))
    return fields


def gpt_neox_export_weights(family: DecoderFamily, model: CausalTransformer, variables: Mapping[str, object],
                            config: Mapping[str, object], *,
                             attention_name: str = 'attention', interleaved: bool = True) -> LazyTensors:
    """The shared writer's tensors with each layer's q, k and v interleaved by
    head into `query_key_value`, the inverse of `gpt_neox_prepare`."""
    tensors = decoder_tensors(family, model, variables, config)
    fused: dict[str, tuple[str, ...]] = {}
    for name in tensors:
        stem, found, leaf = name.partition('.self_attn.q_proj.')
        if found:
            fused[f'{stem}.{attention_name}.query_key_value.{leaf}'] = tuple(
                f'{stem}.self_attn.{part}.{leaf}' for part in ('q_proj', 'k_proj', 'v_proj'))
    parts = {part for names in fused.values() for part in names}
    specs = {name: spec for name, spec in tensors.specs.items() if name not in parts}
    for name, (query, *_) in fused.items():
        spec = tensors.specs[query]
        specs[name] = jax.ShapeDtypeStruct((sum(tensors.specs[part].shape[0] for part in fused[name]),
                                           *spec.shape[1:]), spec.dtype)

    def build(name: str) -> np.ndarray:
        if name not in fused:
            return tensors[name]
        values = [tensors[part] for part in fused[name]]
        if not interleaved:
            return np.concatenate(values, axis=0)
        grouped = np.stack([value.reshape(model.num_heads, model.features_per_head, *value.shape[1:])
                            for value in values], axis=1)
        return grouped.reshape(3 * values[0].shape[0], *values[0].shape[1:])

    return LazyTensors(specs, build)


GPT_NEOX = DecoderFamily(
    ('gpt_neox',),
    _gpt_neox_config,
    lambda fields: bool(
        fields.norm_type == 'layer'
        and fields.norm_bias
        and fields.mlp_bias
        and fields.position_embedding == 'rotary'
    ),
    'gpt_neox',
    'GPTNeoXForCausalLM',
    _gpt_neox_export,
    weight_path=_gpt_neox_path,
    export_path=partial(renamed_name, _GPT_NEOX_NAMES),
    prepare=gpt_neox_prepare,
    export_weights=gpt_neox_export_weights,
    preserve_source_layout=False,
    tied_head_names=('embed_out.weight', 'gpt_neox.embed_in.weight'),
)
