"""GPT-2's learned positions, biased LayerNorm and Conv1D checkpoint layout."""

from collections.abc import Mapping

import numpy as np

from dew import records
from dew.interop.hf_decoders import DecoderFields, _dew_path, _refuse
from dew.nn.backbones.causal_transformer import CausalTransformer


def _gpt2_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    hidden = records.integer(hf.get('n_embd', 768), 'n_embd')
    heads = records.integer(hf.get('n_head', 12), 'n_head')
    layers = records.integer(hf.get('n_layer', 12), 'n_layer')
    positions = records.integer(hf.get('n_positions', 1024), 'n_positions')
    used.update(('n_embd', 'n_head', 'n_layer', 'n_positions', 'n_inner',
                 'vocab_size', 'activation_function', 'layer_norm_epsilon',
                 'tie_word_embeddings', 'scale_attn_weights', 'scale_attn_by_inverse_layer_idx',
                 'reorder_and_upcast_attn', 'add_cross_attention', 'n_ctx',
                 'attn_pdrop', 'embd_pdrop', 'resid_pdrop', 'summary_type',
                 'summary_use_proj', 'summary_activation', 'summary_proj_to_labels',
                 'summary_first_dropout', 'use_cache', 'task_specific_params'))
    if hf.get('add_cross_attention'):
        _refuse('add_cross_attention', 'GPT-2 cross attention has no decoder counterpart')
    if hf.get('scale_attn_by_inverse_layer_idx'):
        _refuse('scale_attn_by_inverse_layer_idx', 'the attention scale does not vary by layer')
    if hf.get('reorder_and_upcast_attn'):
        _refuse('reorder_and_upcast_attn', 'the reordered GPT-2 attention arithmetic is not represented')
    activation = records.text(hf.get('activation_function', 'gelu_new'), 'activation_function')
    activations = {'gelu_new': 'gelu', 'gelu_fast': 'gelu', 'gelu_pytorch_tanh': 'gelu',
                   'gelu': 'gelu_exact', 'relu': 'relu'}
    if activation not in activations:
        _refuse(f'activation_function={activation!r}', 'the ungated MLP uses GELU or ReLU')
    if hidden % heads:
        _refuse('n_embd/n_head', 'the hidden width must divide into whole attention heads')
    return {
        'vocab_size': records.integer(hf.get('vocab_size', 50257), 'vocab_size'),
        'emb_features': hidden, 'num_layers': layers, 'num_heads': heads,
        'num_kv_heads': heads, 'head_dim': hidden // heads,
        'mlp_features': records.integer(hf.get('n_inner') or 4 * hidden, 'n_inner'),
        'mlp': activations[activation], 'mlp_bias': True,
        'norm_type': 'layer', 'norm_bias': True,
        'norm_eps': records.number(hf.get('layer_norm_epsilon', 1e-5), 'layer_norm_epsilon'),
        'qk_norm': False, 'attention_bias': True,
        'attention_scale': None if hf.get('scale_attn_weights', True) else 1.0,
        'position_embedding': 'learned', 'position_embedding_size': positions,
        'max_seq_len': positions, 'tie_embeddings': bool(hf.get('tie_word_embeddings', True)),
        'dropout_rate': records.number(hf.get('resid_pdrop', 0.1), 'resid_pdrop'),
        'embedding_dropout_rate': records.number(hf.get('embd_pdrop', 0.1), 'embd_pdrop'),
        'attention_dropout_rate': records.number(hf.get('attn_pdrop', 0.1), 'attn_pdrop'),
    }


def _gpt2_prepare(tensors: Mapping[str, np.ndarray],
                   _config: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
    """Split Conv1D qkv without copying the mapped checkpoint's bytes."""
    prepared = {}
    for name, tensor in tensors.items():
        # The original gpt2 safetensors stores the bare GPT2Model and its
        # old persistent causal buffers. They describe the fixed mask,
        # not learned attention biases; altered buffers cannot be loaded.
        if name.endswith('.attn.bias'):
            if (tensor.ndim != 4 or tensor.shape[:2] != (1, 1)
                    or tensor.shape[-2] != tensor.shape[-1]
                    or not np.array_equal(tensor, np.tril(np.ones_like(tensor)))):
                raise ValueError(f'{name} must hold the fixed triangular causal mask')
            continue
        if name.endswith('.attn.masked_bias'):
            if tensor.shape != () or tensor != -1e4:
                raise ValueError(f'{name} must hold the historical -10000 mask sentinel')
            continue
        if name.startswith(('h.', 'wte.', 'wpe.', 'ln_f.')):
            name = 'transformer.' + name
        if '.attn.c_attn.' in name:
            axis = 1 if name.endswith('.weight') else 0
            for projection, part in zip(('q_proj', 'k_proj', 'v_proj'),
                                        np.split(tensor, 3, axis=axis), strict=True):
                prepared[name.replace('attn.c_attn', f'self_attn.{projection}')] = (
                    part.T if axis == 1 else part)
        else:
            linear = any(part in name for part in ('.attn.c_proj.', '.mlp.c_fc.', '.mlp.c_proj.'))
            prepared[name] = tensor.T if linear and name.endswith('.weight') else tensor
    return prepared


def _gpt2_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    if name.startswith(('h.', 'wte.', 'wpe.', 'ln_f.')):
        name = 'transformer.' + name
    # Alias inspection sees raw names before preparation. A fused qkv
    # has three leaves, and the fixed buffers are validated by preparation.
    if '.attn.c_attn.' in name or name.endswith(('.attn.bias', '.attn.masked_bias')):
        return None
    if name == 'transformer.wpe.weight':
        return ('params', 'embed_positions', 'embedding')
    name = name.replace('transformer.wte.', 'model.embed_tokens.')
    name = name.replace('transformer.ln_f.', 'model.norm.')
    name = name.replace('transformer.h.', 'model.layers.')
    name = name.replace('.ln_1.', '.input_layernorm.').replace('.ln_2.', '.post_attention_layernorm.')
    name = name.replace('.attn.c_proj.', '.self_attn.o_proj.')
    name = name.replace('.mlp.c_fc.', '.mlp.up_proj.').replace('.mlp.c_proj.', '.mlp.down_proj.')
    return _dew_path(name, config)


def _gpt2_export(model: CausalTransformer) -> Mapping[str, object]:
    activation = model.mlp
    if not isinstance(activation, str) or activation not in ('gelu', 'gelu_exact', 'relu'):
        _refuse('mlp', 'GPT-2 requires an ungated GELU or ReLU feed-forward')
    config: dict[str, object] = {
        'n_embd': model.emb_features, 'n_layer': model.num_layers, 'n_head': model.num_heads,
        'n_inner': model.hidden_features, 'n_positions': model.position_embedding_size or model.max_seq_len,
        'layer_norm_epsilon': model.norm_eps,
        'activation_function': {'gelu': 'gelu_new', 'gelu_exact': 'gelu', 'relu': 'relu'}[activation],
        'scale_attn_weights': model.attention_scale is None,
        'attn_pdrop': model.attention_dropout_rate,
        'embd_pdrop': model.embedding_dropout_rate, 'resid_pdrop': model.dropout_rate,
    }
    config.update(dict.fromkeys((
        'hidden_size', 'num_hidden_layers', 'num_attention_heads',
        'num_key_value_heads', 'head_dim', 'intermediate_size',
        'max_position_embeddings', 'rms_norm_eps', 'attention_bias', 'hidden_act', 'rope_theta')))
    return config


def _gpt2_export_weights(model: CausalTransformer, variables: Mapping[str, object],
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
        if name == 'embed_positions.embedding':
            target = 'transformer.wpe.weight'
        else:
            target = _hf_name(name, config)
            if target is None:
                continue
            target = target.replace('model.embed_tokens.', 'transformer.wte.')
            target = target.replace('model.norm.', 'transformer.ln_f.')
            target = target.replace('model.layers.', 'transformer.h.')
            target = target.replace(".input_layernorm.", ".ln_1.").replace(
                ".post_attention_layernorm.", ".ln_2."
            )
            target = target.replace('.self_attn.o_proj.', '.attn.c_proj.')
            target = target.replace('.mlp.up_proj.', '.mlp.c_fc.').replace('.mlp.down_proj.', '.mlp.c_proj.')
        tensors[target] = np.asarray(value).T if name == 'lm_head.kernel' else np.asarray(value)
    for index in range(model.num_layers):
        for leaf, suffix in (('kernel', 'weight'), ('bias', 'bias')):
            values = [np.asarray(flat[f'layers_{index}.self_attn.{part}.{leaf}'])
                      for part in ('q_proj', 'k_proj', 'v_proj')]
            tensors[f'transformer.h.{index}.attn.c_attn.{suffix}'] = np.concatenate(values, axis=-1)
    return tensors
