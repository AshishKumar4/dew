"""GPT-2's learned positions, biased LayerNorm and Conv1D checkpoint layout."""

from collections.abc import Mapping

import numpy as np

from dew import records
from dew.interop.config_records import native_fields
from dew.interop.decoder_config import DecoderFields, _refuse
from dew.interop.decoder_export import _decoder_tensors
from dew.interop.decoder_paths import Packed, Renames, _renamed_path
from dew.interop.safetensors_io import LazyTensors
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind


def _gpt_neo_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    layers = records.integer(hf.get('num_layers', 24), 'num_layers')
    hidden = records.integer(hf.get('hidden_size', 2048), 'hidden_size')
    positions = records.integer(hf.get('max_position_embeddings', 2048), 'max_position_embeddings')
    config = _gpt2_config({**hf, 'n_embd': hidden, 'n_layer': layers,
                           'n_head': hf.get('num_heads', 16), 'n_positions': positions,
                           'n_inner': hf.get('intermediate_size') or 4 * hidden,
                           'resid_pdrop': hf.get('resid_dropout', 0.),
                           'embd_pdrop': hf.get('embed_dropout', 0.),
                           'attn_pdrop': hf.get('attention_dropout', 0.)}, used)
    stated = hf.get('attention_types', [[['global', 'local'], 12]])
    if not isinstance(stated, (list, tuple)):
        _refuse('attention_types', 'GPT-Neo repeats lists of global/local attention kinds')
    pattern: list[str] = []
    for entry in stated:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            _refuse('attention_types', 'each item holds a kind pattern and its repetition count')
        repeats = records.integer(entry[1], 'attention_types repetitions')
        if repeats < 1:
            _refuse('attention_types', 'repetition counts must be positive')
        pattern.extend(records.strings(entry[0], 'attention_types') * repeats)
    if len(pattern) != layers or set(pattern) - {'global', 'local'}:
        _refuse('attention_types', 'one global or local attention kind is required per layer')
    if (hf.get('attention_layers') is not None
            and records.strings(hf['attention_layers'], 'attention_layers') != tuple(pattern)):
        _refuse('attention_layers', 'the expanded pattern must agree with attention_types')
    used.update(('num_layers', 'num_heads', 'hidden_size', 'intermediate_size', 'max_position_embeddings',
                 'attention_types', 'attention_layers', 'window_size', 'resid_dropout', 'embed_dropout',
                 'attention_dropout', 'classifier_dropout'))
    used.add('gradient_checkpointing')
    config.update({
        'attention_bias': False, 'o_proj_bias': True, 'attention_scale': 1.0,
        'layer_types': tuple('full_attention' if kind == 'global' else 'sliding_attention'
                             for kind in pattern),
        'kinds': {'sliding_attention': native_fields(LayerKind)(
            window=records.integer(hf.get('window_size', 256), 'window_size'))} if 'local' in pattern else {},
    })
    return config


_GPT_NEO_NAMES: Renames = (
    ('attn.attention.out_proj', 'self_attn.o_proj'),
    ('attn.attention', 'self_attn'),
    ('transformer.wte', 'model.embed_tokens'), ('transformer.wpe', 'model.embed_positions'),
    ('transformer.ln_f', 'model.norm'), ('transformer.h', 'model.layers'),
    ('ln_1', 'input_layernorm'), ('ln_2', 'post_attention_layernorm'),
    ('mlp.c_fc', 'mlp.up_proj'), ('mlp.c_proj', 'mlp.down_proj'))


def _gpt_neo_export(model: CausalTransformer) -> Mapping[str, object]:
    fields = dict(_gpt2_export(model))
    fields.update({
        'hidden_size': model.emb_features, 'num_layers': model.num_layers, 'num_heads': model.num_heads,
        'intermediate_size': model.hidden_features,
        'max_position_embeddings': model.position_embedding_size or model.max_seq_len,
        'attention_types': [[[('local' if kind == 'sliding_attention' else 'global')], 1]
                            for kind in model.per_layer_types],
        'window_size': (model.kind_of('sliding_attention').window
                        if 'sliding_attention' in model.per_layer_types else 256),
        'resid_dropout': model.dropout_rate, 'embed_dropout': model.embedding_dropout_rate,
        'attention_dropout': model.attention_dropout_rate,
    })
    fields.update(dict.fromkeys(('n_embd', 'n_layer', 'n_head', 'n_inner', 'n_positions',
                                 'scale_attn_weights', 'attn_pdrop', 'embd_pdrop', 'resid_pdrop',
                                 'layer_types', 'sliding_window')))
    return fields


def _gptj_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    config = _gpt2_config(hf, used)
    head_dim = records.integer(config['head_dim'], 'head_dim')
    rotated = records.integer(hf.get('rotary_dim', 64), 'rotary_dim')
    if rotated < 2 or rotated > head_dim or rotated % 2:
        _refuse('rotary_dim', 'GPT-J rotates an even positive prefix of each attention head')
    used.update(('rotary_dim', 'rotary', 'gradient_checkpointing', 'tokenizer_class'))
    if hf.get('rotary', True) is not True or hf.get('scale_attn_weights', True) is not True:
        _refuse('rotary/scale_attn_weights', 'GPT-J rotates its heads and scales by their dimension')
    config.update({
        'position_embedding': 'rotary', 'position_embedding_size': None,
        'partial_rotary_factor': rotated / head_dim, 'partial_rotary_type': 'default',
        'attention_bias': False, 'parallel_residual': True, 'shared_parallel_norm': True,
        'head_bias': True, 'tie_embeddings': bool(hf.get('tie_word_embeddings', False)),
    })
    return config


_GPTJ_NAMES: Renames = (
    ('transformer.wte', 'model.embed_tokens'), ('transformer.ln_f', 'model.norm'),
    ('transformer.h', 'model.layers'), ('ln_1', 'input_layernorm'),
    ('attn.out_proj', 'self_attn.o_proj'), ('attn', 'self_attn'),
    ('mlp.fc_in', 'mlp.up_proj'), ('mlp.fc_out', 'mlp.down_proj'))


def _gptj_order(head_dim: int, rotated: int) -> np.ndarray:
    """Interleaved rotary pairs as the shared rotate-half arithmetic reads them.

    Applying the same permutation to q and k leaves their dot product
    unchanged. Values keep their original order, so the output projection
    and the residual stream require no permutation or runtime option.
    """
    return np.concatenate((np.arange(0, rotated, 2), np.arange(1, rotated, 2),
                           np.arange(rotated, head_dim)))


def _gptj_prepare(tensors: Mapping[str, np.ndarray],
                  config: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
    if config is None:
        raise ValueError('GPT-J rotary storage requires translated head geometry')
    head_dim = records.integer(config['head_dim'], 'head_dim')
    heads = records.integer(config['num_heads'], 'num_heads')
    rotated = int(head_dim * records.number(config['partial_rotary_factor'], 'partial_rotary_factor'))
    prepared = dict(tensors)
    for name, tensor in tensors.items():
        if name.endswith(('.attn.q_proj.weight', '.attn.k_proj.weight')):
            grouped = tensor.reshape(heads, head_dim, *tensor.shape[1:])
            prepared[name] = grouped[:, _gptj_order(head_dim, rotated)].reshape(tensor.shape)
        elif name.endswith('.attn.bias'):
            if (tensor.ndim != 4 or tensor.shape[:2] != (1, 1)
                    or tensor.shape[-2] != tensor.shape[-1]
                    or not np.array_equal(tensor, np.tril(np.ones_like(tensor)))):
                raise ValueError(f'{name} must hold the fixed triangular causal mask')
            del prepared[name]
        elif name.endswith('.attn.masked_bias'):
            if tensor.shape != () or float(tensor) != -1e9:
                raise ValueError(f'{name} must hold the historical negative mask sentinel')
            del prepared[name]
    return prepared


def _gptj_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    if name.endswith(('.attn.bias', '.attn.masked_bias')):
        return None
    return _renamed_path(_GPTJ_NAMES, name, config)


def _gptj_export(model: CausalTransformer) -> Mapping[str, object]:
    fields = dict(_gpt2_export(model))
    fields.update({
        'rotary_dim': int(model.features_per_head * (model.partial_rotary_factor or 1.)),
        'pad_token_id': None,
    })
    return fields


def _gptj_export_weights(model: CausalTransformer, variables: Mapping[str, object],
                         config: Mapping[str, object]) -> LazyTensors:
    tensors = _decoder_tensors(model, variables, config)
    rotated = int(model.features_per_head * (model.partial_rotary_factor or 1.))
    order = np.argsort(_gptj_order(model.features_per_head, rotated))

    def build(name: str) -> np.ndarray:
        tensor = tensors[name]
        if name.endswith(('.attn.q_proj.weight', '.attn.k_proj.weight')):
            grouped = tensor.reshape(model.num_heads, model.features_per_head, *tensor.shape[1:])
            return grouped[:, order].reshape(tensor.shape)
        return tensor

    return LazyTensors(tensors.specs, build)


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
    return native_fields(CausalTransformer)(
        vocab_size=records.integer(hf.get('vocab_size', 50257), 'vocab_size'),
        emb_features=hidden, num_layers=layers, num_heads=heads,
        num_kv_heads=heads, head_dim=hidden // heads,
        mlp_features=records.integer(hf.get('n_inner') or 4 * hidden, 'n_inner'),
        mlp=activations[activation], mlp_bias=True,
        norm_type='layer', norm_bias=True,
        norm_eps=records.number(hf.get('layer_norm_epsilon', 1e-5), 'layer_norm_epsilon'),
        qk_norm=False, attention_bias=True,
        attention_scale=None if hf.get('scale_attn_weights', True) else 1.0,
        position_embedding='learned', position_embedding_size=positions,
        max_seq_len=positions, tie_embeddings=bool(hf.get('tie_word_embeddings', True)),
        dropout_rate=records.number(hf.get('resid_pdrop', 0.1), 'resid_pdrop'),
        embedding_dropout_rate=records.number(hf.get('embd_pdrop', 0.1), 'embd_pdrop'),
        attention_dropout_rate=records.number(hf.get('attn_pdrop', 0.1), 'attn_pdrop'),
    )


def _gpt2_prepare(tensors: Mapping[str, np.ndarray],
                   _config: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
    """Check and drop the fixed attention buffers, and nest the bare model's names."""
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
        prepared[_nested(name)] = tensor
    return prepared


def _nested(name: str) -> str:
    return 'transformer.' + name if name.startswith(('h.', 'wte.', 'wpe.', 'ln_f.')) else name


_GPT2_NAMES: Renames = (
    ('transformer.wte', 'model.embed_tokens'), ('transformer.wpe', 'model.embed_positions'),
    ('transformer.ln_f', 'model.norm'), ('transformer.h', 'model.layers'),
    ('ln_1', 'input_layernorm'), ('ln_2', 'post_attention_layernorm'),
    ('attn.c_proj', 'self_attn.o_proj'), ('mlp.c_fc', 'mlp.up_proj'), ('mlp.c_proj', 'mlp.down_proj'))

# Conv1D stores `[in, out]` where the path map reads a torch Linear's
# `[out, in]`, and c_attn holds q, k and v side by side on `out`.
_QKV = ('q_proj', 'k_proj', 'v_proj')
_GPT2_PACKED = (
    Packed('.attn.c_attn.weight', tuple(f'.self_attn.{part}.weight' for part in _QKV), 0, (1, 0)),
    Packed('.attn.c_attn.bias', tuple(f'.self_attn.{part}.bias' for part in _QKV), 0),
    *(Packed(f'.{conv}.weight', (f'.{conv}.weight',), 0, (1, 0))
      for conv in ('attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')))


def _gpt2_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    name = _nested(name)
    # Alias inspection sees raw names before preparation. A fused qkv
    # has three leaves, and the fixed buffers are validated by preparation.
    if '.attn.c_attn.' in name or name.endswith(('.attn.bias', '.attn.masked_bias')):
        return None
    return _renamed_path(_GPT2_NAMES, name, config)


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
