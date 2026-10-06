"""ModernBERT's bidirectional encoder and its masked-LM head.

The encoder is a pre-norm block stack over normalized token embeddings, with
the first block reading them unnormed, alternating global and local layers
under separate rope bases, fused qkv and GeGLU projections, and a final norm
(modeling_modernbert.py, transformers 5.16.1). ModernBertForMaskedLM adds a
prediction head (dense, activation, norm) and a tied, biased projection.

The local layers keep their window on both sides of a query
(`LayerKind.bidirectional_window`), which `local_attention` (causal only)
does not run, so they build the dense [S, S] logits a global layer does: at
8192 tokens a local layer costs what a global one costs.
"""

from collections.abc import Mapping

import numpy as np

from dew import records
from dew.interop.config_records import native_fields
from dew.interop.decoder_config import (
    _ACTIVATIONS,
    DEFAULT_MAX_SEQ_LEN,
    DecoderFields,
    _hf_activation,
    _kinds,
    _refuse,
    _rope,
)
from dew.interop.decoder_paths import Packed, Renames, _renamed_name, _renamed_path
from dew.nn.backbones.causal_transformer import CausalTransformer

_ENCODER = 'ModernBertModel'
_MASKED_LM = 'ModernBertForMaskedLM'
_HEAD_ACTIVATIONS = {'gelu': 'gelu_exact', 'gelu_pytorch_tanh': 'gelu'}


def _modernbert_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a ModernBertConfig, either spelling of its rope and pattern.

    Releases before transformers 5 state `global_rope_theta`,
    `local_rope_theta` and `global_attn_every_n_layers`; later ones nest the
    bases under `rope_parameters` and list `layer_types`. The local window
    `local_attention` is the whole span, so a query reads `local_attention // 2`
    keys on either side, which is a symmetric window of that plus one.
    A config whose architectures name the bare `ModernBertModel` has no head.
    """
    used.update((
        'hidden_size', 'num_attention_heads', 'num_hidden_layers', 'vocab_size', 'intermediate_size',
        'hidden_activation', 'norm_eps', 'norm_bias', 'attention_bias', 'max_position_embeddings',
        'tie_word_embeddings', 'local_attention', 'global_attn_every_n_layers', 'layer_types',
        'global_rope_theta', 'local_rope_theta', 'embedding_dropout', 'attention_dropout',
        'mlp_dropout', 'classifier_activation', 'classifier_bias', 'decoder_bias', 'architectures',
        # Read by the sequence-classification heads, which nothing here builds.
        'classifier_pooling', 'classifier_dropout',
        # Training-time and kernel switches of the reference, and token ids.
        'sparse_prediction', 'sparse_pred_ignore_index', 'deterministic_flash_attn',
        'reference_compile', 'repad_logits_with_grad', 'gradient_checkpointing',
        'initializer_cutoff_factor', 'cls_token_id', 'sep_token_id', 'mask_token_id',
        # Pre-5.0 releases carry these, which ModernBertConfig no longer reads:
        # the model norms at norm_eps and always rotates.
        'layer_norm_eps', 'position_embedding_type'))
    for dropout in ('attention_dropout', 'mlp_dropout'):
        if records.number(hf.get(dropout, 0.0), dropout):
            _refuse(dropout, "ModernBERT drops inside its attention and MLP where the blocks here do not")
    if hf.get('classifier_bias'):
        _refuse('classifier_bias', "the prediction head's dense layer is bias-free")
    hidden = records.integer(hf['hidden_size'], 'hidden_size')
    heads = records.integer(hf['num_attention_heads'], 'num_attention_heads')
    layers = records.integer(hf['num_hidden_layers'], 'num_hidden_layers')
    if hidden % heads:
        _refuse('hidden_size/num_attention_heads', 'the hidden width must divide into whole heads')
    activation = records.text(hf.get('hidden_activation', 'gelu'), 'hidden_activation')
    if activation not in _ACTIVATIONS:
        _refuse(f'hidden_activation={activation!r}', f'the GLU supports {sorted(_ACTIVATIONS)}')

    if hf.get('layer_types') is not None:
        layer_types = records.strings(hf['layer_types'], 'layer_types')
    else:
        every = records.integer(hf.get('global_attn_every_n_layers', 3), 'global_attn_every_n_layers')
        layer_types = tuple('sliding_attention' if index % every else 'full_attention'
                            for index in range(layers))
    if isinstance(hf.get('rope_parameters'), Mapping):
        ropes = _rope(hf, used, local=False)
        if ropes.scaling is not None or ropes.local_scaling is not None:
            _refuse('rope_parameters', 'ModernBERT rotates at its plain frequencies')
        theta, local_theta = ropes.theta, ropes.local_theta
    else:
        used.update(('rope_theta', 'rope_parameters', 'rope_scaling'))
        if hf.get('rope_scaling') is not None:
            _refuse('rope_scaling', 'ModernBERT rotates at its plain frequencies')
        theta = records.number(hf.get('global_rope_theta', 160000.0), 'global_rope_theta')
        local = records.number(hf.get('local_rope_theta', 10000.0), 'local_rope_theta')
        local_theta = None if local == theta else local
    window = records.integer(hf.get('local_attention', 128), 'local_attention') // 2 + 1

    head = records.strings(hf.get('architectures', [_MASKED_LM]), 'architectures') != (_ENCODER,)
    head_activation = records.text(hf.get('classifier_activation', 'gelu'), 'classifier_activation')
    if head and head_activation not in _HEAD_ACTIVATIONS:
        _refuse(f'classifier_activation={head_activation!r}',
                f'the prediction head supports {sorted(_HEAD_ACTIVATIONS)}')
    config: DecoderFields = native_fields(CausalTransformer)(
        vocab_size=records.integer(hf['vocab_size'], 'vocab_size'),
        emb_features=hidden, num_layers=layers, num_heads=heads, num_kv_heads=heads,
        head_dim=hidden // heads, mlp=_ACTIVATIONS[activation],
        mlp_features=records.integer(hf['intermediate_size'], 'intermediate_size'),
        max_seq_len=min(records.integer(hf.get('max_position_embeddings', 8192), 'max_position_embeddings'),
                        DEFAULT_MAX_SEQ_LEN),
        rope_theta=theta, layer_types=layer_types,
        norm_type='layer', norm_bias=bool(hf.get('norm_bias', False)),
        norm_eps=records.number(hf.get('norm_eps', 1e-5), 'norm_eps'),
        embedding_norm=True, first_attention_norm=False, qk_norm=False,
        attention_bias=bool(hf.get('attention_bias', False)), causal=False,
        tie_embeddings=bool(hf.get('tie_word_embeddings', True)),
        embedding_dropout_rate=records.number(hf.get('embedding_dropout', 0.0), 'embedding_dropout'),
    )
    kinds = _kinds(layer_types, window, local_theta, None, None)
    if 'sliding_attention' in kinds:
        kinds['sliding_attention']['bidirectional_window'] = True
    config['kinds'] = kinds
    if head:
        config.update(head_transform=_HEAD_ACTIVATIONS[head_activation],
                      head_bias=bool(hf.get('decoder_bias', True)))
    return config


_MODERNBERT_NAMES: Renames = (
    ('model.embeddings.tok_embeddings', 'model.embed_tokens'),
    ('model.embeddings.norm', 'model.embedding_layernorm'), ('model.final_norm', 'model.norm'),
    ('attn_norm', 'input_layernorm'), ('mlp_norm', 'post_attention_layernorm'),
    ('attn.Wo', 'self_attn.o_proj'), ('mlp.Wo', 'mlp.down_proj'), ('decoder', 'lm_head'))

# Wqkv stacks q, k and v in thirds of its output rows, and Wi the activated
# input half over the gate half (ModernBertAttention, ModernBertMLP).
_MODERNBERT_PACKED = (
    Packed('.attn.Wqkv.weight',
           ('.self_attn.q_proj.weight', '.self_attn.k_proj.weight', '.self_attn.v_proj.weight'), 0),
    Packed('.mlp.Wi.weight', ('.mlp.gate_proj.weight', '.mlp.up_proj.weight'), 0))

_HEAD: Mapping[str, tuple[str, ...]] = {
    'head.dense.weight': ('head_dense', 'kernel'), 'head.norm.weight': ('head_norm', 'scale'),
    'head.norm.bias': ('head_norm', 'bias'), 'decoder.bias': ('head_bias',)}
_HEAD_NAMES = {'.'.join(path): name for name, path in _HEAD.items()}


def _nested(name: str) -> str:
    """A bare `ModernBertModel` checkpoint's name as the masked LM nests it."""
    return 'model.' + name if name.startswith(('embeddings.', 'layers.', 'final_norm.')) else name


def _modernbert_prepare(tensors: Mapping[str, np.ndarray],
                        _config: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
    return {_nested(name): tensor for name, tensor in tensors.items()}


def _modernbert_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    name = _nested(name)
    if name in _HEAD:
        return ('params', *_HEAD[name])
    # Alias inspection sees the raw names before preparation splits them.
    if name.endswith(('.attn.Wqkv.weight', '.mlp.Wi.weight')):
        return None
    return _renamed_path(_MODERNBERT_NAMES, name, config)


def _modernbert_export_path(dew_name: str, config: Mapping[str, object]) -> str | None:
    if dew_name in _HEAD_NAMES:
        return _HEAD_NAMES[dew_name]
    return _renamed_name(_MODERNBERT_NAMES, dew_name, config)


def _modernbert_export(model: CausalTransformer) -> Mapping[str, object]:
    types = model.per_layer_types
    sliding = model.kind_of('sliding_attention') if 'sliding_attention' in types else None
    local_theta = model.rope_theta if sliding is None else sliding.rope_theta or model.rope_theta
    fields: dict[str, object] = {
        'architectures': [_ENCODER if model.head_transform is None else _MASKED_LM],
        'hidden_activation': _hf_activation(model.mlp), 'norm_eps': model.norm_eps,
        'norm_bias': model.norm_bias, 'mlp_bias': False, 'layer_types': list(types),
        'rope_parameters': {
            'full_attention': {'rope_type': 'default', 'rope_theta': model.rope_theta},
            'sliding_attention': {'rope_type': 'default', 'rope_theta': local_theta}},
        'embedding_dropout': model.embedding_dropout_rate,
        'attention_dropout': 0.0, 'mlp_dropout': 0.0, 'classifier_bias': False,
        'decoder_bias': model.head_bias,
    }
    if sliding is not None and sliding.window is not None:
        fields['local_attention'] = 2 * (sliding.window - 1)
    if model.head_transform is not None:
        fields['classifier_activation'] = {'gelu_exact': 'gelu', 'gelu': 'gelu_pytorch_tanh'}[
            model.head_transform]
    # ModernBERT's original keys, which transformers 5 no longer writes and older
    # readers still take: llama.cpp's converter reads its norm epsilon from these.
    fields['layer_norm_eps'] = model.norm_eps
    every = next((n for n in range(1, len(types) + 1)
                  if all((kind == 'full_attention') == (index % n == 0) for index, kind in enumerate(types))),
                 None)
    if every is not None:
        fields['global_attn_every_n_layers'] = every
    fields.update(dict.fromkeys(('num_key_value_heads', 'head_dim', 'rms_norm_eps', 'hidden_act',
                                 'rope_theta', 'rope_local_base_freq', 'sliding_window', 'use_cache')))
    return fields
