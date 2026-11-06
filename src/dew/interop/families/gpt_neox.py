"""GPT-NeoX's parallel biased LayerNorm block and head-interleaved qkv storage."""

from collections.abc import Mapping
from functools import partial

import numpy as np

from dew import _records
from dew.interop.decoder_parts import (
    MEGATRON_MLP,
    PLAIN_SPELLING,
    DecoderFamily,
    DecoderFields,
    Renames,
    base_config,
    decoder_tensors,
    fixed_attention_buffer,
    grouped_qkv,
    hf_ungated,
    partial_rope,
    renamed_name,
    renamed_path,
    ungated_mlp,
)
from dew.interop.unread import Stored, equal
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.rope import inverse_frequencies


def _gpt_neox_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    theta, partial, rope = partial_rope(hf, PLAIN_SPELLING, default_factor=.25, factor_field='rotary_pct',
                                        theta_field='rotary_emb_base')
    activation = _records.text(hf.get('hidden_act', 'gelu'), 'hidden_act')
    mlp = ungated_mlp(activation, ('gelu', 'gelu_new', 'gelu_pytorch_tanh', 'relu'), 'hidden_act', 'GPT-NeoX')
    config = base_config({**hf, 'hidden_act': 'gelu', 'rope_theta': theta,
                           'rope_parameters': rope or None}, used, rope=PLAIN_SPELLING,
                         reads=frozenset({'attention_bias'}))
    used.update(('rotary_pct', 'rotary_emb_base', 'use_parallel_residual', 'layer_norm_eps',
                 'hidden_dropout', 'classifier_dropout', 'is_decoder'))
    config.update({
        'mlp': mlp, 'mlp_bias': True,
        'norm_type': 'layer', 'norm_bias': True,
        'norm_eps': _records.number(hf.get('layer_norm_eps', 1e-5), 'layer_norm_eps'),
        'scale_after_cast': False, 'attention_bias': bool(hf.get('attention_bias', True)),
        'partial_rotary_factor': partial, 'partial_rotary_type': 'default',
        'parallel_residual': bool(hf.get('use_parallel_residual', True)),
        'dropout_rate': _records.number(hf.get('hidden_dropout', 0.), 'hidden_dropout'),
        'embedding_dropout_rate': _records.number(hf.get('hidden_dropout', 0.), 'hidden_dropout'),
        'attention_dropout_rate': _records.number(hf.get('attention_dropout', 0.), 'attention_dropout'),
    })
    return config


def _gpt_neox_buffers(tensors: Mapping[str, Stored],
                      config: Mapping[str, object] | None = None) -> Mapping[str, Stored]:
    """The checkpoint without its fixed attention buffers, each checked: the causal mask, the mask
    sentinel and the rotary frequencies, which the config computes."""
    if config is None:
        raise ValueError('GPT-NeoX checks its rotary buffer against translated head geometry')
    dimension = _records.integer(config['head_dim'], 'head_dim')
    prepared = {}
    for name, tensor in tensors.items():
        # FP16 releases cast the historical -1e9 sentinel to -inf.
        if fixed_attention_buffer(name, tensor, 'attention', (-1e9, -np.inf)):
            continue
        if name.endswith('.attention.rotary_emb.inv_freq'):
            rotated = int(
                dimension * _records.number(config.get("partial_rotary_factor", 1.0), "partial_rotary_factor")
            )
            expected = inverse_frequencies(_records.number(config.get('rope_theta', 10000.), 'rope_theta'),
                                           rotated, dtype=np.float32).astype(tensor.dtype)
            if not equal(tensor, expected):
                raise ValueError(
                    f"{name} disagrees with the configured rotary frequencies in its stored dtype"
                )
            continue
        prepared[name] = tensor
    return prepared


_GPT_NEOX_NAMES: Renames = (
    ('embed_out', 'lm_head'), ('gpt_neox.embed_in', 'model.embed_tokens'),
    ('gpt_neox.final_layer_norm', 'model.norm'), ('gpt_neox.layers', 'model.layers'),
    ('attention.dense', 'self_attn.o_proj'), *MEGATRON_MLP)


def _gpt_neox_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    # Alias inspection sees raw names before preparation, which validates and drops the buffers.
    if name.endswith(('.attention.bias', '.attention.masked_bias', '.attention.rotary_emb.inv_freq')):
        return None
    return renamed_path(_GPT_NEOX_NAMES, name, config)


def _gpt_neox_export(model: CausalTransformer) -> Mapping[str, object]:
    fields: dict[str, object] = {
        'layer_norm_eps': model.norm_eps, 'use_parallel_residual': model.parallel_residual,
        'hidden_act': hf_ungated(model.mlp, 'GPT-NeoX'),
        'hidden_dropout': model.dropout_rate, 'attention_dropout': model.attention_dropout_rate,
        'rope_parameters': {'rope_type': 'default', 'rope_theta': model.rope_theta,
                            'partial_rotary_factor': model.partial_rotary_factor or 1.},
    }
    fields.update(dict.fromkeys(('rms_norm_eps', 'rope_theta', 'num_key_value_heads', 'head_dim')))
    return fields


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
    prepare=_gpt_neox_buffers,
    packed=grouped_qkv('.attention.query_key_value'),
    export_weights=decoder_tensors,
    preserve_source_layout=False,
    tied_head_names=('embed_out.weight', 'gpt_neox.embed_in.weight'),
    defaults={'max_position_embeddings': 2048},
)
