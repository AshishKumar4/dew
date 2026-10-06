"""Phi's one-norm parallel biased block, sliced rotary and affine output head."""

from collections.abc import Mapping

from dew import records
from dew.interop.decoder_config import DecoderFields, _base_config, _refuse
from dew.interop.decoder_paths import Renames
from dew.nn.backbones.causal_transformer import CausalTransformer


def _phi_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    parameters = records.record(hf.get('rope_parameters') or {}, 'rope_parameters')
    partial = records.number(parameters.get('partial_rotary_factor', hf.get('partial_rotary_factor', .5)),
                              'partial_rotary_factor')
    theta = records.number(parameters.get('rope_theta', hf.get('rope_theta', 10000.)), 'rope_theta')
    rope = {key: value for key, value in parameters.items() if key != 'partial_rotary_factor'}
    activation = records.text(hf.get('hidden_act', 'gelu_new'), 'hidden_act')
    activations = {'gelu_new': 'gelu', 'gelu_pytorch_tanh': 'gelu', 'gelu': 'gelu_exact', 'relu': 'relu'}
    if activation not in activations:
        _refuse('hidden_act', 'Phi uses an ungated GELU or ReLU feed-forward')
    if hf.get('qk_layernorm', False):
        _refuse('qk_layernorm', 'Phi q/k LayerNorm has no RMSNorm counterpart')
    config = _base_config({**hf, 'hidden_act': 'silu', 'rope_theta': theta,
                           'rope_parameters': rope or None}, used, reads=frozenset())
    used.update(('partial_rotary_factor', 'qk_layernorm', 'layer_norm_eps', 'resid_pdrop',
                 'embd_pdrop', 'attention_dropout'))
    config.update({
        'mlp': activations[activation], 'mlp_bias': True,
        'norm_type': 'layer', 'norm_bias': True, 'scale_after_cast': False,
        'norm_eps': records.number(hf.get('layer_norm_eps', 1e-5), 'layer_norm_eps'),
        'parallel_residual': True, 'shared_parallel_norm': True, 'head_bias': True,
        'attention_bias': True, 'partial_rotary_factor': partial, 'partial_rotary_type': 'default',
        'dropout_rate': records.number(hf.get('resid_pdrop', 0.), 'resid_pdrop'),
        'embedding_dropout_rate': records.number(hf.get('embd_pdrop', 0.), 'embd_pdrop'),
        'attention_dropout_rate': records.number(hf.get('attention_dropout', 0.) or 0., 'attention_dropout'),
    })
    return config


_PHI_NAMES: Renames = (
    ('model.final_layernorm', 'model.norm'), ('self_attn.dense', 'self_attn.o_proj'),
    ('mlp.fc1', 'mlp.up_proj'), ('mlp.fc2', 'mlp.down_proj'))


def _phi_export(model: CausalTransformer) -> Mapping[str, object]:
    return {
        'hidden_act': {'gelu': 'gelu_new', 'gelu_exact': 'gelu', 'relu': 'relu'}[str(model.mlp)],
        'layer_norm_eps': model.norm_eps, 'qk_layernorm': False,
        'resid_pdrop': model.dropout_rate, 'embd_pdrop': model.embedding_dropout_rate,
        'attention_dropout': model.attention_dropout_rate,
        'rope_parameters': {'rope_type': 'default', 'rope_theta': model.rope_theta,
                            'partial_rotary_factor': model.partial_rotary_factor or 1.},
        'attention_bias': None, 'rms_norm_eps': None, 'head_dim': None,
    }
