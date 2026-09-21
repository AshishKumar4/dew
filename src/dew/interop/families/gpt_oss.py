"""GPT OSS: the sliding/full alternation with clamped SwiGLU experts.

The family's own dial is the `swigluoai` mlp value, which is what the export
vocabulary maps back to the reference's `silu`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

from dew.interop.families.deepseek import _deepseek_rope
from dew.interop.hf_decoders import DecoderFields, _base_config, _dew_path, _hf_name, _refuse, _Ropes
from dew.nn.backbones.causal_transformer import CausalTransformer


def _gpt_oss_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    theta, yarn = _deepseek_rope(hf_config, used)
    layers = hf_config['num_hidden_layers']
    experts = hf_config['num_local_experts']
    top_k = hf_config['num_experts_per_tok']
    if not isinstance(layers, int) or not isinstance(experts, int) or not isinstance(top_k, int):
        _refuse('num_hidden_layers/num_local_experts/num_experts_per_tok', 'integer counts are required')
    pattern = tuple('sliding_attention' if index % 2 == 0 else 'full_attention'
                    for index in range(layers))
    config = _base_config(hf_config, used, layer_types=pattern,
                          rope=_Ropes(theta), scale_after_cast=False)
    config.update(mlp='swigluoai', attention_sinks=True, yarn=yarn,
                  mixture={'experts': experts, 'top_k': top_k})
    if hf_config.get('swiglu_limit', 7.0) != 7.0:
        _refuse('swiglu_limit', 'GptOssExperts clamps at 7.0')
    if hf_config.get('experts_per_token', top_k) != top_k:
        _refuse('experts_per_token', 'it disagrees with num_experts_per_tok')
    if hf_config.get('output_router_logits', False):
        _refuse('output_router_logits', 'the decoder returns token logits')
    quantization = hf_config.get('quantization_config')
    if quantization is not None and (not isinstance(quantization, Mapping)
                                    or quantization.get('quant_method') != 'mxfp4'):
        _refuse('quantization_config', 'GPT OSS supports MXFP4 blocks and scales')
    used.update(('num_local_experts', 'num_experts_per_tok', 'experts_per_token',
                 'swiglu_limit', 'initial_context_length', 'router_aux_loss_coef',
                 'output_router_logits', 'quantization_config'))
    return config


def _gpt_oss_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    parts = name.split('.')
    if len(parts) >= 5 and parts[:2] == ['model', 'layers'] and parts[2].isdigit():
        prefix = ('params', f'layers_{parts[2]}')
        if parts[3:] == ['self_attn', 'sinks']:
            return (*prefix, 'self_attn', 'sinks')
        if len(parts) == 6 and parts[3:5] == ['mlp', 'router'] and parts[5] in ('weight', 'bias'):
            return (*prefix, 'mlp', 'router', 'kernel' if parts[5] == 'weight' else 'bias')
        if (len(parts) == 6 and parts[3:5] == ['mlp', 'experts']
                and parts[5] in ('gate_up_proj', 'gate_up_proj_bias', 'down_proj', 'down_proj_bias')):
            return (*prefix, 'mlp', 'experts', parts[5])
    return _dew_path(name, config)


def _gpt_oss_export_path(name: str, config: Mapping[str, object]) -> str | None:
    parts = name.split('.')
    if parts[0].startswith('layers_'):
        prefix = f"model.layers.{parts[0].removeprefix('layers_')}"
        if parts[1:] == ['self_attn', 'sinks']:
            return prefix + '.self_attn.sinks'
        if len(parts) == 4 and parts[1:3] == ['mlp', 'router']:
            return prefix + '.mlp.router.' + ('weight' if parts[3] == 'kernel' else 'bias')
        if len(parts) == 4 and parts[1:3] == ['mlp', 'experts']:
            return prefix + '.mlp.experts.' + parts[3]
    return _hf_name(name, config)


def _gpt_oss_export(model: CausalTransformer) -> Mapping[str, object]:
    mixture = model.mixture
    if mixture is None:
        _refuse('mixture', 'GPT OSS needs routed experts')
    return {'hidden_act': 'silu', 'num_local_experts': mixture.experts,
            'num_experts_per_tok': mixture.top_k, 'swiglu_limit': 7.0,
            'rope_scaling': None if model.yarn is None else asdict(model.yarn)}
