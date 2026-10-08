"""Translate GPT OSS, the sliding/full alternation with clamped SwiGLU experts.

The family's own dial is the `swigluoai` mlp value, which is what the export
vocabulary maps back to the reference's `silu`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

from dew.interop.config_records import native_fields
from dew.interop.families.deepseek import deepseek_rope
from dew.interop.hf_decoders import (
    DecoderFamily,
    DecoderFields,
    Ropes,
    base_config,
    dew_path,
    hf_tensor_name,
    refuse,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture


def _gpt_oss_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    theta, yarn = deepseek_rope(hf_config, used)
    layers = hf_config['num_hidden_layers']
    experts = hf_config['num_local_experts']
    top_k = hf_config['num_experts_per_tok']
    if not isinstance(layers, int) or not isinstance(experts, int) or not isinstance(top_k, int):
        refuse('num_hidden_layers/num_local_experts/num_experts_per_tok', 'integer counts are required')
    pattern = tuple('sliding_attention' if index % 2 == 0 else 'full_attention'
                    for index in range(layers))
    config = base_config(hf_config, used, layer_types=pattern,
                          rope=Ropes(theta), scale_after_cast=False)
    config.update(mlp='swigluoai', attention_sinks=True, yarn=yarn,
                  mixture=native_fields(Mixture)(experts=experts, top_k=top_k))
    if hf_config.get('swiglu_limit', 7.0) != 7.0:
        refuse('swiglu_limit', 'GptOssExperts clamps at 7.0')
    if hf_config.get('experts_per_token', top_k) != top_k:
        refuse('experts_per_token', 'it disagrees with num_experts_per_tok')
    if hf_config.get('output_router_logits', False):
        refuse('output_router_logits', 'the decoder returns token logits')
    quantization = hf_config.get('quantization_config')
    if quantization is not None and (not isinstance(quantization, Mapping)
                                    or quantization.get('quant_method') != 'mxfp4'):
        refuse('quantization_config', 'GPT OSS supports MXFP4 blocks and scales')
    used.update(('num_local_experts', 'num_experts_per_tok', 'experts_per_token',
                 'swiglu_limit', 'initial_context_length', 'router_aux_loss_coef',
                 'output_router_logits', 'quantization_config'))
    return config


# The layer's own tensors beside the shared map's, read one way on load and
# the other on export.
_GPT_OSS_LEAVES: Mapping[str, tuple[str, ...]] = {
    'self_attn.sinks': ('self_attn', 'sinks'),
    'mlp.router.weight': ('mlp', 'router', 'kernel'), 'mlp.router.bias': ('mlp', 'router', 'bias'),
    **{f'mlp.experts.{leaf}': ('mlp', 'experts', leaf)
       for leaf in ('gate_up_proj', 'gate_up_proj_bias', 'down_proj', 'down_proj_bias')}}
_GPT_OSS_NAMES: Mapping[tuple[str, ...], str] = {path: name for name, path in _GPT_OSS_LEAVES.items()}


def _gpt_oss_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    parts = name.split('.')
    if len(parts) >= 5 and parts[:2] == ['model', 'layers'] and parts[2].isdigit():
        leaf = _GPT_OSS_LEAVES.get('.'.join(parts[3:]))
        if leaf is not None:
            return ('params', f'layers_{parts[2]}', *leaf)
    return dew_path(name, config)


def _gpt_oss_export_path(name: str, config: Mapping[str, object]) -> str | None:
    layer, _, rest = name.partition('.')
    tail = tuple(rest.split('.'))
    if layer.startswith('layers_') and tail in _GPT_OSS_NAMES:
        return f"model.layers.{layer.removeprefix('layers_')}.{_GPT_OSS_NAMES[tail]}"
    return hf_tensor_name(name, config)


def _gpt_oss_export(model: CausalTransformer) -> Mapping[str, object]:
    mixture = model.mixture
    if mixture is None:
        refuse('mixture', 'GPT OSS needs routed experts')
    return {'hidden_act': 'silu', 'num_local_experts': mixture.experts,
            'num_experts_per_tok': mixture.top_k, 'swiglu_limit': 7.0,
            'rope_scaling': None if model.yarn is None else asdict(model.yarn)}


GPT_OSS = DecoderFamily(
    ('gpt_oss',),
    _gpt_oss_config,
    lambda fields: fields.mlp == 'swigluoai',
    'gpt_oss',
    'GptOssForCausalLM',
    _gpt_oss_export,
    weight_path=_gpt_oss_path,
    export_path=_gpt_oss_export_path,
    preserve_source_layout=False,
)
