"""Translate GPT OSS, the sliding/full alternation with clamped SwiGLU experts.

The family's own dial is the `swigluoai` mlp value, which is what the export
vocabulary maps back to the reference's `silu`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from functools import partial

from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    Leaves,
    RopeSpelling,
    base_config,
    read_rope,
    refuse,
    renamed_name,
    renamed_path,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture


def _gpt_oss_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    ropes = read_rope(hf_config, used, RopeSpelling(theta=150000.0), ramps=('yarn',), per_kind=False)
    layers = hf_config['num_hidden_layers']
    experts = hf_config['num_local_experts']
    top_k = hf_config['num_experts_per_tok']
    if not isinstance(layers, int) or not isinstance(experts, int) or not isinstance(top_k, int):
        refuse('num_hidden_layers/num_local_experts/num_experts_per_tok', 'integer counts are required')
    config = base_config(hf_config, used, layer_types=('sliding_attention', 'full_attention'),
                          rope=ropes, scale_after_cast=False)
    config.update(mlp='swigluoai', attention_sinks=True,
                  mixture=native_fields(Mixture)(experts=experts, top_k=top_k))
    if hf_config.get('swiglu_limit', 7.0) != 7.0:
        refuse('swiglu_limit', 'GptOssExperts clamps at 7.0')
    if hf_config.get('experts_per_token', top_k) != top_k:
        refuse('experts_per_token', 'it disagrees with num_experts_per_tok')
    if hf_config.get('output_router_logits', False):
        refuse('output_router_logits', 'the decoder returns token logits')
    # quantization_config describes how the weights are stored, which the shared codec reads
    # (`dew.interop.codecs.source_quantization`), the released MXFP4 experts among them.
    used.update(('num_local_experts', 'num_experts_per_tok', 'experts_per_token',
                 'swiglu_limit', 'initial_context_length', 'router_aux_loss_coef',
                 'output_router_logits', 'quantization_config'))
    return config


# The layer's own tensors beside the shared map's, read one way on load and
# the other on export.
_GPT_OSS_LEAVES: Leaves = {
    'self_attn.sinks': ('self_attn', 'sinks'),
    'mlp.router.weight': ('mlp', 'router', 'kernel'), 'mlp.router.bias': ('mlp', 'router', 'bias'),
    **{f'mlp.experts.{leaf}': ('mlp', 'experts', leaf)
       for leaf in ('gate_up_proj', 'gate_up_proj_bias', 'down_proj', 'down_proj_bias')}}


def _gpt_oss_export(model: CausalTransformer) -> Mapping[str, object]:
    mixture = model.mixture
    if mixture is None:
        refuse('mixture', 'GPT OSS needs routed experts')
    # A plain rotary is written as a plain record: an omitted one reads as the class's YaRN.
    rope = ({'rope_type': 'default', 'rope_theta': model.rope_theta} if model.yarn is None
            else {**asdict(model.yarn), 'rope_type': 'yarn', 'rope_theta': model.rope_theta})
    return {'hidden_act': 'silu', 'num_local_experts': mixture.experts,
            'num_experts_per_tok': mixture.top_k, 'swiglu_limit': 7.0, 'rope_parameters': rope}


GPT_OSS = DecoderFamily(
    ('gpt_oss',),
    _gpt_oss_config,
    lambda fields: fields.mlp == 'swigluoai',
    'gpt_oss',
    'GptOssForCausalLM',
    _gpt_oss_export,
    weight_path=partial(renamed_path, (), layer_leaves=_GPT_OSS_LEAVES),
    export_path=partial(renamed_name, (), layer_leaves=_GPT_OSS_LEAVES),
    preserve_source_layout=False,
    # GptOssConfig's defaults: biased attention, its own norm epsilon and expert geometry, a window,
    # and its YaRN record where a config states none.
    defaults={'attention_bias': True, 'rms_norm_eps': 1e-5, 'num_experts_per_tok': 4,
              'num_local_experts': 128, 'sliding_window': 128, 'head_dim': 64, 'num_key_value_heads': 8,
              'max_position_embeddings': 131072,
              'rope_parameters': {'rope_type': 'yarn', 'factor': 32.0, 'beta_fast': 32.0, 'beta_slow': 1.0,
                                  'truncate': False, 'original_max_position_embeddings': 4096}},
    padding_idx=True,
)
