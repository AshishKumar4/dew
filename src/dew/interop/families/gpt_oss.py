"""Translate GPT OSS, the sliding/full alternation with clamped SwiGLU experts.

The family's own dial is the `swigluoai` mlp value, which is what the export
vocabulary maps back to the reference's `silu`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

from dew.interop.decoder_config import LOCAL_EXPERTS, DecoderSpelling, decoder_layout, expect, rope_part
from dew.interop.decoder_parts import DecoderFamily, DecoderFields, Leaves, RopeSpelling, refuse, spelled
from dew.interop.layouts import ConfigLayout, ConfigPart, constant, unread
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.rope import YarnScaling

# The layer's own tensors beside the shared map's, read one way on load and
# the other on export.
_GPT_OSS_LEAVES: Leaves = {
    'self_attn.sinks': ('self_attn', 'sinks'),
    'mlp.router.weight': ('mlp', 'router', 'kernel'), 'mlp.router.bias': ('mlp', 'router', 'bias'),
    **{f'mlp.experts.{leaf}': ('mlp', 'experts', leaf)
       for leaf in ('gate_up_proj', 'gate_up_proj_bias', 'down_proj', 'down_proj_bias')}}


_SPELLING = RopeSpelling(theta=150000.0)


def _plain_or_yarn(model: CausalTransformer) -> Mapping[str, object]:
    # A plain rotary is written as a plain record: an omitted one reads as the class's YaRN.
    yarn = model.rope_scaling if isinstance(model.rope_scaling, YarnScaling) else None
    return {'rope_parameters': {'rope_type': 'default', 'rope_theta': model.rope_theta} if yarn is None
            else {**asdict(yarn), 'rope_type': 'yarn', 'rope_theta': model.rope_theta}}


def _experts_per_token(hf_config: Mapping[str, object], used: set[str],
                       record: DecoderFields) -> DecoderFields:
    used.add('experts_per_token')
    top_k = hf_config['num_experts_per_tok']
    if hf_config.get('experts_per_token', top_k) != top_k:
        refuse('experts_per_token', 'it disagrees with num_experts_per_tok')
    return record


GPT_OSS_CONFIG = ConfigLayout(
    decoder_layout(DecoderSpelling(rope=_SPELLING, pattern=('sliding_attention', 'full_attention'),
                                   activation=None, scale_after_cast=False),
                   rope=ConfigPart(rope_part(_SPELLING, ramps=('yarn',), per_kind=False).read,
                                   _plain_or_yarn)),
    # GptOssExperts gates with its own clamped SiLU, whatever hidden_act names.
    expect('hidden_act', 'GptOssExperts gates with its clamped SiLU', value='silu'),
    expect('swiglu_limit', 'GptOssExperts clamps at 7.0', value=7.0),
    constant('mlp', value='swigluoai'), constant('attention_sinks', value=True),
    LOCAL_EXPERTS, ConfigPart(_experts_per_token, lambda model: {}),
    # quantization_config describes how the weights are stored, which the shared codec reads
    # (`dew.interop.codecs.source_quantization`), the released MXFP4 experts among them.
    unread('initial_context_length', 'quantization_config'))
"""GPT OSS's config, both ways."""


GPT_OSS = DecoderFamily(
    ('gpt_oss',),
    GPT_OSS_CONFIG,
    lambda fields: fields.mlp == 'swigluoai',
    'gpt_oss',
    'GptOssForCausalLM',
    routes=spelled((), layer_leaves=_GPT_OSS_LEAVES),
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
