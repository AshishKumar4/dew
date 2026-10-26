"""Translate Llama, Mistral and Mixtral, the dense block the other families vary.

Llama's own translation is the shared `base_config` over the fields
LlamaConfig declares, so what stands here is only what the variants add to
it. Mistral adds a sliding window on every layer and drops the attention
biases. Ministral names a pattern of sliding and full layers. Mixtral adds a
softmax-routed feed-forward and the expert tensor names that routing brings.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    FUSED_EXPERTS,
    MOE_NAMES,
    PLAIN_SPELLING,
    DecoderFamily,
    DecoderFields,
    Packed,
    RopeSpelling,
    base_config,
    every_layer_windowed,
    read_rope,
    refuse,
    renamed,
    renamed_name,
    renamed_path,
    softmax_top_k,
)
from dew.interop.families.qwen import qwen35_moe_path
from dew.nn.backbones.decoder_block import Mixture


def llama_config(hf_config, used):
    # LlamaConfig declares attention_bias, mlp_bias and no window: LlamaAttention
    # attends every key, and LlamaMLP's projections carry mlp_bias's biases.
    # LlamaRotaryEmbedding builds any rope_type the rope utilities do, YaRN
    # among them (modeling_llama.py:82-86).
    rope = read_rope(hf_config, used, PLAIN_SPELLING)
    return base_config(hf_config, used, rope=rope, reads=frozenset({'attention_bias', 'mlp_bias'}))


def ministral_config(hf_config, used):
    # MinistralConfig declares layer_types and sliding_window, and its
    # attention builds no biases. Without layer_types every layer attends the
    # window where there is one (configuration_ministral.py).
    windowed = hf_config.get('sliding_window') is not None
    return base_config(hf_config, used, rope=PLAIN_SPELLING,
                       layer_types=('sliding_attention' if windowed else 'full_attention',),
                       reads=frozenset({'layer_types', 'sliding_window'}))


# MixtralConfig's default_theta (configuration_mixtral.py:45), the base an omitted rope_theta means.
_MIXTRAL_SPELLING = RopeSpelling(theta=1000000.0)


def _mixtral_config(hf_config, used):
    config = _mistral_config(hf_config, used, _MIXTRAL_SPELLING)
    used.update(('num_local_experts', 'router_jitter_noise'))
    if hf_config.get('router_jitter_noise', 0.0):
        refuse('router_jitter_noise', 'training-time input jitter has no counterpart')
    experts = _records.integer(hf_config['num_local_experts'], 'num_local_experts')
    config['mixture'] = native_fields(Mixture)(
        top_k=softmax_top_k(hf_config, used), experts=experts)
    return config


def _mistral_config(hf_config, used, spelling: RopeSpelling = PLAIN_SPELLING):
    """MistralConfig's decoder, under `spelling`, the class's rope (Mixtral's base differs)."""
    layers = int(hf_config['num_hidden_layers'])
    window = hf_config.get('sliding_window')
    # MistralConfig declares sliding_window alone; MistralAttention builds
    # no biases.
    return base_config(hf_config, used, rope=spelling, layer_types=(
        'full_attention' if window is None else 'sliding_attention',) * layers,
        reads=frozenset({'sliding_window'}))


def _granitemoe_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read Granite's embedding, attention, residual and head multipliers.

    Every layer uses ordinary GQA and a normalized softmax MoE, without
    shared experts or a balancing bias. The reference reads hidden_act, not
    the released activation_function beside it, and applies no
    router_jitter_noise, so both are read as the reference reads them: not at all.
    """
    used.update(('activation_function', 'router_jitter_noise'))
    config = llama_config(hf_config, used)
    for source, target in (('embedding_multiplier', 'embedding_multiplier'),
                           ('residual_multiplier', 'residual_multiplier'),
                           ('logits_scaling', 'logits_scaling'), ('attention_multiplier', 'attention_scale')):
        used.add(source)
        config[target] = _records.number(hf_config.get(source, 1.0), source)

    config['mixture'] = native_fields(Mixture)(
        experts=_records.integer(hf_config.get('num_local_experts', 8), 'num_local_experts'),
        top_k=softmax_top_k(hf_config, used))
    used.add('num_local_experts')
    return config


_GRANITEMOE_NAMES = (
    ('block_sparse_moe.router.layer', 'mlp.gate'), ('block_sparse_moe.router', 'mlp.gate'),
    ('block_sparse_moe', 'mlp'),
)
_GRANITEMOE_PACKED = (
    Packed('.block_sparse_moe.input_linear.weight',
           ('.block_sparse_moe.experts.gate_proj', '.block_sparse_moe.experts.up_proj'), -1, (0, 2, 1)),
    Packed('.block_sparse_moe.output_linear.weight',
           ('.block_sparse_moe.experts.down_proj',), -1, (0, 2, 1)),
    *FUSED_EXPERTS,
)


def _granitemoe_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    return qwen35_moe_path(renamed(name, _GRANITEMOE_NAMES), config)


GRANITEMOE = DecoderFamily(
    ('granitemoe',),
    _granitemoe_config,
    lambda fields: bool(fields.mixture is not None and not fields.qk_norm
                        and (fields.embedding_multiplier != 1.0 or fields.residual_multiplier != 1.0
                             or fields.logits_scaling != 1.0 or fields.attention_scale is not None)),
    'granitemoe',
    'GraniteMoeForCausalLM',
    lambda model: {},
    weight_path=_granitemoe_path,
    export_path=partial(renamed_name, _GRANITEMOE_NAMES),
    packed=_GRANITEMOE_PACKED,
    preserve_source_layout=True,
    padding_idx=True,
    defaults={'max_position_embeddings': 2048},
)

MIXTRAL = DecoderFamily(
    ('mixtral',),
    _mixtral_config,
    lambda fields: fields.mixture is not None,
    'mixtral',
    'MixtralForCausalLM',
    lambda model: {},
    weight_path=partial(renamed_path, MOE_NAMES),
    preserve_source_layout=True,
    # MixtralConfig's defaults (its rope base is its spelling's).
    defaults={'num_experts_per_tok': 2, 'num_local_experts': 8, 'rms_norm_eps': 1e-5,
              'num_key_value_heads': 8, 'max_position_embeddings': 131072},
    padding_idx=True,
)


# MistralConfig has no layer_types: its window is on every layer.
MISTRAL = DecoderFamily(
    ('mistral',),
    _mistral_config,
    every_layer_windowed,
    'mistral',
    'MistralForCausalLM',
    lambda model: {'layer_types': None, 'sliding_window': model.kind_of('sliding_attention').window},
    preserve_source_layout=False,
    # MistralConfig's defaults: it attends a 4096-token window unless a config says otherwise, so
    # an export of a model without one says so with a null.
    defaults={'sliding_window': 4096, 'num_key_value_heads': 8, 'max_position_embeddings': 131072},
    export_nulls=('sliding_window',),
    padding_idx=True,
)
