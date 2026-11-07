"""Translate Llama, Mistral and Mixtral, the dense block the other families vary.

Llama's own config is the shared decoder's layout (`decoder_layout`) over
the fields LlamaConfig declares, so what stands here is only what the
variants add to it, each read and written by one declaration. Mistral adds a
sliding window on every layer and drops the attention biases. Ministral names
a pattern of sliding and full layers. Mixtral adds a softmax-routed
feed-forward and the expert tensor names that routing brings.
"""

from __future__ import annotations

import dataclasses

from dew import _records
from dew.interop.decoder_config import LOCAL_EXPERTS, DecoderSpelling, decoder_layout, inert
from dew.interop.decoder_parts import (
    FUSED_EXPERTS,
    MOE_NAMES,
    DecoderFamily,
    Packed,
    RopeSpelling,
    every_layer_windowed,
    spelled,
)
from dew.interop.families.qwen import QWEN_MOE_ROUTES
from dew.interop.layouts import ConfigField, ConfigLayout, Spelled, unread

# LlamaConfig declares attention_bias, mlp_bias and no window: LlamaAttention attends every key, and
# LlamaMLP's projections carry mlp_bias's biases. LlamaRotaryEmbedding builds any rope_type the rope
# utilities do, YaRN among them (modeling_llama.py:82-86).
LLAMA_DECODER = DecoderSpelling(reads=frozenset({'attention_bias', 'mlp_bias'}))


# MixtralConfig's default_theta (configuration_mixtral.py:45), the base an omitted rope_theta means.
_MIXTRAL_SPELLING = RopeSpelling(theta=1000000.0)


# MistralConfig declares sliding_window alone, which every layer attends where it is set;
# MistralAttention builds no biases.
_MISTRAL_DECODER = DecoderSpelling(reads=frozenset({'sliding_window'}), windowed=True)


_GRANITE_MULTIPLIERS = tuple(
    ConfigField(source, target, default=1.0, parse=_records.number)
    for source, target in (('embedding_multiplier', 'embedding_multiplier'),
                           ('residual_multiplier', 'residual_multiplier'),
                           ('logits_scaling', 'logits_scaling'), ('attention_multiplier', 'attention_scale')))
"""Granite's embedding, residual, head and attention multipliers, each one field both ways."""

_NO_JITTER = inert('router_jitter_noise', 'training-time input jitter has no counterpart', neutral=0.0)


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


GRANITEMOE = DecoderFamily(
    ('granitemoe',),
    # GraniteMoE's reference reads hidden_act, not the released activation_function beside it, and
    # applies no router_jitter_noise: ordinary GQA under a normalized softmax MoE on every layer, without
    # shared experts or a balancing bias.
    ConfigLayout(decoder_layout(LLAMA_DECODER), *_GRANITE_MULTIPLIERS, LOCAL_EXPERTS,
                 unread('activation_function', 'router_jitter_noise')),
    lambda fields: bool(fields.mixture is not None and not fields.qk_norm
                        and (fields.embedding_multiplier != 1.0 or fields.residual_multiplier != 1.0
                             or fields.logits_scaling != 1.0 or fields.attention_scale is not None)),
    'granitemoe',
    'GraniteMoeForCausalLM',
    routes=Spelled(_GRANITEMOE_NAMES, QWEN_MOE_ROUTES),
    packed=_GRANITEMOE_PACKED,
    preserve_source_layout=True,
    padding_idx=True,
    defaults={'max_position_embeddings': 2048, 'num_local_experts': 8},
)

MIXTRAL = DecoderFamily(
    ('mixtral',),
    ConfigLayout(decoder_layout(dataclasses.replace(_MISTRAL_DECODER, rope=_MIXTRAL_SPELLING)), LOCAL_EXPERTS,
                 _NO_JITTER),
    lambda fields: fields.mixture is not None,
    'mixtral',
    'MixtralForCausalLM',
    routes=spelled(MOE_NAMES),
    preserve_source_layout=True,
    # MixtralConfig's defaults (its rope base is its spelling's).
    defaults={'num_experts_per_tok': 2, 'num_local_experts': 8, 'rms_norm_eps': 1e-5,
              'num_key_value_heads': 8, 'max_position_embeddings': 131072},
    padding_idx=True,
)


# MistralConfig has no layer_types: its window is on every layer.
MISTRAL = DecoderFamily(
    ('mistral',),
    ConfigLayout(decoder_layout(_MISTRAL_DECODER)),
    every_layer_windowed,
    'mistral',
    'MistralForCausalLM',
    preserve_source_layout=False,
    # MistralConfig's defaults: it attends a 4096-token window unless a config says otherwise, so
    # an export of a model without one says so with a null.
    defaults={'sliding_window': 4096, 'num_key_value_heads': 8, 'max_position_embeddings': 131072},
    export_nulls=('sliding_window',),
    padding_idx=True,
)


MINISTRAL = DecoderFamily(
    ("ministral",),
    # MinistralConfig declares layer_types and sliding_window, and its attention builds no biases. Without
    # layer_types every layer attends the window where there is one (configuration_ministral.py).
    ConfigLayout(decoder_layout(DecoderSpelling(reads=frozenset({'layer_types', 'sliding_window'}),
                                                windowed=True))),
    lambda fields: "sliding_attention" in (fields.layer_types or ()),
    "ministral",
    "MinistralForCausalLM",
    preserve_source_layout=False,
    # MinistralConfig's defaults for the fields its decoder reads.
    defaults={"sliding_window": 4096, "num_key_value_heads": 8, "max_position_embeddings": 131072},
    padding_idx=True,
)

LLAMA = DecoderFamily(
    ("llama",),
    ConfigLayout(decoder_layout(LLAMA_DECODER)),
    lambda fields: True,
    "llama",
    "LlamaForCausalLM",
    preserve_source_layout=False,
    # LlamaConfig's context (configuration_llama.py:71), which a YaRN record's omitted lengths read.
    defaults={"max_position_embeddings": 2048},
    # Megatron's names for its fields, which Nanbeige's releases repeat beside them.
    repeated={"ffn_hidden_size": "intermediate_size", "num_query_groups": "num_key_value_heads",
              "rotary_base": "rope_theta", "kv_channels": "head_dim",
              "seq_length": "max_position_embeddings"},
    padding_idx=True,
)
