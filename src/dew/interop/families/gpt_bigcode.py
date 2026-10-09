"""GPTBigCode (SantaCoder, StarCoder): GPT-2's block under torch Linears, its
queries, keys and values fused in one `c_attn`, multi-query by default."""

from collections.abc import Mapping
from functools import partial

import numpy as np

from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    Renames,
    refuse,
    renamed_name,
    renamed_path,
)
from dew.interop.families.gpt2 import gpt2_config, gpt2_export
from dew.interop.families.gpt_neox import gpt_neox_export_weights, gpt_neox_prepare
from dew.interop.safetensors_io import LazyTensors
from dew.nn.backbones.causal_transformer import CausalTransformer


def _gpt_bigcode_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    """GPT-2's fields, with one key and value head under `multi_query`.

    transformers 5.16.1 runs the softmax in float32 whatever
    `attention_softmax_in_fp32` and `scale_attention_softmax_in_fp32` say
    (`eager_attention_forward` in modeling_gpt_bigcode.py), as Dew's does,
    and no forward reads SantaCoder's `runner_max_sequence_length`.
    """
    used.update(('multi_query', 'attention_softmax_in_fp32', 'scale_attention_softmax_in_fp32',
                 'runner_max_sequence_length', 'num_key_value_heads'))
    config = gpt2_config(hf, used)
    kv_heads = 1 if hf.get('multi_query', True) else config['num_heads']
    # GPTBigCodeConfig derives num_key_value_heads from multi_query, and 5.x saves it.
    if hf.get('num_key_value_heads', kv_heads) != kv_heads:
        refuse('num_key_value_heads', f'GPTBigCode attends with {kv_heads} key and value heads under '
               f'multi_query={bool(hf.get("multi_query", True))}')
    config['num_kv_heads'] = kv_heads
    return config


_GPT_BIGCODE_NAMES: Renames = (
    ('transformer.wte', 'model.embed_tokens'), ('transformer.wpe', 'model.embed_positions'),
    ('transformer.ln_f', 'model.norm'), ('transformer.h', 'model.layers'),
    ('ln_1', 'input_layernorm'), ('ln_2', 'post_attention_layernorm'),
    ('attn.c_proj', 'self_attn.o_proj'), ('mlp.c_fc', 'mlp.up_proj'), ('mlp.c_proj', 'mlp.down_proj'))


def _gpt_bigcode_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    # The fused qkv has three leaves, split by `_gpt_bigcode_prepare`.
    return None if '.attn.c_attn.' in name else renamed_path(_GPT_BIGCODE_NAMES, name, config)


def _gpt_bigcode_prepare(tensors: Mapping[str, np.ndarray],
                         config: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
    """`c_attn` split into q, k and v: the query's heads, then the one key
    head and the one value head under multi-query, each head's query, key
    and value side by side otherwise (`GPTBigCodeAttention.forward`)."""
    if config is None:
        raise ValueError('GPTBigCode fused qkv requires translated head geometry')
    return gpt_neox_prepare(tensors, config, attention_name='attn', fused_name='c_attn',
                            interleaved=config['num_kv_heads'] != 1)


def _gpt_bigcode_export_weights(family: DecoderFamily, model: CausalTransformer,
                                variables: Mapping[str, object], config: Mapping[str, object]) -> LazyTensors:
    return gpt_neox_export_weights(family, model, variables, config, attention_name='attn',
                                   fused_name='c_attn', interleaved=model.kv_heads != 1)


def _gpt_bigcode_export(model: CausalTransformer) -> Mapping[str, object]:
    fields = dict(gpt2_export(model))
    fields.update({'multi_query': model.kv_heads == 1,
                   'activation_function': {'gelu': 'gelu_pytorch_tanh', 'gelu_exact': 'gelu',
                                           'relu': 'relu'}[str(model.mlp)]})
    return fields


# A multi-head GPTBigCode computes GPT-2's model, which exports as GPT-2.
GPT_BIGCODE = DecoderFamily(
    ('gpt_bigcode',), _gpt_bigcode_config,
    lambda fields: fields.position_embedding == 'learned' and fields.kv_heads == 1 < fields.num_heads,
    'gpt_bigcode', 'GPTBigCodeForCausalLM', _gpt_bigcode_export,
    weight_path=_gpt_bigcode_path, export_path=partial(renamed_name, _GPT_BIGCODE_NAMES),
    prepare=_gpt_bigcode_prepare, export_weights=_gpt_bigcode_export_weights, preserve_source_layout=False,
    tied_head_names=('lm_head.weight', 'transformer.wte.weight'),
)
