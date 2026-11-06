"""GPTBigCode (SantaCoder, StarCoder): GPT-2's block under torch Linears, its
queries, keys and values fused in one `c_attn`, multi-query by default."""

from collections.abc import Mapping
from functools import partial

from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    decoder_tensors,
    grouped_qkv,
    hf_ungated,
    refuse,
    renamed_name,
    renamed_path,
)
from dew.interop.families.gpt2 import GPT2_NAMES, gpt2_config, gpt2_export
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


def _gpt_bigcode_export(model: CausalTransformer) -> Mapping[str, object]:
    fields = dict(gpt2_export(model))
    fields.update({'multi_query': model.kv_heads == 1,
                   'activation_function': hf_ungated(model.mlp, 'GPTBigCode', tanh='gelu_pytorch_tanh')})
    return fields


# A multi-head GPTBigCode computes GPT-2's model, which exports as GPT-2.
GPT_BIGCODE = DecoderFamily(
    ('gpt_bigcode',), _gpt_bigcode_config,
    lambda fields: fields.position_embedding == 'learned' and fields.kv_heads == 1 < fields.num_heads,
    'gpt_bigcode', 'GPTBigCodeForCausalLM', _gpt_bigcode_export,
    weight_path=partial(renamed_path, GPT2_NAMES), export_path=partial(renamed_name, GPT2_NAMES),
    packed=grouped_qkv('.attn.c_attn'), export_weights=decoder_tensors, preserve_source_layout=False,
    tied_head_names=('lm_head.weight', 'transformer.wte.weight'),
)
