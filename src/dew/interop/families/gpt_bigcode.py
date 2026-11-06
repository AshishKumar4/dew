"""GPTBigCode (SantaCoder, StarCoder): GPT-2's block under torch Linears, its
queries, keys and values fused in one `c_attn`, multi-query by default."""

from collections.abc import Mapping

from dew import _records
from dew.interop.decoder_config import ungated
from dew.interop.decoder_parts import DecoderFamily, DecoderFields, grouped_qkv, refuse, spelled
from dew.interop.families.gpt2 import GPT2_NAMES, gpt2_layout
from dew.interop.layouts import ConfigLayout, ConfigPart, unread


def _multi_query(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """One key and value head under `multi_query`, which GPTBigCodeConfig derives num_key_value_heads from
    (and 5.x saves)."""
    used.update(('multi_query', 'num_key_value_heads'))
    kv_heads = 1 if hf_config.get('multi_query', True) else _records.integer(record['num_heads'], 'n_head')
    if hf_config.get('num_key_value_heads', kv_heads) != kv_heads:
        refuse('num_key_value_heads', f'GPTBigCode attends with {kv_heads} key and value heads under '
               f'multi_query={bool(hf_config.get("multi_query", True))}')
    record['num_kv_heads'] = kv_heads
    return record


GPT_BIGCODE_CONFIG = ConfigLayout(
    gpt2_layout(ungated('activation_function',
                                   ('gelu_new', 'gelu_fast', 'gelu_pytorch_tanh', 'gelu', 'relu'),
                                   'GPTBigCode', tanh='gelu_pytorch_tanh')),
    ConfigPart(_multi_query, lambda model: {'multi_query': model.kv_heads == 1}),
    # transformers 5.16.1 runs the softmax in float32 whatever these say (`eager_attention_forward` in
    # modeling_gpt_bigcode.py), as Dew's does, and no forward reads SantaCoder's runner_max_sequence_length.
    unread('attention_softmax_in_fp32', 'scale_attention_softmax_in_fp32', 'runner_max_sequence_length'))
"""GPTBigCode's config, both ways: GPT-2's fields, with one key and value head under `multi_query`."""


# A multi-head GPTBigCode computes GPT-2's model, which exports as GPT-2.
GPT_BIGCODE = DecoderFamily(
    ('gpt_bigcode',), GPT_BIGCODE_CONFIG,
    lambda fields: fields.position_embedding == 'learned' and fields.kv_heads == 1 < fields.num_heads,
    'gpt_bigcode', 'GPTBigCodeForCausalLM',
    routes=spelled(GPT2_NAMES),
    packed=grouped_qkv('.attn.c_attn'), preserve_source_layout=False,
    tied_head_names=('lm_head.weight', 'transformer.wte.weight'),
)
