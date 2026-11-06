"""BLOOM's ALiBi attention, embedding LayerNorm and head-interleaved qkv."""

from collections.abc import Mapping

from dew import _records
from dew.interop.decoder_config import flag, inert, integer
from dew.interop.decoder_parts import (
    DEFAULT_MAX_SEQ_LEN,
    MEGATRON_NAMES,
    DecoderFamily,
    DecoderFields,
    Renames,
    grouped_qkv,
    refuse,
    spelled,
)
from dew.interop.layouts import ConfigField, ConfigLayout, ConfigPart, constant, unread


def _bloom_heads(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    used.update(('num_attention_heads', 'n_head'))
    heads = _records.integer(hf_config.get('num_attention_heads', hf_config.get('n_head', 8)), 'n_head')
    hidden = _records.integer(record['emb_features'], 'hidden_size')
    if hidden % heads:
        refuse('hidden_size/n_head', 'the hidden width must divide into whole attention heads')
    record.update(num_heads=heads, num_kv_heads=heads, head_dim=hidden // heads, mlp_features=4 * hidden)
    return record


def _exact(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    used.update(('pretraining_tp', 'slow_but_exact'))
    tensor_parallel = _records.integer(hf_config.get('pretraining_tp', 1), 'pretraining_tp')
    if hf_config.get('slow_but_exact', False) and tensor_parallel > 1:
        refuse('slow_but_exact', 'the tensor-parallel reference drops projection biases')
    return record


BLOOM_CONFIG = ConfigLayout(
    integer('vocab_size', 'vocab_size', 250880), integer('n_embed', 'emb_features', 64, 'hidden_size'),
    integer('n_layer', 'num_layers', 2, 'num_hidden_layers'),
    ConfigPart(_bloom_heads, lambda model: {'n_head': model.num_heads}),
    ConfigField('layer_norm_epsilon', 'norm_eps', 1e-5, parse=_records.number),
    flag('tie_word_embeddings', 'tie_embeddings', default=True),
    ConfigField('hidden_dropout', 'dropout_rate', 0.0, parse=_records.number),
    ConfigField('attention_dropout', 'attention_dropout_rate', 0.0, parse=_records.number),
    ConfigPart(_exact, lambda model: {'pretraining_tp': 1, 'slow_but_exact': False}),
    inert('apply_residual_connection_post_layernorm', 'BLOOM residuals read the unnormalized input',
          neutral=False),
    constant('mlp', value='gelu'), constant('mlp_bias', value=True), constant('norm_type', value='layer'),
    constant('norm_bias', value=True), constant('embedding_norm', value=True),
    constant('qk_norm', value=False),
    constant('attention_bias', value=True), constant('position_embedding', value='alibi'),
    constant('max_seq_len', value=DEFAULT_MAX_SEQ_LEN),
    # The released 560m config carries Megatron's kernel-fusion flags and ALiBi offset, which
    # transformers' BLOOM forward never reads.
    unread('use_cache', 'attention_softmax_in_fp32', 'bias_dropout_fusion', 'masked_softmax_fusion',
           'skip_bias_add', 'skip_bias_add_qkv', 'offset_alibi', 'n_inner', 'seq_length'))
"""BLOOM's config, both ways."""


_BLOOM_NAMES: Renames = (
    ('transformer.word_embeddings_layernorm', 'model.embedding_layernorm'), *MEGATRON_NAMES)


BLOOM = DecoderFamily(
    ('bloom',), BLOOM_CONFIG,
    lambda fields: fields.position_embedding == 'alibi' and fields.embedding_norm,
    'bloom', 'BloomForCausalLM',
    routes=spelled(_BLOOM_NAMES),
    packed=grouped_qkv('.self_attention.query_key_value'),
    preserve_source_layout=False,
    tied_head_names=('lm_head.weight', 'transformer.word_embeddings.weight'),
)
