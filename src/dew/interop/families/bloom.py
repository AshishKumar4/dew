"""BLOOM's ALiBi attention, embedding LayerNorm and head-interleaved qkv."""

from collections.abc import Mapping
from functools import partial

from dew import records
from dew.interop.config_records import native_fields
from dew.interop.families.gpt_neox import gpt_neox_export_weights, gpt_neox_prepare
from dew.interop.hf_decoders import (
    DEFAULT_MAX_SEQ_LEN,
    DecoderFamily,
    DecoderFields,
    Renames,
    refuse,
    renamed_name,
    renamed_path,
)
from dew.nn.backbones.causal_transformer import CausalTransformer


def _bloom_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    hidden = records.integer(hf.get('n_embed', hf.get('hidden_size', 64)), 'hidden_size/n_embed')
    heads = records.integer(hf.get('num_attention_heads', hf.get('n_head', 8)), 'n_head')
    used.update(('hidden_size', 'n_embed', 'n_head', 'num_attention_heads', 'n_layer', 'vocab_size',
                 'layer_norm_epsilon', 'apply_residual_connection_post_layernorm',
                 'hidden_dropout', 'attention_dropout', 'pretraining_tp', 'slow_but_exact',
                 'tie_word_embeddings', 'use_cache'))
    # The released 560m config carries Megatron's kernel-fusion flags and
    # ALiBi offset, which transformers' BLOOM forward never reads.
    used.update(('attention_softmax_in_fp32', 'bias_dropout_fusion', 'masked_softmax_fusion',
                 'skip_bias_add', 'skip_bias_add_qkv', 'offset_alibi', 'n_inner', 'seq_length'))
    if hf.get('apply_residual_connection_post_layernorm', False):
        refuse('apply_residual_connection_post_layernorm', 'BLOOM residuals read the unnormalized input')
    if hf.get('slow_but_exact', False) and records.integer(hf.get('pretraining_tp', 1), 'pretraining_tp') > 1:
        refuse('slow_but_exact', 'the tensor-parallel reference drops projection biases')
    if hidden % heads:
        refuse('hidden_size/n_head', 'the hidden width must divide into whole attention heads')
    return native_fields(CausalTransformer)(
        vocab_size=records.integer(hf.get('vocab_size', 250880), 'vocab_size'),
        emb_features=hidden,
        num_layers=records.integer(hf.get('num_hidden_layers', hf.get('n_layer', 2)), 'n_layer'),
        num_heads=heads, num_kv_heads=heads, head_dim=hidden // heads,
        mlp_features=4 * hidden, mlp='gelu', mlp_bias=True,
        norm_type='layer', norm_bias=True, embedding_norm=True,
        norm_eps=records.number(hf.get('layer_norm_epsilon', 1e-5), 'layer_norm_epsilon'),
        qk_norm=False, attention_bias=True, position_embedding='alibi', max_seq_len=DEFAULT_MAX_SEQ_LEN,
        tie_embeddings=bool(hf.get('tie_word_embeddings', True)),
        dropout_rate=records.number(hf.get('hidden_dropout', 0.), 'hidden_dropout'),
        attention_dropout_rate=records.number(hf.get('attention_dropout', 0.), 'attention_dropout'),
    )


_BLOOM_NAMES: Renames = (
    ('transformer.word_embeddings_layernorm', 'model.embedding_layernorm'),
    ('transformer.word_embeddings', 'model.embed_tokens'), ('transformer.ln_f', 'model.norm'),
    ('transformer.h', 'model.layers'), ('self_attention.dense', 'self_attn.o_proj'),
    ('mlp.dense_h_to_4h', 'mlp.up_proj'), ('mlp.dense_4h_to_h', 'mlp.down_proj'))


def _bloom_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    if '.self_attention.query_key_value.' in name:
        return None
    return renamed_path(_BLOOM_NAMES, name, config)


def _bloom_export(model: CausalTransformer) -> Mapping[str, object]:
    fields: dict[str, object] = {
        'n_layer': model.num_layers, 'n_head': model.num_heads,
        'layer_norm_epsilon': model.norm_eps, 'apply_residual_connection_post_layernorm': False,
        'hidden_dropout': model.dropout_rate, 'attention_dropout': model.attention_dropout_rate,
        'pretraining_tp': 1, 'slow_but_exact': False,
    }
    fields.update(dict.fromkeys(('num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
                                 'head_dim', 'intermediate_size', 'max_position_embeddings',
                                 'rms_norm_eps', 'attention_bias', 'hidden_act', 'rope_theta')))
    return fields


BLOOM = DecoderFamily(
    ('bloom',), _bloom_config,
    lambda fields: fields.position_embedding == 'alibi' and fields.embedding_norm,
    'bloom', 'BloomForCausalLM', _bloom_export,
    weight_path=_bloom_path, export_path=partial(renamed_name, _BLOOM_NAMES),
    prepare=partial(gpt_neox_prepare, attention_name='self_attention'),
    export_weights=partial(gpt_neox_export_weights, attention_name='self_attention'),
    preserve_source_layout=False,
    tied_head_names=('lm_head.weight', 'transformer.word_embeddings.weight'),
)
