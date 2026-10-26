"""OPT's biased pre-norm decoder and two reserved learned-position rows."""

from collections.abc import Mapping
from functools import partial

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    Renames,
    context_lengths,
    hf_ungated,
    refuse,
    renamed_name,
    renamed_path,
    ungated_mlp,
)
from dew.nn.backbones.causal_transformer import CausalTransformer


def _opt_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    hidden = _records.integer(hf.get('hidden_size', 768), 'hidden_size')
    heads = _records.integer(hf.get('num_attention_heads', 12), 'num_attention_heads')
    positions = _records.integer(hf.get('max_position_embeddings', 2048), 'max_position_embeddings')
    used.update(('hidden_size', 'num_attention_heads', 'num_hidden_layers', 'ffn_dim',
                 'max_position_embeddings', 'do_layer_norm_before', 'word_embed_proj_dim',
                 'dropout', 'layerdrop', 'init_std', 'enable_bias', 'layer_norm_elementwise_affine',
                 'activation_function', 'tie_word_embeddings', 'vocab_size', 'use_cache',
                 'prefix', 'activation_dropout'))
    if hf.get('activation_dropout', 0):
        refuse('activation_dropout', 'OPT only exposes residual and attention-probability dropout')
    if hf.get('do_layer_norm_before', True) is not True:
        refuse('do_layer_norm_before=False', 'OPT post-residual LayerNorm is not represented')
    if hf.get('_remove_final_layer_norm', False):
        refuse('_remove_final_layer_norm', 'the pre-norm decoder ends with LayerNorm')
    if hf.get('layer_norm_elementwise_affine', True) is not True:
        refuse('layer_norm_elementwise_affine=False', 'the decoder LayerNorm carries scale and bias')
    embedding = _records.integer(hf.get('word_embed_proj_dim') or hidden, 'word_embed_proj_dim')
    if embedding != hidden:
        refuse('word_embed_proj_dim', 'the embedding-to-decoder projections are not represented')
    if hf.get('layerdrop', 0):
        refuse('layerdrop', 'stochastic decoder-layer dropping is not represented')
    activation = _records.text(hf.get('activation_function', 'relu'), 'activation_function')
    mlp = ungated_mlp(activation, ('relu', 'gelu', 'gelu_new'), 'activation_function', 'OPT')
    return native_fields(CausalTransformer)(
        vocab_size=_records.integer(hf.get('vocab_size', 50272), 'vocab_size'),
        emb_features=hidden,
        num_layers=_records.integer(hf.get('num_hidden_layers', 12), 'num_hidden_layers'),
        num_heads=heads, num_kv_heads=heads, head_dim=hidden // heads,
        mlp_features=_records.integer(hf.get('ffn_dim', 3072), 'ffn_dim'),
        mlp=mlp, mlp_bias=bool(hf.get('enable_bias', True)),
        norm_type='layer', norm_bias=True, norm_eps=1e-5,
        qk_norm=False, attention_bias=bool(hf.get('enable_bias', True)),
        position_embedding='learned', position_embedding_size=positions + 2,
        max_seq_len=context_lengths(hf, 2048)[0],
        position_embedding_offset=2,
        tie_embeddings=bool(hf.get('tie_word_embeddings', True)),
        dropout_rate=_records.number(hf.get('dropout', .1), 'dropout'),
        attention_dropout_rate=_records.number(hf.get('attention_dropout', 0.), 'attention_dropout'),
    )


_OPT_NAMES: Renames = (
    ('model.decoder.final_layer_norm', 'model.norm'), ('model.decoder', 'model'),
    ('self_attn_layer_norm', 'input_layernorm'), ('final_layer_norm', 'post_attention_layernorm'),
    ('fc1', 'mlp.up_proj'), ('fc2', 'mlp.down_proj'), ('self_attn.out_proj', 'self_attn.o_proj'))


def _opt_export(model: CausalTransformer) -> Mapping[str, object]:
    activation = hf_ungated(model.mlp, 'OPT')
    fields: dict[str, object] = {
        'ffn_dim': model.hidden_features, 'word_embed_proj_dim': model.emb_features,
        'enable_bias': model.attention_bias, 'layer_norm_elementwise_affine': True,
        'do_layer_norm_before': True, 'dropout': model.dropout_rate,
        'attention_dropout': model.attention_dropout_rate,
        'max_position_embeddings': (model.position_embedding_size or model.max_seq_len + 2) - 2,
        'activation_function': activation,
    }
    fields.update(dict.fromkeys(('intermediate_size', 'rms_norm_eps', 'hidden_act', 'head_dim',
                                 'num_key_value_heads', 'attention_bias', 'rope_theta')))
    return fields


OPT = DecoderFamily(
    ('opt',),
    _opt_config,
    lambda fields: fields.position_embedding_offset == 2,
    'opt',
    'OPTForCausalLM',
    _opt_export,
    weight_path=partial(renamed_path, _OPT_NAMES),
    export_path=partial(renamed_name, _OPT_NAMES),
    preserve_source_layout=False,
    tied_head_names=('lm_head.weight', 'model.decoder.embed_tokens.weight'),
    defaults={'pad_token_id': 1},
    padding_idx=True,
)
