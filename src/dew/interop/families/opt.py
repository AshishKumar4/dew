"""OPT's biased decoder, pre-norm or post-LN (OPT-350m's), its embedding at a width of its own where the
release projects it in and out, and two reserved learned-position rows."""

from collections.abc import Mapping

from dew import _records
from dew.interop.decoder_config import biases, expect, flag, inert, integer, ungated
from dew.interop.decoder_parts import DecoderFamily, DecoderFields, Renames, context_lengths, spelled
from dew.interop.layouts import ConfigField, ConfigLayout, ConfigPart, constant, unread
from dew.nn.backbones.causal_transformer import CausalTransformer


def _opt_norms(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """OPT's norm placement, both ways: a post-LN decoder (do_layer_norm_before False, OPT-350m) norms its
    stream after each sublayer and ends with no final norm, as one stated without it does
    (modeling_opt.py, OPTDecoder)."""
    used.update(('do_layer_norm_before', '_remove_final_layer_norm'))
    before = hf_config.get('do_layer_norm_before', True) is True
    record.update(pre_norms=before, post_residual_norms=not before,
                  final_norm=before and not hf_config.get('_remove_final_layer_norm', False))
    return record


def _opt_norms_written(model: CausalTransformer) -> Mapping[str, object]:
    return {'do_layer_norm_before': not model.post_residual_norms,
            '_remove_final_layer_norm': not model.post_residual_norms and not model.final_norm}


def _opt_embedding(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The embedding's own width where it differs from the stream's, projected in and out."""
    used.add('word_embed_proj_dim')
    hidden = _records.integer(record['emb_features'], 'hidden_size')
    embedding = _records.integer(hf_config.get('word_embed_proj_dim') or hidden, 'word_embed_proj_dim')
    record['embedding_features'] = None if embedding == hidden else embedding
    return record


def _opt_positions(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """OPT's learned positions, offset by two rows past the stated context."""
    used.add('max_position_embeddings')
    positions = _records.integer(hf_config.get('max_position_embeddings', 2048), 'max_position_embeddings')
    record.update(position_embedding='learned', position_embedding_size=positions + 2,
                  max_seq_len=context_lengths(hf_config, 2048)[0], position_embedding_offset=2)
    return record


def _opt_heads(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    used.add('num_attention_heads')
    heads = _records.integer(hf_config.get('num_attention_heads', 12), 'num_attention_heads')
    record.update(num_heads=heads, num_kv_heads=heads,
                  head_dim=_records.integer(record['emb_features'], 'hidden_size') // heads)
    return record


OPT_CONFIG = ConfigLayout(
    integer('vocab_size', 'vocab_size', 50272), integer('hidden_size', 'emb_features', 768),
    integer('num_hidden_layers', 'num_layers', 12),
    ConfigPart(_opt_heads, lambda model: {'num_attention_heads': model.num_heads}),
    integer('ffn_dim', 'mlp_features', 3072),
    ungated('activation_function', ('relu', 'gelu', 'gelu_new'), 'OPT', default='relu'),
    biases('enable_bias', default=True),
    ConfigPart(_opt_norms, _opt_norms_written),
    ConfigPart(_opt_embedding, lambda model: {'word_embed_proj_dim': model.embedding_features
                                              or model.emb_features}),
    ConfigPart(_opt_positions, lambda model: {'max_position_embeddings': (
        model.position_embedding_size or model.max_seq_len + 2) - 2}),
    flag('tie_word_embeddings', 'tie_embeddings', default=True),
    ConfigField('dropout', 'dropout_rate', 0.1, parse=_records.number),
    ConfigField('attention_dropout', 'attention_dropout_rate', 0.0, parse=_records.number),
    constant('norm_type', value='layer'), constant('norm_bias', value=True), constant('norm_eps', value=1e-5),
    constant('qk_norm', value=False),
    inert('activation_dropout', 'OPT only exposes residual and attention-probability dropout', neutral=0),
    inert('layerdrop', 'stochastic decoder-layer dropping is not represented', neutral=0),
    expect('layer_norm_elementwise_affine', 'the decoder LayerNorm carries scale and bias', value=True),
    unread('init_std', 'use_cache', 'prefix'))
"""OPT's config, both ways."""


_OPT_NAMES: Renames = (
    ('model.decoder.final_layer_norm', 'model.norm'), ('model.decoder', 'model'),
    ('self_attn_layer_norm', 'input_layernorm'), ('final_layer_norm', 'post_attention_layernorm'),
    ('fc1', 'mlp.up_proj'), ('fc2', 'mlp.down_proj'), ('self_attn.out_proj', 'self_attn.o_proj'))


_OPT_LEAVES = {'model.project_in.weight': ('project_in', 'kernel'),
               'model.project_out.weight': ('project_out', 'kernel')}
"""The projections between an embedding of its own width and the stream, past the shared map."""

OPT = DecoderFamily(
    ('opt',),
    OPT_CONFIG,
    lambda fields: fields.position_embedding_offset == 2,
    'opt',
    'OPTForCausalLM',
    routes=spelled(_OPT_NAMES, leaves=_OPT_LEAVES),
    preserve_source_layout=False,
    tied_head_names=('lm_head.weight', 'model.decoder.embed_tokens.weight'),
    defaults={'pad_token_id': 1},
    padding_idx=True,
)
