"""OPT's biased pre-norm decoder and two reserved learned-position rows."""

from collections.abc import Mapping

from dew import records
from dew.interop.config_records import native_fields
from dew.interop.decoder_config import DEFAULT_MAX_SEQ_LEN, DecoderFields, _refuse
from dew.interop.decoder_paths import Renames
from dew.nn.backbones.causal_transformer import CausalTransformer


def _opt_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    hidden = records.integer(hf.get('hidden_size', 768), 'hidden_size')
    heads = records.integer(hf.get('num_attention_heads', 12), 'num_attention_heads')
    positions = records.integer(hf.get('max_position_embeddings', 2048), 'max_position_embeddings')
    used.update(('hidden_size', 'num_attention_heads', 'num_hidden_layers', 'ffn_dim',
                 'max_position_embeddings', 'do_layer_norm_before', 'word_embed_proj_dim',
                 'dropout', 'layerdrop', 'init_std', 'enable_bias', 'layer_norm_elementwise_affine',
                 'activation_function', 'tie_word_embeddings', 'vocab_size', 'use_cache',
                 'prefix', 'activation_dropout'))
    if hf.get('activation_dropout', 0):
        _refuse('activation_dropout', 'OPT only exposes residual and attention-probability dropout')
    if hf.get('do_layer_norm_before', True) is not True:
        _refuse('do_layer_norm_before=False', 'OPT post-residual LayerNorm is not represented')
    if hf.get('_remove_final_layer_norm', False):
        _refuse('_remove_final_layer_norm', 'the pre-norm decoder ends with LayerNorm')
    if hf.get('layer_norm_elementwise_affine', True) is not True:
        _refuse('layer_norm_elementwise_affine=False', 'the decoder LayerNorm carries scale and bias')
    embedding = records.integer(hf.get('word_embed_proj_dim') or hidden, 'word_embed_proj_dim')
    if embedding != hidden:
        _refuse('word_embed_proj_dim', 'the embedding-to-decoder projections are not represented')
    if hf.get('layerdrop', 0):
        _refuse('layerdrop', 'stochastic decoder-layer dropping is not represented')
    activation = records.text(hf.get('activation_function', 'relu'), 'activation_function')
    activations = {'relu': 'relu', 'gelu': 'gelu_exact', 'gelu_new': 'gelu'}
    if activation not in activations:
        _refuse(f'activation_function={activation!r}', 'the ungated MLP supports ReLU and GELU')
    return native_fields(CausalTransformer)(
        vocab_size=records.integer(hf.get('vocab_size', 50272), 'vocab_size'),
        emb_features=hidden,
        num_layers=records.integer(hf.get('num_hidden_layers', 12), 'num_hidden_layers'),
        num_heads=heads, num_kv_heads=heads, head_dim=hidden // heads,
        mlp_features=records.integer(hf.get('ffn_dim', 3072), 'ffn_dim'),
        mlp=activations[activation], mlp_bias=bool(hf.get('enable_bias', True)),
        norm_type='layer', norm_bias=True, norm_eps=1e-5,
        qk_norm=False, attention_bias=bool(hf.get('enable_bias', True)),
        position_embedding='learned', position_embedding_size=positions + 2,
        max_seq_len=min(positions, DEFAULT_MAX_SEQ_LEN),
        position_embedding_offset=2,
        tie_embeddings=bool(hf.get('tie_word_embeddings', True)),
        dropout_rate=records.number(hf.get('dropout', .1), 'dropout'),
        attention_dropout_rate=records.number(hf.get('attention_dropout', 0.), 'attention_dropout'),
    )


_OPT_NAMES: Renames = (
    ('model.decoder.final_layer_norm', 'model.norm'), ('model.decoder', 'model'),
    ('self_attn_layer_norm', 'input_layernorm'), ('final_layer_norm', 'post_attention_layernorm'),
    ('fc1', 'mlp.up_proj'), ('fc2', 'mlp.down_proj'), ('self_attn.out_proj', 'self_attn.o_proj'))


def _opt_export(model: CausalTransformer) -> Mapping[str, object]:
    activation = model.mlp
    if not isinstance(activation, str) or activation not in ('relu', 'gelu', 'gelu_exact'):
        _refuse('mlp', 'OPT requires an ungated ReLU or GELU MLP')
    fields: dict[str, object] = {
        'ffn_dim': model.hidden_features, 'word_embed_proj_dim': model.emb_features,
        'enable_bias': model.attention_bias, 'layer_norm_elementwise_affine': True,
        'do_layer_norm_before': True, 'dropout': model.dropout_rate,
        'attention_dropout': model.attention_dropout_rate,
        'max_position_embeddings': (model.position_embedding_size or model.max_seq_len + 2) - 2,
        'activation_function': {'relu': 'relu', 'gelu': 'gelu_new', 'gelu_exact': 'gelu'}[activation],
    }
    fields.update(dict.fromkeys(('intermediate_size', 'rms_norm_eps', 'hidden_act', 'head_dim',
                                 'num_key_value_heads', 'attention_bias', 'rope_theta')))
    return fields
