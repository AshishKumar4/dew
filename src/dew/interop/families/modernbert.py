"""ModernBERT's bidirectional encoder and its masked-LM head.

The encoder is a pre-norm block stack over normalized token embeddings, with
the first block reading them unnormed, alternating global and local layers
under separate rope bases, fused qkv and GeGLU projections, and a final norm
(modeling_modernbert.py, transformers 5.16.1). ModernBertForMaskedLM adds a
prediction head (dense, activation, norm) and a tied, biased projection.

The local layers keep their window on both sides of a query
(`LayerKind.bidirectional_window`), which `local_attention` (causal only)
does not run, so they build the dense [S, S] logits a global layer does: at
8192 tokens a local layer costs what a global one costs.
"""

from collections.abc import Mapping

from dew import _records
from dew.interop.decoder_config import flag, inert, integer
from dew.interop.decoder_parts import (
    ACTIVATIONS,
    DecoderFamily,
    DecoderFields,
    Leaves,
    Packed,
    Renames,
    RopeSpelling,
    context_lengths,
    decoder_kinds,
    hf_activation,
    hf_ungated,
    read_rope,
    refuse,
    specified_layer_types,
    spelled,
    ungated_mlp,
)
from dew.interop.layouts import ConfigField, ConfigLayout, ConfigPart, Spelled, Table, constant, unread
from dew.interop.unread import Stored
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.registry import Configured

_ENCODER = 'ModernBertModel'
_MASKED_LM = 'ModernBertForMaskedLM'


_MODERNBERT_SPELLING = RopeSpelling(theta=160000.0, per_kind=True, local_theta=10000.0,
                                    legacy_kinds=('full_attention', 'sliding_attention'),
                                    local_field='local_rope_theta')
"""ModernBertConfig's rope: a record per layer type, at 160000 on the full layers and 10000 on the
sliding ones where the file names no base (its `default_theta`)."""


def _heads(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """Every head its own key/value head, the hidden width split into them."""
    hidden = _records.integer(record['emb_features'], 'emb_features')
    heads = _records.integer(record['num_heads'], 'num_heads')
    if hidden % heads:
        refuse('hidden_size/num_attention_heads', 'the hidden width must divide into whole heads')
    record.update(num_kv_heads=heads, head_dim=hidden // heads)
    return record


def _gated(value: Configured, key: str) -> str:
    activation = _records.text(value, key)
    if activation not in ACTIVATIONS:
        refuse(f'{key}={activation!r}', f'the GLU supports {sorted(ACTIVATIONS)}')
    return ACTIVATIONS[activation]


def _context(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    used.add('max_position_embeddings')
    record['max_seq_len'] = context_lengths(hf_config, 8192)[0]
    return record


def _layers(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The layers' kinds and their rotary, either spelling. Releases before transformers 5 state
    `global_rope_theta`, `local_rope_theta` and `global_attn_every_n_layers`; later ones nest the bases
    under `rope_parameters` and list `layer_types`. The local window `local_attention` is the whole span,
    so a query reads `local_attention // 2` keys on either side, which is a symmetric window of that plus
    one."""
    used.update(('global_attn_every_n_layers', 'global_rope_theta', 'local_attention'))
    every = _records.integer(hf_config.get('global_attn_every_n_layers', 3), 'global_attn_every_n_layers')
    layer_types = specified_layer_types(hf_config, used,
                                        ('full_attention',) + ('sliding_attention',) * (every - 1))
    # ModernBertConfig keeps a record per layer type, filling the full one's base from
    # global_rope_theta and the sliding one's from local_rope_theta, else its defaults
    # (configuration_modernbert.py:77, :141-147).
    ropes = read_rope({'rope_theta': hf_config.get('global_rope_theta'), **hf_config}, used,
                      _MODERNBERT_SPELLING)
    if ropes.scaling is not None or ropes.local_scaling is not None:
        refuse('rope_parameters', 'ModernBERT rotates at its plain frequencies')
    window = _records.integer(hf_config.get('local_attention', 128), 'local_attention') // 2 + 1
    kinds = decoder_kinds(layer_types, window, ropes.local_theta, None, None)
    if 'sliding_attention' in kinds:
        kinds['sliding_attention']['bidirectional_window'] = True
    record.update(rope_theta=ropes.theta, layer_types=layer_types, kinds=kinds)
    return record


def _layers_written(model: CausalTransformer) -> Mapping[str, object]:
    types = model.per_layer_types
    sliding = model.kind_of('sliding_attention') if 'sliding_attention' in types else None
    local_theta = model.rope_theta if sliding is None else sliding.rope_theta or model.rope_theta
    written: dict[str, object] = {
        'layer_types': list(types),
        'rope_parameters': {
            'full_attention': {'rope_type': 'default', 'rope_theta': model.rope_theta},
            'sliding_attention': {'rope_type': 'default', 'rope_theta': local_theta}}}
    if sliding is not None and sliding.window is not None:
        written['local_attention'] = 2 * (sliding.window - 1)
    every = next((n for n in range(1, len(types) + 1)
                  if all((kind == 'full_attention') == (index % n == 0) for index, kind in enumerate(types))),
                 None)
    if every is not None:
        written['global_attn_every_n_layers'] = every
    return written


def _head(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The masked LM's prediction head (dense, activation, norm) and its biased projection; none where the
    config's architectures name the bare `ModernBertModel`."""
    used.update(('architectures', 'classifier_activation', 'decoder_bias'))
    if _records.strings(hf_config.get('architectures', [_MASKED_LM]), 'architectures') == (_ENCODER,):
        return record
    head_activation = _records.text(hf_config.get('classifier_activation', 'gelu'), 'classifier_activation')
    record.update(head_transform=ungated_mlp(head_activation, ('gelu', 'gelu_pytorch_tanh'),
                                             'classifier_activation', "ModernBERT's prediction head"),
                  head_bias=bool(hf_config.get('decoder_bias', True)))
    return record


def _head_written(model: CausalTransformer) -> Mapping[str, object]:
    if model.head_transform is None:
        return {'architectures': [_ENCODER], 'decoder_bias': model.head_bias}
    return {'architectures': [_MASKED_LM], 'decoder_bias': model.head_bias,
            'classifier_activation': hf_ungated(model.head_transform, "ModernBERT's prediction head",
                                                tanh='gelu_pytorch_tanh')}


MODERNBERT_CONFIG = ConfigLayout(
    integer('vocab_size', 'vocab_size'), integer('hidden_size', 'emb_features'),
    integer('num_hidden_layers', 'num_layers'), integer('num_attention_heads', 'num_heads'),
    ConfigPart(_heads, lambda model: {}), integer('intermediate_size', 'mlp_features'),
    ConfigField('hidden_activation', 'mlp', 'gelu', parse=_gated,
                value=lambda model: hf_activation(model.mlp)),
    ConfigPart(_context, lambda model: {'max_position_embeddings': model.max_seq_len}),
    ConfigPart(_layers, _layers_written),
    ConfigField('norm_eps', 'norm_eps', 1e-5, parse=_records.number),
    flag('norm_bias', 'norm_bias', default=False), flag('attention_bias', 'attention_bias', default=False),
    flag('mlp_bias', 'mlp_bias', default=False), flag('tie_word_embeddings', 'tie_embeddings', default=True),
    ConfigField('embedding_dropout', 'embedding_dropout_rate', 0.0, parse=_records.number),
    inert('attention_dropout', "ModernBERT drops inside its attention and MLP where the blocks here do not",
          neutral=0.0),
    inert('mlp_dropout', "ModernBERT drops inside its attention and MLP where the blocks here do not",
          neutral=0.0),
    inert('classifier_bias', "the prediction head's dense layer is bias-free", neutral=False),
    ConfigPart(_head, _head_written),
    constant('norm_type', value='layer'), constant('embedding_norm', value=True),
    constant('first_attention_norm', value=False), constant('qk_norm', value=False),
    constant('causal', value=False),
    # ModernBERT's original key, which transformers 5 no longer writes and older readers still take:
    # llama.cpp's converter reads its norm epsilon from it. The shared header's use_cache, which
    # ModernBertConfig declares none of, is cleared.
    ConfigPart(lambda hf_config, used, record: record,
               lambda model: {'layer_norm_eps': model.norm_eps, 'use_cache': None}),
    unread(
        # Read by the sequence-classification heads, which nothing here builds.
        'classifier_pooling', 'classifier_dropout',
        # Training-time and kernel switches of the reference, and token ids.
        'sparse_prediction', 'sparse_pred_ignore_index', 'deterministic_flash_attn',
        'reference_compile', 'repad_logits_with_grad', 'gradient_checkpointing',
        'initializer_cutoff_factor', 'cls_token_id', 'sep_token_id', 'mask_token_id',
        # Pre-5.0 releases carry these, which ModernBertConfig no longer reads:
        # the model norms at norm_eps and always rotates.
        'layer_norm_eps', 'position_embedding_type',
        # Ettin's encoders state which way they attend, which ModernBertAttention
        # never reads: it attends both ways whatever they say (is_causal = False).
        'causal_mask', 'is_causal'))
"""ModernBertConfig, either spelling of its rope and pattern, both ways."""


_MODERNBERT_NAMES: Renames = (
    ('model.embeddings.tok_embeddings', 'model.embed_tokens'),
    ('model.embeddings.norm', 'model.embedding_layernorm'), ('model.final_norm', 'model.norm'),
    ('attn_norm', 'input_layernorm'), ('mlp_norm', 'post_attention_layernorm'),
    ('attn.Wo', 'self_attn.o_proj'), ('mlp.Wo', 'mlp.down_proj'), ('decoder', 'lm_head'))

# Wqkv stacks q, k and v in thirds of its output rows, and Wi the activated
# input half over the gate half (ModernBertAttention, ModernBertMLP).
_MODERNBERT_PACKED = (
    Packed('.attn.Wqkv.weight',
           ('.self_attn.q_proj.weight', '.self_attn.k_proj.weight', '.self_attn.v_proj.weight'), 0),
    Packed('.mlp.Wi.weight', ('.mlp.gate_proj.weight', '.mlp.up_proj.weight'), 0),
    Packed('.mlp.Wi.bias', ('.mlp.gate_proj.bias', '.mlp.up_proj.bias'), 0))

# The masked LM's head, by its names after the renames (`decoder` is `lm_head`).
_HEAD: Leaves = {
    'head.dense.weight': ('head_dense', 'kernel'), 'head.norm.weight': ('head_norm', 'scale'),
    'head.norm.bias': ('head_norm', 'bias'), 'lm_head.bias': ('head_bias',)}


def _nested(name: str) -> str:
    """A bare `ModernBertModel` checkpoint's name as the masked LM nests it."""
    return 'model.' + name if name.startswith(('embeddings.', 'layers.', 'final_norm.')) else name


def _modernbert_prepare(tensors: Mapping[str, Stored],
                        _config: Mapping[str, object] | None = None) -> Mapping[str, Stored]:
    return {_nested(name): tensor for name, tensor in tensors.items()}


_MASKED_LM_ROUTES = spelled(_MODERNBERT_NAMES, leaves=_HEAD)

_MODERNBERT_ROUTES = Table(
    _MASKED_LM_ROUTES,
    # A bare `ModernBertModel` checkpoint's names, which a load reads as the masked LM nests them.
    Spelled((('embeddings', 'model.embeddings'), ('layers', 'model.layers'),
             ('final_norm', 'model.final_norm')), _MASKED_LM_ROUTES))
"""ModernBERT's names: the masked LM's (`_MODERNBERT_NAMES`, its head's `_HEAD`), which an export writes, and
a bare encoder's, which a load reads too."""


MODERNBERT = DecoderFamily(
    ('modernbert',), MODERNBERT_CONFIG,
    lambda fields: not fields.causal and fields.embedding_norm and not fields.first_attention_norm,
    'modernbert', 'ModernBertForMaskedLM',
    routes=_MODERNBERT_ROUTES,
    prepare=_modernbert_prepare, spelling=_nested, packed=_MODERNBERT_PACKED, preserve_source_layout=False,
    tied_head_names=('decoder.weight', 'model.embeddings.tok_embeddings.weight'),
    defaults={'pad_token_id': 50283},
    padding_idx=True,
)
