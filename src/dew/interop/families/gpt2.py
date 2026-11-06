"""GPT-2's learned positions, biased LayerNorm and Conv1D checkpoint layout, and GPT-1's post-LN blocks over
them."""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_config import flag, four_wide, inert, integer, ungated
from dew.interop.decoder_parts import (
    GPT_MLP,
    GPT_TRUNK,
    DecoderFamily,
    DecoderFields,
    HeadRows,
    Packed,
    Renames,
    fixed_attention_buffer,
    refuse,
    spelled,
)
from dew.interop.layouts import (
    ConfigField,
    ConfigLayout,
    ConfigPart,
    Restated,
    Route,
    Slot,
    Spelled,
    Table,
    constant,
    one_of,
    unread,
)
from dew.interop.unread import Stored
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.registry import Configured


def _heads(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """GPT-2's heads, every one its own key/value head, as wide as the residual over their count."""
    used.add('n_head')
    heads = _records.integer(hf_config.get('n_head', 12), 'n_head')
    hidden = _records.integer(record['emb_features'], 'n_embd')
    if hidden % heads:
        refuse('n_embd/n_head', 'the hidden width must divide into whole attention heads')
    record.update(num_heads=heads, num_kv_heads=heads, head_dim=hidden // heads)
    return record


def _positions(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    used.add('n_positions')
    positions = _records.integer(hf_config.get('n_positions', 1024), 'n_positions')
    record.update(position_embedding='learned', position_embedding_size=positions, max_seq_len=positions)
    return record


def _scaled(value: Configured, key: str) -> float | None:
    """GPT-2's scale_attn_weights: the head width's root, or none at all (1.0)."""
    return None if value is None or _records.boolean(value, key) else 1.0


def gpt2_layout(activation: ConfigPart[CausalTransformer]) -> ConfigLayout[CausalTransformer]:
    """GPT-2's decoder, both ways, its MLP's activation `activation` (GPT-1 spells it its own way)."""
    return ConfigLayout(
        integer('vocab_size', 'vocab_size', 50257), integer('n_embd', 'emb_features', 768),
        integer('n_layer', 'num_layers', 12), ConfigPart(_heads, lambda model: {'n_head': model.num_heads}),
        four_wide('n_inner'), activation,
        ConfigPart(_positions, lambda model: {'n_positions': model.position_embedding_size
                                              or model.max_seq_len}),
        ConfigField('layer_norm_epsilon', 'norm_eps', 1e-5, parse=_records.number),
        ConfigField('scale_attn_weights', 'attention_scale', default=True, parse=_scaled,
                    spell=lambda scale: scale is None),
        flag('tie_word_embeddings', 'tie_embeddings', default=True),
        ConfigField('resid_pdrop', 'dropout_rate', 0.1, parse=_records.number),
        ConfigField('embd_pdrop', 'embedding_dropout_rate', 0.1, parse=_records.number),
        ConfigField('attn_pdrop', 'attention_dropout_rate', 0.1, parse=_records.number),
        constant('mlp_bias', value=True), constant('norm_type', value='layer'),
        constant('norm_bias', value=True), constant('qk_norm', value=False),
        constant('attention_bias', value=True),
        inert('add_cross_attention', 'GPT-2 cross attention has no decoder counterpart', neutral=False),
        inert('scale_attn_by_inverse_layer_idx', 'the attention scale does not vary by layer', neutral=False),
        inert('reorder_and_upcast_attn', 'the reordered GPT-2 attention arithmetic is not represented',
              neutral=False),
        unread('n_ctx', 'summary_type', 'summary_use_proj', 'summary_activation', 'summary_proj_to_labels',
               'summary_first_dropout', 'use_cache', 'task_specific_params'))


GPT2_CONFIG = gpt2_layout(ungated(
    'activation_function', ('gelu_new', 'gelu_fast', 'gelu_pytorch_tanh', 'gelu', 'relu'), 'GPT-2'))
"""GPT-2's config, both ways."""


def _neo_pattern(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """GPT-Neo's attention kinds, both ways: `attention_types`' repeated global and local lists, one kind a
    layer, the local ones attending `window_size`."""
    used.update(('attention_types', 'attention_layers', 'window_size'))
    layers = _records.integer(record['num_layers'], 'num_layers')
    stated = hf_config.get('attention_types', [[['global', 'local'], 12]])
    if not isinstance(stated, (list, tuple)):
        refuse('attention_types', 'GPT-Neo repeats lists of global/local attention kinds')
    pattern: list[str] = []
    for entry in stated:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            refuse('attention_types', 'each item holds a kind pattern and its repetition count')
        repeats = _records.integer(entry[1], 'attention_types repetitions')
        if repeats < 1:
            refuse('attention_types', 'repetition counts must be positive')
        pattern.extend(_records.strings(entry[0], 'attention_types') * repeats)
    if len(pattern) != layers or set(pattern) - {'global', 'local'}:
        refuse('attention_types', 'one global or local attention kind is required per layer')
    if (hf_config.get('attention_layers') is not None
            and _records.strings(hf_config['attention_layers'], 'attention_layers') != tuple(pattern)):
        refuse('attention_layers', 'the expanded pattern must agree with attention_types')
    record['layer_types'] = tuple('full_attention' if kind == 'global' else 'sliding_attention'
                                  for kind in pattern)
    record['kinds'] = ({'sliding_attention': native_fields(LayerKind)(
        window=_records.integer(hf_config.get('window_size', 256), 'window_size'))}
        if 'local' in pattern else {})
    return record


def _neo_pattern_written(model: CausalTransformer) -> Mapping[str, object]:
    return {'attention_types': [[[('local' if kind == 'sliding_attention' else 'global')], 1]
                                for kind in model.per_layer_types],
            'window_size': (model.kind_of('sliding_attention').window
                            if 'sliding_attention' in model.per_layer_types else 256)}


GPT_NEO_CONFIG = ConfigLayout(
    Restated(GPT2_CONFIG, own={'n_embd': 'hidden_size', 'n_layer': 'num_layers', 'n_head': 'num_heads',
                                'n_positions': 'max_position_embeddings', 'n_inner': 'intermediate_size',
                                'resid_pdrop': 'resid_dropout', 'embd_pdrop': 'embed_dropout',
                                'attn_pdrop': 'attention_dropout'},
             defaults={'num_layers': 24, 'hidden_size': 2048, 'num_heads': 16,
                       'max_position_embeddings': 2048, 'resid_dropout': 0.0, 'embed_dropout': 0.0,
                       'attention_dropout': 0.0}),
    ConfigPart(_neo_pattern, _neo_pattern_written),
    constant('attention_bias', value=False), constant('o_proj_bias', value=True),
    constant('attention_scale', value=1.0), unread('classifier_dropout', 'gradient_checkpointing'))
"""GPT-Neo's config, both ways: GPT-2's block under its own names, unscaled attention without q/k/v biases,
and global and local layers."""


_GPT_NEO_NAMES: Renames = (
    ('attn.attention.out_proj', 'self_attn.o_proj'), ('attn.attention', 'self_attn'), *GPT_TRUNK, *GPT_MLP)


def _gptj_rotary_dim(hf_config: Mapping[str, object], used: set[str],
                     record: DecoderFields) -> DecoderFields:
    """GPT-J's rotary, both ways: an even prefix of `rotary_dim` of each head, turned on every layer and
    scaled by the head width."""
    used.update(('rotary_dim', 'rotary'))
    head_dim = _records.integer(record['head_dim'], 'head_dim')
    rotated = _records.integer(hf_config.get('rotary_dim', 64), 'rotary_dim')
    if rotated < 2 or rotated > head_dim or rotated % 2:
        refuse('rotary_dim', 'GPT-J rotates an even positive prefix of each attention head')
    if hf_config.get('rotary', True) is not True or hf_config.get('scale_attn_weights', True) is not True:
        refuse('rotary/scale_attn_weights', 'GPT-J rotates its heads and scales by their dimension')
    record.update(position_embedding='rotary', position_embedding_size=None,
                  partial_rotary_factor=rotated / head_dim, partial_rotary_type='default')
    return record


GPTJ_CONFIG = ConfigLayout(
    GPT2_CONFIG,
    ConfigPart(_gptj_rotary_dim, lambda model: {
        'rotary_dim': int(model.features_per_head * (model.partial_rotary_factor or 1.)),
        'pad_token_id': None}),
    constant('attention_bias', value=False), constant('parallel_residual', value=True),
    constant('shared_parallel_norm', value=True), constant('head_bias', value=True),
    flag('tie_word_embeddings', 'tie_embeddings', default=False),
    unread('gradient_checkpointing', 'tokenizer_class'))
"""GPT-J's config, both ways: GPT-2's block in parallel under one norm, its heads rotated."""


# GPT-J's block has no second norm and its model no learned positions, so of
# GPT-2's trunk names it carries the embeddings, the final norm, the layers and ln_1.
_GPTJ_NAMES: Renames = (
    ('attn.out_proj', 'self_attn.o_proj'), ('attn', 'self_attn'),
    ('mlp.fc_in', 'mlp.up_proj'), ('mlp.fc_out', 'mlp.down_proj'), *GPT_TRUNK)


def _gptj_order(head_dim: int, rotated: int) -> np.ndarray:
    """Interleaved rotary pairs as the shared rotate-half arithmetic reads them.

    Applying the same permutation to q and k leaves their dot product
    unchanged. Values keep their original order, so the output projection
    and the residual stream require no permutation or runtime option.
    """
    return np.concatenate((np.arange(0, rotated, 2), np.arange(1, rotated, 2),
                           np.arange(rotated, head_dim)))


def _gptj_rotary(config: Mapping[str, object]) -> np.ndarray:
    """A GPT-J head's rows as the shared rotate-half arithmetic reads them (`_gptj_order`), by the record."""
    head_dim = _records.integer(config['head_dim'], 'head_dim')
    # A model rotating every channel states no fraction (`_gptj_export` writes it as 1.0).
    fraction = config.get('partial_rotary_factor')
    rotated = int(head_dim * (1.0 if fraction is None
                               else _records.number(fraction, 'partial_rotary_factor')))
    return _gptj_order(head_dim, rotated)


_GPTJ_ROWS = HeadRows(('.attn.q_proj.weight', '.attn.k_proj.weight'), _gptj_rotary)
"""GPT-J's queries' and keys' rows, each head's rotary pairs interleaved where the shared arithmetic
rotates halves."""


def _gptj_prepare(tensors: Mapping[str, Stored],
                  config: Mapping[str, object] | None = None) -> Mapping[str, Stored]:
    """Check and drop the fixed attention buffers."""
    return {name: tensor for name, tensor in tensors.items()
            if not fixed_attention_buffer(name, tensor, 'attn', (-1e9,))}


def _buffers(*names: str) -> Route:
    """A release's fixed attention buffers, by their names' last parts (`names`), which a load checks and
    drops (`fixed_attention_buffer`): no leaf."""
    return Route('{module}.{buffer}', None,
                 MappingProxyType({'module': Slot(r'.+'), 'buffer': one_of(*names)}))


_GPTJ_ROUTES = Table(_buffers('attn.bias', 'attn.masked_bias'), spelled(_GPTJ_NAMES))
"""GPT-J's names: GPT-2's trunk, its own block's (`_GPTJ_NAMES`); its buffers fill no leaf."""


def _gpt2_prepare(tensors: Mapping[str, Stored],
                   _config: Mapping[str, object] | None = None) -> Mapping[str, Stored]:
    """Check and drop the fixed attention buffers, and nest the bare model's names."""
    prepared = {}
    for name, tensor in tensors.items():
        # The original gpt2 safetensors stores the bare GPT2Model and its
        # old persistent causal buffers.
        if not fixed_attention_buffer(name, tensor, 'attn', (-1e4,)):
            prepared[_nested(name)] = tensor
    return prepared


def _nested(name: str) -> str:
    return 'transformer.' + name if name.startswith(('h.', 'wte.', 'wpe.', 'ln_f.')) else name


GPT2_NAMES: Renames = (('attn.c_proj', 'self_attn.o_proj'), *GPT_TRUNK, *GPT_MLP)

# Conv1D stores `[in, out]` where the path map reads a torch Linear's
# `[out, in]`, and c_attn holds q, k and v side by side on `out`.
_QKV = ('q_proj', 'k_proj', 'v_proj')
_GPT2_PACKED = (
    Packed('.attn.c_attn.weight', tuple(f'.self_attn.{part}.weight' for part in _QKV), 0, (1, 0)),
    Packed('.attn.c_attn.bias', tuple(f'.self_attn.{part}.bias' for part in _QKV), 0),
    *(Packed(f'.{conv}.weight', (f'.{conv}.weight',), 0, (1, 0))
      for conv in ('attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')))


_GPT2_ROUTES = Table(
    _buffers('attn.bias', 'attn.masked_bias'), spelled(GPT2_NAMES),
    # The bare GPT2Model's names, which a load nests under the causal LM's `transformer.`.
    Spelled((('h', 'transformer.h'), ('wte', 'transformer.wte'), ('wpe', 'transformer.wpe'),
             ('ln_f', 'transformer.ln_f')), spelled(GPT2_NAMES)))
"""GPT-2's names: the causal LM's, which an export writes, and the bare model's, which a load reads too;
the old persistent causal buffers fill no leaf."""


GPT_NEO = DecoderFamily(
    ('gpt_neo',), GPT_NEO_CONFIG,
    lambda fields: fields.position_embedding == 'learned' and fields.attention_scale == 1.0
                   and fields.attention_bias is False and fields.o_proj_bias is True,
    'gpt_neo', 'GPTNeoForCausalLM',
    routes=spelled(_GPT_NEO_NAMES), preserve_source_layout=False,
    tied_head_names=('lm_head.weight', 'transformer.wte.weight'),
)

GPTJ = DecoderFamily(
    ('gptj',), GPTJ_CONFIG,
    lambda fields: fields.shared_parallel_norm and fields.head_bias and not fields.attention_bias,
    'gptj', 'GPTJForCausalLM',
    routes=_GPTJ_ROUTES, head_rows=_GPTJ_ROWS, prepare=_gptj_prepare, preserve_source_layout=False,
    tied_head_names=('lm_head.weight', 'transformer.wte.weight'),
)

OPENAI_GPT_CONFIG = ConfigLayout(
    # GPT-1's blocks always scale by the head width and read none of GPT-2's other attention switches
    # (modeling_openai.py builds each Block with scale=True), whatever a config carries.
    Restated(gpt2_layout(ungated('afn', ('gelu_new', 'relu'), 'GPT-1',
                                            spelled={'gelu': 'gelu_new', 'relu': 'relu'}, default='gelu')),
             fixed={'n_inner': None, 'scale_attn_weights': True, 'scale_attn_by_inverse_layer_idx': False,
                    'reorder_and_upcast_attn': False, 'add_cross_attention': False},
             defaults={'vocab_size': 40478, 'n_positions': 512}),
    constant('pre_norms', value=False), constant('post_residual_norms', value=True),
    constant('final_norm', value=False),
    # The original release's `n_special`, which transformers' OpenAI GPT reads nowhere: its vocabulary
    # counts the special tokens.
    unread('predict_special_tokens', 'n_special'))
"""GPT-1's config, both ways (modeling_openai.py): GPT-2's attention, MLP and Conv1D layers, each block
norming its stream after each sublayer joins it (`n = ln_1(x + a)`, `h = ln_2(n + m)`) and the trunk ending
in no norm. Its `afn` 'gelu' is OpenAI's tanh GELU (`ACT_FNS`, gelu_new)."""


def _openai_nested(name: str) -> str:
    """A bare OpenAIGPTModel's names nested under the causal LM's `transformer`, as the original release
    stores them."""
    return 'transformer.' + name if name.startswith(('h.', 'tokens_embed.', 'positions_embed.')) else name


def _openai_gpt_prepare(tensors: Mapping[str, Stored],
                        _config: Mapping[str, object] | None = None) -> Mapping[str, Stored]:
    """Check and drop the fixed causal masks, and nest the bare model's names."""
    return {_openai_nested(name): tensor for name, tensor in tensors.items()
            if not fixed_attention_buffer(name, tensor, 'attn', ())}


OPENAI_GPT_NAMES: Renames = (
    ('transformer.tokens_embed', 'model.embed_tokens'),
    ('transformer.positions_embed', 'model.embed_positions'),
    ('transformer.h', 'model.layers'), ('ln_1', 'input_layernorm'), ('ln_2', 'post_attention_layernorm'),
    ('attn.c_proj', 'self_attn.o_proj'), *GPT_MLP)
"""GPT-1's trunk and blocks: GPT-2's but for its embeddings' names; `ln_1` norms the stream after the
attention joins it and `ln_2` after the MLP (`post_residual_norms`), as OPT-350m's norms do."""


_OPENAI_GPT_ROUTES = Table(
    _buffers('attn.bias'), spelled(OPENAI_GPT_NAMES),
    # The bare OpenAIGPTModel's names, which the original release stores.
    Spelled((('h', 'transformer.h'), ('tokens_embed', 'transformer.tokens_embed'),
             ('positions_embed', 'transformer.positions_embed')), spelled(OPENAI_GPT_NAMES)))
"""GPT-1's names: the causal LM's, which an export writes, and the bare model's, which a load reads too;
the fixed causal masks fill no leaf."""


OPENAI_GPT = DecoderFamily(
    ('openai-gpt',),
    OPENAI_GPT_CONFIG,
    lambda fields: (fields.position_embedding == 'learned' and fields.post_residual_norms
                    and not fields.final_norm and fields.position_embedding_offset == 0),
    'openai-gpt',
    'OpenAIGPTLMHeadModel',
    routes=_OPENAI_GPT_ROUTES,
    prepare=_openai_gpt_prepare,
    spelling=_openai_nested,
    packed=_GPT2_PACKED,
    preserve_source_layout=False,
    tied_head_names=('lm_head.weight', 'transformer.tokens_embed.weight'),
)

GPT2 = DecoderFamily(
    ('gpt2',),
    GPT2_CONFIG,
    lambda fields: fields.position_embedding == 'learned',
    'gpt2',
    'GPT2LMHeadModel',
    routes=_GPT2_ROUTES,
    prepare=_gpt2_prepare,
    spelling=_nested,
    packed=_GPT2_PACKED,
    preserve_source_layout=False,
    tied_head_names=('lm_head.weight', 'transformer.wte.weight'),
)
