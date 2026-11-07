"""GPT-NeoX's parallel biased LayerNorm block and head-interleaved qkv storage."""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from dew import _records
from dew.interop.decoder_config import DecoderSpelling, decoder_layout, flag, partial_rope_part, ungated
from dew.interop.decoder_parts import (
    MEGATRON_MLP,
    DecoderFamily,
    DecoderFields,
    Renames,
    fixed_attention_buffer,
    grouped_qkv,
    spelled,
)
from dew.interop.layouts import (
    ConfigField,
    ConfigLayout,
    ConfigPart,
    Restated,
    Route,
    Slot,
    Table,
    constant,
    one_of,
    unread,
)
from dew.interop.unread import Stored, equal
from dew.nn.rope import inverse_frequencies


def _embedding_dropout(hf_config: Mapping[str, object], used: set[str], record: DecoderFields
                       ) -> DecoderFields:
    """GPT-NeoX drops its embeddings at the residual's rate (`hidden_dropout`)."""
    record['embedding_dropout_rate'] = _records.number(hf_config.get('hidden_dropout', 0.), 'hidden_dropout')
    return record


GPT_NEOX_CONFIG = ConfigLayout(
    decoder_layout(DecoderSpelling(reads=frozenset(), scale_after_cast=False, activation=None,
                                   keys={'rms_norm_eps': 'layer_norm_eps'}, defaults={'layer_norm_eps': 1e-5},
                                   undeclared=frozenset({'num_key_value_heads', 'head_dim'})),
                   # GPTNeoXRotaryEmbedding builds any rope_type over the rotated quarter, YaRN among them;
                   # its releases spell the base and the fraction flat (rotary_emb_base, rotary_pct).
                   rope=Restated(ConfigLayout(partial_rope_part(default_factor=.25)),
                                 own={'partial_rotary_factor': 'rotary_pct',
                                      'rope_theta': 'rotary_emb_base'})),
    ungated('hidden_act', ('gelu', 'gelu_new', 'gelu_pytorch_tanh', 'relu'), 'GPT-NeoX', default='gelu'),
    flag('attention_bias', 'attention_bias', default=True), flag('use_parallel_residual', 'parallel_residual',
                                                                 default=True),
    ConfigField('hidden_dropout', 'dropout_rate', 0.0, parse=_records.number),
    ConfigPart(_embedding_dropout, lambda model: {}),
    ConfigField('attention_dropout', 'attention_dropout_rate', 0.0, parse=_records.number),
    constant('mlp_bias', value=True), constant('norm_type', value='layer'), constant('norm_bias', value=True),
    unread('classifier_dropout', 'is_decoder'))
"""GPT-NeoX's config, both ways: a parallel biased-LayerNorm block, a quarter of each head rotated."""


def _gpt_neox_buffers(tensors: Mapping[str, Stored],
                      config: Mapping[str, object] | None = None) -> Mapping[str, Stored]:
    """The checkpoint without its fixed attention buffers, each checked: the causal mask, the mask
    sentinel and the rotary frequencies, which the config computes."""
    if config is None:
        raise ValueError('GPT-NeoX checks its rotary buffer against translated head geometry')
    dimension = _records.integer(config['head_dim'], 'head_dim')
    prepared = {}
    for name, tensor in tensors.items():
        # FP16 releases cast the historical -1e9 sentinel to -inf.
        if fixed_attention_buffer(name, tensor, 'attention', (-1e9, -np.inf)):
            continue
        if name.endswith('.attention.rotary_emb.inv_freq'):
            rotated = int(
                dimension * _records.number(config.get("partial_rotary_factor", 1.0), "partial_rotary_factor")
            )
            expected = inverse_frequencies(_records.number(config.get('rope_theta', 10000.), 'rope_theta'),
                                           rotated, dtype=np.float32).astype(tensor.dtype)
            if not equal(tensor, expected):
                raise ValueError(
                    f"{name} disagrees with the configured rotary frequencies in its stored dtype"
                )
            continue
        prepared[name] = tensor
    return prepared


_GPT_NEOX_NAMES: Renames = (
    ('embed_out', 'lm_head'), ('gpt_neox.embed_in', 'model.embed_tokens'),
    ('gpt_neox.final_layer_norm', 'model.norm'), ('gpt_neox.layers', 'model.layers'),
    ('attention.dense', 'self_attn.o_proj'), *MEGATRON_MLP)


_GPT_NEOX_ROUTES = Table(
    # The fixed attention buffers, which a load checks and drops (`_gpt_neox_buffers`): no leaf.
    Route('{module}.attention.{buffer}', None, MappingProxyType({
        'module': Slot(r'.+'), 'buffer': one_of('bias', 'masked_bias', 'rotary_emb.inv_freq')})),
    spelled(_GPT_NEOX_NAMES))
"""GPT-NeoX's names (`_GPT_NEOX_NAMES`) onto the shared decoder's."""


GPT_NEOX = DecoderFamily(
    ('gpt_neox',),
    GPT_NEOX_CONFIG,
    lambda fields: bool(
        fields.norm_type == 'layer'
        and fields.norm_bias
        and fields.mlp_bias
        and fields.position_embedding == 'rotary'
    ),
    'gpt_neox',
    'GPTNeoXForCausalLM',
    routes=_GPT_NEOX_ROUTES,
    prepare=_gpt_neox_buffers,
    packed=grouped_qkv('.attention.query_key_value'),
    preserve_source_layout=False,
    tied_head_names=('embed_out.weight', 'gpt_neox.embed_in.weight'),
    defaults={'max_position_embeddings': 2048},
)
