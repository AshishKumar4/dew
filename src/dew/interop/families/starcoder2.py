"""Starcoder2's biased LayerNorm block: grouped-query attention over full
rotate-half rotary, a sliding window on every layer when it states one, and
an ungated feed-forward, every map biased under `use_bias`."""


from dew import _records
from dew.interop.decoder_config import DecoderSpelling, biases, decoder_layout, ungated
from dew.interop.decoder_parts import GPT_MLP, DecoderFamily, spelled
from dew.interop.layouts import ConfigField, ConfigLayout, constant, unread
from dew.nn.backbones.causal_transformer import CausalTransformer

STARCODER2_CONFIG = ConfigLayout(
    # Starcoder2Model masks every layer by the window when one is set.
    decoder_layout(DecoderSpelling(reads=frozenset({'sliding_window'}), windowed=True, tie_embeddings=True,
                                   activation=None, keys={'rms_norm_eps': 'norm_epsilon'},
                                   defaults={'norm_epsilon': 1e-5})),
    ungated('hidden_act', ('gelu_pytorch_tanh', 'gelu_new', 'gelu', 'relu'), 'Starcoder2',
            default='gelu_pytorch_tanh', tanh='gelu_pytorch_tanh'),
    biases('use_bias', default=True),
    ConfigField('residual_dropout', 'dropout_rate', 0.0, parse=_records.number),
    ConfigField('embedding_dropout', 'embedding_dropout_rate', 0.0, parse=_records.number),
    ConfigField('attention_dropout', 'attention_dropout_rate', 0.0, parse=_records.number),
    constant('norm_type', value='layer'), constant('norm_bias', value=True),
    # The released configs' mlp_type and norm_type predate Starcoder2Config, whose model reads neither
    # (transformers 5.16.1).
    unread('mlp_type', 'norm_type'))
"""StarCoder2's config, both ways."""


def _matches(fields: CausalTransformer) -> bool:
    """A sequential biased-LayerNorm rotary decoder with an ungated
    feed-forward, grouped or windowed: GPT-NeoX's sequential block has
    neither, and computes the rest. Its config has one `use_bias` for every
    map and one window that Starcoder2Model applies to every layer, so a
    model biasing some maps alone, or windowing some layers alone, is not
    one."""
    kinds = set(fields.per_layer_types)
    return bool(fields.norm_type == 'layer' and fields.norm_bias and fields.position_embedding == 'rotary'
                and fields.mlp in ('gelu', 'gelu_exact', 'relu') and not fields.parallel_residual
                and fields.mlp_bias == fields.attention_bias
                and fields.o_proj_bias in (None, fields.attention_bias)
                and kinds in ({'full_attention'}, {'sliding_attention'})
                and (fields.kv_heads < fields.num_heads or kinds == {'sliding_attention'}))


STARCODER2 = DecoderFamily(
    ('starcoder2',), STARCODER2_CONFIG, _matches,
    'starcoder2', 'Starcoder2ForCausalLM',
    routes=spelled(GPT_MLP),
    preserve_source_layout=False,
    padding_idx=True,
    defaults={'max_position_embeddings': 4096},
)
