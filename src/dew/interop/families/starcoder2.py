"""Starcoder2's biased LayerNorm block: grouped-query attention over full
rotate-half rotary, a sliding window on every layer when it states one, and
an ungated feed-forward, every map biased under `use_bias`."""

from collections.abc import Mapping
from functools import partial

from dew import records
from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    Renames,
    base_config,
    refuse,
    renamed_name,
    renamed_path,
)
from dew.nn.backbones.causal_transformer import CausalTransformer

_ACTIVATIONS = {'gelu_pytorch_tanh': 'gelu', 'gelu_new': 'gelu', 'gelu': 'gelu_exact', 'relu': 'relu'}
"""`hidden_act` as the ungated feed-forward names it."""


def _starcoder2_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    activation = records.text(hf.get('hidden_act', 'gelu_pytorch_tanh'), 'hidden_act')
    if activation not in _ACTIVATIONS:
        refuse(f'hidden_act {activation!r}', f'the ungated Starcoder2 MLP supports {sorted(_ACTIVATIONS)}')
    layers = records.integer(hf['num_hidden_layers'], 'num_hidden_layers')
    # Starcoder2Model masks every layer by the window when one is set.
    window = hf.get('sliding_window')
    # base_config reads the gated activations; the ungated one is set below.
    config = base_config({**hf, 'hidden_act': 'gelu_pytorch_tanh'}, used,
                         layer_types=('full_attention' if window is None else 'sliding_attention',) * layers,
                         tie_embeddings=True, reads=frozenset({'sliding_window'}))
    biased = bool(hf.get('use_bias', True))
    # The released configs' mlp_type and norm_type predate Starcoder2Config,
    # whose model reads neither (transformers 5.16.1).
    used.update(('use_bias', 'norm_epsilon', 'residual_dropout', 'embedding_dropout', 'attention_dropout',
                 'mlp_type', 'norm_type'))
    config.update({
        'mlp': _ACTIVATIONS[activation], 'mlp_bias': biased, 'attention_bias': biased,
        'norm_type': 'layer', 'norm_bias': True,
        'norm_eps': records.number(hf.get('norm_epsilon', 1e-5), 'norm_epsilon'),
        'dropout_rate': records.number(hf.get('residual_dropout', 0.), 'residual_dropout'),
        'embedding_dropout_rate': records.number(hf.get('embedding_dropout', 0.), 'embedding_dropout'),
        'attention_dropout_rate': records.number(hf.get('attention_dropout', 0.), 'attention_dropout'),
    })
    return config


_STARCODER2_NAMES: Renames = (('mlp.c_fc', 'mlp.up_proj'), ('mlp.c_proj', 'mlp.down_proj'))


def _starcoder2_export(model: CausalTransformer) -> Mapping[str, object]:
    windowed = 'sliding_attention' in model.per_layer_types
    return {
        'hidden_act': {'gelu': 'gelu_pytorch_tanh', 'gelu_exact': 'gelu', 'relu': 'relu'}[str(model.mlp)],
        'use_bias': model.attention_bias, 'norm_epsilon': model.norm_eps,
        'sliding_window': model.kind_of('sliding_attention').window if windowed else None,
        'residual_dropout': model.dropout_rate, 'embedding_dropout': model.embedding_dropout_rate,
        'attention_dropout': model.attention_dropout_rate,
        'rms_norm_eps': None, 'layer_types': None, 'attention_bias': None,
    }


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
    ('starcoder2',), _starcoder2_config, _matches, 'starcoder2', 'Starcoder2ForCausalLM', _starcoder2_export,
    weight_path=partial(renamed_path, _STARCODER2_NAMES),
    export_path=partial(renamed_name, _STARCODER2_NAMES),
    preserve_source_layout=False,
)
