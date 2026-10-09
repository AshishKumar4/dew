"""Cohere's one-norm parallel block: a bias-free LayerNorm feeds both the
attention and a gated feed-forward, the rotary turns adjacent channel pairs,
and the head's logits are multiplied by `logit_scale`.

Command R (cohere) can norm each query and key head under that head's own
LayerNorm scale. Command R7B and Command A (cohere2) rotate their sliding
layers alone and leave the full layers unrotated.
"""

from collections.abc import Mapping

from dew import records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    Ropes,
    base_config,
    decoder_kinds,
    plain_partial_rope,
    refuse,
    specified_layer_types,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mixers import AttentionMixer

_ADJACENT = {'class': 'attention', 'fields': {'rotary_pairs': 'adjacent'}}
"""The rotary CohereRotaryEmbedding computes: each frequency repeated over
an adjacent channel pair (`repeat_interleave`), turned by Cohere's
`rotate_half` of the even and odd channels."""


def _cohere_fields(hf: Mapping[str, object], used: set[str], config: DecoderFields, reference: str) -> None:
    """The fields Cohere and Cohere2 share, onto `config`."""
    if hf.get('rope_scaling') is not None:
        refuse('rope_scaling', f'{reference} is the plain rotary')
    _, factor = plain_partial_rope(hf, default_factor=1.0, reference=reference)
    if factor != 1.0:
        refuse(f'partial_rotary_factor {factor}', f'{reference} turns every channel of the head')
    scale = records.number(hf.get('logit_scale', 0.0625), 'logit_scale')
    if scale <= 0:
        refuse(f'logit_scale {scale}', 'the head multiplies its logits by a positive scale')
    used.update(('rope_theta', 'rope_parameters', 'rope_scaling', 'partial_rotary_factor', 'logit_scale',
                 'layer_norm_eps', 'attention_dropout'))
    config.update(
        mixer=_ADJACENT, norm_type='layer', norm_bias=False,
        norm_eps=records.number(hf.get('layer_norm_eps', 1e-5), 'layer_norm_eps'),
        parallel_residual=True, shared_parallel_norm=True, logits_scaling=1 / scale,
        attention_dropout_rate=records.number(hf.get('attention_dropout', 0.), 'attention_dropout'))


def _cohere_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    theta, _ = plain_partial_rope(hf, default_factor=1.0, reference='CohereRotaryEmbedding')
    qk_norm = bool(hf.get('use_qk_norm', False))
    used.add('use_qk_norm')
    config = base_config(hf, used, rope=Ropes(theta), qk_norm=qk_norm, tie_embeddings=True,
                         reads=frozenset({'attention_bias'}))
    _cohere_fields(hf, used, config, 'CohereRotaryEmbedding')
    # CohereLayerNorm over [heads, head_dim]: each head under its own scale.
    config.update(qk_norm_scope='head_layernorm')
    return config


def _cohere2_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    hidden = records.integer(hf['hidden_size'], 'hidden_size')
    heads = records.integer(hf['num_attention_heads'], 'num_attention_heads')
    # Cohere2Config sets head_dim to hidden_size // num_attention_heads over
    # whatever the config states (configuration_cohere2.py:93).
    stated = hf.get('head_dim')
    if stated is not None and records.integer(stated, 'head_dim') != hidden // heads:
        refuse(f'head_dim {stated}',
               f'Cohere2Config sets it to hidden_size // num_attention_heads, {hidden // heads}')
    # An unstated pattern is the legacy sliding_window_pattern's: every
    # fourth layer full, the rest sliding (configuration_cohere2.py:96-102).
    period = records.integer(hf.get('sliding_window_pattern', 4), 'sliding_window_pattern')
    layers = records.integer(hf['num_hidden_layers'], 'num_hidden_layers')
    used.add('sliding_window_pattern')
    pattern = specified_layer_types(hf, used, tuple(
        'sliding_attention' if (index + 1) % period else 'full_attention' for index in range(layers)))
    theta, _ = plain_partial_rope(hf, default_factor=1.0, reference='Cohere2RotaryEmbedding')
    config = base_config({**hf, 'head_dim': hidden // heads}, used, layer_types=pattern, rope=Ropes(theta),
                         tie_embeddings=True,
                         reads=frozenset({'layer_types', 'sliding_window', 'attention_bias'}))
    _cohere_fields(hf, used, config, 'Cohere2RotaryEmbedding')
    # Cohere2Attention rotates a sliding layer alone (modeling_cohere2.py:229-230).
    window = (records.integer(hf['sliding_window'], 'sliding_window')
              if 'sliding_attention' in pattern else None)
    kinds = decoder_kinds(pattern, window, None, None, None)
    if 'full_attention' in pattern:
        kinds['full_attention'] = native_fields(LayerKind)(mixer=None)
        kinds['full_attention'].update(mixer={'class': 'attention',
                                              'fields': {'rotary_pairs': 'adjacent', 'nope': True}})
    config.update(kinds=kinds)
    return config


def _cohere_export(model: CausalTransformer) -> Mapping[str, object]:
    return {
        'layer_norm_eps': model.norm_eps, 'logit_scale': 1 / model.logits_scaling,
        'use_qk_norm': model.qk_norm, 'attention_dropout': model.attention_dropout_rate, 'rms_norm_eps': None,
    }


def _cohere2_export(model: CausalTransformer) -> Mapping[str, object]:
    return {
        'layer_norm_eps': model.norm_eps, 'logit_scale': 1 / model.logits_scaling,
        'attention_dropout': model.attention_dropout_rate, 'rms_norm_eps': None, 'head_dim': None,
    }


def _cohere_block(fields: CausalTransformer) -> bool:
    """A bias-free LayerNorm shared by parallel attention and gated
    feed-forward branches, rotating adjacent pairs."""
    mixer = fields.mixer
    return bool(fields.norm_type == 'layer' and not fields.norm_bias and fields.shared_parallel_norm
                and isinstance(mixer, AttentionMixer) and mixer.rotary_pairs == 'adjacent'
                and fields.mlp in ('swiglu', 'geglu', 'geglu_exact') and not fields.mlp_bias)


def _cohere2_layers(fields: CausalTransformer) -> bool:
    """Sliding layers, and full layers that leave their heads unrotated."""
    types = set(fields.per_layer_types)
    full = fields.kind_of('full_attention').mixer
    return 'sliding_attention' in types and (
        'full_attention' not in types or (isinstance(full, AttentionMixer) and full.nope))


COHERE2 = DecoderFamily(
    ('cohere2',), _cohere2_config,
    lambda fields: _cohere_block(fields) and not fields.qk_norm and _cohere2_layers(fields),
    'cohere2', 'Cohere2ForCausalLM', _cohere2_export, preserve_source_layout=False,
)


COHERE = DecoderFamily(
    ('cohere',), _cohere_config,
    lambda fields: _cohere_block(fields) and set(fields.per_layer_types) == {'full_attention'},
    'cohere', 'CohereForCausalLM', _cohere_export, preserve_source_layout=False,
)
