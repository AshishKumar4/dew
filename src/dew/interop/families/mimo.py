"""Translate MiMo-V2-Flash (`mimo_v2_flash`): full and sliding attention
layers over 192-wide queries and keys and 128-wide values, and DeepSeek V3's
routed experts without a shared one.

The sliding layers hold twice the full layers' key and value heads, add a
learned sink per head and rotate a third of each head under their own base;
every layer scales its values by `attention_value_scale` before attending
(modeling_mimo_v2_flash.py, MiMoV2FlashAttention). The released configs
spell the layout in the authors' own fields (`hybrid_layer_pattern`,
`swa_num_key_value_heads`, ...), which transformers 5.16.1 does not read: it
derives the same model from its defaults, so each such field is accepted
only where it states what the reference computes.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from dew import records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    Renames,
    Ropes,
    base_config,
    decoder_kinds,
    decoder_tensors,
    plain_partial_rope,
    refuse,
    renamed_name,
    renamed_path,
)
from dew.interop.families.deepseek import deepseek_layout
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture

_KINDS = ('full_attention', 'sliding_attention')

_NAMES: Renames = (('self_attn.attention_sink_bias', 'self_attn.sinks'),)
"""The checkpoint's sink logits under the shared map's name for them."""


def _default_layers(layers: int) -> tuple[str, ...]:
    """MiMoV2FlashConfig's pattern: the first layer and every sixth full, the
    rest sliding (configuration_mimo_v2_flash.py:112-116)."""
    return tuple('full_attention' if index == 0 or not (index + 1) % 6 else 'sliding_attention'
                 for index in range(layers))


def _rope(hf: Mapping[str, object], kind: str) -> tuple[float, float]:
    """One kind's (rope_theta, partial_rotary_factor): a `rope_parameters`
    entry, or the released flat fields with the reference's defaults."""
    nested = hf.get('rope_parameters')
    if nested is not None:
        entry = records.record(records.record(nested, 'rope_parameters').get(kind),
                               f'rope_parameters {kind}')
        return plain_partial_rope({'rope_parameters': entry}, default_factor=0.334,
                                  reference='MiMoV2FlashRotaryEmbedding')
    theta = hf.get('rope_theta' if kind == 'full_attention' else 'swa_rope_theta',
                   5_000_000.0 if kind == 'full_attention' else 10_000.0)
    return (records.number(theta, 'rope_theta'),
            records.number(hf.get('partial_rotary_factor', 0.334), 'partial_rotary_factor'))


def _check_released(hf: Mapping[str, object], used: set[str], *, layers: tuple[str, ...],
                    sparse: tuple[int, ...], kv_heads: int) -> None:
    """Accept the authors' own fields where they state what transformers
    computes from its defaults, and refuse any that state another model."""
    heads = records.integer(hf['num_attention_heads'], 'num_attention_heads')
    head_dim = records.integer(hf.get('head_dim', 192), 'head_dim')
    value_dim = records.integer(hf.get('v_head_dim', 128), 'v_head_dim')
    window = hf.get('sliding_window', 128)
    expected = {
        'hybrid_layer_pattern': [0 if kind == 'full_attention' else 1 for kind in layers],
        'moe_layer_freq': [int(index in sparse) for index in range(len(layers))],
        'add_swa_attention_sink_bias': True, 'add_full_attention_sink_bias': False,
        'swa_num_attention_heads': heads, 'swa_num_key_value_heads': 2 * kv_heads,
        'swa_head_dim': head_dim, 'swa_v_head_dim': value_dim,
        'sliding_window_size': window, 'layernorm_epsilon': hf.get('rms_norm_eps', 1e-5),
        'n_shared_experts': None, 'scoring_func': 'sigmoid', 'topk_method': 'noaux_tc',
    }
    for key, value in expected.items():
        if key in hf and hf[key] != value:
            refuse(f'{key}={hf[key]!r}', f'transformers computes {value!r} from MiMoV2FlashConfig')
    # The released configs state the window again as a chunk the reference
    # never reads; attention stays sliding, not chunked.
    used.update((*expected, 'attention_chunk_size', 'swa_rope_theta'))


def _mimo_v2_flash_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    count = records.integer(hf['num_hidden_layers'], 'num_hidden_layers')
    layers = (records.strings(hf['layer_types'], 'layer_types') if hf.get('layer_types') is not None
              else _default_layers(count))
    if set(layers) - set(_KINDS) or len(layers) != count:
        refuse(f'layer_types {list(layers)!r}', 'one full_attention or sliding_attention entry per layer')
    schedule = (records.strings(hf['mlp_layer_types'], 'mlp_layer_types')
                if hf.get('mlp_layer_types') is not None else ('dense',) + ('sparse',) * (count - 1))
    if set(schedule) - {'dense', 'sparse'} or len(schedule) != count:
        refuse(f'mlp_layer_types {list(schedule)!r}', 'one dense or sparse entry per layer')
    sparse = tuple(index for index, kind in enumerate(schedule) if kind == 'sparse')
    kv_heads = records.integer(hf.get('num_key_value_heads', 4), 'num_key_value_heads')
    _check_released(hf, used, layers=layers, sparse=sparse, kv_heads=kv_heads)
    (full_theta, full_factor), (local_theta, local_factor) = (_rope(hf, kind) for kind in _KINDS)
    used.update(('layer_types', 'mlp_layer_types', 'rope_parameters', 'rope_theta', 'partial_rotary_factor',
                 'v_head_dim', 'attention_value_scale', 'norm_topk_prob', 'attention_dropout'))
    config = base_config({**hf, 'head_dim': hf.get('head_dim', 192), 'num_key_value_heads': kv_heads,
                          'rms_norm_eps': hf.get('rms_norm_eps', hf.get('layernorm_epsilon', 1e-5))},
                         used, layer_types=layers, rope=Ropes(full_theta), scale_after_cast=True,
                         reads=frozenset({'sliding_window', 'attention_bias'}))
    window = (records.integer(hf.get('sliding_window', 128), 'sliding_window')
              if 'sliding_attention' in layers else None)
    kinds = decoder_kinds(layers, window, local_theta, None, None)
    if 'sliding_attention' in layers:
        # MiMoV2FlashAttention doubles a sliding layer's key and value heads
        # and gives it the sinks (modeling_mimo_v2_flash.py:348, 363).
        kinds['sliding_attention'].update(num_kv_heads=2 * kv_heads, sinks=True,
                                          partial_rotary_factor=local_factor)
    if hf.get('norm_topk_prob', True) is not True:
        refuse('norm_topk_prob', 'DeepseekV3TopkRouter renormalizes the top-k weights')
    scale = hf.get('attention_value_scale', 0.707)
    config.update(
        kinds=kinds, value_head_dim=records.integer(hf.get('v_head_dim', 128), 'v_head_dim'),
        value_scale=None if scale is None else records.number(scale, 'attention_value_scale'),
        o_proj_bias=False, partial_rotary_factor=None if full_factor == 1.0 else full_factor,
        partial_rotary_type='default',
        attention_dropout_rate=records.number(hf.get('attention_dropout', 0.), 'attention_dropout'),
        mixture=native_fields(Mixture)(
            **deepseek_layout({**hf, 'routed_scaling_factor': hf.get('routed_scaling_factor') or 1.0},
                              count, used, sparse_layers=sparse),
            score_function='sigmoid', bias=True,
            groups=records.integer(hf.get('n_group') or 1, 'n_group'),
            groups_per_token=records.integer(hf.get('topk_group') or 1, 'topk_group')) if sparse else None)
    return config


def _mimo_v2_flash_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """The shared map under the family's names; the released checkpoints'
    `model.mtp.*` prediction layers, which the reference never runs
    (`_keys_to_ignore_on_load_unexpected`), have no place."""
    if name.startswith('model.mtp.'):
        return None
    return renamed_path(_NAMES, name, config)


def _mimo_v2_flash_export(model: CausalTransformer) -> Mapping[str, object]:
    sliding = model.kind_of('sliding_attention')
    full = model.kind_of('full_attention')
    mixture = model.mixture
    return {
        'layer_types': list(model.per_layer_types),
        'mlp_layer_types': ['sparse' if mixture is not None and index in model.sparse_layers else 'dense'
                            for index in range(model.num_layers)],
        'num_key_value_heads': full.num_kv_heads, 'v_head_dim': model.value_head_dim,
        'attention_value_scale': model.value_scale,
        'rope_parameters': {kind: {'rope_type': 'default', 'rope_theta': resolved.rope_theta,
                                   'partial_rotary_factor': resolved.partial_rotary_factor or 1.0}
                            for kind, resolved in (('full_attention', full), ('sliding_attention', sliding))},
        'moe_intermediate_size': None if mixture is None else mixture.expert_features,
        'n_routed_experts': None if mixture is None else mixture.experts,
        'num_experts_per_tok': None if mixture is None else mixture.top_k,
        'routed_scaling_factor': None if mixture is None else mixture.scaling,
        'n_group': None if mixture is None else mixture.groups,
        'topk_group': None if mixture is None else mixture.groups_per_token,
        'norm_topk_prob': True, 'attention_dropout': model.attention_dropout_rate,
        'rope_theta': None, 'rope_local_base_freq': None,
    }


def _matches(fields: CausalTransformer) -> bool:
    """Values narrower than the heads, which no other family's attention has."""
    return fields.value_head_dim is not None and fields.value_head_dim != fields.features_per_head


MIMO_V2_FLASH = DecoderFamily(
    ('mimo_v2_flash',), _mimo_v2_flash_config, _matches, 'mimo_v2_flash', 'MiMoV2FlashForCausalLM',
    _mimo_v2_flash_export, weight_path=_mimo_v2_flash_path, export_path=partial(renamed_name, _NAMES),
    # Its partial rotary turns heads as they are stored, so the shared writer writes them unchanged.
    export_weights=decoder_tensors, preserve_source_layout=False,
)
