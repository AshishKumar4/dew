"""Translate the Gemma decoders: gemma, gemma2, gemma3_text, gemma3n_text, gemma4_text.

Each release keeps the previous block and adds one thing: Gemma 2 the
sandwich norms and the two softcaps, Gemma 3 the q/k norms and the
alternating local window, Gemma 3n AltUp's residual copies with the LAuReL
projections and the per-layer inputs, Gemma 4 the value norm, the shared KV
layers and the routed branch. The export side is here too, because a Gemma
config is written back from the computation rather than from a template.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

from dew import records
from dew.interop.hf_decoders import (
    _MOE_SHARED,
    AltUpFields,
    DecoderFields,
    _base_config,
    _dew_path,
    _fixed_fields,
    _fixed_mixture,
    _flatten,
    _hf_activation,
    _hf_name,
    _kinds,
    _kinds_of,
    _refuse,
    _rope,
    _rope_theta,
    _Ropes,
    _specified_layer_types,
)
from dew.nn.backbones.causal_transformer import CausalTransformer, Mixture
from dew.nn.gemma3n import AltUp
from dew.nn.mixers import AttentionMixer


def _gemma_layer_types(hf_config: Mapping[str, object], used: set[str], *,
                       last_full: bool = False) -> tuple[str, ...]:
    if hf_config.get('layer_types') is not None:
        types = _specified_layer_types(hf_config, used)
    else:
        pattern = 6 if last_full else records.integer(hf_config.get('sliding_window_pattern', 6), 'sliding_window_pattern')
        if not last_full:
            used.add('sliding_window_pattern')
        types = tuple('sliding_attention' if (index + 1) % pattern else 'full_attention'
                      for index in range(records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')))
    # Gemma4TextConfig rewrites the final layer before building the model.
    return (*types[:-1], 'full_attention') if last_full and types else types


def _gemma4_rope(entries: Mapping[str, object]) -> tuple[float, float | None, float | None]:
    """Return (rope_theta, rope_local_theta, partial_rotary_factor) for gemma4.

    The full layers may rotate a fraction of their head dims (proportional
    partial rotary); the sliding layers rotate all of theirs. Anything but
    those two shapes refuses with the entry named.
    """
    full = records.record(entries.get('full_attention') or {}, 'rope_parameters.full_attention')
    sliding = records.record(entries.get('sliding_attention') or {}, 'rope_parameters.sliding_attention')
    for kind, entry in (('sliding_attention', sliding),):
        factor = entry.get('partial_rotary_factor')
        if factor not in (None, 1, 1.0):
            _refuse(f"rope_parameters.{kind} partial_rotary_factor {factor}",
                    "partial rotary applies to the full layers only")
        _rope_theta({**entry, 'rope_type': entry.get('rope_type', entry.get('type', 'default'))},
                    f"rope_parameters.{kind}")
    local = sliding.get('rope_theta', 10000.0)
    kind, entry = 'full_attention', full
    rope_type = entry.get('rope_type', entry.get('type', 'default'))
    factor = entry.get('partial_rotary_factor')
    if rope_type == 'proportional':
        if factor is None:
            _refuse("rope_parameters.full_attention",
                    "proportional rope needs its partial_rotary_factor")
        extra = sorted(set(entry) - {'rope_type', 'type', 'rope_theta',
                                     'partial_rotary_factor', 'factor'})
        if extra or entry.get('factor', 1.0) not in (1, 1.0):
            _refuse("rope_parameters.full_attention scaling",
                    "the backbone applies plain rotary positions at rope_theta")
        partial = records.number(factor, 'rope_parameters.full_attention partial_rotary_factor')
        theta = records.number(entry.get('rope_theta', 1000000.0), 'rope_parameters.full_attention rope_theta')
    elif rope_type in ('default', 'none'):
        if factor not in (None, 1, 1.0):
            _refuse("rope_parameters.full_attention partial_rotary_factor",
                    "partial rotary comes spelled proportional")
        partial = None
        theta = _rope_theta(entry, 'rope_parameters.full_attention') or 10000.0
    else:
        _refuse(f"rope_parameters.full_attention (rope_type {rope_type!r})",
                "the backbone applies plain rotary positions at rope_theta")
    local = records.number(local, 'rope_parameters.sliding_attention rope_theta')

    return theta, (None if local == theta else local), partial


def _gemma_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Gemma 1 config into `CausalTransformer` fields.

    Gemma 1 has (1 + w) norms scaled in fp32, sqrt(d)-scaled embeddings, a
    tied head, and no norms beyond the two pre-norms (modeling_gemma.py:77,
    :374). Its released config names hidden_act 'gelu', which the reference
    computes as the erf gelu (modeling_gemma.py:93, ACT2FN['gelu']).
    """
    config = _base_config(hf_config, used, scale_after_cast=False, tie_embeddings=True)
    config.update(scale_offset=True, embedding_scale=True)
    return config


def _gemma_softcaps(hf_config: Mapping[str, object], used: set[str],
                    config: DecoderFields) -> None:
    """Add the query scalar, the softcaps and the sandwich norms of Gemma 2 and 3.

    Gemma 3 reads attn_logit_softcapping into its attention without passing it
    on (modeling_gemma3.py:334, :370-379), so there it changes nothing. Gemma 2
    applies it (modeling_gemma2.py:282) and its entry maps it below.
    """
    config.update(scale_offset=True, embedding_scale=True, sandwich_norms=True)
    used.update(('query_pre_attn_scalar', 'final_logit_softcapping',
                 'attn_logit_softcapping'))
    scalar = hf_config.get('query_pre_attn_scalar')
    if scalar is not None:
        config['attention_scale'] = records.number(scalar, 'query_pre_attn_scalar') ** -0.5
    softcap = hf_config.get('final_logit_softcapping')
    if softcap is not None:
        config['final_logit_softcap'] = records.number(softcap, 'final_logit_softcapping')


def _gemma2_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Gemma 2 config into `CausalTransformer` fields.

    Gemma 2 is Gemma 3's block without the q/k norms. It alternates sliding and
    full layers at one rope base and softcaps the attention logits with tanh
    (configuration_gemma2.py:95-98, modeling_gemma2.py:203-206).
    """
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    layer_types = _specified_layer_types(hf_config, used, tuple(
        'sliding_attention' if (index + 1) % 2 else 'full_attention'
        for index in range(layers)))
    config = _base_config(hf_config, used, scale_after_cast=False,
                          tie_embeddings=True, layer_types=layer_types)
    _gemma_softcaps(hf_config, used, config)
    softcap = hf_config.get('attn_logit_softcapping')
    if softcap is not None:
        config['attn_logit_softcap'] = records.number(softcap, 'attn_logit_softcapping')
    return config


def _gemma3_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    config = _base_config(hf_config, used, qk_norm=True, scale_after_cast=False,
                          tie_embeddings=True, layer_types=_gemma_layer_types(hf_config, used))
    _gemma_softcaps(hf_config, used, config)
    return config


def _gemma3n_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Gemma 3n config into `CausalTransformer` fields.

    Gemma 3n (E2B, E4B) adds AltUp's stack of residual copies, the LAuReL
    block, gaussian top-k sparsity on the first layers, one feed-forward width
    per layer, per-layer inputs and KV sharing over the last layers.
    """
    if hf_config.get('layer_types') is not None:
        layer_types = _specified_layer_types(hf_config, used)
    else:
        # Gemma3nTextConfig fills every fifth layer full.
        layer_types = tuple('full_attention' if (index + 1) % 5 == 0 else 'sliding_attention'
                            for index in range(records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')))
    # Gemma3nTextConfig folds a flat rope_scaling into the full layers'
    # entry (convert_rope_params_to_dict) and defaults the bases to 1e6 for
    # the full layers and 1e4 for the sliding ones, spelled rope_theta and
    # rope_local_base_freq by the released config and nested by a config
    # transformers wrote.
    nested = hf_config.get('rope_parameters')
    scaling = hf_config.get('rope_scaling')
    if isinstance(nested, Mapping) and 'rope_theta' not in nested and isinstance(scaling, Mapping):
        full = nested.get('full_attention') or {}
        hf_config = {**hf_config, 'rope_scaling': None,
                     'rope_parameters': {**nested, 'full_attention': {**full, **scaling}}}
    ropes = _rope(hf_config, used)
    if not isinstance(nested, Mapping) and 'rope_theta' not in hf_config:
        ropes = dataclasses.replace(ropes, theta=1000000.0)
    if ropes.local_theta is None and not (
            (isinstance(nested, Mapping) and (nested.get('sliding_attention') or {}).get('rope_theta'))
            or hf_config.get('rope_local_base_freq') is not None):
        ropes = dataclasses.replace(ropes, local_theta=None if ropes.theta == 10000.0 else 10000.0)
    config = _base_config(hf_config, used, qk_norm=True, scale_after_cast=False,
                          tie_embeddings=True, layer_types=layer_types, rope=ropes)
    layers = records.integer(config.get('num_layers'), 'num_layers')
    sparsity = hf_config.get('activation_sparsity_pattern')
    if sparsity is None:
        # The reference default is the first ten layers at 0.95 when there
        # are more than ten, else none (configuration_gemma3n.py).
        sparse = 10 if layers > 10 else 0
        sparsity = [0.95] * sparse + [0.0] * (layers - sparse)
    if not isinstance(sparsity, (list, tuple)) or len(sparsity) != layers:
        _refuse(f"activation_sparsity_pattern {sparsity!r}",
                f"the reference takes one fraction per layer of {layers}")
    used.update(('activation_sparsity_pattern', 'laurel_rank', 'altup_num_inputs',
                 'altup_active_idx', 'altup_coef_clip', 'altup_correct_scale',
                 'hidden_size_per_layer_input', 'vocab_size_per_layer_input',
                 'num_kv_shared_layers', 'final_logit_softcapping'))
    # PreTrainedConfig serializes this field; Gemma3nTextMLP never reads it.
    used.add('chunk_size_feed_forward')
    clip = hf_config.get('altup_coef_clip', 120.0)
    # AltUp's own checks name the reference's fields, so a config out of
    # their range is refused here.
    altup = AltUp(num_inputs=records.integer(hf_config.get('altup_num_inputs', 4), 'altup_num_inputs'),
                  active_idx=records.integer(hf_config.get('altup_active_idx', 0), 'altup_active_idx'),
                  coef_clip=None if clip is None else records.number(clip, 'altup_coef_clip'),
                  correct_scale=bool(hf_config.get('altup_correct_scale', True)))
    config.update(
        sandwich_norms=True, embedding_scale=True, attention_scale=1.0, v_norm=True,
        activation_sparsity_pattern=tuple(float(fraction) for fraction in sparsity),
        laurel_rank=records.integer(hf_config.get('laurel_rank', 64), 'laurel_rank'),
        altup=AltUpFields(**dataclasses.asdict(altup)),
        per_layer_input_dim=records.integer(hf_config.get('hidden_size_per_layer_input', 256), 'hidden_size_per_layer_input'),
        per_layer_input_vocab=records.integer(hf_config.get('vocab_size_per_layer_input', 262144), 'vocab_size_per_layer_input'),
        num_kv_shared_layers=records.integer(hf_config.get('num_kv_shared_layers', 15), 'num_kv_shared_layers'),
        final_logit_softcap=records.number(hf_config.get('final_logit_softcapping', 30.0), 'final_logit_softcapping'),
    )
    return config


def _gemma3n_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the variables-tree path for one Gemma 3n tensor name.

    Gemma 3n adds the AltUp and LAuReL leaves beside a Gemma 4 style layer.
    The copies' projections are indexed modules (altup_projections.{i}), which
    land as altup_projections_{i} the way the layers do; the coefficient maps
    are Linears, so they transpose like any kernel.
    """
    parts = name.split('.')
    if len(parts) == 4 and parts[0] == 'model' and parts[2].isdigit() and parts[3] == 'weight' \
            and parts[1] in ('altup_projections', 'altup_unembed_projections'):
        return ('params', f'{parts[1]}_{parts[2]}', 'kernel')
    if len(parts) >= 5 and parts[:2] == ['model', 'layers'] and parts[2].isdigit() \
            and parts[3] in ('altup', 'laurel'):
        layer = ('params', f'layers_{parts[2]}', parts[3])
        tail = parts[4:]
        if tail == ['correct_output_scale']:
            return (*layer, 'correct_output_scale')
        if len(tail) == 2 and tail[1] == 'weight':
            if tail[0] in ('router_norm', 'post_laurel_norm'):
                return (*layer, tail[0], 'scale')
            if tail[0] in ('correction_coefs', 'prediction_coefs', 'modality_router',
                           'linear_left', 'linear_right'):
                return (*layer, tail[0], 'kernel')
        raise ValueError(f"unknown tensor name {name!r}")
    return _dew_path(name, config)


def _gemma4_config(hf_config: Mapping[str, object], used: set[str], *,
                   k_eq_v: bool = False) -> DecoderFields:
    layer_types = _gemma_layer_types(hf_config, used, last_full=True)
    config = _base_config(hf_config, used, qk_norm=True, scale_after_cast=False,
                          tie_embeddings=True, layer_types=layer_types, rope=_Ropes(10000.0))
    # The reference's final-layer rewrite takes precedence over an explicit pattern.
    config['layer_types'] = layer_types
    sliding_dim = records.integer(config.get('head_dim'), 'head_dim')
    kv_heads = records.integer(config.get('num_kv_heads'), 'num_key_value_heads')
    k_eq_v = k_eq_v or bool(hf_config.get('attention_k_eq_v', False))
    used.update(('attention_k_eq_v', 'enable_moe_block', 'per_layer_config',
                 'global_head_dim', 'num_global_key_value_heads'))
    # Every layer reads its geometry from per_layer_config
    # (modeling_gemma4.py, Gemma4TextAttention reads layer_config), whose
    # entries the sliding layers leave at the model's head_dim and
    # num_key_value_heads. Where the config carries no per_layer_config key
    # at all, configuration_gemma4.py builds the full layers' entries from
    # global_head_dim (512 unless named) and, under attention_k_eq_v alone,
    # num_global_key_value_heads; a config that carries the key, null or
    # filled, leaves those two fields unread. The released E2B is one, its
    # global q_proj 8 heads of 256 against the 512 it names.
    full_dim, full_kv = sliding_dim, kv_heads
    if 'per_layer_config' in hf_config:
        entries = hf_config['per_layer_config'] or {}
        per_layer_entries = (entries.values() if isinstance(entries, Mapping)
                             else entries if isinstance(entries, (list, tuple))
                             else _refuse(f"per_layer_config={entries!r}",
                                          "the per-layer overrides are records by layer"))
        for entry in per_layer_entries:
            if not isinstance(entry, Mapping):
                continue
            if entry.get('head_dim') is not None:
                full_dim = records.integer(entry['head_dim'], 'per_layer_config head_dim')
            if entry.get('num_key_value_heads') is not None:
                stated = records.integer(entry['num_key_value_heads'],
                              'per_layer_config num_key_value_heads')
                if full_kv != kv_heads and stated != full_kv:
                    _refuse("per_layer_config num_key_value_heads",
                            f"the full layers name both {full_kv} and {stated}")
                full_kv = stated
    else:
        full_dim = records.integer(hf_config.get('global_head_dim', 512), 'global_head_dim')
        global_kv = hf_config.get('num_global_key_value_heads')
        if global_kv is not None and k_eq_v:
            full_kv = records.integer(global_kv, 'num_global_key_value_heads')
    # Proportional rope rotates a fraction of the full layers' head dims
    # and passes the rest through; sliding layers rotate all of theirs.
    entries = records.record(hf_config.get('rope_parameters') or {}, 'rope_parameters')
    rope_theta, rope_local_theta, partial = _gemma4_rope(entries)
    used.update(('rope_parameters', 'rope_theta'))
    per_layer = records.integer(hf_config.get('hidden_size_per_layer_input', 0), 'hidden_size_per_layer_input')
    config.update(
        sandwich_norms=True, embedding_scale=True, attention_scale=1.0,
        v_norm=True,
        head_dim=sliding_dim,
        rope_theta=rope_theta,
        kinds=_kinds(layer_types, _kinds_of(config).get(
            'sliding_attention', {}).get('window'), rope_local_theta, None,
            None if full_dim == sliding_dim else full_dim),
        partial_rotary_factor=partial,
        use_double_wide_mlp=bool(hf_config.get('use_double_wide_mlp', False)),
        num_kv_shared_layers=records.integer(hf_config.get('num_kv_shared_layers', 0), 'num_kv_shared_layers'),
        per_layer_input_dim=per_layer or None,
        per_layer_input_vocab=records.integer(hf_config.get(
            'vocab_size_per_layer_input',
            records.integer(hf_config['vocab_size'], 'vocab_size')), 'vocab_size_per_layer_input'),
    )
    used.update(('use_double_wide_mlp', 'num_kv_shared_layers',
                 'hidden_size_per_layer_input', 'vocab_size_per_layer_input',
                 'final_logit_softcapping'))
    # attention_logit_cap changes nothing on the text path and maps to
    # nothing. Gemma4TextAttention never passes it to its attention call
    # (modeling_gemma4.py, Gemma4TextAttention.forward). Only the audio
    # attention applies one.
    used.add('attention_logit_cap')
    softcap = hf_config.get('final_logit_softcapping')
    if softcap is not None:
        config['final_logit_softcap'] = records.number(softcap, 'final_logit_softcapping')
    if full_kv != kv_heads:
        _kinds_of(config).setdefault('full_attention', {})['num_kv_heads'] = full_kv
    # Every released Gemma 4 checkpoint carries the layer_scalar buffer the
    # reference initialises to one, so the tree always holds it.
    config.update(attention_k_eq_v=k_eq_v, layer_scalar="frozen")
    used.update(('moe_intermediate_size', 'expert_intermediate_size',
                 'num_experts', 'top_k_experts', 'chunk_size_feed_forward'))
    if hf_config.get('enable_moe_block'):
        # The 26B-A4B routes every layer beside its dense MLP.
        for field in ('num_experts', 'top_k_experts', 'moe_intermediate_size'):
            if hf_config.get(field) is None:
                _refuse("enable_moe_block=True", f"the routed branch needs {field}")
        config['mixture'] = {
            'experts': records.integer(hf_config['num_experts'], 'num_experts'),
            'top_k': records.integer(hf_config['top_k_experts'], 'top_k_experts'),
            'expert_features': records.integer(hf_config['moe_intermediate_size'], 'moe_intermediate_size'),
            'parallel': True,
        }

    return config


def _gemma3_export(model: CausalTransformer) -> Mapping[str, object]:
    return {
        'hidden_activation': _hf_activation(model.mlp),
        'query_pre_attn_scalar': (None if model.attention_scale is None
                                 else round(1.0 / model.attention_scale ** 2)),
        'final_logit_softcapping': model.final_logit_softcap,
        'attn_logit_softcapping': None,
        'use_bidirectional_attention': False,
    }


def _gemma4_export(model: CausalTransformer) -> Mapping[str, object]:
    """Return the Gemma4TextConfig fields that describe `model`'s computation.

    Every field is read off the model, never copied from the config a source
    shipped, so an exported model and a trained one write the same file.
    """
    fixed = {'qk_norm': True, 'v_norm': True, 'sandwich_norms': True, 'pre_norms': True,
             'embedding_scale': True, 'attention_scale': 1.0, 'scale_offset': False,
             'scale_after_cast': False, 'qk_norm_scope': 'head', 'causal': True}
    _fixed_fields(model, fixed, 'Gemma4 computes {0!r} for this field')
    for name in ('output_gate', 'attention_sinks', 'attn_logit_softcap', 'altup',
                 'laurel_rank', 'activation_sparsity_pattern', 'num_nextn_predict_layers', 'dropout_rate'):
        if getattr(model, name):
            _refuse(name, 'the standalone Gemma4 reference has no such computation')
    if model.o_proj_bias is not None and model.o_proj_bias != model.attention_bias:
        _refuse('o_proj_bias', 'Gemma4 uses one attention_bias setting for every projection')
    if model.partial_rotary_factor is not None and model.partial_rotary_type != 'proportional':
        _refuse('partial_rotary_type', 'Gemma4 full layers use proportional rotary')
    if model.layer_scalar not in ('frozen', 'trainable'):
        _refuse('layer_scalar', 'Gemma4 exports an explicit frozen or trainable scalar value')
    types = model.per_layer_types
    if not types or set(types) - {'sliding_attention', 'full_attention'}:
        _refuse('layer_types', 'Gemma4 has only sliding and full attention')
    if types[-1] != 'full_attention':
        _refuse('layer_types', 'Gemma4TextConfig forces the final layer to full attention')
    for name in set(types):
        kind = model.kind_of(name)
        mixer = kind.mixer or model.mixer
        if mixer is not None and not isinstance(mixer, AttentionMixer):
            _refuse(f'kinds.{name}.mixer', 'Gemma4 uses ordinary attention')
        if isinstance(mixer, AttentionMixer) and (mixer.bidirectional_images or mixer.mrope_section is not None):
            _refuse(f'kinds.{name}.mixer', 'multimodal attention metadata needs its source wrapper')
        if kind.rope_scaling is not None or kind.yarn is not None:
            _refuse(f'kinds.{name}.rope', 'Gemma4 uses plain local and proportional global rotary')
    full = model.kind_of('full_attention')
    local = model.kind_of('sliding_attention') if 'sliding_attention' in types else full
    if full.window is not None:
        _refuse('kinds.full_attention.window', 'full attention is unwindowed')
    if 'sliding_attention' in types and (local.window is None or local.window < 1):
        _refuse('kinds.sliding_attention.window', 'sliding attention needs a positive window')
    sharing = model.sharing_layers
    if sharing != tuple(range(model.num_layers - len(sharing), model.num_layers)):
        _refuse('kv_shared_layers', 'Gemma4 can express only a trailing run of shared-KV layers')
    mixture = model.mixture
    fields: dict[str, object] = {
        'hidden_act': None, 'hidden_activation': _hf_activation(model.mlp),
        'layer_types': list(types), 'intermediate_size': model.hidden_features,
        'head_dim': local.head_dim, 'num_key_value_heads': local.num_kv_heads,
        'global_head_dim': full.head_dim, 'num_global_key_value_heads': full.num_kv_heads,
        # Explicit overrides avoid configuration_gemma4.py:210-223 replacing
        # full geometry with global defaults or gating KV heads on K=V.
        'per_layer_config': {str(index): {'head_dim': full.head_dim, 'num_key_value_heads': full.num_kv_heads}
                             for index, name in enumerate(types) if name == 'full_attention'},
        'rope_theta': None, 'rope_scaling': None,
        'rope_parameters': {
            'sliding_attention': {'rope_type': 'default', 'rope_theta': local.rope_theta},
            'full_attention': {'rope_type': 'proportional', 'rope_theta': full.rope_theta,
                               'partial_rotary_factor': model.partial_rotary_factor or 1.0}},
        'sliding_window': local.window if 'sliding_attention' in types else 512,
        'attention_k_eq_v': model.attention_k_eq_v, 'num_kv_shared_layers': len(sharing),
        'use_double_wide_mlp': model.use_double_wide_mlp,
        'hidden_size_per_layer_input': model.per_layer_input_dim or 0,
        'vocab_size_per_layer_input': model.per_layer_input_vocab or model.vocab_size,
        'final_logit_softcapping': model.final_logit_softcap, 'use_bidirectional_attention': None,
        'enable_moe_block': mixture is not None,
    }
    if mixture is not None:
        if not mixture.parallel or tuple(model.sparse_layers) != tuple(range(model.num_layers)):
            _refuse('mixture', 'Gemma4 routes a parallel expert branch on every layer')
        defaults = Mixture(experts=mixture.experts, top_k=mixture.top_k,
                           expert_features=mixture.expert_features, parallel=True)
        represented = {'experts', 'top_k', 'expert_features', 'parallel', 'layers', 'every',
                       'implementation', 'dispatch'}
        _fixed_mixture(mixture, defaults, represented,
                       'Gemma4 has its fixed parallel router and expert computation')
        fields.update(num_experts=mixture.experts, top_k_experts=mixture.top_k,
                      moe_intermediate_size=mixture.expert_features or model.hidden_features)
    return fields


def _gemma4_export_weights(model: CausalTransformer, variables: Mapping[str, object],
                           config: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Return the Gemma 4 checkpoint tensors for `model` and `variables`.

    This is the one text inverse standalone Gemma 4 and DiffusionGemma share.
    `flat` holds the flattened params and `fixed` the constants, so a leaf is
    read from the collection `model.layer_scalar` names.
    """
    mode = model.layer_scalar
    if mode not in ('frozen', 'trainable'):
        raise ValueError('Gemma4 tensor export requires an explicit layer_scalar mode')
    params = variables.get('params', variables)
    constants = variables.get('constants', {})
    if not isinstance(params, Mapping) or not isinstance(constants, Mapping):
        raise ValueError('params and constants must contain native variable trees')
    flat, fixed = dict(_flatten(params)), _flatten(constants)
    scalar_names = {f'layers_{index}.layer_scalar' for index in range(model.num_layers)}
    if set(fixed) - scalar_names:
        raise ValueError(f'unrepresented Gemma4 constants: {sorted(set(fixed) - scalar_names)}')
    for name in scalar_names:
        if mode == 'frozen':
            if name in flat or name not in fixed:
                raise ValueError(f'{name} requires constants only under layer_scalar=frozen')
            flat[name] = fixed[name]
        elif name in fixed or name not in flat:
            raise ValueError(f'{name} requires params only under layer_scalar=trainable')
    inverse = {value: key for key, value in _GEMMA4_MOE.items()}
    per_model = {'embed_tokens_per_layer.embedding': 'model.embed_tokens_per_layer.weight',
                 'per_layer_model_projection.kernel': 'model.per_layer_model_projection.weight',
                 'per_layer_projection_norm.scale': 'model.per_layer_projection_norm.weight'}
    tensors: dict[str, np.ndarray] = {}
    for name, raw in flat.items():
        parts, leaf = name.split('.'), np.asarray(raw)
        if parts[0].startswith('layers_') and tuple(parts[1:]) in inverse:
            layer = parts[0].removeprefix('layers_')
            target = f"model.layers.{layer}." + '.'.join(inverse[tuple(parts[1:])])
        elif len(parts) == 5 and parts[1:3] == ['moe', 'experts']:
            layer = parts[0].removeprefix('layers_')
            stem = f'model.layers.{layer}.experts.'
            if parts[-1] != 'kernel':
                raise ValueError(f'unknown expert parameter {name!r}')
            if parts[3] == 'up_proj':
                if name.replace('.up_proj.', '.gate_proj.') not in flat:
                    raise ValueError(f'{name} has no matching gate projection')
                continue
            if parts[3] == 'gate_proj':
                up = np.asarray(flat[name.replace('.gate_proj.', '.up_proj.')])
                tensors[stem + 'gate_up_proj'] = np.ascontiguousarray(
                    np.swapaxes(np.concatenate([leaf, up], axis=-1), -1, -2))
            elif parts[3] == 'down_proj':
                tensors[stem + 'down_proj'] = np.ascontiguousarray(np.swapaxes(leaf, -1, -2))
            else:
                raise ValueError(f'unknown expert parameter {name!r}')
            continue
        elif name in per_model:
            target = per_model[name]
        elif (len(parts) == 3 and parts[0].startswith('layers_')
              and parts[1] in ('per_layer_input_gate', 'per_layer_projection', 'post_per_layer_input_norm')):
            ending = 'scale' if parts[1] == 'post_per_layer_input_norm' else 'kernel'
            if parts[2] != ending:
                raise ValueError(f'unknown per-layer input parameter {name!r}')
            target = f"model.layers.{parts[0].removeprefix('layers_')}.{parts[1]}.weight"
        else:
            target = _hf_name(name, config)
            if target is None:
                continue
        if target in tensors:
            raise ValueError(f'duplicate canonical tensor {target!r}')
        tensors[target] = np.ascontiguousarray(leaf.T if parts[-1] == 'kernel' else leaf)
    return tensors


def _gemma2_export(model: CausalTransformer) -> Mapping[str, object]:
    return {**_gemma3_export(model), 'attn_logit_softcapping': model.attn_logit_softcap}


def _gemma4_prepare(tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Split the routed branch's fused experts into dew's stacked `[E, in, out]` kernels.

    `Gemma4TextExperts` holds `gate_up_proj` as `[E, 2 * expert, hidden]` with
    the gate in the first rows and `down_proj` as `[E, hidden, expert]`, each
    expert a torch Linear's `[out, in]`.
    """
    prepared: dict[str, np.ndarray] = {}
    for name, tensor in tensors.items():
        if name.endswith('.experts.gate_up_proj'):
            width = tensor.shape[1] // 2
            stem = name[:-len('gate_up_proj')]
            prepared[stem + 'gate_proj'] = np.swapaxes(tensor[:, :width], 1, 2)
            prepared[stem + 'up_proj'] = np.swapaxes(tensor[:, width:], 1, 2)
        elif name.endswith('.experts.down_proj'):
            prepared[name] = np.swapaxes(tensor, 1, 2)
        else:
            prepared[name] = tensor
    return prepared


# Gemma 4's routed branch, named for what each norm normalises, as the
# block's own sandwich norms are.
_GEMMA4_MOE: dict[tuple[str, ...], tuple[str, ...]] = {
    ('router', 'proj', 'weight'): ('moe', 'router', 'proj', 'kernel'),
    ('router', 'scale'): ('moe', 'router', 'scale'),
    ('router', 'per_expert_scale'): ('moe', 'router', 'per_expert_scale'),
    ('post_feedforward_layernorm_1', 'weight'): ('moe', 'mlp_branch_norm', 'scale'),
    ('pre_feedforward_layernorm_2', 'weight'): ('moe', 'experts_input_norm', 'scale'),
    ('post_feedforward_layernorm_2', 'weight'): ('moe', 'experts_output_norm', 'scale'),
    ('layer_scalar',): ('layer_scalar',),
}


def _gemma4_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    parts = name.split('.')
    if len(parts) >= 4 and parts[:2] == ['model', 'layers'] and parts[2].isdigit():
        tail = tuple(parts[3:])
        if tail == ('layer_scalar',):
            mode = config.get("layer_scalar")
            if mode not in ("frozen", "trainable"):
                _refuse("layer_scalar", "the source scalar requires a frozen or trainable model mode")
            collection = "constants" if mode == "frozen" else "params"
            return (collection, f'layers_{parts[2]}', 'layer_scalar')
        layer = ('params', f'layers_{parts[2]}')
        if tail in _GEMMA4_MOE:
            return (*layer, *_GEMMA4_MOE[tail])
        if len(tail) == 2 and tail[0] == 'experts' and tail[1] in _MOE_SHARED:
            return (*layer, 'moe', 'experts', tail[1], 'kernel')
    return _dew_path(name, config)
