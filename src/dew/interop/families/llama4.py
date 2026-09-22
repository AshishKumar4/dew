"""Translate Llama 4's text decoder.

Its rotated layers run chunked local attention around global layers with no
rotary at all. QK-norm is set by layer kind. Its routed feed-forward stores a
fused `gate_up_proj`, one kernel per expert.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

from dew import records
from dew.interop.hf_decoders import (
    _MOE_SHARED,
    DecoderFields,
    KindFields,
    MixtureFields,
    _base_config,
    _dew_path,
    _refuse,
    _rope,
)
from dew.nn import llama4 as llama4_nn


def _llama4_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Llama 4 text config into `CausalTransformer` fields.

    Chunked rotated local layers surround global layers with no rope. Every
    interleaved layer is routed and carries a shared expert. The remaining
    layers take the wider dense MLP.
    """
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    no_rope = hf_config.get('no_rope_layers') or None
    if no_rope is None:
        no_rope = llama4_nn.default_no_rope_layers(
            layers, records.integer(hf_config.get('no_rope_layer_interval', 4), 'no_rope_layer_interval'))
    no_rope = records.integers(no_rope, 'no_rope_layers')
    if len(no_rope) != layers or set(no_rope) - {0, 1}:
        _refuse(f"no_rope_layers {list(no_rope)!r}", f"it names a flag per layer of {layers}")
    layer_types = llama4_nn.rope_layer_types(no_rope)
    stated = hf_config.get('layer_types')
    if stated is not None and records.strings(stated, 'layer_types') != layer_types:
        _refuse(f"layer_types {list(records.strings(stated, 'layer_types'))!r}",
                "it disagrees with no_rope_layers, which is what the reference reads")
    # The released Scout spells its llama3 ramp flat (rope_theta beside
    # rope_scaling); a config transformers wrote nests both. Either way the
    # ramp is the model's, and every rotated layer applies it.
    used.update(('no_rope_layers', 'no_rope_layer_interval', 'layer_types'))
    config = _base_config(hf_config, used, layer_types=layer_types,
                          rope=dataclasses.replace(_rope(hf_config, used), local_theta=None))
    if hf_config.get('router_jitter_noise', 0.0):
        _refuse("router_jitter_noise", "the router selects on the logits alone")
    if hf_config.get('output_router_logits', False):
        _refuse("output_router_logits", "the decoder returns token logits")
    used.update(('router_jitter_noise', 'output_router_logits', 'router_aux_loss_coef',
                 'intermediate_size_mlp', 'num_local_experts', 'num_experts_per_tok',
                 'moe_layers', 'interleave_moe_layer_step', 'attention_chunk_size',
                 'use_qk_norm', 'attn_temperature_tuning', 'floor_scale', 'attn_scale'))
    rule = {
        'kind': 'llama4',
        'use_qk_norm': bool(hf_config.get('use_qk_norm', True)),
        'attn_temperature_tuning': bool(hf_config.get('attn_temperature_tuning', True)),
        'floor_scale': records.number(hf_config.get('floor_scale', 8192), 'floor_scale'),
        'attn_scale': records.number(hf_config.get('attn_scale', 0.1), 'attn_scale'),
    }
    chunk = hf_config.get('attention_chunk_size')
    kinds: dict[str, KindFields] = {
        'full_attention': {'mixer': {**rule, 'use_rope': False}},
        'chunked_attention': {'mixer': {**rule, 'use_rope': True,
                                        'attention_chunk_size': None if chunk is None else records.integer(chunk, 'attention_chunk_size')}},
    }
    moe_layers = hf_config.get('moe_layers')
    step = records.integer(hf_config.get('interleave_moe_layer_step', 1), 'interleave_moe_layer_step')
    mixture: MixtureFields = {
        'experts': records.integer(hf_config['num_local_experts'], 'num_local_experts'),
        'top_k': records.integer(hf_config.get('num_experts_per_tok', 1), 'num_experts_per_tok'),
        'score_function': 'sigmoid',
        'norm_topk_prob': False,
        'scale_inputs': True,
        'expert_features': records.integer(hf_config['intermediate_size'], 'intermediate_size'),
        'shared_features': records.integer(hf_config['intermediate_size'], 'intermediate_size'),
    }
    if moe_layers is not None:
        mixture['layers'] = records.integers(moe_layers, 'moe_layers')
    else:
        mixture['every'] = step
    config.update(
        # The dense layers take intermediate_size_mlp; the experts and the
        # shared expert take intermediate_size.
        mlp_features=records.integer(hf_config['intermediate_size_mlp'], 'intermediate_size_mlp'),
        qk_norm=False,
        kinds={name: kinds[name] for name in kinds if name in layer_types},
        mixture=mixture,
    )
    return config


def _llama4_prepare(tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Split each fused `experts.gate_up_proj` into the two stacked kernels.

    `Llama4TextExperts` holds `[E, hidden, 2 * expert]` with the gate in the
    first half (`gate_up.chunk(2)`), already in the `[E, in, out]` layout dew's
    stacked expert kernels keep.
    """
    prepared: dict[str, np.ndarray] = {}
    for name, tensor in tensors.items():
        if name.endswith('.feed_forward.experts.gate_up_proj'):
            width = tensor.shape[-1] // 2
            stem = name[:-len('gate_up_proj')]
            prepared[stem + 'gate_proj'] = tensor[..., :width]
            prepared[stem + 'up_proj'] = tensor[..., width:]
        else:
            prepared[name] = tensor
    return prepared


def _llama4_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the variables-tree path for one Llama 4 tensor name.

    Llama 4 names its feed-forward `feed_forward`, its router `router` and its
    dense branch `shared_expert`. Its stacked expert kernels arrive without a
    `.weight` suffix.
    """
    parts = name.split('.')
    if len(parts) >= 5 and parts[:2] == ['model', 'layers'] and parts[2].isdigit() and parts[3] == 'feed_forward':
        layer = ('params', f'layers_{parts[2]}', 'mlp')
        if parts[4:] == ['router', 'weight']:
            return (*layer, 'gate', 'kernel')
        if len(parts) == 6 and parts[4] == 'experts' and parts[5] in _MOE_SHARED:
            return (*layer, 'experts', parts[5], 'kernel')
        if len(parts) == 7 and parts[4] == 'shared_expert' and parts[5] in _MOE_SHARED and parts[6] == 'weight':
            return (*layer, 'shared_experts', parts[5], 'kernel')
        if len(parts) == 6 and parts[4] in _MOE_SHARED and parts[5] == 'weight':
            return (*layer, parts[4], 'kernel')
        raise ValueError(f"unknown tensor name {name!r}")
    return _dew_path(name, config)
