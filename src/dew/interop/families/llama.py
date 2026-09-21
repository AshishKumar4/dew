"""Llama, Mistral and Mixtral: the dense block the other families vary.

Llama's own translation is the shared `_base_config`, so what stands here is
what the two variants add to it: Mistral's sliding window on every layer,
Mixtral's softmax-routed feed-forward, and the expert tensor names that
routing brings.
"""

from __future__ import annotations

from collections.abc import Mapping

from dew import records
from dew.interop.hf_decoders import _base_config, _dew_path, _refuse, _softmax_mixture


def _mixtral_config(hf_config, used):
    config = _mistral_config(hf_config, used)
    used.update(('num_local_experts', 'router_jitter_noise'))
    if hf_config.get('router_jitter_noise', 0.0):
        _refuse('router_jitter_noise', 'training-time input jitter has no counterpart')
    config['mixture'] = _softmax_mixture(
        hf_config, used, experts=records.integer(hf_config['num_local_experts'], 'num_local_experts'))
    return config


def _mistral_config(hf_config, used):
    layers = int(hf_config['num_hidden_layers'])
    window = hf_config.get('sliding_window')
    return _base_config(hf_config, used, layer_types=(
        'full_attention' if window is None else 'sliding_attention',) * layers)


def _mixtral_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    name = name.replace('.block_sparse_moe.', '.mlp.')
    for theirs, ours in (('w1', 'gate_proj'), ('w2', 'down_proj'), ('w3', 'up_proj')):
        name = name.replace(f'.{theirs}.weight', f'.{ours}.weight')
    return _dew_path(name, config)
