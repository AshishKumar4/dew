"""Translate Llama, Mistral and Mixtral, the dense block the other families vary.

Llama's own translation is the shared `_base_config` over the fields
LlamaConfig declares, so what stands here is only what the variants add to
it. Mistral adds a sliding window on every layer and drops the attention
biases. Ministral names a pattern of sliding and full layers. Mixtral adds a
softmax-routed feed-forward and the expert tensor names that routing brings.
"""

from __future__ import annotations

from collections.abc import Mapping

from dew import records
from dew.interop.hf_decoders import _base_config, _dew_path, _refuse, _softmax_mixture


def _llama_config(hf_config, used):
    # LlamaConfig declares attention_bias and no window: LlamaAttention
    # attends every key.
    return _base_config(hf_config, used, reads=frozenset({'attention_bias'}))


def _ministral_config(hf_config, used):
    # MinistralConfig declares layer_types and sliding_window, and its
    # attention builds no biases.
    return _base_config(hf_config, used, reads=frozenset({'layer_types', 'sliding_window'}))


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
    # MistralConfig declares sliding_window alone; MistralAttention builds
    # no biases.
    return _base_config(hf_config, used, layer_types=(
        'full_attention' if window is None else 'sliding_attention',) * layers,
        reads=frozenset({'sliding_window'}))


def _mixtral_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    name = name.replace('.block_sparse_moe.', '.mlp.')
    for theirs, ours in (('w1', 'gate_proj'), ('w2', 'down_proj'), ('w3', 'up_proj')):
        name = name.replace(f'.{theirs}.weight', f'.{ours}.weight')
    return _dew_path(name, config)
