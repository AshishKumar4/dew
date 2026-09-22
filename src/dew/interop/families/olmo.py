"""Translate OLMo 3, a post-norm block with one rotary table per layer kind.

The released YaRN sits on the full-attention layers alone, which is why the
kinds carry their own rope rather than the model.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from dew import records
from dew.interop.hf_decoders import (
    DEFAULT_MAX_SEQ_LEN,
    DecoderFields,
    _base_config,
    _rope,
    _specified_layer_types,
)


def _olmo3_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read an OLMo 3 config into `CausalTransformer` fields.

    The block is post-norm: both norms sit on the sublayer outputs
    (modeling_olmo3.py:259-266, the sandwich pair without the input pair), the
    q/k RMSNorms cover the whole projection before the head split (:162-163,
    :178-179), and three sliding layers follow one full one
    (configuration_olmo3.py:96-98).

    `Olmo3RotaryEmbedding` builds one frequency table per layer kind, calling
    `ROPE_INIT_FUNCTIONS[rope_type]` for the kinds whose entry names a type of
    its own (modeling_olmo3.py:277-291). A ramp is therefore a kind's, not the
    model's. A flat `rope_scaling` belongs to the full-attention layers alone,
    where the reference moves it (configuration_olmo3.py:110-113): the released
    7B checkpoints put a YaRN there and the sliding layers rotate plainly at
    rope_theta.
    """
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    layer_types = _specified_layer_types(hf_config, used, tuple(
        'sliding_attention' if (index + 1) % 4 else 'full_attention'
        for index in range(layers)))
    ropes = _rope(hf_config, used, records.integer(hf_config.get(
        'max_position_embeddings', DEFAULT_MAX_SEQ_LEN), 'max_position_embeddings'))
    if ropes.scaling is not None and not isinstance(hf_config.get('rope_parameters'), Mapping):
        ropes = dataclasses.replace(ropes, full_only=True)
    config = _base_config(hf_config, used, qk_norm=True, layer_types=layer_types,
                          scale_after_cast=False, rope=ropes)
    config.update(sandwich_norms=True, pre_norms=False, qk_norm_scope='projection')
    return config
