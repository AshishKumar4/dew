"""Translate Ouro (`ouro`), ByteDance's looped decoder (arXiv 2510.25741).

Its block is Qwen2's without the biases, under sandwich norms that scale after
the cast; the whole stack runs `total_ut_steps` times, the final norm applied
after every pass, and a biased `early_exit_gate` reads each pass's normed
states (modeling_ouro.py at ByteDance/Ouro-1.4B 574fa66c, OuroModel.forward).
The remote code names the norm after the attention `input_layernorm_2` and
the two around the MLP `post_attention_layernorm` and
`post_attention_layernorm_2`.

OuroForCausalLM exits where the cumulative exit probability first reaches
`early_exit_threshold`. At 1.0 that is the last pass, unless an earlier
hazard rounds to 1 in floating point; the decoder here always reads the last
pass, so another threshold, or an `early_exit_step` before the last, is refused.
"""

from collections.abc import Mapping
from functools import partial

from dew import _records
from dew.interop.decoder_parts import (
    PLAIN_SPELLING,
    DecoderFamily,
    DecoderFields,
    Leaves,
    Renames,
    base_config,
    native_layer_types,
    read_rope,
    refuse,
    renamed_name,
    renamed_path,
)
from dew.nn.backbones.causal_transformer import CausalTransformer

_NAMES: Renames = (('input_layernorm_2', 'post_attention_layernorm'),
                   ('post_attention_layernorm', 'pre_feedforward_layernorm'),
                   ('post_attention_layernorm_2', 'post_feedforward_layernorm'))
"""Ouro's norm names onto Gemma 2's, which the shared sandwich map reads."""

_GATE: Leaves = {'model.early_exit_gate.weight': ('early_exit_gate', 'kernel'),
         'model.early_exit_gate.bias': ('early_exit_gate', 'bias')}

_REMOTE = 'ByteDance/Ouro-1.4B'
"""The repository whose remote code an exported config names, as the released ones do."""


def _ouro_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    steps = _records.integer(hf.get('total_ut_steps', 4), 'total_ut_steps')
    threshold = hf.get('early_exit_threshold', 1.0)
    if threshold is not None and _records.number(threshold, 'early_exit_threshold') != 1.0:
        refuse(f'early_exit_threshold {threshold!r}', 'the decoder reads the last pass, which 1.0 exits at')
    exit_step = hf.get('early_exit_step')
    if exit_step is not None and _records.integer(exit_step, 'early_exit_step') != steps - 1:
        refuse(f'early_exit_step {exit_step!r}', f'the decoder reads the last pass, step {steps - 1}')
    used.update(('total_ut_steps', 'early_exit_threshold', 'early_exit_step'))
    # OuroConfig drops the window unless use_sliding_window is set, and
    # derives layer_types with Qwen2Config's rule when none are stated.
    windowed = {**hf, 'sliding_window': hf.get('sliding_window') if hf.get('use_sliding_window') else None}
    # The release's own OuroRotaryEmbedding builds its table through ROPE_INIT_FUNCTIONS
    # (modeling_ouro.py:445-459 of ByteDance/Ouro-1.4B), as transformers' classes do; no check
    # holds a YaRN record against that remote code yet, so its ramps stay the plain table's and
    # Llama 3.1's.
    rope = read_rope(windowed, used, PLAIN_SPELLING, ramps=('llama3',))
    config = base_config(windowed, used, rope=rope, layer_types=native_layer_types(windowed, used, 'qwen2'),
                         reads=frozenset({'layer_types', 'sliding_window'}))
    config.update(sandwich_norms=True, loop={'steps': steps, 'exit_gate': True})
    return config


def _ouro_export(model: CausalTransformer) -> Mapping[str, object]:
    assert model.loop is not None
    return {
        'total_ut_steps': model.loop.steps, 'early_exit_threshold': 1.0,
        'use_sliding_window': 'sliding_attention' in model.per_layer_types, 'attention_bias': None,
        'auto_map': {'AutoConfig': f'{_REMOTE}--configuration_ouro.OuroConfig',
                     'AutoModel': f'{_REMOTE}--modeling_ouro.OuroModel',
                     'AutoModelForCausalLM': f'{_REMOTE}--modeling_ouro.OuroForCausalLM'},
    }


OURO = DecoderFamily(
    ('ouro',), _ouro_config, lambda fields: fields.loop is not None, 'ouro', 'OuroForCausalLM', _ouro_export,
    weight_path=partial(renamed_path, _NAMES, leaves=_GATE),
    export_path=partial(renamed_name, _NAMES, leaves=_GATE, sandwich_norms=True),
    preserve_source_layout=False,
    # OuroModel's table pads config.pad_token_id (modeling_ouro.py:496-500), which OuroConfig leaves
    # to the base class's None.
    padding_idx=True,
)
