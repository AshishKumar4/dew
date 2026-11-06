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

from dew import _records
from dew.interop.decoder_config import DecoderSpelling, decoder_layout, remote_code
from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    Leaves,
    Renames,
    native_layer_types,
    refuse,
    spelled,
)
from dew.interop.layouts import ConfigLayout, ConfigPart, constant
from dew.nn.backbones.causal_transformer import CausalTransformer

_NAMES: Renames = (('input_layernorm_2', 'post_attention_layernorm'),
                   ('post_attention_layernorm', 'pre_feedforward_layernorm'),
                   ('post_attention_layernorm_2', 'post_feedforward_layernorm'))
"""Ouro's norm names onto Gemma 2's, which the shared sandwich map reads."""

_GATE: Leaves = {'model.early_exit_gate.weight': ('early_exit_gate', 'kernel'),
         'model.early_exit_gate.bias': ('early_exit_gate', 'bias')}

_REMOTE = 'ByteDance/Ouro-1.4B'
"""The repository whose remote code an exported config names, as the released ones do."""


def _ouro_window(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """OuroConfig's window, both ways: dropped unless use_sliding_window is set, and layer_types derived
    with Qwen2Config's rule when none are stated (`native_layer_types`)."""
    used.add('use_sliding_window')
    windowed = {**hf_config, 'sliding_window': hf_config.get('sliding_window')
                if hf_config.get('use_sliding_window') else None}
    record['layer_types'] = native_layer_types(windowed, used, 'qwen2')
    if windowed['sliding_window'] is None and 'sliding_attention' in _records.strings(record['layer_types'],
                                                                                      'layer_types'):
        refuse("layer_types with sliding attention",
               "use_sliding_window is not set, so the window has no size")
    return record


def _ouro_loop(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The universal transformer's passes, its decoder reading the last (an early exit at threshold 1.0,
    the last step)."""
    used.update(('total_ut_steps', 'early_exit_threshold', 'early_exit_step'))
    steps = _records.integer(hf_config.get('total_ut_steps', 4), 'total_ut_steps')
    threshold = hf_config.get('early_exit_threshold', 1.0)
    if threshold is not None and _records.number(threshold, 'early_exit_threshold') != 1.0:
        refuse(f'early_exit_threshold {threshold!r}', 'the decoder reads the last pass, which 1.0 exits at')
    exit_step = hf_config.get('early_exit_step')
    if exit_step is not None and _records.integer(exit_step, 'early_exit_step') != steps - 1:
        refuse(f'early_exit_step {exit_step!r}', f'the decoder reads the last pass, step {steps - 1}')
    record['loop'] = {'steps': steps, 'exit_gate': True}
    return record


def _ouro_loop_written(model: CausalTransformer) -> Mapping[str, object]:
    assert model.loop is not None
    return {'total_ut_steps': model.loop.steps, 'early_exit_threshold': 1.0}


OURO_CONFIG = ConfigLayout(
    ConfigPart(_ouro_window, lambda model: {
        'use_sliding_window': 'sliding_attention' in model.per_layer_types}),
    # The release's own OuroRotaryEmbedding builds its table through ROPE_INIT_FUNCTIONS
    # (modeling_ouro.py:445-459 of ByteDance/Ouro-1.4B), as transformers' classes do; no check holds a YaRN
    # record against that remote code yet, so its ramps stay the plain table's and Llama 3.1's.
    decoder_layout(DecoderSpelling(reads=frozenset({'layer_types', 'sliding_window'}),
                                   undeclared=frozenset({'attention_bias'}), ramps=('llama3',))),
    constant('sandwich_norms', value=True), ConfigPart(_ouro_loop, _ouro_loop_written),
    remote_code(_REMOTE, 'ouro', 'Ouro'))
"""Ouro's config, both ways."""


OURO = DecoderFamily(
    ('ouro',), OURO_CONFIG,
    lambda fields: fields.loop is not None, 'ouro', 'OuroForCausalLM',
    routes=spelled(_NAMES, leaves=_GATE),
    preserve_source_layout=False,
    # OuroModel's table pads config.pad_token_id (modeling_ouro.py:496-500), which OuroConfig leaves
    # to the base class's None.
    padding_idx=True,
)
