"""Translate Dory (`dory`), NVIDIA's looped normalized transformer (Nemotron-Labs Dory1.0).

The reference is vLLM's (vllm-project/vllm#60149, vllm/model_executor/models/dory.py and
vllm/transformers_utils/configs/dory.py at 55076950); no transformers class computes it.

A Dory layer is one sublayer, an attention or an MLP, and each is one Dew block (`layers_j`
for layer j) with no second feed-forward: an attention layer's block runs the attention alone, an
'mlp' layer's an `MLPMixer` in the mixer's slot, as Nemotron-H's separate MLP layers do. The
blocks run under nGPT (`dew.nn.ngpt.NGPT`): no norm has a weight, every state is a unit vector
and each sublayer steps the residual along the sphere at its own learned rate. The stack's input,
recurrent and output layers are a `Loop`'s prelude, block and coda, of any sizes and kinds
DoryConfig takes: the block runs `n_recurrent_loops` times, each pass reading the prelude's output
added to the last pass's state under the unit norm (`Loop.reinject`), and the coda reads the last
state. `recurrent_kv_cache_mode` 'last_loop' keeps one cache entry a recurrent layer, every pass
overwriting the step's keys and values (`Loop.shared_cache`); 'per_loop' keeps one a pass.

A run picks its loop count as vLLM's `--hf-overrides` does: the loaded model's
`with_passes(steps)`.
"""

from __future__ import annotations

from collections.abc import Mapping

from dew import _records
from dew.interop.decoder_config import (
    DecoderSpelling,
    decoder_layout,
    expect,
    inert,
    integer,
    kind_mixer,
    place_ropes,
)
from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    RopeSpelling,
    kinds_of,
    layer_type,
    native_fields,
    read_rope,
    refuse,
    refuse_export,
)
from dew.interop.layouts import Config, ConfigField, ConfigLayout, ConfigPart, Nested, Route, Table, constant
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mixers.mlp import MLPMixer
from dew.nn.ngpt import NGPT
from dew.registry import Configured

_SPELLING = RopeSpelling(theta=10000.0, per_kind=True, local_theta=10000.0)
"""DoryConfig's rope: one record per attention kind at the class's default base
(configs/dory.py, DoryConfig.default_theta)."""

_SUBLAYERS = ('sliding_attention', 'full_attention')

_DEFAULTS = {'vocab_size': 131072, 'hidden_size': 2560, 'num_hidden_layers': 72, 'intermediate_size': 7168,
             'num_attention_heads': 16, 'num_key_value_heads': 4, 'head_dim': 256,
             'max_position_embeddings': 8192, 'tie_word_embeddings': False, 'hidden_act': 'silu',
             'initializer_range': 0.02, 'sliding_window': 1024, 'ngpt_alpha_init': 0.05,
             'attention_output_gate_logit_scale_init': None, 'mlp_bias': False, 'attention_bias': False,
             'n_input_layers': 0, 'n_recurrent_layers': None, 'n_output_layers': 0, 'n_recurrent_loops': 1,
             'recurrent_kv_cache_mode': 'per_loop'}
"""DoryConfig's defaults (configs/dory.py:14-38), which a config's absent keys take
(`DecoderFamily.defaults`): transformers writes a config's non-default keys alone."""


_PATTERN = ('sliding_attention', 'mlp') * 3 + ('full_attention', 'mlp')
"""DoryConfig's default cycle of layer kinds (configs/dory.py:41-49): an MLP after each attention,
the fourth attention full."""


_ATTENTION: dict[str, tuple[str, ...]] = {
    **{f'mixer.{name}.weight': ('self_attn', name, 'kernel')
       for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')},
    'mixer.g_proj.weight': ('self_attn', 'gate_proj', 'kernel'),
    'mixer.sqk': ('self_attn', 'sqk'), 'mixer.s_gate': ('self_attn', 's_gate'), 'attn_alpha': ('attn_alpha',)}
"""An attention layer's tensors, by their paths in its block."""
_MLP: dict[str, tuple[str, ...]] = {
    **{f'mixer.{name}.weight': ('self_attn', name, 'kernel')
       for name in ('gate_proj', 'up_proj', 'down_proj')},
    'mixer.suv_gate': ('self_attn', 'suv_gate'), 'mixer.suv_up': ('self_attn', 'suv_up'),
    'mlp_alpha': ('attn_alpha',)}
"""An MLP layer's tensors, by their paths in its block: the MLP in the mixer's slot, and its step's
rate the slot's own."""
_TRUNK: dict[str, tuple[str, ...]] = {
    'backbone.embeddings.weight': ('embed_tokens', 'embedding'),
    'lm_head.weight': ('lm_head', 'kernel'), 'lm_head.logit_scale': ('logit_scale',)}


def _mlp_layer(config: Config, slots: Mapping[str, str]) -> bool:
    """Whether a route's `layer` is an MLP layer, by the record's kinds."""
    return layer_type(config, slots['layer']) == 'mlp'


def _attention_layer(config: Config, slots: Mapping[str, str]) -> bool:
    return not _mlp_layer(config, slots)


_ROUTES = Table(
    *(Route(name, 'params/' + '/'.join(path)) for name, path in _TRUNK.items()),
    *(Route('backbone.layers.{layer}.' + name, 'params/layers_{layer}/' + '/'.join(path), when=held)
      for table, held in ((_MLP, _mlp_layer), (_ATTENTION, _attention_layer))
      for name, path in table.items()))
"""Dory's names: layer j's tensors in block j, an MLP's or an attention's as its kind says."""


def _layers(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """Each layer's kind, the stated ones or DoryConfig's default pattern (`_PATTERN`), as DoryConfig
    validates them (configs/dory.py:63-74): one a layer, an attention, sliding or full, or an MLP."""
    types = _records.strings(record['layer_types'], 'layer_types')
    layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    if len(types) != layers:
        refuse(f'layer_types of {len(types)} entries', f'num_hidden_layers is {layers}')
    unknown = sorted(set(types) - {*_SUBLAYERS, 'mlp'})
    if unknown:
        refuse(f'layer_types {unknown}', 'a Dory layer is sliding_attention, full_attention or mlp')
    return record


def _rope(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The per-kind rope and each kind's `partial_rotary_factor`, which vLLM's `get_rope` reads as
    a rope `int(head_dim * factor)` wide over the head's leading dims (rotary_embedding/__init__.py
    at 55076950, lines 66-76), Dew's 'default' partial rotary. Each record is plain rotary, as
    DoryConfig requires (configs/dory.py:114-137), and a stated set names every attention kind and
    no other."""
    types = _records.strings(record['layer_types'], 'layer_types')
    attention = tuple(kind for kind in _SUBLAYERS if kind in types)
    stated = hf_config.get('rope_parameters')
    parameters = _records.record(stated or {}, 'rope_parameters')
    missing, unknown = sorted(set(attention) - set(parameters)), sorted(set(parameters) - set(_SUBLAYERS))
    if stated is not None and (missing or unknown):
        refuse(f'rope_parameters without {missing} or with {unknown}',
               'Dory reads one rope record per attention kind, sliding or full')
    plain, factors = {}, {}
    for kind, entry in parameters.items():
        entry = dict(_records.record(entry, f'rope_parameters.{kind}'))
        factor = _records.number(entry.pop('partial_rotary_factor', 1.0),
                                 f'rope_parameters.{kind} partial_rotary_factor')
        if not 0.0 < factor <= 1.0:
            refuse(f'rope_parameters.{kind} partial_rotary_factor {factor}',
                   'a fraction of the head in (0, 1]')
        plain[kind], factors[kind] = entry, factor
    place_ropes(record, read_rope({**hf_config, 'rope_parameters': plain or None}, used, _SPELLING, ramps=()))
    kinds = kinds_of(record)
    for kind, factor in factors.items():
        if factor != 1.0 and kind in attention:
            kinds.setdefault(kind, native_fields(LayerKind)())['partial_rotary_factor'] = factor
    record.update(kinds=kinds, partial_rotary_type='default', narrow_rotary_tables=True)
    return record


def _rope_written(model: CausalTransformer) -> Mapping[str, object]:
    """One rope record per attention kind, which DoryConfig reads."""
    def rope(kind: str) -> Mapping[str, object]:
        resolved = model.kind_of(kind)
        return {'rope_type': 'default', 'rope_theta': resolved.rope_theta,
                'partial_rotary_factor': resolved.partial_rotary_factor or 1.0}
    return {'rope_parameters': {kind: rope(kind) for kind in _SUBLAYERS if kind in model.per_layer_types}}


def _loop(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The loop over the recurrent layers (dory.py:53-280), reinjecting the input layers' output, its
    cache per loop or the last one's, refused where DoryConfig refuses it (configs/dory.py:39-133)."""
    used.update(('n_input_layers', 'n_recurrent_layers', 'n_output_layers', 'n_recurrent_loops',
                 'recurrent_kv_cache_mode'))
    layers = _records.integer(record['num_layers'], 'num_layers')
    first = _records.integer(hf_config['n_input_layers'], 'n_input_layers')
    outputs = _records.integer(hf_config['n_output_layers'], 'n_output_layers')
    given = hf_config['n_recurrent_layers']
    recurrent = layers - first - outputs if given is None else _records.integer(given, 'n_recurrent_layers')
    if min(first, recurrent, outputs) < 0 or first + recurrent + outputs != layers:
        refuse(f'n_input_layers {first}, n_recurrent_layers {recurrent}, n_output_layers {outputs}',
               f'they partition the {layers} layers')
    loops = _records.integer(hf_config['n_recurrent_loops'], 'n_recurrent_loops')
    mode = _records.text(hf_config['recurrent_kv_cache_mode'], 'recurrent_kv_cache_mode')
    if mode not in ('per_loop', 'last_loop'):
        refuse(f'recurrent_kv_cache_mode {mode!r}', "Dory caches 'per_loop' or 'last_loop'")
    if loops < 1:
        refuse(f'n_recurrent_loops {loops}', 'a loop runs its block at least once')
    record['loop'] = {'steps': loops, 'layers': [first, first + recurrent], 'reinject': True,
                      'shared_cache': mode == 'last_loop'}
    return record


def _loop_written(model: CausalTransformer) -> Mapping[str, object]:
    """The loop of `model`'s nGPT stack. One Dory does not run is refused: one without its reinjecting
    loop or with a tied head, or whose block holds a feed-forward beside its mixer, which no one Dory
    layer is."""
    ngpt, loop = model.ngpt, model.loop
    mixers = {kind: model.kind_of(kind).mixer for kind in model.per_layer_types}
    mlp = mixers.pop('mlp', None)
    if (ngpt is None or loop is None or not loop.reinject or model.tie_embeddings or model.mlp_features
            or set(mixers) - set(_SUBLAYERS) or not isinstance(mlp, MLPMixer | None)):
        refuse_export('an nGPT decoder without a reinjecting loop, with tied embeddings, or with a block of '
                      'two sublayers or of another kind', "a Dory layer is an attention or an MLP, the "
                      "recurrent ones reinjecting the input layers' output, under a head of its own")
    first, end = loop.layers or (0, model.num_layers)
    return {'n_input_layers': first, 'n_recurrent_layers': end - first,
            'n_output_layers': model.num_layers - end, 'n_recurrent_loops': loop.steps,
            'recurrent_kv_cache_mode': 'last_loop' if loop.shared_cache else 'per_loop'}


def _deviation(value: Configured, key: str) -> float:
    std = _records.number(value, key)
    if std <= 0:
        refuse(f'{key} {std}', "nGPT's gains divide by it")
    return std


DORY_CONFIG = ConfigLayout(
    # Each MLP layer holds its feed-forward in the mixer's slot (`intermediate_size`), its block none.
    decoder_layout(DecoderSpelling(rope=_SPELLING, reads=frozenset({'layer_types', 'sliding_window'}),
                                   pattern=_PATTERN, activation=None, fixed={'intermediate_size': 0}),
                   rope=ConfigPart(_rope, _rope_written)),
    ConfigPart(_layers, lambda model: {}),
    # DoryConfig states each of these, which the release writes.
    expect('hidden_act', 'Dory requires SiLU', value='silu', written=True),
    inert('tie_word_embeddings', 'Dory requires untied embeddings', neutral=False),
    expect('attention_bias', 'Dory projects without biases', value=False, written=True),
    expect('mlp_bias', 'Dory projects without biases', value=False, written=True),
    kind_mixer('mlp', 'mlp', MLPMixer, ConfigLayout(integer('intermediate_size', 'intermediate_size'),
                                                    constant('activation', value='swiglu'))),
    Nested('ngpt', NGPT, ConfigLayout(
        ConfigField('initializer_range', 'std', parse=_deviation),
        ConfigField('ngpt_alpha_init', 'alpha', parse=_records.number),
        ConfigField('attention_output_gate_logit_scale_init', 'gate_logit',
                    parse=lambda value, key: None if value is None else _records.number(value, key))),
           lambda model: model.ngpt),
    ConfigPart(_loop, _loop_written),
    constant('pre_norms', value=False), constant('output_gate', value=True),
    constant('mixer', value={'class': 'attention', 'fields': {'gate_projection': 'own'}}))
"""Dory's config, both ways: a block a layer, the loop over the recurrent ones and nGPT's gains
(dory.py:53-280), refusing what DoryConfig refuses (configs/dory.py:39-133)."""


DORY = DecoderFamily(
    ('dory',), DORY_CONFIG,
    lambda fields: fields.ngpt is not None, 'dory', 'DoryForCausalLM',
    routes=_ROUTES,
    preserve_source_layout=True, defaults=_DEFAULTS,
    tied_head_names=('lm_head.weight', 'backbone.embeddings.weight'),
)
