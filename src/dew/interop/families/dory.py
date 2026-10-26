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

import re
from collections.abc import Mapping

from dew import _records
from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    RopeSpelling,
    base_config,
    decoder_tensors,
    kinds_of,
    native_fields,
    read_rope,
    refuse,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mixers.mlp import MLPMixer

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


def _layer_types(hf_config: Mapping[str, object], used: set[str]) -> tuple[str, ...]:
    """Each layer's kind, the stated ones or DoryConfig's default pattern (configs/dory.py:41-49):
    an attention, sliding or full, or an MLP, as DoryConfig validates them (configs/dory.py:63-74)."""
    layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    stated = hf_config.get('layer_types')
    used.add('layer_types')
    types = (tuple(_records.strings(stated, 'layer_types')) if stated is not None else tuple(
        'mlp' if index % 2 else 'full_attention' if index % 8 == 6 else 'sliding_attention'
        for index in range(layers)))
    if len(types) != layers:
        refuse(f'layer_types of {len(types)} entries', f'num_hidden_layers is {layers}')
    unknown = sorted(set(types) - {*_SUBLAYERS, 'mlp'})
    if unknown:
        refuse(f'layer_types {unknown}', 'a Dory layer is sliding_attention, full_attention or mlp')
    return types


def _rope(hf_config: Mapping[str, object], kinds: tuple[str, ...], used: set[str]):
    """The per-kind rope and each kind's `partial_rotary_factor`, which vLLM's `get_rope` reads as
    a rope `int(head_dim * factor)` wide over the head's leading dims (rotary_embedding/__init__.py
    at 55076950, lines 66-76), Dew's 'default' partial rotary. Each record is plain rotary, as
    DoryConfig requires (configs/dory.py:114-137), and a stated set names every attention kind and
    no other."""
    stated = hf_config.get('rope_parameters')
    parameters = _records.record(stated or {}, 'rope_parameters')
    missing, unknown = sorted(set(kinds) - set(parameters)), sorted(set(parameters) - set(_SUBLAYERS))
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
    ropes = read_rope({**hf_config, 'rope_parameters': plain or None}, used, _SPELLING, ramps=())
    return ropes, factors


def _dory_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Dory config into `CausalTransformer` fields: a block a layer, the loop over the
    recurrent ones and nGPT's gains (dory.py:53-280), refusing what DoryConfig refuses
    (configs/dory.py:39-133)."""
    types = _layer_types(hf_config, used)
    layers = len(types)
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
    std = _records.number(hf_config['initializer_range'], 'initializer_range')
    if std <= 0:
        refuse(f'initializer_range {std}', "nGPT's gains divide by it")
    alpha = _records.number(hf_config['ngpt_alpha_init'], 'ngpt_alpha_init')
    stated_gate = hf_config['attention_output_gate_logit_scale_init']
    gate = (None if stated_gate is None
            else _records.number(stated_gate, 'attention_output_gate_logit_scale_init'))
    used.update(('n_input_layers', 'n_recurrent_layers', 'n_output_layers', 'n_recurrent_loops',
                 'recurrent_kv_cache_mode', 'initializer_range', 'ngpt_alpha_init',
                 'attention_output_gate_logit_scale_init'))
    if hf_config['attention_bias'] or hf_config['mlp_bias']:
        refuse('attention_bias or mlp_bias', 'Dory projects without biases')
    if hf_config['hidden_act'] != 'silu' or hf_config['tie_word_embeddings']:
        refuse(f"hidden_act {hf_config['hidden_act']!r} or tied embeddings",
               'Dory requires SiLU and untied embeddings')
    used.update(('attention_bias', 'mlp_bias'))
    attention = tuple(kind for kind in _SUBLAYERS if kind in types)
    rope, factors = _rope(hf_config, attention, used)
    config = base_config({**hf_config, 'layer_types': list(types)}, used, rope=rope,
                         reads=frozenset({'layer_types', 'sliding_window'}))
    kinds = kinds_of(config)
    for kind, factor in factors.items():
        if factor != 1.0 and kind in attention:
            kinds.setdefault(kind, native_fields(LayerKind)())['partial_rotary_factor'] = factor
    if 'mlp' in types:
        kinds['mlp'] = native_fields(LayerKind)(mixer=None)
        kinds['mlp']['mixer'] = {'class': 'mlp', 'fields': {'intermediate_size': config['mlp_features'],
                                                            'activation': 'swiglu'}}
    config.update(
        mlp_features=0, partial_rotary_type='default', narrow_rotary_tables=True,
        qk_norm=False, pre_norms=False, output_gate=True,
        mixer={'class': 'attention', 'fields': {'gate_projection': 'own'}},
        ngpt={'std': std, 'alpha': alpha, 'gate_logit': gate},
        loop={'steps': loops, 'layers': [first, first + recurrent], 'reinject': True,
              'shared_cache': mode == 'last_loop'})
    return config


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
_LAYER = re.compile(r'backbone\.layers\.(\d+)\.(.+)')


def _sublayer(index: int, config: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    """Layer `index`'s tensors by their paths: an MLP's or an attention's, as its kind says."""
    kind = _records.strings(config['layer_types'], 'layer_types')[index]
    return _MLP if kind == 'mlp' else _ATTENTION


def _dory_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """A Dory tensor's path: layer j's in block j."""
    if name in _TRUNK:
        return ('params', *_TRUNK[name])
    match = _LAYER.fullmatch(name)
    if match is None:
        raise ValueError(f'{name!r} is no Dory tensor')
    index, tail = int(match[1]), match[2]
    table = _sublayer(index, config)
    if tail not in table:
        raise ValueError(f'{name!r} is no tensor of a Dory {"mlp" if table is _MLP else "attention"} layer')
    return ('params', f'layers_{index}', *table[tail])


def _dory_name(dew_name: str, config: Mapping[str, object]) -> str | None:
    """The Dory name of a flat Dew parameter name, `_dory_path`'s inverse."""
    path = tuple(dew_name.split('.'))
    trunk = next((name for name, leaf in _TRUNK.items() if leaf == path), None)
    if trunk is not None:
        return trunk
    index = int(path[0].removeprefix('layers_'))
    tail = next((name for name, leaf in _sublayer(index, config).items() if leaf == path[1:]), None)
    if tail is None:
        raise ValueError(f'{dew_name!r} is no Dory parameter')
    return f'backbone.layers.{index}.{tail}'


def _dory_export(model: CausalTransformer) -> Mapping[str, object]:
    """Dory's config of `model`: its blocks as layers, its loop and nGPT's constants, and one rope
    record per attention kind, which DoryConfig reads (the generic flat rope fields cleared). An
    nGPT stack Dory does not run is refused: one without its reinjecting loop or with a tied head,
    or whose block holds a feed-forward beside its mixer, which no one Dory layer is."""
    ngpt, loop = model.ngpt, model.loop
    mixers = {kind: model.kind_of(kind).mixer for kind in model.per_layer_types}
    mlp = mixers.pop('mlp', None)
    if (ngpt is None or loop is None or not loop.reinject or model.tie_embeddings or model.mlp_features
            or set(mixers) - set(_SUBLAYERS) or not isinstance(mlp, MLPMixer | None)):
        refuse('an nGPT decoder without a reinjecting loop, with tied embeddings, or with a block of '
               'two sublayers or of another kind', "a Dory layer is an attention or an MLP, the "
               "recurrent ones reinjecting the input layers' output, under a head of its own")
    first, end = loop.layers or (0, model.num_layers)

    def rope(kind: str) -> Mapping[str, object]:
        resolved = model.kind_of(kind)
        return {'rope_type': 'default', 'rope_theta': resolved.rope_theta,
                'partial_rotary_factor': resolved.partial_rotary_factor or 1.0}

    return {
        'rope_parameters': {kind: rope(kind) for kind in _SUBLAYERS if kind in model.per_layer_types},
        'rope_theta': None, 'rope_local_base_freq': None, 'rope_scaling': None,
        'num_hidden_layers': model.num_layers, 'layer_types': list(model.per_layer_types),
        'n_input_layers': first, 'n_recurrent_layers': end - first,
        'n_output_layers': model.num_layers - end, 'n_recurrent_loops': loop.steps,
        'recurrent_kv_cache_mode': 'last_loop' if loop.shared_cache else 'per_loop',
        'initializer_range': ngpt.std, 'ngpt_alpha_init': ngpt.alpha,
        'attention_output_gate_logit_scale_init': ngpt.gate_logit,
        'attention_bias': False, 'mlp_bias': False,
        **({} if mlp is None else {'intermediate_size': mlp.intermediate_size}),
    }


DORY = DecoderFamily(
    ('dory',), _dory_config, lambda fields: fields.ngpt is not None, 'dory', 'DoryForCausalLM', _dory_export,
    weight_path=_dory_path, export_path=_dory_name, export_weights=decoder_tensors,
    preserve_source_layout=True, defaults=_DEFAULTS,
    tied_head_names=('lm_head.weight', 'backbone.embeddings.weight'),
)
