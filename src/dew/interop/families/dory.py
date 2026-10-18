"""Translate Dory (`dory`), NVIDIA's looped normalized transformer (Nemotron-Labs Dory1.0).

The reference is vLLM's (vllm-project/vllm#60149, vllm/model_executor/models/dory.py and
vllm/transformers_utils/configs/dory.py at 55076950); no transformers class computes it.

A Dory layer is one sublayer, an attention or an MLP, and the released layout pairs them, an
attention then an MLP, so each pair is one Dew block (`layers_{j // 2}`), the attention's three
sliding to one full. The blocks run under nGPT (`dew.nn.ngpt.NGPT`): no norm has a weight, every
state is a unit vector and each sublayer steps the residual along the sphere at its own learned
rate. The stack's input, recurrent and output layers are a `Loop`'s prelude, block and coda: the
block runs `n_recurrent_loops` times, each pass reading the prelude's output added to the last
pass's state under the unit norm (`Loop.reinject`), and the coda reads the last state.
`recurrent_kv_cache_mode` 'last_loop' keeps one cache entry a recurrent layer, every pass
overwriting the step's keys and values (`Loop.shared_cache`); 'per_loop' keeps one a pass.

A run picks its loop count as vLLM's `--hf-overrides` does: the loaded model's
`with_passes(steps)`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from dew import records
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


def _pairs(hf_config: Mapping[str, object], used: set[str]) -> tuple[str, ...]:
    """The attention kind of each (attention, MLP) pair of layers, which the released layout is."""
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    stated = hf_config.get('layer_types')
    used.add('layer_types')
    types = (tuple(records.strings(stated, 'layer_types')) if stated is not None else tuple(
        'mlp' if index % 2 else 'full_attention' if index % 8 == 6 else 'sliding_attention'
        for index in range(layers)))
    if len(types) != layers:
        refuse(f'layer_types of {len(types)} entries', f'num_hidden_layers is {layers}')
    if layers % 2 or any(types[index] not in _SUBLAYERS or types[index + 1] != 'mlp'
                         for index in range(0, layers, 2)):
        refuse('layer_types other than (attention, mlp) pairs',
               "a Dew block is one attention and one feed-forward, the released Dory's pairs")
    return types[::2]


def _rope(hf_config: Mapping[str, object], kinds: tuple[str, ...], used: set[str]):
    """The per-kind rope and each kind's `partial_rotary_factor`, which vLLM's `get_rope` reads as
    a rope `int(head_dim * factor)` wide over the head's leading dims (rotary_embedding/__init__.py
    at 55076950, lines 66-76), Dew's 'default' partial rotary. Each record is plain rotary, as
    DoryConfig requires (configs/dory.py:114-137), and a stated set names every attention kind."""
    stated = hf_config.get('rope_parameters')
    parameters = records.record(stated or {}, 'rope_parameters')
    missing = sorted(set(kinds) - set(parameters))
    if stated is not None and missing:
        refuse(f'rope_parameters without {missing}', 'Dory reads one rope record per attention kind')
    plain, factors = {}, {}
    for kind, entry in parameters.items():
        entry = dict(records.record(entry, f'rope_parameters.{kind}'))
        factor = records.number(entry.pop('partial_rotary_factor', 1.0),
                                f'rope_parameters.{kind} partial_rotary_factor')
        if not 0.0 < factor <= 1.0:
            refuse(f'rope_parameters.{kind} partial_rotary_factor {factor}',
                   'a fraction of the head in (0, 1]')
        plain[kind], factors[kind] = entry, factor
    ropes = read_rope({**hf_config, 'rope_parameters': plain or None}, used, None, _SPELLING, ramps=())
    return ropes, factors


def _dory_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Dory config into `CausalTransformer` fields: half its layers as blocks, the loop over
    the recurrent ones and nGPT's gains (dory.py:53-280), refusing what DoryConfig refuses
    (configs/dory.py:39-133) and a partition Dew's blocks cannot run."""
    kinds = _pairs(hf_config, used)
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    first = records.integer(hf_config['n_input_layers'], 'n_input_layers')
    outputs = records.integer(hf_config['n_output_layers'], 'n_output_layers')
    given = hf_config['n_recurrent_layers']
    recurrent = layers - first - outputs if given is None else records.integer(given, 'n_recurrent_layers')
    if min(first, recurrent, outputs) < 0 or first + recurrent + outputs != layers:
        refuse(f'n_input_layers {first}, n_recurrent_layers {recurrent}, n_output_layers {outputs}',
               f'they partition the {layers} layers')
    if first % 2 or recurrent % 2 or not recurrent:
        refuse(f'n_input_layers {first} and n_recurrent_layers {recurrent}',
               'a Dew loop runs whole (attention, mlp) pairs, at least one')
    loops = records.integer(hf_config['n_recurrent_loops'], 'n_recurrent_loops')
    mode = records.text(hf_config['recurrent_kv_cache_mode'], 'recurrent_kv_cache_mode')
    if mode not in ('per_loop', 'last_loop'):
        refuse(f'recurrent_kv_cache_mode {mode!r}', "Dory caches 'per_loop' or 'last_loop'")
    if loops < 1:
        refuse(f'n_recurrent_loops {loops}', 'a loop runs its block at least once')
    std = records.number(hf_config['initializer_range'], 'initializer_range')
    if std <= 0:
        refuse(f'initializer_range {std}', "nGPT's gains divide by it")
    alpha = records.number(hf_config['ngpt_alpha_init'], 'ngpt_alpha_init')
    stated_gate = hf_config['attention_output_gate_logit_scale_init']
    gate = (None if stated_gate is None
            else records.number(stated_gate, 'attention_output_gate_logit_scale_init'))
    used.update(('n_input_layers', 'n_recurrent_layers', 'n_output_layers', 'n_recurrent_loops',
                 'recurrent_kv_cache_mode', 'initializer_range', 'ngpt_alpha_init',
                 'attention_output_gate_logit_scale_init'))
    if hf_config['attention_bias'] or hf_config['mlp_bias']:
        refuse('attention_bias or mlp_bias', 'Dory projects without biases')
    if hf_config['hidden_act'] != 'silu' or hf_config['tie_word_embeddings']:
        refuse(f"hidden_act {hf_config['hidden_act']!r} or tied embeddings",
               'Dory requires SiLU and untied embeddings')
    used.update(('attention_bias', 'mlp_bias'))
    pairs = {**hf_config, 'num_hidden_layers': layers // 2, 'layer_types': list(kinds)}
    rope, factors = _rope(hf_config, kinds, used)
    config = base_config(pairs, used, layer_types=kinds, rope=rope,
                         reads=frozenset({'layer_types', 'sliding_window'}))
    for kind, factor in factors.items():
        if factor != 1.0 and kind in kinds:
            kinds_of(config).setdefault(kind, native_fields(LayerKind)())['partial_rotary_factor'] = factor
    config.update(
        partial_rotary_type='default', narrow_rotary_tables=True,
        qk_norm=False, pre_norms=False, output_gate=True,
        mixer={'class': 'attention', 'fields': {'gate_projection': 'own'}},
        ngpt={'std': std, 'alpha': alpha, 'gate_logit': gate},
        loop={'steps': loops, 'layers': [first // 2, (first + recurrent) // 2], 'reinject': True,
              'shared_cache': mode == 'last_loop'})
    return config


_ATTENTION: dict[str, tuple[str, ...]] = {
    **{f'{name}.weight': ('self_attn', name, 'kernel') for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')},
    'g_proj.weight': ('self_attn', 'gate_proj', 'kernel'),
    'sqk': ('self_attn', 'sqk'), 's_gate': ('self_attn', 's_gate')}
"""An attention layer's tensors under `mixer.`, by their paths in its block."""
_MLP: dict[str, tuple[str, ...]] = {
    **{f'{name}.weight': ('mlp', name, 'kernel') for name in ('gate_proj', 'up_proj', 'down_proj')},
    'suv_gate': ('mlp', 'suv_gate'), 'suv_up': ('mlp', 'suv_up')}
"""An MLP layer's tensors under `mixer.`, by their paths in its block."""
_TRUNK: dict[str, tuple[str, ...]] = {
    'backbone.embeddings.weight': ('embed_tokens', 'embedding'),
    'lm_head.weight': ('lm_head', 'kernel'), 'lm_head.logit_scale': ('logit_scale',)}
_LAYER = re.compile(r'backbone\.layers\.(\d+)\.(.+)')


def _dory_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """A Dory tensor's path: layer j's sublayer is block j // 2's attention (even j) or MLP (odd j)."""
    if name in _TRUNK:
        return ('params', *_TRUNK[name])
    match = _LAYER.fullmatch(name)
    if match is None:
        raise ValueError(f'{name!r} is no Dory tensor')
    index, rest = int(match[1]), match[2]
    block = f'layers_{index // 2}'
    if rest == ('mlp_alpha' if index % 2 else 'attn_alpha'):
        return ('params', block, rest)
    table = _MLP if index % 2 else _ATTENTION
    tail = rest.removeprefix('mixer.')
    if not rest.startswith('mixer.') or tail not in table:
        raise ValueError(f'{name!r} is no tensor of a Dory {"mlp" if index % 2 else "attention"} layer')
    return ('params', block, *table[tail])


def _dory_name(dew_name: str, config: Mapping[str, object]) -> str | None:
    """The Dory name of a flat Dew parameter name, `_dory_path`'s inverse."""
    path = tuple(dew_name.split('.'))
    trunk = next((name for name, leaf in _TRUNK.items() if leaf == path), None)
    if trunk is not None:
        return trunk
    block = int(path[0].removeprefix('layers_'))
    if path[1:] in (('attn_alpha',), ('mlp_alpha',)):
        return f'backbone.layers.{2 * block + (path[1] == "mlp_alpha")}.{path[1]}'
    for offset, table in ((0, _ATTENTION), (1, _MLP)):
        tail = next((name for name, leaf in table.items() if leaf == path[1:]), None)
        if tail is not None:
            return f'backbone.layers.{2 * block + offset}.mixer.{tail}'
    raise ValueError(f'{dew_name!r} is no Dory parameter')


def _dory_export(model: CausalTransformer) -> Mapping[str, object]:
    """Dory's config of `model`: its blocks as (attention, mlp) pairs, its loop and nGPT's
    constants, and one rope record per attention kind, which DoryConfig reads (the generic flat
    rope fields cleared). An nGPT stack Dory does not run, one without its reinjecting loop or with
    a tied head, refuses."""
    ngpt, loop = model.ngpt, model.loop
    if ngpt is None or loop is None or not loop.reinject or model.tie_embeddings:
        refuse('an nGPT decoder without a reinjecting loop or with tied embeddings',
               "Dory reinjects its input layers' output into every pass and keeps its own head")
    first, end = loop.layers or (0, model.num_layers)
    layer_types = [kind for attention in model.per_layer_types for kind in (attention, 'mlp')]

    def rope(kind: str) -> Mapping[str, object]:
        resolved = model.kind_of(kind)
        return {'rope_type': 'default', 'rope_theta': resolved.rope_theta,
                'partial_rotary_factor': resolved.partial_rotary_factor or 1.0}

    return {
        'rope_parameters': {kind: rope(kind) for kind in _SUBLAYERS if kind in model.per_layer_types},
        'rope_theta': None, 'rope_local_base_freq': None, 'rope_scaling': None,
        'num_hidden_layers': 2 * model.num_layers, 'layer_types': layer_types,
        'n_input_layers': 2 * first, 'n_recurrent_layers': 2 * (end - first),
        'n_output_layers': 2 * (model.num_layers - end), 'n_recurrent_loops': loop.steps,
        'recurrent_kv_cache_mode': 'last_loop' if loop.shared_cache else 'per_loop',
        'initializer_range': ngpt.std, 'ngpt_alpha_init': ngpt.alpha,
        'attention_output_gate_logit_scale_init': ngpt.gate_logit,
        'attention_bias': False, 'mlp_bias': False,
    }


DORY = DecoderFamily(
    ('dory',), _dory_config, lambda fields: fields.ngpt is not None, 'dory', 'DoryForCausalLM', _dory_export,
    weight_path=_dory_path, export_path=_dory_name, export_weights=decoder_tensors,
    preserve_source_layout=True, defaults=_DEFAULTS,
    tied_head_names=('lm_head.weight', 'backbone.embeddings.weight'),
)
