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

import math
import re
from collections.abc import Mapping

from dew import records
from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    RopeSpelling,
    base_config,
    read_rope,
    refuse,
)
from dew.nn.backbones.causal_transformer import CausalTransformer

_SPELLING = RopeSpelling(theta=10000.0, per_kind=True, local_theta=10000.0)
"""DoryConfig's rope: one record per attention kind at the class's default base
(configs/dory.py, DoryConfig.default_theta)."""

_SUBLAYERS = ('sliding_attention', 'full_attention')


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


def _rope(hf_config: Mapping[str, object], used: set[str]):
    """The per-kind rope, each entry's partial_rotary_factor read: the released Dory rotates whole heads."""
    parameters = records.record(hf_config.get('rope_parameters') or {}, 'rope_parameters')
    whole = {}
    for kind, entry in parameters.items():
        entry = dict(records.record(entry, f'rope_parameters.{kind}'))
        factor = records.number(entry.pop('partial_rotary_factor', 1.0),
                                f'rope_parameters.{kind} partial_rotary_factor')
        if factor != 1.0:
            refuse(f'rope_parameters.{kind} partial_rotary_factor {factor}', 'Dory rotates whole heads')
        whole[kind] = entry
    return read_rope({**hf_config, 'rope_parameters': whole or None}, used, None, _SPELLING)


def _dory_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Dory config into `CausalTransformer` fields: half its layers as blocks, the loop over
    the recurrent ones and nGPT's gains (dory.py:53-280)."""
    kinds = _pairs(hf_config, used)
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    first = records.integer(hf_config.get('n_input_layers', 0), 'n_input_layers')
    outputs = records.integer(hf_config.get('n_output_layers', 0), 'n_output_layers')
    stated = hf_config.get('n_recurrent_layers')
    recurrent = (layers - first - outputs if stated is None
                 else records.integer(stated, 'n_recurrent_layers'))
    if min(first, recurrent, outputs) < 0 or first + recurrent + outputs != layers:
        refuse(f'n_input_layers {first}, n_recurrent_layers {recurrent}, n_output_layers {outputs}',
               f'they partition the {layers} layers')
    if first % 2 or recurrent % 2:
        refuse('a partition that splits an (attention, mlp) pair', 'a Dew block is the whole pair')
    loops = records.integer(hf_config.get('n_recurrent_loops', 1), 'n_recurrent_loops')
    mode = records.text(hf_config.get('recurrent_kv_cache_mode', 'per_loop'), 'recurrent_kv_cache_mode')
    if mode not in ('per_loop', 'last_loop'):
        refuse(f'recurrent_kv_cache_mode {mode!r}', "Dory caches 'per_loop' or 'last_loop'")
    if loops < 1:
        refuse(f'n_recurrent_loops {loops}', 'a loop runs its block at least once')
    std = records.number(hf_config.get('initializer_range', 0.02), 'initializer_range')
    alpha = records.number(hf_config.get('ngpt_alpha_init', 0.05), 'ngpt_alpha_init')
    hidden = records.integer(hf_config['hidden_size'], 'hidden_size')
    heads = records.integer(hf_config['num_attention_heads'], 'num_attention_heads')
    head_dim = records.integer(hf_config.get('head_dim') or hidden // heads, 'head_dim')
    stated_gate = hf_config.get('attention_output_gate_logit_scale_init')
    gate = math.sqrt(hidden) if stated_gate is None else records.number(
        stated_gate, 'attention_output_gate_logit_scale_init')
    used.update(('n_input_layers', 'n_recurrent_layers', 'n_output_layers', 'n_recurrent_loops',
                 'recurrent_kv_cache_mode', 'initializer_range', 'ngpt_alpha_init',
                 'attention_output_gate_logit_scale_init'))
    if records.boolean(hf_config.get('attention_bias', False), 'attention_bias'):
        refuse('attention_bias', 'Dory projects without biases')
    used.add('attention_bias')
    pairs = {**hf_config, 'num_hidden_layers': layers // 2, 'layer_types': list(kinds)}
    config = base_config(pairs, used, layer_types=kinds, rope=_rope(hf_config, used),
                         reads=frozenset({'layer_types', 'sliding_window'}))
    config.update(
        qk_norm=False, pre_norms=False, output_gate=True,
        mixer={'class': 'attention', 'fields': {'gate_projection': 'own'}},
        ngpt={'step': alpha / std, 'step_init': alpha, 'query': math.sqrt(head_dim) / std,
              'gate': math.sqrt(gate) / std, 'hidden': math.sqrt(hidden), 'logits': 1 / std},
        loop={'steps': loops, 'layers': [first // 2, (first + recurrent) // 2], 'reinject': True,
              'shared_cache': mode == 'last_loop'})
    return config


_ATTENTION = {**{f'{name}.weight': ('self_attn', name, 'kernel')
                 for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')},
              'g_proj.weight': ('self_attn', 'gate_proj', 'kernel'),
              'sqk': ('self_attn', 'sqk'), 's_gate': ('self_attn', 's_gate')}
"""An attention layer's tensors under `mixer.`, by their paths in its block."""
_MLP = {**{f'{name}.weight': ('mlp', name, 'kernel') for name in ('gate_proj', 'up_proj', 'down_proj')},
        'suv_gate': ('mlp', 'suv_gate'), 'suv_up': ('mlp', 'suv_up')}
"""An MLP layer's tensors under `mixer.`, by their paths in its block."""
_TRUNK = {'backbone.embeddings.weight': ('embed_tokens', 'embedding'),
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
    assert model.ngpt is not None and model.loop is not None
    first, end = model.loop.layers or (0, model.num_layers)
    std = 1 / model.ngpt.logits
    layer_types = [kind for attention in model.per_layer_types for kind in (attention, 'mlp')]
    return {
        'num_hidden_layers': 2 * model.num_layers, 'layer_types': layer_types,
        'n_input_layers': 2 * first, 'n_recurrent_layers': 2 * (end - first),
        'n_output_layers': 2 * (model.num_layers - end), 'n_recurrent_loops': model.loop.steps,
        'recurrent_kv_cache_mode': 'last_loop' if model.loop.shared_cache else 'per_loop',
        'initializer_range': std, 'ngpt_alpha_init': model.ngpt.step_init,
        'attention_output_gate_logit_scale_init': (model.ngpt.gate * std) ** 2,
        'attention_bias': False, 'mlp_bias': False,
    }


DORY = DecoderFamily(
    ('dory',), _dory_config, lambda fields: fields.ngpt is not None, 'dory', 'DoryForCausalLM', _dory_export,
    weight_path=_dory_path, export_path=_dory_name, preserve_source_layout=True,
    tied_head_names=('lm_head.weight', 'backbone.embeddings.weight'),
)
