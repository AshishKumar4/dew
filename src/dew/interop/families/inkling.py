"""Translate Thinking Machines' Inkling 2 (HF `transformers`' `inkling2_loop`,
`Inkling2LoopForConditionalGeneration`).

Its text decoder (`inkling2_loop_text`, modeling_inkling2_loop.py of transformers' main) is computed by
Dew's backbone under these fields, each a general building block:
- no rotary embedding: positions reach attention through an input-conditioned relative bias
  (`RelativeBias`, `d_rel` states clipped at `rel_l2_clip_threshold` against a learned
  `[d_rel, rel_extent]` bank) and, on its global layers, a log-length query temperature
  (`temperature_form='log_ratio'`, `log_scaling_n_floor`, `log_scaling_alpha`);
- its queries clipped per head at `qk_l2_clip_threshold` after a learned scale and its keys unscaled
  (`QKClip`), its values RMS-normed without a gain (`v_norm`), a per-query projected sink
  (`projected_sinks`) and an elementwise sigmoid output gate of its own projection (`gate_projection='own'`);
- four causal short convolutions a layer (`short_convs` on its keys and values, its global layers' with
  per-token dynamic taps, and `LayerKind.mixer_conv`, `feedforward_conv` on its branches' outputs);
- `local_layer_ids` sliding at `sliding_window_size`, the rest global, each global layer's query, key and
  value read from the depth sources by queries of their own (`LayerKind.depth_sites`);
- block attention residuals of `n_layers_per_attn_res_block` layers, their values normalized and their
  scores over the root of the width (`AttentionResiduals.normalize_values`, `scaled`);
- sigmoid top-k routing over `n_routed_experts`, its correction bias selecting alone, weights renormalized
  and scaled by `route_scale`, experts gated by SiTU (`mlp_gate_softcap`, `mlp_up_softcap`);
- `loop_config`'s core layers run `num_passes` passes, each later one restarting at the prelude's output
  and merging each site's message from the pass before through a rank-`gate_rank` controller (`CrossPass`:
  raw messages, no additive term, joined at unit scale, scores over the root of the width);
- an embedding norm on the lookup, an untied head scoring the first `unpadded_vocab_size` of its rows
  (`output_vocab`) and soft-capped at `final_logit_softcapping`.
The vision tower, the multi-token-prediction draft and the variable-loop variant are read by none of
this yet: a checkpoint holding them loads its text decoder alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from dew import _records
from dew.interop.config_records import HF_METADATA, native_fields
from dew.interop.decoder_config import read_only
from dew.interop.decoder_parts import (
    IGNORED_FIELDS,
    SERIALIZED_TEXT_FIELDS,
    DecoderFamily,
    DecoderFields,
    KindFields,
    Leaves,
    MixtureFields,
    context_lengths,
    record_float,
    record_int,
    refuse,
    spelled,
)
from dew.interop.layouts import Renames, Route, Slot, Table
from dew.interop.unread import Stored
from dew.nn.activations import Situ
from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.attention_terms import QKClip, RelativeBias
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.cross_pass import CrossPass
from dew.nn.short_conv_site import ShortConvs

LOCAL, GLOBAL = 'sliding_attention', 'full_attention'
"""Its layer kinds: `local_layer_ids` slide, the rest attend globally."""

_SLIDING_TYPES = {'hybrid_sliding': True, 'hybrid': False}
"""transformers' spelling of a layer's kind (`layer_types`), by whether it slides."""


def _local_layers(text: Mapping[str, object], layers: int) -> set[int]:
    """The layers that slide: the release's `local_layer_ids`, or `layer_types` where transformers'
    config states them, which then agree. Neither stated, transformers' own pattern: each sixth layer
    global (configuration_inkling2_loop.py:145-148)."""
    stated = text.get('local_layer_ids')
    local = None if stated is None else set(_records.integers(stated, 'local_layer_ids'))
    if local is not None and not local <= set(range(layers)):
        refuse('local_layer_ids', f"its layers lie in the decoder's {layers}")
    if text.get('layer_types') is not None:
        types = _records.strings(text['layer_types'], 'layer_types')
        if len(types) != layers or not set(types) <= set(_SLIDING_TYPES):
            refuse(f'layer_types {types!r}', f"one of {sorted(_SLIDING_TYPES)} for each of {layers} layers")
        typed = {layer for layer, kind in enumerate(types) if _SLIDING_TYPES[kind]}
        if local is not None and typed != local:
            refuse(f'layer_types sliding {sorted(typed)}',
                   f"transformers slides these and the release local_layer_ids' {sorted(local)}")
        return typed
    return local if local is not None else {layer for layer in range(layers) if (layer + 1) % 6}


def _kinds(text: Mapping[str, object], layers: int) -> tuple[tuple[str, ...], dict[str, KindFields]]:
    """Each layer's kind and the two kinds' fields."""
    local = _local_layers(text, layers)
    for field, own in (('swa_head_dim', 'head_dim'), ('swa_num_attention_heads', 'num_attention_heads'),
                       ('swa_num_key_value_heads', 'num_key_value_heads')):
        if text.get(field) is not None and record_int(text, field) != record_int(text, own):
            refuse(field, f"its sliding layers attend at the global layers' {own}")
    kernel = record_int(text, 'sconv_kernel_size')
    window = record_int(text, 'sliding_window_size')
    # A sliding layer's relative bank spans its window, a global layer's `rel_extent`
    # (modeling_inkling2_loop.py:554, :647).
    relative = {'rank': record_int(text, 'd_rel'), 'clip': record_float(text, 'rel_l2_clip_threshold')}
    clip = native_fields(QKClip)(threshold=record_float(text, 'qk_l2_clip_threshold'))
    floor, alpha = record_float(text, 'log_scaling_n_floor'), record_float(text, 'log_scaling_alpha')

    def attention(*, overall: bool) -> Mapping[str, object]:
        return {'nope': True, 'qk_clip': clip, 'projected_sinks': True, 'gate_projection': 'own',
                'relative_bias': native_fields(RelativeBias)(
                    **relative, extent=record_int(text, 'rel_extent') if overall else window,
                    temperature=overall),
                'short_convs': native_fields(ShortConvs)(kernel=kernel, dynamic=overall),
                **({'temperature_tuning': (floor, alpha), 'temperature_form': 'log_ratio'}
                   if overall else {})}
    kinds: dict[str, KindFields] = {
        LOCAL: native_fields(LayerKind)(window=window, mixer=None,
                                        mixer_conv=kernel, feedforward_conv=kernel),
        GLOBAL: native_fields(LayerKind)(mixer=None, mixer_conv=kernel, feedforward_conv=kernel,
                                         depth_sites=True)}
    kinds[LOCAL].update(mixer={'class': 'attention', 'fields': attention(overall=False)})
    kinds[GLOBAL].update(mixer={'class': 'attention', 'fields': attention(overall=True)})
    return tuple(LOCAL if layer in local else GLOBAL for layer in range(layers)), kinds


def _mixture(text: Mapping[str, object], layers: int) -> MixtureFields:
    """Every layer's routed experts: sigmoid scores, the correction bias selecting alone, the chosen
    weights renormalized and scaled by `route_scale` (modeling_inkling2_loop.py:1122-1142)."""
    if record_int(text, 'n_shared_experts', 0):
        refuse('n_shared_experts', "the release's experts are routed alone")
    if not text.get('use_gate_bias', True):
        refuse('use_gate_bias', "the release's router selects on its correction bias")
    return native_fields(Mixture)(
        experts=record_int(text, 'n_routed_experts'), top_k=record_int(text, 'num_experts_per_tok'),
        layers=tuple(range(layers)), score_function='sigmoid', bias=True, norm_topk_prob=True,
        scaling=record_float(text, 'route_scale'), expert_features=record_int(text, 'intermediate_size'))


_LOOP_FIELDS = frozenset({'core_start', 'core_end', 'num_passes', 'gate_rank', 'is_varloop'})
"""`loop_config`'s fields, as the release states them."""


def _loop(text: Mapping[str, object], layers: int) -> Mapping[str, object] | None:
    """`loop_config`'s passes over its core layers, each later pass's sites merging the pass before's
    messages (modeling_inkling2_loop.py:79-96, :142-170, :1320-1393)."""
    loop = text.get('loop_config')
    if loop is None:
        return None
    loop = _records.record(loop, 'loop_config')
    extra = sorted(str(key) for key in set(loop) - _LOOP_FIELDS)
    if extra:
        refuse(f'loop_config fields {extra}', "the release's loop states none of them to compute")
    if loop.get('is_varloop'):
        refuse('loop_config.is_varloop', "a pass count chosen at inference is not read yet")
    start, end = record_int(loop, 'core_start'), record_int(loop, 'core_end')
    if not 0 <= start < end <= layers:
        refuse('loop_config', f"its core layers lie in the decoder's {layers}")
    messages = native_fields(CrossPass)(
        block_size=record_int(text, 'n_layers_per_attn_res_block'), rank=record_int(loop, 'gate_rank'),
        additive=False, normalize_values=bool(text.get('attn_res_normalize_values', True)), scaled=True,
        gained=False, scale_by_input_rms=False, normalize_message=False)
    return {'steps': record_int(loop, 'num_passes'), 'layers': (start, end), 'step_norm': False,
            'messages': messages}


_TEXT_FIELDS = frozenset({
    'model_type', 'model_max_length', 'hidden_size', 'num_hidden_layers', 'vocab_size', 'unpadded_vocab_size',
    'num_attention_heads', 'num_key_value_heads', 'head_dim', 'swa_head_dim', 'swa_num_attention_heads',
    'swa_num_key_value_heads', 'd_rel', 'rel_extent', 'q_bias', 'o_bias', 'log_scaling_n_floor',
    'log_scaling_alpha', 'rms_norm_eps', 'use_value_norm_no_gamma', 'qk_l2_clip_threshold',
    'rel_l2_clip_threshold', 'local_layer_ids', 'layer_types', 'sconv_kernel_size', 'final_logit_softcapping',
    'sliding_window_size', 'n_layers_per_attn_res_block', 'attn_res_normalize_values', 'n_routed_experts',
    'num_experts_per_tok', 'n_shared_experts', 'intermediate_size', 'mlp_gate_softcap', 'mlp_up_softcap',
    'route_scale', 'use_gate_bias', 'scoring_func', 'loop_config', 'tie_word_embeddings',
    # Stated by transformers' class and read by none of its modules (modeling_inkling2_loop.py).
    'hidden_act'})
"""The text config's fields as the release states them (its config.json's text_config), each read below or
inert in the reference."""


def _text_config(text: Mapping[str, object]) -> DecoderFields:
    """An `inkling2_loop_text` config's `CausalTransformer` fields, refusing a field the release states none
    of."""
    if text.get('model_type', 'inkling2_loop_text') != 'inkling2_loop_text':
        refuse(f"model_type {text.get('model_type')!r}", "Inkling 2's text decoder is inkling2_loop_text")
    unknown = sorted(str(key) for key in set(text) - _TEXT_FIELDS - SERIALIZED_TEXT_FIELDS - HF_METADATA
                     - IGNORED_FIELDS if not str(key).startswith('_'))
    if unknown:
        refuse(f"text_config fields {unknown}", "Inkling 2's release states none of them to compute")
    if text.get('scoring_func', 'sigmoid') != 'sigmoid':
        refuse(f"scoring_func {text['scoring_func']!r}", "the release's router scores by sigmoid")
    if text.get('tie_word_embeddings'):
        refuse('tie_word_embeddings', "the release's head is its own, untied from the embedding")
    for field in ('q_bias', 'o_bias'):
        if text.get(field):
            refuse(field, "the release's attention projections are bias-free")
    if not text.get('use_value_norm_no_gamma', True):
        refuse('use_value_norm_no_gamma', "the release norms its values without a gain")
    layers = record_int(text, 'num_hidden_layers')
    types, kinds = _kinds(text, layers)
    max_seq_len, _ = context_lengths({'max_position_embeddings': text.get('model_max_length')})
    vocab = record_int(text, 'vocab_size')
    unpadded = record_int(text, 'unpadded_vocab_size', vocab)
    config: DecoderFields = native_fields(CausalTransformer)(
        vocab_size=vocab, output_vocab=None if unpadded == vocab else unpadded,
        emb_features=record_int(text, 'hidden_size'), num_layers=layers,
        num_heads=record_int(text, 'num_attention_heads'),
        num_kv_heads=record_int(text, 'num_key_value_heads'),
        head_dim=record_int(text, 'head_dim'), mlp_features=record_int(text, 'intermediate_size'),
        max_seq_len=max_seq_len, layer_types=types, kinds={},
        norm_eps=record_float(text, 'rms_norm_eps', 1e-6),
        qk_norm=False, v_norm=True, output_gate=True, embedding_norm=True, embedding_norm_reads='lookup',
        tie_embeddings=False, final_logit_softcap=record_float(text, 'final_logit_softcapping'),
        mixture=None, attention_residuals=None, mlp=None)
    config.update(
        kinds=kinds, mixture=_mixture(text, layers),
        mlp=native_fields(Situ)(beta=record_float(text, 'mlp_gate_softcap'),
                                linear_beta=record_float(text, 'mlp_up_softcap')),
        attention_residuals=native_fields(AttentionResiduals)(
            block_size=record_int(text, 'n_layers_per_attn_res_block'),
            normalize_values=bool(text.get('attn_res_normalize_values', True)), scaled=True,
            gained=False))
    loop = _loop(text, layers)
    if loop is not None:
        config['loop'] = loop
    return config


def _inkling2_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read Inkling 2's config, the release's wrapper (`inkling2_loop`) or its text decoder's alone
    (`inkling2_loop_text`), into its text decoder's fields. The wrapper's vision tower and its
    multi-token-prediction draft (`vision_config`, `mtp_config`) fill no field: a load reads the decoder."""
    if hf_config.get('model_type') != 'inkling2_loop':
        used.update(hf_config)
        return _text_config(hf_config)
    if hf_config.get('tie_word_embeddings'):
        refuse('tie_word_embeddings', "the release's head is its own, untied from the embedding")
    used.update(('text_config', 'vision_config', 'mtp_config', 'tie_word_embeddings'))
    return _text_config(_records.record(hf_config.get('text_config'), 'text_config'))


_SITES = {'q': 'attn', 'k': 'k', 'v': 'v', 'mlp': 'mlp'}
"""Each cross-pass site of Dew's (`cross_pass_<site>`) by the release's name of its controller."""

_NAMES: Renames = (
    ('model.llm.unembed', 'lm_head'), ('model.llm.embed', 'model.embed_tokens'),
    ('model.llm.embed_norm', 'model.embedding_layernorm'), ('model.llm', 'model'),
    ('attn.wq_du', 'self_attn.q_proj'), ('attn.wk_dv', 'self_attn.k_proj'),
    ('attn.wv_dv', 'self_attn.v_proj'), ('attn.wo_ud', 'self_attn.o_proj'),
    ('attn.gate_attn', 'self_attn.gate_proj'), ('attn.attn_sink_proj', 'self_attn.sink_proj'),
    ('attn.wr_du', 'self_attn.relative'), ('attn.q_l2_clip.scale', 'self_attn.q_clip_scale'),
    ('attn.rel_l2_clip.scale', 'self_attn.relative_scale'),
    ('attn.rel_logits_proj.proj', 'self_attn.relative_bank'),
    ('attn.k_sconv', 'self_attn.k_conv'), ('attn.v_sconv', 'self_attn.v_conv'),
    # A global layer's query site is its attention's input norm, its key and value sites their own.
    ('attn_norm', 'input_layernorm'), ('q_input_norm', 'input_layernorm'),
    ('k_input_norm', 'input_layernorm_k'), ('v_input_norm', 'input_layernorm_v'),
    ('mlp_norm', 'post_attention_layernorm'), ('attn_sconv', 'attn_conv'), ('mlp_sconv', 'mlp_conv'),
    *((f'controllers.{theirs}', f'cross_pass_{site}') for site, theirs in _SITES.items()),
    ('mlp.gate.bias', 'mlp.gate.e_score_correction_bias'))
"""The release's names (model.safetensors.index.json of thinkingmachines/Inkling-2-Mini at c9d0283, its text
decoder under `model.llm.`) onto the shared decoder's and this family's own (`_LAYER_LEAVES`), as
transformers' conversion reads them (conversion_mapping.py `inkling2_text`, `inkling2_loop_text`)."""

_LAYER_LEAVES: Leaves = {
    'attn_query': ('attention_res', 'kernel'), 'k_query': ('attention_res_k', 'kernel'),
    'v_query': ('attention_res_v', 'kernel'), 'mlp_query': ('mlp_res', 'kernel'),
    'input_layernorm_k.weight': ('input_layernorm_k', 'scale'),
    'input_layernorm_v.weight': ('input_layernorm_v', 'scale'),
    'attn_conv.weight': ('attn_conv', 'conv', 'weight'), 'mlp_conv.weight': ('mlp_conv', 'conv', 'weight'),
    **{f'self_attn.{site}_conv.weight': ('self_attn', f'{site}_conv', 'conv', 'weight') for site in 'kv'},
    **{f'self_attn.{site}_conv.weight_proj': ('self_attn', f'{site}_conv', 'dynamic', 'kernel')
       for site in 'kv'},
    **{f'self_attn.{leaf}': ('self_attn', leaf)
       for leaf in ('q_clip_scale', 'relative_scale', 'relative_bank')},
    'self_attn.relative.weight': ('self_attn', 'relative', 'kernel'),
    'self_attn.sink_proj.weight': ('self_attn', 'sink_proj', 'kernel'),
    **{f'cross_pass_{site}.query': (f'cross_pass_{site}', 'depth', 'kernel') for site in _SITES},
    **{f'cross_pass_{site}.{leaf}': (f'cross_pass_{site}', leaf) for site in _SITES
       for leaf in ('input_down', 'message_down', 'forget', 'gain')},
}
"""The tensors below a layer the shared decoder has no place for, as `_NAMES` spell them: its depth
queries, convolutions, attention terms and cross-pass controllers, those the release stores otherwise as
`_inkling2_prepare` reads them."""

_ROUTES = Table(
    # The vision tower has no counterpart here: its tensors fill no leaf.
    Route('{media}', None, MappingProxyType({'media': Slot(r'model\.visual\..+')})),
    spelled(_NAMES, leaves={'model.output_query': ('output_res', 'kernel')}, layer_leaves=_LAYER_LEAVES))
"""Inkling 2's names, its text decoder's through `_NAMES`, nothing for its vision tower."""


def _query(value: Stored, name: str, width: int) -> Stored:
    """A depth query `[width]` as the `[1, width]` projection its leaf reads (a kernel, `[out, in]`)."""
    if value.shape != (width,):
        raise ValueError(f"{name} holds {value.shape}, not a [{width}] query")
    return value.reshape(1, width)


def _inkling2_prepare(tensors: Mapping[str, Stored],
                      config: Mapping[str, object] | None = None) -> Mapping[str, Stored]:
    """The release's tensors as `_ROUTES` read them, in its own names (modeling_inkling2_loop.py, its
    conversion in conversion_mapping.py and core_model_loading.py at transformers 3a96649):
    - `attn_res.attn_queries.{i}` is layer `i + 1`'s attention query, a global layer's `[3, D]` its query,
      key and value sites'; layer 0 reads none (its attention's input precedes every block), so its leaves,
      which no read reaches, hold transformers' initial zeros; `mlp_queries.{i}` is layer `i`'s, and
      `output_query` the final read's;
    - `middle_cycle_cross_pass_queries` `[2L, D]` holds each loop layer's query and MLP message queries,
      `middle_cycle_cross_pass_kv_queries` its key and value ones; each controller's `down_weight`
      `[rank, 2D]` reads `[input; message]`, and its `forget_weight`, `gain_weight` `[D, rank]` are the
      leaves' transposes;
    - a global layer's dynamic convolution holds its static taps `static_w` `[K, C]` and the rows of its
      taps' projection `weight_proj` `[heads * K, D]` by lag, the first the token itself, where Dew's run in
      convolution order (`ShortConvSite`);
    - each layer's experts are one `w13_weight` `[E, 2I, D]`, gate and up rows interleaved, and one
      `w2_weight` `[E, D, I]`, read as each expert's projections."""
    if config is None:
        raise ValueError("Inkling 2's queries and loop messages are placed by the decoder's geometry")
    width = _records.integer(config['emb_features'], 'emb_features')
    prepared: dict[str, Stored] = {}
    for name, value in tensors.items():
        stem, _, leaf = name.rpartition('.')
        if stem == 'model.llm.attn_res.attn_queries':
            layer = f'model.llm.layers.{int(leaf) + 1}'
            if value.ndim == 2 and value.shape[0] == 3:
                for index, site in enumerate(('attn', 'k', 'v')):
                    prepared[f'{layer}.{site}_query'] = _query(value[index], name, width)
            else:
                prepared[f'{layer}.attn_query'] = _query(value, name, width)
        elif stem == 'model.llm.attn_res.mlp_queries':
            prepared[f'model.llm.layers.{leaf}.mlp_query'] = _query(value, name, width)
        elif name == 'model.llm.attn_res.output_query':
            prepared['model.llm.output_query'] = _query(value, name, width)
        elif name == 'model.llm.middle_cycle_cross_pass_queries':
            prepared.update(_message_queries(value, tensors, config, width))
        elif name == 'model.llm.middle_cycle_cross_pass_kv_queries':
            continue
        elif '.controllers.' in name and leaf == 'down_weight':
            if value.shape[1] != 2 * width:
                raise ValueError(f"{name} holds {value.shape}, not [rank, {2 * width}]")
            prepared[f'{stem}.input_down'] = value[:, :width].T
            prepared[f'{stem}.message_down'] = value[:, width:].T
        elif '.controllers.' in name and leaf in ('forget_weight', 'gain_weight'):
            prepared[f"{stem}.{leaf.removesuffix('_weight')}"] = value.T
        elif name.endswith(('.k_sconv.static_w', '.v_sconv.static_w')):
            prepared[f'{stem}.weight'] = value[::-1].T[:, None, :]
        elif name.endswith(('.k_sconv.weight_proj', '.v_sconv.weight_proj')):
            static = tensors.get(f'{stem}.static_w')
            if static is None:
                raise ValueError(f"{name} projects a dynamic convolution's taps beside its static_w")
            taps = static.shape[0]
            heads = value.shape[0] // taps
            prepared[name] = value.reshape(heads, taps, value.shape[1])[:, ::-1].reshape(heads * taps, -1)
        elif name.endswith('.mlp.experts.w13_weight'):
            for expert in range(value.shape[0]):
                prepared[f'{stem}.{expert}.gate_proj.weight'] = value[expert, 0::2]
                prepared[f'{stem}.{expert}.up_proj.weight'] = value[expert, 1::2]
        elif name.endswith('.mlp.experts.w2_weight'):
            for expert in range(value.shape[0]):
                prepared[f'{stem}.{expert}.down_proj.weight'] = value[expert]
        else:
            prepared[name] = value
    kinds = _records.record(config['kinds'], 'kinds')
    first = _records.record(kinds[_records.strings(config['layer_types'], 'layer_types')[0]], 'kinds')
    for site in ('attn', 'k', 'v') if first.get('depth_sites') else ('attn',):
        prepared[f'model.llm.layers.0.{site}_query'] = np.zeros((1, width), np.float32)
    return prepared


def _message_queries(queries: Stored, tensors: Mapping[str, Stored], config: Mapping[str, object],
                     width: int) -> dict[str, Stored]:
    """Each loop layer's four message queries from the release's two `[2L, D]` stacks, layer-major:
    `[query, mlp]` and `[key, value]` (core_model_loading.py `Inkling2LoopMergeMessageQueries`)."""
    keys = tensors.get('model.llm.middle_cycle_cross_pass_kv_queries')
    loop = config.get('loop')
    if keys is None or not isinstance(loop, Mapping):
        raise ValueError("middle_cycle_cross_pass_queries are a loop's, beside its kv_queries")
    start, end = loop['layers']
    count = 2 * (end - start)
    for name, held in (('queries', queries), ('kv_queries', keys)):
        if held.shape != (count, width):
            raise ValueError(f"middle_cycle_cross_pass_{name} holds {held.shape}, not [{count}, {width}]")
    placed: dict[str, Stored] = {}
    for layer in range(end - start):
        for stack, sites in ((queries, ('attn', 'mlp')), (keys, ('k', 'v'))):
            for offset, site in enumerate(sites):
                row = 2 * layer + offset
                placed[f'model.llm.layers.{start + layer}.controllers.{site}.query'] = stack[row:row + 1]
    return placed


INKLING2 = DecoderFamily(
    ('inkling2_loop', 'inkling2_loop_text'),
    read_only(_inkling2_config, "Inkling 2 is read from its release alone; no model matches it on export"),
    lambda fields: False,
    'inkling2_loop',
    'Inkling2LoopForConditionalGeneration',
    routes=_ROUTES,
    prepare=_inkling2_prepare,
    preserve_source_layout=True,
)
"""Inkling 2 (`inkling2_loop`): the release's text decoder, its wrapper's or alone."""
