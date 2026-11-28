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

import re
import string
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from dew import _records
from dew.interop.config_records import HF_METADATA, NativeFields, native_fields
from dew.interop.decoder_config import read_only
from dew.interop.decoder_parts import (
    IGNORED_FIELDS,
    NO_AUDIO,
    SERIALIZED_TEXT_FIELDS,
    DecoderFamily,
    DecoderFields,
    Inverted,
    KindFields,
    Leaves,
    MixtureFields,
    Packed,
    Restacked,
    Reversed,
    Split,
    Stacked,
    Transposed,
    WrapperFields,
    context_lengths,
    record_float,
    record_int,
    refuse,
    spelled,
)
from dew.interop.layouts import Config, Renames, Route, Slot, Table
from dew.interop.towers.inkling import translate_inkling_projector_config, translate_inkling_vision_config
from dew.nn.activations import Situ
from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.attention_terms import QKClip, RelativeBias
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.block_draft import MarkovBlockDraft
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
        mixture=None, attention_residuals=None)
    # The gated activation's record beside the typed fields, as a native record (Kimi's SiTU is so too).
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
    (`inkling2_loop_text`), into its text decoder's fields, the wrapper's multi-token-prediction draft
    (`mtp_config`) its block drafter (`_draft`). The wrapper's vision tower fills no field of the decoder."""
    if hf_config.get('model_type') != 'inkling2_loop':
        used.update(hf_config)
        return _text_config(hf_config)
    if hf_config.get('tie_word_embeddings'):
        refuse('tie_word_embeddings', "the release's head is its own, untied from the embedding")
    used.update(('text_config', 'vision_config', 'mtp_config', 'tie_word_embeddings'))
    text = _records.record(hf_config.get('text_config'), 'text_config')
    config = _text_config(text)
    if hf_config.get('mtp_config'):
        config['block_draft'] = _draft(_records.record(hf_config['mtp_config'], 'mtp_config'), text)
    return config


_DRAFT_FIELDS = frozenset({
    'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim',
    'intermediate_size', 'max_position_embeddings', 'q_bias', 'o_bias', 'rms_norm_eps', 'rope_theta',
    'local_layer_ids', 'sliding_window_size', 'block_size', 'markov_rank', 'markov_head_type',
    'target_layer_ids',
    # The tensors' dtype and the classes vLLM's config strips (configs.py `Inkling2DSparkConfig.from_target`).
    'torch_dtype', 'dtype', 'architectures'})
"""`mtp_config`'s fields, as the release states them (thinkingmachines/Inkling-2-Mini at c9d0283)."""


def _draft(mtp: Mapping[str, object], text: Mapping[str, object]) -> NativeFields[MarkovBlockDraft]:
    """The release's multi-token-prediction draft (`mtp_config`, its tensors the release's `mtp.safetensors`)
    as a Markov block drafter (`dew.nn.block_draft`), as vLLM reads it (configs.py
    `Inkling2DSparkConfig.from_target`): its layers at the decoder's width, its taps `target_layer_ids`, a
    recurrent Markov head (`rnn_hidden`) and no attention biases, refusing anything else."""
    extra = sorted(str(key) for key in set(mtp) - _DRAFT_FIELDS)
    if extra:
        refuse(f'mtp_config fields {extra}', "the release's draft states none of them to compute")
    if record_int(mtp, 'hidden_size') != record_int(text, 'hidden_size'):
        refuse('mtp_config.hidden_size', "the draft runs at its decoder's width, reading its taps and head")
    if mtp.get('markov_head_type') != 'rnn_hidden':
        refuse(f"mtp_config.markov_head_type {mtp.get('markov_head_type')!r}",
               "the release's draft corrects its logits by a recurrent head (rnn_hidden)")
    for field in ('q_bias', 'o_bias'):
        if mtp.get(field):
            refuse(f'mtp_config.{field}', "the release's draft projects its attention without biases")
    reach, _ = context_lengths({'max_position_embeddings': mtp.get('max_position_embeddings')})
    decoder, _ = context_lengths({'max_position_embeddings': text.get('model_max_length')})
    if reach < decoder:
        refuse('mtp_config.max_position_embeddings', f"the draft places up to {reach} positions, short of "
               f"its decoder's {decoder}")
    window = record_int(mtp, 'sliding_window_size', 0)
    local = _records.integers(mtp.get('local_layer_ids', ()), 'local_layer_ids')
    return native_fields(MarkovBlockDraft)(
        layers=record_int(mtp, 'num_hidden_layers'), num_heads=record_int(mtp, 'num_attention_heads'),
        num_kv_heads=record_int(mtp, 'num_key_value_heads'), head_dim=record_int(mtp, 'head_dim'),
        mlp_features=record_int(mtp, 'intermediate_size'), block_size=record_int(mtp, 'block_size'),
        taps=_records.integers(mtp.get('target_layer_ids', ()), 'target_layer_ids'),
        markov_rank=record_int(mtp, 'markov_rank'), rope_theta=record_float(mtp, 'rope_theta'),
        norm_eps=record_float(mtp, 'rms_norm_eps', 1e-6), local_layers=local,
        window=window if local else None)


_SITES = {'q': 'attn', 'k': 'k', 'v': 'v', 'mlp': 'mlp'}
"""Each cross-pass site of Dew's (`cross_pass_<site>`) by the release's name of its controller and of its
depth query (`<site>_query`)."""

_RESIDUALS = {'q': 'attention_res', 'k': 'attention_res_k', 'v': 'attention_res_v', 'mlp': 'mlp_res'}
"""Each site's depth query in the shared decoder (`_LAYER_LEAVES`), by Dew's name of the site (`_SITES`)."""

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
    **{f'{theirs}_query': (_RESIDUALS[site], 'kernel') for site, theirs in _SITES.items()},
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
`_RESTACKED` reads them."""

DRAFT = 'mtp.'
"""The prefix the draft's tensors (the release's `mtp.safetensors`) are read under
(`DecoderFamily.sidecars`)."""

_DRAFT_LEAVES: Leaves = {
    'fuse_weight': ('fuse_weight',), 'mask_embed': ('mask_embed',), 'norm.weight': ('norm', 'scale'),
    'markov_head.markov_w1.weight': ('markov_head', 'markov_w1'),
    'markov_head.dflare_norm.weight': ('markov_head', 'dflare_norm', 'scale'),
    'markov_head.hidden_out.weight': ('markov_head', 'hidden_out', 'kernel'),
    'markov_head.hidden_out.bias': ('markov_head', 'hidden_out', 'bias'),
    'markov_head.rnn_step.weight': ('markov_head', 'rnn_step', 'kernel'),
    'markov_head.rnn_context.weight': ('markov_head', 'rnn_context', 'kernel'),
    'markov_head.rnn_context.bias': ('markov_head', 'rnn_context', 'bias'),
}
"""The draft's tensors outside its layers, as `_RESTACKED` reads them (its `rnn_proj` split), onto
the block drafter's leaves (`dew.nn.block_draft.BlockDrafter`)."""

_DRAFT_LAYER_LEAVES: Leaves = {
    **{f'{norm}.weight': (norm, 'scale')
       for norm in ('input_layernorm', 'post_attention_layernorm', 'dflare_norm')},
    **{f'self_attn.{proj}.weight': ('self_attn', proj, 'kernel')
       for proj in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'k_proj_ctx', 'v_proj_ctx')},
    **{f'self_attn.{norm}.weight': ('self_attn', norm, 'scale')
       for norm in ('q_norm', 'k_norm', 'k_norm_ctx')},
    **{f'mlp.{proj}.weight': ('mlp', proj, 'kernel') for proj in ('gate_proj', 'up_proj', 'down_proj')},
}
"""A draft layer's tensors, below `layers.{layer}.`."""


def _draft_kept(config: Mapping[str, object], kept: int) -> dict:
    """The release's config as a checkpoint of its first `kept` prediction depths
    (`DecoderFamily.whole_drafter`): its draft, one drafter, kept at any count and none at 0, where neither
    it nor its `mtp.safetensors` is read."""
    return {'mtp_config': None} if kept == 0 and config.get('mtp_config') else {}


def _rnn_widths(config: Mapping[str, object]) -> tuple[int, int]:
    """The Markov head's `rnn_proj` columns: its step's (the state and the token before's embedding, 2r)
    then its context's (a place's states and the carried taps, 2D), as vLLM's head splits them."""
    draft = _records.record(config['block_draft'], 'block_draft')
    rank = _records.integer(draft['markov_rank'], 'markov_rank')
    return 2 * rank, 2 * _records.integer(config['emb_features'], 'emb_features')


_PACKED = (
    Packed('markov_head.rnn_proj.weight', ('markov_head.rnn_step.weight', 'markov_head.rnn_context.weight'),
           -1, widths=_rnn_widths),
    Packed('markov_head.rnn_proj.bias', ('markov_head.rnn_context.bias',), 0),
)
"""The draft's packed tensors (its `mtp.safetensors`), split as a load reads them and packed back as an
export writes them (row 13538): the Markov head's `rnn_proj` `[3r, 2r + 2D]` its step's and its
context's columns, its bias the context's alone."""


def _drafting(config: Config, slots: Mapping[str, str]) -> bool:
    """Whether the record carries the block drafter the draft's tensors fill."""
    return isinstance(config.get('block_draft'), Mapping)


_ROUTES = Table(
    # The vision tower has no counterpart here: its tensors fill no leaf.
    Route('{media}', None, MappingProxyType({'media': Slot(r'model\.visual\..+')})),
    *(Route(DRAFT + name, 'params/block_drafter/' + '/'.join(path), when=_drafting)
      for name, path in _DRAFT_LEAVES.items()),
    *(Route(DRAFT + 'layers.{layer}.' + name, 'params/block_drafter/layers_{layer}/' + '/'.join(path),
            when=_drafting)
      for name, path in _DRAFT_LAYER_LEAVES.items()),
    spelled(_NAMES, leaves={'model.output_query': ('output_res', 'kernel')}, layer_leaves=_LAYER_LEAVES))
"""Inkling 2's names, its text decoder's through `_NAMES`, its draft's under `DRAFT`, nothing for its vision
tower."""


def _dynamic_taps(record: Mapping[str, object]) -> int | None:
    """The taps of a kind's dynamic short convolutions (`ShortConvs.dynamic`), wherever its record nests
    them, None where it holds none."""
    convs = record.get('short_convs')
    fields = convs.get('fields', convs) if isinstance(convs, Mapping) else None
    if fields is not None and _records.record(fields, 'short_convs').get('dynamic'):
        return _records.integer(fields['kernel'], 'short_convs.kernel')
    return next((taps for value in record.values()
                 if isinstance(value, Mapping) and (taps := _dynamic_taps(value)) is not None), None)


_STACK_NAMES = MappingProxyType({
    'output_query': 'model.llm.attn_res.output_query',
    'output': 'model.llm.output_query',
    'attn_queries': 'model.llm.attn_res.attn_queries.{index}',
    'mlp_queries': 'model.llm.attn_res.mlp_queries.{layer}',
    'query': 'model.llm.layers.{layer}.{site}_query',
    'static_taps': 'model.llm.layers.{layer}.attn.{conv}_sconv.static_w',
    'taps': 'model.llm.layers.{layer}.attn.{conv}_sconv.weight',
    'projection': 'model.llm.layers.{layer}.attn.{conv}_sconv.weight_proj',
    'bank': 'model.llm.layers.{layer}.mlp.experts.{bank}_weight',
    'expert': 'model.llm.layers.{layer}.mlp.experts.{expert}.{part}.weight',
    'controller': 'model.llm.layers.{layer}.controllers.{site}.{leaf}',
    'messages': 'model.llm.middle_cycle_cross_pass_{stack}',
})
"""The names of the release's stacked tensors and of their parts (`_stacks`, `_unstored`), each a template of
the fields `_FIELDS` declares: the one spelling `_stacks` names them by and `_RESERVED` recognizes them by."""

_FIELDS = MappingProxyType({
    'index': r'\d+', 'layer': r'\d+', 'expert': r'\d+', 'site': tuple(_SITES.values()), 'conv': ('k', 'v'),
    'bank': ('w13', 'w2'), 'part': ('gate_proj', 'up_proj', 'down_proj'),
    'leaf': ('down_weight', 'forget_weight', 'gain_weight', 'input_down', 'message_down', 'forget', 'gain',
             'query'),
    'stack': ('queries', 'kv_queries'),
})
"""Each field of `_STACK_NAMES`' templates: a number, or the words it takes."""


def _named(name: str, **fields: str | int) -> str:
    """The tensor name `_STACK_NAMES[name]` spells with `fields`, each one `_FIELDS` declares."""
    for field, value in fields.items():
        domain = _FIELDS[field]
        if not (str(value).isdigit() if isinstance(domain, str) else value in domain):
            raise ValueError(f"{field}={value!r} is no {name} field: {domain}")
    return _STACK_NAMES[name].format(**fields)


def _pattern(template: str) -> str:
    """`template`'s names as a regular expression, each field its declared domain (`_FIELDS`)."""
    pieces = []
    for literal, field, _, _ in string.Formatter().parse(template):
        pieces.append(re.escape(literal))
        if field is not None:
            domain = _FIELDS[field]
            pieces.append(domain if isinstance(domain, str) else f"(?:{'|'.join(map(re.escape, domain))})")
    return ''.join(pieces)


def _query(name: str, parts: tuple[str, ...], width: int) -> Stacked:
    """A depth query `[D]`, or a global layer's sites' `[3, D]`, as the `[1, D]` projection each leaf
    reads (a kernel, `[out, in]`), `D` the decoder's `width`."""
    if len(parts) == 1:
        return Stacked(name, parts, (Split(0),), grid=(), shape=(1, width))
    return Stacked(name, parts, (Split(1),), shape=(1, width))


def _sites(kind: Mapping[str, object]) -> tuple[str, ...]:
    """A layer's attention query sites: a global layer's query, key and value, a local one's one."""
    return tuple(_SITES[site] for site in (('q', 'k', 'v') if kind.get('depth_sites') else ('q',)))


def _stacks(config: Mapping[str, object]) -> list[Stacked]:
    """The release's tensors the decoder's record (`config`) reads otherwise than stored, each as its
    `Stacked` view of those `_ROUTES` read, in its own names (modeling_inkling2_loop.py, its conversion in
    conversion_mapping.py and core_model_loading.py at transformers 3a96649):
    - `attn_res.attn_queries.{i}` is layer `i + 1`'s attention query, a global layer's `[3, D]` its query,
      key and value sites'; layer 0 reads none (its attention's input precedes every block, `_unstored`);
      `mlp_queries.{i}` is layer `i`'s, and `output_query` the final read's;
    - `middle_cycle_cross_pass_queries` `[2L, D]` holds each loop layer's query and MLP message queries,
      layer-major, `middle_cycle_cross_pass_kv_queries` its key and value ones
      (`Inkling2LoopMergeMessageQueries`); each controller's `down_weight` `[rank, 2D]` reads
      `[input; message]`, and its `forget_weight`, `gain_weight` `[D, rank]` are the leaves' transposes;
    - a global layer's dynamic convolution holds its static taps `static_w` `[K, C]`, the `[C, 1, K]`
      `weight` a static one stores reversed, and the rows of its taps' projection `weight_proj`
      `[heads * K, D]` by lag, the first the token itself, where Dew's run in convolution order
      (`ShortConvSite`);
    - each routed layer's experts are one `w13_weight` `[E, 2I, D]`, gate and up rows interleaved, and one
      `w2_weight` `[E, D, I]`, each expert's projections stacked."""
    width = _records.integer(config['emb_features'], 'emb_features')
    kinds = _records.record(config['kinds'], 'kinds')
    types = _records.strings(config['layer_types'], 'layer_types')
    stated = config.get('mixture')
    mixture = None if stated is None else _records.record(stated, 'mixture')
    routed = () if mixture is None else _records.integers(mixture['layers'], 'layers')
    experts = 0 if mixture is None else _records.integer(mixture['experts'], 'experts')
    stacks = [_query(_named('output_query'), (_named('output'),), width)]
    for layer, kind in enumerate(_records.record(kinds[kind], 'kinds') for kind in types):
        if layer:
            sites = tuple(_named('query', layer=layer, site=site) for site in _sites(kind))
            stacks.append(_query(_named('attn_queries', index=layer - 1), sites, width))
        stacks.append(_query(_named('mlp_queries', layer=layer),
                             (_named('query', layer=layer, site=_SITES['mlp']),), width))
        if (taps := _dynamic_taps(kind)) is not None:
            for conv in ('k', 'v'):
                projection = _named('projection', layer=layer, conv=conv)
                stacks += [Stacked(_named('static_taps', layer=layer, conv=conv),
                                   (_named('taps', layer=layer, conv=conv),),
                                   (Reversed(0), Transposed((1, 0)), Split(1)), grid=()),
                           Stacked(projection, (projection,),
                                   (Split(0, taps), Reversed(1), Inverted(Split(0, taps))), grid=())]
        if layer in routed:
            stacks += [Stacked(_named('bank', layer=layer, bank='w13'),
                               tuple(_named('expert', layer=layer, expert=expert, part=part)
                                     for expert in range(experts) for part in ('gate_proj', 'up_proj')),
                               (Split(1, 2), Transposed((0, 2, 1, 3))), grid=(experts, 2)),
                       Stacked(_named('bank', layer=layer, bank='w2'),
                               tuple(_named('expert', layer=layer, expert=expert, part='down_proj')
                                     for expert in range(experts)))]
    loop = config.get('loop')
    if not isinstance(loop, Mapping):
        return stacks
    span = range(*_records.integers(loop['layers'], 'loop.layers'))
    for layer in span:
        for site in _SITES.values():
            def leaf(name: str, layer=layer, site=site) -> str:
                return _named('controller', layer=layer, site=site, leaf=name)

            stacks += [Stacked(leaf('down_weight'), (leaf('input_down'), leaf('message_down')),
                               (Transposed((1, 0)), Split(0, width))),
                       *(Stacked(leaf(f'{held}_weight'), (leaf(held),), (Transposed((1, 0)),), grid=())
                         for held in ('forget', 'gain'))]
    if loop.get('messages') is not None:
        stacks += [Stacked(_named('messages', stack=stack),
                           tuple(_named('controller', layer=layer, site=site, leaf='query')
                                 for layer in span for site in sites), (Split(1),), shape=(1, width))
                   for stack, sites in (('queries', (_SITES['q'], _SITES['mlp'])),
                                        ('kv_queries', (_SITES['k'], _SITES['v'])))]
    return stacks


def _unstored(config: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Layer 0's attention queries, which the release stores none of: no read reaches them (its attention's
    input precedes every block), so their leaves hold transformers' initial zeros."""
    width = _records.integer(config['emb_features'], 'emb_features')
    kinds = _records.record(config['kinds'], 'kinds')
    first = _records.record(kinds[_records.strings(config['layer_types'], 'layer_types')[0]], 'kinds')
    return {_named('query', layer=0, site=site): np.zeros((1, width), np.float32) for site in _sites(first)}


_RESERVED = re.compile('|'.join(_pattern(template) for name, template in _STACK_NAMES.items()
                               if name != 'taps'))
"""The names only `_stacks` and their parts, or `_unstored`, take, whatever the decoder's record, read off
their one spelling (`_STACK_NAMES`): all but a static convolution's taps, which the release stores as they
are where its convolution is not dynamic."""

_RESTACKED = Restacked(_stacks, _unstored, _RESERVED)
"""Inkling 2's release as `_ROUTES` read it and back, one declaration both ways."""


def _inkling2_wrapper(hf_config: Mapping[str, object], used: set[str]) -> WrapperFields:
    """Read Inkling 2's release whole (`inkling2_loop`, Inkling2LoopForConditionalGeneration at transformers
    3a96649): its text decoder (`model.llm.`), its vision trunk (`model.visual.backbone.`) and merger
    (`model.visual.proj_in`, `proj_out`). Each image's merged patches fill its `image_token_id`
    placeholders in order, each token one position on (Inkling2LoopModel.forward, its positions a plain
    count). The multi-token-prediction draft (`mtp_config`, the release's `mtp.safetensors`) is the text
    decoder's block drafter (`_draft`), which a bundle's media do not draft through."""
    text = _inkling2_config(hf_config, used)
    width = record_int(text, 'emb_features')
    vision = bool(hf_config.get('vision_config'))
    used.update(('image_token_id', 'vision_config'))
    return {'model_type': 'inkling2_loop', 'text_model_type': 'inkling2_loop', 'text': text,
            'tower': translate_inkling_vision_config(hf_config) if vision else None,
            'projector': translate_inkling_projector_config(hf_config, width) if vision else None,
            'image_token_id': (_records.integer(hf_config.get('image_token_id', IMAGE_TOKEN),
                                                'image_token_id') if vision else None),
            'tokens_per_image': None, 'video_token_id': None, 'extra_placeholder_ids': (),
            'text_unprefixed': ('llm.',), **NO_AUDIO}


IMAGE_TOKEN = 200054
"""The id of an image's placeholders, Inkling2LoopConfig's default (`<|unused_200054|>`)."""


INKLING2 = DecoderFamily(
    ('inkling2_loop', 'inkling2_loop_text'),
    read_only(_inkling2_config, "Inkling 2 is read from its release alone; no model matches it on export"),
    lambda fields: False,
    'inkling2_loop',
    'Inkling2LoopForConditionalGeneration',
    routes=_ROUTES,
    prepare=_RESTACKED.prepare,
    unprepare=_RESTACKED.unprepare,
    holds_extensions=True,
    packed=_PACKED,
    sidecars=lambda config: {'mtp.safetensors': DRAFT} if config.get('mtp_config') else {},
    whole_drafter=_draft_kept,
    preserve_source_layout=True,
    wrapper=_inkling2_wrapper, wrapper_placeholders=('image_token_id',),
)
"""Inkling 2 (`inkling2_loop`): the release's text decoder, its wrapper's or alone, and the wrapper's
vision tower and merger (`_inkling2_wrapper`)."""
