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

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    DecoderFields,
    KindFields,
    MixtureFields,
    context_lengths,
    record_float,
    record_int,
    refuse,
)
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


def _kinds(text: Mapping[str, object], layers: int) -> tuple[tuple[str, ...], dict[str, KindFields]]:
    """Each layer's kind and the two kinds' fields."""
    local = set(_records.integers(text.get('local_layer_ids', ()), 'local_layer_ids'))
    if not local <= set(range(layers)):
        refuse('local_layer_ids', f"its layers lie in the decoder's {layers}")
    for field, own in (('swa_head_dim', 'head_dim'), ('swa_num_attention_heads', 'num_attention_heads'),
                       ('swa_num_key_value_heads', 'num_key_value_heads')):
        if text.get(field) is not None and record_int(text, field) != record_int(text, own):
            refuse(field, f"its sliding layers attend at the global layers' {own}")
    kernel = record_int(text, 'sconv_kernel_size')
    extent = record_int(text, 'rel_extent')
    relative = {'rank': record_int(text, 'd_rel'), 'extent': extent,
                'clip': record_float(text, 'rel_l2_clip_threshold')}
    clip = native_fields(QKClip)(threshold=record_float(text, 'qk_l2_clip_threshold'))
    floor, alpha = record_float(text, 'log_scaling_n_floor'), record_float(text, 'log_scaling_alpha')

    def attention(*, overall: bool) -> dict[str, object]:
        return {'nope': True, 'qk_clip': clip, 'projected_sinks': True, 'gate_projection': 'own',
                'relative_bias': native_fields(RelativeBias)(**relative, temperature=overall),
                'short_convs': native_fields(ShortConvs)(kernel=kernel, dynamic=overall),
                **({'temperature_tuning': (floor, alpha), 'temperature_form': 'log_ratio'}
                   if overall else {})}
    kinds: dict[str, KindFields] = {
        LOCAL: native_fields(LayerKind)(window=record_int(text, 'sliding_window_size'), mixer=None,
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


def _loop(text: Mapping[str, object], layers: int) -> dict[str, object] | None:
    """`loop_config`'s passes over its core layers, each later pass's sites merging the pass before's
    messages (modeling_inkling2_loop.py:79-96, :142-170, :1320-1393)."""
    loop = text.get('loop_config')
    if loop is None:
        return None
    loop = _records.record(loop, 'loop_config')
    if loop.get('is_varloop'):
        refuse('loop_config.is_varloop', "a pass count chosen at inference is not read yet")
    start, end = record_int(loop, 'core_start'), record_int(loop, 'core_end')
    if not 0 <= start < end <= layers:
        refuse('loop_config', f"its core layers lie in the decoder's {layers}")
    messages = native_fields(CrossPass)(
        block_size=record_int(text, 'n_layers_per_attn_res_block'), rank=record_int(loop, 'gate_rank'),
        additive=False, normalize_values=bool(text.get('attn_res_normalize_values', True)), scaled=True,
        scale_by_input_rms=False, normalize_message=False)
    return {'steps': record_int(loop, 'num_passes'), 'layers': (start, end), 'messages': messages}


def inkling2_text_config(text: Mapping[str, object]) -> DecoderFields:
    """An `inkling2_loop_text` config's `CausalTransformer` fields."""
    if text.get('model_type', 'inkling2_loop_text') != 'inkling2_loop_text':
        refuse(f"model_type {text.get('model_type')!r}", "Inkling 2's text decoder is inkling2_loop_text")
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
            normalize_values=bool(text.get('attn_res_normalize_values', True)), scaled=True))
    loop = _loop(text, layers)
    if loop is not None:
        config['loop'] = loop
    return config
