"""Translate DeepSeek-V4.1-Flash (arXiv 2609.19969).

V4.1 keeps V4's attention layer (`dew.nn.deepseek_v4`) under CSA2's
compressor and adds Single-Pass mHC, engram lookups and the DSpark drafter.
There is no transformers class: the release's inference/model.py is the
reference, cited v41:line at the revision tools/deepseek_v41_reference.py
pins.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from dew.interop.families.deepseek import _V4_SCORES
from dew.interop.hf_decoders import (
    DEFAULT_MAX_SEQ_LEN,
    DecoderFields,
    DSparkFields,
    EngramFields,
    KindFields,
    _base_config,
    _dew_path,
    _record_float,
    _record_int,
    _refuse,
    _Ropes,
    _yarn_record,
)
from dew.nn.dspark import DSpark

# DeepSeek-V4.1-Flash (arXiv 2609.19969). There is no transformers class: the
# release's inference/model.py is the reference (cited v41:line at the
# revision tools/deepseek_v41_reference.py pins), and its config.json nests
# the text model's fields under text_config beside a vision tower.
_V41_TEXT_TYPE = 'deepseek_v41_text'
# The text fields the release ships that no computation reads here: storage
# hints and the attention-shape spellings V4.1 fixes (one KV head, no bias).
_V41_TEXT_INERT = frozenset((
    'model_type', 'attention_dropout', 'initializer_range', 'use_cache', 'hidden_act',
    'num_key_value_heads', 'attention_bias', 'topk_method', 'max_position_embeddings'))


def _v41_modes(text: Mapping[str, object], layers: int) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Every layer's compress ratio and CSA2 mode (section 2.3.1, v41:653-661).

    A layer in both source lists is Full, in the index list alone Reindex,
    and in neither Reuse; a KV source outside the index list would attend
    selections made over another layer's entries, which the paper names no
    mode for. A Reuse or Reindex layer reads the latest KV source's entries,
    so that source must pool at its ratio.
    """
    ratios = text.get('compress_ratios')
    if not isinstance(ratios, (list, tuple)) or len(ratios) < layers or any(
            type(rate) is not int or rate < 0 for rate in ratios):
        _refuse('compress_ratios', 'expected a non-negative compress ratio for every layer')
    kv_sources = set(_int_list(text, 'kv_source_layer_ids'))
    index_sources = set(_int_list(text, 'index_source_layer_ids'))
    modes, latest_kv, latest_index = [], None, None
    for layer, rate in enumerate(ratios[:layers]):
        if rate == 0:
            if layer in kv_sources | index_sources:
                _refuse(f"layer {layer} in the source lists", "a sliding layer compresses nothing")
            modes.append('sliding')
            continue
        if layer in kv_sources and layer not in index_sources:
            _refuse(f"kv_source_layer_ids entry {layer}",
                    "a KV source also selects over its entries (Full mode)")
        mode = ('full' if layer in kv_sources else 'reindex' if layer in index_sources else 'reuse')
        if mode != 'full' and (latest_kv is None or ratios[latest_kv] != rate):
            _refuse(f"layer {layer}",
                    f"a {mode} layer reads the latest KV source's entries, which pool at "
                    f"another ratio or do not exist")
        if mode == 'full':
            latest_kv = layer
        if mode != 'reuse':
            latest_index = layer
        elif latest_index is None or latest_kv is None or latest_index < latest_kv:
            _refuse(f"layer {layer}", "a Reuse layer attends the selection made over the "
                    "entries it reads, and none was made since they were")
        modes.append(mode)
    extra = sorted((kv_sources | index_sources) - set(range(layers)))
    if extra:
        _refuse(f"source layers {extra}", f"the model has {layers} layers")
    return tuple(ratios[:layers]), tuple(modes)


def _int_list(record: Mapping[str, object], field: str) -> tuple[int, ...]:
    value = record.get(field, ())
    if not isinstance(value, (list, tuple)) or any(type(entry) is not int for entry in value):
        _refuse(field, 'expected a list of integers')
    return tuple(value)


def _deepseek_v41_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a DeepSeek-V4.1-Flash config into `CausalTransformer` fields.

    The language model is 40 layers of mHC streams under Single-Pass mixing
    around one CSA2 attention and one routed MoE each (section 2). The
    `compress_ratios` and the two source lists give every layer its kind:
    sliding, then a Full kind per ratio whose Reuse layers the model lists
    as `kv_shared_layers`, and a Reindex kind per ratio; the candidate source
    builds the pool the later Reindex layers search. Engram rides the
    `engram_*` fields, and the tower, its aligner and the image-routing bias
    are the vision half, which has no counterpart here: the loader retains
    their tensors by name and the text model reads image placeholders as
    tokens. The DSpark drafter's fields and tensors are the prediction
    depths, read by the drafter alone.
    """
    text = hf_config.get('text_config')
    if not isinstance(text, Mapping) or text.get('model_type', _V41_TEXT_TYPE) != _V41_TEXT_TYPE:
        _refuse('text_config', f"DeepSeek-V4.1 nests its {_V41_TEXT_TYPE} under text_config")
    used.update(('text_config', 'vision_config', 'image_token_id', 'quantization_config'))
    seen: set[str] = set(_V41_TEXT_INERT)
    if text.get('attention_bias') or _record_int(text, 'num_key_value_heads', 1) != 1:
        _refuse('text_config attention shape', 'V4.1 projects one bias-free KV head (v41:643)')
    if text.get('hidden_act', 'silu') != 'silu':
        _refuse(f"hidden_act {text.get('hidden_act')!r}", "V4.1's experts are SwiGLU (v41:848)")
    layers = _record_int(text, 'num_hidden_layers')
    _record_int(text, 'num_attention_heads')
    _record_int(text, 'head_dim')
    _record_int(text, 'hidden_size')
    rope_width = _record_int(text, 'qk_rope_head_dim')
    window = _record_int(text, 'sliding_window')
    ratios, modes = _v41_modes(text, layers)
    seen.update(('num_hidden_layers', 'num_attention_heads', 'head_dim', 'hidden_size',
                 'qk_rope_head_dim', 'sliding_window', 'compress_ratios',
                 'kv_source_layer_ids', 'index_source_layer_ids'))
    main_theta = _record_float(text, 'rope_theta', 10000.0)
    compress_theta = _record_float(text, 'compress_rope_theta', 160000.0)
    seen.update(('rope_theta', 'compress_rope_theta', 'rope_scaling'))
    scaling = text.get('rope_scaling')
    ramp = None
    if scaling is not None:
        if not isinstance(scaling, Mapping) or scaling.get('rope_type', scaling.get('type')) != 'yarn':
            _refuse('rope_scaling', "V4.1's compressed layers rotate under YaRN or plainly (v41:369-389)")
        entry = {key: value for key, value in scaling.items() if key != 'type'}
        entry.update(rope_type='yarn', rope_theta=compress_theta, attention_factor=1.0)
        ramp = _yarn_record(entry, 'rope_scaling', compress_theta,
                            _record_int(text, 'max_position_embeddings', DEFAULT_MAX_SEQ_LEN))
    index = {'index_topk': _record_int(text, 'index_topk'),
             'index_n_heads': _record_int(text, 'index_n_heads'),
             'index_head_dim': _record_int(text, 'index_head_dim')}
    seen.update(index)
    mixer: dict[str, object] = {
        'kind': 'deepseek_v4', 'q_lora_rank': _record_int(text, 'q_lora_rank'),
        'o_groups': _record_int(text, 'o_groups'), 'o_lora_rank': _record_int(text, 'o_lora_rank'),
        'rope_head_dim': rope_width, 'compressor': None, 'compress_rate': None,
        'query_norm': False, 'kv_qat': True}
    seen.update(('q_lora_rank', 'o_groups', 'o_lora_rank'))
    candidate = text.get('candidate_source_layer_id', -1)
    seen.update(('candidate_source_layer_id', 'candidate_topk_blocks', 'candidate_block_size'))
    pool = {}
    if type(candidate) is not int:
        _refuse('candidate_source_layer_id', 'expected a layer index, or -1 for none')
    if candidate >= 0:
        if candidate >= layers or modes[candidate] != 'full':
            _refuse(f"candidate_source_layer_id {candidate}", "the candidate pool is a Full layer's")
        pool = {'candidate_blocks': _record_int(text, 'candidate_topk_blocks'),
                'candidate_block_size': _record_int(text, 'candidate_block_size')}
    kinds: dict[str, KindFields] = {'sliding_attention': {'window': window, 'mixer': mixer}}
    layer_types, shared = [], []
    for layer, (rate, mode) in enumerate(zip(ratios, modes, strict=True)):
        if mode == 'sliding':
            layer_types.append('sliding_attention')
            continue
        name = f'csa2_ratio_{rate}' + ('_reindex' if mode == 'reindex' else '')
        role = None
        if candidate >= 0 and layer > candidate and mode == 'full':
            _refuse(f"layer {layer}", "a Full layer after the candidate source would search "
                    "a pool it builds no part of")
        if mode == 'full' and layer == candidate:
            role = 'source'
        elif mode == 'reindex' and 0 <= candidate < layer:
            role = 'restrict'
        record: KindFields = {'window': window, 'rope_theta': compress_theta, 'yarn': ramp,
                  'mixer': {**mixer, 'compressor': 'csa2', 'compress_rate': rate, **index,
                            'reindex': mode == 'reindex', 'candidates': role,
                            **(pool if role else {})}}
        held = kinds.setdefault(name, record)
        if mode != 'reuse' and held != record:
            _refuse(f"layer {layer}", f"the {name} layers would need different candidate "
                    "roles, which one kind cannot hold")
        layer_types.append(name)
        if mode == 'reuse':
            shared.append(layer)
    moe = _record_int(text, 'moe_intermediate_size')
    base_used: set[str] = set()
    config = _base_config({**text, 'intermediate_size': moe, 'num_key_value_heads': 1,
                           'tie_word_embeddings': text.get('tie_word_embeddings', False)},
                          base_used, layer_types=tuple(layer_types), rope=_Ropes(main_theta))
    seen.update(base_used - {'intermediate_size'})
    scoring = text.get('scoring_func', 'sqrtsoftplus')
    if scoring not in _V4_SCORES:
        _refuse(f"scoring_func {scoring!r}", f"the router scores with one of {sorted(_V4_SCORES)}")
    if _record_int(text, 'n_shared_experts', 1) != 1:
        _refuse('n_shared_experts', 'V4.1 builds one shared expert of the routed width (v41:886-887)')
    norm_topk = text.get('norm_topk_prob', True)
    if not isinstance(norm_topk, bool):
        _refuse('norm_topk_prob', 'expected a boolean')
    seen.update(('moe_intermediate_size', 'scoring_func', 'n_shared_experts', 'norm_topk_prob',
                 'n_routed_experts', 'num_experts_per_tok', 'routed_scaling_factor', 'swiglu_limit',
                 'hc_mult', 'hc_eps', 'hc_sinkhorn_iters'))
    config.update(
        mixer=mixer, kinds=kinds, kv_shared_layers=tuple(shared),
        mixture={'experts': _record_int(text, 'n_routed_experts'),
                 'top_k': _record_int(text, 'num_experts_per_tok'),
                 'layers': tuple(range(layers)), 'score_function': scoring, 'bias': True,
                 'norm_topk_prob': norm_topk,
                 'scaling': _record_float(text, 'routed_scaling_factor', 1.0),
                 'shared_features': moe, 'expert_features': moe},
        swiglu_limit=_record_float(text, 'swiglu_limit', 10.0) or None,
        hyper_connections={'hc_mult': _record_int(text, 'hc_mult', 4),
                           'hc_eps': _record_float(text, 'hc_eps', 1e-6),
                           'hc_sinkhorn_iters': _record_int(text, 'hc_sinkhorn_iters', 20),
                           'head': 'carried'},
        max_seq_len=min(_record_int(text, 'max_position_embeddings', DEFAULT_MAX_SEQ_LEN),
                        DEFAULT_MAX_SEQ_LEN))
    engram = _v41_engram(text, seen)
    if engram is not None:
        config['engram'] = engram
    dspark = _v41_dspark(text, layers, ratios, seen)
    if dspark is not None:
        config.update(dspark=dspark, mtp_layer_type='sliding_attention')
    unknown = sorted(set(text) - seen - {'vocab_size', 'rms_norm_eps', 'tie_word_embeddings'})
    if unknown:
        _refuse(f"text_config fields {unknown}",
                "CausalTransformer has no counterpart, so translating them would silently change the model")
    return config


def _v41_dspark(text: Mapping[str, object], layers: int, ratios, seen: set[str]) -> DSparkFields | None:
    """The DSpark drafter record (section 2.4.3, v41:129-136), or None when the
    config names no stages. Its stages attend a sliding window (v41:1034)."""
    fields = ('num_nextn_predict_layers', 'dspark_block_size', 'dspark_noise_token_id',
              'dspark_target_layer_ids', 'dspark_markov_rank', 'dspark_n_routed_experts',
              'dspark_num_experts_per_tok')
    seen.update(fields)
    stages = _record_int(text, 'num_nextn_predict_layers', 0)
    block = _record_int(text, 'dspark_block_size', 0)
    if not stages or not block:
        return None
    ratio_list = text['compress_ratios']
    assert isinstance(ratio_list, (list, tuple))
    tail = list(ratio_list)[layers:layers + stages]
    if tail != [0] * stages:
        _refuse('compress_ratios', "the DSpark stages are sliding layers, one trailing 0 each")
    return {'stages': stages, 'block_size': block,
            'noise_token_id': _record_int(text, 'dspark_noise_token_id'),
            'target_layers': _int_list(text, 'dspark_target_layer_ids'),
            'markov_rank': _record_int(text, 'dspark_markov_rank'),
            'experts': _record_int(text, 'dspark_n_routed_experts',
                                   _record_int(text, 'n_routed_experts')),
            'top_k': _record_int(text, 'dspark_num_experts_per_tok',
                                 _record_int(text, 'num_experts_per_tok'))}


def _v41_engram(text: Mapping[str, object], seen: set[str]) -> EngramFields | None:
    """The engram record (section 2.4.2, v41:106-115), or None without layers."""
    fields = ('engram_layer_ids', 'engram_num_embeddings', 'engram_max_ngram_size',
              'engram_vocab_size', 'engram_n_heads', 'engram_head_dim',
              'engram_compressed_vocab_size', 'engram_pad_token_id')
    seen.update(fields)
    layers = _int_list(text, 'engram_layer_ids')
    if not layers:
        return None
    return {'layer_ids': layers, 'num_embeddings': _int_list(text, 'engram_num_embeddings'),
            'max_ngram_size': _record_int(text, 'engram_max_ngram_size'),
            'vocab_size': _record_int(text, 'engram_vocab_size'),
            'n_heads': _record_int(text, 'engram_n_heads'),
            'head_dim': _record_int(text, 'engram_head_dim'),
            'compressed_vocab_size': _record_int(text, 'engram_compressed_vocab_size'),
            'pad_token_id': _record_int(text, 'engram_pad_token_id', 2)}


# The release's names inside a layer onto the module names the tree keeps.
# The indexer keeps V3.2's leaf names (wq_b, wk, k_norm, weights_proj),
# which the shared reader already places under self_attn/indexer.
_DEEPSEEK_V41_NAMES = (
    ('.attn_sink', '.sinks'),
    ('.compressor.norm.', '.compressor.kv_norm.'),
    ('.q_norm.', '.q_a_norm.'),
    ('.attn.wq_a.', '.attn.q_a_proj.'),
    ('.attn.wq_b.', '.attn.q_b_proj.'),
    ('.attn.wkv.', '.attn.kv_proj.'),
    ('.compressor.wkv.', '.compressor.kv_proj.'),
    ('.compressor.wgate.', '.compressor.gate_proj.'),
    ('.wo_a.', '.o_a_proj.'),
    ('.wo_b.', '.o_b_proj.'),
    ('.gate.bias', '.gate.e_score_correction_bias'),
    ('.w1.', '.gate_proj.'),
    ('.w2.', '.down_proj.'),
    ('.w3.', '.up_proj.'),
    ('.attn.', '.self_attn.'),
    ('.ffn.', '.mlp.'),
    ('.attn_norm.', '.input_layernorm.'),
    ('.ffn_norm.', '.post_attention_layernorm.'),
    ('.hc_attn_', '.attn_hc.'),
    ('.hc_ffn_', '.ffn_hc.'),
)
_V41_TRUNK = {'embed.weight': 'model.embed_tokens.weight', 'norm.weight': 'model.norm.weight',
              'head.weight': 'lm_head.weight'}
# The vision half: the tower, its aligner, the image span delimiters and the
# routers' image-token bias, none of which the text model computes.
_V41_VISION = ('vision.', 'aligner.', 'image_start', 'image_end', 'image_newline')


def _deepseek_v41_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the decoder's path for one DeepSeek-V4.1 tensor name, or None
    for a tensor the text model retains by name (the vision half and the
    DSpark drafter's `mtp.*`)."""
    if name.startswith(_V41_VISION) or name.endswith('.gate.bias_vl'):
        return None
    if name.startswith('mtp.'):
        parts = name.split('.')
        dspark = config.get('dspark')
        stages = (0 if dspark is None else dspark['stages'] if isinstance(dspark, Mapping)
                  else dspark.stages if isinstance(dspark, DSpark) else 0)
        if len(parts) < 3 or not parts[1].isdigit() or int(parts[1]) >= stages:
            raise ValueError(f"{name} names an undeclared DSpark stage")
        stage, tail = f'dspark_{parts[1]}', '.'.join(parts[2:])
        own_leaves: dict[str, tuple[str, ...]] = {'main_proj.weight': ('main_proj', 'kernel'), 'main_norm.weight': ('main_norm', 'scale'),
               'norm.weight': ('norm', 'scale'), 'markov_head.embed.weight': ('markov_embed',),
               'markov_head.head.weight': ('markov_head',),
               'confidence_head.proj.weight': ('confidence', 'kernel')}
        own = own_leaves.get(tail)
        if own is not None:
            return ('params', stage, *own)
        path = _deepseek_v41_path('layers.0.' + tail, config)
        if path is None:
            return None
        return (path[0], stage, 'block', *path[2:])
    name = _V41_TRUNK.get(name, name)
    parts = name.split('.')
    if len(parts) >= 4 and parts[0] == 'layers' and parts[1].isdigit() and parts[2] == 'engram':
        leaf = '.'.join(parts[3:])
        engram_leaves: dict[str, tuple[str, ...]] = {'embed.weight': ('embed',), 'wkv.weight': ('wkv', 'kernel'),
                  'q_weight': ('q_weight',), 'k_weight': ('k_weight',)}
        engram = engram_leaves.get(leaf)
        if engram is None:
            raise ValueError(f"unknown tensor name {name!r}")
        return ('params', f'layers_{parts[1]}', 'engram', *engram)
    if not (len(parts) >= 3 and parts[0] == 'layers' and parts[1].isdigit()):
        return _dew_path(name, config)
    tail = '.' + '.'.join(parts[2:])
    for theirs, ours in _DEEPSEEK_V41_NAMES:
        tail = tail.replace(theirs, ours)
    return _dew_path(f"model.layers.{parts[1]}{tail}", config)


def engram_token_map(directory, record: Mapping[str, object]) -> np.ndarray:
    """The compressed vocabulary V4.1's engram hashes over, read off the
    tokenizer the checkpoint ships (engram.py:17-55, :136-146).

    Every hash multiplier derives from the compressed vocabulary's size, so a
    tokenizer that compresses to another size than the config states would
    hash every n-gram elsewhere; it is refused. Ids past the tokenizer's
    length, which no text reaches, map to the pad token's compressed id.
    """
    from transformers import AutoTokenizer

    from dew.nn.engram import compressed_token_map

    engram = record['engram']
    assert isinstance(engram, Mapping)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"engram hashes over the tokenizer's compressed vocabulary, and "
                         f"{directory} ships no tokenizer it can read: {error}") from error
    lookup, size = compressed_token_map(tokenizer)
    if size != engram['compressed_vocab_size']:
        _refuse(f"engram_compressed_vocab_size {engram['compressed_vocab_size']}",
                f"the shipped tokenizer compresses to {size} ids, and every hash "
                "multiplier derives from that size")
    vocab = _record_int(record, 'vocab_size')
    if len(lookup) > vocab:
        _refuse('tokenizer', f"it has {len(lookup)} tokens for a vocabulary of {vocab}")
    pad = lookup[_record_int(engram, 'pad_token_id')]
    return np.concatenate([lookup, np.full(vocab - len(lookup), pad, np.int32)])
