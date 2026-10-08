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

from dew import records
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.decoder_parts import (
    DEFAULT_MAX_SEQ_LEN,
    NO_AUDIO,
    DecoderFamily,
    DecoderFields,
    KindFields,
    Ropes,
    WrapperFields,
    base_config,
    dew_path,
    kind_mixers,
    record_float,
    record_int,
    refuse,
    yarn_record,
)
from dew.interop.families.deepseek import (
    DSPARK_LEAVES,
    V4_LAYER_NAMES,
    V4_SCORES,
    deepseek_v4_prepare,
    dspark_fields,
    dspark_path,
)
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.deepseek_v4 import DeepseekV4Mixer
from dew.nn.engram import Engram
from dew.nn.hyper_connections import HyperConnections
from dew.nn.vision.deepseek_v41 import (
    translate_deepseek_v41_projector_config,
    translate_deepseek_v41_vision_config,
)

# The release's config.json nests the text model's fields under text_config
# beside a vision tower.
_V41_TEXT_TYPE = 'deepseek_v41_text'
# The text fields the release ships that no computation reads here: storage
# hints and the attention-shape spellings V4.1 fixes (one KV head, no bias).
_V41_TEXT_INERT = frozenset((
    'model_type', 'attention_dropout', 'initializer_range', 'use_cache', 'hidden_act',
    'num_key_value_heads', 'attention_bias', 'topk_method', 'max_position_embeddings'))


type EngramFields = NativeFields[Engram]


def _v41_modes(text: Mapping[str, object], layers: int) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Every compress ratio, the DSpark stages' trailing ones included, and
    every layer's CSA2 mode (section 2.3.1, v41:653-661).

    A layer in both source lists is Full, in the index list alone Reindex,
    and in neither Reuse; a KV source outside the index list would attend
    selections made over another layer's entries, which the paper names no
    mode for. A Reuse or Reindex layer reads the latest KV source's entries,
    so that source must pool at its ratio.
    """
    ratios = text.get('compress_ratios')
    if not isinstance(ratios, (list, tuple)) or len(ratios) < layers or any(
            type(rate) is not int or rate < 0 for rate in ratios):
        refuse('compress_ratios', 'expected a non-negative compress ratio for every layer')
    kv_sources = set(records.integers(text.get('kv_source_layer_ids', ()), 'kv_source_layer_ids'))
    index_sources = set(records.integers(text.get('index_source_layer_ids', ()), 'index_source_layer_ids'))
    modes, latest_kv, latest_index = [], None, None
    for layer, rate in enumerate(ratios[:layers]):
        if rate == 0:
            if layer in kv_sources | index_sources:
                refuse(f"layer {layer} in the source lists", "a sliding layer compresses nothing")
            modes.append('sliding')
            continue
        if layer in kv_sources and layer not in index_sources:
            refuse(f"kv_source_layer_ids entry {layer}",
                    "a KV source also selects over its entries (Full mode)")
        mode = ('full' if layer in kv_sources else 'reindex' if layer in index_sources else 'reuse')
        if mode != 'full' and (latest_kv is None or ratios[latest_kv] != rate):
            refuse(f"layer {layer}",
                    f"a {mode} layer reads the latest KV source's entries, which pool at "
                    f"another ratio or do not exist")
        if mode == 'full':
            latest_kv = layer
        if mode != 'reuse':
            latest_index = layer
        elif latest_index is None or latest_kv is None or latest_index < latest_kv:
            refuse(f"layer {layer}", "a Reuse layer attends the selection made over the "
                    "entries it reads, and none was made since they were")
        modes.append(mode)
    extra = sorted((kv_sources | index_sources) - set(range(layers)))
    if extra:
        refuse(f"source layers {extra}", f"the model has {layers} layers")
    return tuple(ratios), tuple(modes)


def _deepseek_v41_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a text-only DeepSeek-V4.1 config into `CausalTransformer` fields
    (`_v41_decoder`); a bundle with its ViT is the wrapper
    (`_deepseek_v41_wrapper`)."""
    if hf_config.get('vision_config') is not None:
        refuse('vision_config', "a DeepSeek-V4.1 bundle with its ViT is a multimodal wrapper, "
                "which translate_wrapper_config reads")
    return _v41_decoder(hf_config, used, media_bias=False)


def _deepseek_v41_wrapper(hf_config: Mapping[str, object], used: set[str]) -> WrapperFields:
    """Read a DeepSeek-V4.1 bundle: the decoder with its routers' image-span
    bias, the ViT and the aligner (v41:1215-1222)."""
    text = _v41_decoder(hf_config, used, media_bias=True)
    return {
        'model_type': 'deepseek_v41', 'text_model_type': 'deepseek_v41', 'text': text,
        'tower': translate_deepseek_v41_vision_config(hf_config),
        'projector': translate_deepseek_v41_projector_config(
            hf_config, record_int(text, 'emb_features')),
        'image_token_id': record_int(hf_config, 'image_token_id'),
        'tokens_per_image': None, **NO_AUDIO}


def _v41_decoder(hf_config: Mapping[str, object], used: set[str], *, media_bias: bool) -> DecoderFields:
    """Read a DeepSeek-V4.1-Flash text_config into `CausalTransformer` fields.

    The language model is 40 layers of mHC streams under Single-Pass mixing
    around one CSA2 attention and one routed MoE each (section 2). The
    `compress_ratios` and the two source lists give every layer its kind:
    sliding, then a Full kind per ratio whose Reuse layers the model lists
    as `kv_shared_layers`, and a Reindex kind per ratio; the candidate source
    builds the pool the later Reindex layers search. Engram rides the
    `engram_*` fields. The DSpark drafter's fields and tensors are the
    prediction depths, read by the drafter alone. `media_bias` gives every
    router the image-span bias a bundle with its ViT carries (v41:807).
    """
    text = hf_config.get('text_config')
    if not isinstance(text, Mapping) or text.get('model_type', _V41_TEXT_TYPE) != _V41_TEXT_TYPE:
        refuse('text_config', f"DeepSeek-V4.1 nests its {_V41_TEXT_TYPE} under text_config")
    used.update(('text_config', 'vision_config', 'image_token_id', 'quantization_config'))
    seen: set[str] = set(_V41_TEXT_INERT)
    if text.get('attention_bias') or record_int(text, 'num_key_value_heads', 1) != 1:
        refuse('text_config attention shape', 'V4.1 projects one bias-free KV head (v41:643)')
    if text.get('hidden_act', 'silu') != 'silu':
        refuse(f"hidden_act {text.get('hidden_act')!r}", "V4.1's experts are SwiGLU (v41:848)")
    layers = record_int(text, 'num_hidden_layers')
    record_int(text, 'num_attention_heads')
    record_int(text, 'head_dim')
    record_int(text, 'hidden_size')
    rope_width = record_int(text, 'qk_rope_head_dim')
    window = record_int(text, 'sliding_window')
    ratios, modes = _v41_modes(text, layers)
    seen.update(('num_hidden_layers', 'num_attention_heads', 'head_dim', 'hidden_size',
                 'qk_rope_head_dim', 'sliding_window', 'compress_ratios',
                 'kv_source_layer_ids', 'index_source_layer_ids'))
    main_theta = record_float(text, 'rope_theta', 10000.0)
    compress_theta = record_float(text, 'compress_rope_theta', 160000.0)
    seen.update(('rope_theta', 'compress_rope_theta', 'rope_scaling'))
    scaling = text.get('rope_scaling')
    ramp = None
    if scaling is not None:
        if not isinstance(scaling, Mapping) or scaling.get('rope_type', scaling.get('type')) != 'yarn':
            refuse('rope_scaling', "V4.1's compressed layers rotate under YaRN or plainly (v41:369-389)")
        entry = {key: value for key, value in scaling.items() if key != 'type'}
        entry.update(rope_type='yarn', rope_theta=compress_theta, attention_factor=1.0)
        ramp = yarn_record(entry, 'rope_scaling', compress_theta,
                            record_int(text, 'max_position_embeddings', DEFAULT_MAX_SEQ_LEN))
    index = {'index_topk': record_int(text, 'index_topk'),
             'index_n_heads': record_int(text, 'index_n_heads'),
             'index_head_dim': record_int(text, 'index_head_dim')}
    seen.update(index)
    fields: dict[str, object] = {
        'q_lora_rank': record_int(text, 'q_lora_rank'),
        'o_groups': record_int(text, 'o_groups'), 'o_lora_rank': record_int(text, 'o_lora_rank'),
        'rope_head_dim': rope_width, 'compressor': None, 'compress_rate': None,
        'query_norm': False, 'kv_qat': True}
    mixer = {'class': 'deepseek_v4', 'fields': fields}
    seen.update(('q_lora_rank', 'o_groups', 'o_lora_rank'))
    compressed = native_fields(LayerKind)(window=window, rope_theta=compress_theta, yarn=None)
    compressed['yarn'] = ramp
    kinds, layer_types, shared = _v41_kinds(text, layers, ratios[:layers], modes, fields,
                                          compressed, index, seen)
    moe = record_int(text, 'moe_intermediate_size')
    base_used: set[str] = set()
    config = base_config({**text, 'intermediate_size': moe, 'num_key_value_heads': 1,
                           'tie_word_embeddings': text.get('tie_word_embeddings', False)},
                          base_used, layer_types=tuple(layer_types), rope=Ropes(main_theta))
    seen.update(base_used - {'intermediate_size'})
    scoring = text.get('scoring_func', 'sqrtsoftplus')
    if scoring not in V4_SCORES:
        refuse(f"scoring_func {scoring!r}", f"the router scores with one of {sorted(V4_SCORES)}")
    if record_int(text, 'n_shared_experts', 1) != 1:
        refuse('n_shared_experts', 'V4.1 builds one shared expert of the routed width (v41:886-887)')
    norm_topk = text.get('norm_topk_prob', True)
    if not isinstance(norm_topk, bool):
        refuse('norm_topk_prob', 'expected a boolean')
    seen.update(('moe_intermediate_size', 'scoring_func', 'n_shared_experts', 'norm_topk_prob',
                 'n_routed_experts', 'num_experts_per_tok', 'routed_scaling_factor', 'swiglu_limit',
                 'hc_mult', 'hc_eps', 'hc_sinkhorn_iters'))
    config.update(
        mixer=mixer, kinds=kinds, kv_shared_layers=tuple(shared),
        mixture=native_fields(Mixture)(experts=record_int(text, 'n_routed_experts'),
                 top_k=record_int(text, 'num_experts_per_tok'),
                 layers=tuple(range(layers)), score_function=scoring, bias=True,
                 norm_topk_prob=norm_topk,
                 scaling=record_float(text, 'routed_scaling_factor', 1.0),
                 shared_features=moe, expert_features=moe, media_bias=media_bias),
        swiglu_limit=record_float(text, 'swiglu_limit', 0.0) or None,
        hyper_connections=native_fields(HyperConnections)(
            hc_mult=record_int(text, 'hc_mult', 4), hc_eps=record_float(text, 'hc_eps', 1e-6),
            hc_sinkhorn_iters=record_int(text, 'hc_sinkhorn_iters', 20), head='carried', single_pass=True),
        max_seq_len=min(record_int(text, 'max_position_embeddings', DEFAULT_MAX_SEQ_LEN),
                        DEFAULT_MAX_SEQ_LEN))
    engram = _v41_engram(text, seen)
    if engram is not None:
        config['engram'] = engram
    dspark = dspark_fields(text, layers, ratios, seen, reads='input')
    if dspark is not None:
        config['dspark'] = dspark
    unknown = sorted(set(text) - seen - {'vocab_size', 'rms_norm_eps', 'tie_word_embeddings'})
    if unknown:
        refuse(f"text_config fields {unknown}",
                "CausalTransformer has no counterpart, so translating them would silently change the model")
    return config


def _v41_kinds(text: Mapping[str, object], layers: int, ratios, modes, mixer: Mapping[str, object],
               compressed: KindFields, index: Mapping[str, int], seen: set[str]):
    """Every layer's kind (section 2.3.1): the sliding kind, a Full kind per
    ratio, and a Reindex kind per ratio, whose layers after the candidate
    source search the pool it builds. Returns the kinds, each layer's kind
    name, and the Reuse layers, which `kv_shared_layers` lists. `compressed`
    holds the window and rope the compressed kinds share."""
    candidate = text.get('candidate_source_layer_id', -1)
    seen.update(('candidate_source_layer_id', 'candidate_topk_blocks', 'candidate_block_size'))
    pool = {}
    if type(candidate) is not int:
        refuse('candidate_source_layer_id', 'expected a layer index, or -1 for none')
    if candidate >= 0:
        if candidate >= layers or modes[candidate] != 'full':
            refuse(f"candidate_source_layer_id {candidate}", "the candidate pool is a Full layer's")
        pool = {'candidate_blocks': record_int(text, 'candidate_topk_blocks'),
                'candidate_block_size': record_int(text, 'candidate_block_size')}
    sliding = native_fields(LayerKind)(window=compressed.value.window, mixer=None)
    sliding['mixer'] = {'class': 'deepseek_v4', 'fields': mixer}
    kinds: dict[str, KindFields] = {'sliding_attention': sliding}
    layer_types, shared = [], []
    for layer, (rate, mode) in enumerate(zip(ratios, modes, strict=True)):
        if mode == 'sliding':
            layer_types.append('sliding_attention')
            continue
        name = f'csa2_ratio_{rate}' + ('_reindex' if mode == 'reindex' else '')
        role = None
        if candidate >= 0 and layer > candidate and mode == 'full':
            refuse(f"layer {layer}", "a Full layer after the candidate source would search "
                    "a pool it builds no part of")
        if mode == 'full' and layer == candidate:
            role = 'source'
        elif mode == 'reindex' and 0 <= candidate < layer:
            role = 'restrict'
        record: KindFields = NativeFields(LayerKind, {**compressed,
                              'mixer': {'class': 'deepseek_v4', 'fields': {
                                        **mixer, 'compressor': 'csa2', 'compress_rate': rate, **index,
                                        'reindex': mode == 'reindex', 'candidates': role,
                                        **(pool if role else {})}}})
        held = kinds.setdefault(name, record)
        if mode != 'reuse' and held != record:
            refuse(f"layer {layer}", f"the {name} layers would need different candidate "
                    "roles, which one kind cannot hold")
        layer_types.append(name)
        if mode == 'reuse':
            shared.append(layer)
    return kinds, layer_types, shared


def _v41_engram(text: Mapping[str, object], seen: set[str]) -> EngramFields | None:
    """The engram record (section 2.4.2, v41:106-115), or None without layers."""
    fields = ('engram_layer_ids', 'engram_num_embeddings', 'engram_max_ngram_size',
              'engram_vocab_size', 'engram_n_heads', 'engram_head_dim',
              'engram_compressed_vocab_size', 'engram_pad_token_id')
    seen.update(fields)
    layers = records.integers(text.get('engram_layer_ids', ()), 'engram_layer_ids')
    if not layers:
        return None
    embeddings = records.integers(text.get('engram_num_embeddings', ()), 'engram_num_embeddings')
    return native_fields(Engram)(layer_ids=layers, num_embeddings=embeddings,
            max_ngram_size=record_int(text, 'engram_max_ngram_size'),
            vocab_size=record_int(text, 'engram_vocab_size'),
            n_heads=record_int(text, 'engram_n_heads'),
            head_dim=record_int(text, 'engram_head_dim'),
            compressed_vocab_size=record_int(text, 'engram_compressed_vocab_size'),
            pad_token_id=record_int(text, 'engram_pad_token_id', 2))


# The release's own names inside a layer, before those V4 shares
# (`V4_LAYER_NAMES`). The indexer keeps V3.2's leaf names (wq_b, wk,
# k_norm, weights_proj), which the shared reader already places under
# self_attn/indexer.
_DEEPSEEK_V41_NAMES = (
    ('.compressor.norm.', '.compressor.kv_norm.'),
    ('.attn.wq_a.', '.attn.q_a_proj.'), ('.attn.wq_b.', '.attn.q_b_proj.'), ('.attn.wkv.', '.attn.kv_proj.'),
    ('.compressor.wkv.', '.compressor.kv_proj.'), ('.compressor.wgate.', '.compressor.gate_proj.'),
    ('.gate.bias_vl', '.gate.media_bias'),
    *V4_LAYER_NAMES,
)
_V41_TRUNK = {'embed.weight': 'model.embed_tokens.weight', 'norm.weight': 'model.norm.weight',
              'head.weight': 'lm_head.weight'}
# V4.1's Markov tables (v41:1077-1081), beside the leaves both releases name.
_V41_DSPARK_LEAVES: dict[str, tuple[str, ...]] = {
    **DSPARK_LEAVES,
    'markov_head.embed.weight': ('markov_embed',), 'markov_head.head.weight': ('markov_head',)}
_ENGRAM_LEAVES: dict[str, tuple[str, ...]] = {
    'embed.weight': ('embed', 'embedding'), 'wkv.weight': ('wkv', 'kernel'),
    'q_weight': ('q_weight',), 'k_weight': ('k_weight',)}


def _deepseek_v41_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the decoder's path for one DeepSeek-V4.1 tensor name, the
    DSpark drafter's `mtp.*` included; the tied head's copy is None."""
    if name.startswith('mtp.'):
        return dspark_path(name, config, _V41_DSPARK_LEAVES, _deepseek_v41_path)
    name = _V41_TRUNK.get(name, name)
    parts = name.split('.')
    if len(parts) >= 4 and parts[0] == 'layers' and parts[1].isdigit() and parts[2] == 'engram':
        leaf = '.'.join(parts[3:])
        if leaf not in _ENGRAM_LEAVES:
            raise ValueError(f"unknown tensor name {name!r}")
        return ('params', f'layers_{parts[1]}', 'engram', *_ENGRAM_LEAVES[leaf])
    if not (len(parts) >= 3 and parts[0] == 'layers' and parts[1].isdigit()):
        return dew_path(name, config)
    tail = '.' + '.'.join(parts[2:])
    for theirs, ours in _DEEPSEEK_V41_NAMES:
        tail = tail.replace(theirs, ours)
    return dew_path(f"model.layers.{parts[1]}{tail}", config)


def _deepseek_v41_constants(directory, record: Mapping[str, object]) -> Mapping[str, object]:
    """The engram hashes' token map: the compressed vocabulary V4.1's engram
    hashes over, read off the tokenizer the checkpoint ships (engram.py:17-55,
    :136-146); nothing for a config without engram.

    Every hash multiplier derives from the compressed vocabulary's size, so a
    tokenizer that compresses to another size than the config states would
    hash every n-gram elsewhere; it is refused. Ids past the tokenizer's
    length, which no text reaches, map to the pad token's compressed id.
    """
    engram = record.get('engram')
    if not isinstance(engram, Mapping):
        return {}
    from dew.data.text import load_tokenizer
    from dew.nn.engram import compressed_token_map

    try:
        tokenizer = load_tokenizer(str(directory), local_files_only=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"engram hashes over the tokenizer's compressed vocabulary, and "
                         f"{directory} ships no tokenizer it can read: {error}") from error
    lookup, size = compressed_token_map(tokenizer)
    if size != engram['compressed_vocab_size']:
        refuse(f"engram_compressed_vocab_size {engram['compressed_vocab_size']}",
                f"the shipped tokenizer compresses to {size} ids, and every hash "
                "multiplier derives from that size")
    vocab = record_int(record, 'vocab_size')
    if len(lookup) > vocab:
        refuse('tokenizer', f"it has {len(lookup)} tokens for a vocabulary of {vocab}")
    pad = lookup[record_int(engram, 'pad_token_id')]
    return {
        "engram_hashes": {"token_map": np.concatenate([lookup, np.full(vocab - len(lookup), pad, np.int32)])}
    }


# V4.1 is V4's block under CSA2's compressor, which names the family; the
# hub's table registers it.
DEEPSEEK_V41 = DecoderFamily(
    ('deepseek_v41',), _deepseek_v41_config,
    lambda fields: any(isinstance(mixer, DeepseekV4Mixer) and mixer.compressor == 'csa2'
                       for mixer in kind_mixers(fields)),
    'deepseek_v41', 'DeepseekV41ForCausalLM', lambda model: {},
    weight_path=_deepseek_v41_path, prepare=deepseek_v4_prepare,
    preserve_source_layout=True, tied_head_names=('head.weight', 'embed.weight'),
    constants=_deepseek_v41_constants, wrapper=_deepseek_v41_wrapper,
    # The image span's learned vectors sit at the top level, beside the
    # aligner (model.py:1201-1222).
    wrapper_projector_names=('image_start', 'image_newline', 'image_end'))
