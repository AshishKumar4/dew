"""Translate the DeepSeek decoders: V2, V3, V3.2, V4 and the Kimi releases.

V2 and V3 differ in how the MoE sizes itself and whether the router carries a
bias. V3.2 adds the sparse indexer over the same MLA block. V4 is its own
layout: three attention kinds, each naming its compressor and its pooling
rate, mHC's residual streams, and hash-routed first layers. Kimi K2 is V3's
computation under its own provenance, and Kimi K2.5 the same decoder nested
in a vision wrapper.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from dew import records
from dew.interop.hf_decoders import (
    DEFAULT_MAX_SEQ_LEN,
    DecoderFields,
    HyperConnectionsFields,
    KindFields,
    MixtureFields,
    Ramp,
    _base_config,
    _dew_path,
    _record_float,
    _record_int,
    _refuse,
    _rope_theta,
    _Ropes,
    _yarn_record,
    translate_config,
)


def _deepseek_rope(hf_config: Mapping[str, object], used: set
                   ) -> tuple[float, Ramp | None]:
    """Return (rope_theta, yarn record) from either rope spelling.

    Both released DeepSeek configs spell it with `rope_scaling` of
    `type: yarn`; transformers prefers `rope_scaling` when both are
    present (convert_rope_params_to_dict), so this does too. Plain rope
    reuses the shared reader; anything but plain or YaRN changes the
    frequencies and refuses with the entry named.
    """
    used.update(('rope_theta', 'rope_parameters', 'rope_scaling'))
    scaling = hf_config.get('rope_scaling')
    parameters = hf_config.get('rope_parameters')
    entry = (scaling if isinstance(scaling, Mapping)
             else parameters if isinstance(parameters, Mapping) else None)
    theta = records.number(hf_config.get('rope_theta', 10000.0), 'rope_theta')
    max_pos = records.integer(hf_config.get('max_position_embeddings',
                                DEFAULT_MAX_SEQ_LEN), 'max_position_embeddings')
    if entry is None:
        return theta, None
    rope_type = entry.get('rope_type', entry.get('type', 'default'))
    field = ('rope_scaling' if scaling is entry else 'rope_parameters')
    if rope_type in ('default', 'none'):
        plain = _rope_theta(dict(entry, rope_theta=entry.get(
            'rope_theta', theta)), field)
        return plain or theta, None
    if rope_type == 'yarn':
        entry_theta = float(entry.get('rope_theta', theta))
        return entry_theta, _yarn_record(
            dict(entry, rope_theta=entry_theta), field, entry_theta, max_pos)
    _refuse(f"rope scaling (rope_type {rope_type!r})",
            "the mixer applies plain or YaRN rotary positions")
    raise AssertionError("unreachable")


def _deepseek_mixture(hf_config: Mapping[str, object], layers: int,
                      used: set) -> MixtureFields:
    """Read a DeepSeek V3 MoE config into the mixture record.

    The reference selects on the biased sigmoid scores inside the best
    groups, each scored by its two best experts, and renormalises.
    `norm_topk_prob: false` would leave the weights unnormalised, which the
    V3 reference never does, so it refuses naming the field.
    """
    used.update(('norm_topk_prob', 'topk_method', 'scoring_func'))
    scoring = hf_config.get('scoring_func', 'sigmoid')
    if scoring != 'sigmoid':
        _refuse(f"scoring_func {scoring!r}",
                "dew's router scores softmax, sigmoid or sqrtsoftplus, and "
                "this family's reference scores sigmoid")
    if hf_config.get('norm_topk_prob', True) is not True:
        _refuse("norm_topk_prob=False",
                "DeepseekV3TopkRouter always renormalises the top-k weights")
    method = hf_config.get('topk_method')
    if method is not None and method != 'noaux_tc':
        _refuse(f"topk_method {method!r}",
                "the reference selects with the bias and the group limit, "
                "which is what noaux_tc names")
    return {
        **_deepseek_layout(hf_config, layers, used),
        'score_function': 'sigmoid',
        'groups': records.integer(hf_config.get('n_group') or 1, 'n_group'),
        'groups_per_token': records.integer(hf_config.get('topk_group') or 1, 'topk_group'),
        'bias': True,
    }


def _deepseek_v2_mixture(hf_config: Mapping[str, object], layers: int,
                         used: set) -> MixtureFields:
    """Read a DeepSeek V2 MoE config into the mixture record.

    `DeepseekV2TopkRouter` softmaxes the logits, selects greedily or inside
    the best groups scored by their best expert, and never renormalises. It
    reads no `norm_topk_prob`. The released `false` translates and a `true`
    refuses.
    """
    used.update(('norm_topk_prob', 'topk_method', 'scoring_func'))
    scoring = hf_config.get('scoring_func', 'softmax')
    if scoring != 'softmax':
        _refuse(f"scoring_func {scoring!r}", "DeepseekV2TopkRouter softmaxes its logits")
    if hf_config.get('norm_topk_prob', False):
        _refuse("norm_topk_prob=True",
                "DeepseekV2TopkRouter never renormalises the top-k weights")
    method = hf_config.get('topk_method', 'greedy')
    if method not in ('greedy', 'group_limited_greedy'):
        _refuse(f"topk_method {method!r}",
                "DeepseekV2TopkRouter selects greedy or group_limited_greedy")
    groups = records.integer(hf_config.get('n_group') or 1, 'n_group')
    per_token = records.integer(hf_config.get('topk_group') or 1, 'topk_group')
    if method == 'greedy' and (groups, per_token) != (1, 1):
        _refuse(f"n_group {groups} with topk_method 'greedy'",
                "the greedy selection ignores the groups")
    return {
        **_deepseek_layout(hf_config, layers, used),
        'score_function': 'softmax',
        'norm_topk_prob': False,
        'groups': groups,
        'groups_per_token': per_token,
        'group_score': 'max',
    }


def _deepseek_layout(hf_config: Mapping[str, object], layers: int,
                     used: set, *, sparse_layers: tuple[int, ...] | None = None) -> MixtureFields:
    """Read the expert counts, widths and sparse layers every DeepSeek MoE shares.

    The first `first_k_dense_replace` layers stay dense and the rest route.
    Transformers builds every layer past the dense ones as MoE whatever
    `moe_layer_freq` says, so anything but that refuses. `aux_loss_alpha`
    and `seq_aux` shape the training loss alone and no forward pass reads
    them; a run sets them on LMObjective, whose `aux_loss_alpha` and
    `seq_aux` compute V2's balance loss. A family with an explicit per-layer
    schedule supplies sparse_layers and reuses only the expert geometry.
    """
    used.update(('n_routed_experts', 'num_local_experts',
                 'num_experts_per_tok', 'routed_scaling_factor',
                 'n_group', 'topk_group',
                 'n_shared_experts', 'moe_intermediate_size',
                 'first_k_dense_replace', 'moe_layer_freq', 'mlp_layer_types',
                 'aux_loss_alpha', 'seq_aux'))
    experts = hf_config.get('n_routed_experts',
                            hf_config.get('num_local_experts'))
    if experts is None:
        _refuse("n_routed_experts",
                "a DeepSeek MoE layer needs its expert count")
    sparse = sparse_layers
    if sparse is None:
        freq = hf_config.get('moe_layer_freq')
        if freq is not None and freq != 1:
            _refuse(f"moe_layer_freq {freq!r}",
                    "transformers builds every layer past the dense ones as MoE, "
                    "whatever this field says")
        first_k = records.integer(hf_config.get('first_k_dense_replace', 0) or 0, 'first_k_dense_replace')
        if not 0 <= first_k <= layers:
            _refuse(f"first_k_dense_replace {first_k!r}",
                    f"it names dense layers of a {layers}-layer model")
        sparse = tuple(range(first_k, layers))
        pattern = hf_config.get('mlp_layer_types')
        if pattern is not None:
            expected = (['dense'] * first_k
                        + ['sparse'] * (layers - first_k))
            if list(records.strings(pattern, 'mlp_layer_types')) != expected:
                _refuse(f"mlp_layer_types {list(records.strings(pattern, 'mlp_layer_types'))!r}",
                        "it disagrees with first_k_dense_replace, which is what "
                        "the reference builds")
    shared = records.integer(hf_config.get('n_shared_experts', 0) or 0, 'n_shared_experts')
    shared_features = 0
    if shared:
        width = hf_config.get('moe_intermediate_size')
        if width is None:
            _refuse("moe_intermediate_size",
                    "the shared experts need their width")
        shared_features = shared * records.integer(width, 'moe_intermediate_size')
    return {
        'experts': records.integer(experts, 'n_routed_experts'),
        'top_k': records.integer(hf_config['num_experts_per_tok'], 'num_experts_per_tok'),
        'layers': sparse,
        'scaling': records.number(hf_config.get('routed_scaling_factor', 1.0), 'routed_scaling_factor'),
        'shared_features': shared_features,
        'expert_features': records.integer(hf_config['moe_intermediate_size'], 'moe_intermediate_size'),
    }


def _deepseek_config(hf_config: Mapping[str, object], used: set[str], *,
                     sparse: bool = False,
                     mixture: Callable[[Mapping[str, object], int, set], MixtureFields]
                     = _deepseek_mixture) -> DecoderFields:
    """Read a DeepSeek V2, V3 or V3.2 config into `CausalTransformer` fields.

    `sparse` picks the V3.2 block, whose every layer is
    deepseek_sparse_attention and whose `index` holds the lightning indexer's
    geometry. `mixture` is the family's own MoE reader. The function builds
    three things from the config: `layer_types`, one entry per layer; the MLA
    `mixer` record; and the routed `mixture`.
    """
    rope_theta, yarn = _deepseek_rope(hf_config, used)
    config = _base_config(hf_config, used, rope=_Ropes(rope_theta))
    layer_types = records.strings(config.get('layer_types'), 'layer_types')
    model_type = hf_config['model_type']
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    sparse_name = ('deepseek_sparse_attention'
                   if sparse else 'full_attention')
    if hf_config.get('layer_types') is None:
        # Neither released config names its pattern. V3 is dense MLA
        # throughout and V3.2 sparse attention throughout.
        layer_types = (sparse_name,) * layers
        config['layer_types'] = layer_types
    for entry in layer_types:
        if entry != sparse_name:
            _refuse(f"layer_types entry {entry!r}",
                    f"a {model_type} model mixes no attention kinds: "
                    f"every layer is {sparse_name}")
    nope = records.integer(hf_config['qk_nope_head_dim'], 'qk_nope_head_dim')
    rope = records.integer(hf_config['qk_rope_head_dim'], 'qk_rope_head_dim')
    head_dim = hf_config.get('head_dim')
    if head_dim is not None and records.integer(head_dim, 'head_dim') != rope:
        _refuse(f"head_dim {head_dim!r}",
                "DeepSeek points head_dim at the rope slice, "
                f"which is {rope} wide here")
    derived = hf_config.get('qk_head_dim')
    if derived is not None and records.integer(derived, 'qk_head_dim') != nope + rope:
        _refuse(f"qk_head_dim {derived!r}",
                f"it derives as qk_nope_head_dim + qk_rope_head_dim, "
                f"which is {nope + rope} here")
    v_dim = hf_config.get('v_head_dim')
    if v_dim is None:
        _refuse("v_head_dim",
                "the values need their width, and no default keeps a "
                "checkpoint's layout")
    kv_rank = hf_config.get('kv_lora_rank')
    if kv_rank is None:
        _refuse("kv_lora_rank",
                "the latent needs its width, and no default keeps a "
                "checkpoint's layout")
    interleave = hf_config.get('rope_interleave', True)
    if sparse and interleave is not True:
        _refuse(f"rope_interleave {interleave!r}",
                "the V3.2 reference always rotates interleaved pairs; a "
                "flag saying otherwise describes no released model")
    index: dict[str, int] | None = None
    if sparse:
        index = {
            'index_topk': records.integer(hf_config['index_topk'], 'index_topk'),
            'index_n_heads': records.integer(hf_config['index_n_heads'], 'index_n_heads'),
            'index_head_dim': records.integer(hf_config['index_head_dim'], 'index_head_dim'),
        }
        used.update(('index_topk', 'index_n_heads', 'index_head_dim'))
    # The released checkpoints ship no mtp.* weights (91991 tensors on
    # DeepSeek-V3 and 92425 on V3.2-Exp, none of them MTP) and
    # transformers builds no MTP module, so the field describes nothing
    # the weights hold and the base model is what loads. Weight
    # translation raises on mtp.* tensors, so a checkpoint that ships them
    # fails at load.
    # The fp8 scales name the stored dtype. dew loads the dequantized
    # weights, and the reader names an unreadable dtype where it meets one.
    # ep_size is a runtime parallel hint.
    used.update(('num_nextn_predict_layers', 'num_mtp_layers'))
    used.update(('quantization_config', 'ep_size'))
    used.update(('qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim',
                 'kv_lora_rank', 'q_lora_rank', 'qk_head_dim',
                 'rope_interleave'))
    config.update(
        head_dim=nope + rope,
        mixer={
            'kind': 'mla',
            'q_lora_rank': (None if hf_config.get('q_lora_rank') is None
                            else records.integer(hf_config['q_lora_rank'], 'q_lora_rank')),
            'kv_lora_rank': records.integer(kv_rank, 'kv_lora_rank'),
            'qk_nope_head_dim': nope,
            'qk_rope_head_dim': rope,
            'v_head_dim': records.integer(v_dim, 'v_head_dim'),
            'rope_interleave': bool(interleave),
            'yarn': yarn,
            'index_topk': None if index is None else index['index_topk'],
            'index_n_heads': None if index is None else index['index_n_heads'],
            'index_head_dim': (None if index is None
                               else index['index_head_dim']),
        },
        mixture=mixture(hf_config, layers, used),
    )
    return config


# The text half of a Kimi K2.5 wrapper, by the model_type its text_config
# names. Kimi_K25Config.__post_init__ rewrites a `kimi_k2` text config, or
# a missing one, into `deepseek_v3` (configuration_kimi_k25.py:80-92), so
# these two spellings name one computation and any other names a decoder
# this wrapper does not carry.
_KIMI_K25_TEXT = ('kimi_k2', 'deepseek_v3')

# moonshotai/Kimi-K2.5's text_config was serialized by transformers 4.56.2,
# which wrote every PreTrainedConfig attribute. 5.16.1's DeepseekV3Config
# reads none of these off a decoder config: the first group is decoding
# policy, which no forward pass consults, and the second is metadata.
_KIMI_K25_TEXT_SERIALIZED = frozenset({
    'bad_words_ids', 'begin_suppress_tokens', 'decoder_start_token_id',
    'diversity_penalty', 'do_sample', 'early_stopping',
    'encoder_no_repeat_ngram_size', 'exponential_decay_length_penalty',
    'forced_bos_token_id', 'forced_eos_token_id', 'length_penalty',
    'max_length', 'min_length', 'no_repeat_ngram_size', 'num_beam_groups',
    'num_beams', 'num_return_sequences', 'output_scores',
    'remove_invalid_values', 'repetition_penalty', 'return_dict_in_generate',
    'sep_token_id', 'suppress_tokens', 'temperature', 'top_k', 'top_p',
    'typical_p',
    'finetuning_task', 'is_decoder', 'prefix', 'task_specific_params',
    'tf_legacy_loss', 'tokenizer_class', 'torchscript', 'use_bfloat16',
})
# The four the same serialization carries that would name another model if
# they were set, so they are read by value rather than accepted by name.
_KIMI_K25_TEXT_ENCODER = ('add_cross_attention', 'cross_attention_hidden_size',
                          'tie_encoder_decoder', 'pruned_heads')


def _kimi_k25_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Kimi K2.5 config into `CausalTransformer` fields.

    moonshotai/Kimi-K2.5 is a vision wrapper whose decoder is Kimi K2's.

    Kimi_K25Model is a vision tower, a language model built from
    text_config and a projector (modeling_kimi_k25.py:590-592), and
    Kimi_K25ForConditionalGeneration puts the head on top
    (:727, :734). The text computation is DeepSeek V3's verbatim, so this
    translates the text half and accounts for the wrapper's own fields by
    name: the tower, the projector and the media placeholders they fill
    have no counterpart here, and nothing they name reaches the decoder.

    The head is the wrapper's, not the nested text model's: the text
    config's own `tie_word_embeddings` describes a DeepseekV3Model with no
    head at all, while the wrapper's ties `lm_head.weight` to
    `model.language_model.embed_tokens.weight` (:727).

    Without pixels the reference looks up image/video placeholders as token
    zero (:686-690). Only the embedding lookup changes; targets retain the
    original vocabulary ids.
    """
    text = hf_config.get('text_config')
    if not isinstance(text, Mapping):
        _refuse('text_config',
                f"the wrapper carries its decoder under text_config, got {text!r}")
    model_type = text.get('model_type', 'deepseek_v3')
    if model_type not in _KIMI_K25_TEXT:
        _refuse(f"text_config model_type {model_type!r}",
                "a Kimi K2.5 wrapper's decoder is Kimi K2's, which the "
                f"reference reads as one of {', '.join(map(repr, _KIMI_K25_TEXT))}")
    for key in _KIMI_K25_TEXT_ENCODER:
        if text.get(key):
            _refuse(f"text_config {key}={text[key]!r}",
                    "the decoder has no cross attention, no encoder to tie "
                    "against and no pruned heads")
    tied = hf_config.get('tie_word_embeddings', True)
    if not isinstance(tied, bool):
        _refuse(f"tie_word_embeddings {tied!r}", "the wrapper head takes a boolean tying policy")
    # The release ships the media placeholder under its remote-code name;
    # transformers' own default for image_token_id is that same 163605, and
    # its video_token_id default sits at the top of the vocabulary, where no
    # id can reach it (configuration_kimi_k25.py:74-77).
    used.update(('text_config', 'tie_word_embeddings', 'vision_config',
                 'projection_hidden_size', 'projection_layer_norm_eps',
                 'image_token_id', 'media_placeholder_token_id', 'video_token_id',
                 'vision_start_token_id', 'vision_end_token_id',
                 'use_unified_vision_chunk', 'video_placeholder', 'ignore_index'))
    nested = {key: value for key, value in text.items()
              if key not in _KIMI_K25_TEXT_SERIALIZED and key not in _KIMI_K25_TEXT_ENCODER}
    config = translate_config({**nested, 'model_type': model_type, 'tie_word_embeddings': tied})
    placeholders = []
    for key, default in (('image_token_id', 163605), ('video_token_id', 163840)):
        value = hf_config.get(key, default)
        if type(value) is not int or value < 0:
            _refuse(key, 'the text-only wrapper requires a nonnegative token id')
        placeholders.append(value)
    config['embedding_zero_ids'] = tuple(placeholders)
    return config


def _kimi_k25_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the text decoder's path for one Kimi K2.5 tensor name.

    The release nests the decoder under `language_model.model.*` with its
    head at `language_model.lm_head.weight`
    (model.safetensors.index.json of moonshotai/Kimi-K2.5 at 4d01dfe0);
    transformers renames those to `model.language_model.*` and
    `lm_head.weight` on the way in (conversion_mapping.py:432-433). The
    tower and the projector have no counterpart here, so their tensors map
    to nothing and the export writes their source bytes back.
    """
    decoder = 'language_model.model.'
    if name.startswith(decoder):
        return _dew_path('model.' + name[len(decoder):], config)
    if name == 'language_model.lm_head.weight':
        return _dew_path('lm_head.weight', config)
    if name.startswith(('vision_tower.', 'mm_projector.')):
        return None
    raise ValueError(f"unknown tensor name {name!r}")


# DeepSeek V4's three attention kinds, onto the compressor each one runs
# (COMPRESSOR_CLASSES, modeling_deepseek_v4.py:788-790): the sliding kind
# runs none, and the two compressed kinds name theirs.
_V4_KINDS = {'sliding_attention': None,
             'compressed_sparse_attention': 'csa',
             'heavily_compressed_attention': 'hca'}
# The legacy per-layer compression rate, onto the kind it names, and the
# rates a kind pools by (configuration_deepseek_v4.py:28-32, :156).
_V4_RATIOS = {0: 'sliding_attention', 4: 'compressed_sparse_attention',
              128: 'heavily_compressed_attention'}
_V4_RATES = {'compressed_sparse_attention': 4, 'heavily_compressed_attention': 128}
_V4_MLP_KINDS = ('hash_moe', 'moe')
# The activations ACT2FN holds for `scoring_func` (modeling_deepseek_v4.py:1031).
_V4_SCORES = ('softmax', 'sigmoid', 'sqrtsoftplus')
# qk_rope_head_dim over head_dim of the released Flash config, which is the
# class default (configuration_deepseek_v4.py:148).
_V4_PARTIAL = 64 / 512


def _string_sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        _refuse(field, 'expected one string entry per layer')
    entries: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            _refuse(field, f'expected a string, got {entry!r}')
        entries.append(entry)
    return tuple(entries)


def _v4_layer_types(hf_config: Mapping[str, object], layers: int,
                    used: set[str]) -> tuple[str, ...]:
    """Return every layer's attention kind.

    `DeepseekV4Config.__post_init__` resolves them in this order
    (configuration_deepseek_v4.py:255-267): an explicit `layer_types`, else the
    legacy `compress_ratios` read through the rate table, else the V4-Pro
    default of two heavily compressed layers and an interleave. Either list is
    truncated to the layer count, which drops the released config's trailing
    entry for its prediction depth.
    """
    used.update(('layer_types', 'compress_ratios'))
    types = hf_config.get('layer_types')
    ratios = hf_config.get('compress_ratios')
    if types is None and ratios is not None:
        if not isinstance(ratios, (list, tuple)) or any(type(rate) is not int for rate in ratios):
            _refuse('compress_ratios', 'expected an integer compression ratio per layer')
        unknown = sorted(set(ratios) - set(_V4_RATIOS))
        if unknown:
            _refuse(f"compress_ratios entries {unknown}",
                    f"a rate names its kind, one of {sorted(_V4_RATIOS)}")
        types = [_V4_RATIOS[rate] for rate in ratios]
    if types is None:
        types = (['heavily_compressed_attention'] * min(layers, 2)
                 + ['compressed_sparse_attention' if index % 2
                    else 'heavily_compressed_attention'
                    for index in range(max(layers - 2, 0))])
    resolved = _string_sequence(types, 'layer_types')[:layers]
    if len(resolved) != layers:
        _refuse(f"layer_types of {len(resolved)} entries",
                f"the model has {layers} layers, one attention kind each")
    unknown = sorted(set(resolved) - set(_V4_KINDS))
    if unknown:
        _refuse(f"layer_types entries {unknown}",
                f"a deepseek_v4 layer is one of {sorted(_V4_KINDS)}")
    return resolved


def _v4_mlp_kinds(hf_config: Mapping[str, object], layers: int,
                  used: set[str]) -> tuple[str, ...]:
    """Return every layer's feed-forward kind.

    `DeepseekV4Config.__post_init__` resolves them in this order
    (configuration_deepseek_v4.py:269-273): an explicit `mlp_layer_types`, else
    the first `num_hash_layers` layers routing by the hash table and the rest
    by the biased top-k.
    """
    used.update(('mlp_layer_types', 'num_hash_layers'))
    types = hf_config.get('mlp_layer_types')
    if types is None:
        hashed = _record_int(hf_config, 'num_hash_layers', 3)
        types = ['hash_moe'] * min(layers, hashed) + ['moe'] * max(layers - hashed, 0)
    resolved = _string_sequence(types, 'mlp_layer_types')[:layers]
    if len(resolved) != layers:
        _refuse(f"mlp_layer_types of {len(resolved)} entries",
                f"the model has {layers} layers, one routing kind each")
    unknown = sorted(set(resolved) - set(_V4_MLP_KINDS))
    if unknown:
        _refuse(f"mlp_layer_types entries {unknown}",
                f"a deepseek_v4 feed-forward is one of {sorted(_V4_MLP_KINDS)}")
    return resolved


def _v4_rope_width(hf_config: Mapping[str, object], used: set[str],
                   head_dim: int) -> tuple[int, float]:
    """Return each head's (rotated width, fraction).

    The width is `int(head_dim * partial_rotary_factor)`
    (configuration_deepseek_v4.py:284-292).

    The legacy `qk_rope_head_dim` names that width and folds into the
    fraction; a config carrying both (the spelling transformers writes
    back) has to agree with what the fraction derives, since that is what
    the rotary tables size themselves by (modeling_deepseek_v4.py:131-134).
    """
    used.update(('partial_rotary_factor', 'qk_rope_head_dim'))
    legacy = hf_config.get('qk_rope_head_dim')
    partial = hf_config.get('partial_rotary_factor')
    if partial is not None:
        fraction = _record_float(hf_config, 'partial_rotary_factor')
    elif legacy is not None:
        fraction = _record_int(hf_config, 'qk_rope_head_dim') / head_dim
    else:
        fraction = _V4_PARTIAL
    width = int(head_dim * fraction)
    if legacy is not None and _record_int(hf_config, 'qk_rope_head_dim') != width:
        _refuse(f"qk_rope_head_dim {legacy!r}",
                f"partial_rotary_factor {partial} rotates {width} of the "
                f"{head_dim} head dims, which is the width the reference derives")
    if not 0 < width <= head_dim or width % 2:
        _refuse(f"partial_rotary_factor {partial}",
                f"it rotates {width} of the {head_dim} head dims, and the "
                "interleaved rotary turns pairs inside the head")
    return width, fraction


def _v4_rope_entries(hf_config: Mapping[str, object], used: set[str],
                     partial: float) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Return the `main` and `compress` rope entries the config resolves to.

    `DeepseekV4Config` resolves them at configuration_deepseek_v4.py:301-321.

    A config that nests both states them. Any other spelling is one flat
    ramp that rides the compressed layers alone, over `compress_rope_theta`,
    leaving the sliding layers the plain base; a YaRN one takes
    `attention_factor` 1.0 there, which is how V4 keeps its cos and sin
    unscaled. `rope_scaling` is the released spelling and wins over
    `rope_parameters`, as transformers' own conversion does
    (modeling_rope_utils.py:742-744).
    """
    used.update(('rope_theta', 'compress_rope_theta',
                 'rope_scaling', 'rope_parameters'))
    scaling = hf_config.get('rope_scaling')
    parameters = hf_config.get('rope_parameters')
    entry: Mapping[str, object] = (scaling if isinstance(scaling, Mapping)
                                else parameters if isinstance(parameters, Mapping) else {})
    theta = _record_float(hf_config, 'rope_theta', 10000.0)
    compress_theta = _record_float(hf_config, 'compress_rope_theta', 160000.0)
    main, compressed = entry.get('main'), entry.get('compress')
    if isinstance(main, Mapping) and isinstance(compressed, Mapping):
        return dict(main), dict(compressed)
    ramp = {key: value for key, value in entry.items()
            if key not in ('main', 'compress')}
    compress = {**ramp, 'rope_theta': compress_theta,
                'partial_rotary_factor': partial}
    compress.setdefault('rope_type', ramp.get('type', 'default'))
    if compress['rope_type'] == 'yarn':
        compress.setdefault('attention_factor', 1.0)
    return ({'rope_type': 'default', 'rope_theta': theta,
             'partial_rotary_factor': partial}, compress)


def _v4_rope(entry: Mapping[str, object], field: str, head_dim: int, width: int,
             max_pos: int) -> tuple[float, Ramp | None]:
    """Read one V4 rope entry into (base, YaRN record or None).

    The entry's own `partial_rotary_factor` sizes its table, and so the
    slice the layers on it rotate (modeling_deepseek_v4.py:131-134); one
    naming another width than the model's has no counterpart in a mixer
    that rotates one. A YaRN entry changes the frequencies alone here, so
    an `attention_factor` other than the 1.0 V4's own configs force
    refuses: the reference multiplies its cos and sin by it (:151-152),
    and an entry that names none derives a factor that does.
    """
    fraction = _record_float(entry, 'partial_rotary_factor', 1.0)
    if int(head_dim * fraction) != width:
        _refuse(f"{field} partial_rotary_factor {entry.get('partial_rotary_factor')!r}",
                f"its table rotates {int(head_dim * fraction)} of the {head_dim} "
                f"head dims where the model's rope width is {width}")
    theta = _record_float(entry, 'rope_theta', 10000.0)
    rope_type = entry.get('rope_type', entry.get('type', 'default'))
    if rope_type in ('default', 'none'):
        extra = sorted(set(entry) - {'rope_type', 'type', 'rope_theta', 'partial_rotary_factor'})
        if extra:
            _refuse(f'{field} fields {extra}', 'the plain rotary has no scaling fields')
        return theta, None
    if rope_type != 'yarn':
        _refuse(f"{field} rope_type {rope_type!r}",
                "the mixer applies plain or YaRN rotary positions")
    if entry.get('attention_factor') != 1.0:
        _refuse(f"{field} attention_factor {entry.get('attention_factor')!r}",
                "V4's rotary scales its cos and sin by this factor, and its "
                "own configs set the 1.0 that leaves them alone")
    ramp = {key: value for key, value in entry.items()
            if key != 'partial_rotary_factor'}
    ramp.setdefault('original_max_position_embeddings', max_pos)
    return theta, _yarn_record(dict(ramp, rope_theta=theta), field, theta, max_pos)


def _v4_compress_rates(hf_config: Mapping[str, object], layer_types: tuple[str, ...],
                       used: set[str]) -> dict[str, int]:
    """Return how many tokens each compressed kind pools into one entry.

    `DeepseekV4Config.__post_init__` resolves them from the `compress_rates`
    dict over the class defaults, with the legacy per-kind scalars folded in
    (configuration_deepseek_v4.py:246-252).
    """
    used.update(('compress_rates', 'compress_rate_csa', 'compress_rate_hca'))
    rates: dict[str, object] = dict(_V4_RATES)
    supplied = hf_config.get('compress_rates')
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            _refuse('compress_rates', 'expected a rate per compressed attention kind')
        rates.update(supplied)
    for legacy, kind in (('compress_rate_csa', 'compressed_sparse_attention'),
                         ('compress_rate_hca', 'heavily_compressed_attention')):
        if hf_config.get(legacy) is not None:
            rates[kind] = hf_config[legacy]
    unknown = sorted(set(rates) - set(_V4_KINDS))
    if unknown:
        _refuse(f"compress_rates keys {unknown}",
                f"a rate belongs to a kind, one of {sorted(_V4_KINDS)}")
    resolved: dict[str, int] = {}
    for kind in dict.fromkeys(layer_types):
        if _V4_KINDS[kind] is None:
            continue
        rate = rates.get(kind)
        if isinstance(rate, bool) or not isinstance(rate, int) or rate < 1:
            _refuse(f"compress_rates[{kind!r}] {rate!r}",
                    "a compressor pools whole windows of at least one token")
        resolved[kind] = int(rate)
    return resolved


def _v4_mixture(hf_config: Mapping[str, object], layers: int,
                mlp_kinds: tuple[str, ...], used: set[str]) -> MixtureFields:
    """Read the mixture every V4 layer routes to.

    The router scores the logits with the config's activation, selects on
    those scores plus its balancing bias and renormalises what it selected
    whatever `norm_topk_prob` says (modeling_deepseek_v4.py:1035-1042); a
    hash layer keeps the frozen table it selects by instead of the bias
    (:1045-1073). There is no group limit, and one shared MLP of the routed
    width whatever the count says (:1082).
    """
    used.update(('n_routed_experts', 'num_experts_per_tok', 'moe_intermediate_size',
                 'n_shared_experts', 'scoring_func', 'norm_topk_prob',
                 'topk_method', 'routed_scaling_factor'))
    for key in ('n_routed_experts', 'num_experts_per_tok', 'moe_intermediate_size'):
        if hf_config.get(key) is None:
            _refuse(key, "every deepseek_v4 layer routes, so the expert "
                           "count, the top-k and the routed width are the "
                           "checkpoint's to state")
    scoring = hf_config.get('scoring_func', 'sqrtsoftplus')
    if scoring not in _V4_SCORES:
        _refuse(f"scoring_func {scoring!r}",
                f"the router scores with one of {sorted(_V4_SCORES)}")
    if hf_config.get('norm_topk_prob', True) is not True:
        _refuse("norm_topk_prob=False",
                "both V4 routers renormalise the weights they selected, "
                "whatever this field says (modeling_deepseek_v4.py:1041, :1072)")
    shared = _record_int(hf_config, 'n_shared_experts', 1)
    if shared != 1:
        _refuse(f"n_shared_experts {shared!r}",
                "DeepseekV4SparseMoeBlock builds one shared MLP of the routed "
                "width (modeling_deepseek_v4.py:1082)")
    method = hf_config.get('topk_method')
    if method is not None and method != 'noaux_tc':
        _refuse(f"topk_method {method!r}",
                "the router selects on the scores plus its balancing bias, "
                "which is what noaux_tc names")
    width = _record_int(hf_config, 'moe_intermediate_size')
    return {
        'experts': _record_int(hf_config, 'n_routed_experts'),
        'top_k': _record_int(hf_config, 'num_experts_per_tok'),
        'layers': tuple(range(layers)),
        'score_function': scoring,
        'bias': True,
        'scaling': _record_float(hf_config, 'routed_scaling_factor', 1.5),
        'shared_features': width,
        'expert_features': width,
        'hash_layers': tuple(index for index, kind in enumerate(mlp_kinds)
                             if kind == 'hash_moe'),
    }


def _deepseek_v4_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a DeepSeek-V4-Flash or V4-Pro config into `CausalTransformer` fields.

    Every layer is mHC's stack of residual streams around one attention over a
    sliding window, with a low-rank query normed per head, one key/value head
    read as both, per-head sinks and a grouped low-rank output projection. The
    two compressed kinds extend those keys with their compressor's pooled
    entries, the sparse one selecting them with its lightning indexer
    (modeling_deepseek_v4.py:746-864).

    Every layer's feed-forward routes, and the first ones route by a frozen
    token-to-expert table rather than the biased top-k (:1045-1073). The rope
    is two entries: the sliding layers rotate at `rope_theta`, the compressed
    ones at `compress_rope_theta` under the ramp they share with their
    compressor (:768, :803-806).
    """
    layers = _record_int(hf_config, 'num_hidden_layers')
    heads = _record_int(hf_config, 'num_attention_heads')
    head_dim = _record_int(hf_config, 'head_dim', _record_int(hf_config, 'hidden_size') // heads)
    layer_types = _v4_layer_types(hf_config, layers, used)
    mlp_kinds = _v4_mlp_kinds(hf_config, layers, used)
    rope_width, partial = _v4_rope_width(hf_config, used, head_dim)
    max_pos = _record_int(hf_config, 'max_position_embeddings', DEFAULT_MAX_SEQ_LEN)
    main, compress = _v4_rope_entries(hf_config, used, partial)
    main_theta, main_ramp = _v4_rope(main, 'rope_parameters main',
                                     head_dim, rope_width, max_pos)
    if main_ramp is not None:
        _refuse("rope_parameters main rope_type 'yarn'",
                "the sliding layers rotate at the plain base, which is what "
                "the config's own main entry names")
    compress_theta, compress_ramp = _v4_rope(compress, 'rope_parameters compress',
                                             head_dim, rope_width, max_pos)
    window = hf_config.get('sliding_window')
    if window is None:
        _refuse("sliding_window", "every deepseek_v4 layer attends a window")
    window = _record_int(hf_config, 'sliding_window')
    kv_heads = _record_int(hf_config, 'num_key_value_heads', 1)
    if kv_heads != 1:
        _refuse(f"num_key_value_heads {kv_heads!r}",
                "V4's attention projects one key/value head and broadcasts it "
                "to every query head (modeling_deepseek_v4.py:749-751, :770)")
    width = _record_int(hf_config, 'moe_intermediate_size')
    # V4 ships no dense feed-forward width and no key/value head count the
    # attention reads: `intermediate_size` is an alias of the routed width
    # (configuration_deepseek_v4.py:101-107) and the one shared head is the
    # block's own.
    config = _base_config({**hf_config, 'intermediate_size': width,
                           'num_key_value_heads': 1}, used,
                          layer_types=layer_types, rope=_Ropes(main_theta))
    if config.get('attention_bias'):
        _refuse("attention_bias=True",
                "V4 builds every attention projection bias-free "
                "(modeling_deepseek_v4.py:777-786)")
    used.update(('q_lora_rank', 'o_groups', 'o_lora_rank',
                 'index_topk', 'index_n_heads', 'index_head_dim',
                 'swiglu_limit', 'hc_mult', 'hc_eps', 'hc_sinkhorn_iters'))
    # Transformers omits the prediction depth (modeling_deepseek_v4.py:1212);
    # the released inference/model.py MTPBlock executes its raw-stream
    # e_proj/h_proj composition. The router's logit output and aux
    # coefficient are training knobs, its jitter is read nowhere, and the
    # storage hints name what a quantized checkpoint is dequantized from.
    used.update(('num_nextn_predict_layers', 'output_router_logits',
                 'router_aux_loss_coef', 'router_jitter_noise',
                 'quantization_config', 'expert_dtype', 'ep_size'))
    groups = _record_int(hf_config, 'o_groups')
    mixer: dict[str, object] = {
        'kind': 'deepseek_v4',
        'q_lora_rank': _record_int(hf_config, 'q_lora_rank'),
        'o_groups': groups,
        'o_lora_rank': _record_int(hf_config, 'o_lora_rank'),
        'rope_head_dim': rope_width,
        'compressor': None, 'compress_rate': None,
        'index_topk': None, 'index_n_heads': None, 'index_head_dim': None}
    if groups < 1 or heads * head_dim % groups:
        _refuse(f"o_groups {mixer['o_groups']}",
                f"the grouped output projection splits the {heads * head_dim} "
                "stacked head dims into equal groups")
    kinds = _v4_attention_kinds(hf_config, used, layer_types, mixer, window,
                                (compress_theta, compress_ramp))
    depth = hf_config.get('num_nextn_predict_layers', 1)
    if type(depth) is not int or depth not in (0, 1):
        _refuse('num_nextn_predict_layers', 'V4 supports the released single prediction depth')
    if depth:
        ratios = hf_config.get('compress_ratios')
        if isinstance(ratios, (list, tuple)) and len(ratios) > layers and ratios[layers] != 0:
            _refuse('compress_ratios prediction depth', 'the released V4 prediction block is sliding-only')
        kinds['mtp_attention'] = {'window': int(window), 'rope_theta': main_theta, 'mixer': mixer}
        config['mtp_layer_type'] = 'mtp_attention'
    config['num_nextn_predict_layers'] = depth
    streams = _v4_streams(hf_config)
    config.update(
        mixer=mixer,
        kinds=kinds,
        mixture=_v4_mixture(hf_config, layers, mlp_kinds, used),
        swiglu_limit=_record_float(hf_config, 'swiglu_limit', 10.0),
        hyper_connections=streams,
    )
    if depth:
        config['mtp_hyper_connections'] = streams.copy()
    return config


def _v4_attention_kinds(hf_config: Mapping[str, object], used: set[str],
                        layer_types: tuple[str, ...], mixer: Mapping[str, object],
                        window: int, compress: tuple[float, Ramp | None]) -> dict[str, KindFields]:
    """Build each attention kind's own record.

    Every V4 layer attends its window; a compressed one rotates at the
    compress base and hands its compressor and rate to the mixer, the sparse
    one its indexer's geometry too.
    """
    compress_theta, compress_ramp = compress
    index = ({field: _record_int(hf_config, field)
              for field in ('index_topk', 'index_n_heads', 'index_head_dim')}
             if 'compressed_sparse_attention' in layer_types else {})
    rates = _v4_compress_rates(hf_config, layer_types, used)
    kinds: dict[str, KindFields] = {}
    for kind in dict.fromkeys(layer_types):
        record: KindFields = {'window': window}
        compressor = _V4_KINDS[kind]
        if compressor is not None:
            layer_mixer = {**mixer, 'compressor': compressor,
                           'compress_rate': rates[kind]}
            if compressor == 'csa':
                layer_mixer.update(index)
            record.update(rope_theta=compress_theta, yarn=compress_ramp,
                          mixer=layer_mixer)
        kinds[kind] = record
    return kinds


def _v4_streams(hf_config: Mapping[str, object]) -> HyperConnectionsFields:
    """Return the residual streams every V4 layer reads and writes.

    A layer collapses them through a learned head of its own
    (DeepseekV4HyperHead, modeling_deepseek_v4.py:946-962).
    """
    return {
        'hc_mult': _record_int(hf_config, 'hc_mult', 4),
        'hc_eps': _record_float(hf_config, 'hc_eps', 1e-6),
        'hc_sinkhorn_iters': _record_int(hf_config, 'hc_sinkhorn_iters', 20),
        'head': 'weighted'}


# A DeepSeek V4 checkpoint's own names onto the module names transformers
# renames them to (conversion_mapping.py:483-534), which is the layout
# `_param_path` reads. The indexer's leaves move before the `attn` and `ffn`
# prefixes they sit under, and the per-expert and shared `w1`/`w2`/`w3` are
# the gate, down and up projections of one gated MLP.
_DEEPSEEK_V4_NAMES = (
    ('.indexer.compressor.', '.compressor.indexer.'),
    ('.indexer.wq_b.', '.compressor.indexer.q_b_proj.'),
    ('.indexer.weights_proj.', '.compressor.indexer.scorer.weights_proj.'),
    ('.attn_sink', '.sinks'),
    ('.norm.', '.kv_norm.'),
    ('.q_norm.', '.q_a_norm.'),
    ('.ape', '.position_bias'),
    ('.wq_a.', '.q_a_proj.'),
    ('.wq_b.', '.q_b_proj.'),
    ('.wkv.', '.kv_proj.'),
    ('.wgate.', '.gate_proj.'),
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


# The trunk's tensors, whose released names the three ^-anchored renames of
# conversion_mapping.py:489-493 rewrite on load and, being anchored, leave
# alone on save: a released checkpoint carries the left column and one
# transformers wrote the right one, so both spellings are read here.
_DEEPSEEK_V4_TRUNK = {
    'embed.weight': 'model.embed_tokens.weight',
    'norm.weight': 'model.norm.weight',
    'head.weight': 'lm_head.weight',
    'hc_head_fn': 'model.hc_head.hc_fn',
    'hc_head_base': 'model.hc_head.hc_base',
    'hc_head_scale': 'model.hc_head.hc_scale',
}


def _deepseek_v4_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the decoder's path for one DeepSeek V4 tensor name.

    The release names the block's halves `attn` and `ffn`, its projections
    `wq_a`/`wq_b`/`wkv`/`wgate`/`wo_a`/`wo_b`, its query latent norm
    `q_norm`, its compressors' entry norms `norm`, its position biases
    `ape` and its sinks `attn_sink`, and it holds the stack at `layers.N.*`
    with no `model.` prefix. transformers renames those onto its module
    names on load and writes most of them back on save, so a released
    checkpoint and an export differ in the trunk's spelling and the prefix
    and agree on the rest; both are read. The renames run over one layer's
    own tail, where the trunk's `norm.weight` cannot meet the compressors'
    pattern.

    The release's prediction depth has its own split input projections,
    mHC block, collapse head and final norm (official inference/model.py,
    MTPBlock), while sharing the trunk embedding and vocabulary head.
    """
    if name.startswith('mtp.'):
        parts = name.split('.')
        if len(parts) < 3 or not parts[1].isdigit() or int(parts[1]) >= _record_int(config, 'num_nextn_predict_layers', 0):
            raise ValueError(f"{name} names an undeclared prediction depth")
        depth = f'mtp_{parts[1]}'
        tail = '.'.join(parts[2:])
        if tail == 'norm.weight':
            return ('params', depth, 'final_norm', 'scale')
        if tail in ('enorm.weight', 'hnorm.weight'):
            return ('params', depth, parts[2], 'scale')
        if tail in ('e_proj.weight', 'h_proj.weight'):
            return ('params', depth, parts[2], 'kernel')
        if tail in ('hc_head_fn', 'hc_head_base', 'hc_head_scale'):
            return ('params', depth, 'hc_head', tail.replace('hc_head_', 'hc_', 1))
        path = _deepseek_v4_path('layers.0.' + tail, config)
        if path is None:
            raise ValueError(f"{name} has no prediction-block parameter")
        return (path[0], depth, 'block', *path[2:])
    name = _DEEPSEEK_V4_TRUNK.get(name, name)
    if name.startswith('layers.'):
        name = 'model.' + name
    parts = name.split('.')
    if not (len(parts) >= 4 and parts[:2] == ['model', 'layers'] and parts[2].isdigit()):
        return _dew_path(name, config)
    tail = '.' + '.'.join(parts[3:])
    for theirs, ours in _DEEPSEEK_V4_NAMES:
        tail = tail.replace(theirs, ours)
    return _dew_path(f"model.layers.{parts[2]}{tail}", config)


def _deepseek_v4_prepare(tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reshape the grouped output projection and the hash table as the tree holds them.

    `DeepseekV4GroupedLinear` stores one block per head group in a matrix of
    `[groups * o_lora_rank, heads * head_dim / groups]` and reads it as
    `weight.view(groups, -1, in).transpose(1, 2)`
    (modeling_deepseek_v4.py:294-323), which is the `[groups, in, rank]`
    kernel the mixer contracts over; the group count is the stacked query
    width over one block's input width. The token-to-expert table narrows to
    the int32 the router's collection holds (:1062).
    """
    prepared = dict(tensors)
    for name, tensor in tensors.items():
        if name.endswith('.attn.wo_a.weight'):
            partner = name.removesuffix('wo_a.weight') + 'wq_b.weight'
            stacked = tensors.get(partner)
            if stacked is None:
                raise ValueError(f"{name} needs {partner} to name its head groups")
            groups = stacked.shape[0] // tensor.shape[1]
            prepared[name] = np.ascontiguousarray(
                tensor.reshape(groups, -1, tensor.shape[1]).transpose(0, 2, 1))
        elif name.endswith('.gate.tid2eid'):
            if not np.issubdtype(tensor.dtype, np.integer):
                raise ValueError(f"{name} must contain integer expert indices")
            router = tensors.get(name.removesuffix('tid2eid') + 'weight')
            if router is None:
                raise ValueError(f"{name} has no gate.weight declaring its expert count")
            if tensor.size:
                lower, upper = np.min(tensor), np.max(tensor)
                limits = np.iinfo(np.int32)
                if lower < limits.min or upper > limits.max:
                    raise ValueError(f"{name} has expert indices outside the lossless int32 range")
                if lower < 0 or upper >= router.shape[0]:
                    raise ValueError(f"{name} has indices outside its {router.shape[0]} experts")
            prepared[name] = np.asarray(tensor, np.int32)
    return prepared
