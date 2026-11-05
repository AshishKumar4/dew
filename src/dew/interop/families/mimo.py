"""Translate MiMo-V2-Flash (`mimo_v2_flash`): full and sliding attention
layers over 192-wide queries and keys and 128-wide values, and DeepSeek V3's
routed experts without a shared one.

The sliding layers hold twice the full layers' key and value heads, add a
learned sink per head and rotate a third of each head under their own base;
every layer scales its values by `attention_value_scale` before attending
(modeling_mimo_v2_flash.py, MiMoV2FlashAttention). The released configs
spell the layout in the authors' own fields (`hybrid_layer_pattern`,
`swa_num_key_value_heads`, ...), which transformers 5.16.1 does not read: it
derives the same model from its defaults, so each such field is accepted
only where it states what the reference computes.

MiMo-V2.6 (`mimo_v2`) is the same decoder under the authors' own config,
which its remote code reads (modeling_mimo_v2.py at 6a7f202): the layout
from `hybrid_layer_pattern` and `moe_layer_freq`, each kind's geometry from
the `swa_*` fields, and a sink per head on the kinds whose flag asks for
one. Its Pro release stores each layer's q, k and v projections fused,
pre-sharded over the key heads (`attention_projection_layout` 'fused_qkv',
vLLM's `_shard_fp8_qkv_proj` at 78b32b3). A release with its towers is a bundle
its family reads whole (`DecoderFamily.wrapper`): the ViT and its merger, and the
audio encoder over the codes of the audio tokenizer the release keeps in a folder
of its own; the decoder read alone leaves them, and its prediction layers, unread
and writes them back.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from dew import _records
from dew.interop import release_pins
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.decoder_parts import (
    NO_AUDIO,
    PLAIN_SPELLING,
    PREDICTION_INPUTS,
    PREDICTION_STEP_END,
    DecoderFamily,
    DecoderFields,
    KindFields,
    Ramp,
    Renames,
    Ropes,
    SourceCode,
    WrapperFields,
    base_config,
    decoder_kinds,
    folder_config,
    folder_name,
    grouped_qkv,
    nested_ramp,
    partial_rope_entry,
    plain_partial_rope,
    prediction_path,
    record_int,
    refuse,
    renamed_name,
    renamed_path,
    translated,
)
from dew.interop.families.deepseek import deepseek_layout, deepseek_mixture
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mimo_audio import translate_mimo_audio_config
from dew.nn.mixers.mlp import MLPMixer
from dew.nn.vision.mimo import translate_mimo_projector_config, translate_mimo_vision_config

_KINDS = ('full_attention', 'sliding_attention')

_NAMES: Renames = (('self_attn.attention_sink_bias', 'self_attn.sinks'),)
"""The checkpoint's sink logits under the shared map's name for them."""


def _default_layers(layers: int) -> tuple[str, ...]:
    """MiMoV2FlashConfig's pattern: the first layer and every sixth full, the
    rest sliding (configuration_mimo_v2_flash.py:112-116)."""
    return tuple('full_attention' if index == 0 or not (index + 1) % 6 else 'sliding_attention'
                 for index in range(layers))


# MiMoV2FlashConfig's per-kind pretrained bases where a config states no rope_parameters
# (configuration_mimo_v2_flash.py:122-126), which the released flat fields override.
_FLASH_THETAS = {'full_attention': 5_000_000.0, 'sliding_attention': 10_000.0}


def _rope(hf: Mapping[str, object], kind: str) -> tuple[float, float, Ramp | None]:
    """One kind's (rope_theta, partial_rotary_factor, ramp): a `rope_parameters` entry, which states
    its base, or the released flat fields with the reference's defaults and no ramp."""
    nested = hf.get('rope_parameters')
    if nested is not None:
        entry = _records.record(_records.record(nested, 'rope_parameters').get(kind),
                               f'rope_parameters {kind}')
        # MiMoV2FlashRotaryEmbedding indexes each entry's base and builds the kind's rope_type over
        # the rotated dims (modeling_mimo_v2_flash.py:80-111): its plain initializer defaults an
        # omitted fraction to 0.334, the scaled ones (`_compute_yarn_parameters`) to the whole head,
        # and a nested record's YaRN always truncates (`nested_ramp`).
        stated = hf.get('max_position_embeddings')
        plain = entry.get('rope_type', entry.get('type', 'default')) in ('default', 'none')
        theta, factor, ramp = partial_rope_entry(
            entry, f'rope_parameters.{kind}', default_factor=0.334 if plain else 1.0,
            max_pos=None if stated is None else _records.integer(stated, 'max_position_embeddings'))
        return theta, factor, nested_ramp(ramp)
    theta = hf.get('rope_theta' if kind == 'full_attention' else 'swa_rope_theta', _FLASH_THETAS[kind])
    return (_records.number(theta, 'rope_theta'),
            _records.number(hf.get('partial_rotary_factor', 0.334), 'partial_rotary_factor'), None)


def _check_released(hf: Mapping[str, object], used: set[str], *, layers: tuple[str, ...],
                    sparse: tuple[int, ...], kv_heads: int) -> None:
    """Accept the authors' own fields where they state what transformers
    computes from its defaults, and refuse any that state another model."""
    heads = _records.integer(hf['num_attention_heads'], 'num_attention_heads')
    head_dim = _records.integer(hf.get('head_dim', 192), 'head_dim')
    value_dim = _records.integer(hf.get('v_head_dim', 128), 'v_head_dim')
    window = hf.get('sliding_window', 128)
    expected = {
        'hybrid_layer_pattern': [0 if kind == 'full_attention' else 1 for kind in layers],
        'moe_layer_freq': [int(index in sparse) for index in range(len(layers))],
        'add_swa_attention_sink_bias': True, 'add_full_attention_sink_bias': False,
        'swa_num_attention_heads': heads, 'swa_num_key_value_heads': 2 * kv_heads,
        'swa_head_dim': head_dim, 'swa_v_head_dim': value_dim,
        'sliding_window_size': window, 'layernorm_epsilon': hf.get('rms_norm_eps', 1e-5),
        'n_shared_experts': None, 'scoring_func': 'sigmoid', 'topk_method': 'noaux_tc',
    }
    for key, value in expected.items():
        if key in hf and hf[key] != value:
            refuse(f'{key}={hf[key]!r}', f'transformers computes {value!r} from MiMoV2FlashConfig')
    # The released configs state the window again as a chunk the reference
    # never reads; attention stays sliding, not chunked.
    used.update((*expected, 'attention_chunk_size', 'swa_rope_theta'))


def _mimo_v2_flash_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    count = _records.integer(hf['num_hidden_layers'], 'num_hidden_layers')
    layers = (_records.strings(hf['layer_types'], 'layer_types') if hf.get('layer_types') is not None
              else _default_layers(count))
    if set(layers) - set(_KINDS) or len(layers) != count:
        refuse(f'layer_types {list(layers)!r}', 'one full_attention or sliding_attention entry per layer')
    schedule = (_records.strings(hf['mlp_layer_types'], 'mlp_layer_types')
                if hf.get('mlp_layer_types') is not None else ('dense',) + ('sparse',) * (count - 1))
    if set(schedule) - {'dense', 'sparse'} or len(schedule) != count:
        refuse(f'mlp_layer_types {list(schedule)!r}', 'one dense or sparse entry per layer')
    sparse = tuple(index for index, kind in enumerate(schedule) if kind == 'sparse')
    kv_heads = _records.integer(hf.get('num_key_value_heads', 4), 'num_key_value_heads')
    _check_released(hf, used, layers=layers, sparse=sparse, kv_heads=kv_heads)
    (full_theta, full_factor, full_ramp), (local_theta, local_factor, local_ramp) = (
        _rope(hf, kind) for kind in _KINDS)
    used.update(('layer_types', 'mlp_layer_types', 'rope_parameters', 'rope_theta', 'partial_rotary_factor',
                 'v_head_dim', 'attention_value_scale', 'norm_topk_prob', 'attention_dropout'))
    config = base_config({**hf, 'head_dim': hf.get('head_dim', 192), 'num_key_value_heads': kv_heads,
                          'rms_norm_eps': hf.get('rms_norm_eps', hf.get('layernorm_epsilon', 1e-5))},
                         used, layer_types=layers, rope=Ropes(full_theta), scale_after_cast=True,
                         reads=frozenset({'sliding_window', 'attention_bias'}))
    window = (_records.integer(hf.get('sliding_window', 128), 'sliding_window')
              if 'sliding_attention' in layers else None)
    kinds = decoder_kinds(layers, window, local_theta, None, None)
    if 'sliding_attention' in layers:
        # MiMoV2FlashAttention doubles a sliding layer's key and value heads
        # and gives it the sinks (modeling_mimo_v2_flash.py:348, 363).
        kinds['sliding_attention'].update(num_kv_heads=2 * kv_heads, sinks=True,
                                          partial_rotary_factor=local_factor)
    for kind, ramp in zip(_KINDS, (full_ramp, local_ramp), strict=True):
        if ramp is not None and kind in layers:
            kinds.setdefault(kind, native_fields(LayerKind)())['rope_scaling'] = ramp
    return _attention_and_experts(
        config, hf, used, kinds, value_dim=_records.integer(hf.get('v_head_dim', 128), 'v_head_dim'),
        scale=0.707, factor=full_factor, count=count, sparse=sparse)


def _attention_and_experts(config: DecoderFields, hf: Mapping[str, object], used: set[str],
                           kinds: Mapping[str, KindFields], *, value_dim: int, scale: float | None,
                           factor: float, count: int, sparse: tuple[int, ...]) -> DecoderFields:
    """`config` with what both MiMo-V2 readers set alike: the layer kinds, the value heads' width
    and scale (`attention_value_scale`, `scale` where the config omits it), the unbiased output
    projection, the 'default' partial rotary of the full layers,
    attention dropout and the routed experts (`_mixture`)."""
    config.update(
        kinds=kinds, value_head_dim=value_dim,
        value_scale=None if (stated := hf.get('attention_value_scale', scale)) is None
        else _records.number(stated, 'attention_value_scale'),
        o_proj_bias=False, partial_rotary_factor=None if factor == 1.0 else factor,
        partial_rotary_type='default',
        attention_dropout_rate=_records.number(hf.get('attention_dropout', 0.), 'attention_dropout'),
        mixture=_mixture(hf, count, used, sparse))
    return config


def _mixture(hf: Mapping[str, object], count: int, used: set[str],
             sparse: tuple[int, ...]) -> NativeFields | None:
    """DeepSeek V3's routed experts on the `sparse` layers: sigmoid scores, a
    balancing bias, grouped choice and renormalized top-k weights. A config
    that states experts no layer routes to builds dense layers alone, as the
    reference does; its expert fields are read all the same."""
    if hf.get('n_routed_experts') is None and not sparse:
        return None
    # A null scaling is the reference's 1.0.
    stated = {**hf, 'routed_scaling_factor': hf.get('routed_scaling_factor') or 1.0}
    if not sparse:
        deepseek_layout(stated, count, used, sparse_layers=sparse)
        return None
    return deepseek_mixture(stated, count, used, sparse_layers=sparse)


def _mimo_v2_flash_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """The shared map under the family's names; the released checkpoints'
    `model.mtp.*` prediction layers, which the reference never runs
    (`_keys_to_ignore_on_load_unexpected`), have no place."""
    if name.startswith('model.mtp.'):
        return None
    return renamed_path(_NAMES, name, config)


def _mimo_v2_flash_export(model: CausalTransformer) -> Mapping[str, object]:
    sliding = model.kind_of('sliding_attention')
    full = model.kind_of('full_attention')
    mixture = model.mixture
    return {
        'layer_types': list(model.per_layer_types),
        'mlp_layer_types': ['sparse' if mixture is not None and index in model.sparse_layers else 'dense'
                            for index in range(model.num_layers)],
        'num_key_value_heads': full.num_kv_heads, 'v_head_dim': model.value_head_dim,
        'attention_value_scale': model.value_scale,
        'rope_parameters': {kind: {'rope_type': 'default', 'rope_theta': resolved.rope_theta,
                                   'partial_rotary_factor': resolved.partial_rotary_factor or 1.0}
                            for kind, resolved in (('full_attention', full), ('sliding_attention', sliding))},
        'attention_dropout': model.attention_dropout_rate, 'rope_theta': None, 'rope_local_base_freq': None,
        **_experts(model)}


def _experts(model: CausalTransformer) -> Mapping[str, object]:
    """Both MiMo-V2 configs' routed experts (`_mixture`), each None without a mixture."""
    mixture = model.mixture
    return {
        'moe_intermediate_size': None if mixture is None else mixture.expert_features,
        'n_routed_experts': None if mixture is None else mixture.experts,
        'num_experts_per_tok': None if mixture is None else mixture.top_k,
        'routed_scaling_factor': None if mixture is None else mixture.scaling,
        'n_group': None if mixture is None else mixture.groups,
        'topk_group': None if mixture is None else mixture.groups_per_token,
        'norm_topk_prob': True,
    }


def _matches(fields: CausalTransformer) -> bool:
    """Values narrower than the heads, which no other family's attention has: MiMo-V2's geometry."""
    return fields.value_head_dim is not None and fields.value_head_dim != fields.features_per_head


def _flash_matches(fields: CausalTransformer) -> bool:
    """MiMo-V2's geometry as MiMo-V2-Flash fixes it (MiMoV2FlashAttention): no sinks on the full layers,
    and the sliding layers' sinks over twice the full layers' key heads. Another holds the authors' own
    config (`MIMO_V2`)."""
    if not _matches(fields):
        return False
    types = set(fields.per_layer_types)
    if 'full_attention' in types and fields.kind_of('full_attention').sinks:
        return False
    sliding = fields.kind_of('sliding_attention') if 'sliding_attention' in types else None
    return sliding is None or (sliding.sinks and sliding.num_kv_heads == 2 * fields.kv_heads)


MIMO_V2_FLASH = DecoderFamily(
    ('mimo_v2_flash',), _mimo_v2_flash_config, _flash_matches, 'mimo_v2_flash', 'MiMoV2FlashForCausalLM',
    _mimo_v2_flash_export, weight_path=_mimo_v2_flash_path, export_path=partial(renamed_name, _NAMES),
    # Its experts are stored one tensor each, so a save writes them back through the source's own names.
    preserve_source_layout=True,
    padding_idx=True,
    defaults={'max_position_embeddings': 131072},
)


_V2_FIELDS = ('swa_num_attention_heads', 'swa_num_key_value_heads', 'swa_head_dim', 'swa_v_head_dim',
              'swa_rope_theta', 'sliding_window_size', 'layernorm_epsilon', 'add_full_attention_sink_bias',
              'add_swa_attention_sink_bias', 'attention_projection_layout', 'attention_value_scale',
              'v_head_dim', 'partial_rotary_factor', 'rope_parameters', 'rope_theta', 'norm_topk_prob',
              'scoring_func', 'topk_method', 'attention_dropout')
_AUDIO_TOKENIZER = 'audio_tokenizer'
"""The folder a MiMo-V2.6 release keeps its audio tokenizer in, its own config and weights, which the
release's code loads beside the model (`MiMoV2ForCausalLM.load_audio_tokenizer`)."""

_V2_UNREAD = ('attention_chunk_size', 'moe_router_dtype', 'audio_config', 'vision_config', 'processor_config',
              'vision_model_type', 'audio_token_id', 'audio_start_token_id', 'audio_end_token_id',
              'image_token_id', 'video_token_id', 'vision_start_token_id', 'vision_end_token_id',
              folder_config(_AUDIO_TOKENIZER))
"""Fields MiMoV2ForCausalLM's text decoder never reads: the chunk the window
is restated as, the router dtype its fp32 gate ignores (modeling_mimo_v2.py:153),
and the towers' sections, tokens and folder, whose tensors ride unread."""


def _layer_flags(hf: Mapping[str, object], used: set[str],
                 count: int) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Which layers slide and which route, as MiMoV2Config derives them
    (configuration_mimo_v2.py:206-232): `hybrid_layer_pattern`, else one full
    layer closing each `hybrid_block_size` block, else every layer full;
    `moe_layer_freq` a flag a layer, or every `moe_layer_freq`-th from the
    first, else none."""
    used.update(('hybrid_layer_pattern', 'hybrid_block_size', 'moe_layer_freq'))
    block = hf.get('hybrid_block_size')
    pattern = hf.get('hybrid_layer_pattern')
    if pattern is None:
        size = None if block is None else _records.integer(block, 'hybrid_block_size')
        sliding = tuple(bool(size and (index + 1) % size) for index in range(count))
    else:
        sliding = tuple(bool(slides) for slides in _records.integers(pattern, 'hybrid_layer_pattern'))
    frequency = hf.get('moe_layer_freq')
    if isinstance(frequency, int) and not isinstance(frequency, bool):
        routed = tuple(frequency > 0 and index % frequency == 0 for index in range(count))
    elif frequency is None:
        routed = (False,) * count
    elif isinstance(frequency, (list, tuple)) and all(routes in (0, 1) for routes in frequency):
        # A saved MiMoV2Config writes the flags it derived as booleans, a release as 0 and 1.
        routed = tuple(bool(routes) for routes in frequency)
    else:
        refuse(f'moe_layer_freq {frequency!r}', 'a period or one 0-or-1 flag a layer')
    if len(sliding) != count or len(routed) != count:
        refuse('hybrid_layer_pattern and moe_layer_freq', f'one entry a layer, {count} of them')
    return sliding, routed


def _mimo_v2_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    count = _records.integer(hf['num_hidden_layers'], 'num_hidden_layers')
    sliding, routed = _layer_flags(hf, used, count)
    layers = tuple('sliding_attention' if slides else 'full_attention' for slides in sliding)
    experts = hf.get('n_routed_experts') is not None
    sparse = tuple(index for index, routes in enumerate(routed) if routes and experts)
    # MiMoV2Config's geometry and its defaults (configuration_mimo_v2.py:156-197).
    heads = _records.integer(hf.get('num_attention_heads', 32), 'num_attention_heads')
    kv_heads = _records.integer(hf.get('num_key_value_heads') or heads, 'num_key_value_heads')
    width = _records.integer(hf['hidden_size'], 'hidden_size')
    head_dim = _records.integer(hf.get('head_dim') or width // heads, 'head_dim')
    value_dim = _records.integer(hf.get('v_head_dim') or head_dim, 'v_head_dim')
    swa_head_dim = _records.integer(hf.get('swa_head_dim') or head_dim, 'swa_head_dim')
    swa = {'swa_num_attention_heads': (_records.integer(hf.get('swa_num_attention_heads') or heads,
                                                       'swa_num_attention_heads'), heads),
           'swa_head_dim': (swa_head_dim, head_dim),
           'swa_v_head_dim': (_records.integer(hf.get('swa_v_head_dim') or swa_head_dim, 'swa_v_head_dim'),
                              value_dim)}
    # The prediction depths: the count a config states (Dew's export of one, 0 where it holds none), else
    # the one depth the release's code skips on load and vLLM drafts with (mimo_v2_mtp.py at 19aba3e,
    # `MiMoV2MTPLayer`, one layer for the Pro and the Flash alike) where its trunk has sliding layers, as
    # every release's does. Each attends with the sliding geometry the config states, whatever the trunk's
    # pattern, which the sliding layers read too.
    slides = 'sliding_attention' in layers
    depths = _records.integer(hf.get('num_nextn_predict_layers', int(slides)), 'num_nextn_predict_layers')
    used.add('num_nextn_predict_layers')
    swa_kv_heads = _records.integer(hf.get('swa_num_key_value_heads') or kv_heads,
                                   'swa_num_key_value_heads')
    layout = hf.get('attention_projection_layout') or 'split'
    if layout not in ('split', 'fused_qkv'):
        refuse(f'attention_projection_layout {layout!r}', "MiMoV2Attention stores 'split' or 'fused_qkv'")
    for key, (stated, full) in swa.items():
        if stated != full and slides:
            refuse(f'{key} {stated}', f"Dew's sliding layers hold the full layers' {full}; "
                                      'its kinds differ in key heads alone')
        # A prediction layer's own kind takes its query heads and width; its values are the model's.
        if stated != full and depths and key == 'swa_v_head_dim':
            refuse(f'{key} {stated}', f"a prediction layer's values are the model's {full} wide")
    if layout == 'fused_qkv' and swa_kv_heads != kv_heads and slides:
        refuse('attention_projection_layout fused_qkv', 'its runs are one geometry for every layer here, '
               f'not {kv_heads} key heads on the full layers and {swa_kv_heads} on the sliding ones')
    window = hf.get('sliding_window') or hf.get('sliding_window_size')
    if (hf.get('scoring_func') or 'sigmoid', hf.get('topk_method') or 'noaux_tc') != ('sigmoid', 'noaux_tc'):
        refuse('scoring_func and topk_method', "MiMoV2MoEGate implements sigmoid scores under 'noaux_tc'")
    theta, factor = plain_partial_rope(hf, PLAIN_SPELLING, default_factor=1.0,
                                       reference='MiMoV2RotaryEmbedding')
    used.update(_V2_FIELDS + _V2_UNREAD)
    config = base_config({**hf, 'sliding_window': window, 'num_attention_heads': heads,
                          'num_key_value_heads': kv_heads, 'head_dim': head_dim,
                          'rms_norm_eps': hf.get('layernorm_epsilon', 1e-6),
                          'moe_intermediate_size': (hf.get('moe_intermediate_size')
                                                    or hf.get('intermediate_size'))},
                         used, layer_types=layers, rope=Ropes(theta), scale_after_cast=True,
                         reads=frozenset({'sliding_window', 'attention_bias'}))
    local_theta = _records.number(hf.get('swa_rope_theta') or theta, 'swa_rope_theta')
    kinds = decoder_kinds(layers, None if window is None else _records.integer(window, 'sliding_window'),
                          local_theta, None, None)
    for kind, key in (('full_attention', 'add_full_attention_sink_bias'),
                      ('sliding_attention', 'add_swa_attention_sink_bias')):
        if _records.boolean(hf.get(key, False), key) and kind in layers:
            kinds.setdefault(kind, native_fields(LayerKind)())['sinks'] = True
    if 'sliding_attention' in kinds:
        # A windowed kind rotates whole heads unless it states its own
        # factor (`LayerKind`); MiMoV2Attention turns the same third of both.
        kinds['sliding_attention'].update(num_kv_heads=swa_kv_heads, partial_rotary_factor=factor)
    # A prediction depth attends with the sliding layers' kind, or where no trunk layer slides with a kind
    # of its own of the sliding geometry (its query and key heads, width, window, base, sinks and rotary),
    # over a dense feed-forward at `intermediate_size`. A checkpoint without its tensors holds none
    # (`hf_decoders.held_depths`).
    attending = 'sliding_attention' if slides else _PREDICTION_ATTENTION
    if depths and not slides:
        predictor = native_fields(LayerKind)(
            window=None if window is None else _records.integer(window, 'sliding_window'),
            rope_theta=local_theta, num_heads=swa['swa_num_attention_heads'][0], head_dim=swa_head_dim,
            num_kv_heads=swa_kv_heads, partial_rotary_factor=factor)
        if _records.boolean(hf.get('add_swa_attention_sink_bias', False), 'add_swa_attention_sink_bias'):
            predictor['sinks'] = True
        kinds[_PREDICTION_ATTENTION] = predictor
    if depths:
        kinds[_PREDICTION_FEEDFORWARD] = native_fields(LayerKind)(mixer=MLPMixer(
            intermediate_size=_records.integer(config['mlp_features'], 'intermediate_size'),
            activation=_records.text(config['mlp'], 'hidden_act')))
    # Each depth's residual plain, as a record a load keeps the trunk alone of states it too.
    config.update(num_nextn_predict_layers=depths, mtp_layer_type=attending if depths else None,
                  mtp_feedforward_type=_PREDICTION_FEEDFORWARD if depths else None,
                  mtp_hyper_connections=None)
    return _attention_and_experts(config, hf, used, kinds, value_dim=value_dim,
                                  scale=None, factor=factor, count=count,
                                  sparse=sparse)


def _prediction_geometry(name: str, config: Mapping[str, object]) -> Mapping[str, object]:
    """The geometry a fused projection's runs take for tensor `name`: a prediction layer's own kind's
    query and key heads and width where it states them (`_PREDICTION_ATTENTION`), else the model's."""
    if _prediction_depth(name, config) is None or config.get('mtp_layer_type') != _PREDICTION_ATTENTION:
        return config
    kind = _records.record(_records.record(config['kinds'], 'kinds')[_PREDICTION_ATTENTION], 'kind')
    return {**config, **{field: kind[field] for field in ('num_heads', 'head_dim', 'num_kv_heads')
                         if kind.get(field) is not None}}


_FUSED_QKV = grouped_qkv('.self_attn.qkv_proj', owner=_prediction_geometry)

_PREDICTION_FEEDFORWARD = 'prediction_feedforward'
"""The kind whose MLP mixer is the prediction depth's dense feed-forward, in no layer of the trunk."""

_PREDICTION_ATTENTION = 'prediction_attention'
"""The kind of a prediction depth's attention where no trunk layer slides: the sliding geometry the config
states, in no layer of the trunk."""

_PREDICTION_LEAVES = {**PREDICTION_INPUTS, **PREDICTION_STEP_END}
"""A prediction depth's own leaves by their names past `model.mtp.layers.{d}.` (vLLM's
`MiMoV2MTPLayer`): its normed inputs, their projection and its output norm, `final_layernorm`."""

_PREDICTION_NAMES = (('pre_mlp_layernorm', 'post_attention_layernorm'),)
"""A prediction depth's block norm under vLLM's name for it, and the trunk layer's."""


def _prediction_depth(name: str, config: Mapping[str, object]) -> int | None:
    """The prediction depth of a `model.mtp.layers.{d}.*` tensor, or None for any other."""
    parts = name.split('.')
    if len(parts) > 4 and parts[:3] == ['model', 'mtp', 'layers'] and parts[3].isdigit():
        return int(parts[3])
    return None


def _mimo_v2_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """The shared map under the family's names; a prediction depth's tensors under its own leaves, its
    block's as the trunk's layer 0's (`prediction_path`), or no place for a model that builds none; and
    no place for the towers and the audio tokenizer's folder, which the text decoder never runs."""
    depth = _prediction_depth(name, config)
    if depth is not None:
        declared = _records.integer(config.get('num_nextn_predict_layers', 0), 'num_nextn_predict_layers')
        if not declared:
            return None
        tail = '.'.join(name.split('.')[4:])

        def block(tail: str) -> tuple[str, ...] | None:
            for vllm, trunk in _PREDICTION_NAMES:
                tail = tail.replace(vllm, trunk)
            return _mimo_v2_path(f'model.layers.0.{tail}', config)
        return prediction_path(name, 'mtp', depth, declared, tail, _PREDICTION_LEAVES, block)
    if name.startswith(('visual.', 'audio_encoder.', 'speech_embeddings.',
                        folder_name(_AUDIO_TOKENIZER, ''))):
        return None
    return renamed_path(_NAMES, name, config)


def _mimo_v2_name(dew_name: str, config: Mapping[str, object]) -> str | None:
    """`_mimo_v2_path` backwards: a prediction depth's leaf under its `model.mtp.layers.{d}.` name, any
    other under the shared map's."""
    parts = dew_name.split('.')
    if not parts[0].startswith('mtp_'):
        return renamed_name(_NAMES, dew_name, config)
    stem = f"model.mtp.layers.{parts[0].removeprefix('mtp_')}."
    own = {path: name for name, path in _PREDICTION_LEAVES.items()}.get(tuple(parts[1:]))
    if own is not None:
        return stem + own
    trunk = renamed_name(_NAMES, '.'.join(('layers_0', *parts[2:])), config)
    if trunk is None:
        return None
    tail = trunk.removeprefix('model.layers.0.')
    for vllm, layer in _PREDICTION_NAMES:
        tail = tail.replace(layer, vllm)
    return stem + tail


def _mimo_v2_wrapper(hf_config: Mapping[str, object], used: set[str]) -> WrapperFields:
    """Read a MiMo-V2.6 bundle (MiMoV2ForCausalLM at 6a7f202): its decoder at the config's root, the ViT
    and its merger (`visual.`) and the audio encoder (`audio_encoder.`, `speech_embeddings.`) over the
    codes of the tokenizer in its folder, whose config the source carries beside its own
    (`DecoderFamily.wrapper_folders`). Images, a video's frames and audio clips fill their placeholders
    (`_get_multimodal_embeds`), each token one position on (`MiMoV2Model.forward`); the encoder projects
    to the decoder's width itself, through no projector of the wrapper's."""
    text = translated(hf_config, MIMO_V2)
    # The decoder's translation read every field at the root or refused it.
    used.update(hf_config)
    width = record_int(text, 'emb_features')
    # A section that states nothing builds no tower (MiMoV2ForCausalLM tests each for truth).
    vision = bool(hf_config.get('vision_config'))
    fields: WrapperFields = {
        'model_type': 'mimo_v2', 'text_model_type': 'mimo_v2', 'text': text,
        'tower': translate_mimo_vision_config(hf_config) if vision else None,
        'projector': translate_mimo_projector_config(hf_config, width) if vision else None,
        'image_token_id': (record_int(hf_config, 'image_token_id')
                           if vision or hf_config.get('image_token_id') is not None else None),
        'tokens_per_image': None,
        'video_token_id': (None if hf_config.get('video_token_id') is None
                           else record_int(hf_config, 'video_token_id')),
        'extra_placeholder_ids': (), 'text_unprefixed': ('embed_tokens.', 'layers.', 'norm.', 'mtp.'),
        **NO_AUDIO}
    stated = hf_config.get('audio_config') or None
    tokenizer = hf_config.get(folder_config(_AUDIO_TOKENIZER))
    if stated is not None and tokenizer is None:
        # The language model alone loads (`text_only`); a whole load is refused by the encoder's name.
        fields['unported'] = dict.fromkeys(('audio_encoder.', 'speech_embeddings.'),
                                           f"the audio encoder reads the codes of the tokenizer in the "
                                           f"release's {_AUDIO_TOKENIZER} folder, which this source lacks")
    elif stated is not None:
        record = translate_mimo_audio_config(_records.record(stated, 'audio_config'),
                                             _records.record(tokenizer, folder_config(_AUDIO_TOKENIZER)))
        projected = record_int(_records.record(record['fields'], 'audio fields'), 'out_hidden_size')
        if projected != width:
            refuse(f'audio_config.out_hidden_size {projected}', f'the decoder embeds {width} wide')
        fields['audio'] = record
        fields['audio_token_id'] = record_int(hf_config, 'audio_token_id')
    return fields


def _mimo_v2_export(model: CausalTransformer) -> Mapping[str, object]:
    """The authors' own config (MiMoV2Config, configuration_mimo_v2.py at 6a7f202) of a MiMo-V2 model: each
    layer's kind and routing as its flags (`hybrid_layer_pattern`, `moe_layer_freq`), each kind's key heads,
    base and sinks in the `swa_*` and sink fields, the epsilon under its name, the q, k and v projections
    stored apart ('split'); the shared fields it does not read (`layer_types`, `rms_norm_eps`,
    `rope_local_base_freq`) written as nothing."""
    types = model.per_layer_types
    # The sliding geometry: the sliding layers', or where none slides, a prediction depth's own.
    stating = ('sliding_attention' if 'sliding_attention' in types
               else model.mtp_layer_type if model.num_nextn_predict_layers else None)
    sliding = None if stating is None else model.kind_of(stating)
    full = model.kind_of('full_attention') if 'full_attention' in types else None
    mixture = model.mixture
    return {
        'layer_types': None, 'rms_norm_eps': None, 'rope_local_base_freq': None,
        'layernorm_epsilon': model.norm_eps,
        'hybrid_layer_pattern': [int(kind == 'sliding_attention') for kind in types],
        'moe_layer_freq': [int(mixture is not None and index in model.sparse_layers)
                           for index in range(model.num_layers)],
        # A sliding kind's geometry, every field stated: its value width defaults to its key width.
        **({} if sliding is None else {
            'swa_num_attention_heads': sliding.num_heads or model.num_heads,
            'swa_head_dim': sliding.head_dim or model.features_per_head,
            'swa_v_head_dim': model.value_head_dim, 'swa_num_key_value_heads': sliding.num_kv_heads}),
        'swa_rope_theta': None if sliding is None else sliding.rope_theta,
        'sliding_window_size': None if sliding is None else sliding.window,
        'add_full_attention_sink_bias': full is not None and full.sinks,
        'add_swa_attention_sink_bias': sliding is not None and sliding.sinks,
        'v_head_dim': model.value_head_dim, 'attention_value_scale': model.value_scale,
        'partial_rotary_factor': model.partial_rotary_factor or 1.0,
        'attention_projection_layout': 'split', 'scoring_func': 'sigmoid', 'topk_method': 'noaux_tc',
        'attention_dropout': model.attention_dropout_rate,
        # The depths it holds, which the release's config does not state: a reader takes one where none is.
        'num_nextn_predict_layers': model.num_nextn_predict_layers, **_experts(model)}


MIMO_V2 = DecoderFamily(
    ('mimo_v2',), _mimo_v2_config, _matches, 'mimo_v2', 'MiMoV2ForCausalLM',
    _mimo_v2_export, weight_path=_mimo_v2_path, export_path=_mimo_v2_name,
    prediction_depth=_prediction_depth,
    packed=_FUSED_QKV, packs=lambda config: config.get('attention_projection_layout') == 'fused_qkv',
    preserve_source_layout=True, release_code=SourceCode(*release_pins.MIMO_V2),
    wrapper=_mimo_v2_wrapper, wrapper_folders=(_AUDIO_TOKENIZER,),
)
