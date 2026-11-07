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

import re
from collections.abc import Callable, Mapping
from types import MappingProxyType

from dew import _records
from dew.interop import release_pins
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.decoder_config import DecoderSpelling, decoder_layout, integer, plain_rope_part
from dew.interop.decoder_parts import (
    NO_AUDIO,
    PREDICTION_INPUTS,
    PREDICTION_STEP_END,
    DecoderFamily,
    DecoderFields,
    Ramp,
    Renames,
    SourceCode,
    WrapperFields,
    declared_depths,
    folder_config,
    folder_name,
    grouped_qkv,
    kinds_of,
    nested_ramp,
    partial_rope_entry,
    prediction_routes,
    record_int,
    refuse,
    spelled,
    translated,
)
from dew.interop.families.deepseek import V3_ROUTER, deepseek_experts
from dew.interop.layouts import (
    INDEX,
    Config,
    ConfigField,
    ConfigLayout,
    ConfigPart,
    Route,
    Slot,
    Spelled,
    Table,
    When,
    unread,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mimo_audio import AUDIO_TOKENIZER, translate_mimo_audio_config
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


def _attention(*, scale: float | None) -> tuple[ConfigField[CausalTransformer], ...]:
    """What both MiMo-V2 configs state of the attention alike: the value heads' scale
    (`attention_value_scale`, `scale` where the config omits it) and the attention dropout."""
    return (ConfigField('attention_value_scale', 'value_scale', scale,
                        parse=lambda value, key: None if value is None else _records.number(value, key)),
            ConfigField('attention_dropout', 'attention_dropout_rate', 0.0, parse=_records.number))


def _experts(routed: Callable[[Mapping[str, object], set[str]], tuple[int, ...]],
             schedule: Callable[[CausalTransformer], Mapping[str, object]]
             ) -> ConfigLayout[CausalTransformer]:
    """Both MiMo-V2 configs' routed experts, both ways: DeepSeek V3's (`deepseek_experts`: sigmoid scores, a
    balancing bias, grouped choice and renormalized top-k weights) on the layers the config's schedule
    routes (`routed`, written back by `schedule`). A config that states no experts, or experts no layer
    routes to, builds dense layers alone, as the reference does. Their shared experts are none, which the
    releases state null."""
    def placed(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        layers = routed(hf_config, used)
        if record.get('mixture') is not None:
            record['mixture'] = NativeFields(Mixture, {**_records.record(record['mixture'], 'mixture'),
                                                       'layers': layers})
        return record

    def present(hf_config: Mapping[str, object]) -> bool:
        return hf_config.get('n_routed_experts') is not None and bool(routed(hf_config, set()))
    return ConfigLayout(deepseek_experts(V3_ROUTER, present),
                        ConfigPart(placed, lambda model: {**schedule(model), 'n_shared_experts': None}))


_MIMO_V2_FLASH_ROUTES = Table(
    # The released checkpoints' `model.mtp.*` prediction layers, which the reference never runs
    # (`_keys_to_ignore_on_load_unexpected`), have no place.
    Route('model.mtp.{rest}', None, MappingProxyType({'rest': Slot(r'.+')})),
    spelled(_NAMES))
"""MiMo-V2-Flash's names: the shared map under the family's (`_NAMES`)."""


def _matches(fields: CausalTransformer) -> bool:
    """Values narrower than the heads, which no other family's attention has: MiMo-V2's geometry."""
    return fields.value_head_dim is not None and fields.value_head_dim != fields.features_per_head


def _flash_matches(fields: CausalTransformer) -> bool:
    """MiMo-V2's geometry as MiMo-V2-Flash fixes it (MiMoV2FlashAttention): no sinks on the full layers,
    and the sliding layers' sinks over twice the full layers' key heads, their query heads and width the
    model's, which its config states no other of. Another holds the authors' own config (`MIMO_V2`)."""
    if not _matches(fields):
        return False
    types = set(fields.per_layer_types)
    if 'full_attention' in types and fields.kind_of('full_attention').sinks:
        return False
    sliding = fields.kind_of('sliding_attention') if 'sliding_attention' in types else None
    return sliding is None or (sliding.sinks and sliding.num_kv_heads == 2 * fields.kv_heads
                               and sliding.num_heads == fields.num_heads
                               and sliding.head_dim == fields.features_per_head)


def _check_released(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """Accept the authors' own fields where they state what transformers
    computes from its defaults, and refuse any that state another model."""
    layers = _records.strings(record['layer_types'], 'layer_types')
    sparse = _flash_sparse(hf_config, len(layers))
    expected = {
        'hybrid_layer_pattern': [0 if kind == 'full_attention' else 1 for kind in layers],
        'moe_layer_freq': [int(index in sparse) for index in range(len(layers))],
        'add_swa_attention_sink_bias': True, 'add_full_attention_sink_bias': False,
        'swa_num_attention_heads': record['num_heads'], 'swa_num_key_value_heads': 2 * _kv_heads(record),
        'swa_head_dim': record['head_dim'], 'swa_v_head_dim': record['value_head_dim'],
        'sliding_window_size': hf_config.get('sliding_window', 128),
        'layernorm_epsilon': hf_config.get('rms_norm_eps', 1e-5),
        'n_shared_experts': None, 'scoring_func': 'sigmoid', 'topk_method': 'noaux_tc',
    }
    for key, value in expected.items():
        if key in hf_config and hf_config[key] != value:
            refuse(f'{key}={hf_config[key]!r}', f'transformers computes {value!r} from MiMoV2FlashConfig')
    # The released configs state the window again as a chunk the reference
    # never reads; attention stays sliding, not chunked.
    used.update((*expected, 'attention_chunk_size', 'swa_rope_theta'))
    return record


def _kv_heads(record: DecoderFields) -> int:
    return _records.integer(record['num_kv_heads'], 'num_kv_heads')


def _flash_layers(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """Each layer's attention: `layer_types`, else MiMoV2FlashConfig's pattern (`_default_layers`)."""
    used.add('layer_types')
    count = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    stated = hf_config.get('layer_types')
    layers = _default_layers(count) if stated is None else _records.strings(stated, 'layer_types')
    if set(layers) - set(_KINDS) or len(layers) != count:
        refuse(f'layer_types {list(layers)!r}', 'one full_attention or sliding_attention entry per layer')
    record['layer_types'] = layers
    return record


def _flash_sparse(hf_config: Mapping[str, object], count: int) -> tuple[int, ...]:
    """The layers that route: `mlp_layer_types`, else every layer but the first."""
    schedule = (_records.strings(hf_config['mlp_layer_types'], 'mlp_layer_types')
                if hf_config.get('mlp_layer_types') is not None else ('dense',) + ('sparse',) * (count - 1))
    if set(schedule) - {'dense', 'sparse'} or len(schedule) != count:
        refuse(f'mlp_layer_types {list(schedule)!r}', 'one dense or sparse entry per layer')
    return tuple(index for index, kind in enumerate(schedule) if kind == 'sparse')


def _flash_rope(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """Each kind's base, rotated fraction and ramp (`_rope`): the full layers' the model's, the sliding
    layers' their kind's."""
    used.update(('rope_parameters', 'rope_theta', 'partial_rotary_factor'))
    layers = _records.strings(record['layer_types'], 'layer_types')
    (full_theta, full_factor, full_ramp), (local_theta, local_factor, local_ramp) = (
        _rope(hf_config, kind) for kind in _KINDS)
    record.update(rope_theta=full_theta, partial_rotary_factor=None if full_factor == 1.0 else full_factor,
                  partial_rotary_type='default')
    kinds = kinds_of(record)
    if 'sliding_attention' in layers:
        kinds['sliding_attention'].update(rope_theta=local_theta, partial_rotary_factor=local_factor)
    for kind, ramp in zip(_KINDS, (full_ramp, local_ramp), strict=True):
        if ramp is not None and kind in layers:
            kinds.setdefault(kind, native_fields(LayerKind)())['rope_scaling'] = ramp
    record['kinds'] = kinds
    return record


def _flash_rope_written(model: CausalTransformer) -> Mapping[str, object]:
    resolved = {kind: model.kind_of(kind) for kind in _KINDS}
    return {'rope_parameters': {kind: {'rope_type': 'default', 'rope_theta': held.rope_theta,
                                       'partial_rotary_factor': held.partial_rotary_factor or 1.0}
                                for kind, held in resolved.items()}}


def _flash_sinks(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """MiMoV2FlashAttention doubles a sliding layer's key and value heads and gives it the sinks
    (modeling_mimo_v2_flash.py:348, 363)."""
    kinds = kinds_of(record)
    if 'sliding_attention' in kinds:
        kinds['sliding_attention'].update(num_kv_heads=2 * _kv_heads(record), sinks=True)
    record['kinds'] = kinds
    return record


def _flash_routed(hf_config: Mapping[str, object], used: set[str]) -> tuple[int, ...]:
    used.add('mlp_layer_types')
    return _flash_sparse(hf_config, _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers'))


def _flash_schedule(model: CausalTransformer) -> Mapping[str, object]:
    routed = set(model.sparse_layers) if model.mixture is not None else set()
    return {'mlp_layer_types': ['sparse' if index in routed else 'dense'
                                for index in range(model.num_layers)]}


MIMO_V2_FLASH_CONFIG = ConfigLayout(
    ConfigPart(_flash_layers, lambda model: {'layer_types': list(model.per_layer_types)}),
    decoder_layout(DecoderSpelling(reads=frozenset({'sliding_window', 'attention_bias'})),
                   rope=ConfigPart(_flash_rope, _flash_rope_written)),
    integer('v_head_dim', 'value_head_dim', 128), *_attention(scale=0.707),
    ConfigPart(_check_released, lambda model: {}),
    ConfigPart(_flash_sinks, lambda model: {}),
    _experts(_flash_routed, _flash_schedule))
"""MiMo-V2-Flash's config, both ways."""


MIMO_V2_FLASH = DecoderFamily(
    ('mimo_v2_flash',), MIMO_V2_FLASH_CONFIG,
    _flash_matches, 'mimo_v2_flash', 'MiMoV2FlashForCausalLM', routes=_MIMO_V2_FLASH_ROUTES,
    # Its experts are stored one tensor each, so a save writes them back through the source's own names.
    preserve_source_layout=True,
    padding_idx=True,
    # MiMoV2FlashConfig's defaults for the fields its decoder reads.
    defaults={'max_position_embeddings': 131072, 'head_dim': 192, 'num_key_value_heads': 4,
              'sliding_window': 128, 'rms_norm_eps': 1e-5},
)


_V2_FIELDS = ('swa_num_attention_heads', 'swa_num_key_value_heads', 'swa_head_dim', 'swa_v_head_dim',
              'swa_rope_theta', 'sliding_window_size', 'layernorm_epsilon', 'add_full_attention_sink_bias',
              'add_swa_attention_sink_bias', 'attention_projection_layout', 'attention_value_scale',
              'v_head_dim', 'partial_rotary_factor', 'rope_parameters', 'rope_theta', 'norm_topk_prob',
              'scoring_func', 'topk_method', 'attention_dropout')
_AUDIO_TOKENIZER = AUDIO_TOKENIZER

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


def _layer_geometry(name: str, config: Mapping[str, object]) -> Mapping[str, object]:
    """The geometry a fused projection's runs take for tensor `name`: its layer's kind's query and key
    heads and width where the kind states them, a sliding layer's (`swa_*`) or a prediction layer's
    (`mtp_layer_type`), else the model's (ledger 4290)."""
    depth, parts = _prediction_depth(name, config), name.split('.')
    if depth is not None:
        kind = _records.text(config['mtp_layer_type'], 'mtp_layer_type')
    elif (len(parts) > 3 and parts[:2] == ['model', 'layers'] and parts[2].isdigit()
          and config.get('layer_types') is not None):
        kind = _records.strings(config['layer_types'], 'layer_types')[int(parts[2])]
    else:
        # Outside a layer, or a geometry stating no layer's kind: every layer's is the model's.
        return config
    kinds = _records.record(config.get('kinds') or {}, 'kinds')
    if kind not in kinds:
        return config
    held = _records.record(kinds[kind], 'kind')
    return {**config, **{field: held[field] for field in ('num_heads', 'head_dim', 'num_kv_heads')
                         if held.get(field) is not None}}


_FUSED_QKV = grouped_qkv('.self_attn.qkv_proj', owner=_layer_geometry)

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


def _builds_depths(config: Config) -> bool:
    return bool(config.get('num_nextn_predict_layers'))


_MIMO_V2_ROUTES = Table(
    When(_builds_depths, prediction_routes(
        'model.mtp.layers.{depth}.', 'mtp', _PREDICTION_LEAVES, Spelled(_PREDICTION_NAMES, spelled(_NAMES)),
        slots={'depth': INDEX}, when=declared_depths())),
    # A model that builds no prediction depth places none of the release's.
    When(lambda config: not _builds_depths(config),
         Route('model.mtp.layers.{rest}', None, MappingProxyType({'rest': Slot(r'.+')}))),
    # The towers and the audio tokenizer's folder, which the text decoder never runs.
    Route('{media}', None, MappingProxyType({'media': Slot(
        r'(?:visual|audio_encoder|speech_embeddings)\..+|'
        + re.escape(folder_name(_AUDIO_TOKENIZER, '')) + '.+')})),
    spelled(_NAMES))
"""MiMo-V2's names: the shared map under the family's (`_NAMES`), and a prediction depth's tensors under
`model.mtp.layers.{d}.` (vLLM's `MiMoV2MTPLayer`): its own leaves, and its block's as the trunk's layer 0's,
its block norm under vLLM's name (`_PREDICTION_NAMES`)."""


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


def _v2_layers(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    sliding, _ = _layer_flags(hf_config, used, _records.integer(hf_config['num_hidden_layers'],
                                                                'num_hidden_layers'))
    record['layer_types'] = tuple('sliding_attention' if slides else 'full_attention' for slides in sliding)
    return record


def _v2_layers_written(model: CausalTransformer) -> Mapping[str, object]:
    return {'hybrid_layer_pattern': [int(kind == 'sliding_attention') for kind in model.per_layer_types],
            'moe_layer_freq': [int(model.mixture is not None and index in model.sparse_layers)
                               for index in range(model.num_layers)]}


def _v2_values(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The value heads' width, the key heads' where unstated."""
    used.add('v_head_dim')
    record['value_head_dim'] = _records.integer(hf_config.get('v_head_dim') or record['head_dim'],
                                                'v_head_dim')
    return record


def _v2_kinds(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """Each kind's geometry, sinks and base (MiMoV2Config, configuration_mimo_v2.py:156-232): the sliding
    layers' query and key heads and width from the `swa_*` fields, the model's where unstated, and the
    prediction depths that attend with them.

    The prediction depths: the count a config states (Dew's export of one, 0 where it holds none), else
    the one depth the release's code skips on load and vLLM drafts with (mimo_v2_mtp.py at 19aba3e,
    `MiMoV2MTPLayer`, one layer for the Pro and the Flash alike) where its trunk has sliding layers, as
    every release's does. Each attends with the sliding geometry the config states, whatever the trunk's
    pattern, which the sliding layers read too."""
    used.update((*_V2_FIELDS, 'num_nextn_predict_layers'))
    layers = _records.strings(record['layer_types'], 'layer_types')
    heads, kv_heads = _records.integer(record['num_heads'], 'num_heads'), _kv_heads(record)
    head_dim = _records.integer(record['head_dim'], 'head_dim')
    swa_heads = _records.integer(hf_config.get('swa_num_attention_heads') or heads, 'swa_num_attention_heads')
    swa_head_dim = _records.integer(hf_config.get('swa_head_dim') or head_dim, 'swa_head_dim')
    swa_kv_heads = _records.integer(hf_config.get('swa_num_key_value_heads') or kv_heads,
                                    'swa_num_key_value_heads')
    slides = 'sliding_attention' in layers
    depths = _records.integer(hf_config.get('num_nextn_predict_layers', int(slides)),
                              'num_nextn_predict_layers')
    layout = hf_config.get('attention_projection_layout') or 'split'
    if layout not in ('split', 'fused_qkv'):
        refuse(f'attention_projection_layout {layout!r}', "MiMoV2Attention stores 'split' or 'fused_qkv'")
    # The sliding layers' kind, and a prediction layer's own, takes its query and key heads and width; its
    # values are the model's, which no kind sets apart.
    values = _records.integer(record['value_head_dim'], 'value_head_dim')
    stated = _records.integer(hf_config.get('swa_v_head_dim') or swa_head_dim, 'swa_v_head_dim')
    if stated != values and (slides or depths):
        refuse(f'swa_v_head_dim {stated}', f"a sliding or prediction layer's values are the model's {values} "
                                           'wide')
    if ((hf_config.get('scoring_func') or 'sigmoid', hf_config.get('topk_method') or 'noaux_tc')
            != ('sigmoid', 'noaux_tc')):
        refuse('scoring_func and topk_method', "MiMoV2MoEGate implements sigmoid scores under 'noaux_tc'")
    window = hf_config.get('sliding_window') or hf_config.get('sliding_window_size')
    factor = _records.number(record['partial_rotary_factor'] or 1.0, 'partial_rotary_factor')
    local_theta = _records.number(hf_config.get('swa_rope_theta') or record['rope_theta'], 'swa_rope_theta')
    kinds = kinds_of(record)
    for kind, key in (('full_attention', 'add_full_attention_sink_bias'),
                      ('sliding_attention', 'add_swa_attention_sink_bias')):
        if _records.boolean(hf_config.get(key, False), key) and kind in layers:
            kinds.setdefault(kind, native_fields(LayerKind)())['sinks'] = True
    if slides:
        # A windowed kind rotates whole heads unless it states its own factor (`LayerKind`); MiMoV2Attention
        # turns the same third of both. Its query heads and width are the `swa_*` geometry, which a fused
        # source's runs read by layer (`_layer_geometry`).
        kinds['sliding_attention'].update(rope_theta=local_theta, num_kv_heads=swa_kv_heads,
                                          partial_rotary_factor=factor, num_heads=swa_heads,
                                          head_dim=swa_head_dim)
    # A prediction depth attends with the sliding layers' kind, or where no trunk layer slides with a kind
    # of its own of the sliding geometry (its query and key heads, width, window, base, sinks and rotary),
    # over a dense feed-forward at `intermediate_size`. A checkpoint without its tensors holds none
    # (`hf_decoders.held_depths`).
    attending = 'sliding_attention' if slides else _PREDICTION_ATTENTION
    if depths and not slides:
        predictor = native_fields(LayerKind)(
            window=None if window is None else _records.integer(window, 'sliding_window'),
            rope_theta=local_theta, num_heads=swa_heads, head_dim=swa_head_dim, num_kv_heads=swa_kv_heads,
            partial_rotary_factor=factor)
        if _records.boolean(hf_config.get('add_swa_attention_sink_bias', False),
                            'add_swa_attention_sink_bias'):
            predictor['sinks'] = True
        kinds[_PREDICTION_ATTENTION] = predictor
    if depths:
        kinds[_PREDICTION_FEEDFORWARD] = native_fields(LayerKind)(mixer=MLPMixer(
            intermediate_size=_records.integer(record['mlp_features'], 'intermediate_size'),
            activation=_records.text(record['mlp'], 'hidden_act')))
    # Each depth's residual plain, as a record a load keeps the trunk alone of states it too.
    record.update(kinds=kinds, num_nextn_predict_layers=depths, mtp_layer_type=attending if depths else None,
                  mtp_feedforward_type=_PREDICTION_FEEDFORWARD if depths else None,
                  mtp_hyper_connections=None)
    return record


def _v2_kinds_written(model: CausalTransformer) -> Mapping[str, object]:
    """Each kind's key heads, base and sinks in the `swa_*` and sink fields, the q, k and v projections
    stored apart ('split'), and the depths it holds, which the release's config does not state: a reader
    takes one where none is."""
    types = model.per_layer_types
    # The sliding geometry: the sliding layers', or where none slides, a prediction depth's own.
    stating = ('sliding_attention' if 'sliding_attention' in types
               else model.mtp_layer_type if model.num_nextn_predict_layers else None)
    sliding = None if stating is None else model.kind_of(stating)
    full = model.kind_of('full_attention') if 'full_attention' in types else None
    return {
        # A sliding kind's geometry, every field stated: its value width defaults to its key width.
        **({} if sliding is None else {
            'swa_num_attention_heads': sliding.num_heads or model.num_heads,
            'swa_head_dim': sliding.head_dim or model.features_per_head,
            'swa_v_head_dim': model.value_head_dim, 'swa_num_key_value_heads': sliding.num_kv_heads}),
        'swa_rope_theta': None if sliding is None else sliding.rope_theta,
        'sliding_window_size': None if sliding is None else sliding.window,
        'add_full_attention_sink_bias': full is not None and full.sinks,
        'add_swa_attention_sink_bias': sliding is not None and sliding.sinks,
        'attention_projection_layout': 'split', 'scoring_func': 'sigmoid', 'topk_method': 'noaux_tc',
        'num_nextn_predict_layers': model.num_nextn_predict_layers}


def _v2_routed(hf_config: Mapping[str, object], used: set[str]) -> tuple[int, ...]:
    layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    return tuple(index for index, routes in enumerate(_layer_flags(hf_config, used, layers)[1]) if routes)


MIMO_V2_CONFIG = ConfigLayout(
    ConfigPart(_v2_layers, _v2_layers_written),
    # MiMoV2Config's geometry and its defaults (configuration_mimo_v2.py:156-197): the epsilon under its
    # own name, the window under either.
    decoder_layout(DecoderSpelling(reads=frozenset({'sliding_window', 'attention_bias'}),
                                   windows=('sliding_window', 'sliding_window_size'),
                                   keys={'rms_norm_eps': 'layernorm_epsilon'},
                                   defaults={'layernorm_epsilon': 1e-6, 'num_attention_heads': 32}),
                   rope=plain_rope_part(default_factor=1.0, reference='MiMoV2RotaryEmbedding')),
    ConfigPart(_v2_values, lambda model: {'v_head_dim': model.value_head_dim}), *_attention(scale=None),
    ConfigPart(_v2_kinds, _v2_kinds_written),
    # The routed layers' schedule is the layers' own (`_v2_layers_written`).
    _experts(_v2_routed, lambda model: {}), unread(*_V2_UNREAD))
"""MiMo-V2.6's config, the authors' own (MiMoV2Config, configuration_mimo_v2.py at 6a7f202), both ways."""


MIMO_V2 = DecoderFamily(
    ('mimo_v2',), MIMO_V2_CONFIG, _matches, 'mimo_v2',
    'MiMoV2ForCausalLM', routes=_MIMO_V2_ROUTES,
    prediction_depth=_prediction_depth,
    packed=_FUSED_QKV, packs=lambda config: config.get('attention_projection_layout') == 'fused_qkv',
    preserve_source_layout=True, release_code=SourceCode(*release_pins.MIMO_V2),
    wrapper=_mimo_v2_wrapper, wrapper_folders=(_AUDIO_TOKENIZER,),
)
