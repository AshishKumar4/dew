"""Translate the DeepSeek decoders: V2, V3, V3.2, V4 and the Kimi releases.

V2 and V3 differ in how the MoE sizes itself and whether the router carries a
bias. V3.2 adds the sparse indexer over the same MLA block. V4 is its own
layout: three attention kinds, each naming its compressor and its pooling
rate, mHC's residual streams, and hash-routed first layers. Kimi K2 is V3's
computation under its own provenance, and Kimi K2.5 the same decoder nested
in a vision wrapper.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import Literal

import numpy as np

from dew import _records
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.decoder_config import (
    ROPE_APART,
    TRAINING_ROUTER,
    DecoderSpelling,
    decoder_layout,
    expect,
    experts,
    flag,
    integer,
    place_ropes,
    read_only,
    rope_part,
    stated_experts,
    unwritten,
    written_ropes,
)
from dew.interop.decoder_parts import (
    DECODER_ROUTES,
    DEEPSEEK_ROUTES,
    DEFAULT_MAX_SEQ_LEN,
    MOE_NAMES,
    NO_AUDIO,
    PLAIN_SPELLING,
    SERIALIZED_ENCODER_FIELDS,
    SERIALIZED_TEXT_FIELDS,
    DecoderFamily,
    DecoderFields,
    HyperConnectionsFields,
    KindFields,
    Leaves,
    MixtureFields,
    Ramp,
    RopeSpelling,
    WeightPreparer,
    WrapperFields,
    declared_depths,
    in_trunk,
    layer_depth,
    prediction_routes,
    read_partial_rope,
    record_float,
    record_int,
    refuse,
    refuse_encoder_fields,
    refuse_export,
    spelled,
    translated,
    yarn_record,
)
from dew.interop.layouts import (
    INDEX,
    REQUIRED,
    Config,
    ConfigEntry,
    ConfigField,
    ConfigLayout,
    ConfigPart,
    Nested,
    Prefixed,
    Route,
    Routes,
    Slot,
    Table,
    constant,
    mapped,
    one_of,
    unread,
)
from dew.interop.unread import Stored, held, valued
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.deepseek_v4 import DeepseekV4Mixer
from dew.nn.dspark import DSpark
from dew.nn.hyper_connections import HyperConnections
from dew.nn.mla import MLAMixer
from dew.nn.vision.moonvit import translate_moonvit_config, translate_moonvit_projector_config
from dew.registry import Configured

# MiniMaxM2Config's default_theta (configuration_minimax_m2.py:74).
_MINIMAX_M2_SPELLING = RopeSpelling(theta=5000000.0)


def _minimax_rope(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """MiniMax-M2's partial rotary, read as its release states it: under rope_scaling, which transformers
    reads as rope_parameters and MiniMaxM2RotaryEmbedding builds through its rope_type
    (modeling_minimax_m2.py:150-165), over rotary_dim of each head (transformers 5.16.1 predates the
    rotary_dim fix, issue 48241, PR 48486; 5.18.0 and the release rotate rotary_dim/head_dim of each
    head)."""
    head_dim = _records.integer(record['head_dim'], 'head_dim')
    rotary_dim = _records.integer(hf_config.get('rotary_dim', head_dim), 'rotary_dim')
    ropes, factor = read_partial_rope(hf_config, used, _MINIMAX_M2_SPELLING,
                                      default_factor=rotary_dim / head_dim)
    if 'rotary_dim' in hf_config and factor * head_dim != rotary_dim:
        refuse('rotary_dim', 'it disagrees with partial_rotary_factor * head_dim')
    if not 0 < factor <= 1 or int(head_dim * factor) % 2:
        refuse('rotary_dim', 'partial rotary must rotate a positive even head slice')
    used.add('rotary_dim')
    place_ropes(record, ropes)
    record.update(partial_rotary_factor=factor, partial_rotary_type='default')
    return record


def _minimax_rope_written(model: CausalTransformer) -> Mapping[str, object]:
    factor = model.partial_rotary_factor or 1.0
    return {**written_ropes(model, _MINIMAX_M2_SPELLING), 'partial_rotary_factor': factor,
            'rotary_dim': int(model.features_per_head * factor)}


def _full_attention_everywhere(hf_config: Mapping[str, object], used: set[str], record: DecoderFields
                               ) -> DecoderFields:
    used.add('attn_type_list')
    layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    if hf_config.get('attn_type_list', [1] * layers) != [1] * layers:
        refuse('attn_type_list', 'MiniMax-M2 uses full attention in every layer')
    return record


MINIMAX_M2_CONFIG = ConfigLayout(
    *(expect(name, reason, value=expected) for name, expected, reason in (
        ('use_qk_norm', True, 'MiniMax-M2 normalizes its query and key projections'),
        ('qk_norm_type', 'per_layer', 'MiniMax-M2 normalizes whole projections before splitting heads'),
        ('use_routing_bias', True, 'MiniMax-M2 selects experts with its balancing bias'),
        ('scoring_func', 'sigmoid', 'MiniMax-M2 normalizes selected sigmoid probabilities'),
        ('shared_intermediate_size', 0, 'MiniMax-M2 has no shared expert branch'),
        ('router_jitter_noise', 0.0, 'training-time input jitter has no counterpart'),
        ('layernorm_full_attention_beta', 1.0, 'the full-attention norm has no extra multiplier'),
        ('layernorm_linear_attention_beta', 1.0, 'MiniMax-M2 has no linear-attention block'),
        ('layernorm_mlp_beta', 1.0, 'the feed-forward norm has no extra multiplier'))),
    ConfigPart(_full_attention_everywhere, lambda model: {}),
    # These release fields are absent from the reference's config/modeling; none changes the forward pass
    # or supplies weights in the released index, which ships no MTP weights: the weight path refuses any
    # MTP tensor instead of dropping a prediction block.
    unread('use_mtp', 'num_mtp_modules', 'mtp_transformer_layers', 'mlp_intermediate_size',
           'shared_moe_mode'),
    decoder_layout(DecoderSpelling(reads=frozenset(), qk_norm=True, defaults={'head_dim': 128}),
                   rope=ConfigPart(_minimax_rope, _minimax_rope_written)),
    constant('qk_norm_scope', value='projection'),
    # The selection-only bias and normalized sigmoid weights reuse DeepSeek's router without groups,
    # scaling or shared experts.
    experts(integer('num_local_experts', 'experts'), integer('num_experts_per_tok', 'top_k'),
            constant('score_function', value='sigmoid'), constant('bias', value=True)),
    TRAINING_ROUTER)
"""MiniMax-M2's config, both ways: projection norms, partial rotary and a sigmoid MoE."""


# MiniMax's balancing bias, under either of its spellings, ahead of Mixtral's
# MoE names, which would otherwise take its block's part first.
_MINIMAX_M2_NAMES = (
    ('block_sparse_moe.e_score_correction_bias', 'mlp.gate.e_score_correction_bias'),
    ('mlp.e_score_correction_bias', 'mlp.gate.e_score_correction_bias'),
    *MOE_NAMES,
)


def deepseek_routed(hf_config: Mapping[str, object], layers: int, used: set) -> tuple[int, ...]:
    """The layers a DeepSeek MoE routes: those past the first `first_k_dense_replace`.

    Transformers builds every layer past the dense ones as MoE whatever
    `moe_layer_freq` says, so anything but that refuses, as does an
    `mlp_layer_types` that disagrees with it.
    """
    used.update(('first_k_dense_replace', 'moe_layer_freq', 'mlp_layer_types'))
    freq = hf_config.get('moe_layer_freq')
    if freq is not None and freq != 1:
        refuse(f"moe_layer_freq {freq!r}",
                "transformers builds every layer past the dense ones as MoE, "
                "whatever this field says")
    # A dense prefix past the stack leaves every layer dense, as DeepseekV3DecoderLayer builds them.
    first_k = min(_records.integer(hf_config.get('first_k_dense_replace', 0) or 0, 'first_k_dense_replace'),
                  layers)
    if first_k < 0:
        refuse(f"first_k_dense_replace {first_k!r}",
                f"it names dense layers of a {layers}-layer model")
    pattern = hf_config.get('mlp_layer_types')
    if pattern is not None:
        expected = ['dense'] * first_k + ['sparse'] * (layers - first_k)
        if list(_records.strings(pattern, 'mlp_layer_types')) != expected:
            refuse(f"mlp_layer_types {list(_records.strings(pattern, 'mlp_layer_types'))!r}",
                    "it disagrees with first_k_dense_replace, which is what "
                    "the reference builds")
    return tuple(range(first_k, layers))


DEEPSEEK_DECODER = decoder_layout(rope=rope_part(PLAIN_SPELLING, ramps=('yarn', 'llama3'), per_kind=False))
"""The shared decoder as DeepSeek V2's, V3's and V3.2's configs state it, both ways."""


def _or_one(value: Configured, key: str) -> int:
    """A group count, one where a config states it null."""
    return _records.integer(value or 1, key)


def _scaling(value: Configured, key: str) -> float:
    """The routed weights' scale, the reference's 1.0 where a config states it null."""
    return _records.number(1.0 if value is None else value, key)


def _shared_experts(hf_config: Mapping[str, object], used: set[str],
                    record: MixtureFields) -> MixtureFields:
    """The shared experts, a count of the routed width (`n_shared_experts`, none where unstated or null)."""
    used.add('n_shared_experts')
    shared = _records.integer(hf_config.get('n_shared_experts', 0) or 0, 'n_shared_experts')
    record['shared_features'] = shared * _records.integer(record['expert_features'], 'moe_intermediate_size')
    return record


def _shared_count(mixture: Mixture) -> Mapping[str, object]:
    width = mixture.expert_features
    if width is None or mixture.shared_features % width:
        refuse_export('mixture.shared_features', 'DeepSeek counts whole shared experts of the routed width')
    return {'n_shared_experts': mixture.shared_features // width}


def _sigmoid_router(hf_config: Mapping[str, object], used: set[str],
                    record: MixtureFields) -> MixtureFields:
    """DeepSeek V3's router: it selects on the biased sigmoid scores inside the best groups, each scored by
    its two best experts (`noaux_tc`)."""
    used.update(('scoring_func', 'topk_method'))
    scoring = hf_config.get('scoring_func', 'sigmoid')
    if scoring != 'sigmoid':
        refuse(f"scoring_func {scoring!r}", "dew's router scores softmax, sigmoid or sqrtsoftplus, and this "
                                            "family's reference scores sigmoid")
    method = hf_config.get('topk_method')
    if method is not None and method != 'noaux_tc':
        refuse(f"topk_method {method!r}",
               "the reference selects with the bias and the group limit, which is what noaux_tc names")
    record.update(score_function='sigmoid', bias=True)
    return record


V3_ROUTER: tuple[ConfigEntry[Mixture], ...] = (
    ConfigPart(_sigmoid_router, lambda mixture: {'scoring_func': 'sigmoid', 'topk_method': 'noaux_tc'},
               keys=('scoring_func', 'topk_method')),
    expect('norm_topk_prob', 'DeepseekV3TopkRouter always renormalises the top-k weights', value=True,
           written=True))
"""DeepSeek V3's router, both ways: sigmoid scores and their bias (`_sigmoid_router`), the top-k weights
always renormalised. `norm_topk_prob: false` would leave them unnormalised, which the V3 reference never
does; a router that reads the flag (GLM-5's) declares it in place of the expectation."""

GLM_ROUTER: tuple[ConfigEntry[Mixture], ...] = (
    V3_ROUTER[0], flag('norm_topk_prob', 'norm_topk_prob', default=True))
"""DeepSeek V3's router as GLM-5's reads it: the top-k weights renormalised where `norm_topk_prob` says."""


def _softmax_router(hf_config: Mapping[str, object], used: set[str],
                    record: MixtureFields) -> MixtureFields:
    """DeepSeek V2's router: `DeepseekV2TopkRouter` softmaxes the logits, selects greedily or inside the
    best groups scored by their best expert, and never renormalises. It reads no `norm_topk_prob`: the
    released `false` translates and a `true` refuses."""
    used.update(('scoring_func', 'topk_method', 'norm_topk_prob'))
    scoring = hf_config.get('scoring_func', 'softmax')
    if scoring != 'softmax':
        refuse(f"scoring_func {scoring!r}", "DeepseekV2TopkRouter softmaxes its logits")
    if hf_config.get('norm_topk_prob', False):
        refuse("norm_topk_prob=True", "DeepseekV2TopkRouter never renormalises the top-k weights")
    method = hf_config.get('topk_method', 'greedy')
    if method not in ('greedy', 'group_limited_greedy'):
        refuse(f"topk_method {method!r}", "DeepseekV2TopkRouter selects greedy or group_limited_greedy")
    if method == 'greedy' and (record['groups'], record['groups_per_token']) != (1, 1):
        refuse(f"n_group {record['groups']} with topk_method 'greedy'",
               "the greedy selection ignores the groups")
    record.update(score_function='softmax', norm_topk_prob=False, group_score='max')
    return record


V2_ROUTER: tuple[ConfigEntry[Mixture], ...] = (ConfigPart(
    _softmax_router, lambda mixture: {
        'scoring_func': 'softmax', 'norm_topk_prob': False,
        'topk_method': 'greedy' if mixture.groups == 1 else 'group_limited_greedy'},
    keys=('scoring_func', 'topk_method', 'norm_topk_prob')),)
"""DeepSeek V2's router, both ways (`_softmax_router`)."""


def deepseek_experts(router: tuple[ConfigEntry[Mixture], ...], present: Callable[[Mapping[str, object]], bool]
                     | None) -> Nested[CausalTransformer, Mixture]:
    """A DeepSeek MoE's routed experts, both ways: their count, top-k and width, the shared ones as a count
    of that width, the routed weights' scale and groups, and how `router` scores and picks them; none
    where `present`, of the source config, says no layer routes (their keys then describe nothing the
    reference computes). Their layers are the family's schedule's (`DENSE_FIRST`, GLM's and MiMo's own).
    `aux_loss_alpha` and `seq_aux` shape the training loss alone and no forward pass reads them; a run
    sets them on LMObjective, whose `aux_loss_alpha` and `seq_aux` compute V2's balance loss."""
    return Nested('mixture', Mixture, ConfigLayout(
        integer('n_routed_experts', 'experts', REQUIRED, 'num_local_experts'),
        integer('num_experts_per_tok', 'top_k'), integer('moe_intermediate_size', 'expert_features'),
        ConfigPart(_shared_experts, _shared_count, keys=('n_shared_experts',)),
        ConfigField('routed_scaling_factor', 'scaling', 1.0, parse=_scaling),
        ConfigField('n_group', 'groups', 1, parse=_or_one),
        ConfigField('topk_group', 'groups_per_token', 1, parse=_or_one), *router,
        unread('aux_loss_alpha', 'seq_aux')), stated_experts, present=present)


def routes_some(hf_config: Mapping[str, object]) -> bool:
    """Whether a DeepSeek MoE config routes any layer (`deepseek_routed`)."""
    layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    return bool(deepseek_routed(hf_config, layers, set()))


def _dense_first(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    routed = deepseek_routed(hf_config, _records.integer(record['num_layers'], 'num_layers'), used)
    if record.get('mixture') is not None:
        record['mixture'] = NativeFields(Mixture, {**_records.record(record['mixture'], 'mixture'),
                                                   'layers': routed})
    return record


def _dense_first_written(model: CausalTransformer) -> Mapping[str, object]:
    if model.mixture is None:
        return {'first_k_dense_replace': model.num_layers}
    routed = tuple(model.sparse_layers)
    first = model.num_layers - len(routed)
    if routed != tuple(range(first, model.num_layers)):
        refuse_export('mixture.layers', 'DeepSeek routes every layer past its first dense ones')
    return {'first_k_dense_replace': first}


DENSE_FIRST = ConfigPart(_dense_first, _dense_first_written,
                         keys=('first_k_dense_replace', 'moe_layer_freq', 'mlp_layer_types'))
"""A DeepSeek MoE's schedule, both ways: the dense layers before the routed ones (`first_k_dense_replace`,
`deepseek_routed`), every layer where the model routes none."""

V3_EXPERTS = ConfigLayout(deepseek_experts(V3_ROUTER, routes_some), DENSE_FIRST)
"""DeepSeek V3's routed experts and their schedule, both ways."""


def _mla(model: CausalTransformer) -> MLAMixer:
    mixer = model.mixer
    if not isinstance(mixer, MLAMixer):
        refuse_export('mixer', 'a DeepSeek layer attends by multi-head latent attention')
    return mixer


def _optional_integer(value: Configured, key: str) -> int | None:
    return None if value is None else _records.integer(value, key)


def mla_fields(*, sparse: bool) -> ConfigLayout[MLAMixer]:
    """The MLA block of a DeepSeek V2, V3 or V3.2 config, both ways: the latents' ranks, the query and key
    heads' rotated and unrotated widths and the values', and where it selects keys (`sparse`: V3.2's,
    whose reference always rotates interleaved pairs) its lightning indexer's geometry."""
    interleave = (expect('rope_interleave', 'the V3.2 reference always rotates interleaved pairs',
                         value=True, written=True)
                  if sparse else flag('rope_interleave', 'rope_interleave', default=True))
    return ConfigLayout(
        ConfigField('q_lora_rank', 'q_lora_rank', None, parse=_optional_integer),
        integer('kv_lora_rank', 'kv_lora_rank'), integer('qk_nope_head_dim', 'qk_nope_head_dim'),
        integer('qk_rope_head_dim', 'qk_rope_head_dim'), integer('v_head_dim', 'v_head_dim'), interleave,
        *((integer('index_topk', 'index_topk'), integer('index_n_heads', 'index_n_heads'),
           integer('index_head_dim', 'index_head_dim')) if sparse else ()))


def _mla_layers(hf_config: Mapping[str, object], used: set[str], record: DecoderFields, *,
                sparse: bool) -> DecoderFields:
    """Every layer the one attention kind: neither released config names its pattern, V3 dense MLA
    throughout and V3.2 sparse attention throughout."""
    kind = 'deepseek_sparse_attention' if sparse else 'full_attention'
    if hf_config.get('layer_types') is None:
        record['layer_types'] = (kind,) * _records.integer(record['num_layers'], 'num_layers')
    for entry in _records.strings(record['layer_types'], 'layer_types'):
        if entry != kind:
            refuse(f"layer_types entry {entry!r}",
                   f"a {hf_config['model_type']} model mixes no attention kinds: every layer is {kind}")
    return record


def _mla_heads(hf_config: Mapping[str, object], used: set[str], record: DecoderFields, *,
               points_head_dim: bool) -> DecoderFields:
    """The heads' width, the MLA's unrotated and rotated slices together (`qk_head_dim` where stated):
    DeepSeek points `head_dim` at the rope slice (a class that points it whatever the config says,
    `points_head_dim` False, GlmMoeDsaConfig's, reads none)."""
    used.add('qk_head_dim')
    mixer = _records.record(_records.record(record['mixer'], 'mixer')['fields'], 'mixer fields')
    nope = _records.integer(mixer['qk_nope_head_dim'], 'qk_nope_head_dim')
    rope = _records.integer(mixer['qk_rope_head_dim'], 'qk_rope_head_dim')
    head_dim = hf_config.get('head_dim') if points_head_dim else None
    if head_dim is not None and _records.integer(head_dim, 'head_dim') != rope:
        refuse(f"head_dim {head_dim!r}",
               f"DeepSeek points head_dim at the rope slice, which is {rope} wide here")
    derived = hf_config.get('qk_head_dim')
    if derived is not None and _records.integer(derived, 'qk_head_dim') != nope + rope:
        refuse(f"qk_head_dim {derived!r}",
               f"it derives as qk_nope_head_dim + qk_rope_head_dim, which is {nope + rope} here")
    record['head_dim'] = nope + rope
    return record


def _mla_heads_written(model: CausalTransformer) -> Mapping[str, object]:
    mixer = _mla(model)
    return {'qk_head_dim': mixer.qk_nope_head_dim + mixer.qk_rope_head_dim,
            'head_dim': mixer.qk_rope_head_dim}


def mla_block(*, sparse: bool, points_head_dim: bool = True) -> ConfigLayout[CausalTransformer]:
    """A DeepSeek V2, V3 or V3.2 attention block, both ways: its layers (`_mla_layers`), its MLA mixer
    (`mla_fields`) and its heads' width (`_mla_heads`), and its prediction depths. The released checkpoints
    ship a depth as the layer past the trunk, model.layers.61.* (1564 of DeepSeek-V3's 91991 tensors, 1571
    of V3.2-Exp's 92425): `DEEPSEEK_ROUTES` reads it. transformers builds no MTP module and saves the trunk
    alone, and a model builds the depths its weights hold (`dew.interop.pretrained`). The fp8 scales name
    the stored dtype: Dew loads the dequantized weights, and the reader names an unreadable dtype where it
    meets one; ep_size is a runtime parallel hint."""
    return ConfigLayout(
        ConfigPart(partial(_mla_layers, sparse=sparse), lambda model: {}),
        Nested('mixer', MLAMixer, mla_fields(sparse=sparse), _mla,
               build=lambda fields: {'class': 'mla', 'fields': fields}),
        ConfigPart(partial(_mla_heads, points_head_dim=points_head_dim), _mla_heads_written),
        integer('num_nextn_predict_layers', 'num_nextn_predict_layers', 0),
        unread('num_mtp_layers', 'quantization_config', 'ep_size'))


def deepseek_layout_of(*, sparse: bool = False, router: tuple[ConfigEntry[Mixture], ...] = V3_ROUTER,
                       points_head_dim: bool = True) -> ConfigLayout[CausalTransformer]:
    """A DeepSeek V2, V3 or V3.2 config, both ways: the shared decoder (`DEEPSEEK_DECODER`), the MLA block
    and its prediction depths (`mla_block`), and the routed experts (`router` their router's) on the layers
    past the dense ones (`DENSE_FIRST`). A class that points head_dim at the rope slice whatever the config
    says (`points_head_dim` False: GlmMoeDsaConfig's) reads no head_dim."""
    decoder = DEEPSEEK_DECODER if points_head_dim else decoder_layout(
        DecoderSpelling(fixed={'head_dim': None}),
        rope=rope_part(PLAIN_SPELLING, ramps=('yarn', 'llama3'), per_kind=False))
    return ConfigLayout(decoder, mla_block(sparse=sparse, points_head_dim=points_head_dim),
                        deepseek_experts(router, routes_some), DENSE_FIRST)


# The text half of a Kimi K2.5 wrapper, by the model_type its text_config
# names. Kimi_K25Config.__post_init__ rewrites a `kimi_k2` text config, or
# a missing one, into `deepseek_v3` (configuration_kimi_k25.py:80-92), so
# these two spellings name one computation and any other names a decoder
# this wrapper does not carry.
_KIMI_K25_TEXT = ('kimi_k2', 'deepseek_v3')


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
        refuse('text_config',
                f"the wrapper carries its decoder under text_config, got {text!r}")
    model_type = text.get('model_type', 'deepseek_v3')
    if model_type not in _KIMI_K25_TEXT:
        refuse(f"text_config model_type {model_type!r}",
                "a Kimi K2.5 wrapper's decoder is Kimi K2's, which the "
                f"reference reads as one of {', '.join(map(repr, _KIMI_K25_TEXT))}")
    refuse_encoder_fields(text)
    tied = hf_config.get('tie_word_embeddings', True)
    if not isinstance(tied, bool):
        refuse(f"tie_word_embeddings {tied!r}", "the wrapper head takes a boolean tying policy")
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
              if key not in SERIALIZED_TEXT_FIELDS and key not in SERIALIZED_ENCODER_FIELDS}
    config = translated({**nested, 'model_type': model_type, 'tie_word_embeddings': tied}, DEEPSEEK_V3)
    placeholders = []
    for key, default in (('image_token_id', 163605), ('video_token_id', 163840)):
        value = hf_config.get(key, default)
        if type(value) is not int or value < 0:
            refuse(key, 'the text-only wrapper requires a nonnegative token id')
        placeholders.append(value)
    config['embedding_zero_ids'] = tuple(placeholders)
    return config


def _kimi_k25_wrapper(hf_config: Mapping[str, object], used: set[str]) -> WrapperFields:
    """Read a Kimi K2.5 bundle whole (transformers 5.17.0's Kimi_K25ForConditionalGeneration, the
    release's own at 4d01dfe): its decoder as the text half reads it (`_kimi_k25_config`), MoonViT and its
    patch merger (`dew.nn.vision.moonvit`), and the media placeholder each of an item's projected merge
    blocks fills, as its processor expands it (`Kimi_K25Processor`); the decoder's tensors, under
    `language_model.`, read as Kimi K2's (`DEEPSEEK_V3`, its text_config's own model type)."""
    text = _kimi_k25_config(hf_config, used)
    # The decoder's own record, which the language model's reader and writer read.
    text_type = _records.text(_records.record(hf_config['text_config'], 'text_config').get(
        'model_type', 'deepseek_v3'), 'model_type')
    width = record_int(text, 'emb_features')
    image = hf_config.get('image_token_id', hf_config.get('media_placeholder_token_id', 163605))
    video = hf_config.get('video_token_id')
    return {'model_type': 'kimi_k25', 'text_model_type': text_type, 'text': text,
            'tower': translate_moonvit_config(hf_config),
            'projector': translate_moonvit_projector_config(hf_config, width),
            'image_token_id': record_int({'image_token_id': image}, 'image_token_id'),
            'tokens_per_image': None,
            'video_token_id': None if video is None else record_int(hf_config, 'video_token_id'),
            'extra_placeholder_ids': (), **NO_AUDIO}


# DeepSeek V4's three attention kinds, each with the compressor it runs
# (COMPRESSOR_CLASSES, modeling_deepseek_v4.py:788-790) and the rate it pools
# by, which is also the legacy per-layer rate that names it
# (configuration_deepseek_v4.py:28-32, :156): the sliding kind runs none, at 0.
_V4_KINDS: dict[str, tuple[str | None, int]] = {
    'sliding_attention': (None, 0),
    'compressed_sparse_attention': ('csa', 4),
    'heavily_compressed_attention': ('hca', 128),
}
_V4_COMPRESSORS = {kind: compressor for kind, (compressor, _) in _V4_KINDS.items()}
_V4_RATIOS = {rate: kind for kind, (_, rate) in _V4_KINDS.items()}
_V4_RATES = {kind: rate for kind, (compressor, rate) in _V4_KINDS.items() if compressor is not None}
_V4_MLP_KINDS = ('hash_moe', 'moe')
# The activations ACT2FN holds for `scoring_func` (modeling_deepseek_v4.py:1031).
V4_SCORES = ('softmax', 'sigmoid', 'sqrtsoftplus')
# qk_rope_head_dim over head_dim of the released Flash config, which is the
# class default (configuration_deepseek_v4.py:148).
_V4_PARTIAL = 64 / 512


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
            refuse('compress_ratios', 'expected an integer compression ratio per layer')
        unknown = sorted(set(ratios) - set(_V4_RATIOS))
        if unknown:
            refuse(f"compress_ratios entries {unknown}",
                    f"a rate names its kind, one of {sorted(_V4_RATIOS)}")
        types = [_V4_RATIOS[rate] for rate in ratios]
    if types is None:
        types = (['heavily_compressed_attention'] * min(layers, 2)
                 + ['compressed_sparse_attention' if index % 2
                    else 'heavily_compressed_attention'
                    for index in range(max(layers - 2, 0))])
    resolved = _records.strings(types, 'layer_types')[:layers]
    if len(resolved) != layers:
        refuse(f"layer_types of {len(resolved)} entries",
                f"the model has {layers} layers, one attention kind each")
    unknown = sorted(set(resolved) - set(_V4_KINDS))
    if unknown:
        refuse(f"layer_types entries {unknown}",
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
        hashed = record_int(hf_config, 'num_hash_layers', 3)
        types = ['hash_moe'] * min(layers, hashed) + ['moe'] * max(layers - hashed, 0)
    resolved = _records.strings(types, 'mlp_layer_types')[:layers]
    if len(resolved) != layers:
        refuse(f"mlp_layer_types of {len(resolved)} entries",
                f"the model has {layers} layers, one routing kind each")
    unknown = sorted(set(resolved) - set(_V4_MLP_KINDS))
    if unknown:
        refuse(f"mlp_layer_types entries {unknown}",
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
        fraction = record_float(hf_config, 'partial_rotary_factor')
    elif legacy is not None:
        fraction = record_int(hf_config, 'qk_rope_head_dim') / head_dim
    else:
        fraction = _V4_PARTIAL
    width = int(head_dim * fraction)
    if legacy is not None and record_int(hf_config, 'qk_rope_head_dim') != width:
        refuse(f"qk_rope_head_dim {legacy!r}",
                f"partial_rotary_factor {partial} rotates {width} of the "
                f"{head_dim} head dims, which is the width the reference derives")
    if not 0 < width <= head_dim or width % 2:
        refuse(f"partial_rotary_factor {partial}",
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
    theta = record_float(hf_config, 'rope_theta', 10000.0)
    compress_theta = record_float(hf_config, 'compress_rope_theta', 160000.0)
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
    that rotates one. A YaRN entry's `attention_factor` scales cos and sin
    (:151-152): the flat spelling's 1.0 leaves them alone, and a nested
    entry that names none derives YaRN's own.
    """
    fraction = record_float(entry, 'partial_rotary_factor', 1.0)
    if int(head_dim * fraction) != width:
        refuse(f"{field} partial_rotary_factor {entry.get('partial_rotary_factor')!r}",
                f"its table rotates {int(head_dim * fraction)} of the {head_dim} "
                f"head dims where the model's rope width is {width}")
    theta = record_float(entry, 'rope_theta', 10000.0)
    rope_type = entry.get('rope_type', entry.get('type', 'default'))
    if rope_type in ('default', 'none'):
        extra = sorted(set(entry) - {'rope_type', 'type', 'rope_theta', 'partial_rotary_factor'})
        if extra:
            refuse(f'{field} fields {extra}', 'the plain rotary has no scaling fields')
        return theta, None
    if rope_type != 'yarn':
        refuse(f"{field} rope_type {rope_type!r}",
                "the mixer applies plain or YaRN rotary positions")
    ramp = {key: value for key, value in entry.items()
            if key != 'partial_rotary_factor'}
    ramp.setdefault('original_max_position_embeddings', max_pos)
    return theta, yarn_record(ramp, field, max_pos)


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
            refuse('compress_rates', 'expected a rate per compressed attention kind')
        rates.update(supplied)
    for kind in _V4_RATES:
        legacy = f"compress_rate_{_V4_COMPRESSORS[kind]}"
        if hf_config.get(legacy) is not None:
            rates[kind] = hf_config[legacy]
    unknown = sorted(set(rates) - set(_V4_KINDS))
    if unknown:
        refuse(f"compress_rates keys {unknown}",
                f"a rate belongs to a kind, one of {sorted(_V4_KINDS)}")
    resolved: dict[str, int] = {}
    for kind in dict.fromkeys(layer_types):
        if _V4_COMPRESSORS[kind] is None:
            continue
        rate = rates.get(kind)
        if isinstance(rate, bool) or not isinstance(rate, int) or rate < 1:
            refuse(f"compress_rates[{kind!r}] {rate!r}",
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
            refuse(key, "every deepseek_v4 layer routes, so the expert "
                           "count, the top-k and the routed width are the "
                           "checkpoint's to state")
    scoring = hf_config.get('scoring_func', 'sqrtsoftplus')
    if scoring not in V4_SCORES:
        refuse(f"scoring_func {scoring!r}",
                f"the router scores with one of {sorted(V4_SCORES)}")
    if hf_config.get('norm_topk_prob', True) is not True:
        refuse("norm_topk_prob=False",
                "both V4 routers renormalise the weights they selected, "
                "whatever this field says (modeling_deepseek_v4.py:1041, :1072)")
    shared = record_int(hf_config, 'n_shared_experts', 1)
    if shared != 1:
        refuse(f"n_shared_experts {shared!r}",
                "DeepseekV4SparseMoeBlock builds one shared MLP of the routed "
                "width (modeling_deepseek_v4.py:1082)")
    method = hf_config.get('topk_method')
    if method is not None and method != 'noaux_tc':
        refuse(f"topk_method {method!r}",
                "the router selects on the scores plus its balancing bias, "
                "which is what noaux_tc names")
    width = record_int(hf_config, 'moe_intermediate_size')
    return native_fields(Mixture)(
        experts=record_int(hf_config, 'n_routed_experts'),
        top_k=record_int(hf_config, 'num_experts_per_tok'),
        layers=tuple(range(layers)),
        score_function=scoring,
        bias=True,
        scaling=record_float(hf_config, 'routed_scaling_factor', 1.5),
        shared_features=width,
        expert_features=width,
        hash_layers=tuple(index for index, kind in enumerate(mlp_kinds)
                             if kind == 'hash_moe'),
    )


type DSparkFields = NativeFields[DSpark]


def dspark_stages(ratios: Sequence[object], layers: int) -> int:
    """The DSpark drafter's stage count: `compress_ratios`' entries past the trunk's `layers`, as the
    release builds stage `s`'s attention from entry `layers + s` (0731 model.py:459, :901-902)."""
    return len(ratios) - layers


def dspark_kept(config: Mapping[str, object], kept: int) -> dict:
    """The fields a V4.1 or V4-Flash-0731 config keeping `kept` prediction depths states differently
    (`DecoderFamily.whole_drafter`): its DSpark stages draft as one, their heads on the last, so the
    drafter is kept whole or, at 0, dropped (its stages' ratios and its block); a count between is
    refused. A config that names no block keeps its fields."""
    ratios, layers = config.get('compress_ratios'), config.get('num_hidden_layers')
    if not (config.get('dspark_block_size') and isinstance(ratios, list) and type(layers) is int):
        return {}
    stages = dspark_stages(ratios, layers)
    if 0 < kept < stages:
        raise ValueError(f"a DSpark drafter's {stages} stages draft as one, its heads on the last: "
                         f"prediction_depths keeps all of them or none (0), not {kept}")
    return {} if kept else {'compress_ratios': ratios[:layers], 'dspark_block_size': 0}


def dspark_fields(text: Mapping[str, object], layers: int, ratios: Sequence[object], seen: set[str], *,
            reads: Literal['input', 'output']) -> DSparkFields | None:
    """The DSpark drafter record of a V4.1 or V4-Flash-0731 config (section
    2.4.3; v41:129-136, 0731 model.py:82-86), or None when it names no
    block. Its stages attend a sliding window (v41:1034, 0731 model.py:753)
    and read their target layers where `reads` says.

    The stages are `compress_ratios`' entries past the trunk's, one sliding
    0 each, as the release builds stage `s`'s attention from entry
    `layers + s` (0731 model.py:459, :901-902). That is the count the
    release runs. V4-Flash-0731's config.json keeps V4's
    `num_nextn_predict_layers` of 1 where its weight index ships mtp.0 to
    mtp.2 and its inference/config.json names n_mtp_layers 3, so a stated
    count of 1 reads as that misstatement and any other count that
    disagrees is refused.
    """
    seen.update(('num_nextn_predict_layers', 'dspark_block_size', 'dspark_noise_token_id',
                 'dspark_target_layer_ids', 'dspark_markov_rank', 'dspark_n_routed_experts',
                 'dspark_num_experts_per_tok'))
    block = record_int(text, 'dspark_block_size', 0)
    if not block:
        return None
    stages = dspark_stages(ratios, layers)
    if stages < 1 or tuple(ratios[layers:]) != (0,) * stages:
        refuse('compress_ratios', "the DSpark stages are sliding layers, one trailing 0 each")
    stated = record_int(text, 'num_nextn_predict_layers', stages)
    if stated not in (stages, 1):
        refuse(f"num_nextn_predict_layers {stated}",
                f"compress_ratios lists {stages} DSpark stages after the {layers} layers")
    return native_fields(DSpark)(stages=stages, block_size=block,
            noise_token_id=record_int(text, 'dspark_noise_token_id'),
            target_layers=_records.integers(text.get('dspark_target_layer_ids', ()),
                                              'dspark_target_layer_ids'),
            markov_rank=record_int(text, 'dspark_markov_rank'),
            experts=record_int(text, 'dspark_n_routed_experts',
                                   record_int(text, 'n_routed_experts')),
            top_k=record_int(text, 'dspark_num_experts_per_tok',
                                 record_int(text, 'num_experts_per_tok')),
            layer_type='sliding_attention', reads=reads)


def _v4_pattern(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """A DeepSeek V4 config's layers' kinds (`_v4_layer_types`), read before the shared decoder's."""
    record['layer_types'] = _v4_layer_types(hf_config, record_int(hf_config, 'num_hidden_layers'), used)
    return record


def _deepseek_v4_config(hf_config: Mapping[str, object], used: set[str],
                        config: DecoderFields) -> DecoderFields:
    """Read a DeepSeek-V4-Flash or V4-Pro config's own fields beside the shared decoder's
    (`DEEPSEEK_V4_CONFIG`).

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
    layers = record_int(hf_config, 'num_hidden_layers')
    heads = record_int(hf_config, 'num_attention_heads')
    head_dim = record_int(hf_config, 'head_dim', record_int(hf_config, 'hidden_size') // heads)
    layer_types = _v4_layer_types(hf_config, layers, used)
    mlp_kinds = _v4_mlp_kinds(hf_config, layers, used)
    rope_width, partial = _v4_rope_width(hf_config, used, head_dim)
    max_pos = record_int(hf_config, 'max_position_embeddings', DEFAULT_MAX_SEQ_LEN)
    main, compress = _v4_rope_entries(hf_config, used, partial)
    main_theta, main_ramp = _v4_rope(main, 'rope_parameters main',
                                     head_dim, rope_width, max_pos)
    if main_ramp is not None:
        refuse("rope_parameters main rope_type 'yarn'",
                "the sliding layers rotate at the plain base, which is what "
                "the config's own main entry names")
    compress_theta, compress_ramp = _v4_rope(compress, 'rope_parameters compress',
                                             head_dim, rope_width, max_pos)
    window = hf_config.get('sliding_window')
    if window is None:
        refuse("sliding_window", "every deepseek_v4 layer attends a window")
    window = record_int(hf_config, 'sliding_window')
    kv_heads = record_int(hf_config, 'num_key_value_heads', 1)
    if kv_heads != 1:
        refuse(f"num_key_value_heads {kv_heads!r}",
                "V4's attention projects one key/value head and broadcasts it "
                "to every query head (modeling_deepseek_v4.py:749-751, :770)")
    config['rope_theta'] = main_theta
    if config.get('attention_bias'):
        refuse("attention_bias=True",
                "V4 builds every attention projection bias-free "
                "(modeling_deepseek_v4.py:777-786)")
    used.update(('q_lora_rank', 'o_groups', 'o_lora_rank',
                 'index_topk', 'index_n_heads', 'index_head_dim',
                 'swiglu_limit', 'hc_mult', 'hc_eps', 'hc_sinkhorn_iters'))
    # Transformers omits the prediction depth (modeling_deepseek_v4.py:1212);
    # the released inference/model.py MTPBlock executes its raw-stream
    # e_proj/h_proj composition, and V4-Flash-0731's replaces it with DSpark's
    # stages (0731 model.py:818-874, :898-904), which `dspark_fields` reads from a
    # config naming a block. The router's logit output and aux coefficient are
    # training knobs, its jitter is read nowhere, and ep_size is a runtime
    # parallel hint.
    used.update(('num_nextn_predict_layers', 'output_router_logits',
                 'router_aux_loss_coef', 'router_jitter_noise', 'ep_size'))
    groups = record_int(hf_config, 'o_groups')
    fields: dict[str, object] = {
        'q_lora_rank': record_int(hf_config, 'q_lora_rank'),
        'o_groups': groups,
        'o_lora_rank': record_int(hf_config, 'o_lora_rank'),
        'rope_head_dim': rope_width,
        'compressor': None, 'compress_rate': None,
        'index_topk': None, 'index_n_heads': None, 'index_head_dim': None}
    mixer = {'class': 'deepseek_v4', 'fields': fields}
    if groups < 1 or heads * head_dim % groups:
        refuse(f"o_groups {groups}",
                f"the grouped output projection splits the {heads * head_dim} "
                "stacked head dims into equal groups")
    kinds = _v4_attention_kinds(hf_config, used, layer_types, fields, window,
                                (compress_theta, compress_ramp))
    ratios = hf_config.get('compress_ratios')
    dspark = dspark_fields(hf_config, layers, ratios if isinstance(ratios, (list, tuple)) else (), used,
                     reads='output')
    if dspark is not None:
        # 0731's stages are V4 sliding layers (0731 model.py:753, :482-485).
        kinds.setdefault('sliding_attention', native_fields(LayerKind)(window=window))
        config['dspark'] = dspark
    depth = 0 if dspark is not None else hf_config.get('num_nextn_predict_layers', 1)
    if type(depth) is not int or depth not in (0, 1):
        refuse('num_nextn_predict_layers', 'V4 supports the released single prediction depth')
    if depth:
        if isinstance(ratios, (list, tuple)) and len(ratios) > layers and ratios[layers] != 0:
            refuse('compress_ratios prediction depth', 'the released V4 prediction block is sliding-only')
        kinds['mtp_attention'] = native_fields(LayerKind)(window=int(window), rope_theta=main_theta,
                                                         mixer=None)
        kinds['mtp_attention']['mixer'] = mixer
        config['mtp_layer_type'] = 'mtp_attention'
    config['num_nextn_predict_layers'] = depth
    streams = _v4_streams(hf_config)
    config.update(
        mixer=mixer,
        kinds=kinds,
        mixture=_v4_mixture(hf_config, layers, mlp_kinds, used),
        swiglu_limit=record_float(hf_config, 'swiglu_limit', 10.0),
        hyper_connections=streams,
    )
    if depth:
        config['mtp_hyper_connections'] = streams.copy()
    return config


def _v4_attention_kinds(hf_config: Mapping[str, object], used: set[str],
                        layer_types: tuple[str, ...], mixer: Mapping[str, object],
                        window: int, compress: tuple[float, Ramp | None]) -> dict[str, KindFields]:
    """Build each attention kind's own record from the shared `mixer` fields.

    Every V4 layer attends its window; a compressed one rotates at the
    compress base and hands its compressor and rate to the mixer, the sparse
    one its indexer's geometry too.
    """
    compress_theta, compress_ramp = compress
    index = ({field: record_int(hf_config, field)
              for field in ('index_topk', 'index_n_heads', 'index_head_dim')}
             if 'compressed_sparse_attention' in layer_types else {})
    rates = _v4_compress_rates(hf_config, layer_types, used)
    kinds: dict[str, KindFields] = {}
    for kind in dict.fromkeys(layer_types):
        record: KindFields = native_fields(LayerKind)(window=window)
        compressor = _V4_COMPRESSORS[kind]
        if compressor is not None:
            layer_mixer = {**mixer, 'compressor': compressor,
                           'compress_rate': rates[kind]}
            if compressor == 'csa':
                layer_mixer.update(index)
            record.update(rope_theta=compress_theta, rope_scaling=compress_ramp,
                          mixer={'class': 'deepseek_v4', 'fields': layer_mixer})
        kinds[kind] = record
    return kinds


def _v4_streams(hf_config: Mapping[str, object]) -> HyperConnectionsFields:
    """Return the residual streams every V4 layer reads and writes.

    A layer collapses them through a learned head of its own
    (DeepseekV4HyperHead, modeling_deepseek_v4.py:946-962).
    """
    return native_fields(HyperConnections)(
        hc_mult=record_int(hf_config, 'hc_mult', 4),
        hc_eps=record_float(hf_config, 'hc_eps', 1e-6),
        hc_sinkhorn_iters=record_int(hf_config, 'hc_sinkhorn_iters', 20),
        head='weighted')


def _v4(stored: str, leaf: str, slots: Mapping[str, Slot] | None = None) -> Route:
    """A V4-family layer's tensor under the release's own names, `stored` past `layers.{layer}.`, onto its
    leaf past the layer's parameters, `leaf`."""
    return Route('layers.{layer}.' + stored, 'params/layers_{layer}/' + leaf,
                 MappingProxyType(dict(slots or {})), in_trunk)


_V4_EXPERTS = mapped({'w1': 'gate_proj', 'w2': 'down_proj', 'w3': 'up_proj'})
"""The per-expert and shared `w1`/`w2`/`w3`: the gate, down and up projections of one gated MLP."""


def v4_layer_routes(compressor: Slot, state: Slot, *own: Route) -> Table:
    """A V4-family layer's tensors under the release's names (DeepSeek V4's and V4.1's, onto the module
    names transformers renames them to, conversion_mapping.py:483-534): the block's halves `attn` and
    `ffn`, its projections `wq_a`/`wq_b`/`wkv`/`wo_a`/`wo_b`, its query latent norm `q_norm`, its sinks
    `attn_sink`, mHC's mappings around each half, and each compressor's (`compressor` names where they
    sit) projections, entry norm `norm` and position bias `ape`; the routers' state (`state`), the
    experts; and the release's own (`own`)."""
    return Table(
        _v4('attn_norm.weight', 'input_layernorm/scale'),
        _v4('ffn_norm.weight', 'post_attention_layernorm/scale'),
        _v4('hc_{site}_{part}', '{site}_hc/{part}', {'site': one_of('attn', 'ffn'),
                                                     'part': one_of('fn', 'base', 'scale')}),
        _v4('attn.{projection}.weight', 'self_attn/{projection}/kernel', {'projection': mapped({
            'wq_a': 'q_a_proj', 'wq_b': 'q_b_proj', 'wkv': 'kv_proj', 'wo_a': 'o_a_proj',
            'wo_b': 'o_b_proj'})}),
        _v4('attn.q_norm.weight', 'self_attn/q_a_norm/scale'),
        _v4('attn.kv_norm.weight', 'self_attn/kv_norm/scale'),
        # transformers' reverse conversion spells it `attn.norm` (its kv_norm rename read backwards).
        _v4('attn.norm.weight', 'self_attn/kv_norm/scale'),
        _v4('attn.attn_sink', 'self_attn/sinks'),
        _v4('attn.{compressor}.{projection}.weight', 'self_attn/{compressor}/{projection}/kernel',
            {'compressor': compressor, 'projection': mapped({'wkv': 'kv_proj', 'wgate': 'gate_proj'})}),
        _v4('attn.{compressor}.norm.weight', 'self_attn/{compressor}/kv_norm/scale',
            {'compressor': compressor}),
        _v4('attn.{compressor}.ape', 'self_attn/{compressor}/position_bias', {'compressor': compressor}),
        *own,
        Route('layers.{layer}.ffn.gate.{state}', 'moe/layers_{layer}/mlp/gate/{state}',
              MappingProxyType({'state': state}), in_trunk),
        _v4('ffn.gate.weight', 'mlp/gate/kernel'),
        _v4('ffn.experts.{expert}.{projection}.weight', 'mlp/experts/{expert}/{projection}/kernel',
            {'projection': _V4_EXPERTS}),
        _v4('ffn.shared_experts.{projection}.weight', 'mlp/shared_experts/{projection}/kernel',
            {'projection': _V4_EXPERTS}),
    )


V4_TRUNK = Table(
    Route('embed.weight', 'params/embed_tokens/embedding'),
    Route('norm.weight', 'params/norm/scale'),
    Route('head.weight', None, when=lambda config, slots: bool(config['tie_embeddings'])),
    Route('head.weight', 'params/lm_head/kernel', when=lambda config, slots: not config['tie_embeddings']),
)
"""The trunk names V4's reference code gives the embedding, final norm and head, which V4.1's shares.
A released checkpoint carries these; one transformers wrote carries the shared decoder's
(conversion_mapping.py:489-493's three ^-anchored renames rewrite them on load and, being anchored, leave
them alone on save), which a V4 family reads too (`DECODER_ROUTES`)."""


# A DSpark stage's own leaves beside its block, under `mtp.{stage}.`, as
# both releases name them; each adds its Markov head's spelling.
DSPARK_LEAVES: dict[str, tuple[str, ...]] = {
    'main_proj.weight': ('main_proj', 'kernel'), 'main_norm.weight': ('main_norm', 'scale'),
    'norm.weight': ('norm', 'scale'), 'confidence_head.proj.weight': ('confidence', 'kernel')}
# V4-Flash-0731's Markov tables (0731 model.py:795-799) and the mHC head its
# last stage collapses by (:838-841).
_V4_DSPARK_LEAVES: dict[str, tuple[str, ...]] = {
    **DSPARK_LEAVES,
    'markov_head.markov_w1.weight': ('markov_embed',), 'markov_head.markov_w2.weight': ('markov_head',),
    'hc_head_fn': ('hc_head', 'hc_fn'), 'hc_head_base': ('hc_head', 'hc_base'),
    'hc_head_scale': ('hc_head', 'hc_scale')}


# A V4 prediction depth's own leaves under `mtp.{depth}.`, beside its block
# (official inference/model.py, MTPBlock).
_V4_PREDICTION_LEAVES: dict[str, tuple[str, ...]] = {
    'norm.weight': ('final_norm', 'scale'),
    **{f'{norm}.weight': (norm, 'scale') for norm in ('enorm', 'hnorm')},
    **{f'{proj}.weight': (proj, 'kernel') for proj in ('e_proj', 'h_proj')},
    **{f'hc_head_{leaf}': ('hc_head', f'hc_{leaf}') for leaf in ('fn', 'base', 'scale')}}


def _stages(config: Config, slots: Mapping[str, str]) -> bool:
    """Whether a route's `depth` is one of the record's DSpark stages."""
    dspark = config.get('dspark')
    return isinstance(dspark, Mapping) and int(slots['depth']) < dspark['stages']


def _depths_without_dspark(config: Config, slots: Mapping[str, str]) -> bool:
    return not isinstance(config.get('dspark'), Mapping) and declared_depths()(config, slots)


def v4_routes(layers: Table, dspark: Leaves, *trunk: Routes, depth: Leaves | None = None) -> Table:
    """A V4-family release's names (`layers`, a layer's, `v4_layer_routes`; `trunk`, the rest of its
    own): its trunk (`V4_TRUNK`); its prediction depths (`depth`'s own leaves) or DSpark's stages
    (`dspark`'s, where the record drafts by DSpark: V4-Flash-0731's `mtp.*` are its stages), each under
    `mtp.{depth}.` beside its block, named as layer 0's; the layers; and the same tensors under the module
    names transformers writes (`DECODER_ROUTES`), which a load reads too."""
    slots = {'depth': INDEX}
    return Table(
        V4_TRUNK, *trunk,
        prediction_routes('mtp.{depth}.', 'dspark', dspark, layers, slots=slots, when=_stages,
                          base='layers.0.'),
        *(() if depth is None else (
            prediction_routes('mtp.{depth}.', 'mtp', depth, layers, slots=slots, when=_depths_without_dspark,
                              base='layers.0.'),
            prediction_routes('mtp.{depth}.', 'mtp', {}, DECODER_ROUTES, slots=slots,
                              when=_depths_without_dspark))),
        # A checkpoint transformers saved: its trunk prefixed `model.`, a layer's tail the release's own.
        layers, Prefixed('model.', layers), DECODER_ROUTES)


def v4_depth(name: str, config: Mapping[str, object]) -> int | None:
    """The prediction depth or DSpark stage of a V4 tensor, the `{index}` of its `mtp.{index}.*`, or
    None for the trunk's (`_deepseek_v4_path`, `dspark_path`)."""
    parts = name.split('.')
    return int(parts[1]) if parts[0] == 'mtp' and len(parts) >= 3 and parts[1].isdigit() else None


DEEPSEEK_V4_ROUTES = v4_routes(
    v4_layer_routes(
        # V4's indexer sits under the layer's compressor, its own compressor under it.
        mapped({'compressor': 'compressor', 'indexer.compressor': 'compressor/indexer'}),
        mapped({'bias': 'e_score_correction_bias', 'tid2eid': 'tid2eid'}),
        _v4('attn.indexer.wq_b.weight', 'self_attn/compressor/indexer/q_b_proj/kernel'),
        _v4('attn.indexer.weights_proj.weight', 'self_attn/compressor/indexer/scorer/weights_proj/kernel')),
    _V4_DSPARK_LEAVES,
    Route('hc_head_{part}', 'params/hc_head/hc_{part}',
          MappingProxyType({'part': one_of('fn', 'base', 'scale')})),
    depth=_V4_PREDICTION_LEAVES)
"""DeepSeek V4's names: the release's, which name the block's halves `attn` and `ffn` and hold the stack
at `layers.N.*` with no `model.` prefix, and the module names transformers renames them to on load and
writes most of back on save, so a released checkpoint and an export differ in the trunk's spelling and
the prefix and agree on the rest; both are read, the release's written. The release's prediction depth
has its own split input projections, mHC block, collapse head and final norm (official
inference/model.py, MTPBlock), while sharing the trunk embedding and vocabulary head."""


def v4_prepare(routes: Routes) -> WeightPreparer:
    """The storage step a V4-family release reads its grouped output projection and hash table through,
    each tensor found by where `routes` place it, in either spelling they read.

    `DeepseekV4GroupedLinear` stores one block per head group in a matrix of
    `[groups * o_lora_rank, heads * head_dim / groups]` and reads it as
    `weight.view(groups, -1, in).transpose(1, 2)`
    (modeling_deepseek_v4.py:294-323), which is the `[groups, in, rank]`
    kernel the mixer contracts over; the group count is the stacked query
    width over one block's input width. An export folds it back
    (`DecoderFamily.folded`). The token-to-expert table narrows to
    the int32 the router's collection holds (:1062).
    """
    def prepare(tensors: Mapping[str, Stored],
                config: Mapping[str, object] | None = None) -> dict[str, Stored]:
        if config is None:
            raise ValueError("a V4 release's grouped projection and hash table are found by the translated "
                             "config")
        paths = {found.path: name for name in tensors
                 if (found := routes.read(name, config)) is not None and found.path is not None}
        prepared = dict(tensors)
        for path, name in paths.items():
            tensor = tensors[name]
            if path[-2:] == ('o_a_proj', 'kernel'):
                partner = paths.get((*path[:-2], 'q_b_proj', 'kernel'))
                if partner is None:
                    raise ValueError(f"{name} needs its block's query projection (wq_b) to name its head "
                                     "groups")
                groups = tensors[partner].shape[0] // tensor.shape[1]
                prepared[name] = held(tensor.reshape(groups, -1, tensor.shape[1]).transpose(0, 2, 1),
                                      contiguous=True)
            elif path[-1] == 'tid2eid':
                if not np.issubdtype(tensor.dtype, np.integer):
                    raise ValueError(f"{name} must contain integer expert indices")
                router_name = paths.get(('params', *path[1:-1], 'kernel'))
                if router_name is None:
                    raise ValueError(f"{name} has no gate.weight declaring its expert count")
                router = tensors[router_name]
                # The indices' range is checked where they are held (`valued`).
                if tensor.size and valued(tensor):
                    lower, upper = np.min(tensor), np.max(tensor)
                    limits = np.iinfo(np.int32)
                    if lower < limits.min or upper > limits.max:
                        raise ValueError(f"{name} has expert indices outside the lossless int32 range")
                    if lower < 0 or upper >= router.shape[0]:
                        raise ValueError(f"{name} has indices outside its {router.shape[0]} experts")
                prepared[name] = held(tensor, np.int32)
        return prepared
    return prepare


MINIMAX_M2 = DecoderFamily(
    ('minimax_m2',),
    MINIMAX_M2_CONFIG,
    lambda fields: bool(fields.qk_norm and fields.qk_norm_scope == 'projection'
                        and fields.pre_norms and fields.mixture is not None),
    'minimax_m2',
    'MiniMaxM2ForCausalLM',
    routes=spelled(_MINIMAX_M2_NAMES),
    preserve_source_layout=True,
    padding_idx=True,
    defaults={'max_position_embeddings': 196608},
)


# V4's block is nothing another family builds: the mixer kind names its
# window, its compressor and its grouped output projection at once.
V4_DECODER = decoder_layout(DecoderSpelling(keys={'intermediate_size': 'moe_intermediate_size'},
                                            fixed={'num_key_value_heads': 1}), rope=ROPE_APART)
"""The shared decoder as DeepSeek V4's block states it, which V4.1's keeps: V4 ships no dense feed-forward
width and no key/value head count the attention reads (`intermediate_size` is an alias of the routed width,
configuration_deepseek_v4.py:101-107, and the one shared head is the block's own), and its rotary entries
are its own part's."""

DEEPSEEK_V4_CONFIG = ConfigLayout(
    ConfigPart(_v4_pattern, lambda model: {}), V4_DECODER,
    ConfigPart(_deepseek_v4_config, unwritten("a V4 model is written as its release, whose config no "
                                              "exporter of Dew's states").write))
"""DeepSeek V4's config: read beside the shared decoder; no V4 model is written."""


DEEPSEEK_V4 = DecoderFamily(
    ('deepseek_v4',),
    DEEPSEEK_V4_CONFIG,
    lambda fields: isinstance(fields.mixer, DeepseekV4Mixer),
    'deepseek_v4',
    'DeepseekV4ForCausalLM',
    routes=DEEPSEEK_V4_ROUTES, prediction_depth=v4_depth,
    prepare=v4_prepare(DEEPSEEK_V4_ROUTES), folded=('o_a_proj/kernel',),
    preserve_source_layout=True,
    tied_head_names=('head.weight', 'embed.weight'),
    padding_idx=True,
    prediction_counts=('num_nextn_predict_layers', 'n_mtp_layers'), whole_drafter=dspark_kept,
)

DEEPSEEK_V32 = DecoderFamily(
    ('deepseek_v32',),
    deepseek_layout_of(sparse=True),
    lambda fields: (isinstance(mixer := fields.mixer, MLAMixer) and mixer.index_topk is not None),
    'deepseek_v32',
    'DeepseekV32ForCausalLM',
    routes=DEEPSEEK_ROUTES, prediction_depth=layer_depth,
    preserve_source_layout=True,
    # A query without its LoRA states its rank null, which the config class's default would fill.
    export_nulls=('q_lora_rank',),
    padding_idx=True,
    defaults={'max_position_embeddings': 163840},
)

DEEPSEEK_V2 = DecoderFamily(
    ('deepseek_v2',),
    deepseek_layout_of(router=V2_ROUTER),
    lambda fields: (
        isinstance(fields.mixer, MLAMixer)
        and (mixture := fields.mixture) is not None
        and not mixture.bias
    ),
    'deepseek_v2',
    'DeepseekV2ForCausalLM',
    routes=DEEPSEEK_ROUTES, prediction_depth=layer_depth,
    preserve_source_layout=True,
    # A query without its LoRA states its rank null, which the config class's default would fill.
    export_nulls=('q_lora_rank',),
    aliases={'num_experts': 'n_routed_experts'},
    # DeepseekV2Config's defaults for the fields its decoder and routing read.
    defaults={'n_shared_experts': 2, 'n_routed_experts': 64, 'routed_scaling_factor': 1.0,
              'topk_method': 'greedy', 'norm_topk_prob': False, 'first_k_dense_replace': 0,
              'q_lora_rank': 1536, 'max_position_embeddings': 2048},
    padding_idx=True,
)


# Kimi K2.5 wraps that same computation in a vision repo, so it is
# provenance-only too, and its own tensor names are the wrapper's. Its pad row
# is the one its text config states, since the language model is built from
# text_config (modeling_kimi_k25.py:586), and DeepSeek V3's reader reads it.
KIMI_K25 = DecoderFamily(
    ('kimi_k25',),
    read_only(_kimi_k25_config, 'a Kimi K2.5 bundle is written whole, its decoder as Kimi K2'),
    lambda fields: False,
    'kimi_k25',
    'Kimi_K25ForConditionalGeneration',
    routes=Table(Prefixed('language_model.', DECODER_ROUTES),
                 # The tower and the projector have no counterpart here: their tensors fill no leaf, and
                 # a save writes their source bytes back.
                 Route('{media}', None,
                       MappingProxyType({'media': Slot(r'(?:vision_tower|mm_projector)\..+')}))),
    preserve_source_layout=True,
    tied_head_names=('language_model.lm_head.weight', 'language_model.model.embed_tokens.weight'),
    wrapper=_kimi_k25_wrapper,
)

# Kimi K2 is DeepSeek V3's computation under another name; a derived model
# exports as DeepSeek V3, the name `export_model_type` gives either source.
DEEPSEEK_V3 = DecoderFamily(
    ('deepseek_v3', 'kimi_k2'),
    deepseek_layout_of(),
    lambda fields: isinstance(fields.mixer, MLAMixer),
    'deepseek_v3',
    'DeepseekV3ForCausalLM',
    routes=DEEPSEEK_ROUTES, prediction_depth=layer_depth,
    preserve_source_layout=True,
    # A query without its LoRA states its rank null, which the config class's default would fill.
    export_nulls=('q_lora_rank',),
    aliases={'num_local_experts': 'n_routed_experts', 'num_mtp_layers': 'num_nextn_predict_layers'},
    # DeepseekV3Config's defaults for the fields its decoder and routing read: its routed weights
    # scale by 2.5.
    defaults={'n_shared_experts': 1, 'n_routed_experts': 256, 'num_experts_per_tok': 8,
              'routed_scaling_factor': 2.5, 'n_group': 8, 'topk_group': 4, 'norm_topk_prob': True,
              'first_k_dense_replace': 3, 'num_key_value_heads': 128, 'q_lora_rank': 1536,
              'max_position_embeddings': 4096, 'num_nextn_predict_layers': 1},
    padding_idx=True,
)
