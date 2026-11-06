"""Translate the GLM decoders: glm4_moe, glm_moe_dsa and glm5_next_text.

GLM 4.5 biases q/k/v over a bias-free output projection and rotates half the
head. glm_moe_dsa is DeepSeek V3.2's sparse MLA block under those names, told
apart by the indexer rotating interleaved pairs. GLM 5.3's text block pools its
MLA keys without position, runs KDA layers beside them and carries mHC, so its
export names every persistent leaf itself, checking that the load reads each
one back, and the shared writer stores them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_config import (
    ROPE_APART,
    DecoderSpelling,
    decoder_layout,
    flag,
    integer,
    partial_rope_part,
)
from dew.interop.decoder_parts import (
    DECODER_ROUTES,
    DEEPSEEK_ROUTES,
    DecoderFamily,
    DecoderFields,
    KindFields,
    MixtureFields,
    fixed_fields,
    kind_mixers,
    layer_depth,
    layer_depth_routes,
    record_float,
    record_int,
    refuse,
    refuse_export,
    specified_layer_types,
)
from dew.interop.families.deepseek import (
    LAYOUT_FIELDS,
    deepseek_layout_of,
    deepseek_mixture,
    deepseek_moe_export,
)
from dew.interop.layouts import ConfigLayout, ConfigPart, Copy, Route, Table, constant
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.dsa_kpool import KPoolSparseAttentionMixer
from dew.nn.hyper_connections import HyperConnections
from dew.nn.kda import KimiDeltaAttentionMixer
from dew.nn.mla import MLAMixer


def _glm5_refusals(hf_config: Mapping[str, object], used: set[str], layers: int) -> None:
    """Refuse a GLM-5.3-Flash config field the reference does not carry.

    Each field the reference fixes is checked against its one value, and the
    indexer schedule against the one selection mode this loader builds.
    """
    for name, expected in (('mhc', True), ('mla_use_nope', True), ('index_kpool_compress', True),
                           ('moe_router_dtype', 'float32'), ('hidden_act', 'silu'),
                           ('qk_rope_head_dim', 0), ('head_dim', 0)):
        used.add(name)
        if hf_config.get(name, expected) != expected:
            refuse(name, f'GLM-5.3-Flash requires {expected!r}')
    if hf_config.get('qk_head_dim', hf_config['qk_nope_head_dim']) != hf_config['qk_nope_head_dim']:
        refuse('qk_head_dim', 'NoPE uses qk_nope_head_dim alone')
    kv_heads = hf_config.get('num_key_value_heads')
    if kv_heads is not None and kv_heads != hf_config['num_attention_heads']:
        refuse('num_key_value_heads', 'the reference requires one KV head per query head')
    if hf_config.get('attention_dropout', 0.0) != 0.0:
        refuse('attention_dropout', 'k-pool attention does not implement training dropout')
    used.add('attention_dropout')
    if 'shared' in _glm_indexer_types(hf_config, layers, used):
        refuse('indexer_types', 'shared k-pool index selections are not implemented')


def _glm5_linear_fields(hf_config: Mapping[str, object],
                        types: tuple[str, ...]) -> Mapping[str, object]:
    """Read the KDA mixer's fields, from `linear_attn_config` or the top level.

    The nested object wins wherever it states a field. Its `kda_layers` and
    `full_attn_layers` are checked against `types` rather than read, since the
    layer kinds are resolved before this runs.
    """
    nested = hf_config.get('linear_attn_config')
    raw = {} if nested is None else nested
    if not isinstance(raw, Mapping):
        refuse('linear_attn_config', 'expected an object')
    fields = {'num_heads': ('linear_num_heads', 64), 'head_dim': ('linear_head_dim', 128),
              'short_conv_kernel_size': ('linear_conv_kernel_dim', 4),
              'gate_lower_bound': ('linear_lower_bound', -5.0)}
    extra = set(raw) - set(fields) - {'safe_gate', 'kda_layers', 'full_attn_layers'}
    if extra:
        refuse('linear_attn_config', f'unknown fields {sorted(extra)}')
    linear = {target: raw.get(source, hf_config.get(target, default))
              for source, (target, default) in fields.items()}
    # The nested safe-gate flag only supplies a missing lower bound (:197-199).
    if nested is not None and raw.get('safe_gate', True) and linear['linear_lower_bound'] is None:
        linear['linear_lower_bound'] = -5.0
    for name, kind in (('kda_layers', 'linear_attention'), ('full_attn_layers', 'full_attention')):
        if name in raw:
            indices = raw[name]
            if not isinstance(indices, (list, tuple)) or list(indices) != [
                    i for i, value in enumerate(types) if value == kind]:
                refuse(f'linear_attn_config.{name}', 'disagrees with layer_types')
    return linear


def _glm5_mixture(hf_config: Mapping[str, object], used: set[str],
                  layers: int) -> tuple[MixtureFields | None, tuple[int, ...]]:
    """Read the routed layers and their mixture, or (None, ()) for a dense stack.

    `routed` is the layers `mlp_layer_types` marks sparse, which the caller
    reads again to place the prediction depth.
    """
    # GLM reads the explicit schedule, not DeepSeek's dense-prefix rule
    # (modeling_glm5_next.py:1270-1272; configuration_glm5_next.py:160-163).
    schedule = hf_config.get('mlp_layer_types')
    if schedule is None:
        schedule = ['dense'] * min(3, layers) + ['sparse'] * max(layers - 3, 0)
    if (not isinstance(schedule, (list, tuple)) or len(schedule) != layers
            or any(kind not in ('dense', 'sparse') for kind in schedule)):
        refuse('mlp_layer_types', 'one dense or sparse entry per layer is required')
    routed = tuple(index for index, kind in enumerate(schedule) if kind == 'sparse')
    if not routed:
        # A stack that routes no layer computes nothing its router's fields describe.
        used.update((*LAYOUT_FIELDS, 'scoring_func', 'topk_method', 'norm_topk_prob'))
        return None, routed
    return deepseek_mixture(hf_config, layers, used, sparse_layers=routed, reads_norm_topk=True), routed


def _glm5_types(hf_config: Mapping[str, object], used: set[str]) -> tuple[str, ...]:
    """A GLM-5.3-Flash config's layers: linear attention, and its k-pool sparse attention as full."""
    layers = record_int(hf_config, 'num_hidden_layers')
    types = specified_layer_types(hf_config, used, ('linear_attention',) * 3 + ('deepseek_sparse_attention',))
    types = tuple('full_attention' if kind == 'deepseek_sparse_attention' else kind for kind in types)
    if len(types) != layers or set(types) - {'linear_attention', 'full_attention'}:
        refuse('layer_types', 'one linear_attention or deepseek_sparse_attention entry per layer')
    return types


def _glm5_pattern(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    record['layer_types'] = _glm5_types(hf_config, used)
    return record


def _glm5_next_config(hf_config: Mapping[str, object], used: set[str],
                      config: DecoderFields) -> DecoderFields:
    """Read a GLM-5.3-Flash text config's own fields beside the shared decoder's (`GLM5_NEXT_CONFIG`).

    The block is NoPE pooled MLA, KDA and mHC
    (modeling_glm5_next.py:1259-1329). The reader builds four things from the
    config: `types`, one attention kind per layer; `linear`, the KDA mixer's
    fields; `sparse`, the k-pool mixer's; and `mixture`, the routed layers'
    geometry. Every field the reference fixes is checked and refused rather
    than dropped.
    """
    layers = record_int(hf_config, 'num_hidden_layers')
    types = _glm5_types(hf_config, used)
    _glm5_refusals(hf_config, used, layers)
    linear = _glm5_linear_fields(hf_config, types)
    sparse_fields = ('q_lora_rank', 'kv_lora_rank', 'qk_nope_head_dim', 'v_head_dim',
                     'index_n_heads', 'index_head_dim', 'index_topk', 'index_kpool')
    sparse = {name: record_int(hf_config, name) for name in sparse_fields}
    if sparse['index_kpool'] < 1 or sparse['index_topk'] % sparse['index_kpool']:
        refuse('index_topk / index_kpool', 'the budget must contain whole positive-sized pools')
    config['rope_theta'] = 10000.0
    kinds: dict[str, KindFields] = {}
    if 'linear_attention' in types:
        kinds['linear_attention'] = native_fields(LayerKind)(mixer=None)
        kinds['linear_attention'].update(mixer={'class': 'kimi_delta_attention', 'fields': {**linear}})
    if 'full_attention' in types:
        kinds['full_attention'] = native_fields(LayerKind)(mixer=None)
        kinds['full_attention'].update(mixer={'class': 'kpool_sparse_attention', 'fields': {**sparse,
            'index_kpool_always_select_tail': hf_config.get('index_kpool_always_select_tail', True)}})
    mixture, routed = _glm5_mixture(hf_config, used, layers)
    hc_fields = ('hc_mult', 'hc_eps', 'hc_sinkhorn_iters')
    config.update(
        kinds=kinds,
        mixture=mixture,
        hyper_connections=native_fields(HyperConnections)(
            hc_mult=record_int(hf_config, "hc_mult"), hc_eps=record_float(hf_config, "hc_eps"),
            hc_sinkhorn_iters=record_int(hf_config, "hc_sinkhorn_iters"), head="mean"),
        swiglu_limit=record_float(hf_config, "swiglu_limit"),
        index_share_for_mtp_iteration=bool(hf_config.get("index_share_for_mtp_iteration", False)),
        num_nextn_predict_layers=_single_depth(hf_config),
    )
    # Native NextN uses normalized trunk states and a plain NoPE depth, not
    # trunk mHC (SGLang 97c6978 deepseek_nextn.py:177-187,247-298). The
    # existing full-attention kind is k-pool/NoPE; mtp_hyper_connections
    # deliberately remains its plain default.
    if (config.get('num_nextn_predict_layers')
            and ('full_attention' not in types or layers - 1 not in routed)):
        refuse('num_nextn_predict_layers', 'the prediction depth requires a routed sparse-attention block')
    used.update((*sparse_fields, *hc_fields, *linear, 'linear_attn_config', 'qk_head_dim',
                 'swiglu_limit', 'index_kpool_always_select_tail', 'output_router_logits',
                 'router_aux_loss_coef', 'index_share_for_mtp_iteration', 'indexer_rope_interleave'))
    return config


def _single_depth(hf_config: Mapping[str, object]) -> int:
    depth = hf_config.get('num_nextn_predict_layers', 0)
    if type(depth) is not int or depth not in (0, 1):
        refuse('num_nextn_predict_layers', 'only a single shared prediction layer is supported')
    return depth


def _glm5_next_export(model: CausalTransformer) -> Mapping[str, object]:
    """Return the config fields a GLM-5.3-Flash checkpoint declares for `model`.

    It first refuses any model setting GLM-5.3 does not carry, then reads the
    two mixers back out of the model's kinds: `linear` for KDA,
    `sparse` for k-pool attention. The routed geometry is added from
    `model.mixture` when the model routes at all.
    """
    # What the config carries is held to the round trip (`refuse_lossy_export`); how the model
    # runs, which it does not carry, is held here.
    fixed_fields(model, {'scale_after_cast': True, 'dropout_rate': 0.0, 'layer_scalar': None},
                 'GLM5 requires {0!r}')
    hc = model.hyper_connections
    if hc is None or hc.head != 'mean':
        refuse_export('hyper_connections', 'GLM5 contracts mHC streams by their unweighted mean')
    if model.sharing_layers:
        refuse_export('kv_shared_layers', 'the supported GLM5 layout owns an indexer per sparse layer')
    if model.num_nextn_predict_layers not in (0, 1):
        refuse_export('num_nextn_predict_layers', 'GLM5 has zero or one plain prediction depth')
    if model.swiglu_limit is None:
        refuse_export('swiglu_limit', 'GLM5 clamps both dense and routed SwiGLU branches')
    types = model.per_layer_types
    if len(types) != model.num_layers or set(types) - {'linear_attention', 'full_attention'}:
        refuse_export('layer_types', 'GLM5 uses linear and NoPE sparse attention')
    linear, sparse = KimiDeltaAttentionMixer(), KPoolSparseAttentionMixer()
    for kind_name in set(types):
        mixer = model.kind_of(kind_name).mixer or model.mixer
        if kind_name == 'linear_attention' and isinstance(mixer, KimiDeltaAttentionMixer):
            linear = mixer
        elif kind_name == 'full_attention' and isinstance(mixer, KPoolSparseAttentionMixer):
            sparse = mixer
        else:
            refuse_export(f'kinds.{kind_name}.mixer', 'GLM5 requires KDA or k-pool attention respectively')
    mixture = model.mixture
    routed = model.sparse_layers
    fields: dict[str, object] = {
        'layer_types': ['deepseek_sparse_attention' if kind == 'full_attention' else kind for kind in types],
        'mlp_layer_types': ['sparse' if index in routed else 'dense' for index in range(model.num_layers)],
        'head_dim': 0, 'qk_rope_head_dim': 0, 'qk_head_dim': sparse.qk_nope_head_dim,
        'num_key_value_heads': model.num_heads, 'attention_dropout': 0.0,
        'rope_theta': None, 'rope_scaling': None, 'rope_parameters': None,
        'mhc': True, 'mla_use_nope': True, 'index_kpool_compress': True,
        'hc_mult': hc.hc_mult, 'hc_eps': hc.hc_eps, 'hc_sinkhorn_iters': hc.hc_sinkhorn_iters,
        'swiglu_limit': model.swiglu_limit, 'num_nextn_predict_layers': model.num_nextn_predict_layers,
        'indexer_types': ['full'] * model.num_layers,
        'index_share_for_mtp_iteration': model.index_share_for_mtp_iteration,
        'linear_attn_config': {
            'num_heads': linear.linear_num_heads, 'head_dim': linear.linear_head_dim,
            'short_conv_kernel_size': linear.linear_conv_kernel_dim,
            'gate_lower_bound': linear.linear_lower_bound,
            # configuration_glm5_next.py:197-199 replaces an absent bound unless disabled.
            'safe_gate': linear.linear_lower_bound is not None,
        },
        **dataclasses.asdict(sparse),
    }
    if mixture is not None:
        width = mixture.expert_features or model.hidden_features
        if mixture.shared_features < width:
            refuse_export("mixture.shared_features", "GLM5 needs a positive count of shared expert widths")
        fields.update(deepseek_moe_export(model, scheduled=False), moe_router_dtype="float32")
    return fields


GLM5_NEXT_ROUTES = Table(
    # Its depths hold no copies of the trunk's embedding and head (`Glm5NextMTP`'s state dict).
    layer_depth_routes(DECODER_ROUTES, copies=False),
    # transformers 5.16.1 ties the head only in the multimodal wrapper (modeling_glm5_next.py:2068), so a
    # reader of the text model alone takes it from its own tensor: a tied export writes it too.
    Copy(Route('lm_head.weight', 'params/embed_tokens/embedding', when=lambda config, slots: bool(
        config['tie_embeddings']))))
"""GLM-5 Next's names: DeepSeek-V3's, prediction depth d stored as layer `num_layers + d`, its block's
leaves named as a trunk layer's and its final norm as `shared_head.norm`, and mHC's site tensors
`hc_{site}_{leaf}`."""


def _glm_experts(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    layers = _records.integer(record['num_layers'], 'num_layers')
    record['mixture'] = deepseek_mixture(hf_config, layers, used)
    return record


GLM4_MOE_CONFIG = ConfigLayout(
    # The released configs spell the rotary flat (rope_theta beside partial_rotary_factor); a config
    # transformers wrote nests both under rope_parameters. Glm4MoeRotaryEmbedding builds any rope_type
    # over the rotated half (modeling_glm4_moe.py:55-59), YaRN among them, and Glm4MoeConfig sets an
    # absent factor to half the head (configuration_glm4_moe.py:123).
    decoder_layout(rope=partial_rope_part(default_factor=0.5, whole_is_none=True)),
    flag('use_qk_norm', 'qk_norm', default=False), constant('o_proj_bias', value=False),
    # Routing is DeepSeek V3's, with the shared experts and the dense first layers.
    ConfigPart(_glm_experts, deepseek_moe_export),
    integer('num_nextn_predict_layers', 'num_nextn_predict_layers', 0))
"""GLM 4.5's and 5's MoE config, both ways: q/k/v biased over a bias-free o_proj, half the head rotated in
the 'default' convention, DeepSeek V3's routing, and the checkpoint's MTP depths."""


def _glm_indexer_types(hf_config: Mapping[str, object], layers: int, used: set[str]) -> tuple[str, ...]:
    """Return each layer's indexer mode, 'full' or 'shared'.

    A 'full' layer runs its own indexer; a 'shared' one reuses the previous
    full layer's top-k. `GlmMoeDsaConfig.__post_init__` resolves them in this
    order (configuration_glm_moe_dsa.py:136-148): an explicit `indexer_types`
    as it stands, else the `index_topk_pattern` string, else the
    `index_topk_freq` / `index_skip_topk_offset` schedule. The released configs
    ship the list beside the schedule that produced it, and the reference reads
    the list alone when both are present.
    """
    used.update(('indexer_types', 'index_topk_pattern', 'index_topk_freq',
                 'index_skip_topk_offset'))
    stated_types = hf_config.get('indexer_types')
    types = None if stated_types is None else list(_records.strings(stated_types, 'indexer_types'))
    if types is None:
        pattern = hf_config.get('index_topk_pattern')
        if pattern is not None:
            letters = {'F': 'full', 'S': 'shared'}
            types = ([letters.get(letter, letter) for letter in pattern]
                     if isinstance(pattern, str)
                     else list(_records.strings(pattern, 'index_topk_pattern')))
        else:
            freq = max(_records.integer(hf_config.get('index_topk_freq', 1), 'index_topk_freq'), 1)
            offset = _records.integer(hf_config.get('index_skip_topk_offset', 2), 'index_skip_topk_offset')
            types = ['full' if max(index - offset + 1, 0) % freq == 0 else 'shared'
                     for index in range(layers)]
    if len(types) != layers:
        refuse(f"indexer_types of {len(types)} entries",
                f"the model has {layers} layers, one indexer mode each")
    unknown = sorted(set(types) - {'full', 'shared'})
    if unknown:
        refuse(f"indexer_types entries {unknown}",
                "a GLM layer runs its indexer ('full') or reuses the previous "
                "full layer's top-k ('shared')")
    if types and types[0] == 'shared':
        refuse("indexer_types starting with 'shared'",
                "the first layer has no earlier indexer to share "
                "(modeling_glm_moe_dsa.py:444-445)")
    return tuple(types)


def _glm_moe_dsa_config(hf_config: Mapping[str, object], used: set[str],
                        config: DecoderFields) -> DecoderFields:
    """Read a glm_moe_dsa config's own fields beside DeepSeek V3.2's (`GLM_MOE_DSA_CONFIG`).

    GLM-5, 5.1, 5.2 and 5.3 are DeepSeek V3.2's sparse MLA block under GLM's
    choices. The indexer rotates interleaved pairs like the main rope head
    (modeling_glm_moe_dsa.py:231-232). IndexShare layers own no indexer and
    attend the previous full layer's top-k (:313-318, :739-748). The MTP depth
    ships past num_hidden_layers as GLM 4.5's does.
    """
    layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    used.add('head_dim')
    # The indexer's rotation is not a dial: GlmMoeDsaIndexer.forward always
    # interleaves, and a flag saying otherwise describes no model the
    # reference builds.
    if hf_config.get('indexer_rope_interleave', True) is not True:
        refuse(f"indexer_rope_interleave {hf_config['indexer_rope_interleave']!r}",
                "GlmMoeDsaIndexer always rotates interleaved pairs")
    # GlmMoeDsaTopkRouter casts to float32 itself (:514), whatever
    # moe_router_dtype says; index_share_for_mtp_iteration tells a
    # speculative-decoding engine to reuse draft step 0's top-k on the later
    # draft steps, and transformers builds no MTP depth to read it.
    used.update(('indexer_rope_interleave', 'moe_router_dtype',
                 'index_share_for_mtp_iteration'))
    shared = tuple(index for index, kind in
                   enumerate(_glm_indexer_types(hf_config, layers, used))
                   if kind == 'shared')
    mixer = _records.record(config.get('mixer'), 'mixer')
    config['mixer'] = {'class': mixer['class'], 'fields': {**_records.record(mixer['fields'], 'mixer fields'),
                                                         'index_rope_interleave': True}}
    if shared:
        config['kv_shared_layers'] = shared
    return config


def _indexer_types(model: CausalTransformer) -> Mapping[str, object]:
    """Each layer's indexer as a full one or one sharing the previous full layer's (its kv-shared layers)."""
    shared = set(model.kv_shared_layers or ())
    return {'indexer_types': ['shared' if index in shared else 'full' for index in range(model.num_layers)]}


GLM_MOE_DSA_CONFIG = ConfigLayout(
    # GlmMoeDsaConfig.__post_init__:152 points head_dim at the rope slice whatever the config says
    # (GLM-5.2 ships 192 over a rope width of 64), so the field describes nothing the reference computes.
    deepseek_layout_of(sparse=True, points_head_dim=False),
    ConfigPart(_glm_moe_dsa_config, _indexer_types))
"""GLM-5's to 5.3's MoE DSA config, both ways: DeepSeek V3.2's sparse MLA block under GLM's choices."""


GLM5_NEXT_CONFIG = ConfigLayout(
    ConfigPart(_glm5_pattern, lambda model: {}),
    # Its heads are as wide as its NoPE latent's, and each attends its own key/value head; it rotates
    # nothing (`_glm5_next_config` states the base the reference never turns).
    decoder_layout(DecoderSpelling(reads=frozenset({'sliding_window', 'attention_bias'}),
                                   keys={'head_dim': 'qk_nope_head_dim',
                                         'num_key_value_heads': 'num_attention_heads'}),
                   rope=ROPE_APART),
    ConfigPart(_glm5_next_config, _glm5_next_export))
"""GLM-5.3-Flash's text config, both ways."""


GLM5_NEXT_TEXT = DecoderFamily(
    ('glm5_next_text',),
    GLM5_NEXT_CONFIG,
    lambda fields: any(
        isinstance(mixer, (KimiDeltaAttentionMixer, KPoolSparseAttentionMixer))
        for mixer in kind_mixers(fields)
    ),
    'glm5_next_text',
    'Glm5NextTextForCausalLM',
    routes=GLM5_NEXT_ROUTES, prediction_depth=layer_depth,
    preserve_source_layout=True,
    # The config class's norm epsilon.
    defaults={'rms_norm_eps': 1e-5, 'pad_token_id': 154820},
    padding_idx=True,
)

GLM4_MOE = DecoderFamily(
    ('glm4_moe',),
    GLM4_MOE_CONFIG,
    lambda fields: (
        fields.partial_rotary_type == 'default'
        and (mixture := fields.mixture) is not None
        and mixture.bias
    ),
    'glm4_moe',
    'Glm4MoeForCausalLM',
    routes=DEEPSEEK_ROUTES, prediction_depth=layer_depth,
    preserve_source_layout=True,
    # Glm4MoeConfig's defaults for the fields its decoder reads.
    defaults={'rms_norm_eps': 1e-5, 'num_key_value_heads': 8, 'max_position_embeddings': 131072,
              'n_shared_experts': 1, 'first_k_dense_replace': 1, 'num_nextn_predict_layers': 1},
    padding_idx=True,
)


# GLM's sparse block is V3.2's with the indexer rotating interleaved
# pairs, which no DeepSeek release does, so that field names the family.
GLM_MOE_DSA = DecoderFamily(
    ('glm_moe_dsa',),
    GLM_MOE_DSA_CONFIG,
    lambda fields: (
        isinstance(mixer := fields.mixer, MLAMixer)
        and mixer.index_topk is not None
        and mixer.index_rope_interleave
    ),
    'glm_moe_dsa',
    'GlmMoeDsaForCausalLM',
    routes=DEEPSEEK_ROUTES, prediction_depth=layer_depth,
    preserve_source_layout=True,
    export_nulls=('q_lora_rank',),
    # The config class's norm epsilon.
    defaults={'max_position_embeddings': 202752, 'rms_norm_eps': 1e-5},
    padding_idx=True,
)
