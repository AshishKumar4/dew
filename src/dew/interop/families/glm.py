"""Translate the GLM decoders: glm4_moe, glm_moe_dsa and glm5_next_text.

GLM 4.5 biases q/k/v over a bias-free output projection and rotates half the
head. glm_moe_dsa is DeepSeek V3.2's sparse MLA block under those names, told
apart by the indexer rotating interleaved pairs. GLM 5.3's text block pools its
MLA keys without position, runs KDA layers beside them and carries mHC, so its
export names every persistent leaf itself, checking that the load reads each
one back, and the shared writer stores them.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from dew import _records
from dew.interop.config_records import NativeFields
from dew.interop.decoder_config import (
    ROPE_APART,
    DecoderSpelling,
    decoder_layout,
    flag,
    integer,
    kind_mixer,
    partial_rope_part,
    single_depth,
)
from dew.interop.decoder_parts import (
    DECODER_ROUTES,
    DEEPSEEK_ROUTES,
    DecoderFamily,
    DecoderFields,
    fixed_fields,
    kind_mixers,
    layer_depth,
    layer_depth_routes,
    record_int,
    refuse,
    refuse_export,
    specified_layer_types,
)
from dew.interop.families.deepseek import GLM_ROUTER, V3_EXPERTS, deepseek_experts, deepseek_layout_of
from dew.interop.layouts import (
    ConfigField,
    ConfigLayout,
    ConfigPart,
    Copy,
    Nested,
    Route,
    Table,
    constant,
    unread,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.dsa_kpool import KPoolSparseAttentionMixer
from dew.nn.hyper_connections import HyperConnections
from dew.nn.kda import KimiDeltaAttentionMixer
from dew.nn.mla import MLAMixer
from dew.registry import Configured, configured


def _glm5_types(hf_config: Mapping[str, object], used: set[str]) -> tuple[str, ...]:
    """A GLM-5.3-Flash config's layers: linear attention, and its k-pool sparse attention as full."""
    layers = record_int(hf_config, 'num_hidden_layers')
    types = specified_layer_types(hf_config, used, ('linear_attention',) * 3 + ('deepseek_sparse_attention',))
    types = tuple('full_attention' if kind == 'deepseek_sparse_attention' else kind for kind in types)
    if len(types) != layers or set(types) - {'linear_attention', 'full_attention'}:
        refuse('layer_types', 'one linear_attention or deepseek_sparse_attention entry per layer')
    return types


def _glm5_pattern(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The layers' kinds (`_glm5_types`), which `linear_attn_config`'s `kda_layers` and `full_attn_layers`
    restate where it states them."""
    types = _glm5_types(hf_config, used)
    nested = hf_config.get('linear_attn_config')
    raw = _records.record({} if nested is None else nested, 'linear_attn_config')
    for name, kind in (('kda_layers', 'linear_attention'), ('full_attn_layers', 'full_attention')):
        if name in raw and list(_records.integers(raw[name], f'linear_attn_config.{name}')) != [
                index for index, value in enumerate(types) if value == kind]:
            refuse(f'linear_attn_config.{name}', 'disagrees with layer_types')
    record['layer_types'] = types
    return record


def _glm5_types_written(model: CausalTransformer) -> Mapping[str, object]:
    types = model.per_layer_types
    if len(types) != model.num_layers or set(types) - {'linear_attention', 'full_attention'}:
        refuse_export('layer_types', 'GLM5 uses linear and NoPE sparse attention')
    return {'layer_types': ['deepseek_sparse_attention' if kind == 'full_attention' else kind
                            for kind in types]}


def _glm5_fixed(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The fields GLM-5.3-Flash's reference fixes, each checked against its one value, and the indexer
    schedule against the one selection mode this loader builds; it rotates nothing, so the base it states
    is one the reference never turns."""
    for name, expected in _GLM5_FIXED.items():
        used.add(name)
        if hf_config.get(name, expected) != expected:
            refuse(name, f'GLM-5.3-Flash requires {expected!r}')
    used.update(('num_key_value_heads', 'attention_dropout'))
    kv_heads = hf_config.get('num_key_value_heads')
    if kv_heads is not None and kv_heads != hf_config['num_attention_heads']:
        refuse('num_key_value_heads', 'the reference requires one KV head per query head')
    if hf_config.get('attention_dropout', 0.0) != 0.0:
        refuse('attention_dropout', 'k-pool attention does not implement training dropout')
    layers = _records.integer(record['num_layers'], 'num_layers')
    if 'shared' in _glm_indexer_types(hf_config, layers, used):
        refuse('indexer_types', 'shared k-pool index selections are not implemented')
    record['rope_theta'] = 10000.0
    return record


def _glm5_fixed_written(model: CausalTransformer) -> Mapping[str, object]:
    """The fixed fields, and what the model runs that its config does not carry, held here: its block's
    cast, dropout and scalar, and a layer of its own indexer each."""
    fixed_fields(model, {'scale_after_cast': True, 'dropout_rate': 0.0, 'layer_scalar': None},
                 'GLM5 requires {0!r}')
    if model.sharing_layers:
        refuse_export('kv_shared_layers', 'the supported GLM5 layout owns an indexer per sparse layer')
    return {**_GLM5_FIXED, 'num_key_value_heads': model.num_heads, 'attention_dropout': 0.0,
            'rope_theta': None, 'rope_scaling': None, 'rope_parameters': None,
            'indexer_types': ['full'] * model.num_layers}


_GLM5_FIXED: Mapping[str, Configured] = MappingProxyType({
    'mhc': True, 'mla_use_nope': True, 'index_kpool_compress': True, 'moe_router_dtype': 'float32',
    'hidden_act': 'silu', 'qk_rope_head_dim': 0, 'head_dim': 0})
"""The fields GLM-5.3-Flash's reference fixes, at their one value (`_glm5_fixed`)."""


_LINEAR_FIELDS: Mapping[str, tuple[str, Configured]] = MappingProxyType({
    'num_heads': ('linear_num_heads', 64), 'head_dim': ('linear_head_dim', 128),
    'short_conv_kernel_size': ('linear_conv_kernel_dim', 4),
    'gate_lower_bound': ('linear_lower_bound', -5.0)})
"""The KDA mixer's fields, each by its key in `linear_attn_config` (its field's name at the top level) and
its class's default."""


def _glm5_linear(hf_config: Mapping[str, object], used: set[str],
                 record: NativeFields[KimiDeltaAttentionMixer]) -> NativeFields[KimiDeltaAttentionMixer]:
    """The KDA mixer's fields, from `linear_attn_config` or the top level; the nested object wins wherever
    it states a field. Its `kda_layers` and `full_attn_layers` are the layer kinds' (`_glm5_pattern`)."""
    used.update(('linear_attn_config', *(target for target, _ in _LINEAR_FIELDS.values())))
    nested = hf_config.get('linear_attn_config')
    raw = _records.record({} if nested is None else nested, 'linear_attn_config')
    extra = set(raw) - set(_LINEAR_FIELDS) - {'safe_gate', 'kda_layers', 'full_attn_layers'}
    if extra:
        refuse('linear_attn_config', f'unknown fields {sorted(extra)}')
    for source, (target, default) in _LINEAR_FIELDS.items():
        record[target] = configured(raw.get(source, hf_config.get(target, default)))
    # The nested safe-gate flag only supplies a missing lower bound (:197-199).
    if nested is not None and raw.get('safe_gate', True) and record['linear_lower_bound'] is None:
        record['linear_lower_bound'] = -5.0
    return record


def _glm5_linear_written(mixer: KimiDeltaAttentionMixer) -> Mapping[str, object]:
    return {'linear_attn_config': {
        **{source: getattr(mixer, target) for source, (target, _) in _LINEAR_FIELDS.items()},
        # configuration_glm5_next.py:197-199 replaces an absent bound unless disabled.
        'safe_gate': mixer.linear_lower_bound is not None}}


def _pools(hf_config: Mapping[str, object], used: set[str],
           record: NativeFields[KPoolSparseAttentionMixer]) -> NativeFields[KPoolSparseAttentionMixer]:
    """The k-pool budget, whole pools of a positive size, and the NoPE heads' width (`qk_head_dim`)."""
    used.add('qk_head_dim')
    pool = _records.integer(record['index_kpool'], 'index_kpool')
    if pool < 1 or _records.integer(record['index_topk'], 'index_topk') % pool:
        refuse('index_topk / index_kpool', 'the budget must contain whole positive-sized pools')
    if hf_config.get('qk_head_dim', record['qk_nope_head_dim']) != record['qk_nope_head_dim']:
        refuse('qk_head_dim', 'NoPE uses qk_nope_head_dim alone')
    return record


_KPOOL = ConfigLayout(
    *(integer(name, name) for name in ('q_lora_rank', 'kv_lora_rank', 'qk_nope_head_dim', 'v_head_dim',
                                       'index_n_heads', 'index_head_dim', 'index_topk', 'index_kpool')),
    flag('index_kpool_always_select_tail', 'index_kpool_always_select_tail', default=True),
    ConfigPart(_pools, lambda mixer: {'qk_head_dim': mixer.qk_nope_head_dim}, keys=('qk_head_dim',)))
"""The k-pool sparse attention's fields, both ways."""


def _glm5_schedule(hf_config: Mapping[str, object]) -> tuple[str, ...]:
    """Each layer's feed-forward, dense or sparse: GLM reads the explicit schedule, not DeepSeek's
    dense-prefix rule (modeling_glm5_next.py:1270-1272; configuration_glm5_next.py:160-163)."""
    layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    stated = hf_config.get('mlp_layer_types')
    schedule = (('dense',) * min(3, layers) + ('sparse',) * max(layers - 3, 0) if stated is None
                else _records.strings(stated, 'mlp_layer_types'))
    if len(schedule) != layers or set(schedule) - {'dense', 'sparse'}:
        refuse('mlp_layer_types', 'one dense or sparse entry per layer is required')
    return schedule


def _glm5_routed(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The routed layers, the schedule's sparse ones; the prediction depth a routed sparse-attention block
    past the last of them."""
    used.add('mlp_layer_types')
    routed = tuple(index for index, kind in enumerate(_glm5_schedule(hf_config)) if kind == 'sparse')
    if record.get('mixture') is not None:
        record['mixture'] = NativeFields(Mixture, {**_records.record(record['mixture'], 'mixture'),
                                                   'layers': routed})
    layers = _records.integer(record['num_layers'], 'num_layers')
    if record.get('num_nextn_predict_layers') and (
            'full_attention' not in _records.strings(record['layer_types'], 'layer_types')
            or layers - 1 not in routed):
        refuse('num_nextn_predict_layers', 'the prediction depth requires a routed sparse-attention block')
    return record


def _glm5_routed_written(model: CausalTransformer) -> Mapping[str, object]:
    mixture = model.mixture
    if mixture is not None and mixture.shared_features < (mixture.expert_features or model.hidden_features):
        refuse_export("mixture.shared_features", "GLM5 needs a positive count of shared expert widths")
    routed = set(model.sparse_layers)
    return {'mlp_layer_types': ['sparse' if index in routed else 'dense'
                                for index in range(model.num_layers)]}


def _mean_streams(model: CausalTransformer) -> HyperConnections:
    streams = model.hyper_connections
    if streams is None or streams.head != 'mean':
        refuse_export('hyper_connections', 'GLM5 contracts mHC streams by their unweighted mean')
    return streams


def _clamped(model: CausalTransformer) -> float:
    if model.swiglu_limit is None:
        refuse_export('swiglu_limit', 'GLM5 clamps both dense and routed SwiGLU branches')
    return model.swiglu_limit


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


GLM4_MOE_CONFIG = ConfigLayout(
    # The released configs spell the rotary flat (rope_theta beside partial_rotary_factor); a config
    # transformers wrote nests both under rope_parameters. Glm4MoeRotaryEmbedding builds any rope_type
    # over the rotated half (modeling_glm4_moe.py:55-59), YaRN among them, and Glm4MoeConfig sets an
    # absent factor to half the head (configuration_glm4_moe.py:123).
    decoder_layout(rope=partial_rope_part(default_factor=0.5, whole_is_none=True)),
    flag('use_qk_norm', 'qk_norm', default=False), constant('o_proj_bias', value=False),
    # Routing is DeepSeek V3's, with the shared experts and the dense first layers.
    V3_EXPERTS,
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
    ConfigPart(_glm5_pattern, _glm5_types_written), single_depth('num_nextn_predict_layers'),
    # Its heads are as wide as its NoPE latent's, and each attends its own key/value head; it rotates
    # nothing (`_glm5_fixed` states the base the reference never turns).
    decoder_layout(DecoderSpelling(reads=frozenset({'sliding_window', 'attention_bias'}),
                                   keys={'head_dim': 'qk_nope_head_dim',
                                         'num_key_value_heads': 'num_attention_heads'}),
                   rope=ROPE_APART),
    ConfigPart(_glm5_fixed, _glm5_fixed_written,
               keys=(*_GLM5_FIXED, 'num_key_value_heads', 'attention_dropout', 'indexer_types')),
    kind_mixer('linear_attention', 'kimi_delta_attention', KimiDeltaAttentionMixer, ConfigLayout(ConfigPart(
        _glm5_linear, _glm5_linear_written,
        keys=('linear_attn_config', *(target for target, _ in _LINEAR_FIELDS.values()))))),
    kind_mixer('full_attention', 'kpool_sparse_attention', KPoolSparseAttentionMixer, _KPOOL),
    deepseek_experts(GLM_ROUTER, lambda hf_config: 'sparse' in _glm5_schedule(hf_config)),
    ConfigPart(_glm5_routed, _glm5_routed_written, keys=('mlp_layer_types',)),
    Nested('hyper_connections', HyperConnections, ConfigLayout(
        integer('hc_mult', 'hc_mult'), ConfigField('hc_eps', 'hc_eps', parse=_records.number),
        integer('hc_sinkhorn_iters', 'hc_sinkhorn_iters'), constant('head', value='mean')), _mean_streams),
    ConfigField('swiglu_limit', 'swiglu_limit', parse=_records.number, value=_clamped),
    flag('index_share_for_mtp_iteration', 'index_share_for_mtp_iteration', default=False),
    # The router's auxiliary loss and logits shape training alone; the indexer rotates nothing; the routed
    # layers are the explicit schedule's, whatever DeepSeek's dense-prefix fields say.
    unread('output_router_logits', 'router_aux_loss_coef', 'indexer_rope_interleave', 'first_k_dense_replace',
           'moe_layer_freq'))
"""GLM-5.3-Flash's text config, both ways: NoPE pooled MLA, KDA and mHC (modeling_glm5_next.py:1259-1329),
every field the reference fixes checked and refused rather than dropped."""


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
