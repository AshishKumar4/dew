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

from flax.traverse_util import flatten_dict

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    PLAIN_SPELLING,
    DecoderFamily,
    DecoderFields,
    KindFields,
    MixtureFields,
    Ropes,
    base_config,
    check_decoder_tree,
    fixed_fields,
    hf_tensor_name,
    kind_mixers,
    layer_depth,
    layer_depth_path,
    layout_tensors,
    leaf_layouts,
    read_partial_rope,
    record_float,
    record_int,
    refuse,
    specified_layer_types,
    translated,
)
from dew.interop.families.deepseek import LAYOUT_FIELDS, deepseek_config, deepseek_mixture
from dew.interop.families.qwen import single_prediction_depth
from dew.interop.safetensors_io import LazyTensors
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


def _glm5_next_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a GLM-5.3-Flash text config into `CausalTransformer` fields.

    The block is NoPE pooled MLA, KDA and mHC
    (modeling_glm5_next.py:1259-1329). The reader builds four things from the
    config: `types`, one attention kind per layer; `linear`, the KDA mixer's
    fields; `sparse`, the k-pool mixer's; and `mixture`, the routed layers'
    geometry. Every field the reference fixes is checked and refused rather
    than dropped.
    """
    layers = record_int(hf_config, 'num_hidden_layers')
    types = specified_layer_types(hf_config, used, ('linear_attention',) * 3 + ('deepseek_sparse_attention',))
    types = tuple('full_attention' if kind == 'deepseek_sparse_attention' else kind for kind in types)
    if len(types) != layers or set(types) - {'linear_attention', 'full_attention'}:
        refuse('layer_types', 'one linear_attention or deepseek_sparse_attention entry per layer')
    _glm5_refusals(hf_config, used, layers)
    linear = _glm5_linear_fields(hf_config, types)
    sparse_fields = ('q_lora_rank', 'kv_lora_rank', 'qk_nope_head_dim', 'v_head_dim',
                     'index_n_heads', 'index_head_dim', 'index_topk', 'index_kpool')
    sparse = {name: record_int(hf_config, name) for name in sparse_fields}
    if sparse['index_kpool'] < 1 or sparse['index_topk'] % sparse['index_kpool']:
        refuse('index_topk / index_kpool', 'the budget must contain whole positive-sized pools')
    config = base_config({**hf_config, 'layer_types': types, 'head_dim': sparse['qk_nope_head_dim'],
                          'num_key_value_heads': hf_config['num_attention_heads']},
                          used, rope=Ropes(10000.0))
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
        num_nextn_predict_layers=single_prediction_depth(hf_config, used, "num_nextn_predict_layers"),
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
        refuse('hyper_connections', 'GLM5 contracts mHC streams by their unweighted mean')
    if model.sharing_layers:
        refuse('kv_shared_layers', 'the supported GLM5 layout owns an indexer per sparse layer')
    if model.num_nextn_predict_layers not in (0, 1):
        refuse('num_nextn_predict_layers', 'GLM5 has zero or one plain prediction depth')
    if model.swiglu_limit is None:
        refuse('swiglu_limit', 'GLM5 clamps both dense and routed SwiGLU branches')
    types = model.per_layer_types
    if len(types) != model.num_layers or set(types) - {'linear_attention', 'full_attention'}:
        refuse('layer_types', 'GLM5 uses linear and NoPE sparse attention')
    linear, sparse = KimiDeltaAttentionMixer(), KPoolSparseAttentionMixer()
    for kind_name in set(types):
        mixer = model.kind_of(kind_name).mixer or model.mixer
        if kind_name == 'linear_attention' and isinstance(mixer, KimiDeltaAttentionMixer):
            linear = mixer
        elif kind_name == 'full_attention' and isinstance(mixer, KPoolSparseAttentionMixer):
            sparse = mixer
        else:
            refuse(f'kinds.{kind_name}.mixer', 'GLM5 requires KDA or k-pool attention respectively')
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
        if mixture.shared_features < width or mixture.shared_features % width:
            refuse(
                "mixture.shared_features", "GLM5 needs an integral positive count of shared expert widths"
            )
        fields.update(
            n_routed_experts=mixture.experts,
            num_experts_per_tok=mixture.top_k,
            moe_intermediate_size=width,
            n_shared_experts=mixture.shared_features // width,
            routed_scaling_factor=mixture.scaling,
            n_group=mixture.groups,
            topk_group=mixture.groups_per_token,
            norm_topk_prob=mixture.norm_topk_prob,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            moe_router_dtype="float32",
        )
    return fields


def _glm5_next_names(path: tuple[str, ...], config: Mapping[str, object],
                     fields: DecoderFields) -> str | tuple[str, ...]:
    """The source name of the leaf at `path`, its collection first, or of each
    expert's slice of a stacked expert kernel, refused unless the load reads
    it back into that leaf (`layer_depth_path`).

    Prediction depth d is layer `num_layers + d`, its block's leaves named as
    a trunk layer's and its final norm as `shared_head.norm`. mHC's site
    tensors are `hc_{site}_{leaf}`.
    """
    def checked(name: str | None, leaf: tuple[str, ...]) -> str:
        if name is None or layer_depth_path(name, fields) != leaf:
            refuse('.'.join(path), 'the source has no matching persistent leaf')
        return name

    head, *tail = path[1:]
    if head.startswith('layers_'):
        prefix = f"model.layers.{int(head.removeprefix('layers_'))}."
    elif head.startswith('mtp_'):
        layer = _records.integer(fields['num_layers'], 'num_layers') + int(head.removeprefix('mtp_'))
        prefix = f'model.layers.{layer}.'
        tail = tail[1:] if tail[0] == 'block' else tail
        if tail == ['final_norm', 'scale']:
            tail = ['shared_head', 'norm', 'scale']
    else:
        return checked(hf_tensor_name('.'.join(path[1:]), config), path)
    if len(tail) == 4 and tail[:2] == ['mlp', 'experts'] and tail[3] == 'kernel':
        return tuple(checked(f'{prefix}mlp.experts.{expert}.{tail[2]}.weight',
                             (*path[:-2], str(expert), *path[-2:]))
                     for expert in range(_records.integer(config['n_routed_experts'], 'n_routed_experts')))
    if len(tail) == 2 and tail[0] in ('attn_hc', 'ffn_hc'):
        return checked(f"{prefix}hc_{tail[0].removesuffix('_hc')}_{tail[1]}", path)
    ending = 'weight' if tail[-1] in ('kernel', 'scale') else tail[-1]
    return checked(prefix + '.'.join([*tail[:-1], ending]), path)


def _glm5_next_export_weights(family: DecoderFamily, model: CausalTransformer,
                              variables: Mapping[str, object], config: Mapping[str, object]) -> LazyTensors:
    """Write every collection but the decode cache under `_glm5_next_names`.

    The collections are first held to the model's own (`check_decoder_tree`),
    so a leaf the model does not declare is refused rather than dropped.
    """
    persistent = {name: tree for name, tree in variables.items() if name != 'cache'}
    check_decoder_tree(persistent, model)
    fields = translated(config, family)
    layouts = leaf_layouts(flatten_dict(persistent), lambda path: _glm5_next_names(path, config, fields))
    if model.tie_embeddings:
        # transformers 5.16.1 ties the head only in the multimodal wrapper
        # (modeling_glm5_next.py:2068), so a reader of the text model alone
        # takes it from its own tensor.
        layouts['lm_head.weight'] = dataclasses.replace(layouts['model.embed_tokens.weight'],
                                                        name='lm_head.weight')
    return layout_tensors(layouts, persistent)


def _glm4_moe_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a glm4_moe config into `CausalTransformer` fields.

    GLM 4.5 and 5 bias q/k/v over a bias-free o_proj and rotate half the head
    in the 'default' convention. Routing is DeepSeek V3's, with the shared
    experts and the dense first layers. The MTP depths are the checkpoint's.
    """
    # The released configs spell the rotary flat (rope_theta beside
    # partial_rotary_factor); a config transformers wrote nests both under
    # rope_parameters. Glm4MoeRotaryEmbedding builds any rope_type over the
    # rotated half (modeling_glm4_moe.py:55-59), YaRN among them, and
    # Glm4MoeConfig sets an absent factor to half the head (configuration_glm4_moe.py:123).
    ropes, factor = read_partial_rope(hf_config, used, PLAIN_SPELLING, default_factor=0.5)
    used.update(('use_qk_norm', 'num_nextn_predict_layers'))
    config = base_config(hf_config, used, rope=ropes,
                          qk_norm=bool(hf_config.get('use_qk_norm', False)))
    layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    config.update(
        o_proj_bias=False,
        partial_rotary_factor=None if factor == 1.0 else factor,
        partial_rotary_type="default",
        mixture=deepseek_mixture(hf_config, layers, used),
        num_nextn_predict_layers=_records.integer(
            hf_config.get("num_nextn_predict_layers", 0), "num_nextn_predict_layers"
        ),
    )
    return config


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


def _glm_moe_dsa_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a glm_moe_dsa config into `CausalTransformer` fields.

    GLM-5, 5.1, 5.2 and 5.3 are DeepSeek V3.2's sparse MLA block under GLM's
    choices. The indexer rotates interleaved pairs like the main rope head
    (modeling_glm_moe_dsa.py:231-232). IndexShare layers own no indexer and
    attend the previous full layer's top-k (:313-318, :739-748). The MTP depth
    ships past num_hidden_layers as GLM 4.5's does.
    """
    layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    # GlmMoeDsaConfig.__post_init__:152 points head_dim at the rope slice
    # whatever the config says (GLM-5.2 ships 192 over a rope width of 64),
    # so the field describes nothing the reference computes.
    config = deepseek_config({**hf_config, 'head_dim': None}, used, sparse=True)
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
    config["num_nextn_predict_layers"] = _records.integer(
        hf_config.get("num_nextn_predict_layers", 0), "num_nextn_predict_layers"
    )
    return config


GLM5_NEXT_TEXT = DecoderFamily(
    ('glm5_next_text',),
    _glm5_next_config,
    lambda fields: any(
        isinstance(mixer, (KimiDeltaAttentionMixer, KPoolSparseAttentionMixer))
        for mixer in kind_mixers(fields)
    ),
    'glm5_next_text',
    'Glm5NextTextForCausalLM',
    _glm5_next_export,
    weight_path=layer_depth_path, prediction_depth=layer_depth,
    export_weights=_glm5_next_export_weights,
    preserve_source_layout=True,
    # The config class's norm epsilon.
    defaults={'rms_norm_eps': 1e-5, 'pad_token_id': 154820},
    padding_idx=True,
)

GLM4_MOE = DecoderFamily(
    ('glm4_moe',),
    _glm4_moe_config,
    lambda fields: (
        fields.partial_rotary_type == 'default'
        and (mixture := fields.mixture) is not None
        and mixture.bias
    ),
    'glm4_moe',
    'Glm4MoeForCausalLM',
    lambda model: {},
    weight_path=layer_depth_path, prediction_depth=layer_depth,
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
    _glm_moe_dsa_config,
    lambda fields: (
        isinstance(mixer := fields.mixer, MLAMixer)
        and mixer.index_topk is not None
        and mixer.index_rope_interleave
    ),
    'glm_moe_dsa',
    'GlmMoeDsaForCausalLM',
    lambda model: {},
    weight_path=layer_depth_path, prediction_depth=layer_depth,
    preserve_source_layout=True,
    # The config class's norm epsilon.
    defaults={'max_position_embeddings': 202752, 'rms_norm_eps': 1e-5},
    padding_idx=True,
)
