"""Translate the GLM decoders: glm4_moe, glm_moe_dsa and glm5_next_text.

GLM 4.5 biases q/k/v over a bias-free output projection and rotates half the
head. glm_moe_dsa is DeepSeek V3.2's sparse MLA block under those names, told
apart by the indexer rotating interleaved pairs. GLM 5.3's text block pools its
MLA keys without position, runs KDA layers beside them and carries mHC, so its
export writes the whole variables tree rather than a leaf at a time.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

from dew import records
from dew.interop.families.deepseek import _deepseek_config, _deepseek_layout, _deepseek_mixture
from dew.interop.families.qwen import _single_prediction_depth
from dew.interop.hf_decoders import (
    DecoderFields,
    KindFields,
    MixtureFields,
    _base_config,
    _check_tree,
    _dew_path,
    _fixed_fields,
    _fixed_mixture,
    _flatten,
    _hf_name,
    _record_float,
    _record_int,
    _refuse,
    _Ropes,
    _specified_layer_types,
    translate_config,
)
from dew.nn.backbones.causal_transformer import CausalTransformer, Mixture
from dew.nn.dsa_kpool import KPoolSparseAttentionMixer
from dew.nn.kda import KimiDeltaAttentionMixer


def _glm5_next_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a GLM-5.3-Flash text config into `CausalTransformer` fields.

    The block is NoPE pooled MLA, KDA and mHC
    (modeling_glm5_next.py:1259-1329). The reader builds four things from the
    config: `types`, one attention kind per layer; `linear`, the KDA mixer's
    fields; `sparse`, the k-pool mixer's; and `mixture`, the routed layers'
    geometry. Every field the reference fixes is checked and refused rather
    than dropped.
    """
    layers = _record_int(hf_config, 'num_hidden_layers')
    types = _specified_layer_types(hf_config, used, tuple(
        'deepseek_sparse_attention' if i % 4 == 3 else 'linear_attention' for i in range(layers)))
    types = tuple('full_attention' if kind == 'deepseek_sparse_attention' else kind for kind in types)
    if len(types) != layers or set(types) - {'linear_attention', 'full_attention'}:
        _refuse('layer_types', 'one linear_attention or deepseek_sparse_attention entry per layer')
    for name, expected in (('mhc', True), ('mla_use_nope', True), ('index_kpool_compress', True),
                           ('moe_router_dtype', 'float32'), ('hidden_act', 'silu'),
                           ('qk_rope_head_dim', 0), ('head_dim', 0)):
        used.add(name)
        if hf_config.get(name, expected) != expected:
            _refuse(name, f'GLM-5.3-Flash requires {expected!r}')
    if hf_config.get('qk_head_dim', hf_config['qk_nope_head_dim']) != hf_config['qk_nope_head_dim']:
        _refuse('qk_head_dim', 'NoPE uses qk_nope_head_dim alone')
    kv_heads = hf_config.get('num_key_value_heads')
    if kv_heads is not None and kv_heads != hf_config['num_attention_heads']:
        _refuse('num_key_value_heads', 'the reference requires one KV head per query head')
    if hf_config.get('attention_dropout', 0.0) != 0.0:
        _refuse('attention_dropout', 'k-pool attention does not implement training dropout')
    used.add('attention_dropout')
    if 'shared' in _glm_indexer_types(hf_config, layers, used):
        _refuse('indexer_types', 'shared k-pool index selections are not implemented')
    nested = hf_config.get('linear_attn_config')
    raw = {} if nested is None else nested
    if not isinstance(raw, Mapping):
        _refuse('linear_attn_config', 'expected an object')
    fields = {'num_heads': ('linear_num_heads', 64), 'head_dim': ('linear_head_dim', 128),
              'short_conv_kernel_size': ('linear_conv_kernel_dim', 4),
              'gate_lower_bound': ('linear_lower_bound', -5.0)}
    extra = set(raw) - set(fields) - {'safe_gate', 'kda_layers', 'full_attn_layers'}
    if extra:
        _refuse('linear_attn_config', f'unknown fields {sorted(extra)}')
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
                _refuse(f'linear_attn_config.{name}', 'disagrees with layer_types')
    sparse_fields = ('q_lora_rank', 'kv_lora_rank', 'qk_nope_head_dim', 'v_head_dim',
                     'index_n_heads', 'index_head_dim', 'index_topk', 'index_kpool')
    sparse = {name: _record_int(hf_config, name) for name in sparse_fields}
    if sparse['index_kpool'] < 1 or sparse['index_topk'] % sparse['index_kpool']:
        _refuse('index_topk / index_kpool', 'the budget must contain whole positive-sized pools')
    config = _base_config({**hf_config, 'layer_types': types, 'head_dim': sparse['qk_nope_head_dim'],
                          'num_key_value_heads': hf_config['num_attention_heads']},
                          used, rope=_Ropes(10000.0))
    kinds: dict[str, KindFields] = {}
    if 'linear_attention' in types:
        kinds['linear_attention'] = {'mixer': {'kind': 'kimi_delta_attention', **linear}}
    if 'full_attention' in types:
        kinds['full_attention'] = {'mixer': {'kind': 'kpool_sparse_attention', **sparse,
            'index_kpool_always_select_tail': hf_config.get('index_kpool_always_select_tail', True)}}
    # GLM reads the explicit schedule, not DeepSeek's dense-prefix rule
    # (modeling_glm5_next.py:1270-1272; configuration_glm5_next.py:160-163).
    schedule = hf_config.get('mlp_layer_types')
    if schedule is None:
        schedule = ['dense'] * min(3, layers) + ['sparse'] * max(layers - 3, 0)
    if (not isinstance(schedule, (list, tuple)) or len(schedule) != layers
            or any(kind not in ('dense', 'sparse') for kind in schedule)):
        _refuse('mlp_layer_types', 'one dense or sparse entry per layer is required')
    for key, expected in (('scoring_func', 'sigmoid'), ('topk_method', 'noaux_tc')):
        if hf_config.get(key, expected) != expected:
            _refuse(key, f'GLM5 uses {expected}')
    norm_topk = hf_config.get('norm_topk_prob', True)
    if not isinstance(norm_topk, bool):
        _refuse('norm_topk_prob', 'expected a boolean')
    routed = tuple(index for index, kind in enumerate(schedule) if kind == 'sparse')
    mixture: MixtureFields | None = None
    if routed:
        geometry = _deepseek_layout(hf_config, layers, used, sparse_layers=routed)
        mixture = {**geometry, 'score_function': 'sigmoid', 'bias': True,
                   'norm_topk_prob': norm_topk,
                   'groups': _record_int({'n_group': hf_config.get('n_group') or 1}, 'n_group'),
                   'groups_per_token': _record_int({'topk_group': hf_config.get('topk_group') or 1}, 'topk_group')}
    else:
        used.update(('n_routed_experts', 'num_local_experts', 'num_experts_per_tok',
                     'routed_scaling_factor', 'n_group', 'topk_group', 'n_shared_experts',
                     'moe_intermediate_size', 'first_k_dense_replace', 'moe_layer_freq',
                     'mlp_layer_types', 'aux_loss_alpha', 'seq_aux'))
    used.update(('scoring_func', 'topk_method', 'norm_topk_prob'))
    hc_fields = ('hc_mult', 'hc_eps', 'hc_sinkhorn_iters')
    config.update(kinds=kinds, mixture=mixture,
                  hyper_connections={'hc_mult': _record_int(hf_config, 'hc_mult'),
                                     'hc_eps': _record_float(hf_config, 'hc_eps'),
                                     'hc_sinkhorn_iters': _record_int(hf_config, 'hc_sinkhorn_iters'),
                                     'head': 'mean'},
                  swiglu_limit=_record_float(hf_config, 'swiglu_limit'),
                  index_share_for_mtp_iteration=bool(hf_config.get('index_share_for_mtp_iteration', False)),
                  num_nextn_predict_layers=_single_prediction_depth(hf_config, used, 'num_nextn_predict_layers'))
    # Native NextN uses normalized trunk states and a plain NoPE depth, not
    # trunk mHC (SGLang 97c6978 deepseek_nextn.py:177-187,247-298). The
    # existing full-attention kind is k-pool/NoPE; mtp_hyper_connections
    # deliberately remains its plain default.
    if (config.get('num_nextn_predict_layers')
            and ('full_attention' not in types or layers - 1 not in routed)):
        _refuse('num_nextn_predict_layers', 'the prediction depth requires a routed sparse-attention block')
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
    fixed = {
        'causal': True, 'pre_norms': True, 'sandwich_norms': False,
        'scale_offset': False, 'scale_after_cast': True, 'embedding_scale': False,
        'mlp': 'swiglu', 'dropout_rate': 0.0, 'layer_scalar': None,
        'altup': None, 'laurel_rank': None, 'per_layer_input_dim': None,
        'activation_sparsity_pattern': None, 'final_logit_softcap': None,
    }
    _fixed_fields(model, fixed, 'GLM5 requires {0!r}')
    hc = model.hyper_connections
    if hc is None or hc.head != 'mean':
        _refuse('hyper_connections', 'GLM5 contracts mHC streams by their unweighted mean')
    if model.sharing_layers:
        _refuse('kv_shared_layers', 'the supported GLM5 layout owns an indexer per sparse layer')
    if model.num_nextn_predict_layers not in (0, 1):
        _refuse('num_nextn_predict_layers', 'GLM5 has zero or one plain prediction depth')
    if model.swiglu_limit is None:
        _refuse('swiglu_limit', 'GLM5 clamps both dense and routed SwiGLU branches')
    types = model.per_layer_types
    if len(types) != model.num_layers or set(types) - {'linear_attention', 'full_attention'}:
        _refuse('layer_types', 'GLM5 uses linear and NoPE sparse attention')
    linear, sparse = KimiDeltaAttentionMixer(), KPoolSparseAttentionMixer()
    for kind_name in set(types):
        mixer = model.kind_of(kind_name).mixer or model.mixer
        if kind_name == 'linear_attention' and isinstance(mixer, KimiDeltaAttentionMixer):
            linear = mixer
        elif kind_name == 'full_attention' and isinstance(mixer, KPoolSparseAttentionMixer):
            sparse = mixer
        else:
            _refuse(f'kinds.{kind_name}.mixer', 'GLM5 requires KDA or k-pool attention respectively')
    mixture = model.mixture
    routed = model.sparse_layers
    if model.num_nextn_predict_layers and ('full_attention' not in types or model.num_layers - 1 not in routed):
        _refuse('num_nextn_predict_layers', 'the source NextN depth requires a routed NoPE block')
    fields: dict[str, object] = {
        'layer_types': ['deepseek_sparse_attention' if kind == 'full_attention' else kind for kind in types],
        'mlp_layer_types': ['sparse' if index in routed else 'dense' for index in range(model.num_layers)],
        'head_dim': 0, 'qk_rope_head_dim': 0, 'qk_head_dim': sparse.qk_nope_head_dim,
        'num_key_value_heads': model.num_heads, 'attention_dropout': 0.0,
        # Native embeddings have no padding row; omission selects the released
        # vocabulary-specific default (configuration_glm5_next.py:126).
        'pad_token_id': None,
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
        represented = {'experts', 'top_k', 'layers', 'every', 'scaling', 'groups',
                       'groups_per_token', 'expert_features', 'shared_features', 'norm_topk_prob',
                       'implementation', 'dispatch'}
        defaults = Mixture(experts=mixture.experts, score_function='sigmoid', bias=True)
        _fixed_mixture(mixture, defaults, represented,
                       'GLM5 uses biased grouped sigmoid routing with top-two group scores')
        width = mixture.expert_features or model.hidden_features
        if mixture.shared_features < width or mixture.shared_features % width:
            _refuse('mixture.shared_features', 'GLM5 needs an integral positive count of shared expert widths')
        fields.update(
            n_routed_experts=mixture.experts, num_experts_per_tok=mixture.top_k,
            moe_intermediate_size=width, n_shared_experts=mixture.shared_features // width,
            routed_scaling_factor=mixture.scaling, n_group=mixture.groups, topk_group=mixture.groups_per_token,
            norm_topk_prob=mixture.norm_topk_prob, scoring_func='sigmoid', topk_method='noaux_tc',
            moe_router_dtype='float32')
    return fields


def _glm5_next_export_weights(model: CausalTransformer, variables: Mapping[str, object],
                              config: Mapping[str, object]) -> dict[str, np.ndarray]:
    persistent = {name: tree for name, tree in variables.items() if name != 'cache'}
    _check_tree(persistent, model)
    fields = translate_config(config)
    tensors: dict[str, np.ndarray] = {}
    for name, raw in _flatten(persistent).items():
        collection, *path = name.split('.')
        leaf = np.asarray(raw)
        original = tuple(path)
        if path[0].startswith('layers_'):
            layer = int(path[0].removeprefix('layers_'))
            tail = path[1:]
        elif path[0].startswith('mtp_'):
            layer = model.num_layers + int(path[0].removeprefix('mtp_'))
            tail = path[2:] if path[1] == 'block' else path[1:]
            if tail == ['final_norm', 'scale']:
                tail = ['shared_head', 'norm', 'scale']
        else:
            target = _hf_name('.'.join(path), config)
            if target is None or _glm4_moe_path(target, fields) != (collection, *original):
                _refuse(name, 'the source has no matching persistent leaf')
            tensors[target] = np.ascontiguousarray(leaf.T if path[-1] == 'kernel' else leaf)
            continue
        prefix = f'model.layers.{layer}.'
        if len(tail) == 4 and tail[:2] == ['mlp', 'experts'] and tail[-1] == 'kernel':
            for expert, value in enumerate(leaf):
                target = prefix + f'mlp.experts.{expert}.{tail[2]}.weight'
                expected = (collection, *original[:-2], str(expert), *original[-2:])
                if _glm4_moe_path(target, fields) != expected:
                    _refuse(name, 'the expert tensor has no matching source path')
                tensors[target] = np.ascontiguousarray(value.T)
            continue
        if len(tail) == 2 and tail[0] in ('attn_hc', 'ffn_hc'):
            target = prefix + f"hc_{tail[0].removesuffix('_hc')}_{tail[1]}"
        else:
            ending = 'weight' if tail[-1] in ('kernel', 'scale') else tail[-1]
            target = prefix + '.'.join([*tail[:-1], ending])
        if _glm4_moe_path(target, fields) != (collection, *original):
            _refuse(name, 'the source has no matching persistent leaf')
        if target in tensors:
            _refuse(name, f'duplicate source tensor {target}')
        tensors[target] = np.ascontiguousarray(leaf.T if tail[-1] == 'kernel' else leaf)
    if model.tie_embeddings:
        tensors['lm_head.weight'] = tensors['model.embed_tokens.weight']
    return tensors


def _glm4_moe_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a glm4_moe config into `CausalTransformer` fields.

    GLM 4.5 and 5 bias q/k/v over a bias-free o_proj and rotate half the head
    in the 'default' convention. Routing is DeepSeek V3's, with the shared
    experts and the dense first layers. The MTP depths are the checkpoint's.
    """
    # The released configs spell the rotary flat (rope_theta beside
    # partial_rotary_factor); a config transformers wrote nests both under
    # rope_parameters, the spelling Glm4MoeRotaryEmbedding reads.
    entry = records.record(hf_config.get('rope_parameters') or {}, 'rope_parameters')
    rope_type = entry.get('rope_type', entry.get('type', 'default'))
    if rope_type not in ('default', 'none') or hf_config.get('rope_scaling') is not None:
        _refuse(f"rope_parameters (rope_type {rope_type!r})",
                "Glm4MoeRotaryEmbedding is the plain rotary")
    scaling = sorted(set(entry) - {'rope_type', 'type', 'rope_theta', 'partial_rotary_factor'})
    if scaling:
        _refuse(f"rope_parameters scaling fields {scaling}",
                "Glm4MoeRotaryEmbedding is the plain rotary")
    theta = records.number(entry.get('rope_theta', hf_config.get('rope_theta', 10000.0)),
                   'rope_parameters rope_theta')
    factor = records.number(entry.get('partial_rotary_factor',
                              hf_config.get('partial_rotary_factor', 1.0)),
                    'rope_parameters partial_rotary_factor')
    used.update(('rope_theta', 'rope_parameters', 'rope_scaling', 'use_qk_norm',
                 'partial_rotary_factor', 'num_nextn_predict_layers'))
    config = _base_config(hf_config, used, rope=_Ropes(theta),
                          qk_norm=bool(hf_config.get('use_qk_norm', False)))
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    config.update(
        o_proj_bias=False,
        partial_rotary_factor=None if factor == 1.0 else factor,
        partial_rotary_type='default',
        mixture=_deepseek_mixture(hf_config, layers, used),
        num_nextn_predict_layers=records.integer(hf_config.get('num_nextn_predict_layers', 0), 'num_nextn_predict_layers'),
    )
    return config


def _glm4_moe_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the variables-tree path for one GLM tensor name.

    GLM's MTP depths are the layers past num_hidden_layers, one block each.
    Depth d arrives as model.layers.{num_layers + d}.*: its own enorm, hnorm,
    eh_proj and shared_head.norm around a decoder block named like any layer,
    plus copies of the trunk's embedding and head. The depth shares those two
    here as in the reference, and `translate_weights` checks the copies.

    The copies map to the trunk's own leaves, since the tree holds one
    embedding and one head for the trunk and every depth. A trained export
    therefore writes the copies from the weights the depth read.
    """
    parts = name.split('.')
    if not (len(parts) >= 4 and parts[:2] == ['model', 'layers'] and parts[2].isdigit()
            and int(parts[2]) >= records.integer(config['num_layers'], 'num_layers')):
        return _dew_path(name, config)
    if int(parts[2]) >= records.integer(config["num_layers"], 'num_layers') + records.integer(config.get("num_nextn_predict_layers", 0), 'num_nextn_predict_layers'):
        raise ValueError(f"{name} names an undeclared prediction depth")
    depth = f"mtp_{int(parts[2]) - records.integer(config['num_layers'], 'num_layers')}"
    tail = parts[3:]
    if tail == ['embed_tokens', 'weight']:
        return ('params', 'embed_tokens', 'embedding')
    if tail == ['shared_head', 'head', 'weight']:
        # A tied trunk keeps the head in the embedding, which stores the
        # same [vocab, features] the copy does; an untied one has the
        # head's own kernel, which stores its transpose.
        return (('params', 'embed_tokens', 'embedding') if config['tie_embeddings']
                else ('params', 'lm_head', 'kernel'))
    if tail == ['shared_head', 'norm', 'weight']:
        return ('params', depth, 'final_norm', 'scale')
    if len(tail) == 2 and tail[0] in ('enorm', 'hnorm') and tail[1] == 'weight':
        return ('params', depth, tail[0], 'scale')
    if tail == ['eh_proj', 'weight']:
        return ('params', depth, 'eh_proj', 'kernel')
    path = _dew_path('.'.join(['model', 'layers', '0', *tail]), config)
    if path is None:
        return None
    return (path[0], depth, 'block', *path[2:])


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
    types = None if stated_types is None else list(records.strings(stated_types, 'indexer_types'))
    if types is None:
        pattern = hf_config.get('index_topk_pattern')
        if pattern is not None:
            letters = {'F': 'full', 'S': 'shared'}
            types = ([letters.get(letter, letter) for letter in pattern]
                     if isinstance(pattern, str)
                     else list(records.strings(pattern, 'index_topk_pattern')))
        else:
            freq = max(records.integer(hf_config.get('index_topk_freq', 1), 'index_topk_freq'), 1)
            offset = records.integer(hf_config.get('index_skip_topk_offset', 2), 'index_skip_topk_offset')
            types = ['full' if max(index - offset + 1, 0) % freq == 0 else 'shared'
                     for index in range(layers)]
    if len(types) != layers:
        _refuse(f"indexer_types of {len(types)} entries",
                f"the model has {layers} layers, one indexer mode each")
    unknown = sorted(set(types) - {'full', 'shared'})
    if unknown:
        _refuse(f"indexer_types entries {unknown}",
                "a GLM layer runs its indexer ('full') or reuses the previous "
                "full layer's top-k ('shared')")
    if types and types[0] == 'shared':
        _refuse("indexer_types starting with 'shared'",
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
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    # GlmMoeDsaConfig.__post_init__:152 points head_dim at the rope slice
    # whatever the config says (GLM-5.2 ships 192 over a rope width of 64),
    # so the field describes nothing the reference computes.
    config = _deepseek_config({**hf_config, 'head_dim': None}, used, sparse=True)
    used.add('head_dim')
    # The indexer's rotation is not a dial: GlmMoeDsaIndexer.forward always
    # interleaves, and a flag saying otherwise describes no model the
    # reference builds.
    if hf_config.get('indexer_rope_interleave', True) is not True:
        _refuse(f"indexer_rope_interleave {hf_config['indexer_rope_interleave']!r}",
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
    config['mixer'] = {**records.record(config.get('mixer'), 'mixer'),
                       'index_rope_interleave': True}
    if shared:
        config['kv_shared_layers'] = shared
    config['num_nextn_predict_layers'] = records.integer(hf_config.get('num_nextn_predict_layers', 0), 'num_nextn_predict_layers')
    return config
