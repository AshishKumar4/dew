"""Translate K2 Horizon (`k2_horizon`): IFM's dense decoders (IFM/K2-Horizon-0.9B, -3.7B and -7B) and
its mixture of value experts (IFM/K2-Horizon-MoVA-36B-A4B).

The block is Llama's, its RMS norms taking their statistics over
`layernorm_num_groups` contiguous groups of the width under one weight, in
fp32, the weight multiplying before the one cast (modeling_k2_horizon.py at
IFM/K2-Horizon-0.9B 383cb663, K2HorizonRMSNorm, :613-636); the 0.9B runs
YaRN. The MoVA release (modeling_k2_horizon.py at cca48b66) routes its sparse
layers' MLP to experts (`K2HorizonSparseMoeBlock`, :531-611) and their values
to value experts (`K2HorizonMoVAAttention`, :351-512, `dew.nn.value_experts`),
both by a sigmoid router whose bias enters the selection alone, normalized over
the chosen and scaled (`calc_router_weights`, :136-167), and gates every
attention output by its own projection under a SiLU or a softplus at beta log 2.
A per-head q/k norm, a partial rotary and a window, which no release turns on,
are refused by name.
"""

from collections.abc import Mapping

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    PLAIN_SPELLING,
    DecoderFamily,
    DecoderFields,
    base_config,
    decoder_tensors,
    dew_path,
    hf_tensor_name,
    read_rope,
    refuse,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture, NormSpec
from dew.nn.moe import EXPERT_BIAS, EXPERT_BIAS_COLLECTION
from dew.nn.value_experts import ValueExperts

_REMOTE = 'IFM/K2-Horizon-0.9B'
"""The repository whose remote code an exported config names, as the released ones do."""

_GROUPED = ('input_layernorm', 'post_attention_layernorm', 'norm')
"""The norms K2HorizonRMSNorm builds, each over `layernorm_num_groups` (:660-667, :836-839)."""

_EXPERTS = ('decoder_sparse_step', 'moe_intermediate_size', 'num_experts_per_tok', 'norm_topk_prob',
            'output_router_logits', 'router_aux_loss_coef', 'moe_gate_bias', 'num_shared_experts',
            'router_score_func', 'router_scaling_factor', 'mlp_only_layers', 'mova_num_experts',
            'mova_num_experts_per_tok')
"""The routed and value experts' fields, which build nothing at num_experts 0 (:651-659)."""

_GATES = {'silu': 'silu', 'softplus': 'softplus2'}
"""The attention gate functions the release names (:500-506), as the mixer's activations."""


def _route(hf: Mapping[str, object], experts: int, used: set[str], config: DecoderFields) -> None:
    """`config` with the sparse layers' MLP experts and value experts (:647-659): every layer outside
    `mlp_only_layers` at the `decoder_sparse_step` cadence routes its MLP, and its values too where
    `mova_num_experts` is set; both routers score by `router_score_func`, the bias (`moe_gate_bias`)
    entering the selection alone, normalized over the chosen and scaled by `router_scaling_factor`."""
    layers = _records.integer(hf['num_hidden_layers'], 'num_hidden_layers')
    dense = set(_records.integers(hf.get('mlp_only_layers', []), 'mlp_only_layers'))
    step = _records.integer(hf.get('decoder_sparse_step', 1), 'decoder_sparse_step')
    score = _records.text(hf.get('router_score_func', 'sigmoid'), 'router_score_func')
    if score not in ('sigmoid', 'softmax'):
        refuse(f'router_score_func {score!r}', 'the release scores its routers by a sigmoid or a softmax')
    scaling = float(_records.number(hf.get('router_scaling_factor', 1.0), 'router_scaling_factor'))
    bias = bool(hf.get('moe_gate_bias', False))
    width = _records.integer(hf['moe_intermediate_size'], 'moe_intermediate_size')
    config['mixture'] = native_fields(Mixture)(
        experts=experts, top_k=_records.integer(hf['num_experts_per_tok'], 'num_experts_per_tok'),
        layers=tuple(index for index in range(layers) if index not in dense and (index + 1) % step == 0),
        score_function=score, norm_topk_prob=bool(hf.get('norm_topk_prob', False)), scaling=scaling,
        bias=bias, expert_features=width,
        shared_features=width * _records.integer(hf.get('num_shared_experts', 0), 'num_shared_experts'))
    values = _records.integer(hf.get('mova_num_experts', 0), 'mova_num_experts')
    if values:
        top_k = _records.integer(hf['mova_num_experts_per_tok'], 'mova_num_experts_per_tok')
        # calc_router_weights normalizes whenever it chooses more than one (:158-159).
        config['value_experts'] = native_fields(ValueExperts)(
            experts=values, top_k=top_k, score_function=score, norm_topk_prob=top_k > 1, scaling=scaling,
            bias=bias)
    used.update(_EXPERTS)


def _k2_horizon_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    # K2HorizonConfig's defaults are a routed, q/k-normed model (configuration_k2_horizon.py:67-84).
    experts = _records.integer(hf.get('num_experts', 128), 'num_experts')
    if hf.get('query_key_norm', True):
        refuse('query_key_norm', 'its per-head norm weighs the whole projection, which no release sets')
    gate = None if hf.get('attention_gate_func') is None else _records.text(hf['attention_gate_func'],
                                                                             'attention_gate_func')
    if gate is not None and gate not in _GATES:
        refuse(f'attention_gate_func {gate!r}', 'the release gates by a SiLU or a softplus at beta log 2')
    head_dim = _records.integer(hf['head_dim'], 'head_dim')
    if hf.get('rope_head_dim') not in (None, head_dim):
        refuse(f"rope_head_dim {hf['rope_head_dim']!r}", f'every dense release rotates all {head_dim} dims')
    if hf.get('use_sliding_window'):
        refuse('use_sliding_window', 'no dense release windows its attention')
    if hf.get('tie_word_embeddings'):
        # K2HorizonForCausalLM declares no tied weights, so a tied checkpoint reloads an untrained head.
        refuse('tie_word_embeddings', 'the remote K2HorizonForCausalLM ties no head to its embedding')
    stated = hf.get('original_max_position_embeddings')
    entry = _records.record(hf.get('rope_parameters') or {}, 'rope_parameters')
    if stated is not None and stated != entry.get('original_max_position_embeddings'):
        refuse(f'original_max_position_embeddings {stated!r}', "the rotary reads its rope_parameters' own")
    used.update(('num_experts', 'query_key_norm', 'attention_gate_func', 'rope_head_dim',
                 'use_sliding_window', 'original_max_position_embeddings', 'layernorm_num_groups', *_EXPERTS))
    # K2HorizonConfig drops the window unless use_sliding_window is set, and its attention builds
    # biases where attention_bias says (:212-223).
    # The rotary builds transformers' YaRN where rope_parameters names it (:740-748).
    rope = read_rope(hf, used, PLAIN_SPELLING)
    config = base_config({**hf, 'sliding_window': None}, used, rope=rope, scale_after_cast=False,
                         reads=frozenset({'attention_bias', 'sliding_window'}))
    groups = _records.integer(hf.get('layernorm_num_groups', 1), 'layernorm_num_groups')
    if groups != 1:
        config['norms'] = {site: {'groups': groups} for site in _GROUPED}
    if gate is not None:
        # The gate is its own projection of the layer's input, one value a head's dimension (:390-394).
        config.update(output_gate=True, output_gate_activation=_GATES[gate],
                      mixer={'class': 'attention', 'fields': {'gate_projection': 'own'}})
    if experts:
        _route(hf, experts, used, config)
    return config


def _groups(model: CausalTransformer) -> int:
    """The groups of the block's input norm, which K2 Horizon's every norm shares."""
    return (model.norms or {}).get('input_layernorm', NormSpec()).groups


def _k2_horizon_export(model: CausalTransformer) -> Mapping[str, object]:
    record: dict[str, object] = {
        'layernorm_num_groups': _groups(model), 'num_experts': 0, 'query_key_norm': False,
        'attention_gate_func': ({value: name for name, value in _GATES.items()}[model.output_gate_activation]
                                if model.output_gate else None),
        'rope_head_dim': model.head_dim, 'use_sliding_window': False, 'mova_num_experts': 0,
        'auto_map': {'AutoConfig': f'{_REMOTE}--configuration_k2_horizon.K2HorizonConfig',
                     'AutoModel': f'{_REMOTE}--modeling_k2_horizon.K2HorizonModel',
                     'AutoModelForCausalLM': f'{_REMOTE}--modeling_k2_horizon.K2HorizonForCausalLM'},
    }
    mixture, values = model.mixture, model.value_experts
    if mixture is not None:
        width = mixture.expert_features or model.hidden_features
        sparse = set(range(model.num_layers) if mixture.layers is None else mixture.layers)
        record.update(
            num_experts=mixture.experts, num_experts_per_tok=mixture.top_k, moe_intermediate_size=width,
            num_shared_experts=mixture.shared_features // width, norm_topk_prob=mixture.norm_topk_prob,
            router_score_func=mixture.score_function, router_scaling_factor=mixture.scaling,
            moe_gate_bias=mixture.bias, decoder_sparse_step=1,
            mlp_only_layers=[index for index in range(model.num_layers) if index not in sparse])
    if values is not None:
        record.update(mova_num_experts=values.experts, mova_num_experts_per_tok=values.top_k)
    return record


def _routed_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """`dew_path` with the routers' own names: an MLP router's bias, which enters its selection alone
    (`Router.expert_bias`), and the value experts' router and projections (`RoutedValues`)."""
    parts = name.split('.')
    if len(parts) >= 6 and parts[:2] == ['model', 'layers'] and parts[2].isdigit():
        layer, tail = f'layers_{parts[2]}', parts[3:]
        values = (layer, 'self_attn', 'experts')
        if tail == ['mlp', 'gate', 'bias']:
            return (EXPERT_BIAS_COLLECTION, layer, 'mlp', 'gate', EXPERT_BIAS)
        if tail == ['self_attn', 'v_router', 'weight']:
            return ('params', *values, 'v_router', 'kernel')
        if tail == ['self_attn', 'v_router', 'bias']:
            return (EXPERT_BIAS_COLLECTION, *values, 'v_router', EXPERT_BIAS)
        if (len(tail) == 4 and tail[:2] == ['self_attn', 'v_experts'] and tail[2].isdigit()
                and tail[3] == 'weight'):
            return ('params', *values, tail[2], 'v_proj', 'kernel')
    return dew_path(name, config)


def _routed_name(dew_name: str, config: Mapping[str, object]) -> str | None:
    """`_routed_path` backwards, every other leaf by the shared names (`hf_tensor_name`)."""
    parts = dew_name.split('.')
    if parts[0].startswith('layers_'):
        layer, tail = f"model.layers.{parts[0].removeprefix('layers_')}", parts[1:]
        if tail == ['mlp', 'gate', EXPERT_BIAS]:
            return f'{layer}.mlp.gate.bias'
        if tail[:3] == ['self_attn', 'experts', 'v_router'] and tail[3:] in (['kernel'], [EXPERT_BIAS]):
            return f"{layer}.self_attn.v_router.{'weight' if tail[3] == 'kernel' else 'bias'}"
        if len(tail) == 5 and tail[:2] == ['self_attn', 'experts'] and tail[3:] == ['v_proj', 'kernel']:
            return f'{layer}.self_attn.v_experts.{tail[2]}.weight'
    return hf_tensor_name(dew_name, config)


# A model whose norms group their statistics; the 0.9B's single group exports as Llama, whose norm
# orders its cast and weight otherwise, the same product in fp32 (`hf_decoders._RUNTIME_FIELDS`),
# and whose rotary builds its YaRN.
K2_HORIZON = DecoderFamily(
    ('k2_horizon',), _k2_horizon_config,
    lambda fields: _groups(fields) != 1 or fields.value_experts is not None,
    'k2_horizon', 'K2HorizonForCausalLM', _k2_horizon_export, preserve_source_layout=False,
    weight_path=_routed_path, export_path=_routed_name, export_weights=decoder_tensors,
    # The remote K2HorizonConfig's context, which a YaRN record's omitted lengths read.
    defaults={'max_position_embeddings': 32768},
)
