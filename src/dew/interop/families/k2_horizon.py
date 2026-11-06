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
from dew.interop.decoder_config import DecoderSpelling, decoder_layout, expect, inert, remote_code
from dew.interop.decoder_parts import DECODER_ROUTES, DecoderFamily, DecoderFields, in_layer, refuse
from dew.interop.layouts import ConfigLayout, ConfigPart, Table
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


def _route(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The sparse layers' MLP experts and value experts (:647-659), none at `num_experts` 0: every layer
    outside `mlp_only_layers` at the `decoder_sparse_step` cadence routes its MLP, and its values too where
    `mova_num_experts` is set; both routers score by `router_score_func`, the bias (`moe_gate_bias`)
    entering the selection alone, normalized over the chosen and scaled by `router_scaling_factor`."""
    used.update(('num_experts', *_EXPERTS))
    # K2HorizonConfig's defaults are a routed model (configuration_k2_horizon.py:67-84).
    experts = _records.integer(hf_config.get('num_experts', 128), 'num_experts')
    if not experts:
        return record
    layers = _records.integer(record['num_layers'], 'num_layers')
    dense = set(_records.integers(hf_config.get('mlp_only_layers', []), 'mlp_only_layers'))
    step = _records.integer(hf_config.get('decoder_sparse_step', 1), 'decoder_sparse_step')
    score = _records.text(hf_config.get('router_score_func', 'sigmoid'), 'router_score_func')
    if score not in ('sigmoid', 'softmax'):
        refuse(f'router_score_func {score!r}', 'the release scores its routers by a sigmoid or a softmax')
    scaling = float(_records.number(hf_config.get('router_scaling_factor', 1.0), 'router_scaling_factor'))
    bias = bool(hf_config.get('moe_gate_bias', False))
    width = _records.integer(hf_config['moe_intermediate_size'], 'moe_intermediate_size')
    record['mixture'] = native_fields(Mixture)(
        experts=experts, top_k=_records.integer(hf_config['num_experts_per_tok'], 'num_experts_per_tok'),
        layers=tuple(index for index in range(layers) if index not in dense and (index + 1) % step == 0),
        score_function=score, norm_topk_prob=bool(hf_config.get('norm_topk_prob', False)), scaling=scaling,
        bias=bias, expert_features=width,
        shared_features=width * _records.integer(hf_config.get('num_shared_experts', 0),
                                                 'num_shared_experts'))
    values = _records.integer(hf_config.get('mova_num_experts', 0), 'mova_num_experts')
    if values:
        top_k = _records.integer(hf_config['mova_num_experts_per_tok'], 'mova_num_experts_per_tok')
        # calc_router_weights normalizes whenever it chooses more than one (:158-159).
        record['value_experts'] = native_fields(ValueExperts)(
            experts=values, top_k=top_k, score_function=score, norm_topk_prob=top_k > 1, scaling=scaling,
            bias=bias)
    return record


def _routed(model: CausalTransformer) -> Mapping[str, object]:
    mixture, values = model.mixture, model.value_experts
    written: dict[str, object] = {'num_experts': 0, 'mova_num_experts': 0}
    if mixture is not None:
        width = mixture.expert_features or model.hidden_features
        sparse = set(model.sparse_layers)
        written.update(
            num_experts=mixture.experts, num_experts_per_tok=mixture.top_k, moe_intermediate_size=width,
            num_shared_experts=mixture.shared_features // width, norm_topk_prob=mixture.norm_topk_prob,
            router_score_func=mixture.score_function, router_scaling_factor=mixture.scaling,
            moe_gate_bias=mixture.bias, decoder_sparse_step=1,
            mlp_only_layers=[index for index in range(model.num_layers) if index not in sparse])
    if values is not None:
        written.update(mova_num_experts=values.experts, mova_num_experts_per_tok=values.top_k)
    return written


def _gate(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The attention output's gate, its own projection of the layer's input, one value a head's dimension
    (:390-394), under `attention_gate_func`; none where unstated."""
    used.add('attention_gate_func')
    stated = hf_config.get('attention_gate_func')
    if stated is None:
        return record
    gate = _records.text(stated, 'attention_gate_func')
    if gate not in _GATES:
        refuse(f'attention_gate_func {gate!r}', 'the release gates by a SiLU or a softplus at beta log 2')
    record.update(output_gate=True, output_gate_activation=_GATES[gate],
                  mixer={'class': 'attention', 'fields': {'gate_projection': 'own'}})
    return record


def _rotated(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The rotated width, every dimension of the head (`rope_head_dim`), and the context its YaRN reads,
    its rope_parameters' own (`original_max_position_embeddings`)."""
    used.update(('rope_head_dim', 'original_max_position_embeddings'))
    head_dim = _records.integer(record['head_dim'], 'head_dim')
    if hf_config.get('rope_head_dim') not in (None, head_dim):
        refuse(f"rope_head_dim {hf_config['rope_head_dim']!r}",
               f'every dense release rotates all {head_dim} dims')
    stated = hf_config.get('original_max_position_embeddings')
    entry = _records.record(hf_config.get('rope_parameters') or {}, 'rope_parameters')
    if stated is not None and stated != entry.get('original_max_position_embeddings'):
        refuse(f'original_max_position_embeddings {stated!r}', "the rotary reads its rope_parameters' own")
    return record


def _grouped(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    used.add('layernorm_num_groups')
    groups = _records.integer(hf_config.get('layernorm_num_groups', 1), 'layernorm_num_groups')
    if groups != 1:
        record['norms'] = {site: {'groups': groups} for site in _GROUPED}
    return record


def _groups(model: CausalTransformer) -> int:
    """The groups of the block's input norm, which K2 Horizon's every norm shares."""
    return (model.norms or {}).get('input_layernorm', NormSpec()).groups


K2_HORIZON_CONFIG = ConfigLayout(
    # K2HorizonConfig drops the window unless use_sliding_window is set, and its attention builds
    # biases where attention_bias says (:212-223). The rotary builds transformers' YaRN where
    # rope_parameters names it (:740-748).
    decoder_layout(DecoderSpelling(reads=frozenset({'attention_bias', 'sliding_window'}),
                                   scale_after_cast=False)),
    inert('use_sliding_window', 'no dense release windows its attention', neutral=False),
    # K2HorizonForCausalLM declares no tied weights, so a tied checkpoint reloads an untrained head.
    inert('tie_word_embeddings', 'the remote K2HorizonForCausalLM ties no head to its embedding',
          neutral=False),
    expect('query_key_norm', 'its per-head norm weighs the whole projection, which no release sets',
           value=False, written=True, required=True),
    ConfigPart(_gate, lambda model: {'attention_gate_func': (
        {value: name for name, value in _GATES.items()}[model.output_gate_activation]
        if model.output_gate else None)}),
    ConfigPart(_rotated, lambda model: {'rope_head_dim': model.head_dim}),
    ConfigPart(_grouped, lambda model: {'layernorm_num_groups': _groups(model)}),
    ConfigPart(_route, _routed),
    remote_code(_REMOTE, 'k2_horizon', 'K2Horizon'))
"""K2 Horizon's config, both ways."""


_ROUTES = Table(
    # An MLP router's bias, which enters its selection alone (`Router.expert_bias`).
    in_layer('mlp.gate.bias', 'mlp/gate/' + EXPERT_BIAS).moved(EXPERT_BIAS_COLLECTION),
    # The value experts' router and projections (`RoutedValues`).
    in_layer('self_attn.v_router.weight', 'self_attn/experts/v_router/kernel'),
    in_layer('self_attn.v_router.bias',
             'self_attn/experts/v_router/' + EXPERT_BIAS).moved(EXPERT_BIAS_COLLECTION),
    in_layer('self_attn.v_experts.{expert}.weight', 'self_attn/experts/{expert}/v_proj/kernel'),
    DECODER_ROUTES)
"""K2 Horizon's names: the shared decoder's, with the routers' own."""


# A model whose norms group their statistics; the 0.9B's single group exports as Llama, whose norm
# orders its cast and weight otherwise, the same product in fp32 (`hf_decoders._RUNTIME_FIELDS`),
# and whose rotary builds its YaRN.
K2_HORIZON = DecoderFamily(
    ('k2_horizon',), K2_HORIZON_CONFIG,
    lambda fields: _groups(fields) != 1 or fields.value_experts is not None,
    'k2_horizon', 'K2HorizonForCausalLM', preserve_source_layout=False,
    routes=_ROUTES,
    # The remote K2HorizonConfig's context, which a YaRN record's omitted lengths read.
    defaults={'max_position_embeddings': 32768},
)
