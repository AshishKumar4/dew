"""Translate K2 Horizon (`k2_horizon`), IFM's dense decoders (IFM/K2-Horizon-0.9B, -3.7B and -7B).

The block is Llama's, its RMS norms taking their statistics over
`layernorm_num_groups` contiguous groups of the width under one weight, in
fp32, the weight multiplying before the one cast (modeling_k2_horizon.py at
IFM/K2-Horizon-0.9B 383cb663, K2HorizonRMSNorm, :613-636); the 0.9B runs
YaRN. The same config class describes routed experts, a mixture of value
experts (MoVA), a gated attention output, a per-head q/k norm, a partial
rotary and a window, which no dense release turns on: each is refused by name.
"""

from collections.abc import Mapping

from dew import records
from dew.interop.decoder_parts import (
    PLAIN_SPELLING,
    DecoderFamily,
    DecoderFields,
    base_config,
    read_rope,
    refuse,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import NormSpec

_REMOTE = 'IFM/K2-Horizon-0.9B'
"""The repository whose remote code an exported config names, as the released ones do."""

_GROUPED = ('input_layernorm', 'post_attention_layernorm', 'norm')
"""The norms K2HorizonRMSNorm builds, each over `layernorm_num_groups` (:660-667, :836-839)."""

_EXPERTS = ('decoder_sparse_step', 'moe_intermediate_size', 'num_experts_per_tok', 'norm_topk_prob',
            'output_router_logits', 'router_aux_loss_coef', 'moe_gate_bias', 'num_shared_experts',
            'router_score_func', 'router_scaling_factor', 'mlp_only_layers', 'mova_num_experts',
            'mova_num_experts_per_tok')
"""The routed and value experts' fields, which build nothing at num_experts 0 (:651-659)."""


def _k2_horizon_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    # K2HorizonConfig's defaults are a routed, q/k-normed model (configuration_k2_horizon.py:67-84).
    if records.integer(hf.get('num_experts', 128), 'num_experts'):
        refuse(f"num_experts {hf.get('num_experts', 128)!r}", 'only the dense K2 Horizon releases translate')
    if hf.get('query_key_norm', True):
        refuse('query_key_norm', 'its per-head norm weighs the whole projection, which no dense release sets')
    if hf.get('attention_gate_func') is not None:
        refuse(f"attention_gate_func {hf['attention_gate_func']!r}", 'no dense release gates its attention')
    head_dim = records.integer(hf['head_dim'], 'head_dim')
    if hf.get('rope_head_dim') not in (None, head_dim):
        refuse(f"rope_head_dim {hf['rope_head_dim']!r}", f'every dense release rotates all {head_dim} dims')
    if hf.get('use_sliding_window'):
        refuse('use_sliding_window', 'no dense release windows its attention')
    if hf.get('tie_word_embeddings'):
        # K2HorizonForCausalLM declares no tied weights, so a tied checkpoint reloads an untrained head.
        refuse('tie_word_embeddings', 'the remote K2HorizonForCausalLM ties no head to its embedding')
    stated = hf.get('original_max_position_embeddings')
    entry = records.record(hf.get('rope_parameters') or {}, 'rope_parameters')
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
    groups = records.integer(hf.get('layernorm_num_groups', 1), 'layernorm_num_groups')
    if groups != 1:
        config['norms'] = {site: {'groups': groups} for site in _GROUPED}
    return config


def _groups(model: CausalTransformer) -> int:
    """The groups of the block's input norm, which K2 Horizon's every norm shares."""
    return (model.norms or {}).get('input_layernorm', NormSpec()).groups


def _k2_horizon_export(model: CausalTransformer) -> Mapping[str, object]:
    return {
        'layernorm_num_groups': _groups(model), 'num_experts': 0, 'query_key_norm': False,
        'attention_gate_func': None, 'rope_head_dim': model.head_dim, 'use_sliding_window': False,
        'auto_map': {'AutoConfig': f'{_REMOTE}--configuration_k2_horizon.K2HorizonConfig',
                     'AutoModel': f'{_REMOTE}--modeling_k2_horizon.K2HorizonModel',
                     'AutoModelForCausalLM': f'{_REMOTE}--modeling_k2_horizon.K2HorizonForCausalLM'},
    }


# A model whose norms group their statistics; the 0.9B's single group exports as Llama, whose norm
# orders its cast and weight otherwise, the same product in fp32 (`hf_decoders._RUNTIME_FIELDS`),
# and whose rotary builds its YaRN.
K2_HORIZON = DecoderFamily(
    ('k2_horizon',), _k2_horizon_config, lambda fields: _groups(fields) != 1,
    'k2_horizon', 'K2HorizonForCausalLM', _k2_horizon_export, preserve_source_layout=False,
    # The remote K2HorizonConfig's context, which a YaRN record's omitted lengths read.
    defaults={'max_position_embeddings': 32768},
)
