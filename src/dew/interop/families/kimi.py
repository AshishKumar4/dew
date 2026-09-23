"""Translate Kimi K3: the Kimi Linear decoder of moonshotai/Kimi-K3.

The release (revision f831ab66814297da540d832a5235f8e904f29d06, whose
modeling_kimi_linear.py and configuration_kimi_k3.py are unchanged since the
initial commit c5d1dd4) is a vision wrapper, `KimiK3ForConditionalGeneration`,
around `KimiLinearForCausalLM` built from `text_config`. Without pixels the
wrapper embeds the ids and runs the language model as it is
(modeling_kimi_k3.py:1145-1218), so the text half is the model here. Its
tower and projector tensors map to nothing and export writes their bytes back.

The decoder (modeling_kimi_linear.py) is Dew's backbone under these fields:

- KDA on the `linear_attn_config.kda_layers` (1-based) and NoPE MLA with a
  sigmoid output gate on the `full_attn_layers`: `kimi_delta_attention` with
  the full-rank output gate, and `mla` with `mla_use_nope` and
  `mla_use_output_gate`.
- A dense first `first_k_dense_replace` layers, then Stable LatentMoE:
  `num_experts` sigmoid-scored experts with DeepSeek's balancing bias, top
  `num_experts_per_token`, run at `routed_expert_hidden_size` between a down
  and an up projection with an RMSNorm on their sum, and `num_shared_experts`
  expert widths of shared branch at the model width.
- SiTU on every gated MLP (`activation_situ_beta`, `activation_situ_linear_beta`).
- Attention Residuals over blocks of `attn_res_block_size` layers.

The routed experts ship as compressed-tensors MXFP4 (`text_config.
quantization_config`), which `load_pretrained` decodes before this map runs.
Each KDA layer's `A_log` ships padded from `num_heads` to a longer zero tail
(128 for 96 heads), which `_kimi_k3_prepare` checks and trims.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from dew import records
from dew.interop.hf_decoders import (
    _ACTIVATIONS,
    _CODEC_FIELDS,
    DecoderFields,
    KindFields,
    MixtureFields,
    SituFields,
    _dew_path,
    _record_float,
    _record_int,
    _refuse,
)

# The text config's own fields the reader consumes, KimiLinearConfig's
# __init__ arguments (configuration_kimi_k3.py:1-100).
_K3_TEXT_FIELDS = frozenset({
    'activation_situ_beta', 'activation_situ_linear_beta', 'attn_res_block_size',
    'first_k_dense_replace', 'hidden_act', 'hidden_size', 'intermediate_size',
    'kv_lora_rank', 'latent_moe_use_norm', 'linear_attn_config', 'max_position_embeddings',
    'mla_use_nope', 'mla_use_output_gate', 'model_type', 'moe_intermediate_size',
    'moe_layer_freq', 'moe_renormalize', 'moe_router_activation_func', 'num_attention_heads',
    'num_expert_group', 'num_experts', 'num_experts_per_token', 'num_hidden_layers',
    'num_key_value_heads', 'num_nextn_predict_layers', 'num_shared_experts', 'q_lora_rank',
    'qk_nope_head_dim', 'qk_rope_head_dim', 'rms_norm_eps', 'routed_expert_hidden_size',
    'routed_scaling_factor', 'tie_word_embeddings', 'topk_group', 'topk_method',
    'use_grouped_topk', 'v_head_dim', 'vocab_size',
    # Rotary settings KimiLinearConfig stores and no layer reads: MLA runs
    # NoPE and KDA has no positions (modeling_kimi_linear.py:403, 437-440).
    'rope_theta', 'rope_scaling', 'head_dim',
})
# moonshotai/Kimi-K3's text_config was serialized by transformers 4.56.2,
# which writes every PreTrainedConfig attribute: decoding policy, token ids
# and metadata that no forward pass consults.
_K3_TEXT_SERIALIZED = frozenset({
    '_name_or_path', 'architectures', 'auto_map', 'bad_words_ids', 'begin_suppress_tokens',
    'bos_token_id', 'chunk_size_feed_forward', 'decoder_start_token_id', 'diversity_penalty',
    'do_sample', 'dtype', 'early_stopping', 'encoder_no_repeat_ngram_size', 'eos_token_id',
    'exponential_decay_length_penalty', 'finetuning_task', 'forced_bos_token_id',
    'forced_eos_token_id', 'id2label', 'initializer_range', 'is_decoder', 'is_encoder_decoder',
    'label2id', 'length_penalty', 'max_length', 'min_length', 'no_repeat_ngram_size',
    'num_beam_groups', 'num_beams', 'num_return_sequences', 'output_attentions',
    'output_hidden_states', 'output_scores', 'pad_token_id', 'prefix', 'problem_type',
    'remove_invalid_values', 'repetition_penalty', 'return_dict', 'return_dict_in_generate',
    'sep_token_id', 'suppress_tokens', 'task_specific_params', 'temperature', 'tf_legacy_loss',
    'tokenizer_class', 'top_k', 'top_p', 'torchscript', 'transformers_version', 'typical_p',
    'use_bfloat16', 'use_cache', 'torch_dtype',
})
# Serialized fields that would name another model if set, read by value.
_K3_TEXT_ENCODER = ('add_cross_attention', 'cross_attention_hidden_size',
                    'tie_encoder_decoder', 'pruned_heads')
_K3_LINEAR_FIELDS = frozenset({'full_attn_layers', 'kda_layers', 'num_heads', 'head_dim',
                               'short_conv_kernel_size', 'gate_lower_bound', 'use_full_rank_gate'})
# The wrapper's own fields: the head's tying, the tower and projector, and
# the media placeholders they fill, none of which reaches the text path.
_K3_WRAPPER_FIELDS = ('text_config', 'tie_word_embeddings', 'vision_config',
                      'media_placeholder_token_id', 'image_placeholder', 'ignore_index')


def _layer_schedule(linear: Mapping[str, object], layers: int) -> tuple[str, ...]:
    """One kind per layer from the 1-based `kda_layers` and `full_attn_layers`
    (`is_kda_layer`, configuration_kimi_k3.py:152-156; any other layer is MLA)."""
    kda = records.integers(linear.get('kda_layers'), 'linear_attn_config.kda_layers')
    full = records.integers(linear.get('full_attn_layers'), 'linear_attn_config.full_attn_layers')
    if sorted((*kda, *full)) != list(range(1, layers + 1)):
        _refuse('linear_attn_config.kda_layers/full_attn_layers',
                f'the two 1-based lists must name each of the {layers} layers once')
    return tuple('linear_attention' if index + 1 in kda else 'full_attention' for index in range(layers))


def _situ(text: Mapping[str, object]) -> SituFields:
    """The SiTU betas `_get_situ_activation_params` reads: an unset or zero
    beta is 1.0 (`beta or 1.0`), an unset linear beta leaves up uncapped."""
    beta = text.get('activation_situ_beta')
    linear = text.get('activation_situ_linear_beta')
    return {'beta': float(records.number(beta, 'activation_situ_beta')) if beta else 1.0,
            'linear_beta': None if linear is None else float(records.number(linear, 'activation_situ_linear_beta'))}


def _k3_mixture(text: Mapping[str, object], layers: int) -> MixtureFields | None:
    """The routed layers and their Stable LatentMoE, or None for a dense stack.

    A layer routes when `layer_idx >= first_k_dense_replace` and
    `layer_idx % moe_layer_freq == 0` (modeling_kimi_linear.py:893-900). The
    gate always selects on the bias-shifted scores and renormalizes above one
    expert (:703-759); it limits groups only when `num_expert_group` exceeds
    `topk_group`, scoring a group by its two best experts.
    """
    if text.get('num_experts') is None:
        return None
    experts = _record_int(text, 'num_experts')
    top_k = _record_int(text, 'num_experts_per_token')
    first = _record_int(text, 'first_k_dense_replace', 0)
    every = _record_int(text, 'moe_layer_freq', 1)
    score = records.text(text.get('moe_router_activation_func', 'sigmoid'), 'moe_router_activation_func')
    if score not in ('sigmoid', 'softmax'):
        _refuse('moe_router_activation_func', 'the reference scores by sigmoid or softmax')
    groups, per_token = _record_int(text, 'num_expert_group', 1), _record_int(text, 'topk_group', 1)
    width = _record_int(text, 'moe_intermediate_size')
    shared = _record_int(text, 'num_shared_experts', 0)
    latent = text.get('routed_expert_hidden_size')
    mixture: MixtureFields = {
        'experts': experts, 'top_k': top_k,
        'layers': tuple(index for index in range(layers) if index >= first and index % every == 0),
        'score_function': score, 'bias': True,
        'norm_topk_prob': bool(text.get('moe_renormalize', True)) and top_k > 1,
        'scaling': _record_float(text, 'routed_scaling_factor', 1.0),
        'groups': groups if groups > per_token else 1,
        'groups_per_token': per_token if groups > per_token else 1,
        'expert_features': width, 'shared_features': width * shared,
        'latent_features': None if latent is None else records.integer(latent, 'routed_expert_hidden_size'),
        'latent_norm': bool(text.get('latent_moe_use_norm', False)) and latent is not None,
    }
    return mixture


def _kimi_k3_text(text: Mapping[str, object], tied: bool) -> DecoderFields:
    """Read KimiLinearConfig's fields into `CausalTransformer` fields."""
    unknown = sorted(key for key in set(text) - _K3_TEXT_FIELDS - _K3_TEXT_SERIALIZED - set(_K3_TEXT_ENCODER)
                     - _CODEC_FIELDS
                     if not str(key).startswith('_'))
    if unknown:
        _refuse(f"text_config fields {unknown}", "KimiLinearConfig has no such field to compute")
    for key in _K3_TEXT_ENCODER:
        if text.get(key):
            _refuse(f"text_config {key}={text[key]!r}", "the decoder has no cross attention, encoder or pruned heads")
    if text.get('model_type', 'kimi_linear') != 'kimi_linear':
        _refuse(f"text_config model_type {text.get('model_type')!r}", "a Kimi K3 decoder is kimi_linear")
    if not text.get('mla_use_nope'):
        _refuse('mla_use_nope', 'KimiMLAAttention asserts NoPE (modeling_kimi_linear.py:396)')
    if _record_int(text, 'num_nextn_predict_layers', 0):
        _refuse('num_nextn_predict_layers', 'KimiLinearForCausalLM builds no prediction depth')
    heads = _record_int(text, 'num_attention_heads')
    if text.get('num_key_value_heads') not in (None, heads):
        _refuse('num_key_value_heads', 'the MLA expands one key and value per query head')
    activation = records.text(text.get('hidden_act', 'silu'), 'hidden_act')
    if activation != 'situ' and activation not in _ACTIVATIONS:
        _refuse(f"hidden_act {activation!r}", f"the gated MLP supports situ and {sorted(_ACTIVATIONS)}")
    layers = _record_int(text, 'num_hidden_layers')
    linear = records.record(text.get('linear_attn_config'), 'linear_attn_config')
    extra = sorted(set(linear) - _K3_LINEAR_FIELDS)
    if extra:
        _refuse(f'linear_attn_config fields {extra}', 'KimiDeltaAttention reads no such field')
    types = _layer_schedule(linear, layers)
    hidden = _record_int(text, 'hidden_size')
    kinds: dict[str, KindFields] = {}
    if 'linear_attention' in types:
        bound = linear.get('gate_lower_bound')
        kinds['linear_attention'] = {'mixer': {
            'kind': 'kimi_delta_attention',
            'linear_num_heads': _record_int(linear, 'num_heads'),
            'linear_head_dim': _record_int(linear, 'head_dim'),
            'linear_conv_kernel_dim': _record_int(linear, 'short_conv_kernel_size'),
            'linear_lower_bound': None if bound is None else float(records.number(bound, 'gate_lower_bound')),
            'use_full_rank_gate': bool(linear.get('use_full_rank_gate', False))}}
    if 'full_attention' in types:
        kinds['full_attention'] = {'mixer': {
            'kind': 'mla',
            'q_lora_rank': None if text.get('q_lora_rank') is None else _record_int(text, 'q_lora_rank'),
            'kv_lora_rank': _record_int(text, 'kv_lora_rank'),
            'qk_nope_head_dim': _record_int(text, 'qk_nope_head_dim'),
            'qk_rope_head_dim': _record_int(text, 'qk_rope_head_dim'),
            'v_head_dim': _record_int(text, 'v_head_dim'),
            'mla_use_nope': True,
            'mla_use_output_gate': bool(text.get('mla_use_output_gate', False))}}
    block = text.get('attn_res_block_size')
    config: DecoderFields = {
        'vocab_size': _record_int(text, 'vocab_size'),
        'emb_features': hidden,
        'num_layers': layers,
        'num_heads': heads,
        'num_kv_heads': heads,
        'head_dim': hidden // heads,
        'mlp': _situ(text) if activation == 'situ' else _ACTIVATIONS[activation],
        'mlp_features': _record_int(text, 'intermediate_size'),
        'max_seq_len': min(_record_int(text, 'max_position_embeddings', 4096), 8192),
        'layer_types': types,
        'kinds': kinds,
        # KimiRMSNorm scales after the cast (modeling_kimi_linear.py:232-236).
        'norm_eps': _record_float(text, 'rms_norm_eps', 1e-6),
        'scale_after_cast': True,
        'qk_norm': False,
        'tie_embeddings': tied,
        'mixture': _k3_mixture(text, layers),
        'attention_residuals': None if block is None else {'block_size': records.integer(block, 'attn_res_block_size')},
    }
    return config


def _kimi_k3_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Kimi K3 wrapper config into its text decoder's fields.

    The head is the text model's own `lm_head`, untied in the release; the
    wrapper's `tie_word_embeddings` is what the head is built under.
    """
    text = hf_config.get('text_config')
    if not isinstance(text, Mapping):
        _refuse('text_config', f"the wrapper carries its decoder under text_config, got {text!r}")
    tied = hf_config.get('tie_word_embeddings', False)
    if not isinstance(tied, bool):
        _refuse(f"tie_word_embeddings {tied!r}", "the head takes a boolean tying policy")
    used.update(_K3_WRAPPER_FIELDS)
    return _kimi_k3_text(text, tied)


_K3_LAYER_SITES = {'self_attention_res_norm': ('attention_res', 'scale'),
                   'self_attention_res_proj': ('attention_res', 'kernel'),
                   'mlp_res_norm': ('mlp_res', 'scale'),
                   'mlp_res_proj': ('mlp_res', 'kernel')}
_K3_MOE_PROJECTIONS = {'w1': 'gate_proj', 'w3': 'up_proj', 'w2': 'down_proj'}
_K3_KDA_LEAVES = ('q_proj', 'k_proj', 'v_proj', 'f_a_proj', 'f_b_proj', 'b_proj')


def _kimi_k3_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the variables-tree path of one Kimi K3 tensor name.

    The decoder sits under `language_model.model.*` with its head at
    `language_model.lm_head.weight` (model.safetensors.index.json at
    f831ab6). The standard norms, MLA projections and dense MLP map as
    DeepSeek's do; K3's own names are the attention residual sites, the
    KDA projections, the full-rank gate and the `block_sparse_moe` block.
    """
    if name.startswith(('vision_tower.', 'mm_projector.')):
        return None
    if name == 'language_model.lm_head.weight':
        return _dew_path('lm_head.weight', config)
    decoder = 'language_model.model.'
    if not name.startswith(decoder):
        raise ValueError(f"unknown tensor name {name!r}")
    parts = name[len(decoder):].split('.')
    if parts in (['output_attn_res_norm', 'weight'], ['output_attn_res_proj', 'weight']):
        return ('params', 'output_res', 'scale' if parts[0].endswith('norm') else 'kernel')
    if len(parts) < 3 or parts[0] != 'layers' or not parts[1].isdigit():
        return _dew_path('model.' + '.'.join(parts), config)
    layer, module, tail = f'layers_{parts[1]}', parts[2], parts[3:]
    if tail == ['weight'] and module in _K3_LAYER_SITES:
        return ('params', layer, *_K3_LAYER_SITES[module])
    if module == 'self_attn':
        if tail in (['A_log'], ['dt_bias'], ['o_norm', 'weight']):
            return ('params', layer, 'self_attn', *tail)
        if len(tail) == 2 and tail[1] == 'weight' and tail[0] in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
            return ('params', layer, 'self_attn', *tail)
        if len(tail) == 2 and tail[1] == 'weight' and tail[0] in (*_K3_KDA_LEAVES, 'g_proj'):
            return ('params', layer, 'self_attn', tail[0], 'kernel')
    if module == 'block_sparse_moe':
        if tail == ['gate', 'weight']:
            return ('params', layer, 'mlp', 'gate', 'kernel')
        if tail == ['gate', 'e_score_correction_bias']:
            return ('moe', layer, 'mlp', 'gate', 'e_score_correction_bias')
        if (len(tail) == 4 and tail[0] == 'experts' and tail[1].isdigit()
                and tail[2] in _K3_MOE_PROJECTIONS and tail[3] == 'weight'):
            return ('params', layer, 'mlp', 'experts', tail[1], _K3_MOE_PROJECTIONS[tail[2]], 'kernel')
        if len(tail) == 3 and tail[0] == 'shared_experts' and tail[2] == 'weight' and tail[1] in _K3_MOE_PROJECTIONS.values():
            return ('params', layer, 'mlp', 'shared_experts', tail[1], 'kernel')
        if tail in (['routed_expert_down_proj', 'weight'], ['routed_expert_up_proj', 'weight']):
            return ('params', layer, 'mlp', tail[0], 'kernel')
        if tail == ['routed_expert_norm', 'weight']:
            return ('params', layer, 'mlp', 'routed_expert_norm', 'scale')
        raise ValueError(f"unknown tensor name {name!r}")
    return _dew_path('model.' + '.'.join(parts), config)


_KDA_ZERO_PADDED = ('.self_attn.A_log',)
"""The tensors the release stores past their heads as zeros (`DecoderFamily.zero_padded`)."""


def _kimi_k3_prepare(tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    """Trim each KDA layer's zero-padded `A_log` to its heads.

    The release stores `A_log` `[128]` beside `dt_bias` `[96 * 128]`: 96
    heads' values then zeros (read off model-00001-of-000096.safetensors at
    f831ab6), where `KimiDeltaAttention` holds one per head
    (modeling_kimi_linear.py:520-521). The head count is `dt_bias` over the
    `f_a_proj` width; a nonzero tail is a checkpoint this loader would misread
    and is refused.
    """
    prepared = dict(tensors)
    for name, value in tensors.items():
        if not name.endswith(_KDA_ZERO_PADDED):
            continue
        stem = name.removesuffix('A_log')
        rank = tensors.get(stem + 'f_a_proj.weight')
        bias = tensors.get(stem + 'dt_bias')
        if rank is None or bias is None or bias.shape[0] % rank.shape[0]:
            raise ValueError(f"{name} has no dt_bias and f_a_proj to count its heads by")
        heads = bias.shape[0] // rank.shape[0]
        log = np.asarray(value)
        if log.ndim != 1 or log.shape[0] < heads:
            raise ValueError(f"{name} holds {log.shape}, fewer than its {heads} heads")
        if np.any(log[heads:]):
            raise ValueError(f"{name} pads its {heads} heads with nonzero values")
        prepared[name] = log[:heads]
    return prepared
