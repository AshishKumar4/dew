"""Hugging Face decoder checkpoints into CausalTransformer trees, and back.

translate_config and translate_weights are the map: a decoder config dict into
CausalTransformer kwargs, and HF-named tensors into a dew params tree. The
wrappers around them fetch a repo (or read a local directory), read the
safetensors shards as fp32 without torch, and build the model, so
load_pretrained_decoder returns a (model, variables, config) triple that a
forward pass takes straight away, config being the dew config the model was
built from.

The families covered are the ones CausalTransformer can express: llama, qwen3,
gemma3_text, gemma4_text, deepseek_v3 and deepseek_v32. qwen2 is refused
rather than half-loaded, since its q/k/v biases without an o_proj bias have
no counterpart in the backbone's one attention_bias flag, and a multimodal
wrapper config is refused rather than loading its text half. DeepSeek loads
through the MLA mixer with DeepSeek's MoE sizing, and its released
checkpoints carry `num_nextn_predict_layers: 1` with no `mtp.*` weights, so
translation builds the base model the weights describe. A config field that
changes what the model computes and has no dew counterpart raises a
ValueError naming it, rather than loading a model that silently computes
something else.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, NoReturn, Optional, Tuple

import numpy as np

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.registry import models, with_precision

CONFIG_FILE = "config.json"
GENERATION_CONFIG_FILE = "generation_config.json"

# The KV cache is allocated at the full decode length, so a 128k-context
# checkpoint would allocate one of those whether the caller asked or not.
DEFAULT_MAX_SEQ_LEN = 8192

# hidden_act / hidden_activation values, onto the GatedMLP activations. These
# are the two the covered families use; anything else is refused by name.
_ACTIVATIONS = {'silu': 'swiglu', 'gelu_pytorch_tanh': 'geglu'}

_QK_NORM_FAMILIES = ('qwen3', 'gemma3_text', 'gemma4_text')
_GEMMA = 'gemma3_text'
_DEEPSEEK = ('deepseek_v3', 'deepseek_v32')
_FAMILIES = ('llama', 'qwen3', 'gemma3_text', 'gemma4_text') + _DEEPSEEK
# A multimodal repo's config.json is a wrapper whose model_type names the
# whole model and whose text_config holds the decoder. Its own weights live
# under model.language_model.*, next to vision and audio towers this has no
# counterpart for, so the wrapper is refused by name.
_WRAPPERS = ('gemma3', 'gemma4', 'gemma4_unified', 'gemma3n')

_IGNORED_FIELDS = {
    'architectures', 'attention_dropout', 'attn_implementation', 'auto_map',
    'bos_token_id', 'cache_implementation', 'dtype', 'eos_token_id',
    'id2label', 'initializer_range', 'is_encoder_decoder', 'label2id',
    'max_window_layers', 'mlp_bias', 'output_attentions',
    'output_hidden_states', 'pad_token_id', 'pretraining_tp',
    'problem_type', 'return_dict', 'use_cache', 'use_sliding_window',
    'torch_dtype', 'transformers_version',
}

# The fields above have no effect on an eval-time forward pass: metadata,
# token ids, or runtime knobs of the reference implementation (Gemma 3 ships
# cache_implementation 'hybrid', which describes transformers' KV cache).


def _refuse(field: str, detail: str) -> NoReturn:
    raise ValueError(f"{field} is not expressible: {detail}")


def _rope_theta(entry: Optional[Mapping[str, Any]], field: str) -> Optional[float]:
    """One rope base frequency out of a rope_parameters entry, if it names one.

    Only plain rope ('default' or 'none') maps; a scaled variant changes what
    the model computes, so the caller refuses it with the field named. 'type'
    is the older spelling of rope_type and transformers still reads it
    (modeling_rope_utils.py:785, 839). Plain rope takes no field beyond those
    two and rope_theta, which is what its validator accepts
    (modeling_rope_utils.py:850-857), so a 'factor' or an
    'original_max_position_embeddings' names a scaling whatever the type says.
    """
    if entry is None:
        return None
    rope_type = entry.get('rope_type', entry.get('type', 'default'))
    if rope_type not in ('default', 'none'):
        _refuse(f"{field} (rope_type {rope_type!r})",
                "the backbone applies plain rotary positions at rope_theta")
    scaling = sorted(set(entry) - {'rope_type', 'type', 'rope_theta'})
    if scaling:
        _refuse(f"{field} scaling fields {scaling}",
                "the backbone applies plain rotary positions at rope_theta")
    theta = entry.get('rope_theta')
    return None if theta is None else float(theta)


def _rope(hf_config: Mapping[str, Any], used: set) -> Tuple[float, Optional[float]]:
    """(rope_theta, rope_local_theta) from any of the three HF spellings.

    Old configs carry flat rope_theta, gemma3 text configs add
    rope_local_base_freq, new configs nest per-layer-type rope_parameters.
    rope_scaling, when present, is the old name for the same thing.
    """
    used.update(('rope_theta', 'rope_local_base_freq', 'rope_parameters', 'rope_scaling'))
    rope_parameters = hf_config.get('rope_parameters')

    if isinstance(rope_parameters, Mapping) and 'rope_theta' not in rope_parameters:
        theta = (_rope_theta(rope_parameters.get('full_attention'),
                             'rope_parameters.full_attention') or 10000.0)
        local = (_rope_theta(rope_parameters.get('sliding_attention'),
                             'rope_parameters.sliding_attention') or theta)
        return theta, (None if local == theta else local)

    # Flat spellings: either field may carry the base frequency, and either
    # may carry a scaling type, which _rope_theta refuses when it is not plain.
    theta = None
    for field in ('rope_parameters', 'rope_scaling'):
        entry = hf_config.get(field)
        if isinstance(entry, Mapping):
            theta = _rope_theta(entry, field) or theta
    if theta is None:
        theta = float(hf_config.get('rope_theta', 10000.0))
    local = hf_config.get('rope_local_base_freq')
    return theta, (None if local is None else float(local))


def _layer_types(hf_config: Mapping[str, Any], used: set) -> Tuple[str, ...]:
    """Per-layer attention windows, derived the way the family's config does.

    A config that carries layer_types is taken at its word, except that
    gemma4_text forces its last layer full whatever the config says, which is
    what its own config class does (configuration_gemma4.py:197-201). Without
    layer_types, qwen3 makes every layer from max_window_layers on sliding,
    and only when use_sliding_window says so; gemma3_text repeats its
    sliding_window_pattern and gemma4_text a fixed period of six, five
    sliding layers to one full; llama has no sliding layers at all.
    """
    layers = int(hf_config['num_hidden_layers'])
    model_type = hf_config.get('model_type')
    layer_types = hf_config.get('layer_types')
    if layer_types is not None:
        used.add('layer_types')
        return _last_layer_full(tuple(layer_types), model_type)

    if model_type == 'qwen3':
        used.update(('use_sliding_window', 'sliding_window'))
        if (not hf_config.get('use_sliding_window', False)
                or hf_config.get('sliding_window') is None):
            return ('full_attention',) * layers
        first_sliding = int(hf_config.get('max_window_layers', layers))
        return tuple('sliding_attention' if index >= first_sliding else 'full_attention'
                     for index in range(layers))

    if model_type in (_GEMMA, 'gemma4_text'):
        # Gemma 3 reads its period from the config; Gemma 4 fixes it at six
        # (configuration_gemma4.py:190-195), so a gemma4 config that omits
        # the pattern is a 5:1 stack and not the all-full model this read as
        # one before.
        if model_type == _GEMMA:
            pattern = int(hf_config.get('sliding_window_pattern', 6))
            used.add('sliding_window_pattern')
        else:
            pattern = 6
        return _last_layer_full(
            tuple('sliding_attention' if (index + 1) % pattern else 'full_attention'
                  for index in range(layers)), model_type)

    return ('full_attention',) * layers


def _last_layer_full(layer_types: Tuple[str, ...], model_type: Optional[str]
                     ) -> Tuple[str, ...]:
    """Gemma 4's rule that the last layer attends the whole sequence.

    Gemma4TextConfig rewrites a trailing sliding layer to full and warns
    (configuration_gemma4.py:197-201), so the weights of a checkpoint whose
    config ends in a sliding layer were trained with a full one, and reading
    that config at its word would build a different model.
    """
    if model_type != 'gemma4_text' or not layer_types:
        return layer_types
    return layer_types[:-1] + ('full_attention',)


def _kinds(layer_types: Tuple[str, ...], window: Optional[int],
           local_theta: Optional[float], full_theta: Optional[float],
           full_head_dim: Optional[int]) -> Dict[str, Dict[str, Any]]:
    """What each named kind of the pattern does, as records.

    A family states its window and its local rope base for the sliding
    layers and its own head dim for the global ones; the pattern already
    names which layer is which, so each of those lands on that kind and the
    model's own `rope_theta` and `head_dim` stay the defaults.
    """
    kinds: Dict[str, Dict[str, Any]] = {}
    if 'sliding_attention' in layer_types:
        sliding: Dict[str, Any] = {'window': window}
        if local_theta is not None:
            sliding['rope_theta'] = local_theta
        kinds['sliding_attention'] = sliding
    if 'full_attention' in layer_types:
        full: Dict[str, Any] = {}
        if full_theta is not None:
            full['rope_theta'] = full_theta
        if full_head_dim is not None:
            full['head_dim'] = full_head_dim
        if full:
            kinds['full_attention'] = full
    return kinds


def _gemma4_rope(entries: Mapping[str, Any]) -> Tuple[float, Optional[float], Optional[float]]:
    """(rope_theta, rope_local_theta, partial_rotary_factor) for gemma4.

    The full layers may rotate a fraction of their head dims (proportional
    partial rotary); the sliding layers rotate all of theirs. Anything but
    those two shapes refuses with the entry named.
    """
    full = entries.get('full_attention') or {}
    sliding = entries.get('sliding_attention') or {}
    for kind, entry in (('sliding_attention', sliding),):
        factor = entry.get('partial_rotary_factor')
        if factor not in (None, 1, 1.0):
            _refuse(f"rope_parameters.{kind} partial_rotary_factor {factor}",
                    "partial rotary applies to the full layers only")
        _rope_theta({**entry, 'rope_type': entry.get('rope_type', entry.get('type', 'default'))},
                    f"rope_parameters.{kind}")
    local = sliding.get('rope_theta', 10000.0)
    kind, entry = 'full_attention', full
    rope_type = entry.get('rope_type', entry.get('type', 'default'))
    factor = entry.get('partial_rotary_factor')
    if rope_type == 'proportional':
        if factor is None:
            _refuse("rope_parameters.full_attention",
                    "proportional rope needs its partial_rotary_factor")
        extra = sorted(set(entry) - {'rope_type', 'type', 'rope_theta',
                                     'partial_rotary_factor', 'factor'})
        if extra or entry.get('factor', 1.0) not in (1, 1.0):
            _refuse("rope_parameters.full_attention scaling",
                    "the backbone applies plain rotary positions at rope_theta")
        partial = float(factor)
        theta = float(entry.get('rope_theta', 1000000.0))
    elif rope_type in ('default', 'none'):
        if factor not in (None, 1, 1.0):
            _refuse("rope_parameters.full_attention partial_rotary_factor",
                    "partial rotary comes spelled proportional")
        partial = None
        theta = _rope_theta(entry, 'rope_parameters.full_attention') or 10000.0
    else:
        _refuse(f"rope_parameters.full_attention (rope_type {rope_type!r})",
                "the backbone applies plain rotary positions at rope_theta")
    local = float(local)

    return theta, (None if local == theta else local), partial

_YARN_FIELDS = frozenset({
    'rope_type', 'type', 'rope_theta', 'factor', 'beta_fast', 'beta_slow',
    'mscale', 'mscale_all_dim', 'original_max_position_embeddings',
    'truncate', 'attention_factor', 'partial_rotary_factor',
})


def _yarn_record(entry: Mapping[str, Any], field: str, theta: float,
                 max_pos: int) -> Dict[str, Any]:
    """The mixer's yarn record out of a YaRN rope entry.

    Keeps the reference's names, so translation renames nothing; the
    mixer's YarnScaling is built from these keys. An explicit
    `attention_factor` rides along (the reference scales cos/sin by it
    instead of deriving one), while a partial rotary inside a YaRN entry
    has no counterpart in the mixer's full-width ramp and refuses. A
    missing factor falls back the way the reference does, to the context
    ratio off the original length.
    """
    unknown = sorted(set(entry) - _YARN_FIELDS)
    if unknown:
        _refuse(f"{field} fields {unknown}",
                "the YaRN ramp reads no such fields")
    partial = entry.get('partial_rotary_factor')
    if partial not in (None, 1, 1.0):
        _refuse(f"{field} partial_rotary_factor {partial}",
                "the mixer's YaRN ramp runs over the whole rope width")
    factor = entry.get('factor')
    if factor is None:
        factor = (float(max_pos)
                  / float(entry['original_max_position_embeddings']))
    return {
        'rope_type': 'yarn',
        'rope_theta': theta,
        'factor': float(factor),
        'original_max_position_embeddings': int(
            entry['original_max_position_embeddings']),
        'beta_fast': float(entry.get('beta_fast') or 32),
        'beta_slow': float(entry.get('beta_slow') or 1),
        'mscale': (None if entry.get('mscale') is None
                   else float(entry['mscale'])),
        'mscale_all_dim': (None if entry.get('mscale_all_dim') is None
                           else float(entry['mscale_all_dim'])),
        'truncate': bool(entry.get('truncate', True)),
        'attention_factor': (None if entry.get('attention_factor') is None
                             else float(entry['attention_factor'])),
    }


def _deepseek_rope(hf_config: Mapping[str, Any], used: set
                   ) -> Tuple[float, Optional[Dict[str, Any]]]:
    """(rope_theta, yarn record) from either rope spelling.

    Both released DeepSeek configs spell it the old way (`rope_scaling`
    with `type: yarn`); transformers prefers `rope_scaling` when both are
    present (convert_rope_params_to_dict), so this does too. Plain rope
    reuses the shared reader; anything but plain or YaRN changes the
    frequencies and refuses with the entry named.
    """
    used.update(('rope_theta', 'rope_parameters', 'rope_scaling'))
    scaling = hf_config.get('rope_scaling')
    parameters = hf_config.get('rope_parameters')
    entry = (scaling if isinstance(scaling, Mapping)
             else parameters if isinstance(parameters, Mapping) else None)
    theta = float(hf_config.get('rope_theta', 10000.0))
    max_pos = int(hf_config.get('max_position_embeddings',
                                DEFAULT_MAX_SEQ_LEN))
    if entry is None:
        return theta, None
    rope_type = entry.get('rope_type', entry.get('type', 'default'))
    field = ('rope_scaling' if scaling is entry else 'rope_parameters')
    if rope_type in ('default', 'none'):
        plain = _rope_theta(dict(entry, rope_theta=entry.get(
            'rope_theta', theta)), field)
        return plain or theta, None
    if rope_type == 'yarn':
        entry_theta = float(entry.get('rope_theta', theta))
        return entry_theta, _yarn_record(
            dict(entry, rope_theta=entry_theta), field, entry_theta, max_pos)
    _refuse(f"rope scaling (rope_type {rope_type!r})",
            "the mixer applies plain or YaRN rotary positions")
    raise AssertionError("unreachable")


def _deepseek_mixture(hf_config: Mapping[str, Any], layers: int,
                      used: set) -> Dict[str, Any]:
    """The mixture record out of a DeepSeek MoE config.

    The first `first_k_dense_replace` layers stay dense and the rest route;
    transformers never reads `moe_layer_freq`, so anything but every layer
    past the dense ones refuses, since the reference would build something
    else. `norm_topk_prob: false` has no Mixture knob yet and refuses with
    the field named rather than renormalising behind the config's back.
    """
    used.update(('n_routed_experts', 'num_local_experts',
                 'num_experts_per_tok', 'routed_scaling_factor',
                 'norm_topk_prob', 'n_group', 'topk_group',
                 'n_shared_experts', 'moe_intermediate_size',
                 'first_k_dense_replace', 'moe_layer_freq', 'topk_method',
                 'scoring_func', 'mlp_layer_types'))
    experts = hf_config.get('n_routed_experts',
                            hf_config.get('num_local_experts'))
    if experts is None:
        _refuse("n_routed_experts",
                "a DeepSeek MoE layer needs its expert count")
    scoring = hf_config.get('scoring_func', 'sigmoid')
    if scoring != 'sigmoid':
        _refuse(f"scoring_func {scoring!r}",
                "dew's router scores softmax, sigmoid or sqrtsoftplus, and "
                "this family's reference scores sigmoid")
    if hf_config.get('norm_topk_prob', True) is not True:
        _refuse("norm_topk_prob=False",
                "the mixture always renormalises the top-k weights; a knob "
                "not to is what this field would need")
    method = hf_config.get('topk_method')
    if method is not None and method != 'noaux_tc':
        _refuse(f"topk_method {method!r}",
                "the reference selects with the bias and the group limit, "
                "which is what noaux_tc names")
    freq = hf_config.get('moe_layer_freq')
    if freq is not None and freq != 1:
        _refuse(f"moe_layer_freq {freq!r}",
                "transformers builds every layer past the dense ones as MoE, "
                "whatever this field says")
    first_k = int(hf_config.get('first_k_dense_replace', 0) or 0)
    if not 0 <= first_k <= layers:
        _refuse(f"first_k_dense_replace {first_k!r}",
                f"it names dense layers of a {layers}-layer model")
    sparse = tuple(range(first_k, layers))
    pattern = hf_config.get('mlp_layer_types')
    if pattern is not None:
        expected = (['dense'] * first_k
                    + ['sparse'] * (layers - first_k))
        if list(pattern) != expected:
            _refuse(f"mlp_layer_types {list(pattern)!r}",
                    "it disagrees with first_k_dense_replace, which is what "
                    "the reference builds")
    shared = int(hf_config.get('n_shared_experts', 0) or 0)
    shared_features = 0
    if shared:
        width = hf_config.get('moe_intermediate_size')
        if width is None:
            _refuse("moe_intermediate_size",
                    "the shared experts need their width")
        shared_features = shared * int(width)
    return {
        'experts': int(experts),
        'top_k': int(hf_config['num_experts_per_tok']),
        'layers': sparse,
        'score_function': 'sigmoid',
        'scaling': float(hf_config.get('routed_scaling_factor', 1.0)),
        'groups': int(hf_config.get('n_group') or 1),
        'groups_per_token': int(hf_config.get('topk_group') or 1),
        'bias': True,
        'shared_features': shared_features,
        'expert_features': int(hf_config['moe_intermediate_size']),
    }


def translate_config(hf_config: Mapping[str, Any]) -> Dict[str, Any]:
    """A decoder config dict into CausalTransformer kwargs.

    Accepts the text decoder families CausalTransformer can express: llama,
    qwen3, gemma3_text, gemma4_text, deepseek_v3 and deepseek_v32. Every field
    that changes what a forward pass computes and has no dew counterpart
    raises, naming the field."""

    model_type = hf_config.get('model_type')
    if model_type == 'qwen2':
        # Qwen2 biases q/k/v and leaves o_proj bias-free (modeling_qwen2.py
        # 189-192), and CausalSelfAttention has one attention_bias flag for
        # all four projections, so its checkpoints cannot load unchanged.
        _refuse("model_type 'qwen2'",
                "its q/k/v projections carry biases and o_proj does not, which "
                "the one attention_bias flag cannot say")
    if model_type in _WRAPPERS or (model_type not in _FAMILIES
                                   and 'text_config' in hf_config):
        # google/gemma-4-E2B is one of these: the decoder is real and its
        # text_config translates, but the repo is a multimodal model whose
        # weights sit under model.language_model.* beside vision and audio
        # towers, and loading the text half would build something that is not
        # the checkpoint. The refusal names the text config so a caller who
        # wants the decoder alone asks for it deliberately.
        _refuse(f"model_type {model_type!r}",
                "it is a multimodal wrapper whose vision and audio towers have "
                "no counterpart here; its decoder is the text_config, which "
                "translates on its own, and its weights are the "
                "model.language_model.* half of the checkpoint")
    if model_type not in _FAMILIES:
        _refuse(f"model_type {model_type!r}",
                f"expected one of {', '.join(repr(name) for name in _FAMILIES)}")

    if hf_config.get('use_bidirectional_attention', False):
        _refuse("use_bidirectional_attention=True", "the backbone is causal")
    if hf_config.get('attn_logit_softcapping') is not None:
        _refuse("attn_logit_softcapping",
                "the attention kernels apply no softcap to the logits")
    if hf_config.get('mlp_bias'):
        _refuse("mlp_bias=True", "the gated MLP is bias-free")

    used = {'model_type', 'use_bidirectional_attention', 'attn_logit_softcapping',
            'mlp_bias', 'num_hidden_layers'}

    hidden = int(hf_config['hidden_size'])
    heads = int(hf_config['num_attention_heads'])
    kv_heads = hf_config.get('num_key_value_heads')
    head_dim = int(hf_config.get('head_dim') or hidden // heads)
    used.update(('hidden_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim'))

    activation = hf_config.get('hidden_act', hf_config.get('hidden_activation', 'silu'))
    used.update(('hidden_act', 'hidden_activation'))
    mapped = _ACTIVATIONS.get(activation)
    if mapped is None:
        _refuse(f"hidden_act {activation!r}",
                "the gated MLP supports 'swiglu' and 'geglu'")

    if model_type == 'gemma4_text':
        # Proportional partial rotary is a gemma4 shape with its own reader below.
        rope_theta, rope_local_theta = 10000.0, None
        yarn = None
    elif model_type in _DEEPSEEK:
        rope_theta, yarn = _deepseek_rope(hf_config, used)
        rope_local_theta = None
    else:
        rope_theta, rope_local_theta = _rope(hf_config, used)
        yarn = None
    layer_types = _layer_types(hf_config, used)
    sliding_window = hf_config.get('sliding_window')
    used.add('sliding_window')
    if 'sliding_attention' in layer_types and sliding_window is None:
        _refuse("layer_types with sliding attention",
                "sliding_window is not set, so the window has no size")
    if 'sliding_attention' not in layer_types:
        sliding_window = None

    config: Dict[str, Any] = {
        'vocab_size': int(hf_config['vocab_size']),
        'emb_features': hidden,
        'num_layers': int(hf_config['num_hidden_layers']),
        'num_heads': heads,
        'num_kv_heads': heads if kv_heads is None else int(kv_heads),
        'head_dim': head_dim,
        'mlp': mapped,
        'mlp_features': int(hf_config['intermediate_size']),
        'max_seq_len': min(int(hf_config.get('max_position_embeddings',
                                             DEFAULT_MAX_SEQ_LEN)),
                           DEFAULT_MAX_SEQ_LEN),
        'rope_theta': rope_theta,
        'layer_types': layer_types,
        'kinds': _kinds(layer_types, sliding_window, rope_local_theta, None, None),
        'norm_eps': float(hf_config.get('rms_norm_eps', 1e-6)),
        # LlamaRMSNorm and Qwen3RMSNorm multiply the scale into the
        # activations after casting them (modeling_qwen3.py:61-64); Gemma3's
        # and Gemma4's norms scale in fp32 and cast the product
        # (modeling_gemma3.py:147-150, modeling_gemma4.py:197-215).
        'scale_after_cast': model_type in ('llama', 'qwen3'),
        'qk_norm': model_type in _QK_NORM_FAMILIES,
        'attention_bias': bool(hf_config.get('attention_bias', False)),
        # Gemma3TextConfig ties by default, and so does Gemma4TextConfig; the
        # others do not, so a config that omits the field (gemma-3-1b-pt
        # does) has to take its family's default rather than a single one
        # here.
        'tie_embeddings': bool(hf_config.get(
            'tie_word_embeddings', model_type in (_GEMMA, 'gemma4_text'))),
    }
    used.update(('vocab_size', 'intermediate_size', 'max_position_embeddings',
                 'rms_norm_eps', 'attention_bias', 'tie_word_embeddings'))

    if model_type == _GEMMA:
        config.update(scale_offset=True, embedding_scale=True, sandwich_norms=True)
        used.update(('query_pre_attn_scalar', 'final_logit_softcapping'))
        scalar = hf_config.get('query_pre_attn_scalar')
        if scalar is not None:
            config['attention_scale'] = float(scalar) ** -0.5
        softcap = hf_config.get('final_logit_softcapping')
        if softcap is not None:
            config['final_logit_softcap'] = float(softcap)
    if model_type == 'gemma4_text':
        if hf_config.get('attention_k_eq_v'):
            _refuse("attention_k_eq_v=True",
                    "the backbone always projects its own values")
        if hf_config.get('enable_moe_block'):
            _refuse("enable_moe_block=True",
                    "the parallel MoE branch has no counterpart here")
        used.update(('attention_k_eq_v', 'enable_moe_block'))
        # The full layers may override the head dim; the sliding ones use
        # hidden // heads, so that is the model's and the override lands on
        # the full kind.
        sliding_dim = hidden // heads
        if hidden % heads:
            _refuse(f"hidden_size {hidden}",
                    f"it does not divide into {heads} heads")
        full_dim = sliding_dim
        entries = hf_config.get('per_layer_config') or {}
        model_kv = heads if kv_heads is None else int(kv_heads)
        for entry in (entries.values() if isinstance(entries, Mapping) else entries):
            if not isinstance(entry, Mapping):
                continue
            if entry.get('head_dim') is not None:
                full_dim = int(entry['head_dim'])
            # The reference gives a layer kind its own key/value head count as
            # well as its own head dim (modeling_gemma4.py,
            # Gemma4TextAttention reads layer_config.num_key_value_heads).
            # The backbone has one count for the model, so a config that
            # varies it is refused rather than built at the wrong K/V width.
            if (entry.get('num_key_value_heads') is not None
                    and int(entry['num_key_value_heads']) != model_kv):
                _refuse(f"per_layer_config num_key_value_heads "
                        f"{int(entry['num_key_value_heads'])}",
                        f"the backbone has one key/value head count for the "
                        f"model, which this config sets to {model_kv}")
        global_kv = hf_config.get('num_global_key_value_heads')
        if global_kv is not None and int(global_kv) != model_kv:
            _refuse(f"num_global_key_value_heads {int(global_kv)}",
                    f"the backbone has one key/value head count for the model, "
                    f"which this config sets to {model_kv}")
        if hf_config.get('global_head_dim') is not None:
            full_dim = int(hf_config['global_head_dim'])
        used.update(('per_layer_config', 'global_head_dim',
                     'num_global_key_value_heads'))
        # Proportional rope rotates a fraction of the full layers' head dims
        # and passes the rest through; sliding layers rotate all of theirs.
        entries = hf_config.get('rope_parameters') or {}
        rope_theta, rope_local_theta, partial = _gemma4_rope(entries)
        used.update(('rope_parameters', 'rope_theta'))
        per_layer = int(hf_config.get('hidden_size_per_layer_input', 0))
        config.update(
            sandwich_norms=True, embedding_scale=True, attention_scale=1.0,
            v_norm=True,
            head_dim=sliding_dim,
            rope_theta=rope_theta,
            kinds=_kinds(layer_types, config['kinds'].get(
                'sliding_attention', {}).get('window'), rope_local_theta, None,
                None if full_dim == sliding_dim else full_dim),
            partial_rotary_factor=partial,
            use_double_wide_mlp=bool(hf_config.get('use_double_wide_mlp', False)),
            num_kv_shared_layers=int(hf_config.get('num_kv_shared_layers', 0)),
            per_layer_input_dim=per_layer or None,
            per_layer_input_vocab=int(hf_config.get(
                'vocab_size_per_layer_input', int(hf_config['vocab_size']))),
        )
        used.update(('use_double_wide_mlp', 'num_kv_shared_layers',
                     'hidden_size_per_layer_input', 'vocab_size_per_layer_input',
                     'final_logit_softcapping'))
        # attention_logit_cap is read by no text path: Gemma4TextAttention
        # never passes it to its attention call (modeling_gemma4.py,
        # Gemma4TextAttention.forward), so it changes nothing and maps to
        # nothing. Only the audio attention applies one.
        used.add('attention_logit_cap')
        softcap = hf_config.get('final_logit_softcapping')
        if softcap is not None:
            config['final_logit_softcap'] = float(softcap)
        # MoE sizing keys with the branch off: refused above when it is on.
        used.update(('moe_intermediate_size', 'expert_intermediate_size',
                     'num_experts', 'top_k_experts', 'chunk_size_feed_forward'))

    if model_type in _DEEPSEEK:
        layers = int(hf_config['num_hidden_layers'])
        sparse_name = ('deepseek_sparse_attention'
                       if model_type == 'deepseek_v32' else 'full_attention')
        if hf_config.get('layer_types') is None:
            # Neither released config names its pattern: V3 is dense MLA
            # throughout and V3.2 sparse attention throughout.
            layer_types = (sparse_name,) * layers
            config['layer_types'] = layer_types
        for entry in layer_types:
            if entry != sparse_name:
                _refuse(f"layer_types entry {entry!r}",
                        f"a {model_type} model mixes no attention kinds: "
                        f"every layer is {sparse_name}")
        nope = int(hf_config['qk_nope_head_dim'])
        rope = int(hf_config['qk_rope_head_dim'])
        head_dim = hf_config.get('head_dim')
        if head_dim is not None and int(head_dim) != rope:
            _refuse(f"head_dim {head_dim!r}",
                    "DeepSeek points head_dim at the rope slice, "
                    f"which is {rope} wide here")
        derived = hf_config.get('qk_head_dim')
        if derived is not None and int(derived) != nope + rope:
            _refuse(f"qk_head_dim {derived!r}",
                    f"it derives as qk_nope_head_dim + qk_rope_head_dim, "
                    f"which is {nope + rope} here")
        v_dim = hf_config.get('v_head_dim')
        if v_dim is None:
            _refuse("v_head_dim",
                    "the values need their width, and no default keeps a "
                    "checkpoint's layout")
        kv_rank = hf_config.get('kv_lora_rank')
        if kv_rank is None:
            _refuse("kv_lora_rank",
                    "the latent needs its width, and no default keeps a "
                    "checkpoint's layout")
        interleave = hf_config.get('rope_interleave', True)
        if model_type == 'deepseek_v32' and interleave is not True:
            _refuse(f"rope_interleave {interleave!r}",
                    "the V3.2 reference always rotates interleaved pairs; a "
                    "flag saying otherwise describes no released model")
        index: Optional[Dict[str, int]] = None
        if model_type == 'deepseek_v32':
            index = {
                'index_topk': int(hf_config['index_topk']),
                'index_n_heads': int(hf_config['index_n_heads']),
                'index_head_dim': int(hf_config['index_head_dim']),
            }
            used.update(('index_topk', 'index_n_heads', 'index_head_dim'))
        # The released checkpoints ship no mtp.* weights (91991 tensors on
        # DeepSeek-V3 and 92425 on V3.2-Exp, none of them MTP), so the field
        # builds the base model the weights describe. Weight translation
        # refuses mtp.* tensors loudly, so a checkpoint that ships them
        # cannot drop them silently.
        # The fp8 scales name the stored dtype, not the computation: dew
        # loads the dequantized weights, and the reader names an unreadable
        # dtype where it meets one. ep_size is a runtime parallel hint.
        used.update(('num_nextn_predict_layers', 'num_mtp_layers'))
        used.update(('quantization_config', 'ep_size'))
        used.update(('qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim',
                     'kv_lora_rank', 'q_lora_rank', 'qk_head_dim',
                     'rope_interleave'))
        config.update(
            head_dim=nope + rope,
            mixer={
                'kind': 'mla',
                'q_lora_rank': (None if hf_config.get('q_lora_rank') is None
                                else int(hf_config['q_lora_rank'])),
                'kv_lora_rank': int(kv_rank),
                'qk_nope_head_dim': nope,
                'qk_rope_head_dim': rope,
                'v_head_dim': int(v_dim),
                'rope_interleave': bool(interleave),
                'yarn': yarn,
                'index_topk': None if index is None else index['index_topk'],
                'index_n_heads': None if index is None else index['index_n_heads'],
                'index_head_dim': (None if index is None
                                   else index['index_head_dim']),
            },
            mixture=_deepseek_mixture(hf_config, layers, used),
            num_nextn_predict_layers=0,
        )

    unknown = (set(hf_config) - used - _IGNORED_FIELDS
               - {key for key in hf_config if str(key).startswith('_')})
    if unknown:
        _refuse(f"config fields {sorted(unknown)}",
                "CausalTransformer has no counterpart, so translating them "
                "would silently change the model")
    return config


# Where a layer's norms sit in the two trees. Without the sandwich the names
# are the same; with it three of the four move, because HF names its norms
# after the sublayer they follow while dew names them after what they
# normalize: HF's post_attention_layernorm normalizes the attention output
# (our attention_output_norm), its pre_feedforward_layernorm is the MLP's
# pre-norm (our post_attention_layernorm) and its post_feedforward_layernorm
# normalizes the MLP output (our mlp_output_norm).
_PRE_NORMS = {
    'input_layernorm': 'input_layernorm',
    'post_attention_layernorm': 'post_attention_layernorm',
}
_SANDWICH_NORMS = {
    'input_layernorm': 'input_layernorm',
    'post_attention_layernorm': 'attention_output_norm',
    'pre_feedforward_layernorm': 'post_attention_layernorm',
    'post_feedforward_layernorm': 'mlp_output_norm',
}
_PROJECTIONS = {'self_attn': ('q_proj', 'k_proj', 'v_proj', 'o_proj'),
                'mlp': ('gate_proj', 'up_proj', 'down_proj', 'gate')}
_HEAD_NORMS = ('q_norm', 'k_norm')
# The MLA projections and norms live under self_attn beside the standard
# ones, with no counterpart in another family, so they extend the map by
# pattern: a tensor that is present maps, whatever the family.
_MLA_PROJECTIONS = ('q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa',
                    'kv_b_proj', 'o_proj')
_MLA_NORMS = ('q_a_layernorm', 'kv_a_layernorm')
# One leaf per projection for the router and the shared experts; the routed
# experts stack per-expert tensors (see _stack_experts).
_MOE_SHARED = ('gate_proj', 'up_proj', 'down_proj')


def _norm_names(sandwich: bool) -> Dict[str, str]:
    return _SANDWICH_NORMS if sandwich else _PRE_NORMS


def _dew_path(hf_name: str, config: Mapping[str, Any]) -> Optional[Tuple[str, ...]]:
    """One HF tensor name into its path in a CausalTransformer tree.

    None means the tensor has no place in the tree: the tied lm_head a
    checkpoint carries as a copy of the embedding. A name the map cannot
    explain at all raises, so an unfamiliar checkpoint fails here instead of
    loading a model with half its weights.
    """
    parts = hf_name.split('.')
    if parts == ['model', 'norm', 'weight']:
        return ('norm', 'scale')
    if parts == ['model', 'embed_tokens', 'weight']:
        return ('embed_tokens', 'embedding')
    if parts == ['model', 'embed_tokens_per_layer', 'weight']:
        return ('embed_tokens_per_layer', 'embedding')
    if parts == ['model', 'per_layer_model_projection', 'weight']:
        return ('per_layer_model_projection', 'kernel')
    if parts == ['model', 'per_layer_projection_norm', 'weight']:
        return ('per_layer_projection_norm', 'scale')
    if parts == ['lm_head', 'weight']:
        return None if config['tie_embeddings'] else ('lm_head', 'kernel')

    if len(parts) >= 5 and parts[:2] == ['model', 'layers'] and parts[2].isdigit():
        layer, module, leaf = f'layers_{parts[2]}', parts[3], parts[-1]
        if module in _PROJECTIONS and len(parts) == 6:
            sublayer = parts[4]
            if sublayer in _PROJECTIONS[module] and leaf in ('weight', 'bias'):
                # torch Linear holds [out, in]; nn.Dense keeps [in, out]
                return (layer, module, sublayer,
                        'kernel' if leaf == 'weight' else 'bias')
            if (module == 'self_attn' and sublayer in _HEAD_NORMS
                    and leaf == 'weight'):
                return (layer, module, sublayer, 'scale')
            if (module == 'self_attn' and sublayer in _MLA_PROJECTIONS
                    and leaf in ('weight', 'bias')):
                return (layer, module, sublayer,
                        'kernel' if leaf == 'weight' else 'bias')
            if (module == 'self_attn' and sublayer in _MLA_NORMS
                    and leaf == 'weight'):
                return (layer, module, sublayer, 'scale')
        if (len(parts) == 7 and module == 'self_attn'
                and parts[4] == 'indexer'):
            # model.layers.N.self_attn.indexer.{wq_b,wk,weights_proj}.weight
            # and k_norm.{weight,bias}: the sparse selector's own tensors.
            sublayer, leaf = parts[5], parts[6]
            if sublayer in ('wq_b', 'wk', 'weights_proj') and leaf == 'weight':
                return (layer, 'self_attn', 'indexer', sublayer, 'kernel')
            if sublayer == 'k_norm' and leaf in ('weight', 'bias'):
                return (layer, 'self_attn', 'indexer', sublayer,
                        'scale' if leaf == 'weight' else 'bias')
        if (len(parts) == 8 and module == 'mlp'
                and parts[4] in ('experts', 'shared_experts')):
            # model.layers.N.mlp.experts.K.{gate,up,down}_proj.weight, one
            # tensor per expert, stacked by _stack_experts below; and the
            # dense shared experts beside them.
            group, index, projection = parts[4], parts[5], parts[6]
            if projection in _MOE_SHARED and leaf == 'weight':
                if group == 'shared_experts':
                    return (layer, 'mlp', group, projection, 'kernel')
                if group == 'experts' and index.isdigit():
                    return (layer, 'mlp', group, index, projection, 'kernel')
        # Gemma 4's per-layer residual: gate and projection are kernels, the
        # post norm is a scale. The values norm carries no weight, so it maps
        # nothing.
        if len(parts) == 5 and leaf == 'weight':
            if module in ('per_layer_input_gate', 'per_layer_projection'):
                return (layer, module, 'kernel')
            if module == 'post_per_layer_input_norm':
                return (layer, module, 'scale')
        norms = _norm_names(bool(config.get('sandwich_norms')))
        if len(parts) == 5 and module in norms and leaf == 'weight':
            return (layer, norms[module], 'scale')
    raise ValueError(f"unknown tensor name {hf_name!r}")


def _stack_experts(params: Dict[str, Any]) -> None:
    """Per-expert `experts/K/projection` dicts into stacked `[E, ...]` leaves.

    A checkpoint names one tensor per expert while the tree keeps one leaf
    per projection stacked on an expert dimension, so after the flat map
    each sparse layer's digit-keyed dicts stack in expert order. A layer
    whose experts do not form a dense `0..E-1` run refuses rather than
    stacking a shuffled or partial set.
    """
    for layer, block in params.items():
        if not (isinstance(block, dict) and layer.startswith('layers_')):
            continue
        mlp = block.get('mlp')
        if not isinstance(mlp, dict):
            continue
        experts = mlp.get('experts')
        if not isinstance(experts, dict):
            continue
        indices = sorted(experts, key=int)
        if ([int(index) for index in indices]
                != list(range(len(indices)))):
            raise ValueError(
                f"{layer} experts {indices} are not a dense 0..E-1 run")
        stacked = {}
        for projection in experts[indices[0]]:
            leaves = [np.ascontiguousarray(experts[index][projection]['kernel'])
                      for index in indices]
            shapes = {leaf.shape for leaf in leaves}
            if len(shapes) != 1:
                raise ValueError(
                    f"{layer} experts disagree on {projection}: "
                    f"{sorted(shapes)}")
            stacked[projection] = {'kernel': np.stack(leaves)}
        mlp['experts'] = stacked


def translate_weights(hf_tensors: Mapping[str, np.ndarray],
                      config: Mapping[str, Any]) -> Dict[str, Any]:
    """HF-named tensors into a CausalTransformer params tree, in fp32.

    Linear weights arrive as [out, in] and nn.Dense keeps [in, out], so every
    `.kernel` is transposed; norm `.weight` becomes `.scale`; Gemma's
    post_attention_layernorm and post_feedforward_layernorm land on the
    sandwich norms, which is where Gemma applies them.

    A tied checkpoint carries lm_head.weight as well, as a copy of the
    embedding (Qwen3-0.6B does). The copy is checked and dropped: the tree has
    one leaf for the two, and a checkpoint whose "tied" head is a different
    matrix would otherwise load as a model that computes something else.
    DeepSeek's routed experts arrive one tensor per expert and stack onto
    an expert dimension here; its dense shared experts, MLA projections
    and indexer map by pattern like everything else.
    """
    tied_head = hf_tensors.get('lm_head.weight')
    if config['tie_embeddings'] and tied_head is not None:
        embedding = hf_tensors.get('model.embed_tokens.weight')
        if embedding is None or not np.array_equal(tied_head, embedding):
            raise ValueError(
                "tie_word_embeddings is set but lm_head.weight is not the "
                "embedding it claims to copy")

    params: Dict[str, Any] = {}
    for name, tensor in hf_tensors.items():
        path = _dew_path(name, config)
        if path is None:
            continue
        leaf = np.asarray(tensor, np.float32)
        if path[-1] == 'kernel':
            leaf = np.ascontiguousarray(leaf.T)
        node = params
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = leaf
    _stack_experts(params)
    return {'params': params}


_DTYPES = {'F32': np.float32, 'F16': np.float16}


def _read_shard(path: Path) -> Dict[str, np.ndarray]:
    """Every tensor of one safetensors file as fp32, without torch.

    safetensors.numpy cannot read bfloat16 and most decoder checkpoints are
    bfloat16, so those leaves are widened here the way every bf16 reader does:
    the 16 payload bits shifted into the top half of an fp32 word. The file is
    opened once and read in header order, which is offset order.
    """
    tensors: Dict[str, np.ndarray] = {}
    with open(path, 'rb') as handle:
        length = int.from_bytes(handle.read(8), 'little')
        header = json.loads(handle.read(length))
        data = 8 + length
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            dtype, shape = meta['dtype'], tuple(meta['shape'])
            start, end = meta['data_offsets']
            handle.seek(data + start)
            raw = handle.read(end - start)
            if dtype == 'BF16':
                widened = np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16
                tensors[name] = widened.view(np.float32).reshape(shape)
            elif dtype in _DTYPES:
                tensors[name] = np.frombuffer(
                    raw, dtype=_DTYPES[dtype]).astype(np.float32).reshape(shape)
            else:
                raise ValueError(
                    f"tensor {name} in {path.name} has dtype {dtype}, which this "
                    "loader cannot read")
    return tensors


def _load_shards(directory: Path) -> Dict[str, np.ndarray]:
    """Every tensor of a checkpoint directory, shard by shard, as fp32."""
    shards = sorted(directory.glob('*.safetensors'))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors under {directory}")
    tensors: Dict[str, np.ndarray] = {}
    for shard in shards:
        tensors.update(_read_shard(shard))
    return tensors


def _snapshot(name_or_dir: str, revision: Optional[str]) -> Path:
    """A local directory as given, or a hub snapshot downloaded once."""
    if os.path.isdir(name_or_dir):
        return Path(name_or_dir)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        name_or_dir, revision=revision,
        allow_patterns=["*.safetensors", "*.json"]))


def load_pretrained_decoder(name_or_dir: str, *, dtype: str = 'bfloat16',
                            attention_impl: str = 'auto',
                            max_seq_len: Optional[int] = None,
                            revision: Optional[str] = None
                            ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """A Hugging Face decoder checkpoint, as (model, variables, config).

    `name_or_dir` is a hub repo id or a local directory in the HF layout.
    The config is translated, the weights mapped onto that tree in fp32, and
    the model built with the run's precision policy, so `variables` fits the
    model and the policy's `dtype` reaches every module the same way it does
    in a training run. max_seq_len defaults to the config's context clamped
    to 8192, because the KV cache is allocated at that length. `config` is
    what the model was built from, in dew's own vocabulary, so a caller logs
    the model it ran rather than the checkpoint's own fields.
    """
    directory = _snapshot(name_or_dir, revision)
    with open(directory / CONFIG_FILE) as handle:
        hf_config = json.load(handle)

    config = translate_config(hf_config)
    if max_seq_len is not None:
        config['max_seq_len'] = int(max_seq_len)

    tensors = _load_shards(directory)
    params = translate_weights(tensors, config)['params']

    built = with_precision('causal_transformer', config,
                           dtype=dtype, attention_impl=attention_impl)
    model = models.build('causal_transformer', **built)
    _check_tree(params, model)
    return model, {'params': params}, built


def save_pretrained_decoder(model, variables, directory, *,
                            tokenizer_name: Optional[str] = None) -> None:
    """Write a decoder back out in the HF layout: config.json, model.safetensors.

    The inverse of load_pretrained_decoder: the same field map, run backwards,
    so a round-trip through dew hands transformers a checkpoint it accepts and
    a load hands back bitwise-equal parameters. model_type is gemma3_text when
    the sandwich norms are on, qwen3 when the q/k norms are, llama otherwise.
    All three references build q/k/v/o with bias=config.attention_bias, so the
    flag exports as it stands.
    """
    from dew.interop.safetensors_io import save_hf_layout

    if not isinstance(model, CausalTransformer):
        raise ValueError(
            f"save_pretrained_decoder takes a CausalTransformer, got {type(model).__name__}")
    if model.per_layer_input_dim or model.num_kv_shared_layers or model.v_norm:
        raise ValueError(
            "per-layer input embeddings, KV sharing and the values norm have "
            "no counterpart in the llama, qwen3 and gemma3_text families this "
            "exports, so a model with per_layer_input_dim, num_kv_shared_layers "
            "or v_norm set cannot be written back to the HF layout")
    if getattr(model, 'mixer', None) is not None or model.mixture is not None:
        raise ValueError(
            "a mixer other than standard attention and routed experts have "
            "no writer here yet: the config half round-trips through "
            "_export_config, but the MLA, indexer and per-expert tensors "
            "have no _hf_name, so a model with mixer or mixture set cannot "
            "be written back to the HF layout")
    params = variables.get('params', variables)
    config = _export_config(model)

    hf_tensors: Dict[str, np.ndarray] = {}
    for name, leaf in _flatten(params).items():
        hf_name = _hf_name(name, config)
        if hf_name is None:
            continue
        leaf = np.asarray(leaf)
        hf_tensors[hf_name] = np.ascontiguousarray(
            leaf.T if name.endswith('.kernel') else leaf)

    os.makedirs(directory, exist_ok=True)
    save_hf_layout(hf_tensors, config, directory)
    generation_config: Dict[str, Any] = {'do_sample': True, 'use_cache': True}
    if tokenizer_name is not None:
        generation_config['tokenizer_name'] = tokenizer_name
    with open(os.path.join(directory, GENERATION_CONFIG_FILE), 'w') as handle:
        json.dump(generation_config, handle, indent=2)


def _flatten(tree: Mapping[str, Any], prefix: str = '') -> Dict[str, Any]:
    """A params tree as '.'-joined names, leaves untouched.

    Untouched matters: the shape check flattens a jax.eval_shape template,
    whose leaves carry a shape but no data to convert.
    """
    flat: Dict[str, Any] = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _export_config(model) -> Dict[str, Any]:
    """A CausalTransformer's fields back into HF vocabulary."""
    sandwich = bool(model.sandwich_norms)
    if sandwich:
        model_type = 'gemma3_text'
    elif model.qk_norm:
        model_type = 'qwen3'
    else:
        model_type = 'llama'

    config: Dict[str, Any] = {
        'model_type': model_type,
        'architectures': ['Gemma3ForCausalLM' if sandwich else
                          ('Qwen3ForCausalLM' if model.qk_norm else
                           'LlamaForCausalLM')],
        'hidden_size': model.emb_features,
        'num_hidden_layers': model.num_layers,
        'num_attention_heads': model.num_heads,
        'num_key_value_heads': model.kv_heads,
        'head_dim': model.features_per_head,
        'intermediate_size': model.hidden_features,
        'vocab_size': model.vocab_size,
        'max_position_embeddings': model.max_seq_len,
        'rms_norm_eps': model.norm_eps,
        'attention_bias': model.attention_bias,
        'tie_word_embeddings': model.tie_embeddings,
        'hidden_act': 'silu' if model.mlp == 'swiglu' else 'gelu_pytorch_tanh',
        'use_cache': True,
    }
    types = model.per_layer_types
    if any(layer != 'full_attention' for layer in types):
        config['layer_types'] = list(types)
    sliding = model.kind_of('sliding_attention') if 'sliding_attention' in types else None
    local_theta = None if sliding is None or sliding.rope_theta == model.rope_theta else sliding.rope_theta
    if local_theta is not None:
        if sandwich:
            config['rope_parameters'] = {
                'full_attention': {'rope_type': 'default',
                                   'rope_theta': model.rope_theta},
                'sliding_attention': {'rope_type': 'default',
                                      'rope_theta': local_theta},
            }
        else:
            config['rope_theta'] = model.rope_theta
            config['rope_local_base_freq'] = local_theta
    else:
        config['rope_theta'] = model.rope_theta
    if sliding is not None and sliding.window is not None:
        config['sliding_window'] = sliding.window
    if model_type == 'gemma3_text':
        config.update(
            hidden_activation='gelu_pytorch_tanh',
            query_pre_attn_scalar=(None if model.attention_scale is None
                                   else round(1.0 / model.attention_scale ** 2)),
            final_logit_softcapping=model.final_logit_softcap,
            attn_logit_softcapping=None,
            use_bidirectional_attention=False,
        )
    if model_type == 'qwen3':
        window = None if sliding is None else sliding.window
        config['use_sliding_window'] = window is not None
        if window is not None:
            config['max_window_layers'] = 0
    return {key: value for key, value in config.items() if value is not None}


def _hf_name(dew_name: str, config: Mapping[str, Any]) -> Optional[str]:
    """One flattened dew param path into its HF tensor name, or None.

    None is the tied lm_head: the embedding it copies is written instead.
    """
    parts = dew_name.split('.')
    if parts == ['norm', 'scale']:
        return 'model.norm.weight'
    if parts == ['embed_tokens', 'embedding']:
        return 'model.embed_tokens.weight'
    if parts == ['lm_head', 'kernel']:
        return None if config['tie_word_embeddings'] else 'lm_head.weight'

    if parts[0].startswith('layers_'):
        index = parts[0].removeprefix('layers_')
        module, leaf = parts[1], parts[-1]
        if len(parts) == 4 and module in _PROJECTIONS:
            if parts[2] in _PROJECTIONS[module] and leaf in ('kernel', 'bias'):
                return (f'model.layers.{index}.{module}.{parts[2]}.'
                        + ('weight' if leaf == 'kernel' else 'bias'))
            if module == 'self_attn' and parts[2] in _HEAD_NORMS and leaf == 'scale':
                return f'model.layers.{index}.self_attn.{parts[2]}.weight'
        theirs = {ours: hf for hf, ours in
                  _norm_names(config['model_type'] == _GEMMA).items()}
        if len(parts) == 3 and module in theirs and leaf == 'scale':
            return f'model.layers.{index}.{theirs[module]}.weight'
    raise ValueError(f"unknown parameter path {dew_name!r}")


def _check_tree(params: Mapping[str, Any], model) -> None:
    """Refuse a tree the model would not accept, naming what is off.

    jax.eval_shape builds the template without allocating it, so checking a
    0.6B checkpoint costs shapes rather than a second copy of the weights.
    """
    import jax
    import jax.numpy as jnp

    template = jax.eval_shape(
        lambda: model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2), jnp.int32)))
    expected = {name: leaf.shape
                for name, leaf in _flatten(template["params"]).items()}
    loaded = _flatten(params)

    missing = sorted(set(expected) - set(loaded))
    unexpected = sorted(set(loaded) - set(expected))
    mismatched = sorted(
        f"{name} is {loaded[name].shape}, the model takes {shape}"
        for name, shape in expected.items()
        if name in loaded and loaded[name].shape != shape)
    if missing or unexpected or mismatched:
        raise ValueError(
            f"the checkpoint does not fit the model: missing {missing}, "
            f"unexpected {unexpected}, mismatched {mismatched}")
