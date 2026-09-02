"""Hugging Face decoder checkpoints into CausalTransformer trees, and back.

translate_config and translate_weights are the map: a decoder config dict into
CausalTransformer kwargs, and HF-named tensors into a dew params tree. The
wrappers around them fetch a repo (or read a local directory), read the
safetensors shards as fp32 without torch, and build the model, so
load_pretrained_decoder returns a (model, variables, hf_config) triple that a
forward pass takes straight away.

The families covered are the ones CausalTransformer can express: llama, qwen2,
qwen3 and gemma3_text, including the text_config of a gemma3 multimodal
checkpoint. A config field that changes what the model computes and has no dew
counterpart raises a ValueError naming it, rather than loading a model that
silently computes something else.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from dew.registry import apply_precision_policy, build_model

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
GENERATION_CONFIG_FILE = "generation_config.json"

# The KV cache is allocated at the full decode length, so a 128k-context
# checkpoint would allocate one of those whether the caller asked or not.
DEFAULT_MAX_SEQ_LEN = 8192

# hidden_act / hidden_activation values, onto the GatedMLP activations.
_ACTIVATIONS = {
    'silu': 'swiglu',
    'swish': 'swiglu',
    'gelu_pytorch_tanh': 'geglu',
    'gelu_new': 'geglu',
}

_QK_NORM_FAMILIES = ('qwen3', 'gemma3_text')
_GEMMA = 'gemma3_text'

_IGNORED_FIELDS = {
    'architectures', 'attention_dropout', 'attn_implementation', 'bos_token_id',
    'dtype', 'eos_token_id', 'id2label', 'initializer_range', 'is_encoder_decoder',
    'label2id', 'max_window_layers', 'mlp_bias', 'output_attentions',
    'output_hidden_states', 'pad_token_id', 'pretraining_tp', 'problem_type',
    'return_dict', 'use_cache', 'use_sliding_window',
    'torch_dtype', 'transformers_version',
}

# A gemma3 multimodal config describes a vision tower and a splicer too; only
# the text decoder maps, and these are the fields that name the rest.
_IGNORED_MULTIMODAL_FIELDS = {
    'architectures', 'boi_token_index', 'eoi_token_index', 'eos_token_id',
    'image_token_index', 'mm_tokens_per_image', 'model_type', 'torch_dtype',
    'transformers_version', 'vision_config',
}


def _refuse(field: str, detail: str) -> None:
    raise ValueError(f"{field} is not expressible: {detail}")


def _rope_theta(entry: Optional[Mapping[str, Any]], field: str) -> Optional[float]:
    """One rope base frequency out of a rope_parameters entry, if it names one.

    Only plain rope ('default' or 'none') maps; a scaled variant changes what
    the model computes, so the caller refuses it with the field named.
    """
    if entry is None:
        return None
    rope_type = entry.get('rope_type', 'default')
    if rope_type not in ('default', 'none'):
        _refuse(f"{field} (rope_type {rope_type!r})",
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

    scaling = hf_config.get('rope_scaling')
    if scaling is not None:
        theta_from_scaling = _rope_theta(scaling, 'rope_scaling')
        if theta_from_scaling is None:
            _refuse(f"rope_scaling {dict(scaling)!r}",
                    "it carries no rope_theta to read")
    theta = (_rope_theta(rope_parameters, 'rope_parameters')
             if rope_parameters is not None else None)
    theta = theta or theta_from_scaling or float(hf_config.get('rope_theta', 10000.0))
    local = hf_config.get('rope_local_base_freq')
    return theta, (None if local is None else float(local))


def _layer_types(hf_config: Mapping[str, Any], used: set) -> Tuple[str, ...]:
    """Per-layer attention windows, derived the way the family's config does.

    qwen2/qwen3 configs in the wild leave layer_types unset and derive it from
    use_sliding_window, sliding_window and max_window_layers. gemma3_text
    derives its sliding_window_pattern repetition when layer_types is missing.
    A config that carries layer_types is taken at its word, for both.
    """
    layers = int(hf_config['num_hidden_layers'])
    layer_types = hf_config.get('layer_types')
    if layer_types is not None:
        used.add('layer_types')
        return tuple(layer_types)

    if hf_config.get('model_type') in ('qwen2', 'qwen3'):
        used.update(('use_sliding_window', 'sliding_window'))
        if not hf_config.get('use_sliding_window', False):
            return ('full_attention',) * layers
        if hf_config.get('sliding_window') is None:
            return ('full_attention',) * layers
        first_sliding = int(hf_config.get('max_window_layers', layers))
        return tuple('sliding_attention' if index >= first_sliding else 'full_attention'
                     for index in range(layers))

    pattern = int(hf_config.get('sliding_window_pattern', 6))
    used.add('sliding_window_pattern')
    return tuple('sliding_attention' if (index + 1) % pattern else 'full_attention'
                 for index in range(layers))


def translate_config(hf_config: Mapping[str, Any]) -> Dict[str, Any]:
    """A decoder config dict into CausalTransformer kwargs.

    Accepts the text decoder families CausalTransformer can express: llama,
    qwen2, qwen3 and gemma3_text, plus a gemma3 multimodal config by reading
    its text_config. Every field that changes what a forward pass computes and
    has no dew counterpart raises, naming the field.
    """
    hf_config = dict(hf_config)

    if hf_config.get('model_type') == 'gemma3' and 'text_config' in hf_config:
        unknown = set(hf_config) - _IGNORED_MULTIMODAL_FIELDS
        if unknown:
            _refuse(f"gemma3 config fields {sorted(unknown)}",
                    "only the text_config of a multimodal Gemma maps onto a "
                    "text decoder")
        hf_config = dict(hf_config['text_config'])
        if hf_config.get('model_type') not in (None, 'gemma3_text'):
            _refuse(f"text_config model_type {hf_config.get('model_type')!r}",
                    "expected 'gemma3_text'")

    model_type = hf_config.get('model_type')
    if model_type not in ('llama', 'qwen2', 'qwen3', 'gemma3_text'):
        _refuse(f"model_type {model_type!r}",
                "expected one of 'llama', 'qwen2', 'qwen3', 'gemma3_text'")

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
    head_dim = int(hf_config.get('head_dim', hidden // heads))
    used.update(('hidden_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim'))

    activation = hf_config.get('hidden_act', hf_config.get('hidden_activation', 'silu'))
    used.update(('hidden_act', 'hidden_activation'))
    mapped = _ACTIVATIONS.get(activation)
    if mapped is None:
        _refuse(f"hidden_act {activation!r}",
                "the gated MLP supports 'swiglu' and 'geglu'")

    rope_theta, rope_local_theta = _rope(hf_config, used)
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
        'rope_local_theta': rope_local_theta,
        'layer_types': layer_types,
        'sliding_window': sliding_window,
        'norm_eps': float(hf_config.get('rms_norm_eps', 1e-6)),
        'qk_norm': model_type in _QK_NORM_FAMILIES,
        'attention_bias': bool(hf_config.get('attention_bias', False)),
        'tie_embeddings': bool(hf_config.get('tie_word_embeddings', False)),
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

    unknown = (set(hf_config) - used - _IGNORED_FIELDS
               - {key for key in hf_config if str(key).startswith('_')})
    if unknown:
        _refuse(f"config fields {sorted(unknown)}",
                "CausalTransformer has no counterpart, so translating them "
                "would silently change the model")
    return config


# Where a layer's sublayers and norms sit in the two trees. Gemma's two
# output norms are the renames: HF's post_attention_layernorm normalizes the
# attention output where every other family puts its MLP-input pre-norm, and
# its pre_feedforward_layernorm normalizes the MLP output where ours puts the
# residual the MLP adds to.
_LAYER_MODULES = {
    'input_layernorm': 'input_layernorm',
    'pre_feedforward_layernorm': 'post_attention_layernorm',
    'self_attn': 'self_attn',
    'mlp': 'mlp',
    'post_attention_layernorm': 'attention_output_norm',
    'post_feedforward_layernorm': 'mlp_output_norm',
}
_SUBLAYERS = {'self_attn': ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                            'q_norm', 'k_norm'),
              'mlp': ('gate_proj', 'up_proj', 'down_proj')}


def _dew_path(hf_name: str, config: Mapping[str, Any]) -> Optional[Tuple[str, ...]]:
    """One HF tensor name into its path in a CausalTransformer tree.

    None means the tensor has no place in the tree: the tied lm_head a
    checkpoint carries as a copy of the embedding. A name the map cannot
    explain at all raises, so an unfamiliar checkpoint fails here instead of
    producing a half-loaded model.
    """
    parts = hf_name.split('.')
    if parts == ['model', 'norm', 'weight']:
        return ('norm', 'scale')
    if parts == ['model', 'embed_tokens', 'weight']:
        return ('embed_tokens', 'embedding')
    if parts == ['lm_head', 'weight']:
        return None if config['tie_embeddings'] else ('lm_head', 'kernel')

    if len(parts) == 4 and parts[0] == 'model' and parts[1] == 'layers':
        layer, module, leaf = parts[2:]
        if (not layer.isdigit() or module not in _LAYER_MODULES
                or leaf != 'weight' or _LAYER_MODULES[module] in _SUBLAYERS):
            raise ValueError(f"unknown tensor name {hf_name!r}")
        return (f'layers_{layer}', _LAYER_MODULES[module], 'scale')

    if len(parts) == 5 and parts[0] == 'model' and parts[1] == 'layers':
        layer, module, sublayer, leaf = parts[2:]
        if not layer.isdigit() or module not in _LAYER_MODULES:
            raise ValueError(f"unknown tensor name {hf_name!r}")
        ours = _LAYER_MODULES[module]
        if module in _SUBLAYERS:
            if sublayer not in _SUBLAYERS[module] or leaf not in ('weight', 'bias'):
                raise ValueError(f"unknown tensor name {hf_name!r}")
            # torch Linear stores [out, in]; nn.Dense keeps [in, out] in .kernel
            return (f'layers_{layer}', ours, sublayer,
                    'kernel' if leaf == 'weight' else 'bias')
        if sublayer or leaf != 'weight':
            raise ValueError(f"unknown tensor name {hf_name!r}")
        return (f'layers_{layer}', ours, 'scale')
    raise ValueError(f"unknown tensor name {hf_name!r}")


def translate_weights(hf_tensors: Mapping[str, np.ndarray],
                      config: Mapping[str, Any]) -> Dict[str, Any]:
    """HF-named tensors into a CausalTransformer params tree, in fp32.

    Linear weights arrive as [out, in] and nn.Dense keeps [in, out], so every
    `.kernel` is transposed; norm `.weight` becomes `.scale`; Gemma's
    post_attention_layernorm and post_feedforward_layernorm land on the
    sandwich norms, which is where Gemma applies them. A tied lm_head is
    skipped, since the embedding it copies is the tree's leaf already.
    """
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
    return {'params': params}


def _read_header(path: Path) -> Dict[str, Any]:
    with open(path, 'rb') as handle:
        length = int.from_bytes(handle.read(8), 'little')
        return json.loads(handle.read(length))


_DTYPES = {'F32': np.float32, 'F16': np.float16, 'I64': np.int64,
           'I32': np.int32, 'U8': np.uint8, 'BOOL': np.bool_}


def _read_tensor(path: Path, header: Mapping[str, Any],
                 name: str) -> np.ndarray:
    """One tensor out of a safetensors file as fp32, without torch.

    safetensors.numpy cannot read bfloat16 and most decoder checkpoints are
    bfloat16, so those leaves are widened here the way every bf16 reader does:
    the 16 payload bits shifted into the top half of an fp32 word.
    """
    meta = header[name]
    dtype, shape = meta['dtype'], tuple(meta['shape'])
    start, end = meta['data_offsets']
    header_length = int.from_bytes((path.read_bytes()[:8]), 'little')
    with open(path, 'rb') as handle:
        handle.seek(8 + header_length + start)
        raw = handle.read(end - start)
    if dtype == 'BF16':
        widened = np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16
        return widened.view(np.float32).reshape(shape)
    if dtype not in _DTYPES:
        raise ValueError(
            f"tensor {name} has dtype {dtype}, which this loader cannot read")
    return np.frombuffer(raw, dtype=_DTYPES[dtype]).reshape(shape)


def _load_shards(directory: Path) -> Dict[str, np.ndarray]:
    """Every tensor of a checkpoint directory, shard by shard, as fp32."""
    shards = sorted(directory.glob('*.safetensors'))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors under {directory}")
    tensors: Dict[str, np.ndarray] = {}
    for shard in shards:
        header = _read_header(shard)
        if shards[0].name == WEIGHTS_FILE and len(shards) > 1:
            # a single-file export and sharded remnants do not mix
            pass
        tensors.update({name: _read_tensor(shard, header, name)
                        for name in header if name != '__metadata__'})
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
    """A Hugging Face decoder checkpoint, as (model, variables, hf_config).

    `name_or_dir` is a hub repo id or a local directory in the HF layout.
    The config is translated, the weights mapped onto that tree in fp32, and
    the model built with the run's precision policy, so `variables` fits the
    model and the policy's `dtype` reaches every module the same way it does
    in a training run. max_seq_len defaults to the config's context clamped
    to 8192, because the KV cache is allocated at that length.
    """
    directory = _snapshot(name_or_dir, revision)
    with open(directory / CONFIG_FILE) as handle:
        hf_config = json.load(handle)

    config = translate_config(hf_config)
    if max_seq_len is not None:
        config['max_seq_len'] = int(max_seq_len)

    tensors = _load_shards(directory)
    params = translate_weights(tensors, config)['params']

    architecture = 'causal_transformer'
    built = apply_precision_policy(architecture, dict(config),
                                   dtype=dtype, attention_impl=attention_impl)
    model = build_model(architecture, built)
    missing, unexpected = _tree_gaps(params, model)
    if missing or unexpected:
        raise ValueError(
            f"the checkpoint and {architecture} disagree: missing {sorted(missing)}, "
            f"unexpected {sorted(unexpected)}")
    return model, {'params': params}, hf_config


def save_pretrained_decoder(model, variables, directory, *,
                            tokenizer_name: Optional[str] = None) -> None:
    """Write a decoder back out in the HF layout: config.json, model.safetensors.

    The inverse of load_pretrained_decoder: the same field map, run backwards,
    so a round-trip through dew hands transformers a checkpoint it accepts and
    a load hands back bitwise-equal parameters. model_type is gemma3_text when
    the sandwich norms are on, qwen3 when the q/k norms are, llama otherwise;
    attention-biased qwen2 checkpoints are not written, since the exported
    family would not carry them.
    """
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.interop.safetensors_io import save_hf_layout

    if not isinstance(model, CausalTransformer):
        raise ValueError(
            f"save_pretrained_decoder takes a CausalTransformer, got {type(model).__name__}")
    params = variables.get('params', variables)
    config = _export_config(model)

    hf_tensors: Dict[str, np.ndarray] = {}
    for name, leaf in _flat_params(params).items():
        hf_name = _hf_name(name, config)
        if hf_name is None:
            continue
        if name.endswith('.kernel') or name in ('lm_head.kernel',):
            leaf = leaf.T
        hf_tensors[hf_name] = np.ascontiguousarray(leaf)

    os.makedirs(directory, exist_ok=True)
    save_hf_layout(hf_tensors, config, directory)
    generation_config = {'do_sample': True, 'use_cache': True}
    if tokenizer_name is not None:
        generation_config['tokenizer_name'] = tokenizer_name
    with open(os.path.join(directory, GENERATION_CONFIG_FILE), 'w') as handle:
        json.dump(generation_config, handle, indent=2)


def _flat_params(params: Mapping[str, Any], prefix: str = '') -> Dict[str, np.ndarray]:
    flat: Dict[str, np.ndarray] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flat_params(value, f"{name}."))
        else:
            flat[name] = np.asarray(value)
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
    if model.attention_bias and model_type != 'qwen2':
        # An attention-biased export cannot be qwen3 or llama, which have no
        # such checkpoints in the wild; refuse rather than mislabel.
        raise ValueError(
            "attention_bias=True exports as model_type qwen2, whose layout "
            "this writer does not target")
    if model.rope_local_theta is not None:
        if sandwich:
            config['rope_parameters'] = {
                'full_attention': {'rope_type': 'default',
                                   'rope_theta': model.rope_theta},
                'sliding_attention': {'rope_type': 'default',
                                      'rope_theta': model.rope_local_theta},
            }
        else:
            config['rope_theta'] = model.rope_theta
            config['rope_local_base_freq'] = model.rope_local_theta
    else:
        config['rope_theta'] = model.rope_theta
    if model.sliding_window is not None:
        config['sliding_window'] = model.sliding_window
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
        config['use_sliding_window'] = model.sliding_window is not None
        if model.sliding_window is not None:
            config['max_window_layers'] = 0
    return {key: value for key, value in config.items() if value is not None}


def _hf_name(dew_name: str, config: Mapping[str, Any]) -> Optional[str]:
    """One flattened dew param path into its HF tensor name, or None."""
    parts = dew_name.split('.')
    if parts == ['norm', 'scale']:
        return 'model.norm.weight'
    if parts == ['embed_tokens', 'embedding']:
        return 'model.embed_tokens.weight'
    if parts == ['lm_head', 'kernel']:
        return None if config['tie_word_embeddings'] else 'lm_head.weight'

    if len(parts) == 4 and parts[0].startswith('layers_'):
        layer, module, sublayer, leaf = parts
        index = layer.removeprefix('layers_')
        ours_to_hf = {ours: hf for hf, ours in _LAYER_MODULES.items()}
        hf_module = ours_to_hf[module]
        if module in ('self_attn', 'mlp'):
            return f'model.layers.{index}.{hf_module}.{sublayer}.' + (
                'weight' if leaf == 'kernel' else 'bias')
        if leaf != 'scale':
            raise ValueError(f"unknown parameter path {dew_name!r}")
        return f'model.layers.{index}.{hf_module}.weight'
    raise ValueError(f"unknown parameter path {dew_name!r}")


def _tree_gaps(params: Mapping[str, Any], model) -> Tuple[List[str], List[str]]:
    """Leaves the checkpoint did not fill, and tree paths with no leaf."""
    import jax
    import jax.numpy as jnp

    template = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2), jnp.int32))
    expected = _flat_params(template['params'])
    loaded = _flat_params(params)
    missing = sorted(set(expected) - set(loaded))
    unexpected = sorted(set(loaded) - set(expected))
    if not missing and not unexpected:
        mismatched = [name for name, shape in expected.items()
                      if tuple(loaded[name].shape) != tuple(shape.shape)]
        return [], mismatched
    return missing, unexpected
