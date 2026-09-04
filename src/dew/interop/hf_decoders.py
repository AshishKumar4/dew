"""Hugging Face decoder checkpoints into CausalTransformer trees, and back.

translate_config and translate_weights are the map: a decoder config dict into
CausalTransformer kwargs, and HF-named tensors into a dew params tree. The
wrappers around them fetch a repo (or read a local directory), read the
safetensors shards as fp32 without torch, and build the model, so
load_pretrained_decoder returns a (model, variables, config) triple that a
forward pass takes straight away, config being the dew config the model was
built from.

The families covered are the ones CausalTransformer can express: llama, qwen3
and gemma3_text. qwen2 is refused rather than half-loaded, since its q/k/v
biases without an o_proj bias have no counterpart in the backbone's one
attention_bias flag. A config field that changes what the model computes and
has no dew counterpart raises a ValueError naming it, rather than loading a
model that silently computes something else.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

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

_QK_NORM_FAMILIES = ('qwen3', 'gemma3_text', 'gemma4', 'gemma4_text')
_GEMMA = 'gemma3_text'

_IGNORED_FIELDS = {
    'architectures', 'attention_dropout', 'attn_implementation', 'bos_token_id',
    'cache_implementation', 'dtype', 'eos_token_id', 'id2label',
    'initializer_range', 'is_encoder_decoder', 'label2id', 'max_window_layers',
    'mlp_bias', 'output_attentions', 'output_hidden_states', 'pad_token_id',
    'pretraining_tp', 'problem_type', 'return_dict', 'use_cache',
    'use_sliding_window', 'torch_dtype', 'transformers_version',
}

# The fields above have no effect on an eval-time forward pass: metadata,
# token ids, or runtime knobs of the reference implementation (Gemma 3 ships
# cache_implementation 'hybrid', which describes transformers' KV cache).


def _refuse(field: str, detail: str) -> None:
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

    A config that carries layer_types is taken at its word. Without it, qwen3
    makes every layer from max_window_layers on sliding, and only when
    use_sliding_window says so; gemma3_text repeats its sliding_window_pattern,
    five sliding layers to one full; llama has no sliding layers at all.
    """
    layers = int(hf_config['num_hidden_layers'])
    model_type = hf_config.get('model_type')
    layer_types = hf_config.get('layer_types')
    if layer_types is not None:
        used.add('layer_types')
        return tuple(layer_types)

    if model_type == 'qwen3':
        used.update(('use_sliding_window', 'sliding_window'))
        if (not hf_config.get('use_sliding_window', False)
                or hf_config.get('sliding_window') is None):
            return ('full_attention',) * layers
        first_sliding = int(hf_config.get('max_window_layers', layers))
        return tuple('sliding_attention' if index >= first_sliding else 'full_attention'
                     for index in range(layers))

    if model_type == _GEMMA:
        pattern = int(hf_config.get('sliding_window_pattern', 6))
        used.add('sliding_window_pattern')
        return tuple('sliding_attention' if (index + 1) % pattern else 'full_attention'
                     for index in range(layers))

    return ('full_attention',) * layers

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


def translate_config(hf_config: Mapping[str, Any]) -> Dict[str, Any]:
    """A decoder config dict into CausalTransformer kwargs.

    Accepts the text decoder families CausalTransformer can express: llama,
    qwen3, gemma3_text and gemma4_text. Every field that changes what a
    forward pass computes and has no dew counterpart raises, naming the
    field."""

    model_type = hf_config.get('model_type')
    if model_type == 'qwen2':
        # Qwen2 biases q/k/v and leaves o_proj bias-free (modeling_qwen2.py
        # 189-192), and CausalSelfAttention has one attention_bias flag for
        # all four projections, so its checkpoints cannot load unchanged.
        _refuse("model_type 'qwen2'",
                "its q/k/v projections carry biases and o_proj does not, which "
                "the one attention_bias flag cannot say")
    if model_type not in ('llama', 'qwen3', 'gemma3_text', 'gemma4', 'gemma4_text'):
        _refuse(f"model_type {model_type!r}",
                "expected one of 'llama', 'qwen3', 'gemma3_text', 'gemma4', 'gemma4_text'")

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

    if model_type in ('gemma4', 'gemma4_text'):
        # Proportional partial rotary is a gemma4 shape with its own reader below.
        rope_theta, rope_local_theta = 10000.0, None
    else:
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
            'tie_word_embeddings', model_type in (_GEMMA, 'gemma4', 'gemma4_text'))),
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
    if model_type in ('gemma4', 'gemma4_text'):
        if hf_config.get('attention_k_eq_v'):
            _refuse("attention_k_eq_v=True",
                    "the backbone always projects its own values")
        if hf_config.get('enable_moe_block'):
            _refuse("enable_moe_block=True",
                    "the parallel MoE branch has no counterpart here")
        used.update(('attention_k_eq_v', 'enable_moe_block'))
        # Full layers may override the head dim and the sliding layers use
        # hidden // heads; the two live side by side as head_dim and
        # global_head_dim.
        sliding_dim = hidden // heads
        if hidden % heads:
            _refuse(f"hidden_size {hidden}",
                    f"it does not divide into {heads} heads")
        full_dim = sliding_dim
        entries = hf_config.get('per_layer_config') or {}
        for entry in (entries.values() if isinstance(entries, Mapping) else entries):
            if isinstance(entry, Mapping) and entry.get('head_dim') is not None:
                full_dim = int(entry['head_dim'])
        if hf_config.get('global_head_dim') is not None:
            full_dim = int(hf_config['global_head_dim'])
        used.update(('per_layer_config', 'global_head_dim',
                     'num_global_key_value_heads'))
        # Proportional rope rotates a fraction of the full layers' head dims
        # and passes the rest through; sliding layers rotate all of theirs.
        entries = hf_config.get('rope_parameters') or {}
        rope_theta, rope_local_theta, partial = _gemma4_rope(entries)
        used.update(('rope_parameters', 'rope_theta'))
        config.update(
            sandwich_norms=True, embedding_scale=True, attention_scale=1.0,
            v_norm=True,
            head_dim=sliding_dim,
            global_head_dim=(None if full_dim == sliding_dim else full_dim),
            rope_theta=rope_theta, rope_local_theta=rope_local_theta,
            partial_rotary_factor=partial,
            use_double_wide_mlp=bool(hf_config.get('use_double_wide_mlp', False)),
            num_kv_shared_layers=int(hf_config.get('num_kv_shared_layers', 0)),
            per_layer_input_dim=int(hf_config.get('hidden_size_per_layer_input', 0)),
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
                'mlp': ('gate_proj', 'up_proj', 'down_proj')}
_HEAD_NORMS = ('q_norm', 'k_norm')


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
    generation_config = {'do_sample': True, 'use_cache': True}
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
