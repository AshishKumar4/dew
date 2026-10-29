"""Translate the Gemma decoders: gemma, gemma2, gemma3_text, gemma3n_text, gemma4_text.

Each release keeps the previous block and adds one thing: Gemma 2 the
sandwich norms and the two softcaps, Gemma 3 the q/k norms and the
alternating local window, Gemma 3n AltUp's residual copies with the LAuReL
projections and the per-layer inputs, Gemma 4 the value norm, the shared KV
layers and the routed branch. The export side is here too, because a Gemma
config is written back from the computation rather than from a template.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from flax.traverse_util import flatten_dict

from dew import _records
from dew._model_types import LAYER_SCALAR_COLLECTIONS
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.decoder_parts import (
    FUSED_EXPERTS,
    GEMMA3_MODEL_TYPE,
    MOE_SHARED,
    PLAIN_SPELLING,
    DecoderFamily,
    DecoderFields,
    MixtureFields,
    Ropes,
    RopeSpelling,
    base_config,
    decoder_kinds,
    decoder_tensors,
    dew_path,
    fixed_fields,
    hf_activation,
    hf_tensor_name,
    kinds_of,
    read_rope,
    read_rope_theta,
    refuse,
    specified_layer_types,
)
from dew.interop.safetensors_io import LazyTensors
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.gemma3n import AltUp
from dew.nn.mixers import AttentionMixer


def _trailing_sharers(hf_config: Mapping[str, object], layers: int, default: int) -> tuple[int, ...] | None:
    """`num_kv_shared_layers` as the indices of the trailing layers that share
    their providers' keys and values (`first_kv_shared_layer_idx`,
    modeling_gemma3n.py:1178, modeling_gemma4.py:1068), None for none."""
    count = _records.integer(hf_config.get("num_kv_shared_layers", default), "num_kv_shared_layers")
    if not 0 <= count < layers:
        refuse(f"num_kv_shared_layers {count}", f"it has to leave a provider among {layers} layers")
    return tuple(range(layers - count, layers)) or None


_SHARED_UNUSED = ('k_proj', 'v_proj', 'k_norm')
"""A kv-shared layer's own key and value projections and key norm, which transformers builds and a
release may store (google/gemma-4-E4B-it @ee0ef60 does, layers 24-41) but never computes with: the
layer reads its provider's keys and values (modeling_gemma4.py, Gemma4TextAttention.forward)."""


def _shared_unused(name: str, config: Mapping[str, object]) -> bool:
    """Whether `name` is one of a kv-shared layer's unused tensors (`_SHARED_UNUSED`), which a load
    leaves unplaced and a save writes back as stored."""
    parts = name.split('.')
    sharing = config.get('kv_shared_layers') or ()
    return (len(parts) == 6 and parts[:2] == ['model', 'layers'] and parts[2].isdigit()
            and parts[3] == 'self_attn' and parts[4] in _SHARED_UNUSED and parts[5] == 'weight'
            and isinstance(sharing, (tuple, list)) and int(parts[2]) in sharing)


def _gemma_layer_types(hf_config: Mapping[str, object], used: set[str], *,
                       last_full: bool = False) -> tuple[str, ...]:
    period = 6
    if not last_full and hf_config.get('layer_types') is None:
        used.add('sliding_window_pattern')
        period = _records.integer(hf_config.get("sliding_window_pattern", 6), "sliding_window_pattern")
    types = specified_layer_types(hf_config, used,
                                  ('sliding_attention',) * (period - 1) + ('full_attention',))
    # Gemma4TextConfig rewrites the final layer before building the model.
    return (*types[:-1], 'full_attention') if last_full and types else types


GEMMA4_ROPE = {'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.0},
               'full_attention': {'rope_type': 'proportional', 'partial_rotary_factor': 0.25,
                                  'rope_theta': 1000000.0}}
"""The records Gemma4TextConfig and DiffusionGemmaTextConfig install when
rope_parameters is absent or null (configuration_gemma4.py:203-208)."""

GEMMA4_DEFAULTS = {'hidden_activation': 'gelu_pytorch_tanh', 'head_dim': 256, 'num_key_value_heads': 4,
                   'sliding_window': 512, 'pad_token_id': 0}
"""The class defaults Gemma 4's text configs share where Dew's readers would
otherwise fall back to another value (configuration_gemma4.py:152-183)."""


def _gemma4_rope(entries: Mapping[str, object]) -> tuple[float, float | None, float | None]:
    """Return (rope_theta, rope_local_theta, partial_rotary_factor) for gemma4.

    The full layers may rotate a fraction of their head dims (proportional
    partial rotary); the sliding layers rotate all of theirs. Anything but
    those two shapes refuses with the entry named.
    """
    full = _records.record(entries.get('full_attention') or {}, 'rope_parameters.full_attention')
    sliding = _records.record(entries.get('sliding_attention') or {}, 'rope_parameters.sliding_attention')
    factor = sliding.get('partial_rotary_factor')
    if factor not in (None, 1, 1.0):
        refuse(f"rope_parameters.sliding_attention partial_rotary_factor {factor}",
                "partial rotary applies to the full layers only")
    read_rope_theta({**sliding, 'rope_type': sliding.get('rope_type', sliding.get('type', 'default'))},
                "rope_parameters.sliding_attention")
    local = sliding.get('rope_theta', 10000.0)
    rope_type = full.get('rope_type', full.get('type', 'default'))
    factor = full.get('partial_rotary_factor')
    if rope_type == 'proportional':
        if factor is None:
            refuse("rope_parameters.full_attention",
                    "proportional rope needs its partial_rotary_factor")
        extra = sorted(set(full) - {'rope_type', 'type', 'rope_theta',
                                    'partial_rotary_factor', 'factor'})
        if extra or full.get('factor', 1.0) not in (1, 1.0):
            refuse("rope_parameters.full_attention scaling",
                    "the backbone applies plain rotary positions at rope_theta")
        partial = _records.number(factor, 'rope_parameters.full_attention partial_rotary_factor')
        theta = _records.number(
            full.get("rope_theta", 1000000.0), "rope_parameters.full_attention rope_theta"
        )
    elif rope_type in ('default', 'none'):
        if factor not in (None, 1, 1.0):
            refuse("rope_parameters.full_attention partial_rotary_factor",
                    "partial rotary comes spelled proportional")
        partial = None
        theta = read_rope_theta(full, 'rope_parameters.full_attention') or 10000.0
    else:
        refuse(f"rope_parameters.full_attention (rope_type {rope_type!r})",
                "the backbone applies plain rotary positions at rope_theta")
    local = _records.number(local, 'rope_parameters.sliding_attention rope_theta')

    return theta, (None if local == theta else local), partial


def _gemma_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Gemma 1 config into `CausalTransformer` fields.

    Gemma 1 has (1 + w) norms scaled in fp32, sqrt(d)-scaled embeddings, a
    tied head, and no norms beyond the two pre-norms (modeling_gemma.py:77,
    :374). Its released config names hidden_act 'gelu', which the reference
    computes as the erf gelu (modeling_gemma.py:93, ACT2FN['gelu']).
    """
    config = base_config(hf_config, used, rope=PLAIN_SPELLING, scale_after_cast=False, tie_embeddings=True,
                          reads=frozenset({'attention_bias'}))
    config.update(scale_offset=True, embedding_scale=True)
    return config


def _gemma_softcaps(hf_config: Mapping[str, object], used: set[str],
                    config: DecoderFields) -> None:
    """Add the query scalar, the softcaps and the sandwich norms of Gemma 2 and 3.

    Gemma 3 reads attn_logit_softcapping into its attention without passing it
    on (modeling_gemma3.py:334, :370-379), so there it changes nothing. Gemma 2
    applies it (modeling_gemma2.py:282) and its entry maps it below.
    """
    config.update(scale_offset=True, embedding_scale=True, sandwich_norms=True)
    used.update(('query_pre_attn_scalar', 'final_logit_softcapping',
                 'attn_logit_softcapping'))
    scalar = hf_config.get('query_pre_attn_scalar')
    if scalar is not None:
        config['attention_scale'] = _records.number(scalar, 'query_pre_attn_scalar') ** -0.5
    softcap = hf_config.get('final_logit_softcapping')
    if softcap is not None:
        config['final_logit_softcap'] = _records.number(softcap, 'final_logit_softcapping')


def _gemma2_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Gemma 2 config into `CausalTransformer` fields.

    Gemma 2 is Gemma 3's block without the q/k norms. It alternates sliding and
    full layers at one rope base and softcaps the attention logits with tanh
    (configuration_gemma2.py:95-98, modeling_gemma2.py:203-206).
    """
    layer_types = specified_layer_types(hf_config, used, ('sliding_attention', 'full_attention'))
    config = base_config(hf_config, used, rope=PLAIN_SPELLING, scale_after_cast=False,
                          tie_embeddings=True, layer_types=layer_types,
                          reads=frozenset({'layer_types', 'sliding_window', 'attention_bias'}))
    sliding = kinds_of(config).get('sliding_attention', native_fields(LayerKind)())
    if 'rope_theta' in sliding or 'rope_scaling' in sliding:
        refuse("rope_parameters.sliding_attention",
                "Gemma2RotaryEmbedding rotates every layer at one base and ramp")
    _gemma_softcaps(hf_config, used, config)
    softcap = hf_config.get('attn_logit_softcapping')
    if softcap is not None:
        config['attn_logit_softcap'] = _records.number(softcap, 'attn_logit_softcapping')
    return config


_GEMMA3_ROPE = RopeSpelling(theta=1000000.0, per_kind=True, local_theta=10000.0,
                            local_field='rope_local_base_freq')
"""Gemma3TextConfig's and Gemma3nTextConfig's rope: a record per layer type,
the full layers at 1e6 and the sliding ones at rope_local_base_freq or 1e4
(configuration_gemma3.py:102, 136-163; configuration_gemma3n.py:120, 179-206)."""


def _gemma3_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    rope = read_rope(hf_config, used, _GEMMA3_ROPE)
    config = base_config(hf_config, used, qk_norm=True, scale_after_cast=False, rope=rope,
                          tie_embeddings=True, layer_types=_gemma_layer_types(hf_config, used))
    _gemma_softcaps(hf_config, used, config)
    return config


def _gemma3n_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Gemma 3n config into `CausalTransformer` fields.

    Gemma 3n (E2B, E4B) adds AltUp's stack of residual copies, the LAuReL
    block, gaussian top-k sparsity on the first layers, one feed-forward width
    per layer, per-layer inputs and KV sharing over the last layers.
    """
    # Gemma3nTextConfig fills every fifth layer full.
    layer_types = specified_layer_types(hf_config, used, ('sliding_attention',) * 4 + ('full_attention',))
    ropes = read_rope(hf_config, used, _GEMMA3_ROPE)
    config = base_config(hf_config, used, qk_norm=True, scale_after_cast=False,
                          tie_embeddings=True, layer_types=layer_types, rope=ropes)
    layers = _records.integer(config.get('num_layers'), 'num_layers')
    sparsity = hf_config.get('activation_sparsity_pattern')
    if sparsity is None:
        # The reference default is the first ten layers at 0.95 when there
        # are more than ten, else none (configuration_gemma3n.py).
        sparse = 10 if layers > 10 else 0
        sparsity = [0.95] * sparse + [0.0] * (layers - sparse)
    if not isinstance(sparsity, (list, tuple)) or len(sparsity) != layers:
        refuse(f"activation_sparsity_pattern {sparsity!r}",
                f"the reference takes one fraction per layer of {layers}")
    used.update(('activation_sparsity_pattern', 'laurel_rank', 'altup_num_inputs',
                 'altup_active_idx', 'altup_coef_clip', 'altup_correct_scale',
                 'hidden_size_per_layer_input', 'vocab_size_per_layer_input',
                 'num_kv_shared_layers', 'final_logit_softcapping'))
    # PreTrainedConfig serializes this field; Gemma3nTextMLP never reads it.
    used.add('chunk_size_feed_forward')
    clip = hf_config.get('altup_coef_clip', 120.0)
    # AltUp's own checks name the reference's fields, so a config out of
    # their range is refused here; that is why the value is built before its
    # record, which states all four of AltUp's fields.
    altup = AltUp(num_inputs=_records.integer(hf_config.get('altup_num_inputs', 4), 'altup_num_inputs'),
                  active_idx=_records.integer(hf_config.get('altup_active_idx', 0), 'altup_active_idx'),
                  coef_clip=None if clip is None else _records.number(clip, 'altup_coef_clip'),
                  correct_scale=bool(hf_config.get('altup_correct_scale', True)))
    config.update(
        sandwich_norms=True,
        embedding_scale=True,
        attention_scale=1.0,
        v_norm=True,
        activation_sparsity_pattern=tuple(float(fraction) for fraction in sparsity),
        laurel_rank=_records.integer(hf_config.get("laurel_rank", 64), "laurel_rank"),
        altup=NativeFields(AltUp, dataclasses.asdict(altup)),
        per_layer_input_dim=_records.integer(
            hf_config.get("hidden_size_per_layer_input", 256), "hidden_size_per_layer_input"
        ),
        per_layer_input_vocab=_records.integer(
            hf_config.get("vocab_size_per_layer_input", 262144), "vocab_size_per_layer_input"
        ),
        kv_shared_layers=_trailing_sharers(hf_config, layers, 15),
        final_logit_softcap=_records.number(
            hf_config.get("final_logit_softcapping", 30.0), "final_logit_softcapping"
        ),
    )
    return config


def _gemma3n_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the variables-tree path for one Gemma 3n tensor name.

    Gemma 3n adds the AltUp and LAuReL leaves beside a Gemma 4 style layer.
    The copies' projections are indexed modules (altup_projections.{i}), which
    land as altup_projections_{i} the way the layers do; the coefficient maps
    are Linears, so they transpose like any kernel. A kv-shared layer's unused
    keys, values and key norm land nowhere (`_shared_unused`).
    """
    if _shared_unused(name, config):
        return None
    parts = name.split('.')
    if len(parts) == 4 and parts[0] == 'model' and parts[2].isdigit() and parts[3] == 'weight' \
            and parts[1] in ('altup_projections', 'altup_unembed_projections'):
        return ('params', f'{parts[1]}_{parts[2]}', 'kernel')
    if len(parts) >= 5 and parts[:2] == ['model', 'layers'] and parts[2].isdigit() \
            and parts[3] in ('altup', 'laurel'):
        layer = ('params', f'layers_{parts[2]}', parts[3])
        tail = parts[4:]
        if tail == ['correct_output_scale']:
            return (*layer, 'correct_output_scale')
        if len(tail) == 2 and tail[1] == 'weight':
            if tail[0] in ('router_norm', 'post_laurel_norm'):
                return (*layer, tail[0], 'scale')
            if tail[0] in ('correction_coefs', 'prediction_coefs', 'modality_router',
                           'linear_left', 'linear_right'):
                return (*layer, tail[0], 'kernel')
        raise ValueError(f"unknown tensor name {name!r}")
    return dew_path(name, config)


def gemma4_config(hf_config: Mapping[str, object], used: set[str], *,
                   k_eq_v: bool = False) -> DecoderFields:
    layer_types = _gemma_layer_types(hf_config, used, last_full=True)
    config = base_config(hf_config, used, qk_norm=True, scale_after_cast=False,
                          tie_embeddings=True, layer_types=layer_types, rope=Ropes(10000.0))
    # The reference's final-layer rewrite takes precedence over an explicit pattern.
    config['layer_types'] = layer_types
    sliding_dim = _records.integer(config.get('head_dim'), 'head_dim')
    kv_heads = _records.integer(config.get('num_kv_heads'), 'num_key_value_heads')
    k_eq_v = k_eq_v or bool(hf_config.get('attention_k_eq_v', False))
    used.update(('attention_k_eq_v', 'enable_moe_block', 'per_layer_config',
                 'global_head_dim', 'num_global_key_value_heads'))
    # Every layer reads its geometry from per_layer_config
    # (modeling_gemma4.py, Gemma4TextAttention reads layer_config), whose
    # entries the sliding layers leave at the model's head_dim and
    # num_key_value_heads. Where the config carries no per_layer_config key
    # at all, configuration_gemma4.py builds the full layers' entries from
    # global_head_dim (512 unless named) and, under attention_k_eq_v alone,
    # num_global_key_value_heads; a config that carries the key, null or
    # filled, leaves those two fields unread. The released E2B is one, its
    # global q_proj 8 heads of 256 against the 512 it names.
    full_dim, full_kv = sliding_dim, kv_heads
    if 'per_layer_config' in hf_config:
        entries = hf_config['per_layer_config'] or {}
        per_layer_entries = (entries.values() if isinstance(entries, Mapping)
                             else entries if isinstance(entries, (list, tuple))
                             else refuse(f"per_layer_config={entries!r}",
                                          "the per-layer overrides are records by layer"))
        for entry in per_layer_entries:
            if not isinstance(entry, Mapping):
                continue
            if entry.get('head_dim') is not None:
                full_dim = _records.integer(entry['head_dim'], 'per_layer_config head_dim')
            if entry.get('num_key_value_heads') is not None:
                stated = _records.integer(entry['num_key_value_heads'],
                              'per_layer_config num_key_value_heads')
                if full_kv != kv_heads and stated != full_kv:
                    refuse("per_layer_config num_key_value_heads",
                            f"the full layers name both {full_kv} and {stated}")
                full_kv = stated
    else:
        full_dim = _records.integer(hf_config.get('global_head_dim', 512), 'global_head_dim')
        global_kv = hf_config.get('num_global_key_value_heads')
        if global_kv is not None and k_eq_v:
            full_kv = _records.integer(global_kv, 'num_global_key_value_heads')
    # Proportional rope rotates a fraction of the full layers' head dims
    # and passes the rest through; sliding layers rotate all of theirs.
    stated = hf_config.get('rope_parameters')
    entries = GEMMA4_ROPE if stated is None else _records.record(stated, 'rope_parameters')
    rope_theta, rope_local_theta, partial = _gemma4_rope(entries)
    used.update(('rope_parameters', 'rope_theta'))
    per_layer = _records.integer(
        hf_config.get("hidden_size_per_layer_input", 0), "hidden_size_per_layer_input"
    )
    sharing = _trailing_sharers(hf_config, len(layer_types), 0)
    kinds = decoder_kinds(
        layer_types,
        kinds_of(config).get("sliding_attention", native_fields(LayerKind)()).value.window,
        rope_local_theta,
        None,
        None if full_dim == sliding_dim else full_dim,
    )
    if partial is not None and "sliding_attention" in kinds:
        # Gemma 4's sliding layers turn whole heads; its partial factor is the full layers' alone.
        kinds["sliding_attention"]["partial_rotary_factor"] = 1.0
    config.update(
        sandwich_norms=True,
        embedding_scale=True,
        attention_scale=1.0,
        v_norm=True,
        head_dim=sliding_dim,
        rope_theta=rope_theta,
        kinds=kinds,
        partial_rotary_factor=partial,
        # The reference widens only sharing layers. With none, this flag
        # changes neither the weights nor the forward operation.
        use_double_wide_mlp=bool(hf_config.get("use_double_wide_mlp", False)) and sharing is not None,
        kv_shared_layers=sharing,
        per_layer_input_dim=per_layer or None,
        per_layer_input_vocab=_records.integer(
            hf_config.get(
                "vocab_size_per_layer_input", _records.integer(hf_config["vocab_size"], "vocab_size")
            ),
            "vocab_size_per_layer_input",
        ),
    )
    used.update(('use_double_wide_mlp', 'num_kv_shared_layers',
                 'hidden_size_per_layer_input', 'vocab_size_per_layer_input',
                 'final_logit_softcapping'))
    # attention_logit_cap changes nothing on the text path and maps to
    # nothing. Gemma4TextAttention never passes it to its attention call
    # (modeling_gemma4.py, Gemma4TextAttention.forward). Only the audio
    # attention applies one.
    used.add('attention_logit_cap')
    softcap = hf_config.get('final_logit_softcapping')
    if softcap is not None:
        config['final_logit_softcap'] = _records.number(softcap, 'final_logit_softcapping')
    if full_kv != kv_heads:
        kinds_of(config).setdefault('full_attention', native_fields(LayerKind)())['num_kv_heads'] = full_kv
    # Every released Gemma 4 checkpoint carries the layer_scalar buffer the
    # reference initialises to one, so the tree always holds it.
    config.update(attention_k_eq_v=k_eq_v, layer_scalar="frozen")
    used.update(('moe_intermediate_size', 'expert_intermediate_size',
                 'num_experts', 'top_k_experts', 'chunk_size_feed_forward'))
    if hf_config.get('enable_moe_block'):
        # The 26B-A4B routes every layer beside its dense MLP.
        for field in ('num_experts', 'top_k_experts', 'moe_intermediate_size'):
            if hf_config.get(field) is None:
                refuse("enable_moe_block=True", f"the routed branch needs {field}")
        config['mixture'] = parallel_experts(hf_config)

    return config


def parallel_experts(hf_config: Mapping[str, object]) -> MixtureFields:
    """The routed branch a Gemma 4 layer runs beside its dense MLP."""
    return native_fields(Mixture)(experts=_records.integer(hf_config['num_experts'], 'num_experts'),
            top_k=_records.integer(hf_config['top_k_experts'], 'top_k_experts'),
            expert_features=_records.integer(hf_config['moe_intermediate_size'], 'moe_intermediate_size'),
            parallel=True)


def _gemma3_export(model: CausalTransformer) -> Mapping[str, object]:
    return {
        'hidden_activation': hf_activation(model.mlp),
        'query_pre_attn_scalar': (None if model.attention_scale is None
                                 else round(1.0 / model.attention_scale ** 2)),
        'final_logit_softcapping': model.final_logit_softcap,
        'attn_logit_softcapping': None,
        'use_bidirectional_attention': False,
    }


def _gemma4_export(model: CausalTransformer) -> Mapping[str, object]:
    """Return the Gemma4TextConfig fields that describe `model`'s computation.

    Every field is read off the model, never copied from the config a source
    shipped, so an exported model and a trained one write the same file.
    """
    # What the config carries is held to the round trip (`refuse_lossy_export`); how the model
    # runs, which it does not carry, is held here.
    fixed_fields(model, {'scale_after_cast': False}, 'Gemma4 computes {0!r} for this field')
    for name in ('output_gate', 'attention_sinks', 'attn_logit_softcap', 'altup',
                 'laurel_rank', 'activation_sparsity_pattern', 'num_nextn_predict_layers', 'dropout_rate'):
        if getattr(model, name):
            refuse(name, 'the standalone Gemma4 reference has no such computation')
    if model.partial_rotary_factor is not None and model.partial_rotary_type != 'proportional':
        refuse('partial_rotary_type', 'Gemma4 full layers use proportional rotary')
    if model.layer_scalar not in ('frozen', 'trainable'):
        refuse('layer_scalar', 'Gemma4 exports an explicit frozen or trainable scalar value')
    types = model.per_layer_types
    if not types or set(types) - {'sliding_attention', 'full_attention'}:
        refuse('layer_types', 'Gemma4 has only sliding and full attention')
    if types[-1] != 'full_attention':
        refuse('layer_types', 'Gemma4TextConfig forces the final layer to full attention')
    for name in set(types):
        kind = model.kind_of(name)
        mixer = kind.mixer or model.mixer
        if mixer is not None and not isinstance(mixer, AttentionMixer):
            refuse(f'kinds.{name}.mixer', 'Gemma4 uses ordinary attention')
        if isinstance(mixer, AttentionMixer) and (
            mixer.bidirectional_images or mixer.bidirectional_blocks or mixer.mrope_section is not None
        ):
            refuse(f'kinds.{name}.mixer', 'multimodal attention metadata needs its source wrapper')
        if kind.rope_scaling is not None:
            refuse(f'kinds.{name}.rope', 'Gemma4 uses plain local and proportional global rotary')
    full = model.kind_of('full_attention')
    local = model.kind_of('sliding_attention') if 'sliding_attention' in types else full
    if full.window is not None:
        refuse('kinds.full_attention.window', 'full attention is unwindowed')
    if 'sliding_attention' in types and (local.window is None or local.window < 1):
        refuse('kinds.sliding_attention.window', 'sliding attention needs a positive window')
    sharing = model.sharing_layers
    if sharing != tuple(range(model.num_layers - len(sharing), model.num_layers)):
        refuse('kv_shared_layers', 'Gemma4 can express only a trailing run of shared-KV layers')
    mixture = model.mixture
    fields: dict[str, object] = {
        'hidden_act': None, 'hidden_activation': hf_activation(model.mlp),
        'layer_types': list(types), 'intermediate_size': model.hidden_features,
        'head_dim': local.head_dim, 'num_key_value_heads': local.num_kv_heads,
        'global_head_dim': full.head_dim, 'num_global_key_value_heads': full.num_kv_heads,
        # Explicit overrides avoid configuration_gemma4.py:210-223 replacing
        # full geometry with global defaults or gating KV heads on K=V.
        'per_layer_config': {str(index): {'head_dim': full.head_dim, 'num_key_value_heads': full.num_kv_heads}
                             for index, name in enumerate(types) if name == 'full_attention'},
        'rope_theta': None, 'rope_scaling': None,
        'rope_parameters': {
            'sliding_attention': {'rope_type': 'default', 'rope_theta': local.rope_theta},
            'full_attention': {'rope_type': 'proportional', 'rope_theta': full.rope_theta,
                               'partial_rotary_factor': model.partial_rotary_factor or 1.0}},
        'sliding_window': local.window if 'sliding_attention' in types else 512,
        'attention_k_eq_v': model.attention_k_eq_v, 'num_kv_shared_layers': len(sharing),
        'use_double_wide_mlp': model.use_double_wide_mlp,
        'hidden_size_per_layer_input': model.per_layer_input_dim or 0,
        'vocab_size_per_layer_input': model.per_layer_input_vocab or model.vocab_size,
        'final_logit_softcapping': model.final_logit_softcap, 'use_bidirectional_attention': None,
        'enable_moe_block': mixture is not None,
    }
    if mixture is not None:
        if not mixture.parallel or tuple(model.sparse_layers) != tuple(range(model.num_layers)):
            refuse('mixture', 'Gemma4 routes a parallel expert branch on every layer')
        fields.update(num_experts=mixture.experts, top_k_experts=mixture.top_k,
                      moe_intermediate_size=mixture.expert_features or model.hidden_features)
    return fields


def gemma4_export_weights(family: DecoderFamily, model: CausalTransformer, variables: Mapping[str, object],
                           config: Mapping[str, object]) -> LazyTensors:
    """Return the Gemma 4 checkpoint tensors for `model` and `variables`.

    This is the one text inverse standalone Gemma 4 and DiffusionGemma share.
    Every layer scalar must sit in the collection `model.layer_scalar` names,
    and nothing else in `constants`, before the shared writer reads them.
    """
    mode = model.layer_scalar
    if mode not in LAYER_SCALAR_COLLECTIONS:
        raise ValueError('Gemma4 tensor export requires an explicit layer_scalar mode')
    found: dict[str, set[str]] = {}
    for each, collection in LAYER_SCALAR_COLLECTIONS.items():
        # A bare parameter tree is its own `params`.
        tree = variables.get(collection, variables if collection == 'params' else {})
        if not isinstance(tree, Mapping):
            raise ValueError('params and constants must contain native variable trees')
        found[each] = set(flatten_dict(dict(tree), sep='.'))
    scalars = {f'layers_{index}.layer_scalar' for index in range(model.num_layers)}
    if found['frozen'] - scalars:
        raise ValueError(f'unrepresented Gemma4 constants: {sorted(found["frozen"] - scalars)}')
    held = found[mode] & scalars
    stray = set().union(*(names & scalars for each, names in found.items() if each != mode))
    if held != scalars or stray:
        raise ValueError(f"layer_scalar={mode} keeps every layer scalar in "
                         f"{LAYER_SCALAR_COLLECTIONS[mode]} alone")
    return decoder_tensors(family, model, variables, config)


def _gemma2_export(model: CausalTransformer) -> Mapping[str, object]:
    return {**_gemma3_export(model), 'attn_logit_softcapping': model.attn_logit_softcap}


# Gemma 4's routed branch, named for what each norm normalises, as the
# block's own sandwich norms are.
_GEMMA4_MOE: dict[tuple[str, ...], tuple[str, ...]] = {
    ('router', 'proj', 'weight'): ('moe', 'router', 'proj', 'kernel'),
    ('router', 'scale'): ('moe', 'router', 'scale'),
    ('router', 'per_expert_scale'): ('moe', 'router', 'per_expert_scale'),
    ('post_feedforward_layernorm_1', 'weight'): ('moe', 'mlp_branch_norm', 'scale'),
    ('pre_feedforward_layernorm_2', 'weight'): ('moe', 'experts_input_norm', 'scale'),
    ('post_feedforward_layernorm_2', 'weight'): ('moe', 'experts_output_norm', 'scale'),
    ('layer_scalar',): ('layer_scalar',),
}


def gemma4_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """The variables-tree path of one Gemma 4 tensor: its routed branch's (`_GEMMA4_MOE`), its layer
    scalar's collection, the shared names' otherwise; a kv-shared layer's unused keys, values and key
    norm land nowhere (`_shared_unused`)."""
    if _shared_unused(name, config):
        return None
    parts = name.split('.')
    if len(parts) >= 4 and parts[:2] == ['model', 'layers'] and parts[2].isdigit():
        tail = tuple(parts[3:])
        if tail == ('layer_scalar',):
            mode = config.get("layer_scalar")
            if not isinstance(mode, str) or mode not in LAYER_SCALAR_COLLECTIONS:
                return refuse("layer_scalar", "the source scalar requires a frozen or trainable model mode")
            return (LAYER_SCALAR_COLLECTIONS[mode], f'layers_{parts[2]}', 'layer_scalar')
        layer = ('params', f'layers_{parts[2]}')
        if tail in _GEMMA4_MOE:
            return (*layer, *_GEMMA4_MOE[tail])
        if len(tail) == 2 and tail[0] == 'experts' and tail[1] in MOE_SHARED:
            return (*layer, 'moe', 'experts', tail[1], 'kernel')
    return dew_path(name, config)


_GEMMA4_MOE_NAMES = {path: name for name, path in _GEMMA4_MOE.items()}


def gemma4_export_path(name: str, config: Mapping[str, object]) -> str | None:
    """`gemma4_path` backwards; each stacked expert kernel takes the name its
    fused tensor is packed from (`FUSED_EXPERTS`)."""
    layer, _, rest = name.partition('.')
    tail = tuple(rest.split('.'))
    if layer.startswith('layers_'):
        stem = f"model.layers.{layer.removeprefix('layers_')}."
        if tail in _GEMMA4_MOE_NAMES:
            return stem + '.'.join(_GEMMA4_MOE_NAMES[tail])
        if len(tail) == 4 and tail[:2] == ('moe', 'experts') and tail[3] == 'kernel':
            return f'{stem}experts.{tail[2]}'
    return hf_tensor_name(name, config, sandwich_norms=True)


GEMMA3N_TEXT = DecoderFamily(
    ('gemma3n_text',),
    _gemma3n_config,
    lambda fields: fields.altup is not None,
    'gemma3n_text',
    'Gemma3nForCausalLM',
    _gemma3_export,
    sandwich_norms=True,
    weight_path=_gemma3n_path,
    preserve_source_layout=True,
    # Gemma3nTextConfig's window, which an omitted sliding_window means.
    defaults={'max_position_embeddings': 32768, 'hidden_activation': 'gelu_pytorch_tanh',
              'final_logit_softcapping': 30.0, 'pad_token_id': 0,
              'sliding_window': 512},
    padding_idx=True,
)

GEMMA4_TEXT = DecoderFamily(
    ('gemma4_text',),
    gemma4_config,
    lambda fields: bool(
        fields.v_norm or fields.per_layer_input_dim or fields.kv_shared_layers
    ),
    'gemma4_text',
    'Gemma4ForCausalLM',
    _gemma4_export,
    sandwich_norms=True,
    weight_path=gemma4_path,
    export_path=gemma4_export_path,
    packed=FUSED_EXPERTS,
    export_weights=gemma4_export_weights,
    preserve_source_layout=True,
    defaults={**GEMMA4_DEFAULTS, 'hidden_size_per_layer_input': 256,
              'vocab_size_per_layer_input': 262144},
    padding_idx=True,
)

def gemma4_unified_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    used.add('is_causal')
    if hf_config.get('is_causal', True) is not True:
        refuse('Gemma4Unified text configuration',
               'only its causal text path with optional vision blocks is mapped')
    if hf_config.get('enable_moe_block') or hf_config.get('hidden_size_per_layer_input'):
        refuse('Gemma4Unified text configuration', 'the unified decoder has no MoE or per-layer input branch')
    return gemma4_config(hf_config, used)


GEMMA4_UNIFIED_TEXT = dataclasses.replace(
    GEMMA4_TEXT, model_types=('gemma4_unified_text',), export_model_type='gemma4_unified_text',
    translate_config=gemma4_unified_config,
    architecture='Gemma4UnifiedForCausalLM',
    defaults={**GEMMA4_DEFAULTS, 'sliding_window': 1024, 'max_position_embeddings': 262144,
              'use_bidirectional_attention': 'vision'})


GEMMA3_TEXT = DecoderFamily(
    (GEMMA3_MODEL_TYPE,),
    _gemma3_config,
    lambda fields: bool(fields.sandwich_norms and fields.qk_norm),
    GEMMA3_MODEL_TYPE,
    'Gemma3ForCausalLM',
    _gemma3_export,
    sandwich_norms=True,
    preserve_source_layout=False,
    # Gemma3TextConfig's defaults for the fields its decoder reads.
    defaults={'hidden_activation': 'gelu_pytorch_tanh', 'query_pre_attn_scalar': 256, 'pad_token_id': 0,
              'sliding_window': 4096, 'head_dim': 256, 'max_position_embeddings': 131072},
    padding_idx=True,
)

GEMMA2 = DecoderFamily(
    ('gemma2',),
    _gemma2_config,
    lambda fields: bool(fields.sandwich_norms),
    'gemma2',
    'Gemma2ForCausalLM',
    _gemma2_export,
    sandwich_norms=True,
    preserve_source_layout=False,
    # Gemma2Config's defaults for the fields its decoder reads.
    defaults={'hidden_activation': 'gelu_pytorch_tanh', 'query_pre_attn_scalar': 256, 'pad_token_id': 0,
              'final_logit_softcapping': 30.0, 'attn_logit_softcapping': 50.0, 'sliding_window': 4096,
              'head_dim': 256, 'max_position_embeddings': 8192},
    padding_idx=True,
)

GEMMA = DecoderFamily(
    ('gemma',),
    _gemma_config,
    lambda fields: bool(fields.embedding_scale),
    'gemma',
    'GemmaForCausalLM',
    lambda model: {},
    preserve_source_layout=False,
    defaults={'max_position_embeddings': 8192, 'hidden_act': 'gelu_pytorch_tanh', 'pad_token_id': 0},
    padding_idx=True,
)
