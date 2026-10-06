"""Write a CausalTransformer back out in its family's Hugging Face layout: config, tensors and assets."""

import dataclasses
import json
import operator
import os
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Protocol

import jax
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

from dew import records
from dew.interop.decoder_config import _family_for_model, _hf_activation, families, translate_config
from dew.interop.decoder_paths import Packed, _hf_name
from dew.interop.safetensors_io import LazyTensors
from dew.interop.streaming import WeightLayout
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers import AttentionMixer
from dew.registry import from_record

GENERATION_CONFIG_FILE = "generation_config.json"


class ExportTokenizer(Protocol):
    """A tokenizer that writes its own HF files. The byte vocabulary has
    none, so it is recorded by name only."""

    def save_pretrained(self, directory: str, /) -> tuple[str, ...] | None:
        """Return the files it wrote, which transformers returns and this module does not read."""
        ...


GENERATION_DEFAULTS: Mapping[str, object] = MappingProxyType({"do_sample": True, "use_cache": True})
"""The generation_config.json an export writes when nothing names one: sampling
with the KV cache, which is what transformers' generate reads by default."""


def save_export_assets(
    directory,
    *,
    tokenizer: str | ExportTokenizer | None = None,
    generation_config: Mapping[str, object] | None = None,
    named: bool = True,
) -> None:
    """Write the tokenizer files and generation_config.json beside exported weights.

    Readers of the HF layout (transformers, llama.cpp and the runtimes on it) locate the
    vocabulary through tokenizer_config.json, so the tokenizer writes its own files here
    (`save_pretrained`) and the directory is the whole record of it: a name, a hub repo
    or a path on the machine that exported it, is recorded nowhere, and a
    `tokenizer_name` the source's generation config carried is dropped. A name is
    resolved through `tokenizer_for` from local files only. Dew's byte vocabulary has
    no files, so it alone is recorded, as `tokenizer_name: "byte"`; unless `named`,
    where the layout has no such field and it is refused before anything is written.
    """
    values = dict(GENERATION_DEFAULTS if generation_config is None else generation_config)
    values.pop('tokenizer_name', None)
    byte = False
    writer: ExportTokenizer | None = None
    if isinstance(tokenizer, str):
        from dew.data.text import ByteTokenizer, tokenizer_for

        resolved = tokenizer_for(tokenizer, local_files_only=True)
        byte = isinstance(resolved, ByteTokenizer)
        writer = None if isinstance(resolved, ByteTokenizer) else resolved
    elif tokenizer is not None:
        writer = tokenizer
    if byte and not named:
        raise ValueError("the byte vocabulary has no tokenizer files, and this layout's "
                         "generation_config.json has no field to name it by; export a run trained "
                         "on a Hugging Face tokenizer")
    os.makedirs(directory, exist_ok=True)
    if writer is not None:
        writer.save_pretrained(str(directory))
    if byte:
        values['tokenizer_name'] = "byte"
    with open(os.path.join(directory, GENERATION_CONFIG_FILE), 'w') as handle:
        json.dump(values, handle, indent=2)


def export_decoder_weights(model: nn.Module, variables: Mapping[str, object],
                           config: Mapping[str, object]) -> Mapping[str, np.ndarray]:
    """Encode whole native variables as canonical model.* / lm_head.* tensors.

    The family owns collection packing and any fused tensor geometry. A
    wrapper adds only its naming envelope after this shared inverse.

    A config that carries `tie_word_embeddings` is read exactly as
    `_base_config` reads it, an explicit null included; only an absent key
    asks the family for its own default. A derived config therefore reaches
    its weight encoder without translating geometry that encoder may not
    support.
    """
    if not isinstance(model, CausalTransformer):
        raise TypeError('decoder weight export requires a CausalTransformer')
    model_type = config.get('model_type')
    if not isinstance(model_type, str) or model_type not in families():
        raise ValueError(f'no decoder tensor encoder for model_type {model_type!r}')
    family = families()[model_type]
    tied = (bool(config['tie_word_embeddings']) if 'tie_word_embeddings' in config
            else records.boolean(family.translate_config(config, set()).get('tie_embeddings'),
                       'tie_embeddings'))
    if tied != model.tie_embeddings:
        raise ValueError('tie_word_embeddings disagrees with the native model')
    return family.export_weights(model, variables, {**config, 'tie_word_embeddings': tied})


def _dense_decoder_weights(model: CausalTransformer, variables: Mapping[str, object],
                           config: Mapping[str, object]) -> Mapping[str, np.ndarray]:
    if model.per_layer_input_dim or model.sharing_layers or model.v_norm:
        raise ValueError(
            'per-layer input embeddings, KV sharing and the values norm have '
            'no counterpart in this dense tensor encoder: per_layer_input_dim, '
            'kv_shared_layers or v_norm is set')
    mixers = [model.mixer] + [kind.mixer for kind in (model.kinds or {}).values()]
    if (model.output_gate or model.partial_rotary_factor is not None
            or any(mixer is not None and not isinstance(mixer, AttentionMixer) for mixer in mixers)):
        raise ValueError(
            'the attention output gate, a partial rotary and a mixer other than attention '
            'have no counterpart in this dense tensor encoder')
    family = families()[records.text(config['model_type'], 'model_type')]
    if model.mixture is not None and family.export_path is _hf_name:
        raise ValueError('a model with a mixture has no routed tensor writer in this family')
    return _decoder_tensors(model, variables, config)


def _decoder_tensors(model: CausalTransformer, variables: Mapping[str, object],
                     config: Mapping[str, object]) -> LazyTensors:
    """Write every leaf under the name the family's `export_path` gives it.

    Each leaf is stored as the load oriented it, a 2-D kernel transposed, and
    the family's `packed` tensors are built from their parts. Gemma 4's layer
    scalars are read from the collection `model.layer_scalar` names.
    """
    family = families()[records.text(config['model_type'], 'model_type')]
    params = variables.get('params', variables)
    if not isinstance(params, Mapping):
        raise ValueError('params must contain the decoder parameter tree')
    tree = variables if 'params' in variables else {'params': params}
    leaves = dict(flatten_dict(dict(params), sep='.'))
    constants = tree.get('constants')
    if model.layer_scalar == 'frozen' and isinstance(constants, Mapping):
        leaves.update((name, value) for name, value in flatten_dict(dict(constants), sep='.').items()
                      if name.endswith('.layer_scalar'))
    layouts: dict[str, WeightLayout] = {}
    for name, value in leaves.items():
        target = family.export_path(name, config)
        if target is None:
            continue
        if not isinstance(value, (jax.Array, np.ndarray)):
            raise TypeError(f'{name} is a {type(value).__name__}, not an array')
        if target in layouts:
            raise ValueError(f'{name} and {layouts[target].paths[0]} both write {target}')
        kernel = name.endswith('.kernel') and value.ndim == 2
        layouts[target] = WeightLayout(target, (('params', *name.split('.')),), value.shape[::-1]
                                       if kernel else value.shape, (1, 0) if kernel else None)
    return _layout_tensors(_packed_layouts(layouts, family.packed), tree, model.layer_scalar)


def _packed_layouts(layouts: Mapping[str, WeightLayout],
                    packed: Sequence["Packed"]) -> dict[str, WeightLayout]:
    """`layouts` with each `packed` source tensor built from its parts' layouts,
    in the place of its first part."""
    source: dict[str, WeightLayout] = {}
    for name, layout in layouts.items():
        found = next(((packing, part) for packing in packed for part in packing.parts
                      if name.endswith(part)), None)
        if found is None:
            source[name] = layout
            continue
        packing, part = found
        stem = name.removesuffix(part)
        missing = [stem + other for other in packing.parts if stem + other not in layouts]
        if missing:
            raise ValueError(f'{stem}{packing.name} packs {name} with {missing}, which nothing writes')
        if part == packing.parts[0]:
            source[stem + packing.name] = packing.layout(
                stem + packing.name, [layouts[stem + other] for other in packing.parts])
    return source


def _layout_tensors(layouts: Mapping[str, WeightLayout], variables: Mapping[str, object],
                    scalar_mode: str | None = None,
                    retained: Mapping[str, np.ndarray] | None = None) -> LazyTensors:
    """The layouts' tensors and `retained` beside them, each built when it is
    read, so a sharded writer holds one shard of the export, not all of it."""
    kept = retained or {}
    specs = {**{name: jax.ShapeDtypeStruct(np.shape(value), np.asarray(value).dtype)
                for name, value in kept.items()},
             **{name: jax.ShapeDtypeStruct(layout.shape, layout.stored_dtype(variables, scalar_mode))
                for name, layout in layouts.items()}}
    return LazyTensors(specs, lambda name: (
        layouts[name].export(variables, scalar_mode) if name in layouts else kept[name]))


def _export_config(model) -> Mapping[str, object]:
    """Write a CausalTransformer's fields back into HF vocabulary."""
    family = _family_for_model(model)
    exported = family.export_fields(model)
    chunked = sorted(name for name, kind in (model.kinds or {}).items() if kind.chunk is not None)
    if chunked and 'attention_chunk_size' not in exported:
        # Llama 4's attention_chunk_size is the one config field a chunk
        # goes back out under.
        raise ValueError(
            f"kinds {chunked} attend by chunk, which the {family.export_model_type} "
            f"config does not carry")
    sandwich = bool(model.sandwich_norms)
    config: dict[str, object] = {
        'model_type': family.export_model_type,
        'architectures': [family.architecture],
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
        'hidden_act': _hf_activation(model.mlp),
        'use_cache': True,
    }
    # A dial only some families' references read cannot ride in another
    # family's config. Qwen2 splits the o_proj bias from the others, and
    # Dream's reference is that block with the causal mask dropped
    # (modeling_dream.py, DreamAttention builds o_proj bias-free over
    # biased q/k/v); Gemma 2 alone applies the attention softcap (Gemma 3
    # reads the field without passing it on). A checkpoint written under a
    # family that would drop the dial is refused naming it.
    if (model.o_proj_bias is not None and model.o_proj_bias != model.attention_bias
            and family.export_model_type not in ('qwen2', 'dream', 'gpt_neo')):
        raise ValueError(
                "o_proj_bias differs from attention_bias, which only the qwen2 "
                "and dream references build, so the model cannot be written as "
                f"{family.export_model_type}")
    if model.attn_logit_softcap is not None and family.export_model_type != 'gemma2':
        raise ValueError(
            "attn_logit_softcap is applied by the gemma2 reference alone, so "
            f"the model cannot be written as {family.export_model_type}")
    types = model.per_layer_types
    if any(layer != 'full_attention' for layer in types):
        config['layer_types'] = list(types)
    sliding = model.kind_of('sliding_attention') if 'sliding_attention' in types else None
    local_theta = None if sliding is None or sliding.rope_theta == model.rope_theta else sliding.rope_theta
    # Gemma3TextConfig and Olmo3Config give an unstated sliding base their
    # own default rather than rope_theta (`_rope`'s local_default), so a
    # sliding model of theirs states both bases.
    if sliding is not None and family.export_model_type in ('gemma3_text', 'olmo3'):
        local_theta = sliding.rope_theta or model.rope_theta
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
    # The ramp writes in the flat spelling every family that reads one
    # accepts (rope_scaling beside rope_theta, as Llama 3.1 ships it); a
    # ramp that differs between kinds has that spelling in no reference
    # this exports, so it is refused naming the kinds.
    ramps = {kind: model.kind_of(kind).rope_scaling for kind in set(types)}
    if len(set(ramps.values())) > 1:
        raise ValueError(
            f"rope_scaling differs between layer kinds ({sorted(ramps)}), which "
            "no exported family spells; the model cannot be written back")
    ramp = model.kind_of(types[0]).rope_scaling
    if ramp is not None:
        config['rope_scaling'] = dataclasses.asdict(ramp)
    # A YaRN table replaces the frequencies rather than riding over them.
    # One table the whole model shares writes flat beside rope_theta, which
    # is where the single-table references read it. A table that differs
    # between the kinds is what Olmo3RotaryEmbedding builds, one per kind
    # out of nested rope_parameters (modeling_olmo3.py:277-291, the
    # spelling Olmo3Config.to_dict writes); a reference with one table for
    # the whole model cannot say it, so that is refused naming the kinds.
    yarns = {kind: model.kind_of(kind).yarn for kind in sorted(set(types))}
    ramped = {kind: yarn for kind, yarn in yarns.items() if yarn is not None}
    per_kind = bool(ramped) and (set(ramped) != set(yarns)
                                 or len(set(ramped.values())) > 1
                                 or local_theta is not None)
    if per_kind:
        if not sandwich:
            raise ValueError(
                f"the yarn table differs between the layer kinds {sorted(yarns)}, "
                f"and the {family.export_model_type} reference rotates every layer "
                "at one table; the model cannot be written back")
        config['rope_parameters'] = {
            kind: (dataclasses.asdict(yarn) if yarn is not None else
                   {'rope_type': 'default', 'rope_theta': model.kind_of(kind).rope_theta})
            for kind, yarn in yarns.items()}
        config.pop('rope_local_base_freq', None)
    elif ramped:
        config['rope_scaling'] = dataclasses.asdict(next(iter(ramped.values())))
    config.update(exported)
    return {key: value for key, value in config.items() if value is not None or key == 'pad_token_id'}


_RUNTIME_FIELDS = frozenset({
    'parent', 'name', 'dtype', 'precision', 'attention_impl', 'kv_cache', 'remat', 'scan_layers',
    'bank_layers', 'dropout_rate', 'embedding_dropout_rate', 'attention_dropout_rate',
    'max_seq_len', 'mask_token_id', 'layer_scalar', 'scale_after_cast'})
"""CausalTransformer fields that say how a model runs or trains, not what it
computes. `layer_scalar` is whether Gemma 4's scalars train; either way the
forward multiplies by them. `scale_after_cast` orders a norm's scale and its
cast to the compute dtype, which are the same product in fp32."""

_RESOLVED: Mapping[str, Callable[[CausalTransformer], object]] = {
    "num_kv_heads": lambda model: model.kv_heads,
    "head_dim": lambda model: model.features_per_head,
    "layer_types": lambda model: model.per_layer_types,
    "kinds": lambda model: tuple(model.kind_of(kind) for kind in sorted(set(model.per_layer_types))),
    "partial_rotary_factor": lambda model: model.partial_rotary_factor or 1.0,
    "per_layer_input_vocab": lambda model: model.per_layer_input_vocab or model.vocab_size,
    "position_embedding_size": lambda model: (
        model.position_embedding_size or model.max_seq_len if model.position_embedding == "learned" else None
    ),
}
"""Fields whose None stands for a value the forward derives, spelled out."""


def _refuse_lossy_export(model: CausalTransformer, config: Mapping[str, object]) -> None:
    """Refuse an exported config that reads back as a different computation.

    The config is translated again by the family it names, which is the
    reading the parity fixtures hold to transformers. A field the family's
    config does not carry comes back at the backbone default instead of the
    model's value, and the export would load in transformers, and here, as
    another model. The differing fields are named.
    """
    try:
        rebuilt = from_record(CausalTransformer, {**translate_config(config), 'dtype': model.dtype})
    except KeyError as error:
        raise ValueError(f"the {config['model_type']} config written for this model lacks {error}, which "
                         f"that family reads: this model's computation (its mixer or experts) has no "
                         f"exported config; no exported family carries it") from error
    except ValueError as error:
        raise ValueError(f"the {config['model_type']} config written for this model does not read back: "
                         f"{error}") from error
    lost: dict[str, tuple[str, str]] = {}
    for declared in dataclasses.fields(model):
        if declared.name in _RUNTIME_FIELDS:
            continue
        resolve = _RESOLVED.get(declared.name, operator.attrgetter(declared.name))
        ours, theirs = resolve(model), resolve(rebuilt)
        if ours != theirs:
            lost[declared.name] = (repr(theirs), repr(ours))
    if lost:
        raise ValueError(
            f"{sorted(lost)} would not survive an export as {config['model_type']}: its config reads back "
            f"{', '.join(f'{name}={read}' for name, (read, _) in lost.items())} where this model has "
            f"{', '.join(f'{name}={held}' for name, (_, held) in lost.items())}, "
            "so transformers would compute "
            "another model; no exported family carries this computation"
        )
