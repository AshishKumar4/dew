"""Assemble a DiffusionGemma for the shared pretrained-model loader.

The checkpoint stores its text weights under two prefixes, the encoder's and
the decoder's, and both map into one Flax subtree. The vision and projection
weights keep subtrees of their own. This module opens no checkpoint or
processor, and it adds no public loading function for this family; the
shared loader calls it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np
from flax.traverse_util import flatten_dict

from dew import records
from dew.diffusion.block import BlockProcess
from dew.interop.config_records import NativeFields
from dew.interop.hf_decoders import _export_config, translate_config, translate_denoiser_weights
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.multimodal import VisionConditioner
from dew.nn.vision import tower_variables as vision_variables
from dew.nn.vision.gemma4 import (
    Gemma4Projector,
    Gemma4Vision,
    export_gemma4_vision_config,
    translate_gemma4_projector_config,
    translate_gemma4_projector_weights,
    translate_gemma4_vision_config,
)
from dew.objectives.base import Variables
from dew.registry import from_record, projectors, towers


# The wrapper config states these three with a default this assembly supplies
# when the file omits them, so each reads `config.get` and narrows the answer.
def _section(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    return records.record(config[name], name)


def _integer(config: Mapping[str, object], name: str, default: int | None = None) -> int:
    return records.integer(config.get(name, default), name)


def _number(config: Mapping[str, object], name: str, default: float) -> float:
    return records.number(config.get(name, default), name)


def text_config(config: Mapping[str, object]) -> Mapping[str, object]:
    if config.get("model_type") != "diffusion_gemma":
        raise ValueError("the DiffusionGemma assembly requires a diffusion_gemma wrapper")
    return _section(config, "text_config")


def build(config: Mapping[str, object], *, dtype: str | None = "bfloat16",
          attention_impl: str = "auto", max_seq_len: int | None = None) -> DiffusionGemma:
    """Build the native `DiffusionGemma` module from a checkpoint's config, without allocating parameters.

    `config` is the `diffusion_gemma` wrapper config; any other config, or
    one with an audio encoder, raises ValueError. `canvas_length` is 256
    when the config omits it.
    """
    fields = translate_config(text_config(config))
    fields["causal"] = True
    if max_seq_len is not None:
        fields["max_seq_len"] = max_seq_len
    precise = NativeFields(CausalTransformer, {**fields, "dtype": dtype, "attention_impl": attention_impl})
    text = from_record(CausalTransformer, precise)
    conditioner = None
    if config.get("vision_config") is not None:
        tower = translate_gemma4_vision_config(_section(config, "vision_config"))
        projector = translate_gemma4_projector_config(tower["fields"], text.emb_features)
        conditioner = VisionConditioner(
            vision=towers.from_record(tower),
            projection=projectors.from_record(projector), dtype=text.dtype,
            precision=text.precision)
    if config.get("audio_config") is not None:
        raise ValueError("DiffusionGemma publishes no audio encoder")
    return DiffusionGemma(text=text, canvas_length=_integer(config, "canvas_length", 256),
                          conditioner=conditioner)


def translate_weights(
    tensors: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    *,
    param_dtype: str = "float32",
) -> Variables:
    """Map a DiffusionGemma checkpoint's tensors into its variables tree.

    Text weights route through the shared decoder map, vision and projection
    weights through their own. Each component casts its own leaves to
    `param_dtype` as it binds them, so no full FP32 tree is built and narrowed.
    """
    text: dict[str, np.ndarray] = {}
    vision: dict[str, np.ndarray] = {}
    projection: dict[str, np.ndarray] = {}
    for name, tensor in tensors.items():
        if name.startswith("model.encoder.vision_tower."):
            vision[name.removeprefix("model.encoder.vision_tower.")] = tensor
        elif name.startswith("model.encoder.embed_vision."):
            projection[name.removeprefix("model.encoder.embed_vision.")] = tensor
        else:
            text[name] = tensor
    fields = translate_config(text_config(config))
    mapped = translate_denoiser_weights(text, fields, param_dtype=param_dtype)
    params = {"text": mapped["text"]["params"],
              "self_conditioning": mapped["self_conditioning"]["params"]}
    variables = {"params": params}
    for collection, tree in mapped["text"].items():
        if collection != "params":
            variables[collection] = {"text": tree}
    if config.get("vision_config") is not None:
        if not vision or not projection:
            raise ValueError("vision_config requires both vision_tower and embed_vision tensors")
        tower_variables = vision_variables("gemma4", vision, param_dtype)
        params["conditioner"] = {
            "tower": tower_variables["params"],
            "projector": translate_gemma4_projector_weights(projection, param_dtype=param_dtype)}
        if "constants" in tower_variables:
            variables.setdefault("constants", {})["conditioner"] = {"tower": tower_variables["constants"]}
    elif vision or projection:
        raise ValueError("vision weights require vision_config")
    return variables


def generation_process(config: Mapping[str, object], generation: Mapping[str, object]) -> BlockProcess:
    """Build the block-diffusion process the published generation config describes.

    Each field falls back to the reference's own default when
    `generation_config.json` omits it.
    """
    sampler = generation.get("sampler_config")
    if sampler is None:
        budget = 0.1
    else:
        sampler_fields = _section(generation, "sampler_config")
        kind = sampler_fields.get("_cls_name", "EntropyBoundSamplerConfig")
        if kind != "EntropyBoundSamplerConfig":
            raise ValueError(f"DiffusionGemma sampler {kind!r} is not entropy-bounded sampling")
        budget = _number(sampler_fields, "entropy_bound", 0.1)
    return BlockProcess(
        canvas_length=_integer(config, "canvas_length", 256),
        vocab_size=_integer(text_config(config), "vocab_size"),
        max_steps=_integer(generation, "max_denoising_steps", 48),
        t_min=_number(generation, "t_min", 0.4),
        t_max=_number(generation, "t_max", 0.8),
        entropy_bound=budget,
        stability_threshold=_integer(generation, "stability_threshold", 1),
        confidence_threshold=_number(generation, "confidence_threshold", 0.005))


def scalar_placement(text: CausalTransformer, variables: Variables) -> CausalTransformer:
    """Return `text` with its layer-scalar policy set to the one `variables` keeps.

    Google makes the per-layer skip scale a parameter and Transformers calls
    the same tensor a buffer, so a source declares which it is and the export
    reads the collection that policy names. `BlockDiffusionObjective` trains
    the scalar, which moves every one into `params`, so a finished SFT's tree
    disagrees with the source it loaded. The tree being written decides, so an
    exported run and a saved source agree.
    """
    if text.layer_scalar is None:
        return text
    for mode, collection in (("trainable", "params"), ("frozen", "constants")):
        held = variables.get(collection, {})
        if any(isinstance(node, Mapping) and "layer_scalar" in node for node in held.values()):
            return text if text.layer_scalar == mode else text.clone(layer_scalar=mode)
    return text


# Gemma 4 text fields DiffusionGemma's text config has none of: its modules
# fix them (keys read as values on full layers, a routed mixture beside every
# dense MLP, the 30 softcap) or never build them (per-layer inputs, shared KV
# layers, the double-wide MLP).
_GEMMA4_ONLY = ("architectures", "use_cache", "attention_k_eq_v", "enable_moe_block",
                "final_logit_softcapping", "hidden_size_per_layer_input", "vocab_size_per_layer_input",
                "num_kv_shared_layers", "use_double_wide_mlp")
# The run's own settings, which no config carries: precision and kernels,
# training-time dropout and initialization, layout, and where the layer
# scalar is stored (`scalar_placement`). `causal` is no setting but is fixed
# by the classes on both sides: Dew's text stack is the causal encoder view
# DiffusionGemma requires, and transformers' encoder attends causally unless
# use_bidirectional_attention is "all", which this config never writes,
# while its decoder never does (modeling_diffusion_gemma.py:281, :383, 5.16.1).
_RUN_SETTINGS = ("dtype", "precision", "force_fp32_for_softmax", "attention_impl", "kv_cache", "causal",
                 "scan_layers", "bank_layers", "remat", "dropout_rate", "embedding_dropout_rate",
                 "attention_dropout_rate", "initializer_range", "depth_scaled_init", "layer_scalar")


def published_config(model: DiffusionGemma) -> Mapping[str, object]:
    """The config transformers' DiffusionGemmaForBlockDiffusion reads for
    `model`: its text stack in diffusion_gemma_text's fields, its Gemma 4
    tower's vision_config and its canvas length.

    Dew places images by position and keeps no image token ids, so the config
    names none and transformers' defaults stand. A model the written fields
    would rebuild differently is refused, naming the fields that differ.
    """
    text = {name: value for name, value in _export_config(model.text).items() if name not in _GEMMA4_ONLY}
    text.update(model_type="diffusion_gemma_text",
                use_bidirectional_attention=None if model.conditioner is None else "vision")
    rebuilt = from_record(CausalTransformer, {
        **translate_config(text), **{name: getattr(model.text, name) for name in _RUN_SETTINGS}})
    lost = [field.name for field in dataclasses.fields(CausalTransformer)
            if field.name not in ("parent", "name")
            and getattr(rebuilt, field.name) != getattr(model.text, field.name)]
    if lost:
        raise ValueError(f"DiffusionGemma's text config cannot carry this model's {', '.join(lost)}")
    config: dict[str, object] = {"model_type": "diffusion_gemma",
                                 "architectures": ["DiffusionGemmaForBlockDiffusion"],
                                 "text_config": text, "canvas_length": model.canvas_length,
                                 "tie_word_embeddings": model.text.tie_embeddings}
    if model.conditioner is not None:
        tower, projection = model.conditioner.vision, model.conditioner.projection
        # transformers builds the tower from vision_config and the projector
        # from the tower's epsilon and the text width.
        if not (isinstance(tower, Gemma4Vision) and projection == Gemma4Projector(
                text_width=model.text.emb_features, norm_eps=tower.rms_norm_eps)):
            raise ValueError("DiffusionGemma's vision_config describes a Gemma 4 tower and its projector, "
                             f"not {tower!r} with {projection!r}")
        config["vision_config"] = export_gemma4_vision_config(tower)
    return config


def _refuse_unreadable(config: Mapping[str, object]) -> None:
    """The published implementation, transformers' DiffusionGemmaForBlockDiffusion
    (5.16.1), builds a routed mixture on every layer and a vision tower: a
    config without `text_config.num_experts` or `vision_config` is one it
    cannot build (modeling_diffusion_gemma.py:616, 1019), so files written in
    its layout would be read by no one."""
    text = config.get("text_config")
    missing = [name for name, present in (
        ("text_config.num_experts", isinstance(text, Mapping) and text.get("num_experts")),
        ("vision_config", config.get("vision_config"))) if not present]
    if missing:
        raise ValueError(
            f"this DiffusionGemma has no {' or '.join(missing)}, and transformers' "
            "DiffusionGemmaForBlockDiffusion, the implementation its checkpoint layout is for, builds "
            "both, so it cannot read an export of it; keep training it from the run's checkpoint "
            "(dew.checkpoints) rather than a published directory")


def export_weights(
    model: DiffusionGemma, variables: Variables, config: Mapping[str, object]
) -> dict[str, np.ndarray]:
    """Return the checkpoint tensors for a DiffusionGemma, keyed by source name.

    The decoder half goes through `export_decoder_weights` and is renamed under
    the reference's `model.decoder.` and `model.encoder.` prefixes. A config
    the published implementation cannot build is refused (`_refuse_unreadable`).
    """
    from dew.interop.hf_decoders import export_decoder_weights
    from dew.nn.vision.gemma4 import _GEMMA4_VISION_NORMS, _GEMMA4_VISION_TENSORS

    _refuse_unreadable(config)
    params = variables["params"]
    text_variables = {collection: tree["text"] for collection, tree in variables.items()
                      if "text" in tree}
    text = scalar_placement(model.text, text_variables)
    tensors: dict[str, np.ndarray] = {}
    for name, tensor in export_decoder_weights(text, text_variables, text_config(config)).items():
        target = "model.decoder." + name.removeprefix("model.") if name.startswith("model.") else name
        tensors[target] = tensor
        if name.endswith(".layer_scalar"):
            tensors[target.replace("model.decoder.", "model.encoder.language_model.")] = tensor
    for name, leaf in flatten_dict(params["self_conditioning"], sep=".").items():
        module, kind = name.split(".")
        tensors[f"model.decoder.self_conditioning.{module}.weight"] = np.ascontiguousarray(
            np.asarray(leaf).T if kind == "kernel" else np.asarray(leaf))
    if "conditioner" in params:
        inverse = {value: key for key, value in _GEMMA4_VISION_TENSORS.items()}
        norms = {value: key for key, value in _GEMMA4_VISION_NORMS.items()}
        for collection in ("params", "constants"):
            tower = variables.get(collection, {}).get("conditioner", {}).get("tower", {})
            for name, raw in flatten_dict(tower, sep=".").items():
                parts = name.split(".")
                if tuple(parts) in inverse:
                    target = inverse[tuple(parts)]
                elif parts[0].startswith("layers_"):
                    index = parts[0].removeprefix("layers_")
                    tail = [norms.get(part, part) for part in parts[1:-1]]
                    if parts[-1] == "kernel":
                        tail = [*tail, "linear"]
                    ending = parts[-1] if collection == "constants" else "weight"
                    target = f"encoder.layers.{index}." + ".".join([*tail, ending])
                else:
                    raise ValueError(f"unknown vision parameter {name!r}")
                leaf = np.asarray(raw)
                tensors["model.encoder.vision_tower." + target] = np.ascontiguousarray(
                    leaf.T if parts[-1] == "kernel" else leaf)
        tensors["model.encoder.embed_vision.embedding_projection.weight"] = np.ascontiguousarray(
            np.asarray(params["conditioner"]["projector"]["projection"]["kernel"]).T)
    return tensors


__all__ = ["build"]
