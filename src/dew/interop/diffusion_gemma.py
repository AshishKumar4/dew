"""DiffusionGemma assembly for the shared pretrained-model loader.

Checkpoint text aliases collapse to one Flax subtree; vision and projection
weights retain their separate subtrees. This module opens no checkpoint or
processor and introduces no family-specific public loading entry point.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from flax import linen as nn

from dew.diffusion.block import BlockProcess
from dew.interop.hf_decoders import translate_config, translate_denoiser_weights
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.multimodal import VisionConditioner
from dew.nn.vision import (
    projector_from_record, tower_from_record, translate_gemma4_projector_config,
    translate_gemma4_projector_weights, translate_gemma4_vision_config,
    translate_gemma4_vision_weights,
)
from dew.objectives.base import Variables
from dew.registry import with_precision


def _section(config: Mapping[str, object], name: str) -> dict[str, object]:
    value = config[name]
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed configuration record")
    return dict(value)


def _integer(config: Mapping[str, object], name: str, default: int | None = None) -> int:
    value = config.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(config: Mapping[str, object], name: str, default: float) -> float:
    value = config.get(name, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


def text_config(config: Mapping[str, object]) -> dict[str, object]:
    if config.get("model_type") != "diffusion_gemma":
        raise ValueError("the DiffusionGemma assembly requires a diffusion_gemma wrapper")
    return _section(config, "text_config")


def build(config: Mapping[str, object], *, dtype: str = "bfloat16",
          attention_impl: str = "auto", max_seq_len: int | None = None) -> DiffusionGemma:
    """Build native model values without allocating parameters."""
    fields = translate_config(text_config(config))
    fields["causal"] = True
    if max_seq_len is not None:
        fields["max_seq_len"] = max_seq_len
    text = CausalTransformer(**with_precision(
        "causal_transformer", fields, dtype=dtype, attention_impl=attention_impl))
    conditioner = None
    if config.get("vision_config") is not None:
        tower = translate_gemma4_vision_config(_section(config, "vision_config"))
        projector = translate_gemma4_projector_config(tower, text.emb_features)
        conditioner = VisionConditioner(
            family="diffusion_gemma", vision=tower_from_record(tower),
            projection=projector_from_record(projector), dtype=text.dtype,
            precision=text.precision)
    if config.get("audio_config") is not None:
        raise ValueError("DiffusionGemma publishes no audio encoder")
    return DiffusionGemma(text=text, canvas_length=_integer(config, "canvas_length", 256),
                          conditioner=conditioner)


def translate_weights(tensors: Mapping[str, np.ndarray], config: Mapping[str, object]) -> Variables:
    """Map published aliases and media tensors into one native parameter tree."""
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
    mapped = translate_denoiser_weights(text, fields)
    params = {"text": mapped["text"]["params"],
              "self_conditioning": mapped["self_conditioning"]["params"]}
    variables = {"params": params}
    for collection, tree in mapped["text"].items():
        if collection != "params":
            variables[collection] = {"text": tree}
    if config.get("vision_config") is not None:
        if not vision or not projection:
            raise ValueError("vision_config requires both vision_tower and embed_vision tensors")
        tower_variables = translate_gemma4_vision_weights(vision)
        params["conditioner"] = {
            "tower": tower_variables["params"],
            "projector": translate_gemma4_projector_weights(projection)}
        if "constants" in tower_variables:
            variables.setdefault("constants", {})["conditioner"] = {"tower": tower_variables["constants"]}
    elif vision or projection:
        raise ValueError("vision weights require vision_config")
    return variables


def generation_process(config: Mapping[str, object], generation: Mapping[str, object]) -> BlockProcess:
    """Published inference defaults overridden by generation_config.json."""
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


def export_weights(model: nn.Module, variables: Variables, config: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Write canonical decoder tensors under the native model's explicit scalar policy."""
    from dew.interop.hf_decoders import _GEMMA4_MOE, _flatten, _hf_name
    from dew.nn.vision import _GEMMA4_VISION_TENSORS

    if not isinstance(model, DiffusionGemma):
        raise TypeError("DiffusionGemma export requires its native model value")
    mode = model.text.layer_scalar
    if mode not in ("frozen", "trainable"):
        raise ValueError("DiffusionGemma export requires an explicit layer_scalar mode")
    params = variables["params"]
    flat = _flatten(params["text"])
    scalar_collection = "params" if mode == "trainable" else "constants"
    other_collection = "constants" if mode == "trainable" else "params"
    for index in range(model.text.num_layers):
        layer = f"layers_{index}"
        wrong = variables.get(other_collection, {}).get("text", {}).get(layer, {})
        if "layer_scalar" in wrong:
            raise ValueError(f"layer_scalar in {other_collection} disagrees with model mode {mode}")
        value = variables[scalar_collection]["text"][layer]["layer_scalar"]
        flat[f"{layer}.layer_scalar"] = value
    inverse_moe = {value: key for key, value in _GEMMA4_MOE.items()}
    hf_text = text_config(config)
    result: dict[str, np.ndarray] = {}
    for name, raw in flat.items():
        leaf = np.asarray(raw)
        parts = name.split(".")
        if parts[0].startswith("layers_") and tuple(parts[1:]) in inverse_moe:
            index = parts[0].removeprefix("layers_")
            tail = ".".join(inverse_moe[tuple(parts[1:])])
            target = f"model.decoder.layers.{index}.{tail}"
        elif len(parts) == 5 and parts[1:3] == ["moe", "experts"]:
            index = parts[0].removeprefix("layers_")
            stem = f"model.decoder.layers.{index}.experts."
            if parts[3] == "up_proj":
                continue
            if parts[3] == "gate_proj":
                up = np.asarray(flat[name.replace("gate_proj", "up_proj")])
                result[stem + "gate_up_proj"] = np.ascontiguousarray(
                    np.swapaxes(np.concatenate([leaf, up], axis=-1), -1, -2))
            elif parts[3] == "down_proj":
                result[stem + "down_proj"] = np.ascontiguousarray(np.swapaxes(leaf, -1, -2))
            else:
                raise ValueError(f"unknown expert parameter {name!r}")
            continue
        else:
            mapped = _hf_name(name, hf_text)
            if mapped is None:
                continue
            target = "model.decoder." + mapped.removeprefix("model.") if mapped.startswith("model.") else mapped
        result[target] = np.ascontiguousarray(leaf.T if parts[-1] == "kernel" else leaf)
        if parts[-1] == "layer_scalar":
            result[target.replace("model.decoder.", "model.encoder.language_model.")] = result[target]
    for name, leaf in _flatten(params["self_conditioning"]).items():
        module, kind = name.split(".")
        result[f"model.decoder.self_conditioning.{module}.weight"] = np.ascontiguousarray(
            np.asarray(leaf).T if kind == "kernel" else np.asarray(leaf))
    if "conditioner" in params:
        inverse = {value: key for key, value in _GEMMA4_VISION_TENSORS.items()}
        for name, raw in _flatten(params["conditioner"]["tower"]).items():
            parts = name.split(".")
            if tuple(parts) in inverse:
                target = inverse[tuple(parts)]
            elif parts[0].startswith("layers_"):
                index = parts[0].removeprefix("layers_")
                tail = parts[1:-1]
                if parts[-1] == "kernel":
                    tail = [*tail, "linear"]
                target = f"encoder.layers.{index}." + ".".join([*tail, "weight"])
            else:
                raise ValueError(f"unknown vision parameter {name!r}")
            leaf = np.asarray(raw)
            result["model.encoder.vision_tower." + target] = np.ascontiguousarray(
                leaf.T if parts[-1] == "kernel" else leaf)
        result["model.encoder.embed_vision.embedding_projection.weight"] = np.ascontiguousarray(
            np.asarray(params["conditioner"]["projector"]["projection"]["kernel"]).T)
    return result
