"""DiffusionGemma assembly for the shared pretrained-model loader.

Checkpoint text aliases collapse to one Flax subtree; vision and projection
weights retain their separate subtrees. This module opens no checkpoint or
processor and introduces no family-specific public loading entry point.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from flax import linen as nn

from dew import records
from dew.diffusion.block import BlockProcess
from dew.interop.hf_decoders import DecoderFields, translate_config, translate_denoiser_weights
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.multimodal import VisionConditioner
from dew.nn.vision import (
    projector_from_record,
    tower_from_record,
    translate_gemma4_projector_config,
    translate_gemma4_projector_weights,
    translate_gemma4_vision_config,
    translate_gemma4_vision_weights,
)
from dew.objectives.base import Variables
from dew.registry import models, precision_fields


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


def build(config: Mapping[str, object], *, dtype: str = "bfloat16",
          attention_impl: str = "auto", max_seq_len: int | None = None) -> DiffusionGemma:
    """Build native model values without allocating parameters."""
    fields = translate_config(text_config(config))
    fields["causal"] = True
    if max_seq_len is not None:
        fields["max_seq_len"] = max_seq_len
    precise: DecoderFields = {**fields, **precision_fields(
        "causal_transformer", fields, dtype=dtype, attention_impl=attention_impl)}
    text = models.build("causal_transformer", precise)
    if not isinstance(text, CausalTransformer):
        raise TypeError("the causal_transformer registry entry must build a CausalTransformer")
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


def translate_weights(
    tensors: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    *,
    param_dtype: str = "float32",
) -> Variables:
    """Map shared aliases and media weights with independent parameter storage.

    Each component casts parameters at its binding site; frozen state remains
    native FP32. No completed FP32 variables tree is narrowed afterward.
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
        tower_variables = translate_gemma4_vision_weights(vision, param_dtype=param_dtype)
        params["conditioner"] = {
            "tower": tower_variables["params"],
            "projector": translate_gemma4_projector_weights(projection, param_dtype=param_dtype)}
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


def scalar_placement(text: CausalTransformer, variables: Variables) -> CausalTransformer:
    """`text` under the layer-scalar policy the tree being written actually keeps.

    Google makes the per-layer skip scale a parameter and Transformers calls
    the same tensor a buffer, so a source declares which of the two it is and
    the export reads the collection that policy names. `BlockDiffusionObjective`
    trains the scalar, which moves every one of them into `params`, so the tree
    a finished SFT hands back disagrees with the source it was loaded from.
    The tree is what is being written, so the tree decides: exporting a run
    and saving the source it trained both come through here and agree.
    """
    if text.layer_scalar is None:
        return text
    for mode, collection in (("trainable", "params"), ("frozen", "constants")):
        held = variables.get(collection, {})
        if any(isinstance(node, Mapping) and "layer_scalar" in node for node in held.values()):
            return text if text.layer_scalar == mode else text.clone(layer_scalar=mode)
    return text


def export_weights(model: nn.Module, variables: Variables, config: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Write canonical decoder tensors under the scalar policy the tree keeps."""
    from dew.interop.hf_decoders import _flatten, export_decoder_weights
    from dew.nn.vision import _GEMMA4_VISION_TENSORS

    if not isinstance(model, DiffusionGemma):
        raise TypeError("DiffusionGemma export requires its native model value")
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
    for name, leaf in _flatten(params["self_conditioning"]).items():
        module, kind = name.split(".")
        tensors[f"model.decoder.self_conditioning.{module}.weight"] = np.ascontiguousarray(
            np.asarray(leaf).T if kind == "kernel" else np.asarray(leaf))
    if "conditioner" in params:
        inverse = {value: key for key, value in _GEMMA4_VISION_TENSORS.items()}
        for collection in ("params", "constants"):
            tower = variables.get(collection, {}).get("conditioner", {}).get("tower", {})
            for name, raw in _flatten(tower).items():
                parts = name.split(".")
                if tuple(parts) in inverse:
                    target = inverse[tuple(parts)]
                elif parts[0].startswith("layers_"):
                    index = parts[0].removeprefix("layers_")
                    tail = parts[1:-1]
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
