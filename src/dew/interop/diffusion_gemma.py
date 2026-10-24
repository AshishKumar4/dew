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
from dew.interop.hf_decoders import (
    NOTHING_CHECKED,
    Checked,
    export_config,
    refuse_lossy_export,
    translate_config,
    translate_denoiser_weights,
)
from dew.interop.safetensors_io import LazyTensors, tensor_specs
from dew.interop.streaming import WeightLayout
from dew.interop.weights import nested
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import layer_scalar_mode
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.multimodal import VisionConditioner
from dew.nn.vision import tower_variables as vision_variables
from dew.nn.vision.gemma4 import (
    Gemma4Projector,
    Gemma4Vision,
    export_gemma4_vision_config,
    gemma4_projector_layouts,
    gemma4_tower_layouts,
    translate_gemma4_projector_config,
    translate_gemma4_projector_weights,
    translate_gemma4_vision_config,
)
from dew.objectives.base import Variables
from dew.registry import from_record, projectors, towers

VISION_PREFIX = "model.encoder.vision_tower."
PROJECTION_PREFIX = "model.encoder.embed_vision."
"""Where a DiffusionGemma checkpoint keeps its Gemma 4 tower's and embedder's tensors: under its
encoder, by the names a Gemma 4 wrapper gives them (`Gemma4Vision.checkpoint`,
`Gemma4Projector.checkpoint`)."""


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
    lazy: bool = False,
    checked: Checked = NOTHING_CHECKED,
) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """Map a DiffusionGemma checkpoint's tensors into its variables tree, with the layouts its
    vision readers placed the tower's and the embedder's tensors by, under their source names, which
    `export_weights` writes them back through.

    Text weights route through the shared decoder map, vision and projection
    weights through their own. Each component casts its own leaves to
    `param_dtype` as it binds them, so no full FP32 tree is built and narrowed.
    """
    text: dict[str, np.ndarray] = {}
    vision: dict[str, np.ndarray] = {}
    projection: dict[str, np.ndarray] = {}
    for name, tensor in tensors.items():
        if name.startswith(VISION_PREFIX):
            vision[name.removeprefix(VISION_PREFIX)] = tensor
        elif name.startswith(PROJECTION_PREFIX):
            projection[name.removeprefix(PROJECTION_PREFIX)] = tensor
        else:
            text[name] = tensor
    fields = translate_config(text_config(config))
    mapped = translate_denoiser_weights(text, fields, param_dtype=param_dtype, lazy=lazy, checked=checked)
    params = {"text": mapped["text"]["params"],
              "self_conditioning": mapped["self_conditioning"]["params"]}
    variables = {"params": params}
    for collection, tree in mapped["text"].items():
        if collection != "params":
            variables[collection] = {"text": tree}
    layouts: tuple[WeightLayout, ...] = ()
    if config.get("vision_config") is not None:
        if not vision or not projection:
            raise ValueError("vision_config requires both vision_tower and embed_vision tensors")
        tower_variables, tower_layouts = vision_variables("gemma4", vision, param_dtype)
        projector, projector_layouts = translate_gemma4_projector_weights(projection, param_dtype=param_dtype)
        params["conditioner"] = {"tower": tower_variables["params"], "projector": projector}
        if "constants" in tower_variables:
            variables.setdefault("constants", {})["conditioner"] = {"tower": tower_variables["constants"]}
        layouts = _named(VISION_PREFIX, nested(tower_layouts, 1, "conditioner", "tower")) + _named(
            PROJECTION_PREFIX, nested(projector_layouts, 0, "params", "conditioner", "projector"))
    elif vision or projection:
        raise ValueError("vision weights require vision_config")
    return variables, layouts


def _named(prefix: str, layouts: tuple[WeightLayout, ...]) -> tuple[WeightLayout, ...]:
    """`layouts` under the checkpoint names their component's tensors carry below `prefix`."""
    return tuple(dataclasses.replace(layout, name=prefix + layout.name) for layout in layouts)


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


def generation_config_of(process: BlockProcess, *, max_new_tokens: int | None = None,
                         pad_token_id: int | None = None) -> dict[str, records.JSON]:
    """The generation_config.json transformers' DiffusionGemmaGenerationConfig
    reads, and `generation_process` reads back as `process`: its inverse. The
    canvas and vocabulary are the model's in both, as a block run's process
    takes them from its model (`BlockDiffusionObjective`)."""
    # Floats where BlockProcess also takes an int, as transformers' reader
    # requires (generation_diffusion_gemma.py:181-183, 333).
    sampler: dict[str, records.JSON] = {"_cls_name": "EntropyBoundSamplerConfig",
                                        "entropy_bound": float(process.entropy_bound)}
    values: dict[str, records.JSON] = {
        "max_denoising_steps": process.max_steps, "t_min": float(process.t_min),
        "t_max": float(process.t_max), "stability_threshold": process.stability_threshold,
        "confidence_threshold": float(process.confidence_threshold), "sampler_config": sampler,
    }
    if max_new_tokens:
        values["max_new_tokens"] = max_new_tokens
    if pad_token_id is not None:
        values["pad_token_id"] = pad_token_id
    return values


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
    mode = layer_scalar_mode(variables, ("layers_0", "layer_scalar"))
    return text if mode in (None, text.layer_scalar) else text.clone(layer_scalar=mode)


# Gemma 4 text fields DiffusionGemma's text config has none of: its modules
# fix them (keys read as values on full layers, a routed mixture beside every
# dense MLP, the 30 softcap) or never build them (per-layer inputs, shared KV
# layers, the double-wide MLP).
_GEMMA4_ONLY = ("architectures", "use_cache", "attention_k_eq_v", "enable_moe_block",
                "final_logit_softcapping", "hidden_size_per_layer_input", "vocab_size_per_layer_input",
                "num_kv_shared_layers", "use_double_wide_mlp")


def published_config(model: DiffusionGemma) -> Mapping[str, object]:
    """The config transformers' DiffusionGemmaForBlockDiffusion reads for
    `model`: its text stack in diffusion_gemma_text's fields, its Gemma 4
    tower's vision_config and its canvas length.

    Dew places images by position and keeps no image token ids, so the config
    names none and transformers' defaults stand. A model the written fields
    would rebuild differently is refused, naming the fields that differ
    (`refuse_lossy_export`). The text record is the decoder view, as
    `causal` is fixed by the classes on both sides: Dew's text stack is the
    causal encoder view DiffusionGemma requires, and transformers' encoder
    attends causally unless use_bidirectional_attention is "all", which this
    config never writes, while its decoder never does
    (modeling_diffusion_gemma.py:281, :383, 5.16.1).
    """
    text = {name: value for name, value in export_config(model.text).items() if name not in _GEMMA4_ONLY}
    text.update(model_type="diffusion_gemma_text",
                use_bidirectional_attention=None if model.conditioner is None else "vision")
    refuse_lossy_export(model.text.clone(causal=False), text)
    config: dict[str, object] = {"model_type": "diffusion_gemma",
                                 "architectures": ["DiffusionGemmaForBlockDiffusion"],
                                 "text_config": text, "canvas_length": model.canvas_length,
                                 "tie_word_embeddings": model.text.tie_embeddings}
    if model.conditioner is not None:
        tower, projection = model.conditioner.vision, model.conditioner.projection
        # transformers builds the tower from vision_config and the projector
        # from the tower's epsilon and the text width.
        expected = Gemma4Projector(text_width=model.text.emb_features,
                                   norm_eps=tower.rms_norm_eps if isinstance(tower, Gemma4Vision) else 1e-6)
        if not (isinstance(tower, Gemma4Vision)
                and projection.clone(parent=None, name=None, dtype=None, precision=None) == expected):
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
    model: DiffusionGemma, variables: Variables, config: Mapping[str, object],
    vision: tuple[WeightLayout, ...] | None = None,
) -> LazyTensors:
    """Return the checkpoint tensors for a DiffusionGemma, keyed by source name.

    The decoder half goes through `export_decoder_weights` and is renamed under
    the reference's `model.decoder.` and `model.encoder.` prefixes. The tower
    and the embedder go back through `vision`, the layouts their readers placed
    a loaded checkpoint's tensors by (`translate_weights`); a model built in Dew
    has none, and writes them through the layouts its own tree gives. A config
    the published implementation cannot build is refused (`_refuse_unreadable`).
    """
    from dew.interop.decoder_parts import layout_tensors, leaf_layouts
    from dew.interop.hf_decoders import export_decoder_weights

    _refuse_unreadable(config)
    params = variables["params"]
    text_variables = {collection: tree["text"] for collection, tree in variables.items()
                      if "text" in tree}
    text = scalar_placement(model.text, text_variables)
    decoded = export_decoder_weights(text, text_variables, text_config(config))
    names = {}
    specs = {}
    for name, spec in tensor_specs(decoded).items():
        target = "model.decoder." + name.removeprefix("model.") if name.startswith("model.") else name
        names[target] = name
        specs[target] = spec
        if name.endswith(".layer_scalar"):
            encoder = target.replace("model.decoder.", "model.encoder.language_model.")
            names[encoder], specs[encoder] = name, spec
    leaves: dict[tuple[str, ...], object] = {}
    for path, leaf in flatten_dict(dict(params["self_conditioning"])).items():
        leaves[("params", "self_conditioning", *(str(part) for part in path))] = leaf
    layouts = leaf_layouts(leaves, lambda path: f"model.decoder.self_conditioning.{path[-2]}.weight")
    if "conditioner" in params:
        if vision is None:
            tower = {collection: tree["conditioner"]["tower"] for collection, tree in variables.items()
                     if "tower" in tree.get("conditioner", {})}
            projector = params["conditioner"]["projector"]
            vision = (_named(VISION_PREFIX, nested(gemma4_tower_layouts(tower), 1, "conditioner", "tower"))
                      + _named(PROJECTION_PREFIX, nested(gemma4_projector_layouts(projector),
                                                         1, "conditioner", "projector")))
        layouts.update({layout.name: layout for layout in vision})
    auxiliary = layout_tensors(layouts, variables)
    specs.update(tensor_specs(auxiliary))

    def read(name: str) -> np.ndarray:
        if name in names:
            return decoded[names[name]]
        return auxiliary[name]

    return LazyTensors(specs, read)


__all__ = ["build"]
