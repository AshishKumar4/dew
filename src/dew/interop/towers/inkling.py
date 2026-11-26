"""Inkling 2's vision tower and projector as its release stores them: their tensors' names and their config's
fields (`dew.interop.towers.common`), for the models `dew.nn.vision.inkling` computes; read from the release
alone (thinkingmachines/Inkling-2-Mini), as its decoder is (`dew.interop.families.inkling`)."""

from collections.abc import Callable, Mapping
from types import MappingProxyType

import numpy as np

from dew import _records
from dew.interop.config_records import NativeFields
from dew.interop.layouts import ConfigField, ConfigLayout, ConfigPart, Route, Table, mapped
from dew.interop.unread import Stored, held
from dew.interop.weights import placed
from dew.nn.vision.inkling import InklingProjector, InklingVision
from dew.registry import Record, towers

from .common import ProjectorLayout, TowerLayout, _vision_section, block_routes, rope_base

_TENSORS = MappingProxyType({
    "embeddings.patch_embedding.weight": ("patch_embed", "kernel"),
    "embeddings.class_embedding": ("class_token",),
    "embeddings.position_embedding": ("position_table", "embedding"),
    "ln_pre.weight": ("pre_norm", "scale"), "ln_pre.bias": ("pre_norm", "bias"),
})
"""The trunk's tensors outside its blocks past `visual.backbone.`, as `_inkling_weights` reads them: the
class position joins the position table as its first row (`_CLASS_POSITION`)."""

_CLASS_POSITION = "embeddings.cls_position_embedding"
"""The class token's own position, which the tower holds as its position table's first row (`ViTLayout`'s
leading class token)."""

_BLOCK_MAPS = {("self_attn", "q_proj"): ("self_attn", "q_proj"),
               ("self_attn", "k_proj"): ("self_attn", "k_proj"),
               ("self_attn", "v_proj"): ("self_attn", "v_proj"),
               ("self_attn", "out_proj"): ("self_attn", "o_proj"),
               ("mlp", "c_fc"): ("mlp", "up_proj"), ("mlp", "c_proj"): ("mlp", "down_proj")}

_BLOCK_NORMS = {"ln_1": "input_layernorm", "ln_2": "post_attention_layernorm"}

_SCALES = {"ls_1": "attention_scale", "ls_2": "mlp_scale"}
"""Each branch's learned per-channel scale (`lambda_1`, `lambda_2` in the reference), as the block names
it."""

_ROUTES = Table(
    *(Route(name, "/".join(("params", *path))) for name, path in _TENSORS.items()),
    block_routes(_BLOCK_MAPS, prefix="layers", norms=_BLOCK_NORMS),
    Route("layers.{index}.{scale}.gamma", "params/blocks_{index}/{scale}",
          MappingProxyType({"scale": mapped(_SCALES)})))
"""An Inkling 2 trunk's tensor names past `visual.backbone.`, both ways."""


def _inkling_weights(hf_tensors: Mapping[str, Stored], path_of: Callable[[str], tuple[str, ...] | None],
                     param_dtype: str = "float32"):
    """The trunk's variables (`placed`), the patch map read in the tower's pixel order and the class
    position the position table's first row: the reference flattens each patch's rows, columns and channels
    (Inkling2VisionEmbeddings: `[rows, columns, channels]`), the tower each channel's rows and columns
    (`ViTLayout`), and adds the class position to the class token apart from the table, which the tower
    reads from the table's first row."""
    tensors = dict(hf_tensors)
    patch = "embeddings.patch_embedding.weight"
    if patch in tensors:
        weight = held(tensors[patch])
        width, pixels = weight.shape
        side = round((pixels // 3) ** 0.5)
        if 3 * side * side != pixels:
            raise ValueError(f"{patch} holds {weight.shape}, not a square patch of three channels")
        tensors[patch] = held(np.asarray(weight).reshape(width, side, side, 3).transpose(0, 3, 1, 2)
                              .reshape(width, pixels), contiguous=True)
    if _CLASS_POSITION in tensors:
        table = "embeddings.position_embedding"
        if table not in tensors:
            raise ValueError(f"{_CLASS_POSITION} is the position table's first row, beside {table}")
        tensors[table] = held(np.concatenate([np.asarray(held(tensors.pop(_CLASS_POSITION)))[None],
                                              np.asarray(held(tensors[table]))]), contiguous=True)
    return placed(tensors, path_of, param_dtype)


def _read_rope(vision: Mapping[str, object], used: set[str],
               record: NativeFields[InklingVision]) -> NativeFields[InklingVision]:
    """The axial rotary's base: Inkling 2's trunk turns no other (Inkling2VisionRotaryEmbedding)."""
    used.update(("rope_parameters", "rope_theta"))
    rope = _records.record(vision.get("rope_parameters") or {"rope_type": "axial"}, "rope_parameters")
    if rope.get("rope_type", "axial") != "axial":
        raise ValueError(f"Inkling 2's trunk turns its axial rotary alone, not {rope.get('rope_type')!r}")
    record["rope_theta"] = rope_base({**vision, "rope_parameters": rope}, InklingVision.rope_theta,
                                     axial=True)
    return record


def _write_rope(tower: InklingVision) -> Mapping[str, object]:
    return {"rope_parameters": {"rope_type": "axial", "rope_theta": tower.rope_theta}}


def _integer(key: str, field: str | None = None) -> ConfigField:
    """A key the release states, the tower's default where it does not."""
    return ConfigField(key, field or key, getattr(InklingVision, field or key), parse=_records.integer)


def _mlp_width(vision: Mapping[str, object], used: set[str],
               record: NativeFields[InklingVision]) -> NativeFields[InklingVision]:
    """The feed-forward width: `intermediate_size`, else `mlp_ratio` times the width (the config class's
    rule)."""
    used.update(("intermediate_size", "mlp_ratio"))
    width = _records.integer(vision.get("hidden_size", InklingVision.hidden_size), "hidden_size")
    stated = vision.get("intermediate_size")
    ratio = _records.number(vision.get("mlp_ratio", 4.0), "mlp_ratio")
    record["intermediate_size"] = (_records.integer(stated, "intermediate_size") if stated is not None
                                   else int(width * ratio))
    return record


_CONFIG = ConfigLayout(
    _integer("hidden_size"), _integer("num_hidden_layers", "depth"),
    _integer("num_attention_heads", "num_heads"),
    _integer("num_channels", "in_channels"), _integer("native_patch_size", "patch_size"),
    _integer("adapter_merge_size", "spatial_merge_size"), _integer("posemb_grid_size"),
    ConfigField("layer_norm_eps", "layer_norm_eps", InklingVision.layer_norm_eps, parse=_records.number),
    ConfigField("layer_scale_init_value", "layer_scale_init_value", InklingVision.layer_scale_init_value,
                parse=_records.number),
    ConfigPart(_mlp_width, lambda tower: {"intermediate_size": tower.intermediate_size}),
    ConfigPart(_read_rope, _write_rope))
"""An Inkling2VisionConfig's fields as the release spells them (`native_patch_size`, `adapter_merge_size`,
the config class's aliases of `patch_size` and `spatial_merge_size`)."""

_INERT = frozenset({"model_type", "temporal_patch_size", "hidden_act", "projection_hidden_dim",
                    "projector_bias", "text_hidden_size", "initializer_range", "attention_dropout"})
"""Fields the trunk computes nothing by: its projector's (read by `translate_inkling_projector_config`), the
release's `temporal_patch_size`, which the reference's patch map never reads (a still image's patch, one
frame), its activation (the exact GELU it states alone) and its training-time fields."""


def translate_inkling_vision_config(hf_config: Mapping[str, object]) -> Record:
    """An Inkling2VisionConfig into an InklingVision value's fields, refusing a field it states that no
    field reads, and an activation other than the exact GELU its trunk runs."""
    vision = _vision_section(hf_config)
    if vision.get("model_type", "inkling2_vision") != "inkling2_vision":
        raise ValueError(f"vision model_type {vision.get('model_type')!r} is not Inkling 2's trunk")
    if vision.get("hidden_act", "gelu") != "gelu":
        raise ValueError(f"Inkling 2's trunk runs the exact GELU, not {vision.get('hidden_act')!r}")
    used: set[str] = set()
    fields = _CONFIG.read(vision, used, NativeFields(InklingVision, {}))
    unread = sorted(set(vision) - used - _INERT)
    if unread:
        raise ValueError(f"Inkling 2's vision_config states {unread}, which its trunk computes nothing by")
    record: Record = {"class": "inkling2", "fields": fields}
    towers.from_record(record)
    return record


def translate_inkling_projector_config(hf_config: Mapping[str, object], text_width: int) -> Record:
    """Inkling 2's merger fields: each merge block's patches joined, `projection_hidden_dim` wide between
    two maps, biased where `projector_bias`, into the decoder's `text_width`, which `text_hidden_size`
    states."""
    vision = _vision_section(hf_config)
    stated = _records.integer(vision.get("text_hidden_size", text_width), "text_hidden_size")
    if stated != int(text_width):
        raise ValueError(f"text_hidden_size ({stated}) is not the decoder width ({text_width})")
    width = _records.integer(vision.get("hidden_size", InklingVision.hidden_size), "hidden_size")
    merge = _records.integer(vision.get("adapter_merge_size", InklingVision.spatial_merge_size),
                             "adapter_merge_size")
    return {"class": "inkling2", "fields": {
        "hidden_size": width, "spatial_merge_size": merge,
        "width": _records.integer(vision.get("projection_hidden_dim", 4096), "projection_hidden_dim"),
        "out_width": int(text_width), "norm": False, "activation": "gelu",
        "bias": bool(vision.get("projector_bias", False))}}


INKLING_VISION = TowerLayout("visual.backbone.", _ROUTES, read=_inkling_weights, config=_CONFIG,
                             model_type="inkling2_vision")

INKLING_PROJECTOR = ProjectorLayout("visual.proj_", {
    "in.weight": ("fc1", "kernel"), "in.bias": ("fc1", "bias"),
    "out.weight": ("fc2", "kernel"), "out.bias": ("fc2", "bias")})
"""The merger's two maps (`visual.proj_in`, `visual.proj_out`), apart from the trunk under
`visual.backbone.`, which a bundle routes after its projector's prefix (`hf_decoders.wrapper_route`)."""

TOWERS = {InklingVision: INKLING_VISION}
"""Where Inkling 2's tower tensors sit in its release (`dew.interop.towers.tower_layout`)."""

PROJECTORS = {InklingProjector: INKLING_PROJECTOR}
"""Where Inkling 2's projector tensors sit in its release (`dew.interop.towers.projector_layout`)."""
