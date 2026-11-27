"""Inkling 2's vision tower and projector as its release stores them: their tensors' names and their config's
fields (`dew.interop.towers.common`), for the models `dew.nn.vision.inkling` computes; read from the release
alone (thinkingmachines/Inkling-2-Mini), as its decoder is (`dew.interop.families.inkling`)."""

from collections.abc import Callable, Mapping
from types import MappingProxyType

import jax
import numpy as np

from dew import _records
from dew.interop.config_records import NativeFields
from dew.interop.layouts import ConfigField, ConfigLayout, ConfigPart, Route, Table, mapped
from dew.interop.safetensors_io import LazyTensors, tensor_specs
from dew.interop.unread import Stored, held
from dew.interop.weights import placed
from dew.nn.vision.inkling import InklingProjector, InklingVision
from dew.registry import Record, towers

from .common import (
    ProjectorLayout,
    TowerLayout,
    _vision_section,
    axial_rope_part,
    block_routes,
    integer_field,
    number_field,
)

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


def _inkling_unread(tensors: Mapping[str, Stored], prefix: str) -> LazyTensors:
    """`_inkling_weights`' rewrites undone (`TowerLayout.unread`), each tensor built as it is read, by each
    tensor's name past `prefix` (its local name), every spelling of one name alike (a checked alias, `model.`
    put on or taken off): the patch map back in the reference's pixel order, and where the table holds the
    class position (its source stores one, so the table's tensors are beside it), the class position the
    table's first row and the table the rest. A class position beside no table is refused."""
    specs = tensor_specs(tensors)
    spellings: dict[str, list[str]] = {}
    for name in specs:
        if prefix in name:
            spellings.setdefault(name[name.index(prefix) + len(prefix):], []).append(name)
    built: dict[str, tuple[jax.ShapeDtypeStruct, Callable[[], np.ndarray]]] = {
        name: (spec, lambda name=name: np.asarray(tensors[name])) for name, spec in specs.items()}
    for name in spellings.get("embeddings.patch_embedding.weight", []):
        width, pixels = specs[name].shape
        side = round((pixels // 3) ** 0.5)
        built[name] = (specs[name], lambda name=name, width=width, side=side: np.ascontiguousarray(
            np.asarray(tensors[name]).reshape(width, 3, side, side).transpose(0, 2, 3, 1).reshape(width, -1)))
    if _CLASS_POSITION in spellings:
        tables = spellings.get("embeddings.position_embedding", [])
        if not tables:
            raise ValueError(f"{spellings[_CLASS_POSITION]} hold the position table's first row, beside no "
                             "embeddings.position_embedding")
        rows, first = specs[tables[0]], tables[0]
        for name in spellings[_CLASS_POSITION]:
            built[name] = (jax.ShapeDtypeStruct(rows.shape[1:], rows.dtype),
                           lambda: np.asarray(tensors[first])[0])
        for table in tables:
            built[table] = (jax.ShapeDtypeStruct((rows.shape[0] - 1, *rows.shape[1:]), rows.dtype),
                            lambda table=table: np.asarray(tensors[table])[1:])
    return LazyTensors({name: spec for name, (spec, _) in built.items()}, lambda name: built[name][1]())


def _mlp_width(vision: Mapping[str, object], used: set[str],
               record: NativeFields[InklingVision]) -> NativeFields[InklingVision]:
    """The feed-forward width: `intermediate_size`, else `mlp_ratio` times the width (the config class's
    rule)."""
    used.update(("intermediate_size", "mlp_ratio"))
    width = _records.integer(vision.get("hidden_size", 1024), "hidden_size")
    stated = vision.get("intermediate_size")
    ratio = _records.number(vision.get("mlp_ratio", 4.0), "mlp_ratio")
    record["intermediate_size"] = (_records.integer(stated, "intermediate_size") if stated is not None
                                   else int(width * ratio))
    return record


_CONFIG = ConfigLayout(
    integer_field("hidden_size", default=1024), integer_field("num_hidden_layers", "depth", 23),
    integer_field("num_attention_heads", "num_heads", 16), integer_field("num_channels", "in_channels", 3),
    ConfigField("native_patch_size", "patch_size", 20, parse=_records.integer, aliases=("patch_size",)),
    ConfigField("adapter_merge_size", "spatial_merge_size", 2, parse=_records.integer,
                aliases=("spatial_merge_size",)),
    integer_field("posemb_grid_size", default=32), number_field("layer_norm_eps", default=1e-5),
    number_field("layer_scale_init_value", default=0.1),
    ConfigPart(_mlp_width, lambda tower: {"intermediate_size": tower.intermediate_size}),
    axial_rope_part(InklingVision.rope_theta, "Inkling 2"))
"""An Inkling2VisionConfig's fields and its defaults (configuration_inkling2.py), as the release spells
them (`native_patch_size`, `adapter_merge_size`), or as transformers writes them (`patch_size`,
`spatial_merge_size`, the config class's names for those)."""

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
    width = _records.integer(vision.get("hidden_size", 1024), "hidden_size")
    merge = _records.integer(vision.get("adapter_merge_size", vision.get("spatial_merge_size", 2)),
                             "adapter_merge_size")
    return {"class": "inkling2", "fields": {
        "hidden_size": width, "spatial_merge_size": merge,
        "width": _records.integer(vision.get("projection_hidden_dim", 4096), "projection_hidden_dim"),
        "out_width": int(text_width), "norm": False, "activation": "gelu",
        "bias": bool(vision.get("projector_bias", False))}}


INKLING_VISION = TowerLayout("visual.backbone.", _ROUTES, read=_inkling_weights, unread=_inkling_unread,
                             config=_CONFIG, model_type="inkling2_vision")

INKLING_PROJECTOR = ProjectorLayout("visual.proj_", {
    "in.weight": ("fc1", "kernel"), "in.bias": ("fc1", "bias"),
    "out.weight": ("fc2", "kernel"), "out.bias": ("fc2", "bias")})
"""The merger's two maps (`visual.proj_in`, `visual.proj_out`), apart from the trunk under
`visual.backbone.`, which a bundle routes after its projector's prefix (`hf_decoders.wrapper_route`)."""

TOWERS = {InklingVision: INKLING_VISION}
"""Where Inkling 2's tower tensors sit in its release (`dew.interop.towers.tower_layout`)."""

PROJECTORS = {InklingProjector: INKLING_PROJECTOR}
"""Where Inkling 2's projector tensors sit in its release (`dew.interop.towers.projector_layout`)."""
