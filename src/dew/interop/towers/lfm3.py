"""LFM3-VL's vision tower and projector as a release stores them: their tensors' names and their
config's fields, both ways (`dew.interop.towers.common`), for the models `dew.nn.vision.lfm3` computes."""

from collections.abc import Mapping
from types import MappingProxyType

from dew import _records
from dew.interop.config_records import NativeFields
from dew.interop.layouts import ConfigField, ConfigLayout, ConfigPart, Route, Table, mapped
from dew.nn.activations import HIDDEN_ACT
from dew.nn.vision.lfm3 import Lfm3VlProjector, Lfm3VlVision
from dew.registry import Record, towers

from .common import (
    ProjectorLayout,
    TowerLayout,
    _vision_section,
    activation_name,
    block_routes,
    rope_base,
    tensor_routes,
)

_TENSORS = {
    "patchifier.weight": ("patch_embed", "kernel"),
    "embedding_norm.weight": ("pre_norm", "scale"),
}


_BLOCK_MAPS = {("self_attn", "q_proj"): ("self_attn", "q_proj"),
               ("self_attn", "k_proj"): ("self_attn", "k_proj"),
               ("self_attn", "v_proj"): ("self_attn", "v_proj"),
               ("self_attn", "out_proj"): ("self_attn", "o_proj"),
               ("conv", "in_proj"): ("self_attn", "in_proj"), ("conv", "out_proj"): ("self_attn", "out_proj"),
               ("conv", "spatial_conv"): ("self_attn", "spatial_conv"),
               ("feed_forward", "w1"): ("mlp", "gate_proj"), ("feed_forward", "w3"): ("mlp", "up_proj"),
               ("feed_forward", "w2"): ("mlp", "down_proj")}


_BLOCK_NORMS = {"operator_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}


_HEAD_NORMS = {"q_layernorm": "q_norm", "k_layernorm": "k_norm"}


_LFM3_ROUTES = Table(
    *tensor_routes(_TENSORS),
    block_routes(_BLOCK_MAPS, prefix="layers", norm_bias=False, norms=_BLOCK_NORMS),
    Route("layers.{index}.self_attn.{norm}.weight", "params/blocks_{index}/self_attn/{norm}/scale",
          MappingProxyType({"norm": mapped(_HEAD_NORMS)})))
"""An LFM3-VL trunk's tensor names past `vision_tower.`, both ways."""


_INTEGERS = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
             "num_key_value_heads", "patch_size", "temporal_patch_size", "in_channels", "spatial_merge_size",
             "conv_kernel_size")
"""The Lfm3VlVision fields an Lfm3VlVisionConfig states as integers under the same names."""


def _stated(key: str, parse) -> ConfigField:
    """A key the config states or leaves null, the value's own default where it is null or absent."""
    default = getattr(Lfm3VlVision, key)
    return ConfigField(key, key, None,
                       parse=lambda value, key: default if value is None else parse(value, key))


def _read_rope(vision: Mapping[str, object], used: set[str],
               record: NativeFields[Lfm3VlVision]) -> NativeFields[Lfm3VlVision]:
    """The axial rotary's base: LFM3-VL's trunk turns no other."""
    used.update(("rope_parameters", "rope_theta"))
    rope = _records.record(vision.get("rope_parameters") or {"rope_type": "axial"}, "rope_parameters")
    if rope.get("rope_type", "axial") != "axial":
        raise ValueError(f"LFM3-VL's trunk turns its axial rotary alone, not {rope.get('rope_type')!r}")
    record["rope_theta"] = rope_base({**vision, "rope_parameters": rope}, Lfm3VlVision.rope_theta, axial=True)
    return record


def _write_rope(tower: Lfm3VlVision) -> Mapping[str, object]:
    return {"rope_parameters": {"rope_type": "axial", "rope_theta": tower.rope_theta}}


_LFM3_CONFIG = ConfigLayout(
    *(_stated(name, _records.integer) for name in _INTEGERS), _stated("norm_eps", _records.number),
    ConfigField("layer_types", "layer_types", None,
                parse=lambda value, key: _records.strings(value, key) if value else None),
    ConfigPart(_read_rope, _write_rope))
"""An Lfm3VlVisionConfig's fields, both ways: those it states, the value's own defaults for the rest
(its cadence over the depth among them, `Lfm3VlVision.kinds`)."""


def translate_lfm3_vl_vision_config(hf_config: Mapping[str, object]) -> Record:
    """An Lfm3VlVisionConfig into an Lfm3VlVision value's fields: those it states, checked, the value's
    own defaults for the rest (its cadence over the depth among them, `Lfm3VlVision.kinds`); the rotary
    is the axial one over the patch grid alone, as the release's Lfm3VlVisionRotaryEmbedding runs no
    other."""
    vision = _vision_section(hf_config)
    if vision.get("model_type", "lfm3_vl_vision") != "lfm3_vl_vision":
        raise ValueError(f"vision model_type {vision.get('model_type')!r} is not LFM3-VL's trunk")
    fields = _LFM3_CONFIG.read(vision, set(), NativeFields(Lfm3VlVision, {}))
    record: Record = {"class": "lfm3_vl", "fields": fields}
    # Built here, so a config the value refuses (a cadence of another depth, an unknown kind) is refused
    # as it is read.
    towers.from_record(record)
    return record


def translate_lfm3_vl_projector_config(hf_config: Mapping[str, object], text_width: int) -> Record:
    """An Lfm3VlConfig's projector fields: each `downsample_factor` square of trunk tokens, which the
    config holds to the trunk's merge blocks, mapped to `projector_hidden_size` and to the decoder's
    `text_width`."""
    vision = _vision_section(hf_config)
    merge = _records.integer(hf_config.get("downsample_factor", 2), "downsample_factor")
    if merge != vision.get("spatial_merge_size", 2):
        raise ValueError(f"downsample_factor ({merge}) is not the trunk's spatial_merge_size "
                         f"({vision.get('spatial_merge_size', 2)}), as Lfm3VlConfig requires")
    return {"class": "lfm3_vl", "fields": {
        "hidden_size": _records.integer(vision.get("hidden_size", 1024), "hidden_size"),
        "spatial_merge_size": merge,
        "width": _records.integer(hf_config.get("projector_hidden_size", 4096), "projector_hidden_size"),
        "out_width": int(text_width),
        "bias": _records.boolean(hf_config.get("projector_bias", True), "projector_bias"),
        "norm_bias": True,
        "norm_eps": _records.number(hf_config.get("projector_layer_norm_eps", 1e-6),
                                    "projector_layer_norm_eps"),
        "activation": activation_name(hf_config, HIDDEN_ACT, "gelu", ("projector_hidden_act",)),
    }}


LFM3_VL_VISION = TowerLayout("vision_tower.", _LFM3_ROUTES, config=_LFM3_CONFIG,
                                      model_type="lfm3_vl_vision")


LFM3_VL_PROJECTOR = ProjectorLayout("multi_modal_projector.", {
    "layer_norm.weight": ("norm", "scale"), "layer_norm.bias": ("norm", "bias"),
    "linear_1.weight": ("fc1", "kernel"), "linear_1.bias": ("fc1", "bias"),
    "linear_2.weight": ("fc2", "kernel"), "linear_2.bias": ("fc2", "bias"),
})

TOWERS = {Lfm3VlVision: LFM3_VL_VISION}
"""Where each of these tower kinds' tensors sit in a media checkpoint (`dew.interop.towers.tower_layout`)."""


PROJECTORS = {Lfm3VlProjector: LFM3_VL_PROJECTOR}
"""Where each of these projector kinds' tensors sit in a media checkpoint
(`dew.interop.towers.projector_layout`)."""
