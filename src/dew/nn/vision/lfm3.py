"""LFM3-VL's vision trunk and projector, with their checkpoint maps (LiquidAI/LFM3-VL-2B@f83d2b0,
`modeling_lfm3_vl.py`: Lfm3VlVisionModel and Lfm3VlMultiModalProjector).
"""

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

from flax import linen as nn

from dew import _records
from dew.interop.config_records import NativeFields
from dew.interop.layouts import ConfigField, ConfigLayout, ConfigPart, Route, Table, mapped
from dew.nn.activations import HIDDEN_ACT
from dew.registry import Record, towers

from .common import (
    ProjectorBase,
    ProjectorLayout,
    TowerBase,
    TowerGeometry,
    TowerLayout,
    _vision_section,
    activation_name,
    block_routes,
    head_width,
    rope_base,
    tensor_routes,
)
from .vit import MergeProjector, VisionTransformer, ViTLayout


def default_layer_types(layers: int) -> tuple[str, ...]:
    """LFM3-VL's cadence over `layers` layers: two convolutions, then attention and convolution in turn
    (Lfm3VlVisionConfig's default `layer_types`)."""
    return tuple("conv" if i < 2 or i % 2 else "full_attention" for i in range(layers))


@dataclasses.dataclass(frozen=True)
class Lfm3VlVision(TowerBase):
    """LFM3-VL's trunk geometry under its vision_config's names.

    Its layout (`ViTLayout`): a bias-free map of each patch's channel-then-time pixels in merge-block
    order (`patchifier`), an RMS norm after it (`embedding_norm`), and pre-norm blocks of RMS norms at
    `norm_eps` with a bias-free SwiGLU feed-forward (`w1` its gate, `w3` its up map), each mixing its
    frame's patches by full attention, grouped and unbiased with an RMS norm on each head's queries and
    keys before the rotary turns them by the patch grid, rows then columns (`full_attention`), or by a
    gated depthwise convolution over the frame's grid (`conv`, `GatedGridConv`); no norm after the
    blocks. Every RMS norm weighs its input after its cast back, as the release's `Lfm3VlRMSNorm` does.
    `layer_types` None is the release's cadence over its depth (`default_layer_types`).
    """

    hidden_size: int = 1024
    intermediate_size: int = 2560
    num_hidden_layers: int = 14
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    conv_kernel_size: int = 3
    norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    layer_types: tuple[str, ...] | None = None

    def __post_init__(self):
        unknown = sorted(set(self.kinds) - {"conv", "full_attention"})
        if unknown:
            raise ValueError(f"layer_types {unknown} are neither conv nor full_attention")
        if len(self.kinds) != self.num_hidden_layers:
            raise ValueError(f"layer_types names {len(self.kinds)} layers of {self.num_hidden_layers}")

    @property
    def kinds(self) -> tuple[str, ...]:
        """Each layer's kind: `layer_types`, or the release's cadence over the depth."""
        return default_layer_types(self.num_hidden_layers) if self.layer_types is None else self.layer_types

    def layout(self) -> ViTLayout:
        depth = self.num_hidden_layers
        return ViTLayout(
            depth=depth, hidden_size=self.hidden_size, num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            head_dim=head_width(self.hidden_size, self.num_attention_heads),
            intermediate_size=self.intermediate_size, activation="swiglu", in_channels=self.in_channels,
            patch_size=self.patch_size, temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.spatial_merge_size, norm_type="rms", norm_eps=self.norm_eps,
            norm_bias=False, mlp_bias=False, packed=False, patch_bias=False,
            attention=tuple("conv" if kind == "conv" else "full" for kind in self.kinds),
            order=("patches",) * depth, sinks=(False,) * depth, pre_norm="rms", outer_norm_eps=self.norm_eps,
            rope_theta=self.rope_theta, attention_bias=False, qk_norm=True, conv_kernel=self.conv_kernel_size,
            scale_after_cast=True)

    def build(self) -> nn.Module:
        return VisionTransformer(self.layout())

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.spatial_merge_size,
                             channels=self.in_channels)


class Lfm3VlProjector(MergeProjector, ProjectorBase):
    """LFM3-VL's projector (Lfm3VlMultiModalProjector) is `MergeProjector`: a layer norm of each
    trunk token at `projector_layer_norm_eps`, its bias kept whatever `projector_bias` says, each merge
    block's tokens joined unit by unit, and two maps, biased by `projector_bias`, around the
    `projector_hidden_act` activation."""


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


Lfm3VlVision.checkpoint = TowerLayout("vision_tower.", _LFM3_ROUTES, config=_LFM3_CONFIG,
                                      model_type="lfm3_vl_vision")
Lfm3VlProjector.checkpoint = ProjectorLayout("multi_modal_projector.", {
    "layer_norm.weight": ("norm", "scale"), "layer_norm.bias": ("norm", "bias"),
    "linear_1.weight": ("fc1", "kernel"), "linear_1.bias": ("fc1", "bias"),
    "linear_2.weight": ("fc2", "kernel"), "linear_2.bias": ("fc2", "bias"),
})
