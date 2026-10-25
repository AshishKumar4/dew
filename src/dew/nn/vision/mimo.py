"""MiMo-V2.6's vision trunk (MiMo ViT) and its merger, with their checkpoint maps
(XiaomiMiMo/MiMo-V2.6-Pro-MOPD's `modeling_mimo_v2.py`, MiMoVisionTransformer).
"""

import dataclasses
from collections.abc import Mapping

from flax import linen as nn

from dew import records
from dew.interop.weights import in_params
from dew.registry import Record

from .common import (
    ProjectorBase,
    ProjectorLayout,
    TowerBase,
    TowerGeometry,
    TowerLayout,
    _vision_section,
    activation_name,
    block_tensor_path,
    merger_projector,
)
from .vit import MergeProjector, VisionTransformer, ViTLayout, translate_vit_weights


@dataclasses.dataclass(frozen=True)
class MiMoVision(TowerBase):
    """MiMo ViT's geometry under its vision_config's names.

    Its layout (`ViTLayout`): a bias-free patch map and no position table; RMS-normed blocks whose
    grouped queries, keys and values come from one biased map, rotated by the patch grid as Qwen 3.5's
    are, then a biased SwiGLU. Each layer of `fullatt_block_indexes` attends over its whole frame,
    every other one over the `visual_token_window_size` tokens either side of each query, plus a
    learned logit per head on each frame's first key where `use_sink`; a layer whose
    `vit_window_attn_types` entry is 1 runs with the merge blocks taken column by column.
    """

    depth: int = 28
    hidden_size: int = 1280
    intermediate_size: int = 4608
    num_heads: int = 32
    num_key_value_heads: int = 8
    qk_channels: int = 64
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    rms_norm_eps: float = 1e-6
    fullatt_block_indexes: tuple[int, ...] = (0, 9, 18, 27)
    vit_window_attn_types: tuple[int, ...] = (-1, 0, 0, 0, 0, 1, 1, 1, 1) * 3 + (-1,)
    use_sink: bool = True
    visual_token_window_size: int = 64

    def __post_init__(self):
        if len(self.vit_window_attn_types) != self.depth:
            raise ValueError(f"vit_window_attn_types names {len(self.vit_window_attn_types)} layers of "
                             f"{self.depth}")
        if self.vit_window_attn_types[-1] == 1:
            raise ValueError("the last layer runs column by column, so the reference merges column-ordered "
                             "tokens; the tower returns patches in their merge blocks' order")

    def layout(self) -> ViTLayout:
        full = set(self.fullatt_block_indexes)
        banded = self.visual_token_window_size > 0
        return ViTLayout(
            depth=self.depth, hidden_size=self.hidden_size, num_heads=self.num_heads,
            num_kv_heads=self.num_key_value_heads, head_dim=self.qk_channels,
            intermediate_size=self.intermediate_size, activation="swiglu", in_channels=self.in_channels,
            patch_size=self.patch_size, temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.spatial_merge_size, norm_type="rms", norm_eps=self.rms_norm_eps,
            norm_bias=False, mlp_bias=True, packed=True, patch_bias=False,
            attention=tuple("full" if i in full or not banded else "band" for i in range(self.depth)),
            order=tuple("columns" if kind == 1 else "patches" for kind in self.vit_window_attn_types),
            sinks=tuple(self.use_sink and i not in full for i in range(self.depth)),
            band=self.visual_token_window_size)

    def build(self) -> nn.Module:
        return VisionTransformer(self.layout())

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.spatial_merge_size,
                             channels=self.in_channels)


class MiMoProjector(MergeProjector, ProjectorBase):
    """MiMo's merger (MiMoVisionPatchMerger): a layer norm, each merge block grouped, and two maps
    around an exact GELU to the decoder width. The release stores none of its biases."""

    bias: bool = False

_BLOCK_MAPS = {("attn", "qkv"): ("self_attn", "qkv_proj"), ("attn", "proj"): ("self_attn", "o_proj"),
               ("mlp", "gate_proj"): ("mlp", "gate_proj"), ("mlp", "up_proj"): ("mlp", "up_proj"),
               ("mlp", "down_proj"): ("mlp", "down_proj")}


def mimo_vision_path(hf_name: str) -> tuple[str, ...] | None:
    """One MiMo ViT tensor name, its `visual.` prefix off, into its path in the trunk tree. The
    merger maps with the projector; anything else unknown raises ValueError."""
    parts = hf_name.split(".")
    if parts[0] == "merger":
        return None
    if len(parts) == 4 and parts[0] == "blocks" and parts[1].isdigit() and parts[2:] == ["attn", "sinks"]:
        return (f"sinks_{parts[1]}",)
    path = block_tensor_path(parts, _BLOCK_MAPS, norm_bias=False)
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def translate_mimo_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A MiMo-V2 vision_config into a MiMoVision value's fields. The trunk computes SwiGLU, as the
    release's silu names it, and no other activation."""
    vision = _vision_section(hf_config)
    activation_name(vision, ("silu",), "silu")
    depth = records.integer(vision["depth"], "depth")
    types = records.integers(vision.get("vit_window_attn_types") or [-1] * depth, "vit_window_attn_types")
    return {"class": "mimo", "fields": {
        "depth": depth,
        "hidden_size": records.integer(vision["hidden_size"], "hidden_size"),
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_heads": records.integer(vision["num_heads"], "num_heads"),
        "num_key_value_heads": records.integer(vision.get("num_key_value_heads", vision["num_heads"]),
                                               "num_key_value_heads"),
        "qk_channels": records.integer(vision.get("qk_channels", 64), "qk_channels"),
        "in_channels": records.integer(vision.get("in_channels") or vision.get("in_chans", 3), "in_chans"),
        "patch_size": records.integer(vision["patch_size"], "patch_size"),
        "spatial_merge_size": records.integer(vision.get("spatial_merge_size", 2), "spatial_merge_size"),
        "temporal_patch_size": records.integer(vision["temporal_patch_size"], "temporal_patch_size"),
        "rms_norm_eps": records.number(vision.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
        "fullatt_block_indexes": records.integers(vision.get("fullatt_block_indexes", []),
                                                  "fullatt_block_indexes"),
        "vit_window_attn_types": types,
        "use_sink": bool(vision.get("use_sink", False)),
        "visual_token_window_size": records.integer(vision.get("visual_token_window_size", -1),
                                                    "visual_token_window_size"),
    }}


def translate_mimo_projector_config(hf_config: Mapping[str, object], text_width: int) -> Record:
    """A MiMo-V2 wrapper's merger fields (`merger_projector`)."""
    return merger_projector(hf_config, text_width, "mimo", bias=False)


MiMoVision.checkpoint = TowerLayout("visual.", in_params(mimo_vision_path), read=translate_vit_weights)
# MiMo's merger stores no biases (MiMoVisionPatchMerger's released tensors).
MiMoProjector.checkpoint = ProjectorLayout("visual.merger.", {
    "ln_q.weight": ("norm", "scale"), "mlp.0.weight": ("fc1", "kernel"), "mlp.2.weight": ("fc2", "kernel"),
}, strip="merger.")
