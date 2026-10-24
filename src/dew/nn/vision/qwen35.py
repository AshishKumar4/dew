"""The Qwen 3.5 vision trunk and its merger, with their checkpoint maps
(transformers 5.16.1 `models/qwen3_5/modeling_qwen3_5.py`).
"""

import dataclasses
from collections.abc import Mapping

from flax import linen as nn

from dew import records
from dew._model_types import _QWEN35_VISION_TYPES
from dew.interop.weights import in_params
from dew.nn.text_encoders import TOWER_ACTIVATIONS
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
    head_width,
    merger_projector,
)
from .vit import MergeProjector, VisionTransformer, ViTLayout, translate_vit_weights


@dataclasses.dataclass(frozen=True)
class Qwen35Vision(TowerBase):
    """A Qwen 3.5 trunk's geometry, under the reference's field names.

    Its layout (`ViTLayout`): a biased patch map, the learned table resampled with aligned corners,
    CLIP's layer at a 1e-6 epsilon (flax's one-pass layer norm with its bias, the queries, keys and
    values in one biased map, an ungated feed-forward) and full attention over each frame, rotated
    by the patch grid, heights then widths (modeling_qwen3_5.py, Qwen3_5VisionRotaryEmbedding and
    apply_rotary_pos_emb_vision). The trunk sequence leaves for the merger, its projector.
    """

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int = 2304

    def layout(self) -> ViTLayout:
        return ViTLayout(
            depth=self.depth, hidden_size=self.hidden_size, num_heads=self.num_heads,
            num_kv_heads=self.num_heads, head_dim=head_width(self.hidden_size, self.num_heads),
            intermediate_size=self.intermediate_size, activation=TOWER_ACTIVATIONS[self.hidden_act],
            in_channels=self.in_channels, patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size, spatial_merge_size=self.spatial_merge_size,
            norm_type="fast_layer", norm_eps=1e-6, norm_bias=True, mlp_bias=True, packed=True,
            patch_bias=True,
            attention=("full",) * self.depth, order=("patches",) * self.depth, sinks=(False,) * self.depth,
            position_table=self.num_position_embeddings)

    def build(self) -> nn.Module:
        return VisionTransformer(self.layout())

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.spatial_merge_size,
                             channels=self.in_channels)


@dataclasses.dataclass(frozen=True)
class Qwen35Projector(ProjectorBase):
    """Qwen 3.5's projector fields: the trunk width, the merge size, the
    merged width the text embeddings read directly. The merger
    (modeling_qwen3_5.py, Qwen3_5VisionPatchMerger) is `MergeProjector`: norm,
    each merge block grouped, two maps around an exact GELU."""

    vision_width: int
    merge_size: int
    out_width: int

    def build(self) -> nn.Module:
        return MergeProjector(hidden_size=self.vision_width, spatial_merge_size=self.merge_size,
                              width=self.vision_width * self.merge_size ** 2, out_width=self.out_width)


_QWEN35_VISION_TENSORS = {
    "patch_embed.proj.weight": ("patch_embed", "kernel"),
    "patch_embed.proj.bias": ("patch_embed", "bias"),
    "pos_embed.weight": ("position_table", "embedding"),
}


# A block's maps as `clip_layer`'s decoder block names them (`block_tensor_path`).
_BLOCK_MAPS = {("attn", "qkv"): ("self_attn", "qkv_proj"), ("attn", "proj"): ("self_attn", "o_proj"),
               ("mlp", "linear_fc1"): ("mlp", "up_proj"), ("mlp", "linear_fc2"): ("mlp", "down_proj")}


def qwen35_vision_path(hf_name: str) -> tuple[str, ...] | None:
    """One Qwen 3.5 vision tensor name into its path in a trunk tree.

    The merger lives under its own prefix and maps with the projector; the
    rotary table is a buffer the checkpoint leaves out. Anything else unknown
    raises ValueError.
    """
    if hf_name.split(".")[0] == "merger":
        return None
    path = _QWEN35_VISION_TENSORS.get(hf_name) or block_tensor_path(hf_name.split("."), _BLOCK_MAPS)
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def translate_qwen35_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A Qwen3_5VisionConfig into a Qwen35Vision value's fields.

    Reads the vision_config of a wrapper or a bare vision config. The
    released 0.8B checkpoint spells the tower's model_type qwen3_5, which the
    config class rewrites on load; both spellings map. The position table
    must be square, and the activation one the shared MLP runs.
    """
    vision = _vision_section(hf_config)
    if vision.get("model_type", "qwen3_5_vision") not in _QWEN35_VISION_TYPES:
        raise ValueError(
            f"vision model_type {vision.get('model_type')!r} is not the Qwen 3.5 tower")
    table = records.integer(vision["num_position_embeddings"], "num_position_embeddings")
    if int(table ** 0.5) ** 2 != table:
        raise ValueError(
            f"num_position_embeddings ({table}) is not a square, this trunk "
            "resamples a square table")
    return {
        "class": "qwen3_5", "fields": {
        "depth": records.integer(vision["depth"], "depth"),
        "hidden_size": records.integer(vision["hidden_size"], "hidden_size"),
        "hidden_act": activation_name(vision, TOWER_ACTIVATIONS, "gelu_pytorch_tanh"),
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_heads": records.integer(vision["num_heads"], "num_heads"),
        "in_channels": records.integer(vision.get("in_channels", 3), "in_channels"),
        "patch_size": records.integer(vision["patch_size"], "patch_size"),
        "spatial_merge_size": records.integer(vision.get("spatial_merge_size", 2), "spatial_merge_size"),
        "temporal_patch_size": records.integer(vision["temporal_patch_size"], "temporal_patch_size"),
        "num_position_embeddings": table,
    }}


def translate_qwen35_projector_config(hf_config: Mapping[str, object],
                                      text_width: int) -> Record:
    """A Qwen 3.5 wrapper's projector fields: trunk width, merge, output (`merger_projector`)."""
    return merger_projector(hf_config, text_width, "qwen3_5")


Qwen35Vision.checkpoint = TowerLayout("visual.", in_params(qwen35_vision_path), read=translate_vit_weights)
Qwen35Projector.checkpoint = ProjectorLayout("visual.merger.", {
    "norm.weight": ("norm", "scale"), "norm.bias": ("norm", "bias"),
    "linear_fc1.weight": ("fc1", "kernel"), "linear_fc1.bias": ("fc1", "bias"),
    "linear_fc2.weight": ("fc2", "kernel"), "linear_fc2.bias": ("fc2", "bias"),
}, strip="merger.")
