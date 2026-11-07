"""MiMo-V2.6's vision trunk (MiMo ViT) and its merger, with their checkpoint maps
(XiaomiMiMo/MiMo-V2.6-Pro-MOPD's `modeling_mimo_v2.py`, MiMoVisionTransformer).
"""

import dataclasses

from flax import linen as nn

from .common import ProjectorBase, TowerBase, TowerGeometry
from .vit import MergeProjector, VisionTransformer, ViTLayout


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











# MiMo's merger stores no biases (MiMoVisionPatchMerger's released tensors).
