"""Inkling 2's vision trunk and its merger
(transformers 3a96649 `models/inkling2/modeling_inkling2.py`, Inkling2VisionModel and Inkling2VisionMerger).
"""

import dataclasses

from flax import linen as nn

from .common import ProjectorBase, TowerBase, TowerGeometry, head_width
from .vit import LearnedPositions, MergeProjector, SpecialTokens, VisionTransformer, ViTLayout


@dataclasses.dataclass(frozen=True)
class InklingVision(TowerBase):
    """Inkling 2's trunk geometry under its vision_config's names.

    Its layout (`ViTLayout`): a bias-free patch map over each patch's channels, its rows and columns; the
    learned `posemb_grid_size`-square table resampled without aligned corners, its taps held at the edge
    (`get_vision_interpolation_indices_and_weights`' border padding), and a leading class token, its own
    position the table's first row; a layer norm before the blocks; CLIP's layer at `layer_norm_eps` with
    its queries, keys and values apart and biased, its feed-forward an exact GELU between two biased maps,
    each branch times its learned per-channel scale (`ls_1`, `ls_2`, from `layer_scale_init_value`); full
    attention over each image, rotated over the whole head by the patch grid, columns then rows in adjacent
    pairs, the patches counted from one and the class token at the origin; no norm after the blocks. The
    patches leave for the merger, its projector (`InklingProjector`), the class token left behind.
    """

    depth: int = 23
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 20
    spatial_merge_size: int = 2
    posemb_grid_size: int = 32
    layer_norm_eps: float = 1e-5
    layer_scale_init_value: float = 0.1
    rope_theta: float = 10000.0

    temporal_patch_size: int = 1
    """A patch spans one frame: the reference's patch map reads a still image's rows, columns and channels."""

    def layout(self) -> ViTLayout:
        return ViTLayout(
            depth=self.depth, hidden_size=self.hidden_size, num_heads=self.num_heads,
            num_kv_heads=self.num_heads, head_dim=head_width(self.hidden_size, self.num_heads),
            intermediate_size=self.intermediate_size, activation="gelu", in_channels=self.in_channels,
            patch_size=self.patch_size, temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.spatial_merge_size, norm_type="layer", norm_eps=self.layer_norm_eps,
            norm_bias=True, mlp_bias=True, packed=False, patch_bias=False,
            attention=("full",) * self.depth, order=("patches",) * self.depth, sinks=(False,) * self.depth,
            positions=LearnedPositions(self.posemb_grid_size, align_corners=False, border=True),
            special_tokens=SpecialTokens(cls="first"), rotary="adjacent", rotary_columns_first=True,
            rotary_origin=1, rope_theta=self.rope_theta, pre_norm="layer", outer_norm_eps=self.layer_norm_eps,
            output="patches", branch_scale=self.layer_scale_init_value)

    def build(self) -> nn.Module:
        return VisionTransformer(self.layout())

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.spatial_merge_size,
                             channels=self.in_channels)


class InklingProjector(MergeProjector, ProjectorBase):
    """Inkling 2's projector fields: the trunk width, the merge size, the projection width
    (`projection_hidden_dim`) and the text width. The merger (Inkling2VisionMerger) is `MergeProjector`
    without a norm: each merge block's patches joined, two maps around the trunk's exact GELU, biased where
    `projector_bias`."""
