"""The Qwen 3.5 vision trunk and its merger
(transformers 5.16.1 `models/qwen3_5/modeling_qwen3_5.py`).
"""

import dataclasses

from flax import linen as nn

from dew.nn.text_encoders import TOWER_ACTIVATIONS

from .common import ProjectorBase, TowerBase, TowerGeometry, head_width
from .vit import LearnedPositions, MergeProjector, VisionTransformer, ViTLayout


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
            positions=LearnedPositions(round(self.num_position_embeddings ** 0.5)))

    def build(self) -> nn.Module:
        return VisionTransformer(self.layout())

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.spatial_merge_size,
                             channels=self.in_channels)


class Qwen35Projector(MergeProjector, ProjectorBase):
    """Qwen 3.5's projector fields: the trunk width, the merge size, the
    merged width the text embeddings read directly. The merger
    (modeling_qwen3_5.py, Qwen3_5VisionPatchMerger) is `MergeProjector`: norm,
    each merge block grouped, two maps around an exact GELU."""


