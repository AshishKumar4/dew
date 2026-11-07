"""Muse Glimmer's vision trunk and projector, with their checkpoint maps
(transformers 5.16.1 `models/muse_glimmer/modeling_muse_glimmer.py`).
"""

import dataclasses

import jax
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.dense import Dense
from dew.nn.norms import RMSNorm
from dew.nn.precision import float32_dot_general
from dew.nn.text_encoders import TOWER_ACTIVATIONS

from .common import ProjectorBase, TowerBase, TowerGeometry
from .vit import LearnedPositions, MergeProjector, VisionTransformer, ViTLayout


def default_layer_types(layers: int) -> tuple[str, ...]:
    """Muse Glimmer's attention cadence over `layers` layers: full attention every fourth layer and
    at the last, window attention elsewhere (MuseGlimmerVisionConfig's default `layer_types`)."""
    return tuple("full_attention" if (i + 1) % 4 == 0 or i == layers - 1 else "window_attention"
                 for i in range(layers))


@dataclasses.dataclass(frozen=True)
class MuseGlimmerVision(TowerBase):
    """Muse Glimmer's trunk geometry under its vision_config's names.

    Its layout (`ViTLayout`): a bias-free map of each raster patch's time-then-channel pixels, the
    learned `pos_emb_height` x `pos_emb_width` table resampled with half-pixel centres and zero
    padding (MuseGlimmerVisionPatchEmbedder), a layer norm before the blocks and one after, and
    CLIP-style blocks (layer norms at 1e-5, biased query, key and value maps, an ungated
    feed-forward) rotated by the patch grid, columns then rows, each counted from one. The tokens run
    window by window, `pos_emb_height` patches square; a `window_attention` layer attends within its
    window, a `full_attention` one over its frame.
    """

    num_hidden_layers: int = 50
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_attention_heads: int = 16
    hidden_act: str = "gelu"
    patch_size: int = 14
    patch_temporal: int = 2
    merge_size: int = 2
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    layer_types: tuple[str, ...] = default_layer_types(50)

    def __post_init__(self):
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"layer_types names {len(self.layer_types)} layers of {self.num_hidden_layers}")
        if self.pos_emb_height != self.pos_emb_width:
            raise ValueError("the reference resamples a square position table only")

    def layout(self) -> ViTLayout:
        heads = self.num_attention_heads
        return ViTLayout(
            depth=self.num_hidden_layers, hidden_size=self.hidden_size, num_heads=heads, num_kv_heads=heads,
            head_dim=self.hidden_size // heads, intermediate_size=self.intermediate_size,
            activation=TOWER_ACTIVATIONS[self.hidden_act], in_channels=3, patch_size=self.patch_size,
            temporal_patch_size=self.patch_temporal, spatial_merge_size=self.merge_size, norm_type="layer",
            norm_eps=1e-5, norm_bias=True, mlp_bias=True, packed=False, patch_bias=False,
            attention=tuple("window" if kind == "window_attention" else "full" for kind in self.layer_types),
            order=("windows",) * self.num_hidden_layers, sinks=(False,) * self.num_hidden_layers,
            positions=LearnedPositions(self.pos_emb_height, align_corners=False), pre_norm="layer",
            post_norm="layer", outer_norm_eps=self.layer_norm_eps, rope_theta=self.rope_theta,
            rotary_columns_first=True, rotary_origin=1, window=self.pos_emb_height, merge_blocks=False,
            channels_first=False)

    def build(self) -> nn.Module:
        return VisionTransformer(self.layout())

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.merge_size, channels=3)


class MuseGlimmerProjector(ProjectorBase):
    """The merged image tokens as the decoder reads them (MuseGlimmerModel.get_image_features): each
    merge block's features channel by channel through the adapter's two bias-free maps, each followed
    by the activation (`MergeProjector`), a bias-free map to the decoder width, and a scale-free RMS
    norm (`perception_emb_norm`)."""

    vision_width: int
    merge_size: int
    adapter_width: int
    text_width: int
    activation: str = "gelu"
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, image_features) -> jax.Array:
        adapted = MergeProjector(hidden_size=self.vision_width, spatial_merge_size=self.merge_size,
                                 width=self.adapter_width, out_width=self.adapter_width, norm=False,
                                 channel_major=True, bias=False, activation=self.activation,
                                 activation_after=True, dtype=self.dtype, precision=self.precision,
                                 name="adapter")(image_features)
        projected = Dense(self.text_width, use_bias=False, dtype=self.dtype, precision=self.precision,
                          name="projection", dot_general=float32_dot_general)(adapted)
        return RMSNorm(epsilon=self.norm_eps, with_scale=False, dtype=self.dtype, name="norm")(projected)














# The adapter and the projection, each name its prefix off.
