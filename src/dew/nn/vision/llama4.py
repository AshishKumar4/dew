"""The Llama 4 vision trunk and its projector
(transformers 5.16.1 `models/llama4/modeling_llama4.py`).
"""

import dataclasses

import jax
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.dense import Dense
from dew.nn.precision import float32_dot_general
from dew.nn.scan_orders import patchify
from dew.nn.vision.vit import LearnedPositions, MergeProjector, SpecialTokens, VisionTransformer, ViTLayout

from .common import ProjectorBase, TowerBase, TowerGeometry, head_width


def pixel_shuffle(patches: jax.Array, ratio: float) -> jax.Array:
    """Space to depth: each block of `1 / ratio` patches a side of the square
    patch grid becomes one thicker token, its patches row by row (`patchify`).

    The inverse of a pixel shuffle at the same ratio (modeling_llama4.py,
    pixel_shuffle): tokens shrink by ratio squared and channels grow by it.
    """
    batch, count, channels = patches.shape
    side = round(count ** 0.5)
    block = round(1 / ratio)
    if block * ratio != 1:
        raise ValueError(f"a shuffle ratio of {ratio} does not gather whole blocks of patches")
    return patchify(patches.reshape(batch, side, side, channels), block)


class Llama4VisionTransformer(nn.Module):
    """The Llama 4 vision trunk, param layout of `Llama4VisionConfig`.

    `pixel_values` are what the checkpoint's image processor emits: [B, C, H,
    W] at `image_size`. The patches embed through one bias-free map, the class
    token rides last through the trunk and is dropped before the adapter, and
    what returns is the pixel-shuffled MLP output the outer projector maps to
    text width.
    """

    config: "Llama4Vision"
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        cfg = self.config
        self.trunk = VisionTransformer(cfg.layout(), dtype=self.dtype, precision=self.precision)
        nn.share_scope(self, self.trunk)
        # The reference's adapter maps the shuffled patches through two bias-free maps with an
        # exact GELU after each, the last one included (modeling_llama4.py, Llama4VisionMLP2).
        self.vision_adapter = MergeProjector(
            round(cfg.hidden_size / cfg.pixel_shuffle_ratio ** 2), 1, cfg.projector_input_dim,
            cfg.projector_output_dim, norm=False, bias=False, activation="gelu", activation_after=True,
            dtype=self.dtype, precision=self.precision, name="vision_adapter")

    def __call__(self, pixel_values) -> jax.Array:
        cfg = self.config
        hidden_states = self.trunk(pixel_values)
        return self.vision_adapter(pixel_shuffle(hidden_states, cfg.pixel_shuffle_ratio))


@dataclasses.dataclass(frozen=True)
class Llama4Vision(TowerBase):
    """A Llama 4 trunk's geometry, under the reference's field names."""

    hidden_size: int = 1408
    intermediate_size: int = 5632
    num_layers: int = 32
    num_heads: int = 16
    image_size: int = 336
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    pixel_shuffle_ratio: float = 0.5
    projector_input_dim: int = 4096
    projector_output_dim: int = 4096

    def layout(self) -> ViTLayout:
        initialize = nn.initializers.normal(self.hidden_size ** -0.5)
        return ViTLayout(
            depth=self.num_layers, hidden_size=self.hidden_size, num_heads=self.num_heads,
            num_kv_heads=self.num_heads, head_dim=head_width(self.hidden_size, self.num_heads),
            intermediate_size=self.intermediate_size, activation="gelu_exact", in_channels=self.num_channels,
            patch_size=self.patch_size, temporal_patch_size=1, spatial_merge_size=1,
            norm_type="fast_layer", norm_eps=self.layer_norm_eps, norm_bias=True, mlp_bias=True,
            packed=False, patch_bias=False, attention=("full",) * self.num_layers,
            order=("patches",) * self.num_layers, sinks=(False,) * self.num_layers,
            positions=LearnedPositions(self.image_size // self.patch_size, "fixed", initializer=initialize),
            special_tokens=SpecialTokens("last", initializer=initialize), image_size=self.image_size,
            rotary="adjacent", pre_norm="layer", post_norm="layer", outer_norm_eps=self.layer_norm_eps,
            output="patches", rope_theta=self.rope_theta, rotary_columns_first=True, rotary_origin=1,
            merge_blocks=False)

    def build(self) -> nn.Module:
        return Llama4VisionTransformer(self)

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(image_size=self.image_size, patch_size=self.patch_size,
                             channels=self.num_channels)


class Llama4Projector(ProjectorBase):
    """Tower output to text width: one bias-free map."""

    text_width: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.linear = Dense(
            self.text_width, use_bias=False, dtype=self.dtype,
            precision=self.precision, name="linear", dot_general=float32_dot_general)

    def __call__(self, image_features) -> jax.Array:
        return self.linear(image_features)


