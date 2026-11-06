"""The Llama 4 vision trunk and its projector, with their checkpoint maps
(transformers 5.16.1 `models/llama4/modeling_llama4.py`).
"""

import dataclasses
from collections.abc import Mapping

import jax
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew import _records
from dew.interop.config_records import NativeFields
from dew.interop.layouts import ConfigField, ConfigLayout, Table
from dew.nn.dense import Dense
from dew.nn.precision import float32_dot_general
from dew.nn.scan_orders import patchify
from dew.nn.text_encoders import encoder_layer_routes
from dew.nn.vision.vit import LearnedPositions, MergeProjector, SpecialTokens, VisionTransformer, ViTLayout
from dew.registry import Record

from .common import (
    TRUNK_FIELDS,
    ProjectorBase,
    ProjectorLayout,
    TowerBase,
    TowerGeometry,
    TowerLayout,
    _vision_section,
    head_width,
    integer_field,
    number_field,
    rope_part,
    side_field,
    tensor_routes,
)


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


# Llama 4 names its layers' norms and maps as the decoder block does.
_LAYER_NORMS = {name: name for name in ("input_layernorm", "post_attention_layernorm")}
_LAYER_PROJECTIONS = {name: name for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
_LLAMA4_VISION_TENSORS = {
    "patch_embedding.linear.weight": ("patch_embed", "kernel"),
    "class_embedding": ("class_token",),
    "positional_embedding_vlm": ("position_table", "embedding"),
    "layernorm_pre.weight": ("pre_norm", "scale"),
    "layernorm_pre.bias": ("pre_norm", "bias"),
    "layernorm_post.weight": ("post_norm", "scale"),
    "layernorm_post.bias": ("post_norm", "bias"),
    "vision_adapter.mlp.fc1.weight": ("vision_adapter", "fc1", "kernel"),
    "vision_adapter.mlp.fc2.weight": ("vision_adapter", "fc2", "kernel"),
}


_LLAMA4_ROUTES = Table(*tensor_routes(_LLAMA4_VISION_TENSORS),
                       encoder_layer_routes("model", _LAYER_NORMS, _LAYER_PROJECTIONS, block="blocks"))
"""A Llama 4 trunk's tensor names, both ways."""


def translate_llama4_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A Llama4VisionConfig into a Llama4Vision value's fields.

    Reads the vision_config of a wrapper or a bare vision config. The feature
    selection defaults to the last layer; the concatenated-layer strategy has
    no counterpart and refuses. Rope reads the nested spelling transformers
    writes or the flat theta the released Scout carries.
    """
    vision = _vision_section(hf_config)
    if vision.get("vision_feature_select_strategy", "default") != "default":
        raise ValueError(
            f"vision_feature_select_strategy "
            f"{vision.get('vision_feature_select_strategy')!r} concatenates layers "
            "this trunk never builds")
    if vision.get("vision_feature_layer", -1) not in (-1, None):
        raise ValueError(
            f"vision_feature_layer {vision.get('vision_feature_layer')!r} reads a "
            "middle layer this trunk never returns")
    if vision.get("multi_modal_projector_bias", False):
        raise ValueError("multi_modal_projector_bias=True needs a projector bias this map lacks")
    output_dim = _records.integer(
        vision.get("vision_output_dim", vision.get("projector_output_dim", 0)), "vision_output_dim"
    )
    if output_dim != _records.integer(vision.get("projector_output_dim", output_dim), "projector_output_dim"):
        raise ValueError(
            f"vision_output_dim ({output_dim}) disagrees with projector_output_dim "
            f"({vision.get('projector_output_dim')}), the adapter's width")
    return {"class": "llama4", "fields": _LLAMA4_CONFIG.read(vision, set(), NativeFields(Llama4Vision, {}))}


_LLAMA4_CONFIG = ConfigLayout(
    *TRUNK_FIELDS, side_field("image_size", 336), integer_field("patch_size", default=14),
    integer_field("num_channels", default=3),
    ConfigField("norm_eps", "layer_norm_eps", 1e-5, parse=_records.number, aliases=("layer_norm_eps",)),
    rope_part(10000.0), number_field("pixel_shuffle_ratio", default=0.5),
    integer_field("projector_input_dim", required=True),
    # The adapter's width, which the vision output's width states again.
    ConfigField("projector_output_dim", "projector_output_dim", parse=_records.integer),
    ConfigField("vision_output_dim", "projector_output_dim", parse=_records.integer,
                aliases=("projector_output_dim",)))
"""A Llama4VisionConfig's fields, both ways."""


def translate_llama4_projector_config(text_width: int) -> Record:
    """A Llama 4 wrapper's projector fields: the text width."""
    return {"class": "llama4", "fields": {"text_width": int(text_width)}}


Llama4Vision.checkpoint = TowerLayout("vision_model.", _LLAMA4_ROUTES, config=_LLAMA4_CONFIG,
                                      model_type="llama4_vision_model")
Llama4Projector.checkpoint = ProjectorLayout("multi_modal_projector.",
                                             {"linear_1.weight": ("linear", "kernel")})
