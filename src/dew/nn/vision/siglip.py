"""The SigLIP vision trunk and Gemma 3's projector, with their checkpoint maps
(transformers 5.16.1 `models/siglip/modeling_siglip.py`, `models/gemma3/modeling_gemma3.py`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew import _records
from dew.batch import Variables
from dew.interop.config_records import NativeFields
from dew.interop.layouts import ConfigLayout, ConfigPart, Table
from dew.interop.unread import Stored, held
from dew.interop.weights import checkpoint_array
from dew.nn.dense import Dense
from dew.nn.norms import RMSNorm
from dew.nn.precision import float32_dot_general
from dew.nn.text_encoders import TOWER_ACTIVATIONS, encoder_layer_routes, tower_activation
from dew.registry import Record

from .common import (
    TRUNK_FIELDS,
    ProjectorBase,
    ProjectorLayout,
    TowerBase,
    TowerGeometry,
    TowerLayout,
    _vision_section,
    activation_field,
    head_width,
    integer_field,
    number_field,
    side_field,
    tensor_routes,
    unplaced,
)
from .vit import LearnedPositions, VisionTransformer, ViTLayout

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout


@dataclasses.dataclass(frozen=True)
class SiglipVision(TowerBase):
    """A SigLIP trunk's geometry, under the reference's field names."""

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6

    def layout(self) -> ViTLayout:
        return ViTLayout(
            depth=self.num_layers, hidden_size=self.hidden_size, num_heads=self.num_heads,
            num_kv_heads=self.num_heads, head_dim=head_width(self.hidden_size, self.num_heads),
            intermediate_size=self.intermediate_size, activation=tower_activation(self.hidden_act),
            in_channels=self.num_channels, patch_size=self.patch_size, temporal_patch_size=1,
            spatial_merge_size=1, norm_type="fast_layer", norm_eps=self.layer_norm_eps, norm_bias=True,
            mlp_bias=True, packed=False, patch_bias=True, attention=("full",) * self.num_layers,
            order=("patches",) * self.num_layers, sinks=(False,) * self.num_layers,
            positions=LearnedPositions(self.image_size // self.patch_size, "fixed"),
            patch_projection="conv", image_size=self.image_size, rotary=None, post_norm="layer",
            outer_norm_eps=self.layer_norm_eps, merge_blocks=False)

    def build(self) -> nn.Module:
        return VisionTransformer(self.layout())

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(image_size=self.image_size, patch_size=self.patch_size,
                             channels=self.num_channels)


class GemmaProjector(ProjectorBase):
    """Patch features into soft tokens: block average, norm, map to text width.

    The reference reshapes the trunk sequence into its patch grid, averages
    each kernel block, norms and multiplies by its (vision, text) matrix
    (modeling_gemma3.py, Gemma3MultiModalProjector).
    """

    text_width: int
    patches_per_side: int
    tokens_per_side: int
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        if self.patches_per_side % self.tokens_per_side:
            raise ValueError(
                f"patches_per_side ({self.patches_per_side}) must split over "
                f"tokens_per_side ({self.tokens_per_side})")
        self.mm_soft_emb_norm = RMSNorm(
            epsilon=self.norm_eps, scale_offset=True, dtype=self.dtype,
            name="mm_soft_emb_norm")
        self.mm_input_projection = Dense(
            self.text_width, use_bias=False, dtype=self.dtype,
            precision=self.precision, name="mm_input_projection", dot_general=float32_dot_general)

    def __call__(self, vision_outputs) -> jax.Array:
        batch, length, width = vision_outputs.shape
        patches, tokens = self.patches_per_side, self.tokens_per_side
        if length != patches * patches:
            raise ValueError(
                f"{length} patch features are not the {patches}x{patches} grid "
                "this projector pools")
        kernel = patches // tokens
        grid = vision_outputs.reshape(batch, patches, patches, width)
        pooled = grid.reshape(batch, tokens, kernel, tokens, kernel, width).mean(axis=(2, 4))
        return self.mm_input_projection(
            self.mm_soft_emb_norm(pooled.reshape(batch, tokens * tokens, width)))


_SIGLIP_TENSORS = {
    "embeddings.patch_embedding.weight": ("patch_embed", "kernel"),
    "embeddings.patch_embedding.bias": ("patch_embed", "bias"),
    "embeddings.position_embedding.weight": ("position_table", "embedding"),
    "post_layernorm.weight": ("post_norm", "scale"),
    "post_layernorm.bias": ("post_norm", "bias"),
}


_SIGLIP_ROUTES = Table(unplaced(r"embeddings\.position_ids|head\..+"), *tensor_routes(_SIGLIP_TENSORS),
                       encoder_layer_routes("encoder", block="blocks"))
"""A SigLIP trunk's tensor names, both ways. position_ids is an arange buffer, not a parameter; the
attention pooling head some checkpoints carry maps to nothing, the Gemma path reading the trunk
sequence alone."""


GEMMA_PROJECTOR_TENSORS = {
    "mm_soft_emb_norm.weight": ("mm_soft_emb_norm", "scale"),
    "mm_input_projection_weight": ("mm_input_projection", "kernel"),
}
"""The Gemma 3 projector's tensors past `multi_modal_projector.`, each to its leaf."""


def translate_gemma_projector_weights(
    hf_tensors: Mapping[str, Stored], *, param_dtype: str = "float32"
) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """A Gemma projector's two tensors into its parameter tree, with the layouts that write them
    back.

    The norm's weight becomes its scale; the projection matrix is a plain
    parameter the reference multiplies as is, so unlike a Linear kernel it
    keeps its layout.
    """
    from dew.interop.streaming import WeightLayout

    known = set(GEMMA_PROJECTOR_TENSORS)
    unknown = sorted(set(hf_tensors) - known)
    if unknown:
        raise ValueError(f"unknown tensor names {unknown}")
    paths = GEMMA_PROJECTOR_TENSORS
    return ({module: {leaf: held(checkpoint_array(hf_tensors[name], param_dtype), contiguous=True)}
             for name, (module, leaf) in paths.items()},
            tuple(WeightLayout(name, (path,), tuple(np.shape(hf_tensors[name])))
                  for name, path in paths.items()))


def translate_siglip_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A SiglipVisionConfig into a SiglipVision value's fields.

    Reads the vision_config of a multimodal wrapper or a bare vision config.
    Only square images map, and an activation the trunk does not compute
    (`TOWER_ACTIVATIONS`) refuses with its name. Dropout, which the eval trunk
    never applies, is left unread.
    """
    fields = _SIGLIP_CONFIG.read(_vision_section(hf_config), set(), NativeFields(SiglipVision, {}))
    return {"class": "siglip", "fields": fields}


_SIGLIP_CONFIG = ConfigLayout(
    *TRUNK_FIELDS, side_field("image_size", 224), side_field("patch_size", 16),
    integer_field("num_channels", default=3),
    activation_field(TOWER_ACTIVATIONS, "gelu_pytorch_tanh", "hidden_act", "hidden_activation"),
    number_field("layer_norm_eps", default=1e-6),
    # The pooling head some releases carry, which the trunk does not compute with: a written tower has none.
    ConfigPart(lambda vision, used, record: (used.add("vision_use_head"), record)[1],
               lambda tower: {"vision_use_head": False}))
"""A SiglipVisionConfig's fields, both ways."""


def translate_gemma_projector_config(vision: Mapping[str, object], text_width: int,
                                     mm_tokens_per_image: object) -> Record:
    """A Gemma wrapper's projector fields: decoder width, grids."""
    if isinstance(mm_tokens_per_image, bool) or not isinstance(mm_tokens_per_image, int):
        raise ValueError(
            f"mm_tokens_per_image is {mm_tokens_per_image!r}, the soft-token count "
            "is an int")
    side = int(mm_tokens_per_image ** 0.5)
    if side * side != mm_tokens_per_image:
        raise ValueError(
            f"mm_tokens_per_image ({mm_tokens_per_image}) is not a square, this "
            "projector pools a grid into a grid")
    patches = _records.integer(vision["image_size"], "image_size") // _records.integer(
        vision["patch_size"], "patch_size"
    )
    if patches % side:
        raise ValueError(
            f"{patches} patches per side do not split over {side} soft tokens per side")
    return {
        "class": "gemma", "fields": {
        "text_width": int(text_width),
        "patches_per_side": patches,
        "tokens_per_side": side,
        "norm_eps": _records.number(vision.get("layer_norm_eps", 1e-6), "layer_norm_eps"),
    }}


SiglipVision.checkpoint = TowerLayout("vision_tower.", _SIGLIP_ROUTES, config=_SIGLIP_CONFIG,
                                      model_type="siglip_vision_model")
GemmaProjector.checkpoint = ProjectorLayout("multi_modal_projector.", GEMMA_PROJECTOR_TENSORS,
                                            kept=frozenset({"mm_input_projection_weight"}),
                                            read=translate_gemma_projector_weights)
