"""SigLIP and Gemma 3's vision tower and projector as a release stores them: their tensors' names and their
config's fields, both ways (`dew.interop.towers.common`), for the models `dew.nn.vision.siglip` computes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from dew import _records
from dew.batch import Variables
from dew.interop.config_records import NativeFields
from dew.interop.layouts import ConfigLayout, ConfigPart, Table
from dew.interop.unread import Stored, held
from dew.interop.weights import checkpoint_array
from dew.nn.text_encoders import TOWER_ACTIVATIONS, encoder_layer_routes
from dew.nn.vision.siglip import GemmaProjector, SiglipVision
from dew.registry import Record

from .common import (
    TRUNK_FIELDS,
    ProjectorLayout,
    TowerLayout,
    _vision_section,
    activation_field,
    integer_field,
    number_field,
    side_field,
    tensor_routes,
    unplaced,
)

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

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


SIGLIP_VISION = TowerLayout("vision_tower.", _SIGLIP_ROUTES, config=_SIGLIP_CONFIG,
                                      model_type="siglip_vision_model")


GEMMA_PROJECTOR = ProjectorLayout("multi_modal_projector.", GEMMA_PROJECTOR_TENSORS,
                                            kept=frozenset({"mm_input_projection_weight"}),
                                            read=translate_gemma_projector_weights)

TOWERS = {SiglipVision: SIGLIP_VISION}
"""Where each of these tower kinds' tensors sit in a media checkpoint (`dew.interop.towers.tower_layout`)."""


PROJECTORS = {GemmaProjector: GEMMA_PROJECTOR}
"""Where each of these projector kinds' tensors sit in a media checkpoint
(`dew.interop.towers.projector_layout`)."""
