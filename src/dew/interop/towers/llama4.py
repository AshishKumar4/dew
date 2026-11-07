"""Llama 4's vision tower and projector as a release stores them: their tensors' names and their
config's fields, both ways (`dew.interop.towers.common`), for the models `dew.nn.vision.llama4` computes."""

from collections.abc import Mapping

from dew import _records
from dew.interop.config_records import NativeFields
from dew.interop.layouts import ConfigField, ConfigLayout, Table
from dew.interop.text_encoders import encoder_layer_routes
from dew.nn.vision.llama4 import Llama4Projector, Llama4Vision
from dew.registry import Record

from .common import (
    TRUNK_FIELDS,
    ProjectorLayout,
    TowerLayout,
    _vision_section,
    integer_field,
    number_field,
    rope_part,
    side_field,
    tensor_routes,
)

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


LLAMA4_VISION = TowerLayout("vision_model.", _LLAMA4_ROUTES, config=_LLAMA4_CONFIG,
                                      model_type="llama4_vision_model")


LLAMA4_PROJECTOR = ProjectorLayout("multi_modal_projector.",
                                             {"linear_1.weight": ("linear", "kernel")})

TOWERS = {Llama4Vision: LLAMA4_VISION}
"""Where each of these tower kinds' tensors sit in a media checkpoint (`dew.interop.towers.tower_layout`)."""


PROJECTORS = {Llama4Projector: LLAMA4_PROJECTOR}
"""Where each of these projector kinds' tensors sit in a media checkpoint
(`dew.interop.towers.projector_layout`)."""
