"""Qwen 3.5's vision tower and projector as a release stores them: their tensors' names and their
config's fields, both ways (`dew.interop.towers.common`), for the models `dew.nn.vision.qwen35` computes."""

from collections.abc import Mapping

from dew import _records
from dew._model_types import _QWEN35_VISION_TYPES
from dew.interop.config_records import NativeFields
from dew.interop.layouts import ConfigLayout, Table
from dew.nn.text_encoders import TOWER_ACTIVATIONS
from dew.nn.vision.qwen35 import Qwen35Projector, Qwen35Vision
from dew.registry import Record

from .common import (
    ProjectorLayout,
    TowerBase,
    TowerLayout,
    _vision_section,
    activation_field,
    block_routes,
    integer_field,
    merger_projector,
    tensor_routes,
    unplaced,
)
from .vit import PATCH_CONVOLUTION, translate_vit_weights

_QWEN35_VISION_TENSORS = {
    "patch_embed.proj.weight": ("patch_embed", "kernel"),
    "patch_embed.proj.bias": ("patch_embed", "bias"),
    "pos_embed.weight": ("position_table", "embedding"),
}


# A block's maps as `clip_layer`'s decoder block names them (`block_routes`).
_BLOCK_MAPS = {("attn", "qkv"): ("self_attn", "qkv_proj"), ("attn", "proj"): ("self_attn", "o_proj"),
               ("mlp", "linear_fc1"): ("mlp", "up_proj"), ("mlp", "linear_fc2"): ("mlp", "down_proj")}


_QWEN35_ROUTES = Table(unplaced(r"merger\..+"), *tensor_routes(_QWEN35_VISION_TENSORS),
                       block_routes(_BLOCK_MAPS))
"""A Qwen 3.5 trunk's tensor names past `visual.`, both ways: the merger maps with the projector, and the
rotary table is a buffer the checkpoint leaves out."""


def _patch_convolution(tower: TowerBase) -> Mapping[str, tuple[int, ...]]:
    """The patch map as the release stores it, a `[width, channels, time, patch, patch]` convolution."""
    assert isinstance(tower, Qwen35Vision)
    return {PATCH_CONVOLUTION: (tower.hidden_size, tower.in_channels, tower.temporal_patch_size,
                                tower.patch_size, tower.patch_size)}


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
    table = _records.integer(vision["num_position_embeddings"], "num_position_embeddings")
    if int(table ** 0.5) ** 2 != table:
        raise ValueError(
            f"num_position_embeddings ({table}) is not a square, this trunk "
            "resamples a square table")
    return {"class": "qwen3_5", "fields": _QWEN35_CONFIG.read(vision, set(), NativeFields(Qwen35Vision, {}))}


_QWEN35_CONFIG = ConfigLayout(
    integer_field("depth", required=True), integer_field("hidden_size", required=True),
    activation_field(TOWER_ACTIVATIONS, "gelu_pytorch_tanh", "hidden_act"),
    integer_field("intermediate_size", required=True), integer_field("num_heads", required=True),
    integer_field("in_channels", default=3), integer_field("patch_size", required=True),
    integer_field("spatial_merge_size", default=2), integer_field("temporal_patch_size", required=True),
    integer_field("num_position_embeddings", required=True))
"""A Qwen3_5VisionConfig's fields, both ways: the trunk's geometry under the reference's names."""


def translate_qwen35_projector_config(hf_config: Mapping[str, object],
                                      text_width: int) -> Record:
    """A Qwen 3.5 wrapper's projector fields: trunk width, merge, output (`merger_projector`)."""
    return merger_projector(hf_config, text_width, "qwen3_5")


QWEN35_VISION = TowerLayout("visual.", _QWEN35_ROUTES, read=translate_vit_weights,
                                      shapes=_patch_convolution, config=_QWEN35_CONFIG,
                                      model_type="qwen3_5_vision")


QWEN35_PROJECTOR = ProjectorLayout("visual.merger.", {
    "norm.weight": ("norm", "scale"), "norm.bias": ("norm", "bias"),
    "linear_fc1.weight": ("fc1", "kernel"), "linear_fc1.bias": ("fc1", "bias"),
    "linear_fc2.weight": ("fc2", "kernel"), "linear_fc2.bias": ("fc2", "bias"),
}, strip="merger.")

TOWERS = {Qwen35Vision: QWEN35_VISION}
"""Where each of these tower kinds' tensors sit in a media checkpoint (`dew.interop.towers.tower_layout`)."""


PROJECTORS = {Qwen35Projector: QWEN35_PROJECTOR}
"""Where each of these projector kinds' tensors sit in a media checkpoint
(`dew.interop.towers.projector_layout`)."""
