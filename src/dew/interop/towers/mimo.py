"""MiMo's vision tower and projector as a release stores them: their tensors' names and their
config's fields, both ways (`dew.interop.towers.common`), for the models `dew.nn.vision.mimo` computes."""

from collections.abc import Mapping

from dew import _records
from dew.interop.layouts import Route, Table
from dew.nn.vision.mimo import MiMoProjector, MiMoVision
from dew.registry import Record

from .common import (
    ProjectorLayout,
    TowerLayout,
    _vision_section,
    activation_name,
    block_routes,
    merger_projector,
    unplaced,
)
from .vit import PATCH_CONVOLUTION, patch_convolution, translate_vit_weights

_BLOCK_MAPS = {("attn", "qkv"): ("self_attn", "qkv_proj"), ("attn", "proj"): ("self_attn", "o_proj"),
               ("mlp", "gate_proj"): ("mlp", "gate_proj"), ("mlp", "up_proj"): ("mlp", "up_proj"),
               ("mlp", "down_proj"): ("mlp", "down_proj")}


_MIMO_ROUTES = Table(
    unplaced(r"merger\..+"), Route(PATCH_CONVOLUTION, "params/patch_embed/kernel"),
    Route("blocks.{index}.attn.sinks", "params/sinks_{index}"), block_routes(_BLOCK_MAPS, norm_bias=False))
"""A MiMo ViT's tensor names past `visual.`, both ways; the merger maps with the projector."""


def translate_mimo_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A MiMo-V2 vision_config into a MiMoVision value's fields. The trunk computes SwiGLU, as the
    release's silu names it, and no other activation."""
    vision = _vision_section(hf_config)
    activation_name(vision, ("silu",), "silu")
    depth = _records.integer(vision["depth"], "depth")
    types = _records.integers(vision.get("vit_window_attn_types") or [-1] * depth, "vit_window_attn_types")
    return {"class": "mimo", "fields": {
        "depth": depth,
        "hidden_size": _records.integer(vision["hidden_size"], "hidden_size"),
        "intermediate_size": _records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_heads": _records.integer(vision["num_heads"], "num_heads"),
        "num_key_value_heads": _records.integer(vision.get("num_key_value_heads", vision["num_heads"]),
                                               "num_key_value_heads"),
        "qk_channels": _records.integer(vision.get("qk_channels", 64), "qk_channels"),
        "in_channels": _records.integer(vision.get("in_channels") or vision.get("in_chans", 3), "in_chans"),
        "patch_size": _records.integer(vision["patch_size"], "patch_size"),
        "spatial_merge_size": _records.integer(vision.get("spatial_merge_size", 2), "spatial_merge_size"),
        "temporal_patch_size": _records.integer(vision["temporal_patch_size"], "temporal_patch_size"),
        "rms_norm_eps": _records.number(vision.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
        "fullatt_block_indexes": _records.integers(vision.get("fullatt_block_indexes", []),
                                                  "fullatt_block_indexes"),
        "vit_window_attn_types": types,
        "use_sink": bool(vision.get("use_sink", False)),
        "visual_token_window_size": _records.integer(vision.get("visual_token_window_size", -1),
                                                    "visual_token_window_size"),
    }}


def translate_mimo_projector_config(hf_config: Mapping[str, object], text_width: int) -> Record:
    """A MiMo-V2 wrapper's merger fields (`merger_projector`)."""
    return merger_projector(hf_config, text_width, "mimo", bias=False)


MIMO_VISION = TowerLayout("visual.", _MIMO_ROUTES, read=translate_vit_weights,
                                    shapes=patch_convolution)


# MiMo's merger stores no biases (MiMoVisionPatchMerger's released tensors).
MIMO_PROJECTOR = ProjectorLayout("visual.merger.", {
    "ln_q.weight": ("norm", "scale"), "mlp.0.weight": ("fc1", "kernel"), "mlp.2.weight": ("fc2", "kernel"),
}, strip="merger.")

TOWERS = {MiMoVision: MIMO_VISION}
"""Where each of these tower kinds' tensors sit in a media checkpoint (`dew.interop.towers.tower_layout`)."""


PROJECTORS = {MiMoProjector: MIMO_PROJECTOR}
"""Where each of these projector kinds' tensors sit in a media checkpoint
(`dew.interop.towers.projector_layout`)."""
