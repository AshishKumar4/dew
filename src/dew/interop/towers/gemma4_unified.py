"""Gemma 4 unified's vision tower and projector as a release stores them: their tensors' names and their
config's fields, both ways (`dew.interop.towers.common`), for the models
`dew.nn.vision.gemma4_unified` computes."""

import dataclasses
from collections.abc import Mapping

from dew import _records
from dew.interop.config_records import NativeFields
from dew.interop.layouts import ConfigField, ConfigLayout, Table
from dew.nn.vision.gemma4_unified import Gemma4UnifiedVision
from dew.registry import Record

from .common import TowerLayout, tensor_routes

_UNIFIED_CONFIG = ConfigLayout(*(ConfigField(field.name, field.name, field.default, parse=_records.integer)
                                  for field in dataclasses.fields(Gemma4UnifiedVision)))
"""A Unified vision_config's embedding fields, both ways, each at Gemma4UnifiedVisionConfig's default where
the config leaves it out."""


def translate_gemma4_unified_vision_config(config: Mapping[str, object]) -> Record:
    """A Unified vision_config's embedding fields, each at Gemma4UnifiedVisionConfig's default where the
    config leaves it out. Its num_soft_tokens is the processor's budget, which this embedding never
    reads: Dew takes the processor's patches as they are."""
    fields = _UNIFIED_CONFIG.read(config, set(), NativeFields(Gemma4UnifiedVision, {}))
    width = fields["mm_embed_dim"]
    if _records.integer(config.get("output_proj_dims", width), "output_proj_dims") != width:
        raise ValueError("output_proj_dims must equal mm_embed_dim for the vision embedding projection")
    return {"class": "gemma4_unified", "fields": fields}


_TENSORS = {
    "patch_ln1.weight": ("patch_norm1", "scale"), "patch_ln1.bias": ("patch_norm1", "bias"),
    "patch_dense.weight": ("patch_embed", "kernel"), "patch_dense.bias": ("patch_embed", "bias"),
    "patch_ln2.weight": ("patch_norm2", "scale"), "patch_ln2.bias": ("patch_norm2", "bias"),
    "pos_embedding": ("position_table",),
    "pos_norm.weight": ("position_norm", "scale"), "pos_norm.bias": ("position_norm", "bias"),
}


GEMMA4_UNIFIED_VISION = TowerLayout("vision_embedder.", Table(*tensor_routes(_TENSORS)),
                                             config=_UNIFIED_CONFIG, model_type="gemma4_unified_vision")

TOWERS = {Gemma4UnifiedVision: GEMMA4_UNIFIED_VISION}
"""Where each of these tower kinds' tensors sit in a media checkpoint (`dew.interop.towers.tower_layout`)."""
