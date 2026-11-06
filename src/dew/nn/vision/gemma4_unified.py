"""Encoder-free Gemma 4 media embeddings, from transformers 5.17.0's gemma4_unified.

The processor merges image patches before the axial embedding. Vision states and raw waveform
frames then use the same scale-free RMS projector, before entering the shared text decoder.
"""

import dataclasses
import functools
from collections.abc import Mapping

from flax import linen as nn

from dew import _records
from dew.interop.config_records import NativeFields
from dew.interop.layouts import ConfigField, ConfigLayout, Table
from dew.nn.norms import LayerNorm
from dew.registry import Record

from .common import TowerBase, TowerGeometry, TowerLayout, tensor_routes
from .gemma4 import AxialPatchEmbedding


@dataclasses.dataclass(frozen=True)
class Gemma4UnifiedVision(TowerBase):
    """Merged-patch geometry and the learned per-axis position table."""

    patch_size: int = 16
    pooling_kernel_size: int = 3
    mm_embed_dim: int = 3840
    mm_posemb_size: int = 1120

    def build(self) -> nn.Module:
        return AxialPatchEmbedding(
            self.mm_embed_dim, self.patch_size * self.pooling_kernel_size, self.mm_posemb_size,
            use_bias=True, table_axis=1, position_validity="axis",
            norm=functools.partial(LayerNorm, epsilon=1e-5, use_fast_variance=False),
            initializer=nn.initializers.zeros)

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size * self.pooling_kernel_size, block_size=1,
                             channels=3, position_size=self.mm_posemb_size, retains_patch_slots=True)


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


Gemma4UnifiedVision.checkpoint = TowerLayout("vision_embedder.", Table(*tensor_routes(_TENSORS)),
                                             config=_UNIFIED_CONFIG, model_type="gemma4_unified_vision")
