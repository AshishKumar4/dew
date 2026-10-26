"""Encoder-free Gemma 4 media embeddings, from transformers 5.16.1's gemma4_unified.

The processor merges image patches before the axial embedding. Vision states and raw waveform
frames then use the same scale-free RMS projector, before entering the shared text decoder.
"""

import dataclasses
import functools
from collections.abc import Mapping

from flax import linen as nn

from dew import _records
from dew.interop.weights import in_params
from dew.registry import Record

from .common import TowerBase, TowerGeometry, TowerLayout
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
            norm=functools.partial(nn.LayerNorm, epsilon=1e-5, use_fast_variance=False),
            initializer=nn.initializers.zeros)

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size * self.pooling_kernel_size, block_size=1,
                             channels=3, position_size=self.mm_posemb_size, retains_patch_slots=True)


def translate_vision(config: Mapping[str, object]) -> Record:
    fields = {field.name: _records.integer(config.get(field.name, field.default), field.name)
              for field in dataclasses.fields(Gemma4UnifiedVision)}
    width = fields["mm_embed_dim"]
    if _records.integer(config.get("output_proj_dims", width), "output_proj_dims") != width:
        raise ValueError("output_proj_dims must equal mm_embed_dim for the vision embedding projection")
    return {"class": "gemma4_unified", "fields": fields}


def translate_projector(config: Mapping[str, object], text_width: int, input_width: int) -> Record:
    return {"class": "gemma4", "fields": {
        "text_width": text_width, "input_features": input_width,
        "norm_eps": _records.number(config.get("rms_norm_eps", 1e-6), "rms_norm_eps")}}


_TENSORS = {
    "patch_ln1.weight": ("patch_norm1", "scale"), "patch_ln1.bias": ("patch_norm1", "bias"),
    "patch_dense.weight": ("patch_embed", "kernel"), "patch_dense.bias": ("patch_embed", "bias"),
    "patch_ln2.weight": ("patch_norm2", "scale"), "patch_ln2.bias": ("patch_norm2", "bias"),
    "pos_embedding": ("position_table",),
    "pos_norm.weight": ("position_norm", "scale"), "pos_norm.bias": ("position_norm", "bias"),
}


def vision_path(name: str) -> tuple[str, ...]:
    if name not in _TENSORS:
        raise ValueError(f"unknown unified vision tensor {name!r}")
    return _TENSORS[name]


Gemma4UnifiedVision.checkpoint = TowerLayout("vision_embedder.", in_params(vision_path))
