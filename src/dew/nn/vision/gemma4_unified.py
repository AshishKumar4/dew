"""Encoder-free Gemma 4 media embeddings, from transformers 5.17.0's gemma4_unified.

The processor merges image patches before the axial embedding. Vision states and raw waveform
frames then use the same scale-free RMS projector, before entering the shared text decoder.
"""

import dataclasses
import functools

from flax import linen as nn

from dew.nn.norms import LayerNorm

from .common import TowerBase, TowerGeometry
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








