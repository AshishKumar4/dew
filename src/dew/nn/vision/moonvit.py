"""Kimi K2.5's and Kimi K3's vision trunk (MoonViT) and projector, with their checkpoint maps
(moonshotai/Kimi-K2.5@4d01dfe `modeling_kimi_k25.py`: MoonViT3dPretrainedModel, PatchMergerMLP;
moonshotai/Kimi-K3@f831ab6 `modeling_kimi_k3.py`, its options: RMS norms, bias-free maps, a wider
attention, PatchMergerMLPV2).
"""

import dataclasses
from typing import Literal

from flax import linen as nn

from .common import ProjectorBase, TowerBase, TowerGeometry, head_width
from .vit import LearnedPositions, MergeProjector, VisionTransformer, ViTLayout


@dataclasses.dataclass(frozen=True)
class MoonViT(TowerBase):
    """MoonViT's geometry under its vision_config's names.

    Its layout (`ViTLayout`): a convolution of each raster 14x14 patch, its channels then its pixels
    (`MoonVision3dPatchEmbed`), plus the learned `init_pos_emb_height` square table resampled to the
    frame's patch grid as `F.interpolate` resamples it (`pos_emb_interpolation_mode`, half-pixel centres,
    edges held) and, for an item of more than one frame, a fixed sinusoidal table of
    `init_pos_emb_time` frames (`Learnable2DInterpPosEmbDivided_fixed`); pre-norm blocks over every patch
    of an item, its frames together (`spatial_temporal`), their queries, keys and values one map
    rotated in adjacent pairs by the patch grid, a column's pair then a row's (`Rope2DPosEmbRepeated`),
    and an ungated tanh-GELU feed-forward (`MLP2`); a norm after the blocks; then each merge block's
    patches averaged over the item's frames (`tpool_patch_merger`, `sd2_tpool`). K2.5's norms are layer
    norms and its maps biased; K3 states RMS norms (`norm_type`), each at `rms_eps`, by default (None) the
    epsilon of the dtype it accumulates in as torch's RMSNorm(eps=None) takes it (float32's, float64's in
    float64), unbiased maps and a wider attention (`qkv_hidden_size`).
    """

    vt_hidden_size: int = 1152
    vt_intermediate_size: int = 4304
    vt_num_attention_heads: int = 16
    vt_num_hidden_layers: int = 27
    patch_size: int = 14
    merge_kernel_size: int = 2
    init_pos_emb_height: int = 64
    init_pos_emb_time: int = 4
    pos_emb_interpolation_mode: Literal["bicubic", "bilinear"] = "bicubic"
    qkv_hidden_size: int | None = None
    attn_bias: bool = True
    linear_bias: bool = True
    patch_embed_proj_bias: bool = True
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    rms_eps: float | None = None

    def __post_init__(self):
        if self.pos_emb_interpolation_mode not in ("bicubic", "bilinear"):
            raise ValueError(f"pos_emb_interpolation_mode {self.pos_emb_interpolation_mode!r} is neither "
                             "bicubic nor bilinear")
        if self.norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError(f"norm_type {self.norm_type!r} is neither layernorm nor rmsnorm")

    def layout(self) -> ViTLayout:
        depth, heads = self.vt_num_hidden_layers, self.vt_num_attention_heads
        width = self.vt_hidden_size if self.qkv_hidden_size is None else self.qkv_hidden_size
        rms = self.norm_type == "rmsnorm"
        # torch's LayerNorm at its default epsilon (K2.5), or its RMSNorm at its accumulating dtype's (K3).
        eps = self.rms_eps if rms else 1e-5
        return ViTLayout(
            depth=depth, hidden_size=self.vt_hidden_size, num_heads=heads, num_kv_heads=heads,
            head_dim=head_width(width, heads), intermediate_size=self.vt_intermediate_size,
            activation="gelu", in_channels=3, patch_size=self.patch_size, temporal_patch_size=1,
            spatial_merge_size=self.merge_kernel_size, norm_type="rms" if rms else "layer", norm_eps=eps,
            norm_bias=not rms,
            mlp_bias=self.linear_bias, packed=True, patch_bias=self.patch_embed_proj_bias,
            attention=("item",) * depth, order=("patches",) * depth, sinks=(False,) * depth,
            positions=LearnedPositions(self.init_pos_emb_height, mode=self.pos_emb_interpolation_mode,
                                       align_corners=False, border=True,
                                       initializer=nn.initializers.normal(1.0)),
            rotary="adjacent", rotary_interleaved=True, rotary_columns_first=True,
            post_norm="rms" if rms else "layer", outer_norm_eps=eps, attention_bias=self.attn_bias,
            merge_blocks=False,
            frame_positions=self.init_pos_emb_time, pool_frames=True)

    def build(self) -> nn.Module:
        return VisionTransformer(self.layout())

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.merge_kernel_size, channels=3,
                             pools_frames=True)


class MoonViTProjector(MergeProjector, ProjectorBase):
    """MoonViT's projector (`PatchMergerMLP`): a layer norm of each trunk token at `projector_ln_eps`,
    each merge block's tokens joined unit by unit, and two biased maps around an exact GELU, the first
    as wide as the block, the second to the decoder width."""


class MoonViTProjectorV2(MergeProjector, ProjectorBase):
    """Kimi K3's projector (`PatchMergerMLPV2`): each merge block's trunk tokens joined unit by unit, two
    unbiased maps around an exact GELU, the first as wide as the block, the second to the decoder width,
    and an RMS norm after them at `projector_ln_eps`; no norm before."""



















