"""LFM3-VL's vision trunk and projector, with their checkpoint maps (LiquidAI/LFM3-VL-2B@f83d2b0,
`modeling_lfm3_vl.py`: Lfm3VlVisionModel and Lfm3VlMultiModalProjector).
"""

import dataclasses

from flax import linen as nn

from .common import ProjectorBase, TowerBase, TowerGeometry, head_width
from .vit import MergeProjector, VisionTransformer, ViTLayout


def default_layer_types(layers: int) -> tuple[str, ...]:
    """LFM3-VL's cadence over `layers` layers: two convolutions, then attention and convolution in turn
    (Lfm3VlVisionConfig's default `layer_types`)."""
    return tuple("conv" if i < 2 or i % 2 else "full_attention" for i in range(layers))


@dataclasses.dataclass(frozen=True)
class Lfm3VlVision(TowerBase):
    """LFM3-VL's trunk geometry under its vision_config's names.

    Its layout (`ViTLayout`): a bias-free map of each patch's channel-then-time pixels in merge-block
    order (`patchifier`), an RMS norm after it (`embedding_norm`), and pre-norm blocks of RMS norms at
    `norm_eps` with a bias-free SwiGLU feed-forward (`w1` its gate, `w3` its up map), each mixing its
    frame's patches by full attention, grouped and unbiased with an RMS norm on each head's queries and
    keys before the rotary turns them by the patch grid, rows then columns (`full_attention`), or by a
    gated depthwise convolution over the frame's grid (`conv`, `GatedGridConv`); no norm after the
    blocks. Every RMS norm weighs its input after its cast back, as the release's `Lfm3VlRMSNorm` does.
    `layer_types` None is the release's cadence over its depth (`default_layer_types`).
    """

    hidden_size: int = 1024
    intermediate_size: int = 2560
    num_hidden_layers: int = 14
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    conv_kernel_size: int = 3
    norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    layer_types: tuple[str, ...] | None = None

    def __post_init__(self):
        unknown = sorted(set(self.kinds) - {"conv", "full_attention"})
        if unknown:
            raise ValueError(f"layer_types {unknown} are neither conv nor full_attention")
        if len(self.kinds) != self.num_hidden_layers:
            raise ValueError(f"layer_types names {len(self.kinds)} layers of {self.num_hidden_layers}")

    @property
    def kinds(self) -> tuple[str, ...]:
        """Each layer's kind: `layer_types`, or the release's cadence over the depth."""
        return default_layer_types(self.num_hidden_layers) if self.layer_types is None else self.layer_types

    def layout(self) -> ViTLayout:
        depth = self.num_hidden_layers
        return ViTLayout(
            depth=depth, hidden_size=self.hidden_size, num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            head_dim=head_width(self.hidden_size, self.num_attention_heads),
            intermediate_size=self.intermediate_size, activation="swiglu", in_channels=self.in_channels,
            patch_size=self.patch_size, temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.spatial_merge_size, norm_type="rms", norm_eps=self.norm_eps,
            norm_bias=False, mlp_bias=False, packed=False, patch_bias=False,
            attention=tuple("conv" if kind == "conv" else "full" for kind in self.kinds),
            order=("patches",) * depth, sinks=(False,) * depth, pre_norm="rms", outer_norm_eps=self.norm_eps,
            rope_theta=self.rope_theta, attention_bias=False, qk_norm=True, conv_kernel=self.conv_kernel_size,
            scale_after_cast=True)

    def build(self) -> nn.Module:
        return VisionTransformer(self.layout())

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.spatial_merge_size,
                             channels=self.in_channels)


class Lfm3VlProjector(MergeProjector, ProjectorBase):
    """LFM3-VL's projector (Lfm3VlMultiModalProjector) is `MergeProjector`: a layer norm of each
    trunk token at `projector_layer_norm_eps`, its bias kept whatever `projector_bias` says, each merge
    block's tokens joined unit by unit, and two maps, biased by `projector_bias`, around the
    `projector_hidden_act` activation."""




















