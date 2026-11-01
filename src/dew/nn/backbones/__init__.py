"""Model backbones.

You build a model by calling its class, as in `SimpleDiT(patch_size=4, ...)`.
A run record names the class by its import path, and `dew.registry.models`
holds short aliases such as `simple_dit`, so either and its fields rebuild
the same model.

Each backbone is imported only when it is used (`dew._exports`), so a module that reads one
backbone imports that backbone alone."""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_block import Mixture
    from dew.nn.backbones.dit import SimpleDiT
    from dew.nn.backbones.edm2 import EDM2UNet
    from dew.nn.backbones.flux import FluxTransformer
    from dew.nn.backbones.flux2 import Flux2Transformer
    from dew.nn.backbones.mmdit import HierarchicalMMDiT, SimpleMMDiT
    from dew.nn.backbones.qwen_image import QwenImageTransformer
    from dew.nn.backbones.sd3 import SD3Transformer
    from dew.nn.backbones.ssm_dit import HybridSSMAttentionDiT
    from dew.nn.backbones.unet import Unet
    from dew.nn.backbones.unet3d import UNet3D
    from dew.nn.backbones.unet_condition import UNet2DCondition, UNetStage
    from dew.nn.backbones.uvit import SimpleUDiT, UViT
    from dew.nn.backbones.video_dit import VideoDiT
    from dew.nn.backbones.wan import WanTransformer
    from dew.nn.backbones.z_image import ZImageTransformer

_EXPORTS = {
    "CausalTransformer": "dew.nn.backbones.causal_transformer",
    "Mixture": "dew.nn.backbones.decoder_block",
    "SimpleDiT": "dew.nn.backbones.dit",
    "EDM2UNet": "dew.nn.backbones.edm2",
    "FluxTransformer": "dew.nn.backbones.flux",
    "Flux2Transformer": "dew.nn.backbones.flux2",
    "HierarchicalMMDiT": "dew.nn.backbones.mmdit",
    "SimpleMMDiT": "dew.nn.backbones.mmdit",
    "QwenImageTransformer": "dew.nn.backbones.qwen_image",
    "SD3Transformer": "dew.nn.backbones.sd3",
    "HybridSSMAttentionDiT": "dew.nn.backbones.ssm_dit",
    "Unet": "dew.nn.backbones.unet",
    "UNet3D": "dew.nn.backbones.unet3d",
    "UNet2DCondition": "dew.nn.backbones.unet_condition",
    "UNetStage": "dew.nn.backbones.unet_condition",
    "SimpleUDiT": "dew.nn.backbones.uvit",
    "UViT": "dew.nn.backbones.uvit",
    "VideoDiT": "dew.nn.backbones.video_dit",
    "WanTransformer": "dew.nn.backbones.wan",
    "ZImageTransformer": "dew.nn.backbones.z_image",
}


__getattr__ = _lookup(__name__, _EXPORTS)


def __dir__() -> list[str]:
    return list(__all__)


__all__ = [
    "CausalTransformer",
    "EDM2UNet",
    "Flux2Transformer",
    "FluxTransformer",
    "HierarchicalMMDiT",
    "HybridSSMAttentionDiT",
    "Mixture",
    "QwenImageTransformer",
    "SD3Transformer",
    "SimpleDiT",
    "SimpleMMDiT",
    "SimpleUDiT",
    "UNet2DCondition",
    "UNet3D",
    "UNetStage",
    "UViT",
    "Unet",
    "VideoDiT",
    "WanTransformer",
    "ZImageTransformer",
]
