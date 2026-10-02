"""Model backbones.

Code builds a model by its class: `SimpleDiT(patch_size=4, ...)`. Each class
also registers itself with `dew.registry.models` where it is defined, so a
run record that names `simple_dit` and its fields rebuilds the same model.
"""

from .causal_transformer import CausalTransformer
from .decoder_block import Mixture
from .dit import SimpleDiT
from .edm2 import EDM2UNet
from .flux import FluxTransformer
from .flux2 import Flux2Transformer
from .mmdit import HierarchicalMMDiT, SimpleMMDiT
from .qwen_image import QwenImageTransformer
from .sd3 import SD3Transformer
from .ssm_dit import HybridSSMAttentionDiT
from .unet import Unet
from .unet3d import UNet3D
from .unet_condition import UNet2DCondition, UNetStage
from .uvit import SimpleUDiT, UViT
from .video_dit import VideoDiT
from .z_image import ZImageTransformer

__all__ = ["CausalTransformer", "EDM2UNet", "Flux2Transformer", "FluxTransformer", "HierarchicalMMDiT",
           "HybridSSMAttentionDiT", "Mixture", "QwenImageTransformer", "SD3Transformer", "SimpleDiT",
           "SimpleMMDiT", "SimpleUDiT", "UNet2DCondition", "UNet3D", "UNetStage", "UViT", "Unet", "VideoDiT",
           "ZImageTransformer"]
