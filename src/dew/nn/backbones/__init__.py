"""Model backbones.

Each class registers itself with `dew.registry.models` where it is defined,
so `models["simple_dit"]`, `models.SimpleDiT` and the class are one object and
training and inference build the same model from a logged name and fields.
The classes are exported for direct use in notebooks and tests.
"""

from .causal_transformer import CausalTransformer
from .dit import SimpleDiT
from .mmdit import HierarchicalMMDiT, SimpleMMDiT
from .ssm_dit import HybridSSMAttentionDiT
from .unet import Unet
from .unet3d import UNet3D
from .unet_condition import UNet2DCondition, UNetStage
from .uvit import SimpleUDiT, UViT
from .video_dit import VideoDiT

__all__ = [
    # Image backbones
    "Unet",
    "UNet2DCondition", "UNetStage",
    "UViT",
    "SimpleUDiT",
    "SimpleDiT",
    "SimpleMMDiT",
    "HierarchicalMMDiT",
    "HybridSSMAttentionDiT",
    # Language backbones
    "CausalTransformer",
    # Video backbones
    "VideoDiT",
    "UNet3D",
]
