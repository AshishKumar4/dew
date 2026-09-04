"""Model backbones.

Each class registers itself with `dew.registry.models` where it is defined,
so `models["simple_dit"]`, `models.SimpleDiT` and the class are one object and
training and inference build the same model from a logged name and fields.
The classes are exported for direct use in notebooks and tests.
"""

from .unet import Unet
from .uvit import UViT, SimpleUDiT
from .dit import SimpleDiT
from .causal_transformer import CausalTransformer
from .mmdit import SimpleMMDiT, HierarchicalMMDiT
from .ssm_dit import HybridSSMAttentionDiT
from .video_dit import VideoDiT
from .unet3d import UNet3D

__all__ = [
    # Image backbones
    "Unet",
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
