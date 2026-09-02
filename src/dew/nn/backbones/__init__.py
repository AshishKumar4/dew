"""Model backbones.

`dew.registry.build_model` is the primary entry point: training and inference both
construct models from a logged architecture name plus config, so a checkpoint
always reconstructs the model it was trained with. The classes are exported for
direct use in notebooks and tests.
"""

from typing import Any

from .unet import Unet
from .uvit import UViT, SimpleUDiT
from .dit import SimpleDiT
from .causal_transformer import CausalTransformer
from .mmdit import SimpleMMDiT, HierarchicalMMDiT
from .ssm_dit import HybridSSMAttentionDiT
from .video_dit import VideoDiT
from .unet3d import UNet3D

__all__ = [
    # Primary construction API
    "build_model",
    "canonicalize_architecture",
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

_REGISTRY_EXPORTS = ("build_model", "canonicalize_architecture", "MODEL_REGISTRY")


def __getattr__(name: str) -> Any:
    """Resolve the registry exports on first use.

    dew.registry builds its table out of the backbone modules in this package,
    so importing it eagerly here would close that loop and break
    `import dew.nn.backbones.dit` on a cold interpreter. Deferring it keeps
    `from dew.nn.backbones import build_model` working without the cycle.
    """
    if name in _REGISTRY_EXPORTS:
        from dew import registry
        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
