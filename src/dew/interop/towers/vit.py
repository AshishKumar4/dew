"""A ViT's patch convolution as a release stores it, folded into the Dense map the tower holds, both
ways (`translate_vit_weights`), for the towers whose patch embedding is a convolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dew.interop.unread import Stored, held
from dew.interop.weights import Tree, checkpoint_array, insert, placed
from dew.nn.vision.common import TowerBase

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

PATCH_CONVOLUTION = "patch_embed.proj.weight"
"""Where the Qwen-style towers keep their patch map: a `[out, in, time, h, w]` convolution, or MoonViT's
`[out, in, h, w]`."""


@runtime_checkable
class PatchedTower(Protocol):
    """A tower whose release stores its patch map as a `[width, channels, time, patch, patch]` convolution
    (`patch_convolution`)."""

    hidden_size: int
    in_channels: int
    temporal_patch_size: int
    patch_size: int


def patch_convolution(tower: TowerBase) -> Mapping[str, tuple[int, ...]]:
    """The patch map as a Qwen-style release stores it (`PATCH_CONVOLUTION`), the convolution
    `translate_vit_weights` folds into the tower's Dense map: `[width, channels, time, patch, patch]`, a
    tower's `TowerLayout.shapes`."""
    if not isinstance(tower, PatchedTower):
        raise TypeError(f"{type(tower).__name__} states no patch convolution's geometry")
    return {PATCH_CONVOLUTION: (tower.hidden_size, tower.in_channels, tower.temporal_patch_size,
                                tower.patch_size, tower.patch_size)}


def translate_vit_weights(hf_tensors: Mapping[str, Stored],
                          path_of: Callable[[str], tuple[str, ...] | None],
                          param_dtype: str = "float32") -> tuple[Tree[Stored], tuple[WeightLayout, ...]]:
    """A trunk's variables from its tensors through its family's `path_of`, rooted at the collection,
    the patch convolution `PATCH_CONVOLUTION`, where the checkpoint has one, landing as one Dense map
    over the processor's channel, time and pixel order at `params/patch_embed/kernel`; with the
    layouts that write each tensor back from them (`placed`)."""
    from dew.interop.streaming import WeightLayout

    variables, layouts = placed({name: tensor for name, tensor in hf_tensors.items()
                                 if name != PATCH_CONVOLUTION}, path_of, param_dtype)
    if PATCH_CONVOLUTION in hf_tensors:
        conv = checkpoint_array(hf_tensors[PATCH_CONVOLUTION], param_dtype)
        dense = held(conv.transpose(*range(1, conv.ndim), 0).reshape(-1, conv.shape[0]), contiguous=True)
        insert(variables, ("params", "patch_embed", "kernel"), dense, PATCH_CONVOLUTION)
        # The Dense map's transpose is the convolution, its input axes flattened.
        kernel = ("params", "patch_embed", "kernel")
        layouts += (WeightLayout(PATCH_CONVOLUTION, (kernel,), conv.shape, (1, 0)),)
    return variables, layouts

