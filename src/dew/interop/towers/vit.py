"""A ViT's patch convolution as a release stores it, folded into the Dense map the tower holds, both
ways (`translate_vit_weights`), for the towers whose patch embedding is a convolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from dew.interop.unread import Stored, held
from dew.interop.weights import Tree, checkpoint_array, insert, placed

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

PATCH_CONVOLUTION = "patch_embed.proj.weight"
"""Where the Qwen-style towers keep their patch map: a `[out, in, time, h, w]` convolution, or MoonViT's
`[out, in, h, w]`."""


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

