"""What every vision family shares: the tower and projector bases `dew.registry`'s kinds name, and the
geometry a conditioner reads off a tower. Where a release stores a kind's tensors and config is
`dew.interop.towers`'.
"""

from __future__ import annotations

import dataclasses

from flax import linen as nn

PIXEL_VALUES_KEY = "pixel_values"
"""The batch field carrying images as the checkpoint's processor emitted them."""


class ProjectorBase(nn.Module):
    """A native projector module whose constructor is its record schema. The registry names its kind;
    where a release stores its tensors is `dew.interop.towers`'."""

    @property
    def input_width(self) -> int | None:
        """Declared feature width when this projector consumes raw media without an encoder."""
        return None


def head_width(hidden_size: int, num_heads: int) -> int:
    """Each head's width where `num_heads` split a tower's `hidden_size`
    evenly, and a refusal where they leave a remainder."""
    if hidden_size % num_heads:
        raise ValueError(f"hidden_size ({hidden_size}) must split over num_heads ({num_heads})")
    return hidden_size // num_heads


@dataclasses.dataclass(frozen=True)
class TowerGeometry:
    """The shapes a conditioner has to invent to create a tower's media leaves.

    A token-only init has no batch, so `VisionConditioner` and
    `AudioConditioner` build one smallest input the tower accepts. What that
    is differs by kind: a fixed-resolution tower states its `image_size`, a
    patch-and-pool tower states the patch and block that make one, and an
    audio tower states how many mel bins a frame carries. None means this
    tower has no such field, and the conditioner uses its own default.
    """

    image_size: int | None = None
    patch_size: int | None = None
    block_size: int | None = None
    channels: int | None = None
    mel_features: int | None = None
    frames_per_token: int | None = None
    """How many input frames an audio tower encodes into one token: its convolutions' time strides
    times any frame reduction after them."""
    position_size: int | None = None
    retains_patch_slots: bool = False
    pools_frames: bool = False
    """Whether an item's frames pool into one frame's tokens (MoonViT's `tpool_patch_merger`): its tokens
    are its first frame's merge blocks, wherever it holds more."""


class TowerBase:
    """One tower kind's value: its fields and how it builds its module. Where a release stores its tensors
    is `dew.interop.towers`'."""

    def build(self) -> nn.Module:
        """The Flax module for this value."""
        raise NotImplementedError(
            f"{type(self).__name__} names a tower kind but builds no module")

    def geometry(self) -> TowerGeometry:
        """What an initialising input has to look like for this tower.

        Nothing by default: a tower states the fields it has, and a
        conditioner reads them here instead of asking the value at runtime
        whether it carries each one.
        """
        return TowerGeometry()


