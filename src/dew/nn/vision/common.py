"""What every vision family shares: the tower and projector bases, the
geometry a conditioner reads off a tower, the vision-config readers and the
projector tensor table.
"""

import dataclasses
from collections.abc import Mapping

from flax import linen as nn

from dew import records

PIXEL_VALUES_KEY = "pixel_values"
"""The batch field carrying images as the checkpoint's processor emitted them."""


class ProjectorBase:
    """One projector kind's value: its fields, and how it builds its module.

    Each kind is a frozen dataclass of the reference's field names, with an
    alias in `dew.registry.projectors` (`"gemma"`). `build` turns the value
    into the Flax module. A record that names no known kind raises.
    """

    def build(self) -> nn.Module:
        """The Flax module for this value."""
        raise NotImplementedError(
            f"{type(self).__name__} names a projector kind but builds no module")


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


class TowerBase:
    """One tower kind's value: its fields, and how it builds its module."""

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


_PROJECTOR_PATHS: dict[str, dict[str, tuple[str, ...]]] = {
    "gemma": {
        "mm_soft_emb_norm.weight": ("mm_soft_emb_norm", "scale"),
        "mm_input_projection_weight": ("mm_input_projection", "kernel"),
    },
    "llama4": {"linear_1.weight": ("linear", "kernel")},
    "gemma4": {"embedding_projection.weight": ("projection", "kernel")},
    "qwen3_5": {
        "norm.weight": ("norm", "scale"), "norm.bias": ("norm", "bias"),
        "linear_fc1.weight": ("fc1", "kernel"), "linear_fc1.bias": ("fc1", "bias"),
        "linear_fc2.weight": ("fc2", "kernel"), "linear_fc2.bias": ("fc2", "bias"),
    },
    "gemma3n": {
        "embedding.weight": ("embedding", "embedding"),
        "hard_embedding_norm.weight": ("hard_embedding_norm", "scale"),
        "soft_embedding_norm.weight": ("soft_embedding_norm", "scale"),
        "embedding_projection.weight": ("embedding_projection", "kernel"),
    },
    # The aligner's maps under its `aligner.` prefix, and the span's learned
    # vectors, which the release keeps at its top level (model.py:1215-1222).
    "deepseek_v41": {
        "w1.weight": ("w1", "kernel"), "w1.bias": ("w1", "bias"),
        "w2.weight": ("w2", "kernel"), "w2.bias": ("w2", "bias"),
        "image_start": ("image_start",), "image_newline": ("image_newline",),
        "image_end": ("image_end",),
    },
}


def projector_weight_path(kind: str, name: str) -> tuple[str, ...]:
    """The projector's canonical tensor path, shared by load and source export."""
    if kind == "qwen3_5":
        name = name.removeprefix("merger.")
    if kind not in _PROJECTOR_PATHS or name not in _PROJECTOR_PATHS[kind]:
        raise ValueError(f"unknown {kind} projector tensor {name!r}")
    return _PROJECTOR_PATHS[kind][name]


def _vision_section(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    """The vision half of a wrapper config, or a bare vision config."""
    return records.record(hf_config.get("vision_config", hf_config), "vision_config")


def _image_size(value: object, field: str) -> int:
    """A square image size as one side: an int, or a pair with equal sides."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(
                f"{field} {list(value)!r} is not square, this trunk tiles squares")
        value = value[0]
    return records.integer(value, field)
