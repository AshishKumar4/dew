"""What every vision family shares: the tower and projector bases, the
geometry a conditioner reads off a tower, the vision-config readers, and where
a kind's tensors sit in a media checkpoint (`TowerLayout`, `ProjectorLayout`),
which each family's class holds and `dew.registry`'s kinds name.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Collection, Mapping
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from flax import linen as nn

from dew import records
from dew.interop.decoder_parts import read_rope_theta
from dew.interop.weights import placed
from dew.objectives.base import Variables
from dew.registry import Record, projectors, towers

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

type Reader = Callable[[Mapping[str, np.ndarray], Callable[[str], tuple[str, ...] | None], str],
                       tuple[Variables, tuple[WeightLayout, ...]]]
"""A tower's tensors into its variables in a storage dtype, with the layouts that write them back:
`dew.interop.weights.placed`, or a patch-folding ViT's `translate_vit_weights`."""


@dataclasses.dataclass(frozen=True)
class TowerLayout:
    """Where one tower kind's tensors sit in a media checkpoint and the leaves they fill: `prefix`
    before every name, and `path` from a name past it to its leaf, rooted at its collection
    (`params`, and Gemma 4's `constants` beside it). `read` binds them (`placed`, or, for a tower that
    runs its patch convolution as one Dense map, `translate_vit_weights`). Loading
    (`tower_variables`), routing a checkpoint's names (`hf_decoders`) and binding an export
    (`interop.pretrained`) read it alike."""

    prefix: str
    path: Callable[[str], tuple[str, ...] | None]
    read: Reader = placed

    def variables(self, hf_tensors: Mapping[str, np.ndarray], param_dtype: str
                  ) -> tuple[Variables, tuple[WeightLayout, ...]]:
        """The tower's variables from its tensors, in the requested storage, and the layouts that
        write each tensor it placed back from them; a tensor the tower does not compute with, as
        SigLIP's pooling head, has none."""
        return self.read(hf_tensors, self.path, param_dtype)



@dataclasses.dataclass(frozen=True)
class ProjectorLayout:
    """Where one projector kind's tensors sit in a media checkpoint and the leaves they fill:
    `prefix` before every name (or several, whose tensors each keep the name past their own),
    `tensors` each name's leaf path, `strip` a further prefix a name may carry (a merger's), and
    `read` the projector's own reader where it checks its tensors (`placed` otherwise)."""

    prefix: str | tuple[str, ...]
    tensors: Mapping[str, tuple[str, ...]]
    strip: str = ""
    read: Callable[..., tuple[Variables, tuple[WeightLayout, ...]]] | None = None

    def path(self, name: str) -> tuple[str, ...]:
        """The leaf of tensor `name`, shared by load and source export."""
        name = name.removeprefix(self.strip) if self.strip else name
        if name not in self.tensors:
            raise ValueError(f"unknown projector tensor {name!r}")
        return self.tensors[name]

    def variables(self, hf_tensors: Mapping[str, np.ndarray], param_dtype: str
                  ) -> tuple[Variables, tuple[WeightLayout, ...]]:
        """The projector's parameters, in the requested storage, and the layouts that write its
        tensors back from them."""
        if self.read is not None:
            return self.read(hf_tensors, param_dtype=param_dtype)
        return placed(hf_tensors, self.path, param_dtype)

PIXEL_VALUES_KEY = "pixel_values"
"""The batch field carrying images as the checkpoint's processor emitted them."""


class ProjectorBase:
    """One projector kind's value: its fields, and how it builds its module.

    Each kind is a frozen dataclass of the reference's field names, with an
    alias in `dew.registry.projectors` (`"gemma"`). `build` turns the value
    into the Flax module, and `checkpoint` says where its tensors sit. A
    record that names no known kind raises.
    """

    checkpoint: ClassVar[ProjectorLayout | None] = None

    def build(self) -> nn.Module:
        """The Flax module for this value."""
        raise NotImplementedError(
            f"{type(self).__name__} names a projector kind but builds no module")


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


class TowerBase:
    """One tower kind's value: its fields, how it builds its module, and where its tensors sit in a
    media checkpoint (`checkpoint`, None for a tower read another way, as the audio towers are)."""

    checkpoint: ClassVar[TowerLayout | None] = None

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


_BLOCK_NORMS = {"norm1": "input_layernorm", "norm2": "post_attention_layernorm"}
"""A ViT block's two norms as `clip_layer`'s decoder block names them."""


def block_tensor_path(parts: list[str], maps: Mapping[tuple[str, str], tuple[str, ...]], *,
                      prefix: str = "blocks", norm_bias: bool = True, block: str = "blocks",
                      norms: Mapping[str, str] = _BLOCK_NORMS) -> tuple[str, ...] | None:
    """An indexed block's tensor, `{prefix}.N.<norm>.weight|bias` or
    `{prefix}.N.<module>.<linear>.weight|bias`, its name split on dots, into its path under
    `{block}_N`: `norms` names each norm's module (a shared-trunk ViT's `norm1` and `norm2` by
    default), `maps` places each release's (module, linear) pair, and a release whose norms carry
    no bias says so (`norm_bias`). `prefix` may hold dots (a text encoder's `encoder.layers`). None
    for any other name, which the family reads or refuses."""
    root = prefix.split(".")
    depth = len(root)
    if (len(parts) not in (depth + 3, depth + 4) or parts[:depth] != root or not parts[depth].isdigit()
            or parts[-1] not in ("weight", "bias")):
        return None
    name, module, leaf = f"{block}_{parts[depth]}", parts[depth + 1], parts[-1]
    if len(parts) == depth + 3 and module in norms and (norm_bias or leaf == "weight"):
        return (name, norms[module], "scale" if leaf == "weight" else "bias")
    if len(parts) == depth + 4 and (module, parts[depth + 2]) in maps:
        return (name, *maps[module, parts[depth + 2]], "kernel" if leaf == "weight" else "bias")
    return None


def tower_layout(kind: str) -> TowerLayout:
    """Where tower kind `kind`'s tensors sit: its class's `checkpoint` (`dew.registry.towers`)."""
    return _held(towers[kind].checkpoint if kind in towers else None, "tower", kind)


def projector_layout(kind: str) -> ProjectorLayout:
    """Where projector kind `kind`'s tensors sit: its class's `checkpoint`
    (`dew.registry.projectors`)."""
    return _held(projectors[kind].checkpoint if kind in projectors else None, "projector", kind)


def _held[Layout](layout: Layout | None, what: str, kind: str) -> Layout:
    if layout is None:
        raise ValueError(f"{what} kind {kind!r} has no weight map here")
    return layout


def _vision_section(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    """The vision half of a wrapper config, or a bare vision config."""
    return records.record(hf_config.get("vision_config", hf_config), "vision_config")


def merger_projector(hf_config: Mapping[str, object], text_width: int, kind: str) -> Record:
    """The `kind` projector record of a wrapper whose vision tower ends in its own merger
    (`out_hidden_size`, `spatial_merge_size`), as Qwen 3.5's and MiMo-V2's do: the merged
    features enter the text embeddings as they are, so a merger width beside the decoder width
    refuses."""
    vision = _vision_section(hf_config)
    merged = records.integer(vision["out_hidden_size"], "out_hidden_size")
    if merged != int(text_width):
        raise ValueError(f"out_hidden_size ({merged}) is not the decoder width ({text_width}), "
                         "the merger output enters the text embeddings as it is")
    return {"class": kind, "fields": {
        "vision_width": records.integer(vision["hidden_size"], "hidden_size"),
        "merge_size": records.integer(vision.get("spatial_merge_size", 2), "spatial_merge_size"),
        "out_width": merged,
    }}


def _side(value: object, field: str) -> int:
    """A square image or patch size as one side: an int, or a pair with equal sides."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(
                f"{field} {list(value)!r} is not square, this trunk tiles squares")
        value = value[0]
    return records.integer(value, field)


def trunk_geometry(vision: Mapping[str, object]) -> dict[str, int]:
    """The width, MLP width, depth and head count a ViT config states, under
    the trunk fields of the families that share the reference's names."""
    return {
        "hidden_size": records.integer(vision["hidden_size"], "hidden_size"),
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_layers": records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        "num_heads": records.integer(vision["num_attention_heads"], "num_attention_heads"),
    }


def activation_name(config: Mapping[str, object], runs: Collection[str], default: str,
                    fields: tuple[str, ...] = ("hidden_act",)) -> str:
    """The activation the first of `fields` that `config` states names, else
    the family's `default`; a name outside `runs`, the activations its trunk
    computes, refuses."""
    name = str(next((config[field] for field in fields if field in config), default))
    if name not in runs:
        raise ValueError(f"{fields[0]} {name!r} is not expressible: this trunk runs {', '.join(runs)}")
    return name




def rope_base(vision: Mapping[str, object], default: float, *, axial: bool = False) -> float:
    """A trunk's plain rotary base: `rope_parameters`' rope_theta, else the
    flat field, else the family's `default`. A scaled rope refuses, as any
    rope does where only plain rotary has a place (`read_rope_theta`), and so
    does the decoders' 'none' alias: the towers dispatch any type but
    'default' to Transformers' rope initializers, which name no 'none'
    (modeling_gemma4.py:716-719). An `axial` tower's rotary, its grid's rows
    and columns each over half the head, is named 'axial' too: Transformers
    5.17 writes that name where earlier releases wrote 'default', and reads
    either as the one its vision rotary runs (configuration_gemma4.py's
    `default_rope_type`, modeling_gemma4.py's Gemma4VisionRotaryEmbedding)."""
    entry = records.record(vision.get("rope_parameters") or {}, "rope_parameters")
    kind = entry.get("rope_type", entry.get("type"))
    if kind == "none":
        raise ValueError("rope_parameters: a vision tower's rotary type is 'default', not 'none'")
    if axial and kind == "axial":
        entry = {**{name: value for name, value in entry.items() if name != "type"}, "rope_type": "default"}
    theta = read_rope_theta(entry, "rope_parameters")
    return records.number(vision.get("rope_theta", default), "rope_theta") if theta is None else theta
