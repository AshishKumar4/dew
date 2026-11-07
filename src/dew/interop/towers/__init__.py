"""Where each media tower's and projector's tensors sit in a media checkpoint, both ways: a family's
module declares its kinds' layouts (`TowerLayout`, `ProjectorLayout`), on EX02's one bidirectional
form, beside the configuration it reads them by, and the tables here find a kind's by its class
(`dew.registry.towers`, `dew.registry.projectors`). The models in `dew.nn.vision` and `dew.nn.mimo_audio`
compute; they read no release's names.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from dew.batch import Variables
from dew.interop.towers import (
    deepseek_v41,
    gemma3n,
    gemma4,
    gemma4_unified,
    lfm3,
    llama4,
    mimo,
    mimo_audio,
    moonvit,
    muse_glimmer,
    qwen35,
    siglip,
)
from dew.interop.towers.common import ProjectorLayout, TowerLayout
from dew.interop.unread import Stored
from dew.nn.vision.common import ProjectorBase, TowerBase
from dew.registry import projectors, towers

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

_PROJECTED = (deepseek_v41, gemma3n, gemma4, lfm3, llama4, mimo, moonvit, muse_glimmer, qwen35, siglip)
"""The family modules that declare projector layouts; each declares its towers' too."""

TOWER_LAYOUTS: Mapping[type[TowerBase], TowerLayout] = MappingProxyType(
    {kind: layout for family in (*_PROJECTED, gemma4_unified, mimo_audio)
     for kind, layout in family.TOWERS.items()})
"""Each tower class's layout, by the family modules that declare them: a tower read another way (Gemma's
audio towers) has none."""

PROJECTOR_LAYOUTS: Mapping[type[ProjectorBase], ProjectorLayout] = MappingProxyType(
    {kind: layout for family in _PROJECTED for kind, layout in family.PROJECTORS.items()})
"""Each projector class's layout, by the family modules that declare them."""


def _held[Layout](layout: Layout | None, what: str, kind: str) -> Layout:
    if layout is None:
        raise ValueError(f"{what} kind {kind!r} has no weight map here")
    return layout


def tower_layout(kind: str) -> TowerLayout:
    """Where tower kind `kind`'s tensors sit: its class's layout (`TOWER_LAYOUTS`, `dew.registry.towers`)."""
    return _held(TOWER_LAYOUTS.get(towers[kind]) if kind in towers else None, "tower", kind)


def projector_layout(kind: str) -> ProjectorLayout:
    """Where projector kind `kind`'s tensors sit: its class's layout (`PROJECTOR_LAYOUTS`,
    `dew.registry.projectors`)."""
    return _held(PROJECTOR_LAYOUTS.get(projectors[kind]) if kind in projectors else None, "projector", kind)


def tower_variables(kind: str, hf_tensors: Mapping[str, Stored], param_dtype: str
                    ) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """One vision tower's variables, in the requested storage (`tower_layout`), and the layouts
    that write its tensors back from them."""
    return tower_layout(kind).variables(hf_tensors, param_dtype)


def projector_variables(kind: str, hf_tensors: Mapping[str, Stored], param_dtype: str
                        ) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """One projector kind's parameters, in the requested storage (`projector_layout`), and the
    layouts that write its tensors back from them."""
    return projector_layout(kind).variables(hf_tensors, param_dtype)
