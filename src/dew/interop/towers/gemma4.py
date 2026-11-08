"""Gemma 4's vision tower and projector as a release stores them: their tensors' names and their
config's fields, both ways (`dew.interop.towers.common`), for the models `dew.nn.vision.gemma4` computes."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from dew import _records
from dew._model_types import SANDWICH_NORMS
from dew.batch import Variables
from dew.interop.config_records import NativeFields
from dew.interop.layouts import ConfigField, ConfigLayout, ConfigPart, Route, Table, mapped, one_of
from dew.interop.unread import Stored
from dew.interop.weights import placed
from dew.nn.vision.gemma4 import Gemma4Projector, Gemma4Vision
from dew.registry import Record

from .common import (
    TRUNK_FIELDS,
    ProjectorLayout,
    TowerLayout,
    _vision_section,
    activation_field,
    integer_field,
    number_field,
    rope_part,
    written,
)

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout

_GEMMA4_VISION_TENSORS = {
    "patch_embedder.input_proj.weight": ("patch_embed", "kernel"),
    "patch_embedder.position_embedding_table": ("position_table",),
    "std_bias": ("std_bias",),
    "std_scale": ("std_scale",),
}


_GEMMA4_VISION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


_GEMMA4_VISION_MLP = ("gate_proj", "up_proj", "down_proj")


_CONSTANTS = ("std_bias", "std_scale")
"""The tower's standardization buffers, which it holds in `constants` beside its parameters."""


_CLIPS = one_of("input_min", "input_max", "output_min", "output_max")
"""A clipped map's activation bounds, held in `constants`."""


_GEMMA4_LAYER = "encoder.layers.{index}."


_GEMMA4_LEAF = "layers_{index}/"


_GEMMA4_ROUTES = Table(
    *(Route(name, "/".join(("constants" if path[0] in _CONSTANTS else "params", *path)))
      for name, path in _GEMMA4_VISION_TENSORS.items()),
    Route(_GEMMA4_LAYER + "{norm}.weight", "params/" + _GEMMA4_LEAF + "{norm}/scale",
          MappingProxyType({"norm": mapped(SANDWICH_NORMS)})),
    *(route for module, projections in (("self_attn", _GEMMA4_VISION_PROJECTIONS),
                                        ("mlp", _GEMMA4_VISION_MLP))
      for route in (
          Route(_GEMMA4_LAYER + module + ".{projection}.linear.weight",
                "params/" + _GEMMA4_LEAF + module + "/{projection}/kernel",
                MappingProxyType({"projection": one_of(*projections)})),
          Route(_GEMMA4_LAYER + module + ".{projection}.{clip}",
                "constants/" + _GEMMA4_LEAF + module + "/{projection}/{clip}",
                MappingProxyType({"projection": one_of(*projections), "clip": _CLIPS})))),
    # The value norm carries no scale (modeling_gemma4.py, Gemma4VisionAttention), so a weight under its
    # name is unknown.
    Route(_GEMMA4_LAYER + "self_attn.{norm}.weight", "params/" + _GEMMA4_LEAF + "self_attn/{norm}/scale",
          MappingProxyType({"norm": one_of("q_norm", "k_norm")})))
"""A Gemma 4 tower's tensor names, both ways, each leaf in its collection (`params`, and `constants` for
the standardization and the clip bounds). The rotary tables are buffers the checkpoint leaves out,
recomputed from the grid at call time."""


GEMMA4_PROJECTOR_TENSORS = {"embedding_projection.weight": ("projection", "kernel")}
"""The Gemma 4 embedder's tensors past `embed_vision.` (or its audio's), each to its leaf."""


def gemma4_tower_layouts(variables: Variables) -> tuple[WeightLayout, ...]:
    """The layouts that write a Gemma 4 tower's tensors from its collections (`_GEMMA4_ROUTES`)."""
    return written(_GEMMA4_ROUTES, variables)


def gemma4_projector_layouts(params: Variables) -> tuple[WeightLayout, ...]:
    """The layouts that write a Gemma 4 embedder's projection from its parameters, rooted at `params`."""
    return GEMMA4_PROJECTOR.written(params)


def translate_gemma4_projector_weights(
    hf_tensors: Mapping[str, Stored], *, param_dtype: str = "float32"
) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """A Gemma 4 embedder's map into its parameter tree, with the layouts that write it back.

    The pre-projection norm carries no scale, so the projection weight is
    the only tensor.
    """
    if set(hf_tensors) != set(GEMMA4_PROJECTOR_TENSORS):
        raise ValueError(f"unknown tensor names {sorted(hf_tensors)}")
    return placed(hf_tensors, GEMMA4_PROJECTOR_TENSORS.__getitem__, param_dtype)


def translate_gemma4_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A Gemma4VisionConfig into a Gemma4Vision value's fields.

    Reads the vision_config of a wrapper or a bare vision config. The head
    width comes from head_dim or the hidden size divided by the head count.
    Standardization and activation clipping retain
    their reference buffers outside the trainable parameter collection.
    """
    vision = _vision_section(hf_config)
    if vision.get("attention_bias", False):
        raise ValueError("attention_bias=True needs biased maps this trunk lacks")
    if "output_proj_dims" in vision:
        raise ValueError(
            f"output_proj_dims ({vision['output_proj_dims']!r}) changes the "
            "projector width this record leaves to the text width")
    return {"class": "gemma4", "fields": _GEMMA4_CONFIG.read(vision, set(), NativeFields(Gemma4Vision, {}))}


def _read_heads(vision: Mapping[str, object], used: set[str],
                record: NativeFields[Gemma4Vision]) -> NativeFields[Gemma4Vision]:
    """The key/value heads, the query heads' count where unstated, and each head's width, the trunk's
    over its heads where unstated."""
    used.update(("num_key_value_heads", "head_dim"))
    heads = _records.integer(record["num_heads"], "num_heads")
    width = _records.integer(record["hidden_size"], "hidden_size")
    record["num_key_value_heads"] = _records.integer(vision.get("num_key_value_heads", heads),
                                                     "num_key_value_heads")
    record["head_dim"] = _records.integer(vision.get("head_dim", width // heads), "head_dim")
    return record


def _write_heads(tower: Gemma4Vision) -> Mapping[str, object]:
    """The heads `_read_heads` reads back: an unset head width as the width it stands for."""
    return {"num_key_value_heads": tower.num_key_value_heads,
            "head_dim": tower.hidden_size // tower.num_heads if tower.head_dim is None else tower.head_dim}


def _flag(key: str) -> ConfigField:
    """A switch the config states as a boolean, false where omitted."""
    return ConfigField(key, key, default=False, parse=lambda value, key: bool(value))


_GEMMA4_CONFIG = ConfigLayout(
    *TRUNK_FIELDS, ConfigPart(_read_heads, _write_heads),
    integer_field("patch_size", default=16), integer_field("pooling_kernel_size", default=3),
    integer_field("position_embedding_size", default=10240),
    activation_field(("gelu_pytorch_tanh", "gelu"), "gelu_pytorch_tanh", "hidden_activation",
                     "hidden_act"),
    number_field("rms_norm_eps", default=1e-6), rope_part(100.0, axial=True),
    _flag("standardize"), _flag("use_clipped_linears"))
"""A Gemma4VisionConfig's fields, both ways."""


def translate_gemma4_projector_config(vision: Mapping[str, object], text_width: int,
                                      input_width: int | None = None) -> Record:
    """A Gemma 4 wrapper's projector fields: decoder width, norm epsilon, and the width of the raw media
    it projects where no encoder precedes it (`input_width`, Gemma 4 Unified's)."""
    fields: dict[str, object] = {
        "text_width": int(text_width),
        "norm_eps": _records.number(vision.get("rms_norm_eps", 1e-6), "rms_norm_eps")}
    if input_width is not None:
        fields["input_features"] = int(input_width)
    return {"class": "gemma4", "fields": fields}


GEMMA4_VISION = TowerLayout("vision_tower.", _GEMMA4_ROUTES, config=_GEMMA4_CONFIG,
                                      model_type="gemma4_vision")


GEMMA4_PROJECTOR = ProjectorLayout("embed_vision.", GEMMA4_PROJECTOR_TENSORS,
                                    read=translate_gemma4_projector_weights)



TOWERS = {Gemma4Vision: GEMMA4_VISION}
"""Where each of these tower kinds' tensors sit in a media checkpoint (`dew.interop.towers.tower_layout`)."""


PROJECTORS = {Gemma4Projector: GEMMA4_PROJECTOR}
"""Where each of these projector kinds' tensors sit in a media checkpoint
(`dew.interop.towers.projector_layout`)."""
