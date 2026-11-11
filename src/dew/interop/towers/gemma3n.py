"""Gemma 3n's vision tower and projector as a release stores them: their tensors' names and their
config's fields, both ways (`dew.interop.towers.common`), for the models `dew.nn.vision.gemma3n` computes."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING

from dew import _records
from dew.interop.config_records import NativeFields, hf_metadata
from dew.interop.layouts import Prefixed, Route, Table
from dew.interop.unread import Stored
from dew.interop.weights import placed
from dew.nn.mobilenet import MOBILENET_ARCHITECTURE
from dew.nn.vision.gemma3n import Gemma3nProjector, Gemma3nVision
from dew.registry import Record, from_record

from .common import ProjectorLayout, TowerLayout, _vision_section

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout


def _tails() -> dict[tuple[str, ...], frozenset[tuple[str, ...]]]:
    """Each module's tensors past its name, by its place in the MobileNet (`MOBILENET_ARCHITECTURE`): a
    block's by its kind, the stem's and the multi-scale fusion's."""
    held: dict[tuple[str, ...], frozenset[tuple[str, ...]]] = {
        ("conv_stem",): frozenset({("conv", "weight"), ("conv", "bias"), ("bn", "weight")}),
        ("msfa",): frozenset({("norm", "weight"), *(
            ("ffn", name, child, "weight") for name in ("pw_exp", "pw_proj") for child in ("conv", "bn"))}),
    }
    for stage, blocks in enumerate(MOBILENET_ARCHITECTURE):
        for index, spec in enumerate(blocks):
            tails: set[tuple[str, ...]]
            if spec.kind == "edge":
                tails = {(name, "weight") for name in ("conv_exp", "conv_pwl", "bn1", "bn2")}
            elif spec.kind == "inverted":
                modules = ["pw_exp", "pw_proj", *(["dw_start"] if spec.start_kernel else []),
                           *(["dw_mid"] if spec.middle_kernel else [])]
                tails = {(name, child, "weight") for name in modules for child in ("conv", "bn")}
                tails.add(("layer_scale", "gamma"))
            else:
                tails = {("norm", "weight"), ("layer_scale", "gamma")}
                tails.update(("attn", name, "proj", "weight") for name in ("query", "key", "value", "output"))
                if spec.kv_stride > 1:
                    tails.update(("attn", name, child, "weight") for name in ("key", "value")
                                 for child in ("down_conv", "norm"))
            held[("blocks", str(stage), str(index))] = frozenset(tails)
    return held


def _leaf(module: tuple[str, ...], tail: tuple[str, ...]) -> str:
    """The leaf of a MobileNet tensor: a block's under its stage's and its own module, a norm's weight its
    scale, any other weight its kernel."""
    place = (f"stages_{module[1]}", f"blocks_{module[2]}") if module[0] == "blocks" else module
    if tail[-1] != "weight":
        return "/".join(("params", *place, *tail))
    norm = tail[-2] in ("bn", "bn1", "bn2", "norm")
    return "/".join(("params", *place, *tail[:-1], "scale" if norm else "kernel"))


_MOBILENET = Table(*(Route(".".join((*module, *tail)), _leaf(module, tail))
                     for module, tails in _tails().items() for tail in sorted(tails)))


_GEMMA3N_ROUTES = Table(Prefixed("timm_model.", _MOBILENET), _MOBILENET)
"""A Gemma 3n MobileNet-v5's tensor names past `vision_tower.`, both ways, each a module's at its place
in the architecture (`_tails`): under timm's `timm_model.`, as the release stores and an export writes
them, or without it, which a load reads too."""


GEMMA3N_PROJECTOR_TENSORS = {
    "embedding.weight": ("embedding", "embedding"),
    "hard_embedding_norm.weight": ("hard_embedding_norm", "scale"),
    "soft_embedding_norm.weight": ("soft_embedding_norm", "scale"),
    "embedding_projection.weight": ("embedding_projection", "kernel"),
}
"""The Gemma 3n embedder's tensors past `embed_vision.` (or its audio's), each to its leaf."""


def translate_gemma3n_projector_weights(
    hf_tensors: Mapping[str, Stored], *, param_dtype: str = "float32"
) -> tuple[Mapping[str, object], tuple[WeightLayout, ...]]:
    """A Gemma 3n vision embedder's parameter tree, with the layouts that write it back."""
    paths = GEMMA3N_PROJECTOR_TENSORS
    if set(hf_tensors) != set(paths):
        raise ValueError(f"vision embedder tensors differ: missing {sorted(set(paths) - set(hf_tensors))}, "
                         f"unknown {sorted(set(hf_tensors) - set(paths))}")
    return placed(hf_tensors, paths.__getitem__, param_dtype)


def _gemma3n_vision_record(
        hf_config: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """The vision embedder's fields and the encoder's model_args, the whole
    record validated before either component consumes it."""
    vision = _vision_section(hf_config)
    if vision.get("model_type", "gemma3n_vision") != "gemma3n_vision":
        raise ValueError(f"vision model_type {vision.get('model_type')!r} is not gemma3n_vision")
    used = {"model_type", "architecture", "hidden_size", "do_pooling", "model_args",
            "vocab_size", "vocab_offset", "rms_norm_eps"}
    # These are serialized HF metadata. Vocabulary and RMS fields above feed
    # the vision embedder; construction fields feed the MobileNet encoder.
    metadata = hf_metadata() | {"initializer_range", "label_names", "num_classes"}
    unknown = (set(vision) - used - metadata
               - {key for key in vision if str(key).startswith("_")})
    if unknown:
        raise ValueError(f"vision_config fields {sorted(unknown)} have no counterpart")
    if vision.get("architecture", "mobilenetv5_300m_enc") != "mobilenetv5_300m_enc":
        raise ValueError(f"architecture {vision.get('architecture')!r} is not the MobileNet-v5 encoder")
    embedder = {
        "vision_width": _records.integer(vision.get("hidden_size", 2048), "hidden_size"),
        "vocab_size": _records.integer(vision.get("vocab_size", 128), "vocab_size"),
        "vocab_offset": _records.integer(vision.get("vocab_offset", 262144), "vocab_offset"),
        "norm_eps": _records.number(vision.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
    }
    if embedder["vision_width"] != 2048:
        raise ValueError("hidden_size must be 2048; timm's MobileNet-v5 encoder fixes its adapter width")
    if vision.get("do_pooling", False):
        raise ValueError("do_pooling=True requests a classifier head the encoder does not have")
    options = vision.get("model_args")
    options = _records.record({} if options is None else options, "model_args")
    allowed = {field.name for field in dataclasses.fields(Gemma3nVision)}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"MobileNet-v5 model_args {sorted(unknown)} are not supported")
    return embedder, options


def translate_gemma3n_vision_config(hf_config: Mapping[str, object]) -> Record:
    _, options = _gemma3n_vision_record(hf_config)
    value: Gemma3nVision = from_record(Gemma3nVision, options)
    return {"class": "gemma3n", "fields": {**dataclasses.asdict(value)}}


def translate_gemma3n_projector_config(hf_config: Mapping[str, object],
                                       text_width: int) -> Record:
    embedder, _ = _gemma3n_vision_record(hf_config)
    return {"class": "gemma3n", "fields": NativeFields(Gemma3nProjector,
                                                       {**embedder, "text_width": text_width})}


GEMMA3N_VISION = TowerLayout("vision_tower.", _GEMMA3N_ROUTES)


GEMMA3N_PROJECTOR = ProjectorLayout("embed_vision.", GEMMA3N_PROJECTOR_TENSORS,
                                              read=translate_gemma3n_projector_weights)

TOWERS = {Gemma3nVision: GEMMA3N_VISION}
"""Where each of these tower kinds' tensors sit in a media checkpoint (`dew.interop.towers.tower_layout`)."""


PROJECTORS = {Gemma3nProjector: GEMMA3N_PROJECTOR}
"""Where each of these projector kinds' tensors sit in a media checkpoint
(`dew.interop.towers.projector_layout`)."""
