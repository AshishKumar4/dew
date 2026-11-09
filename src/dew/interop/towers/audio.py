"""Gemma 3n's and Gemma 4's audio towers as a release stores them: their config's fields, read one way, and
their tensors' names, each read onto the encoder's own leaf and checked against its shape, with the layouts
that write it back, for the models `dew.nn.audio` computes."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from dew.batch import unboxed
from dew.interop.config_records import hf_metadata
from dew.interop.unread import Stored
from dew.interop.weights import Tree, check_shapes, native_shape, placed
from dew.nn.audio import Gemma3nAudio, Gemma4Audio
from dew.nn.sharding import abstract
from dew.registry import from_record

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout


def audio_config(record: Mapping[str, object]) -> Gemma3nAudio | Gemma4Audio:
    """Validate an HF audio record and retain every encoder computation field."""
    model_type = record.get("model_type")
    if model_type == "gemma3n_audio":
        cls = Gemma3nAudio
        other = {"vocab_size", "vocab_offset"}  # consumed by the shared multimodal embedder
    elif model_type == "gemma4_audio":
        cls = Gemma4Audio
        other = set()
    else:
        raise ValueError(f"audio model_type {model_type!r} is not supported")
    fields = {field.name for field in dataclasses.fields(cls)}
    unknown = (set(record) - fields - hf_metadata() - other
               - {key for key in record if key.startswith("_")})
    if unknown:
        raise ValueError(f"audio config fields {sorted(unknown)} have no counterpart")
    return from_record(cls, {key: value for key, value in record.items() if key in fields})


@functools.cache
def _audio_template(config: Gemma3nAudio | Gemma4Audio):
    features = (
        config.input_feat_size if isinstance(config, Gemma3nAudio) else config.subsampling_conv_channels[0]
    )
    template = unboxed(abstract(config.build().init, jax.random.key(0),
                                       jax.ShapeDtypeStruct((1, 16, features), jnp.float32),
                                       jax.ShapeDtypeStruct((1, 16), jnp.bool_)))
    return {tuple(str(k.key) for k in path): value.shape
            for path, value in jax.tree_util.tree_leaves_with_path(template)}


def audio_weight_path(name: str, config: Gemma3nAudio | Gemma4Audio) -> tuple[str, ...]:
    """One bare HF audio tensor name into its complete native collection path."""
    parts = name.split(".")
    if len(parts) > 1 and parts[0] in ("conformer", "layers") and parts[1].isdigit():
        parts = [f"{parts[0]}_{parts[1]}", *parts[2:]]
    collection = (
        "constants" if parts[-1] in ("input_min", "input_max", "output_min", "output_max") else "params"
    )
    if parts[-1] == "weight":
        norms = {"norm", "pre_attn_norm", "post_norm", "pre_layer_norm", "post_layer_norm",
                 "norm_pre_attn", "norm_post_attn", "norm_out", "conv_norm"}
        parts[-1] = "scale" if len(parts) > 1 and parts[-2] in norms else "kernel"
    path = (collection, *parts)
    if path not in _audio_template(config):
        raise ValueError(f"audio tensor {name!r} does not match the encoder")
    return path


def audio_weights(
    tensors: Mapping[str, Stored],
    config: Gemma3nAudio | Gemma4Audio,
    *,
    param_dtype: str = "float32",
) -> tuple[Tree[Stored], tuple[WeightLayout, ...]]:
    """Translate reference weights into params and frozen clipping constants, with the layouts
    that write each tensor back from them (`placed`).

    Prefix stripping belongs to the shared wrapper loader. Parameters use the
    requested storage precision; clipping state stays FP32. Integer tensors
    retain their native dtype, and every shape is checked against the encoder.
    """
    paths = {name: audio_weight_path(name, config) for name in tensors}
    check_shapes(((path, native_shape(path, tensors[name].shape)) for name, path in paths.items()),
                 _audio_template(config))
    return placed(tensors, paths.__getitem__, param_dtype)

