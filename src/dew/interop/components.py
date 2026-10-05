"""Bind configured native components without replacing supplied parameters."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np
from flax import linen as nn

from dew.interop.weights import record_layouts
from dew.objectives.base import Variables

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout


def component_source(name_or_dir: str | Path, revision: str | None = None, subfolder: str = "", *,
                     weights: bool) -> tuple[Path, dict]:
    """Resolve one component's metadata and only the weights its caller needs."""
    from dew.interop.sources import snapshot

    directory = snapshot(str(name_or_dir), revision, weights=(subfolder,) if weights else False)
    return directory / subfolder, json.loads((directory / subfolder / "config.json").read_text())


def bind_component[ComponentT, ConfigT: Mapping[str, object]](
    directory: Path, component: str, config: ConfigT, model: nn.Module,
    path_of: Callable[[str, int], tuple[str, ...] | None], wrap: Callable[[Variables], ComponentT], *,
    prefix: tuple[str, ...], param_dtype: str = "float32", params: Variables | None = None,
    lazy: bool = False, inputs: tuple[jax.Array | np.ndarray | jax.ShapeDtypeStruct, ...] = (),
    tensors: Mapping[str, np.ndarray] | None = None,
    validate: Callable[[Mapping[str, np.ndarray]], None] | None = None,
) -> tuple[ComponentT, Variables, tuple[WeightLayout, ...], ConfigT]:
    """Translate absent parameters, validate the native geometry and bind a wrapper.

    Supplied parameters remain authoritative and return no source layouts.
    Explicit tensors let components read non-parameter state, such as FLUX.2's
    latent statistics, even when their parameters were supplied.
    """
    from dew.interop.safetensors_io import read_weights
    from dew.nn.text_encoders import check_tree

    layouts: tuple[WeightLayout, ...] = ()
    if params is None:
        stored = read_weights(directory) if tensors is None else tensors
        if validate is not None:
            validate(stored)
        params, layouts = record_layouts(
            component, stored, lambda name: path_of(name, np.ndim(stored[name])), prefix,
            param_dtype=param_dtype, lazy=lazy)
    if inputs:
        check_tree({"params": params}, model, *inputs)
    return wrap(params), params, layouts, config
