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
    """Resolve required metadata and weights, using the root's conversion fallback."""
    from dew.interop.sources import snapshot

    requested = (subfolder,) if subfolder else True
    directory = snapshot(str(name_or_dir), revision, weights=requested if weights else False)
    return directory / subfolder, json.loads((directory / subfolder / "config.json").read_text())


def bind_component[ComponentT, ConfigT: Mapping[str, object]](
    directory: Path, component: str, config: ConfigT, model: nn.Module,
    path_of: Callable[[str, int], tuple[str, ...] | None], wrap: Callable[[Variables], ComponentT], *,
    prefix: tuple[str, ...], param_dtype: str = "float32", params: Variables | None = None,
    lazy: bool = False, inputs: tuple[jax.Array | np.ndarray | jax.ShapeDtypeStruct, ...] = (),
    tensors: Mapping[str, np.ndarray] | None = None,
    validate: Callable[[Mapping[str, np.ndarray]], None] | None = None,
    state: Callable[[str], bool] | None = None,
    shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[ComponentT, Variables, tuple[WeightLayout, ...], ConfigT]:
    """Translate absent parameters, validate the native geometry and bind a wrapper.

    Supplied parameters remain authoritative, and their layouts come from the
    stored tensors' `shapes` (`weights.source_layouts`), with no weight read; none
    where those are not given. Explicit tensors let components read
    non-parameter state, such as FLUX.2's latent statistics, even when their
    parameters were supplied. The tensors `state` selects are state rather than
    weights, such as a safety head's scoring vectors and thresholds, and keep
    FP32 whatever `param_dtype` says.
    """
    from dew.interop.safetensors_io import read_weights
    from dew.interop.weights import check_tree, source_layouts

    layouts: tuple[WeightLayout, ...] = ()
    if params is not None and shapes is not None:
        layouts = source_layouts(component, shapes, lambda name: path_of(name, len(shapes[name])), prefix)
    if params is None:
        stored = read_weights(directory) if tensors is None else tensors
        if validate is not None:
            validate(stored)

        def path(name: str) -> tuple[str, ...] | None:
            return path_of(name, np.ndim(stored[name]))

        held = {} if state is None else {name: tensor for name, tensor in stored.items() if state(name)}
        params, layouts = record_layouts(
            component, {name: tensor for name, tensor in stored.items() if name not in held}, path, prefix,
            param_dtype=param_dtype, lazy=lazy)
        if held:
            scoring, state_layouts = record_layouts(component, held, path, prefix, param_dtype="float32",
                                                    lazy=lazy)
            params.update(scoring)
            layouts += state_layouts
    if inputs:
        check_tree({"params": params}, model, *inputs)
    return wrap(params), params, layouts, config
