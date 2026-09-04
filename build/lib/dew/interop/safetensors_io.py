"""safetensors for Flax parameter trees.

A parameter tree is nested dicts, a safetensors file is a flat table of named
tensors. The two meet at the '/'-joined path, the same naming the Hugging Face
checkpoints use, so a tree written here opens in anything that reads
safetensors and a file read back keeps its nesting.

Only the container is handled here. No leaf is renamed, transposed or cast:
the names on disk are the module names in the tree.
"""

import json
import os
from typing import Any, Dict, Mapping

import jax
import numpy as np

SEPARATOR = "/"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"


def _safetensors():
    """The numpy backend, imported on use since safetensors is an extra."""
    try:
        from safetensors import numpy as safetensors_numpy
    except ImportError as error:
        raise ImportError(
            "dew.interop needs safetensors: pip install dew-ml[interop]"
        ) from error
    return safetensors_numpy


def _leaf_name(path) -> str:
    names = []
    for entry in path:
        if not isinstance(entry, jax.tree_util.DictKey):
            raise TypeError(
                f"safetensors names come from dict keys, this tree has a "
                f"{type(entry).__name__} in its path")
        if not isinstance(entry.key, str) or SEPARATOR in entry.key:
            raise ValueError(
                f"parameter key {entry.key!r} cannot go into a {SEPARATOR!r}-joined name")
        names.append(entry.key)
    return SEPARATOR.join(names)


def _host_array(leaf) -> np.ndarray:
    """Host copy of a leaf. safetensors writes raw bytes, so it must be dense."""
    array = np.asarray(leaf)
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def _flatten(params) -> Dict[str, np.ndarray]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    return {_leaf_name(path): _host_array(leaf) for path, leaf in leaves}


def _unflatten(tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for name, tensor in tensors.items():
        *branches, leaf = name.split(SEPARATOR)
        node = tree
        for branch in branches:
            node = node.setdefault(branch, {})
        node[leaf] = tensor
    return tree


def save_params(params, path) -> None:
    """Write a parameter tree to a safetensors file, one tensor per leaf."""
    _safetensors().save_file(_flatten(params), os.fspath(path))


def load_params(path) -> Dict[str, Any]:
    """Read a safetensors file back into a nested parameter dict.

    Leaves arrive as numpy arrays, so nothing is placed on a device until the
    caller asks for it.
    """
    return _unflatten(_safetensors().load_file(os.fspath(path)))


def save_hf_layout(params, config: Dict[str, Any], directory) -> None:
    """Write model.safetensors and config.json into `directory`.

    That pair is what a Hugging Face style loader looks for. The config is
    written as given: dew does not translate its own config vocabulary into
    anyone else's.
    """
    os.makedirs(directory, exist_ok=True)
    save_params(params, os.path.join(directory, WEIGHTS_FILE))
    with open(os.path.join(directory, CONFIG_FILE), "w") as handle:
        json.dump(config, handle, indent=2)
