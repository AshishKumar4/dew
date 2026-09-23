"""Read PyTorch pickle checkpoints through a one-time safetensors conversion.

transformers writes a pickle checkpoint as `pytorch_model.bin`, or as the
shards `pytorch_model.bin.index.json` maps; `safetensors_io.weight_files`
with `STEMS` and `SUFFIX` selects them by the same rule it reads
safetensors by. `converted` unpickles each shard once with
`torch.load(weights_only=True)`, which runs no code from the file, and
memory maps a zip-format file (every `torch.save` since PyTorch 1.6) rather
than reading it. Each shard is written as safetensors into Dew's cache, and
the loader then maps that folder like any other checkpoint.

The cache is keyed by the pickles' identity: each file's resolved path, size
and modification time. A Hub snapshot's files resolve to blobs named by
their content hash. A later load of the same files maps the cache and never
imports torch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import ml_dtypes
import numpy as np

from dew.interop.safetensors_io import WEIGHTS_FILE, weight_files, write_file
from dew.telemetry.instrumentation import dew_cache_dir

if TYPE_CHECKING:
    import torch

STEMS = ("pytorch_model",)
"""The stem of transformers' pickle checkpoint and of its index."""

SUFFIX = ".bin"

CONVERT_SPACE = "https://huggingface.co/spaces/safetensors/convert"
"""SFconvertbot's space, which opens a repo's safetensors conversion as a pull request."""


def converted(directory: Path, shards: Sequence[str]) -> Path:
    """Return a folder of safetensors holding the pickle `shards` of `directory`.

    The first call for these files converts them, which needs torch: the
    `pickles` extra. One shard becomes model.safetensors. Several become
    numbered shards and an index, written last, so a folder whose files
    `weight_files` selects is a finished conversion. Each file is published
    atomically, so a failed conversion leaves nothing a later load would
    read.
    """
    identity = []
    for name in shards:
        status = (directory / name).stat()
        identity.append([name, os.path.realpath(directory / name), status.st_size, status.st_mtime_ns])
    target = Path(dew_cache_dir()) / "converted" / hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:32]
    if target.is_dir() and weight_files({entry.name for entry in target.iterdir()}, "",
                                        lambda name: json.loads((target / name).read_text())):
        return target
    if importlib.util.find_spec("torch") is None:
        raise ImportError(
            f"{directory} holds PyTorch pickles ({', '.join(shards)}), and converting them needs torch: "
            f"pip install 'dew-ml[torch]', or open their safetensors conversion at {CONVERT_SPACE} "
            "and load its refs/pr/N revision")
    target.mkdir(parents=True, exist_ok=True)
    outputs = ([WEIGHTS_FILE] if len(shards) == 1 else
               [f"model-{number:05d}-of-{len(shards):05d}.safetensors" for number in range(1, len(shards) + 1)])
    weight_map: dict[str, str] = {}
    for shard, output in zip(shards, outputs, strict=True):
        tensors = _state_dict(directory / shard)
        write_file(tensors, target / output, {"format": "pt"})
        weight_map.update(dict.fromkeys(tensors, output))
    if len(shards) > 1:
        index = target / "model.safetensors.index.json"
        temporary = index.with_name(f".{index.name}.tmp")
        temporary.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
        os.replace(temporary, index)
    return target


def _state_dict(path: Path) -> dict[str, np.ndarray]:
    """Unpickle one shard's state dict as NumPy views of its tensors' bytes."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path))
    if not isinstance(state, dict):
        raise ValueError(f"{path} holds a {type(state).__name__}, not a state dict of named tensors; "
                         "save the model's state_dict() instead")
    tensors: dict[str, np.ndarray] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{path} holds {name!r} as a {type(tensor).__name__}, not a tensor; "
                             "save the model's state_dict() instead")
        tensors[name] = host_view(tensor, f"{path}'s {name!r}")
    return tensors


def host_view(tensor: torch.Tensor, what: str) -> np.ndarray:
    """A CPU torch tensor's bytes as a NumPy array of its dtype, bfloat16 and
    float8 through ml_dtypes, sharing its storage where it is contiguous."""
    import torch

    kind = str(tensor.dtype).removeprefix("torch.")
    try:
        dtype = np.dtype(getattr(ml_dtypes, kind, kind))
    except TypeError as error:
        raise ValueError(f"{what} is torch.{kind}, which NumPy and safetensors have no dtype for; "
                         "cast it to a floating or integer dtype") from error
    flat = tensor.detach().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().view(dtype).reshape(tuple(tensor.shape))
