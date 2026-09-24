"""Original-format single-file diffusion checkpoints (LDM and SGM `.safetensors`).

A single-file checkpoint holds one or more pipeline components under the
names of the codebase that trained it: `model.diffusion_model.*`,
`first_stage_model.*`, `cond_stage_model.*`, `conditioner.embedders.*`.
diffusers reads these with `from_single_file`, and this module reuses
diffusers' own key maps, not a copy of them:

- `infer_diffusers_model_type` and `DIFFUSERS_DEFAULT_PIPELINE_PATHS` name
  the diffusers repo whose configs describe the checkpoint;
- each component class's `checkpoint_mapping_fn` (`SINGLE_FILE_LOADABLE_CLASSES`)
  converts its tensors;
- `convert_ldm_clip_checkpoint` and `convert_open_clip_checkpoint` convert
  the CLIP and OpenCLIP text encoders.

The file is converted once into Dew's cache (`dew_cache_dir()/single_file/`)
as the diffusers directory it describes: the config repo's configs,
tokenizers and scheduler, each converted component's weights, and the weights
of any component the file does not carry, downloaded from the config repo
(a Flux transformer file ships without its text encoders or VAE) and linked,
not copied, into the directory. The safety checker is left out, as
`from_single_file` leaves it (diffusers' SINGLE_FILE_OPTIONAL_COMPONENTS).
The pipeline then loads from there. diffusers' conversions are written in torch,
so this route needs `pip install 'dew-ml[diffusers]'`; the loaded pipeline
itself runs in JAX.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from dew.interop.pickles import host_view
from dew.interop.safetensors_io import read_file, write_file
from dew.telemetry.instrumentation import dew_cache_dir

if TYPE_CHECKING:
    import torch

INSTALL = "pip install 'dew-ml[diffusers]'"

_CLIP_ENCODERS = ("CLIPTextModel", "CLIPTextModelWithProjection")


def _torch_view(array: np.ndarray) -> torch.Tensor:
    """A torch tensor sharing `array`'s bytes, bfloat16 included."""
    import torch

    dtype = {"bfloat16": torch.bfloat16, "float8_e4m3fn": torch.float8_e4m3fn}.get(array.dtype.name)
    if dtype is None:
        return torch.from_numpy(np.ascontiguousarray(array))
    return torch.frombuffer(bytearray(np.ascontiguousarray(array).tobytes()), dtype=dtype).reshape(array.shape)


def _key(path: Path, config_repo: str) -> str:
    status = path.stat()
    identity = f"{path.resolve()}\0{status.st_size}\0{status.st_mtime_ns}\0{config_repo}"
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def config_repo(checkpoint: Mapping[str, torch.Tensor]) -> str:
    """The diffusers repo diffusers takes this checkpoint's configs from."""
    from diffusers.loaders.single_file_utils import fetch_diffusers_config

    return fetch_diffusers_config(checkpoint)["pretrained_model_name_or_path"]


def _text_encoder(name: str, class_name: str, checkpoint: Mapping[str, torch.Tensor],
                  configs: Path) -> dict[str, torch.Tensor] | None:
    """A CLIP or OpenCLIP text encoder's tensors, converted as
    `create_diffusers_clip_model_from_ldm` converts them, or None when the
    checkpoint does not carry this one."""
    from diffusers.loaders import single_file_utils as utils
    from transformers import CLIPTextConfig

    if class_name not in _CLIP_ENCODERS:
        return None
    config = CLIPTextConfig.from_pretrained(configs / name)
    width = config.max_position_embeddings
    keys = utils.CHECKPOINT_KEY_NAMES

    def fits(kind: str) -> bool:
        return keys[kind] in checkpoint and checkpoint[keys[kind]].shape[-1] == width

    if utils.is_clip_model(checkpoint) or fits("clip_sdxl"):
        return utils.convert_ldm_clip_checkpoint(checkpoint)
    stand_in = SimpleNamespace(config=config)
    if utils.is_open_clip_model(checkpoint):
        return utils.convert_open_clip_checkpoint(stand_in, checkpoint, prefix="cond_stage_model.model.")
    if fits("open_clip_sdxl"):
        return utils.convert_open_clip_checkpoint(stand_in, checkpoint, prefix="conditioner.embedders.1.model.")
    if utils.is_open_clip_sdxl_refiner_model(checkpoint):
        return utils.convert_open_clip_checkpoint(stand_in, checkpoint, prefix="conditioner.embedders.0.model.")
    return None


def _component(name: str, library: str, class_name: str, checkpoint: Mapping[str, torch.Tensor],
               configs: Path) -> dict[str, torch.Tensor] | None:
    """One component's diffusers-format tensors from the checkpoint, or None
    when the checkpoint does not carry it."""
    if library == "diffusers":
        from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES

        entry = SINGLE_FILE_LOADABLE_CLASSES.get(class_name)
        if entry is None:
            return None
        config = json.loads((configs / name / "config.json").read_text())
        return entry["checkpoint_mapping_fn"](config=config, checkpoint=dict(checkpoint)) or None
    if library == "transformers":
        return _text_encoder(name, class_name, checkpoint, configs)
    return None


def unpacked(path: str | os.PathLike[str], configs: Path | None = None) -> Path:
    """The diffusers directory a single-file checkpoint describes, written
    once into Dew's cache.

    `configs` is a diffusers directory whose model_index.json and component
    configs describe the checkpoint, as `from_single_file(config=...)` takes
    one; without it they come from the diffusers repo diffusers infers from
    the checkpoint's keys and shapes. The directory is published with one
    rename once every file is written.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(f"a single-file checkpoint loads through diffusers' key maps; {INSTALL}") from error
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as error:
        raise ImportError(f"a single-file checkpoint converts through diffusers' key maps, which run in torch; "
                          f"{INSTALL}") from error

    source = Path(path)
    stored, _ = read_file(source)
    checkpoint = {name: _torch_view(np.asarray(tensor)) for name, tensor in stored.items()}
    local = configs is not None
    repo = str(configs.resolve()) if configs is not None else config_repo(checkpoint)
    target = Path(dew_cache_dir()) / "single_file" / _key(source, repo)
    if (target / "model_index.json").is_file():
        return target
    if configs is None:
        configs = Path(snapshot_download(repo, allow_patterns=["model_index.json", "*/*.json", "*/*.txt",
                                                               "*/*.model"]))
    from diffusers.loaders.single_file import SINGLE_FILE_OPTIONAL_COMPONENTS

    index = json.loads((configs / "model_index.json").read_text())
    for name in SINGLE_FILE_OPTIONAL_COMPONENTS:
        if name in index:
            index[name] = [None, None]
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=".converting-"))
    try:
        # The config snapshot may also hold weights an earlier download left.
        shutil.copytree(configs, staging, dirs_exist_ok=True, symlinks=False,
                        ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.ckpt", "*.msgpack"))
        (staging / "model_index.json").write_text(json.dumps(index, indent=2))
        missing = []
        for name, spec in index.items():
            if name.startswith("_") or not isinstance(spec, list) or spec[0] is None:
                continue
            library, class_name = spec
            converted = _component(name, library, class_name, checkpoint, configs)
            if converted is None:
                if (configs / name / "config.json").is_file():
                    missing.append(name)
                continue
            weights = "diffusion_pytorch_model.safetensors" if library == "diffusers" else "model.safetensors"
            write_file({key: host_view(value.contiguous(), f"{name}'s {key}") for key, value in converted.items()},
                       staging / name / weights, {"format": "pt"})
        if missing:
            # Components the file does not carry come from the configs' own weights.
            weights = configs if local else Path(snapshot_download(
                repo, allow_patterns=[f"{name}/*.safetensors" for name in missing]))
            for name in missing:
                for file in (weights / name).glob("*.safetensors"):
                    os.link(file.resolve(), staging / name / file.name)
        os.replace(staging, target)
    except OSError:
        if not (target / "model_index.json").is_file():
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target
