"""Original-format single-file diffusion checkpoints: Stable Diffusion 1.x,
SDXL and BFL's FLUX.1 `.safetensors`.

A single-file checkpoint holds one or more pipeline components under the
names of the codebase that trained it: `model.diffusion_model.*`,
`first_stage_model.*`, `cond_stage_model.*`, `conditioner.embedders.*`,
BFL's `double_blocks.*` and `single_blocks.*`.
diffusers reads these with `from_single_file`, and this module reuses
diffusers' own key maps, not a copy of them:

- `infer_diffusers_model_type` and `DIFFUSERS_DEFAULT_PIPELINE_PATHS` name
  the diffusers repo whose configs describe the checkpoint;
- each component class's `checkpoint_mapping_fn` (`SINGLE_FILE_LOADABLE_CLASSES`)
  converts its tensors;
- `convert_ldm_clip_checkpoint` and `convert_open_clip_checkpoint` convert
  the CLIP and OpenCLIP text encoders, chosen in the order
  `create_diffusers_clip_model_from_ldm` chooses them.

The file is converted once into Dew's cache (`dew_cache_dir()/single_file/`)
as the diffusers directory it describes: the configs, tokenizers and
scheduler, each converted component's weights, and the weights of any
component the file does not carry (a Flux transformer file ships without its
text encoders or VAE), taken from where the configs came from and linked,
not copied, into the directory. The safety checker is left out, as
`from_single_file` leaves it (diffusers' SINGLE_FILE_OPTIONAL_COMPONENTS).
The pipeline then loads from there, and the entry is kept only once it has. diffusers' conversions are written in torch,
so this route needs `pip install 'dew-ml[diffusers]'`; the loaded pipeline
itself runs in JAX.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
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
    """A torch tensor sharing `array`'s bytes, bfloat16 and float8 included."""
    import torch

    raw = torch.from_numpy(np.ascontiguousarray(array).reshape(-1).view(np.uint8))
    return raw.view(getattr(torch, array.dtype.name)).reshape(array.shape)


def config_repo(checkpoint: Mapping[str, torch.Tensor]) -> str:
    """The diffusers repo diffusers takes this checkpoint's configs from."""
    from diffusers.loaders.single_file_utils import fetch_diffusers_config

    return fetch_diffusers_config(checkpoint)["pretrained_model_name_or_path"]


def _text_encoder(name: str, class_name: str, checkpoint: Mapping[str, torch.Tensor],
                  configs: Path) -> dict[str, torch.Tensor] | None:
    """A CLIP or OpenCLIP text encoder's tensors, converted as
    `create_diffusers_clip_model_from_ldm` converts them, or None when the
    checkpoint does not carry this one.

    An SDXL file carries two CLIP towers, CLIP-L and OpenCLIP-G; as in
    diffusers, the tower whose position embedding is as wide as this
    encoder's hidden size is its own. The `text_encoders.*` towers of SD3 and
    ComfyUI's all-in-one files are not read (their encoders count as not
    carried).
    """
    from diffusers.loaders import single_file_utils as utils

    if class_name not in _CLIP_ENCODERS:
        return None
    from transformers import CLIPTextConfig

    config = CLIPTextConfig.from_pretrained(configs / name)
    keys = utils.CHECKPOINT_KEY_NAMES

    def fits(kind: str) -> bool:
        return keys[kind] in checkpoint and checkpoint[keys[kind]].shape[-1] == config.hidden_size

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
    when the checkpoint does not carry it (a mapping that finds none of its
    tensors returns nothing, as `from_single_file` reads it)."""
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


def pipeline_index(configs: Path) -> dict[str, object]:
    """The configs' model_index.json without the components `from_single_file`
    leaves out (diffusers' SINGLE_FILE_OPTIONAL_COMPONENTS: the safety checker
    and its feature extractor)."""
    from diffusers.loaders.single_file import SINGLE_FILE_OPTIONAL_COMPONENTS

    index = json.loads((configs / "model_index.json").read_text())
    for name in SINGLE_FILE_OPTIONAL_COMPONENTS:
        if name in index:
            index[name] = [None, None]
    return index


def converted(checkpoint: Mapping[str, torch.Tensor], configs: Path,
              index: Mapping[str, object]) -> Iterator[tuple[str, str, dict[str, torch.Tensor] | None]]:
    """Each model component of the pipeline `index` describes, as
    `(name, library, tensors)`: its diffusers-format tensors, or None when
    the checkpoint does not carry it. One component is converted at a time,
    so a caller that writes each before asking for the next holds one."""
    for name, spec in index.items():
        if name.startswith("_") or not isinstance(spec, list) or spec[0] is None:
            continue
        library, class_name = spec
        if (configs / name / "config.json").is_file():
            yield name, library, _component(name, library, class_name, checkpoint, configs)


def _configs(repo: str) -> Path:
    """The config repo's configs, tokenizers and scheduler, without weights,
    in a snapshot directory named by the commit they came from."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    try:
        return Path(snapshot_download(repo, allow_patterns=["model_index.json", "*/*.json", "*/*.txt", "*/*.model"]))
    except GatedRepoError as error:
        raise PermissionError(
            f"this checkpoint's configs come from {repo}, which is gated on the Hub: accept its license at "
            f"https://huggingface.co/{repo} and log in (`hf auth login`), or put the file in a local diffusers "
            "directory holding model_index.json and its component configs and load that directory with "
            "single_file=") from error


def _hub_weights(repo: str, revision: str, names: Collection[str]) -> Path:
    """The snapshot of `repo` at `revision` holding these components' weights."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=[
        pattern for name in names for pattern in (f"{name}/*.safetensors", f"{name}/*.safetensors.index.json")]))


def _link(source: Path, destination: Path) -> None:
    """A hard link, or a copy where the two sit on different filesystems."""
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, destination)


def _key(path: Path, configs: str) -> str:
    """The cache entry's name: the file (a Hub file resolves to its
    content-addressed blob), where the configs came from, and the diffusers
    whose key maps convert it."""
    import diffusers

    status = path.stat()
    identity = f"{path.resolve()}\0{status.st_size}\0{status.st_mtime_ns}\0{configs}\0{diffusers.__version__}"
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


@contextmanager
def unpacked(path: str | os.PathLike[str], configs: Path | None = None,
             hub: tuple[str, str] | None = None) -> Iterator[tuple[Path, Path]]:
    """The diffusers directory a single-file checkpoint describes, as
    `(directory, published)`: the directory to load now and where it lives
    in Dew's cache once the load succeeds.

    `configs` is a diffusers directory whose model_index.json and component
    configs describe the checkpoint, as `from_single_file(config=...)` takes
    one. Without it they come from the diffusers repo diffusers infers from
    the checkpoint's keys and shapes, at the commit that fetch resolves.
    A component the file does not carry takes its weights from where the
    configs came from: `configs` itself, or the Hub repo and commit `hub`
    names when `configs` is that repo's metadata snapshot. One that has none
    there is refused by name.

    A cached conversion is returned before the file is read. A new one is
    written into a staging directory, which is published with one rename
    when the caller's block exits without an error, so an interrupted or
    failed load leaves nothing cached.
    """
    try:
        import diffusers  # noqa: F401
        import huggingface_hub  # noqa: F401
        import torch  # noqa: F401
    except ImportError as error:
        raise ImportError(f"a single-file checkpoint converts through diffusers' key maps, which run in torch; "
                          f"{INSTALL}") from error

    source = Path(path)
    checkpoint: dict[str, torch.Tensor] = {}
    if configs is None:
        # Copy-on-write, so a conversion that writes into a tensor never writes the file.
        stored, _ = read_file(source, copy_on_write=True)
        checkpoint = {name: _torch_view(np.asarray(tensor)) for name, tensor in stored.items()}
        repo = config_repo(checkpoint)
        configs = _configs(repo)
        hub = (repo, configs.name)
    origin = f"{hub[0]}@{hub[1]}" if hub is not None else str(configs.resolve())
    target = Path(dew_cache_dir()) / "single_file" / _key(source, origin)
    if (target / "model_index.json").is_file():
        yield target, target
        return
    if not checkpoint:
        stored, _ = read_file(source, copy_on_write=True)
        checkpoint = {name: _torch_view(np.asarray(tensor)) for name, tensor in stored.items()}
    index = pipeline_index(configs)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=".converting-"))
    try:
        # The config snapshot may also hold weights an earlier download left.
        shutil.copytree(configs, staging, dirs_exist_ok=True, symlinks=False,
                        ignore=shutil.ignore_patterns("*.safetensors", "*.safetensors.index.json", "*.bin",
                                                      "*.ckpt", "*.msgpack"))
        (staging / "model_index.json").write_text(json.dumps(index, indent=2))
        missing = []
        for name, library, tensors in converted(checkpoint, configs, index):
            if tensors is None:
                missing.append(name)
                continue
            weights = "diffusion_pytorch_model.safetensors" if library == "diffusers" else "model.safetensors"
            write_file({key: host_view(value.contiguous(), f"{name}'s {key}") for key, value in tensors.items()},
                       staging / name / weights, {"format": "pt"})
        if missing:
            weights = configs if hub is None else _hub_weights(*hub, missing)
            for name in missing:
                for file in (weights / name).glob("*.safetensors*"):
                    _link(file.resolve(), staging / name / file.name)
            empty = [name for name in missing if not any((staging / name).glob("*.safetensors"))]
            if empty:
                raise ValueError(
                    f"{source.name} carries no weights for {empty}, and {origin} has none for them either; "
                    f"put their diffusers-format weights under {configs}/<component>/ or load the file from "
                    "a repo that ships them")
        yield staging, target
        try:
            os.replace(staging, target)
        except OSError:
            # Another process published the same conversion first.
            if not (target / "model_index.json").is_file():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
