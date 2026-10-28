"""An original-format single-file checkpoint loads as the diffusers pipeline it describes.

The fixtures (tools/single_file_reference.py) are tiny random Stable Diffusion
1.x and SDXL pipelines, each written as one original-format file by
diffusers' own reverse script, next to the pipeline's configs and a SHA-256
of every tensor diffusers' `from_single_file(file, config=...)` converted it
back to. diffusers has no reverse script for Flux, and its Flux key map
hard-codes FLUX.1's 3072-wide blocks, so the Flux file is built here at that
width with one block of each kind, under BFL's tensor names, and checked
against `from_single_file` in the same process. The network tests check the
released SD 1.5 file the same way, and the released SDXL and FLUX.1 files'
headers against the converted key map.
"""

import errno
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest

from dew.interop import Pretrained, single_file
from dew.interop.safetensors_io import read_file, write_file

FIXTURE = Path(__file__).parent / "fixtures" / "single_file"
WEIGHTS = {"unet": "diffusion_pytorch_model.safetensors", "vae": "diffusion_pytorch_model.safetensors",
           "transformer": "diffusion_pytorch_model.safetensors", "text_encoder": "model.safetensors",
           "text_encoder_2": "model.safetensors"}


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    return tmp_path / "xdg" / "dew" / "single_file"


def _repo(tmp_path: Path, kind: str) -> Path:
    directory = tmp_path / kind
    shutil.copytree(FIXTURE / kind / "configs", directory)
    shutil.copy(FIXTURE / kind / f"{kind}.safetensors", directory / f"{kind}.safetensors")
    return directory


def _load(directory: Path, kind: str):
    return Pretrained.load(directory, single_file=f"{kind}.safetensors", dtype="float32", param_dtype="auto")


def _digest(value: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() + f":torch.{dtype}"


def _entries(cache: Path) -> list[Path]:
    return sorted(cache.iterdir()) if cache.is_dir() else []


def _assert_converted(converted: Path, expected: dict[str, dict[str, str]]) -> None:
    for component, digests in expected.items():
        tensors, _ = read_file(converted / component / WEIGHTS[component])
        # transformers 5.6 dropped CLIPTextModel's `text_model.` level, so a
        # name compares without it. CLIP's position_ids is a buffer rebuilt
        # from the config, not a weight: a released file may store it and
        # diffusers' state dict lists it whether or not it does.
        ours = {name.removeprefix("text_model."): _digest(value, value.dtype.name)
                for name, value in tensors.items() if not name.endswith("position_ids")}
        theirs = {name.removeprefix("text_model."): digest for name, digest in digests.items()
                  if not name.endswith("position_ids")}
        assert ours == theirs, component


@pytest.mark.parametrize("kind", ["sd", "sdxl"])
def test_a_single_file_converts_to_the_tensors_diffusers_from_single_file_does(
        kind: str, tmp_path: Path, cache: Path) -> None:
    loaded = _load(_repo(tmp_path, kind), kind)
    [converted] = _entries(cache)
    assert loaded.source == converted
    _assert_converted(converted, json.loads((FIXTURE / kind / "from_single_file.json").read_text()))


def _without_vae(tmp_path: Path, vae_weights: bool) -> Path:
    """The SD fixture's file without its VAE, beside configs that hold the
    VAE's diffusers weights or, like a Hub repo's metadata snapshot, none."""
    repo = _repo(tmp_path, "sd")
    tensors, _ = read_file(repo / "sd.safetensors")
    write_file({name: value for name, value in tensors.items() if not name.startswith("first_stage_model.")},
               repo / "sd.safetensors", {"format": "pt"})
    if vae_weights:
        _vae_weights(tmp_path / "whole", repo / "vae")
    return repo


def _vae_weights(scratch: Path, destination: Path) -> None:
    whole = _repo(scratch, "sd")
    with single_file.unpacked(whole / "sd.safetensors", whole) as (converted, _):
        shutil.copy(converted / "vae" / WEIGHTS["vae"], destination / WEIGHTS["vae"])


def test_a_component_the_file_lacks_comes_from_the_local_configs(tmp_path: Path, cache: Path) -> None:
    repo = _without_vae(tmp_path, vae_weights=True)
    loaded = _load(repo, "sd")
    vae = loaded.source / "vae" / WEIGHTS["vae"]
    assert vae.read_bytes() == (repo / "vae" / WEIGHTS["vae"]).read_bytes()
    assert vae.stat().st_nlink == 2


def test_only_the_configs_metadata_is_copied(tmp_path: Path, cache: Path) -> None:
    """Other formats of the weights beside the configs (an ONNX export, a
    precision variant, a weights index) and anything else stay out."""
    repo = _repo(tmp_path, "sd")
    for extra in ("unet/model.onnx", "unet/model.onnx_data", "unet/diffusion_pytorch_model.fp16.safetensors",
                  "unet/diffusion_pytorch_model.safetensors.index.json", "README.md"):
        (repo / extra).write_bytes(b"{}")
    loaded = _load(repo, "sd")
    copied = {
        path.relative_to(loaded.source).as_posix() for path in loaded.source.rglob("*") if path.is_file()
    }
    configs = {path.relative_to(FIXTURE / "sd" / "configs").as_posix()
               for path in (FIXTURE / "sd" / "configs").rglob("*") if path.is_file()}
    assert copied == configs | {f"{name}/{WEIGHTS[name]}" for name in ("unet", "vae", "text_encoder")}


def test_an_edited_local_config_converts_again(tmp_path: Path, cache: Path) -> None:
    repo = _repo(tmp_path, "sd")
    _load(repo, "sd")
    scheduler = repo / "scheduler" / "scheduler_config.json"
    scheduler.write_text(json.dumps(json.loads(scheduler.read_text()) | {"beta_end": 0.02}))
    loaded = _load(repo, "sd")
    assert len(_entries(cache)) == 2
    assert json.loads((loaded.source / "scheduler" / "scheduler_config.json").read_text())["beta_end"] == 0.02


def test_a_replaced_lent_weight_converts_again(tmp_path: Path, cache: Path) -> None:
    """The VAE the configs lend a file without one is replaced (a new file):
    the next load converts again and links the new weights, not the old."""
    repo = _without_vae(tmp_path, vae_weights=True)
    _load(repo, "sd")
    before = len(_entries(cache))
    lent = repo / "vae" / WEIGHTS["vae"]
    tensors, _ = read_file(lent)
    replaced = tmp_path / "replaced.safetensors"
    write_file({name: np.asarray(value) * 2 for name, value in tensors.items()}, replaced, {"format": "pt"})
    os.replace(replaced, lent)
    loaded = _load(repo, "sd")
    assert len(_entries(cache)) == before + 1
    assert (loaded.source / "vae" / WEIGHTS["vae"]).read_bytes() == lent.read_bytes()


@pytest.mark.parametrize("code", [errno.EXDEV, errno.EPERM, errno.ENOTSUP])
def test_weights_that_cannot_be_hard_linked_are_copied(code: int, tmp_path: Path, cache: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _without_vae(tmp_path, vae_weights=True)

    def cross_device(source: object, destination: object) -> None:
        raise OSError(code, os.strerror(code), source, None, destination)

    monkeypatch.setattr(os, "link", cross_device)
    vae = _load(repo, "sd").source / "vae" / WEIGHTS["vae"]
    assert vae.read_bytes() == (repo / "vae" / WEIGHTS["vae"]).read_bytes()
    assert vae.stat().st_nlink == 1


# FLUX.1's width and head size, which diffusers' key map hard-codes; one block of each kind.
FLUX = {"attention_head_dim": 128, "num_attention_heads": 24, "axes_dims_rope": [16, 56, 56],
        "num_layers": 1, "num_single_layers": 1}


def _bfl_names(width: int, config: dict[str, object]) -> dict[str, tuple[int, ...]]:
    """BFL's Flux tensor names and shapes (black-forest-labs/flux, model.py)."""
    mlp, head = 4 * width, int(config["attention_head_dim"])
    linear = {
        "time_in.in_layer": (width, 256),
        "time_in.out_layer": (width, width),
        "vector_in.in_layer": (width, int(config["pooled_projection_dim"])),
        "vector_in.out_layer": (width, width),
        "guidance_in.in_layer": (width, 256),
        "guidance_in.out_layer": (width, width),
        "txt_in": (width, int(config["joint_attention_dim"])),
        "img_in": (width, int(config["in_channels"])),
        "single_blocks.0.modulation.lin": (3 * width, width),
        "single_blocks.0.linear1": (3 * width + mlp, width),
        "single_blocks.0.linear2": (width, width + mlp),
        "final_layer.linear": (int(config["in_channels"]), width),
        "final_layer.adaLN_modulation.1": (2 * width, width),
    }
    norms = ["single_blocks.0.norm.query_norm.scale", "single_blocks.0.norm.key_norm.scale"]
    for stream in ("img", "txt"):
        block = f"double_blocks.0.{stream}"
        linear |= {f"{block}_mod.lin": (6 * width, width), f"{block}_attn.qkv": (3 * width, width),
                   f"{block}_attn.proj": (width, width), f"{block}_mlp.0": (mlp, width),
                   f"{block}_mlp.2": (width, mlp)}
        norms += [f"{block}_attn.norm.query_norm.scale", f"{block}_attn.norm.key_norm.scale"]
    shapes = {f"{name}.weight": shape for name, shape in linear.items()}
    shapes |= {f"{name}.bias": shape[:1] for name, shape in linear.items()}
    return shapes | dict.fromkeys(norms, (head,))


def test_a_transformer_only_flux_file_converts_as_from_single_file_does_and_loads(
        tmp_path: Path, cache: Path) -> None:
    """The file carries only the transformer; the text encoders and VAE come
    from the configs' own weights (the committed tiny Flux pipeline's)."""
    import torch
    from diffusers import FluxTransformer2DModel

    repo = tmp_path / "flux"
    with tarfile.open(Path(__file__).parent / "fixtures" / "flux_source.tar.xz") as archive:
        archive.extractall(tmp_path / "source", filter="data")
    shutil.copytree(tmp_path / "source" / "pipeline", repo)
    (repo / "transformer" / WEIGHTS["transformer"]).unlink()
    config = json.loads((repo / "transformer" / "config.json").read_text()) | FLUX
    (repo / "transformer" / "config.json").write_text(json.dumps(config))
    width = FLUX["attention_head_dim"] * FLUX["num_attention_heads"]
    generator = torch.Generator().manual_seed(0)
    state = {name: (0.02 * torch.randn(shape, generator=generator)).to(torch.bfloat16)
             for name, shape in _bfl_names(width, config).items()}
    from safetensors.torch import save_file

    save_file(state, repo / "flux1.safetensors")
    del state
    expected = FluxTransformer2DModel.from_single_file(
        repo / "flux1.safetensors", config=str(repo), subfolder="transformer", torch_dtype=torch.bfloat16)
    digests = {name: hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
               + f":{value.dtype}" for name, value in expected.state_dict().items()}
    del expected
    loaded = Pretrained.load(repo, single_file="flux1.safetensors", dtype="float32", param_dtype="auto")
    _assert_converted(loaded.source, {"transformer": digests})
    for component in ("vae", "text_encoder", "text_encoder_2"):
        assert (loaded.source / component / WEIGHTS[component]).read_bytes() == (
            repo / component / WEIGHTS[component]).read_bytes()


@pytest.mark.network
def test_the_released_sd15_file_converts_to_the_tensors_diffusers_from_single_file_does(cache: Path) -> None:
    loaded = Pretrained.load("Comfy-Org/stable-diffusion-v1-5-archive",
                             single_file="v1-5-pruned-emaonly-fp16.safetensors",
                             revision="9cfd069101959ca3828bf9c04a4419870832b74f", param_dtype="auto")
    _assert_converted(loaded.source, json.loads((FIXTURE / "sd15_from_single_file.json").read_text()))
