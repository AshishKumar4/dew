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

from dew.interop import load_pretrained, pretrained
from dew.interop.safetensors_io import read_file, write_file

pytest.importorskip("diffusers", reason="single-file conversion runs diffusers' own key maps")
single_file = pytest.importorskip("dew.interop.single_file")

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
    return load_pretrained(directory, single_file=f"{kind}.safetensors", dtype="float32", param_dtype="auto")


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


def test_a_cached_load_reads_nothing_from_the_file(tmp_path: Path, cache: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, "sd")
    _load(repo, "sd")

    def refused(*_: object, **__: object) -> None:
        raise AssertionError("the single file was read again")

    monkeypatch.setattr(single_file, "read_file", refused)
    _load(repo, "sd")
    assert len(_entries(cache)) == 1


def test_another_diffusers_converts_again(tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import diffusers

    repo = _repo(tmp_path, "sd")
    _load(repo, "sd")
    monkeypatch.setattr(diffusers, "__version__", "0.0.0-other")
    _load(repo, "sd")
    assert len(_entries(cache)) == 2


def test_a_failed_load_caches_nothing(tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, "sd")
    load = pretrained._load_diffusion_source

    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(pretrained, "_load_diffusion_source", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _load(repo, "sd")
    assert list(cache.iterdir()) == []
    monkeypatch.setattr(pretrained, "_load_diffusion_source", load)
    _load(repo, "sd")
    assert [entry.name.startswith(".") for entry in cache.iterdir()] == [False]


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


def test_a_component_with_no_weights_anywhere_is_refused_by_name(tmp_path: Path, cache: Path) -> None:
    repo = _without_vae(tmp_path, vae_weights=False)
    with pytest.raises(ValueError, match=r"no weights for \['vae'\]"):
        _load(repo, "sd")
    assert _entries(cache) == []


def test_a_hub_repo_gives_the_missing_weights_by_the_snapshot_rule_at_its_commit(
        tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_pretrained("org/repo", single_file=...) for a repo whose metadata
    snapshot holds the configs and a file without its VAE: the VAE comes from
    the repo at the snapshot's commit, fetched and linked by `weight_files`'
    rule, so the fp16 variant beside it stays behind."""
    from dew.interop import hf_decoders

    metadata = _without_vae(tmp_path, vae_weights=False)
    hub = tmp_path / "hub-weights"
    (hub / "vae").mkdir(parents=True)
    _vae_weights(tmp_path / "whole", hub / "vae")
    shutil.copy(hub / "vae" / WEIGHTS["vae"], hub / "vae" / "diffusion_pytorch_model.fp16.safetensors")
    asked = []

    def snapshot(name: str, revision: str | None, *, weights: bool | tuple[str, ...] = True) -> Path:
        asked.append((name, revision, weights))
        return metadata if weights is False else hub

    monkeypatch.setattr(hf_decoders, "_snapshot", snapshot)
    monkeypatch.setattr(hf_decoders, "repo_file", lambda name, directory, filename: metadata / filename)
    loaded = load_pretrained("org/sd-tiny", single_file="sd.safetensors", dtype="float32", param_dtype="auto")
    assert asked == [("org/sd-tiny", None, False), ("org/sd-tiny", metadata.name, ("vae",))]
    assert loaded.revision == metadata.name
    assert sorted(path.name for path in (loaded.source / "vae").iterdir()) == ["config.json", WEIGHTS["vae"]]
    assert (loaded.source / "vae" / WEIGHTS["vae"]).read_bytes() == (hub / "vae" / WEIGHTS["vae"]).read_bytes()


def test_only_the_configs_metadata_is_copied(tmp_path: Path, cache: Path) -> None:
    """Other formats of the weights beside the configs (an ONNX export, a
    precision variant, a weights index) and anything else stay out."""
    repo = _repo(tmp_path, "sd")
    for extra in ("unet/model.onnx", "unet/model.onnx_data", "unet/diffusion_pytorch_model.fp16.safetensors",
                  "unet/diffusion_pytorch_model.safetensors.index.json", "README.md"):
        (repo / extra).write_bytes(b"{}")
    loaded = _load(repo, "sd")
    copied = {path.relative_to(loaded.source).as_posix() for path in loaded.source.rglob("*") if path.is_file()}
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


@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e4m3fn", "float16"])
def test_a_stored_tensor_is_viewed_in_torch_without_a_copy(dtype: str) -> None:
    import ml_dtypes
    import torch

    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4).astype(getattr(ml_dtypes, dtype, dtype))
    tensor = single_file._torch_view(array)
    assert tensor.data_ptr() == array.ctypes.data
    assert tensor.dtype == getattr(torch, dtype)
    np.testing.assert_array_equal(tensor.float().numpy(), array.astype(np.float32))


# FLUX.1's width and head size, which diffusers' key map hard-codes; one block of each kind.
FLUX = {"attention_head_dim": 128, "num_attention_heads": 24, "axes_dims_rope": [16, 56, 56],
        "num_layers": 1, "num_single_layers": 1}


def _bfl_names(width: int, config: dict[str, object]) -> dict[str, tuple[int, ...]]:
    """BFL's Flux tensor names and shapes (black-forest-labs/flux, model.py)."""
    mlp, head = 4 * width, int(config["attention_head_dim"])
    linear = {"time_in.in_layer": (width, 256), "time_in.out_layer": (width, width),
              "vector_in.in_layer": (width, int(config["pooled_projection_dim"])),
              "vector_in.out_layer": (width, width), "guidance_in.in_layer": (width, 256),
              "guidance_in.out_layer": (width, width), "txt_in": (width, int(config["joint_attention_dim"])),
              "img_in": (width, int(config["in_channels"])),
              "single_blocks.0.modulation.lin": (3 * width, width),
              "single_blocks.0.linear1": (3 * width + mlp, width), "single_blocks.0.linear2": (width, width + mlp),
              "final_layer.linear": (int(config["in_channels"]), width),
              "final_layer.adaLN_modulation.1": (2 * width, width)}
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
    loaded = load_pretrained(repo, single_file="flux1.safetensors", dtype="float32", param_dtype="auto")
    _assert_converted(loaded.source, {"transformer": digests})
    for component in ("vae", "text_encoder", "text_encoder_2"):
        assert (loaded.source / component / WEIGHTS[component]).read_bytes() == (
            repo / component / WEIGHTS[component]).read_bytes()


@pytest.mark.network
def test_the_released_sd15_file_converts_to_the_tensors_diffusers_from_single_file_does(cache: Path) -> None:
    loaded = load_pretrained("Comfy-Org/stable-diffusion-v1-5-archive",
                             single_file="v1-5-pruned-emaonly-fp16.safetensors",
                             revision="9cfd069101959ca3828bf9c04a4419870832b74f", param_dtype="auto")
    _assert_converted(loaded.source, json.loads((FIXTURE / "sd15_from_single_file.json").read_text()))


RELEASED = {
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors",
             "462165984030d82259a11f4367a4eed129e94a7b"),
    "flux-dev": ("Comfy-Org/flux1-dev", "flux1-dev.safetensors", "83c446ef27a6ac1e9e36ecf13257283aa12cf22a"),
    "flux-schnell": ("Comfy-Org/flux1-schnell", "flux1-schnell.safetensors",
                     "c2b683ea00713d6feadcd54b39e3725bbc78638b"),
}


def _header(repo: str, filename: str, revision: str) -> dict:
    """A released file's tensors as meta tensors, from its safetensors header alone."""
    import torch
    from huggingface_hub import HfApi

    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    header = HfApi().parse_safetensors_file_metadata(repo, filename, revision=revision)
    return {name: torch.empty(info.shape, dtype=dtypes[info.dtype], device="meta")
            for name, info in header.tensors.items()}


def _shapes(tensors) -> dict[str, tuple[int, ...]]:
    return {name.removeprefix("text_model."): tuple(value.shape) for name, value in tensors.items()
            if not name.endswith("position_ids")}


@pytest.mark.network
def test_the_released_sdxl_header_converts_to_the_repo_s_own_diffusers_layout() -> None:
    """Every component sd_xl_base_1.0.safetensors carries converts to exactly
    the names and shapes of the same repo's diffusers-format weights."""
    from huggingface_hub import HfApi

    repo, filename, revision = RELEASED["sdxl"]
    checkpoint = _header(repo, filename, revision)
    assert single_file.config_repo(checkpoint) == repo
    configs = single_file._configs(repo)
    index = single_file.pipeline_index(configs)
    seen = []
    for name, _, tensors in single_file.converted(checkpoint, configs, index):
        assert tensors is not None, name
        published = HfApi().parse_safetensors_file_metadata(repo, f"{name}/{WEIGHTS[name]}", revision=revision)
        assert _shapes(tensors) == {key.removeprefix("text_model."): tuple(info.shape)
                                    for key, info in published.tensors.items()
                                    if not key.endswith("position_ids")}, name
        assert {str(value.dtype) for value in tensors.values()} <= {"torch.float16"}, name
        seen.append(name)
    assert sorted(seen) == ["text_encoder", "text_encoder_2", "unet", "vae"]


@pytest.mark.network
@pytest.mark.parametrize("kind", ["flux-dev", "flux-schnell"])
def test_the_released_flux_header_converts_to_the_diffusers_transformer(kind: str, tmp_path: Path) -> None:
    """black-forest-labs' config repos are gated, so the configs are diffusers'
    own FLUX.1 defaults; the converted names and shapes are exactly those of
    FluxTransformer2DModel built from them."""
    import torch
    from diffusers import FluxTransformer2DModel

    checkpoint = _header(*RELEASED[kind])
    guided = kind == "flux-dev"
    assert single_file.config_repo(checkpoint) == ("black-forest-labs/FLUX.1-dev" if guided
                                                   else "black-forest-labs/FLUX.1-schnell")
    config = {"guidance_embeds": guided}
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(config))
    index = {"transformer": ["diffusers", "FluxTransformer2DModel"]}
    [(_, _, tensors)] = single_file.converted(checkpoint, tmp_path, index)
    assert tensors is not None
    with torch.device("meta"):
        model = FluxTransformer2DModel(**config)
    assert _shapes(tensors) == _shapes(model.state_dict())
