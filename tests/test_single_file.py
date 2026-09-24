"""An original-format single-file checkpoint loads as the diffusers pipeline it describes.

The fixture (tools/single_file_fixture.py) is a tiny random Stable Diffusion
1.x pipeline written as one LDM-format file by diffusers' own reverse script,
next to the pipeline's configs, and a SHA-256 of every tensor diffusers'
`from_single_file(file, config=...)` converted it back to. The network test
checks the released SD 1.5 fp16 file the same way.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from dew.interop import load_pretrained
from dew.interop.safetensors_io import read_file

pytest.importorskip("diffusers", reason="single-file conversion runs diffusers' own key maps")

FIXTURE = Path(__file__).parent / "fixtures" / "single_file"
FILE = "tiny-sd-ldm.safetensors"
WEIGHTS = {"unet": "diffusion_pytorch_model.safetensors", "vae": "diffusion_pytorch_model.safetensors",
           "text_encoder": "model.safetensors"}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    directory = tmp_path / "repo"
    shutil.copytree(FIXTURE / "configs", directory)
    shutil.copy(FIXTURE / FILE, directory / FILE)
    return directory


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() + f":torch.{value.dtype.name}"


def test_a_single_file_converts_to_the_tensors_diffusers_from_single_file_does(repo: Path) -> None:
    load_pretrained(repo, single_file=FILE, dtype="float32", param_dtype="auto")
    [converted] = (repo.parent / "cache" / "dew" / "single_file").iterdir()
    _assert_converted(converted, json.loads((FIXTURE / "from_single_file.json").read_text()))


def _assert_converted(converted: Path, expected: dict[str, dict[str, str]]) -> None:
    for component, filename in WEIGHTS.items():
        tensors, _ = read_file(converted / component / filename)
        # transformers 5.6 dropped CLIPTextModel's `text_model.` level, so a
        # name compares without it. CLIP's position_ids is a buffer rebuilt
        # from the config, not a weight: a released file may store it and
        # diffusers' state dict lists it whether or not it does.
        ours = {name.removeprefix("text_model."): _digest(np.asarray(value)) for name, value in tensors.items()
                if not name.endswith("position_ids")}
        theirs = {name.removeprefix("text_model."): digest for name, digest in expected[component].items()
                  if not name.endswith("position_ids")}
        assert ours == theirs, component


def test_a_second_load_reads_the_converted_directory(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dew.interop import single_file

    load_pretrained(repo, single_file=FILE, dtype="float32", param_dtype="auto")

    def refused(*_: object) -> None:
        raise AssertionError("the file was converted twice")

    monkeypatch.setattr(single_file, "_component", refused)
    load_pretrained(repo, single_file=FILE, dtype="float32", param_dtype="auto")


@pytest.mark.network
def test_the_released_sd15_file_converts_to_the_tensors_diffusers_from_single_file_does(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    load_pretrained("Comfy-Org/stable-diffusion-v1-5-archive", single_file="v1-5-pruned-emaonly-fp16.safetensors",
                    revision="9cfd069101959ca3828bf9c04a4419870832b74f", param_dtype="auto")
    [converted] = (tmp_path / "dew" / "single_file").iterdir()
    _assert_converted(converted, json.loads((FIXTURE / "sd15_from_single_file.json").read_text()))
