"""A DDUF file loads as the diffusers pipeline it packs, as Diffusers reads it.

The committed Flux pipeline (tests/fixtures/flux_source.tar.xz) is packed with
huggingface_hub's own `export_folder_as_dduf`. `Pretrained.load(...,
dduf_file=)` unpacks it once into Dew's cache and loads it as that directory;
Diffusers 0.34.0's own `DiffusionPipeline.from_pretrained(..., dduf_file=)`
reads the same file, and its transformer is the reference Dew's prediction
is held to (tests/reference_error.py).
"""

import json
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import extract_fixture, packed, unpacked
from reference_error import assert_as_exact_as_the_reference

from dew.diffusion.process import DenoisingCondition
from dew.interop import Pretrained
from tools.diffusers_wan_reference import float64

PUBLISHED = ("DDUF/tiny-flux-dev-pipe-dduf", "fluxpipeline.dduf", "4fa81aa2c667f7f09ea7f2913be771e6f927dcd9")


@pytest.fixture
def packed_pipeline(tmp_path, monkeypatch):
    """The committed Flux pipeline packed into `repo/pipeline.dduf`, with
    Dew's cache under `tmp_path`."""
    from huggingface_hub import export_folder_as_dduf

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    extract_fixture(Path(__file__).parent / "fixtures" / "flux_source.tar.xz", tmp_path / "source")
    repo = tmp_path / "repo"
    repo.mkdir()
    export_folder_as_dduf(repo / "pipeline.dduf", tmp_path / "source" / "pipeline")
    return tmp_path / "source" / "pipeline", repo


def flux_inputs(bundle, seed: int = 0) -> dict[str, np.ndarray]:
    """A packed latent, a text context, a pooled row, the distilled guidance
    where the transformer embeds one, and two model times, at the widths of
    the transformer `bundle` loaded."""
    transformer_config = json.loads((Path(bundle.source) / "transformer" / "config.json").read_text())
    generator = np.random.default_rng(seed)
    grid = 4
    latent = generator.standard_normal((2, 2 * grid, 2 * grid, transformer_config["in_channels"] // 4))
    widths = (transformer_config["joint_attention_dim"], transformer_config["pooled_projection_dim"])
    return {"packed": packed(latent.astype(np.float32)),
            "context": generator.standard_normal((2, 6, widths[0]), np.float32),
            "pooled": generator.standard_normal((2, widths[1]), np.float32),
            "guidance": np.full((2,), 3.5, np.float32) if transformer_config.get("guidance_embeds") else None,
            "times": np.asarray([700.0, 150.0], np.float32)}


def dew_prediction(bundle, inputs) -> np.ndarray:
    grid = int(np.sqrt(inputs["packed"].shape[1]))
    guidance = None if inputs["guidance"] is None else jnp.asarray(inputs["guidance"])
    condition = DenoisingCondition(jnp.asarray(inputs["context"]), jnp.asarray(inputs["pooled"]),
                                   guidance=guidance)
    own = {"params": bundle.variables["params"]}
    return np.asarray(bundle.model.apply(own, jnp.asarray(unpacked(inputs["packed"], grid, grid)),
                                         jnp.asarray(inputs["times"]), condition))


def diffusers_prediction(repo: str | Path, dduf_file: str, inputs, *, wide: bool,
                         revision: str | None = None):
    """Diffusers' own DDUF load of the pipeline, and its transformer's
    prediction on `inputs` with FluxPipeline's own position ids, in float32
    or in float64, read back into Dew's layout."""
    import contextlib

    import torch
    from diffusers import DiffusionPipeline, FluxPipeline

    dtype = torch.float64 if wide else torch.float32
    with float64() if wide else contextlib.nullcontext():
        pipe = DiffusionPipeline.from_pretrained(str(repo), dduf_file=dduf_file, torch_dtype=dtype,
                                                 revision=revision)
        model = pipe.transformer.eval()
        grid = int(np.sqrt(inputs["packed"].shape[1]))

        def tensor(value):
            return torch.from_numpy(np.asarray(value)).to(dtype)

        positions = FluxPipeline._prepare_latent_image_ids(1, grid, grid, "cpu", dtype)
        with torch.no_grad():
            output = model(hidden_states=tensor(inputs["packed"]),
                           encoder_hidden_states=tensor(inputs["context"]),
                           pooled_projections=tensor(inputs["pooled"]),
                           timestep=tensor(inputs["times"]) / 1000,
                           guidance=None if inputs["guidance"] is None else tensor(inputs["guidance"]),
                           img_ids=positions,
                           txt_ids=tensor(np.zeros((inputs["context"].shape[1], 3)))).sample
    return unpacked(output.numpy(), grid, grid)


def test_a_dduf_file_loads_what_diffusers_reads_from_it_and_unpacks_once(packed_pipeline, tmp_path,
                                                                         monkeypatch):
    """Dew's load of the packed file is its load of the directory the file
    packs, leaf for leaf; each unpacked file is the bytes huggingface_hub's
    own reader maps for that entry; Dew's Flux prediction holds the float64
    rule against the transformer Diffusers' own DDUF load builds; and a
    second load reads the archive no more."""
    from huggingface_hub import read_dduf_file

    from dew.interop import dduf

    directory, repo = packed_pipeline
    loaded = Pretrained.load(repo, dduf_file="pipeline.dduf", dtype="float32", attention_impl="xla")
    direct = Pretrained.load(directory, dtype="float32", attention_impl="xla")
    assert type(loaded.model) is type(direct.model)
    paths = jax.tree_util.tree_leaves_with_path(loaded.variables)
    expected_paths = [path for path, _ in jax.tree_util.tree_leaves_with_path(direct.variables)]
    assert [path for path, _ in paths] == expected_paths
    for (path, value), expected in zip(paths, jax.tree.leaves(direct.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(value), np.asarray(expected),
                                      err_msg=jax.tree_util.keystr(path))

    [written] = sorted((tmp_path / "cache" / "dew" / "dduf").iterdir())
    entries = read_dduf_file(repo / "pipeline.dduf")
    files = sorted(path.relative_to(written).as_posix() for path in written.rglob("*") if path.is_file())
    assert files == sorted(entries)
    for name, entry in entries.items():
        with entry.as_mmap() as mapped:
            assert (written / name).read_bytes() == bytes(mapped), name

    inputs = flux_inputs(loaded)
    theirs, truth = (diffusers_prediction(repo, "pipeline.dduf", inputs, wide=wide) for wide in (False, True))
    assert_as_exact_as_the_reference(dew_prediction(loaded, inputs), theirs, truth, "DDUF Flux")

    import huggingface_hub

    def unread(path):
        raise AssertionError(f"{path} was read again")

    monkeypatch.setattr(huggingface_hub, "read_dduf_file", unread)
    assert dduf.unpacked(repo / "pipeline.dduf") == written


def test_a_dduf_file_the_repo_lacks_or_beside_a_single_file_is_refused(tmp_path):
    (tmp_path / "other.dduf").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match=r"the \.dduf files in .* are \['other\.dduf'\]"):
        Pretrained.load(tmp_path, dduf_file="pipeline.dduf")
    with pytest.raises(ValueError, match="pass one"):
        Pretrained.load(tmp_path, dduf_file="other.dduf", single_file="model.safetensors")


@pytest.mark.parametrize("name", ["../escaped.json", "/escaped.json"])
def test_an_entry_named_outside_the_archive_directory_is_refused(tmp_path, monkeypatch, name):
    """huggingface_hub 1.30.0's `read_dduf_file` checks each name with its
    slashes stripped but keys the entry by the name as written, so it lists
    these; the unpack refuses to write outside its own directory."""
    from dew.interop import dduf

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    archive = tmp_path / "crafted.dduf"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as out:
        # Its structure check wants the folder ('..', or '' before a leading
        # slash) named in the index and holding a config.
        out.writestr("model_index.json", '{"_class_name": "FluxPipeline", "": ["diffusers", "X"], '
                                         '"..": ["diffusers", "X"]}')
        out.writestr(name, "{}")
        out.writestr("/config.json" if name.startswith("/") else "../config.json", "{}")
    with pytest.raises(ValueError, match="outside its own directory"):
        dduf.unpacked(archive)
    # '..' from the staging directory is the cache's dduf directory.
    assert not (tmp_path / "cache" / "dew" / "dduf" / "escaped.json").exists()


@pytest.mark.network
def test_a_published_dduf_file_loads_what_diffusers_reads_from_it():
    """The same comparison on a DDUF file published on the Hub, at a pinned
    commit: Dew's Flux prediction against Diffusers' own DDUF load."""
    repo, file, revision = PUBLISHED
    loaded = Pretrained.load(repo, dduf_file=file, revision=revision, dtype="float32", attention_impl="xla")
    inputs = flux_inputs(loaded)
    assert_as_exact_as_the_reference(dew_prediction(loaded, inputs),
                                     diffusers_prediction(repo, file, inputs, wide=False, revision=revision),
                                     diffusers_prediction(repo, file, inputs, wide=True, revision=revision),
                                     "published DDUF Flux")
