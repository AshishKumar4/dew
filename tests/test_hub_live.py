"""Hub resolution, download and push through the real `huggingface_hub` client.

The other Hub tests stand a recording fake in for the client
(tests/test_pretrained_sources.py) or for `HfApi` (tests/test_interop.py).
Here the client is the real one, against the Hub: a tiny decoder bundle is
pushed with `push_to_hub`, and a tiny Stable Diffusion pipeline is uploaded,
each to a throwaway private repo under dewml with other copies of its
weights beside it (Mistral's consolidated file; fp16 variants, Flax copies,
a root single-file checkpoint and a folder the index does not declare).
Resolving them reads the commit's real listing, downloads into a fresh hub
cache only the files that hold the weights, loads what the same files give
locally, and loads again offline from that cache. The repos are deleted at
the end.

Network-marked, and run with DEW_NETWORK_TESTS=1 and a Hub token that can
write to dewml.
"""

import json
import os
import shutil
import tarfile
import uuid
from pathlib import Path

import jax
import numpy as np
import pytest

safetensors_numpy = pytest.importorskip("safetensors.numpy")

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get("DEW_NETWORK_TESTS") != "1",
                       reason="DEW_NETWORK_TESTS=1 pushes to and reads from throwaway Hub repos"),
]
ORGANIZATION = "dewml"
# Every repo these tests create is named under this prefix, and the cleanup
# deletes nothing that is not.
PREFIX = f"{ORGANIZATION}/dew-hub-test-"


@pytest.fixture(scope="module")
def api():
    from huggingface_hub import HfApi, get_token

    if get_token() is None:
        pytest.skip("no Hub token to push the throwaway repos with")
    return HfApi()


@pytest.fixture(scope="module")
def throwaway(api):
    """A fresh repo name under dewml, deleted when the module is done: only
    names this fixture made, each under PREFIX with a random suffix."""
    created = []

    def name(kind: str) -> str:
        repo = f"{PREFIX}{kind}-{uuid.uuid4().hex[:8]}"
        created.append(repo)
        return repo

    yield name
    for repo in created:
        if not repo.startswith(PREFIX):
            raise AssertionError(f"refusing to delete {repo}, which these tests did not name")
        api.delete_repo(repo, missing_ok=True)


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    """An empty hub cache, so every file a load reads is a file it fetched."""
    from huggingface_hub import constants

    cache = tmp_path / "hub-cache"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    return cache


def files_under(directory: Path) -> set[str]:
    return {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}


def offline(monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)


def test_a_pushed_decoder_resolves_downloads_and_loads_through_the_hub(api, throwaway, fresh_cache, tmp_path,
                                                                       monkeypatch):
    from dew.interop import PretrainedDecoder, sources
    from dew.interop.hub import pull_from_hub
    from dew.interop.pretrained import Pretrained
    from dew.nn.backbones import CausalTransformer

    repo = throwaway("decoder")
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1, num_heads=2, mlp_features=16,
                              max_seq_len=8, attention_impl="reference")
    bundle = PretrainedDecoder.from_model(model, model.init(jax.random.key(0), np.zeros((1, 2), np.int32)),
                                          tokenizer="byte")
    bundle.push_to_hub(repo, private=True, commit_message="dew live hub test")
    bundle.save(tmp_path / "saved")
    pushed = api.model_info(repo)
    assert pushed.private
    assert api.list_repo_commits(repo)[0].title == "dew live hub test"
    assert set(api.list_repo_files(repo)) - {".gitattributes"} == files_under(tmp_path / "saved")

    decoy = tmp_path / "consolidated.safetensors"
    safetensors_numpy.save_file({"tok_embeddings.weight": np.ones((16, 8), np.float32)}, str(decoy))
    api.upload_file(path_or_fileobj=str(decoy), path_in_repo="consolidated.safetensors", repo_id=repo)
    commit = api.model_info(repo).sha

    directory = sources.snapshot(repo, None)
    assert directory.name == commit and directory.is_relative_to(fresh_cache)
    assert files_under(directory) == files_under(tmp_path / "saved")

    ids = np.arange(6, dtype=np.int32).reshape(1, 6) % 16
    local = Pretrained.load(str(tmp_path / "saved"), dtype="float32")
    remote = Pretrained.load(repo, dtype="float32")
    assert remote.revision == commit
    expected = np.asarray(local.model.apply(local.variables, ids))
    np.testing.assert_array_equal(np.asarray(remote.model.apply(remote.variables, ids)), expected)

    with monkeypatch.context() as patched:
        offline(patched)
        cached = Pretrained.load(repo, dtype="float32")
        assert cached.revision == commit
        np.testing.assert_array_equal(np.asarray(cached.model.apply(cached.variables, ids)), expected)

    pulled = pull_from_hub(repo, revision=commit)
    assert pulled == directory and "consolidated.safetensors" in files_under(pulled)


def test_a_pipeline_downloads_only_the_component_weights_it_reads(api, throwaway, fresh_cache, tmp_path,
                                                                  monkeypatch):
    from dew.interop.pretrained import Pretrained

    repo = throwaway("pipeline")
    with tarfile.open(ROOT / "tests" / "fixtures" / "tiny_diffusers.tar.xz") as archive:
        archive.extractall(tmp_path, filter="data")
    folder = tmp_path / "sd"
    published = folder.parent / "published"
    shutil.copytree(folder, published)
    unet = published / "unet" / "diffusion_pytorch_model.safetensors"
    shutil.copyfile(unet, published / "unet" / "diffusion_pytorch_model.fp16.safetensors")
    shutil.copytree(published / "vae", published / "vae_1_0")
    shutil.copyfile(unet, published / "v1-5-pruned.safetensors")
    declared = set(json.loads((folder / "model_index.json").read_text()))
    declared -= {"_class_name", "_diffusers_version"}
    assert "vae_1_0" not in declared
    api.create_repo(repo, private=True)
    api.upload_folder(repo_id=repo, folder_path=str(published), commit_message="dew live hub test")
    commit = api.model_info(repo).sha

    remote = Pretrained.load(repo, dtype="float32")
    assert remote.revision == commit and remote.source.name == commit
    weights = {name for name in files_under(remote.source)
               if name.endswith((".safetensors", ".msgpack", ".bin"))}
    assert weights == {name for name in files_under(folder)
                       if name.endswith("model.safetensors") and name.split("/")[0] in declared}
    local = Pretrained.load(str(folder), dtype="float32")
    for (path, mine), theirs in zip(jax.tree_util.tree_leaves_with_path(remote.variables),
                                    jax.tree_util.tree_leaves(local.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs),
                                      err_msg=jax.tree_util.keystr(path))

    with monkeypatch.context() as patched:
        offline(patched)
        cached = Pretrained.load(repo, dtype="float32")
        assert cached.revision == commit and cached.source == remote.source
