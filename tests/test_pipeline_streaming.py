"""A diffusion pipeline loaded onto a mesh streams its weights.

`Pretrained.load(..., mesh=)` reads a pipeline's denoiser and text encoders
as `SourceLeaf` recipes over the mapped files and places them one leaf at a
time, as it already does a decoder's. These tests hold the streamed load to
the eager one on the committed tiny pipelines of every family: the same
tree, leaf for leaf and bit for bit, and the same samples.
"""

import tarfile
from pathlib import Path

import jax
import numpy as np
import pytest

from dew.interop.diffusion import component_tensors, translate_wan_weights
from dew.interop.pretrained import Pretrained, load_diffusion_source
from dew.interop.streaming import SourceLeaf
from dew.training import Layout, MeshSpec

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = Layout(min_shard=1, tolerance=1.0)
# Each family's tiny pipeline: its archive and the directory inside it.
PIPELINES = {
    "sd": ("tiny_diffusers", "sd"),
    "xl": ("tiny_diffusers", "xl"),
    "safety": ("tiny_diffusers", "safety"),
    "flux": ("flux_source", "pipeline"),
    "sd3": ("sd3_source", "pipeline"),
    "z_image": ("z_image_source", "pipeline"),
    "wan": ("wan_pipeline", "pipeline"),
}


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipelines")
    for archive in sorted({name for name, _ in PIPELINES.values()}):
        with tarfile.open(ROOT / f"tests/fixtures/{archive}.tar.xz") as opened:
            opened.extractall(root / archive, filter="data")
    return {family: root / archive / inside for family, (archive, inside) in PIPELINES.items()}


def flat(tree) -> dict[str, object]:
    return {jax.tree_util.keystr(path): leaf for path, leaf in jax.tree_util.tree_leaves_with_path(tree)}


@pytest.mark.mesh
@pytest.mark.parametrize("family", sorted(PIPELINES))
@pytest.mark.parametrize("param_dtype", ["float32", "bfloat16"])
def test_a_streamed_pipeline_holds_the_eager_tree_and_reads_one_shard_at_a_time(
        extracted, family, param_dtype, monkeypatch):
    """Every placed leaf is the eager load's, bit for bit, carries the
    layout's sharding, and every read of a recipe is one device's shard of
    it; the UNet's per-head attention kernels and every convolution are
    read whole and land the same."""
    eager = Pretrained.load(str(extracted[family]), dtype="float32", param_dtype=param_dtype,
                            attention_impl="xla")
    reads: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    read = SourceLeaf.read

    def recorded(leaf: SourceLeaf, index: tuple[slice, ...] | None = None) -> np.ndarray:
        value = read(leaf, index)
        reads.append((leaf.shape, value.shape))
        return value

    monkeypatch.setattr(SourceLeaf, "read", recorded)
    streamed = Pretrained.load(str(extracted[family]), dtype="float32", param_dtype=param_dtype,
                               attention_impl="xla", mesh=MeshSpec(fsdp=jax.device_count()), layout=LAYOUT)
    expected, placed = flat(eager.variables), flat(streamed.variables)
    assert placed.keys() == expected.keys()
    shards = set()
    for name, leaf in placed.items():
        assert isinstance(leaf, jax.Array), name
        want = np.asarray(expected[name])
        got = np.asarray(jax.device_get(leaf))
        assert got.dtype == want.dtype and got.shape == want.shape, name
        assert got.tobytes() == want.tobytes(), name
        shards.add((leaf.shape, leaf.sharding.shard_shape(leaf.shape)))
    assert any(shard != shape for shape, shard in shards)
    assert reads and set(reads) <= shards, sorted(set(reads) - shards)[:3]


@pytest.mark.parametrize("family", ["flux", "z_image", "wan"])
def test_a_streamed_pipeline_samples_what_the_eager_one_does(extracted, family):
    """The conditioner and the autoencoder hold the placed leaves, so a
    prompt runs end to end on them."""
    eager = Pretrained.load(str(extracted[family]), dtype="float32", attention_impl="xla")
    streamed = Pretrained.load(str(extracted[family]), dtype="float32", attention_impl="xla", mesh=MeshSpec())

    def sampled(source):
        return np.asarray(source.text_to_image()(["a red bird"], steps=2, key=3).host().images)

    np.testing.assert_array_equal(sampled(streamed), sampled(eager))


def test_load_diffusion_source_streams_at_its_own_geometry(extracted):
    """`load_diffusion_source` takes the mesh too, beside the clip geometry
    only it takes."""
    loaded = load_diffusion_source(str(extracted["wan"]), dtype="float32", size=(5, 16, 32), mesh=MeshSpec())
    assert loaded.inputs.sample.shape == (5, 16, 32, 3)
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(loaded.variables))


def test_a_lazy_translation_reads_nothing_until_placed(extracted):
    """Lazily, a linear kernel and an untransposed leaf are recipes over the
    stored tensor; a convolution, whose transpose a recipe does not express,
    is read whole. Eagerly, nothing is a recipe."""
    tensors = component_tensors(extracted["wan"], "transformer")
    lazy, _ = translate_wan_weights(tensors, lazy=True)
    kinds = {name: type(leaf) for name, leaf in flat(lazy).items()}
    assert kinds["['blocks_0']['attn1']['to_q']['kernel']"] is SourceLeaf
    assert kinds["['scale_shift_table']"] is SourceLeaf
    assert kinds["['patch_embedding_3d']['kernel']"] is np.ndarray
    eager, _ = translate_wan_weights(tensors)
    assert all(type(leaf) is np.ndarray for leaf in jax.tree.leaves(eager))
