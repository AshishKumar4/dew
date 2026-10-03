"""A diffusion pipeline loads in pieces that fit: streamed, and without its text encoder.

`Pretrained.load(..., mesh=)` reads a pipeline's weights as `SourceLeaf`
recipes over the mapped files and places them one leaf at a time, as it
already does a decoder's. These tests hold the streamed load to the eager
one on the committed tiny pipelines of every family: the same tree, leaf for
leaf and bit for bit, and the same samples.

`load_diffusion_source(text=False)` leaves the text encoder out, and a call
takes prompts its conditioner encoded alone (`prepare(conditions=...)`): the
same images as the prompted call, bit for bit.
"""

import shutil
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference

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


ENCODED = {"z_image": "HiddenStatesConditioner", "wan": "WanConditioner"}
PROMPTS = ["a red bird", "two cats on a mat"]


def encoded(directory: Path, conditioner: str, rows: list, placement: dict, *, widened: bool = False) -> dict:
    """`rows` encoded with the pipeline's conditioner loaded alone; `widened`
    runs its tower in float64 over float64 parameters (under x64), the truth
    tests/reference_error.py measures from, since a compute dtype is at most
    float32."""
    import dataclasses

    import dew.inputs.diffusion as conditioners

    encoder = getattr(conditioners, conditioner).from_pretrained(str(directory), dtype="float32", **placement)
    if widened:
        modules = [field.name for field in dataclasses.fields(encoder)
                   if isinstance(getattr(encoder, field.name), nn.Module)]
        towers = {name: getattr(encoder, name).clone(dtype=jnp.float64) for name in modules}
        params = jax.tree.map(lambda leaf: np.asarray(leaf, np.float64)
                              if np.issubdtype(leaf.dtype, np.floating) else np.asarray(leaf), encoder.params)
        encoder = dataclasses.replace(encoder, **towers, params=params)
    return {"conditioning": jax.jit(encoder.encode)(encoder.params, encoder.tokenize(rows))}


@pytest.mark.parametrize("family", sorted(ENCODED))
def test_an_encoded_call_without_the_text_encoder_draws_the_prompted_images(extracted, family):
    """The prompts and the blank negative encoded by the conditioner alone,
    walked by the pipeline loaded without it, guided at its own scale: the
    prompted call's images, bit for bit, from the same per-row noise."""
    directory = extracted[family]
    whole = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla")
    task = whole.text_to_image()
    prompted = task(task.prepare(PROMPTS, key=3, steps=2), key=3).host().images
    lean = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla", text=False)
    assert "conditioning" not in lean.variables["encoders"]
    blank = whole.inputs.conditions["conditioning"].unconditional
    lean_task = lean.text_to_image()
    conditioner = ENCODED[family]
    prepared = lean_task.prepare(conditions=encoded(directory, conditioner, PROMPTS, {}),
                                 unconditional=encoded(directory, conditioner, [blank], {}), key=3, steps=2)
    walked = lean_task(prepared, key=3).host().images
    assert np.asarray(walked).tobytes() == np.asarray(prompted).tobytes()


@pytest.mark.mesh
@pytest.mark.parametrize("family", sorted(ENCODED))
def test_on_a_mesh_an_encoded_call_is_the_prompted_call(extracted, family):
    """Over an FSDP mesh, which pads the rows to its devices: the encoded
    call draws the prompted call's noise bit for bit, and given the
    encodings the prompted call made it walks to its images bit for bit.
    The conditioner alone encodes two rows where the task encodes them
    padded and sharded, so XLA sums its contractions in another order: those
    encodings are held to tests/reference_error.py's rule against the same
    conditioner's float64 encoding, no further from it than twice the
    task's own, and their masks equal. Observed (RMS distance from float64,
    the conditioner alone's over the task's): Wan 1.008, Z-Image 1.053; the
    two float32 encodings lie at most 8.9e-7 apart."""
    directory = extracted[family]
    placement = {"mesh": MeshSpec(fsdp=jax.device_count()), "layout": LAYOUT}
    whole = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla", **placement)
    task = whole.text_to_image()
    prompted = task.prepare(PROMPTS, key=3, steps=2)
    lean = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla", text=False,
                                 **placement)
    lean_task = lean.text_to_image()
    rows = len(PROMPTS)
    own = lean_task.prepare(conditions=jax.tree.map(lambda leaf: leaf[:rows], prompted.conditions),
                            unconditional=prompted.unconditional, key=3, steps=2)
    assert np.asarray(own.noise).tobytes() == np.asarray(prompted.noise).tobytes()
    walked = lean_task(own, key=3).host().images
    assert np.asarray(walked).tobytes() == np.asarray(task(prompted, key=3).host().images).tobytes()
    alone = encoded(directory, ENCODED[family], PROMPTS, placement)["conditioning"]
    with jax.enable_x64(new_val=True):
        truth = encoded(directory, ENCODED[family], PROMPTS, {}, widened=True)["conditioning"]
        truth = [np.asarray(leaf) for leaf in jax.tree.leaves(truth)]
    task_own = [np.asarray(leaf)[:rows] for leaf in jax.tree.leaves(prompted.conditions["conditioning"])]
    for got, want, wide in zip(jax.tree.leaves(alone), task_own, truth, strict=True):
        if want.dtype == bool:
            np.testing.assert_array_equal(np.asarray(got), want)
            np.testing.assert_array_equal(wide, want)
        else:
            assert wide.dtype == np.float64
            assert_as_exact_as_the_reference(np.asarray(got), want, wide, f"{family} conditioner alone")


def test_a_pipeline_without_its_text_encoder_reads_none_of_its_weights(extracted, tmp_path):
    """With the text encoder's weight files gone, `text=False` loads and the
    whole pipeline does not."""
    directory = tmp_path / "wan"
    shutil.copytree(extracted["wan"], directory)
    for weights in (directory / "text_encoder").glob("*.safetensors*"):
        weights.unlink()
    load_diffusion_source(str(directory), dtype="float32", text=False)
    with pytest.raises(FileNotFoundError):
        load_diffusion_source(str(directory), dtype="float32")


def test_a_pipeline_without_its_text_encoder_refuses_what_reads_it(extracted, tmp_path):
    """A prompt, training and saving each read the text encoder; each is
    refused by name, never at a missing key. An encoded call with no
    unconditional branch walks unguided and refuses guidance."""
    lean = load_diffusion_source(str(extracted["wan"]), dtype="float32", attention_impl="xla", text=False)
    task = lean.text_to_image()
    with pytest.raises(ValueError, match="text=False"):
        task(PROMPTS, steps=2, key=0)
    with pytest.raises(ValueError, match="text=False"):
        lean.diffusion_objective()
    with pytest.raises(ValueError, match="text=False"):
        lean.save(tmp_path / "saved")
    given = encoded(extracted["wan"], "WanConditioner", PROMPTS, {})
    with pytest.raises(ValueError, match="one of the two"):
        task.prepare(PROMPTS, conditions=given, key=0, steps=2)
    unguided = task.prepare(conditions=given, key=0, steps=2)
    assert not unguided.unconditional
    with pytest.raises(ValueError, match="unconditional branch"):
        task(unguided, key=0)
    assert np.isfinite(np.asarray(task(unguided, guidance=None, key=0).host().images)).all()
