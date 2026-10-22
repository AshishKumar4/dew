"""A diffusion pipeline loads in pieces that fit: streamed, and without its text encoder.

`Pretrained.load(..., mesh=)` reads a pipeline's weights as `SourceLeaf`
recipes over the mapped files and places them one leaf at a time, as it
already does a decoder's. These tests hold the streamed load to the eager
one on the committed tiny pipelines of every family: the same tree, leaf for
leaf and bit for bit, and the same samples. Each family's upstream walk runs
on the streamed load too, against the source's own recorded outputs
(test_native_diffusion's streamed bundles for SD, XL and the safety
checker; the pipeline walks of test_flux_source, test_sd3_source,
test_z_image_source and test_wan_source).

`load_diffusion_source(text=False)` leaves the text encoder out, and a call
takes prompts its conditioner encoded alone (`prepare(conditions=...)`): the
same images as the prompted call, bit for bit.
"""

import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import streamed_export
from reference_error import assert_as_exact_as_the_reference

from dew.interop.pretrained import Pretrained, load_diffusion_source
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


@pytest.mark.parametrize("family", ["flux", "z_image", "wan"])
def test_a_streamed_pipeline_samples_what_the_eager_one_does(extracted, family):
    """The conditioner and the autoencoder hold the placed leaves, so a
    prompt runs end to end on them."""
    eager = Pretrained.load(str(extracted[family]), dtype="float32", attention_impl="xla")
    streamed = Pretrained.load(str(extracted[family]), dtype="float32", attention_impl="xla", mesh=MeshSpec())

    def sampled(source):
        return np.asarray(source.text_to_image()(["a red bird"], steps=2, key=3).host().images)

    np.testing.assert_array_equal(sampled(streamed), sampled(eager))


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


WIDER_FLUX = """
import json, shutil, sys
from pathlib import Path
import torch
from diffusers import FluxTransformer2DModel

source, out = Path(sys.argv[1]), Path(sys.argv[2])
shutil.copytree(source, out)
for stale in (out / "transformer").glob("diffusion_pytorch_model*"):
    stale.unlink()
stated = json.loads((source / "transformer" / "config.json").read_text())
config = {key: value for key, value in stated.items() if not key.startswith("_")}
config.update(num_attention_heads=64, num_layers=4, num_single_layers=8)
FluxTransformer2DModel(**config).to(torch.bfloat16).save_pretrained(out / "transformer")
"""


def test_a_sharded_pipeline_export_holds_a_few_shards_on_the_host(extracted, tmp_path):
    """The tiny Flux pipeline with a 300 MB bfloat16 transformer, written by
    Diffusers in one process, then loaded onto a mesh and saved in 25 MB
    shards by a fresh one (tools/benchmark_streaming_export.py). Each
    component's tensors are built as their shard is written, so the memory
    the process allocates over the save rises by a few shards (measured
    0.047-0.048 GB, 0.029-0.036 GB in 10 MB shards); building the whole
    export first rose 0.30 GB. Its resident memory also holds the pages of the
    source files it maps, which the host's page cache and load move, so the
    bound reads its anonymous memory."""
    measured = streamed_export(WIDER_FLUX, extracted["flux"], tmp_path, "25MB")
    assert measured["shards"] > 8
    assert measured["anon_rise_gb"] < 4 * 25e6 / 2**30, measured
