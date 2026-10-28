"""The Frechet distance, FID and the InceptionV3 weights it reads.

The Frechet distance itself is checked against closed forms that need no
weights and against pytorch-fid's on its own statistics. The end-to-end
InceptionV3 path downloads the FID checkpoint and is network-marked: it is
held to pytorch-fid's features and distance on the published weights
(tests/fixtures/inception/pytorch_fid_reference.npz, written by
tools/pytorch_fid_reference.py). The offline case reads the drawn
sixteenth-width extractor committed under tests/fixtures/inception. The
weights loader reads arrays alone, by a pinned digest.
"""

import json
import warnings
from pathlib import Path

import jax
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.artifacts import ImageGrid
from dew.data.dataset import DataPartition
from dew.eval import FID

FLOWERS = "datasets/nkirschi/oxford-flowers@a37b1891609c0376fa81eced756e7863e1bd873b"
"""Oxford Flowers 102 as published Parquet on the Hub, at a pinned commit."""

INCEPTION_TINY = (Path(__file__).resolve().parent / "fixtures" / "inception" / "tiny"
                  / "inception_v3_fid.safetensors")


PYTORCH_FID = Path(__file__).resolve().parent / "fixtures" / "inception" / "pytorch_fid_reference.npz"


def pytorch_fid_reference():
    """pytorch-fid 0.3.0's features and FID on its published weights, written by
    tools/pytorch_fid_reference.py: set "a" at 64x64 (upsampled to 299) and
    set "b" scored at 400x400 (downsampled), with the arrays both sides read."""
    reference = np.load(PYTORCH_FID)
    upscale = int(reference["upscale_b"])
    scored_b = np.repeat(np.repeat(reference["images_b"], upscale, axis=1), upscale, axis=2)
    return reference, reference["images_a"], scored_b


@pytest.mark.network
def test_the_extractor_gives_pytorch_fids_features_and_distance_on_the_published_weights():
    """The converted jax-fid checkpoint against pytorch-fid on the weights both
    come from, through each side's own input path (uint8 to [-1, 1], bilinear
    to 299x299 with no antialiasing).

    Features: fp32 convolutions summed in a different order by torch and XLA
    differ at the last bits; observed 5.7e-06 at most on features up to 3.7,
    so 1e-4 absolute is the float32 bound, and a wrong pool, norm or resize
    moves features by 1e-2 and more (an antialiased downsample moved set "b"
    by up to 0.55 and FID from 195.789 to 182.878). Distance: observed
    195.788897 against pytorch-fid's 195.788855, 2.2e-07 relative; eight
    images a set leave rank-7 covariances, whose square root amplifies the
    feature bits, so the bound is 1e-5 relative."""
    from dew.eval.fid import _get_activations
    from dew.inputs import unit_range

    reference, images_a, images_b = pytorch_fid_reference()
    extract = _get_activations(None)
    np.testing.assert_allclose(np.asarray(extract(unit_range(images_a))), reference["features_a"],
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(extract(unit_range(images_b))), reference["features_b"],
                               rtol=0, atol=1e-4)
    assert FID().score(images_a, images_b) == pytest.approx(float(reference["fid"]), rel=1e-5)



def test_the_distance_of_pytorch_fids_own_features_is_its_fid_with_no_warning():
    """Eight images a set leave rank-7 covariances of 2048 features, a singular
    product whose square root scipy's `sqrtm` warns about; the distance takes
    its trace from the product's eigenvalues instead, with no warning raised,
    to pytorch-fid's own FID within the 1e-5 relative the extractor test holds."""
    from dew.eval import frechet_distance

    reference = np.load(PYTORCH_FID)
    stats = [(features.mean(0), np.cov(features, rowvar=False))
             for features in (reference["features_a"].astype(np.float64),
                              reference["features_b"].astype(np.float64))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        distance = frechet_distance(*stats[0], *stats[1])
    assert distance == pytest.approx(float(reference["fid"]), rel=1e-5)

def fid_sets():
    """Sixteen uint8 images of 32x32, and the same pixels brightened by 40."""
    images = np.random.default_rng(93).integers(0, 256, (16, 32, 32, 3), dtype=np.uint8)
    return images, np.clip(images.astype(np.int32) + 40, 0, 255).astype(np.uint8)


def test_fid_takes_its_extractor_from_a_file_and_orders_populations_offline():
    """`weights` names the extractor's variables as a safetensors file, the way
    `CLIPScore(modelname)` names a local CLIP, so a distance is computable
    with no download.

    The committed fixture is this module's own InceptionV3 at a sixteenth of
    every channel width, its parameters drawn rather than trained, so the
    values are its own and not the published checkpoint's. For these images,
    a population against itself scores zero to matrix-square-root rounding;
    brightening the pixels by 40 counts scores above it, and a flat gray
    field further out. The registered metric reads the same file and lands
    on the same number. `source.json` states the feature width, and the
    extractor agrees.
    """
    from dew.eval.fid import _get_activations
    from dew.inputs import unit_range

    images, brighter = fid_sets()
    weights = str(INCEPTION_TINY)
    record = json.loads((INCEPTION_TINY.parent / "source.json").read_text())

    assert abs(FID(weights=weights).score(images, images)) < 1e-6
    shifted = FID(weights=weights).score(brighter, images)
    assert 0 < shifted < FID(weights=weights).score(np.full_like(images, 128), images)

    metric = FID(weights=weights)
    pooled = metric.finalize(metric(ImageGrid(unit_range(brighter)), {"image": images}))
    assert pooled == pytest.approx(shifted, rel=1e-6)
    assert _get_activations(weights)(unit_range(images[:1])).shape[1] == record["pool3_features"]


PYTORCH_FID_TINY = Path(__file__).resolve().parent / "fixtures" / "pytorch_fid_tiny" / "reference.npz"


# pytorch-fid's BasicConv2d leaves by the names jax-fid's pickle gives them.
JAX_FID_LEAVES = {"conv.weight": ("conv", "kernel"), "bn.weight": ("bn", "scale"), "bn.bias": ("bn", "bias"),
                  "bn.running_mean": ("bn", "mean"), "bn.running_var": ("bn", "var")}


def jax_fid_layout(state: dict) -> dict:
    """A pytorch-fid state dict as jax-fid's pickle holds it: nested by module,
    kernels HWIO. tools/pytorch_fid_tiny_reference.py's names; checked against the
    two published files (pt_inception-2015-12-05 and jax-fid's pickle), which
    it maps tensor for tensor."""
    tree: dict = {}
    for name, value in state.items():
        if name.endswith("num_batches_tracked") or name.startswith("fc."):
            continue
        *modules, layer, leaf = name.split(".")
        node = tree
        for step in (*modules, JAX_FID_LEAVES[f"{layer}.{leaf}"][0]):
            node = node.setdefault(step, {})
        node[JAX_FID_LEAVES[f"{layer}.{leaf}"][1]] = value.transpose(2, 3, 1, 0) if value.ndim == 4 else value
    return tree


@pytest.mark.parametrize("size", [64, 320])
def test_the_converted_extractor_is_as_exact_as_pytorch_fid(tmp_path, size):
    """pytorch-fid's own InceptionV3 at the tiny extractor's width
    (tools/pytorch_fid_tiny_reference.py), its drawn weights written in jax-fid's
    layout and read by `dew.interop.inception_fid.convert`, the way the
    published checkpoint reaches the extractor: pool3 features of images it
    resizes up and down, by the float64 rule. pytorch-fid scales [0, 1] after
    resizing; the extractor takes [-1, 1] and resizes."""
    import pickle

    from dew.eval.fid import _get_activations
    from dew.interop.inception_fid import convert, save

    with np.load(PYTORCH_FID_TINY) as loaded:
        arrays = dict(loaded)
    divisor = json.loads(arrays["meta"].tobytes())["divisor"]
    state = {name.removeprefix("state/"): value for name, value in arrays.items()
             if name.startswith("state/")}
    source = tmp_path / "inception_v3_fid.pickle"
    source.write_bytes(pickle.dumps(jax_fid_layout(state), protocol=4))
    weights = tmp_path / "inception_v3_fid.safetensors"
    save(convert(source), weights, divisor)
    images = arrays[f"{size}/pixels"].transpose(0, 2, 3, 1).astype(np.float32) / 255
    features = _get_activations(str(weights))(2 * images - 1)
    assert_as_exact_as_the_reference(np.asarray(features), arrays[f"{size}/fp32.features"],
                                     arrays[f"{size}/fp64.features"], f"{size}x{size} pool3")

    legacy_pixels = np.random.default_rng(0).uniform(-1, 1, (4, 299, 299, 3)).astype(np.float32)
    legacy_features = np.asarray(_get_activations(str(INCEPTION_TINY))(legacy_pixels))
    legacy_reference = np.load(INCEPTION_TINY.parent / "reference_features.npy")
    np.testing.assert_allclose(legacy_features, legacy_reference, rtol=1e-5, atol=1e-7)


def test_fid_extraction_is_independent_of_small_batch_boundaries():
    """The mean of repeated copies of one image is that image's feature.

    Compare these public FID statistics across the batch-eight boundary.
    Unlike a relative covariance check, this tests the extracted features
    themselves: centering nearly constant features magnifies their rounding.
    """
    from dew.inputs import unit_range

    images, brighter = fid_sets()
    metric = FID(weights=str(INCEPTION_TINY))

    def extract(rows):
        return metric(ImageGrid(unit_range(np.repeat(brighter[:1], rows, axis=0))),
                      {"image": np.repeat(images[:1], rows, axis=0)})

    reference = extract(8)
    for rows in (1, 3, 7, 9):
        actual = extract(rows)
        for part, whole in ((actual.generated, reference.generated), (actual.real, reference.real)):
            assert part.count == rows
            np.testing.assert_allclose(part.mean, whole.mean, rtol=1e-5, atol=1e-7)


@pytest.mark.network
def test_fid_pools_a_streamed_set_into_the_distance_of_the_whole_set():
    """A set can arrive as an iterable of arrays, and `batch_size` splits
    whatever arrives, so a directory of samples never has to be held at once.

    Pooling makes the split invisible. Dropping a block or losing a merge
    moves the number; six blocks of at most three rows came within 3.8e-09
    relative of the whole set on CPU, which is what the extractor's
    batch-shape rounding costs. The bound is 1e-5 to leave a backend that
    rounds differently the same headroom."""
    images, brighter = fid_sets()

    whole = FID().score(brighter, images)
    streamed = FID().score([brighter[:7], brighter[7:]], images, batch_size=3)

    assert streamed == pytest.approx(whole, rel=1e-5)


def test_the_weights_loader_reads_arrays_and_refuses_the_rest(tmp_path):
    """T28: the FID weights are a pickle written under numpy 1, whose array
    reconstructor now lives behind a shim that warns on every attribute read,
    so `pickle.load` could not read them under numpy 2 with the suite's
    filters. The loader builds the arrays its allowed numpy names stand for from
    numpy's public API and refuses every other name, so a downloaded pickle
    cannot run code. Protocol 4 pickles an array through numpy's
    `_reconstruct`, protocol 5 through `_frombuffer`, whose 'K' layout carries
    the axis order of a transposed array."""
    import pickle

    from dew.interop.inception_fid import load_arrays

    kernel = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    tree = {"conv": {"kernel": kernel, "bias": np.zeros(3, np.float32),
                     "fortran": np.asfortranarray(kernel), "permuted": kernel.transpose(2, 0, 1)}}
    for protocol in (4, 5):
        path = tmp_path / f"weights-{protocol}.pickle"
        path.write_bytes(pickle.dumps(tree, protocol=protocol))
        loaded = load_arrays(path)
        assert set(loaded) == {"conv"} and set(loaded["conv"]) == set(tree["conv"])
        for name, value in tree["conv"].items():
            np.testing.assert_array_equal(loaded["conv"][name], value)

    hostile = tmp_path / "hostile.pickle"
    hostile.write_bytes(pickle.dumps(print))
    with pytest.raises(pickle.UnpicklingError):
        load_arrays(hostile)


@pytest.mark.network
def test_fid_is_far_smaller_between_halves_of_real_data_than_against_noise():
    """The calibration the metric exists for: two disjoint halves of the same
    photographs score close, unrelated noise scores far.

    Measured on Oxford Flowers at 64px with the released weights, 64 images a
    side (TFDS's copy, bilinear): 123 between halves against 505 for noise. The absolute number is
    finite-sample bias, not the distance between the halves, and it falls with
    the sample count on the same photographs: 176 at 16 a side, 89 at 128, 63
    at 256, 41 at 512, while noise stays near 500 throughout. A comparison
    therefore needs the consumed population counts alongside its FID value.
    """
    from dew.data import Loading, presets
    from dew.data.compile import compile
    from dew.data.sources import source
    from dew.inputs import unit_range

    # Oxford Flowers' published Parquet copy, read and resized by dew.data's images preset: no
    # dataset preparation, which TFDS runs through TensorFlow. The halves are the run's own, so it
    # holds out nothing.
    photographs = source(f"hf://{FLOWERS}").shuffle(0, window=64).repeat().pipe(presets.images, 64, batch=128)
    data = compile(photographs, read="streamed", loading=Loading(workers=0, threads=1, read_buffer=2))
    stream = data.train(DataPartition())
    try:
        photos = np.asarray(next(stream)["image"])
    finally:
        stream.close()
    metric = FID()
    first, second = photos[:64], photos[64:]
    noise = jax.random.normal(jax.random.PRNGKey(1), (64, 64, 64, 3)).clip(-1.0, 1.0)

    halves = metric.finalize(metric(ImageGrid(unit_range(first)), {"image": second}))
    unrelated = metric.finalize(metric(ImageGrid(noise), {"image": second}))
    itself = metric.finalize(metric(ImageGrid(unit_range(second)), {"image": second}))

    assert abs(itself) < 1.0, f"the same images do not score zero: {itself:.3f}"
    assert halves < 0.5 * unrelated, f"halves {halves:.1f} against noise {unrelated:.1f}"
