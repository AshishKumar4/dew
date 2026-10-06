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
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.artifacts import ImageGrid
from dew.eval import FID
from dew.eval.fid import frechet_distance

INCEPTION_TINY = (Path(__file__).resolve().parent / "fixtures" / "inception" / "tiny"
                  / "inception_v3_fid.safetensors")


def test_frechet_distance_of_a_distribution_with_itself_is_zero(rng):
    features = np.asarray(jax.random.normal(rng, (256, 16)))
    mu, sigma = features.mean(axis=0), np.cov(features, rowvar=False)
    assert frechet_distance(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-6)


def test_frechet_distance_of_shifted_gaussians_is_the_squared_mean_gap():
    sigma = np.eye(8)
    mu_a = np.zeros(8)
    mu_b = np.full(8, 0.5)
    assert frechet_distance(mu_a, sigma, mu_b, sigma) == pytest.approx(8 * 0.25, abs=1e-6)


def test_frechet_distance_of_a_scaled_covariance_matches_the_closed_form():
    """Centred gaussians with covariances I and cI are 8 (1 - sqrt(c))^2 apart.
    Dropping the (sigma_a sigma_b)^1/2 term leaves tr(sigma_a) + tr(sigma_b),
    which grows with c as well."""
    mu = np.zeros(8)
    identity = np.eye(8)
    assert frechet_distance(mu, identity, mu, identity * 1.5) == pytest.approx(
        8 * (1 - np.sqrt(1.5)) ** 2, abs=1e-6)
    assert frechet_distance(mu, identity, mu, identity * 4.0) == pytest.approx(
        8 * (1 - 2.0) ** 2, abs=1e-6)


PYTORCH_FID = Path(__file__).resolve().parent / "fixtures" / "inception" / "pytorch_fid_reference.npz"


def pytorch_fid_reference():
    """pytorch-fid 0.3.0's features and FID on its published weights, written by
    tools/pytorch_fid_reference.py: set "a" at 64x64 (upsampled to 299) and
    set "b" scored at 400x400 (downsampled), with the arrays both sides read."""
    reference = np.load(PYTORCH_FID)
    upscale = int(reference["upscale_b"])
    scored_b = np.repeat(np.repeat(reference["images_b"], upscale, axis=1), upscale, axis=2)
    return reference, reference["images_a"], scored_b


def test_frechet_distance_equals_pytorch_fids_on_the_same_statistics():
    """The distance alone, on the statistics of pytorch-fid's own features:
    both take scipy's `sqrtm` of the covariance product and nudge the
    diagonal when it is singular.

    Eight images a set leave rank-7 covariances, so 2041 of the product's
    2048 eigenvalues are rounding, each at most eps * |Sa| * |Sb| (spectral
    norms). sqrtm turns each into up to the square root of that, and FID
    subtracts twice the trace, so two correct runs (another scipy, another
    BLAS) can differ by 2 * (n - r) * sqrt(eps * |Sa| * |Sb|): 7.8e-4 here.
    CI's scipy landed 4.2e-6 from pytorch-fid's number, a Colab CPU 1.8e-8.
    Covariances without Bessel's correction move the distance by 7.6, one
    trace term instead of two by 9.3, and dropping the mean term by 135."""
    reference, _, _ = pytorch_fid_reference()
    (mu_a, sigma_a), (mu_b, sigma_b) = [
        (features.astype(np.float64).mean(axis=0), np.cov(features.astype(np.float64), rowvar=False))
        for features in (reference["features_a"], reference["features_b"])]
    width = sigma_a.shape[0]
    rank = min(len(reference["features_a"]), len(reference["features_b"])) - 1
    rounding = np.finfo(np.float64).eps * np.linalg.norm(sigma_a, 2) * np.linalg.norm(sigma_b, 2)
    bound = 2 * (width - rank) * np.sqrt(rounding)
    assert frechet_distance(mu_a, sigma_a, mu_b, sigma_b) == pytest.approx(float(reference["fid"]), abs=bound)


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


@pytest.mark.network
def test_fid_metric_scores_real_images_better_than_noise(rng):
    metric = FID()
    assert metric.name == 'fid' and metric.reads is ImageGrid

    key_real, key_noise = jax.random.split(rng)
    real = jax.random.randint(key_real, (8, 64, 64, 3), 0, 256, dtype=jnp.int32).astype(jnp.uint8)
    batch = {'image': real}

    # Generated samples live in [-1, 1]; the same images should score far
    # closer to the batch than unrelated noise does
    matching = metric.finalize(metric(ImageGrid((jnp.asarray(real, jnp.float32) - 127.5) / 127.5), batch))
    unrelated = metric.finalize(metric(ImageGrid(jax.random.normal(key_noise, (8, 64, 64, 3))), batch))
    assert np.isfinite(matching) and np.isfinite(unrelated)
    assert matching < unrelated


def fid_sets():
    """Sixteen uint8 images of 32x32, and the same pixels brightened by 40."""
    images = np.random.default_rng(93).integers(0, 256, (16, 32, 32, 3), dtype=np.uint8)
    return images, np.clip(images.astype(np.int32) + 40, 0, 255).astype(np.uint8)


def test_fid_refuses_an_image_set_it_cannot_score():
    """Both sides of `FID.score` are uint8 [N, H, W, 3]. The refusal comes out of the
    batch parser before the extractor is asked for, so a call that cannot be
    scored never pays for the 90 MB of Inception weights, which is why this
    test needs no network."""
    images = np.zeros((4, 8, 8, 3), np.uint8)
    with pytest.raises(ValueError, match="generated: expected uint8"):
        FID().score(images.astype(np.float32), images)
    with pytest.raises(ValueError, match="real: expected uint8"):
        FID().score(images, images[0])
    with pytest.raises(ValueError, match="generated: no images"):
        FID().score([], images)
    with pytest.raises(ValueError, match="at least one image"):
        FID().score(images, images, batch_size=0)


@pytest.mark.network
def test_fid_of_a_set_against_itself_is_zero_and_a_shifted_set_scores_above_it():
    """`FID.score` scores two image sets with no objective, no dataset and no batch.

    One set twice has identical pooled statistics, so the distance is zero up
    to the rounding in the matrix square root. Observed -2.0e-05 with the
    released weights on 16 images of 32x32 on CPU, against a bound of 1e-3.
    Adding 40 counts to every pixel moves the population, observed 10.6."""
    images, brighter = fid_sets()

    assert abs(FID().score(images, images)) < 1e-3
    assert FID().score(brighter, images) > 0


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


def test_the_converted_extractor_reproduces_the_features_it_gave_as_a_pickle():
    """The extractor used to get its weights by unpickling a nested dict and
    handing every convolution and norm an initializer that returned the stored
    array. It is an ordinary Flax module now, applied to an ordinary variables
    tree read from safetensors, and `reference_features.npy` is what the old
    path gave for this fixture: the same arrays reach the same operations.

    The two are equal to the bit when the reductions behind them are split the
    same way, which is how this was checked when the extractor changed. XLA's
    CPU backend splits by its thread pool, and the thread pool is the machine's
    rather than the code's: the same weights on the same pixels move by 6e-08
    between one host and another, so what is asserted here is a tolerance a
    changed network could not sit inside.
    """
    from dew.eval.fid import _get_activations

    images = np.random.default_rng(0).uniform(-1, 1, (4, 299, 299, 3)).astype(np.float32)
    features = np.asarray(_get_activations(str(INCEPTION_TINY))(images))
    reference = np.load(INCEPTION_TINY.parent / "reference_features.npy")
    np.testing.assert_allclose(features, reference, rtol=1e-5, atol=1e-7)


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
def test_the_fid_metric_and_the_function_report_the_same_distance():
    """The registered metric is that same path with a trainer's artifact and
    batch in front of it, so a validation pass lands on the number `FID.score` gives
    for the same pixels.

    One pass over one batch splits nothing, so the features and the statistics
    are the same arrays: the two distances were equal to the last bit on CPU,
    and the bound is 1e-6 relative for a backend that reassociates."""
    from dew.inputs import unit_range

    images, brighter = fid_sets()
    metric = FID()

    pooled = metric.finalize(metric(ImageGrid(unit_range(brighter)), {"image": images}))

    assert pooled == pytest.approx(FID().score(brighter, images), rel=1e-6)


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
    filters. The loader resolves the numpy names at their current home, which
    also means it can allow exactly those names and refuse the rest, so a
    downloaded pickle cannot run code."""
    import pickle

    from dew.interop.inception_fid import load_arrays

    tree = {"conv": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3),
                     "bias": np.zeros(3, np.float32)}}
    path = tmp_path / "weights.pickle"
    path.write_bytes(pickle.dumps(tree))
    loaded = load_arrays(path)
    assert set(loaded) == {"conv"} and set(loaded["conv"]) == {"kernel", "bias"}
    assert np.array_equal(loaded["conv"]["kernel"], tree["conv"]["kernel"])

    hostile = tmp_path / "hostile.pickle"
    hostile.write_bytes(pickle.dumps(print))
    with pytest.raises(pickle.UnpicklingError, match=r"builtins.print"):
        load_arrays(hostile)


def test_a_wrong_digest_is_refused_before_anything_reads_the_file(tmp_path):
    """The pin is a check over local bytes, so it is tested over local bytes:
    a file whose hash is not the pinned one is refused, and the pinned file
    passes through untouched."""
    import hashlib

    from dew.interop.inception_fid import _check_digest

    path = tmp_path / "weights.pickle"
    path.write_bytes(b"not the weights")
    digest = hashlib.sha256(b"not the weights").hexdigest()

    assert _check_digest(str(path), "repo", "file", "revision", digest) == str(path)
    with pytest.raises(ValueError, match="hashes to"):
        _check_digest(str(path), "repo", "file", "revision", "0" * 64)


@pytest.mark.network
def test_the_weights_are_pinned_by_digest():
    """The weights came from a consumer file-sharing link with no checksum,
    which a released package unpickled, so whoever held the link chose what
    ran. They come from a Hub revision now and the digest is checked as well,
    so a file that is not the one this code was written against is refused
    before anything reads it."""
    from dew.interop import inception_fid

    assert len(inception_fid.FID_WEIGHTS_DIGEST) == 64
    assert len(inception_fid.FID_WEIGHTS_REVISION) == 40
    with pytest.raises(ValueError, match="hashes to"):
        inception_fid.fetch(inception_fid.FID_WEIGHTS_REPO, inception_fid.FID_WEIGHTS_FILE,
                            inception_fid.FID_WEIGHTS_REVISION, "0" * 64)


@pytest.mark.network
def test_fid_is_far_smaller_between_halves_of_real_data_than_against_noise():
    """The calibration the metric exists for: two disjoint halves of the same
    photographs score close, unrelated noise scores far.

    Measured on Oxford Flowers at 64px with the released weights, 64 images a
    side: 123 between halves against 505 for noise. The absolute number is
    finite-sample bias, not the distance between the halves, and it falls with
    the sample count on the same photographs: 176 at 16 a side, 89 at 128, 63
    at 256, 41 at 512, while noise stays near 500 throughout. A comparison
    therefore needs the consumed population counts alongside its FID value.
    """
    tfds = pytest.importorskip("tensorflow_datasets", reason="needs the tfds extra")
    from dew.inputs import unit_range

    source = tfds.data_source("oxford_flowers102", split="all", try_gcs=False)
    photos = np.stack([
        np.asarray(jax.image.resize(jnp.asarray(source[index]["image"], jnp.float32),
                                    (64, 64, 3), method="bilinear"))
        for index in range(128)]).astype(np.uint8)
    metric = FID()
    first, second = photos[:64], photos[64:]
    noise = jax.random.normal(jax.random.PRNGKey(1), (64, 64, 64, 3)).clip(-1.0, 1.0)

    halves = metric.finalize(metric(ImageGrid(unit_range(first)), {"image": second}))
    unrelated = metric.finalize(metric(ImageGrid(noise), {"image": second}))
    itself = metric.finalize(metric(ImageGrid(unit_range(second)), {"image": second}))

    assert abs(itself) < 1.0, f"the same images do not score zero: {itself:.3f}"
    assert halves < 0.5 * unrelated, f"halves {halves:.1f} against noise {unrelated:.1f}"
