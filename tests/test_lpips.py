"""LPIPS against REPA-E's own perceptual loss (`tools/lpips_reference.py`):
its `LPIPS` on VGG16, run as published, on drawn weights and on the
published ones."""

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.artifacts import ImageGrid
from dew.eval import LPIPS
from dew.eval.lpips import LPIPSNetwork, variables_from_torch

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "lpips"
CONVOLUTIONS = (0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28)
WIDTHS = (64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512)
CHANNELS = (64, 128, 256, 512, 512)


def drawn_weights(seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The tool's `drawn_weights`: torchvision's order from
    `np.random.default_rng(seed)`'s float32 uniforms, kernels He uniform,
    biases in [-0.01, 0.01), heads in [0, 0.2): integer bits and float32
    arithmetic, the same bits on every machine."""
    rng = np.random.default_rng(seed)

    def uniform(shape, bound: float) -> np.ndarray:
        return (rng.random(shape, dtype=np.float32) * np.float32(2) - np.float32(1)) * np.float32(bound)

    vgg, inputs = {}, 3
    for index, width in zip(CONVOLUTIONS, WIDTHS, strict=True):
        vgg[f"features.{index}.weight"] = uniform((width, inputs, 3, 3), np.sqrt(6 / (9 * inputs)))
        vgg[f"features.{index}.bias"] = uniform(width, 0.01)
        inputs = width
    linear = {f"lin{stage}.model.1.weight": rng.random((1, width, 1, 1), dtype=np.float32) * np.float32(0.2)
              for stage, width in enumerate(CHANNELS)}
    return vgg, linear


def test_the_distance_and_its_gradient_are_repa_es():
    """On weights both sides draw (bitwise the fixture's, by SHA-256),
    converted by the same `variables_from_torch` the published weights go
    through: each pair's distance is within 1e-6 of REPA-E's float64 run,
    and the gradient of their mean in the first image, what the perceptual
    loss trains an autoencoder by, is Dew's float64 run within the float64
    rounding of the computation of REPA-E's (24,576 entries). The gradient
    passes VGG's ReLUs and pools, and one ReLU input (conv_6's) sits 0.64 of
    Dew's own float32 error from its kink, so Dew's float32 takes the other
    branch there, a step the float32 rule does not model; in float64, whose
    margin there is 2.4 float32 spacings, both runs take one.

    Two distances are too few for the rule's RMS, so they take a bound:
    Dew's are 2.1e-8 and 5.8e-8 relative from float64, the reference's own
    float32 7.1e-8 and 1.2e-7, and a wrong shift or scale digit or a dropped
    stage moves a distance by 1e-3 or more."""
    reference = dict(np.load(FIXTURES / "drawn.npz"))
    vgg, linear = drawn_weights()
    for key, value in {**vgg, **linear}.items():
        assert hashlib.sha256(value.tobytes()).hexdigest() == str(reference[f"sha256/{key}"]), key
    network, variables = LPIPSNetwork(), variables_from_torch(vgg, linear)
    images, references = (jnp.asarray(reference[key], jnp.float32) for key in ("images", "references"))
    distance = network.apply(variables, images, references)
    np.testing.assert_allclose(np.asarray(distance), reference["distance_f64"], rtol=1e-6)

    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
        first, second = (jnp.asarray(reference[key], jnp.float64) for key in ("images", "references"))

        def mean(images):
            return jnp.mean(network.apply(wide, images, second))

        gradient = np.asarray(jax.grad(mean)(first))
        roundings = chain_roundings(jax.make_jaxpr(jax.grad(mean))(first))
    assert_computes_the_oracle(gradient, reference["gradient_f64"], "the gradient", roundings=roundings)


@pytest.mark.network
def test_the_published_network_measures_as_repa_es():
    """The published weights, the Hub copies downloaded and checked on first
    use: four pairs' distances within 1e-6 of REPA-E's float64 run on the
    original files, and the `LPIPS` metric over an image grid is their mean, the
    batch holding the references as pixels in [0, 255] as a loader does.
    Dew's distances are at most 2.5e-7 relative from float64, the
    reference's own float32 at most 1.9e-7."""
    reference = dict(np.load(FIXTURES / "published.npz"))
    network, variables = LPIPSNetwork.published()
    images, references = (jnp.asarray(reference[key], jnp.float32) for key in ("images", "references"))
    distance = network.apply(variables, images, references)
    np.testing.assert_allclose(np.asarray(distance), reference["distance_f64"], rtol=1e-6)
    total, count = LPIPS()(ImageGrid(images=images), {"image": reference["references"] * 127.5 + 127.5})
    np.testing.assert_allclose(total / count, reference["distance_f64"].mean(), rtol=1e-6)
