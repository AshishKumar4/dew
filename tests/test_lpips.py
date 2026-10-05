"""LPIPS against REPA-E's own perceptual loss (`tools/lpips_reference.py`):
its `LPIPS` on VGG16, run as published, on drawn weights and on the
published ones."""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.artifacts import ImageGrid
from dew.eval import LPIPS
from dew.eval.lpips import LPIPSNetwork, variables_from_torch

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "lpips"
CONVOLUTIONS = (0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28)
WIDTHS = (64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512)
CHANNELS = (64, 128, 256, 512, 512)


def drawn_weights(seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The tool's `drawn_weights`: torchvision's order from
    `np.random.default_rng(seed)`, kernels He normal, biases normal at 0.01,
    heads the absolute value of a normal at 0.1, in float32."""
    rng = np.random.default_rng(seed)
    vgg, inputs = {}, 3
    for index, width in zip(CONVOLUTIONS, WIDTHS, strict=True):
        vgg[f"features.{index}.weight"] = (rng.standard_normal((width, inputs, 3, 3), dtype=np.float32)
                                           * np.float32(np.sqrt(2 / (9 * inputs))))
        vgg[f"features.{index}.bias"] = rng.standard_normal(width, dtype=np.float32) * np.float32(0.01)
        inputs = width
    linear = {f"lin{stage}.model.1.weight": np.abs(rng.standard_normal((1, width, 1, 1), dtype=np.float32))
              * np.float32(0.1) for stage, width in enumerate(CHANNELS)}
    return vgg, linear


def test_the_distance_and_its_gradient_are_repa_es():
    """On weights both sides draw (their per-tensor sums are the fixture's),
    converted by the same `variables_from_torch` the published weights go
    through: each pair's distance is within 1e-6 of REPA-E's float64 run,
    and the gradient of their mean in the first image, what the perceptual
    loss trains an autoencoder by, is held to it by the float64 rule over
    24,576 entries.

    Two distances are too few for the rule's RMS, so they take a bound:
    Dew's are 5.7e-8 and 4.4e-8 relative from float64, the reference's own
    float32 1.3e-7 and 3.0e-8, and a wrong shift or scale digit or a dropped
    stage moves a distance by 1e-3 or more."""
    reference = dict(np.load(FIXTURES / "drawn.npz"))
    vgg, linear = drawn_weights()
    for key, value in {**vgg, **linear}.items():
        # The fixture's sum is numpy's, whose order follows the CPU's SIMD
        # width, so it carries up to n·u·Σ|x| of float64 rounding (7.5e-9 on
        # one conv's 1.2M entries, one CI runner against another); ours is
        # math.fsum's, rounded once. A different draw moves a sum by ~30.
        entries = value.astype(np.float64).ravel()
        bound = entries.size * np.finfo(np.float64).eps * np.abs(entries).sum()
        assert abs(math.fsum(entries) - reference[f"sum/{key}"]) <= bound, key
    network, variables = LPIPSNetwork(), variables_from_torch(vgg, linear)
    images, references = (jnp.asarray(reference[key], jnp.float32) for key in ("images", "references"))

    def mean(first):
        return jnp.mean(network.apply(variables, first, references))

    distance = network.apply(variables, images, references)
    gradient = jax.grad(mean)(images)
    np.testing.assert_allclose(np.asarray(distance), reference["distance_f64"], rtol=1e-6)
    assert_as_exact_as_the_reference(gradient, reference["gradient"], reference["gradient_f64"],
                                     "the gradient")


@pytest.mark.network
def test_the_published_network_measures_as_repa_es():
    """The published weights, downloaded, checked and converted on first use:
    four pairs' distances within 1e-6 of REPA-E's float64 run on the same
    files, and the `LPIPS` metric over an image grid is their mean, the
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
