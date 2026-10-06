"""Measure FID between two populations of images.

`FID().score(generated, reference)` scores two image sets against each other,
and the same `FID` as a registered metric pools the same features, statistics
and distance over the populations a validation pass consumes.
"""

import functools
import logging
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from dew.artifacts import ImageGrid
from dew.inputs import unit_range
from dew.objectives.base import Shown
from dew.registry import metrics

from .common import metric_device

_log = logging.getLogger(__name__)

# Two FID values are comparable only when the features behind them and the
# population counts agree, so every distance is logged with this line.
FEATURES = "pytorch-fid InceptionV3 pool3, bilinear 299x299 without antialiasing, [-1, 1]"


def _features(weights: str | None) -> str:
    """Names what a distance was measured with, for the line every one is logged
    with: two FID values are comparable only when this string matches."""
    return FEATURES if weights is None else f"{FEATURES}, weights {weights}"


def _extractor(weights: str | None):
    """Build the feature extractor and the variables to apply it with.

    The module is an ordinary Flax module, so its parameters are an ordinary
    variables tree in safetensors: the file named, or the converted copy of the
    published checkpoint that `dew.interop.inception_fid` keeps in the Hub
    cache. Its header says which width to build the extractor at, since a Flax
    parameter has to be the shape its module declares.
    """
    from dew.interop.inception_fid import cached_weights, channel_divisor
    from dew.interop.safetensors_io import load_params

    from .inception import InceptionV3

    path = cached_weights() if weights is None else weights
    # The file is read as memory maps. They land on the device here, once, so
    # the jitted extractor closes over arrays instead of compiling 90 MB of
    # weights into every kernel as constants.
    return (InceptionV3(channel_divisor=channel_divisor(path)),
            jax.tree.map(jnp.asarray, load_params(path)))


def _sqrtm(product):
    """Take `sqrtm` without its singularity warning. A singular product is the
    case the finiteness check in `frechet_distance` handles."""
    from scipy import linalg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        return linalg.sqrtm(product)


def frechet_distance(mu_a, sigma_a, mu_b, sigma_b, eps=1e-6) -> float:
    """Return the Frechet distance between two multivariate gaussians.

    It runs on the host through scipy, once per validation pass, because the
    matrix square root of the covariance product has no JAX equivalent. When
    the covariance product is singular, `eps` is added to both diagonals, as
    the reference implementations do.
    """

    mu_a, mu_b = np.atleast_1d(mu_a), np.atleast_1d(mu_b)
    sigma_a, sigma_b = np.atleast_2d(sigma_a), np.atleast_2d(sigma_b)

    # sqrtm's result is complex when rounding leaves the product with a
    # negative eigenvalue; the imaginary part is rounding noise.
    covmean = _sqrtm(sigma_a.dot(sigma_b))
    if not np.isfinite(covmean).all():
        # Singular product covariance. Nudge the diagonal as the reference
        # implementations do.
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = _sqrtm((sigma_a + offset).dot(sigma_b + offset))
    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean))


@dataclass
class GaussianStats:
    """Holds a population's count, float64 mean and centered sum of outer products."""

    count: int
    mean: NDArray[np.float64]
    m2: NDArray[np.float64]

    @classmethod
    def from_features(cls, features, *, population: str) -> "GaussianStats":
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError(f"fid {population}: expected finite [N, D] features")
        count, width = x.shape
        mean = x.mean(axis=0) if count else np.zeros(width, dtype=np.float64)
        centered = x - mean
        m2 = centered.T @ centered
        if not np.isfinite(mean).all() or not np.isfinite(m2).all():
            raise ValueError(f"fid {population}: non-finite feature statistics")
        return cls(count, mean, m2)

    def merge(self, other: "GaussianStats") -> "GaussianStats":
        if self.mean.shape != other.mean.shape:
            raise ValueError("fid: feature dimensions differ between batches")
        if other.count == 0:
            return self
        if self.count == 0:
            return other
        count = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2
        self.m2 += np.outer(delta, delta) * (self.count * other.count / count)
        self.mean += delta * (other.count / count)
        self.count = count
        if not np.isfinite(self.mean).all() or not np.isfinite(self.m2).all():
            raise ValueError("fid: non-finite pooled feature statistics")
        return self

    def covariance(self, *, population: str) -> NDArray[np.float64]:
        if self.count < 2:
            raise ValueError(f"fid {population}: at least two rows required, got {self.count}")
        return self.m2 / (self.count - 1)


@dataclass
class FIDStats:
    generated: GaussianStats
    real: GaussianStats


@functools.cache
def _get_activations(weights: str | None = None):
    """Return the jitted pool3 feature extractor, built on first use.

    Building it loads the ~90 MB of InceptionV3 weights, once per process and
    per weights, which every metric shares; constructing a metric opens nothing.
    """
    _log.info("loading InceptionV3 FID weights from %s (cached for reuse)",
              "the hub" if weights is None else weights)
    model, variables = _extractor(weights)

    @jax.jit
    def activations(images):
        # pytorch-fid's F.interpolate(bilinear, align_corners=False) does not
        # antialias; jax.image.resize does unless told not to.
        resized = jax.image.resize(images, (images.shape[0], 299, 299, 3), method='bilinear',
                                   antialias=False)
        features = model.apply(variables, resized)
        assert isinstance(features, jax.Array)
        return features.reshape(features.shape[0], -1)

    return activations


def _pooled_stats(batches: Iterable[ArrayLike], *, population: str,
                  weights: str | None = None) -> GaussianStats:
    """Pool pool3 statistics over batches of pixels in [-1, 1], as they come.

    The extractor is asked for inside the loop, so the weights load with the
    first batch and an image set that is refused costs no download.
    """
    pooled: GaussianStats | None = None
    for images in batches:
        contribution = GaussianStats.from_features(_get_activations(weights)(images),
                                                   population=population)
        pooled = contribution if pooled is None else pooled.merge(contribution)
    if pooled is None:
        raise ValueError(f"fid {population}: no images to score")
    return pooled


def _unit_range_batches(images: NDArray[np.uint8] | jax.Array | Iterable[ArrayLike], *,
                        population: str, batch_size: int) -> Iterator[jax.Array]:
    """Yield [-1, 1] batches from a uint8 [N, H, W, 3] array or an iterable of them.
    of at most `batch_size` rows."""
    blocks = [images] if isinstance(images, np.ndarray | jax.Array) else images
    for block in blocks:
        pixels = np.asarray(block)
        if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
            raise ValueError(f"fid {population}: expected uint8 [N, H, W, 3] images, got "
                             f"{pixels.dtype} {list(pixels.shape)}")
        for start in range(0, pixels.shape[0], batch_size):
            yield unit_range(pixels[start:start + batch_size])


def _pooled_distance(stats: FIDStats, weights: str | None = None) -> float:
    """Return the distance between two pooled populations, logged with the counts and
    the features it holds for."""
    generated, real = stats.generated, stats.real
    if generated.count < 2 or real.count < 2:
        raise ValueError(
            "fid generated and real populations require at least two rows each; "
            f"got generated={generated.count}, real={real.count}")
    distance = frechet_distance(generated.mean, generated.covariance(population="generated"),
                                real.mean, real.covariance(population="real"))
    _log.info("FID populations: generated=%d, real=%d; features=%s",
              generated.count, real.count, _features(weights))
    return distance


@metrics("fid")
@dataclass(frozen=True)
class FID:
    """Measures the Fréchet Inception Distance between two image sets (`score`) or over a validation pass.

    As a validation metric, each call reads the sampled grid and the batch's
    reference `field` and pools their feature statistics, and the pass ends
    with one distance. The features, statistics and distance are the ones
    `score` computes, and `weights` is described there.
    """

    field: str = "image"
    weights: str | None = None
    name = "fid"
    reads = ImageGrid
    shown = Shown(better="lower")

    def __call__(self, artifact: ImageGrid, batch) -> FIDStats:
        with metric_device():
            return FIDStats(_pooled_stats([artifact.images], population="generated",
                                          weights=self.weights),
                            _pooled_stats([unit_range(batch[self.field])], population="real",
                                          weights=self.weights))

    def merge(self, accumulated: FIDStats, contribution: FIDStats) -> FIDStats:
        accumulated.generated = accumulated.generated.merge(contribution.generated)
        accumulated.real = accumulated.real.merge(contribution.real)
        return accumulated

    def finalize(self, accumulated: FIDStats) -> float:
        return _pooled_distance(accumulated, self.weights)

    def score(self, generated: NDArray[np.uint8] | jax.Array | Iterable[ArrayLike],
              reference: NDArray[np.uint8] | jax.Array | Iterable[ArrayLike], *,
              batch_size: int = 64) -> float:
        """Measure FID between two sets of uint8 [N, H, W, 3] images.

        Each side is one array or an iterable of arrays, so a directory of
        samples can stream through in blocks of `batch_size` rows without
        being held in memory at once. Each side needs at least two images.
        The value is the distance between the two populations passed in, so it
        is FID-50k only with 50,000 images a side. A validation pass over
        50,000 images a side reports the same number.

        `weights` is a file holding the feature extractor's parameters, as
        `CLIPScore(modelname)` names a local CLIP. The file is the InceptionV3
        variables tree in safetensors, which `tools/convert_inception_weights.py`
        writes. When `weights` is unset, the published checkpoint is
        downloaded and converted. Two distances are comparable only when both
        used the same weights, so every distance logs which weights it used.
        With the published weights, the features and distance reproduce
        pytorch-fid 0.3.0's (bilinear resize without antialiasing) to within
        1e-5 relative.
        """
        # tests/test_fid.py holds the distance to pytorch-fid 0.3.0's within 1e-5 relative.
        if batch_size < 1:
            raise ValueError(f"fid: a batch holds at least one image, got batch_size={batch_size}")
        with metric_device():
            stats = FIDStats(
                _pooled_stats(_unit_range_batches(generated, population="generated",
                                                  batch_size=batch_size),
                              population="generated", weights=self.weights),
                _pooled_stats(_unit_range_batches(reference, population="real",
                                                  batch_size=batch_size),
                              population="real", weights=self.weights))
        return _pooled_distance(stats, self.weights)
