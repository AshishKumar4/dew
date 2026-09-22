"""Measure FID between two populations of images.

`fid(generated, reference)` scores two image sets against each other, and the
registered `fid` metric pools the same features, statistics and distance over
the populations a validation pass consumes.
"""

import functools
import logging
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from dew.artifacts import ImageGrid
from dew.inputs import unit_range
from dew.registry import metrics

from .common import metric_device

_log = logging.getLogger(__name__)

# Two FID values are comparable only when the features behind them and the
# population counts agree, so every distance is logged with this line.
FEATURES = "FID InceptionV3 pool3, bilinear 299x299, [-1, 1]"


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
    from dew.interop.inception_fid import cached_weights, channel_divisor, load

    from .inception import InceptionV3

    path = cached_weights() if weights is None else weights
    # The file is read as memory maps. They land on the device here, once, so
    # the jitted extractor closes over arrays instead of compiling 90 MB of
    # weights into every kernel as constants.
    return (InceptionV3(channel_divisor=channel_divisor(path)),
            jax.tree.map(jnp.asarray, load(path)))


@functools.cache
def _get_inception(weights: str | None = None):
    """Load the pool3 feature extractor and its variables, once per process
    and per weights. The FID InceptionV3 is about 90 MB of weights,
    and every metric built from this module shares the copy."""
    _log.info("loading InceptionV3 FID weights from %s (cached for reuse)",
              "the hub" if weights is None else weights)
    return _extractor(weights)


def _sqrtm(product):
    """Take `sqrtm` without its singularity warning. A singular product is the
    case the finiteness check in `frechet_distance` handles."""
    from scipy import linalg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        return linalg.sqrtm(product)


def frechet_distance(mu_a, sigma_a, mu_b, sigma_b, eps=1e-6) -> float:
    """Return the Frechet distance between two multivariate gaussians.

    Runs once per consumed validation pass on the host through scipy. The
    matrix square root of the covariance product has no JAX equivalent.
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

    Building it loads the ~90MB weights, so it happens here, on first use.
    Constructing the metric opens nothing.
    """
    model, variables = _get_inception(weights)

    @jax.jit
    def activations(images):
        # Inception wants [-1, 1] at 299x299; pool3 output is [B, 1, 1, 2048]
        resized = jax.image.resize(images, (images.shape[0], 299, 299, 3), method='bilinear')
        features = model.apply(variables, resized, train=False)
        # apply returns the output alone, since no mutable collections are
        # asked for.
        assert not isinstance(features, tuple)
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


def fid(generated: NDArray[np.uint8] | jax.Array | Iterable[ArrayLike],
        reference: NDArray[np.uint8] | jax.Array | Iterable[ArrayLike],
        *, batch_size: int = 64, weights: str | Path | None = None) -> float:
    """Measure FID between two sets of uint8 [N, H, W, 3] images.

    Each side is one array or an iterable of arrays, so a directory of samples
    can stream past in blocks of `batch_size` rows instead of being held at
    once. The value is the distance between the two populations passed in,
    which is FID-50k only at 50,000 images a side.

    `weights` is the feature extractor's parameters as a file, the way
    `clip_score(modelname=)` names a local CLIP: the InceptionV3 variables tree
    in safetensors, which `tools/convert_inception_weights.py` writes. Unset
    downloads the published checkpoint and converts it. Two distances are
    comparable only when both were measured with the same one, which is why
    every distance logs which it was.
    """
    if batch_size < 1:
        raise ValueError(f"fid: a batch holds at least one image, got batch_size={batch_size}")
    named = None if weights is None else str(weights)
    with metric_device():
        stats = FIDStats(
            _pooled_stats(_unit_range_batches(generated, population="generated",
                                              batch_size=batch_size),
                          population="generated", weights=named),
            _pooled_stats(_unit_range_batches(reference, population="real",
                                              batch_size=batch_size),
                          population="real", weights=named))
    return _pooled_distance(stats, named)


@metrics("fid")
@dataclass(frozen=True)
class FID:
    """Scores FID over a pass, pooling statistics and taking one final distance.

    The call gathers the sampled grid and the batch's reference field. The
    features, the statistics and the distance are the ones `fid` runs, so a
    pass over 50,000 images a side reports the number `fid` reports, and
    `weights` names the extractor's parameters there the same way.
    """

    field: str = "image"
    weights: str | None = None
    name = "fid"
    reads = ImageGrid

    @property
    def feature_identity(self) -> str:
        return _features(self.weights)

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
