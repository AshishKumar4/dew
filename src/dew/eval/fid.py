import functools
import warnings
from dataclasses import dataclass

from numpy.typing import NDArray

import jax
import jax.numpy as jnp
import numpy as np

from dew.artifacts import ImageGrid
from dew.inputs import unit_range
from dew.registry import metrics
from .common import metric_device


@functools.lru_cache(maxsize=None)
def _get_inception():
    """The pool3 feature extractor and its parameters, loaded once per
    process. The FID InceptionV3 is about 90 MB of weights, and every metric
    built from this module shares the copy."""
    from .inception import InceptionV3
    print("[metrics] Loading InceptionV3 FID weights (cached for reuse)...")
    model = InceptionV3(pretrained=True)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3)))
    return model, params


def _sqrtm(product):
    """`sqrtm` without its singularity warning. A singular product is the
    case the finiteness check in `frechet_distance` handles."""
    from scipy import linalg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        return linalg.sqrtm(product)


def frechet_distance(mu_a, sigma_a, mu_b, sigma_b, eps=1e-6) -> float:
    """Frechet distance between two multivariate gaussians.

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
    """A population's count, float64 mean and centered sum of outer products."""

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


@functools.lru_cache(maxsize=None)
def _get_activations():
    """The jitted pool3 feature extractor, built on first use.

    Building it loads the ~90MB weights, so it happens here, on first use.
    Constructing the metric opens nothing.
    """
    model, params = _get_inception()

    @jax.jit
    def activations(images):
        # Inception wants [-1, 1] at 299x299; pool3 output is [B, 1, 1, 2048]
        resized = jax.image.resize(images, (images.shape[0], 299, 299, 3), method='bilinear')
        features = model.apply(params, resized, train=False)
        # apply returns the output alone, since no mutable collections are
        # asked for.
        assert not isinstance(features, tuple)
        return features.reshape(features.shape[0], -1)

    return activations


@dataclass(frozen=True)
class FID:
    """Pooled-population FID with O(D²) pass state and one final distance."""

    field: str = "image"
    name = "fid"
    reads = ImageGrid
    feature_identity = "FID InceptionV3 pool3, bilinear 299x299, [-1, 1]"

    def __call__(self, artifact: ImageGrid, batch) -> FIDStats:
        with metric_device():
            activations = _get_activations()
            generated = GaussianStats.from_features(activations(artifact.images), population="generated")
            real = GaussianStats.from_features(activations(unit_range(batch[self.field])), population="real")
        return FIDStats(generated, real)

    def merge(self, accumulated: FIDStats, contribution: FIDStats) -> FIDStats:
        accumulated.generated = accumulated.generated.merge(contribution.generated)
        accumulated.real = accumulated.real.merge(contribution.real)
        return accumulated

    def finalize(self, accumulated: FIDStats) -> float:
        generated, real = accumulated.generated, accumulated.real
        if generated.count < 2 or real.count < 2:
            raise ValueError(
                "fid generated and real populations require at least two rows each; "
                f"got generated={generated.count}, real={real.count}")
        result = frechet_distance(generated.mean, generated.covariance(population="generated"),
                                  real.mean, real.covariance(population="real"))
        print(f"FID populations: generated={generated.count}, real={real.count}; "
              f"features={self.feature_identity}")
        return result


@metrics("fid")
def fid(field: str = "image") -> FID:
    """FID of the generated and real populations actually consumed in the pass.

    This is FID-50k only when 50,000 generated and real observations were
    consumed with the matching feature and preprocessing definition.
    """
    return FID(field)
