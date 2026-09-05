import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from dew.inputs import unit_range
from dew.registry import metrics
from .common import ImageMetric


@functools.lru_cache(maxsize=None)
def _get_inception():
    """The pool3 feature extractor and its parameters, loaded once per
    process: the FID InceptionV3 is about 90 MB of weights, and every metric
    built from this module shares the copy."""
    from .inception import InceptionV3
    print("[metrics] Loading InceptionV3 FID weights (cached for reuse)...")
    model = InceptionV3(pretrained=True)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3)))
    return model, params


def _sqrtm(product):
    """`sqrtm` without its singularity warning: a singular product is the
    case the finiteness check in `frechet_distance` handles, so the warning
    is noise."""
    from scipy import linalg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        return linalg.sqrtm(product)


def frechet_distance(mu_a, sigma_a, mu_b, sigma_b, eps=1e-6) -> float:
    """Frechet distance between two multivariate gaussians.

    Runs on the host through scipy: the matrix square root of the covariance
    product has no jax equivalent, and FID is computed once per validation
    batch so the transfer is irrelevant.
    """

    mu_a, mu_b = np.atleast_1d(mu_a), np.atleast_1d(mu_b)
    sigma_a, sigma_b = np.atleast_2d(sigma_a), np.atleast_2d(sigma_b)

    # sqrtm's result is complex when rounding leaves the product with a
    # negative eigenvalue; the imaginary part is that noise.
    covmean = _sqrtm(sigma_a.dot(sigma_b))
    if not np.isfinite(covmean).all():
        # Singular product covariance, nudge the diagonal as in the reference
        # implementations rather than returning a nan
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = _sqrtm((sigma_a + offset).dot(sigma_b + offset))
    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean))


def _gaussian_stats(activations):
    activations = np.asarray(activations, dtype=np.float64)
    return activations.mean(axis=0), np.cov(activations, rowvar=False)


@functools.lru_cache(maxsize=None)
def _get_activations():
    """The jitted pool3 feature extractor, built on first use.

    Building it loads the ~90MB weights, so it happens here rather than in
    the factory: constructing the metric opens nothing.
    """
    model, params = _get_inception()

    @jax.jit
    def activations(images):
        # Inception wants [-1, 1] at 299x299; pool3 output is [B, 1, 1, 2048]
        resized = jax.image.resize(images, (images.shape[0], 299, 299, 3), method='bilinear')
        features = model.apply(params, resized, train=False)
        # apply returns the output alone unless mutable collections were asked
        # for, and none were.
        assert not isinstance(features, tuple)
        return features.reshape(features.shape[0], -1)

    return activations


@metrics("fid")
def fid(field: str = "image") -> ImageMetric:
    """FID between the sampled images and the batch's, lower is better.

    Per-batch FID is noisy at typical validation batch sizes and is only
    meaningful as a relative trend across checkpoints, not as a headline number
    comparable to published FID-50k.
    """

    def measure(artifact, batch):
        activations = _get_activations()
        mu_gen, sigma_gen = _gaussian_stats(activations(artifact.images))
        mu_real, sigma_real = _gaussian_stats(activations(unit_range(batch[field])))
        return frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)

    return ImageMetric(name="fid", measure=measure)
