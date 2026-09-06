"""Host-local image metric sufficient statistics."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np
from jax.typing import ArrayLike

from dew.artifacts import ImageGrid, VideoGrid
from dew.objectives.base import Batch


@contextmanager
def metric_device():
    """Keep host metric kernels and lazily loaded weights on one local device."""
    device = jax.local_devices()[0]
    mesh = jax.sharding.Mesh(np.asarray([device]), ("metric",))
    with jax.set_mesh(mesh), jax.default_device(device):
        yield


def frames(artifact: ImageGrid | VideoGrid) -> jax.Array:
    """Pixels in [-1, 1], with videos retaining their frame axis."""
    return artifact.videos if isinstance(artifact, VideoGrid) else artifact.images


def paired(artifact: ImageGrid | VideoGrid, batch: Batch, field: str):
    """Generated and reference pixels aligned over the complete batch."""
    from dew.inputs import unit_range

    samples = frames(artifact)
    targets = unit_range(batch[field])
    if targets.shape[0] != samples.shape[0]:
        raise ValueError(
            f"the artifact holds {samples.shape[0]} rows and batch[{field!r}] "
            f"{targets.shape[0]}; paired metrics require equal counts")
    return samples, targets


@dataclass(frozen=True, eq=False)
class ImageMetric:
    """Per-image (per-frame for video) means over the consumed pass."""

    name: str
    measure: Callable[[ImageGrid | VideoGrid, Batch], ArrayLike]
    """One measurement per image or frame, never an already averaged scalar."""
    reads: type = ImageGrid

    def __call__(self, artifact, batch) -> tuple[float, int]:
        with metric_device():
            values = np.asarray(self.measure(artifact, batch), dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"{self.name} must measure one value per image or frame")
        return float(values.sum()), values.size

    def merge(self, accumulated: tuple[float, int],
              contribution: tuple[float, int]) -> tuple[float, int]:
        return accumulated[0] + contribution[0], accumulated[1] + contribution[1]

    def finalize(self, accumulated: tuple[float, int]) -> float:
        total, count = accumulated
        if count == 0:
            raise ValueError(f"{self.name}: no images or frames in the validation pass")
        return total / count
