"""Accumulate host-local sufficient statistics for the image metrics."""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np
from jax.typing import ArrayLike

from dew.artifacts import Artifact, ImageGrid, VideoGrid
from dew.objectives.base import Batch, Shown, mean_of_totals, merge_totals


@dataclass(frozen=True, eq=False)
class Mean[Scored: Artifact]:
    """Average per-example values or additive (total, count) contributions.

    `better` is required so checkpoint ranking cannot infer the opposite
    direction. `reads` explicitly names the scoring artifact the function
    consumes. State belongs to the evaluation pass.
    """

    fn: Callable[[Scored, Batch], ArrayLike | tuple[float, float]]
    name: str
    better: Literal["higher", "lower"]
    reads: type[Scored]

    def __post_init__(self) -> None:
        if self.better not in ("higher", "lower"):
            raise ValueError("better must be higher or lower")
        if "/" in self.name:
            raise ValueError(
                "name must be unprefixed, such as accuracy; evaluation adds val/ or the split name"
            )

    @property
    def shown(self) -> Shown:
        return Shown(better=self.better)

    def __call__(self, artifact: Artifact, batch: Batch) -> tuple[float, float]:
        if not isinstance(artifact, self.reads):
            raise TypeError(f"{self.name} reads {self.reads.__name__}, not {type(artifact).__name__}")
        measured = self.fn(artifact, batch)
        if isinstance(measured, tuple):
            if len(measured) != 2:
                raise ValueError(f"{self.name}: totals must be a (total, count) pair")
            total, count = float(measured[0]), float(measured[1])
            if not np.isfinite(count) or count < 0:
                raise ValueError(f"{self.name}: count must be finite and nonnegative")
            return total, count
        values = np.asarray(measured, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"{self.name}: expected per-example values, not an already averaged scalar")
        return float(values.sum()), float(values.size)

    def merge(
        self, accumulated: tuple[float, float], contribution: tuple[float, float]
    ) -> tuple[float, float]:
        return merge_totals(accumulated, contribution)

    def finalize(self, accumulated: tuple[float, float]) -> float:
        if accumulated[1] <= 0:
            raise ValueError(f"{self.name}: no counted examples in the validation pass")
        return mean_of_totals(accumulated)


@contextmanager
def metric_device():
    """Keep host metric kernels and lazily loaded weights on one local device."""
    device = jax.local_devices()[0]
    mesh = jax.sharding.Mesh(np.asarray([device]), ("metric",))
    with jax.set_mesh(mesh), jax.default_device(device):
        yield


def frames(artifact: ImageGrid | VideoGrid) -> jax.Array:
    """Return pixels in [-1, 1], with videos keeping their frame axis."""
    return artifact.videos if isinstance(artifact, VideoGrid) else artifact.images


def paired(artifact: ImageGrid | VideoGrid, batch: Batch, field: str):
    """Align generated and reference pixels over the complete batch."""
    from dew.inputs import unit_range

    samples = frames(artifact)
    targets = batch[field]
    target_shape = np.shape(targets)
    if target_shape != samples.shape:
        raise ValueError(
            f"the artifact has pixel shape {samples.shape} and batch[{field!r}] "
            f"{target_shape}; paired metrics require equal counts and pixel shapes")
    return samples, unit_range(targets)


class ImageMetric(Mean[ImageGrid | VideoGrid]):
    """A `Mean` of one measurement per image, or per frame for video, taken on
    one local device. Flow-GRPO reads `fn` as a per-sample reward."""

    def __init__(self, name: str, measure: Callable[[ImageGrid | VideoGrid, Batch], ArrayLike], *,
                 better: Literal["higher", "lower"] = "higher",
                 reads: type[ImageGrid | VideoGrid] = ImageGrid):
        super().__init__(measure, name, better, reads)

    def __call__(self, artifact: Artifact, batch: Batch) -> tuple[float, float]:
        with metric_device():
            return super().__call__(artifact, batch)
