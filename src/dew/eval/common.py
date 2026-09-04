"""The metric shape the trainer consumes: one artifact type, a per-batch
measurement, the mean over the pass."""

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from dew.artifacts import ImageGrid, VideoGrid


def frames(artifact) -> Any:
    """The pixels an image metric scores, `[N, H, W, C]` or `[N, T, H, W, C]`
    in [-1, 1], from either grid."""
    return artifact.videos if isinstance(artifact, VideoGrid) else artifact.images


@dataclass(frozen=True, eq=False)
class ImageMetric:
    """A measurement of an `ImageGrid` against the batch it was sampled for,
    averaged over the validation pass."""

    name: str
    measure: Callable[[Any, Any], float]
    reads: type = ImageGrid

    def __call__(self, artifact, batch) -> float:
        return float(self.measure(artifact, batch))

    def reduce(self, values: Sequence[float]) -> float:
        return float(np.mean(values))
