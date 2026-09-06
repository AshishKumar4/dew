"""What an objective's evaluation produces, as typed values.

Objectives return these values from scoring and preview hooks. A metric
consumes scoring artifacts; a tracker renders previews by type. Array leaves
cross jit, while optional captions and decoded text remain host metadata.
"""

from __future__ import annotations

from typing import TypeVar

from flax import struct
import jax
import numpy as np

T = TypeVar("T")

@struct.dataclass
class ImageGrid:
    """Images in [-1, 1], `[N, H, W, C]`, with the text each was conditioned on
    where there was any."""
    images: jax.Array
    captions: tuple[str, ...] = struct.field(pytree_node=False, default=())


@struct.dataclass
class VideoGrid:
    """Clips in [-1, 1], `[N, T, H, W, C]`."""
    videos: jax.Array
    captions: tuple[str, ...] = struct.field(pytree_node=False, default=())


@struct.dataclass
class TextSamples:
    """Generated token rows, with optional decoded preview text and prompt."""
    tokens: jax.Array
    prompt: str = struct.field(pytree_node=False, default="")
    texts: tuple[str, ...] = struct.field(pytree_node=False, default=())


@struct.dataclass
class Representations:
    """Encoder outputs `[N, D]` and the labels of the records they came from,
    for a probe to score."""
    features: jax.Array
    labels: jax.Array


@struct.dataclass
class TokenScores:
    """Teacher-forced per-token losses `[N, L]` and the weight of each target,
    1 where it counts and 0 where it is padding or a document's first token.
    A perplexity is exp of the weighted mean over a whole pass, so a batch
    with no counted target weighs nothing."""
    losses: jax.Array
    weights: jax.Array


# Scoring and preview hooks each return one artifact or a tuple. Metrics
# pick exactly one scoring artifact by type; previews never satisfy metrics.
Artifact = ImageGrid | VideoGrid | TextSamples | Representations | TokenScores
Artifacts = Artifact | tuple[Artifact, ...]


def _addressable(leaf: jax.Array | np.ndarray) -> np.ndarray:
    """`leaf` as numpy, gathering it across the pool when it is a global
    array this process holds only a shard of."""
    if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
        from jax.experimental import multihost_utils

        return np.asarray(multihost_utils.process_allgather(leaf, tiled=True))
    return np.asarray(leaf)


def host(value: T) -> T:
    """An artifact whose arrays are host-local numpy.

    Scoring and drawing happen on the host: a metric reads the arrays with
    numpy, a tracker draws them. On one process that is a device transfer. On
    a pool the arrays are shards of a global array, which numpy cannot read at
    all, and the gather that completes them is a collective, so every process
    has to make the same call. That is why the trainer brings an artifact home
    once for the whole pool before any metric or tracker, which run on one
    process, sees it.
    """
    return jax.tree.map(_addressable, value)
