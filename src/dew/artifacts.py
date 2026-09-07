"""What an objective's evaluation produces, as typed values.

Objectives return these values from scoring and preview hooks. A metric
consumes scoring artifacts; a tracker renders previews by type. Array leaves
cross jit, while optional captions and decoded text remain host metadata.
"""

from __future__ import annotations

import json
from typing import TypeVar

from flax import struct
import jax
from jax.experimental import multihost_utils
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
    tokens: jax.Array | np.ndarray
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


def collective_host(value: T, *, phase: str) -> T:
    """Materialize an evaluation tree on every rank with transfer consensus.

    All ranks must call this, even for entirely local trees. Local leaves and
    global arrays' addressable shards are checked before any data gather.
    Ranks then agree the ordered global gather plan and each transfer outcome.
    Local-only trees may differ, as with root-only decoded previews. A device
    failure inside an in-flight collective still needs runtime termination.
    """
    leaves = []
    tree = None
    global_indices = []
    plan = []
    error = None
    try:
        paths, tree = jax.tree_util.tree_flatten_with_path(value)
        for path, leaf in paths:
            if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
                for shard in leaf.addressable_shards:
                    np.asarray(shard.data)
                global_indices.append(len(leaves))
                plan.append([jax.tree_util.keystr(path), list(leaf.shape), str(leaf.dtype),
                             str(leaf.sharding)])
                leaves.append(leaf)
            else:
                leaves.append(np.asarray(leaf))
    except BaseException as failure:
        error = failure
    agree_process_phase(error, phase=f"{phase} transfer preflight")
    root_plan = broadcast_from_process_zero(plan)
    error = None if plan == root_plan else ValueError("global array gather plans differ across ranks")
    agree_process_phase(error, phase=f"{phase} gather plan")
    for index in global_indices:
        error = None
        try:
            leaves[index] = _addressable(leaves[index])
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase=f"{phase} transfer leaf {index}")
    error = None
    result = value
    try:
        assert tree is not None
        result = jax.tree.unflatten(tree, leaves)
    except BaseException as failure:
        error = failure
    agree_process_phase(error, phase=f"{phase} tree reconstruction")
    return result


def broadcast_from_process_zero(value):
    """A JSON-encodable value broadcast from rank zero to every rank."""
    payload = np.frombuffer(json.dumps(value).encode(), np.uint8)
    length = int(multihost_utils.broadcast_one_to_all(np.asarray(len(payload), np.int64)))
    if jax.process_index() != 0:
        payload = np.zeros(length, np.uint8)
    return json.loads(multihost_utils.broadcast_one_to_all(payload).tobytes())


def agree_process_phase(error: BaseException | None, *, phase: str,
                        available: bool = True) -> int:
    """Propagate host errors, then count ranks declaring availability.

    All live ranks must reach this boundary. It cannot rescue a failed or
    blocked device collective. Errors take priority over unavailable input.
    """
    if jax.process_count() == 1:
        if error is not None:
            raise error
        return int(available)
    status = 2 if error is not None else int(available)
    statuses = np.asarray(multihost_utils.process_allgather(np.asarray(status, np.int32))).reshape(-1)
    failed = np.flatnonzero(statuses == 2)
    if not failed.size:
        return int(np.count_nonzero(statuses))
    source = int(failed[0])
    is_source = jax.process_index() == source
    message = b""
    if is_source:
        assert error is not None
        message = f"{type(error).__name__}: {error}".encode("utf-8", errors="replace")
        if len(message) > 4096:
            message = message[:4064] + b" ... [diagnostic truncated]"
    length = int(multihost_utils.broadcast_one_to_all(
        np.asarray(len(message), np.int32), is_source=is_source))
    payload = np.frombuffer(message, np.uint8) if is_source else np.zeros(length, np.uint8)
    diagnostic = multihost_utils.broadcast_one_to_all(payload, is_source=is_source).tobytes()
    context = f"Process phase {phase} failed on rank {source}: {diagnostic.decode('utf-8', errors='replace')}"
    if error is not None:
        error.add_note(context)
        raise error
    raise RuntimeError(context)
