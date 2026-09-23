"""What an objective's evaluation produces, as typed values.

Objectives return these values from scoring and preview hooks. A metric
consumes scoring artifacts; a tracker renders previews by type. Array leaves
cross jit, while optional captions and decoded text remain host metadata.
"""

from __future__ import annotations

import itertools
import json
import os
import socket
import sys
import threading
import time
import types
from collections.abc import Callable
from typing import TypeVar

import jax
import numpy as np
from flax import struct
from jax.experimental import multihost_utils
from jax.typing import ArrayLike
from numpy.typing import NDArray

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


def uint8_pixels(images: ArrayLike) -> NDArray[np.uint8]:
    """[-1, 1] pixels, as `ImageGrid` and `VideoGrid` hold them, as uint8 in [0, 255].

    Each value goes to its nearest level, `(x + 1) * 127.5` in float32
    rounded half to even (`np.rint`), then clipped, because a sample can
    leave the range. A metric scores and a tracker previews these bytes, so
    the two see the same image.
    """
    levels = np.rint((np.asarray(images, np.float32) + 1.0) * 127.5)
    return np.clip(levels, 0, 255).astype(np.uint8)


def _addressable(leaf: jax.Array | np.ndarray) -> np.ndarray:
    """`leaf` as numpy, gathering it across the pool when it is a global
    array this process holds only a shard of."""
    if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
        return np.asarray(multihost_utils.process_allgather(leaf, tiled=True))
    return np.asarray(leaf)


def host[T](value: T) -> T:
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


def collective_host[T](value: T, *, phase: str) -> T:
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
    materialized = value
    try:
        assert tree is not None
        materialized = jax.tree.unflatten(tree, leaves)
    except BaseException as failure:
        error = failure
    agree_process_phase(error, phase=f"{phase} tree reconstruction")
    return materialized


def broadcast_from_process_zero(value):
    """A JSON-encodable value broadcast from rank zero to every rank."""
    payload = np.frombuffer(json.dumps(value).encode(), np.uint8)
    length = int(multihost_utils.broadcast_one_to_all(np.asarray(len(payload), np.int64)))
    if jax.process_index() != 0:
        payload = np.zeros(length, np.uint8)
    return json.loads(multihost_utils.broadcast_one_to_all(payload).tobytes())


def agreed[T](phase: str, operation: Callable[[], T]) -> T:
    """Run `operation` on every rank, then agree on the outcome before going on.

    A rank that fails reports its error at the agreement point instead of
    raising alone, so its peers hear about it there rather than hanging at
    the next collective. The peers raise `PeerFailure`; the failing rank
    re-raises its own error. On one process this is a plain call.
    """
    held: tuple[T] | None = None
    error: BaseException | None = None
    try:
        held = (operation(),)
    except BaseException as failure:
        error = failure
    try:
        agree_process_phase(error, phase=phase)
    except BaseException as failure:
        if error is None:
            raise PeerFailure(str(failure)) from failure
        raise
    assert held is not None
    return held[0]


class PeerFailure(RuntimeError):
    """Another rank failed at a phase agreement this rank passed."""


AGREEMENT_PATIENCE_SECONDS = 24 * 3600
"""How long ranks that reached an agreement wait on the host for the rest.

Host work before an agreement, such as process 0 uploading the final
checkpoint, can take hours; the pool bounds device executions, not this."""

_agreements = itertools.count()


def agree_process_phase(error: BaseException | None, *, phase: str,
                        available: bool = True) -> int:
    """Propagate host errors, then count ranks declaring availability.

    All live ranks must reach this boundary. It cannot rescue a blocked
    device collective: a rank that failed between steps meets peers still
    inside the next step's collectives, which wait for it for ever on a GPU.
    So a failing rank first publishes its error (`publish_failure`), which
    ends the pool within `FAILURE_GRACE_SECONDS` unless the agreement
    completes and withdraws it. Errors take priority over unavailable input.

    The ranks meet on the host, at a coordination-service barrier, before
    the device collectives that carry the outcome. A rank still busy on its
    host keeps the others waiting there rather than inside an execution,
    which the pool's bound (`dew.training.runtime.EXECUTION_TIMEOUT`) would
    end.
    """
    if jax.process_count() == 1:
        if error is not None:
            raise error
        return int(available)
    published = error is not None and publish_failure(error, f"phase {phase}")
    status = 2 if error is not None else int(available)
    _client().wait_at_barrier(f"dew/agreement/{next(_agreements)}",
                              int(AGREEMENT_PATIENCE_SECONDS * 1000))
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
    if published:
        # Every rank took part in the agreement's collectives and holds the
        # failure; one that met a peer's pending step instead never gets here.
        withdraw_failure()
    context = f"Process phase {phase} failed on rank {source}: {diagnostic.decode('utf-8', errors='replace')}"
    if error is not None:
        error.add_note(context)
        raise error
    raise RuntimeError(context)


FAILURE_DIRECTORY = "dew/failure/"
FAILURE_KEY = FAILURE_DIRECTORY + "published"
"""The coordination-service key a failing process writes its error under."""

FAILURE_GRACE_SECONDS = 60.0
"""How long a published failure may go unheard before every process ends."""


def _client():
    """The jax.distributed client, through orbax's public accessor for it."""
    from orbax.checkpoint import multihost

    return multihost.get_jax_distributed_client()


def this_process() -> str:
    """This process as `host:pid`, named without the backend, which may be
    the thing that failed or be opening in another thread."""
    return f"{socket.gethostname()}:{os.getpid()}"


def publish_failure(error: BaseException, where: str) -> bool:
    """Write this process's failure where every process's watch sees it.

    Returns False when another failure is already there: the first stands.
    """
    text = f"{os.urandom(4).hex()} {this_process()} failed in {where}: "
    try:
        _client().key_value_set(FAILURE_KEY, (text + f"{type(error).__name__}: {error}")[:2048])
    except jax.errors.JaxRuntimeError:  # the key exists: a failure is already published
        return False
    return True


def withdraw_failure() -> None:
    """Remove the published failure once the pool has heard it at an agreement."""
    _client().key_value_delete(FAILURE_KEY)


def end_pool_on_failure(grace: float = FAILURE_GRACE_SECONDS) -> None:
    """End this process when a failure goes unheard, or when it fails itself.

    A process of a pool that raises past its program, or meets a peer's
    failure it cannot hear, would otherwise hang: its peers wait for it in a
    collective no GPU backend times out, and jax.distributed's shutdown
    barrier holds an exiting process up to shutdown_timeout_seconds (300 s)
    for them. So an uncaught exception prints, publishes and leaves at once,
    and a watch thread ends the process `grace` seconds after any published
    failure that no agreement withdrew. `dew launch`, srun and a pod's
    scheduler then see the failure and stop the rest.

    It needs only the pool's coordination service, not the backend, so it
    goes in before the backend opens: a process whose devices fail to open
    after the pool has formed would otherwise wait in that barrier for peers
    that wait for its devices.
    """
    previous = sys.excepthook
    client = _client()

    def leave(kind: type[BaseException], value: BaseException,
              trace: types.TracebackType | None) -> None:
        previous(kind, value, trace)
        publish_failure(value, "the program")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130 if issubclass(kind, KeyboardInterrupt) else 1)

    def published() -> str | None:
        # A directory read finds nothing without an error, where reading the
        # key alone raises every time no failure is there.
        return dict(client.key_value_dir_get(FAILURE_DIRECTORY)).get(FAILURE_KEY)

    def watch() -> None:
        while True:
            time.sleep(5.0)
            try:
                seen = published()
                if seen is None:
                    continue
                time.sleep(grace)
                if published() != seen:  # an agreement heard and withdrew it
                    continue
            except jax.errors.JaxRuntimeError:  # the pool's coordination service is gone
                continue
            sys.stderr.write(f"{seen.split(' ', 1)[1]}\nNo agreement heard that failure in "
                             f"{grace:.0f} s, so this process waits in a collective that will "
                             f"not complete; {this_process()} ends.\n")
            sys.stderr.flush()
            os._exit(1)

    sys.excepthook = leave
    threading.Thread(target=watch, name="dew-failure-watch", daemon=True).start()
