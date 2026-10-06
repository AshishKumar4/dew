"""How a pool of processes agrees, gathers to the host and ends together.

`agreed` and `agree_process_phase` run a phase on every process and agree on
its outcome, `collective_host` gathers a tree's global leaves to the host
with that agreement, and a process that fails publishes it so its peers end
too (`end_pool_on_failure`). Training, evaluation, loading and serving all
coordinate through these, so they live apart from any one of them.
"""


from __future__ import annotations

import atexit
import itertools
import json
import os
import socket
import sys
import threading
import types
from collections.abc import Callable
from typing import Literal, overload

import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec

GATHER_BYTES = 256 * 2 ** 20
"""The bytes of global leaves `collective_host` gathers in one computation.

A group's leaves are replicated by one computation and agreed on once, so a
tree of many small leaves pays one round trip a group rather than one a leaf,
and a device holds at most one group's values, or its one larger leaf, beside
the tree's shards. With `held_by="first"` a rank that holds nothing still
holds one replicated group on its devices while that group's computation
runs, as `process_allgather` replicated each leaf.
"""


def _mesh(leaf: jax.Array) -> jax.sharding.Mesh | None:
    """The concrete mesh `leaf` is laid out on, if it is laid out on one."""
    mesh = leaf.sharding.mesh if isinstance(leaf.sharding, NamedSharding) else None
    return mesh if isinstance(mesh, jax.sharding.Mesh) else None


def _identity(*values: jax.Array) -> tuple[jax.Array, ...]:
    return values


def _gather_groups(leaves: list[jax.Array]) -> list[list[int]]:
    """`leaves`' positions in consecutive groups on one mesh, each of at most
    GATHER_BYTES or one larger leaf. The same on every rank: it reads the
    shapes, dtypes and shardings the ranks agreed in the gather plan."""
    groups: list[list[int]] = []
    size = 0
    for index, leaf in enumerate(leaves):
        mesh = _mesh(leaf)
        if (groups and mesh is not None and mesh == _mesh(leaves[groups[-1][0]])
                and size + leaf.nbytes <= GATHER_BYTES):
            groups[-1].append(index)
            size += leaf.nbytes
        else:
            groups.append([index])
            size = leaf.nbytes
    return groups


def _gathered(group: list[jax.Array], held: bool) -> list[np.ndarray] | None:
    """`group`'s global arrays whole, on this host where `held`. One mesh's
    arrays are replicated by one computation; a leaf on another sharding
    goes alone through `process_allgather`. A rank that holds nothing still
    completes the computation, which is a collective."""
    mesh = _mesh(group[0])
    if mesh is None:
        return [np.asarray(multihost_utils.process_allgather(group[0], tiled=True))]
    replicated = NamedSharding(mesh, PartitionSpec())
    outputs = jax.jit(_identity, out_shardings=(replicated,) * len(group))(*group)
    if not held:
        jax.block_until_ready(outputs)
        return None
    whole = [output.addressable_data(0) for output in outputs]
    for value in whole:
        value.copy_to_host_async()
    return [np.asarray(value) for value in whole]


@overload
def collective_host[T](value: T, *, phase: str, held_by: Literal["every"] = "every") -> T: ...
@overload
def collective_host[T](value: T, *, phase: str, held_by: Literal["first"]) -> T | None: ...
def collective_host[T](value: T, *, phase: str, held_by: Literal["every", "first"] = "every") -> T | None:
    """Materialize an evaluation tree on every rank with transfer consensus.

    All ranks must call this, even for entirely local trees. Every leaf is
    waited on first, so a computation that failed on a rank reports at the
    preflight rather than inside a gather collective. Ranks then agree the
    ordered global gather plan, and gather global leaves in groups
    (`GATHER_BYTES`), agreeing each group's outcome before the next starts.
    Local-only leaves may differ across ranks. A device failure inside an
    in-flight collective still needs runtime termination. With `held_by`
    "first" only process 0 copies the tree home and returns it; the others
    take part in every computation and agreement and return None.
    """
    held = held_by == "every" or jax.process_index() == 0
    leaves = []
    tree = None
    global_indices = []
    plan = []
    error = None
    try:
        paths, tree = jax.tree_util.tree_flatten_with_path(value)
        local = []
        for path, leaf in paths:
            if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
                global_indices.append(len(leaves))
                plan.append([jax.tree_util.keystr(path), list(leaf.shape), str(leaf.dtype),
                             str(leaf.sharding)])
            else:
                local.append(len(leaves))
            leaves.append(leaf)
        if held:
            # One read for every local leaf: their copies run together.
            for index, home in zip(local, jax.device_get([leaves[index] for index in local]), strict=True):
                leaves[index] = np.asarray(home)
        jax.block_until_ready(leaves)
    except BaseException as failure:
        error = failure
    agree_process_phase(error, phase=f"{phase} transfer preflight")
    root_plan = broadcast_from_process_zero(plan)
    error = None if plan == root_plan else ValueError("global array gather plans differ across ranks")
    agree_process_phase(error, phase=f"{phase} gather plan")
    for group in _gather_groups([leaves[index] for index in global_indices]):
        indices = [global_indices[position] for position in group]
        error = None
        try:
            gathered = _gathered([leaves[index] for index in indices], held)
            if gathered is not None:
                for index, whole in zip(indices, gathered, strict=True):
                    leaves[index] = whole
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase=f"{phase} transfer leaves {indices[0]}-{indices[-1]}")
    error = None
    materialized = value
    try:
        assert tree is not None
        if held:
            materialized = jax.tree.unflatten(tree, leaves)
    except BaseException as failure:
        error = failure
    agree_process_phase(error, phase=f"{phase} tree reconstruction")
    return materialized if held else None


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
    """Raised when another process failed at a phase agreement that this process passed."""


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

FAILURE_POLL_SECONDS = 5.0
"""How often a pool's failure watch reads the coordination service."""


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


def stop_at_exit(thread: threading.Thread, stop: Callable[[], None], *, timeout: float) -> Callable[[], None]:
    """End `thread` before Python finalizes, and return what withdraws that.

    A thread still inside jaxlib when Python finalizes is ended with
    pthread_exit (CPython before 3.14), and the unwind through jaxlib's C++
    GIL guard aborts a finished program with SIGABRT. The handler calls
    `stop` and waits up to `timeout` seconds; atexit runs it before jax's own,
    registered earlier, so jax's clients are still open.
    """

    def finish() -> None:
        try:
            stop()
        finally:
            thread.join(timeout)

    atexit.register(finish)
    return lambda: atexit.unregister(finish)


def end_pool_on_failure(grace: float = FAILURE_GRACE_SECONDS) -> None:
    """End this process when a failure goes unheard, or when it fails itself.

    A process that raises past its program, or meets a peer's failure it
    cannot hear, would otherwise hang in a collective no GPU backend times
    out, or in jax.distributed's 300 s shutdown barrier. So an uncaught
    exception prints, publishes and leaves at once, and a watch thread ends
    the process `grace` seconds after any published failure no agreement
    withdrew; `dew launch`, srun and a pod's scheduler then stop the rest.

    It needs only the coordination service, so it goes in before the backend
    opens, and its watch ends with the program (`stop_at_exit`).
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

    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(FAILURE_POLL_SECONDS):
            try:
                seen = published()
                if seen is None:
                    continue
                if stop.wait(grace):
                    return
                if published() != seen:  # an agreement heard and withdrew it
                    continue
            except jax.errors.JaxRuntimeError:  # the pool's coordination service is gone
                continue
            sys.stderr.write(f"{seen.split(' ', 1)[1]}\nNo agreement heard that failure in "
                             f"{grace:.0f} s, so this process waits in a collective that will "
                             f"not complete; {this_process()} ends.\n")
            sys.stderr.flush()
            os._exit(1)

    watcher = threading.Thread(target=watch, name="dew-failure-watch", daemon=True)
    sys.excepthook = leave
    watcher.start()
    # Every client leaves the pool's service in jax's exit handler, after this
    # one, so the service still answers a read under way within milliseconds.
    # The bound only keeps an exit from waiting on a service that stopped
    # answering.
    stop_at_exit(watcher, stop.set, timeout=5.0)


__all__ = [
    "PeerFailure",
    "agree_process_phase",
    "agreed",
    "broadcast_from_process_zero",
    "collective_host",
    "end_pool_on_failure",
]
