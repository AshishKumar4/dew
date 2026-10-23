"""What every serving kind shares, and the engine that serves kinds together.

A serving kind (`Kind`) admits the requests waiting for it within its own bounds, runs them with one
device call a `step`, and settles their futures: `dew.inference.serving.Server` is autoregressive
generation's kind, a slot scheduler over a resident cache, and `dew.inference.scoring.ScoringServer`
scoring's, one forward a batch. A kind of block or masked generation joins as one more. Each holds its
weights resident on its mesh (`dew.inference.tasks.resident`), reloads new weights through
`normalized` and `reloaded`, and answers each request through a `Timed` future that `settled`
completes. `Engine` serves kinds from one thread, a bounded step of each busy kind a turn, so none
starves another.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from dew.objectives.base import Variables


class Timed[T](Future[T]):
    """A request's future, with when it was submitted, admitted to a step and finished, in
    `time.perf_counter` seconds; `admitted` and `finished` are None until they happen."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted = time.perf_counter()
        self.admitted: float | None = None
        self.finished: float | None = None


def settled[T](future: Future[T], outcome: T | BaseException) -> None:
    """Complete `future` with `outcome`, a result or an error, unless its caller cancelled it first.

    A future its owner claimed (`set_running_or_notify_cancel`) can no
    longer be cancelled; one still pending is claimed here.
    """
    if not future.running() and not future.set_running_or_notify_cancel():
        return
    if isinstance(outcome, BaseException):
        future.set_exception(outcome)
    else:
        future.set_result(outcome)


def normalized(variables: Variables, source: list) -> Variables:
    """`variables` checked against the served tree's leaves `source` (`tree_flatten_with_path` of
    their shapes) and cast to their dtypes: the tree must match leaf for leaf in path and shape, and
    a leaf that is not floating point must match its dtype too, so the compiled programs keep running
    without a recompile."""
    # A frozen and a plain mapping flatten to different tree structures but
    # the same leaf paths, so the paths are what must match.
    incoming, incoming_structure = jax.tree_util.tree_flatten_with_path(variables)
    if [path for path, _ in incoming] != [path for path, _ in source]:
        raise ValueError("reloaded variables must have the served tree structure")
    for (_, new), (_, old) in zip(incoming, source, strict=True):
        kind, served_kind = jnp.result_type(new), old.dtype
        if np.shape(new) != np.shape(old) or (kind != served_kind and not (
                jnp.issubdtype(kind, jnp.floating) and jnp.issubdtype(served_kind, jnp.floating))):
            raise ValueError(
                f"a reloaded leaf is {kind}{list(np.shape(new))}, "
                f"the served leaf {served_kind}{list(np.shape(old))}")
    return jax.tree.unflatten(
        incoming_structure, [jnp.asarray(np.asarray(new, dtype=old.dtype)
                                         if isinstance(new, np.ndarray) else new, dtype=old.dtype)
                             for (_, new), (_, old) in zip(incoming, source, strict=True)])


def reloaded(incoming: Variables, served: Variables) -> Variables:
    """`incoming`, a tree of `served`'s leaves (`normalized`), copied onto `served`'s placement and
    dtypes, so the caller may donate its buffers right after."""
    leaves = []
    incoming_leaves, _ = jax.tree_util.tree_flatten_with_path(incoming)
    served_leaves, structure = jax.tree_util.tree_flatten_with_path(served)
    for (_, new), (_, old) in zip(incoming_leaves, served_leaves, strict=True):
        served_kind = jnp.result_type(old)
        # A placement can alias a shard of the caller's array; the copy
        # owns its buffer whatever the caller donates next.
        placement = old.sharding if isinstance(old, jax.Array) else None
        leaves.append(jnp.array(jax.device_put(new, placement), dtype=served_kind, copy=True))
    return jax.tree.unflatten(structure, leaves)


class Kind(Protocol):
    """One serving kind, as `Engine` drives it."""

    arrived: Callable[[], None]
    """Called after each submission, so an engine waiting for work wakes to it."""

    @property
    def busy(self) -> bool:
        """Whether a request waits for, or runs in, a step."""
        ...

    def step(self) -> None:
        """One bounded device call: admit what fits, run it, settle what finished."""
        ...

    def fail(self, failure: BaseException) -> None:
        """Stop after `failure`: every waiting and running request settles with it."""
        ...


class Engine:
    """Serves `kinds` from one thread of its own between `start` and `close`: each turn, every busy
    kind takes one step, so a kind's long queue never starves another's; with none busy the thread
    waits for a submission. A step that fails stops every kind, each request settled with the
    failure, as one device's failure is every kind's."""

    def __init__(self, *kinds: Kind) -> None:
        if not kinds:
            raise ValueError("an engine serves one kind or more")
        self.kinds = kinds
        self._arrived = threading.Event()
        self._stepping = threading.Lock()
        """Held for each step and each `reload`, which a kind serializes with its steps."""
        self._closed = False
        self._thread: threading.Thread | None = None
        self.failure: BaseException | None = None
        for kind in kinds:
            kind.arrived = self._arrived.set

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("the engine is serving already")
        self._closed = False
        self._thread = threading.Thread(target=self._serve, name="serving-engine")
        self._thread.start()

    def close(self) -> None:
        """Stop serving once what waits is answered."""
        self._closed = True
        self._arrived.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def reload(self, reloading: Callable[[], None]) -> None:
        """Run `reloading`, a kind's reload, between two steps."""
        with self._stepping:
            reloading()

    def _serve(self) -> None:
        try:
            while True:
                # Cleared before the look, so a submission after it wakes the wait.
                self._arrived.clear()
                busy = [kind for kind in self.kinds if kind.busy]
                for kind in busy:
                    with self._stepping:
                        kind.step()
                if not busy:
                    if self._closed:
                        return
                    self._arrived.wait()
        except BaseException as failure:
            # One device serves every kind: a failed step stops them all, each request told why.
            self.failure = failure
            for kind in self.kinds:
                kind.fail(failure)
