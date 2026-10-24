"""A fit's profiler window: the request (`ProfileWindow`) and the capture's life on every rank.

A window is owned before the dataset or the mesh do any work (`owned_capture`), started on every rank
together once its warmup steps have run (`start_capture`), and stopped once its steps have (`stop_capture`).
Each phase runs under `dew._coordination.agreed`, so a rank that fails raises on all of them, and none
proceeds alone to deadlock its peers at the next collective. The trainer reports what the window wrote.
"""

import dataclasses
import sys
from typing import TYPE_CHECKING

from dew._coordination import agreed
from dew.telemetry import profile as telemetry_profile

if TYPE_CHECKING:
    import jax

    from dew.telemetry.profile import Profiler


@dataclasses.dataclass(frozen=True)
class ProfileWindow:
    """A request for one profiler window per fit.

    It traces `steps` steps into `directory` after `warmup` steps have run, so the
    trace shows the loop and not the compile.

    `dew.Profiler` is the other way to capture a trace, as a context manager
    around any code; a fit refuses to schedule a window inside one. The loop
    reports the window it wrote as the `ProfileWindow` record of
    `dew.telemetry.records`.
    """
    directory: str
    steps: int
    warmup: int = 2


def owned_capture(window: ProfileWindow | None) -> "Profiler | None":
    """Take ownership of the capture `window` will write, or None without one.

    A configured window must own the capture. When another profiler
    already holds it, refuse before the dataset or the mesh do any work,
    so neither trace is silently dropped or cut short. The window also
    resolves its optional dependency here, rather than after warmup steps
    have run.

    Every rank either leaves owning a stopped Profiler or raises
    together; a rank that proceeded alone would deadlock its peers at the
    first collective of training.
    """
    if window is None:
        return None

    def own_window() -> "Profiler":
        if telemetry_profile.active_profile() is not None:
            raise ValueError(
                "Trainer.profile cannot schedule a window while an explicit "
                "dew.Profiler capture is active; drop one or stop the outer "
                "profiler before fitting")
        telemetry_profile.require_profile_support()
        return telemetry_profile.Profiler(window.directory)

    return agreed("profiling window setup", own_window)


def start_capture(profiler: "Profiler") -> None:
    """Open the profiler's capture on every rank.

    A peer that failed to start raises here. A capture this rank did
    start is closed first, so no live trace outlives the aborted run.
    """
    capturing = False

    def start() -> None:
        nonlocal capturing
        profiler.start()
        capturing = True

    try:
        agreed("profiling window start", start)
    except BaseException as primary:
        if capturing:
            try:
                profiler.stop()
            except BaseException as failure:
                primary.add_note(f"Profiler stop failed: {failure!r}")
        raise


def stop_capture(profiler: "Profiler", loss: "jax.Array | None") -> None:
    """Stop the window's owned capture on every rank.

    The core Profiler drains the backend and exports the native reports
    on `stop`. The run's last `loss` is blocked on only to order the
    primary's failure ahead of the profiler's own drain."""
    def stop() -> None:
        try:
            if loss is not None:
                loss.block_until_ready()
        finally:
            primary = sys.exception()
            try:
                profiler.stop()
            except BaseException as failure:
                if primary is None:
                    raise
                primary.add_note(f"Profiler stop failed: {failure!r}")

    agreed("profile stop", stop)
