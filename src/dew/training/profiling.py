"""A fit's profiler window: the request (`ProfileWindow`) and the capture's life on every rank.

A window is owned before the dataset or the mesh do any work (`owned_capture`), started on every rank
together once its warmup steps have run (`start_capture`), and stopped once its steps have (`stop_capture`).
Each phase runs under `dew._coordination.agreed`, so a rank that fails raises on all of them, and none
proceeds alone to deadlock its peers at the next collective. The trainer reports what the window wrote.
"""

import dataclasses
import gzip
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from dew._coordination import agreed
from dew.telemetry import profile as telemetry_profile
from dew.training.tracker import LocalTracker

if TYPE_CHECKING:
    import jax

    from dew.telemetry.profile import Profiler


@dataclasses.dataclass(frozen=True)
class ProfileWindow:
    """A request for one profiler window per fit.

    It traces `steps` steps into `directory` after `warmup` steps have run and the
    run's first step has compiled, so the trace shows the loop and not the compile.
    A fit under JAX's AutoPGLE, which profiles programs in sessions of its own,
    refuses a window (`PGLE_CAPTURE`).

    `dew.Profiler` is the other way to capture a trace, as a context manager
    around any code; a fit refuses to schedule a window inside one. The loop
    reports the window it wrote as the `ProfileWindow` record of
    `dew.telemetry.records`.
    """
    directory: str
    steps: int
    warmup: int = 2


def pgle_runs() -> int:
    """The runs of a program JAX's AutoPGLE profiles before compiling it again guided by their profile
    (`jax_enable_pgle`, `jax_pgle_profiling_runs`), 0 where it is off. JAX's jit guides the programs it
    compiles at a call; a step the trainer compiles ahead of its run (`Lowered.compile`) is no such call,
    which JAX compiles for the profile's sake (command buffers off) and never guides, so the trainer
    guides it (`dew.training.planning.Planned.guided`)."""
    import jax

    return jax.config.jax_pgle_profiling_runs if jax.config.jax_enable_pgle else 0


PGLE_CAPTURE = (
    "JAX's AutoPGLE (jax_enable_pgle) profiles each program's first runs in a profiling session of its own, "
    "and a process holds one session at a time, so no capture runs beside it: train once under "
    "JAX_ENABLE_PGLE=1 to fill the compilation cache with the guided programs, then capture a run under "
    "JAX_COMPILATION_CACHE_EXPECT_PGLE=1, which loads them")
"""Why a capture refuses to run where JAX's AutoPGLE profiles programs, and JAX's own way round it."""


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
        if pgle_runs():
            raise ValueError(PGLE_CAPTURE)
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


def traced_steps(capture: Path) -> tuple[int, ...]:
    """The steps a capture's training annotations hold (`jax.profiler.StepTraceAnnotation`, which a fit
    names `train`), each by the steps the run had taken as it began, its `step_num`, in order, read from
    the capture's native trace as it exports it (`*.trace.json.gz`). Reading an event's stats through
    `jax.profiler.ProfileData` instead crashed the process on this jaxlib (a CPU capture's annotations)."""
    steps = set()
    for trace in capture.glob("plugins/profile/*/*.trace.json.gz"):
        with gzip.open(trace, "rt") as stream:
            events = json.load(stream)["traceEvents"]
        steps.update(int(event["args"]["step_num"]) for event in events
                     if event.get("name") == "train" and "step_num" in event.get("args", {}))
    return tuple(sorted(steps))


def device_time(devices: Sequence[Sequence[tuple[str, int, int]]], *, steps: int,
                step_ms: float) -> dict[str, float]:
    """A traced window's device time per step and device, in milliseconds, from each device's kernels
    (`(name, start_ns, end_ns)`): `compute` and `communication` the unions of its compute and its NCCL
    kernels, `exposed_communication` the communication no compute ran beside
    (`dew.telemetry.timeline.window_split`), so time two kernels overlap counts once, and `device_idle`
    the rest of the step's `step_ms` of wall time, which no kernel ran in: dispatch, host work and the
    device's wait on its input."""
    from dew.telemetry.timeline import window_totals

    totals = window_totals(devices)
    scale = 1e-6 / steps / len(devices)
    busy = (totals["compute"] + totals["exposed_communication"]) * scale
    return {"compute_ms": totals["compute"] * scale, "communication_ms": totals["communication"] * scale,
            "exposed_communication_ms": totals["exposed_communication"] * scale,
            "device_idle_ms": step_ms - busy}


@dataclasses.dataclass(frozen=True)
class ProfileReport:
    """Where the steps of a run's profile window spent their time (`dew.config.RunConfig.profile`): the
    window's own log, `Trainer.fit`'s scalars over exactly the steps its capture traced (`traced`), and the
    device time its devices' kernels took, overlapping kernels counted once (`device_time`). Each figure
    is per step, and the device's per device; None where the run reports none: no analytic work for its
    model (`Objective.training_flops`, never the compiled program's count), no device peak to rate it
    against, or no device kernels in the capture, a CPU's (`attribution` says why). The window holds no
    checkpoint, validation or log read but its own, so their time is outside every figure here.
    `directory` is the profile's, its `report.json` beside the run."""

    directory: str
    steps: int
    warmup: int
    traced: tuple[int, ...]
    capture: str
    step_ms: float
    input_wait_ms: float
    compile_s: float | None
    compute_ms: float | None
    communication_ms: float | None
    exposed_communication_ms: float | None
    device_idle_ms: float | None
    flops: float | None
    mfu: float | None
    hfu: float | None
    tokens_per_s: float | None
    attribution: str | None

    @classmethod
    def read(cls, directory: Path, name: str, window: Path, *, warmup: int, steps: int) -> "ProfileReport":
        """The report of the window of `steps` steps after `warmup` that the run `name` under `directory`
        traced into `window`, this process's newest capture's, its device time read off each local device's
        kernels (`dew.telemetry.timeline.device_events`). The run logged its steps once at the window's end,
        so that log's scalars are the steps the capture's annotations hold (`traced_steps`), or the report
        is refused."""
        from dew.telemetry.profile import process_directory
        from dew.telemetry.timeline import device_events

        capture = max(process_directory(window).glob("capture-*"), key=lambda path: path.stat().st_mtime)
        traced = traced_steps(capture)
        if traced != tuple(range(warmup, warmup + steps)):
            raise ValueError(f"the window traced the steps begun after {list(traced)} steps, and its log "
                             f"covers those begun after {warmup} to {warmup + steps - 1}")
        run = directory / name
        # The window's log, and any score the run's end logged under its last step beside it.
        logged = {key: value for step, scalars in LocalTracker(run / "tracking").scalars()
                  if step == warmup + steps for key, value in scalars.items()}
        records = (run / "tracking" / "records.jsonl").read_text().splitlines()
        compiles = [entry["value"]["seconds"] for entry in map(json.loads, records)
                    if entry["type"] == "StepCompiled"]
        planes, _ = device_events(capture)
        kernels = [[(event.name, event.start_ns, event.end_ns) for event in events]
                   for events in planes.values()]
        unknown = dict.fromkeys(("compute_ms", "communication_ms", "exposed_communication_ms",
                                 "device_idle_ms"))
        step_ms = logged["train/step_time_ms"]
        device = device_time(kernels, steps=steps, step_ms=step_ms) if kernels else unknown
        return cls(str(directory), steps, warmup, traced, str(capture), step_ms, logged["train/data_wait_ms"],
                   compiles[0] if compiles else None, **device,
                   flops=logged.get("train/flops"), mfu=logged.get("train/mfu"), hfu=logged.get("train/hfu"),
                   tokens_per_s=logged.get("train/tokens_per_s"),
                   attribution=None if kernels else "the capture holds no GPU or TPU kernels to attribute")

    def text(self) -> str:
        """The report as lines a person reads, the largest share of a step named last."""
        lines = [f"Profiled {self.steps} steps after {self.warmup} of warmup, the capture in {self.capture}, "
                 f"this report in {self.directory}/report.json",
                 f"  step          {self.step_ms:10.2f} ms  wall clock a step",
                 f"  input wait    {self.input_wait_ms:10.2f} ms  the loop waiting on its next batch"]
        shares: dict[str, float] = {}
        if (self.compute_ms is None or self.communication_ms is None or self.exposed_communication_ms is None
                or self.device_idle_ms is None):
            lines.append(f"  device        unattributed: {self.attribution}")
        else:
            # The step's partition: compute, the communication it hid nothing of, and the rest.
            shares = {"compute": self.compute_ms, "exposed communication": self.exposed_communication_ms,
                      "device idle": self.device_idle_ms}
            lines += [f"  compute       {self.compute_ms:10.2f} ms  device time in compute kernels",
                      f"  communication {self.communication_ms:10.2f} ms  device time in collectives, "
                      f"{self.exposed_communication_ms:.2f} ms of it beside no compute",
                      f"  device idle   {self.device_idle_ms:10.2f} ms  the rest of the step: dispatch, host "
                      "work and input"]
        if self.compile_s is not None:
            lines.append(f"  compile       {self.compile_s:10.2f} s   the step's first compile, in warmup")
        if self.flops is None:
            lines.append("  work          unknown: the objective counts no analytical FLOPs for this model "
                         "(Supervised(work=...) declares them); no compiled count stands in")
        else:
            rated = "" if self.mfu is None else f", MFU {100 * self.mfu:.1f}%"
            rated += "" if self.hfu is None else f", HFU {100 * self.hfu:.1f}%"
            lines.append(f"  work          {self.flops / 1e12:10.4f} TFLOP a step{rated}")
        if self.tokens_per_s is not None:
            lines.append(f"  tokens        {self.tokens_per_s:10.1f} real tokens a second")
        lines.append("  checkpoints, validation and logging: none inside the window")
        if shares:
            largest = max(shares, key=shares.__getitem__)
            share = 100 * shares[largest] / self.step_ms
            lines.append(f"The largest share of a step: {largest}, {share:.0f}%.")
        return "\n".join(lines)
