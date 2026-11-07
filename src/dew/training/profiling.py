"""A fit's profiler window: the request (`ProfileWindow`) and the capture's life on every rank.

A window is owned before the dataset or the mesh do any work (`owned_capture`), started on every rank
together once its warmup steps have run (`start_capture`), and stopped once its steps have (`stop_capture`).
Each phase runs under `dew._coordination.agreed`, so a rank that fails raises on all of them, and none
proceeds alone to deadlock its peers at the next collective. The trainer reports what the window wrote.
"""

import dataclasses
import functools
import gzip
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from dew._coordination import agreed, from_every_process
from dew.telemetry import profile as telemetry_profile
from dew.telemetry.records import ProfileTraced, Record
from dew.training.tracker import LocalTracker

if TYPE_CHECKING:
    import jax

    from dew.telemetry.profile import Profiler


_log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ProfileWindow:
    """A request for one profiler window per fit.

    It traces `steps` steps into `directory` after `warmup` steps have run and the
    run's first step has compiled, so the trace shows the loop and not the compile, and it holds the
    training steps and the loop's own cadenced reads of them (its log, an observer's call): the work between
    them that is no step's (an evaluation, a checkpoint save, a cooldown branch) runs outside its capture,
    which starts again at the next step's read, so a window that spans such work writes a capture of each
    run of its steps (`dew.training._fit.FitRun.outside_window`). A fit under JAX's AutoPGLE, which
    profiles programs in sessions of its own, refuses a window (`PGLE_CAPTURE`).

    The capture records what `capture` asks (`dew.telemetry.profile.CaptureOptions`): by default
    host events at level 2, no Python calls (the Python tracer slows Python-heavy host work several
    times over), and on GPU at most its `activity_cap` CUDA activity events.

    A window that `reports` nothing keeps its trace and manifest alone, which needs no XProf and its stop
    writes at once; one that reports converts its capture's reports in a process of its own while the run
    trains on, waited for at the fit's end, minutes for a training window, or, where it `wait`s, in the
    run's own process as its capture stops, before the run's next step, as `Profiler.stop(wait=True)`
    does: a short interactive run's, or one whose sandbox forbids it to create a process (a live page's
    kernel). A run's own window (`RunProfile`, every run's by default) is
    resolved into one by its config (`dew.config.TrainerConfig.window`), kept with its records.

    `dew.Profiler` is the other way to capture a trace, as a context manager
    around any code; a fit refuses to schedule a window inside one. The loop
    reports the window it wrote as the `ProfileTraced` record of
    `dew.telemetry.records`.
    """
    directory: str
    steps: int
    warmup: int = 2
    capture: telemetry_profile.CaptureOptions = dataclasses.field(
        default_factory=telemetry_profile.CaptureOptions)
    reports: bool = True
    wait: bool = False


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


@dataclasses.dataclass(frozen=True)
class RunProfile:
    """A run's own profile window, every run's by default (`dew.config.TrainerConfig.run_profile`): `steps`
    traced once `warmup` have run past its first compile, kept with the run's records, its trace and
    manifest alone (no reports: no XProf asked, no conversion the run's end waits for), and taken where the
    process can take one (`capturable`), none where it cannot, where a `ProfileWindow` a run names refuses.
    Its config resolves it into its window (`dew.config.TrainerConfig.window`); a trainer is given that."""

    steps: int = 2
    warmup: int = 10
    capture: telemetry_profile.CaptureOptions = dataclasses.field(
        default_factory=telemetry_profile.CaptureOptions)

    def window(self, directory: str) -> ProfileWindow:
        """This window as a trainer traces it, into `directory`."""
        return ProfileWindow(directory, self.steps, warmup=self.warmup, capture=self.capture, reports=False)


def capturable() -> bool:
    """Whether every process can take a capture: none holds another (`dew.Profiler`) and none runs under
    JAX's AutoPGLE (`pgle_runs`); a run's own window (`RunProfile`), which reports nothing and so asks no
    XProf, is taken where they can, a window a run names refuses where they cannot (`owned_capture`)."""
    local = telemetry_profile.active_profile() is None and not pgle_runs()
    return all(from_every_process(local))


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
        if window.reports:
            telemetry_profile.require_profile_support()
        return telemetry_profile.Profiler(window.directory, options=window.capture.profile_options(),
                                          reports=window.reports)

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


def stop_capture(profiler: "Profiler", loss: "jax.Array | None", *, wait: bool = False) -> None:
    """Stop the window's owned capture on every rank.

    The core Profiler drains the backend and stops the trace here, and
    exports its native reports in a process of its own while the run
    trains on (`Profiler.stop`, row 2017), which the run's cleanup waits
    for (`dew.telemetry.profile.exported`), or here where the window
    `wait`s (`ProfileWindow.wait`). The run's last `loss` is blocked
    on only to order the primary's failure ahead of the profiler's own drain.
    A window that reports nothing writes its manifest as it stops, a
    diagnostic of the run's: its write failing is logged, and the run trains
    on (`dew.telemetry.profile.ExportFailed`, which only such a window's stop
    raises, a reporting window's export running in a process of its own)."""
    def stop() -> None:
        try:
            if loss is not None:
                loss.block_until_ready()
        finally:
            primary = sys.exception()
            try:
                profiler.stop(wait=wait)
            except telemetry_profile.ExportFailed as failure:
                _log.warning("the profile window traced, and its manifest was not written: %s (%s)",
                             failure, failure.exceptions)
            except BaseException as failure:
                if primary is None:
                    raise
                primary.add_note(f"Profiler stop failed: {failure!r}")

    agreed("profile stop", stop)


def waited_export(window: ProfileWindow | None, profiler: "Profiler | None") -> Callable[[], None] | None:
    """What a fit's cleanup waits for of its `window`'s capture by `profiler`: this rank's export of the
    reports it converts, of the captures under the profiler's own directory alone
    (`dew.telemetry.profile.exported`), a failed one failing every rank at the run's outcome's agreement;
    none for a window that reports nothing, whose manifest its stop wrote, or that never started."""
    if window is None or not window.reports or profiler is None:
        return None
    return functools.partial(_exported_under, profiler)


def _exported_under(profiler: "Profiler") -> None:
    """`profiler`'s exports waited for, under the directory it captured into once it started; none where it
    never did."""
    if profiler.directory is not None:
        telemetry_profile.exported(profiler.directory)


def reported_window(report: Callable[[Record, int], None], note: Callable[[str], None],
                    profiled: tuple[int, int, str] | None,
                    error: BaseException | None) -> BaseException | None:
    """Report a fit's window that stopped, `profiled` its steps traced, the step it stopped at and the
    directory its capture is in, once every rank's export is written and the pool agreed the run's
    outcome (`Trainer._reported_outcome`), whatever that outcome: its record (`report`) and its note on
    process zero (`note`). Returns the run's `error`, or a failed report as the error of a run that had
    none, noted on one that had."""
    if profiled is None:
        return error
    traced, step, directory = profiled
    try:
        report(ProfileTraced(directory, traced), step)
        wrote = f"Wrote profile for {traced} steps to {directory}"
        agreed("profile announcement", lambda: note(wrote))
    except BaseException as failure:
        if error is None:
            return failure
        error.add_note(f"Reporting the profile window failed: {failure!r}")
    return error


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


def device_figures(devices: "list[list[tuple[str, float, float]]]", steps: int) -> dict[str, float | None]:
    """A window's device time a step from each device's kernels (`(name, start ns, end ns)`), the devices'
    mean (`dew.telemetry.trace.window_split`): the window, its compute, its communication whole and the part
    no compute ran beside, and the gaps no kernel ran in; overlapped kernels count once. Each None where no
    device holds a kernel."""
    from dew.telemetry.trace import window_split

    splits = [window_split(kernels) for kernels in devices if kernels]

    def per_step(*keys: str) -> float | None:
        if not splits:
            return None
        return sum(sum(split[key] for key in keys) for split in splits) / len(splits) / steps / 1e6

    return {"device_window_ms": per_step("window"), "compute_ms": per_step("compute"),
            "communication_ms": per_step("communication"),
            "exposed_communication_ms": per_step("exposed_communication"),
            "device_idle_ms": per_step("idle_input", "idle_host")}


@dataclasses.dataclass(frozen=True)
class ProfileReport:
    """Where the steps of a run's profile window spent their time (`dew.config.RunConfig.profile`): the
    window's own log, `Trainer.fit`'s scalars over exactly the steps its capture traced (`traced`), and
    the device's time from its kernels' intervals in the capture (`dew.telemetry.trace.window_split`),
    averaged over the devices whose tracks it holds: the window from the first kernel's start to the
    last one's end; compute and the collective time no compute ran beside (`exposed_communication_ms`),
    which with the device's idle gaps make up the window, overlapped kernels counted once; and
    communication whole. Input wait is the host's, which the device's queue can hide, so no device
    figure is taken from it. Each figure is per step; None where the run reports none: no analytic work
    for its model (`Objective.training_flops`, never the compiled program's count), no device peak to
    rate it against, or no device track in the capture, or a capture that holds less than the window
    ran (its manifest's `capture_status` "partial", CUPTI's activity cap reached or a device's track
    missing), whose surviving kernels no full-window figure may be read from (`device` says why). The
    window holds no checkpoint, validation or log read but its own, so their time is outside every
    figure here.
    `directory` is the profile's, its `report.json` beside the run."""

    directory: str
    steps: int
    warmup: int
    traced: tuple[int, ...]
    capture: str
    step_ms: float
    input_wait_ms: float
    compile_s: float | None
    device_window_ms: float | None
    compute_ms: float | None
    communication_ms: float | None
    exposed_communication_ms: float | None
    device_idle_ms: float | None
    flops: float | None
    mfu: float | None
    hfu: float | None
    tokens_per_s: float | None
    device: str | None

    @classmethod
    def read(cls, directory: Path, name: str, window: Path, *, warmup: int, steps: int) -> "ProfileReport":
        """The report of the window of `steps` steps after `warmup` that the run `name` under `directory`
        traced into `window`, this process's newest capture's. The run logged its steps once at the
        window's end, so that log's scalars are the steps the capture's annotations hold
        (`traced_steps`), or the report is refused."""
        from dew.telemetry.profile import process_directory
        from dew.telemetry.trace import device_events

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
        manifest = json.loads((capture / "manifest.json").read_text())
        reasons = manifest.get("capture_reasons") or []
        planes, _ = device_events(capture)
        figures = device_figures([[(event.name, event.start_ns, event.end_ns) for event in events]
                                  for events in ([] if reasons else planes.values())], steps)
        device = (f"the capture holds less than the window ran: {'; '.join(reasons)}" if reasons
                  else None if figures["device_window_ms"] is not None
                  else "the capture holds no device track (a CPU run, or its trace has none)")
        return cls(str(directory), steps, warmup, traced, str(capture), logged["train/step_time_ms"],
                   logged["train/data_wait_ms"], compiles[0] if compiles else None, **figures,
                   flops=logged.get("train/flops"), mfu=logged.get("train/mfu"), hfu=logged.get("train/hfu"),
                   tokens_per_s=logged.get("train/tokens_per_s"), device=device)

    def text(self) -> str:
        """The report as lines a person reads, the largest share of a step named last."""
        lines = [f"Profiled {self.steps} steps after {self.warmup} of warmup, the capture in {self.capture}, "
                 f"this report in {self.directory}/report.json",
                 f"  step          {self.step_ms:10.2f} ms  wall clock a step",
                 f"  input wait    {self.input_wait_ms:10.2f} ms  the loop waiting on its next batch"]
        shares = {"input wait": self.input_wait_ms}
        if (self.device_window_ms is None or self.compute_ms is None or self.communication_ms is None
                or self.exposed_communication_ms is None or self.device_idle_ms is None):
            lines.append(f"  device        unknown: {self.device}")
        else:
            shares.update({"compute": self.compute_ms, "exposed communication": self.exposed_communication_ms,
                           "device idle": self.device_idle_ms})
            lines += [f"  device        {self.device_window_ms:10.2f} ms  first kernel's start to last's end",
                      f"  compute       {self.compute_ms:10.2f} ms  kernels outside collectives, once each",
                      f"  exposed comm  {self.exposed_communication_ms:10.2f} ms  collectives no compute ran "
                      f"beside (of {self.communication_ms:.2f} ms in all)",
                      f"  device idle   {self.device_idle_ms:10.2f} ms  gaps no kernel ran in: dispatch, "
                      "host work, input"]
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
        largest = max(shares, key=shares.__getitem__)
        lines.append(f"The largest share of a step: {largest}, {100 * shares[largest] / self.step_ms:.0f}%.")
        return "\n".join(lines)
