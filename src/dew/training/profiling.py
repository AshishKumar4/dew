"""A fit's profiler window: the request (`ProfileWindow`) and the capture's life on every rank.

A window is owned before the dataset or the mesh do any work (`owned_capture`), started on every rank
together once its warmup steps have run (`start_capture`), and stopped once its steps have (`stop_capture`).
Each phase runs under `dew._coordination.agreed`, so a rank that fails raises on all of them, and none
proceeds alone to deadlock its peers at the next collective. The trainer reports what the window wrote.
"""

import dataclasses
import json
import sys
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


@dataclasses.dataclass(frozen=True)
class ProfileReport:
    """Where the steps of a run's profile window spent their time (`dew.config.RunConfig.profile`): the
    window's own log, `Trainer.fit`'s scalars over exactly its steps, and the device time XProf's
    hlo_stats attributes to the capture's ops, collectives apart from the rest. Each figure is per
    step; None where the run reports none: no analytic work for its model (`Objective.training_flops`,
    never the compiled program's count), no device peak to rate it against, or no op the capture
    attributes time to (`attribution` says why). The window holds no checkpoint, validation or log
    read but its own, so their time is outside every figure here."""

    steps: int
    warmup: int
    capture: str
    step_ms: float
    input_wait_ms: float
    compile_s: float | None
    device_ms: float | None
    communication_ms: float | None
    flops: float | None
    mfu: float | None
    hfu: float | None
    tokens_per_s: float | None
    attribution: str | None

    @classmethod
    def read(cls, run: Path, window: Path, *, warmup: int, steps: int, devices: int) -> "ProfileReport":
        """The report of the window of `steps` steps after `warmup` that a run under `run` traced into
        `window`, its newest capture's, over `devices` local devices; the run logged its steps once
        at the window's end, so that log's scalars are the window's."""
        from dew.telemetry.instrumentation import collective, reported_instruction
        from dew.telemetry.profile import report_table

        # The window's log, and any score the run's end logged under its last step beside it.
        logged = {name: value for step, scalars in LocalTracker(run / "tracking").scalars()
                  if step == warmup + steps for name, value in scalars.items()}
        records = (run / "tracking" / "records.jsonl").read_text().splitlines()
        compiles = [entry["value"]["seconds"] for entry in map(json.loads, records)
                    if entry["type"] == "StepCompiled"]
        capture = max(window.rglob("capture-*"), key=lambda path: path.stat().st_mtime)
        _, rows = report_table(capture / "reports" / "hlo_stats.json")
        ops = [row for row in rows if not str(row.get("hlo_op_name") or "").startswith("XlaModule")]
        # Why no op holds time, where the export could tell (`dew.telemetry.profile.COMMAND_BUFFERS`).
        manifest = json.loads((capture / "manifest.json").read_text())
        unattributed = manifest.get("op_attribution") or "XProf's hlo_stats names no op of this backend's"

        def per_step(chosen: list) -> float:
            return sum(float(str(row["total_self_time"])) for row in chosen) / 1000 / steps / devices

        instructions = [(row, reported_instruction(str(row.get("hlo_op_expression") or ""))) for row in ops]
        communication = [row for row, op in instructions if op is not None and collective(op.op) is not None]
        return cls(steps, warmup, str(capture), logged["train/step_time_ms"], logged["train/data_wait_ms"],
                   compiles[0] if compiles else None, per_step(ops) if ops else None,
                   per_step(communication) if ops else None, logged.get("train/flops"),
                   logged.get("train/mfu"), logged.get("train/hfu"), logged.get("train/tokens_per_s"),
                   None if ops else unattributed)

    def text(self) -> str:
        """The report as lines a person reads, the largest share of a step named last."""
        lines = [f"Profiled {self.steps} steps after {self.warmup} of warmup, the capture in {self.capture}",
                 f"  step          {self.step_ms:10.2f} ms  wall clock a step",
                 f"  input wait    {self.input_wait_ms:10.2f} ms  the loop waiting on its next batch"]
        shares = {"input wait": self.input_wait_ms}
        if self.device_ms is None or self.communication_ms is None:
            lines.append(f"  device        unattributed: {self.attribution}")
        else:
            compute = self.device_ms - self.communication_ms
            idle = max(self.step_ms - self.device_ms - self.input_wait_ms, 0.0)
            shares.update({"compute": compute, "communication": self.communication_ms, "device idle": idle})
            lines += [f"  compute       {compute:10.2f} ms  device time outside collectives",
                      f"  communication {self.communication_ms:10.2f} ms  device time in collectives",
                      f"  device idle   {idle:10.2f} ms  the rest of the step: dispatch and host work"]
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
