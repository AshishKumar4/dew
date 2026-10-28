"""Process-local JAX profile captures, exported as XProf's own reports.

A capture runs only when you start a `Profiler`. Starting and stopping it
wait for the default backend's live arrays and effects, but they do not
synchronize with other processes. When no capture is running, Dew installs
no instrumentation hooks.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import shlex
import tempfile
import threading
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from importlib.metadata import version
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, Self, runtime_checkable

import jax

from dew._records import JSON


@runtime_checkable
class _Converter(Protocol):
    def xspace_to_tool_names(self, paths: Sequence[str]) -> list[str]: ...

    def xspace_to_tool_data(
        self, paths: Sequence[str], tool: str, params: Mapping[str, object]
    ) -> tuple[bytes | str | None, str]: ...


_active: Profiler | None = None
_lifecycle = threading.Lock()
_logger = logging.getLogger(__name__)


def require_profile_support() -> _Converter:
    """Resolve the optional native report converter before starting work."""
    try:
        converter = importlib.import_module("xprof.convert.raw_to_tool_data")
    except ImportError as error:
        raise ImportError("Profiling requires XProf; install 'dewml[profile]'.") from error
    if not isinstance(converter, _Converter):
        raise ImportError("Installed XProf lacks native converters; install 'dewml[profile]'.")
    return converter


def capture_options() -> jax.profiler.ProfileOptions:
    """The options a Dew capture starts with, for any JAX trace that should read like one.

    Host events at level 2 and the compiled HLO, and no Python tracer: it
    records every Python and C call, which slows Python-heavy host work
    several times over (a server's 512-row settle loop took 0.85 ms a step
    traced against 0.18 ms untraced), so the trace would show host time,
    and the device idle behind it, that the run never spends. On TPU the
    trace also carries compute and sync events.
    """
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 2
    options.python_tracer_level = 0
    # Exporting every live executable's HLO cost 0.95 s and 3.4 MB for a
    # 4-layer decoder's fit that had evaluated and sampled first (RTX 4080).
    # 400 unrelated compiled programs made a CPU capture of small work take
    # 15 s and 75 MB.
    options.enable_hlo_proto = True
    if jax.default_backend() == "tpu":
        options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}
    return options


def active_profile() -> Profiler | None:
    """The explicitly enabled process-local profiler, or None."""
    owner = _active
    return owner if owner is not None and owner._phase == "running" else None


def _drain() -> None:
    """Wait for the work in flight: every live array, then the effects.

    An array can list as live with one of its buffers deleted, one whose
    shard's buffer its holder deleted, and waiting on it raises. The step's
    outputs are the arrays to wait on, so it is passed over."""
    jax.block_until_ready([array for array in jax.live_arrays()
                           if not any(shard.data.is_deleted() for shard in array.addressable_shards)])
    jax.effects_barrier()


COMMAND_BUFFERS = (
    "no device time is attributed to HLO ops (reports/hlo_stats.json): the kernels XLA ran as CUDA "
    "command buffers stay under their module. Profile with "
    "XLA_FLAGS=--xla_gpu_enable_command_buffer= for op-level reports, and time runs without it")


def process_directory(directory: Path) -> Path:
    """Where this process's captures under `directory` go: `directory` itself, or in a pool of processes
    `process-<rank>` within it, since each process captures its own devices."""
    return directory if jax.process_count() == 1 else directory / f"process-{jax.process_index()}"


def report_table(path: Path, index: int = 0) -> tuple[list[str], list[dict[str, JSON]]]:
    """The `index`-th DataTable of an exported XProf report (`reports/hlo_stats.json`,
    `framework_op_stats.json`, ...): its column ids, and its rows as values by column.
    A report that is absent or holds no such table has neither."""
    exported = json.loads(path.read_text()) if path.exists() else []
    tables = exported if isinstance(exported, list) else [exported]
    if index >= len(tables) or not isinstance(tables[index], dict) or "cols" not in tables[index]:
        return [], []
    columns = [column.get("id") for column in tables[index]["cols"]]
    return columns, [{name: (cell.get("v") if isinstance(cell, dict) else cell)
                      for name, cell in zip(columns, row["c"], strict=False)}
                     for row in tables[index].get("rows", [])]


def program_kernels(capture: Path, programs: Sequence[str]) -> dict[str, list[dict[str, JSON]]]:
    """A capture's kernels (one of `Profiler.captures`; XProf's kernel_stats, `report_table`), each row
    filed under the first of `programs`, in their order, whose name its op name holds: the jitted program
    it ran in, which XProf's `op_name` begins with (`jit(<name>)/...`), so callers name each program for
    what it runs (`fn.__name__`). The order decides between names one holds another of (`step` before
    `step_eval` files both under `step`). The rest go under `''`; each list by total time, longest first.
    A capture whose export failed has no kernel_stats: FileNotFoundError naming it."""
    stats = capture / "reports" / "kernel_stats.json"
    if not stats.is_file():
        raise FileNotFoundError(f"capture {capture} has no kernel_stats report (its manifest's "
                                "capture_errors and export_status say why)")
    _, rows = report_table(stats)
    found: dict[str, list[dict[str, JSON]]] = {name: [] for name in (*programs, "")}
    for row in rows:
        op = str(row.get("op_name") or "")
        owner = next((name for name in programs if name in op), "")
        found[owner].append(row)
    for kernels in found.values():
        kernels.sort(key=lambda row: -float(str(row.get("total_duration_us") or 0)))
    return found


def attributed_ops(reports: Path) -> int | None:
    """How many HLO ops XProf's hlo_stats in `reports` attributes device time
    to, not counting whole modules (`XlaModule`), as it reports kernels run as
    a command buffer; None where it has no hlo_op_name column to tell by."""
    columns, rows = report_table(reports / "hlo_stats.json")
    if "hlo_op_name" not in columns:
        return None
    return sum(not str(row.get("hlo_op_name") or "").startswith("XlaModule") for row in rows)


def _reports(directory: Path, converter: _Converter, metadata: dict[str, JSON]) -> None:
    paths = sorted(directory.glob("plugins/profile/*/*.xplane.pb"))
    failures: list[Exception] = []
    records: list[JSON] = []
    metadata["reports"] = records
    reports = directory / "reports"
    reports.mkdir()
    if not paths:
        failures.append(RuntimeError("JAX capture produced no native XPlane files"))
    sessions = sorted({path.parent for path in paths})
    for index, session in enumerate(sessions):
        hosts = [str(path) for path in paths if path.parent == session]
        try:
            tools = converter.xspace_to_tool_names(hosts)
            if not tools:
                raise RuntimeError("Native XProf tool discovery returned no tools")
        except Exception as error:
            records.append({"session": str(session.relative_to(directory)),
                            "status": "error", "error": str(error)})
            failures.append(error)
            continue
        for tool in tools:
            record: dict[str, JSON] = {"tool": tool, "session": str(session.relative_to(directory))}
            records.append(record)
            if tool in ("graph_viewer", "memory_viewer", "trace_viewer@", "trace_viewer"):
                record.update(
                    status="interactive", reason="Use the native XProf viewer and retained trace/HLO files"
                )
                continue
            if not tool or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in tool
            ):
                error = ValueError(f"Unexpected native tool name: {tool!r}")
                record.update(status="error", error=str(error))
                failures.append(error)
                continue
            groups = [[host] for host in hosts] if tool == "memory_profile" else [hosts]
            artifacts: list[JSON] = []
            record["artifacts"] = artifacts
            record["status"] = "completed"
            for host_index, group in enumerate(groups):
                try:
                    payload, mime = converter.xspace_to_tool_data(group, tool, {})
                    if payload is None:
                        record.update(status="unavailable", reason="Native converter returned no data")
                        continue
                    if not isinstance(payload, (bytes, str)):
                        raise TypeError(
                            f"Native {tool} returned {type(payload).__name__}; expected bytes or str"
                        )
                    extension = {"application/json": ".json", "text/html": ".html",
                                 "text/plain": ".txt", "application/octet-stream": ".bin"}.get(mime, ".bin")
                    suffix = (f"-session-{index}" if len(sessions) > 1 else "")
                    suffix += f"-host-{host_index}" if len(groups) > 1 else ""
                    destination = reports / (tool + suffix + extension)
                    destination.write_bytes(payload.encode("utf-8") if isinstance(payload, str) else payload)
                    artifacts.append(str(destination.relative_to(directory)))
                    record["content_type"] = mime
                except Exception as error:
                    record.update(status="error", error=str(error))
                    failures.append(error)
    attributed = attributed_ops(reports) if metadata.get("backend") == "gpu" else 1
    if attributed is None:
        metadata["op_attribution"] = "unknown: hlo_stats has no hlo_op_name column"
    elif attributed == 0:
        metadata["op_attribution"] = COMMAND_BUFFERS
        _logger.warning("Profile %s: %s", directory, COMMAND_BUFFERS)
    written: list[JSON] = [str(path.relative_to(directory))
                           for path in sorted(directory.rglob("*")) if path.is_file()]
    metadata["artifacts"] = written
    metadata["export_status"] = "failed" if failures else "completed"
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if failures:
        raise ExceptionGroup("Native profile report export failed", failures)


class Profiler:
    """Captures a JAX profile of this process, as a context manager or through
    `start` and `stop`.

    One profiler can capture more than once. Each `start` creates a new
    capture directory under `directory`. Without a directory, the first start
    creates a temporary one, which is kept after the run. A capture covers
    only this process, so other processes must start their own. Starting and
    stopping wait for the work on JAX's default backend to finish, without
    copying array values to the host. Only one capture can run in a process
    at a time, and `start` raises `RuntimeError` while another is running.

    A capture includes the HLO of every executable alive in the process, not
    only of the work it traced, and `stop` exports all of it. The op_profile,
    hlo_stats and roofline_model reports are built from that HLO and come out
    nearly empty without it. In a training run's window this costs little.
    A long-lived process that has compiled many other programs pays for each
    of them, so there pass `options` with `enable_hlo_proto` off when the
    trace and kernel reports are enough.

    On a GPU the reports also come out empty for work that already ran before
    the capture, as a Trainer's profile window does: XLA runs a step's
    kernels as CUDA command buffers, and their device time is attributed to
    their module, not their ops (on JAX 0.11.2 an RTX 4080's warmed step
    named no op; 63 when its first run fell inside the capture; 166 with
    command buffers off). XLA means to run command buffers op by op while a
    profiling session is active (command_buffer_thunk.cc, ExecuteOnStream),
    and here does not; why is open, and tools/gpu_profile_check.py records
    whether it does on every GPU session. The manifest's `op_attribution`
    then says so and a warning names the fix, profiling with
    `XLA_FLAGS=--xla_gpu_enable_command_buffer=`; time runs without that
    flag, since launching kernels one by one costs time. Where hlo_stats has
    no column naming the ops, `op_attribution` records that it could not tell.
    """

    def __init__(self, directory: str | os.PathLike[str] | None = None, *,
                 options: jax.profiler.ProfileOptions | None = None) -> None:
        self._directory = None if directory is None else Path(directory)
        self._options = options
        self._capture: Path | None = None
        self._converter: _Converter | None = None
        self._metadata: dict[str, JSON] = {}
        self._temporary = directory is None
        self._phase: Literal["starting", "running", "stopping"] | None = None
        self._captures: list[Path] = []

    @property
    def directory(self) -> Path | None:
        return self._directory

    @property
    def captures(self) -> tuple[Path, ...]:
        """The capture directories this profiler's stops exported, in order: each holds its manifest and
        its `reports/` (`report_table`, `program_kernels`). Where an export failed (the manifest's
        `capture_errors`, `export_status`) reports/ may lack a report."""
        return tuple(self._captures)

    def region(self, name: str) -> AbstractContextManager[None]:
        """Return a context manager that marks a named span in the trace while
        this profiler is capturing.

        Wrap the work whose device and host time should show under `name`.
        Outside a capture the span does nothing, so the same code runs
        unprofiled without a branch at every call site.
        """
        if not self.running:
            return nullcontext()
        return jax.profiler.TraceAnnotation(name)

    @property
    def running(self) -> bool:
        return active_profile() is self

    def start(self) -> Self:
        global _active
        with _lifecycle:
            if _active is not None:
                raise RuntimeError("A Dew profile is already running in this process")
            _active = self
            self._phase = "starting"
        owned_start = False
        try:
            converter = require_profile_support()
            backend = jax.default_backend()
            options = capture_options() if self._options is None else self._options
            _drain()
            if self._directory is None:
                self._directory = Path(tempfile.mkdtemp(prefix="dew-profile-"))
            self._directory = self._directory.expanduser().resolve()
            root = process_directory(self._directory)
            count, rank = jax.process_count(), jax.process_index()
            root.mkdir(parents=True, exist_ok=True)
            capture = Path(tempfile.mkdtemp(prefix="capture-", dir=root))
            metadata: dict[str, JSON] = {
                "versions": {name: version(name) for name in ("jax", "jaxlib", "xprof")},
                "backend": backend, "devices": [str(device) for device in jax.local_devices()],
                "device_kinds": [device.device_kind for device in jax.local_devices()],
                "process_index": rank, "process_count": count,
                "scope": "process-local; default-backend drain; no distributed barrier",
                "default_matmul_precision": jax.config.jax_default_matmul_precision,
                "jax_enable_x64": jax.config.jax_enable_x64,
                "options": {"host_tracer_level": options.host_tracer_level,
                            "python_tracer_level": options.python_tracer_level,
                            "enable_hlo_proto": options.enable_hlo_proto,
                            "advanced_configuration": options.advanced_configuration},
                "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
                "view_command": "xprof --logdir=" + shlex.quote(str(capture)),
            }
            jax.profiler.start_trace(capture, profiler_options=options)
            owned_start = True
            with _lifecycle:
                self._capture, self._converter, self._metadata = capture, converter, metadata
                self._phase = "running"
        except BaseException as primary:
            try:
                if owned_start:
                    try:
                        jax.profiler.stop_trace()
                    except BaseException as cleanup:
                        primary.add_note(f"Owned profile start cleanup failed: {cleanup}")
            finally:
                with _lifecycle:
                    _active = None
                    self._phase = None
                    self._capture = None
                    self._converter = None
                    self._metadata = {}
            raise
        return self

    def stop(self) -> None:
        self._stop("completed")

    def _stop(self, body: str) -> None:
        global _active
        failures: list[BaseException] = []
        with _lifecycle:
            if _active is not self or self._phase != "running":
                raise RuntimeError("This profiler does not own a running capture")
            self._phase = "stopping"
            capture, converter, metadata = self._capture, self._converter, self._metadata
        try:
            try:
                _drain()
            except BaseException as error:
                failures.append(error)
            try:
                jax.profiler.stop_trace()
            except BaseException as error:
                failures.append(error)
        finally:
            with _lifecycle:
                _active = None
                self._phase = None
                self._capture = None
                self._converter = None
                self._metadata = {}
        assert capture is not None and converter is not None
        self._captures.append(capture)
        metadata["body_status"] = body
        metadata["capture_status"] = "failed" if failures else "completed"
        metadata["capture_errors"] = [str(error) for error in failures]
        try:
            _reports(capture, converter, metadata)
        except BaseException as error:
            failures.append(error)
        if self._temporary:
            _logger.info("Profile saved under %s", self._directory)
        if failures:
            raise BaseExceptionGroup("Profile cleanup or export failed", failures)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, traceback: TracebackType | None) -> None:
        if self.running:
            try:
                self._stop("failed" if exc is not None else "completed")
            except BaseException as error:
                if exc is None:
                    raise
                exc.add_note(f"Profile cleanup/report failure: {error}")


def region(name: str) -> AbstractContextManager[None]:
    """Return a context manager that marks a named span in the running
    capture, or does nothing when no capture is running.

    The call site does not need to know whether a profiler is running or
    which one; `with region("input.wait"):` reads the same either way.
    """
    active = active_profile()
    return nullcontext() if active is None else active.region(name)


__all__ = ["Profiler", "region"]
